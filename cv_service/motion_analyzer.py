"""
cv_service/motion_analyzer.py
─────────────────────────────────────────────────────────────
Articulated Motion Detection Engine
"""
from __future__ import annotations

import collections
import os
import sys
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.schema import (
    Activity, EquipmentState, MotionSource, TimeAnalytics, UtilizationInfo
)

# ── LSTM configuration ─────────────────────────────────────────────────────────
INPUT_DIM           = 6
HIDDEN_DIM          = 64
NUM_CLASSES         = 4      # DIGGING, SWINGING, DUMPING, WAITING
SEQ_LEN             = 16
LSTM_CONF_THRESHOLD = 0.90   # below this → rule-based fallback


class ActivityLSTM(nn.Module):
    """
    2-layer LSTM for excavator activity classification.
    Architecture must match train_lstm_colab.py exactly.
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = INPUT_DIM,
            hidden_size = HIDDEN_DIM,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
        )
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])


# LSTM output index → Activity label (excavator only)
_ACTIVITY_LABELS = [
    Activity.DIGGING,
    Activity.SWINGING,
    Activity.DUMPING,
    Activity.WAITING,
]


@dataclass
class EquipmentTracker:
    """
    Per-machine state tracker keyed by canonical business ID.

    Dwell time NEVER resets — the caller (_resolve_id in run_local.py)
    is responsible for mapping ByteTrack IDs to stable canonical IDs
    using IoU-based re-association. This tracker just accumulates.

    Excavator: uses LSTM (if loaded) for activity classification.
    Dump truck / mixer: uses simplified rule-based (MOVING / WAITING only).
    """
    equipment_id:    str
    equipment_class: str
    fps:             float = 30.0

    _sub_arm:     cv2.BackgroundSubtractorMOG2 = field(init=False)
    _sub_body:    cv2.BackgroundSubtractorMOG2 = field(init=False)
    _feature_buf: Deque[List[float]]            = field(init=False)

    # Cumulative time — never reset
    total_tracked: float = 0.0
    total_active:  float = 0.0
    total_idle:    float = 0.0

    # Last known bbox — used by run_local.py for IoU re-association
    last_bbox: Tuple[int, int, int, int] = field(default=(0, 0, 0, 0), init=False)

    # MOG2 warm-up — background model unreliable for first N frames
    WARMUP_FRAMES:  int = 30
    _warmup_frames: int = field(default=0, init=False)

    # State debouncer — prevents ACTIVE/INACTIVE flickering
    # State only commits after N consecutive frames agree
    DEBOUNCE_FRAMES: int = 10
    _pending_state:  str = field(default="INACTIVE", init=False)
    _pending_count:  int = field(default=0, init=False)
    _stable_state:   str = field(default="INACTIVE", init=False)

    def __post_init__(self):
        self._sub_arm     = _make_mog2()
        self._sub_body    = _make_mog2()
        self._feature_buf = collections.deque(maxlen=SEQ_LEN)

    def process_frame(
        self,
        frame:      np.ndarray,
        bbox:       Tuple[int, int, int, int],
        lstm_model: ActivityLSTM,
        use_lstm:   bool = True,
    ) -> UtilizationInfo:
        """
        Run motion analysis for one frame.

        Steps:
          1. Clamp bbox, crop frame
          2. Split crop into ARM / BODY zones
          3. Run MOG2 per zone → motion scores
          4. Build feature vector
          5. Classify activity:
               - Excavator with trained LSTM → LSTM (fallback to rule-based)
               - Excavator without LSTM      → rule-based
               - Dump truck / mixer          → simplified rule-based (MOVING/WAITING)
          6. Debounce state
          7. Accumulate time
        """
        x1, y1, x2, y2 = _clamp_bbox(bbox, frame.shape)
        self.last_bbox  = (x1, y1, x2, y2)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return _waiting_info()

        arm_crop, body_crop = _split_regions(crop, self.equipment_class)
        arm_score  = _motion_score(arm_crop,  self._sub_arm)
        body_score = _motion_score(body_crop, self._sub_body)

        # During warm-up the background model is still learning —
        # every pixel looks like motion, results are meaningless
        self._warmup_frames += 1
        if self._warmup_frames < self.WARMUP_FRAMES:
            self._accumulate(EquipmentState.INACTIVE)
            return _waiting_info()

        # Build feature vector
        h, w     = crop.shape[:2]
        aspect   = w / max(h, 1)
        area_n   = (w * h) / (frame.shape[0] * frame.shape[1] + 1e-6)
        arm_rel  = arm_score  / (arm_score + body_score + 1e-6)
        body_rel = body_score / (arm_score + body_score + 1e-6)
        self._feature_buf.append(
            [arm_score, body_score, arm_rel, body_rel, aspect, area_n]
        )

        # Classify activity based on equipment type
        if self.equipment_class == "excavator":
            # Use LSTM if trained and buffer full, else rule-based
            if use_lstm and len(self._feature_buf) == SEQ_LEN:
                activity, motion_source = _lstm_classify(
                    self._feature_buf, lstm_model, arm_score, body_score
                )
            else:
                activity, motion_source = _excavator_rule_based(
                    arm_score, body_score
                )
        else:
            # Dump truck and mixer: simplified — only MOVING or WAITING
            # No point in complex classification without trained LSTM for them
            activity, motion_source = _simple_rule_based(
                arm_score, body_score
            )

        # Debug logging (set DEBUG_MOTION=1 env var to enable)
        if os.getenv("DEBUG_MOTION") == "1" and self._warmup_frames % 60 == 0:
            ratio = body_score / (arm_score + 1e-6)
            import logging
            logging.getLogger(__name__).debug(
                "%s | arm=%.3f body=%.3f ratio=%.2f → %s",
                self.equipment_id, arm_score, body_score, ratio, activity.value
            )

        # Derive raw state from motion source
        raw_state = (
            EquipmentState.ACTIVE
            if motion_source != MotionSource.NONE
            else EquipmentState.INACTIVE
        )

        # Debounce: only commit state change after DEBOUNCE_FRAMES agree
        raw_str = raw_state.value
        if raw_str == self._pending_state:
            self._pending_count += 1
        else:
            self._pending_state = raw_str
            self._pending_count = 1

        if self._pending_count >= self.DEBOUNCE_FRAMES:
            self._stable_state = raw_str

        state = EquipmentState(self._stable_state)

        # When INACTIVE, always report WAITING for consistency
        if state == EquipmentState.INACTIVE:
            activity      = Activity.WAITING
            motion_source = MotionSource.NONE

        self._accumulate(state)

        return UtilizationInfo(
            current_state    = state,
            current_activity = activity,
            motion_source    = motion_source,
        )

    def _accumulate(self, state: EquipmentState) -> None:
        dt = 1.0 / self.fps
        self.total_tracked += dt
        if state == EquipmentState.ACTIVE:
            self.total_active += dt
        else:
            self.total_idle += dt

    def time_analytics(self) -> TimeAnalytics:
        pct = (
            self.total_active / self.total_tracked * 100
            if self.total_tracked > 0 else 0.0
        )
        return TimeAnalytics(
            total_tracked_seconds = self.total_tracked,
            total_active_seconds  = self.total_active,
            total_idle_seconds    = self.total_idle,
            utilization_percent   = pct,
        )


# ── Classifiers ────────────────────────────────────────────────────────────────

def _excavator_rule_based(
    arm_score:  float,
    body_score: float,
    arm_th:     float = 0.08,
    body_th:    float = 0.06,
) -> Tuple[Activity, MotionSource]:
    """
    Rule-based classifier for excavator.
    Used as LSTM fallback or when buffer not full.

    DIGGING:  arm active, body minor (arm dominates)
    SWINGING: arm active, body nearly equal to arm (≥85%)
    WAITING:  nothing significant moving
    """
    arm_active  = arm_score  > arm_th
    body_active = body_score > body_th

    if not arm_active and not body_active:
        return Activity.WAITING, MotionSource.NONE

    # SWINGING needs body ≥ 85% of arm — true rotation vs dig+cab vibration
    body_dominant = body_score > max(body_th, arm_score * 0.85)

    if arm_active and body_dominant:
        return Activity.SWINGING, MotionSource.FULL_BODY
    if arm_active:
        return Activity.DIGGING,  MotionSource.ARM_ONLY
    if body_active:
        # Tracks moving, arm folded → tramming
        return Activity.WAITING,  MotionSource.TRACKS_ONLY
    return Activity.WAITING, MotionSource.NONE


def _simple_rule_based(
    arm_score:  float,
    body_score: float,
    arm_th:     float = 0.08,
    body_th:    float = 0.06,
) -> Tuple[Activity, MotionSource]:
    """
    Simplified classifier for dump truck and concrete mixer.
    Only two states: MOVING (machine in motion) or WAITING (idle).

    We don't attempt DUMPING/MIXING without a trained LSTM for those classes.
    The utilization metric (active vs idle time) is what matters for the client,
    and MOVING vs WAITING captures that correctly.
    """
    arm_active  = arm_score  > arm_th
    body_active = body_score > body_th

    if arm_active or body_active:
        return Activity.MOVING, MotionSource.FULL_BODY
    return Activity.WAITING, MotionSource.NONE


def _lstm_classify(
    buf:        Deque[List[float]],
    model:      ActivityLSTM,
    arm_score:  float,
    body_score: float,
) -> Tuple[Activity, MotionSource]:
    """
    Run LSTM inference on the current sliding window.
    Falls back to rule-based if confidence < LSTM_CONF_THRESHOLD.
    This ensures UNKNOWN is never returned even with untrained weights.
    """
    try:
        x = torch.tensor([list(buf)], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]

        if float(probs.max()) < LSTM_CONF_THRESHOLD:
            return _excavator_rule_based(arm_score, body_score)

        activity = _ACTIVITY_LABELS[int(probs.argmax())]
        _, source = _excavator_rule_based(arm_score, body_score)
        return activity, source

    except Exception:
        return _excavator_rule_based(arm_score, body_score)


# ── MOG2 helpers ───────────────────────────────────────────────────────────────

def _make_mog2() -> cv2.BackgroundSubtractorMOG2:
    sub = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=40, detectShadows=False
    )
    sub.setNMixtures(3)
    return sub


def _motion_score(region: np.ndarray, sub: cv2.BackgroundSubtractorMOG2) -> float:
    if region.size == 0:
        return 0.0
    gray   = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask   = sub.apply(gray)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return float(np.clip(np.count_nonzero(mask) / (mask.size + 1e-6), 0.0, 1.0))


def _split_regions(
    crop: np.ndarray, equipment_class: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split bounding box into ARM and BODY zones per equipment type.

    Excavator:
      ARM  = top 55%    — boom, stick, bucket
      BODY = bottom 45% — undercarriage, tracks

    Dump truck:
      ARM  = rear 50%   — tipper bed
      BODY = front 50%  — cab, chassis

    Concrete mixer:
      ARM  = rear 60%   — rotating drum
      BODY = front 40%  — cab, chassis
    """
    h, w = crop.shape[:2]
    if equipment_class == "excavator":
        split = int(h * 0.55)
        return crop[:split, :], crop[split:, :]
    elif equipment_class == "dump_truck":
        split = w // 2
        return crop[:, split:], crop[:, :split]
    elif equipment_class == "concrete_mixer_truck":
        split = int(w * 0.4)
        return crop[:, split:], crop[:, :split]
    else:
        split = h // 2
        return crop[:split, :], crop[split:, :]


def _waiting_info() -> UtilizationInfo:
    return UtilizationInfo(
        current_state    = EquipmentState.INACTIVE,
        current_activity = Activity.WAITING,
        motion_source    = MotionSource.NONE,
    )


def _clamp_bbox(
    bbox: Tuple[int, int, int, int], shape: Tuple
) -> Tuple[int, int, int, int]:
    H, W = shape[:2]
    x1, y1, x2, y2 = bbox
    return max(0, int(x1)), max(0, int(y1)), min(W, int(x2)), min(H, int(y2))
