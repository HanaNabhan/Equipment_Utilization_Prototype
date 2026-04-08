"""
run_local.py — Equipment Utilization & Activity Classification

Architecture:
  - YOLO + ByteTrack detects and tracks equipment per frame
  - MachineRegistry maps ByteTrack IDs → stable canonical IDs
    using IoU re-association so dwell time never resets
  - EquipmentTracker (motion_analyzer.py) classifies activity per machine:
      Excavator  → DIGGING / SWINGING / DUMPING / WAITING  (LSTM)
      Dump truck → MOVING / WAITING                        (rule-based)
      Mixer      → MOVING / WAITING                        (rule-based)
  - ClipWriter buffers 1 min of annotated frames → writes H.264 MP4
  - Streamlit dashboard reads DB + clip file

Usage:
    python run_local.py --video data/input.mp4 --fresh
    python run_local.py --video data/input.mp4 --model model/weights/best_model.pt
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from cv_service.preprocess import preprocess_frame, is_blurry
try:
    from infra.kafka_producer import EquipmentProducer
except ImportError:
    class EquipmentProducer:
        def __init__(self, *a, **kw): pass
        def send(self, *a): pass
        def flush(self): pass
        def close(self): pass

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

DB_PATH      = "local_dev.db"
CLIP_PATH    = "latest_clip.mp4"
FULL_VID_PATH = "output_full.mp4"   # complete annotated video
CLIP_SECONDS = 60     # write one clip per minute of processed video
CLIP_FPS_OUT = 20     # playback FPS of the output clip
MIN_FRAMES   = 100    # ignore tracks shorter than this in dashboard

# Your trained model's class IDs — from equipment-4 dataset
# Only excavator (class 3) is tracked for prototype
# Dump trucks and mixers are detected by YOLO but ignored by the pipeline
COCO_MAP = {
    3: "excavator",
}


# ══════════════════════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════════════════════

def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS equipment_telemetry (
            ts                TEXT,
            frame_id          INTEGER,
            equipment_id      TEXT,
            equipment_class   TEXT,
            current_state     TEXT,
            current_activity  TEXT,
            motion_source     TEXT,
            confidence        REAL,
            total_tracked_sec REAL,
            total_active_sec  REAL,
            total_idle_sec    REAL,
            utilization_pct   REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS processing_status (
            key TEXT PRIMARY KEY, value TEXT
        )
    """)
    conn.commit()
    return conn


def update_status(conn, **kwargs):
    for k, v in kwargs.items():
        conn.execute(
            "INSERT OR REPLACE INTO processing_status VALUES (?,?)",
            (k, str(v))
        )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
# IoU helper
# ══════════════════════════════════════════════════════════════════════════════

def _iou(a, b) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    aa = max(1, (a[2]-a[0])*(a[3]-a[1]))
    ab = max(1, (b[2]-b[0])*(b[3]-b[1]))
    return inter / (aa + ab - inter + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# Machine Registry
# ══════════════════════════════════════════════════════════════════════════════

class MachineRegistry:
    """
    Maps ByteTrack IDs → stable canonical IDs.

    ByteTrack drops and reassigns IDs on occlusion or reappearance.
    We re-associate based on IoU between the new detection bbox and
    the history of recent bboxes of each existing tracker.

    New machine creation rules:
      - A ByteTrack ID must appear for CONFIRM_FRAMES before we create
        a new canonical ID (kills one-frame ghost detections)
      - We only suppress a new ID if we're confident it is the same
        physical machine: IoU ≥ IOU_THRESH with an existing tracker.
        If IoU is low and the detection is confirmed → it IS a new machine.
      - Scene change (global frame diff > threshold) resets suppression.

    This is deliberately simple — the IoU check is the gating condition.
    If IoU ≥ threshold → same machine. If not → new machine (after confirmation).
    """

    CONFIRM_FRAMES = 6     # detections needed before creating a new canonical ID
    IOU_THRESH     = 0.35  # minimum IoU to call it the same machine
    HIST_LEN       = 60    # how many recent bboxes to keep per tracker

    def __init__(self, eff_fps: float):
        from cv_service.motion_analyzer import EquipmentTracker
        self._ET          = EquipmentTracker
        self._eff_fps     = eff_fps
        self._trackers:  dict = {}               # canonical_id → EquipmentTracker
        self._id_map:    dict = {}               # byte_id → canonical_id
        self._hist:      dict = {}               # canonical_id → deque of bboxes
        self._pending:   dict = {}               # byte_id → {count, class, bbox}
        self._prev_gray       = None
        self._scene_unlocked  = False

    def update_scene(self, gray_small: np.ndarray):
        """Call every N frames with a small grayscale version of the frame."""
        if self._prev_gray is not None:
            diff = float(cv2.absdiff(self._prev_gray, gray_small).mean())
            if diff > 28.0:
                self._scene_unlocked = True
                log.info("Scene change detected (diff=%.1f)", diff)
        self._prev_gray = gray_small.copy()

    def resolve(self, byte_id: int, eq_class: str,
                bbox: tuple, conf: float):
        """
        Returns (canonical_id, tracker) or (None, None) if confirming.
        """
        # ── Already mapped ─────────────────────────────────────────
        if byte_id in self._id_map:
            cid = self._id_map[byte_id]
            if cid in self._trackers:
                self._push(cid, bbox)
                return cid, self._trackers[cid]

        # ── IoU search in confirmed trackers of same class ─────────
        best_iou, best_cid = 0.0, None
        for cid, tracker in self._trackers.items():
            if tracker.equipment_class != eq_class:
                continue
            # Check last known bbox
            iou = _iou(bbox, tracker.last_bbox)
            # Also check recent history
            for hbbox in self._hist.get(cid, []):
                iou = max(iou, _iou(bbox, hbbox))
            if iou > best_iou:
                best_iou, best_cid = iou, cid

        if best_iou >= self.IOU_THRESH:
            # Same physical machine — re-associate
            log.info("Re-associated byte_id=%d → %s (IoU=%.2f)",
                     byte_id, best_cid, best_iou)
            self._id_map[byte_id] = best_cid
            self._push(best_cid, bbox)
            self._pending.pop(byte_id, None)
            return best_cid, self._trackers[best_cid]

        # ── No IoU match — accumulate confirmation buffer ──────────
        if byte_id not in self._pending:
            self._pending[byte_id] = {
                "class": eq_class, "bbox": bbox, "count": 0
            }
        p = self._pending[byte_id]
        p["count"] += 1
        p["bbox"]   = bbox

        if p["count"] < self.CONFIRM_FRAMES:
            return None, None  # still waiting for confirmation

        # ── Confirmed new detection — create canonical ID ──────────
        prefix = {
            "excavator":            "EX",
            "dump_truck":           "DT",
            "concrete_mixer_truck": "CMT",
        }.get(eq_class, "EQ")

        n      = sum(1 for t in self._trackers.values()
                     if t.equipment_class == eq_class)
        new_id = f"{prefix}-{n+1:03d}"

        tracker = self._ET(new_id, eq_class, self._eff_fps)
        self._trackers[new_id] = tracker
        self._id_map[byte_id]  = new_id
        self._hist[new_id]     = collections.deque(maxlen=self.HIST_LEN)
        self._scene_unlocked   = False
        self._pending.pop(byte_id, None)

        log.info("New machine: %s (%s)", new_id, eq_class)
        return new_id, tracker

    def _push(self, cid: str, bbox: tuple):
        if cid not in self._hist:
            self._hist[cid] = collections.deque(maxlen=self.HIST_LEN)
        self._hist[cid].append(bbox)

    def all_trackers(self) -> dict:
        return dict(self._trackers)


# ══════════════════════════════════════════════════════════════════════════════
# Frame annotation
# ══════════════════════════════════════════════════════════════════════════════

# Activity → display color
ACTIVITY_COLOR = {
    "DIGGING":  (59,  130, 246),   # blue
    "SWINGING": (139,  92, 246),   # purple
    "DUMPING":  (245, 158,  11),   # amber
    "MOVING":   (6,   182, 212),   # cyan
    "WAITING":  (71,   85, 105),   # slate
    "MIXING":   (16,  185, 129),   # green
}


def draw_annotations(frame: np.ndarray, detections: list):
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]

        # Color by activity
        activity = d["activity"]
        bgr      = ACTIVITY_COLOR.get(activity, (71, 85, 105))
        color    = (bgr[2], bgr[1], bgr[0])  # RGB → BGR for OpenCV

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Two label lines:
        # Line 1: ID  |  ACTIVITY
        # Line 2: util%  state
        line1 = f"{d['eq_id']}  {activity}"
        line2 = f"{d['util_pct']:.0f}%  {d['state']}"

        for i, txt in enumerate([line1, line2]):
            yp = y1 - 22 + i * 16
            if yp < 14:
                yp = y2 + 16 + i * 16
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            cv2.rectangle(frame, (x1, yp-th-2),
                          (x1+tw+4, yp+2), color, -1)
            cv2.putText(frame, txt, (x1+2, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                        (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"Equipment Utilization  {time.strftime('%H:%M:%S')}",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
        (160, 160, 160), 1, cv2.LINE_AA,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Clip writer — 1 minute → H.264 MP4
# ══════════════════════════════════════════════════════════════════════════════

class ClipWriter:
    """
    Buffers CLIP_SECONDS of annotated frames then writes an MP4.
    Uses ffmpeg for H.264 encoding (browser-compatible).
    Falls back to mp4v if ffmpeg not available.
    """

    def __init__(self, out_path: str, clip_sec: float,
                 src_fps: float, out_fps: float = CLIP_FPS_OUT):
        import shutil
        self._out    = out_path
        self._fps    = out_fps
        self._maxbuf = max(1, int(src_fps * clip_sec))
        self._buf:list= []
        self._w      = 0
        self._h      = 0
        self._ffmpeg = shutil.which("ffmpeg")
        log.info("ClipWriter: buf=%d frames, ffmpeg=%s",
                 self._maxbuf, "yes" if self._ffmpeg else "no")

    def add(self, frame: np.ndarray):
        if not self._w:
            self._h, self._w = frame.shape[:2]
        self._buf.append(frame.copy())
        if len(self._buf) >= self._maxbuf:
            self._flush()
            self._buf.clear()

    def _flush(self):
        if not self._buf or not self._w:
            return

        tmp_raw = self._out + ".raw.mp4"

        # Write raw mp4v first
        wrt = cv2.VideoWriter(
            tmp_raw,
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps, (self._w, self._h)
        )
        if not wrt.isOpened():
            log.warning("ClipWriter: VideoWriter failed")
            return
        for f in self._buf:
            wrt.write(f)
        wrt.release()

        final = tmp_raw

        # Re-encode to H.264 with ffmpeg for browser compatibility
        if self._ffmpeg:
            tmp_h264 = self._out + ".h264.mp4"
            ret = subprocess.run([
                self._ffmpeg, "-y", "-loglevel", "error",
                "-i", tmp_raw,
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "28",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                tmp_h264,
            ], capture_output=True)

            if ret.returncode == 0 and os.path.exists(tmp_h264):
                final = tmp_h264
                try:
                    os.remove(tmp_raw)
                except OSError:
                    pass
            else:
                log.warning("ffmpeg failed: %s",
                            ret.stderr.decode()[:200])

        # Atomic replace
        try:
            if os.path.exists(self._out):
                os.remove(self._out)
            os.rename(final, self._out)
        except OSError as e:
            log.warning("ClipWriter rename: %s", e)

        # Clean leftovers
        for p in [tmp_raw, self._out + ".raw.mp4",
                  self._out + ".h264.mp4"]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    def finalize(self):
        if self._buf:
            self._flush()
            self._buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# CV thread
# ══════════════════════════════════════════════════════════════════════════════

def cv_thread(video_path, model_path, conf, skip, conn, do_preprocess=True, use_kafka=False):
    from ultralytics import YOLO
    from cv_service.motion_analyzer import ActivityLSTM
    import torch

    try:
        log.info("Loading YOLO: %s", model_path)
        yolo = YOLO(model_path)

        # Load LSTM for excavator activity classification
        lstm     = ActivityLSTM()
        lstm.eval()
        ckpt     = Path("best_lstm.pth")
        use_lstm = False
        if ckpt.exists():
            lstm.load_state_dict(
                torch.load(str(ckpt), map_location="cpu"))
            use_lstm = True
            log.info("LSTM loaded — excavator activity: DIGGING/SWINGING/DUMPING/WAITING")
        else:
            log.info("No LSTM — excavator falls back to rule-based")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            update_status(conn, status="error",
                          error=f"Cannot open: {video_path}")
            return

        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        eff_fps = fps / max(skip, 1)

        update_status(conn, status="processing",
                      total_frames=total_f, fps=fps)
        log.info("Video %.0f fps | %d frames | eff_fps=%.1f",
                 fps, total_f, eff_fps)

        # Kafka producer — graceful fallback if unavailable
        producer = EquipmentProducer() if use_kafka else None
        if producer:
            log.info("Kafka streaming enabled")

        registry = MachineRegistry(eff_fps)
        writer   = ClipWriter(CLIP_PATH, CLIP_SECONDS, eff_fps)

        # Full annotated video writer
        # Writes entire processed video to output_full.mp4
        # Use this to pick the best segments for your demo
        cap_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_out = cv2.VideoWriter(
            FULL_VID_PATH,
            cv2.VideoWriter_fourcc(*"mp4v"),
            eff_fps,
            (cap_w, cap_h)
        )
        if full_out.isOpened():
            log.info("Full video writer: %s  (%dx%d @ %.0ffps)",
                     FULL_VID_PATH, cap_w, cap_h, eff_fps)
        else:
            log.warning("Full video writer failed to open — skipping")
            full_out = None
        batch:list = []
        idx        = 0
        scene_tick = max(1, int(eff_fps * 2))  # check scene every 2s

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % skip != 0:
                continue

            # Preprocess frame for better detection
            if do_preprocess:
                frame = preprocess_frame(frame)

            # Scene change check
            if idx % scene_tick == 0:
                sm = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    (160, 90)
                )
                registry.update_scene(sm)

            # Detect + track
            results = yolo.track(
                frame, conf=conf, persist=True, verbose=False)
            detections = []

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id   = int(box.cls[0].item())
                    eq_class = COCO_MAP.get(cls_id)
                    if not eq_class:
                        continue

                    byte_id = (
                        int(box.id[0].item())
                        if box.id is not None
                        else cls_id * 100000 + idx
                    )
                    x1, y1, x2, y2 = [int(v)
                                       for v in box.xyxy[0].tolist()]
                    bbox     = (x1, y1, x2, y2)
                    conf_val = float(box.conf[0].item())

                    eq_id, tracker = registry.resolve(
                        byte_id, eq_class, bbox, conf_val)
                    if eq_id is None:
                        continue  # still confirming

                    info = tracker.process_frame(
                        frame, bbox, lstm, use_lstm=use_lstm)
                    ta   = tracker.time_analytics()

                    row_data = {
                        "ts":               time.strftime(
                            "%Y-%m-%dT%H:%M:%S"),
                        "frame_id":         idx,
                        "equipment_id":     eq_id,
                        "equipment_class":  eq_class,
                        "current_state":    info.current_state.value,
                        "current_activity": info.current_activity.value,
                        "motion_source":    info.motion_source.value,
                        "confidence":       conf_val,
                        "total_tracked_sec":ta.total_tracked_seconds,
                        "total_active_sec": ta.total_active_seconds,
                        "total_idle_sec":   ta.total_idle_seconds,
                        "utilization_pct":  ta.utilization_percent,
                    }
                    batch.append(row_data)

                    # Stream to Kafka (exact required payload format)
                    if producer:
                        producer.send({
                            "frame_id":        row_data["frame_id"],
                            "equipment_id":    row_data["equipment_id"],
                            "equipment_class": row_data["equipment_class"],
                            "timestamp":       row_data["ts"],
                            "utilization": {
                                "current_state":    row_data["current_state"],
                                "current_activity": row_data["current_activity"],
                                "motion_source":    row_data["motion_source"],
                            },
                            "bbox": {
                                "x1": detections[-1]["x1"] if detections else 0,
                                "y1": detections[-1]["y1"] if detections else 0,
                                "x2": detections[-1]["x2"] if detections else 0,
                                "y2": detections[-1]["y2"] if detections else 0,
                                "confidence": row_data["confidence"],
                            },
                            "time_analytics": {
                                "total_tracked_seconds": row_data["total_tracked_sec"],
                                "total_active_seconds":  row_data["total_active_sec"],
                                "total_idle_seconds":    row_data["total_idle_sec"],
                                "utilization_percent":   row_data["utilization_pct"],
                            },
                        })

                    detections.append({
                        "eq_id":    eq_id,
                        "state":    info.current_state.value,
                        "activity": info.current_activity.value,
                        "util_pct": ta.utilization_percent,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })

            draw_annotations(frame, detections)
            writer.add(frame)
            if full_out:
                full_out.write(frame)

            # Flush DB every 30 rows
            if len(batch) >= 30:
                for row in batch:
                    conn.execute("""
                        INSERT INTO equipment_telemetry VALUES (
                            :ts,:frame_id,:equipment_id,:equipment_class,
                            :current_state,:current_activity,:motion_source,
                            :confidence,:total_tracked_sec,:total_active_sec,
                            :total_idle_sec,:utilization_pct)""", row)
                conn.commit()
                batch.clear()

            if idx % 60 == 0:
                real = registry.all_trackers()
                pct  = round(idx / max(total_f, 1) * 100, 1)
                update_status(conn, current_frame=idx,
                              progress_pct=pct, machines=len(real))
                log.info("Progress: %d/%d (%.0f%%) | %s",
                         idx, total_f, pct, list(real.keys()))

        # Finalize
        writer.finalize()
        if full_out:
            full_out.release()
            log.info("Full annotated video saved → %s", FULL_VID_PATH)
            # Re-encode to H.264 if ffmpeg available (better compatibility)
            import shutil
            if shutil.which("ffmpeg"):
                h264_out = FULL_VID_PATH.replace(".mp4", "_h264.mp4")
                import subprocess as _sp
                ret = _sp.run([
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-i", FULL_VID_PATH,
                    "-c:v", "libx264", "-preset", "fast",
                    "-crf", "23", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    h264_out,
                ], capture_output=True)
                if ret.returncode == 0:
                    import os as _os
                    _os.replace(h264_out, FULL_VID_PATH)
                    log.info("Re-encoded to H.264: %s", FULL_VID_PATH)
        if producer:
            producer.flush()
            producer.close()
        for row in batch:
            conn.execute("""
                INSERT INTO equipment_telemetry VALUES (
                    :ts,:frame_id,:equipment_id,:equipment_class,
                    :current_state,:current_activity,:motion_source,
                    :confidence,:total_tracked_sec,:total_active_sec,
                    :total_idle_sec,:utilization_pct)""", row)
        conn.commit()
        cap.release()

        real = registry.all_trackers()
        update_status(conn, status="done",
                      progress_pct=100, current_frame=total_f)
        log.info("Done. Machines: %s", list(real.keys()))
        for eid, t in real.items():
            ta = t.time_analytics()
            log.info("  %s  active=%.1fs  idle=%.1fs  util=%.1f%%",
                     eid, ta.total_active_seconds,
                     ta.total_idle_seconds, ta.utilization_percent)

    except Exception as e:
        log.exception("CV thread crashed: %s", e)
        update_status(conn, status="error", error=str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video",  default="data/input.mp4")
    p.add_argument("--model",  default="model/weights/best_model.pt")
    p.add_argument("--conf",   type=float, default=0.50)
    p.add_argument("--skip",   type=int,   default=1)
    p.add_argument("--fresh",      action="store_true")
    p.add_argument("--use-kafka",     action="store_true",
                   help="Stream results to Kafka")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Skip frame preprocessing")
    a = p.parse_args()

    if a.fresh and Path(DB_PATH).exists():
        Path(DB_PATH).unlink()
        log.info("DB cleared.")

    conn = init_db()
    update_status(conn, status="starting", progress_pct=0, total_frames=0)

    log.info("Launching dashboard → http://localhost:8501")
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         "streamlit_ui/app_local.py",
         "--server.port=8501", "--server.headless=true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    t = threading.Thread(
        target=cv_thread,
        args=(a.video, a.model, a.conf, a.skip, conn,
              not a.no_preprocess, a.use_kafka),
        daemon=True,
    )
    t.start()
    log.info("Dashboard: http://localhost:8501  |  Ctrl+C to stop.")
    try:
        t.join()
        log.info("Complete. Dashboard still running.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Stopped.")


if __name__ == "__main__":
    main()