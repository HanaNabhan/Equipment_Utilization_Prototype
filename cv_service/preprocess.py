"""
cv_service/preprocess.py
─────────────────────────────────────────────────────────────
Input video preprocessing pipeline.

Applies before YOLO inference to improve detection quality:
  1. Resolution normalization  — resize to 1280px wide max
  2. Contrast enhancement      — CLAHE on luminance channel
  3. Denoising                 — light Gaussian blur for noisy cameras
  4. Stabilization check       — warn if frame is blurry (motion blur)

Used automatically by run_local.py on every frame before detection.
Can be disabled with --no-preprocess flag.

Standalone usage (inspect what preprocessing does to your video):
    python cv_service/preprocess.py --video data/input.mp4 --frames 5
"""
from __future__ import annotations

import argparse
import cv2
import numpy as np


# ── Configuration ──────────────────────────────────────────────────────────────
MAX_WIDTH    = 1280   # max frame width — larger frames slow YOLO
CLAHE_CLIP   = 2.0    # CLAHE clip limit (higher = more contrast)
CLAHE_GRID   = (8, 8) # CLAHE tile grid size
BLUR_SIGMA   = 0.5    # Gaussian blur sigma (0 = disabled)
BLUR_THRESH  = 80.0   # Laplacian variance below this = blurry frame warning


def preprocess_frame(
    frame:         np.ndarray,
    max_width:     int   = MAX_WIDTH,
    enhance:       bool  = True,
    denoise:       bool  = True,
) -> np.ndarray:
    """
    Apply preprocessing pipeline to a single BGR frame.

    Args:
        frame:     Input BGR frame from cv2.VideoCapture
        max_width: Resize if frame wider than this
        enhance:   Apply CLAHE contrast enhancement
        denoise:   Apply light Gaussian blur

    Returns:
        Preprocessed BGR frame, same dtype as input
    """
    if frame is None or frame.size == 0:
        return frame

    # ── 1. Resize if too large ─────────────────────────────────────────────
    h, w = frame.shape[:2]
    if w > max_width:
        scale  = max_width / w
        new_w  = max_width
        new_h  = int(h * scale)
        frame  = cv2.resize(frame, (new_w, new_h),
                            interpolation=cv2.INTER_LINEAR)

    # ── 2. CLAHE contrast enhancement ─────────────────────────────────────
    # Improves detection in dark/shadowed construction site footage
    # Applied on luminance channel only to preserve color balance
    if enhance:
        lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit    = CLAHE_CLIP,
            tileGridSize = CLAHE_GRID
        )
        l     = clahe.apply(l)
        lab   = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── 3. Light denoising ─────────────────────────────────────────────────
    # Reduces compression artifacts from site cameras
    # Very light sigma — we don't want to blur out details
    if denoise and BLUR_SIGMA > 0:
        frame = cv2.GaussianBlur(frame, (0, 0), BLUR_SIGMA)

    return frame


def is_blurry(frame: np.ndarray, threshold: float = BLUR_THRESH) -> bool:
    """
    Check if a frame is too blurry to get reliable detections.
    Uses Laplacian variance — low variance = blurry.

    Returns True if frame is likely too blurry for good detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var()) < threshold


# ── Standalone inspection ──────────────────────────────────────────────────────

def _inspect(video_path: str, n_frames: int = 5):
    """
    Save N frames before/after preprocessing for visual inspection.
    Output: data/debug_preprocess/frame_N_{raw,processed}.jpg
    """
    import os
    out_dir = "data/debug_preprocess"
    os.makedirs(out_dir, exist_ok=True)

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = max(1, total // (n_frames + 1))

    print(f"Video: {video_path}")
    print(f"Total frames: {total}")
    print(f"Saving {n_frames} frame pairs to {out_dir}/\n")

    for i in range(n_frames):
        frame_no = step * (i + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess_frame(frame.copy())
        blurry    = is_blurry(frame)

        raw_path  = f"{out_dir}/frame_{i+1:02d}_raw.jpg"
        proc_path = f"{out_dir}/frame_{i+1:02d}_processed.jpg"
        cv2.imwrite(raw_path,  frame)
        cv2.imwrite(proc_path, processed)

        h, w = frame.shape[:2]
        ph, pw = processed.shape[:2]
        print(f"Frame {frame_no:>6} | raw={w}x{h} → proc={pw}x{ph} "
              f"| blurry={'YES ⚠' if blurry else 'no'} "
              f"| saved {raw_path}")

    cap.release()
    print(f"\nDone. Open {out_dir}/ to compare raw vs processed frames.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Inspect video preprocessing output")
    p.add_argument("--video",  required=True)
    p.add_argument("--frames", type=int, default=5)
    a = p.parse_args()
    _inspect(a.video, a.frames)
