"""
cv_service/extract_features.py
─────────────────────────────────────────────────────────────
Feature extraction for excavator-only LSTM training.

"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent))
from cv_service.motion_analyzer import (
    _make_mog2, _motion_score, _split_regions, _clamp_bbox
)

# Excavator only — 4 classes
ACTIVITY_TO_IDX = {
    "DIGGING":  0,
    "SWINGING": 1,
    "DUMPING":  2,
    "WAITING":  3,
}
CLASS_NAMES   = ["DIGGING", "SWINGING", "DUMPING", "WAITING"]
EXCAVATOR_CLS = 3   # class ID in your trained model


def load_labels(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["activity"] != "REJECT"].copy()
    df["activity"] = df["activity"].str.upper().str.strip()
    df = df[df["activity"].isin(ACTIVITY_TO_IDX)].copy()
    df = df.sort_values("start_sec").reset_index(drop=True)

    print(f"\nLabels loaded from {path}:")
    total_labeled = 0.0
    for act in CLASS_NAMES:
        sub  = df[df["activity"] == act]
        secs = (sub["end_sec"] - sub["start_sec"]).sum()
        total_labeled += secs
        warn = "" if secs >= 60 else "  ⚠ aim for 60s+"
        print(f"  {act:<12} {len(sub):>3} segments  {secs:.1f}s{warn}")
    print(f"  Total labeled: {total_labeled:.1f}s")
    return df


def label_at(df: pd.DataFrame, t: float):
    """Return label index at time t, or None if gap/unlabeled."""
    row = df[(df["start_sec"] <= t) & (df["end_sec"] > t)]
    if row.empty:
        return None
    return ACTIVITY_TO_IDX.get(row.iloc[0]["activity"])


def extract(video_path, labels_path, model_path, conf=0.40, out_path="data/features/features.pt"):
    from ultralytics import YOLO

    print(f"\nVideo : {video_path}")
    print(f"Model : {model_path}")

    df = load_labels(labels_path)

    yolo    = YOLO(model_path)
    cap     = cv2.VideoCapture(video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nProcessing: {total_f} frames @ {fps:.0f}fps "
          f"({total_f/fps/60:.1f} min)")
    print("This may take a few minutes...\n")

    sub_arm  = _make_mog2()
    sub_body = _make_mog2()

    all_features = []
    all_labels   = []
    stats        = {"warmup": 0, "unlabeled": 0, "no_det": 0, "saved": 0}
    frame_idx    = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        t = frame_idx / fps

        # Warm-up: MOG2 needs ~30 frames to learn background
        if frame_idx <= 30:
            stats["warmup"] += 1
            continue

        lbl = label_at(df, t)

        # Detect excavator in this frame
        results  = yolo(frame, conf=conf, verbose=False)
        best_box = None
        best_c   = 0.0

        if results[0].boxes:
            for box in results[0].boxes:
                # Only consider excavator class
                if int(box.cls[0].item()) == EXCAVATOR_CLS:
                    c_val = float(box.conf[0].item())
                    if c_val > best_c:
                        best_c, best_box = c_val, box

        # Always update MOG2 when excavator is detected
        # (keeps background model warm even on unlabeled frames)
        if best_box is not None:
            x1, y1, x2, y2 = _clamp_bbox(
                [int(v) for v in best_box.xyxy[0].tolist()], frame.shape
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                arm_crop, body_crop = _split_regions(crop, "excavator")
                arm_score  = _motion_score(arm_crop,  sub_arm)
                body_score = _motion_score(body_crop, sub_body)

                # Only save if this frame has a label
                if lbl is not None:
                    h, w   = crop.shape[:2]
                    aspect = w / max(h, 1)
                    area_n = (w * h) / (frame.shape[0]*frame.shape[1] + 1e-6)
                    arm_rel  = arm_score  / (arm_score + body_score + 1e-6)
                    body_rel = body_score / (arm_score + body_score + 1e-6)

                    all_features.append(
                        [arm_score, body_score, arm_rel, body_rel, aspect, area_n]
                    )
                    all_labels.append(lbl)
                    stats["saved"] += 1
                else:
                    stats["unlabeled"] += 1
            else:
                if lbl is not None:
                    stats["no_det"] += 1
        else:
            if lbl is not None:
                stats["no_det"] += 1
            else:
                stats["unlabeled"] += 1

        if frame_idx % 300 == 0:
            pct = frame_idx / total_f * 100
            print(f"  {pct:>5.1f}%  |  saved={stats['saved']}  "
                  f"unlabeled={stats['unlabeled']}  no_det={stats['no_det']}")

    cap.release()

    print(f"\n{'='*45}")
    print(f"Extraction complete")
    print(f"  Saved frames : {stats['saved']}")
    print(f"  Unlabeled    : {stats['unlabeled']} (gaps — normal)")
    print(f"  No detection : {stats['no_det']}")

    if stats["saved"] < 100:
        print(f"\n⚠ Only {stats['saved']} labeled frames.")
        print("  Label more segments with label_tapper.py and retry.")
        return

    ft = torch.tensor(np.array(all_features, dtype=np.float32))
    lt = torch.tensor(np.array(all_labels,   dtype=np.int64))

    print(f"\nClass distribution:")
    unique, counts = torch.unique(lt, return_counts=True)
    for u, c in zip(unique.tolist(), counts.tolist()):
        bar = "█" * int(c / max(counts).item() * 20)
        print(f"  {CLASS_NAMES[u]:<12} {c:>5} frames  {bar}")

    min_c = counts.min().item()
    if min_c < 200:
        print(f"\n⚠ Smallest class has {min_c} frames — label more if possible")

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features":    ft,
        "labels":      lt,
        "class_names": CLASS_NAMES,
        "video":       str(video_path),
        "fps":         fps,
        "num_classes": len(CLASS_NAMES),
    }, str(out))

    size_kb = out.stat().st_size / 1024
    print(f"\nSaved → {out}  ({size_kb:.0f} KB)")
    print("\nNext steps:")
    print("  1. Upload features.pt to Google Colab")
    print("  2. Run cv_service/train_lstm_colab.py")
    print("  3. Download best_lstm.pth")
    print("  4. Place best_lstm.pth in project root")
    print("  5. Set use_lstm=True in run_local.py")


def main():
    p = argparse.ArgumentParser(
        description="Extract LSTM features from labelled excavator video"
    )
    p.add_argument("--video",  default="data/input.mp4")
    p.add_argument("--labels", default="data/labels.csv")
    p.add_argument("--model",  default="model/weights/best_model.pt")
    p.add_argument("--out",    default="data/features/features.pt")
    p.add_argument("--conf",   type=float, default=0.40)
    a = p.parse_args()
    extract(a.video, a.labels, a.model, a.conf, a.out)


if __name__ == "__main__":
    main()
