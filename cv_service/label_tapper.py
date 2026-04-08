"""
cv_service/label_tapper.py
─────────────────────────────────────────────────────────────
Speed-labeling tool — Excavator, 4 classes.

HOW TO USE:
  1. Open your video in VLC
  2. Run this script:  python cv_service/label_tapper.py
  3. Press SPACE here AND in VLC at the same time (sync)
  4. Hold a key while the activity is happening:
         D = DIGGING  (arm/bucket into ground)
         S = SWINGING (whole body rotating)
         Q = DUMPING  (bucket emptying into truck)
         W = WAITING  (machine idle/still)
  5. Release the key when the activity ends
  6. Press R immediately after a mistake to reject it
  7. Press ESC when done → saves data/labels.csv

TIPS:
  - VLC at 0.75x speed helps with precise transitions
  - Leave gaps between labels — unlabeled frames are simply skipped
  - The full cycle is: DIGGING → SWINGING → DUMPING → SWINGING → repeat
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

try:
    import keyboard
except ImportError:
    print("Installing keyboard...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "keyboard"])
    import keyboard

KEY_MAP = {
    "d": "DIGGING",
    "s": "SWINGING",
    "q": "DUMPING",
    "w": "WAITING",
    "r": "REJECT",
}

COLORS = {
    "DIGGING":  "\033[94m",
    "SWINGING": "\033[95m",
    "DUMPING":  "\033[93m",
    "WAITING":  "\033[90m",
    "REJECT":   "\033[91m",
    "RESET":    "\033[0m",
}


def c(text, label):
    return f"{COLORS.get(label, '')}{text}{COLORS['RESET']}"


def fmt(sec):
    m = int(sec // 60)
    s = sec % 60
    return f"{m:02d}:{s:06.3f}"


def main():
    out_path = Path("data/labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*54)
    print("  EXCAVATOR LABEL TAPPER  —  4 classes")
    print("="*54)
    print(f"\n  [D]  {c('DIGGING',  'DIGGING')}  — arm/bucket into ground")
    print(f"  [S]  {c('SWINGING', 'SWINGING')} — whole body rotating")
    print(f"  [Q]  {c('DUMPING',  'DUMPING')}  — bucket emptying into truck")
    print(f"  [W]  {c('WAITING',  'WAITING')}  — machine idle/still")
    print(f"  [R]  REJECT   — mark bad segment")
    print(f"\n  [SPACE] Start / pause timer")
    print(f"  [ESC]   Save and quit")
    print("\n" + "="*54)
    print("\n  Open VLC, then press SPACE here + in VLC together\n")

    labels        = []
    timer_running = False
    elapsed       = 0.0
    timer_start   = None
    current_label = None
    label_start   = None

    def now():
        if timer_running and timer_start:
            return elapsed + (time.time() - timer_start)
        return elapsed

    def finish():
        nonlocal current_label, label_start
        if current_label is None:
            return
        end = now()
        dur = end - label_start
        if dur >= 0.3:
            labels.append({
                "start_sec": round(label_start, 3),
                "end_sec":   round(end, 3),
                "activity":  current_label,
                "duration":  round(dur, 3),
            })
            tag = "SAVED" if current_label != "REJECT" else "REJECTED"
            print(
                f"  {c(current_label, current_label):<22}"
                f"  {fmt(label_start)} → {fmt(end)}"
                f"  ({dur:.2f}s)  [{tag}]"
            )
        else:
            print(f"  [too short {dur:.3f}s — ignored]")
        current_label = None
        label_start   = None

    def on_space():
        nonlocal timer_running, elapsed, timer_start
        if timer_running:
            elapsed += time.time() - timer_start
            timer_start   = None
            timer_running = False
            print(f"\n  ⏸  Paused  @ {fmt(elapsed)}\n")
        else:
            timer_start   = time.time()
            timer_running = True
            print(f"\n  ▶  Running @ {fmt(elapsed)}\n")

    keyboard.on_press_key("space", lambda _: on_space())

    def press(label):
        def h(_):
            nonlocal current_label, label_start
            if not timer_running:
                print("  [not running — press SPACE first]")
                return
            if current_label == label:
                return
            finish()
            current_label = label
            label_start   = now()
        return h

    def release(label):
        def h(_):
            nonlocal current_label
            if current_label == label:
                finish()
        return h

    for key, label in KEY_MAP.items():
        keyboard.on_press_key(key,   press(label))
        keyboard.on_release_key(key, release(label))

    print(f"  {'Activity':<22}  {'Start':>9} → {'End':>9}  {'Dur':>6}  Status")
    print("  " + "-"*58)

    keyboard.wait("esc")
    finish()

    valid    = [l for l in labels if l["activity"] != "REJECT"]
    rejected = len(labels) - len(valid)

    print("\n" + "="*54)
    print(f"  Done — {len(valid)} segments saved, {rejected} rejected")
    print()

    from collections import Counter
    dist = Counter(l["activity"] for l in valid)
    for act in ["DIGGING", "SWINGING", "DUMPING", "WAITING"]:
        count     = dist.get(act, 0)
        total_sec = sum(l["duration"] for l in valid if l["activity"] == act)
        bar       = "█" * min(int(total_sec / 5), 20)
        warn      = "" if total_sec >= 60 else "  ⚠ aim for 60s+"
        print(f"  {c(act, act):<22} {count:>3} segs  "
              f"{total_sec:>6.1f}s  {bar}{warn}")

    if not valid:
        print("\n  No labels — nothing saved.")
        return

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["start_sec", "end_sec", "activity", "duration"]
        )
        w.writeheader()
        w.writerows(valid)

    print(f"\n  Saved → {out_path}")
    print("="*54)
    print("\n  Next:")
    print("  python cv_service/extract_features.py \\")
    print("      --video  data/input.mp4 \\")
    print("      --labels data/labels.csv \\")
    print("      --model  model/weights/best_model.pt\n")


if __name__ == "__main__":
    main()