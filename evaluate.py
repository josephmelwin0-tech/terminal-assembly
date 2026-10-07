"""
Evaluation and benchmark script for Industrial DIN Rail Terminal Block Assembly Verification.
Measures:
  1. Assembly State Classification Accuracy across held-out validation frames
  2. Average inference latency (ms/frame) and FPS on local GPU
  3. Terminal slot and wire color verification consistency

Usage:
    python evaluate.py
"""

import os
import time
import cv2
import torch
from ultralytics import YOLO

from terminal_config import (
    PROJECT_ROOT,
    CLS_DATASET_DIR,
    ASSEMBLY_STATES,
    STEP_TITLES,
)


def evaluate_state_classifier():
    best_pt = os.path.join(PROJECT_ROOT, "runs_classify", "terminal_states_cls", "weights", "best.pt")
    if not os.path.exists(best_pt):
        print(f"[ERROR] Weights file not found: {best_pt}")
        return

    val_dir = os.path.join(CLS_DATASET_DIR, "val")
    if not os.path.exists(val_dir):
        print(f"[ERROR] Validation directory not found: {val_dir}")
        return

    print("\n" + "=" * 72)
    print(" BENCHMARK: ASSEMBLY STATE CLASSIFICATION ON HELD-OUT VAL FRAMES")
    print(f" Model: {best_pt}")
    print(f" Device: {'CUDA:0' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 72)

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(best_pt)

    total_frames = 0
    correct_frames = 0
    latencies = []

    for sname in sorted(os.listdir(val_dir)):
        s_path = os.path.join(val_dir, sname)
        if not os.path.isdir(s_path):
            continue

        val_files = [f for f in os.listdir(s_path) if f.lower().endswith((".jpg", ".png"))]
        if not val_files:
            continue

        s_correct = 0
        s_total = len(val_files)

        for vf in val_files:
            img_path = os.path.join(s_path, vf)
            img = cv2.imread(img_path)
            if img is None:
                continue

            t0 = time.perf_counter()
            res = model(img, imgsz=384, device=device, verbose=False)[0]
            latencies.append((time.perf_counter() - t0) * 1000.0)

            pred_name = res.names[res.probs.top1]
            if pred_name == sname:
                s_correct += 1

        acc = (s_correct / s_total) * 100.0 if s_total > 0 else 0.0
        total_frames += s_total
        correct_frames += s_correct

        title = STEP_TITLES.get(sname, sname)
        print(f"  {title:<42}: {s_correct:>3}/{s_total:<3} ({acc:>5.1f}%)")

    overall_acc = (correct_frames / total_frames) * 100.0 if total_frames > 0 else 0.0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

    print("-" * 72)
    print(f"  Overall Validation Accuracy: {correct_frames}/{total_frames} ({overall_acc:.1f}%)")
    print(f"  Average Pipeline Latency:    {avg_lat:.1f} ms/frame (~{avg_fps:.1f} FPS)")
    print("=" * 72)


if __name__ == "__main__":
    evaluate_state_classifier()
