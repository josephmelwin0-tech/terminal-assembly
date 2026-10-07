"""
Trains a YOLO Component Detector on terminal assembly components:
  0: din_rail
  1: terminal_block
  2: closer
  3: end_clamp
  4: blue_wire
  5: red_wire
  6: yellow_wire

Usage:
    python train_detector.py
    python train_detector.py --epochs 25 --batch 16
"""

import os
import argparse
import torch
from ultralytics import YOLO

from terminal_config import PROJECT_ROOT, DET_DATASET_DIR


def main():
    parser = argparse.ArgumentParser(description="Train YOLO Component Detector for Terminal Assembly")
    parser.add_argument("--data", type=str, default=os.path.join(DET_DATASET_DIR, "data.yaml"), help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--imgsz", type=int, default=512, help="Input resolution (default: 512)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--name", type=str, default="terminal_detector", help="Run folder name")
    args = parser.parse_args()

    data_yaml = os.path.abspath(args.data).replace("\\", "/")
    if not os.path.exists(data_yaml):
        print(f"[ERROR] data.yaml not found: {data_yaml}")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print("=" * 72)
    print(" TERMINAL COMPONENT DETECTOR TRAINING")
    print(f" Dataset:     {data_yaml}")
    print(f" Device:      {device_name} (CUDA={torch.cuda.is_available()})")
    print(f" Base Model:  {args.model}")
    print(f" Epochs:      {args.epochs}")
    print(f" Image Size:  {args.imgsz}")
    print(f" Batch Size:  {args.batch}")
    print("=" * 72)

    model = YOLO(args.model)
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=os.path.join(PROJECT_ROOT, "runs_detect"),
        name=args.name,
        exist_ok=True,
        workers=4,
        # Augmentations for industrial robustness
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=8.0,
        translate=0.08,
        scale=0.15,
        shear=1.5,
        fliplr=0.0,    # DO NOT flip horizontally (preserves TB1..TB6 terminal ordering)
        flipud=0.0,
        mosaic=0.5,
        erasing=0.10,
        patience=15,
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
    )

    best_pt = os.path.join(PROJECT_ROOT, "runs_detect", args.name, "weights", "best.pt")
    print("\n" + "=" * 72)
    print(" TRAINING COMPLETE")
    print(f" Best Weights Saved to: {best_pt}")
    print("=" * 72)


if __name__ == "__main__":
    main()
