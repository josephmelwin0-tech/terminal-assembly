"""
Trains a YOLO State Classification model on terminal assembly stages.
Optimized for high-accuracy industrial inspection:
  - Input resolution 384x384 (fine wire, clamp and block detail)
  - Color, lighting, and perspective invariance
  - Uses local NVIDIA RTX 4050 Laptop GPU (CUDA:0)

Usage:
    python train_classifier.py
    python train_classifier.py --epochs 35 --batch 16
"""

import os
import argparse
import torch
from ultralytics import YOLO

from terminal_config import PROJECT_ROOT, CLS_DATASET_DIR


def main():
    parser = argparse.ArgumentParser(description="Train YOLO State Classifier on Terminal Assembly")
    parser.add_argument("--dataset", type=str, default=CLS_DATASET_DIR, help="Path to classification dataset")
    parser.add_argument("--model", type=str, default="yolo11s-cls.pt", help="Pretrained weights")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=384, help="Input resolution (default: 384)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--name", type=str, default="terminal_states_cls", help="Run folder name")
    args = parser.parse_args()

    dataset_path = os.path.abspath(args.dataset).replace("\\", "/")
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory '{dataset_path}' not found.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if device == 0 else "CPU"
    print("=" * 72)
    print(" TERMINAL ASSEMBLY STATE CLASSIFIER TRAINING")
    print(f" Dataset:     {dataset_path}")
    print(f" Device:      {device_name} (CUDA={torch.cuda.is_available()})")
    print(f" Base Model:  {args.model}")
    print(f" Epochs:      {args.epochs}")
    print(f" Image Size:  {args.imgsz}")
    print(f" Batch Size:  {args.batch}")
    print("=" * 72)

    model = YOLO(args.model)
    results = model.train(
        data=dataset_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=os.path.join(PROJECT_ROOT, "runs_classify"),
        name=args.name,
        exist_ok=True,
        workers=4,
        # Industrial augmentations
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=12.0,
        translate=0.08,
        scale=0.15,
        shear=2.0,
        fliplr=0.0,    # DO NOT horizontal-flip terminal order!
        flipud=0.0,
        mosaic=0.0,
        erasing=0.15,
        patience=15,
        save=True,
        save_period=5,
        plots=True,
        verbose=True,
    )

    best_pt = os.path.join(PROJECT_ROOT, "runs_classify", args.name, "weights", "best.pt")
    print("\n" + "=" * 72)
    print(" TRAINING COMPLETE")
    print(f" Best Weights Saved to: {best_pt}")
    print("=" * 72)


if __name__ == "__main__":
    main()
