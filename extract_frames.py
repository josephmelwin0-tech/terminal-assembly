"""
Frame extraction script for DIN Rail Terminal Block Assembly videos.
Extracts clean, motion-filtered frames from all videos in data/individual_parts/ and data/states/
with Laplacian blur filtering and temporal 80/20 train/val partitioning.

Usage:
    python extract_frames.py
    python extract_frames.py --fps 4.0 --blur_thresh 20.0
"""

import os
import argparse
import cv2
import numpy as np

from terminal_config import (
    PARTS_VIDEO_DIR,
    STATES_VIDEO_DIR,
    EXTRACTED_DIR,
    PARTS_VIDEO_MAPPING,
    STATES_VIDEO_MAPPING,
)


def is_blurry(gray_frame, threshold=20.0):
    """Calculates Laplacian variance. Low variance indicates motion blur."""
    if threshold <= 0:
        return False
    var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return var < threshold


def clean_existing_frames(target_dir):
    """Removes previously extracted vid_* frames from target directory."""
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        return 0

    removed = 0
    for f in os.listdir(target_dir):
        if f.startswith("vid_") and f.lower().endswith((".jpg", ".jpeg", ".png")):
            os.remove(os.path.join(target_dir, f))
            removed += 1
    return removed


def extract_from_video(
    video_path,
    target_category,
    is_part_video=False,
    fps=4.0,
    blur_thresh=20.0,
    val_split=0.20,
    clean_old=True,
):
    """
    Extracts frames from a single video, filtering blur and applying temporal train/val split.
    """
    if not os.path.exists(video_path):
        print(f"  [ERROR] Video file not found: {video_path}")
        return None

    subfolder = "parts" if is_part_video else "states"
    out_dir = os.path.join(EXTRACTED_DIR, subfolder, target_category)
    os.makedirs(out_dir, exist_ok=True)

    if clean_old:
        num_cleaned = clean_existing_frames(out_dir)
        if num_cleaned > 0:
            print(f"  [INFO] Cleaned {num_cleaned} old video frames from '{out_dir}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Could not open video: {video_path}")
        return None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps if total_frames > 0 else 0

    frame_interval = max(1, int(round(video_fps / fps)))

    print(f"\n  Processing: {os.path.basename(video_path)}")
    print(f"    Target: '{target_category}' ({'Part' if is_part_video else 'State'})")
    print(f"    Source: {total_frames} frames @ {video_fps:.1f} fps ({duration_sec:.1f}s)")
    print(f"    Extracting every {frame_interval} frames (~{video_fps/frame_interval:.1f} fps)")

    # Read and buffer candidates
    candidates = []
    f_idx = 0
    skipped_blur = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if f_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if is_blurry(gray, blur_thresh):
                skipped_blur += 1
            else:
                candidates.append((f_idx, frame))

        f_idx += 1

    cap.release()

    total_kept = len(candidates)
    print(f"    Kept {total_kept} clean frames (Filtered {skipped_blur} blurry frames)")

    if total_kept == 0:
        print("    [WARN] No clean frames passed filters!")
        return {"category": target_category, "train": 0, "val": 0, "total": 0}

    # Temporal split: middle 80% train, last 20% val
    split_idx = int(total_kept * (1.0 - val_split))

    saved_train = 0
    saved_val = 0

    for i, (orig_fidx, frame) in enumerate(candidates):
        is_val = i >= split_idx
        prefix = "vid_val" if is_val else "vid_train"
        filename = f"{prefix}_{orig_fidx:06d}.jpg"
        save_path = os.path.join(out_dir, filename)

        cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if is_val:
            saved_val += 1
        else:
            saved_train += 1

    print(f"    Saved -> Train: {saved_train} | Val: {saved_val} | Total: {saved_train + saved_val}")
    return {"category": target_category, "train": saved_train, "val": saved_val, "total": total_kept}


def main():
    parser = argparse.ArgumentParser(description="Extract clean frames from terminal assembly videos.")
    parser.add_argument("--fps", type=float, default=3.5, help="Extraction frame rate (default: 3.5)")
    parser.add_argument("--blur_thresh", type=float, default=18.0, help="Laplacian variance blur threshold")
    parser.add_argument("--val_split", type=float, default=0.20, help="Fraction for validation split (default: 0.20)")
    args = parser.parse_args()

    print("=" * 72)
    print(" DIN RAIL TERMINAL BLOCK ASSEMBLY - DATASET EXTRACTION")
    print("=" * 72)

    stats = []

    # 1. Process Individual Part Videos
    print("\n--- EXTRACTING INDIVIDUAL PART VIDEOS ---")
    for vname, info in PARTS_VIDEO_MAPPING.items():
        vpath = os.path.join(PARTS_VIDEO_DIR, vname)
        res = extract_from_video(
            vpath,
            info["target"],
            is_part_video=True,
            fps=args.fps,
            blur_thresh=args.blur_thresh,
            val_split=args.val_split,
        )
        if res:
            stats.append(res)

    # 2. Process Assembly State Videos
    print("\n--- EXTRACTING ASSEMBLY STATE VIDEOS ---")
    for vname, info in STATES_VIDEO_MAPPING.items():
        vpath = os.path.join(STATES_VIDEO_DIR, vname)
        res = extract_from_video(
            vpath,
            info["target"],
            is_part_video=False,
            fps=args.fps,
            blur_thresh=args.blur_thresh,
            val_split=args.val_split,
        )
        if res:
            stats.append(res)

    print("\n" + "=" * 72)
    print(" EXTRACTION SUMMARY")
    print("=" * 72)
    tot_tr = sum(s["train"] for s in stats)
    tot_va = sum(s["val"] for s in stats)
    for s in stats:
        print(f"  {s['category']:<28} : Train={s['train']:>4} | Val={s['val']:>4} | Total={s['total']:>4}")
    print("-" * 72)
    print(f"  TOTAL DATASET FRAMES        : Train={tot_tr:>4} | Val={tot_va:>4} | Total={tot_tr + tot_va:>4}")
    print("=" * 72)


if __name__ == "__main__":
    main()
