"""
Dataset preparation for Assembly State Classifier.
Organizes extracted state frames into standard YOLO classification structure:
    yolo_dataset_cls/
        train/<state_name>/*.jpg
        val/<state_name>/*.jpg
"""

import os
import shutil
import cv2
from terminal_config import EXTRACTED_DIR, CLS_DATASET_DIR, ASSEMBLY_STATES


def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def main():
    states_dir = os.path.join(EXTRACTED_DIR, "states")
    if not os.path.exists(states_dir):
        print(f"[ERROR] '{states_dir}' not found. Run extract_frames.py first.")
        return

    print("=" * 72)
    print(" PREPARING ASSEMBLY STATE CLASSIFICATION DATASET")
    print(f" Target Directory: {CLS_DATASET_DIR}")
    print("=" * 72)

    clear_and_make(CLS_DATASET_DIR)
    for split in ["train", "val"]:
        for sname in ASSEMBLY_STATES:
            if sname == "state_0_unstarted":
                continue
            os.makedirs(os.path.join(CLS_DATASET_DIR, split, sname), exist_ok=True)

    summary = {}

    for folder_name in sorted(os.listdir(states_dir)):
        src_dir = os.path.join(states_dir, folder_name)
        if not os.path.isdir(src_dir):
            continue

        canonical_name = folder_name
        train_dst = os.path.join(CLS_DATASET_DIR, "train", canonical_name)
        val_dst = os.path.join(CLS_DATASET_DIR, "val", canonical_name)

        n_train = 0
        n_val = 0

        for fname in sorted(os.listdir(src_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            split = "val" if "vid_val_" in fname else "train"
            dst_dir = val_dst if split == "val" else train_dst

            src_img_path = os.path.join(src_dir, fname)
            dst_img_path = os.path.join(dst_dir, fname)

            shutil.copy2(src_img_path, dst_img_path)
            if split == "train":
                n_train += 1
            else:
                n_val += 1

        summary[canonical_name] = (n_train, n_val)
        print(f"  {canonical_name:<28}: Train={n_train:>3} | Val={n_val:>3} | Total={n_train + n_val:>3}")

    total_train = sum(s[0] for s in summary.values())
    total_val = sum(s[1] for s in summary.values())
    print("-" * 72)
    print(f"  Total Processed: {total_train} train frames, {total_val} validation frames.")
    print("=" * 72)


if __name__ == "__main__":
    main()
