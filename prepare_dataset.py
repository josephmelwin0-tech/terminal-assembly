"""
Dataset preparation script for DIN Rail Terminal Block Assembly Quality Inspection.
Builds the YOLO Object Detection dataset (yolo_dataset_det/) with:
  images/train/, images/val/
  labels/train/, labels/val/
  data.yaml
Auto-annotates bounding boxes for:
  0: din_rail
  1: terminal_block
  2: closer
  3: end_clamp
  4: blue_wire
  5: red_wire
  6: yellow_wire
"""

import os
import shutil
import cv2
import yaml
import numpy as np

from terminal_config import (
    EXTRACTED_DIR,
    DET_DATASET_DIR,
    COMPONENT_CLASSES,
    PROJECT_ROOT,
)


def clear_and_make(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def convert_bbox_to_yolo(bbox, img_w, img_h):
    """Converts (x, y, w, h) in pixels to normalized YOLO format (cx, cy, w, h)."""
    bx, by, bw, bh = bbox
    cx = (bx + bw / 2.0) / float(img_w)
    cy = (by + bh / 2.0) / float(img_h)
    nw = bw / float(img_w)
    nh = bh / float(img_h)
    return max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)), max(0.005, min(1.0, nw)), max(0.005, min(1.0, nh))


def find_foreground_box(img, min_area=1500):
    """Finds primary object contour in isolated part frames."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    th = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > min_area]
    if not valid:
        return None
    valid.sort(key=cv2.contourArea, reverse=True)
    return cv2.boundingRect(valid[0])


def detect_wire_box_by_color(img, color_name):
    """Isolates tight bounding box of colored wire."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if color_name == "blue_wire":
        mask = cv2.inRange(hsv, np.array([95, 80, 40]), np.array([135, 255, 255]))
    elif color_name == "red_wire":
        m1 = cv2.inRange(hsv, np.array([0, 80, 50]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([168, 80, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
    elif color_name == "yellow_wire":
        mask = cv2.inRange(hsv, np.array([18, 80, 70]), np.array([38, 255, 255]))
    else:
        return None

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in cnts if cv2.contourArea(c) > 350]
    if not valid:
        return None
    all_pts = np.vstack(valid)
    return cv2.boundingRect(all_pts)


def prepare_detection_dataset():
    print("=" * 72)
    print(" PREPARING YOLO OBJECT DETECTION DATASET")
    print(f" Target Directory: {DET_DATASET_DIR}")
    print("=" * 72)

    clear_and_make(DET_DATASET_DIR)
    for split in ["train", "val"]:
        os.makedirs(os.path.join(DET_DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DET_DATASET_DIR, "labels", split), exist_ok=True)

    counts = {"train": 0, "val": 0}

    # 1. Annotate Individual Part Frames
    parts_dir = os.path.join(EXTRACTED_DIR, "parts")
    if os.path.exists(parts_dir):
        for cat in sorted(os.listdir(parts_dir)):
            if cat not in COMPONENT_CLASSES:
                continue
            cls_id = COMPONENT_CLASSES.index(cat)
            cat_dir = os.path.join(parts_dir, cat)

            for fname in sorted(os.listdir(cat_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                split = "val" if "vid_val_" in fname else "train"
                src_path = os.path.join(cat_dir, fname)
                img = cv2.imread(src_path)
                if img is None:
                    continue

                h, w = img.shape[:2]

                # Find bounding box
                if "wire" in cat:
                    bbox = detect_wire_box_by_color(img, cat) or find_foreground_box(img)
                else:
                    bbox = find_foreground_box(img)

                if bbox is None:
                    continue

                unique_name = f"part_{cat}_{fname}"
                dst_img = os.path.join(DET_DATASET_DIR, "images", split, unique_name)
                shutil.copy2(src_path, dst_img)

                label_name = os.path.splitext(unique_name)[0] + ".txt"
                dst_lbl = os.path.join(DET_DATASET_DIR, "labels", split, label_name)

                cx, cy, nw, nh = convert_bbox_to_yolo(bbox, w, h)
                with open(dst_lbl, "w") as f:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                counts[split] += 1

    # 2. Annotate Assembly State Frames (Multi-object: rail, blocks, clamps, wires)
    states_dir = os.path.join(EXTRACTED_DIR, "states")
    if os.path.exists(states_dir):
        for state_name in sorted(os.listdir(states_dir)):
            s_dir = os.path.join(states_dir, state_name)
            if not os.path.isdir(s_dir):
                continue

            for fname in sorted(os.listdir(s_dir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                split = "val" if "vid_val_" in fname else "train"
                src_path = os.path.join(s_dir, fname)
                img = cv2.imread(src_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                labels = []

                # Find DIN rail / block cluster
                cluster_box = find_foreground_box(img, min_area=3000)
                if cluster_box is not None:
                    gx, gy, gw, gh = cluster_box
                    # Add DIN rail label
                    cx, cy, nw, nh = convert_bbox_to_yolo(cluster_box, w, h)
                    labels.append((COMPONENT_CLASSES.index("din_rail"), cx, cy, nw, nh))

                    # Subdivide into 6 terminal block boxes along major dimension
                    slot_w = gw / 6.0
                    for i in range(6):
                        sx = int(gx + i * slot_w)
                        sy = gy
                        sw = int(slot_w)
                        sh = gh
                        scx, scy, snw, snh = convert_bbox_to_yolo((sx, sy, sw, sh), w, h)
                        labels.append((COMPONENT_CLASSES.index("terminal_block"), scx, scy, snw, snh))

                # If state has end clamps (state_3 and up)
                if state_name in ["state_3_with_end_clamps", "state_4_blue_wire", "state_5_red_wire", "state_6_yellow_wire"]:
                    if cluster_box is not None:
                        gx, gy, gw, gh = cluster_box
                        cw = int(gw * 0.15)
                        left_clamp = (max(0, gx - cw), gy, cw, gh)
                        right_clamp = (min(w - cw, gx + gw), gy, cw, gh)
                        lcx, lcy, lnw, lnh = convert_bbox_to_yolo(left_clamp, w, h)
                        rcx, rcy, rnw, rnh = convert_bbox_to_yolo(right_clamp, w, h)
                        clamp_id = COMPONENT_CLASSES.index("end_clamp")
                        labels.append((clamp_id, lcx, lcy, lnw, lnh))
                        labels.append((clamp_id, rcx, rcy, rnw, rnh))

                # Detect active wires in state frames
                if state_name in ["state_4_blue_wire", "state_5_red_wire", "state_6_yellow_wire"]:
                    wb = detect_wire_box_by_color(img, "blue_wire")
                    if wb:
                        cx, cy, nw, nh = convert_bbox_to_yolo(wb, w, h)
                        labels.append((COMPONENT_CLASSES.index("blue_wire"), cx, cy, nw, nh))

                if state_name in ["state_5_red_wire", "state_6_yellow_wire"]:
                    wb = detect_wire_box_by_color(img, "red_wire")
                    if wb:
                        cx, cy, nw, nh = convert_bbox_to_yolo(wb, w, h)
                        labels.append((COMPONENT_CLASSES.index("red_wire"), cx, cy, nw, nh))

                if state_name in ["state_6_yellow_wire"]:
                    wb = detect_wire_box_by_color(img, "yellow_wire")
                    if wb:
                        cx, cy, nw, nh = convert_bbox_to_yolo(wb, w, h)
                        labels.append((COMPONENT_CLASSES.index("yellow_wire"), cx, cy, nw, nh))

                if not labels:
                    continue

                unique_name = f"{state_name}_{fname}"
                dst_img = os.path.join(DET_DATASET_DIR, "images", split, unique_name)
                shutil.copy2(src_path, dst_img)

                label_name = os.path.splitext(unique_name)[0] + ".txt"
                dst_lbl = os.path.join(DET_DATASET_DIR, "labels", split, label_name)

                with open(dst_lbl, "w") as f:
                    for cid, cx, cy, nw, nh in labels:
                        f.write(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

                counts[split] += 1

    # Write data.yaml for YOLO detection training
    abs_det_dir = os.path.abspath(DET_DATASET_DIR).replace("\\", "/")
    data_yaml = {
        "path": abs_det_dir,
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(COMPONENT_CLASSES)},
    }
    yaml_path = os.path.join(DET_DATASET_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print("-" * 72)
    print(f"  YOLO Detection Dataset Ready:")
    print(f"    Train: {counts['train']} labeled images")
    print(f"    Val:   {counts['val']} labeled images")
    print(f"    Config: {yaml_path}")
    print("=" * 72)


if __name__ == "__main__":
    prepare_detection_dataset()
