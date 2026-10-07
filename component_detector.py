"""
Component Detector & Dual-Perception Engine for DIN Rail Terminal Assembly.
Integrates:
1. YOLO Component Detector (din_rail, terminal_block, closer, end_clamp, wires)
2. Wire Color & Ferrule Port Association
3. Terminal Assembly Graph (Physical topological verification)
4. Trained YOLO State Classifier for secondary macro-state corroboration
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

from terminal_config import (
    PROJECT_ROOT,
    COMPONENT_CLASSES,
    ASSEMBLY_STATES,
    STEP_TITLES,
    TOTAL_TERMINAL_BLOCKS,
)
from assembly_graph import TerminalAssemblyGraph
from backend.wire_detector import WireDetector


class ComponentDetector:
    def __init__(self, det_weights=None, cls_weights=None, conf_thresh=0.40):
        self.conf_thresh = conf_thresh
        self.assembly_graph = TerminalAssemblyGraph()
        self.wire_detector = WireDetector()

        # 1. Load YOLO Component Detector
        det_path = det_weights or os.path.join(PROJECT_ROOT, "runs_detect", "terminal_detector", "weights", "best.pt")
        if not os.path.exists(det_path):
            alt_path = os.path.join(PROJECT_ROOT, "runs_detect", "terminal_detector", "weights", "last.pt")
            if os.path.exists(alt_path):
                det_path = alt_path

        self.det_model = None
        if os.path.exists(det_path):
            try:
                self.det_model = YOLO(det_path)
                print(f"[ComponentDetector] Loaded YOLO Detector from '{det_path}'")
            except Exception as e:
                print(f"[ComponentDetector] Could not load detector: {e}")
        else:
            print(f"[ComponentDetector] Detector weights '{det_path}' not found yet. Using hybrid color/contour locator.")

        # 2. Load YOLO State Classifier
        cls_path = cls_weights or os.path.join(PROJECT_ROOT, "runs_classify", "terminal_states_cls", "weights", "best.pt")
        self.cls_model = None
        if os.path.exists(cls_path):
            try:
                self.cls_model = YOLO(cls_path)
                print(f"[ComponentDetector] Loaded YOLO Classifier from '{cls_path}'")
            except Exception as e:
                print(f"[ComponentDetector] Could not load classifier: {e}")

    def detect_components(self, img):
        """Runs component detection on frame."""
        if img is None or img.size == 0:
            return []

        h, w = img.shape[:2]
        detections = []

        # Use trained YOLO detector if available
        if self.det_model is not None:
            results = self.det_model(img, imgsz=512, conf=self.conf_thresh, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                cname = results.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                detections.append({
                    "class_name": cname,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                })
        else:
            # Hybrid fallback: Wire detector + geometric cluster
            wires = self.wire_detector.detect_wires(img)
            for w_item in wires:
                col = w_item["color"].lower()
                cname = f"{col}_wire"
                detections.append({
                    "class_name": cname,
                    "class_id": COMPONENT_CLASSES.index(cname) if cname in COMPONENT_CLASSES else 4,
                    "confidence": w_item["confidence"],
                    "bbox": w_item["bbox"],
                })

        return detections

    def analyze(self, img, current_step_index=0):
        """
        Runs dual-perception analysis combining component detector, spatial graph, and state classifier.
        """
        if img is None or img.size == 0:
            return {
                "predicted_state": "state_0_unstarted",
                "confidence": 0.0,
                "is_valid": False,
                "diagnostic": "Image frame empty",
                "detections": [],
                "classifier_pred": None,
                "classifier_conf": 0.0,
            }

        # 1. Detect individual components
        detections = self.detect_components(img)

        # 2. Run classifier macro-state prediction
        cls_pred = None
        cls_conf = 0.0
        if self.cls_model is not None:
            try:
                res_cls = self.cls_model(img, imgsz=384, verbose=False)[0]
                cls_pred = res_cls.names[res_cls.probs.top1]
                cls_conf = float(res_cls.probs.top1conf)
            except Exception:
                pass

        # 3. Spatial Assembly Graph Evaluation
        target_state = ASSEMBLY_STATES[current_step_index] if current_step_index < len(ASSEMBLY_STATES) else "state_0_unstarted"
        graph_eval = self.assembly_graph.evaluate(detections, current_target_step=target_state)

        inferred_state = graph_eval["inferred_state"]
        conf = graph_eval["confidence"]
        is_valid = graph_eval["is_valid"]
        diagnostic = graph_eval["diagnostic"]

        # 4. Dual Perception Corroboration
        if cls_pred is not None and cls_conf >= 0.70:
            # If classifier agrees with spatial graph or supports advancement
            if cls_pred == inferred_state and is_valid:
                conf = min(0.99, max(conf, (conf + cls_conf) / 2.0))
            elif cls_pred == target_state and not is_valid:
                # Keep spatial graph defect message! (Physical rules take precedence)
                pass

        return {
            "predicted_state": inferred_state,
            "confidence": conf,
            "is_valid": is_valid,
            "diagnostic": diagnostic,
            "detections": detections,
            "classifier_pred": cls_pred,
            "classifier_conf": cls_conf,
            "part_counts": graph_eval.get("part_counts", {}),
            "spatial_checks": graph_eval.get("spatial_checks", []),
        }
