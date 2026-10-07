"""
Standalone OpenCV Live Inspection Demo for DIN Rail Terminal Block Assembly.
Directly connects to USB Phone Camera (Device 1) with DirectShow, runs real-time
inspection, and displays full HUD overlay.

Controls:
    'q' / ESC: Quit
    'r': Reset state machine & counters
    'c': Switch camera (0 <-> 1)
    's': Save high-res inspection snapshot
"""

import cv2
import yaml
import time
import os
import numpy as np

from terminal_config import PROJECT_ROOT, DEFAULT_WIRING_RECIPE
from backend.detector import BlockDetector
from backend.wire_detector import WireDetector
from backend.inspection import InspectionEngine
from backend.state_machine import StateMachine

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False


def load_config():
    p = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if os.path.exists(p):
        with open(p, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def main():
    config = load_config()
    cam_cfg = config.get("camera", {})
    dev_idx = cam_cfg.get("device_index", 1)  # Default to 1 for USB Phone Camera
    width = cam_cfg.get("width", 1280)
    height = cam_cfg.get("height", 720)

    print("=" * 72)
    print(" INDUSTRIAL DIN RAIL ASSEMBLY VERIFICATION - LIVE CAMERA DEMO")
    print(f" Connecting to Camera Device {dev_idx}...")
    print(" Controls: [q] Quit  |  [r] Reset  |  [c] Switch Cam  |  [s] Snapshot")
    print("=" * 72)

    # Initialize Camera
    cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] Failed to open Device {dev_idx} with DSHOW, trying default backend...")
        cap = cv2.VideoCapture(dev_idx)
    if not cap.isOpened():
        print(f"[WARN] Device {dev_idx} unavailable, falling back to Device 0...")
        dev_idx = 0
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Initialize Pipeline Components
    block_detector = BlockDetector(config=config)
    wire_detector = WireDetector(config=config)
    inspection_engine = InspectionEngine(config=config)
    state_machine = StateMachine(
        window_size=config.get("stabilization", {}).get("history_window_size", 12),
        pass_ratio=config.get("stabilization", {}).get("pass_threshold_ratio", 0.80),
        motion_gating=config.get("stabilization", {}).get("motion_gating", True),
    )

    # Load State Classifier if weights exist
    classifier = None
    best_pt = os.path.join(PROJECT_ROOT, "runs_classify", "terminal_states_cls", "weights", "best.pt")
    if HAS_YOLO and os.path.exists(best_pt):
        try:
            classifier = YOLO(best_pt)
            print(f"[INFO] Loaded trained classifier from '{best_pt}'")
        except Exception as e:
            print(f"[WARN] Could not load classifier: {e}")

    fps_count = 0
    fps_start = time.time()
    current_fps = 0.0

    cv2.namedWindow("Industrial Assembly Verification (DIN Rail)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Industrial Assembly Verification (DIN Rail)", 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect Cluster & Slots
        assembly_info = block_detector.detect_assembly_cluster(frame)

        # 2. Detect Wires
        tb_boxes = [s["bbox"] for s in assembly_info["slots"]] if assembly_info else []
        detected_wires = wire_detector.detect_wires(frame, terminal_boxes=tb_boxes)

        # 3. Macro State Classifier
        cls_pred = None
        cls_conf = 0.0
        if classifier is not None:
            try:
                res_cls = classifier(frame, imgsz=384, verbose=False)[0]
                cls_pred = res_cls.names[res_cls.probs.top1]
                cls_conf = float(res_cls.probs.top1conf)
            except Exception:
                pass

        # 4. Inspection Engine
        inspection_res = inspection_engine.inspect(
            assembly_info=assembly_info,
            wire_detector=wire_detector,
            frame=frame,
            classifier_pred=cls_pred,
            classifier_conf=cls_conf,
        )

        # 5. Temporal Smoothing
        sm_state = state_machine.update(inspection_res, frame_gray=gray)

        # FPS calculation
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()

        # Render HUD Overlay
        status = sm_state.get("status", "SEARCHING")
        diag = sm_state.get("diagnostic", "")

        if status == "PASS":
            bar_col = (40, 180, 50)
            status_txt = "VERIFIED: PASS"
        elif status == "FAIL":
            bar_col = (40, 40, 220)
            status_txt = "DEFECT: REJECTED"
        elif status == "ASSEMBLING":
            bar_col = (40, 190, 240)
            status_txt = "ASSEMBLING IN PROGRESS"
        else:
            bar_col = (180, 120, 40)
            status_txt = "SEARCHING FOR ASSEMBLY"

        header_h = 75
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 25), -1)
        cv2.rectangle(overlay, (0, header_h - 4), (w, header_h), bar_col, -1)
        frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

        cv2.putText(frame, status_txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, bar_col, 2, cv2.LINE_AA)
        disp_diag = diag[:65] + "..." if len(diag) > 65 else diag
        cv2.putText(frame, disp_diag, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

        ratio_txt = f"VOTE: {sm_state.get('consensus_ratio', 'N/A')} | {current_fps:.1f} FPS (Cam {dev_idx})"
        cv2.putText(frame, ratio_txt, (w - 320, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw Terminal Slots
        if assembly_info and "slots" in assembly_info:
            gx, gy, gw, gh = assembly_info["cluster_bbox"]
            cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (160, 160, 160), 1)

            terminal_results = inspection_res.get("terminals", {})
            for slot in assembly_info["slots"]:
                tb_name = slot["name"]
                sx, sy, sw, sh = slot["bbox"]
                px, py, pw, ph = slot["port_roi"]

                res = terminal_results.get(tb_name, {})
                is_match = res.get("match", False)
                exp_col = res.get("expected", "EMPTY")
                det_col = res.get("detected", "EMPTY")

                box_col = (40, 200, 60) if is_match else (40, 40, 220)
                cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), box_col, 2)
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (255, 255, 255), 1)

                cv2.putText(frame, tb_name, (sx + 4, sy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"{det_col[:3]}", (sx + 4, sy + sh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_col, 1, cv2.LINE_AA)

        # Draw Wires
        for wire in detected_wires:
            wx, wy, ww, wh = wire["bbox"]
            tx, ty = wire["tip"]
            col_name = wire["color"]
            c_bgr = (40, 40, 220) if col_name == "RED" else (220, 140, 40) if col_name == "BLUE" else (40, 220, 240) if col_name == "YELLOW" else (50, 50, 50)
            cv2.circle(frame, (tx, ty), 5, c_bgr, -1)
            cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), c_bgr, 1)

        cv2.imshow("Industrial Assembly Verification (DIN Rail)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            state_machine.reset()
            print("[INFO] State machine reset.")
        elif key == ord('s'):
            snap_path = os.path.join(PROJECT_ROOT, f"inspection_snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[INFO] Snapshot saved to {snap_path}")
        elif key == ord('c'):
            dev_idx = 0 if dev_idx == 1 else 1
            cap.release()
            cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"[INFO] Switched to Camera Device {dev_idx}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
