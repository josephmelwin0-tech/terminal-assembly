"""
Industrial DIN Rail Terminal Block Assembly Verification - Live Camera HUD Demo.
Exact production HUD from Block Assembly Quality Inspection System:
- Multi-tier dual-perception (YOLO Component Detector + Spatial Graph + Trained Classifier)
- Real-time right-side 7-step sequential checklist ([PASS], [NOW ], [NEXT], [ERR ])
- Hand-stillness & optical motion energy gating
- Color-coded component bounding boxes and wire connection vectors
- Dwell consensus smoothing & defect error latching

Controls:
    'q' / ESC: Quit
    'r': Reset inspection state machine & counters
    'c': Cycle camera source (Device 1 <-> Device 0)
    's': Save timestamped inspection snapshot
"""

import os
import sys
import time
import cv2
import numpy as np

from terminal_config import (
    PROJECT_ROOT,
    COMPONENT_CLASSES,
    COMPONENT_COLORS_BGR,
    ASSEMBLY_STATES,
    STEP_TITLES,
    DEFAULT_WIRING_RECIPE,
)
from component_detector import ComponentDetector
from state_machine import AssemblyStateMachine


def draw_industrial_hud(frame, res, sm_state, state_machine, fps, dev_idx):
    """
    Renders industrial inspection HUD overlay on top of frame.
    Matches the Block Assembly Quality Inspection design.
    """
    h, w = frame.shape[:2]
    status, detail, consensus_state, votes_ratio = sm_state
    diagnostic = res.get("diagnostic", "")
    conf = res.get("confidence", 0.0)
    display_state = res.get("predicted_state", "state_0_unstarted")

    # 1. Component Bounding Boxes
    for det in res.get("detections", []):
        cname = det["class_name"]
        dconf = det["confidence"]
        bx, by, bw, bh = det["bbox"]
        col = COMPONENT_COLORS_BGR.get(cname, (200, 200, 200))
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), col, 2)
        lbl = f"{cname} {dconf*100:.0f}%"
        cv2.putText(frame, lbl, (bx, max(15, by - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # 2. Semi-Transparent Header Banner
    header_h = 160
    scale = 0.65 if w >= 800 else 0.50
    sub_scale = 0.45 if w >= 800 else 0.38
    start_x = w - 260

    overlay = frame.copy()
    is_error = state_machine.error_active or status == "error"
    is_done = state_machine.is_complete()

    if is_error:
        bg_color = (25, 25, 140)   # Deep Red
    elif is_done:
        bg_color = (25, 120, 25)   # Deep Green
    elif status == "assembling" or getattr(state_machine, "is_motion_active", False):
        bg_color = (20, 90, 140)   # Amber / Orange
    else:
        bg_color = (20, 20, 25)    # Industrial Dark Slate

    cv2.rectangle(overlay, (0, 0), (w, header_h), bg_color, -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Right side checklist background container
    sub_chk = frame[4:header_h - 4, max(0, start_x - 6):w - 4]
    chk_bg = np.full(sub_chk.shape, (10, 10, 10), dtype=np.uint8)
    frame[4:header_h - 4, max(0, start_x - 6):w - 4] = cv2.addWeighted(sub_chk, 0.20, chk_bg, 0.80, 0)

    # 3. Left Side Status Text
    if is_error:
        cv2.putText(frame, "SEQUENCE REJECTED!", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        err_msg = detail or state_machine.error_detail or diagnostic or "Sequence defect detected"
        cv2.putText(frame, err_msg[:46], (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (200, 230, 255), 1)
        cv2.putText(frame, f"Detected: {display_state} ({conf*100:.0f}%)", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (255, 255, 255), 1)
        cv2.putText(frame, f"FPS: {fps:.1f} (Cam {dev_idx}) | Press 'r' to reset", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (0, 255, 255), 1)
    elif is_done:
        cv2.putText(frame, "100% COMPLETE & VERIFIED!", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        cv2.putText(frame, "All 7 assembly stages verified in order.", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 255, 220), 1)
        cv2.putText(frame, "DIN Rail terminal assembly complete!", (12, 102), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 255, 220), 1)
        cv2.putText(frame, f"FPS: {fps:.1f} | Ready for next unit", (12, 138), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (100, 255, 100), 1)
    elif status == "assembling" or getattr(state_machine, "is_motion_active", False):
        cur_title = state_machine.get_step_title(state_machine.current_index)
        cv2.putText(frame, f"INSPECTION: {cur_title.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        prompt_txt = detail or diagnostic or "Operator assembling parts..."
        if prompt_txt.startswith("ASSEMBLING: "):
            prompt_txt = prompt_txt[12:]
        cv2.putText(frame, f"ACTION: {prompt_txt[:48]}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (0, 220, 255), 1)
        cv2.putText(frame, "STATUS: ASSEMBLING (Hold steady to inspect)", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (50, 200, 255), 1)
        next_step = state_machine.get_step_title(state_machine.current_index + 1)
        cv2.putText(frame, f"Target: [{next_step}] | FPS: {fps:.1f} (Cam {dev_idx})", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (180, 180, 180), 1)
    else:
        cur_title = state_machine.get_step_title(state_machine.current_index)
        cv2.putText(frame, f"INSPECTION: {cur_title.upper()}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
        if state_machine.current_index > 0:
            cv2.putText(frame, f"VERIFIED: {cur_title} PASSED", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (100, 255, 100), 1)
            next_step = state_machine.get_step_title(state_machine.current_index + 1)
            cv2.putText(frame, "STATUS: PASSED (Waiting for next stage)", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (80, 240, 120), 1)
            cv2.putText(frame, f"Next: [{next_step}] | FPS: {fps:.1f}", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (200, 220, 255), 1)
        else:
            cv2.putText(frame, f"LIVE: {display_state} ({conf*100:.0f}%)", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 220, 220), 1)
            cv2.putText(frame, f"STATUS: {status.upper()} (Dwell: {votes_ratio})", (12, 88), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (220, 180, 50), 1)
            next_step = state_machine.get_step_title(state_machine.current_index + 1)
            cv2.putText(frame, f"Target: [{next_step}] | FPS: {fps:.1f} (Cam {dev_idx})", (12, 118), cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (180, 180, 180), 1)

    # 4. Right Side Step Checklist (Exact Block Assembly layout)
    steps = state_machine.get_steps_for_hud()
    for i, s in enumerate(steps):
        st = s["status"]
        if st in ["verified", "current_passed"]:
            tag = "[PASS] "
            col = (60, 235, 60)
        elif st == "next":
            tag = "[NEXT] "
            col = (240, 210, 40)
        elif st == "error_current":
            tag = "[ERR ] "
            col = (40, 40, 255)
        elif st == "current":
            tag = "[NOW ] "
            col = (200, 200, 200)
        else:
            tag = "[    ] "
            col = (110, 110, 110)

        title = s["title"].split(". ", 1)[-1]
        text = f"{tag}{i}.{title[:16]}"
        cv2.putText(frame, text, (start_x, 18 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)

    # 5. Flash banner if reset just occurred
    if getattr(state_machine, "reset_flash", 0) > 0:
        state_machine.reset_flash -= 1
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 165, 255), -1)
        cv2.putText(frame, "CYCLE RESET: TABLE CLEARED -> RESTARTED AT STEP 0", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 2)

    return frame


def main():
    import yaml
    dev_idx = 0
    cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
                dev_idx = cfg.get("camera", {}).get("device_index", 0)
        except Exception:
            pass

    if len(sys.argv) > 1:
        try:
            dev_idx = int(sys.argv[1])
        except ValueError:
            pass

    width, height = 1280, 720

    print("=" * 72)
    print(" INDUSTRIAL DIN RAIL ASSEMBLY VERIFIER - LIVE WEBCAM HUD")
    print(f" Connecting to Camera Device {dev_idx}...")
    print(" Controls: [q] Quit  |  [r] Reset  |  [c] Cycle Camera  |  [s] Snapshot")
    print("=" * 72)

    cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] Failed to open Device {dev_idx} with DSHOW, trying default backend...")
        cap = cv2.VideoCapture(dev_idx)
    if not cap.isOpened():
        alt_idx = 1 if dev_idx == 0 else 0
        print(f"[WARN] Device {dev_idx} unavailable, falling back to Device {alt_idx}...")
        dev_idx = alt_idx
        cap = cv2.VideoCapture(dev_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(dev_idx)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    detector = ComponentDetector()
    state_machine = AssemblyStateMachine(window_size=12, min_consensus=8, motion_gating=True)

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

        # 1. Dual-perception analysis
        res = detector.analyze(frame, current_step_index=state_machine.current_index)

        # 2. Sequential State Machine update
        sm_state = state_machine.update_smoothed(
            raw_state=res["predicted_state"],
            is_valid_spatial=res["is_valid"],
            diagnostic=res["diagnostic"],
            frame_gray=gray,
        )

        # 3. FPS Calculation
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            current_fps = fps_count / (time.time() - fps_start)
            fps_count = 0
            fps_start = time.time()

        # 4. Render Industrial HUD
        annotated = draw_industrial_hud(frame, res, sm_state, state_machine, current_fps, dev_idx)

        cv2.imshow("Industrial Assembly Verification (DIN Rail)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            state_machine.reset()
            print("[INFO] Reset to Step 0.")
        elif key == ord('s'):
            snap_path = os.path.join(PROJECT_ROOT, f"snapshot_{int(time.time())}.jpg")
            cv2.imwrite(snap_path, frame)
            print(f"[INFO] Saved snapshot to {snap_path}")
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
