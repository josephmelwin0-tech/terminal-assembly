"""
Main FastAPI server for Industrial DIN Rail Terminal Block Assembly Verification.
Provides:
- Real-time MJPEG live stream with rich bounding box annotations and terminal slot HUD
- Comprehensive REST API for dashboard telemetry
- Dynamic camera source switching (Device 1 USB Phone, Device 0 Laptop, or Test Video)
- Recipe configuration hot-reloading
"""

import cv2
import yaml
import time
import os
import numpy as np
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.camera import CameraStream
from backend.detector import BlockDetector
from backend.wire_detector import WireDetector
from backend.inspection import InspectionEngine
from backend.state_machine import StateMachine

# YOLO Classifier
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

app = FastAPI(title="Industrial DIN Rail Assembly Verifier", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path("config/config.yaml")


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


config = load_config()

# Initialize core inspection pipeline components
camera_stream = CameraStream(config_path=str(CONFIG_PATH))
block_detector = BlockDetector(config=config)
wire_detector = WireDetector(config=config)
inspection_engine = InspectionEngine(config=config)
state_machine = StateMachine(
    window_size=config.get("stabilization", {}).get("history_window_size", 12),
    pass_ratio=config.get("stabilization", {}).get("pass_threshold_ratio", 0.80),
    motion_gating=config.get("stabilization", {}).get("motion_gating", True),
)

# Load trained state classifier if available
state_classifier = None
CLASSIFIER_PATH = Path("runs_classify/terminal_states_cls/weights/best.pt")
if HAS_YOLO and CLASSIFIER_PATH.exists():
    try:
        state_classifier = YOLO(str(CLASSIFIER_PATH))
        print(f"[INFO] Loaded trained state classifier from '{CLASSIFIER_PATH}'")
    except Exception as e:
        print(f"[WARN] Could not load state classifier: {e}")

# Global state cache for telemetry API
latest_telemetry = {
    "status": "SEARCHING",
    "diagnostic": "Align DIN rail under camera",
    "consensus_ratio": "0/12",
    "terminals": {},
    "metrics": {"total_cycles": 0, "passed_cycles": 0, "failed_cycles": 0, "pass_rate": 0.0, "cycle_time_ms": 0},
    "reasons": [],
}

# Mount static frontend
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.on_event("startup")
def startup_event():
    camera_stream.start()


@app.on_event("shutdown")
def shutdown_event():
    camera_stream.stop()


def draw_hud(frame, assembly_info, detected_wires, inspection_res, sm_state, fps):
    """Draws industrial inspection HUD overlay on top of camera frame."""
    h, w = frame.shape[:2]
    status = sm_state.get("status", "SEARCHING")
    diag = sm_state.get("diagnostic", "")

    # Status color scheme
    if status == "PASS":
        bar_col = (40, 180, 50)       # Green
        status_txt = "VERIFIED: PASS"
    elif status == "FAIL":
        bar_col = (40, 40, 220)       # Red
        status_txt = "DEFECT: REJECTED"
    elif status == "ASSEMBLING":
        bar_col = (40, 190, 240)      # Amber / Yellow
        status_txt = "ASSEMBLING IN PROGRESS"
    else:
        bar_col = (180, 120, 40)      # Blue
        status_txt = "SEARCHING FOR ASSEMBLY"

    # 1. Top Status Banner
    header_h = 75
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, header_h), (20, 20, 25), -1)
    cv2.rectangle(overlay, (0, header_h - 4), (w, header_h), bar_col, -1)
    frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

    # Banner text
    cv2.putText(frame, status_txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, bar_col, 2, cv2.LINE_AA)
    disp_diag = diag[:65] + "..." if len(diag) > 65 else diag
    cv2.putText(frame, disp_diag, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # FPS & Consensus Tag
    ratio_txt = f"VOTE: {sm_state.get('consensus_ratio', 'N/A')} | {fps:.1f} FPS"
    cv2.putText(frame, ratio_txt, (w - 230, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    # 2. Draw Terminal Slots TB1 - TB6
    if assembly_info and "slots" in assembly_info:
        # Assembly cluster box
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

            # Slot badge
            cv2.putText(frame, tb_name, (sx + 4, sy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"{det_col[:3]}", (sx + 4, sy + sh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_col, 1, cv2.LINE_AA)

    # 3. Draw Detected Wires
    for wire in detected_wires:
        wx, wy, ww, wh = wire["bbox"]
        tx, ty = wire["tip"]
        col_name = wire["color"]
        c_bgr = (40, 40, 220) if col_name == "RED" else (220, 140, 40) if col_name == "BLUE" else (40, 220, 240) if col_name == "YELLOW" else (50, 50, 50)
        cv2.circle(frame, (tx, ty), 5, c_bgr, -1)
        cv2.rectangle(frame, (wx, wy), (wx + ww, wy + wh), c_bgr, 1)

    return frame


def generate_video_stream():
    """Generator for annotated MJPEG video stream."""
    global latest_telemetry

    while True:
        success, frame = camera_stream.get_frame()
        if not success or frame is None:
            time.sleep(0.03)
            continue

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Component & Terminal Block Locator
        assembly_info = block_detector.detect_assembly_cluster(frame)

        # 2. Wire & Color Presence
        tb_boxes = [s["bbox"] for s in assembly_info["slots"]] if assembly_info else []
        detected_wires = wire_detector.detect_wires(frame, terminal_boxes=tb_boxes)

        # 3. State Classifier Prediction (if available)
        cls_pred = None
        cls_conf = 0.0
        if state_classifier is not None:
            try:
                res_cls = state_classifier(frame, imgsz=384, verbose=False)[0]
                cls_pred = res_cls.names[res_cls.probs.top1]
                cls_conf = float(res_cls.probs.top1conf)
            except Exception:
                pass

        # 4. Inspection Engine Verification
        inspection_res = inspection_engine.inspect(
            assembly_info=assembly_info,
            wire_detector=wire_detector,
            frame=frame,
            classifier_pred=cls_pred,
            classifier_conf=cls_conf,
        )

        # 5. Temporal Consensus Smoothing
        sm_state = state_machine.update(inspection_res, frame_gray=gray)

        # Cache latest telemetry for frontend API
        latest_telemetry = {
            "status": sm_state.get("status", "SEARCHING"),
            "diagnostic": sm_state.get("diagnostic", ""),
            "consensus_ratio": sm_state.get("consensus_ratio", "N/A"),
            "terminals": inspection_res.get("terminals", {}),
            "metrics": sm_state.get("metrics", {}),
            "reasons": inspection_res.get("reasons", []),
            "fps": round(camera_stream.actual_fps, 1),
            "device_index": camera_stream.device_index,
            "frame_count": camera_stream.frame_count,
        }

        # 6. Render HUD on frame
        annotated_frame = draw_hud(frame, assembly_info, detected_wires, inspection_res, sm_state, camera_stream.actual_fps)

        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.015)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_video_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/status")
def get_status():
    global latest_telemetry
    return latest_telemetry


@app.post("/api/switch_camera/{device_index}")
def switch_camera(device_index: int):
    success = camera_stream.switch_camera(device_index)
    return {"status": "ok" if success else "failed", "device_index": camera_stream.device_index}


@app.post("/api/reset")
def reset_inspection():
    state_machine.reset()
    return {"status": "ok", "message": "State machine reset successfully"}


@app.post("/api/update_recipe")
async def update_recipe(request: Request):
    data = await request.json()
    new_recipe = data.get("expected_configuration", {})
    if new_recipe:
        inspection_engine.expected_recipe = new_recipe
        state_machine.reset()
    return {"status": "ok", "expected_configuration": inspection_engine.expected_recipe}


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = Path("frontend/index.html")
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Industrial Assembly Verification - Server Running</h1>")


if __name__ == "__main__":
    cfg = load_config()
    server_cfg = cfg.get("server", {})
    host = server_cfg.get("host", "127.0.0.1")
    port = server_cfg.get("port", 8000)
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
