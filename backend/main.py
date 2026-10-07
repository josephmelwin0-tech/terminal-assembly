"""
Main FastAPI server for Industrial DIN Rail Terminal Block Assembly Verification.
Provides:
- Real-time MJPEG live stream with the exact Block Assembly HUD (right-side checklist, bounding boxes, banners)
- Telemetry endpoint for Web Dashboard (/api/status)
- Dynamic camera source switching and recipe updating
"""

import cv2
import yaml
import time
import os
import numpy as np
from pathlib import Path
from fastapi import FastAPI, Response, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.camera import CameraStream
from component_detector import ComponentDetector
from state_machine import AssemblyStateMachine
from live_demo import draw_industrial_hud

app = FastAPI(title="Industrial DIN Rail Assembly Verifier", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONFIG_PATH = Path("config/config.yaml")

# Initialize shared components
camera_stream = CameraStream(config_path=str(CONFIG_PATH))
detector = ComponentDetector()
state_machine = AssemblyStateMachine(window_size=12, min_consensus=8, motion_gating=True)

latest_telemetry = {
    "status": "SEARCHING",
    "diagnostic": "Align DIN rail under camera",
    "consensus_ratio": "0/12",
    "current_step": 0,
    "metrics": {"total_cycles": 0, "passed_cycles": 0, "failed_cycles": 0, "pass_rate": 0.0, "cycle_time_ms": 0},
    "reasons": [],
}

frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.on_event("startup")
def startup_event():
    camera_stream.start()


@app.on_event("shutdown")
def shutdown_event():
    camera_stream.stop()


def generate_video_stream():
    global latest_telemetry

    while True:
        success, frame = camera_stream.get_frame()
        if not success or frame is None:
            time.sleep(0.03)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Analyze frame with dual perception
        res = detector.analyze(frame, current_step_index=state_machine.current_index)

        # 2. Update sequential state machine
        sm_state = state_machine.update_smoothed(
            raw_state=res["predicted_state"],
            is_valid_spatial=res["is_valid"],
            diagnostic=res["diagnostic"],
            frame_gray=gray,
        )

        status, detail, consensus_state, votes_ratio = sm_state

        # Update telemetry
        latest_telemetry = {
            "status": "PASS" if status in ["PASS", "advanced"] or state_machine.is_complete() else "FAIL" if status == "error" else "ASSEMBLING" if status == "assembling" else "SEARCHING",
            "diagnostic": detail or res["diagnostic"],
            "consensus_ratio": votes_ratio,
            "current_step": state_machine.current_index,
            "step_title": state_machine.get_step_title(state_machine.current_index),
            "metrics": {
                "total_cycles": state_machine.total_cycles,
                "passed_cycles": state_machine.passed_cycles,
                "failed_cycles": state_machine.failed_cycles,
                "pass_rate": round((state_machine.passed_cycles / max(1, state_machine.total_cycles)) * 100, 1),
                "cycle_time_ms": round(state_machine.last_cycle_duration_ms, 0),
            },
            "fps": round(camera_stream.actual_fps, 1),
            "device_index": camera_stream.device_index,
            "frame_count": camera_stream.frame_count,
        }

        # 3. Draw identical industrial HUD
        annotated = draw_industrial_hud(frame, res, sm_state, state_machine, camera_stream.actual_fps, camera_stream.device_index)

        ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
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
    return {"status": "ok", "message": "State machine reset to Step 0"}


@app.get("/", response_class=HTMLResponse)
def index_page():
    index_file = Path("frontend/index.html")
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Industrial Assembly Verification - Server Running</h1>")


if __name__ == "__main__":
    port = 8000
    uvicorn.run("backend.main:app", host="127.0.0.1", port=port, reload=False)
