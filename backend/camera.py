import cv2
import yaml
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Camera")

class CameraStream:
    """
    Thread-safe camera capture class supporting DirectShow on Windows,
    with auto-reconnect and frame timestamping.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        cam_cfg = self.config.get("camera", {})

        self.device_index = cam_cfg.get("device_index", 0)
        self.width = cam_cfg.get("width", 1280)
        self.height = cam_cfg.get("height", 720)
        self.fps = cam_cfg.get("fps", 30)

        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.actual_fps = 0.0

        self.thread: Optional[threading.Thread] = None

    def _load_config(self) -> dict:
        path = Path(self.config_path)
        if path.exists():
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def start(self) -> bool:
        """Start the camera capture loop in a background thread."""
        logger.info(f"Opening camera index {self.device_index} (backend: DSHOW)...")
        # DirectShow backend for faster startup and resolution control on Windows
        self.cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            logger.warning(f"Failed to open camera {self.device_index} with CAP_DSHOW, trying default backend...")
            self.cap = cv2.VideoCapture(self.device_index)

        if not self.cap.isOpened():
            logger.error(f"Cannot open camera index {self.device_index}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera initialized successfully. Resolution: {actual_w}x{actual_h}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return True

    def _capture_loop(self):
        fps_counter = 0
        fps_start = time.time()

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.warning("Dropped camera frame, attempting reconnect...")
                time.sleep(0.05)
                continue

            with self.lock:
                self.current_frame = frame
                self.frame_count += 1
                self.last_frame_time = time.time()

            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                self.actual_fps = fps_counter / (time.time() - fps_start)
                fps_counter = 0
                fps_start = time.time()

            # Slight sleep to yield CPU if high frame rate
            time.sleep(0.005)

    def get_frame(self) -> Tuple[bool, Optional[any]]:
        """Return a copy of the latest captured frame."""
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None

    def switch_camera(self, new_index: int) -> bool:
        """Switch camera device dynamically."""
        logger.info(f"Switching camera to index {new_index}...")
        self.stop()
        self.device_index = new_index
        return self.start()

    def stop(self):
        """Stop capture thread and release hardware resources."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("Camera stream stopped.")
