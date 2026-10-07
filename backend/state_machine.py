"""
Temporal State Machine and Hand-Stillness Gating Module.
Enforces multi-frame rolling window consensus smoothing to eliminate single-frame flickering,
and pauses evaluation while operator hands are in motion.
"""

from collections import deque, Counter
import time
import numpy as np


class StateMachine:
    def __init__(self, window_size: int = 12, pass_ratio: float = 0.80, motion_gating: bool = True):
        self.window_size = window_size
        self.pass_ratio = pass_ratio
        self.motion_gating = motion_gating

        self.history = deque(maxlen=window_size)
        self.state_history = deque(maxlen=window_size)
        self.current_status = "SEARCHING"  # SEARCHING, ASSEMBLING, PASS, FAIL
        self.current_diagnostic = "System Initializing"

        # Motion gating tracking
        self.prev_gray = None
        self.motion_streak = 0
        self.is_motion_active = False

        # Production metrics
        self.total_cycles = 0
        self.passed_cycles = 0
        self.failed_cycles = 0
        self.cycle_start_time = time.time()
        self.last_cycle_duration_ms = 0.0
        self.last_latched_status = None

    def compute_motion(self, frame_gray) -> float:
        """Computes frame difference motion metric."""
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return 0.0

        diff = np.abs(frame_gray.astype(np.float32) - self.prev_gray.astype(np.float32))
        motion_score = float(np.mean(diff))
        self.prev_gray = frame_gray
        return motion_score

    def update(self, inspection_result: dict, frame_gray=None) -> dict:
        """
        Updates temporal buffer and returns stabilized state machine status.
        """
        raw_status = inspection_result.get("status", "SEARCHING")
        raw_diagnostic = inspection_result.get("diagnostic", "")

        # 1. Motion Gating Check
        if self.motion_gating and frame_gray is not None:
            motion = self.compute_motion(frame_gray)
            # If motion exceeds threshold, operator is actively handling the assembly
            if motion > 6.5:
                self.motion_streak += 1
                self.is_motion_active = True
                return {
                    "status": "ASSEMBLING",
                    "diagnostic": "ASSEMBLING: Operator Motion Active (Stabilizing...)",
                    "consensus_ratio": "Holding",
                    "is_motion_active": True,
                    "metrics": self.get_metrics(),
                }
            else:
                self.motion_streak = 0
                self.is_motion_active = False

        # 2. Push to temporal history
        self.history.append(raw_status)
        self.state_history.append(inspection_result.get("current_state", "state_0_unstarted"))

        # 3. Majority Voting
        counts = Counter(self.history)
        top_status, votes = counts.most_common(1)[0]
        consensus_ratio = f"{votes}/{len(self.history)}"

        # If we have reached consensus threshold
        if votes / float(len(self.history)) >= self.pass_ratio:
            prev_status = self.current_status
            self.current_status = top_status
            self.current_diagnostic = raw_diagnostic

            # Latch cycle metrics when transitioning to terminal PASS or FAIL
            if self.current_status in ["PASS", "FAIL"] and self.current_status != self.last_latched_status:
                self.total_cycles += 1
                if self.current_status == "PASS":
                    self.passed_cycles += 1
                else:
                    self.failed_cycles += 1
                self.last_cycle_duration_ms = (time.time() - self.cycle_start_time) * 1000.0
                self.cycle_start_time = time.time()
                self.last_latched_status = self.current_status
        else:
            # Buffer stabilizing
            if self.current_status not in ["PASS", "FAIL"]:
                self.current_status = top_status

        return {
            "status": self.current_status,
            "diagnostic": self.current_diagnostic,
            "consensus_ratio": consensus_ratio,
            "is_motion_active": self.is_motion_active,
            "metrics": self.get_metrics(),
        }

    def get_metrics(self) -> dict:
        pass_rate = (self.passed_cycles / self.total_cycles * 100.0) if self.total_cycles > 0 else 0.0
        return {
            "total_cycles": self.total_cycles,
            "passed_cycles": self.passed_cycles,
            "failed_cycles": self.failed_cycles,
            "pass_rate": round(pass_rate, 1),
            "cycle_time_ms": round(self.last_cycle_duration_ms, 0),
        }

    def reset(self):
        """Resets state machine buffer and metrics."""
        self.history.clear()
        self.state_history.clear()
        self.current_status = "SEARCHING"
        self.current_diagnostic = "Reset Complete"
        self.cycle_start_time = time.time()
        self.last_latched_status = None
