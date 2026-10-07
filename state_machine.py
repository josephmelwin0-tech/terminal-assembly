"""
Assembly State Machine for DIN Rail Terminal Block Verification.
Enforces sequential assembly progression, implements rolling-window temporal consensus smoothing,
hand-motion gating, and strictly latches diagnostic errors when out-of-order steps or defects occur.
"""

from collections import deque, Counter
import time
import numpy as np

from terminal_config import ASSEMBLY_STATES, STEP_TITLES


class AssemblyStateMachine:
    def __init__(self, state_order=None, window_size=12, min_consensus=8, motion_gating=True):
        self.state_order = state_order or ASSEMBLY_STATES
        self.current_index = 0
        self.window_size = window_size
        self.min_consensus = min_consensus
        self.motion_gating = motion_gating
        self.history = deque(maxlen=window_size)

        # Strict error tracking
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None
        self.skip_streak = 0
        self.empty_workspace_frames = 0
        self.completion_frames = 0
        self.reset_flash = 0

        # Motion gating tracking
        self.prev_gray = None
        self.is_motion_active = False

        # Production metrics
        self.total_cycles = 0
        self.passed_cycles = 0
        self.failed_cycles = 0
        self.cycle_start_time = time.time()
        self.last_cycle_duration_ms = 0.0

    def reset(self):
        """Resets assembly tracker back to Step 0."""
        self.current_index = 0
        self.history.clear()
        self.error_active = False
        self.error_detail = ""
        self.skipped_step_index = None
        self.skip_streak = 0
        self.empty_workspace_frames = 0
        self.completion_frames = 0
        self.reset_flash = 30
        self.cycle_start_time = time.time()

    def get_consensus(self):
        valid_votes = [s for s in self.history if s is not None and s in self.state_order]
        if not valid_votes:
            return None, 0
        counts = Counter(valid_votes)
        top_state, count = counts.most_common(1)[0]
        return top_state, count

    def get_step_title(self, index):
        if 0 <= index < len(self.state_order):
            s = self.state_order[index]
            return STEP_TITLES.get(s, s)
        return "Complete"

    def compute_motion(self, frame_gray) -> float:
        if frame_gray is None:
            return 0.0
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return 0.0
        diff = np.abs(frame_gray.astype(np.float32) - self.prev_gray.astype(np.float32))
        score = float(np.mean(diff))
        self.prev_gray = frame_gray
        return score

    def update_smoothed(self, raw_state, is_valid_spatial=True, diagnostic="", frame_gray=None):
        """
        Pushes a new frame prediction into rolling buffer, performs majority voting,
        and triggers state advance or error latching.
        """
        # Motion gating check
        if self.motion_gating and frame_gray is not None:
            motion = self.compute_motion(frame_gray)
            if motion > 6.5:
                self.is_motion_active = True
                return "assembling", "ASSEMBLING: Operator Motion Active (Stabilizing...)", self.current_state(), f"{len(self.history)}/{self.window_size}"
            else:
                self.is_motion_active = False

        self.history.append(raw_state)
        consensus_state, votes = self.get_consensus()
        ratio = f"{votes}/{len(self.history)}"

        # Workspace clear tracking
        if raw_state == "state_0_unstarted":
            self.empty_workspace_frames += 1
        else:
            self.empty_workspace_frames = 0

        # Auto-reset check when cycle completes
        if self.is_complete():
            self.completion_frames += 1
            if self.empty_workspace_frames >= 10 or self.completion_frames >= 90:
                self.total_cycles += 1
                self.passed_cycles += 1
                self.last_cycle_duration_ms = (time.time() - self.cycle_start_time) * 1000.0
                self.reset()
                return "reset", "Cycle Complete - Ready for Next Unit", "state_0_unstarted", "1/1"

        # Mid-assembly reset when table is cleared
        if self.empty_workspace_frames >= (15 if self.error_active else 25):
            if self.current_index > 0 or self.error_active:
                if self.error_active:
                    self.failed_cycles += 1
                    self.total_cycles += 1
                self.reset()
                return "reset", "Workspace Cleared - Reset to Step 0", "state_0_unstarted", "1/1"

        # Assembling prompt
        if diagnostic and diagnostic.startswith("ASSEMBLING:"):
            self.error_active = False
            self.error_detail = ""
            return "assembling", diagnostic, consensus_state, ratio

        # Spatial constraint violation
        if not is_valid_spatial and diagnostic and not diagnostic.startswith("PASS"):
            self.error_active = True
            self.error_detail = diagnostic
            return "error", self.error_detail, consensus_state, ratio

        # Hold if clear
        if consensus_state == "state_0_unstarted" and self.current_index > 0:
            return "holding", "Assembly in hand / Workspace clear", self.current_state(), ratio

        if consensus_state is None or votes < self.min_consensus:
            if self.error_active:
                return "error", self.error_detail, consensus_state, ratio
            return "holding", "stabilizing", consensus_state, ratio

        status, detail = self.update(consensus_state)
        return status, detail, consensus_state, ratio

    def current_state(self):
        return self.state_order[self.current_index]

    def update(self, matched_state, is_valid_spatial=True, diagnostic=""):
        if diagnostic and diagnostic.startswith("ASSEMBLING:"):
            self.error_active = False
            self.error_detail = ""
            return "assembling", diagnostic

        if not is_valid_spatial and diagnostic and not diagnostic.startswith("PASS"):
            self.error_active = True
            self.error_detail = diagnostic
            return "error", self.error_detail

        if matched_state is None:
            if self.error_active:
                return "error", self.error_detail
            return "holding", None

        if matched_state not in self.state_order:
            self.error_active = True
            self.error_detail = f"Unrecognized state: {matched_state}"
            return "error", self.error_detail

        matched_index = self.state_order.index(matched_state)

        # Clear table mid-assembly
        if matched_index == 0 and self.current_index > 0:
            return "holding", "Assembly in hand / Workspace clear"

        # Same as current step
        if matched_index == self.current_index:
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            self.skip_streak = 0
            return "holding", None

        # Sequential advance to next step
        elif matched_index == self.current_index + 1:
            self.current_index = matched_index
            self.error_active = False
            self.error_detail = ""
            self.skipped_step_index = None
            self.skip_streak = 0
            return "advanced", self.current_state()

        # Step regression
        elif matched_index < self.current_index:
            return "holding", f"Regressed to Step {matched_index} ({matched_state})"

        # Skipped step
        else:
            expected = self.state_order[self.current_index + 1] if self.current_index + 1 < len(self.state_order) else "Complete"
            detected = matched_state
            self.skip_streak += 1
            if self.skip_streak >= 3:
                self.error_active = True
                self.skipped_step_index = self.current_index + 1
                self.error_detail = f"SKIPPED STEP! Missing [{expected}], but found [{detected}]"
                return "error", self.error_detail
            else:
                return "holding", f"Verifying [{expected}]..."

    def get_steps_for_hud(self):
        """Returns 7-step checklist for HUD."""
        steps = []
        for i, sname in enumerate(self.state_order):
            title = STEP_TITLES.get(sname, sname)
            if i < self.current_index:
                st = "verified"
            elif i == self.current_index:
                if self.error_active:
                    st = "error_current"
                elif self.current_index > 0 or self.is_complete():
                    st = "current_passed"
                else:
                    st = "current"
            elif i == self.current_index + 1:
                st = "next" if not self.error_active else "pending"
            else:
                st = "pending"
            steps.append({"title": title, "state": sname, "status": st, "index": i})
        return steps

    def is_complete(self):
        return self.current_index == len(self.state_order) - 1 and not self.error_active
