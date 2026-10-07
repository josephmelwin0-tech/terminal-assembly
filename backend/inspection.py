"""
Industrial Inspection Engine for DIN Rail Terminal Block Assembly.
Performs deterministic multi-tier verification:
1. 6 Terminal blocks geometry and spatial ordering (TB1 to TB6)
2. Wire presence and color classification at each slot
3. Comparison against configurable recipe in config.yaml
4. Defect isolation and diagnostic reason reporting
"""

from typing import Dict, Any, List
from terminal_config import STEP_TITLES, ASSEMBLY_STATES


class InspectionEngine:
    def __init__(self, config: dict = None):
        self.config = config or {}
        assembly_cfg = self.config.get("assembly", {})
        self.total_terminals = assembly_cfg.get("total_terminals", 6)
        self.expected_recipe = assembly_cfg.get("expected_configuration", {
            "TB1": "BLUE",
            "TB2": "RED",
            "TB3": "YELLOW",
            "TB4": "EMPTY",
            "TB5": "EMPTY",
            "TB6": "EMPTY",
        })
        self.check_end_clamps = assembly_cfg.get("check_end_clamps", True)

    def inspect(
        self,
        assembly_info: Dict[str, Any],
        wire_detector,
        frame,
        classifier_pred: str = None,
        classifier_conf: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive quality inspection.
        Returns:
            {
                'status': 'PASS' | 'FAIL' | 'ASSEMBLING' | 'SEARCHING',
                'pass': bool,
                'overall_confidence': float,
                'reasons': List[str],
                'diagnostic': str,
                'terminals': Dict[str, dict], # TB1..TB6 details
                'current_state': str,
                'macro_prediction': str,
            }
        """
        # Case 0: No DIN rail or terminal blocks visible in frame
        if assembly_info is None or "slots" not in assembly_info:
            return {
                "status": "SEARCHING",
                "pass": False,
                "overall_confidence": 0.0,
                "reasons": ["Present DIN rail and terminal assembly under camera"],
                "diagnostic": "SEARCHING: Align assembly in camera view",
                "terminals": {f"TB{i+1}": {"expected": self.expected_recipe.get(f"TB{i+1}", "EMPTY"), "detected": "NONE", "match": False} for i in range(self.total_terminals)},
                "current_state": "state_0_unstarted",
                "macro_prediction": classifier_pred or "None",
            }

        slots = assembly_info.get("slots", [])
        num_slots = len(slots)
        reasons = []
        is_pass = True

        # Check 1: Terminal Count Verification
        if num_slots < self.total_terminals:
            is_pass = False
            reasons.append(f"DEFECT: Missing terminal block(s)! Found {num_slots}, expected {self.total_terminals}.")
        elif num_slots > self.total_terminals:
            is_pass = False
            reasons.append(f"DEFECT: Extra terminal block(s) detected! Found {num_slots}, expected {self.total_terminals}.")

        # Check 2: Terminal-by-Terminal Wire Color Verification
        terminal_results = {}
        total_conf = 0.0
        wires_present_count = 0

        for slot in slots:
            tb_name = slot["name"]
            expected_color = self.expected_recipe.get(tb_name, "EMPTY").upper()
            port_roi = slot["port_roi"]

            # Sample the wire color at the port ROI
            det_color, conf = wire_detector.inspect_terminal_slot_color(frame, port_roi)
            det_color = det_color.upper()
            total_conf += conf

            if det_color != "EMPTY":
                wires_present_count += 1

            # Match evaluation
            match = (det_color == expected_color)
            if not match:
                is_pass = False
                if expected_color == "EMPTY":
                    reasons.append(f"DEFECT: Unexpected wire in {tb_name}! Expected EMPTY, detected {det_color}.")
                elif det_color == "EMPTY":
                    reasons.append(f"DEFECT: Missing wire in {tb_name}! Expected {expected_color}, but terminal is EMPTY.")
                else:
                    reasons.append(f"DEFECT: Wrong wire color in {tb_name}! Expected {expected_color}, detected {det_color}.")

            terminal_results[tb_name] = {
                "expected": expected_color,
                "detected": det_color,
                "confidence": round(conf, 2),
                "match": match,
                "bbox": slot["bbox"],
                "port_roi": port_roi,
            }

        avg_conf = total_conf / float(num_slots) if num_slots > 0 else 0.0

        # Check 3: End Clamps Check
        clamps = assembly_info.get("end_clamps", {})
        if self.check_end_clamps:
            if not clamps.get("left", False) and not clamps.get("right", False):
                # Only flag clamp defect if wires are already fully attached (final stage)
                if wires_present_count >= 2:
                    is_pass = False
                    reasons.append("DEFECT: End clamps missing on DIN rail!")

        # Corroborate macro progression state
        inferred_state = "state_1_blocks_on_rail"
        if wires_present_count == 1:
            inferred_state = "state_4_blue_wire"
        elif wires_present_count == 2:
            inferred_state = "state_5_red_wire"
        elif wires_present_count >= 3:
            inferred_state = "state_6_yellow_wire"

        # Determine overall status
        if is_pass:
            status = "PASS"
            diagnostic = "PASS: All 6 terminals and wiring connections fully verified!"
        else:
            # If parts are present but assembly is partially wired in progress
            if wires_present_count < 3 and all("Missing wire" in r for r in reasons if "DEFECT" in r):
                status = "ASSEMBLING"
                diagnostic = f"ASSEMBLING: {wires_present_count}/3 wires inserted. Continue assembly."
            else:
                status = "FAIL"
                diagnostic = reasons[0] if reasons else "DEFECT: Quality inspection failed."

        return {
            "status": status,
            "pass": is_pass,
            "overall_confidence": round(avg_conf, 2),
            "reasons": reasons,
            "diagnostic": diagnostic,
            "terminals": terminal_results,
            "current_state": inferred_state,
            "macro_prediction": classifier_pred or inferred_state,
        }
