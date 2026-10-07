"""
Wire & Color Detection Module for DIN Rail Terminal Block Assembly.
Extracts colored wire contours (Blue, Red, Yellow, Black), isolates the wire tip / ferrule,
and associates the wire connection with the respective terminal block port.
"""

import cv2
import numpy as np


class WireDetector:
    def __init__(self, config: dict = None):
        self.config = config or {}
        color_cfg = self.config.get("color_thresholds", {})

        # HSV Thresholds
        self.red_l1 = np.array(color_cfg.get("red", {}).get("lower1", [0, 90, 60]), dtype=np.uint8)
        self.red_u1 = np.array(color_cfg.get("red", {}).get("upper1", [10, 255, 255]), dtype=np.uint8)
        self.red_l2 = np.array(color_cfg.get("red", {}).get("lower2", [168, 90, 60]), dtype=np.uint8)
        self.red_u2 = np.array(color_cfg.get("red", {}).get("upper2", [180, 255, 255]), dtype=np.uint8)

        self.blue_l = np.array(color_cfg.get("blue", {}).get("lower", [95, 100, 50]), dtype=np.uint8)
        self.blue_u = np.array(color_cfg.get("blue", {}).get("upper", [135, 255, 255]), dtype=np.uint8)

        self.yellow_l = np.array(color_cfg.get("yellow", {}).get("lower", [18, 90, 90]), dtype=np.uint8)
        self.yellow_u = np.array(color_cfg.get("yellow", {}).get("upper", [38, 255, 255]), dtype=np.uint8)

        self.black_l = np.array(color_cfg.get("black", {}).get("lower", [0, 0, 0]), dtype=np.uint8)
        self.black_u = np.array(color_cfg.get("black", {}).get("upper", [180, 80, 60]), dtype=np.uint8)

    def detect_color_masks(self, frame, roi_mask=None):
        """Generates binary masks for RED, BLUE, YELLOW, BLACK in frame."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red dual hue mask
        mask_red1 = cv2.inRange(hsv, self.red_l1, self.red_u1)
        mask_red2 = cv2.inRange(hsv, self.red_l2, self.red_u2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Blue mask
        mask_blue = cv2.inRange(hsv, self.blue_l, self.blue_u)

        # Yellow mask
        mask_yellow = cv2.inRange(hsv, self.yellow_l, self.yellow_u)

        # Black mask
        mask_black = cv2.inRange(hsv, self.black_l, self.black_u)

        if roi_mask is not None:
            mask_red = cv2.bitwise_and(mask_red, mask_red, mask=roi_mask)
            mask_blue = cv2.bitwise_and(mask_blue, mask_blue, mask=roi_mask)
            mask_yellow = cv2.bitwise_and(mask_yellow, mask_yellow, mask=roi_mask)
            mask_black = cv2.bitwise_and(mask_black, mask_black, mask=roi_mask)

        # Clean noise with morphological opening and closing
        k_wire = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        masks = {
            "RED": cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, k_wire),
            "BLUE": cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k_wire),
            "YELLOW": cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, k_wire),
            "BLACK": cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, k_wire),
        }
        return masks

    def detect_wires(self, frame, terminal_boxes=None):
        """
        Detects wire segments and finds their connection endpoints.
        Returns a list of detected wire dicts:
        [{
            'color': 'BLUE',
            'bbox': [x, y, w, h],
            'tip': (x, y),
            'confidence': float,
            'contour': ndarray
        }]
        """
        h, w = frame.shape[:2]
        masks = self.detect_color_masks(frame)
        detected_wires = []

        for color_name, mask in masks.items():
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < 250:  # Ignore tiny noise specs
                    continue

                x, y, bw, bh = cv2.boundingRect(c)

                # Find endpoint/tip of wire (the point closest to top or bottom terminal entry)
                # Terminal blocks usually have wires entering from top or bottom
                pts = c.reshape(-1, 2)
                # If terminal_boxes given, find point closest to the terminal block area
                if terminal_boxes and len(terminal_boxes) > 0:
                    t_y_centers = [tb[1] + tb[3] // 2 for tb in terminal_boxes]
                    avg_tb_y = sum(t_y_centers) / len(t_y_centers)
                    # Wire tip is closest to the terminal center line
                    dists = np.abs(pts[:, 1] - avg_tb_y)
                    best_idx = np.argmin(dists)
                    tip = (int(pts[best_idx, 0]), int(pts[best_idx, 1]))
                else:
                    # Default: uppermost point if wire comes from below, or lowermost if comes from above
                    top_pt = pts[np.argmin(pts[:, 1])]
                    tip = (int(top_pt[0]), int(top_pt[1]))

                conf = min(0.99, max(0.65, area / 1500.0))
                detected_wires.append({
                    "color": color_name,
                    "bbox": [x, y, bw, bh],
                    "tip": tip,
                    "confidence": float(conf),
                    "contour": c,
                    "area": float(area),
                })

        return detected_wires

    def inspect_terminal_slot_color(self, frame, slot_bbox):
        """
        Directly samples the terminal wire port ROI to classify the wire color present.
        Returns ('RED' | 'BLUE' | 'YELLOW' | 'BLACK' | 'EMPTY', confidence)
        """
        x, y, w, h = slot_bbox
        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(fw, x + w)
        y2 = min(fh, y + h)

        if x2 <= x1 or y2 <= y1:
            return "EMPTY", 0.0

        slot_crop = frame[y1:y2, x1:x2]
        masks = self.detect_color_masks(slot_crop)

        slot_area = float((x2 - x1) * (y2 - y1))
        scores = {}
        for cname, mask in masks.items():
            px_count = cv2.countNonZero(mask)
            coverage = px_count / slot_area
            scores[cname] = coverage

        # Sort by coverage
        best_color, best_cov = max(scores.items(), key=lambda item: item[1])

        # Require minimum pixel coverage inside the port ROI to claim a wire is connected
        if best_cov >= 0.07:
            conf = min(0.98, max(0.60, best_cov * 3.5))
            return best_color, float(conf)
        else:
            return "EMPTY", 0.90
