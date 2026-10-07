"""
Central configuration for Industrial DIN Rail Terminal Block Assembly Verification System.
Defines video mapping, component classes, assembly states, and spatial graph rules.
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PARTS_VIDEO_DIR = os.path.join(DATA_DIR, "individual_parts")
STATES_VIDEO_DIR = os.path.join(DATA_DIR, "states")
EXTRACTED_DIR = os.path.join(PROJECT_ROOT, "extracted_frames")
DET_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_det")
CLS_DATASET_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset_cls")

# Component Classes for Object Detection
COMPONENT_CLASSES = [
    "din_rail",        # 0: 35mm slotted steel/aluminum DIN rail
    "terminal_block",  # 1: Individual DIN rail terminal block
    "closer",          # 2: Insulating end cover / closer plate
    "end_clamp",       # 3: Mechanical end clamp / bracket
    "blue_wire",       # 4: Blue wire with ferrule
    "red_wire",        # 5: Red wire with ferrule
    "yellow_wire",     # 6: Yellow wire with ferrule
]

# Color palette for HUD visualization (BGR)
COMPONENT_COLORS_BGR = {
    "din_rail": (180, 180, 180),        # Silver/Gray
    "terminal_block": (140, 140, 140),  # Dark Gray
    "closer": (100, 180, 100),          # Muted Green
    "end_clamp": (200, 130, 70),        # Steel Blue
    "blue_wire": (235, 130, 40),        # Bright Blue/Cyan
    "red_wire": (40, 40, 235),          # Bright Red
    "yellow_wire": (40, 220, 240),      # Bright Yellow
}

# Sequential Assembly States (7 stages, 0 to 6)
ASSEMBLY_STATES = [
    "state_0_unstarted",
    "state_1_blocks_on_rail",
    "state_2_closed",
    "state_3_with_end_clamps",
    "state_4_blue_wire",
    "state_5_red_wire",
    "state_6_yellow_wire",
]

# Human-readable step titles for HUD and Dashboard
STEP_TITLES = {
    "state_0_unstarted": "0. Unstarted / Present DIN Rail",
    "state_1_blocks_on_rail": "1. 6 Terminal Blocks Mounted on Rail",
    "state_2_closed": "2. End Closer Plate Installed",
    "state_3_with_end_clamps": "3. End Clamps Fastened on Both Sides",
    "state_4_blue_wire": "4. Blue Wire Connected",
    "state_5_red_wire": "5. Red Wire Connected",
    "state_6_yellow_wire": "6. Yellow Wire Connected (Complete)",
}

# Video to Category Mapping
PARTS_VIDEO_MAPPING = {
    "din_rail.mp4": {"target": "din_rail", "desc": "35mm DIN rail"},
    "terminal_block.mp4": {"target": "terminal_block", "desc": "Individual terminal block"},
    "closer.mp4": {"target": "closer", "desc": "Insulating end plate / closer"},
    "end_clamps.mp4": {"target": "end_clamp", "desc": "Mechanical end clamp"},
    "blue_wire.mp4": {"target": "blue_wire", "desc": "Blue wire with ferrule"},
    "red_wire.mp4": {"target": "red_wire", "desc": "Red wire with ferrule"},
    "yellow_wire.mp4": {"target": "yellow_wire", "desc": "Yellow wire with ferrule"},
}

STATES_VIDEO_MAPPING = {
    "state1_blocks_on_rail.mp4": {"target": "state_1_blocks_on_rail", "desc": "6 Terminal blocks mounted on rail"},
    "state2_closed.mp4": {"target": "state_2_closed", "desc": "End closer attached to block group"},
    "state3_with_end_clamps.mp4": {"target": "state_3_with_end_clamps", "desc": "End clamps tightened on both ends"},
    "state4_blue_wire.mp4": {"target": "state_4_blue_wire", "desc": "Blue wire inserted into designated terminal"},
    "state5_red_wire.mp4": {"target": "state_5_red_wire", "desc": "Red wire inserted into designated terminal"},
    "state6_yellow_wire.mp4": {"target": "state_6_yellow_wire", "desc": "Yellow wire inserted into designated terminal"},
}

# Physical Assembly Parameters
TOTAL_TERMINAL_BLOCKS = 6

# Default expected wiring recipe per terminal slot
DEFAULT_WIRING_RECIPE = {
    "TB1": "BLUE",
    "TB2": "RED",
    "TB3": "YELLOW",
    "TB4": "EMPTY",
    "TB5": "EMPTY",
    "TB6": "EMPTY",
}
