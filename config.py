"""
Centralized Configuration for JLR VIN Project
==============================================
All paths and constants are defined here.
Import this module in all scripts to ensure consistency.

Usage:
    from config import *
    # or
    import config
    model = YOLO(config.YOLO_WEIGHTS)
"""

import os
from pathlib import Path

# =============================================================================
# BASE PATHS - ONLY CHANGE THESE IF YOU MOVE THE PROJECT
# =============================================================================

# Auto-detect project root (where this config.py file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data directory
DATA_ROOT = PROJECT_ROOT / "Data"

# =============================================================================
# DATASET PATHS
# =============================================================================

TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TRAIN_LABELS = DATA_ROOT / "train" / "labels"
VAL_IMAGES = DATA_ROOT / "val" / "images"
VAL_LABELS = DATA_ROOT / "val" / "labels"
TEST_IMAGES = DATA_ROOT / "test" / "images"
TEST_LABELS = DATA_ROOT / "test" / "labels"

# YOLO data config
DATA_YAML = PROJECT_ROOT / "data.yaml"

# =============================================================================
# MODEL PATHS
# =============================================================================

# Base weights
YOLOV8S_WEIGHTS = PROJECT_ROOT / "yolov8s.pt"
YOLOV8N_WEIGHTS = PROJECT_ROOT / "yolov11n.pt"

# Training output directories
YOLO_FIXED_SEED_DIR = PROJECT_ROOT / "YOLO_trained_fixed_seed"
YOLO_OPTUNA_DIR = PROJECT_ROOT / "YOLO_trained_optuna"

# Best trained models
YOLO_BEST_FIXED_SEED = YOLO_FIXED_SEED_DIR / "vin_yolov8s_best_fixed_seed.pt"
YOLO_BEST_OPTUNA = YOLO_OPTUNA_DIR / "vin_yolov8s_best_autotuned.pt"

# TrOCR model
TROCR_MODEL_DIR = PROJECT_ROOT / "best_trocr_v1" / "best_trocr_v1"

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

# Common settings
SEED = 12345
IMGSZ = 1024
BATCH = 16
DEVICE = 0  # GPU index, or "cpu"

# Fixed seed training
FIXED_SEED_EPOCHS = 50
F1_THRESHOLD = 0.997

# Optuna tuning
OPTUNA_N_TRIALS = 15
OPTUNA_EPOCHS_PER_TRIAL = 20
OPTUNA_FINAL_EPOCHS = 100

# =============================================================================
# CLASS NAMES (VIN Characters)
# =============================================================================

# Order MUST match data.yaml
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
    'Y', 'Z', '*'
]
NUM_CLASSES = len(CLASS_NAMES)  # 33

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_dirs():
    """Create necessary directories if they don't exist."""
    dirs = [
        DATA_ROOT,
        TRAIN_IMAGES, TRAIN_LABELS,
        VAL_IMAGES, VAL_LABELS,
        TEST_IMAGES, TEST_LABELS,
        YOLO_FIXED_SEED_DIR,
        YOLO_OPTUNA_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_data_yaml_content():
    """Generate data.yaml content with correct paths."""
    return f"""train: {TRAIN_IMAGES.as_posix()}
val: {VAL_IMAGES.as_posix()}
test: {TEST_IMAGES.as_posix()}
nc: {NUM_CLASSES}
names:
{chr(10).join(f"- '{n}'" for n in CLASS_NAMES)}
"""


def update_data_yaml():
    """Update data.yaml with current paths."""
    content = get_data_yaml_content()
    with open(DATA_YAML, 'w') as f:
        f.write(content)
    print(f"Updated {DATA_YAML}")


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("JLR VIN PROJECT CONFIGURATION")
    print("=" * 60)
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"DATA_ROOT:       {DATA_ROOT}")
    print(f"DATA_YAML:       {DATA_YAML}")
    print()
    print(f"TRAIN_IMAGES:    {TRAIN_IMAGES} (exists: {TRAIN_IMAGES.exists()})")
    print(f"VAL_IMAGES:      {VAL_IMAGES} (exists: {VAL_IMAGES.exists()})")
    print(f"TEST_IMAGES:     {TEST_IMAGES} (exists: {TEST_IMAGES.exists()})")
    print()
    print(f"YOLO_WEIGHTS:    {YOLOV8S_WEIGHTS} (exists: {YOLOV8S_WEIGHTS.exists()})")
    print(f"BEST_FIXED_SEED: {YOLO_BEST_FIXED_SEED} (exists: {YOLO_BEST_FIXED_SEED.exists()})")
    print(f"BEST_OPTUNA:     {YOLO_BEST_OPTUNA} (exists: {YOLO_BEST_OPTUNA.exists()})")
    print("=" * 60)


# =============================================================================
# AUTO-RUN ON IMPORT (optional)
# =============================================================================

if __name__ == "__main__":
    print_config()
    
    # Update data.yaml if requested
    import sys
    if "--update-yaml" in sys.argv:
        update_data_yaml()
