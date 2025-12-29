"""
Configuration file for Fashion Detection Demo.
Centralized settings for model paths, thresholds, and server configuration.
"""
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model configuration
MODEL_DIR = BASE_DIR / "models"
MODEL_CHECKPOINT = MODEL_DIR / "epoch_30.pth"

# Model settings
MODEL_NAME = "hustvl/yolos-small"
NUM_LABELS = 47  # Fashionpedia categories
NUM_ATTRIBUTES = 294  # Fashionpedia attributes
HIDDEN_SIZE = 384  # YOLOS-small hidden dimension
DROPOUT = 0.1

# Inference settings
CONFIDENCE_THRESHOLD = 0.6  # Confidence threshold for detections (0-1)
NMS_THRESHOLD = 0.5         # IoU threshold for Non-Maximum Suppression
ATTRIBUTE_THRESHOLD = 0.5   # Threshold for attribute predictions (0-1)
IMAGE_SIZE = 512            # Target image size

# Data paths
DATA_DIR = BASE_DIR.parent / "data"
ANNOTATIONS_DIR = DATA_DIR / "raw" / "annotations"
ANNOTATION_FILE = ANNOTATIONS_DIR / "instances_attributes_val2020.json"
LABEL_DESCRIPTIONS_FILE = BASE_DIR / "label_descriptions.json"

# Alternative annotation file (if main dataset not available)
ANNOTATION_FILE_TRAIN = ANNOTATIONS_DIR / "instances_attributes_train2020.json"

# Server settings
HOST = "127.0.0.1"
PORT = 8000
RELOAD = True  # Auto-reload on code changes (development only)

# Upload settings
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB

# Create necessary directories
MODEL_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# Device configuration
DEVICE = "cuda"  # Will fall back to "cpu" if CUDA not available
