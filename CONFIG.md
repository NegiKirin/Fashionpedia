# Configuration Guide

This demo app uses a centralized configuration system for easy customization.

## Configuration File: `config.py`

All settings are managed in a single file for convenience.

### Model Settings

```python
# Model paths
MODEL_DIR = BASE_DIR / "models"
MODEL_CHECKPOINT = MODEL_DIR / "best_model.pth"

# Model architecture
MODEL_NAME = "hustvl/yolos-small"
NUM_LABELS = 47          # Fashion categories
NUM_ATTRIBUTES = 294     # Fashion attributes
HIDDEN_SIZE = 384        # Model hidden dimension
DROPOUT = 0.1           # Dropout rate
```

### Inference Settings

```python
CONFIDENCE_THRESHOLD = 0.3  # Detection confidence (0-1)
ATTRIBUTE_THRESHOLD = 0.5   # Attribute threshold (0-1)
IMAGE_SIZE = 512           # Target image size
```

### Data Paths

```python
ANNOTATION_FILE = ANNOTATIONS_DIR / "instances_attributes_val2020.json"
ANNOTATION_FILE_TRAIN = ANNOTATIONS_DIR / "instances_attributes_train2020.json"
```

### Server Settings

```python
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True  # Auto-reload on code changes
```

### Upload Settings

```python
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
```

## How Configuration is Used

### In `app.py` (FastAPI backend):
- Loads model from `config.MODEL_CHECKPOINT`
- Uses `config.CONFIDENCE_THRESHOLD` and `config.ATTRIBUTE_THRESHOLD`
- Runs server on `config.HOST:config.PORT`

### In `inference.py` (Detector):
- Initializes model with `config.MODEL_NAME`, `config.NUM_LABELS`, etc.
- Uses `config.IMAGE_SIZE` for preprocessing
- Applies thresholds from config

### In `preprocessing.py`:
- Uses `config.IMAGE_SIZE` for image resizing

## Customization Examples

### Change Detection Sensitivity

To get more detections (lower precision):
```python
CONFIDENCE_THRESHOLD = 0.2  # Lower = more detections
```

To get fewer, higher-quality detections:
```python
CONFIDENCE_THRESHOLD = 0.5  # Higher = fewer but more confident
```

### Change Attribute Predictions

```python
ATTRIBUTE_THRESHOLD = 0.3  # Lower = more attributes per item
```

### Use Different Model

```python
MODEL_NAME = "hustvl/yolos-tiny"  # Smaller, faster model
# or
MODEL_NAME = "hustvl/yolos-base"  # Larger, more accurate model
```

### Change Server Port

```python
PORT = 5000  # Run on different port
```

### Use Custom Checkpoint

```python
MODEL_CHECKPOINT = MODEL_DIR / "my_custom_model.pth"
```

## Benefits of Centralized Config

✅ **Single source of truth**: All settings in one place
✅ **Easy experimentation**: Change thresholds without editing code
✅ **Deployment flexibility**: Different configs for dev/prod
✅ **Type safety**: Pathlib for file paths
✅ **Auto-create directories**: Models and uploads folders created automatically

## Advanced: Environment-based Config

You can extend `config.py` to use environment variables:

```python
import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.3"))
PORT = int(os.getenv("PORT", "8000"))
```

Then run:
```bash
CONF_THRESHOLD=0.4 PORT=5000 python app.py
```
