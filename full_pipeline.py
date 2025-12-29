# %%
# ============================================================
# Import Libraries
# ============================================================
import os
import json
import requests
import zipfile
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import io
from PIL import Image as PILImage
from PIL import Image
from tqdm.auto import tqdm
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, ToPILImage
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers import AutoModelForObjectDetection

import albumentations as A

# Configuration
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """
    Global Configuration for Fashion Detection System.
    Merges system config, model config, and training config.
    """
    # System
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Data Paths
    data_dir: str = 'data'
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    label_descriptions_path: str = 'label_descriptions.json'
    
    # Data Processing
    image_size: int = 512
    batch_size: int = 8
    use_autoaugment: bool = True
    normalize: bool = True
    
    # Model
    model_name: str = "hustvl/yolos-small"
    num_labels: int = 46  # Default Fashionpedia categories (0-45)
    num_attributes: int = 294
    hidden_size: int = 384
    dropout: float = 0.1
    
    # Training
    num_epochs: int = 15
    learning_rate: float = 2.5e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 0
    scheduler_type: str = 'cosine_restarts'
    num_cycles: int = 3
    grad_clip_norm: float = 1.0
    mixed_precision: bool = True
    
    # Loss Weights
    detection_weight: float = 1.0
    attribute_weight: float = 1.0
    
    # Checkpointing & Logging
    save_every_n_epochs: int = 5
    save_best: bool = True
    save_last: bool = True
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1

    # Target Categories (from label_descriptions.json)
    # Only these categories will be used for training/inference
    # To limit classes, comment out or remove items from this list
    target_categories: List[str] = field(default_factory=lambda: [
        "shirt, blouse",
        "top, t-shirt, sweatshirt",
        "sweater",
        "cardigan",
        "jacket",
        "vest",
        "pants",
        "shorts",
        "skirt",
        "coat",
        "dress",
        "jumpsuit",
        "cape",
        "glasses",
        "hat",
        "headband, head covering, hair accessory",
        "tie",
        "glove",
        "watch",
        "belt",
        "leg warmer",
        "tights, stockings",
        "sock",
        "shoe",
        "bag, wallet",
        "scarf",
        "umbrella",
        "hood",
        "collar",
        "lapel",
        "epaulette",
        "sleeve",
        "pocket",
        "neckline",
        "buckle",
        "zipper",
        "applique",
        "bead",
        "bow",
        "flower",
        "fringe",
        "ribbon",
        "rivet",
        "ruffle",
        "sequin",
        "tassel"
    ])

    def __post_init__(self):
        """Create directories after initialization."""
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

# Initialize global config
CONFIG = Config()

#%%
# ============================================================
# Downloader
# ============================================================
class FashionpediaDownloader:
    """
    Handles downloading and extracting Fashionpedia dataset.
    Combines all download logic into a reusable class.
    """
    
    # Dataset URLs
    ANNOTATION_URLS = {
        'train': 'https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json',
        'val': 'https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json',
    }
    
    IMAGE_URLS = {
        'train': 'https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip',
        'val_test': 'https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip',
    }
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize downloader with data directory.
        
        Args:
            data_dir: Root directory for data storage
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / 'raw'
        self.images_dir = self.raw_data_dir / 'images'
        self.annotations_dir = self.raw_data_dir / 'annotations'
        
        # Create directories
        for dir_path in [self.data_dir, self.raw_data_dir, self.images_dir, self.annotations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Data directory initialized: {self.data_dir}")
    
    @staticmethod
    def download_file(url: str, dest_path: Path, desc: str = "Downloading") -> bool:
        """
        Download file with progress bar.
        
        Args:
            url: URL to download from
            dest_path: Destination file path
            desc: Description for progress bar
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dest_path.exists():
            print(f"Already exists: {dest_path.name}")
            return True
        
        try:
            print(f"Downloading: {desc}")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f, tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=desc
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
            
            print(f"Downloaded: {dest_path.name} ({total_size / 1024**2:.1f} MB)")
            return True
            
        except Exception as e:
            print(f"Error downloading {dest_path.name}: {e}")
            if dest_path.exists():
                dest_path.unlink()
            return False
    
    @staticmethod
    def extract_zip(zip_path: Path, extract_to: Path, desc: str = "Extracting") -> bool:
        """
        Extract zip file with progress.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            desc: Description for progress bar
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not zip_path.exists():
            print(f"File not found: {zip_path}")
            return False
        
        try:
            print(f"{desc}: {zip_path.name}")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc=desc):
                    zip_ref.extract(member, extract_to)
            
            print(f"Extracted to: {extract_to}")
            return True
            
        except Exception as e:
            print(f"Error extracting {zip_path.name}: {e}")
            return False
    
    def download_annotations(self) -> bool:
        """
        Download all annotation files.
        
        Returns:
            bool: True if all successful
        """
        print("\nDownloading Annotations...")
        
        success = True
        for name, url in self.ANNOTATION_URLS.items():
            filename = url.split('/')[-1]
            dest_path = self.annotations_dir / filename
            if not self.download_file(url, dest_path, desc=f"Annotation - {name}"):
                success = False
        
        if success:
            print("All annotations downloaded!")
        return success
    
    def download_images(self, download: bool = False) -> bool:
        """
        Download and extract image files.
        
        Args:
            download: If False, skip download (default)
            
        Returns:
            bool: True if successful or skipped
        """
        if not download:
            print("Skipping image download (download=False). Set download=True to download images.")
            return True
        
        print("Downloading Images (this will take a while...)")
        
        success = True
        for name, url in self.IMAGE_URLS.items():
            filename = url.split('/')[-1]
            zip_path = self.raw_data_dir / filename
            
            # Download
            if not self.download_file(url, zip_path, desc=f"Images - {name}"):
                success = False
                continue
            
            # Extract
            if not self.extract_zip(zip_path, self.images_dir, desc=f"Extracting {name}"):
                success = False
            print()
        
        if success:
            print("All images downloaded and extracted!")
        return success
    
    def download_all(self, download_images: bool = False) -> Dict[str, Any]:
        """
        Download all dataset files.
        
        Args:
            download_images: Whether to download images (large files ~20GB)
            
        Returns:
            dict: Status of downloads with paths
        """
        # Download annotations
        annotations_success = self.download_annotations()
        
        # Download images
        images_success = self.download_images(download=download_images)
        
        return {
            'annotations_success': annotations_success,
            'images_success': images_success,
            'annotations_dir': str(self.annotations_dir),
            'images_dir': str(self.images_dir),
        }

#%%
# ============================================================
# Preprocessing
# ============================================================
def fix_channels(t: torch.Tensor) -> Image.Image:
    """
    Convert images to 3-channel RGB format.
    Handles grayscale (1 channel) and RGBA (4 channels).
    
    Args:
        t: Image tensor
        
    Returns:
        PIL Image in RGB format
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)


def xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox from [x1, y1, x2, y2] to [center_x, center_y, width, height].
    
    Args:
        box: Bounding boxes tensor
        
    Returns:
        Converted bounding boxes
    """
    x1, y1, x2, y2 = box.unbind(dim=1)
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width * 0.5
    cy = y1 + height * 0.5
    return torch.stack([cx, cy, width, height], dim=1)


def cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Convert bbox from [center_x, center_y, width, height] to [x1, y1, x2, y2].
    
    Args:
        x: Bounding boxes tensor
        
    Returns:
        Converted bounding boxes
    """
    cx, cy, w, h = x.unbind(1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size: Tuple[int, int], down: bool = True) -> torch.Tensor:
    """
    Rescale bounding boxes between normalized [0,1] and pixel coordinates.
    
    Args:
        out_bbox: Bounding boxes
        size: (width, height) of image
        down: If True, normalize to [0,1]. If False, convert to pixels.
        
    Returns:
        Rescaled bounding boxes
    """
    img_w, img_h = size
    if down:
        b = torch.Tensor(out_bbox) / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    else:
        b = torch.Tensor(out_bbox) * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def create_id_to_idx_mapping(items: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Create bidirectional mapping between IDs and continuous indices.
    
    This is useful when IDs have gaps (e.g., attributes with IDs [0,1,2,5,6,10])
    to convert to continuous indices [0,1,2,3,4,5].
    
    Args:
        items: List of items with 'id' field
        
    Returns:
        Tuple of (id_to_idx, idx_to_id) mappings
    """
    # Sort by ID to ensure consistent ordering
    sorted_items = sorted(items, key=lambda x: x['id'])
    
    id_to_idx = {}
    idx_to_id = {}
    
    for idx, item in enumerate(sorted_items):
        item_id = item['id']
        id_to_idx[item_id] = idx
        idx_to_id[idx] = item_id
    
    return id_to_idx, idx_to_id


def create_attribute_mapping(attributes: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, Any]:
    """
    Create attribute ID to index mapping and return useful info.
    
    Args:
        attributes: List of attribute dictionaries
        verbose: Whether to print mapping info
        
    Returns:
        dict: Mapping information including id_to_idx, idx_to_id, and stats
    """
    attr_id_to_idx, attr_idx_to_id = create_id_to_idx_mapping(attributes)
    
    # Extract IDs and compute stats
    ids = sorted([attr['id'] for attr in attributes])
    min_id = min(ids)
    max_id = max(ids)
    num_attrs = len(attributes)
    
    # Check for gaps
    expected_count = max_id + 1 if min_id == 0 else max_id - min_id + 1
    has_gaps = (num_attrs != expected_count)
    
    result = {
        'id_to_idx': attr_id_to_idx,
        'idx_to_id': attr_idx_to_id,
        'num_attributes': num_attrs,
        'min_id': min_id,
        'max_id': max_id,
        'has_gaps': has_gaps,
        'expected_count': expected_count,
        'gap_count': expected_count - num_attrs if has_gaps else 0,
    }
    
    if verbose:
        print("=" * 70)
        print("ATTRIBUTE ID -> INDEX MAPPING")
        print("=" * 70)
        print(f"Total attributes: {num_attrs}")
        print(f"ID range: {min_id} to {max_id}")
        print(f"Expected count (if continuous): {expected_count}")
        
        if has_gaps:
            print(f"Has {result['gap_count']} gaps in ID sequence")
            print(f"Created mapping: sparse IDs -> continuous indices [0, {num_attrs-1}]")
            print(f"\nExample mappings:")
            for i, (id_val, idx) in enumerate(list(attr_id_to_idx.items())[:5]):
                attr_name = next(a['name'] for a in attributes if a['id'] == id_val)
                print(f"  ID {id_val:3d} -> idx {idx:3d} : {attr_name}")
            if len(attr_id_to_idx) > 5:
                print(f"  ... and {len(attr_id_to_idx) - 5} more")
        else:
            print(f"IDs are continuous, mapping is identity function")
        
        print("=" * 70)
    
    return result


class AutoAugmentBBox:
    """
    AutoAugment for Object Detection with bounding box awareness.
    Uses albumentations for bbox-aware transformations.
    """
    
    def __init__(self, policies: str = 'coco'):
        """
        Initialize AutoAugment with predefined policies.
        
        Args:
            policies: 'coco' for AutoAugment-COCO policies (default)
        """
        self.policies = self._get_policies(policies)
    
    def _get_policies(self, policy_name: str) -> List[List[Tuple]]:
        """
        Get AutoAugment policies optimized for object detection.
        Based on AutoAugment-COCO from the paper.
        """
        if policy_name == 'coco':
            # AutoAugment-COCO policies
            # Each policy contains 2 operations: (operation, probability, magnitude)
            policies = [
                # Policy 1: Basic geometric
                [
                    ('TranslateX', 0.6, 4),
                    ('Equalize', 0.8, 0),
                ],
                # Policy 2: Color + geometric
                [
                    ('TranslateY', 0.2, 9),
                    ('Color', 0.6, 6),
                ],
                # Policy 3: Rotation
                [
                    ('Rotate', 0.8, 9),
                    ('Equalize', 0.4, 0),
                ],
                # Policy 4: Shear + color
                [
                    ('ShearX', 0.4, 7),
                    ('Brightness', 0.6, 7),
                ],
                # Policy 5: Color enhancement
                [
                    ('Sharpness', 0.8, 8),
                    ('Color', 0.4, 3),
                ],
                # Policy 6: Rotation + Shear
                [
                    ('Rotate', 0.7, 5),
                    ('ShearY', 0.5, 6),
                ],
                # Policy 7: Equalize + Brightness
                [
                    ('Equalize', 0.8, 0),
                    ('Brightness', 0.6, 5),
                ],
                # Policy 8: Color manipulations
                [
                    ('Color', 0.3, 9),
                    ('Sharpness', 0.7, 4),
                ],
                # Policy 9: Strong geometric
                [
                    ('Rotate', 0.8, 7),
                    ('TranslateY', 0.9, 6),
                ],
                # Policy 10: Contrast + Shear
                [
                    ('Contrast', 0.5, 8),
                    ('ShearX', 0.7, 3),
                ],
            ]
        else:
            raise ValueError(f"Unknown policy: {policy_name}")
        
        return policies
    
    def _operation_to_albumentation(self, operation: str, magnitude: int) -> Optional[A.BasicTransform]:
        """
        Convert operation name and magnitude to albumentations transform.
        Magnitude ranges from 0-10, we scale it appropriately for each operation.
        """
        import random
        
        # Magnitude scaling for different operations
        mag_scale = magnitude / 10.0  # Normalize to [0, 1]
        
        if operation == 'Rotate':
            limit = int(30 * mag_scale)
            return A.Rotate(limit=limit, border_mode=0, p=1.0)
        
        elif operation == 'ShearX':
            shear = 0.3 * mag_scale
            return A.Affine(shear={'x': (-shear, shear)}, mode=0, p=1.0)
        
        elif operation == 'ShearY':
            shear = 0.3 * mag_scale
            return A.Affine(shear={'y': (-shear, shear)}, mode=0, p=1.0)
        
        elif operation == 'TranslateX':
            shift = 0.3 * mag_scale
            return A.Affine(translate_percent={'x': (-shift, shift)}, mode=0, p=1.0)
        
        elif operation == 'TranslateY':
            shift = 0.3 * mag_scale
            return A.Affine(translate_percent={'y': (-shift, shift)}, mode=0, p=1.0)
        
        elif operation == 'Color':
            factor = 0.9 * mag_scale
            return A.ColorJitter(
                brightness=0, contrast=0, saturation=factor, hue=factor * 0.1, p=1.0
            )
        
        elif operation == 'Brightness':
            limit = 0.4 * mag_scale
            return A.RandomBrightnessContrast(
                brightness_limit=limit, contrast_limit=0, p=1.0
            )
        
        elif operation == 'Contrast':
            limit = 0.4 * mag_scale
            return A.RandomBrightnessContrast(
                brightness_limit=0, contrast_limit=limit, p=1.0
            )
        
        elif operation == 'Sharpness':
            alpha = (0.1, 0.5 * mag_scale + 0.5)
            return A.Sharpen(alpha=alpha, lightness=(0.5, 1.5), p=1.0)
        
        elif operation == 'Equalize':
            return A.Equalize(p=1.0)
        
        else:
            return None
    
    def __call__(self, image: np.ndarray, bboxes: List, category_ids: List) -> Tuple:
        """
        Apply a random sub-policy to image and bboxes.
        
        Args:
            image: numpy array (H, W, C)
            bboxes: list of bboxes in format [x_min, y_min, x_max, y_max] (normalized 0-1)
            category_ids: list of category IDs for each bbox
        
        Returns:
            Augmented image, bboxes, category_ids
        """
        import random
        
        # Select random sub-policy
        policy = random.choice(self.policies)
        
        # Build albumentations pipeline for this sub-policy
        transforms_list = []
        for operation, prob, magnitude in policy:
            if random.random() < prob:
                transform = self._operation_to_albumentation(operation, magnitude)
                if transform is not None:
                    transforms_list.append(transform)
        
        if len(transforms_list) == 0:
            return image, bboxes, category_ids
        
        # Create composition with bbox support
        transform = A.Compose(
            transforms_list,
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['category_ids'],
                min_visibility=0.3,
            )
        )
        
        # Convert normalized bboxes to pixel coordinates
        h, w = image.shape[:2]
        bboxes_pixel = []
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            bboxes_pixel.append([
                x_min * w,
                y_min * h,
                x_max * w,
                y_max * h,
            ])
        
        # Apply transformation
        try:
            transformed = transform(
                image=image,
                bboxes=bboxes_pixel,
                category_ids=category_ids
            )
            
            # Convert back to normalized coordinates
            image_aug = transformed['image']
            bboxes_aug = []
            h_aug, w_aug = image_aug.shape[:2]
            
            for bbox in transformed['bboxes']:
                x_min, y_min, x_max, y_max = bbox
                bboxes_aug.append([
                    x_min / w_aug,
                    y_min / h_aug,
                    x_max / w_aug,
                    y_max / h_aug,
                ])
            
            return image_aug, bboxes_aug, transformed['category_ids']
        
        except Exception as e:
            # If augmentation fails, return original
            print(f"Warning: AutoAugment failed with error {e}, returning original")
            return image, bboxes, category_ids

class FashionpediaPreprocessor:
    """
    Complete preprocessing pipeline for Fashionpedia dataset.
    Handles image loading, augmentation, and label preparation.
    """
    
    def __init__(
        self,
        use_autoaugment: bool = True,
        image_size: int = 512,
        num_attributes: int = 294,
        attr_id_to_idx: Optional[Dict[int, int]] = None,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize preprocessor.
        
        Args:
            use_autoaugment: Whether to use AutoAugment
            image_size: Target image size
            num_attributes: Number of attribute classes
            attr_id_to_idx: Mapping from attribute IDs to continuous indices
            normalize: Whether to normalize images (default: True)
            mean: Mean for normalization (default: ImageNet)
            std: Std for normalization (default: ImageNet)
        """
        self.use_autoaugment = use_autoaugment
        self.image_size = image_size
        self.num_attributes = num_attributes
        self.attr_id_to_idx = attr_id_to_idx
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Transforms
        from torchvision import transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=mean, std=std) if normalize else None
        
        # Initialize AutoAugment
        self.autoaugment = AutoAugmentBBox(policies='coco') if use_autoaugment else None
        
        print(f"Preprocessor initialized")
        print(f"  - AutoAugment: {use_autoaugment}")
        print(f"  - Image size: {image_size}")
        print(f"  - Num attributes: {num_attributes}")
        if attr_id_to_idx is not None:
            print(f"  - Using attribute ID->idx mapping ({len(attr_id_to_idx)} entries)")
        if normalize:
            print(f"  - Normalization: Enabled (mean={mean}, std={std})")
    
    def resize_with_padding(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        target_size: int = None,
    ) -> Tuple[Image.Image, List[List[float]], Tuple[int, int]]:
        """
        Resize image to target size with padding to preserve aspect ratio.
        Also adjusts bounding boxes accordingly.
        
        Args:
            image: PIL Image
            bboxes: List of bboxes [x1, y1, x2, y2] in pixel coordinates
            target_size: Target size (default: self.image_size)
        
        Returns:
            Tuple of (resized_image, adjusted_bboxes, (new_width, new_height))
        """
        if target_size is None:
            target_size = self.image_size
        
        # Get original size
        orig_width, orig_height = image.size
        
        # Calculate resize ratio (keep aspect ratio)
        ratio = min(target_size / orig_width, target_size / orig_height)
        new_width = int(orig_width * ratio)
        new_height = int(orig_height * ratio)
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create padded image
        padded_image = Image.new('RGB', (target_size, target_size), color=(0, 0, 0))
        
        # Calculate padding
        pad_left = (target_size - new_width) // 2
        pad_top = (target_size - new_height) // 2
        
        # Paste resized image onto padded image
        padded_image.paste(resized_image, (pad_left, pad_top))
        
        # Adjust bboxes
        adjusted_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Scale coordinates
            x1_new = x1 * ratio + pad_left
            y1_new = y1 * ratio + pad_top
            x2_new = x2 * ratio + pad_left
            y2_new = y2 * ratio + pad_top
            
            # Ensure valid bbox (width and height > 0)
            if x2_new <= x1_new:
                x2_new = x1_new + 1.0
            if y2_new <= y1_new:
                y2_new = y1_new + 1.0
                
            adjusted_bboxes.append([x1_new, y1_new, x2_new, y2_new])
        
        return padded_image, adjusted_bboxes, (target_size, target_size)
    
    def preprocess_image(
        self,
        image: Image.Image,
        bboxes: List[List[float]],
        categories: List[int],
        attributes: List[List[int]] = None,
        width: int = None,
        height: int = None,
        is_training: bool = True,
    ) -> Dict[str, Any]:
        """
        Preprocess a single image with its annotations.
        
        Args:
            image: PIL Image
            bboxes: List of bboxes [x1, y1, x2, y2] in pixel coordinates
            categories: List of category IDs
            attributes: List of attribute ID lists for each object
            width: Original image width
            height: Original image height
            is_training: Whether to apply training augmentations
            
        Returns:
            dict: Preprocessed data with image and labels
        """
        # Fix channels
        image_tensor = self.to_tensor(image)
        image = fix_channels(image_tensor)
        image_np = np.array(image)
        
        # Get dimensions
        if width is None or height is None:
            height, width = image_np.shape[:2]
        
        # Resize with padding to ensure consistent size
        # This prevents batching errors from variable image sizes
        image, bboxes, (new_width, new_height) = self.resize_with_padding(
            image=image,
            bboxes=bboxes,
            target_size=self.image_size,
        )
        
        # Update dimensions after resize
        width, height = new_width, new_height
        
        # Convert back to numpy for augmentation
        image_np = np.array(image)
        
        # Normalize bboxes to [0, 1]
        bboxes_normalized = []
        valid_indices = []
        
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            
            # Additional safety check
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Clamp to image boundaries
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            bboxes_normalized.append([
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ])
            valid_indices.append(i)
            
        # Filter associated data
        categories = [categories[i] for i in valid_indices]
        if attributes is not None:
             attributes = [attributes[i] for i in valid_indices]
        
        # Apply AutoAugment during training
        if is_training and self.autoaugment is not None:
            image_np, bboxes_normalized, categories = self.autoaugment(
                image_np, bboxes_normalized, categories
            )
            image = Image.fromarray(image_np)
        
        # Convert to tensor and normalize
        # Note: image_np is uint8 [0, 255], ToTensor converts to float [0, 1]
        final_image_tensor = self.to_tensor(image)
        
        if self.normalize_transform is not None:
            final_image_tensor = self.normalize_transform(final_image_tensor)
            
        # Update image_np for consistency (though tensor is usually what we want)
        # If normalized, image_np will differ from tensor
        if not self.normalize:
             image_np = (final_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Prepare attribute labels
        attribute_labels = torch.zeros((len(categories), self.num_attributes), dtype=torch.float32)
        if attributes and len(attributes) == len(categories):
            for i, attr_list in enumerate(attributes):
                if attr_list:
                    # Convert attribute IDs to indices if mapping exists
                    if self.attr_id_to_idx is not None:
                        # Use mapping to convert sparse IDs to continuous indices
                        valid_indices = []
                        for attr_id in attr_list:
                            if attr_id in self.attr_id_to_idx:
                                idx = self.attr_id_to_idx[attr_id]
                                if 0 <= idx < self.num_attributes:
                                    valid_indices.append(idx)
                        if valid_indices:
                            attribute_labels[i, valid_indices] = 1.0
                    else:
                        # Assume IDs are already continuous indices
                        valid_attrs = [a for a in attr_list if 0 <= a < self.num_attributes]
                        if valid_attrs:
                            attribute_labels[i, valid_attrs] = 1.0
        
        # Convert to tensors
        bboxes_tensor = torch.tensor(bboxes_normalized, dtype=torch.float32)
        if len(bboxes_tensor) > 0:
            bboxes_cxcywh = xyxy_to_cxcywh(bboxes_tensor)
        else:
            bboxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
        
        return {
            'image': final_image_tensor,  # Now a normalized Tensor [C, H, W]
            'image_np': image_np,         # Numpy array for visualization
            'bboxes': bboxes_cxcywh,
            'bboxes_xyxy': bboxes_tensor,
            'categories': torch.tensor(categories, dtype=torch.long),
            'attributes': attribute_labels,
            'orig_size': (height, width),  # Padded size (will be same for all images)
        }

#%%
# =============
# Dataset
# =============

class FashionpediaDataset(Dataset):
    """
    PyTorch Dataset for Fashionpedia.
    
    Features:
    - Lazy loading of images
    - Optional image caching
    - Integrated preprocessing and augmentation
    - COCO format support
    - Attribute labels support
    
    Usage:
        dataset = FashionpediaDataset(
            annotation_file='data/annotations/train.json',
            images_dir='data/images',
            split='train',
            use_autoaugment=True,
        )
        dataloader = dataset.get_dataloader(batch_size=16)
    """
    
    def __init__(
        self,
        annotation_file: str,
        images_dir: str = 'data/raw/images',
        split: str = 'train',
        use_autoaugment: bool = True,
        image_size: int = 512,
        num_attributes: int = 294,
        cache_images: bool = False,
        load_attributes: bool = True,
        min_object_size: int = 0,
        transform: Optional[FashionpediaPreprocessor] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            annotation_file: Path to COCO format annotation JSON file
            images_dir: Directory containing images
            split: Dataset split ('train', 'val', 'test')
            use_autoaugment: Whether to use AutoAugment
            image_size: Target image size
            num_attributes: Number of attribute classes
            cache_images: Whether to cache images in memory
            load_attributes: Whether to load attribute labels
            min_object_size: Minimum object area for filtering
            transform: Optional FashionpediaPreprocessor, will create default if None
        """
        self.annotation_file = annotation_file
        self.images_dir = Path(images_dir)
        self.split = split
        self.use_autoaugment = use_autoaugment
        self.image_size = image_size
        self.num_attributes = num_attributes
        self.cache_images = cache_images
        self.load_attributes = load_attributes
        self.min_object_size = min_object_size
        
        # Load annotations
        print(f"Loading annotations from: {annotation_file}")
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Extract data
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        self.attributes = self.coco_data['attributes']
        
        # Create attribute ID to index mapping
        self._create_attribute_mapping()
        
        # Create other mappings
        self._create_mappings()
        
        # Setup preprocessor
        if transform is None:
            self.preprocessor = FashionpediaPreprocessor(
                use_autoaugment=use_autoaugment,
                image_size=image_size,
                num_attributes=self.num_attributes,  # Use actual count after mapping
                attr_id_to_idx=self.attr_id_to_idx,  # Pass the mapping
                normalize=True,
            )
        else:
            self.preprocessor = transform
        
        # Image cache
        self.image_cache = {} if cache_images else None
        
        print(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
        print(f"Categories: {len(self.categories)}, Attributes: {len(self.attributes)}")
    
    def _create_attribute_mapping(self):
        """Create attribute ID to index mapping to handle sparse IDs."""
        
        # Create mapping
        mapping_info = create_attribute_mapping(self.attributes, verbose=True)
        
        self.attr_id_to_idx = mapping_info['id_to_idx']
        self.attr_idx_to_id = mapping_info['idx_to_id']
        self.has_attribute_gaps = mapping_info['has_gaps']
        
        # Verify actual number of attributes matches parameter
        actual_num_attrs = len(self.attributes)
        if actual_num_attrs != self.num_attributes:
            print(f"Warning: num_attributes parameter ({self.num_attributes}) "
                  f"!= actual attributes ({actual_num_attrs})")
            print(f"   Updating to match actual count: {actual_num_attrs}")
            self.num_attributes = actual_num_attrs
    
    def _create_mappings(self):
        """Create useful mappings for fast lookup."""
        # Image ID to annotations
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)
        
        # Category ID to name
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in self.categories}
        self.cat_name_to_id = {cat['name']: cat['id'] for cat in self.categories}
        
        # Attribute ID to name
        self.attr_id_to_name = {attr['id']: attr['name'] for attr in self.attributes}
        
        # Filter images with no annotations
        self.valid_image_ids = [
            img['id'] for img in self.images 
            if img['id'] in self.image_id_to_annotations
        ]
        
        # Image ID to index mapping
        self.image_id_to_idx = {img_id: idx for idx, img_id in enumerate(self.valid_image_ids)}
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.valid_image_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Sample with image and labels
        """
        # Get image ID
        image_id = self.valid_image_ids[idx]
        
        # Get image info
        image_info = next(img for img in self.images if img['id'] == image_id)
        
        # Load image
        image = self._load_image(image_info)
        
        # Get annotations for this image
        annotations = self.image_id_to_annotations[image_id]
        
        # Extract bboxes, categories, attributes
        bboxes = []
        categories = []
        attributes = []
        areas = []
        
        for ann in annotations:
            # COCO format: [x, y, w, h] -> convert to [x1, y1, x2, y2]
            x, y, w, h = ann['bbox']
            
            # Filter small objects if needed
            if self.min_object_size > 0:
                area = w * h
                if area < self.min_object_size:
                    continue
            
            bboxes.append([x, y, x + w, y + h])
            categories.append(ann['category_id'])
            
            if self.load_attributes:
                attributes.append(ann.get('attribute_ids', []))
            
            areas.append(ann.get('area', w * h))
        
        # Skip if no valid objects
        if len(bboxes) == 0:
            # Return empty tensors
            return {
                'image_id': image_id,
                'image': image,
                'image_np': np.array(image),
                'bboxes': torch.zeros((0, 4), dtype=torch.float32),
                'bboxes_xyxy': torch.zeros((0, 4), dtype=torch.float32),
                'categories': torch.zeros((0,), dtype=torch.long),
                'attributes': torch.zeros((0, self.num_attributes), dtype=torch.float32),
                'areas': torch.zeros((0,), dtype=torch.float32),
                'orig_size': (image_info['height'], image_info['width']),
            }
        
        # Preprocess
        is_training = (self.split == 'train')
        preprocessed = self.preprocessor.preprocess_image(
            image=image,
            bboxes=bboxes,
            categories=categories,
            attributes=attributes if self.load_attributes else None,
            width=image_info['width'],
            height=image_info['height'],
            is_training=is_training,
        )
        
        # Add extra info
        preprocessed['image_id'] = image_id
        preprocessed['areas'] = torch.tensor(areas, dtype=torch.float32)
        
        return preprocessed
    
    def _load_image(self, image_info: Dict) -> Image.Image:
        """
        Load image from disk or cache.
        
        Args:
            image_info: Image metadata dict
            
        Returns:
            PIL Image
        """
        image_id = image_info['id']
        
        # Check cache
        if self.image_cache is not None and image_id in self.image_cache:
            return self.image_cache[image_id]
        
        # Construct image path
        filename = image_info['file_name']
        
        # Try different possible paths
        possible_paths = [
            self.images_dir / filename,
            self.images_dir / 'train' / filename,
            self.images_dir / 'test' / filename,
        ]
        
        image_path = None
        for path in possible_paths:
            if path.exists():
                image_path = path
                break
        
        if image_path is None:
            # Return dummy image if not found
            print(f"Warning: Image not found: {filename}, using dummy image")
            image = Image.new('RGB', (image_info['width'], image_info['height']), color='gray')
        else:
            image = Image.open(image_path).convert('RGB')
        
        # Cache if enabled
        if self.image_cache is not None:
            self.image_cache[image_id] = image
        
        return image
    
    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return [cat['name'] for cat in self.categories]
    
    def get_category_name(self, cat_id: int) -> str:
        """Get category name by ID."""
        return self.cat_id_to_name.get(cat_id, 'Unknown')
    
    def get_dataloader(
        self,
        batch_size: int = 8,
        shuffle: bool = None,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader for this dataset.
        
        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle (auto-set based on split if None)
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            
        Returns:
            DataLoader instance
        """
        if shuffle is None:
            shuffle = (self.split == 'train')
        
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collate_fn,
        )
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function for DataLoader.
        Handles variable number of objects per image.
        
        Args:
            batch: List of samples
            
        Returns:
            Batched data
        """
        # Stack images
        images = [sample['image'] for sample in batch]
        
        # Images might have different sizes, so we keep them as list
        # Or you can pad/resize them here
        
        # Collect per-image data
        collated = {
            'image_ids': [sample['image_id'] for sample in batch],
            'images': images,   # List of Tensors now
            'bboxes': [sample['bboxes'] for sample in batch],
            'bboxes_xyxy': [sample['bboxes_xyxy'] for sample in batch],
            'categories': [sample['categories'] for sample in batch],
            'attributes': [sample['attributes'] for sample in batch],
            'areas': [sample['areas'] for sample in batch],
            'orig_sizes': [sample['orig_size'] for sample in batch],
        }
        
        return collated
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            dict: Statistics about the dataset
        """
        # Category distribution
        cat_counts = {}
        for ann in self.annotations:
            cat_id = ann['category_id']
            cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
        
        # Object sizes
        areas = [ann.get('area', ann['bbox'][2] * ann['bbox'][3]) for ann in self.annotations]
        
        SMALL_THRESHOLD = 32 * 32
        MEDIUM_THRESHOLD = 96 * 96
        
        small_count = sum(1 for area in areas if area < SMALL_THRESHOLD)
        medium_count = sum(1 for area in areas if SMALL_THRESHOLD <= area < MEDIUM_THRESHOLD)
        large_count = sum(1 for area in areas if area >= MEDIUM_THRESHOLD)
        
        return {
            'num_images': len(self.valid_image_ids),
            'num_annotations': len(self.annotations),
            'num_categories': len(self.categories),
            'num_attributes': len(self.attributes),
            'avg_objects_per_image': len(self.annotations) / len(self.valid_image_ids),
            'category_distribution': cat_counts,
            'size_distribution': {
                'small': small_count,
                'medium': medium_count,
                'large': large_count,
                'small_pct': small_count / len(areas) * 100,
                'medium_pct': medium_count / len(areas) * 100,
                'large_pct': large_count / len(areas) * 100,
            }
        }

def create_train_val_datasets(
    data_dir: str = 'data',
    batch_size: int = 8,
    use_autoaugment: bool = True,
    **kwargs
) -> Tuple[FashionpediaDataset, FashionpediaDataset]:
    """
    Create train and validation datasets with matching parameters.
    
    Args:
        data_dir: Data directory
        batch_size: Batch size (passed to get_dataloader)
        use_autoaugment: Whether to use AutoAugment
        **kwargs: Additional parameters for FashionpediaDataset
        
    Returns:
        Tuple of (train_dataset, val_dataset)
        
    Note:
        batch_size is stored but not used by Dataset itself,
        pass it to get_dataloader() when needed
    """
    annotations_dir = f"{data_dir}/raw/annotations"
    images_dir = f"{data_dir}/raw/images"
    
    # Train dataset
    train_dataset = FashionpediaDataset(
        annotation_file=f"{annotations_dir}/instances_attributes_train2020.json",
        images_dir=images_dir,
        split='train',
        use_autoaugment=use_autoaugment,
        **kwargs
    )
    
    # Val dataset (no augmentation)
    val_dataset = FashionpediaDataset(
        annotation_file=f"{annotations_dir}/instances_attributes_val2020.json",
        images_dir=images_dir,
        split='val',
        use_autoaugment=False,  # No augmentation for validation
        **kwargs
    )
    
    # Store batch_size for convenience (can be accessed later)
    train_dataset.batch_size = batch_size
    val_dataset.batch_size = batch_size
    
    return train_dataset, val_dataset


def print_dataset_info(dataset: FashionpediaDataset):
    """
    Print detailed information about a dataset.
    
    Args:
        dataset: FashionpediaDataset instance
    """
    print("=" * 70)
    print(f"DATASET INFO - {dataset.split.upper()}")
    print("=" * 70)
    
    stats = dataset.get_stats()
    
    print(f"\nBasic Statistics:")
    print(f"   Images: {stats['num_images']:,}")
    print(f"   Annotations: {stats['num_annotations']:,}")
    print(f"   Categories: {stats['num_categories']}")
    print(f"   Attributes: {stats['num_attributes']}")
    print(f"   Avg objects/image: {stats['avg_objects_per_image']:.2f}")
    
    print(f"\nObject Size Distribution:")
    print(f"   Small (<32x32): {stats['size_distribution']['small']:,} ({stats['size_distribution']['small_pct']:.1f}%)")
    print(f"   Medium (32-96): {stats['size_distribution']['medium']:,} ({stats['size_distribution']['medium_pct']:.1f}%)")
    print(f"   Large (>96x96): {stats['size_distribution']['large']:,} ({stats['size_distribution']['large_pct']:.1f}%)")
    
    print(f"\nDataset Parameters:")
    batch_size = getattr(dataset, 'batch_size', 'Not set')
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {dataset.image_size}")
    print(f"   AutoAugment: {dataset.use_autoaugment}")
    print(f"   Cache images: {dataset.cache_images}")
    print(f"   Load attributes: {dataset.load_attributes}")
    
    print("=" * 70)

#%%
# ============================================================
# Model
# ============================================================
class FashionDetectionModel(nn.Module):
    """
    YOLOS-based Fashion Detection Model with Attribute Prediction.
    
    This is a pure PyTorch implementation that extends YOLOS with
    an attribute classification head for fine-grained attribute recognition.
    
    Architecture:
        YOLOS Base Model (Vision Transformer)
            ↓
        Detection Head (bounding boxes + category labels)
            ↓
        Attribute Head (multi-label attribute classification)
    
    Args:
        model_name: HuggingFace model name (default: 'hustvl/yolos-small')
        num_labels: Number of object categories (default: 47 for Fashionpedia)
        num_attributes: Number of fine-grained attributes (default: 294)
        hidden_size: Hidden dimension size (default: 384 for YOLOS-small)
        dropout: Dropout probability (default: 0.1)
    
    Inputs:
        pixel_values: Tensor [B, 3, H, W] - Batch of images
        labels: Optional list of dicts with ground truth (for training)
    
    Outputs:
        Dictionary containing:
            - logits: [B, num_queries, num_labels+1] - Detection logits
            - pred_boxes: [B, num_queries, 4] - Predicted boxes (cxcywh normalized)
            - attribute_logits: [B, num_queries, num_attributes] - Attribute logits
            - loss: Scalar (only if labels provided)
            - last_hidden_state: [B, seq_len, hidden_size] - Encoder outputs
    """
    
    def __init__(
        self,
        model_name: str = 'hustvl/yolos-small',
        num_labels: int = 47,
        num_attributes: int = 294,
        hidden_size: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.num_attributes = num_attributes
        self.hidden_size = hidden_size
        
        # Load pretrained YOLOS model
        print(f"\nLoading YOLOS model: {model_name}")
        self.yolos = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        
        # Attribute prediction head
        # Takes encoder outputs and predicts multi-label attributes
        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_attributes),
        )
        
        print(f"Initialized Fashion Detection Model")
        print(f"  - Detection categories: {num_labels}")
        print(f"  - Attribute classes: {num_attributes}")
        print(f"  - Hidden dimension: {hidden_size}")
        print(f"  - Dropout: {dropout}")
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            pixel_values: Input images [B, 3, H, W]
            labels: Optional ground truth labels (for training)
        
        Returns:
            Dictionary with model outputs
        """
        # Detection branch - forward through YOLOS
        detection_outputs = self.yolos(
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=True,
        )
        
        # Extract encoder outputs for attribute prediction
        # Shape: [B, sequence_length, hidden_size]
        encoder_outputs = detection_outputs.last_hidden_state
        
        # Get the detection token queries
        # YOLOS architecture: [CLS] + [patch_tokens] + [DET_tokens]
        # We need only the [DET_tokens] which correspond to object queries
        num_queries = detection_outputs.logits.shape[1]
        detection_tokens = encoder_outputs[:, -num_queries:, :]
        
        # Attribute prediction
        # Predict attributes for each detection query
        # Shape: [B, num_queries, num_attributes]
        attribute_logits = self.attribute_head(detection_tokens)
        
        # Prepare output dictionary
        outputs = {
            'logits': detection_outputs.logits,  # [B, num_queries, num_labels+1]
            'pred_boxes': detection_outputs.pred_boxes,  # [B, num_queries, 4]
            'attribute_logits': attribute_logits,  # [B, num_queries, num_attributes]
            'last_hidden_state': encoder_outputs,  # [B, seq_len, hidden_size]
        }
        
        # Include detection loss if labels provided
        if labels is not None and hasattr(detection_outputs, 'loss'):
            outputs['detection_loss'] = detection_outputs.loss
            if hasattr(detection_outputs, 'loss_dict'):
                outputs['loss_dict'] = detection_outputs.loss_dict
        
        return outputs

#%%
# ============================================================
# Loss
# ============================================================
def compute_fashion_loss(
    outputs: Dict[str, torch.Tensor],
    labels: List[Dict[str, torch.Tensor]],
    detection_weight: float = 1.0,
    attribute_weight: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined loss for fashion detection.
    
    The total loss is a weighted combination of:
    1. Detection loss (from YOLOS): bbox regression + classification
    2. Attribute loss: multi-label BCE with Hungarian matching
    
    Hungarian Matching Strategy:
    - Match predicted queries to ground truth objects based on bbox L1 distance
    - Only compute attribute loss for matched pairs
    - This ensures each ground truth object is matched to exactly one prediction
    
    Args:
        outputs: Model outputs dictionary containing:
            - detection_loss: YOLOS detection loss (if available)
            - pred_boxes: [B, num_queries, 4] predicted boxes in cxcywh
            - attribute_logits: [B, num_queries, num_attributes] attribute predictions
        labels: List of ground truth dictionaries, each containing:
            - boxes: [num_objects, 4] ground truth boxes in cxcywh
            - class_labels: [num_objects] category labels
            - attribute_labels: [num_objects, num_attributes] binary attribute labels
        detection_weight: Weight for detection loss (default: 1.0)
        attribute_weight: Weight for attribute loss (default: 0.5)
    
    Returns:
        total_loss: Weighted combined loss
        loss_dict: Dictionary with individual loss components
    """
    
    # 1. Detection Loss
    # This comes directly from YOLOS model
    if 'detection_loss' in outputs:
        detection_loss = outputs['detection_loss']
    else:
        # If not available, we can't compute it here
        detection_loss = torch.tensor(0.0, device=outputs['pred_boxes'].device)
    
    # 2. Attribute Loss with Hungarian Matching
    pred_boxes = outputs['pred_boxes']  # [B, num_queries, 4]
    attribute_logits = outputs['attribute_logits']  # [B, num_queries, num_attributes]
    
    batch_size = pred_boxes.shape[0]
    attribute_losses = []
    
    for i in range(batch_size):
        # Ground truth for this sample
        gt_boxes = labels[i]['boxes']  # [num_gt, 4] in cxcywh format
        gt_attributes = labels[i]['attribute_labels']  # [num_gt, num_attributes]
        
        # Skip if no ground truth objects
        if len(gt_boxes) == 0:
            continue
        
        # Predicted boxes and attributes for this sample
        pred_boxes_i = pred_boxes[i]  # [num_queries, 4]
        pred_attrs_i = attribute_logits[i]  # [num_queries, num_attributes]
        
        # Hungarian matching based on L1 distance between boxes
        # Cost matrix: L1 distance between all pred-gt box pairs
        with torch.no_grad():
            cost_matrix = torch.cdist(pred_boxes_i, gt_boxes, p=1)  # [num_queries, num_gt]
            
            # Solve assignment problem
            # Returns indices: src_idx (predictions), tgt_idx (ground truth)
            src_idx, tgt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
        
        # Compute attribute loss only for matched pairs
        # Get matched predictions and ground truth
        matched_pred_attrs = pred_attrs_i[src_idx]  # [num_matched, num_attributes]
        matched_gt_attrs = gt_attributes[tgt_idx]  # [num_matched, num_attributes]
        
        # Binary cross-entropy with logits
        # This handles multi-label classification
        loss = F.binary_cross_entropy_with_logits(
            matched_pred_attrs,
            matched_gt_attrs,
            reduction='mean'
        )
        
        attribute_losses.append(loss)
    
    # Average attribute loss across batch
    if attribute_losses:
        attribute_loss = torch.stack(attribute_losses).mean()
    else:
        # No valid samples in batch
        attribute_loss = torch.tensor(0.0, device=detection_loss.device)
    
    # 3. Combined Loss
    total_loss = (
        detection_weight * detection_loss +
        attribute_weight * attribute_loss
    )
    
    # 4. Loss dictionary for logging
    loss_dict = {
        'total_loss': total_loss.item(),
        'detection_loss': detection_loss.item(),
        'attribute_loss': attribute_loss.item(),
    }
    
    # Add YOLOS sub-losses if available
    if 'loss_dict' in outputs:
        for k, v in outputs['loss_dict'].items():
            loss_dict[f'yolos_{k}'] = v.item() if isinstance(v, torch.Tensor) else v
    
    return total_loss, loss_dict

#%%
# ============================================================
# Trainer
# ============================================================
# Config moved to top of file

class FashionTrainer:
    """
    Trainer class for Fashion Detection Model.
    
    Handles complete training pipeline including:
    - Training and validation loops
    - Checkpointing (best and last models)
    - Learning rate scheduling
    - Logging to TensorBoard
    - Early stopping
    - Mixed precision training (optional)
    
    Example:
        config = TrainingConfig(num_epochs=30, batch_size=4)
        trainer = FashionTrainer(config)
        trainer.fit(train_dataset, val_dataset)
    """
    
    def __init__(
        self,
        config: Config,
        model: Optional[nn.Module] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
            model: Optional pre-initialized model. If None, creates new model.
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Set random seed for reproducibility
        self._set_seed(config.seed)
        
        # Initialize model
        if model is None:
            print(f"\nInitializing model: {config.model_name}")
            self.model = FashionDetectionModel(
                model_name=config.model_name,
                num_labels=config.num_labels,
                num_attributes=config.num_attributes,
                hidden_size=config.hidden_size,
                dropout=config.dropout,
            )
        else:
            self.model = model
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Initialize scheduler (will be finalized when we know dataset size)
        self.scheduler = None
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Initialize scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
        
        # Tracking metrics
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_detection_loss': [],
            'train_attribute_loss': [],
            'val_loss': [],
            'val_detection_loss': [],
            'val_attribute_loss': [],
            'val_mAP': [],
            'val_mAP_50': [],
            'val_mAP_75': [],
            'learning_rate': [],
        }
        
        # Initialize mAP metric
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            iou_thresholds=None,  # Use COCO default [0.5:0.95]
        ).to(self.device)
        
        print(f"\nTrainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Mixed precision: {config.mixed_precision and self.scaler is not None}")
        print(f"  Checkpoint directory: {config.checkpoint_dir}")
        print(f"  Log directory: {config.log_dir}")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_scheduler(self, num_training_steps: int):
        """
        Setup learning rate scheduler.
        
        Args:
            num_training_steps: Total number of training steps
        """
        num_warmup_steps = int(num_training_steps * self.config.warmup_epochs / self.config.num_epochs)
        
        if self.config.scheduler_type == 'cosine':
            # Cosine annealing with warmup
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    # Linear warmup from 0.1 to 1.0 (avoid starting from 0)
                    warmup_ratio = float(current_step) / float(max(1, num_warmup_steps))
                    return 0.1 + (1.0 - 0.1) * warmup_ratio
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        elif self.config.scheduler_type == 'cosine_restarts':
            # Cosine Annealing with Warm Restarts (SGDR)
            # T_0: Number of iterations for the first restart
            # Start from high LR, decrease, then restart
            # We split total steps into 'num_cycles' cycles
            if not hasattr(self.config, 'num_cycles'):
                self.config.num_cycles = 3
                
            T_0 = max(1, num_training_steps // self.config.num_cycles)
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=1,  # Can be 2 to double cycle length each time
                eta_min=0.0  # Min LR
            )
        
        elif self.config.scheduler_type == 'step':
            # Step decay
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=num_training_steps // 3,
                gamma=0.1
            )
        
        else:
            # No scheduler
            self.scheduler = None
    
    def _convert_batch_format(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert FashionpediaDataset batch format to model input format.
        
        Args:
            batch: Batch from FashionpediaDataset with keys:
                'images', 'bboxes', 'categories', 'attributes', etc.
        
        Returns:
            Batch with keys:
                'pixel_values': Tensor [B, 3, H, W]
                'labels': List of dicts with ground truth
        """
        batch_size = len(batch['images'])
        
        # Stack images to tensor [B, 3, H, W]
        pixel_values = torch.stack(batch['images'])
        
        # Create labels list
        labels = []
        for i in range(batch_size):
            label = {
                'boxes': batch['bboxes'][i],  # Already in cxcywh format
                'class_labels': batch['categories'][i],
                'attribute_labels': batch['attributes'][i],
                'image_id': torch.tensor([batch['image_ids'][i]], dtype=torch.int64),
                'area': batch['areas'][i],
                'iscrowd': torch.zeros(len(batch['bboxes'][i]), dtype=torch.int64),
                'orig_size': torch.tensor(batch['orig_sizes'][i], dtype=torch.int64),
            }
            labels.append(label)
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
        }
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Dictionary with average training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_detection_loss = 0.0
        total_attribute_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Convert batch format
            batch = self._convert_batch_format(batch)
            
            # Move data to device
            pixel_values = batch['pixel_values'].to(self.device)
            labels = [
                {k: v.to(self.device) for k, v in label.items()}
                for label in batch['labels']
            ]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss, loss_dict = compute_fashion_loss(
                        outputs,
                        labels,
                        detection_weight=self.config.detection_weight,
                        attribute_weight=self.config.attribute_weight,
                    )
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                # Standard training without mixed precision
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss, loss_dict = compute_fashion_loss(
                    outputs,
                    labels,
                    detection_weight=self.config.detection_weight,
                    attribute_weight=self.config.attribute_weight,
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Scheduler step (if using step-based scheduler)
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Accumulate metrics
            total_loss += loss_dict['total_loss']
            total_detection_loss += loss_dict['detection_loss']
            total_attribute_loss += loss_dict['attribute_loss']
            num_batches += 1
            
            # Update global step
            self.global_step += 1
            
            # Log to TensorBoard
            if self.global_step % self.config.log_every_n_steps == 0:
                self.writer.add_scalar('train/loss', loss_dict['total_loss'], self.global_step)
                self.writer.add_scalar('train/detection_loss', loss_dict['detection_loss'], self.global_step)
                self.writer.add_scalar('train/attribute_loss', loss_dict['attribute_loss'], self.global_step)
                
                if self.scheduler is not None:
                    self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'det': f"{loss_dict['detection_loss']:.4f}",
                'attr': f"{loss_dict['attribute_loss']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
        
        # Calculate averages
        avg_metrics = {
            'train_loss': total_loss / num_batches,
            'train_detection_loss': total_detection_loss / num_batches,
            'train_attribute_loss': total_attribute_loss / num_batches,
        }
        
        return avg_metrics
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model and compute mAP.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Dictionary with average validation metrics including mAP
        """
        self.model.eval()
        
        total_loss = 0.0
        total_detection_loss = 0.0
        total_attribute_loss = 0.0
        num_batches = 0
        
        # Reset mAP metric
        self.map_metric.reset()
        
        # Collect predictions and targets for mAP
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Convert batch format
                batch = self._convert_batch_format(batch)
                
                # Move data to device
                pixel_values = batch['pixel_values'].to(self.device)
                labels = [
                    {k: v.to(self.device) for k, v in label.items()}
                    for label in batch['labels']
                ]
                
                # Forward pass
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss, loss_dict = compute_fashion_loss(
                    outputs,
                    labels,
                    detection_weight=self.config.detection_weight,
                    attribute_weight=self.config.attribute_weight,
                )
                
                # Accumulate loss metrics
                total_loss += loss_dict['total_loss']
                total_detection_loss += loss_dict['detection_loss']
                total_attribute_loss += loss_dict['attribute_loss']
                num_batches += 1
                
                # Process predictions for mAP
                # outputs.logits: [B, num_queries, num_classes+1]
                # outputs.pred_boxes: [B, num_queries, 4] in cxcywh normalized
                batch_size = len(labels)
                
                for i in range(batch_size):
                    # Get predictions
                    logits = outputs['logits'][i]  # [num_queries, num_classes+1]
                    boxes = outputs['pred_boxes'][i]  # [num_queries, 4] cxcywh normalized
                    
                    # Convert to probabilities
                    probs = logits.softmax(-1)
                    
                    # Get scores and class predictions (exclude background class)
                    scores, pred_labels = probs[:, :-1].max(-1)
                    
                    # Filter by confidence threshold
                    keep = scores > 0.05  # Low threshold for mAP calculation
                    
                    if keep.sum() > 0:
                        # Convert boxes from cxcywh to xyxy format
                        boxes_xyxy = cxcywh_to_xyxy(boxes[keep])
                        
                        # Scale to image size (assuming square images)
                        img_size = labels[i]['orig_size'][0].item()  # Get height (same as width for square)
                        boxes_xyxy = boxes_xyxy * img_size
                        
                        pred_dict = {
                            'boxes': boxes_xyxy,
                            'scores': scores[keep],
                            'labels': pred_labels[keep],
                        }
                    else:
                        # Empty prediction
                        pred_dict = {
                            'boxes': torch.zeros((0, 4)),
                            'scores': torch.zeros((0,)),
                            'labels': torch.zeros((0,), dtype=torch.int64),
                        }
                    
                    all_predictions.append(pred_dict)
                    
                    # Get ground truth
                    gt_boxes = labels[i]['boxes']  # cxcywh normalized
                    if len(gt_boxes) > 0:
                        # Convert to xyxy
                        gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes)
                        
                        # Scale to image size
                        img_size = labels[i]['orig_size'][0].item()
                        gt_boxes_xyxy = gt_boxes_xyxy * img_size
                        
                        target_dict = {
                            'boxes': gt_boxes_xyxy,
                            'labels': labels[i]['class_labels'],
                        }
                    else:
                        # Empty ground truth
                        target_dict = {
                            'boxes': torch.zeros((0, 4)),
                            'labels': torch.zeros((0,), dtype=torch.int64),
                        }
                    
                    all_targets.append(target_dict)
        
        # Compute mAP
        if len(all_predictions) > 0:
            self.map_metric.update(all_predictions, all_targets)
            map_results = self.map_metric.compute()
            
            # Extract metrics
            map_score = map_results['map'].item() if 'map' in map_results else 0.0
            map_50 = map_results['map_50'].item() if 'map_50' in map_results else 0.0
            map_75 = map_results['map_75'].item() if 'map_75' in map_results else 0.0
            
            # Clear metric to free memory immediately
            self.map_metric.reset()
        else:
            map_score = 0.0
            map_50 = 0.0
            map_75 = 0.0
        
        # Calculate averages
        avg_metrics = {
            'val_loss': total_loss / num_batches,
            'val_detection_loss': total_detection_loss / num_batches,
            'val_attribute_loss': total_attribute_loss / num_batches,
            'val_mAP': map_score,
            'val_mAP_50': map_50,
            'val_mAP_75': map_75,
        }
        
        return avg_metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step,
            'config': self.config.to_dict(),
            'history': self.history,
        }
        
        torch.save(checkpoint, path)
        print(f"  Checkpoint saved: {path}")
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  Best model saved: {best_path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model from checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        print(f"\nLoading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if checkpoint['scaler_state_dict'] and self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint['global_step']
        self.history = checkpoint.get('history', self.history)
        
        print(f"  Resumed from epoch {self.current_epoch}")
        print(f"  Best validation loss: {self.best_val_loss:.4f}")
    
    def visualize_predictions(
        self,
        val_loader: DataLoader,
        num_samples: int = 4,
        conf_threshold: float = 0.5,
        epoch: int = 0,
    ):
        """
        Visualize predictions vs ground truth and log to TensorBoard.
        
        Args:
            val_loader: Validation data loader
            num_samples: Number of samples to visualize
            conf_threshold: Confidence threshold for predictions
            epoch: Current epoch for logging
        """
        self.model.eval()
        
        # Get one batch
        batch = next(iter(val_loader))
        batch = self._convert_batch_format(batch)
        
        pixel_values = batch['pixel_values'].to(self.device)
        labels = [
            {k: v.to(self.device) for k, v in label.items()}
            for label in batch['labels']
        ]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        # Process samples
        batch_size = len(labels)
        num_samples = min(num_samples, batch_size)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(20, 10 * num_samples))
        if num_samples == 1:
            axes = np.expand_dims(axes, 0)
            
        for i in range(num_samples):
            # Denormalize image
            # Assumes standard ImageNet mean/std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            
            img_tensor = pixel_values[i] * std + mean
            img_tensor = torch.clamp(img_tensor, 0, 1)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            
            height, width = img_np.shape[:2]
            
            # --- Plot Ground Truth (Left) ---
            ax_gt = axes[i, 0]
            ax_gt.imshow(img_np)
            ax_gt.set_title(f"Sample {i}: Ground Truth")
            ax_gt.axis('off')
            
            # Get GT boxes and convert to pixel coordinates
            gt_boxes = labels[i]['boxes']
            if len(gt_boxes) > 0:
                gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes)
                # Scale from normalized [0,1] to pixels
                gt_boxes_xyxy[:, [0, 2]] *= width
                gt_boxes_xyxy[:, [1, 3]] *= height
                
                gt_labels = labels[i]['class_labels']
                
                # Get dataset from loader to access category names
                dataset = val_loader.dataset
                
                for box, label in zip(gt_boxes_xyxy, gt_labels):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    cat_name = dataset.get_category_name(label.item())
                    
                    # Draw box
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='g', facecolor='none'
                    )
                    ax_gt.add_patch(rect)
                    
                    # Draw text
                    ax_gt.text(
                        x1, y1, cat_name,
                        color='white', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='g', alpha=0.5, pad=0)
                    )
            
            # --- Plot Predictions (Right) ---
            ax_pred = axes[i, 1]
            ax_pred.imshow(img_np)
            ax_pred.set_title(f"Sample {i}: Predictions")
            ax_pred.axis('off')
            
            # Get predictions
            logits = outputs['logits'][i]
            pred_boxes = outputs['pred_boxes'][i]
            
            probs = logits.softmax(-1)
            scores, pred_labels = probs[:, :-1].max(-1)
            
            keep = scores > conf_threshold
            
            if keep.sum() > 0:
                pred_boxes_keep = pred_boxes[keep]
                pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes_keep)
                # Scale to pixels
                pred_boxes_xyxy[:, [0, 2]] *= width
                pred_boxes_xyxy[:, [1, 3]] *= height
                
                pred_scores = scores[keep]
                pred_labels_keep = pred_labels[keep]
                
                # Get attribute logits if available
                # Note: outputs.attribute_logits is [B, num_queries, num_attributes]
                if hasattr(outputs, 'attribute_logits'):
                    attr_logits = outputs.attribute_logits[i][keep]
                    # Apply sigmoid
                    attr_probs = attr_logits.sigmoid()
                    # Threshold attributes
                    attr_keep = attr_probs > 0.5
                else:
                    attr_keep = None
                
                for j, (box, label, score) in enumerate(zip(pred_boxes_xyxy, pred_labels_keep, pred_scores)):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    cat_name = dataset.get_category_name(label.item())
                    
                    # Draw box
                    rect = patches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax_pred.add_patch(rect)
                    
                    # Build caption
                    caption = f"{cat_name} {score:.2f}"
                    
                    # Add attributes if available
                    if attr_keep is not None:
                         # We need attribute names from dataset if possible, 
                         # but dataset.attributes is a list of dicts.
                         # This part might be tricky if we don't have attr ID map handy.
                         # For now just show number of attributes predicted
                         num_attrs = attr_keep[j].sum().item()
                         if num_attrs > 0:
                             caption += f"\n+{num_attrs} attrs"
                    
                    # Draw text
                    ax_pred.text(
                        x1, y1, caption,
                        color='white', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='r', alpha=0.5, pad=0)
                    )
        
        plt.tight_layout()
        
        # Log to TensorBoard
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        image = PILImage.open(buf)
        image_tensor = ToTensor()(image)
        
        self.writer.add_image('val/predictions', image_tensor, epoch)
        plt.show()
        plt.close(fig)
        print("  Visualization logged to TensorBoard")
    
    def fit(
        self,
        train_dataset: FashionpediaDataset,
        val_dataset: FashionpediaDataset,
        resume_from: Optional[str] = None,
    ):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            resume_from: Optional path to checkpoint to resume from
        """
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        
        # Create data loaders
        train_loader = train_dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        
        val_loader = val_dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )
        
        print(f"\nDataset sizes:")
        print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
        
        # Setup scheduler
        num_training_steps = len(train_loader) * self.config.num_epochs
        self._setup_scheduler(num_training_steps)
        print(f"\nScheduler: {self.config.scheduler_type}")
        print(f"  Total training steps: {num_training_steps}")
        print(f"  Warmup steps: {int(num_training_steps * self.config.warmup_epochs / self.config.num_epochs)}")
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Save config
        config_path = Path(self.config.checkpoint_dir) / 'config.json'
        self.config.save(config_path)
        print(f"\nConfiguration saved: {config_path}")
        
        # Training loop
        print("\n" + "=" * 80)
        print("TRAINING LOOP")
        print("=" * 80)
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\n{'=' * 80}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'=' * 80}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Log training metrics
            self.history['train_loss'].append(train_metrics['train_loss'])
            self.history['train_detection_loss'].append(train_metrics['train_detection_loss'])
            self.history['train_attribute_loss'].append(train_metrics['train_attribute_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"\nTraining metrics:")
            print(f"  Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Detection loss: {train_metrics['train_detection_loss']:.4f}")
            print(f"  Attribute loss: {train_metrics['train_attribute_loss']:.4f}")
            print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Validate
            should_validate = (epoch + 1) % self.config.val_every_n_epochs == 0
            if should_validate:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['val_detection_loss'].append(val_metrics['val_detection_loss'])
                self.history['val_attribute_loss'].append(val_metrics['val_attribute_loss'])
                self.history['val_mAP'].append(val_metrics['val_mAP'])
                self.history['val_mAP_50'].append(val_metrics['val_mAP_50'])
                self.history['val_mAP_75'].append(val_metrics['val_mAP_75'])
                
                self.writer.add_scalar('val/loss', val_metrics['val_loss'], epoch)
                self.writer.add_scalar('val/detection_loss', val_metrics['val_detection_loss'], epoch)
                self.writer.add_scalar('val/attribute_loss', val_metrics['val_attribute_loss'], epoch)
                self.writer.add_scalar('val/mAP', val_metrics['val_mAP'], epoch)
                self.writer.add_scalar('val/mAP_50', val_metrics['val_mAP_50'], epoch)
                self.writer.add_scalar('val/mAP_75', val_metrics['val_mAP_75'], epoch)
                
                print(f"\nValidation metrics:")
                print(f"  Loss: {val_metrics['val_loss']:.4f}")
                print(f"  Detection loss: {val_metrics['val_detection_loss']:.4f}")
                print(f"  Attribute loss: {val_metrics['val_attribute_loss']:.4f}")
                print(f"  mAP: {val_metrics['val_mAP']:.4f}")
                print(f"  mAP@50: {val_metrics['val_mAP_50']:.4f}")
                print(f"  mAP@75: {val_metrics['val_mAP_75']:.4f}")
                print(f"  mAP@75: {val_metrics['val_mAP_75']:.4f}")
                
                # Visualize predictions
                try:
                    self.visualize_predictions(val_loader, num_samples=4, epoch=epoch)
                except Exception as e:
                    print(f"Visualization failed: {e}")
                
                # Check if this is the best model
                is_best = val_metrics['val_loss'] < self.best_val_loss
                if is_best:
                    improvement = self.best_val_loss - val_metrics['val_loss']
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    print(f"\n✓ New best model! (improved by {improvement:.4f})")
                else:
                    self.epochs_without_improvement += 1
                    print(f"\n  No improvement for {self.epochs_without_improvement} epochs")
                
                # Save checkpoint (Best Model only)
                if self.config.save_best and is_best:
                    best_path = Path(self.config.checkpoint_dir) / 'best_model.pth'
                    self.save_checkpoint(best_path, is_best=False)
                
                # Early stopping
                if self.config.early_stopping and self.epochs_without_improvement >= self.config.patience:
                    print(f"\n" + "=" * 80)
                    print(f"EARLY STOPPING: No improvement for {self.config.patience} epochs")
                    print(f"=" * 80)
                    break
            
            # Save per-epoch checkpoint (Always)
            epoch_path = Path(self.config.checkpoint_dir) / f'epoch_{epoch + 1}.pth'
            self.save_checkpoint(epoch_path, is_best=False)

            # Update last model (Always)
            if self.config.save_last:
                last_path = Path(self.config.checkpoint_dir) / 'last_model.pth'
                self.save_checkpoint(last_path, is_best=False)
        
        # Save final checkpoint
        if self.config.save_last:
            final_path = Path(self.config.checkpoint_dir) / 'last_model.pth'
            self.save_checkpoint(final_path)
        
        # Close writer
        self.writer.close()
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total epochs: {self.current_epoch + 1}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")

#%%
# =======================================
# Run
# =======================================
# Initialize downloader
downloader = FashionpediaDownloader(data_dir='data')
# Download annotations and images
status = downloader.download_all(download_images=True)

# Create configuration
config = CONFIG  # Use global config
# config = Config(num_epochs=20)  # Or create custom config

print("\nConfiguration created:")
print(f"  Epochs: {config.num_epochs}")
print(f"  Batch size: {config.batch_size}")
print(f"  Learning rate: {config.learning_rate}")
print(f"  Device: {config.device}")
print(f"  Target categories: {len(config.target_categories)} classes")

# Training dataset
train_dataset = FashionpediaDataset(
    annotation_file=str(Path(config.data_dir) / 'raw/annotations/instances_attributes_train2020.json'),
    images_dir=str(Path(config.data_dir) / 'raw/images'),
    split='train',
    use_autoaugment=config.use_autoaugment,
    image_size=config.image_size,
    num_attributes=config.num_attributes,
)

# Validation dataset
val_dataset = FashionpediaDataset(
    annotation_file=str(Path(config.data_dir) / 'raw/annotations/instances_attributes_val2020.json'),
    images_dir=str(Path(config.data_dir) / 'raw/images'),
    split='val',
    use_autoaugment=False,  # No augmentation for validation
    image_size=config.image_size,
    num_attributes=config.num_attributes,
)

# Initialize trainer
trainer = FashionTrainer(config)

# Start training
trainer.fit(train_dataset, val_dataset)

# Or resume from checkpoint
# trainer.fit(train_dataset, val_dataset, resume_from='/kaggle/input/fashionpedia/pytorch/default/2/epoch_30.pth')