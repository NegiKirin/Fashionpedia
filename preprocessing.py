"""
Preprocessing utilities for fashion detection.
Contains image preprocessing, bounding box utilities, and transformations.
"""
from typing import Tuple, List, Dict, Any, Optional
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np


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


class FashionpediaPreprocessor:
    """
    Preprocessing pipeline for Fashionpedia dataset.
    Handles image loading, resizing, and normalization.
    """
    
    def __init__(
        self,
        image_size: int = 512,
        normalize: bool = True,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Initialize preprocessor.
        
        Args:
            image_size: Target image size
            normalize: Whether to normalize images (default: True)
            mean: Mean for normalization (default: ImageNet)
            std: Std for normalization (default: ImageNet)
        """
        self.image_size = image_size
        self.normalize = normalize
        self.mean = mean
        self.std = std
        
        # Transforms
        from torchvision import transforms
        self.to_tensor = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=mean, std=std) if normalize else None
        
    def resize_with_padding(
        self,
        image: Image.Image,
        target_size: int = None,
    ) -> Tuple[Image.Image, Tuple[int, int], Tuple[int, int]]:
        """
        Resize image to target size with padding to preserve aspect ratio.
        
        Args:
            image: PIL Image
            target_size: Target size (default: self.image_size)
        
        Returns:
            Tuple of (resized_image, (pad_left, pad_top), (new_width, new_height))
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
        
        return padded_image, (pad_left, pad_top), (new_width, new_height)
    
    def preprocess_image(
        self,
        image: Image.Image,
    ) -> Dict[str, Any]:
        """
        Preprocess a single image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            dict: Preprocessed data with image tensor and metadata
        """
        # Fix channels
        image_tensor = self.to_tensor(image)
        image = fix_channels(image_tensor)
        
        # Resize with padding
        image, (pad_left, pad_top), (new_width, new_height) = self.resize_with_padding(
            image=image,
            target_size=self.image_size,
        )
        
        # Convert to tensor and normalize
        final_image_tensor = self.to_tensor(image)
        
        if self.normalize_transform is not None:
            final_image_tensor = self.normalize_transform(final_image_tensor)
            
        return {
            'image': final_image_tensor,  # Tensor [C, H, W]
            'padding': (pad_left, pad_top),
            'resized_size': (new_width, new_height),
            'original_size': image.size,
        }


def denormalize_image(tensor: torch.Tensor, mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                     std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
        
    Returns:
        Denormalized image as numpy array [H, W, C] in range [0, 255]
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    img_tensor = tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    
    return img_np
