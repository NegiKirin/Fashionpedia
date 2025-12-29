"""
Inference module for fashion detection.
Handles model loading, prediction, and visualization.
"""
from typing import Dict, List, Any, Tuple
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
from torchvision.ops import nms

from preprocessing import FashionpediaPreprocessor, cxcywh_to_xyxy, denormalize_image
from model import FashionDetectionModel
import config


class FashionDetector:
    """
    Fashion detection inference service.
    Loads model checkpoint and performs inference on images.
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = None,
        conf_threshold: float = None,
        attr_threshold: float = None,
    ):
        """
        Initialize detector.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file). Uses config default if None.
            device: Device to run inference on ('cuda' or 'cpu'). Uses config default if None.
            conf_threshold: Confidence threshold for detections. Uses config default if None.
            attr_threshold: Threshold for attribute predictions. Uses config default if None.
        """
        self.checkpoint_path = checkpoint_path or str(config.MODEL_CHECKPOINT)
        self.device = device or config.DEVICE
        # Fall back to CPU if CUDA not available
        if self.device == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            print("CUDA not available, using CPU")
        
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.attr_threshold = attr_threshold or config.ATTRIBUTE_THRESHOLD
        self.nms_threshold = config.NMS_THRESHOLD
        
        # Initialize preprocessor
        self.preprocessor = FashionpediaPreprocessor(
            image_size=config.IMAGE_SIZE,
            normalize=True,
        )
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Category and attribute mappings (will be loaded from annotations)
        self.category_names = {}
        self.attribute_names = {}
        
    def _load_model(self) -> FashionDetectionModel:
        """Load model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        # Initialize model with config parameters
        model = FashionDetectionModel(
            model_name=config.MODEL_NAME,
            num_labels=config.NUM_LABELS,
            num_attributes=config.NUM_ATTRIBUTES,
            hidden_size=config.HIDDEN_SIZE,
            dropout=config.DROPOUT,
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        print(f"Model loaded successfully on {self.device}")
        
        return model
    
    def load_label_mappings(self, category_names: Dict[int, str], attribute_names: Dict[int, str]):
        """
        Load category and attribute name mappings.
        
        Args:
            category_names: Mapping from category ID to name
            attribute_names: Mapping from attribute ID to name
        """
        self.category_names = category_names
        self.attribute_names = attribute_names
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run inference on an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with detection results
        """
        # Preprocess image
        preprocessed = self.preprocessor.preprocess_image(image)
        pixel_values = preprocessed['image'].unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
        
        # Post-process results
        detections = self._post_process(outputs, preprocessed)
        
        return {
            'detections': detections,
            'preprocessed': preprocessed,
            'image_size': image.size,
        }
    
    def _post_process(self, outputs: Dict[str, torch.Tensor], preprocessed: Dict) -> List[Dict[str, Any]]:
        """
        Post-process model outputs to extract detections.
        
        Args:
            outputs: Model outputs
            preprocessed: Preprocessed image data
            
        Returns:
            List of detection dictionaries
        """
        # Get predictions (batch size = 1)
        logits = outputs['logits'][0]  # [num_queries, num_labels+1]
        pred_boxes = outputs['pred_boxes'][0]  # [num_queries, 4] in cxcywh format
        attribute_logits = outputs['attribute_logits'][0]  # [num_queries, num_attributes]
        
        # Get class predictions
        probs = logits.softmax(-1)
        scores, pred_labels = probs[:, :-1].max(-1)  # Exclude background class
        
        # Filter by confidence
        keep = scores > self.conf_threshold
        
        if keep.sum() == 0:
            return []
        
        # Get filtered predictions
        pred_boxes_keep = pred_boxes[keep]
        pred_scores = scores[keep]
        pred_labels_keep = pred_labels[keep]
        attribute_logits_keep = attribute_logits[keep]
        
        # Convert boxes from cxcywh to xyxy for NMS
        pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes_keep)
        
        # Apply Non-Maximum Suppression (NMS)
        keep_nms = nms(pred_boxes_xyxy, pred_scores, self.nms_threshold)
        
        # Filter again after NMS
        pred_boxes_xyxy = pred_boxes_xyxy[keep_nms]
        pred_scores = pred_scores[keep_nms]
        pred_labels_keep = pred_labels_keep[keep_nms]
        attribute_logits_keep = attribute_logits_keep[keep_nms]
        
        # Get predicted attributes (apply sigmoid and threshold)
        attribute_probs = attribute_logits_keep.sigmoid()
        
        # Build detection list
        detections = []
        for i in range(len(pred_boxes_xyxy)):
            box = pred_boxes_xyxy[i].cpu().tolist()
            label_id = pred_labels_keep[i].item()
            confidence = pred_scores[i].item()
            
            # Get attributes above threshold
            attr_mask = attribute_probs[i] > self.attr_threshold
            attr_indices = torch.where(attr_mask)[0].cpu().tolist()
            
            # Map attribute indices to names
            attribute_names_list = [
                self.attribute_names.get(idx, f"Attribute_{idx}")
                for idx in attr_indices
            ]
            
            detection = {
                'label': self.category_names.get(label_id, f"Category_{label_id}"),
                'label_id': label_id,
                'confidence': confidence,
                'box': box,  # [x_min, y_min, x_max, y_max] in normalized coords [0, 1]
                'attributes': attribute_names_list,
                'attribute_ids': attr_indices,
            }
            
            detections.append(detection)
        
        return detections
    
    def visualize_detections(
        self,
        image: Image.Image,
        detections: List[Dict[str, Any]],
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Original PIL Image
            detections: List of detections
            
        Returns:
            Annotated image
        """
        # Create a copy
        img_draw = image.copy()
        draw = ImageDraw.Draw(img_draw)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        width, height = image.size
        
        # Define colors for boxes
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52D3AA'
        ]
        
        for idx, detection in enumerate(detections):
            box = detection['box']
            label = detection['label']
            confidence = detection['confidence']
            
            # Convert normalized coords to pixels
            x_min = int(box[0] * width)
            y_min = int(box[1] * height)
            x_max = int(box[2] * width)
            y_max = int(box[3] * height)
            
            # Choose color
            color = colors[idx % len(colors)]
            
            # Draw rectangle
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)
            
            # Create label text
            text = f"{label} {confidence:.2f}"
            
            # Get text bounding box
            bbox = draw.textbbox((x_min, y_min), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background for text
            draw.rectangle(
                [x_min, y_min - text_height - 4, x_min + text_width + 4, y_min],
                fill=color
            )
            
            # Draw text
            draw.text((x_min + 2, y_min - text_height - 2), text, fill='white', font=font)
        
        return img_draw
    
    def convert_to_pixel_coords(
        self,
        detections: List[Dict[str, Any]],
        image_size: Tuple[int, int],
        padding: Tuple[int, int] = None,
        resized_size: Tuple[int, int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convert detection coordinates from normalized [0,1] to pixel coordinates.
        
        Args:
            detections: List of detections with normalized coords
            image_size: (width, height) of original image
            padding: (pad_left, pad_top) used in preprocessing
            resized_size: (new_width, new_height) after resizing
            
        Returns:
            List of detections with pixel coordinates
        """
        width, height = image_size
        
        result = []
        for det in detections:
            det_copy = det.copy()
            box = det['box']
            
            if padding is not None and resized_size is not None:
                # Get preprocessing params
                pad_left, pad_top = padding
                new_w, new_h = resized_size
                target_size = config.IMAGE_SIZE
                
                # Convert normalized coords (0-1) to padded image coords (pixels)
                x1 = box[0] * target_size
                y1 = box[1] * target_size
                x2 = box[2] * target_size
                y2 = box[3] * target_size
                
                # Remove padding
                x1 = x1 - pad_left
                y1 = y1 - pad_top
                x2 = x2 - pad_left
                y2 = y2 - pad_top
                
                # Scale back to original image size
                # scale = original / resized
                scale_x = width / new_w
                scale_y = height / new_h
                
                x1 = x1 * scale_x
                y1 = y1 * scale_y
                x2 = x2 * scale_x
                y2 = y2 * scale_y
                
                # Clamp to image boundaries and convert to int
                det_copy['box'] = [
                    min(max(0, int(x1)), width),
                    min(max(0, int(y1)), height),
                    min(max(0, int(x2)), width),
                    min(max(0, int(y2)), height),
                ]
            else:
                # Fallback for simple resize (no padding)
                det_copy['box'] = [
                    int(box[0] * width),
                    int(box[1] * height),
                    int(box[2] * width),
                    int(box[3] * height),
                ]
            
            result.append(det_copy)
        
        return result
