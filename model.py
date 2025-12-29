"""
Model architecture for fashion detection.
YOLOS-based object detection with attribute classification head.
"""
from typing import Optional, List, Dict
import torch
import torch.nn as nn
from transformers import AutoModelForObjectDetection


class FashionDetectionModel(nn.Module):
    """
    Fashion Detection Model with YOLOS backbone and attribute head.
    
    Architecture:
    - YOLOS (You Only Look at One Sequence) for object detection
    - Custom MLP head for attribute classification
    
    Args:
        model_name: HuggingFace model name (default: 'hustvl/yolos-small')
        num_labels: Number of object categories
        num_attributes: Number of attribute classes
        hidden_size: Hidden dimension size
        dropout: Dropout probability
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
        
        # Load YOLOS model for object detection
        self.yolos = AutoModelForObjectDetection.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        
        # Attribute classification head
        # Takes detection token features and predicts attributes
        self.attribute_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_attributes),
        )
        
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
