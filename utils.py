"""
Utility functions for the demo application.
Handles label mappings, JSON formatting, and encoding.
"""
from typing import Dict, List, Any
import json
import base64
from io import BytesIO
from PIL import Image


def load_category_names(annotation_file: str) -> Dict[int, str]:
    """
    Load category names from COCO annotation file.
    
    Args:
        annotation_file: Path to annotation JSON file
        
    Returns:
        Dictionary mapping category ID to name
    """
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        category_names = {cat['id']: cat['name'] for cat in data['categories']}
        return category_names
    except Exception as e:
        print(f"Warning: Could not load categories from {annotation_file}: {e}")
        return {}


def load_attribute_names(annotation_file: str) -> Dict[int, str]:
    """
    Load attribute names from COCO annotation file.
    
    Args:
        annotation_file: Path to annotation JSON file
        
    Returns:
        Dictionary mapping attribute ID to name
    """
    try:
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Sort attributes by ID to ensure consistency with training
        sorted_attributes = sorted(data['attributes'], key=lambda x: x['id'])
        
        # Create mapping from Index -> Name
        # The model outputs correspond to indices in this sorted list
        attribute_names = {idx: attr['name'] for idx, attr in enumerate(sorted_attributes)}
        return attribute_names
    except Exception as e:
        print(f"Warning: Could not load attributes from {annotation_file}: {e}")
        return {}


def format_detection_results(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format detection results for JSON response.
    
    Args:
        detections: List of raw detection dictionaries
        
    Returns:
        Formatted detection list
    """
    formatted = []
    
    for det in detections:
        formatted_det = {
            'label': det['label'],
            'confidence': round(det['confidence'], 2),
            'box': [round(x) for x in det['box']],  # Round to integers
            'attributes': det.get('attributes', []),
        }
        formatted.append(formatted_det)
    
    return formatted


def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """
    Convert PIL Image to base64 string.
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
        
    Returns:
        Base64 encoded string
    """
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{format.lower()};base64,{img_str}"


def create_default_mappings() -> tuple[Dict[int, str], Dict[int, str]]:
    """
    Create default category and attribute mappings if annotation files are not available.
    
    Returns:
        Tuple of (category_names, attribute_names)
    """
    # Fashionpedia categories (47 total)
    categories = [
        "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
        "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape", "glasses", "hat",
        "headband, head covering, hair accessory", "tie", "glove", "watch", "belt", "leg warmer",
        "tights, stockings", "sock", "shoe", "bag, wallet", "scarf", "umbrella", "hood", "collar",
        "lapel", "epaulette", "sleeve", "pocket", "neckline", "buckle", "zipper", "applique",
        "bead", "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel"
    ]
    
    category_names = {i: name for i, name in enumerate(categories)}
    
    # Create some default attribute names
    attribute_names = {i: f"Attribute_{i}" for i in range(294)}
    
    return category_names, attribute_names
