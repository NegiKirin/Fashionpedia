"""
FastAPI application for fashion detection demo.
Provides web interface and API endpoints for image upload and detection.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from PIL import Image
import io
import traceback

from inference import FashionDetector
from utils import (
    load_category_names,
    load_attribute_names,
    format_detection_results,
    image_to_base64,
    create_default_mappings,
)
import config

# Initialize FastAPI app
app = FastAPI(
    title="Fashion Detection Demo",
    description="AI-powered fashion item detection with attributes",
    version="1.0.0"
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize detector (will be loaded on startup)
detector = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global detector
    
    print("=" * 80)
    print("INITIALIZING FASHION DETECTION DEMO")
    print("=" * 80)
    
    # Check if checkpoint exists
    if not config.MODEL_CHECKPOINT.exists():
        print(f"WARNING: Checkpoint not found at {config.MODEL_CHECKPOINT}")
        print(f"Please place your trained model checkpoint at: {config.MODEL_CHECKPOINT}")
        print("The demo will start but predictions will fail until a checkpoint is available.")
        return
    
    try:
        # Initialize detector with config settings
        detector = FashionDetector(
            checkpoint_path=str(config.MODEL_CHECKPOINT),
            conf_threshold=config.CONFIDENCE_THRESHOLD,
            attr_threshold=config.ATTRIBUTE_THRESHOLD,
        )
        
        # Try to load labels from annotation file
        annotation_loaded = False
        possible_annotation_files = [
            config.LABEL_DESCRIPTIONS_FILE,
            config.ANNOTATION_FILE,
            config.ANNOTATION_FILE_TRAIN
        ]
        
        for annotation_path in possible_annotation_files:
            if annotation_path.exists():
                print(f"Loading labels from {annotation_path}")
                category_names = load_category_names(str(annotation_path))
                attribute_names = load_attribute_names(str(annotation_path))
                annotation_loaded = True
                break
        
        if not annotation_loaded:
            print("Annotation files not found, using default mappings")
            category_names, attribute_names = create_default_mappings()
        
        detector.load_label_mappings(category_names, attribute_names)
        
        print("=" * 80)
        print("MODEL LOADED SUCCESSFULLY")
        print(f"  Checkpoint: {config.MODEL_CHECKPOINT}")
        print(f"  Categories: {len(category_names)}")
        print(f"  Attributes: {len(attribute_names)}")
        print(f"  Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
        print(f"  Attribute Threshold: {config.ATTRIBUTE_THRESHOLD}")
        print("=" * 80)
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        traceback.print_exc()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_file = static_dir / "index.html"
    
    if html_file.exists():
        return html_file.read_text(encoding='utf-8')
    else:
        return """
        <html>
            <body>
                <h1>Fashion Detection Demo</h1>
                <p>Static files not found. Please ensure static/index.html exists.</p>
            </body>
        </html>
        """


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    """
    Detect fashion items in uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON with detection results and annotated images
    """
    global detector
    
    if detector is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure checkpoint exists."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run detection
        results = detector.predict(image)
        detections = results['detections']
        
        # Convert to pixel coordinates for visualization
        preprocessed = results.get('preprocessed', {})
        detections_pixel = detector.convert_to_pixel_coords(
            detections,
            results['image_size'],
            padding=preprocessed.get('padding'),
            resized_size=preprocessed.get('resized_size')
        )
        
        # Draw detections on image
        annotated_image = detector.visualize_detections(image, detections)
        
        # Format results
        formatted_detections = format_detection_results(detections_pixel)
        
        # Convert images to base64
        original_b64 = image_to_base64(image)
        annotated_b64 = image_to_base64(annotated_image)
        
        return JSONResponse({
            'success': True,
            'detections': formatted_detections,
            'images': {
                'original': original_b64,
                'annotated': annotated_b64,
            },
            'metadata': {
                'num_detections': len(formatted_detections),
                'image_size': results['image_size'],
            }
        })
        
    except Exception as e:
        print(f"Error during detection: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        'status': 'healthy',
        'model_loaded': detector is not None,
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        reload=config.RELOAD
    )
