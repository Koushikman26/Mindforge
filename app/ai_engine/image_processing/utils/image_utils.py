import cv2
import numpy as np
import time
import logging
from typing import Tuple, Optional, Any, Union
from functools import wraps
from pathlib import Path

# Import your existing logging configuration
from ...utils.logging_config import get_logger

logger = get_logger(__name__)

def timing_decorator(func):
    """Decorator to measure and log processing time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            processing_time = end_time - start_time
            logger.info(f"{func.__name__} completed in {processing_time:.4f} seconds")
            return result
        except Exception as e:
            end_time = time.time()
            processing_time = end_time - start_time
            logger.error(f"{func.__name__} failed after {processing_time:.4f} seconds: {str(e)}")
            raise
    return wrapper

def validate_image(image: np.ndarray) -> bool:
    """Validate if image is suitable for processing"""
    if image is None:
        logger.warning("Image is None")
        return False
    
    if not isinstance(image, np.ndarray):
        logger.warning(f"Image is not numpy array, got {type(image)}")
        return False
    
    if len(image.shape) not in [2, 3]:
        logger.warning(f"Invalid image dimensions: {image.shape}")
        return False
    
    if image.size == 0:
        logger.warning("Image is empty")
        return False
    
    # Check for minimum size requirements for stock analysis
    min_height, min_width = 100, 100
    if image.shape[0] < min_height or image.shape[1] < min_width:
        logger.warning(f"Image too small: {image.shape[:2]}, minimum: {min_height}x{min_width}")
        return False
    
    return True

def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load image with error handling"""
    try:
        image_path = str(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Could not load image from: {image_path}")
            return None
        
        logger.info(f"Loaded image from {image_path}, shape: {image.shape}")
        return image
    
    except Exception as e:
        logger.error(f"Error loading image from {image_path}: {str(e)}")
        return None

def save_image(image: np.ndarray, output_path: Union[str, Path], quality: int = 95) -> bool:
    """Save image with compression settings optimized for stock analysis"""
    try:
        output_path = str(output_path)
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Set compression parameters based on file extension
        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif output_path.lower().endswith('.png'):
            params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, (100-quality)//10))]
        else:
            params = []
        
        success = cv2.imwrite(output_path, image, params)
        
        if success:
            logger.info(f"Image saved to: {output_path}")
        else:
            logger.error(f"Failed to save image to: {output_path}")
        
        return success
    
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {str(e)}")
        return False

def resize_image(image: np.ndarray, target_size: Tuple[int, int], maintain_aspect: bool = True) -> np.ndarray:
    """Resize image with options for stock analysis optimization"""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if not maintain_aspect:
        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # Maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create canvas and center the image
    if len(image.shape) == 3:
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas

def extract_roi(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract region of interest for focused processing"""
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]

def calculate_image_metrics(image: np.ndarray) -> dict:
    """Calculate comprehensive image quality metrics for stock analysis"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return {
        'mean_brightness': float(np.mean(gray)),
        'std_brightness': float(np.std(gray)),
        'contrast': float(gray.max() - gray.min()),
        'dynamic_range': float(np.percentile(gray, 99) - np.percentile(gray, 1)),
        'sharpness': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        'noise_level': float(np.std(gray - cv2.medianBlur(gray, 5))),
        'edge_density': float(np.sum(cv2.Canny(gray, 50, 150) > 0) / gray.size),
        'image_size': image.shape[:2]
    }

class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

class ValidationError(ImageProcessingError):
    """Exception for image validation errors"""
    pass

class ProcessingTimeoutError(ImageProcessingError):
    """Exception for processing timeout errors"""
    pass