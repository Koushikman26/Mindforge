from dataclasses import dataclass
from typing import Tuple, Dict, Any
import os

@dataclass
class ImageProcessingConfig:
    """Configuration for the advanced image processing pipeline"""
    
    # Performance settings aligned with your existing AI engine
    MAX_PROCESSING_TIME: int = int(os.getenv('MAX_IMAGE_PROCESSING_TIME', 10))
    TARGET_SIZE: Tuple[int, int] = (1920, 1080)
    BATCH_SIZE: int = int(os.getenv('IMAGE_BATCH_SIZE', 4))
    
    # Lighting correction settings
    GAMMA_RANGE: Tuple[float, float] = (0.4, 2.5)
    CONTRAST_ALPHA_RANGE: Tuple[float, float] = (0.7, 1.8)
    BRIGHTNESS_BETA_RANGE: Tuple[int, int] = (-60, 60)
    CLAHE_CLIP_LIMIT: float = 3.0
    CLAHE_TILE_SIZE: Tuple[int, int] = (8, 8)
    
    # Perspective settings for ceiling cameras (stock analysis specific)
    PERSPECTIVE_THRESHOLD: int = 80
    MIN_CONTOUR_AREA: int = 800
    SHELF_DETECTION_MIN_AREA: int = 5000
    PRODUCT_DETECTION_MIN_AREA: int = 100
    
    # Quality enhancement settings
    SHARPEN_KERNEL_SIZE: int = 3
    DENOISE_H: float = 12.0
    DENOISE_TEMPLATE_SIZE: int = 7
    DENOISE_SEARCH_SIZE: int = 21
    BILATERAL_D: int = 9
    BILATERAL_SIGMA_COLOR: int = 75
    BILATERAL_SIGMA_SPACE: int = 75
    
    # Multi-angle processing for stock shelves
    SUPPORTED_ANGLES: Tuple[int, ...] = (0, 90, 180, 270)
    ROTATION_TOLERANCE: float = 3.0
    
    # Stock analysis specific settings
    ENABLE_SHELF_DETECTION: bool = True
    ENABLE_PRODUCT_SEGMENTATION: bool = True
    ENABLE_INVENTORY_OPTIMIZATION: bool = True
    
    # Integration with existing AI engine
    LOG_PROCESSING_STEPS: bool = True
    SAVE_INTERMEDIATE_RESULTS: bool = bool(os.getenv('DEBUG_MODE', False))
    OUTPUT_DIR: str = os.getenv('PROCESSING_OUTPUT_DIR', '/tmp/image_processing')

    def __post_init__(self):
        """Ensure output directory exists"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)