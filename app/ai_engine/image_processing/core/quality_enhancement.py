import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Optional, Dict
from ..utils.image_utils import timing_decorator, validate_image, calculate_image_metrics, logger
from ..configs.processing_config import ImageProcessingConfig

class QualityEnhancer:
    """Advanced quality enhancement optimized for stock analysis imagery"""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        logger.info("Initialized QualityEnhancer")
    
    @timing_decorator
    def enhance_image_quality(self, image: np.ndarray, enhancement_level: str = "auto") -> np.ndarray:
        """
        Comprehensive image quality enhancement pipeline
        """
        if not validate_image(image):
            raise ValueError("Invalid image provided for quality enhancement")
        
        original_metrics = calculate_image_metrics(image)
        logger.info(f"Original quality metrics: sharpness={original_metrics['sharpness']:.2f}, "
                   f"noise_level={original_metrics['noise_level']:.2f}")
        
        # Determine enhancement strategy
        if enhancement_level == "auto":
            enhancement_level = self._determine_enhancement_strategy(image, original_metrics)
        
        result = image.copy()
        
        # Step 1: Noise reduction (preserves important details)
        result = self._intelligent_noise_reduction(result, original_metrics)
        
        # Step 2: Sharpening based on content analysis
        result = self._adaptive_sharpening(result, original_metrics)
        
        # Step 3: Detail enhancement
        result = self._enhance_fine_details(result)
        
        # Step 4: Color and contrast optimization
        result = self._optimize_colors_contrast(result)
        
        # Step 5: Final quality validation
        result = self._validate_and_refine(result, original_metrics)
        
        enhanced_metrics = calculate_image_metrics(result)
        logger.info(f"Enhanced quality metrics: sharpness={enhanced_metrics['sharpness']:.2f}, "
                   f"noise_level={enhanced_metrics['noise_level']:.2f}")
        
        return result
    
    def _determine_enhancement_strategy(self, image: np.ndarray, metrics: Dict) -> str:
        """Determine the best enhancement strategy based on image analysis"""
        sharpness = metrics['sharpness']
        noise_level = metrics['noise_level']
        edge_density = metrics['edge_density']
        
        if noise_level > 15:
            return "noise_heavy"
        elif sharpness < 100:
            return "sharpness_focus"
        elif edge_density < 0.1:
            return "detail_enhancement"
        else:
            return "balanced"
    
    def _intelligent_noise_reduction(self, image: np.ndarray, metrics: Dict) -> np.ndarray:
        """Apply noise reduction while preserving important details"""
        noise_level = metrics['noise_level']
        
        if noise_level < 5:
            # Minimal noise - light filtering only
            return cv2.bilateralFilter(image, 5, 30, 30)
        elif noise_level < 15:
            # Moderate noise - standard bilateral filter
            return cv2.bilateralFilter(
                image, 
                self.config.BILATERAL_D, 
                self.config.BILATERAL_SIGMA_COLOR, 
                self.config.BILATERAL_SIGMA_SPACE
            )
        else:
            # Heavy noise - non-local means denoising
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(
                    image, None, 
                    self.config.DENOISE_H, 
                    self.config.DENOISE_H,
                    self.config.DENOISE_TEMPLATE_SIZE, 
                    self.config.DENOISE_SEARCH_SIZE
                )
            else:
                return cv2.fastNlMeansDenoising(
                    image, None, 
                    self.config.DENOISE_H,
                    self.config.DENOISE_TEMPLATE_SIZE, 
                    self.config.DENOISE_SEARCH_SIZE
                )
    
    def _adaptive_sharpening(self, image: np.ndarray, metrics: Dict) -> np.ndarray:
        """Apply adaptive sharpening based on image characteristics"""
        sharpness = metrics['sharpness']
        
        if sharpness > 200:
            # Already sharp - minimal sharpening
            kernel = np.array([[-0.5, -0.5, -0.5],
                              [-0.5,  5.0, -0.5],
                              [-0.5, -0.5, -0.5]], dtype=np.float32)
        elif sharpness > 100:
            # Moderate sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]], dtype=np.float32)
        else:
            # Aggressive sharpening for blurry images
            kernel = np.array([[-1, -2, -1],
                              [-2, 13, -2],
                              [-1, -2, -1]], dtype=np.float32)
        
        sharpened = cv2.filter2D(image, -1, kernel)
        
        # Unsharp masking for better control
        gaussian = cv2.GaussianBlur(image, (0, 0), 1.5)
        unsharp_mask = cv2.addWeighted(image, 1.8, gaussian, -0.8, 0)
        
        # Combine both techniques
        return cv2.addWeighted(sharpened, 0.6, unsharp_mask, 0.4, 0)
    
    def _enhance_fine_details(self, image: np.ndarray) -> np.ndarray:
        """Enhance fine details using multi-scale processing"""
        # Convert to LAB for better luminance processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Multi-scale detail enhancement
        details_enhanced = l.copy().astype(np.float32)
        
        for scale in [1, 2, 4]:
            # Create detail layer at different scales
            blurred = cv2.GaussianBlur(l, (0, 0), scale)
            detail = l.astype(np.float32) - blurred.astype(np.float32)
            
            # Enhance detail with controlled amplification
            enhanced_detail = detail * (1.0 + 0.3 / scale)
            details_enhanced += enhanced_detail
        
        # Clamp values
        details_enhanced = np.clip(details_enhanced, 0, 255).astype(np.uint8)
        
        # Apply CLAHE for local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(details_enhanced)
        
        # Merge back
        lab_enhanced = cv2.merge([l_clahe, a, b])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    def _optimize_colors_contrast(self, image: np.ndarray) -> np.ndarray:
        """Optimize colors and contrast for better visibility"""
        # Convert to HSV for color enhancement
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance saturation moderately (important for product identification)
        s_factor = 1.15
        s_enhanced = cv2.multiply(s, s_factor)
        s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
        
        # Enhance value channel with histogram equalization
        v_eq = cv2.equalizeHist(v)
        v_enhanced = cv2.addWeighted(v, 0.7, v_eq, 0.3, 0)
        
        # Merge and convert back
        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
        
        # Final contrast adjustment
        alpha = 1.1  # Contrast
        beta = 5     # Brightness
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        return result
    
    def _validate_and_refine(self, image: np.ndarray, original_metrics: Dict) -> np.ndarray:
        """Validate enhancement results and apply refinements if needed"""
        enhanced_metrics = calculate_image_metrics(image)
        
        # Check if enhancement was beneficial
        sharpness_improvement = enhanced_metrics['sharpness'] - original_metrics['sharpness']
        noise_change = enhanced_metrics['noise_level'] - original_metrics['noise_level']
        
        logger.info(f"Enhancement validation: sharpness_change={sharpness_improvement:.2f}, "
                   f"noise_change={noise_change:.2f}")
        
        # If noise increased significantly without much sharpness gain, apply additional denoising
        if noise_change > 5 and sharpness_improvement < 50:
            logger.info("Applying additional noise reduction")
            image = cv2.bilateralFilter(image, 5, 50, 50)
        
        # If over-sharpened (artifacts visible), apply smoothing
        if enhanced_metrics['sharpness'] > 500:
            logger.info("Reducing over-sharpening artifacts")
            image = cv2.GaussianBlur(image, (3, 3), 0.5)
        
        return image
    
    def enhance_for_stock_analysis(self, image: np.ndarray) -> np.ndarray:
        """Specialized enhancement for stock analysis tasks"""
        logger.info("Applying stock analysis specific enhancements")
        
        # First apply standard enhancement
        enhanced = self.enhance_image_quality(image, "balanced")
        
        # Additional product visibility enhancement
        enhanced = self._enhance_product_visibility(enhanced)
        
        # Improve text readability on products
        enhanced = self._enhance_text_readability(enhanced)
        
        return enhanced
    
    def _enhance_product_visibility(self, image: np.ndarray) -> np.ndarray:
        """Enhance visibility of products on shelves"""
        # Edge enhancement to better define product boundaries
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Dilate edges slightly
        kernel = np.ones((2, 2), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Create edge mask
        edge_mask = edges_dilated.astype(np.float32) / 255.0
        
        # Enhance areas near edges
        result = image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] = result[:, :, i] * (1.0 + edge_mask * 0.2)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _enhance_text_readability(self, image: np.ndarray) -> np.ndarray:
        """Enhance text readability on product labels"""
        # Convert to grayscale for text detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect potential text regions using morphological operations
        # Horizontal kernel to detect text lines
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
        text_regions = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_h)
        
        # Vertical kernel to detect text blocks
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
        text_regions = cv2.morphologyEx(text_regions, cv2.MORPH_OPEN, kernel_v)
        
        # Create text enhancement mask
        text_mask = cv2.threshold(text_regions, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        text_mask = text_mask.astype(np.float32) / 255.0
        
        # Apply local contrast enhancement to text regions
        result = image.copy().astype(np.float32)
        
        # Enhance contrast in text regions
        for i in range(3):
            enhanced_channel = cv2.addWeighted(
                result[:, :, i], 1.3, 
                cv2.GaussianBlur(result[:, :, i], (5, 5), 0), -0.3, 
                0
            )
            result[:, :, i] = result[:, :, i] * (1 - text_mask) + enhanced_channel * text_mask
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result