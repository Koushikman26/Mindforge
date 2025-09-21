import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from ..utils.image_utils import timing_decorator, validate_image, calculate_image_metrics, logger
from ..configs.processing_config import ImageProcessingConfig

class AdvancedLightingCorrector:
    """Advanced lighting correction optimized for stock shelf imaging"""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        logger.info("Initialized AdvancedLightingCorrector")
    
    @timing_decorator
    def correct_lighting(self, image: np.ndarray, correction_type: str = "auto") -> np.ndarray:
        """
        Main lighting correction function with multiple correction strategies
        """
        if not validate_image(image):
            raise ValueError("Invalid image provided for lighting correction")
        
        original_metrics = calculate_image_metrics(image)
        logger.info(f"Original image metrics: brightness={original_metrics['mean_brightness']:.2f}, "
                   f"contrast={original_metrics['contrast']:.2f}")
        
        if correction_type == "auto":
            correction_type = self._detect_lighting_condition(image)
        
        if correction_type == "low_light":
            result = self._correct_low_light(image)
        elif correction_type == "high_light":
            result = self._correct_overexposure(image)
        elif correction_type == "uneven":
            result = self._correct_uneven_lighting(image)
        else:
            result = self._general_correction(image)
        
        # Validate improvement
        improved_metrics = calculate_image_metrics(result)
        logger.info(f"Corrected image metrics: brightness={improved_metrics['mean_brightness']:.2f}, "
                   f"contrast={improved_metrics['contrast']:.2f}")
        
        return result
    
    def _detect_lighting_condition(self, image: np.ndarray) -> str:
        """Detect the type of lighting issue in the image"""
        metrics = calculate_image_metrics(image)
        
        mean_brightness = metrics['mean_brightness']
        contrast = metrics['contrast']
        std_brightness = metrics['std_brightness']
        
        # Analyze lighting conditions
        if mean_brightness < 80:
            return "low_light"
        elif mean_brightness > 180:
            return "high_light"
        elif std_brightness > 80 and contrast > 150:
            return "uneven"
        else:
            return "general"
    
    def _correct_low_light(self, image: np.ndarray) -> np.ndarray:
        """Specialized correction for low-light conditions (common in stock rooms)"""
        # Convert to LAB for better luminance control
        l_channel, a_channel, b_channel = self._convert_to_lab_channels(image)
        
        # Apply adaptive histogram equalization
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT * 1.5,  # More aggressive for low light
            tileGridSize=self.config.CLAHE_TILE_SIZE
        )
        l_enhanced = clahe.apply(l_channel)
        
        # Gamma correction for low light
        gamma = 0.6  # Brighten the image
        l_gamma = self._apply_gamma_correction(l_enhanced, gamma)
        
        # Merge back
        lab_corrected = cv2.merge([l_gamma, a_channel, b_channel])
        result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        # Additional brightness boost if needed
        mean_brightness = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
        if mean_brightness < 100:
            alpha = 1.2
            beta = 20
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        return result
    
    def _correct_overexposure(self, image: np.ndarray) -> np.ndarray:
        """Correct overexposed images (bright lighting conditions)"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = self._convert_to_lab_channels(image)
        
        # Gentle CLAHE for overexposed images
        clahe = cv2.createCLAHE(
            tileGridSize=self.config.CLAHE_TILE_SIZE
        )
        l_enhanced = clahe.apply(l_channel)
        
        # Gamma correction to reduce brightness
        gamma = 1.3
        l_gamma = self._apply_gamma_correction(l_enhanced, gamma)
        
        # Merge back
        lab_corrected = cv2.merge([l_gamma, a_channel, b_channel])
        result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        # Additional contrast enhancement
        alpha = 1.1
        beta = -10
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        
        return result
    
    def _correct_uneven_lighting(self, image: np.ndarray) -> np.ndarray:
        """Correct uneven lighting (shadows, hot spots)"""
        # Convert to grayscale for background estimation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Estimate background using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Normalize by background
        normalized = cv2.divide(gray, background, scale=255.0)
        
        # Apply to all channels
        result = image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] = cv2.divide(result[:, :, i], background, scale=255.0)
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Fine-tune with CLAHE
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        clahe = cv2.createCLAHE(
            clipLimit=self.config.CLAHE_CLIP_LIMIT,
            tileGridSize=self.config.CLAHE_TILE_SIZE
        )
        l_enhanced = clahe.apply(l_channel)
        
        lab_corrected = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _general_correction(self, image: np.ndarray) -> np.ndarray:
        """General lighting correction for normal conditions"""
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel, a_channel, b_channel = self._convert_to_lab_channels(image)
        
        # Standard CLAHE application
        clahe = cv2.createCLAHE(
        )
        l_enhanced = clahe.apply(l_channel)
        
        # Optimal gamma based on image analysis
        gamma = self._calculate_optimal_gamma(image)
        l_gamma = self._apply_gamma_correction(l_enhanced, gamma)
        
        # Merge back
        lab_corrected = cv2.merge([l_gamma, a_channel, b_channel])
        result = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
        
        # Auto contrast adjustment
        result = self._auto_contrast_brightness(result)
        
        return result
    
    def _apply_gamma_correction(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """Apply gamma correction with lookup table for efficiency"""
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)
    
    def _calculate_optimal_gamma(self, image: np.ndarray) -> float:
        """Calculate optimal gamma value based on image histogram"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Adaptive gamma based on brightness distribution
        if mean_brightness < 60:
            return 0.7
        elif mean_brightness < 100:
            return 0.8
        elif mean_brightness > 220:
            return 1.4
        elif mean_brightness > 180:
            return 1.2
        else:
            return 1.0
    
    def _auto_contrast_brightness(self, image: np.ndarray) -> np.ndarray:
        """Automatically adjust contrast and brightness using histogram analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def _auto_contrast_brightness(self, image: np.ndarray) -> np.ndarray:
        """Automatically adjust contrast and brightness using histogram analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate percentiles for robust contrast stretching
        p2 = np.percentile(gray, 2)
        p98 = np.percentile(gray, 98)
        
        # Calculate alpha (contrast) and beta (brightness)
        alpha = 255.0 / (p98 - p2) if p98 - p2 > 0 else 1.0
        beta = -p2 * alpha
        
        # Constrain to reasonable ranges
        alpha = np.clip(alpha, self.config.CONTRAST_ALPHA_RANGE[0], self.config.CONTRAST_ALPHA_RANGE[1])
        beta = np.clip(beta, self.config.BRIGHTNESS_BETA_RANGE[0], self.config.BRIGHTNESS_BETA_RANGE[1])
        
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def _convert_to_lab_channels(self, image: np.ndarray):
        """Convert BGR image to LAB and split channels."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        return l_channel, a_channel, b_channel
    
    def correct_shelf_lighting(self, image: np.ndarray) -> np.ndarray:
        # Detect if this looks like a shelf image
        if not self._is_shelf_image(image):
            return self.correct_lighting(image, "auto")
        
        # Use uneven lighting correction as shelves often have uneven lighting
        result = self._correct_uneven_lighting(image)
        
        # Additional product visibility enhancement
        result = self._enhance_product_visibility(result)
        
        return result
    
    def _is_shelf_image(self, image: np.ndarray) -> bool:
        """Detect if image contains retail shelves"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect horizontal lines (typical of shelves)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            horizontal_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                if abs(angle) < 15 or abs(angle) > 165:  # Nearly horizontal
                    horizontal_lines += 1
            
            # If we have multiple horizontal lines, likely a shelf
            return horizontal_lines > 3
        
        return False
    
    def _enhance_product_visibility(self, image: np.ndarray) -> np.ndarray:
        """Enhance visibility of products on shelves"""
        # Convert to HSV for better color manipulation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Enhance saturation to make products more visible
        s_enhanced = cv2.multiply(s, 1.15)
        s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
        
        # Slight value enhancement
        v_enhanced = cv2.multiply(v, 1.05)
        v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)
        
        # Merge and convert back
        hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)