import cv2
import numpy as np
from scipy.signal import wiener
from typing import Tuple, Optional, Dict
from ..utils.image_utils import timing_decorator, validate_image, calculate_image_metrics, logger
from ..configs.processing_config import ImageProcessingConfig

class AdvancedNoiseReducer:
    """Advanced noise reduction with automatic noise type detection"""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        logger.info("Initialized AdvancedNoiseReducer")
    
    @timing_decorator
    def reduce_noise(self, image: np.ndarray, noise_type: str = "auto", preserve_details: bool = True) -> np.ndarray:
        """
        Main noise reduction function with automatic detection and preservation options
        """
        if not validate_image(image):
            raise ValueError("Invalid image provided for noise reduction")
        
        original_metrics = calculate_image_metrics(image)
        logger.info(f"Original noise metrics: noise_level={original_metrics['noise_level']:.2f}, "
                   f"sharpness={original_metrics['sharpness']:.2f}")
        
        if noise_type == "auto":
            noise_type = self._detect_noise_type(image, original_metrics)
        
        logger.info(f"Applying {noise_type} noise reduction, preserve_details={preserve_details}")
        
        if noise_type == "gaussian":
            result = self._reduce_gaussian_noise(image, preserve_details)
        elif noise_type == "salt_pepper":
            result = self._reduce_salt_pepper_noise(image, preserve_details)
        elif noise_type == "poisson":
            result = self._reduce_poisson_noise(image, preserve_details)
        elif noise_type == "mixed":
            result = self._reduce_mixed_noise(image, preserve_details)
        else:
            result = self._general_noise_reduction(image, preserve_details)
        
        # Validate improvement
        final_metrics = calculate_image_metrics(result)
        noise_reduction = original_metrics['noise_level'] - final_metrics['noise_level']
        detail_preservation = final_metrics['sharpness'] / original_metrics['sharpness']
        
        logger.info(f"Noise reduction results: noise_reduced={noise_reduction:.2f}, "
                   f"detail_preservation={detail_preservation:.3f}")
        
        return result
    
    def _detect_noise_type(self, image: np.ndarray, metrics: Dict) -> str:
        """Advanced noise type detection using multiple indicators"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Test different filters and measure their effectiveness
        gaussian_filtered = cv2.GaussianBlur(gray, (5, 5), 1.0)
        median_filtered = cv2.medianBlur(gray, 5)
        bilateral_filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Calculate MSE for each filter
        mse_gaussian = np.mean((gray.astype(np.float32) - gaussian_filtered.astype(np.float32)) ** 2)
        mse_median = np.mean((gray.astype(np.float32) - median_filtered.astype(np.float32)) ** 2)
        mse_bilateral = np.mean((gray.astype(np.float32) - bilateral_filtered.astype(np.float32)) ** 2)
        
        # Detect impulse noise (salt and pepper)
        diff_median = np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32))
        impulse_ratio = np.sum(diff_median > 20) / gray.size
        
        # Detect periodic noise patterns
        f_transform = np.fft.fft2(gray)
        f_magnitude = np.abs(f_transform)
        high_freq_energy = np.sum(f_magnitude[gray.shape[0]//4:3*gray.shape[0]//4, 
                                              gray.shape[1]//4:3*gray.shape[1]//4])
        total_energy = np.sum(f_magnitude)
        freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # Decision logic
        if impulse_ratio > 0.01:  # More than 1% impulse noise
            return "salt_pepper"
        elif mse_median < mse_gaussian * 0.7:
            return "salt_pepper"
        elif freq_ratio > 0.3:  # High frequency content suggests Poisson
            return "poisson"
        elif metrics['noise_level'] > 20:
            return "mixed"
        else:
            return "gaussian"
    
    def _reduce_gaussian_noise(self, image: np.ndarray, preserve_details: bool) -> np.ndarray:
        """Specialized Gaussian noise reduction"""
        if preserve_details:
            # Edge-preserving bilateral filter
            result = cv2.bilateralFilter(
                image, 
                self.config.BILATERAL_D, 
                self.config.BILATERAL_SIGMA_COLOR, 
                self.config.BILATERAL_SIGMA_SPACE
            )
            
            # Additional non-local means for heavy noise
            noise_level = calculate_image_metrics(image)['noise_level']
            if noise_level > 15:
                if len(image.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(
                        result, None, 
                        self.config.DENOISE_H * 0.8, 
                        self.config.DENOISE_H * 0.8,
                        self.config.DENOISE_TEMPLATE_SIZE, 
                        self.config.DENOISE_SEARCH_SIZE
                    )
                else:
                    result = cv2.fastNlMeansDenoising(
                        result, None, 
                        self.config.DENOISE_H * 0.8,
                        self.config.DENOISE_TEMPLATE_SIZE, 
                        self.config.DENOISE_SEARCH_SIZE
                    )
        else:
            # More aggressive denoising
            if len(image.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(
                    image, None, 
                    self.config.DENOISE_H, 
                    self.config.DENOISE_H,
                    self.config.DENOISE_TEMPLATE_SIZE, 
                    self.config.DENOISE_SEARCH_SIZE
                )
            else:
                result = cv2.fastNlMeansDenoising(
                    image, None, 
                    self.config.DENOISE_H,
                    self.config.DENOISE_TEMPLATE_SIZE, 
                    self.config.DENOISE_SEARCH_SIZE
                )
        
        return result
    
    def _reduce_salt_pepper_noise(self, image: np.ndarray, preserve_details: bool) -> np.ndarray:
        """Specialized salt and pepper noise reduction"""
        if len(image.shape) == 3:
            # Apply median filter to each channel
            result = np.zeros_like(image)
            kernel_size = 3 if preserve_details else 5
            
            for i in range(3):
                result[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
        else:
            kernel_size = 3 if preserve_details else 5
            result = cv2.medianBlur(image, kernel_size)
        
        # Morphological operations for stubborn noise
        if not preserve_details:
            kernel = np.ones((3, 3), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        
        return result
    
    def _reduce_poisson_noise(self, image: np.ndarray, preserve_details: bool) -> np.ndarray:
        """Poisson noise reduction using Anscombe transform"""
        # Convert to float for processing
        img_float = image.astype(np.float32)
        
        # Anscombe transform to convert Poisson to approximately Gaussian
        epsilon = 3/8
        transformed = 2 * np.sqrt(img_float + epsilon)
        
        # Apply Gaussian denoising in transformed domain
        h_value = self.config.DENOISE_H * (0.8 if preserve_details else 1.0)
        
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                transformed.astype(np.uint8), None, h_value, h_value, 7, 21
            )
        else:
            denoised = cv2.fastNlMeansDenoising(
                transformed.astype(np.uint8), None, h_value, 7, 21
            )
        
        # Inverse Anscombe transform
        denoised_float = denoised.astype(np.float32)
        result = (denoised_float / 2) ** 2 - epsilon
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def _reduce_mixed_noise(self, image: np.ndarray, preserve_details: bool) -> np.ndarray:
        """Handle mixed noise types with multi-stage approach"""
        # Stage 1: Remove impulse noise with median filter
        if len(image.shape) == 3:
            stage1 = np.zeros_like(image)
            for i in range(3):
                stage1[:, :, i] = cv2.medianBlur(image[:, :, i], 3)
        else:
            stage1 = cv2.medianBlur(image, 3)
        
        # Stage 2: Remove Gaussian noise with bilateral filter
        stage2 = cv2.bilateralFilter(stage1, 5, 50, 50)
        
        # Stage 3: Final cleanup with non-local means if needed
        if not preserve_details:
            if len(image.shape) == 3:
                stage3 = cv2.fastNlMeansDenoisingColored(
                    stage2, None, 
                    self.config.DENOISE_H * 0.6, 
                    self.config.DENOISE_H * 0.6,
                    5, 15
                )
            else:
                stage3 = cv2.fastNlMeansDenoising(
                    stage2, None, 
                    self.config.DENOISE_H * 0.6,
                    5, 15
                )
            return stage3
        
        return stage2
    
    def _general_noise_reduction(self, image: np.ndarray, preserve_details: bool) -> np.ndarray:
        """General noise reduction for unknown noise types"""
        # Start with bilateral filter for edge preservation
        result = cv2.bilateralFilter(
            image, 
            5 if preserve_details else 9, 
            50 if preserve_details else 75, 
            50 if preserve_details else 75
        )
        
        # Additional processing based on noise level
        noise_level = calculate_image_metrics(result)['noise_level']
        
        if noise_level > 10:
            # Apply non-local means for remaining noise
            h_value = self.config.DENOISE_H * (0.7 if preserve_details else 1.0)
            
            if len(image.shape) == 3:
                result = cv2.fastNlMeansDenoisingColored(
                    result, None, h_value, h_value, 5, 15
                )
            else:
                result = cv2.fastNlMeansDenoising(
                    result, None, h_value, 5, 15
                )
        
        return result
    
    def adaptive_noise_reduction(self, image: np.ndarray, block_size: int = 64) -> np.ndarray:
        """Adaptive noise reduction based on local image characteristics"""
        h, w = image.shape[:2]
        result = image.copy()
        
        # Process image in blocks for local adaptation
        for i in range(0, h - block_size + 1, block_size // 2):
            for j in range(0, w - block_size + 1, block_size // 2):
                # Extract block
                end_i = min(i + block_size, h)
                end_j = min(j + block_size, w)
                block = image[i:end_i, j:end_j]
                
                # Analyze local noise characteristics
                local_metrics = calculate_image_metrics(block)
                local_noise = local_metrics['noise_level']
                local_sharpness = local_metrics['sharpness']
                
                # Apply appropriate denoising based on local characteristics
                if local_noise > 20:
                    # High noise area - aggressive denoising
                    processed_block = self._reduce_mixed_noise(block, preserve_details=False)
                elif local_noise > 10:
                    # Medium noise - balanced approach
                    processed_block = self._reduce_gaussian_noise(block, preserve_details=True)
                elif local_sharpness < 50:
                    # Low detail area - can afford more denoising
                    processed_block = cv2.bilateralFilter(block, 9, 75, 75)
                else:
                    # High detail area - minimal processing
                    processed_block = cv2.bilateralFilter(block, 5, 30, 30)
                
                # Blend processed block back into result
                alpha = 0.8  # Blending factor to avoid artifacts
                result[i:end_i, j:end_j] = cv2.addWeighted(
                    result[i:end_i, j:end_j], 1 - alpha,
                    processed_block, alpha, 0
                )
        
        return result
    
    def estimate_noise_characteristics(self, image: np.ndarray) -> Dict:
        """Comprehensive noise analysis for debugging and optimization"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # Basic noise level using high-pass filter
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        high_pass = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        noise_variance = np.var(high_pass)
        
        # Noise estimation using Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        
        # Estimate using median absolute deviation
        median_filtered = cv2.medianBlur(gray, 5)
        noise_mad = np.median(np.abs(gray.astype(np.float32) - median_filtered.astype(np.float32)))
        
        # Frequency domain analysis
        f_transform = np.fft.fft2(gray)
        f_magnitude = np.abs(f_transform)
        
        # High frequency energy ratio
        h, w = gray.shape
        high_freq_mask = np.zeros_like(f_magnitude)
        high_freq_mask[h//4:3*h//4, w//4:3*w//4] = 1
        high_freq_energy = np.sum(f_magnitude * high_freq_mask)
        total_energy = np.sum(f_magnitude)
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        return {
            'noise_variance': float(noise_variance),
            'laplacian_variance': float(laplacian_variance),
            'noise_mad': float(noise_mad),
            'high_freq_ratio': float(high_freq_ratio),
            'estimated_noise_level': float(noise_mad * 1.4826),  # MAD to std conversion
            'image_size': gray.shape,
            'recommended_filter': self._recommend_filter(noise_mad, high_freq_ratio)
        }
    
    def _recommend_filter(self, noise_mad: float, high_freq_ratio: float) -> str:
        """Recommend optimal filter based on noise characteristics"""
        if noise_mad > 15:
            return "aggressive_nlm"  # Non-local means
        elif high_freq_ratio > 0.3:
            return "bilateral"       # Good for texture preservation
        elif noise_mad > 8:
            return "moderate_nlm"    # Moderate non-local means
        else:
            return "light_bilateral" # Light bilateral filtering