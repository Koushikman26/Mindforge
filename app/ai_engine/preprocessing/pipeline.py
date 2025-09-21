import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, Tuple, Dict, Any
import asyncio

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self):
        self.initialized = False
    
    def correct_fisheye_distortion(self, image: np.ndarray) -> np.ndarray:
        """Apply basic fisheye correction for ceiling cameras"""
        try:
            h, w = image.shape[:2]
            
            # Simple radial distortion correction
            # Create distortion map
            center_x, center_y = w // 2, h // 2
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Calculate distance from center
            dx = x - center_x
            dy = y - center_y
            distance = np.sqrt(dx**2 + dy**2)
            
            # Apply correction (simple barrel distortion model)
            max_distance = min(center_x, center_y)
            normalized_distance = distance / max_distance
            
            # Correction factor (adjust these parameters based on your camera)
            k1, k2 = 0.1, 0.02
            correction_factor = 1 + k1 * normalized_distance**2 + k2 * normalized_distance**4
            
            # Apply correction
            corrected_x = center_x + dx / correction_factor
            corrected_y = center_y + dy / correction_factor
            
            # Ensure coordinates are within bounds
            corrected_x = np.clip(corrected_x, 0, w-1).astype(np.float32)
            corrected_y = np.clip(corrected_y, 0, h-1).astype(np.float32)
            
            # Remap the image
            corrected_image = cv2.remap(image, corrected_x, corrected_y, cv2.INTER_LINEAR)
            
            logger.debug("Fisheye distortion corrected")
            return corrected_image
            
        except Exception as e:
            logger.error(f"Fisheye correction failed: {str(e)}")
            return image
    
    def apply_perspective_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply perspective correction for overhead view"""
        try:
            h, w = image.shape[:2]
            
            # Define source points (assuming overhead camera with slight angle)
            src_points = np.float32([
                [w * 0.15, h * 0.15],   # Top-left
                [w * 0.85, h * 0.15],   # Top-right
                [w * 0.85, h * 0.85],   # Bottom-right
                [w * 0.15, h * 0.85]    # Bottom-left
            ])
            
            # Destination points (rectangular view)
            dst_points = np.float32([
                [0, 0],
                [w, 0],
                [w, h],
                [0, h]
            ])
            
            # Calculate transformation matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply transformation
            corrected = cv2.warpPerspective(image, matrix, (w, h))
            
            logger.debug("Perspective correction applied")
            return corrected
            
        except Exception as e:
            logger.error(f"Perspective correction failed: {str(e)}")
            return image
    
    def enhance_lighting(self, image: np.ndarray) -> np.ndarray:
        """Enhance lighting and contrast"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge channels and convert back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            logger.debug("Lighting enhancement applied")
            return enhanced
            
        except Exception as e:
            logger.error(f"Lighting enhancement failed: {str(e)}")
            return image
    
    def resize_for_model(self, image: np.ndarray, target_size: int = 1024) -> np.ndarray:
        """Resize image for model input"""
        try:
            h, w = image.shape[:2]
            
            # Calculate scaling to fit within target_size while maintaining aspect ratio
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Pad to square if needed
            if new_w != target_size or new_h != target_size:
                # Create padded image
                padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                resized = padded
            
            logger.debug(f"Image resized to {target_size}x{target_size}")
            return resized
            
        except Exception as e:
            logger.error(f"Image resizing failed: {str(e)}")
            return image
    
    async def preprocess_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Complete preprocessing pipeline"""
        try:
            # Run preprocessing in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._preprocess_sync, image)
            return result
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_image": image
            }
    
    def _preprocess_sync(self, image: np.ndarray) -> Dict[str, Any]:
        """Synchronous preprocessing"""
        try:
            processed_image = image.copy()
            
            # Step 1: Fisheye correction
            processed_image = self.correct_fisheye_distortion(processed_image)
            
            # Step 2: Perspective correction
            processed_image = self.apply_perspective_correction(processed_image)
            
            # Step 3: Lighting enhancement
            processed_image = self.enhance_lighting(processed_image)
            
            # Step 4: Resize for model
            processed_image = self.resize_for_model(processed_image)
            
            return {
                "success": True,
                "processed_image": processed_image,
                "original_shape": image.shape,
                "final_shape": processed_image.shape
            }
            
        except Exception as e:
            logger.error(f"Synchronous preprocessing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processed_image": image
            }

# Global preprocessor instance
image_preprocessor = ImagePreprocessor()
