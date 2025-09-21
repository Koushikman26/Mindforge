import cv2
import numpy as np
from typing import Dict, List, Tuple
import logging
from dataclasses import dataclass
from PIL import Image
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    product_name: str
    bounding_boxes: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    confidence_scores: List[float]
    total_area: float
    shelf_coverage: float

class ProductDetector:
    def __init__(self):
        # HSV color ranges for each product (these may need tuning based on your images)
        self.color_ranges = {
            "banana": {
                "lower": np.array([15, 100, 100], dtype=np.uint8),  # Yellow range
                "upper": np.array([35, 255, 255], dtype=np.uint8)
            },
            "broccoli": {
                "lower": np.array([35, 50, 50], dtype=np.uint8),   # Green range
                "upper": np.array([85, 255, 255], dtype=np.uint8)
            },
            "avocado": {
                "lower": np.array([35, 30, 30], dtype=np.uint8),   # Dark green range
                "upper": np.array([85, 200, 150], dtype=np.uint8)
            }
        }
    
    async def detect_product(self, image: np.ndarray, product: str) -> DetectionResult:
        """Detect specific product using color-based segmentation"""
        try:
            # Run detection in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._detect_product_sync, image, product)
            return result
            
        except Exception as e:
            logger.error(f"Product detection failed for {product}: {str(e)}")
            return DetectionResult(product, [], [], 0.0, 0.0)
    
    def _detect_product_sync(self, image: np.ndarray, product: str) -> DetectionResult:
        """Synchronous product detection"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Get color range
            color_range = self.color_ranges.get(product, {})
            if not color_range:
                logger.warning(f"No color range defined for {product}")
                return DetectionResult(product, [], [], 0.0, 0.0)
            
            # Create mask
            mask = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
            
            # Morphological operations to clean mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bounding_boxes = []
            confidence_scores = []
            total_area = 0
            
            # Process contours
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))
                    
                    # Calculate confidence based on area and compactness
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        compactness = 4 * np.pi * area / (perimeter * perimeter)
                        confidence = min(compactness * (area / 1000), 1.0)
                    else:
                        confidence = 0.5
                    
                    confidence_scores.append(confidence)
                    total_area += area
            
            # Calculate shelf coverage
            image_area = image.shape[0] * image.shape[1]
            shelf_coverage = total_area / image_area if image_area > 0 else 0.0
            
            logger.debug(f"Detected {len(bounding_boxes)} {product} objects")
            return DetectionResult(
                product_name=product,
                bounding_boxes=bounding_boxes,
                confidence_scores=confidence_scores,
                total_area=total_area,
                shelf_coverage=shelf_coverage
            )
            
        except Exception as e:
            logger.error(f"Synchronous detection failed for {product}: {str(e)}")
            return DetectionResult(product, [], [], 0.0, 0.0)
    
    async def detect_all_products(self, image: np.ndarray) -> Dict[str, DetectionResult]:
        """Detect all target products"""
        products = ["banana", "broccoli", "avocado"]
        results = {}
        
        # Run detections in parallel
        tasks = [self.detect_product(image, product) for product in products]
        detection_results = await asyncio.gather(*tasks)
        
        # Map results to products
        for product, result in zip(products, detection_results):
            results[product] = result
        
        return results

# Global detector instance
product_detector = ProductDetector()
