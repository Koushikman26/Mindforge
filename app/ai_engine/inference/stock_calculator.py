import numpy as np
from typing import Dict, Any
import logging
from dataclasses import dataclass
from .product_detection import DetectionResult

logger = logging.getLogger(__name__)

@dataclass
class StockLevel:
    product_name: str
    percentage: float  # 0-100
    abundance_category: str  # "sparse", "adequate", "abundant"
    confidence: float  # 0-1
    detection_info: DetectionResult

class StockCalculator:
    def __init__(self):
        # Abundance thresholds
        self.thresholds = {
            "sparse": 30.0,
            "adequate": 70.0,
            "abundant": 70.0
        }
        
        # Product-specific parameters
        self.product_params = {
            "banana": {
                "ideal_coverage": 0.4,
                "density_multiplier": 1.2,
                "confidence_boost": 1.1
            },
            "broccoli": {
                "ideal_coverage": 0.3,
                "density_multiplier": 1.5,
                "confidence_boost": 0.9
            },
            "avocado": {
                "ideal_coverage": 0.35,
                "density_multiplier": 1.3,
                "confidence_boost": 0.95
            }
        }
    
    def calculate_stock_percentage(self, detection_result: DetectionResult) -> float:
        """Calculate stock percentage based on detection results"""
        try:
            if not detection_result.bounding_boxes:
                return 0.0
            
            # Get product parameters
            params = self.product_params.get(detection_result.product_name, {})
            ideal_coverage = params.get("ideal_coverage", 0.35)
            density_multiplier = params.get("density_multiplier", 1.0)
            
            # Calculate coverage-based percentage
            coverage_percentage = (detection_result.shelf_coverage / ideal_coverage) * 100
            
            # Apply density multiplier
            adjusted_percentage = coverage_percentage * density_multiplier
            
            # Consider number of detected objects (more objects = fuller shelf)
            object_count_factor = min(len(detection_result.bounding_boxes) / 10, 1.0)
            final_percentage = adjusted_percentage * (0.7 + 0.3 * object_count_factor)
            
            # Cap at 100%
            return min(final_percentage, 100.0)
            
        except Exception as e:
            logger.error(f"Stock percentage calculation failed: {str(e)}")
            return 0.0
    
    def determine_abundance_category(self, percentage: float) -> str:
        """Determine abundance category"""
        if percentage < self.thresholds["sparse"]:
            return "sparse"
        elif percentage < self.thresholds["adequate"]:
            return "adequate"
        else:
            return "abundant"
    
    def calculate_confidence(self, detection_result: DetectionResult, percentage: float) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence from detections
            if not detection_result.confidence_scores:
                base_confidence = 0.0
            else:
                base_confidence = np.mean(detection_result.confidence_scores)
            
            # Product-specific confidence boost
            params = self.product_params.get(detection_result.product_name, {})
            confidence_boost = params.get("confidence_boost", 1.0)
            
            # Detection count factor
            count_factor = min(len(detection_result.bounding_boxes) / 5, 1.0)
            
            # Percentage confidence (extreme values are more confident)
            if percentage < 20 or percentage > 80:
                percentage_factor = 1.1
            else:
                percentage_factor = 1.0
            
            # Final confidence calculation
            final_confidence = base_confidence * confidence_boost * (0.6 + 0.4 * count_factor) * percentage_factor
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {str(e)}")
            return 0.0
    
    def calculate_stock_level(self, detection_result: DetectionResult) -> StockLevel:
        """Calculate complete stock level assessment"""
        try:
            # Calculate percentage
            percentage = self.calculate_stock_percentage(detection_result)
            
            # Determine category
            abundance_category = self.determine_abundance_category(percentage)
            
            # Calculate confidence
            confidence = self.calculate_confidence(detection_result, percentage)
            
            logger.debug(f"Stock level for {detection_result.product_name}: "
                        f"{percentage:.1f}% ({abundance_category}), confidence: {confidence:.2f}")
            
            return StockLevel(
                product_name=detection_result.product_name,
                percentage=percentage,
                abundance_category=abundance_category,
                confidence=confidence,
                detection_info=detection_result
            )
            
        except Exception as e:
            logger.error(f"Stock level calculation failed: {str(e)}")
            return StockLevel(
                product_name=detection_result.product_name,
                percentage=0.0,
                abundance_category="unknown",
                confidence=0.0,
                detection_info=detection_result
            )
    
    def calculate_multiple_stock_levels(self, detection_results: Dict[str, DetectionResult]) -> Dict[str, StockLevel]:
        """Calculate stock levels for multiple products"""
        stock_levels = {}
        
        for product_name, detection_result in detection_results.items():
            stock_levels[product_name] = self.calculate_stock_level(detection_result)
        
        return stock_levels

# Global calculator instance
stock_calculator = StockCalculator()
