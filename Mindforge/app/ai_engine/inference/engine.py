import cv2
import numpy as np
from PIL import Image
import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
import time

from models.model_loader import vl_model
from preprocessing.pipeline import image_preprocessor
from .product_detection import product_detector
from .stock_calculator import stock_calculator
from configs.config import config

logger = logging.getLogger(__name__)

class InferenceEngine:
    def __init__(self):
        self.model = vl_model
        self.preprocessor = image_preprocessor
        self.detector = product_detector
        self.calculator = stock_calculator
        self.is_initialized = False
        self.stats = {
            "total_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "average_processing_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the inference engine"""
        try:
            logger.info("Initializing inference engine...")
            
            # Load the VL model
            success = await self.model.load_model()
            if not success:
                logger.warning("VL model failed to load, using computer vision fallback")
            
            self.is_initialized = True
            logger.info("Inference engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Inference engine initialization failed: {str(e)}")
            return False
    
    async def analyze_stock_levels(self, image: np.ndarray) -> Dict[str, Any]:
        """Complete stock level analysis pipeline"""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                return {"success": False, "error": "Inference engine not initialized"}
            
            logger.info("Starting stock level analysis")
            
            # Step 1: Preprocess image
            preprocess_result = await self.preprocessor.preprocess_image(image)
            if not preprocess_result["success"]:
                logger.warning("Preprocessing failed, using original image")
                processed_image = image
            else:
                processed_image = preprocess_result["processed_image"]
            
            # Step 2: Computer vision detection
            cv_detection_results = await self.detector.detect_all_products(processed_image)
            
            # Step 3: Calculate stock levels
            stock_levels = self.calculator.calculate_multiple_stock_levels(cv_detection_results)
            
            # Step 4: Generate VL model analysis (if available)
            vl_analysis = None
            if self.model.is_loaded():
                try:
                    vl_analysis = await self._get_vl_analysis(processed_image)
                except Exception as e:
                    logger.warning(f"VL model analysis failed: {str(e)}")
            
            # Step 5: Generate summary
            summary = self._generate_analysis_summary(stock_levels)
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(True, processing_time)
            
            return {
                "success": True,
                "stock_levels": {name: self._format_stock_level(sl) for name, sl in stock_levels.items()},
                "summary": summary,
                "vl_analysis": vl_analysis,
                "processing_info": {
                    "image_shape": image.shape,
                    "preprocessed": preprocess_result["success"],
                    "vl_model_used": vl_analysis is not None,
                    "processing_time": round(processing_time, 2)
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_stats(False, processing_time)
            logger.error(f"Stock level analysis failed: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _get_vl_analysis(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Get Vision-Language model analysis"""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Create analysis prompt
            prompt = """
            Analyze this supermarket shelf image from an overhead camera view. 
            Identify and assess stock levels for banana, broccoli, and avocado.
            
            For each visible product, provide:
            1. Stock level (percentage: 0-100%)
            2. Abundance assessment (sparse/adequate/abundant)
            3. Location description
            
            Respond in JSON format with structure:
            {
                "banana": {"percentage": 75, "abundance": "abundant", "location": "left side"},
                "broccoli": {"percentage": 30, "abundance": "sparse", "location": "center"},
                "avocado": {"percentage": 60, "abundance": "adequate", "location": "right side"}
            }
            """
            
            # Get VL model analysis
            result = await self.model.process_image_with_prompt(pil_image, prompt)
            
            if result["success"]:
                return {
                    "vl_response": result["response"],
                    "vl_confidence": result["confidence"],
                    "analysis_attempted": True
                }
            else:
                return {
                    "vl_response": None,
                    "vl_confidence": 0.0,
                    "analysis_attempted": True,
                    "error": result.get("error", "VL analysis failed")
                }
                
        except Exception as e:
            logger.error(f"VL analysis failed: {str(e)}")
            return {
                "vl_response": None,
                "vl_confidence": 0.0,
                "analysis_attempted": False,
                "error": str(e)
            }
    
    def _format_stock_level(self, stock_level) -> Dict[str, Any]:
        """Format stock level for API response"""
        return {
            "product_name": stock_level.product_name,
            "stock_percentage": round(stock_level.percentage, 1),
            "abundance_level": stock_level.abundance_category,
            "confidence": round(stock_level.confidence, 2),
            "detected_objects": len(stock_level.detection_info.bounding_boxes),
            "needs_restocking": stock_level.abundance_category == "sparse",
            "shelf_coverage": round(stock_level.detection_info.shelf_coverage * 100, 1),
            "bounding_boxes": stock_level.detection_info.bounding_boxes[:5]  # Limit for API response
        }
    
    def _generate_analysis_summary(self, stock_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Generate analysis summary"""
        try:
            total_products = len(stock_levels)
            sparse_count = sum(1 for sl in stock_levels.values() if sl.abundance_category == "sparse")
            adequate_count = sum(1 for sl in stock_levels.values() if sl.abundance_category == "adequate")
            abundant_count = sum(1 for sl in stock_levels.values() if sl.abundance_category == "abundant")
            
            # Overall status determination
            if sparse_count > total_products / 2:
                overall_status = "needs_restocking"
                priority_level = "high"
            elif sparse_count > 0:
                overall_status = "mixed"
                priority_level = "medium"
            elif abundant_count > total_products / 2:
                overall_status = "well_stocked"
                priority_level = "low"
            else:
                overall_status = "adequate"
                priority_level = "low"
            
            # Products needing immediate attention
            needs_attention = []
            for sl in stock_levels.values():
                if sl.abundance_category == "sparse":
                    needs_attention.append({
                        "product": sl.product_name,
                        "percentage": round(sl.percentage, 1),
                        "urgency": "high" if sl.percentage < 15 else "medium"
                    })
            
            # Average confidence across all detections
            confidences = [sl.confidence for sl in stock_levels.values()]
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Generate actionable recommendations
            recommendations = []
            for product_name, stock_level in stock_levels.items():
                if stock_level.abundance_category == "sparse":
                    if stock_level.percentage < 15:
                        recommendations.append(f"URGENT: Restock {product_name} immediately - critically low at {stock_level.percentage:.0f}%")
                    else:
                        recommendations.append(f"Restock {product_name} soon - low at {stock_level.percentage:.0f}%")
                elif stock_level.abundance_category == "adequate" and stock_level.percentage < 45:
                    recommendations.append(f"Monitor {product_name} - may need restocking within next few hours ({stock_level.percentage:.0f}% full)")
            
            if not recommendations:
                recommendations.append("All products are well stocked - no immediate action required")
            
            # Calculate total detected objects across all products
            total_detected_objects = sum(len(sl.detection_info.bounding_boxes) for sl in stock_levels.values())
            
            return {
                "overall_status": overall_status,
                "priority_level": priority_level,
                "total_products": total_products,
                "sparse_count": sparse_count,
                "adequate_count": adequate_count,
                "abundant_count": abundant_count,
                "needs_attention": needs_attention,
                "average_confidence": round(avg_confidence, 2),
                "recommendations": recommendations,
                "total_detected_objects": total_detected_objects,
                "analysis_quality": "high" if avg_confidence > 0.7 else "medium" if avg_confidence > 0.4 else "low"
            }
            
        except Exception as e:
            logger.error(f"Summary generation failed: {str(e)}")
            return {
                "overall_status": "error",
                "priority_level": "unknown",
                "error": str(e),
                "recommendations": ["Unable to generate summary - check system logs"]
            }
    
    async def analyze_single_product(self, image: np.ndarray, product: str) -> Dict[str, Any]:
        """Analyze stock level for a single specific product"""
        try:
            if not self.is_initialized:
                return {"success": False, "error": "Inference engine not initialized"}
            
            if product not in ["banana", "broccoli", "avocado"]:
                return {"success": False, "error": f"Unsupported product: {product}"}
            
            logger.info(f"Analyzing stock level for {product}")
            
            # Preprocess image
            preprocess_result = await self.preprocessor.preprocess_image(image)
            processed_image = preprocess_result.get("processed_image", image)
            
            # Detect specific product
            detection_result = await self.detector.detect_product(processed_image, product)
            
            # Calculate stock level
            stock_level = self.calculator.calculate_stock_level(detection_result)
            
            return {
                "success": True,
                "product": product,
                "stock_level": self._format_stock_level(stock_level),
                "processing_info": {
                    "image_shape": image.shape,
                    "preprocessed": preprocess_result["success"]
                }
            }
            
        except Exception as e:
            logger.error(f"Single product analysis failed for {product}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics"""
        try:
            return {
                "is_initialized": self.is_initialized,
                "model_loaded": self.model.is_loaded() if self.is_initialized else False,
                "supported_products": ["banana", "broccoli", "avocado"],
                "device": config.DEVICE,
                "model_name": config.MODEL_NAME,
                "thresholds": {
                    "sparse": config.LOW_STOCK_THRESHOLD * 100,
                    "adequate": config.MEDIUM_STOCK_THRESHOLD * 100,
                    "abundant": config.HIGH_STOCK_THRESHOLD * 100
                },
                "max_image_size": config.MAX_IMAGE_SIZE,
                "processing_stats": self.stats
            }
        except Exception as e:
            logger.error(f"Failed to get system stats: {str(e)}")
            return {"error": str(e)}
    
    def _update_stats(self, success: bool, processing_time: float):
        """Update processing statistics"""
        try:
            self.stats["total_processed"] += 1
            
            if success:
                self.stats["successful_analyses"] += 1
            else:
                self.stats["failed_analyses"] += 1
            
            # Update average processing time
            current_avg = self.stats["average_processing_time"]
            total = self.stats["total_processed"]
            self.stats["average_processing_time"] = ((current_avg * (total - 1)) + processing_time) / total
            
        except Exception as e:
            logger.error(f"Failed to update stats: {str(e)}")

# Global inference engine
inference_engine = InferenceEngine()
