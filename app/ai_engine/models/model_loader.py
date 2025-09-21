import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import logging
from typing import Optional, Dict, Any
from configs.config import config
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import base64

logger = logging.getLogger(__name__)

class VisionLanguageModel:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = config.DEVICE
        self.model_name = config.MODEL_NAME
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def load_model(self) -> bool:
        """Load the vision-language model asynchronously"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Run model loading in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model_sync)
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        try:
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir=config.MODEL_CACHE_DIR
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=config.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            if self.device == "cpu":
                self.model = self.model.to("cpu")
                
        except Exception as e:
            logger.error(f"Synchronous model loading failed: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None and self.processor is not None
    
    async def process_image_with_prompt(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Process image with text prompt asynchronously"""
        if not self.is_loaded():
            raise ValueError("Model not loaded")
        
        try:
            # Run inference in thread to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._process_image_sync, 
                image, 
                prompt
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }
    
    def _process_image_sync(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Synchronous image processing"""
        try:
            # Prepare inputs
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Process with model
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up the response (remove the prompt part)
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return {
                "success": True,
                "response": response,
                "confidence": 0.8  # Placeholder - implement proper confidence scoring
            }
            
        except Exception as e:
            logger.error(f"Synchronous processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }

# Global model instance
vl_model = VisionLanguageModel()
