from pydantic import BaseSettings
from typing import Optional
import torch
import os

class AIConfig(BaseSettings):
    # Model Configuration
    MODEL_NAME: str = "microsoft/Florence-2-base-ft"  # Using Florence-2 as fallback for Qwen-VL
    MODEL_CACHE_DIR: str = "./model_cache"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Processing Configuration
    MAX_IMAGE_SIZE: int = 1024
    MIN_IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 1
    
    # Stock Detection Thresholds
    LOW_STOCK_THRESHOLD: float = 0.25
    MEDIUM_STOCK_THRESHOLD: float = 0.60
    HIGH_STOCK_THRESHOLD: float = 0.85
    
    # Confidence Thresholds
    MIN_CONFIDENCE: float = 0.5
    HIGH_CONFIDENCE: float = 0.8
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    class Config:
        env_file = ".env"
        env_prefix = "AI_"

config = AIConfig()