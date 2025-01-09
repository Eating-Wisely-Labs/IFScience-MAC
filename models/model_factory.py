"""
Model Factory Module.

This module provides a factory for creating different multimodal model instances.
"""

from typing import Dict, Any
from .base_multimodal_model import BaseMultiModalModel
from .implementations.gpt4_vision import GPT4VisionModel
from .implementations.claude_vision import ClaudeVisionModel
from .implementations.gemini_vision import GeminiVisionModel

class ModelFactory:
    """Factory class for creating multimodal model instances."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> BaseMultiModalModel:
        """
        Create a model instance based on the specified type.

        Args:
            model_type (str): Type of model to create ('gpt4', 'claude', 'gemini')
            **kwargs: Model-specific parameters

        Returns:
            BaseMultiModalModel: An instance of the specified model

        Raises:
            ValueError: If the model type is not supported
        """
        model_map = {
            'gpt4': GPT4VisionModel,
            'claude': ClaudeVisionModel,
            'gemini': GeminiVisionModel
        }
        
        model_class = model_map.get(model_type.lower())
        if model_class:
            return model_class(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {', '.join(model_map.keys())}")
