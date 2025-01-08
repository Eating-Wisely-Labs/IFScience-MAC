"""
Base Multimodal Model Module.

This module provides the base class for multimodal models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseMultiModalModel(ABC):
    """Abstract base class for multimodal models."""

    @abstractmethod
    async def run_model(self, task: str, image_url: str, **kwargs) -> Dict[str, Any]:
        """Run the model on a task and image.

        Args:
            task (str): Task description
            image_url (str): URL of the image
            **kwargs: Additional model-specific parameters

        Returns:
            Dict[str, Any]: Model response
        """
        pass
