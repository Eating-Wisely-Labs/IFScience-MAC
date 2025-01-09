"""
Base Trainer Module.

This module provides the base class for model training implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    batch_size: int
    learning_rate: float
    num_epochs: int
    validation_split: float
    checkpoint_dir: str
    
class BaseTrainer(ABC):
    """Base class for all model trainers."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize the trainer with configuration."""
        self.config = config
        self.training_history = []
    
    @abstractmethod
    async def prepare_data(self, data: Any) -> Dict[str, Any]:
        """Prepare data for training."""
        pass
    
    @abstractmethod
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model."""
        pass
    
    @abstractmethod
    async def save_model(self, path: str):
        """Save the trained model."""
        pass
    
    @abstractmethod
    async def load_model(self, path: str):
        """Load a trained model."""
        pass
