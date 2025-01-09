"""
Base Data Service Module.

This module provides the base class for data service implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class DataServiceConfig:
    """Configuration for data service."""
    name: str
    connection_string: str
    cache_size: int = 1000
    batch_size: int = 32

class BaseDataService(ABC):
    """Base class for all data services."""
    
    def __init__(self, config: DataServiceConfig):
        """Initialize the data service with configuration."""
        self.config = config
        self.cache = {}
    
    @abstractmethod
    async def connect(self):
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def fetch_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch data from the source."""
        pass
    
    @abstractmethod
    async def store_data(self, data: Dict[str, Any]) -> bool:
        """Store data to the source."""
        pass
    
    @abstractmethod
    async def prepare_training_data(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for model training."""
        pass
    
    def cache_data(self, key: str, data: Any):
        """Cache data for faster access."""
        if len(self.cache) >= self.config.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = data
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Retrieve data from cache."""
        return self.cache.get(key)
