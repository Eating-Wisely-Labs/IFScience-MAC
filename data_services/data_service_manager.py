"""
Data Service Manager Module.

This module provides functionality to manage different data services.
"""

from typing import Dict, List, Any, Optional
from .base_data_service import BaseDataService, DataServiceConfig

class DataServiceManager:
    """Manages multiple data services."""
    
    def __init__(self):
        """Initialize the data service manager."""
        self.services: Dict[str, BaseDataService] = {}
        self.active_connections: Dict[str, bool] = {}
    
    def register_service(self, service: BaseDataService):
        """Register a new data service."""
        self.services[service.config.name] = service
        self.active_connections[service.config.name] = False
    
    def unregister_service(self, service_name: str):
        """Unregister a data service."""
        if service_name in self.services:
            del self.services[service_name]
            del self.active_connections[service_name]
    
    async def connect_service(self, service_name: str):
        """Connect to a specific data service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        service = self.services[service_name]
        await service.connect()
        self.active_connections[service_name] = True
    
    async def disconnect_service(self, service_name: str):
        """Disconnect from a specific data service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        service = self.services[service_name]
        await service.disconnect()
        self.active_connections[service_name] = False
    
    async def fetch_data(
        self,
        service_name: str,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch data from a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        if not self.active_connections[service_name]:
            await self.connect_service(service_name)
        
        service = self.services[service_name]
        return await service.fetch_data(query)
    
    async def store_data(
        self,
        service_name: str,
        data: Dict[str, Any]
    ) -> bool:
        """Store data to a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        if not self.active_connections[service_name]:
            await self.connect_service(service_name)
        
        service = self.services[service_name]
        return await service.store_data(data)
    
    async def prepare_training_data(
        self,
        service_name: str,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare training data from a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        if not self.active_connections[service_name]:
            await self.connect_service(service_name)
        
        service = self.services[service_name]
        return await service.prepare_training_data(query)
    
    def list_available_services(self) -> List[str]:
        """List all available data services."""
        return list(self.services.keys())
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get the status of a specific service."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} not found")
        
        return {
            "name": service_name,
            "connected": self.active_connections[service_name],
            "config": self.services[service_name].config
        }
