"""
Test cases for data service manager.
"""

import pytest
from data_services.data_service_manager import DataServiceManager
from data_services.base_data_service import BaseDataService, DataServiceConfig

class TestDataService(BaseDataService):
    """Test data service implementation."""
    
    async def connect(self):
        self._connected = True
    
    async def disconnect(self):
        self._connected = False
    
    async def fetch_data(self, query):
        if not hasattr(self, '_connected') or not self._connected:
            raise ConnectionError("Not connected")
        return {"data": f"test_data_{query['id']}"}
    
    async def store_data(self, data):
        if not hasattr(self, '_connected') or not self._connected:
            raise ConnectionError("Not connected")
        return True
    
    async def prepare_training_data(self, query):
        if not hasattr(self, '_connected') or not self._connected:
            raise ConnectionError("Not connected")
        return {
            "train": [f"train_data_{query['id']}"],
            "validation": [f"val_data_{query['id']}"]
        }

@pytest.fixture
def service_config():
    """Create a test service configuration."""
    return DataServiceConfig(
        name="test_service",
        connection_string="test://localhost",
        cache_size=100
    )

@pytest.fixture
def data_manager():
    """Create a test data service manager."""
    return DataServiceManager()

@pytest.mark.asyncio
async def test_service_registration(data_manager, service_config):
    """Test service registration and unregistration."""
    service = TestDataService(service_config)
    
    # Test registration
    data_manager.register_service(service)
    assert service.config.name in data_manager.services
    assert not data_manager.active_connections[service.config.name]
    
    # Test unregistration
    data_manager.unregister_service(service.config.name)
    assert service.config.name not in data_manager.services
    assert service.config.name not in data_manager.active_connections

@pytest.mark.asyncio
async def test_service_connection(data_manager, service_config):
    """Test service connection management."""
    service = TestDataService(service_config)
    data_manager.register_service(service.config.name)
    
    # Test connection
    await data_manager.connect_service(service.config.name)
    assert data_manager.active_connections[service.config.name]
    
    # Test disconnection
    await data_manager.disconnect_service(service.config.name)
    assert not data_manager.active_connections[service.config.name]

@pytest.mark.asyncio
async def test_data_operations(data_manager, service_config):
    """Test data operations."""
    service = TestDataService(service_config)
    data_manager.register_service(service)
    
    # Test fetch data
    query = {"id": 1}
    data = await data_manager.fetch_data(service.config.name, query)
    assert data["data"] == "test_data_1"
    
    # Test store data
    success = await data_manager.store_data(service.config.name, {"test": "data"})
    assert success
    
    # Test prepare training data
    training_data = await data_manager.prepare_training_data(
        service.config.name,
        {"id": 2}
    )
    assert training_data["train"][0] == "train_data_2"
    assert training_data["validation"][0] == "val_data_2"

@pytest.mark.asyncio
async def test_error_handling(data_manager):
    """Test error handling."""
    # Test invalid service
    with pytest.raises(ValueError):
        await data_manager.fetch_data("nonexistent_service", {})
    
    with pytest.raises(ValueError):
        await data_manager.store_data("nonexistent_service", {})
    
    with pytest.raises(ValueError):
        await data_manager.prepare_training_data("nonexistent_service", {})

def test_service_status(data_manager, service_config):
    """Test service status retrieval."""
    service = TestDataService(service_config)
    data_manager.register_service(service)
    
    status = data_manager.get_service_status(service.config.name)
    assert status["name"] == service.config.name
    assert not status["connected"]
    assert status["config"] == service.config

def test_available_services(data_manager, service_config):
    """Test listing available services."""
    service1 = TestDataService(
        DataServiceConfig(**{**service_config.__dict__, "name": "service1"})
    )
    service2 = TestDataService(
        DataServiceConfig(**{**service_config.__dict__, "name": "service2"})
    )
    
    data_manager.register_service(service1)
    data_manager.register_service(service2)
    
    available_services = data_manager.list_available_services()
    assert len(available_services) == 2
    assert "service1" in available_services
    assert "service2" in available_services
