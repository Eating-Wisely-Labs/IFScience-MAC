"""
Test cases for model factory.
"""

import pytest
from models.model_factory import ModelFactory
from models.implementations.gpt4_vision import GPT4VisionModel
from models.implementations.claude_vision import ClaudeVisionModel
from models.implementations.gemini_vision import GeminiVisionModel

def test_model_factory_creation():
    """Test model factory can create different model types."""
    factory = ModelFactory()
    
    # Test GPT-4 model creation
    gpt4_model = factory.create_model('gpt4')
    assert isinstance(gpt4_model, GPT4VisionModel)
    
    # Test Claude model creation
    claude_model = factory.create_model('claude')
    assert isinstance(claude_model, ClaudeVisionModel)
    
    # Test Gemini model creation
    gemini_model = factory.create_model('gemini')
    assert isinstance(gemini_model, GeminiVisionModel)

def test_model_factory_invalid_type():
    """Test model factory handles invalid model types."""
    factory = ModelFactory()
    
    with pytest.raises(ValueError) as exc_info:
        factory.create_model('invalid_model')
    assert "Unsupported model type" in str(exc_info.value)

def test_model_factory_case_insensitive():
    """Test model factory handles case-insensitive model types."""
    factory = ModelFactory()
    
    gpt4_model = factory.create_model('GPT4')
    assert isinstance(gpt4_model, GPT4VisionModel)
    
    claude_model = factory.create_model('CLAUDE')
    assert isinstance(claude_model, ClaudeVisionModel)
