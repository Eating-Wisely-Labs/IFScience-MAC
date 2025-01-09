"""
Test configuration and fixtures.
"""

import os
import pytest
from dotenv import load_dotenv

# Load environment variables for testing
load_dotenv(".env.test", override=True)

@pytest.fixture(scope="session")
def test_data_dir():
    """Return the path to the test data directory."""
    return os.path.join(os.path.dirname(__file__), "data")

@pytest.fixture(scope="session")
def test_models_dir():
    """Return the path to the test models directory."""
    return os.path.join(os.path.dirname(__file__), "models")
