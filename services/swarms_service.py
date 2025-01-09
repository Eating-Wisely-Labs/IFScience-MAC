"""
Swarms Service Module.

This module provides functionality for interacting with Swarms API.
"""

# Standard library imports
import asyncio
import os
from dotenv import load_dotenv
import yaml
from typing import List, Dict, Any

# Local imports
from agents.implementations.nutrition_orchestrator_agent import NutritionOrchestratorAgent

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is required. Please set it in your .env file.")


def load_config(file_path: str) -> Dict:
    """Attempt to load a configuration from a YAML file.
    
    Args:
        file_path (str): Path to the YAML configuration file.
    
    Returns:
        Dict: Parsed YAML configuration as a dictionary.
    
    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The {file_path} file is missing.") from e
    except yaml.YAMLError as e:
        raise yaml.YAMLError("Failed to parse YAML file.") from e


config = load_config("../configs/ifs_reply_agent.yaml")


async def analyze_food_image(
    query_text: str,
    query_images: List[str]
) -> Dict[str, Any]:
    """
    Analyze food image using the nutrition orchestrator agent.
    
    Args:
        query_text (str): Text query about the food
        query_images (list): List of image paths
        
    Returns:
        dict: Comprehensive food analysis including ingredients,
              portions, and nutritional information
    """
    try:
        # Initialize the orchestrator agent
        orchestrator = NutritionOrchestratorAgent()
        
        # Let the orchestrator handle the entire analysis process
        return await orchestrator.analyze_meal(query_text, query_images)
        
    except Exception as e:
        raise Exception(f"Food analysis failed: {e}") from e


if __name__ == '__main__':
    query_text = "What's this?"
    query_images = ["https://pbs.twimg.com/media/GfPLHWNbIAAyk8g?format=jpg&name=medium"]
    try:
        content = asyncio.run(analyze_food_image(query_text, query_images))
        print(content)
    except Exception as e:
        print(f"An error occurred: {e}")
