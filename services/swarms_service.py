"""
Swarms Service Module.

This module provides functionality for interacting with Swarms API.
"""

# Standard library imports
import asyncio
import os
from dotenv import load_dotenv
import yaml
from typing import List, Dict

# Third-party imports
from swarms import Agent

# Local imports
from services.gpt4_vision_api import GPT4VisionAPI
from models.model_factory import ModelFactory

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


async def get_food_reply(query_text: str, query_images: list) -> dict:
    """
    Get food analysis reply using Swarms Agent with GPT-4 Vision model.

    Args:
        query_text (str): Text query about the food
        query_images (list): List of image paths

    Returns:
        dict: Response from the model
    """
    try:
        # Create GPT-4 Vision model instance
        vision_model = ModelFactory.create_model('gpt4')
        
        # Create Swarms agent with the vision model
        agent = Agent(
            role="Food Analysis Expert",
            goal="Analyze food images and provide detailed information",
            backstory="I am an expert in food analysis, capable of identifying ingredients, estimating nutritional content, and providing detailed descriptions of food items.",
            allow_delegation=False,
            tools=[vision_model.run_model],
        )
        
        messages = [{"role": "system", "content": config["system_prompt"]}, 
                    {"role": "user", "content": query_text}]
        return await get_swarms_response(messages, agent)
    except Exception as e:
        raise Exception(f"Agent processing failed: {e}") from e


async def get_swarms_response(messages: List[Dict], agent: Agent) -> Dict:
    """Process messages using the Swarms Agent and return the response.
    
    Args:
        messages (List[Dict]): List of message dictionaries to process.
        agent (Agent): Swarms Agent instance.
    
    Returns:
        Dict: The response from the Swarms Agent.
    
    Raises:
        Exception: Raises an exception if the agent encounters an error during processing.
    """
    try:
        return agent.run(messages=messages)
    except Exception as e:
        raise Exception(f"Agent processing failed: {e}") from e


if __name__ == '__main__':
    """Main function to orchestrate fetching replies for predefined comments."""
    query_text = "What's this?"
    query_images = ["https://pbs.twimg.com/media/GfPLHWNbIAAyk8g?format=jpg&name=medium"]
    try:
        content = asyncio.run(get_food_reply(query_text, query_images))
        print(content)
    except Exception as e:
        print(f"An error occurred: {e}")
