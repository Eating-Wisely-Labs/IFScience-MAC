"""
Swarms Service Module.

This module provides functionality for interacting with Swarms API.
"""

# Standard library imports
import os
from typing import Optional, Dict, Any

# Third-party imports
import aiohttp
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SWARMS_API_KEY = os.getenv("SWARMS_API_KEY")

# Constants
SWARMS_API_BASE = "https://api.swarms.ai/v1"

class SwarmsService:
    """A class for interacting with the Swarms API."""

    def __init__(self, api_key: str = SWARMS_API_KEY):
        """Initialize the Swarms service.

        Args:
            api_key (str): Swarms API key
        """
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Swarms API key is required")

    async def get_response(
        self,
        prompt: str,
        image_url: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Get response from Swarms API.

        Args:
            prompt (str): Input prompt
            image_url (str, optional): URL of image for multi-modal tasks
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for sampling

        Returns:
            Dict[str, Any]: API response
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            if image_url:
                payload["image_url"] = image_url

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{SWARMS_API_BASE}/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    return await response.json()

        except Exception as e:
            logger.error(f"Error in SwarmsService.get_response: {str(e)}")
            raise
