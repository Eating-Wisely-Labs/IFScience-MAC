"""
GPT-4 Vision API Module.

This module provides functionality for interacting with OpenAI's GPT-4V API.
"""

# Standard library imports
import base64
import logging
import os
from typing import Optional, Dict, Any, List

# Third-party imports
import aiohttp
import requests
from dotenv import load_dotenv
from loguru import logger

# Local imports
from ..models.base_multimodal_model import BaseMultiModalModel

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
SYSTEM_PROMPT = """
You are a multi-modal autonomous agent. You are given a task and an image. 
You must generate a response to the task and image.
"""

class GPT4VisionAPI(BaseMultiModalModel):
    """A class for interacting with the OpenAI GPT-4 Vision API."""

    def __init__(
        self,
        openai_api_key: str = OPENAI_API_KEY,
        model_name: str = "gpt-4-vision-preview",
        logging_enabled: bool = False,
        max_workers: int = 10,
        max_retries: int = 3,
        beautify: bool = False,
        streaming_enabled: Optional[bool] = False,
        meta_prompt: Optional[bool] = False,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        *args,
        **kwargs,
    ):
        """Initialize the GPT-4 Vision API.

        Args:
            openai_api_key (str): OpenAI API key
            model_name (str): Model name to use
            logging_enabled (bool): Whether to enable logging
            max_workers (int): Maximum number of workers
            max_retries (int): Maximum number of retries
            beautify (bool): Whether to beautify output
            streaming_enabled (bool): Whether to enable streaming
            meta_prompt (bool): Whether to use meta prompting
            system_prompt (str): System prompt to use
        """
        super().__init__(*args, **kwargs)
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.logging_enabled = logging_enabled
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.beautify = beautify
        self.streaming_enabled = streaming_enabled
        self.meta_prompt = meta_prompt
        self.system_prompt = system_prompt

    async def encode_image_from_url(self, image_url: str) -> str:
        """Encode image from URL to base64.

        Args:
            image_url (str): URL of the image

        Returns:
            str: Base64 encoded image
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as response:
                image_data = await response.read()
                return base64.b64encode(image_data).decode('utf-8')

    async def run_model(
        self,
        task: str,
        image_url: str,
        max_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
    ) -> Dict[str, Any]:
        """Run the GPT-4V model on a task and image.

        Args:
            task (str): Task description
            image_url (str): URL of the image
            max_tokens (int): Maximum tokens to generate
            temperature (float): Temperature for sampling
            top_p (float): Top p for nucleus sampling
            frequency_penalty (float): Frequency penalty
            presence_penalty (float): Presence penalty

        Returns:
            Dict[str, Any]: Model response
        """
        try:
            base64_image = await self.encode_image_from_url(image_url)
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": task
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
                ) as response:
                    return await response.json()

        except Exception as e:
            logger.error(f"Error in GPT4VisionAPI.run_model: {str(e)}")
            raise
