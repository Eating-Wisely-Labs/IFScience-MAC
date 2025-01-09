"""
GPT-4 Vision Model Implementation.

This module implements the GPT-4 Vision model for multimodal tasks.
"""

import base64
import os
import requests
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from ..base_multimodal_model import BaseMultiModalModel

# Load environment variables
load_dotenv()

class GPT4VisionModel(BaseMultiModalModel):
    """Implementation of GPT-4 Vision model."""

    def __init__(
            self,
            openai_api_key: str = None,
            model: str = "gpt-4-vision-preview",
            max_tokens: int = 300,
            temperature: float = 0.7,
            openai_proxy: str = None,
            timeout: int = 60,
            max_retries: int = 3
    ):
        """
        Initialize the GPT4VisionModel instance.

        Args:
            openai_api_key (str, optional): OpenAI API key. Defaults to environment variable.
            model (str, optional): Model to use. Defaults to "gpt-4-vision-preview".
            max_tokens (int, optional): Maximum tokens in response. Defaults to 300.
            temperature (float, optional): Response randomness. Defaults to 0.7.
            openai_proxy (str, optional): OpenAI API proxy URL. Defaults to environment variable.
            timeout (int, optional): Request timeout in seconds. Defaults to 60.
            max_retries (int, optional): Maximum retry attempts. Defaults to 3.
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set it in .env file or pass it to the constructor.")
            
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.openai_proxy = openai_proxy or os.getenv('OPENAI_API_PROXY', "https://api.openai.com/v1/chat/completions")
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encode an image file to base64 string.

        Args:
            image_path (str): Path to the image file

        Returns:
            str: Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def run_model(self, task: str, image_path: str, **kwargs) -> Dict[str, Any]:
        """
        Run the GPT-4 Vision model on a task and image.

        Args:
            task (str): Task description or prompt
            image_path (str): Path to the image file
            **kwargs: Additional model parameters

        Returns:
            Dict[str, Any]: Model response
        """
        try:
            base64_image = self.encode_image(image_path)
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.openai_api_key}",
            }
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": task},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }

            response = requests.post(
                self.openai_proxy,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            return response.json()

        except Exception as e:
            self.logger.error(f"Error in GPT-4 Vision API call: {str(e)}")
            raise
