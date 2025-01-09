"""
Food Analyzer Agent.

This agent specializes in analyzing food ingredients and nutritional information
from images.
"""

from typing import Dict, List, Any
from swarms import Agent
from models.model_factory import ModelFactory

class FoodAnalyzerAgent:
    """Agent for analyzing food ingredients and nutritional content."""
    
    def __init__(self):
        """Initialize the food analyzer agent."""
        self.vision_model = ModelFactory.create_model('gpt4')
        self.agent = Agent(
            role="Nutritional Analysis Expert",
            goal=(
                "Analyze food images to identify ingredients and their "
                "nutritional properties"
            ),
            backstory=(
                "I am a professional nutritionist with expertise in identifying "
                "ingredients and their nutritional content from visual inspection. "
                "I can break down complex dishes into their components and "
                "provide detailed nutritional information."
            ),
            allow_delegation=False,
            tools=[self.vision_model.run_model],
        )
    
    async def analyze(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze food image to identify ingredients and nutrition info.
        
        Args:
            image_data: Dictionary containing image and context information
            
        Returns:
            Dictionary containing:
                - identified_ingredients: List of ingredients
                - nutritional_info: Basic nutritional information per ingredient
                - confidence_scores: Confidence levels for identifications
        """
        system_prompt = (
            "Analyze the food image and provide:\n"
            "1. List of all visible ingredients\n"
            "2. Estimated nutritional content per ingredient\n"
            "3. Confidence score for each identification\n"
            "Format the response as a structured JSON"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "Please analyze this food image and provide detailed "
                "ingredient and nutritional information."
            )}
        ]
        
        try:
            response = await self.agent.run(messages=messages)
            return self._process_response(response)
        except Exception as e:
            raise Exception(f"Food analysis failed: {e}")
    
    def _process_response(self, response: Dict) -> Dict[str, Any]:
        """Process and structure the agent's response."""
        # Add any additional processing logic here
        return {
            "identified_ingredients": response.get("ingredients", []),
            "nutritional_info": response.get("nutrition", {}),
            "confidence_scores": response.get("confidence", {}),
            "raw_response": response
        }
