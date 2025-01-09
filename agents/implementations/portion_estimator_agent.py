"""
Portion Estimator Agent.

This agent specializes in estimating food portions using reference objects
in images.
"""

from typing import Dict, List, Any
from swarms import Agent
from models.model_factory import ModelFactory

class PortionEstimatorAgent:
    """Agent for estimating food portions using visual references."""
    
    # Known reference object dimensions (in centimeters)
    REFERENCE_OBJECTS = {
        "iphone_15": {"width": 7.17, "height": 14.66},
        "credit_card": {"width": 8.56, "height": 5.39},
        "dinner_plate": {"diameter": 25.4},  # Standard dinner plate
        "wine_glass": {"height": 15.24},
        "soda_can": {"height": 12.2, "diameter": 6.6},
    }
    
    def __init__(self):
        """Initialize the portion estimator agent."""
        self.vision_model = ModelFactory.create_model('claude')  # Using Claude for precise measurements
        self.agent = Agent(
            role="Portion Size Expert",
            goal=(
                "Estimate food portions by analyzing images with reference "
                "objects of known dimensions"
            ),
            backstory=(
                "I am a food portion specialist with extensive experience in "
                "estimating serving sizes using visual references. I use common "
                "objects with known dimensions to calculate accurate portion sizes."
            ),
            allow_delegation=False,
            tools=[self.vision_model.run_model],
        )
    
    async def estimate(self, image_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate food portions using reference objects in the image.
        
        Args:
            image_data: Dictionary containing image and context information
            
        Returns:
            Dictionary containing:
                - portion_sizes: Estimated portion sizes in grams/ml
                - reference_objects: Identified reference objects
                - scale_factors: Calculated scale factors
                - confidence_scores: Confidence in estimations
        """
        system_prompt = (
            "Analyze the food image and:\n"
            "1. Identify any reference objects (phones, plates, cards, etc)\n"
            "2. Calculate scale factors using known object dimensions\n"
            "3. Estimate portion sizes of food items\n"
            f"Reference object dimensions: {self.REFERENCE_OBJECTS}\n"
            "Format the response as a structured JSON"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "Please analyze this image and estimate food portions using "
                "any visible reference objects."
            )}
        ]
        
        try:
            response = await self.agent.run(messages=messages)
            return self._process_response(response)
        except Exception as e:
            raise Exception(f"Portion estimation failed: {e}")
    
    def _process_response(self, response: Dict) -> Dict[str, Any]:
        """Process and structure the agent's response."""
        return {
            "portion_sizes": response.get("portions", {}),
            "reference_objects": response.get("references", {}),
            "scale_factors": response.get("scale_factors", {}),
            "confidence_scores": response.get("confidence", {}),
            "raw_response": response
        }
