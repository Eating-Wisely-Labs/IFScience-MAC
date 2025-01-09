"""
Nutrition Orchestrator Agent.

This agent orchestrates the collaboration between multiple specialized agents
to provide comprehensive nutritional analysis for intermittent fasting.
"""

from typing import Dict, List, Any
from swarms import Agent, MultiAgentController
from models.model_factory import ModelFactory
from agents.implementations.food_analyzer_agent import FoodAnalyzerAgent
from agents.implementations.portion_estimator_agent import PortionEstimatorAgent

class NutritionOrchestratorAgent:
    """
    Orchestrator agent that coordinates multiple specialized agents
    for comprehensive nutritional analysis.
    """
    
    def __init__(self):
        """Initialize the nutrition orchestrator agent."""
        self.vision_model = ModelFactory.create_model('gemini')
        self.controller = MultiAgentController()
        self.food_analyzer = FoodAnalyzerAgent()
        self.portion_estimator = PortionEstimatorAgent()
        self.agent = Agent(
            role="Nutrition Analysis Orchestrator",
            goal=(
                "Coordinate multiple specialized agents to provide comprehensive "
                "nutritional analysis for intermittent fasting"
            ),
            backstory=(
                "I am an AI orchestrator specialized in coordinating multiple "
                "expert agents for nutritional analysis. I understand how to "
                "combine and validate insights from different specialists to "
                "create accurate and comprehensive dietary assessments for "
                "intermittent fasting practitioners."
            ),
            allow_delegation=True,  # Enable delegation to sub-agents
            tools=[
                self.vision_model.run_model,
                self.controller.delegate_task,
                self.controller.aggregate_results
            ],
        )
    
    async def analyze_meal(
        self,
        query_text: str,
        query_images: List[str]
    ) -> Dict[str, Any]:
        """
        Orchestrate complete meal analysis using specialized agents.
        
        Args:
            query_text: Text query about the food
            query_images: List of image paths
            
        Returns:
            Complete analysis including ingredients, portions, and IF recommendations
        """
        try:
            # Prepare image data
            image_data = {
                "query_text": query_text,
                "images": query_images
            }
            
            # Step 1: Analyze ingredients and basic nutrition
            ingredient_analysis = await self.food_analyzer.analyze(image_data)
            
            # Step 2: Estimate portion sizes
            portion_estimates = await self.portion_estimator.estimate(image_data)
            
            # Step 3: Orchestrate comprehensive analysis
            final_analysis = await self.orchestrate_analysis(
                ingredient_analysis,
                portion_estimates
            )
            
            return {
                "ingredient_analysis": ingredient_analysis,
                "portion_estimates": portion_estimates,
                "nutritional_analysis": final_analysis,
                "summary": {
                    "total_calories": final_analysis["total_nutrition"]["calories"],
                    "meal_type": final_analysis["meal_analysis"]["meal_type"],
                    "if_window": final_analysis["if_recommendations"]["eating_window"],
                    "confidence": final_analysis["confidence_metrics"]["overall_confidence"]
                }
            }
            
        except Exception as e:
            raise Exception(f"Meal analysis orchestration failed: {e}")
    
    async def orchestrate_analysis(
        self,
        ingredient_analysis: Dict[str, Any],
        portion_estimates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Orchestrate the nutritional analysis process using multiple agents.
        
        Args:
            ingredient_analysis: Output from FoodAnalyzerAgent
            portion_estimates: Output from PortionEstimatorAgent
            
        Returns:
            Dictionary containing:
                - total_nutrition: Aggregated nutritional values
                - meal_analysis: Comprehensive meal assessment
                - if_recommendations: Intermittent fasting specific advice
                - agent_contributions: Individual agent insights
                - confidence_metrics: Multi-agent confidence scores
        """
        system_prompt = (
            "Orchestrate nutritional analysis by:\n"
            "1. Validating and cross-referencing agent inputs\n"
            "2. Resolving any conflicts in analysis\n"
            "3. Generating IF-specific insights\n"
            "4. Providing confidence metrics for each insight\n"
            "Format the response as a structured JSON"
        )
        
        # Prepare context from agent analyses
        context = {
            "ingredients": {
                "data": ingredient_analysis["identified_ingredients"],
                "confidence": ingredient_analysis["confidence_scores"],
                "source": "food_analyzer_agent"
            },
            "portions": {
                "data": portion_estimates["portion_sizes"],
                "confidence": portion_estimates["confidence_scores"],
                "source": "portion_estimator_agent"
            }
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                "Please orchestrate a comprehensive nutritional analysis "
                f"using the following agent outputs: {context}"
            )}
        ]
        
        try:
            # Delegate sub-tasks to specialized processing agents
            macro_analysis = await self.controller.delegate_task(
                "macro_calculator",
                ingredient_analysis,
                portion_estimates
            )
            
            micro_analysis = await self.controller.delegate_task(
                "micro_calculator",
                ingredient_analysis,
                portion_estimates
            )
            
            if_impact = await self.controller.delegate_task(
                "if_advisor",
                {
                    "macro_analysis": macro_analysis,
                    "micro_analysis": micro_analysis
                }
            )
            
            # Aggregate all results
            final_results = await self.controller.aggregate_results([
                macro_analysis,
                micro_analysis,
                if_impact
            ])
            
            # Get orchestrator's final analysis
            orchestrator_response = await self.agent.run(messages=messages)
            
            return self._process_response(orchestrator_response, final_results)
            
        except Exception as e:
            raise Exception(f"Nutrition orchestration failed: {e}")
    
    def _process_response(
        self,
        orchestrator_response: Dict,
        agent_results: Dict
    ) -> Dict[str, Any]:
        """Process and structure the orchestrated response."""
        return {
            "total_nutrition": {
                "calories": agent_results.get("total_calories", 0),
                "macronutrients": agent_results.get("macros", {}),
                "micronutrients": agent_results.get("micros", {})
            },
            "meal_analysis": {
                "meal_type": agent_results.get("meal_type", "unknown"),
                "portion_size": agent_results.get("total_portion", "unknown"),
                "glycemic_load": agent_results.get("glycemic_load", "unknown"),
                "dietary_tags": agent_results.get("dietary_tags", [])
            },
            "if_recommendations": {
                "eating_window": agent_results.get("if_window", {}),
                "meal_timing": agent_results.get("meal_timing", {}),
                "next_meal_advice": agent_results.get("next_meal_advice", "")
            },
            "agent_contributions": {
                "macro_calculator": agent_results.get("macro_insights", {}),
                "micro_calculator": agent_results.get("micro_insights", {}),
                "if_advisor": agent_results.get("if_recommendations", {})
            },
            "confidence_metrics": {
                "overall_confidence": orchestrator_response.get("confidence", 0.0),
                "agent_confidence": agent_results.get("confidence_scores", {}),
                "conflict_resolution": orchestrator_response.get("conflicts_resolved", [])
            },
            "raw_response": orchestrator_response
        }
