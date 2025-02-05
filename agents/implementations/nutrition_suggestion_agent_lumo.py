import torch
from transformers import LlamaForCausalLM, AutoTokenizer
import logging
from typing import List, Dict

class NutritionSuggestionAgent:
    def __init__(self):
        self.model_name = "lumolabs-ai/Lumo-DeepSeek-R1-8B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initialize model and tokenizer
        self.model = LlamaForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # System prompt
        self.system_prompt = """You are a nutritionist assistant based on Lumo-DeepSeek-R1-8B model.
            When given a dish description along with per serving [calories] kcal, [fat]g, [carbs]g, [protein]g,
            provide a concise dietary tip in ≤140 characters. Highlight key nutrients or balance for a healthy meal plan."""

    async def complete_chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 128) -> str:
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True
            ).to(self.device)

            async with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logging.error(f"Error in chat completion: {e}")
            raise

    async def get_nutrition_suggestion(self, dish_description: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": dish_description}
        ]
        return await self.complete_chat(messages)

# Example usage:
async def main():
    agent = NutritionSuggestionAgent()
    response = await agent.get_nutrition_suggestion(
        "Hot pot with various meats and veggies. Estimated total: 2500 kcal, 120g fat, 200g carbs, 150g protein."
    )
    print(response)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

