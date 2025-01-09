"""
Fine-tuning Trainer Implementation.

This module provides functionality for fine-tuning language models.
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from ..base_trainer import BaseTrainer, TrainingConfig

@dataclass
class FineTuningConfig(TrainingConfig):
    """Configuration for fine-tuning."""
    base_model: str
    tokenizer_name: str = None  # If None, use base_model
    lora_config: Optional[Dict[str, Any]] = None
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    max_seq_length: int = 512
    use_8bit: bool = False
    use_4bit: bool = False

class FineTuningTrainer(BaseTrainer):
    """Trainer for fine-tuning language models."""
    
    def __init__(self, config: FineTuningConfig):
        """Initialize the fine-tuning trainer."""
        super().__init__(config)
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    async def prepare_data(self, data: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Prepare data for fine-tuning.
        
        Args:
            data: Dictionary containing 'train' and 'validation' lists of text
            
        Returns:
            Dictionary containing prepared datasets and tokenizer
        """
        # Initialize tokenizer
        tokenizer_name = self.config.tokenizer_name or self.config.base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        def tokenize_function(examples):
            """Tokenize text examples."""
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
        
        # Create datasets
        train_dataset = Dataset.from_dict({"text": data["train"]})
        val_dataset = Dataset.from_dict({"text": data["validation"]})
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
        
        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "tokenizer": self.tokenizer
        }
    
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the model using prepared data.
        
        Args:
            training_data: Dictionary containing prepared datasets and tokenizer
            
        Returns:
            Dictionary containing training results
        """
        # Initialize model
        model_kwargs = {}
        if self.config.use_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.config.use_4bit:
            model_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            **model_kwargs
        )
        
        # Apply LoRA if configured
        if self.config.lora_config:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **self.config.lora_config
            )
            self.model = get_peft_model(self.model, peft_config)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=self.config.checkpoint_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            learning_rate=self.config.learning_rate,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="none"  # Disable wandb/tensorboard logging
        )
        
        # Initialize trainer
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=training_data["train_dataset"],
            eval_dataset=training_data["val_dataset"],
            data_collator=data_collator,
        )
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save training metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_steps": train_result.global_step,
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"]
        }
        
        return metrics
    
    async def evaluate(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the fine-tuned model.
        
        Args:
            test_data: Dictionary containing test dataset
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Model must be trained before evaluation")
        
        # Prepare test dataset
        test_dataset = Dataset.from_dict({"text": test_data["text"]})
        test_dataset = test_dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                padding="max_length",
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            ),
            batched=True,
            remove_columns=test_dataset.column_names
        )
        
        # Run evaluation
        eval_results = self.trainer.evaluate(eval_dataset=test_dataset)
        
        return {
            "perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item(),
            "eval_loss": eval_results["eval_loss"],
            "eval_runtime": eval_results["eval_runtime"]
        }
    
    async def save_model(self, path: str):
        """
        Save the fine-tuned model.
        
        Args:
            path: Path to save the model
        """
        if not self.model:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config_path = os.path.join(path, "finetuning_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "base_model": self.config.base_model,
                "tokenizer_name": self.config.tokenizer_name,
                "lora_config": self.config.lora_config,
                "max_seq_length": self.config.max_seq_length,
                "use_8bit": self.config.use_8bit,
                "use_4bit": self.config.use_4bit
            }, f)
    
    async def load_model(self, path: str):
        """
        Load a fine-tuned model.
        
        Args:
            path: Path to load the model from
        """
        # Load config
        config_path = os.path.join(path, "finetuning_config.json")
        with open(config_path, "r") as f:
            saved_config = json.load(f)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Initialize model
        model_kwargs = {}
        if saved_config["use_8bit"]:
            model_kwargs["load_in_8bit"] = True
        elif saved_config["use_4bit"]:
            model_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            **model_kwargs
        )
        
        # Apply LoRA if it was used
        if saved_config["lora_config"]:
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                **saved_config["lora_config"]
            )
            self.model = get_peft_model(self.model, peft_config)
