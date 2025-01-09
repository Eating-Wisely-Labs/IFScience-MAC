"""
Test cases for fine-tuning trainer.
"""

import pytest
import os
import torch
from training.implementations.finetuning_trainer import FineTuningTrainer, FineTuningConfig

@pytest.fixture
def trainer_config():
    """Create a test configuration for the trainer."""
    return FineTuningConfig(
        model_name="test-model",
        base_model="gpt2",  # Use small model for testing
        batch_size=2,
        learning_rate=1e-4,
        num_epochs=1,
        checkpoint_dir="./test_checkpoints",
        max_seq_length=128  # Small sequence length for testing
    )

@pytest.fixture
def test_data():
    """Create test data for training."""
    return {
        "train": [
            "This is a test sentence for training.",
            "Another test sentence to train on."
        ],
        "validation": [
            "This is a validation sentence.",
            "Another validation example."
        ]
    }

@pytest.mark.asyncio
async def test_prepare_data(trainer_config, test_data):
    """Test data preparation functionality."""
    trainer = FineTuningTrainer(trainer_config)
    prepared_data = await trainer.prepare_data(test_data)
    
    assert "train_dataset" in prepared_data
    assert "val_dataset" in prepared_data
    assert "tokenizer" in prepared_data
    
    # Check dataset properties
    assert len(prepared_data["train_dataset"]) == len(test_data["train"])
    assert len(prepared_data["val_dataset"]) == len(test_data["validation"])

@pytest.mark.asyncio
async def test_training_workflow(trainer_config, test_data):
    """Test the complete training workflow."""
    trainer = FineTuningTrainer(trainer_config)
    
    # Prepare data
    prepared_data = await trainer.prepare_data(test_data)
    
    # Train model
    metrics = await trainer.train(prepared_data)
    assert "train_loss" in metrics
    assert "train_steps" in metrics
    
    # Test saving
    save_path = os.path.join(trainer_config.checkpoint_dir, "test_save")
    await trainer.save_model(save_path)
    assert os.path.exists(save_path)
    
    # Test loading
    new_trainer = FineTuningTrainer(trainer_config)
    await new_trainer.load_model(save_path)
    assert new_trainer.model is not None
    assert new_trainer.tokenizer is not None

@pytest.mark.asyncio
async def test_evaluation(trainer_config, test_data):
    """Test model evaluation."""
    trainer = FineTuningTrainer(trainer_config)
    prepared_data = await trainer.prepare_data(test_data)
    await trainer.train(prepared_data)
    
    test_data = {"text": ["This is a test sentence."]}
    eval_results = await trainer.evaluate(test_data)
    
    assert "perplexity" in eval_results
    assert "eval_loss" in eval_results
    assert eval_results["perplexity"] > 0

def test_config_validation(trainer_config):
    """Test configuration validation."""
    # Test invalid batch size
    with pytest.raises(ValueError):
        invalid_config = FineTuningConfig(
            **{**trainer_config.__dict__, "batch_size": 0}
        )
        FineTuningTrainer(invalid_config)
    
    # Test invalid learning rate
    with pytest.raises(ValueError):
        invalid_config = FineTuningConfig(
            **{**trainer_config.__dict__, "learning_rate": -1}
        )
        FineTuningTrainer(invalid_config)

@pytest.mark.asyncio
async def test_lora_config(trainer_config):
    """Test LoRA configuration."""
    lora_config = {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj"]
    }
    config_with_lora = FineTuningConfig(
        **{**trainer_config.__dict__, "lora_config": lora_config}
    )
    
    trainer = FineTuningTrainer(config_with_lora)
    prepared_data = await trainer.prepare_data({"train": ["Test"], "validation": ["Test"]})
    
    # Train with LoRA
    metrics = await trainer.train(prepared_data)
    assert "train_loss" in metrics
    
    # Verify LoRA parameters exist
    for name, param in trainer.model.named_parameters():
        if "lora" in name:
            assert param.requires_grad
