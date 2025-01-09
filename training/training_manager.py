"""
Training Manager Module.

This module provides functionality to manage model training processes.
"""

import os
from typing import Dict, List, Any, Optional
from .base_trainer import BaseTrainer, TrainingConfig

class TrainingManager:
    """Manages model training processes."""
    
    def __init__(self):
        """Initialize the training manager."""
        self.trainers: Dict[str, BaseTrainer] = {}
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
    
    def register_trainer(self, name: str, trainer: BaseTrainer):
        """Register a new trainer."""
        self.trainers[name] = trainer
    
    def unregister_trainer(self, name: str):
        """Unregister a trainer."""
        if name in self.trainers:
            del self.trainers[name]
    
    async def start_training_job(
        self,
        trainer_name: str,
        data: Any,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new training job."""
        if trainer_name not in self.trainers:
            raise ValueError(f"Trainer {trainer_name} not found")
        
        job_id = f"job_{len(self.active_jobs)}"
        trainer = self.trainers[trainer_name]
        
        self.active_jobs[job_id] = {
            "trainer": trainer_name,
            "status": "preparing",
            "config": config or {}
        }
        
        try:
            # Prepare training data
            training_data = await trainer.prepare_data(data)
            
            # Update job status
            self.active_jobs[job_id].update({
                "status": "training",
                "training_data": training_data
            })
            
            # Start training
            results = await trainer.train(training_data)
            
            # Save the model
            model_path = os.path.join(trainer.config.checkpoint_dir, job_id)
            await trainer.save_model(model_path)
            
            # Update job status
            self.active_jobs[job_id].update({
                "status": "completed",
                "results": results,
                "model_path": model_path
            })
            
            return job_id
            
        except Exception as e:
            self.active_jobs[job_id].update({
                "status": "failed",
                "error": str(e)
            })
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a training job."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.active_jobs[job_id]
    
    def list_available_trainers(self) -> List[str]:
        """List all available trainers."""
        return list(self.trainers.keys())
    
    async def evaluate_model(self, job_id: str, test_data: Any) -> Dict[str, Any]:
        """Evaluate a trained model."""
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job_info = self.active_jobs[job_id]
        if job_info["status"] != "completed":
            raise ValueError(f"Job {job_id} is not completed")
        
        trainer = self.trainers[job_info["trainer"]]
        await trainer.load_model(job_info["model_path"])
        
        return await trainer.evaluate(test_data)
