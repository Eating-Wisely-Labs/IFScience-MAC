"""
Base Agent Module.

This module provides the base class for all agents in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    role: str
    goal: str
    tools: List[Any]
    memory_size: int = 100
    max_iterations: int = 10

class BaseAgent(ABC):
    """Base class for all agents."""
    
    def __init__(self, config: AgentConfig):
        """Initialize the agent with configuration."""
        self.config = config
        self.memory = []
        self.current_state = {}
    
    @abstractmethod
    async def plan(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan the steps to complete a task."""
        pass
    
    @abstractmethod
    async def execute(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute a plan."""
        pass
    
    @abstractmethod
    async def reflect(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Reflect on the execution results."""
        pass
    
    def update_memory(self, event: Dict[str, Any]):
        """Update agent's memory with new events."""
        self.memory.append(event)
        if len(self.memory) > self.config.memory_size:
            self.memory.pop(0)
    
    @abstractmethod
    async def collaborate(self, other_agent: 'BaseAgent', task: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with another agent."""
        pass
