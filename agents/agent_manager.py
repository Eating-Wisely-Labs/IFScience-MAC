"""
Agent Manager Module.

This module provides functionality to manage and coordinate multiple agents.
"""

from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentConfig

class AgentManager:
    """Manages multiple agents and their interactions."""
    
    def __init__(self):
        """Initialize the agent manager."""
        self.agents: Dict[str, BaseAgent] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent with the manager."""
        self.agents[agent.config.name] = agent
    
    def unregister_agent(self, agent_name: str):
        """Unregister an agent from the manager."""
        if agent_name in self.agents:
            del self.agents[agent_name]
    
    async def create_team(self, task: Dict[str, Any], agent_names: List[str]) -> str:
        """Create a team of agents for a specific task."""
        team_id = f"team_{len(self.active_tasks)}"
        team_agents = {name: self.agents[name] for name in agent_names if name in self.agents}
        
        if not team_agents:
            raise ValueError("No valid agents specified for team creation")
        
        self.active_tasks[team_id] = {
            "task": task,
            "agents": team_agents,
            "status": "created"
        }
        
        return team_id
    
    async def execute_team_task(self, team_id: str) -> Dict[str, Any]:
        """Execute a task with a team of agents."""
        if team_id not in self.active_tasks:
            raise ValueError(f"Team {team_id} not found")
        
        task_info = self.active_tasks[team_id]
        task_info["status"] = "running"
        
        try:
            # Create initial plan with the first agent
            primary_agent = list(task_info["agents"].values())[0]
            plan = await primary_agent.plan(task_info["task"])
            
            # Collaborate with other agents
            results = []
            for agent in task_info["agents"].values():
                if agent != primary_agent:
                    result = await agent.collaborate(primary_agent, task_info["task"])
                    results.append(result)
            
            # Execute the plan
            final_result = await primary_agent.execute(plan)
            
            # Reflect on results
            reflection = await primary_agent.reflect(final_result)
            
            task_info["status"] = "completed"
            task_info["result"] = {
                "plan": plan,
                "execution": final_result,
                "reflection": reflection,
                "collaborations": results
            }
            
            return task_info["result"]
            
        except Exception as e:
            task_info["status"] = "failed"
            task_info["error"] = str(e)
            raise
    
    def get_task_status(self, team_id: str) -> Dict[str, Any]:
        """Get the status of a team task."""
        if team_id not in self.active_tasks:
            raise ValueError(f"Team {team_id} not found")
        return self.active_tasks[team_id]
    
    def list_available_agents(self) -> List[str]:
        """List all available agents."""
        return list(self.agents.keys())
