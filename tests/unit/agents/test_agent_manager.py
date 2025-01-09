"""
Test cases for agent manager.
"""

import pytest
from agents.agent_manager import AgentManager
from agents.base_agent import BaseAgent, AgentConfig

class TestAgent(BaseAgent):
    """Test agent implementation."""
    
    async def plan(self, task):
        return [{"step": 1, "action": "test"}]
    
    async def execute(self, plan):
        return {"status": "success", "result": plan}
    
    async def reflect(self, result):
        return {"reflection": "test completed"}
    
    async def collaborate(self, other_agent, task):
        return {"collaboration": "success"}

@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return AgentConfig(
        name="test_agent",
        role="tester",
        goal="testing",
        tools=["test_tool"],
        memory_size=10
    )

@pytest.fixture
def agent_manager():
    """Create a test agent manager."""
    return AgentManager()

@pytest.mark.asyncio
async def test_agent_registration(agent_manager, agent_config):
    """Test agent registration and unregistration."""
    agent = TestAgent(agent_config)
    
    # Test registration
    agent_manager.register_agent(agent)
    assert agent.config.name in agent_manager.agents
    
    # Test unregistration
    agent_manager.unregister_agent(agent.config.name)
    assert agent.config.name not in agent_manager.agents

@pytest.mark.asyncio
async def test_team_creation(agent_manager, agent_config):
    """Test team creation."""
    agent1 = TestAgent(AgentConfig(**{**agent_config.__dict__, "name": "agent1"}))
    agent2 = TestAgent(AgentConfig(**{**agent_config.__dict__, "name": "agent2"}))
    
    agent_manager.register_agent(agent1)
    agent_manager.register_agent(agent2)
    
    task = {"objective": "test task"}
    team_id = await agent_manager.create_team(task, ["agent1", "agent2"])
    
    assert team_id in agent_manager.active_tasks
    assert len(agent_manager.active_tasks[team_id]["agents"]) == 2

@pytest.mark.asyncio
async def test_team_execution(agent_manager, agent_config):
    """Test team task execution."""
    agent1 = TestAgent(AgentConfig(**{**agent_config.__dict__, "name": "agent1"}))
    agent2 = TestAgent(AgentConfig(**{**agent_config.__dict__, "name": "agent2"}))
    
    agent_manager.register_agent(agent1)
    agent_manager.register_agent(agent2)
    
    task = {"objective": "test task"}
    team_id = await agent_manager.create_team(task, ["agent1", "agent2"])
    
    result = await agent_manager.execute_team_task(team_id)
    assert result["plan"] is not None
    assert result["execution"] is not None
    assert result["reflection"] is not None
    assert len(result["collaborations"]) == 1

@pytest.mark.asyncio
async def test_invalid_team_creation(agent_manager):
    """Test team creation with invalid agents."""
    task = {"objective": "test task"}
    
    with pytest.raises(ValueError):
        await agent_manager.create_team(task, ["nonexistent_agent"])

def test_task_status(agent_manager, agent_config):
    """Test task status retrieval."""
    agent = TestAgent(agent_config)
    agent_manager.register_agent(agent)
    
    with pytest.raises(ValueError):
        agent_manager.get_task_status("nonexistent_team")

def test_available_agents(agent_manager, agent_config):
    """Test listing available agents."""
    agent1 = TestAgent(AgentConfig(**{**agent_config.__dict__, "name": "agent1"}))
    agent2 = TestAgent(AgentConfig(**{**agent_config.__dict__, "name": "agent2"}))
    
    agent_manager.register_agent(agent1)
    agent_manager.register_agent(agent2)
    
    available_agents = agent_manager.list_available_agents()
    assert len(available_agents) == 2
    assert "agent1" in available_agents
    assert "agent2" in available_agents
