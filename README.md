# IFScience-MAC (Multi-Agent Control)

A sophisticated food recognition framework for Intermittent Fasting (IF), built on the innovative [Swarms platform](https://github.com/kyegomez/swarms). This framework provides a comprehensive multi-model, multi-agent system specifically designed for food analysis and dietary tracking in IF contexts. By leveraging Swarms' powerful agent orchestration capabilities, it enables accurate food identification, nutritional analysis, and dietary compliance verification.

## Features

### 1. Swarms Integration
- **Built on Swarms Platform**:
  - Leverages Swarms' agent orchestration
  - Inherits Swarms' parallel processing capabilities
  - Utilizes Swarms' communication protocols
  - Supports all Swarms-compatible models and tools

### 2. Food Recognition Models
- **Multiple Vision Models**:
  - GPT-4 Vision for detailed food analysis
  - Claude Vision for nutritional assessment
  - Gemini Vision for ingredient identification
- Factory pattern for easy model instantiation
- Unified interface for consistent food analysis

### 3. Multi-Agent Control
- Specialized agents for different aspects of food analysis:
  - Recognition agents for visual identification
  - Nutrition agents for dietary analysis
  - Compliance agents for IF schedule tracking
- Team-based task execution
- Collaborative food analysis
- Configurable agent memory and iteration limits

### 4. Fine-tuning Capabilities
- Custom model training for specific food types
- LoRA (Low-Rank Adaptation) integration
- Configurable training parameters
- Model evaluation and metrics tracking
- Support for 8-bit and 4-bit quantization

### 5. Data Services
- Food database integration
- Nutritional information caching
- Support for multiple dietary databases
- Training data preparation for food recognition

## Project Structure
```
IFScience-MAC/
├── agents/                    # Multi-agent control system
│   ├── base_agent.py         # Base agent interface
│   └── agent_manager.py      # Agent coordination
├── models/                    # Model implementations
│   ├── implementations/      # Specific model implementations
│   ├── base_multimodal_model.py
│   └── model_factory.py      # Factory for model creation
├── training/                  # Training services
│   ├── implementations/      # Training implementations
│   ├── base_trainer.py       # Base trainer interface
│   └── training_manager.py   # Training coordination
├── data_services/            # Data service components
│   ├── base_data_service.py  # Base data service interface
│   └── data_service_manager.py # Data service coordination
└── tests/                    # Test suite
    └── unit/                 # Unit tests
        ├── models/           # Model tests
        ├── agents/           # Agent tests
        ├── training/         # Training tests
        └── data_services/    # Data service tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/IFScience-MAC.git
cd IFScience-MAC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Usage

### Basic Usage
```python
from models.model_factory import ModelFactory
from agents.agent_manager import AgentManager

# Create a model
factory = ModelFactory()
model = factory.create_model('gpt4')

# Set up agents
manager = AgentManager()
agent = YourAgent(config)
manager.register_agent(agent)

# Execute team task
team_id = await manager.create_team(task, ["agent1", "agent2"])
result = await manager.execute_team_task(team_id)
```

### Fine-tuning Models
```python
from training.implementations.finetuning_trainer import FineTuningTrainer, FineTuningConfig

# Configure training
config = FineTuningConfig(
    model_name="custom-model",
    base_model="gpt2",
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    checkpoint_dir="./checkpoints"
)

# Initialize trainer
trainer = FineTuningTrainer(config)

# Prepare and train
data = {"train": [...], "validation": [...]}
prepared_data = await trainer.prepare_data(data)
metrics = await trainer.train(prepared_data)
```

### Using Data Services
```python
from data_services.data_service_manager import DataServiceManager

# Initialize manager
manager = DataServiceManager()
service = YourDataService(config)
manager.register_service(service)

# Fetch and prepare data
data = await manager.fetch_data("service_name", query)
training_data = await manager.prepare_training_data("service_name", query)
```

## Testing

Run the test suite:
```bash
# Run all tests
pytest tests/

# Run specific test modules
pytest tests/unit/models/
pytest tests/unit/agents/
pytest tests/unit/training/
pytest tests/unit/data_services/

# Run with coverage report
pytest --cov=. tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 Vision API
- Anthropic for Claude Vision API
- Google for Gemini Vision API
