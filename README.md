# IFSci Agentic

IFSci Agentic is a Large Language Model (LLM) framework designed to serve IFSci Server. It provides a unified interface for various LLM operations including direct model calls, fine-tuning, and RAG implementations.

## Features

- Direct LLM API calls (GPT-4, GPT-4V, etc.)
- Support for fine-tuning operations
- RAG (Retrieval Augmented Generation) implementations
- Swarms integration
- Unified interface for all LLM operations

## Installation

```bash
pip install -e .
```

## Usage

```python
from ifsci_agentic.services.gpt4_vision import GPT4VisionAPI
from ifsci_agentic.services.swarms import SwarmsService

# Using GPT-4V
vision_api = GPT4VisionAPI()
response = await vision_api.run_model(task="Describe this image", image_url="...")

# Using Swarms
swarms_service = SwarmsService()
response = await swarms_service.get_response(prompt="...", image_url="...")
```

## Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_key_here
SWARMS_API_KEY=your_key_here
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
