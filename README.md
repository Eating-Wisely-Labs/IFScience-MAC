# IFSci Agentic

IFSci Agentic is a sophisticated Large Language Model (LLM) framework developed specifically for the IFSci Server. Built on the innovative [Swarms platform](https://github.com/kyegomez/swarms), it is designed to enhance and streamline the deployment of LLM agents. This framework provides a comprehensive interface that supports a variety of LLM operations, including direct model calls and on-the-fly fine-tuning.

## Features

- **Direct LLM API Calls**: Supports models like GPT-4, GPT-4V, etc.
- **Fine-Tuning Support**: Customize models to meet specific needs.
- **Swarms Integration**: Seamlessly integrate with Swarms services.
- **Unified Interface**: Simplifies all LLM operations.


## Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_key_here
```


## Usage

Here is an example of using IFSci Agentic for food analysis.

```python
"""
Example usage of the Swarms Service Module with a custom function.

This example demonstrates how to use the module to send a request
to the Swarms API and receive a response using a customized function that
takes text and image URLs as input.
"""

# Standard library imports
import asyncio

# Import the module functions and classes from your Swarms service module
# Adjust the import path as necessary
from ifsci_agentic.services.swarms_service import get_food_reply


def prepare_comments(query_text: str, query_images: list) -> list:
    """
    Prepare comments for the Swarms API based on provided text and image URLs.

    Args:
        query_text (str): Text content for the query.
        query_images (list): List of image URLs to be included in the query.

    Returns:
        list: A structured list of comments suitable for the Swarms API.
    """
    comments = {"role": "user", "content": []}
    # Add text content
    if query_text:
        comments["content"].append({"type": "text", "text": query_text})
    # Add images
    for image_url in query_images:
        comments["content"].append({"type": "image_url", "image_url": {"url": image_url}})
    return [comments]


# Main function to execute the example
async def main():
    # Example usage of the prepare_comments function
    query_text = "What's this?"
    query_images = ["https://pbs.twimg.com/media/GfPLHWNbIAAyk8g?format=jpg&name=medium"]

    # Prepare the comments using the function
    comments = prepare_comments(query_text, query_images)

    try:
        # Invoke the get_food_reply function with the prepared comments
        response = await get_food_reply(comments)
        print("Response from Swarms Agent:")
        print(response)
    except Exception as e:
        print(f"An error occurred while fetching the reply: {e}")


# Run the main function as an asynchronous event loop
if __name__ == "__main__":
    asyncio.run(main())
```

## Contributing

To contribute to this project:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add some amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
