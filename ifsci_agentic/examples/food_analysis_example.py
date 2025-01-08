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
