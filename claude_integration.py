import asyncio
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple

import anthropic
from anthropic.types import MessageParam
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Constants
MAX_RETRIES = 5
MIN_RETRY_WAIT = 1  # seconds
MAX_RETRY_WAIT = 60  # seconds

class ClaudeClient:
    """Client for interacting with Anthropic's Claude API with retry logic and error handling."""
    
    def __init__(self, api_key: str):
        """Initialize the Claude client with the provided API key."""
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = "claude-3-opus-20240229"  # Default model, can be overridden
    
    def set_model(self, model: str):
        """Set the Claude model to use."""
        self.model = model
    
    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            asyncio.TimeoutError
        )),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def process_image(
        self,
        image_data: bytes,
        content_type: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 4096
    ) -> Tuple[str, bool]:
        """
        Process an image with Claude API.
        
        Args:
            image_data: Raw image data bytes
            content_type: MIME type of the image
            prompt: User prompt to accompany the image
            system_prompt: System prompt to guide Claude's response
            max_tokens: Maximum tokens in the response
            
        Returns:
            Tuple of (response_text, success_flag)
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            
            # Prepare messages
            messages: List[MessageParam] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
            
            # Make API request
            response = await self.client.messages.create(
                model=self.model,
                messages=messages,
                system=system_prompt,
                max_tokens=max_tokens
            )
            
            return response.content[0].text, True
            
        except anthropic.RateLimitError as e:
            logging.warning(f"Rate limit exceeded: {e}. Retrying with exponential backoff...")
            raise  # Will be caught by retry decorator
            
        except anthropic.APITimeoutError as e:
            logging.warning(f"API timeout: {e}. Retrying...")
            raise  # Will be caught by retry decorator
            
        except anthropic.APIConnectionError as e:
            logging.warning(f"API connection error: {e}. Retrying...")
            raise  # Will be caught by retry decorator
            
        except anthropic.BadRequestError as e:
            logging.error(f"Bad request error: {e}")
            return f"Error processing image: {str(e)}", False
            
        except anthropic.AuthenticationError as e:
            logging.error(f"Authentication error: {e}")
            return "Error: Claude API authentication failed. Please check your API key.", False
            
        except Exception as e:
            logging.exception(f"Unexpected error processing image: {e}")
            return f"An unexpected error occurred: {str(e)}", False

    @retry(
        retry=retry_if_exception_type((
            anthropic.RateLimitError,
            anthropic.APITimeoutError,
            anthropic.APIConnectionError,
            asyncio.TimeoutError
        )),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def stream_process_image(
        self,
        image_data: bytes,
        content_type: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int = 4096
    ):
        """
        Process an image with Claude API and stream the response.
        
        Args:
            image_data: Raw image data bytes
            content_type: MIME type of the image
            prompt: User prompt to accompany the image
            system_prompt: System prompt to guide Claude's response
            max_tokens: Maximum tokens in the response
            
        Returns:
            AsyncGenerator yielding response chunks
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            
            # Prepare messages
            messages: List[MessageParam] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": content_type,
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
            
            # Make streaming API request
            stream = await self.client.messages.create(
                model=self.model,
                messages=messages,
                system=system_prompt,
                max_tokens=max_tokens,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                    yield chunk.delta.text
                    
        except Exception as e:
            logging.exception(f"Error in stream_process_image: {e}")
            yield f"Error processing image: {str(e)}"