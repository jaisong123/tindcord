#!/usr/bin/env python3
"""
Test script for Claude API integration.
This script tests the Claude API integration by sending a sample image to the API.

Usage:
    python test_claude_api.py <image_path>

Example:
    python test_claude_api.py test_image.jpg
"""

import asyncio
import argparse
import logging
import os
import sys
import yaml

from claude_integration import ClaudeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

async def test_claude_api(image_path, api_key=None, model=None):
    """Test Claude API integration with a sample image."""
    
    # Get API key from config or argument
    if not api_key:
        try:
            with open("config.yaml", "r") as file:
                cfg = yaml.safe_load(file)
                api_key = cfg["providers"]["anthropic"].get("api_key")
                if not api_key:
                    logging.error("No Claude API key found in config.yaml")
                    return False
                
                # Get model if not provided
                if not model and "/" in cfg["model"] and cfg["model"].split("/", 1)[0] == "anthropic":
                    model = cfg["model"].split("/", 1)[1]
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return False
    
    # Check if image exists
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return False
    
    # Get image content type
    content_type = "image/jpeg"  # Default
    if image_path.lower().endswith(".png"):
        content_type = "image/png"
    elif image_path.lower().endswith(".gif"):
        content_type = "image/gif"
    elif image_path.lower().endswith(".webp"):
        content_type = "image/webp"
    
    # Read image data
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Initialize Claude client
    claude_client = ClaudeClient(api_key)
    
    # Set model if provided
    if model:
        claude_client.set_model(model)
        logging.info(f"Using Claude model: {model}")
    
    # Test prompt - simulating a message Blake would receive on a dating app
    prompt = "Hey there! I like your profile pic. What are you up to this weekend?"
    
    # Get system prompt from config
    try:
        with open("config.yaml", "r") as file:
            cfg = yaml.safe_load(file)
            system_prompt = cfg.get("system_prompt", "You are a helpful assistant that analyzes images.")
    except Exception as e:
        logging.warning(f"Could not load system prompt from config: {e}")
        system_prompt = "You are a helpful assistant that analyzes images."
    
    logging.info("Using system prompt from config.yaml")
    
    logging.info(f"Sending image to Claude API: {image_path}")
    logging.info(f"Content type: {content_type}")
    logging.info(f"Image size: {len(image_data)} bytes")
    
    try:
        # Process image
        response, success = await claude_client.process_image(
            image_data=image_data,
            content_type=content_type,
            prompt=prompt,
            system_prompt=system_prompt
        )
        
        if success:
            logging.info("Claude API response successful!")
            print("\n--- Claude API Response ---\n")
            print(response)
            print("\n--------------------------\n")
            return True
        else:
            logging.error(f"Claude API response failed: {response}")
            return False
            
    except Exception as e:
        logging.exception(f"Error testing Claude API: {e}")
        return False

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Claude API integration")
    parser.add_argument("image_path", help="Path to the image file to test")
    parser.add_argument("--api-key", help="Claude API key (optional, will use config.yaml if not provided)")
    parser.add_argument("--model", help="Claude model to use (optional)")
    
    args = parser.parse_args()
    
    success = await test_claude_api(args.image_path, args.api_key, args.model)
    
    if success:
        logging.info("Claude API test completed successfully!")
        sys.exit(0)
    else:
        logging.error("Claude API test failed!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())