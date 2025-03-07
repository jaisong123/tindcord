#!/usr/bin/env python3
"""
Test script for using Mistral OCR API to extract text from dating app screenshots.
"""

import asyncio
import base64
import logging
import os
import sys
from typing import Optional

import httpx
import yaml
from mistralai import Mistral
from mistralai.models.ocr import OcrProcessParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Mistral API key
MISTRAL_API_KEY = "MjOLQjULNySrGTTsErxXE8RiE2iSxz97"

async def download_image(image_url):
    """Download an image from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(image_url)
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/png")

async def process_image_with_mistral(image_path_or_url):
    """
    Process an image with Mistral OCR API.
    
    Args:
        image_path_or_url: Path to local image file or URL to image
    """
    try:
        # Initialize Mistral client
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Check if input is a URL or local file path
        if image_path_or_url.startswith(("http://", "https://", "file://")):
            # Download image from URL
            logging.info(f"Downloading image from URL: {image_path_or_url}")
            image_data, content_type = await download_image(image_path_or_url)
            
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{content_type};base64,{base64_image}"
        else:
            # Read local file
            logging.info(f"Reading image from local file: {image_path_or_url}")
            with open(image_path_or_url, "rb") as f:
                image_data = f.read()
            
            # Determine content type based on file extension
            if image_path_or_url.lower().endswith(".png"):
                content_type = "image/png"
            elif image_path_or_url.lower().endswith((".jpg", ".jpeg")):
                content_type = "image/jpeg"
            elif image_path_or_url.lower().endswith(".gif"):
                content_type = "image/gif"
            else:
                content_type = "image/png"  # Default
            
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{content_type};base64,{base64_image}"
        
        # Process with Mistral OCR API
        logging.info("Sending request to Mistral OCR API")
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": image_url
            }
        )
        
        # Extract raw text
        raw_text = ocr_response.text
        logging.info(f"Raw OCR text:\n{raw_text}")
        
        # Process the raw text to extract the conversation
        extraction_prompt = """
Extract the text from this dating app screenshot and organize it into a conversation format.
1. Identify messages from Blake (right side, blue) and the other user (left side)
2. Arrange messages in chronological order
3. Format as "User: [message]" and "Blake: [message]"
4. ONLY extract the conversation, do not add any additional text or response

Here is the raw OCR text from the image:

"""
        
        # Use Mistral for text processing as well
        chat_response = client.chat(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": extraction_prompt + raw_text}
            ]
        )
        
        extracted_conversation = chat_response.messages[0].content
        logging.info(f"Extracted conversation:\n{extracted_conversation}")
        
        return extracted_conversation
        
    except Exception as e:
        logging.exception(f"Error processing image with Mistral OCR API: {e}")
        return None

async def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path_or_url>")
        sys.exit(1)
    
    image_path_or_url = sys.argv[1]
    result = await process_image_with_mistral(image_path_or_url)
    
    if result:
        print("\n--- Extracted Conversation ---\n")
        print(result)
        print("\n-----------------------------\n")
    else:
        print("Failed to extract conversation from image.")

if __name__ == "__main__":
    asyncio.run(main())