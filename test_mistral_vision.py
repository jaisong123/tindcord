#!/usr/bin/env python3
"""
Test script for using Mistral Vision API to extract text from dating app screenshots.
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

def process_image_with_mistral(image_path_or_url):
    """
    Process an image with Mistral Vision API.
    
    Args:
        image_path_or_url: Path to local image file or URL to image
    """
    try:
        # Initialize Mistral client
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Check if input is a URL or local file path
        if image_path_or_url.startswith(("http://", "https://", "file://")):
            # For URLs, we'll use the URL directly
            image_url = image_path_or_url
            logging.info(f"Using image URL: {image_url}")
        else:
            # For local files, we need to read and encode
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
        
        # Process with Mistral Vision API
        logging.info("Sending request to Mistral Vision API")
        
        # First, extract text from the image
        extraction_prompt = """
Extract the text from this dating app screenshot and organize it into a conversation format.
1. Identify messages from Blake (right side, blue) and the other user (left side)
2. Arrange messages in chronological order
3. Format as "User: [message]" and "Blake: [message]"
4. ONLY extract the conversation, do not add any additional text or response
"""
        
        # Use Mistral for text extraction
        chat_response = client.chat(
            model="mistral-large-latest",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": extraction_prompt},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ]
        )
        
        extracted_conversation = chat_response.choices[0].message.content
        logging.info(f"Extracted conversation:\n{extracted_conversation}")
        
        # Now generate Blake's response based on the extracted conversation
        response_prompt = f"""
Here is the extracted conversation from a dating app:

{extracted_conversation}

IMPORTANT: Respond ONLY as Blake to continue this conversation. 
DO NOT repeat the conversation above.
DO NOT include "User:" or "Blake:" prefixes in your response.
Simply write what Blake would say next in his characteristic style.
"""
        
        # Use Mistral to generate Blake's response
        response = client.chat(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": """You are Blake, a confident, flirtatious communicator on a dating app. Your responses are characterized by:
                
1. Extreme Brevity (MOST CRITICAL)
   - Messages rarely exceed 8 words
   - Multiple short texts instead of one long message
   - Every word has purpose, no unnecessary elaboration

2. Confident Frame Control
   - You are the selector, not the selected
   - Subtly challenge others to meet your standards
   - Never appear overly eager or invested

3. Direct Communication Style
   - Make statements, not questions (unless leading somewhere)
   - Express interest without neediness
   - Comfortable with silence

4. Flirtatious Approach
   - Light teasing and playful challenges
   - Suggestive undertones without being explicit
   - Balance flirtation with genuine connection

5. Goal-Oriented Messaging
   - Move toward meeting up
   - Qualify matches with "intentions" check
   - Direct about phone numbers: "Shoot me your number"

Your Messaging Patterns:
- Double-texts: Brief response (3-8 words) followed by conversation advancement (3-8 words)
- Use signature phrases: "I feel that," "Shoot me your number," "Let's see what happens," "You're cute, what are you looking for on here?"
- Abbreviate words ("def," "tbh," "w the flow")
- Minimal punctuation, sparse emoji (mainly "😉")"""},
                {"role": "user", "content": response_prompt}
            ]
        )
        
        blake_response = response.choices[0].message.content
        logging.info(f"Blake's response:\n{blake_response}")
        
        return extracted_conversation, blake_response
        
    except Exception as e:
        logging.exception(f"Error processing image with Mistral Vision API: {e}")
        return None, None

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path_or_url>")
        sys.exit(1)
    
    image_path_or_url = sys.argv[1]
    extracted_conversation, blake_response = process_image_with_mistral(image_path_or_url)
    
    if extracted_conversation and blake_response:
        print("\n--- Extracted Conversation ---\n")
        print(extracted_conversation)
        print("\n--- Blake's Response ---\n")
        print(blake_response)
        print("\n-----------------------------\n")
    else:
        print("Failed to process image.")

if __name__ == "__main__":
    main()