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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Mistral API key
MISTRAL_API_KEY = "MjOLQjULNySrGTTsErxXE8RiE2iSxz97"

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        logging.error(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

def process_image_with_mistral_ocr(image_path):
    """
    Process an image with Mistral OCR API.
    
    Args:
        image_path: Path to local image file
    """
    try:
        # Initialize Mistral client
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Read and encode image
        logging.info(f"Reading image from local file: {image_path}")
        base64_image = encode_image(image_path)
        
        if not base64_image:
            return None
        
        # Determine content type based on file extension
        if image_path.lower().endswith(".png"):
            content_type = "image/png"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            content_type = "image/jpeg"
        elif image_path.lower().endswith(".gif"):
            content_type = "image/gif"
        else:
            content_type = "image/png"  # Default
        
        # Process with Mistral OCR API
        logging.info("Sending request to Mistral OCR API")
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": f"data:{content_type};base64,{base64_image}"
            }
        )
        
        # Extract raw text
        raw_text = ocr_response.text
        logging.info(f"Raw OCR text:\n{raw_text}")
        
        # Process the raw text to extract the conversation
        # Use Mistral for text processing
        extraction_prompt = """
Extract the text from this dating app screenshot and organize it into a conversation format.
1. Identify messages from Blake (right side, blue) and the other user (left side)
2. Arrange messages in chronological order
3. Format as "User: [message]" and "Blake: [message]"
4. ONLY extract the conversation, do not add any additional text or response

Here is the raw OCR text from the image:

"""
        
        # Use Mistral for text processing
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": extraction_prompt + raw_text}
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
        response = client.chat.complete(
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
        logging.exception(f"Error processing image with Mistral OCR API: {e}")
        return None, None

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    extracted_conversation, blake_response = process_image_with_mistral_ocr(image_path)
    
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