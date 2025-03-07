import asyncio
import base64
import logging
import os
import sys
from typing import Optional

import discord
import httpx
import yaml
from anthropic import AsyncAnthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

# Constants
COMMAND_PATTERN = "reply?"

class TinderBot:
    def __init__(self, config_path="config.yaml"):
        # Load config
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
        
        # Set up Discord client
        intents = discord.Intents.default()
        intents.message_content = True
        self.discord_client = discord.Client(intents=intents)
        
        # Set up Anthropic client
        self.anthropic_client = AsyncAnthropic(api_key=self.config["providers"]["anthropic"]["api_key"])
        
        # Set up HTTP client for downloading images
        self.http_client = httpx.AsyncClient()
        
        # Set up event handlers
        self.discord_client.event(self.on_ready)
        self.discord_client.event(self.on_message)
        
        # Processing lock
        self.processing = False
    
    async def on_ready(self):
        logging.info(f"Logged in as {self.discord_client.user}")
    
    async def on_message(self, message):
        # Skip messages from bots
        if message.author.bot:
            return
        
        # Check if the message mentions the bot and contains the command pattern
        is_command = (
            self.discord_client.user in message.mentions and 
            COMMAND_PATTERN in message.content.lower()
        )
        
        if not is_command:
            return
        
        # Check if there's an image attachment
        image_attachments = [
            att for att in message.attachments 
            if att.content_type and "image" in att.content_type
        ]
        
        if not image_attachments:
            await message.reply("Please attach an image to analyze.")
            return
        
        # Check if already processing a message
        if self.processing:
            await message.reply("I'm already processing another request. Please wait.")
            return
        
        # Set processing flag
        self.processing = True
        
        try:
            # Send initial response
            response_message = await message.reply("Processing image... ⏳")
            
            # Download image
            image_attachment = image_attachments[0]
            image_response = await self.http_client.get(image_attachment.url)
            image_data = image_response.content
            content_type = image_attachment.content_type
            
            logging.info(f"Downloaded image: {len(image_data)} bytes")
            
            # Step 1: Extract conversation from image
            extraction_prompt = """
Extract the text from this dating app screenshot and organize it into a conversation format.
1. Identify messages from Blake (right side, blue) and the other user (left side)
2. Arrange messages in chronological order
3. Format as "User: [message]" and "Blake: [message]"
4. ONLY extract the conversation, do not add any additional text or response
"""
            
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode("utf-8")
            
            # First API call to extract conversation
            extraction_response = await self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system="You are a helpful assistant that extracts text from images accurately.",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompt},
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
            )
            
            extracted_conversation = extraction_response.content[0].text
            logging.info(f"Extracted conversation: {extracted_conversation}")
            
            # Step 2: Generate Blake's response
            response_prompt = f"""
Here is the extracted conversation from a dating app:

{extracted_conversation}

Respond as Blake to continue this conversation. Remember to use Blake's messaging style.
"""
            
            # Get system prompt from config
            system_prompt = self.config["system_prompt"]
            
            # Second API call to generate Blake's response
            response = await self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": response_prompt}
                        ]
                    }
                ]
            )
            
            blake_response = response.content[0].text
            logging.info(f"Blake's response: {blake_response}")
            
            # Update the response message with Blake's response
            await response_message.edit(content=blake_response)
            
        except Exception as e:
            logging.exception(f"Error processing image: {e}")
            await message.reply(f"Error processing image: {str(e)}")
        finally:
            # Reset processing flag
            self.processing = False
    
    async def start(self):
        await self.discord_client.start(self.config["bot_token"])

async def main():
    bot = TinderBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())