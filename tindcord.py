import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
import re
from typing import Literal, Optional, Union, Dict, Any

import discord
import httpx
from openai import AsyncOpenAI
import yaml

from claude_integration import ClaudeClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "claude-3", "gemini", "pixtral", "llava", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

ALLOWED_FILE_TYPES = ("image", "text")

# Claude API specific constants
CLAUDE_COMMAND_PATTERN = r"@bot\s+reply\?"
CLAUDE_PROVIDER = "anthropic"

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100


def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jaisong123/tindcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0
processing_locks = {}  # Dictionary to track which messages are being processed


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    # Skip messages from bots
    if new_msg.author.bot:
        return
        
    # Check for "@bot reply?" command with image
    is_claude_image_request = False
    if re.search(CLAUDE_COMMAND_PATTERN, new_msg.content, re.IGNORECASE) or (discord_client.user in new_msg.mentions and "reply?" in new_msg.content.lower()):
        # Check if there's an image attachment
        image_attachments = [att for att in new_msg.attachments if att.content_type and "image" in att.content_type]
        if image_attachments:
            is_claude_image_request = True
            logging.info(f"Detected Claude image request: {new_msg.content}")
    
    # Skip messages that don't mention the bot and aren't Claude image requests
    if not is_dm and not is_claude_image_request and discord_client.user not in new_msg.mentions:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(id for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None)) if id)

    cfg = get_config()

    allow_dms = cfg["allow_dms"]
    permissions = cfg["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return
        
    # Handle Claude image request
    if is_claude_image_request:
        await process_claude_image_request(new_msg, image_attachments[0])
        return

    provider, model = cfg["model"].split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()

                good_attachments = {type: [att for att in curr_msg.attachments if att.content_type and type in att.content_type] for type in ALLOWED_FILE_TYPES}

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + [embed.description for embed in curr_msg.embeds if embed.description]
                    + [(await httpx_client.get(att.url)).text for att in good_attachments["text"]]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode((await httpx_client.get(att.url)).content).decode('utf-8')}"))
                    for att in good_attachments["image"]
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(len(att_list) for att_list in good_attachments.values())

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
        messages.append(dict(role="system", content=full_system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=cfg["extra_api_parameters"])
    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                if finish_reason != None:
                    break

                finish_reason = curr_chunk.choices[0].finish_reason

                prev_content = curr_content or ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        if edit_task != None:
                            await edit_task

                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)

                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                        last_task_time = dt.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def process_claude_image_request(new_msg, image_attachment):
    """
    Process an image with Claude API when the "@bot reply?" command is detected.
    
    Args:
        new_msg: The Discord message containing the command and image
        image_attachment: The image attachment to process
    """
    global processing_locks
    
    # Check if this message is already being processed
    if new_msg.id in processing_locks:
        logging.info(f"Message {new_msg.id} is already being processed, skipping")
        return
    
    # Mark this message as being processed
    processing_locks[new_msg.id] = True
    
    try:
        logging.info(f"Processing image with Claude API (user ID: {new_msg.author.id}, attachment: {image_attachment.filename})")
        
        # Get configuration
        cfg = get_config()
        
        # Get Claude API key
        api_key = cfg["providers"][CLAUDE_PROVIDER].get("api_key")
        if not api_key:
            await new_msg.reply("Error: Claude API key not configured. Please add it to your config.yaml file.")
            return
        
        # Initialize Claude client
        claude_client = ClaudeClient(api_key)
        
        # Set model if specified in config
        if "/" in cfg["model"] and cfg["model"].split("/", 1)[0] == CLAUDE_PROVIDER:
            claude_client.set_model(cfg["model"].split("/", 1)[1])
        
        # Get system prompt
        system_prompt = cfg["system_prompt"] or "You are a helpful assistant that analyzes images."
        
        # Extract user prompt (remove the command pattern and bot mention)
        user_content = new_msg.content
        user_prompt = re.sub(CLAUDE_COMMAND_PATTERN, "", user_content, flags=re.IGNORECASE).strip()
        
        # Also remove bot mention if present
        if discord_client.user:
            user_prompt = user_prompt.replace(discord_client.user.mention, "").strip()
        
        # Default prompt if empty or just "reply?"
        if not user_prompt or user_prompt.lower() == "reply?":
            user_prompt = """
Extract the text from the image and organize it into a conversation format. Follow these steps to ensure accuracy:

Identify Users: Clearly distinguish between messages from two different users based on their positions in the image. Label them as 'User 1 (Right side)' and 'User 2 (Left side)' based on their placement.

Chronological Order: Arrange the messages in chronological order according to the timestamps provided in the image. If timestamps are not visible, use the logical sequence of messages to determine the order.

Attribution: Ensure each message is correctly attributed to the corresponding user. Maintain the integrity of the conversation by preserving the flow and context.

Natural Flow: Present the conversation in a natural, readable format without any numbering or additional markers. The output should be a clean, chronological conversation.

Accuracy: Double-check the arrangement to ensure there are no misplaced messages or attribution errors.

Example Format:

Blake (Right side): [Message]
User 1 (Left side): [Message]
Follow these guidelines to produce an accurate and coherent conversation transcript from the image.
"""
        
        logging.info(f"User prompt after cleaning: '{user_prompt}'")
        
        # Get max tokens
        max_tokens = cfg["extra_api_parameters"].get("max_tokens", 4096)
        
        # Download image
        image_response = await httpx_client.get(image_attachment.url)
        image_data = image_response.content
        content_type = image_attachment.content_type
        
        logging.info(f"Downloaded image: {len(image_data)} bytes, content type: {content_type}")
        
        # Process with Claude API
        embed = discord.Embed()
        embed.description = "Thinking..." + STREAMING_INDICATOR  # Add placeholder description
        embed.color = EMBED_COLOR_INCOMPLETE
        
        # Send initial response
        response_msg = await new_msg.reply(embed=embed, silent=True)
        
        # Step 1: Extract conversation from image
        extraction_prompt = """
Extract the text from this dating app screenshot and organize it into a conversation format.
1. Identify messages from Blake (right side, blue) and the other user (left side)
2. Arrange messages in chronological order
3. Format as "User: [message]" and "Blake: [message]"
4. ONLY extract the conversation, do not add any additional text or response
"""
        logging.info(f"Sending extraction request to Claude API")
        logging.info(f"System prompt length: {len(system_prompt)} characters")
        
        # First API call to extract conversation
        extracted_conversation, extraction_success = await claude_client.process_image(
            image_data=image_data,
            content_type=content_type,
            prompt=extraction_prompt,
            system_prompt="You are a helpful assistant that extracts text from images accurately.",
            max_tokens=max_tokens
        )
        
        logging.info(f"Extraction response: success={extraction_success}, length={len(extracted_conversation)}")
        
        if not extraction_success or not extracted_conversation:
            logging.error(f"Failed to extract conversation: {extracted_conversation}")
            embed.description = "Failed to extract conversation from image. Please try again."
            embed.color = discord.Color.red()
            await response_msg.edit(embed=embed)
            return
        
        # Step 2: Generate Blake's response based on extracted conversation
        response_prompt = f"""
Here is the extracted conversation from a dating app:

{extracted_conversation}

IMPORTANT: Respond ONLY as Blake to continue this conversation.
DO NOT repeat the conversation above.
DO NOT include "User:" or "Blake:" prefixes in your response.
Simply write what Blake would say next in his characteristic style.
"""
        logging.info(f"Sending response request to Claude API")
        
        # Second API call to generate Blake's response
        blake_response, response_success = await claude_client.process_image(
            image_data=image_data,
            content_type=content_type,
            prompt=response_prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens
        )
        
        logging.info(f"Blake response: success={response_success}, length={len(blake_response)}")
        
        if response_success and blake_response:
            # Clean up Blake's response to remove any extracted conversation parts
            # Look for patterns like "User: ... Blake: ..." and remove them
            cleaned_response = blake_response
            
            # Remove any lines that start with "User:" or contain conversation formatting
            lines = cleaned_response.split('\n')
            cleaned_lines = []
            
            # Flag to indicate we've found Blake's actual response (after any extracted conversation)
            found_blake_response = False
            
            for line in lines:
                # Skip empty lines at the beginning
                if not line.strip() and not found_blake_response:
                    continue
                    
                # Skip lines that look like part of the extracted conversation
                if line.strip().startswith("User:") or line.strip().startswith("Blake:"):
                    # But if it's a new Blake response after the conversation, keep it
                    if found_blake_response:
                        cleaned_lines.append(line)
                else:
                    # This is likely Blake's actual response
                    found_blake_response = True
                    cleaned_lines.append(line)
            
            # If we couldn't clean it properly, just use the original response
            if not cleaned_lines:
                cleaned_response = blake_response
            else:
                cleaned_response = '\n'.join(cleaned_lines)
            
            full_response = cleaned_response
            embed.description = full_response
            embed.color = EMBED_COLOR_COMPLETE
            await response_msg.edit(embed=embed)
            
            # Log both the extracted conversation and Blake's response for debugging
            logging.info(f"Extracted conversation: {extracted_conversation}")
            logging.info(f"Blake's response: {blake_response}")
            logging.info(f"Cleaned response: {cleaned_response}")
            
            # Store in message nodes
            msg_nodes[response_msg.id] = MsgNode(
                text=full_response,
                role="assistant",
                parent_msg=new_msg
            )
        else:
            logging.error(f"Failed to generate Blake's response: {blake_response}")
            embed.description = "Failed to generate response. Please try again."
            embed.color = discord.Color.red()
            await response_msg.edit(embed=embed)
    except Exception as e:
        logging.exception(f"Error processing image with Claude API: {e}")
        try:
            embed.description = f"Error: {str(e)}"
            embed.color = discord.Color.red()
            await response_msg.edit(embed=embed)
        except:
            await new_msg.reply(f"Error: {str(e)}", silent=True)
    finally:
        # Always release the lock when done
        if new_msg.id in processing_locks:
            del processing_locks[new_msg.id]
            logging.info(f"Released processing lock for message {new_msg.id}")


async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())
