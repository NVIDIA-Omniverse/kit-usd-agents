## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.messages import convert_to_openai_messages
from nat.data_models.api_server import (
    AIQChatRequest,
    Message,
    TextContent,
    ImageContent,
    ImageUrl,
    ChatContentType
)
from pydantic import HttpUrl

# Custom ImageData class extending ImageUrl to handle base64 data
class ImageData(ImageUrl):
    data: str  # Base64 image data

def convert_langchain_to_nat_messages(lc_messages: List[BaseMessage]) -> AIQChatRequest:
    """Convert LangChain messages to NAT chat request format.

    This function takes a list of LangChain BaseMessage objects and converts them
    to the NAT message format required for NAT chat requests. It properly handles
    complex content including text and images.

    Args:
        lc_messages: List of LangChain BaseMessage objects

    Returns:
        AIQChatRequest: A properly formatted NAT chat request object
    """
    # First convert LangChain messages to OpenAI format
    openai_messages = convert_to_openai_messages(lc_messages)
    nat_messages = []

    # Convert each OpenAI message to AIQ message format
    for openai_msg in openai_messages:
        content = openai_msg["content"]

        # Handle complex content (list of text/image objects)
        if isinstance(content, list):
            aiq_content = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        # Create TextContent object
                        text_content = TextContent(
                            type=ChatContentType.TEXT,
                            text=item.get("text", "")
                        )
                        aiq_content.append(text_content)
                    elif item.get("type") == "image" or item.get("type") == "image_url":
                        # Create ImageContent object
                        image_url_data = item.get("image_url", {})
                        if isinstance(image_url_data, dict):
                            url_str = image_url_data.get("url", "https://placeholder.invalid")
                        else:
                            url_str = str(image_url_data)

                        # AIQ only supports HTTP/HTTPS URLs for ImageContent
                        # For base64 and file:// URLs, preserve the path information
                        if url_str.startswith("data:image"):
                            # Base64 image - use ImageContent with our custom ImageData
                            # Note: placeholder URL required by HttpUrl validation, actual data is in 'data' field
                            image_content = ImageContent(
                                type=ChatContentType.IMAGE_URL,
                                image_url=ImageData(url=HttpUrl("https://placeholder.invalid"), data=url_str)
                            )
                            aiq_content.append(image_content)
                        elif len(url_str) < 2083 and (url_str.startswith("https://") or url_str.startswith("http://")):
                            # Valid HTTP/HTTPS URL - use standard ImageContent
                            try:
                                image_content = ImageContent(
                                    type=ChatContentType.IMAGE_URL,
                                    image_url=ImageUrl(url=HttpUrl(url_str))
                                )
                                aiq_content.append(image_content)
                            except:
                                # If URL is invalid, convert to text
                                text_content = TextContent(
                                    type=ChatContentType.TEXT,
                                    text=f"[Image: {url_str}]"
                                )
                                aiq_content.append(text_content)
                        else:
                            # File paths or other formats - convert to text
                            text_content = TextContent(
                                type=ChatContentType.TEXT,
                                text=f"[Image: {url_str}]" if url_str else "[Image]"
                            )
                            aiq_content.append(text_content)
                else:
                    # Simple string item - convert to TextContent
                    text_content = TextContent(
                        type=ChatContentType.TEXT,
                        text=str(item)
                    )
                    aiq_content.append(text_content)

            # Create Message with list of UserContent
            nat_messages.append(Message(content=aiq_content, role=openai_msg["role"]))
        else:
            # Simple string content
            nat_messages.append(Message(content=str(content), role=openai_msg["role"]))

    # Return a properly formatted NAT chat request
    return AIQChatRequest(messages=nat_messages)