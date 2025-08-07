## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .runnable_node import RunnableNode
from .utils.pydantic import BaseModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.base import RunnableLambda
from langchain_core.runnables.base import RunnableLike
from pydantic import model_serializer
from typing import Callable, Optional, List, Literal
import base64
import httpx
import mimetypes
import os


class RunnableAppend(BaseModel, RunnableLambda):
    # We need to keep the args for serailization and deserialization
    message: BaseMessage
    func: Optional[Callable] = None
    name: Literal["runnable_append"] = "runnable_append"

    def __init__(self, message: BaseMessage, **kwargs):
        # RunnableLambda doesn't pass args to Runnable
        BaseModel.__init__(self, message=message, **kwargs)

        RunnableLambda.__init__(self, lambda x, s=self, **kwargs: s._execute(x))

    def _iter(self, *args, **kwargs):
        """Pydantic serialization method"""
        # No func
        kwargs["exclude"] = (kwargs.get("exclude", None) or set()) | {"func"}

        # Call super
        yield from super()._iter(*args, **kwargs)

    @model_serializer
    def serialize_model(self) -> dict:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except func
        result = {}
        for field_name, field_value in self:
            if field_name != "func":
                result[field_name] = field_value

        return result

    def _append(self, input, message):
        if isinstance(input, ChatPromptValue):
            input.messages.append(message)
            return input
        elif isinstance(input, list):
            return input + [message]
        elif isinstance(input, dict):
            return [input, message]

        return ChatPromptValue(messages=[message])

    def _execute(self, input):
        return self._append(input, self.message)


class RunnableSystemAppend(RunnableAppend):
    # We need to keep the args for serailization and deserialization
    system_message: str
    name: Literal["system_append"] = "system_append"

    def __init__(self, system_message: str, **kwargs):
        super().__init__(message=SystemMessage(content=system_message), system_message=system_message, **kwargs)


class RunnableHumanNode(RunnableNode):
    # We need to keep the args for serailization and deserialization
    human_message: str

    def __init__(self, human_message: str, **kwargs):
        super().__init__(human_message=human_message, **kwargs)
        self.outputs = HumanMessage(content=human_message)


class RunnableAINode(RunnableNode):
    # We need to keep the args for serailization and deserialization
    ai_message: str

    def __init__(self, ai_message: str, **kwargs):
        super().__init__(ai_message=ai_message, **kwargs)
        self.outputs = AIMessage(content=ai_message)


class _ImageHelper:
    @classmethod
    def get_content(cls, text: str, images: List[str], **kwargs):
        content = []
        content.append({"type": "text", "text": text})

        for image in images:
            image_data = cls.get_image_data(image)

            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"{image_data}"},
                }
            )

        return content

    @classmethod
    def get_image_data(cls, image: str):
        if image.startswith("http"):
            return cls.get_url_data(image)
        elif image.startswith("data:"):
            return image
        elif os.path.exists(image):
            return cls.get_local_data(image)
        else:
            return ""

    @classmethod
    def get_url_data(cls, image_path):
        response = httpx.get(image_path)
        response.raise_for_status()  # Ensure the request was successful

        # Determine MIME type from URL
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            raise ValueError("Unable to determine the MIME type of the file")

        image_data = base64.b64encode(response.content).decode("utf-8")

        return f"data:{mime_type};base64,{image_data}"

    @classmethod
    def get_local_data(cls, image_path):
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            raise ValueError("Unable to determine the MIME type of the file")

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{image_data}"


class RunnableHumanImageNode(RunnableHumanNode):
    """
    Creates a human message with images.

    Args:
        human_message (str): The text message.
        images (List[str]): A list of images, each image could be either a local path, remote URL, or image data embedded in request format.
    """

    # We need to keep the args for serailization and deserialization
    images: List[str]

    def __init__(self, human_message: str, images: List[str], **kwargs):
        super().__init__(human_message=human_message, images=images, **kwargs)
        self.images = images
        self.outputs = HumanMessage(content=_ImageHelper.get_content(human_message, images))


class RunnableAIImageNode(RunnableAINode):
    """
    Creates an AI message with images.

    Args:
        ai_message (str): The text message.
        images (List[str]): A list of images, each image could be either a local path, remote URL, or image data embedded in request format.
    """

    # We need to keep the args for serailization and deserialization
    images: List[str]

    def __init__(self, ai_message: str, images: List[str], **kwargs):
        super().__init__(ai_message=ai_message, images=images, **kwargs)
        self.images = images
        self.outputs = AIMessage(content=_ImageHelper.get_content(ai_message, images))


class RunnableToolNode(RunnableNode):
    """
    This node represents the result of a tool invocation.
    """

    tool_message: str
    tool_call_id: str

    def __init__(self, tool_message: str, tool_call_id: str, status: Literal["success", "error"] = "success", **kwargs):
        super().__init__(tool_message=tool_message, tool_call_id=tool_call_id, **kwargs)
        self.outputs = ToolMessage(content=tool_message, tool_call_id=tool_call_id, status=status)
