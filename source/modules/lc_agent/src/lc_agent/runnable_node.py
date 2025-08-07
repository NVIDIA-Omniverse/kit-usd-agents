## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .chat_model_registry import get_chat_model_registry
from .node_factory import get_node_factory
from .utils.culling import _cull_messages
from .utils.profiling_utils import Profiler
from .uuid_utils import UUIDMixin
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages import messages_to_dict as langchain_messages_to_dict
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables import ensure_config
from langchain_core.runnables.base import RunnableBinding
from langchain_core.runnables.utils import Input, Output
from pydantic import model_serializer, model_validator
from typing import Any, Dict, Iterator, List, Optional, AsyncIterator, ForwardRef, Type, Union
from typing_extensions import Self
import os
import time


def _messages_to_dict(messages: List[BaseMessage]):
    """
    Convert a sequence of messages to a list of dictionaries and clean the resulting dictionaries.

    This function converts messages to a dictionary format using the `langchain_messages_to_dict` function
    and then recursively cleans the resulting dictionary by removing all None values, empty strings, empty lists,
    and nested dictionaries that become empty after cleaning.

    Args:
        messages (List[BaseMessage]): Sequence of messages to convert.

    Returns:
        list: A list of cleaned dictionaries representing the messages.

    Example:
        input_messages = [Message1, Message2, ...]
        cleaned_dict_list = _messages_to_dict(input_messages)
    """

    def _clean_dict(d):
        """
        Recursively clean a dictionary by removing all None values, empty strings, empty lists,
        and nested dictionaries that become empty after cleaning.

        Args:
            d (dict): The dictionary to clean.

        Returns:
            dict: The cleaned dictionary.
        """
        if not isinstance(d, dict):
            return d

        cleaned_dict = {}
        for key, value in d.items():
            # Recursively clean nested dictionaries
            if isinstance(value, dict):
                nested_cleaned = _clean_dict(value)
                if nested_cleaned:  # Only add non-empty nested dictionaries
                    cleaned_dict[key] = nested_cleaned
            # Remove None, empty strings, empty lists, and empty sets
            elif isinstance(value, bool) or value:
                cleaned_dict[key] = value

        return cleaned_dict

    # Convert messages to dictionaries
    result = langchain_messages_to_dict(messages)

    # Clean the resulting dictionary
    cleaned_result = [_clean_dict(r) for r in result]

    return cleaned_result


def _is_message(message):
    if isinstance(message, BaseMessage):
        return True

    if isinstance(message, ChatPromptTemplate):
        return True

    if isinstance(message, ChatPromptValue):
        return True

    if isinstance(message, dict) and "role" in message and "content" in message:
        return True

    if isinstance(message, str):
        return True

    if isinstance(message, tuple) and all(isinstance(i, str) for i in message) and len(message) == 2:
        return True


class ModelNotFoundError(Exception):
    pass


class AINodeMessage(AIMessage):
    node: Optional[Any] = None

    def copy_from(self, other: AIMessage):
        self.content = other.content
        self.name = other.name
        self.response_metadata = other.response_metadata
        self.additional_kwargs = other.additional_kwargs
        self.tool_calls = other.tool_calls
        self.invalid_tool_calls = other.invalid_tool_calls
        self.usage_metadata = other.usage_metadata
        if isinstance(other, AINodeMessage):
            self.node = other.node


class AINodeMessageChunk(AIMessageChunk, AINodeMessage):
    def copy_from(self, other: AIMessage):
        super().copy_from(other)

        if isinstance(other, AIMessageChunk):
            self.tool_call_chunks = other.tool_call_chunks


class CountTokensCallbackHandler(BaseCallbackHandler):
    """Callback to count tokens"""

    def __init__(self):
        super().__init__()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._model = None
        self._token_encoding = None
        self._first_token_time = None
        self._tokens_per_second_with_ttf = None
        self._tokens_per_second_wo_ttf = None
        self._start_time = time.time()
        self._first_token_received_time = None
        self._elapsed_time = None
        self._elapsed_time_wo_ttf = None

    def _count_prompt_tokens(self, prompts: List[str]):
        if not self._token_encoding:
            return 0

        if isinstance(prompts, str):
            return len(self._token_encoding.encode(prompts))

        # Count the number of roles in the prompt.
        # At this moment we receive all the messages combined to one single prompt.
        roles = ["Human: ", "AI: ", "System: "]
        number_of_roles = 0
        for prompt in prompts:
            for role in roles:
                number_of_roles += prompt.count(role)

        result = number_of_roles * 2

        for prompt in prompts:
            result += len(self._token_encoding.encode(prompt))

        return result

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        invoc_params = kwargs.get("invocation_params", {})
        self._model = invoc_params.get("model_name") or invoc_params.get("model")

        if not self._model:
            # Model is unknown
            # TODO: Warning?
            self._model = "cl100k_base"

        try:
            import tiktoken

            self._token_encoding = tiktoken.encoding_for_model(self._model)
        except KeyError:
            self._model = "cl100k_base"
            self._token_encoding = tiktoken.get_encoding(self._model)

        self._prompt_tokens = self._count_prompt_tokens(prompts)
        # Start the timer
        self._start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token:
            if self._first_token_time is None:
                # Time to first token
                self._first_token_time = time.time() - self._start_time
                self._first_token_received_time = time.time()
            self._completion_tokens += len(self._token_encoding.encode(token))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if not self._completion_tokens:
            if not response.generations or not response.generations[-1]:
                # No text generated
                return

            model_output: Optional[str] = response.generations[-1][-1].text

            if model_output and self._token_encoding:
                self._completion_tokens = len(self._token_encoding.encode(model_output))

        # Stop the timer
        end_time = time.time()
        # Calculate elapsed time in seconds
        self._elapsed_time = end_time - self._start_time
        # Calculate elapsed time excluding time to first token
        self._elapsed_time_wo_ttf = end_time - (self._first_token_received_time or self._start_time)

        # Save tokens per second (with and without time to first token)
        self._tokens_per_second_with_ttf = (
            self._completion_tokens / self._elapsed_time if self._elapsed_time > 0 else None
        )
        self._tokens_per_second_wo_ttf = (
            self._completion_tokens / self._elapsed_time_wo_ttf if self._elapsed_time_wo_ttf > 0 else None
        )

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def tokens_per_second_with_ttf(self) -> float:
        return self._tokens_per_second_with_ttf

    @property
    def tokens_per_second_wo_ttf(self) -> float:
        return self._tokens_per_second_wo_ttf

    @property
    def time_to_first_token(self) -> float:
        return self._first_token_time

    @property
    def elapsed_time(self) -> float:
        return self._elapsed_time

    @property
    def elapsed_time_wo_ttf(self) -> float:
        return self._elapsed_time_wo_ttf


RunnableNode = ForwardRef("RunnableNode")
OutputType = Union[HumanMessage, AIMessage, SystemMessage, ToolMessage, ChatPromptTemplate]


class RunnableNode(RunnableSerializable[Input, Output], UUIDMixin):
    # We have inputs, parents and outputs
    # Closest alternative is RunnableSequence. They have first, middle, last
    parents: List[RunnableNode] = []
    inputs: List = []
    outputs: Optional[Union[List[OutputType], OutputType]] = None

    metadata: Dict[str, Any] = {}

    verbose: bool = False

    invoked: bool = False
    chat_model_name: Optional[str] = None

    # Add class variable for debug file
    _debug_payload_file = os.environ.get("LC_AGENT_DEBUG_PAYLOAD")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uuid()

        from .runnable_network import RunnableNetwork

        current_network = RunnableNetwork.get_active_network()
        if current_network:
            current_network.add_node(self)

    def _iter(self, *args, **kwargs):
        """Pydantic 1 serialization method"""
        # No parents
        kwargs["exclude"] = (kwargs.get("exclude", None) or set()) | {"parents"}

        # Call super
        yield from super()._iter(*args, **kwargs)

        # Save the type
        yield "__node_type__", self.__class__.__name__

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except parents
        result = {}
        for field_name, field_value in self:
            if field_name not in ["modifiers", "callbacks", "parents"]:
                result[field_name] = field_value

        # Add type information
        result["__node_type__"] = self.__class__.__name__

        return result

    @classmethod
    def _deserialize_message(cls, message_dict: Dict[str, Any]) -> BaseMessage:
        """
        Deserialize a message dictionary into the appropriate message class.

        Args:
            message_dict: Dictionary representation of a message

        Returns:
            BaseMessage: The deserialized message
        """
        if not isinstance(message_dict, dict):
            return message_dict

        # If it's already a BaseMessage instance, return it
        if isinstance(message_dict, BaseMessage):
            return message_dict

        # Check the type field to determine the message class
        message_type = message_dict.get("type")

        if message_type == "human":
            # Create a HumanMessage
            content = message_dict.get("content", "")
            kwargs = {k: v for k, v in message_dict.items() if k not in ["type", "content"]}
            return HumanMessage(content=content, **kwargs)
        elif message_type == "ai":
            # Create an AIMessage
            content = message_dict.get("content", "")
            kwargs = {k: v for k, v in message_dict.items() if k not in ["type", "content"]}
            return AIMessage(content=content, **kwargs)
        elif message_type == "system":
            # Create a SystemMessage
            content = message_dict.get("content", "")
            kwargs = {k: v for k, v in message_dict.items() if k not in ["type", "content"]}
            return SystemMessage(content=content, **kwargs)
        elif message_type == "tool" and "tool_call_id" in message_dict:
            # Create a ToolMessage only if tool_call_id is present
            content = message_dict.get("content", "")
            kwargs = {k: v for k, v in message_dict.items() if k not in ["type", "content"]}
            return ToolMessage(content=content, **kwargs)

        # If we can't determine the type or it's not a valid message dict,
        # return the original dict
        return message_dict

    @classmethod
    def _deserialize_outputs(cls, outputs):
        """
        Deserialize the outputs field which can be a single message or a list of messages.

        Args:
            outputs: A message dictionary or list of message dictionaries

        Returns:
            The deserialized message(s)
        """
        if outputs is None:
            return None

        if isinstance(outputs, list):
            return [cls._deserialize_message(msg) for msg in outputs]
        else:
            return cls._deserialize_message(outputs)

    @classmethod
    def parse_obj(cls: Type["RunnableNode"], obj: Any) -> "RunnableNode":
        """Pydantic deserialization method"""
        edited_obj = obj.copy()

        # Handle outputs field specially to avoid message deserialization issues
        # if "outputs" in edited_obj:
        #     edited_obj["outputs"] = cls._deserialize_outputs(edited_obj["outputs"])

        # Remove the type from the object. So it will recusively call the
        # correct parse_obj
        # Check for both __node_type__ (new) and __type__ (old) for backward compatibility
        need_type_name = edited_obj.pop("__node_type__", None)
        if need_type_name is None:
            # Try the old field name for backward compatibility
            need_type_name = edited_obj.pop("__type__", None)

        if need_type_name is None:
            # If no type is specified, default to the current class
            return super(RunnableNode, cls).parse_obj(obj)

        # Use a factory or some registry to get the correct type
        # Check by name first and if it's not there, check by type
        name = edited_obj.get("name", None)
        if name:
            node_type = get_node_factory().get_registered_node_type(name)
        else:
            node_type = None
        if node_type is None:
            # Couldn't find by name, try by type
            node_type = get_node_factory().get_registered_node_type(need_type_name)

        if node_type is None:
            # Still not found, create RunableNode and assign all the fields
            node_type = RunnableNode

        return node_type.parse_obj(edited_obj)

    @model_validator(mode="wrap")
    @classmethod
    def deserialize(cls, data: Any, handler) -> Self:
        """Pydantic v2 deserialization method using model_validator with wrap mode"""
        # Handle dictionary input
        if isinstance(data, dict):
            edited_data = data.copy()

            # Remove the type from the object so it will recursively call the
            # correct model_validate
            # Check for both __node_type__ (new) and __type__ (old) for backward compatibility
            need_type_name = edited_data.pop("__node_type__", None)
            if need_type_name is None:
                # Try the old field name for backward compatibility
                need_type_name = edited_data.pop("__type__", None)

            if need_type_name is None:
                # If no type is specified, use the default handler
                return handler(data)

            # Use a factory or some registry to get the correct type
            # Check by name first and if it's not there, check by type
            name = edited_data.get("name", None)
            if name:
                node_type = get_node_factory().get_registered_node_type(name)
            else:
                node_type = None
            if node_type is None:
                # Couldn't find by name, try by type
                node_type = get_node_factory().get_registered_node_type(need_type_name)

            if node_type is None:
                # Still not found, create RunnableNode and assign all the fields
                node_type = RunnableNode

            outputs = None
            # Handle outputs field specially to avoid message deserialization issues
            if "outputs" in edited_data:
                outputs = cls._deserialize_outputs(edited_data.pop("outputs"))

            # Use the appropriate class's model_validate method
            if node_type is cls:
                # If we're already using the right class, use the handler
                result = handler(edited_data)
            else:
                # Otherwise, use the node_type's model_validate method
                result = node_type.model_validate(edited_data)

            if outputs is not None:
                result.outputs = outputs

            return result

        # Handle case where obj is already a RunnableNode instance
        if isinstance(data, RunnableNode):
            return data

        # For other types, use the default handler
        return handler(data)

    def __repr__(self):
        if isinstance(self.outputs, list):
            repr_output = f"({len(self.outputs)} outputs)"
        elif self.outputs and isinstance(self.outputs, BaseMessage):
            repr_output = f"{self.outputs.content}"
            repr_output = repr_output[: min(50, len(repr_output))]
            repr_output = f"'{repr_output}'"
        else:
            repr_output = "(no outputs)"

        repr_name = f"'{self.name}'" if self.name else ""

        type_name = type(self).__name__

        return f"<{type_name} {repr_name} {repr_output}>"

    def __str__(self):
        return self.__repr__()

    def invoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        self._pre_invoke(input, config, **kwargs)

        if self.invoked or self.outputs is not None:
            self.invoked = True
            return self.outputs

        result = None
        count_tokens = CountTokensCallbackHandler()
        config = self._get_config(config, count_tokens)
        chat_model_input = None

        try:
            parents_result = self._process_parents(input, config, **kwargs)
            chat_model_input = self._combine_inputs(input, config, parents_result, **kwargs)

            if self.verbose:
                print(f"[{self.name}] Input:", type(input))
                print(f"[{self.name}] Chat model input:", chat_model_input)

            chat_model_name = self._get_chat_model_name(chat_model_input, input, config)
            chat_model = self._get_chat_model(chat_model_name, chat_model_input, input, config)

            # Tool messages
            chat_model_input = self._sanitize_messages_for_chat_model(chat_model_input, chat_model_name, chat_model)

            max_tokens = get_chat_model_registry().get_max_tokens(chat_model_name)
            tokenizer = get_chat_model_registry().get_tokenizer(chat_model_name)
            if max_tokens is not None and tokenizer is not None:
                chat_model_input = _cull_messages(chat_model_input, max_tokens, tokenizer)

            self.metadata["chat_model_input"] = _messages_to_dict(chat_model_input)

            result = self._invoke_chat_model(chat_model, chat_model_input, input, config, **kwargs)
        except BaseException as e:
            self.metadata["error"] = str(e)
            raise

        self.metadata["token_usage"] = {
            "total_tokens": count_tokens.total_tokens,
            "prompt_tokens": count_tokens.prompt_tokens,
            "completion_tokens": count_tokens.completion_tokens,
            "tokens_per_second": count_tokens.tokens_per_second_with_ttf,
            "tokens_per_second_wo_ttf": count_tokens.tokens_per_second_wo_ttf,
            "time_to_first_token": count_tokens.time_to_first_token,
            "elapsed_time": count_tokens.elapsed_time,
            "elapsed_time_wo_ttf": count_tokens.elapsed_time_wo_ttf,
        }

        if input:
            self.metadata["invoke_input"] = input

        if self.verbose:
            # Print token usage
            print(f"[{self.name}] Token usage:", self.metadata["token_usage"])

        self.outputs = result
        self.invoked = True
        return self.outputs

    async def ainvoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        self._pre_invoke(input, config, **kwargs)

        if self.invoked or self.outputs is not None:
            self.invoked = True
            return self.outputs

        result = None
        count_tokens = CountTokensCallbackHandler()
        config = self._get_config(config, count_tokens)
        chat_model_input = None

        try:
            parents_result = await self._aprocess_parents(input, config, **kwargs)
            chat_model_input = await self._acombine_inputs(input, config, parents_result, **kwargs)

            if self.verbose:
                print(f"[{self.name}] Input:", type(input))
                print(f"[{self.name}] Chat model input:", chat_model_input)

            chat_model_name = self._get_chat_model_name(chat_model_input, input, config)
            chat_model = self._get_chat_model(chat_model_name, chat_model_input, input, config)

            # Tool messages
            chat_model_input = self._sanitize_messages_for_chat_model(chat_model_input, chat_model_name, chat_model)

            max_tokens = get_chat_model_registry().get_max_tokens(chat_model_name)
            tokenizer = get_chat_model_registry().get_tokenizer(chat_model_name)
            if max_tokens is not None and tokenizer is not None:
                chat_model_input = _cull_messages(chat_model_input, max_tokens, tokenizer)

            self.metadata["chat_model_input"] = _messages_to_dict(chat_model_input)

            result = await self._ainvoke_chat_model(chat_model, chat_model_input, input, config, **kwargs)
        except BaseException as e:
            self.metadata["error"] = str(e)
            raise

        self.metadata["token_usage"] = {
            "total_tokens": count_tokens.total_tokens,
            "prompt_tokens": count_tokens.prompt_tokens,
            "completion_tokens": count_tokens.completion_tokens,
            "tokens_per_second": count_tokens.tokens_per_second_with_ttf,
            "tokens_per_second_wo_ttf": count_tokens.tokens_per_second_wo_ttf,
            "time_to_first_token": count_tokens.time_to_first_token,
            "elapsed_time": count_tokens.elapsed_time,
            "elapsed_time_wo_ttf": count_tokens.elapsed_time_wo_ttf,
        }

        if input:
            self.metadata["invoke_input"] = input

        if self.verbose:
            # Print token usage
            print(f"[{self.name}] Token usage:", self.metadata["token_usage"])

        self.outputs = result
        self.invoked = True
        return self.outputs

    async def astream(
        self,
        input: Input = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        self._pre_invoke(input, config, **kwargs)

        if self.invoked or self.outputs is not None:
            self.invoked = True
            return

        count_tokens = CountTokensCallbackHandler()
        config = self._get_config(config, count_tokens)

        try:
            parents_result = await self._aprocess_parents(input, config, **kwargs)
            chat_model_input = await self._acombine_inputs(input, config, parents_result, **kwargs)

            if self.verbose:
                print(f"[{self.name}] Input:", type(input))
                print(f"[{self.name}] Chat model input:", chat_model_input)

            outputs = AINodeMessage(content="")
            merged_chunks = AINodeMessageChunk(content="")
            self.outputs = outputs

            chat_model_name = self._get_chat_model_name(chat_model_input, input, config)
            chat_model = self._get_chat_model(chat_model_name, chat_model_input, input, config)

            # Tool messages
            chat_model_input = self._sanitize_messages_for_chat_model(chat_model_input, chat_model_name, chat_model)

            max_tokens = get_chat_model_registry().get_max_tokens(chat_model_name)
            tokenizer = get_chat_model_registry().get_tokenizer(chat_model_name)
            if max_tokens is not None and tokenizer is not None:
                chat_model_input = _cull_messages(chat_model_input, max_tokens, tokenizer)

            self.metadata["chat_model_input"] = _messages_to_dict(chat_model_input)

            latest_node = None
            async for item in self._astream_chat_model(chat_model, chat_model_input, input, config, **kwargs):
                if isinstance(item.content, list):
                    # Sonnet 3.7 returns a list of dicts with text instead of a string
                    # We need to merge them into a single string otherwise we
                    # have langchain error when merging chunks line this:
                    # merged_chunks += item
                    merged = ""
                    for content_item in item.content:
                        if isinstance(content_item, dict) and "text" in content_item:
                            merged += content_item["text"]
                    item.content = merged

                if isinstance(item, AINodeMessageChunk):
                    # It happens when the item is passed from other node
                    # We don't replace it with the new one
                    if latest_node is not item.node:
                        latest_node = item.node
                        # Reset merged chunk
                        merged_chunks = AINodeMessageChunk(content="")

                    merged_chunks += item
                    outputs.copy_from(merged_chunks)

                    yield item
                    continue

                # else:
                # It happens when the item is passed from chat_model
                # We replace it with the new one that has the node info
                if isinstance(item, BaseMessageChunk):
                    merged_chunks += item
                else:
                    merged_chunks.copy_from(item)
                outputs.copy_from(merged_chunks)

                new_item = AINodeMessageChunk(content=item.content, node=self)
                new_item.copy_from(item)

                yield new_item

            outputs.copy_from(merged_chunks)

        except BaseException as e:
            self.metadata["error"] = str(e)
            raise

        # TODO: Realtime token counting
        self.metadata["token_usage"] = {
            "total_tokens": count_tokens.total_tokens,
            "prompt_tokens": count_tokens.prompt_tokens,
            "completion_tokens": count_tokens.completion_tokens,
            "tokens_per_second": count_tokens.tokens_per_second_with_ttf,
            "tokens_per_second_wo_ttf": count_tokens.tokens_per_second_wo_ttf,
            "time_to_first_token": count_tokens.time_to_first_token,
            "elapsed_time": count_tokens.elapsed_time,
            "elapsed_time_wo_ttf": count_tokens.elapsed_time_wo_ttf,
        }

        if self.verbose:
            # Print token usage
            print(f"[{self.name}] Token usage:", self.metadata["token_usage"])

        self.invoked = True

    def _get_chat_model_name(self, chat_model_input, invoke_input, config):
        if self.chat_model_name:
            return self.chat_model_name

        # Get from network
        from .runnable_network import RunnableNetwork

        current_network = RunnableNetwork.get_active_network()
        if current_network:
            chat_model_name = current_network.chat_model_name
            if chat_model_name:
                return chat_model_name

        # Get anything we can find
        registered_names = get_chat_model_registry().get_registered_names()
        if registered_names:
            return registered_names[0]

    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        # at this point we will get it from registry
        if chat_model_name:
            chat_model = get_chat_model_registry().get_model(chat_model_name)
        else:
            # Fall back to default
            from langchain_openai import ChatOpenAI

            chat_model = ChatOpenAI(model="gpt-3.5-turbo")

        if chat_model is None:
            raise ModelNotFoundError(f"Chat model '{chat_model_name}' not found in the registry")

        return chat_model

    def _save_chat_model_input_to_payload(self, chat_model_input):
        """
        Save chat model input to payload.txt with descriptive metadata.
        Only saves if LC_AGENT_DEBUG_PAYLOAD environment variable is set.

        Args:
            chat_model_input: The input messages to be sent to the chat model
        """
        # Check if debug payload is enabled
        if not self._debug_payload_file:
            return

        import datetime
        import json

        # Create description with node info and timestamp
        description = {
            "node_type": self.__class__.__name__,
            "node_name": self.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "metadata": self.metadata,
        }

        # Format the messages in a more readable way
        formatted_messages = []
        for msg in chat_model_input:
            formatted_messages.append(f"{msg.__class__.__name__}: {msg.content}")

        # Format the content to save
        content = f"\n\n{'='*80}\n"
        content += f"{json.dumps(description, indent=2)}\n\n"
        content += "Chat Model Input:\n"
        content += "\n".join(f"[{i+1}] {msg}" for i, msg in enumerate(formatted_messages))
        content += "\n"

        # Append to the file specified in the environment variable
        with open(self._debug_payload_file, "a", encoding="utf-8") as f:
            f.write(content)

    async def _astream_chat_model(self, chat_model, chat_model_input, invoke_input, config, **kwargs):
        # Save chat model input to payload.txt
        self._save_chat_model_input_to_payload(chat_model_input)

        # Original streaming logic
        async for item in chat_model.astream(chat_model_input, config, **kwargs):
            yield item

    def _invoke_chat_model(self, chat_model, chat_model_input, invoke_input, config, **kwargs):
        # Save chat model input to payload.txt
        self._save_chat_model_input_to_payload(chat_model_input)

        # TODO: Use chat model generator if available
        result = chat_model.invoke(chat_model_input, config, **kwargs)

        return result

    async def _ainvoke_chat_model(self, chat_model, chat_model_input, invoke_input, config, **kwargs):
        # Save chat model input to payload.txt
        self._save_chat_model_input_to_payload(chat_model_input)

        # TODO: Use chat model generator if available
        result = await chat_model.ainvoke(chat_model_input, config, **kwargs)

        return result

    def _iterate_chain(self, iterated) -> Iterator["RunnableNode"]:
        """
        Provides an iterator over the network, yielding each node
        in a defined sequence. This method emphasizes the modular design of nodes
        by allowing for the ordered iteration over parent nodes within the network.

        Yields:
            RunnableNode: The next nodes in the sequence.

        Raises:
            NotImplementedError: If attempting to iterate over non-RunnableNode.
        """
        iterated.add(self)

        # Iterate over the parents of this node.
        for parent in self.parents:
            # If the parent is an LLM-based node, recursively iterate over its parents.
            if isinstance(parent, RunnableNode):
                if parent not in iterated:
                    yield from parent._iterate_chain(iterated)
            else:
                raise NotImplementedError("Iteration of non-RunnableNodes is not implemented")

        if not self.metadata.get("contribute_to_history", True):
            # Don't yield the current node if it doesn't contribute to the history.
            return

        # Yield the current node.
        yield self

    def _add_parent(self, parent: "RunnableNode", parent_index: int = -1):
        """
        Adds one more parent

        It's protected because only RunnableNetwork can call it. To set the parent
        we need to use `network.add_node`

        Args:
        parent: the parent node to add
        parent_index: the order of parents is important
        """
        if parent in self.parents:
            self.parents.remove(parent)

        if parent_index < 0:
            self.parents.append(parent)
        else:
            self.parents.insert(parent_index, parent)

    def _clear_parents(self):
        """
        Removes parents

        It's protected because only RunnableNetwork can call it.
        """
        self.parents.clear()

    def on_before_node_added(self, network: "RunnableNetwork"):
        pass

    def on_node_added(self, network: "RunnableNetwork"):
        """
        Called by the network when the node is added to the network.

        Should be re-implemented.

        Can be used to register new modifier in the network.

        Args:
        network: The network it's added to.
        """
        pass

    def on_before_node_removed(self, network: "RunnableNetwork"):
        pass

    def on_node_removed(self, network: "RunnableNetwork"):
        """
        Called by the network when the node is removed from the network.

        Should be re-implemented.

        Can be used to remove new modifier in the network.

        Args:
        network: The network it's removed from.
        """
        pass

    def __hash__(self):
        return hash(id(self))

    def __rshift__(self, other):
        if isinstance(other, RunnableNode):
            # OK
            pass
        elif isinstance(other, Runnable):
            from .from_runnable_node import FromRunnableNode

            other = FromRunnableNode(other)
        else:
            raise ValueError(f"Invalid child type: {type(other)}")

        other._clear_parents()
        other._add_parent(self)

        return other

    def __rrshift__(self, other):
        if isinstance(other, RunnableNode):
            self._clear_parents()
            self._add_parent(other)
        if isinstance(other, Runnable):
            from .from_runnable_node import FromRunnableNode

            self._clear_parents()
            self._add_parent(FromRunnableNode(other))
        elif isinstance(other, list) and all(isinstance(i, RunnableNode) for i in other):
            self._clear_parents()
            for o in other:
                self._add_parent(o)
        elif other is None:
            self._clear_parents()
        else:
            raise ValueError(f"Invalid parent type: {type(other)}")

        return self

    def __lshift__(self, other):
        if isinstance(other, RunnableNode):
            # OK
            pass
        elif isinstance(other, Runnable):
            from .from_runnable_node import FromRunnableNode

            other = FromRunnableNode(other)
        else:
            raise ValueError(f"Invalid parent type: {type(other)}")

        self._clear_parents()
        self._add_parent(other)

        return other

    def _process_parents(self, input: Dict[str, Any], config: Optional[RunnableConfig], **kwargs: Any) -> list:
        parents_result = []
        iterated = set()
        for p in self._iterate_chain(iterated):
            if p is self:
                continue

            result = p.invoke(input, config, **kwargs)
            if isinstance(result, list):
                parents_result.extend(result)
            else:
                parents_result.append(result)
        return parents_result

    async def _aprocess_parents(self, input: Dict[str, Any], config: Optional[RunnableConfig], **kwargs: Any) -> list:
        with Profiler(
            f"process_parents_{self.__class__.__name__}",
            "process_parents",
            node_id=self.uuid(),
            node_name=self.name or self.__class__.__name__,
            parent_count=len(self.parents),
        ):
            parents_result = []
            iterated = set()
            for p in self._iterate_chain(iterated):
                if p is self:
                    continue

                result = await p.ainvoke(input, config, **kwargs)
                if isinstance(result, list):
                    parents_result.extend(result)
                else:
                    parents_result.append(result)
            return parents_result

    def _reorder_tool_messages(self, messages):
        """
        Reorders messages so that each AINodeMessage with tool calls is grouped with its
        corresponding ToolMessages. The group is placed at the position of the latest
        message in the group. Unmatched AINodeMessages with tool calls and unmatched
        ToolMessages are removed. Messages without tool calls or those that don't
        participate in tool interactions are kept in their original positions.

        Args:
            messages (List[BaseMessage]): The list of messages to reorder.

        Returns:
            List[BaseMessage]: The reordered list of messages.
        """
        # Step 1: Build mappings of tool_call IDs to message indices
        tool_id_to_toolmsg_idx = {}
        ainode_idx_to_tool_ids = {}
        ainode_messages_with_tool_calls = []

        for idx, msg in enumerate(messages):
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    tool_ids = [tool_call.get("id") for tool_call in tool_calls if "id" in tool_call]
                    ainode_idx_to_tool_ids[idx] = set(tool_ids)
                    ainode_messages_with_tool_calls.append(idx)
            elif isinstance(msg, ToolMessage):
                tool_call_id = getattr(msg, "tool_call_id", None)
                if tool_call_id:
                    tool_id_to_toolmsg_idx[tool_call_id] = idx

        # Step 2: Build groups of messages
        groups = []
        messages_in_groups = set()
        for ainode_idx in ainode_messages_with_tool_calls:
            tool_ids = ainode_idx_to_tool_ids[ainode_idx]
            group_indices = [ainode_idx]
            valid_group = True
            for tool_id in tool_ids:
                tool_msg_idx = tool_id_to_toolmsg_idx.get(tool_id)
                if tool_msg_idx is not None:
                    group_indices.append(tool_msg_idx)
                else:
                    # Missing ToolMessage, skip this AIMessage
                    valid_group = False
                    break
            if valid_group:
                latest_index = max(group_indices)
                group_indices.sort()
                groups.append((latest_index, group_indices))
                messages_in_groups.update(group_indices)

        # Step 3: Remove unmatched AINodeMessages and ToolMessages
        indices_to_remove = set()
        for idx, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and idx in ainode_idx_to_tool_ids and idx not in messages_in_groups:
                indices_to_remove.add(idx)
            elif isinstance(msg, ToolMessage) and idx not in messages_in_groups:
                indices_to_remove.add(idx)

        # Step 4: Sort groups based on the latest position in the group
        groups.sort(key=lambda x: x[0])

        # Step 5: Build the final reordered message list
        output_messages = []
        idx = 0
        total_len = len(messages)
        group_idx = 0

        while idx < total_len:
            # If current index matches the latest index of a group, insert the group
            if group_idx < len(groups) and idx == groups[group_idx][0]:
                group_indices = groups[group_idx][1]
                for g_idx in group_indices:
                    output_messages.append(messages[g_idx])
                group_idx += 1
                idx += 1
                continue

            # Skip messages that are part of groups or to be removed
            if idx in messages_in_groups or idx in indices_to_remove:
                idx += 1
                continue

            output_messages.append(messages[idx])
            idx += 1

        return output_messages

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """
        Sanitize messages for the chat model if the model is not in function calling mode.

        This method processes the messages and removes or converts certain types of messages (ToolMessage, empty AIMessage with tool_calls)
        to regular messages to ensure compatibility with the chat model, especially for Llama3.1 which doesn't handle ToolMessage well
        when not in function calling mode.

        Parameters:
        messages (list): The list of messages to be sanitized.
        chat_model_name (str): The name of the chat model.
        chat_model: The chat model instance.

        Returns:
        list: The sanitized list of messages.
        """

        # If the chat model has tools bound, return the messages as they are.
        if isinstance(chat_model, RunnableBinding) and "tools" in chat_model.kwargs:
            return messages

        # Initialize the result list to store sanitized messages.
        result = []

        for message in messages:
            # Convert ToolMessage to HumanMessage if the chat model is not in function calling mode.
            if isinstance(message, ToolMessage):
                result.append(HumanMessage(content=message.content))
            elif isinstance(message, AIMessage) and message.tool_calls:
                # Skip empty AIMessage with tool_calls.
                if not message.content:
                    pass
                else:
                    # Remove tool_calls argument if the chat model isn't expecting tools
                    result.append(AIMessage(content=message.content))
            # Append other types of messages as they are.
            else:
                result.append(message)

        return result

    def __handle_chat_model_input(
        self,
        input_result,
        parents_result: List[BaseMessage],
        input: Dict[str, Any],
    ) -> List:
        chat_model_input = parents_result[:]

        if isinstance(input_result, ChatPromptValue):
            chat_model_input.extend(input_result.messages)
        elif isinstance(input_result, list):
            chat_model_input.extend([i for i in input_result if _is_message(i)])
        elif _is_message(input_result):
            chat_model_input.append(input_result)

        # some model require for the System Message to be the first input
        # message can be HumanMessage, "AssistantMessage", "SystemMessage"
        # we leave all the other message in the same order
        system_messages = []
        other_messages = []
        for message in chat_model_input:
            if isinstance(message, SystemMessage):
                system_messages.append(message)
            elif isinstance(message, ChatPromptTemplate):
                other_messages += message.format_messages(**input)
            else:
                other_messages.append(message)

        # Some model also don't like to have 2 HumanMessage in a row, we need to merge them
        chat_model_input = []
        last_message = None
        for message in system_messages + other_messages:
            if last_message is None:
                last_message = message.copy()
                chat_model_input.append(last_message)
            else:
                # if the same type
                if type(last_message) is type(message) and not isinstance(message, ToolMessage):
                    last_message.content += "\n\n" + str(message.content)
                elif message:
                    last_message = message.copy()
                    chat_model_input.append(last_message)

        chat_model_input = self._reorder_tool_messages(chat_model_input)

        return chat_model_input

    def _combine_inputs(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig],
        parents_result: List[BaseMessage],
        **kwargs: Any,
    ) -> List:
        input_result = input
        for step in self.inputs:
            input_result = step.invoke(input_result, config, **kwargs)

        return self.__handle_chat_model_input(input_result, parents_result, input)

    async def _acombine_inputs(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig],
        parents_result: List[BaseMessage],
        **kwargs: Any,
    ) -> List:
        with Profiler(
            f"combine_inputs_{self.__class__.__name__}",
            "combine_inputs",
            node_id=self.uuid(),
            node_name=self.name or self.__class__.__name__,
            input_count=len(self.inputs),
        ):
            input_result = input
            for step in self.inputs:
                input_result = await step.ainvoke(input_result, config, **kwargs)

            return self.__handle_chat_model_input(input_result, parents_result, input)

    def _get_config(self, config, count_tokens: BaseCallbackHandler):
        if count_tokens is None:
            return config

        # Assign the stream handler to the config
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        if callbacks is None:
            config["callbacks"] = [count_tokens]
        elif isinstance(callbacks, list):
            config["callbacks"] = callbacks + [count_tokens]
        elif isinstance(callbacks, BaseCallbackManager):
            callbacks = callbacks.copy()
            callbacks.add_handler(count_tokens, inherit=True)
            config["callbacks"] = callbacks
        else:
            raise ValueError(
                f"Unexpected type for callbacks: {callbacks}." "Expected None, list or AsyncCallbackManager."
            )

        return config

    def _pre_invoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        """
        Prepares the node for invocation.
        """
        pass

    def find_metadata(self, key: str) -> Any:
        """
        Gets metadata by key, checking first in the node itself, then in active networks.

        Args:
            key (str): The metadata key to search for

        Returns:
            Any: The metadata value if found, None otherwise
        """
        # First check node's own metadata
        if key in self.metadata:
            return self.metadata[key]

        # Then check active networks from most recent to oldest
        from .runnable_network import RunnableNetwork

        for network in RunnableNetwork.get_active_networks():
            if key in network.metadata:
                return network.metadata[key]

        return None


RunnableNode.update_forward_refs()
