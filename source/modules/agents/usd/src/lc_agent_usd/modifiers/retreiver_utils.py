## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import _convert_to_message as convert_to_message
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables.base import RunnableLambda
from lc_agent import RunnableHumanNode, RunnableNetwork
from lc_agent.code_atlas import USDAtlasTool
from lc_agent.utils.profiling_utils import Profiler
from lc_agent.utils.pydantic import BaseModel
from lc_agent_rag_modifiers import BaseRetrieverMessage
from pydantic import model_serializer
from typing import Any
from typing import Callable
from typing import Literal
from typing import Optional
import re
import time

# Global variables
_CODE_ATLAS_TOOL = None
DEBUG_TIMING = False

# Constants for messages
CLASSES_MESSAGE = """
Some USD classes that might be helpful are:

{classes}
"""

USD_CLASS_QUERY_PROMPT = """This is the list of USD classes:

{classes}

TASK: Identify the most relevant USD classes for the following question.

Question: "{question}"

INSTRUCTIONS:
1. Return ONLY the class names that are directly relevant to solving this question
2. Return a MAXIMUM of 10 classes
3. Sort classes by relevance (most relevant first)
4. Format: Return class names only, separated by commas
5. Do not include any explanations, descriptions, or additional text
6. If no classes are relevant, return an empty response

Example good response:
UsdGeom, UsdShade, UsdSkel

Example bad response:
Here are the relevant classes: UsdGeom, UsdShade (these would be useful because...)
"""


def _get_code_atlas_tool() -> USDAtlasTool:
    """Initialize and return the USDAtlasTool."""
    global _CODE_ATLAS_TOOL
    if _CODE_ATLAS_TOOL is None:
        _CODE_ATLAS_TOOL = USDAtlasTool()
    return _CODE_ATLAS_TOOL


def _split_string(input_string: str) -> list:
    """Split the input string into a list of words."""
    # Replace newline characters and commas with spaces
    input_string = input_string.replace("\n", " ").replace(",", " ")
    # Split the string by spaces and remove leading/trailing whitespace from each word
    words = [word.strip() for word in input_string.split()]
    # Remove dots at the beginning and end of each word
    words = [word.strip(".") for word in words]
    return words


class AsyncUSDClassAppender(BaseModel, RunnableLambda):
    """Class to handle asynchronous appending of messages with relevant USD classes."""

    question: str
    type: str = "system"
    func: Optional[Callable] = None
    afunc: Optional[Any] = None
    name: Optional[str] = None

    def __init__(self, question: str, **kwargs):
        BaseModel.__init__(self, question=question, **kwargs)
        RunnableLambda.__init__(self, func=self._aexecute)

    def _iter(self, *args, **kwargs):
        """Pydantic serialization method."""
        kwargs["exclude"] = (kwargs.get("exclude", None) or set()) | {"func", "afunc"}
        yield from super()._iter(*args, **kwargs)

    @model_serializer
    def serialize_model(self) -> dict:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except excluded ones
        result = {}
        for field_name, field_value in self:
            if field_name not in ["func", "afunc"]:
                result[field_name] = field_value

        return result

    async def _aexecute(self, input, **kwargs):
        """Execute the append operation asynchronously."""
        with Profiler(
            "retriever_execute_" + type(self).__name__,
            "retriever",
            retriever_name=type(input).__name__,
            question_type=self.type,
            question=self.question,
        ):
            return await self._append_relevant_classes(input)

    async def _append_relevant_classes(self, input):
        """Append relevant USD classes to the input."""
        active_network = RunnableNetwork.get_active_network()
        if not active_network:
            return input

        default_node: str = active_network.default_node
        chat_model_name: Optional[str] = active_network.chat_model_name

        classes = self._get_all_classes()
        prompt = self._prepare_prompt(classes)

        with Profiler(
            "get_relevant_classes_" + type(self).__name__,
            "chunk",
            default_node=default_node,
            chat_model_name=chat_model_name,
        ):
            found_classes = await self._fetch_relevant_classes_from_network(prompt, default_node, chat_model_name)
        if found_classes:
            message = self._create_message_from_classes(found_classes)
            input = self._append_message_to_input(input, message)

        return input

    def _get_all_classes(self) -> str:
        """Retrieve all USD classes."""
        classes = _get_code_atlas_tool().cache._classes
        classes = [c.full_name for _, c in classes.items() if "_" not in c.name]
        return "\n".join(classes)

    def _prepare_prompt(self, classes: str) -> str:
        """Prepare the prompt for fetching relevant classes."""
        return USD_CLASS_QUERY_PROMPT.replace("{question}", self.question).replace("{classes}", classes)

    async def _fetch_relevant_classes_from_network(
        self, prompt: str, default_node: str, chat_model_name: Optional[str]
    ) -> list:
        """Fetch relevant classes from the network."""
        start_time = time.time() if DEBUG_TIMING else None

        with RunnableNetwork(chat_model_name=chat_model_name, default_node=default_node) as network:
            RunnableHumanNode(human_message=prompt)
            result = await network.ainvoke()

        if DEBUG_TIMING:
            elapsed_time = time.time() - start_time
            print(f"[AsyncUSDClassAppender] Network fetch took {elapsed_time:.2f} seconds")

        return _split_string(result.content)

    def _create_message_from_classes(self, classes: list) -> BaseMessage:
        """Create a message from the found USD classes."""
        found_classes = [self._lookup_class(class_name) for class_name in classes]
        found_classes = [cls for cls in found_classes if cls]
        found_classes_text = "\n".join(found_classes)
        return convert_to_message((self.type, CLASSES_MESSAGE.replace("{classes}", found_classes_text)))

    def _lookup_class(self, class_name: str) -> Optional[str]:
        """Lookup a class in the USDAtlasTool cache."""
        return _get_code_atlas_tool().cache.lookup_class(
            class_name,
            methods=True,
            method_bodies=False,
            docs=False,
            pass_in_body=False,
        )

    def _append_message_to_input(self, input, message: BaseMessage):
        """Append the message to the input."""
        if isinstance(input, ChatPromptValue):
            input.messages.append(message)
        elif isinstance(input, list):
            input.append(message)
        elif isinstance(input, dict):
            input = [input, message]
        else:
            input = ChatPromptValue(messages=[message])
        return input


class CodeAtlasErrorMessage(BaseRetrieverMessage):
    """Class to handle retrieving and formatting messages with code atlas."""

    name: Literal["code_atlas_error_message"] = "code_atlas_error_message"

    def _process_question(self, question):
        """Process the question and return the formatted message."""
        error_text = question
        return self._handle_error(error_text)

    def _handle_error(self, error_text: str) -> Optional[str]:
        patterns = [
            (
                r"module '(\S+)' has no attribute '(\S+)'",
                self._lookup_module,
            ),
            (
                r"Python argument types in\n\s+(\S+)\.(\S+)\(([\s\S]+?)\)\n\s+did not match C\+\+ signature",
                self._lookup_class_method,
            ),
            (
                r"'(\S+)' object has no attribute '(\S+)'",
                self._lookup_class_method,
            ),
        ]

        for pattern, handler in patterns:
            match = re.search(pattern, error_text)
            if match:
                groups = match.groups()
                return handler(*groups)

        return None

    def _lookup_module(self, module_name: str, attribute: str) -> Optional[str]:
        return _get_code_atlas_tool().cache.lookup_module(module_name, classes=True, methods=False, docs=False)

    def _lookup_class_method(self, class_name: str, method_name: str, *args) -> Optional[str]:
        return _get_code_atlas_tool().cache.lookup_class(
            class_name,
            methods=True,
            method_bodies=False,
            docs=False,
            pass_in_body=False,
        )
