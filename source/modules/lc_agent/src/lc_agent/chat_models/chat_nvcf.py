## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import run_in_executor
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional
from typing import Any, Dict, List
import aiohttp
import json
import os
import requests


class NvcfCallAsync:
    def __init__(
        self, payload: Dict[str, Any], headers: Dict[str, str], invoke_url: str
    ) -> None:
        self._started = False
        self._finished = False

        self._payload = payload
        self._headers = headers
        self._invoke_url = invoke_url

        self._response = None
        self._session: aiohttp.ClientSession = None

        self._stack: List[Dict[str, Any]] = []

    def __aiter__(self) -> "NvcfCall":
        """Make this class an asynchronous iterator."""
        return self

    async def __anext__(self) -> Dict[str, Any]:
        """Asynchronous iterator method for fetching response chunks."""
        if self._stack:
            return self._stack.pop(0)

        if not self._started:
            await self._session_begin()

        if self._started and not self._finished:
            await self._fetch_next_chunk()

        if self._stack:
            return self._stack.pop(0)
        else:
            raise StopAsyncIteration

    async def _session_begin(self) -> None:
        """Initializes the session and makes the first invocation to get the response."""
        if not self._started:
            self._session = aiohttp.ClientSession()
            self._response = await self._session.post(
                self._invoke_url, headers=self._headers, json=self._payload
            )
            self._started = True

    async def _session_end(self) -> None:
        """Closes the session and associated resources."""
        if not self._finished:
            if self._response:
                await self._response.release()
            if self._session:
                await self._session.close()
            self._finished = True

    async def _fetch_next_chunk(self) -> None:
        """Fetch the next chunk and process it."""
        chunk = await self._response.content.readany()
        if chunk:
            decoded_chunk = chunk.decode("utf-8")
            for raw_str in decoded_chunk.split("data:"):
                raw_str = raw_str.strip()
                if not raw_str:
                    continue

                # Assuming JSON data starts with "{"
                raw_str = raw_str[raw_str.find("{") :]
                try:
                    json_data = json.loads(raw_str)
                    self._stack.append(json_data)
                    if (
                        "choices" in json_data
                        and json_data["choices"][0].get("finish_reason") is not None
                    ):
                        await self._session_end()
                        break
                except json.JSONDecodeError:
                    pass  # Handle cases where JSON is invalid
        else:
            await self._session_end()


class NvcfCallSync:
    def __init__(self, payload: dict, headers: dict, invoke_url: str) -> None:
        self._started = False
        self._finished = False

        self._payload = payload
        self._headers = headers
        self._invoke_url = invoke_url

        self._response = None
        self._session: requests.Session = requests.Session()

        self._stack: list[dict] = []

    def __iter__(self) -> "NvcfCallSync":
        """Make this class a synchronous iterator."""
        return self

    def __next__(self) -> dict:
        """Iterator method for fetching response chunks."""
        if self._stack:
            return self._stack.pop(0)

        if not self._started:
            self._session_begin()

        if self._started and not self._finished:
            self._fetch_next_chunk()

        if self._stack:
            return self._stack.pop(0)
        else:
            raise StopIteration

    def _session_begin(self) -> None:
        """Initializes the session and makes the first invocation to get the response."""
        if not self._started:
            self._response = self._session.post(
                self._invoke_url, headers=self._headers, json=self._payload
            )
            self._started = True

    def _session_end(self) -> None:
        """Closes the session and associated resources."""
        if not self._finished:
            self._session.close()
            self._finished = True

    def _fetch_next_chunk(self) -> None:
        """Fetch the next chunk and process it."""
        if self._response:
            chunk = self._response.content.decode("utf-8")
            # print("chunk", chunk)
            for raw_str in chunk.split("data:"):
                raw_str = raw_str.strip()
                if not raw_str:
                    continue

                # Assuming JSON data starts with "{"
                raw_str = raw_str[raw_str.find("{") :]
                try:
                    json_data = json.loads(raw_str)
                    self._stack.append(json_data)
                    if (
                        "choices" in json_data
                        and json_data["choices"][0].get("finish_reason") is not None
                    ):
                        self._session_end()
                        break
                except json.JSONDecodeError:
                    pass  # Handle cases where JSON is invalid
        else:
            self._session_end()


class ChatNVCF(SimpleChatModel):
    model: Optional[str] = None
    max_tokens: Optional[int] = 1024
    temperature: float = 0.1
    top_p: float = 1.0
    top_k: float = 1.0
    invoke_url: str = "https://api.nvcf.nvidia.com/v2/nvcf/pexec"
    api_token: Optional[str] = None

    call_depth: int = 0

    def __get_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        def get_message_role(message: BaseMessage) -> str:
            if isinstance(message, AIMessage):
                return "assistant"
            elif isinstance(message, HumanMessage):
                return "user"
            elif isinstance(message, SystemMessage):
                return "system"

            # Default to user
            return "user"

        messages = [
            {"role": get_message_role(m), "content": str(m.content)}
            for m in messages
            if m
        ]

        system_messages = []
        other_messages = []

        # System messages should be first
        for message in messages:
            if message["role"] == "system":
                system_messages.append(message)
            else:
                other_messages.append(message)

        return system_messages + other_messages

    @property
    def _invoke_url(self) -> str:
        invoke_url = self.invoke_url

        if self.model:
            invoke_url += f"/functions/{self.model}"

        return invoke_url

    @property
    def _api_token(self) -> str:
        if self.api_token is not None:
            return self.api_token

        # Get NVIDIA_API_KEY envvar if possible
        api_token = os.getenv("NVIDIA_API_KEY")
        if api_token:
            return api_token

    @property
    def _header(self) -> Dict[str, str]:
        header = {"Accept": "application/json"}
        api_token = self._api_token
        if api_token:
            header["Authorization"] = f"Bearer {api_token}"

        return header

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        converted_messages = self.__get_messages(messages)
        stream = True
        payload = {
            "messages": converted_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
        }
        if stream:
            payload["stream"] = True

        nvcf_call = NvcfCallSync(payload, self._header, self._invoke_url)

        response = None
        result = ""
        for response in nvcf_call:
            if response is None or "choices" not in response or not response["choices"]:
                continue

            if stream:
                result += response["choices"][0]["delta"]["content"]
            else:
                result += response["choices"][0]["message"]["content"]

        if response is None:
            if self.call_depth >= 10:
                return "Error: [ChatNVCF] No response received from NVCF."

            print("Error: No response received from NVCF, trying again...")
            if self.call_depth == 2:
                print("URL", self._invoke_url)
                print("The header is:")
                print(json.dumps(self._header, indent=2))
                print("The payload is:")
                print(json.dumps(payload, indent=2))

            self.call_depth += 1
            result = self._call(messages, stop, run_manager, **kwargs)
            self.call_depth -= 1

            return result

        if response:
            return result

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        payload = {
            "messages": self.__get_messages(messages),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stream": True,
        }

        header = self._header
        header["accept"] = "text/event-stream"

        nvcf_call = NvcfCallAsync(payload, header, self._invoke_url)

        response = None
        # print("payload", json.dumps(payload, indent=2))
        async for response in nvcf_call:
            if response is None or "choices" not in response or not response["choices"]:
                if response and "error" in response:
                    print("Error: [ChatNVCF]", response["error"])
                continue

            content = response["choices"][0]["delta"]["content"]
            yield ChatGenerationChunk(message=AIMessageChunk(content=str(content)))

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "invoke_url": self.invoke_url,
            "api_token": self.api_token,
        }
