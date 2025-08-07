## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Any, Dict, List, Optional

from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI


class ChatNVNIM(ChatOpenAI):
    """ChatNVNIM extends ChatOpenAI to support custom payload modifications, specifically for expert_type.

    This class provides the only mechanism to inject expert_type into the LLM payload, which controls
    the AI expert's behavior. The expert_type can be set to:
        - "knowledge": For general knowledge-based responses
        - "code": For code-related tasks
        - "metafunction": For function-related operations

    This approach allows for specialization of node behavior without requiring multiple chat model
    registrations in the system.
    """

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = super()._default_params
        # Keep max_tokens as is, don't convert to max_completion_tokens
        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        # Call parent of ChatOpenAI to avoid the max_tokens conversion
        payload = super(ChatOpenAI, self)._get_request_payload(input_, stop=stop, **kwargs)

        # Convert max_completion_tokens back to max_tokens if present
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")

        # Extract any custom payload from the latest message
        # This is where the expert_type is passed from the RunnableNode
        latest_message = input_[-1]
        extra_body = latest_message.additional_kwargs.get("extra_body", None)

        # If custom payload exists and is a dictionary, update the main payload
        # This allows injection of expert_type and other custom parameters
        if extra_body and isinstance(extra_body, dict):
            if "extra_body" not in payload:
                payload["extra_body"] = {}
            payload["extra_body"].update(extra_body)

        return payload
