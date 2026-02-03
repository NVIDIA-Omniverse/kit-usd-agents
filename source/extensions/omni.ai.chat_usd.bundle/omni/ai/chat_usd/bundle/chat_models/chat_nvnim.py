# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
