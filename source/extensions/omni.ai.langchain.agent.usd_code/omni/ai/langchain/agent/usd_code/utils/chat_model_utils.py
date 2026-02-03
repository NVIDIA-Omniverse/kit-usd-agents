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


def sanitize_messages_with_expert_type(messages, expert_type: str, **kwargs):
    """Injects expert_type into message payload for specialized AI behavior.

    This is the only way to provide the expert_type to the LLM payload through ChatNVNIM.
    The expert_type controls how the AI responds:
    - "knowledge": For general knowledge-based responses
    - "code": For code-related tasks
    - "metafunction": For function-related operations

    This approach allows specialization of node behavior without requiring multiple chat model
    registrations. The expert_type will be used by ChatNVNIM but safely ignored by regular ChatOpenAI.

    Args:
        messages: List of chat messages to be sanitized
        expert_type: Type of expert behavior to inject ("knowledge", "code", or "metafunction")

    Returns:
        List of sanitized messages with expert_type injected
    """
    # Inject expert_type into the latest message's additional_kwargs
    # This will be picked up by ChatNVNIM._get_request_payload() and ignored by regular ChatOpenAI
    latest_message = messages[-1]
    extra_body = {"expert_type": expert_type}

    if kwargs:
        extra_body.update(kwargs)

    latest_message.additional_kwargs["extra_body"] = extra_body
    return messages
