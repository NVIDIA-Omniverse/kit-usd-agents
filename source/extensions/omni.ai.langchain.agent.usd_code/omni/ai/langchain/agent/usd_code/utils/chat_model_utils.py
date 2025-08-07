## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##


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
