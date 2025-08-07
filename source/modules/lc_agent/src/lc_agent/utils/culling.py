## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)


def _get_message_role(message):
    """
    Extracts the role of a given message.

    Args:
        message: The message object to extract the role from.

    Returns:
        The role of the message. Possible values are "assistant", "user", "system", or "none".
    """
    if isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, HumanMessage):
        return "user"
    elif isinstance(message, SystemMessage):
        return "system"
    return "none"


def _get_message_tokens(message, tokenizer):
    role = _get_message_role(message)
    role_tokens = len(tokenizer.encode(f"<{role}>: "))
    content_tokens = len(tokenizer.encode(message.content))
    additional_tokens = 6
    return role_tokens + content_tokens + additional_tokens


def _cull_message(message, tokenizer, max_tokens, remove_from_start=True):
    role = _get_message_role(message)
    role_tokens = len(tokenizer.encode(f"<{role}>: "))
    # This is the constant number of tokens that are added to the message content
    # We found this number empirically by testing the culling on a variety of messages
    additional_tokens = 6
    max_content_tokens = max_tokens - role_tokens - additional_tokens

    content_tokens = tokenizer.encode(message.content)
    if len(content_tokens) <= max_content_tokens:
        return message

    if remove_from_start:
        culled_content = tokenizer.decode(content_tokens[-max_content_tokens:])
    else:
        culled_content = tokenizer.decode(content_tokens[:max_content_tokens])

    culled_message = message.copy()
    culled_message.content = culled_content
    return culled_message


def _cull_messages(messages, max_tokens, tokenizer, remove_from_start=True):
    if not max_tokens or not tokenizer:
        return messages

    # Determine the priority order for messages
    priority_order = []
    messages_length = len(messages)
    latest_message_index = messages_length - 1 if messages_length > 0 else -1
    
    # Only add latest_message_index if it's valid
    if latest_message_index >= 0:
        priority_order.append(latest_message_index)
    
    system_indices = [i for i in range(messages_length) if isinstance(messages[i], SystemMessage)]
    other_indices = [i for i in range(messages_length) if i != latest_message_index and i not in system_indices]

    priority_order.extend(reversed(system_indices))
    priority_order.extend(reversed(other_indices))

    total_tokens = 0
    culled_indices = set()
    culled_messages = {}

    for i in priority_order:
        current_message = messages[i]
        num_tokens = _get_message_tokens(current_message, tokenizer)
        if total_tokens + num_tokens > max_tokens:
            culled_message = _cull_message(current_message, tokenizer, max_tokens - total_tokens, remove_from_start)
            culled_messages[i] = culled_message
            culled_indices.add(i)
            break  # No need to process further messages as max_tokens is reached
        else:
            culled_messages[i] = current_message
            culled_indices.add(i)
            total_tokens += num_tokens

    # Preserve the original order of messages
    result = [culled_messages[i] for i in range(len(messages)) if i in culled_indices]

    return result
