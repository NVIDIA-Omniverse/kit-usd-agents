## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Utilities for multi-agent processing in the LC Agent framework.

This module provides helper classes and functions to support message routing,
tool call history, and classification nodes for a multi-agent network.
"""

from ..network_node import NetworkNode
from ..node_factory import get_node_factory
from ..runnable_network import RunnableNetwork
from ..runnable_node import RunnableNode
from ..runnable_utils import RunnableHumanNode
from ..runnable_utils import RunnableSystemAppend
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
import time
from typing import Optional, Tuple, List, Dict

# Debug flag for printing execution time.
PRINT_TIME = False
# Maximum iterations to avoid infinite loops.
MAX_ITERATIONS = 100

# Routing system prompts for long and short versions.
ROUTING_SYSTEM_LONG = """
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Respond with ONE LINE in either of these formats:
{tool_names}
- FINAL <answer>
- <tool_name> <short question>

Example responses:
{first_tool} How do I start?
FINAL This solves the problem

Begin! Remember to respond with exactly one line in the allowed format.
"""

ROUTING_SYSTEM_SHORT = """
Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Respond with ONLY ONE WORD from these options:
{tool_names}
- FINAL (if you can answer directly)

Example responses:
{first_tool}
FINAL It solves the problem

Begin! Remember to respond with exactly one word from the allowed options.
"""

# Templates for tool call instructions.
LONG_FIRST_TOOL_CALL_TEMPLATE = (
    '(Reminder to respond in one line only. Format: "<tool_name> <question>". '
    "The only available options are: {options})"
)

LONG_SUBSEQUENT_TOOL_CALL_TEMPLATE = (
    '(Reminder to respond in one line only. Either "<tool_name> <question>" or "FINAL <answer>". '
    "The only available options are: {options}, FINAL. "
    'If there is an answer, respond with "FINAL <answer>".)'
)

SHORT_FIRST_TOOL_CALL_TEMPLATE = (
    "(Reminder to respond in one word only no matter what. The only available options are: {options})"
)

SHORT_SUBSEQUENT_TOOL_CALL_TEMPLATE = (
    '(Reminder to respond in one word only no matter what. If there is answer, respond with "FINAL". '
    'If you see the same question and result repeating, respond with "FINAL" to avoid loops.)'
)


class RunnableRoutingNode(RunnableNode):
    """
    A routing node that appends a system message to its inputs.

    This node is used within the network to route messages based on the provided system instructions.
    """

    def __init__(self, system_message: str, **kwargs):
        super().__init__(**kwargs)
        # Append the routing system message as a system addition.
        self.inputs.append(RunnableSystemAppend(system_message=system_message))


def get_routing_tools_info(network: "MultiAgentNetworkNode", use_long_prompt: bool) -> Tuple[str, str, str]:
    """
    Helper function to build the tools descriptions, call formats, and example tool name
    for the routing prompt.
    """
    node_factory = get_node_factory()
    tools_descriptions = ""
    tool_call_formats = ""
    example_tool_name = ""

    # Loop over each registered route node to build the list of tools.
    for route in network.route_nodes:
        # Retrieve the node type registered with the node factory.
        node_type = node_factory.get_registered_node_type(route)
        if not node_type:
            continue  # Skip if the node type is not registered.

        # Extract the node description via its field or docstring.
        description = None
        description_field = node_type.__fields__.get("description")
        if description_field:
            description = description_field.default

        if not description:
            node_kwargs = node_factory.get_registered_node_kwargs(route)
            if node_kwargs:
                description = node_kwargs.get("description")

        if not description:
            description = node_type.__doc__

        # Get schema
        function = node_factory.get_registered_node_kwargs(route).get("function", None)
        example_fields = None
        if function and hasattr(function, "input_schema"):
            input_schema = function.input_schema
            model_fields = input_schema.model_fields
            field_names = list(model_fields.keys())
            if len(field_names) > 1:
                # Add schema information for tools with multiple input fields
                schema_info = "\nInput Schema (JSON format):\n"
                schema_info += "{\n"
                for field_name in field_names:
                    field = model_fields[field_name]
                    field_type = (
                        field.annotation.__name__ if hasattr(field.annotation, "__name__") else str(field.annotation)
                    )
                    field_description = field.description or ""
                    # Only show default if it exists and is not any form of Pydantic's undefined value
                    has_real_default = (
                        field.default is not None
                        and "Undefined" not in str(field.default)
                        and "undefined" not in str(field.default).lower()
                    )
                    default_value = f" (default: {field.default})" if has_real_default else ""
                    schema_info += f'  "{field_name}": {field_type}{default_value} - {field_description}\n'
                schema_info += "}\n"

                # Create example with actual field names
                example_fields = ", ".join([f'"{name}": ...' for name in field_names[:2]])
                if len(field_names) > 2:
                    example_fields += ", ..."
                schema_info += f"JSON is expected as a dictionary: {{{example_fields}}}\n"

                # Append schema information to the tool description
                description += schema_info

        tools_descriptions += f"\n{route} - {description or ''}\n"
        if example_fields:
            tool_call_formats += f"- {route} {{{example_fields}}}\n"
        elif use_long_prompt:
            tool_call_formats += f"- {route} <question>\n"
        else:
            tool_call_formats += f"- {route}\n"

        # Capture the name of the first tool (used in examples).
        if not example_tool_name:
            example_tool_name = route

    return tools_descriptions, tool_call_formats, example_tool_name


def _get_routing_prompt(network: "MultiAgentNetworkNode", use_long_prompt: bool) -> str:
    """
    Constructs and returns the routing prompt for the network.

    It gathers all available tools from the network and fills in the corresponding routing template.

    Args:
        network: The multi-agent network node.
        use_long_prompt: Boolean flag indicating whether to generate a long or short prompt.

    Returns:
        The routing prompt string.
    """
    tools_descriptions, tool_call_formats, example_tool_name = get_routing_tools_info(network, use_long_prompt)

    # Select the routing prompt template based on the flag.
    prompt = ROUTING_SYSTEM_LONG if use_long_prompt else ROUTING_SYSTEM_SHORT

    # Replace placeholders with the generated tools, tool names, and first tool.
    prompt = (
        prompt.replace("{tools}", tools_descriptions)
        .replace("{tool_names}", tool_call_formats)
        .replace("{first_tool}", example_tool_name)
    )

    return prompt


def _get_previous_classifications(network: "MultiAgentNetworkNode", node: "RunnableNode") -> List[str]:
    """
    Traverses up the node chain to collect previous tool call classifications.

    It gathers any tool call full strings stored in a node's metadata, stopping at the first human node.

    Args:
        network: The multi-agent network node.
        node: The starting runnable node.

    Returns:
        A list of previous classification strings.
    """
    iteration_node = node
    previous_classifications: List[str] = []
    while iteration_node:
        tool_call_full = iteration_node.metadata.get("tool_call_full")
        if tool_call_full:
            previous_classifications.append(tool_call_full)

        # Get the parent nodes.
        parent_nodes = network.get_parents(iteration_node)
        if not parent_nodes:
            break

        # Proceed with the first parent.
        iteration_node = parent_nodes[0]
    return previous_classifications


async def _process_network_stream(
    tmp_network: RunnableNetwork,
    network: "MultiAgentNetworkNode",
    node: "RunnableNode",
    all_active_networks: List[RunnableNetwork],
) -> str:
    """
    Asynchronously process the network stream and update the output.

    Streams the content as it arrives. Once the output starts with "final", it updates the network outputs
    and notifies the UI via the event callback.

    Args:
        tmp_network: The temporary network used for streaming.
        network: The main multi-agent network.
        node: The current runnable node.
        all_active_networks: List of all active networks for UI updating.

    Returns:
        The accumulated result as a string.
    """
    result = ""
    # Stream results as they are generated.
    async for c in tmp_network.astream():
        result += c.content

        # When the result starts with "final", update UI outputs.
        if result[:5].lower() == "final":
            text = result[6:]  # Extract text after "FINAL "
            if not network.outputs or not isinstance(network.outputs, AIMessage):
                network.outputs = AIMessage(text)
            else:
                network.outputs.content = text

            # Update all active networks.
            update_node = node
            for update_network in all_active_networks:
                update_network._event_callback(
                    RunnableNetwork.Event.NODE_INVOKED,
                    {"node": update_node, "network": update_network},
                )
                update_node = update_network
    return result


async def determine_next_action(
    network: "MultiAgentNetworkNode", node: Optional["RunnableNode"] = None
) -> Optional[dict]:
    """
    Determines the next action by classifying the human's question and tool call history.

    The function constructs a human message with routing instructions and streams the generated result.
    It checks whether the same tool call has already been made to avoid loops. Finally, it parses
    the classification result into an actionable dictionary.

    Args:
        network: The multi-agent network node.
        node: Optional starting runnable node; if not provided, determined from network.

    Returns:
        A dictionary with the parsed result if successful, otherwise None.
    """
    # Retrieve the human node and its tool history.
    human_node, tools_called = _get_human_node_and_tools(network, node)
    if not human_node:
        return False

    # Abort if the tool call count exceeds maximum allowed iterations.
    if len(tools_called) > MAX_ITERATIONS:
        return False
    question = str(human_node.outputs.content)
    if not question:
        return False

    # Determine whether to use the long version of the prompt.
    use_long_prompt = network.generate_prompt_per_agent
    chat_model_name = network._get_chat_model_name(None, None, None)
    routing_prompt = _get_routing_prompt(network, use_long_prompt)
    human_message = _compose_human_message(question, tools_called, network, use_long_prompt)

    if PRINT_TIME:
        print(f"ToolModifier._get_classification human_message: '{human_message}'")

    start_time = time.time()

    # Determine the node to attach the new message.
    inner_node = node
    if isinstance(node, NetworkNode):
        inner_node = node.get_leaf_node()

    # Create a temporary network for classification.
    default_node = network.default_node
    with RunnableNetwork(chat_model_name=chat_model_name) as tmp_network:
        inner_node >> RunnableHumanNode(human_message=human_message)
        if default_node:
            # Use the default node to include the supervisor node's system message.
            get_node_factory().create_node(default_node, inputs=[RunnableSystemAppend(system_message=routing_prompt)])
        else:
            RunnableRoutingNode(routing_prompt)

    all_active_networks = list(RunnableNetwork.get_active_networks())
    previous_classifications = _get_previous_classifications(network, node)

    # Process the streaming output and check for repeated tool calls.
    while True:
        try:
            result = await _process_network_stream(tmp_network, network, node, all_active_networks)
        except Exception as e:
            return False

        if result in previous_classifications:
            # If the result has already been seen, prompt for a different tool or question.
            with tmp_network:
                text = "This tool was already called. Previous attempts:\n"
                text += "\n".join(previous_classifications)
                text += (
                    "\nThe result is provided and it is the same as the previous attempts. "
                    "Please try a different tool or a different question. "
                    "Please respond with FINAL to provide the final answer. "
                )
                # Note: human_message_footer is assumed to be defined elsewhere in the context.
                text += human_message_footer
                RunnableHumanNode(human_message=text)

                if default_node:
                    get_node_factory().create_node(
                        default_node, inputs=[RunnableSystemAppend(system_message=routing_prompt)]
                    )
                else:
                    RunnableRoutingNode(routing_prompt)
        else:
            break

    if PRINT_TIME:
        print(f"ToolModifier._get_classification took {time.time() - start_time} seconds. {result}")

    # Parse the result using the provided parser function.
    parsed_result = parse_classification_result(result, network)
    if parsed_result:
        return parsed_result

    return None


async def create_classification_node(
    network: "MultiAgentNetworkNode", node: Optional["RunnableNode"]
) -> Optional["RunnableNode"]:
    """
    Creates a classification node within the network.

    It reconstructs a human message that contains the initial question and the history of tool calls,
    then creates and configures a classification node with the proper routing prompt. Additionally,
    it marks intermediate nodes to not contribute to the history.

    Args:
        network: The multi-agent network node.
        node: The current runnable node that triggered classification.

    Returns:
        The created classification node if successful, otherwise False.
    """
    # Retrieve the human node and tool call history.
    human_node, tools_called = _get_human_node_and_tools(network, node)
    if not human_node:
        return False

    if len(tools_called) > MAX_ITERATIONS:
        return False
    question = str(human_node.outputs.content)
    if not question:
        return False

    use_long_prompt = network.generate_prompt_per_agent
    routing_prompt = _get_routing_prompt(network, use_long_prompt)
    human_message = _compose_human_message(question, tools_called, network, use_long_prompt)

    if PRINT_TIME:
        print(f">>> create_classification_node human_message:\n{human_message}\n>>>")

    with network:
        metadata = {"multi_agent_classification": True}

        # Create a new human node for classification.
        classification_human_node = RunnableHumanNode(human_message=human_message, metadata=metadata)
        classification_human_node.metadata["contribute_to_ui"] = False
        node >> classification_human_node

        # Create the classification node with the routing prompt.
        if network.default_node:
            classification_node = get_node_factory().create_node(
                network.default_node, inputs=[RunnableSystemAppend(system_message=routing_prompt)], metadata=metadata
            )
        else:
            classification_node = RunnableRoutingNode(routing_prompt, metadata=metadata)

    # Mark intermediate nodes as non-contributive to the history.
    set_nodes_history_contribution(classification_human_node, human_node, contribute=False)

    return classification_node


def set_nodes_history_contribution(
    start_node: "RunnableNode", end_node: "RunnableNode", contribute: bool = False
) -> None:
    """
    Sets the 'contribute_to_history' flag for all nodes between two given nodes.

    This helps in controlling which nodes are considered part of the conversation history.

    Args:
        start_node: The starting node (exclusive).
        end_node: The target ending node (exclusive).
        contribute: Boolean flag to set in the metadata.
    """
    iterator_node = start_node
    while True:
        parents = iterator_node.parents
        if not parents:
            break
        iterator_node = parents[0]

        if iterator_node is end_node:
            break

        iterator_node.metadata["contribute_to_history"] = contribute


def parse_classification_result(result: str, network: "MultiAgentNetworkNode") -> Optional[Dict[str, Optional[str]]]:
    """
    Parses the classification node output and returns an actionable dictionary.

    The expected result is a single line in one of the formats:
      - "FINAL <answer>"
      - "<tool_name> <short question>"

    Args:
        result: The raw result string from the classification output.
        network: The multi-agent network node to validate against its route nodes.

    Returns:
        A dictionary with keys "action", "content", and "full" (original result),
        or None if parsing fails.
    """
    result = result.strip()
    if not result:
        return None
    parts = result.split(maxsplit=1)
    action = parts[0].upper()
    content = parts[1] if len(parts) > 1 else None
    if action == "FINAL":
        return {"action": "FINAL", "content": content, "full": result, "is_loop": False}
    # Check if the action matches any of the valid route nodes.
    for route in network.route_nodes:
        if action == route.upper():
            is_loop = False

            # Loop detection
            human_node, tools_called = _get_human_node_and_tools(network, network.get_leaf_node())
            if human_node and tools_called:
                last_tool_call = tools_called[-1]
                last_tool_call_name = last_tool_call["tool_call_name"]
                last_tool_call_content = last_tool_call["tool_call_content"]
                if last_tool_call_name == route and last_tool_call_content == content:
                    is_loop = True

            return {"action": route, "content": content, "full": result, "is_loop": is_loop}
    return None


def _get_human_node_and_tools(
    network: "MultiAgentNetworkNode",
    node: Optional["RunnableNode"],
) -> Tuple[Optional["RunnableNode"], List[Dict[str, Optional[str]]]]:
    """
    Traverses up the node chain to find the human node along with tool call history.

    When extracting for classification, the human node must:
      - Have contribute_to_history set to True (default True)
      - Not have multi_agent_classification set (or set to False)

    Args:
        network: The multi-agent network node.
        node: The starting node. If not provided, the search begins from the network.

    Returns:
        A tuple containing the human node and a list of tool call dictionaries. Each dictionary
        contains 'tool_call_name', 'tool_call_content', and 'result'.
    """
    # If node is not provided, search the network for a human node.
    if not node:
        node = network
        while node and not isinstance(node, RunnableHumanNode):
            node = node.parents[0] if node.parents else None

    if not node:
        return None, []

    # Identify the target human node and accumulate tool calls.
    human_node = node
    tools_called: List[Dict[str, Optional[str]]] = []
    while human_node:
        metadata = human_node.metadata
        tool_call_name = metadata.get("tool_call_name")
        tool_call_content = metadata.get("tool_call_content")
        if tool_call_name:
            tools_called.append(
                {
                    "tool_call_name": tool_call_name,
                    "tool_call_content": tool_call_content,
                    "result": human_node.outputs.content,
                }
            )

        if (
            isinstance(human_node, RunnableHumanNode)
            and human_node.metadata.get("contribute_to_history", True)
            and not human_node.metadata.get("multi_agent_classification", False)
        ):
            break

        human_node = human_node.parents[0] if human_node.parents else None

    return human_node, tools_called


def _compose_human_message(
    question: str, tools_called: List[Dict[str, Optional[str]]], network: "MultiAgentNetworkNode", use_long_prompt: bool
) -> str:
    """
    Composes the human message for routing based on the original question and tool call history.

    The message begins with the question, then appends details about previous tool calls,
    and finally adds routing instructions depending on whether it is the first call or a subsequent one.

    Args:
        question: The original human question.
        tools_called: A list of tool call dictionaries.
        network: The multi-agent network node.
        use_long_prompt: Boolean indicating whether to use the long prompt template.

    Returns:
        The complete human message string ready for classification.
    """
    message = ""
    if tools_called:
        # Start with the original question.
        message += f"Question: {question}\n"
        # Append previous tool call details in reverse order.
        for tool in reversed(tools_called):
            tool_call_name = tool["tool_call_name"]
            tool_call_content = tool["tool_call_content"]
            tool_call_result = tool["result"]
            message += f"\nAction: {tool_call_name}\n"
            if tool_call_content:
                message += f"Question: {tool_call_content}\n"
            if tool_call_result:
                message += f"Result: {tool_call_result}\n"
        message += "\nAction:\n"
    # Combine the available route node names as options.
    options = ", ".join(network.route_nodes)
    # Use custom routing instructions if provided; otherwise, select based on the prompt length.
    if use_long_prompt:
        footer = LONG_FIRST_TOOL_CALL_TEMPLATE if not tools_called else LONG_SUBSEQUENT_TOOL_CALL_TEMPLATE
    else:
        footer = SHORT_FIRST_TOOL_CALL_TEMPLATE if not tools_called else SHORT_SUBSEQUENT_TOOL_CALL_TEMPLATE
    if network.first_routing_instruction and network.subsequent_routing_instruction:
        footer = network.first_routing_instruction if not tools_called else network.subsequent_routing_instruction
    else:
        # Reapply based on the use_long_prompt flag if custom instructions are not set.
        footer = (
            LONG_FIRST_TOOL_CALL_TEMPLATE
            if (use_long_prompt and not tools_called)
            else (
                LONG_SUBSEQUENT_TOOL_CALL_TEMPLATE
                if use_long_prompt
                else SHORT_FIRST_TOOL_CALL_TEMPLATE if not tools_called else SHORT_SUBSEQUENT_TOOL_CALL_TEMPLATE
            )
        )
    footer = footer.replace("{options}", options)
    message += footer
    return message
