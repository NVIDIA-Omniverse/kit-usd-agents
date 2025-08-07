## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..nodes.usd_code_gen_node import USDCodeGenNode
from langchain_core.messages import AIMessage
from lc_agent import NetworkModifier
from lc_agent import NetworkNode
from lc_agent import RunnableAINode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode

FINAL_NODE_TEXT = (
    "Maximum retry limit reached: "
    "Unable to generate error-free code after multiple attempts. "
    "Please review the code manually or consider rephrasing your request."
)


class NetworkLenghtModifierException(Exception):
    def __init__(self, lenght: int, max_length: int):
        self.lenght = lenght
        self.max_length = max_length
        super().__init__(f"Network length {lenght} exceeds the maximum length of {max_length}")


def is_countable(network, node):
    return isinstance(node, USDCodeGenNode)


def get_max_countable_nodes(network, node):
    node_countable_value = 1 if is_countable(network, node) else 0
    parents = network.get_parents(node)
    if not parents:
        return node_countable_value

    max_countable = 0
    for parent in parents:
        parent_countable = get_max_countable_nodes(network, parent)
        max_countable = max(max_countable, parent_countable)

    return max_countable + node_countable_value


class NetworkLenghtModifier(NetworkModifier):
    def __init__(self, max_length, produce_exception=False, final_node_text=FINAL_NODE_TEXT):
        self.max_length = max_length
        self.produce_exception = produce_exception
        self.final_node_text = final_node_text

    def on_pre_invoke(self, network: RunnableNetwork, node: RunnableNode):
        if isinstance(node, USDCodeGenNode) and not network.get_children(node):
            length = get_max_countable_nodes(network, node)
            if length > self.max_length:
                parents = network.get_parents(node)[0]
                parent = parents if parents else None
                network.remove_node(node)

                stop_reason = (
                    f"[NetworkLenghtModifier] Network length {length} exceeds the maximum length of {self.max_length}"
                )
                if self.produce_exception:
                    if parent:
                        node.metadata["stop_reason"] = stop_reason
                    # This will prevent the node from being invoked
                    raise NetworkLenghtModifierException(length, self.max_length)
                else:

                    # Get the function name of this branch
                    function_name = None
                    current_node = node
                    while current_node:
                        if "function_name" in current_node.metadata:
                            function_name = current_node.metadata["function_name"]
                            break
                        if not current_node.parents:
                            break
                        current_node = current_node.parents[0]
                    if function_name:
                        print(f"[NetworkLenghtModifier] Failed: {function_name}")

                    if self.max_length > 1 and self.final_node_text:
                        created_node = RunnableAINode(ai_message=self.final_node_text)
                        created_node.metadata["stop_reason"] = stop_reason
                        if isinstance(network, NetworkNode):
                            network.outputs = created_node.outputs
