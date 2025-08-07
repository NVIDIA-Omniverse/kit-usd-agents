## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from abc import ABC
from abc import abstractmethod
from langchain_core.messages import HumanMessage
from lc_agent import MultiAgentNetworkNode
from lc_agent import NetworkModifier
from lc_agent import NetworkNode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from typing import Optional


class BaseRagModifier(NetworkModifier, ABC):
    """Class to modify the network with RAG functionality."""

    def _needs_rag(self, network: RunnableNetwork, node: RunnableNode) -> bool:
        """Check if RAG functionality is needed for the given node."""
        if isinstance(node, NetworkNode):
            return False

        if not isinstance(node, RunnableNode) or network.get_children(node) or node.outputs is not None:
            return False

        parents = network.get_parents(node)
        if not parents:
            return False

        return True

    def _find_question(self, node: RunnableNode, network: RunnableNetwork) -> Optional[str]:
        """Find the human message in the node outputs."""
        while True:
            if (
                node.outputs
                and not isinstance(node, NetworkNode)
                and not isinstance(node, MultiAgentNetworkNode.RunnableSupervisorNode)
            ):
                if isinstance(node.outputs, HumanMessage):
                    return node.outputs.content
                elif isinstance(node.outputs, list):
                    for output in node.outputs:
                        if isinstance(output, HumanMessage):
                            return output.content

            parents = network.get_parents(node)
            if not parents:
                return

            node = parents[0]

    @abstractmethod
    def _inject_rag(self, network: RunnableNetwork, node: RunnableNode, question: str):
        """Inject RAG functionality into the network."""
        pass

    def on_pre_invoke(self, network: RunnableNetwork, node: RunnableNode):
        """Called before invoking a node."""
        if not self._needs_rag(network, node):
            return

        parent: RunnableNode = network.get_parents(node)[0]
        question = self._find_question(parent, network)
        # print(f"[{type(self).__name__}] Injecting RAG for {type(node).__name__} with question: {question}")
        if not question:
            return

        # We are here because we have a human question for LLM
        self._inject_rag(network, node, question)
