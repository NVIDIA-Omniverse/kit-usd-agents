## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .base_rag_modifier import BaseRagModifier
from .retriever_message import RetrieverMessage
from lc_agent import FromRunnableNode
from lc_agent import RunnableHumanNode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from typing import Optional

# pyright: reportUnusedExpression=false


class HumanRagModifier(BaseRagModifier):
    """
    Class to modify the network with RAG functionality. It creates a human
    node, so the RAG will stay in the history.
    """

    def __init__(self, retriever_name: str, top_k: Optional[int] = None, max_tokens: Optional[int] = None) -> None:
        super().__init__()
        self._retriever_name = retriever_name
        self._top_k = top_k
        self._max_tokens = max_tokens

    def _inject_rag(self, network: RunnableNetwork, node: RunnableNode, question: str):
        """Inject RAG functionality into the network."""

        # Looking for human node, because we need to inject the question before it
        human_node = node
        while human_node and not isinstance(human_node, RunnableHumanNode):
            parents = network.get_parents(human_node)
            human_node = parents[0] if parents else None

        # This is the node before the human node
        parents = network.get_parents(human_node)
        parent_node = parents[0] if parents else None

        # Put the RAG to the separate node
        with network:
            (
                parent_node
                >> FromRunnableNode(
                    RetrieverMessage(
                        question=question,
                        retriever_name=self._retriever_name,
                        type="human",
                        top_k=self._top_k,
                        max_tokens=self._max_tokens,
                    )
                )
                >> human_node
            )
