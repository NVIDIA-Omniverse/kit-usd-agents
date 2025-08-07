## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Optional
from .base_rag_modifier import BaseRagModifier
from .retriever_message import RetrieverMessage
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode


class SystemRagModifier(BaseRagModifier):
    """Class to modify the network with RAG functionality."""

    def __init__(self, retriever_name: str, top_k: Optional[int] = None, max_tokens: Optional[int] = None) -> None:
        super().__init__()
        self._retriever_name = retriever_name
        self._top_k = top_k
        self._max_tokens = max_tokens

    def _inject_rag(self, network: RunnableNetwork, node: RunnableNode, question: str):
        """Inject RAG functionality into the network."""
        # Put the RAG to the system prompt
        node.inputs.insert(
            1,
            RetrieverMessage(
                question=question,
                retriever_name=self._retriever_name,
                type="system",
                top_k=self._top_k,
                max_tokens=self._max_tokens,
            ),
        )
