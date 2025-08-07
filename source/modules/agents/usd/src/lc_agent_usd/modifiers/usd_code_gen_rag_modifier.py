## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .retreiver_utils import AsyncUSDClassAppender
from .retreiver_utils import CodeAtlasErrorMessage
from lc_agent import RunnableHumanNode, RunnableNetwork, RunnableNode
from lc_agent_rag_modifiers import SystemRagModifier


class USDCodeGenRagModifier(SystemRagModifier):
    def __init__(
        self,
        code_atlas_for_errors: bool = True,
        code_atlas_for_human: bool = False,
        retriever_name="usd_code06262024",
        **kwargs
    ):
        super().__init__(retriever_name=retriever_name, **kwargs)
        self._code_atlas_for_errors = code_atlas_for_errors
        self._code_atlas_for_human = code_atlas_for_human

    def _inject_rag(self, network: RunnableNetwork, node: RunnableNode, question: str):
        """Inject RAG functionality into the network."""
        parents = network.get_parents(node)
        if parents and isinstance(parents[0], RunnableHumanNode):
            # Check if there was an error while interpreting the code
            error_text = parents[0].metadata.get("interpreter_error", None)
            if error_text:
                if self._code_atlas_for_errors:
                    # Put the error message at the beginning of the inputs
                    node.inputs.insert(
                        0,
                        CodeAtlasErrorMessage(
                            question=error_text,
                            type="system",
                        ),
                    )
            elif self._top_k != 0 and self._max_tokens != 0:
                if self._code_atlas_for_human:
                    # Put the question at the beginning of the inputs
                    question = parents[0].outputs.content
                    node.inputs.insert(0, AsyncUSDClassAppender(question=question))

        if self._retriever_name:
            return super()._inject_rag(network, node, question)
