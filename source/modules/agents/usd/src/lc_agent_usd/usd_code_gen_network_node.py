## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .modifiers.code_extractor_modifier import CodeExtractorModifier
from .modifiers.code_fixer_modifier import CodeFixerModifier
from .modifiers.code_interpreter_modifier import CodeInterpreterModifier
from .modifiers.network_length_modifier import NetworkLenghtModifier
from .modifiers.usd_code_gen_interpreter_modifier import USDCodeGenInterpreterModifier
from .modifiers.usd_code_gen_patcher_modifier import USDCodeGenPatcherModifier
from .modifiers.usd_code_gen_rag_modifier import USDCodeGenRagModifier
from .nodes.usd_code_gen_node import USDCodeGenNode
from lc_agent import NetworkNode
from lc_agent import get_node_factory
from typing import List
from typing import Optional

get_node_factory().register(USDCodeGenNode)


class USDCodeGenNetworkNode(NetworkNode):
    """
    A specialized code generation tool focused exclusively on OpenUSD (Universal Scene Description) 
    Python programming tasks. This tool assists with generating, modifying, and explaining code that 
    interacts with USD APIs.

    IMPORTANT: This tool is strictly limited to USD-related code generation and cannot:
    - Answer general Python programming questions
    - Provide explanations about USD concepts or terminology
    - Handle Omniverse-related queries
    - Generate code for other 3D file formats or graphics systems

    Example appropriate questions:
    - "How do I create a USD file with a sphere primitive?"
    - "Generate code to read transform values from a USD stage"
    - "Show me how to set custom attributes on a USD prim"
    - "Write code to traverse a USD stage hierarchy"
    """

    default_node: str = "USDCodeGenNode"
    code_interpreter_hide_items: Optional[List[str]] = None

    def __init__(
        self,
        show_stdout=True,
        code_atlas_for_human: bool = False,
        snippet_verification: bool = False,
        shippet_language_check: bool = False,
        use_code_fixer: bool = False,
        retriever_name="usd_code06262024",
        enable_code_interpreter=True,
        enable_code_patcher=True,
        max_network_length=5,
        **kwargs
    ):
        """
        Initialize the USDCodeGenNetworkNode.

        Args:
            show_stdout: Whether to show stdout output.
            code_atlas_for_human: Whether to use code atlas for human messages.
            snippet_verification: Whether verification of running of the code
                                  snippet is needed. When False, all the snippets
                                  are considered correct.
            shippet_language_check: Whether to check the language tag of the snippet.
            use_code_fixer: Whether to use the code fixer that produces diff.

        """
        super().__init__(**kwargs)

        if max_network_length:
            self.add_modifier(NetworkLenghtModifier(max_length=max_network_length))
        self.add_modifier(
            CodeExtractorModifier(
                snippet_verification=snippet_verification, shippet_language_check=shippet_language_check
            )
        )
        if enable_code_patcher:
            self.add_modifier(USDCodeGenPatcherModifier())
        # this doesn't work in linux/services
        # self.add_modifier(USDCodeGenInterpreterModifier(show_stdout=show_stdout))
        if enable_code_interpreter:
            self.add_modifier(
                CodeInterpreterModifier(show_stdout=show_stdout, hide_items=self.code_interpreter_hide_items)
            )
        if use_code_fixer:
            self.add_modifier(CodeFixerModifier())
        if retriever_name:
            self.add_modifier(USDCodeGenRagModifier(code_atlas_for_human, retriever_name=retriever_name))
