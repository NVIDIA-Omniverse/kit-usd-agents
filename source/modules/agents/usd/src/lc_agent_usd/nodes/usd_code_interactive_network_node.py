## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import List, Optional

from lc_agent import NetworkNode
from ..modifiers.usd_code_gen_rag_modifier import USDCodeGenRagModifier


class USDCodeInteractiveNetworkNode(NetworkNode):
    """
    "USD Code Interactive" node. Use it to modify USD stage in real-time and import assets that was found with another tools.
    """

    default_node: str = "USDCodeInteractiveNode"
    code_interpreter_hide_items: Optional[List[str]] = None

    def __init__(
        self,
        enable_code_atlas=True,
        enable_metafunctions=True,
        retriever_name=None,
        usdcode_retriever_name=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        rag_top_k = self.find_metadata("rag_top_k")
        rag_max_tokens = self.find_metadata("rag_max_tokens")

        if enable_code_atlas or retriever_name:
            self.add_modifier(
                USDCodeGenRagModifier(
                    code_atlas_for_human=enable_code_atlas,
                    code_atlas_for_errors=enable_code_atlas,
                    retriever_name=retriever_name,
                    top_k=rag_top_k,
                    max_tokens=rag_max_tokens,
                )
            )

        # Metafunctions
        if enable_metafunctions:
            from ..modifiers.mf_rag_modifier import MFRagModifier

            args = {}
            if usdcode_retriever_name:
                args["retriever_name"] = usdcode_retriever_name

            self.add_modifier(
                MFRagModifier(
                    top_k=rag_top_k,
                    max_tokens=rag_max_tokens,
                    **args,
                )
            )
