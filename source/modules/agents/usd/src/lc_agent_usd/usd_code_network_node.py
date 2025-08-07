## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .modifiers.usd_code_default_node_modifier import USDCodeDefaultNodeModifier
from lc_agent import NetworkNode
from typing import List
from typing import Optional


class USDCodeNetworkNode(NetworkNode):
    code_interpreter_hide_items: Optional[List[str]] = None

    def __init__(self, enable_code_interpreter=True, enable_code_patcher=True, max_network_length=5, **kwargs):
        super().__init__(**kwargs)
        self.add_modifier(
            USDCodeDefaultNodeModifier(
                enable_code_interpreter=enable_code_interpreter,
                enable_code_patcher=enable_code_patcher,
                max_network_length=max_network_length,
                code_interpreter_hide_items=self.code_interpreter_hide_items,
            )
        )

        self.metadata["description"] = (
            "USD code generation and validation assistant. "
            "Creates general USD code snippets and performs checks to ensure code executability. "
            "Ideal for developing and testing USD scripts."
        )
        self.metadata["examples"] = [
            "Write a function that randomizes light intensities",
            "How to create a mesh in USD?",
        ]
