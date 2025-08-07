## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Optional

import carb.settings
from lc_agent import NetworkNode
from lc_agent_usd import CodeExtractorModifier, NetworkLenghtModifier, USDCodeGenPatcherModifier
from lc_agent_usd.modifiers.code_interpreter_modifier import CodeInterpreterModifier

MAX_RETRIES_SETTINGS = "/exts/omni.ai.langchain.agent.usd_code/max_retries"
MAX_RETRIES_DEFAULT = 3


class USDCodeNetworkNode(NetworkNode):
    default_node: str = "USDCodeNode"

    def __init__(self, snippet_verification=False, max_retries: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        if max_retries is None:
            max_retries = carb.settings.get_settings().get(MAX_RETRIES_SETTINGS) or MAX_RETRIES_DEFAULT

        self.add_modifier(NetworkLenghtModifier(max_length=max_retries))
        self.add_modifier(CodeExtractorModifier(snippet_verification=snippet_verification))
        # self.add_modifier(USDCodeGenPatcherModifier())
        self.add_modifier(CodeInterpreterModifier())

        self.metadata["description"] = "Agent to answer USD knowledge and generate Python USD code."
        self.metadata["examples"] = [
            "How to create a new layer?",
            "How do I read attributes from a prim?",
            "Show me how to traverse a stage?",
        ]
