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
from lc_agent import get_chat_model_registry
from lc_agent_usd import CodeExtractorModifier, NetworkLenghtModifier
from lc_agent_usd import USDCodeInteractiveNetworkNode as USDCodeInteractiveNetworkNodeBase

from ..modifiers.double_run_usd_code_gen_interpreter_modifier import DoubleRunUSDCodeGenInterpreterModifier
from ..modifiers.scene_info_modifier import SceneInfoModifier
from ..modifiers.scene_info_promote_last_node_modifier import SceneInfoPromoteLastNodeModifier

MAX_RETRIES_SETTINGS = "/exts/omni.ai.langchain.agent.usd_code/max_retries"
MAX_RETRIES_DEFAULT = 3


class USDCodeInteractiveNetworkNode(USDCodeInteractiveNetworkNodeBase):
    """
    "USD Code Interactive" node. Use it to modify USD stage in real-time and import assets that was found with another tools.

    Important:
    - Never use `stage = omni.usd.get_context().get_stage()` in the code. The global variable `stage` is already defined.
    """

    def __init__(
        self,
        snippet_verification=False,
        scene_info=True,
        max_retries: Optional[int] = None,
        enable_code_interpreter=True,
        enable_code_atlas=True,
        enable_metafunctions=True,
        enable_interpreter_undo_stack=True,
        enable_code_promoting=False,
        double_run_first=True,
        double_run_second=True,
        **kwargs,
    ):
        super().__init__(enable_code_atlas=enable_code_atlas, enable_metafunctions=enable_metafunctions, **kwargs)

        if max_retries is None:
            max_retries = carb.settings.get_settings().get(MAX_RETRIES_SETTINGS) or MAX_RETRIES_DEFAULT

        if scene_info and enable_code_interpreter:
            self.add_modifier(
                SceneInfoModifier(
                    code_interpreter_hide_items=self.code_interpreter_hide_items,
                    enable_interpreter_undo_stack=enable_interpreter_undo_stack,
                    enable_rag=enable_code_atlas,
                    max_retries=max_retries,
                ),
                priority=-100,
            )

        if enable_code_promoting:
            # It will find the code and error message in the subnetwork and copy it to this network output
            self.add_modifier(SceneInfoPromoteLastNodeModifier())
        self.add_modifier(NetworkLenghtModifier(max_length=max_retries))
        self.add_modifier(CodeExtractorModifier(snippet_verification=snippet_verification))
        if enable_code_interpreter:
            self.add_modifier(
                DoubleRunUSDCodeGenInterpreterModifier(
                    hide_items=self.code_interpreter_hide_items,
                    undo_stack=enable_interpreter_undo_stack,
                    first_run=double_run_first,
                    second_run=double_run_second,
                )
            )

        self.metadata["description"] = "Agent to interact and modify the USD stage."
        self.metadata["examples"] = [
            "Create a red sphere",
            "Rotate the sphere 90 degrees around Y axis",
            "Move the selected object up 100 units",
        ]

    def _pre_invoke_network(self):
        """Called before invoking the network."""
        super()._pre_invoke_network()

        if self.chat_model_name == "nvidia/usdcode-llama3-70b-instruct":
            # In the case of Chat USD, we want to use the interactive version of
            # ChatUSD if available
            chat_model_name = "nvidia/usdcode-llama3-70b-instruct-interactive"
            model = get_chat_model_registry().get_model(chat_model_name)
            if model:
                self.chat_model_name = chat_model_name
