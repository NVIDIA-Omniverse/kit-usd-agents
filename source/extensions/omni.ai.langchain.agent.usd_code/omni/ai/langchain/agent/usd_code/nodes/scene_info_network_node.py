## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
from typing import List, Optional

from lc_agent import NetworkNode, RunnableHumanNode
from lc_agent_usd import CodeExtractorModifier, NetworkLenghtModifier, USDCodeGenRagModifier

from ..modifiers.double_run_usd_code_gen_interpreter_modifier import DoubleRunUSDCodeGenInterpreterModifier
from ..modifiers.scene_info_promote_last_node_modifier import SceneInfoPromoteLastNodeModifier

human_message = """Consider the following task:
{question}

Generate a script that prints only the necessary information about the scene required to perform this task. Keep the script as short and concise as possible.
"""


class SceneInfoNetworkNode(NetworkNode):
    """
    Use this function to get any information about the scene.

    Use it only in the case if the user wants to write ascript and you need information about the scene.

    For example use it if you need the scene hirarchy or the position of objects.

    Always use it if the user asked for an object in the scene and didn't provide the exact name.

    Never use USDCodeInteractiveNetworkNode if the exact name is not known.
    Use SceneInfoNetworkNode to get the name first and use USDCodeInteractiveNetworkNode when the name is known.

    Use this function to get any size or position of the object.
    """

    default_node: str = "SceneInfoGenNode"
    code_interpreter_hide_items: Optional[List[str]] = None

    def __init__(
        self,
        question: Optional[str] = None,
        enable_interpreter_undo_stack=True,
        enable_rag=True,
        max_retries: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if max_retries is None:
            max_retries = 15

        self.add_modifier(SceneInfoPromoteLastNodeModifier())
        self.add_modifier(NetworkLenghtModifier(max_length=max_retries))
        self.add_modifier(CodeExtractorModifier())
        self.add_modifier(
            DoubleRunUSDCodeGenInterpreterModifier(
                success_message="This is some scene information:\n",
                hide_items=self.code_interpreter_hide_items,
                undo_stack=enable_interpreter_undo_stack,
                first_run=False,
            )
        )
        if enable_rag:
            self.add_modifier(USDCodeGenRagModifier())

        if question:
            with self:
                RunnableHumanNode(human_message=human_message.format(question=question))
