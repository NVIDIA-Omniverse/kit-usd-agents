# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
from typing import List, Optional

from lc_agent import NetworkModifier, RunnableHumanNode, RunnableNetwork, get_node_factory

from ..nodes.scene_info_network_node import SceneInfoNetworkNode
from ..nodes.scene_info_verify_node import SceneInfoVerifyNode

# pyright: reportUnusedExpression=false

human_message = """
Consider the following question:

{question}

Based on this question, do you need information about the scene to write the script? Respond with "yes" or "no".
"""


class SceneInfoModifier(NetworkModifier):
    def __init__(
        self,
        code_interpreter_hide_items: Optional[List[str]] = None,
        enable_interpreter_undo_stack=True,
        enable_rag=True,
        max_retries: Optional[int] = None,
    ):
        super().__init__()
        self.code_interpreter_hide_items = code_interpreter_hide_items
        self.enable_interpreter_undo_stack = enable_interpreter_undo_stack
        self.enable_rag = enable_rag
        self.max_retries = max_retries

    async def has_inject_info(self, network: RunnableNetwork):
        """
        Check if the given network needs information about the scene
        """
        if len(network.parents) != 1:
            return False

        parent_node = network.parents[0]
        question = str(parent_node.outputs.content)
        if not question:
            return False

        start_time = time.time()

        # Classification step. It's yes/no ATM, but it will be more cases in the future
        with RunnableNetwork(chat_model_name=network.chat_model_name) as tmp_network:
            parent_node >> RunnableHumanNode(human_message=human_message.format(question=str(question)))
            SceneInfoVerifyNode()

        result = await tmp_network.ainvoke()
        result = str(result.content).strip().lower()

        print(f"SceneInfoModifier.has_inject_info took {time.time() - start_time} seconds, result: {result}")

        return result.lower() == "yes"

    async def on_begin_invoke_async(self, network: "USDCodeInteractiveNetworkNode"):
        if not network.nodes:
            if await self.has_inject_info(network):
                parent_node = network.parents[0]
                question = str(parent_node.outputs.content)
                parent_node >> SceneInfoNetworkNode(
                    question=question,
                    enable_rag=self.enable_rag,
                    code_interpreter_hide_items=self.code_interpreter_hide_items,
                    max_retries=self.max_retries,
                )
