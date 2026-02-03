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

from langchain_core.messages import AIMessage, HumanMessage
from lc_agent import NetworkModifier, NetworkNode, RunnableNetwork


class SceneInfoPromoteLastNodeModifier(NetworkModifier):
    async def on_end_invoke_async(self, network: "USDCodeInteractiveNetworkNode"):
        leafs = network.get_leaf_nodes()
        if len(leafs) != 1:
            return

        node = leafs[0]

        if isinstance(network, NetworkNode):
            metadata = node.metadata
            interpreter_code = metadata.get("interpreter_code")
            interpreter_error = metadata.get("interpreter_error")
            interpreter_result = metadata.get("interpreter_result")

            is_multi_agent = ("tool_call_id" in metadata) or ("tool_call_name" in network.metadata)
            if interpreter_code and interpreter_error and is_multi_agent:
                network.metadata["interpreter_code"] = interpreter_code
                network.metadata["interpreter_error"] = interpreter_error
                network.outputs = AIMessage(
                    content=f"```python\n{interpreter_code}\n```\n\n"
                    f"Executed with error:\n```{interpreter_error}\n```\n"
                )
                network._event_callback(
                    RunnableNetwork.Event.NODE_INVOKED,
                    {"node": network, "network": network},
                )

                return
            elif interpreter_code and interpreter_result and is_multi_agent:
                network.metadata["interpreter_code"] = interpreter_code
                network.metadata["interpreter_result"] = interpreter_result
                network.outputs = AIMessage(
                    content=f"```python\n{interpreter_code}\n```\n\nOutput:\n```{interpreter_result}\n```"
                )
                network._event_callback(
                    RunnableNetwork.Event.NODE_INVOKED,
                    {"node": network, "network": network},
                )

                return

        # Use the output of the last node as the output of the network
        network.outputs = HumanMessage(content=node.outputs.content)
        network._event_callback(
            RunnableNetwork.Event.NODE_INVOKED,
            {"node": network, "network": network},
        )
