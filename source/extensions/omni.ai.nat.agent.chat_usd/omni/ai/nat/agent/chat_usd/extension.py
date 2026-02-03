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

import carb
import omni.ext
from lc_agent import MultiAgentNetworkNode, RunnableToolNode, get_node_factory

# Register all NAT components
from nat.agent.react_agent.register import *

# from nat.embedder.langchain_client import *
# from nat.embedder.nim_embedder import *
from nat.llm.nim_llm import *
from nat.plugins.langchain.register import *
from nat.retriever.milvus.register import *
from nat.runtime.loader import PluginTypes, discover_and_register_plugins
from nat.tool.retriever import *

try:
    from lc_agent_nat import RunnableNATNode
    from lc_agent_nat.register import *
except ImportError:
    RunnableNATNode = None


class ChatUsdExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        """
        Called when the extension is started.
        """
        if RunnableNATNode is None:
            carb.log_warn("LangChain agent is not installed, skipping USD code function registration")
            return

        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

        nat_config = {
            "functions": {
                "ChatUSD_USDCodeInteractive": {
                    "_type": "ChatUSD_USDCodeInteractive",
                },
                "ChatUSD_SceneInfo": {
                    "_type": "ChatUSD_SceneInfo",
                },
            },
            "workflow": {
                "_type": "ChatUSD",
                "tool_names": [
                    "ChatUSD_USDCodeInteractive",
                    "ChatUSD_SceneInfo",
                ],
            },
        }

        get_node_factory().register(RunnableNATNode, name="ChatUSD NAT", nat_config=nat_config)
        get_node_factory().register(MultiAgentNetworkNode, hidden=True)

    def on_shutdown(self):
        """
        Called when the extension is shut down.
        """
        get_node_factory().unregister("ChatUSD NAT")
        get_node_factory().unregister(MultiAgentNetworkNode)
