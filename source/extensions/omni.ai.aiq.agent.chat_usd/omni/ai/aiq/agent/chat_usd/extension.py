## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import carb
import omni.ext

# Register all AIQ components
from aiq.agent.react_agent.register import *
from aiq.embedder.langchain_client import *
from aiq.embedder.nim_embedder import *
from aiq.llm.nim_llm import *
from aiq.plugins.langchain.register import *
from aiq.retriever.milvus.register import *
from aiq.runtime.loader import PluginTypes, discover_and_register_plugins
from aiq.tool.retriever import *
from lc_agent import MultiAgentNetworkNode, RunnableToolNode, get_node_factory

try:
    from lc_agent_aiq import RunnableAIQNode
    from lc_agent_aiq.register import *
except ImportError:
    RunnableAIQNode = None


class ChatUsdExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        """
        Called when the extension is started.
        """
        if RunnableAIQNode is None:
            carb.log_warn("LangChain agent is not installed, skipping USD code function registration")
            return

        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

        aiq_config = {
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

        get_node_factory().register(RunnableAIQNode, name="ChatUSD AIQ", aiq_config=aiq_config)
        get_node_factory().register(MultiAgentNetworkNode, hidden=True)

    def on_shutdown(self):
        """
        Called when the extension is shut down.
        """
        get_node_factory().unregister("ChatUSD AIQ")
        get_node_factory().unregister(MultiAgentNetworkNode)
