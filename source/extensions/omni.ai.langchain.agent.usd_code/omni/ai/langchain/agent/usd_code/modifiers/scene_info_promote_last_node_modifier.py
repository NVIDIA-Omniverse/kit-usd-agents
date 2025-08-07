## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

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
