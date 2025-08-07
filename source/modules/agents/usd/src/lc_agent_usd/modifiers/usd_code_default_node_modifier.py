## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..usd_code_gen_network_node import USDCodeGenNetworkNode
from ..usd_knowledge_network_node import USDKnowledgeNetworkNode
from lc_agent import RunnableHumanNode, RunnableNetwork, RunnableNode, NetworkModifier
from typing import List
from typing import Optional
import time

# pyright: reportUnusedExpression=false


class USDCodeDefaultNodeModifier(NetworkModifier):
    def __init__(
        self,
        enable_code_interpreter=True,
        enable_code_patcher=True,
        max_network_length=5,
        code_interpreter_hide_items: Optional[List[str]] = None,
    ):
        super().__init__()
        self.enable_code_interpreter = enable_code_interpreter
        self.enable_code_patcher = enable_code_patcher
        self.max_network_length = max_network_length
        self.code_interpreter_hide_items = code_interpreter_hide_items

    def on_begin_invoke(self, network: "RunnableNetwork"):
        node = network.parents[0]

        question = node.outputs.content

        system_question = f"""You have 2 type of USD Agents that can help answer questions
Code: Can answer any type of USD code related questions
Knowledge: Can answer any type of USD knowledge related questions

considering that this is the question:
{question}

what the Agent we should call?
just write Code or Knowledge
"""
        with RunnableNetwork(chat_model_name=network.chat_model_name) as fast_network:
            RunnableHumanNode(human_message=system_question)
            RunnableNode()

        start_time = time.time()
        answer = fast_network.invoke()
        print(f"USDCodeDefaultNodeModifier fast_network.invoke() took {time.time() - start_time} seconds")

        with network:
            if "Code" in answer.content:
                node >> USDCodeGenNetworkNode(
                    name="USDCode",
                    show_stdout=False,
                    enable_code_interpreter=self.enable_code_interpreter,
                    enable_code_patcher=self.enable_code_patcher,
                    max_network_length=self.max_network_length,
                    code_interpreter_hide_items=self.code_interpreter_hide_items,
                )
            else:
                node >> USDKnowledgeNetworkNode(name="USDKnowledge")
