## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import re
from pathlib import Path
from typing import List, Optional

from lc_agent import MultiAgentNetworkNode, NetworkModifier, RunnableNode, RunnableSystemAppend
from omni.ai.langchain.agent.usd_code.modifiers.double_run_usd_code_gen_interpreter_modifier import (
    DoubleRunUSDCodeGenInterpreterModifier,
)
from omni.ai.langchain.agent.usd_code.utils.chat_model_utils import sanitize_messages_with_expert_type

SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


identity = read_md_file(f"{SYSTEM_PATH}/chat_usd_supervisor_identity.md")
identity_omniui = read_md_file(f"{SYSTEM_PATH}/chat_usd_supervisor_identity_with_omni_ui.md")


class RunScriptModifier(NetworkModifier):
    async def _run(self, message):
        def remove_brackets(s):
            # Pattern to remove double brackets
            pattern = r"```(?:\w+)?\n(.*?)```"
            match = re.search(pattern, s, re.DOTALL)
            if match:
                return match.group(1).strip()
            return s.strip()

        code = remove_brackets(message)
        if not code:
            return

        code_interpreter = DoubleRunUSDCodeGenInterpreterModifier(first_run=False)
        code_snippet_run = code_interpreter._fix_before_run(code)
        execution_result = code_interpreter._run(code_snippet_run)

        return execution_result

    async def on_post_invoke_async(self, network: "RunnableNetwork", node: "RunnableNode"):
        """
        Post-invoke hook that modifies the network based on the outputs of a node.

        Args:
            network (RunnableNetwork): The current network being executed.
            node (RunnableNode): The node that was just invoked.
        """
        if (
            isinstance(network, MultiAgentNetworkNode)
            and not network.function_calling
            and node.invoked
            and not network.get_children(node)
            and network.multishot
        ):
            metadata = node.metadata
            tool_call_name = metadata.get("tool_call_name")
            if tool_call_name == "FINAL":
                final_message = node.outputs.content
                result = await self._run(final_message)
                print(f"Final run: {result}")


class ChatUSDSupervisorNode(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.append(RunnableSystemAppend(system_message=identity))

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """Sanitizes messages and adds metafunction expert type for USD operations."""
        messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        return sanitize_messages_with_expert_type(messages, "knowledge", rag_max_tokens=0, rag_top_k=0)


class ChatUSDWithOmniUISupervisorNode(RunnableNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs.append(RunnableSystemAppend(system_message=identity_omniui))

    def _sanitize_messages_for_chat_model(self, messages, chat_model_name, chat_model):
        """Sanitizes messages and adds metafunction expert type for USD operations."""
        messages = super()._sanitize_messages_for_chat_model(messages, chat_model_name, chat_model)
        return sanitize_messages_with_expert_type(messages, "knowledge", rag_max_tokens=0, rag_top_k=0)


class ChatUSDNetworkNode(MultiAgentNetworkNode):
    """
    ChatUSDNetworkNode is a specialized network node designed to handle conversations related to USD (Universal Scene Description).
    It utilizes the ChatUSDNodeModifier to dynamically modify the scene or search for USD assets based on the conversation's context.

    This class is an example of how to implement a multi-agent system where different tasks are handled by specialized agents (nodes)
    based on the user's input.
    """

    default_node: str = "ChatUSDSupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
    ]
    function_calling: bool = False
    generate_prompt_per_agent: bool = True
    multishot: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metadata["description"] = "Multi-agents to modify the scene and search USD assets."
        self.metadata["examples"] = [
            "Find me a traffic cone",
            "Create a sphere with a red material",
            "Find an orange and import it to the scene",
        ]


class ChatUSDWithOmniUINetworkNode(ChatUSDNetworkNode):
    """
    ChatUSDNetworkNode is a specialized network node designed to handle conversations related to USD (Universal Scene Description).
    It utilizes the ChatUSDNodeModifier to dynamically modify the scene or search for USD assets based on the conversation's context.

    This class is an example of how to implement a multi-agent system where different tasks are handled by specialized agents (nodes)
    based on the user's input.
    """

    default_node: str = "ChatUSDWithOmniUISupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCode",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "OmniUI_Code",
    ]

    first_routing_instruction: str = (
        '(Reminder to respond in one line only. Format: "<tool_name> <question>". '
        "The only available options are: {options})"
    )

    subsequent_routing_instruction: str = (
        '(Reminder to respond in one line only. Either "<tool_name> <question>" or "FINAL <answer>". '
        'If there is answer, respond with "FINAL <code>". The final answer should contain the final code no matter what. '
        "Remember to use UI code from OmniUI_Code and USD code from ChatUSD_USDCode. "
        "Never use UI code from ChatUSD_USDCode and ChatUSD_SceneInfo. "
        "Never use USD code from OmniUI_Code. If OmniUI_Code provides USD code, ask ChatUSD_USDCode to redo the USD code. )"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_modifier(RunScriptModifier())

        self.metadata["description"] = "Chat USD Agent can modify the scene and create UI elements using omni.ui"
        self.metadata["examples"] = [
            "Create a window with a slider that moves the sphere\nin the current USD stage up and down",
            "Create a window with a button that creates a red sphere",
        ]
