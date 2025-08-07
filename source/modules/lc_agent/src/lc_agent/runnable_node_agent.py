## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .runnable_node import RunnableNode
from langchain import hub
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.base import RunnableLambda
from typing import List
from typing import Union
from typing import Dict
from typing import Any


class RunnableNodeAgent(RunnableNode):
    """
    A wrapper for LC Agents that can be used as RunnableNode in a RunnableNetwork.

    The developer only needs to create tools in init.

    Tools are LC compatible classes that can be used in the agent.
    """

    tools: List = []

    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        from langchain.agents import AgentExecutor
        from langchain.agents import create_structured_chat_agent

        def convert_agent_in(chat_model_input: List[Union[AIMessage, HumanMessage, SystemMessage]], **kwargs):
            messages = chat_model_input[:]

            # Get the latest HumanMessage from messages and remove it from messages
            human_message = None
            for i, message in enumerate(messages):
                if isinstance(message, HumanMessage):
                    human_message = i

            if human_message is not None:
                human_message = messages.pop(human_message)

            return {
                "input": human_message.content if human_message else "",
                "chat_history": messages,
            }

        def convert_agent_out(agent_result: Dict[str, Any], **kwargs):
            agent_output = agent_result.get("output", None)
            return AIMessage(content=str(agent_output))

        chat_model = super()._get_chat_model(chat_model_name, chat_model_input, invoke_input, config)

        prompt = hub.pull("hwchase17/structured-chat-agent")

        agent = create_structured_chat_agent(chat_model, self.tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            return_intermediate_steps=True,
        )

        # We need to follow input and output types of RunnableNode
        runnable_in = RunnableLambda(convert_agent_in)
        runnable_out = RunnableLambda(convert_agent_out)

        return runnable_in | agent_executor | runnable_out
