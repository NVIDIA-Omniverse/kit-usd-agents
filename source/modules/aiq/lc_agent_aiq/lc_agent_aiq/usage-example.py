## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
This script demonstrates how to use AgentIQ to create a simple workflow.
"""

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.data_models.config import AIQConfig
from aiq.llm.nim_llm import NIMModelConfig
from aiq.runtime.loader import PluginTypes
from aiq.runtime.loader import discover_and_register_plugins
from aiq.data_models.api_server import AIQChatRequest
from lc_agent_aiq import SimpleFunctionConfig
import asyncio
import sys


async def main():
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    try:
        print("Initializing AgentIQ workflow...")

        # Create a NIM LLM configuration
        nim_config = NIMModelConfig(
            model_name="meta/llama-3.1-70b-instruct",
            temperature=0.0,
        )

        system_prompt = "You are a dog. Answer like a dog. Woof woof!"

        # Configure the workflow
        workflow_config = SimpleFunctionConfig(
            system_message=system_prompt,
            llm_name="nim_llm",
            verbose=True,
        )

        # Create an AIQ configuration
        config = AIQConfig(
            llms={"nim_llm": nim_config},
            workflow=workflow_config,
        )

        print("Creating workflow builder...")
        async with WorkflowBuilder.from_config(config) as builder:
            print("Building workflow...")
            workflow = builder.build()

            message = AIQChatRequest.from_string("Who are you?")

            print("Executing workflow with input: 'Who are you?'")
            async with workflow.run(message) as runner:
                print("Waiting for response...")
                async for result in runner.result_stream(to_type=str):
                    print("> Chunk:", result)

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting AgentIQ script...")
    asyncio.run(main())
