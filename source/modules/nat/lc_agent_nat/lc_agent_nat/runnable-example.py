## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
This script demonstrates how to create a RunnableNode with NAT.
"""

from nat.runtime.loader import PluginTypes
from nat.runtime.loader import discover_and_register_plugins
import asyncio

from lc_agent import RunnableHumanNode
from lc_agent import RunnableNetwork
from lc_agent import RunnableAINode

# Import our new components from the package
from lc_agent_nat import RunnableNATNode


async def main():
    discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

    # Create a dictionary configuration for NAT
    nat_config = {
        "workflow": {
            "_type": "lc_agent_simple",
            "system_message": "You are a dog. Answer like a dog. Woof woof!",
        }
    }

    with RunnableNetwork(chat_model_name="P-GPT4o") as network:
        RunnableHumanNode("Who are you?")
        RunnableNATNode(nat_config=nat_config)

    async for c in network.astream():
        print(c.content, end="")
    print("\n")


if __name__ == "__main__":
    print("Starting NAT integration example...")
    asyncio.run(main())
