## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..runnable_utils import RunnableSystemAppend
from ..runnable_node_agent import RunnableNodeAgent
from .usd_atlas_tool import USDAtlasTool
from .codeinterpreter_tool import CodeInterpreterTool
import os


class USDAtlasAgent(RunnableNodeAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Get the current directory
        current_dir = os.path.dirname(os.path.realpath(__file__))

        # Load the system
        with open(
            os.path.join(current_dir, f"{current_dir}/../data/usd_agent_system.md"), "r"
        ) as file:
            system_prompt = file.read()

        self.inputs.append(RunnableSystemAppend(system_message=system_prompt))

        # Create tools
        self.tools = [USDAtlasTool(), CodeInterpreterTool()]
