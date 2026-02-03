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

from pathlib import Path
from typing import List

from lc_agent import NetworkNode, RunnableNode, RunnableSystemAppend
from omni.ai.chat_usd.bundle.chat.chat_usd_network_node import ChatUSDNetworkNode, ChatUSDSupervisorNode

from ..modifiers.planning_modifier import PlanningModifier

SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


# Load the system messages from external files
PLANNING_SUPERVISOR_IDENTITY_PATH = SYSTEM_PATH.joinpath("chat_usd_planning_supervisor_identity.md")
PLANNING_SUPERVISOR_IDENTITY = read_md_file(str(PLANNING_SUPERVISOR_IDENTITY_PATH))

PLANNING_GEN_SYSTEM_PATH = SYSTEM_PATH.joinpath("planning_gen_system.md")
PLANNING_GEN_SYSTEM = read_md_file(str(PLANNING_GEN_SYSTEM_PATH))


class PlanningGenNode(RunnableNode):
    """
    Node responsible for creating detailed plans for scene creation/modification.

    This node creates comprehensive, step-by-step plans that include:
    - All objects to create/modify/delete
    - Specific properties (dimensions, position, orientation, color)
    - Dependencies between objects
    - Execution sequence
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add system message from external file
        self.inputs.append(RunnableSystemAppend(system_message=PLANNING_GEN_SYSTEM))


class PlanningNetworkNode(NetworkNode):
    """
    Tool for creating detailed plans for USD scene creation and modification.

    ## Use Case: Scene Planning

    This tool creates comprehensive, detailed plans for scene creation and
    modification tasks. It acts as the project architect, detailing all objects,
    properties, and steps needed for scene modifications.

    The planning system is designed to:
    - Break down user requests into specific, actionable steps
    - Specify all objects that need to be created or modified
    - Define exact properties for each object (size, position, color, etc.)
    - Establish dependencies between objects
    - Create an optimal execution sequence
    - Provide success criteria for each step

    ## When to Use This Tool:

    Use this tool when the user request involves:
    1. Creating a new scene with multiple objects
    2. Modifying an existing scene with several changes
    3. Creating complex objects with multiple components
    4. Any task requiring detailed planning before execution
    5. Any scene manipulation where specific properties are important
    6. Tasks that involve multiple steps or dependencies between objects
    7. ANY scene modification task - this should ALWAYS be the first agent called

    ## Capabilities:

    This tool creates detailed plans that include:
    - List of all objects to create/modify/delete
    - Specific properties for each object
    - Exact numerical values for transforms and sizes
    - Dependencies between objects
    - Execution sequence
    - Success criteria for each step

    ## Examples:

    - "Create an office scene with a desk, chair, and computer"
    - "Build a kitchen with appliances and cabinets"
    - "Add lighting to my living room scene"
    - "Create a playground with swings, a slide, and a sandbox"
    - "Build a simple car model with wheels and windows"
    - "Design a garden with plants, paths, and a small pond"
    """

    default_node: str = "PlanningGenNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add system message from external file
        # self.inputs.append(RunnableSystemAppend(system_message=PLANNING_GEN_SYSTEM))

        # Add the planning modifier
        self.add_modifier(PlanningModifier())


class ChatUSDPlanningSupervisorNode(ChatUSDSupervisorNode):
    """
    Supervisor node that combines USD capabilities with planning capabilities.

    This node extends the ChatUSDSupervisorNode to add planning capabilities.
    It ensures that the planning agent is called before any scene modification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Override the system message to include planning capabilities
        self.inputs.append(RunnableSystemAppend(system_message=PLANNING_SUPERVISOR_IDENTITY))


class ChatUSDPlanningNetworkNode(ChatUSDNetworkNode):
    """
    ChatUSDPlanningNetworkNode is a specialized network node that extends ChatUSDNetworkNode
    to include planning capabilities.

    It allows users to:
    - Create detailed plans for scene creation/modification
    - Modify USD scenes based on those plans
    - Search for USD assets
    - Get information about the scene

    Planning is the central component of this system, ensuring that all scene
    modifications are well-structured and complete.
    """

    default_node: str = "ChatUSDPlanningSupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "ChatUSD_Planning",
    ]
    function_calling: bool = False
    generate_prompt_per_agent: bool = True
    multishot: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metadata["description"] = "Chat USD with detailed planning capabilities."
        self.metadata["examples"].append("Create a plan for a complete office scene")
