## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from pathlib import Path
from typing import List

from lc_agent import NetworkNode, RunnableNode, RunnableSystemAppend
from omni.ai.chat_usd.bundle.chat.chat_usd_network_node import ChatUSDNetworkNode, ChatUSDSupervisorNode

from ..modifiers.navigation_modifier import NavigationModifier

SYSTEM_PATH = Path(__file__).parent.joinpath("systems")


def read_md_file(file_path: str):
    with open(file_path, "r") as file:
        return file.read()


# Load the system messages from external files
NAVIGATION_SUPERVISOR_IDENTITY_PATH = SYSTEM_PATH.joinpath("chat_usd_navigation_supervisor_identity.md")
NAVIGATION_SUPERVISOR_IDENTITY = read_md_file(str(NAVIGATION_SUPERVISOR_IDENTITY_PATH))

NAVIGATION_GEN_SYSTEM_PATH = SYSTEM_PATH.joinpath("navigation_gen_system.md")
NAVIGATION_GEN_SYSTEM = read_md_file(str(NAVIGATION_GEN_SYSTEM_PATH))


class NavigationGenNode(RunnableNode):
    """
    Node responsible for scene navigation operations.

    This node handles:
    - Listing all points of interest in a scene
    - Navigating to specific points by setting camera transforms
    - Saving current camera positions as new points of interest
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add system message from external file
        self.inputs.append(RunnableSystemAppend(system_message=NAVIGATION_GEN_SYSTEM))


class NavigationNetworkNode(NetworkNode):
    """
    Tool for scene navigation in USD scenes.

    ## Use Case: Scene Navigation

    This tool enables natural language navigation within USD scenes, allowing users to navigate to
    Points of Interest (POIs) that can be specific locations or objects within their facilities.

    The navigation system is designed to:
    - Work with facility-specific 3D environments
    - Store and retrieve contextual information about important locations
    - Provide intuitive camera control through natural language commands
    - Support exploration of complex environments through saved viewpoints

    ## When to Use This Tool:

    Use this tool when the user request involves:
    1. Navigating to a specific location or viewpoint in the scene
    2. Listing available points of interest in the scene
    3. Saving the current camera position as a point of interest
    4. Requesting information about scene navigation capabilities
    5. Exploring a digital twin facility through different viewpoints
    6. Requesting to move the camera to view specific objects or areas

    ## Capabilities:

    This tool coordinates navigation operations in USD scenes by:
    - Listing all available points of interest with their descriptions
    - Navigating to specific points by setting camera transforms
    - Saving current camera positions as new points of interest
    - Storing facility-specific metadata for each point of interest
    - Supporting natural language queries about scene navigation

    Points of interest are stored as custom metadata in the scene with attributes like
    name, position, and look-at coordinates, ensuring persistence across sessions.

    ## Examples:

    - "Show me all the available viewpoints in this facility"
    - "Take me to the kitchen area"
    - "Navigate to the main entrance"
    - "Save this view as 'production line overview'"
    - "I want to see the building from the south entrance"
    - "Move the camera to focus on the assembly station"
    - "What points of interest are available in this scene?"
    - "Can you save my current position as 'maintenance access point'?"
    - "Give me a tour of the facility"
    - "Show me the view from the security camera position"
    - "Navigate to the loading dock"
    - "Save this perspective as 'optimal inspection angle'"
    """

    default_node: str = "NavigationGenNode"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Add the navigation modifier
        self.add_modifier(NavigationModifier())

        self.metadata["description"] = "Scene navigation operations for USD scenes."
        self.metadata["examples"] = [
            "List all points of interest",
            "Navigate to the kitchen",
            "Save this view as 'front entrance'",
        ]


class ChatUSDNavigationSupervisorNode(ChatUSDSupervisorNode):
    """
    Supervisor node that combines USD capabilities with scene navigation.

    This node extends the ChatUSDSupervisorNode to add navigation capabilities.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Override the system message to include navigation capabilities
        self.inputs.append(RunnableSystemAppend(system_message=NAVIGATION_SUPERVISOR_IDENTITY))


class ChatUSDNavigationNetworkNode(ChatUSDNetworkNode):
    """
    ChatUSDNavigationNetworkNode is a specialized network node that extends ChatUSDNetworkNode
    to include navigation capabilities.

    It allows users to:
    - Modify USD scenes
    - Search for USD assets
    - Get information about the scene
    - Navigate the scene using points of interest
    """

    default_node: str = "ChatUSDNavigationSupervisorNode"
    route_nodes: List[str] = [
        "ChatUSD_USDCodeInteractive",
        "ChatUSD_USDSearch",
        "ChatUSD_SceneInfo",
        "ChatUSD_Navigation",
    ]
    function_calling: bool = False
    generate_prompt_per_agent: bool = True
    multishot: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.metadata["examples"].append("Show me all the viewpoints in this scene")
