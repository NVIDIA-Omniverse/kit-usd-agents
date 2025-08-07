## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent import NetworkNode
from omni.ai.langchain.widget.core.network_node_delegate import NetworkNodeDelegate

from ..search.usd_search_delegate import USDSearchImageDelegate


class ChatUSDNetworkNodeDelegate(NetworkNodeDelegate, USDSearchImageDelegate):
    """
    ChatUSDNetworkNodeDelegate is a delegate class that extends both NetworkNodeDelegate and USDSearchImageDelegate.

    It is designed to handle the dynamic UI representation of a ChatUSDNetworkNode's output in a multi-agent system,
    specifically catering to outputs that are either images (handled by USDSearchImageDelegate) or other types of content
    (handled by NetworkNodeDelegate). This class demonstrates an example of polymorphism and delegation in handling
    different types of outputs from a network node.

    This class is an example of how to implement a delegate that can handle different types of data and dynamically
    adjust the UI representation in a multi-agent system.
    """

    def __init__(self):
        super().__init__()

    def __is_image(self, output: str):
        """Private method to determine if the given output string represents an image. It checks if the output
        is enclosed in square brackets as a simple heuristic. A more precise but slower check (commented out) involves
        attempting to parse the output as JSON.
        """

        output: str = output.strip()
        return output.startswith("{") and output.endswith("}")

        # More presice check but slower
        # try:
        #     json.loads(output)
        #     return True
        # except json.JSONDecodeError:
        #     return False

    def need_rebuild_agent_widget(self, network, node: NetworkNode, data) -> bool:
        """Determines whether the agent's widget needs to be rebuilt based on the type of output.
        If the output is an image, it delegates the decision to USDSearchImageDelegate; otherwise, it falls back to
        NetworkNodeDelegate.
        """
        output = node.outputs.content.strip()
        is_image = self.__is_image(output)
        if is_image:
            return USDSearchImageDelegate.need_rebuild_agent_widget(self, network, node, data)
        else:
            return NetworkNodeDelegate.need_rebuild_agent_widget(self, network, node, data)

    def _build_agent_body(self, network, node: NetworkNode):
        """Constructs the body of the agent's widget based on the type of output. If the output is an image,
        it uses USDSearchImageDelegate's method; otherwise, it uses NetworkNodeDelegate's method.
        """
        output = node.outputs.content.strip()
        is_image = self.__is_image(output)

        if is_image:
            return USDSearchImageDelegate._build_agent_body(self, network, node)
        else:
            return NetworkNodeDelegate._build_agent_body(self, network, node)
