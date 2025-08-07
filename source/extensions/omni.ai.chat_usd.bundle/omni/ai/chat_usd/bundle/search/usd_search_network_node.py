## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent import NetworkNode

from .usd_search_modifier import USDSearchModifier


class USDSearchNetworkNode(NetworkNode):
    """
    Use this node to search any asset in Deep Search. It can search, to import call another tool after this one.
    """

    def __init__(self, host_url=None, api_key=None, username=None, url_replacements=None, search_path=None, **kwargs):
        """Initialize USDSearchNetworkNode with optional configuration parameters.

        Note: The api_key parameter is intentionally accepted directly to support flexible
        configuration scenarios. Security is maintained through multiple methods:
        - Direct parameter passing (for dynamic/programmatic configuration)
        - AIQ configuration file
        - Environment variable fallback
        This design allows for both secure production deployments and flexible development workflows.
        """
        super().__init__(**kwargs)

        # Add the USDSearchModifier to the network
        self.add_modifier(
            USDSearchModifier(
                host_url=host_url,
                api_key=api_key,
                username=username,
                url_replacements=url_replacements,
                search_path=search_path,
            )
        )

        # Set the default node to USDSearchNode
        self.default_node = "USDSearchNode"

        self.metadata[
            "description"
        ] = """Agent to search and Import Assets.
Connect to the USD Search NIM to find USD assets base on the natural language query.
Drag and drop discovered assets directly into your scene for seamless integration"""

        self.metadata["examples"] = [
            "What can you do?",
            "Find 3 traffic cones and 2 Boxes",
            "I need 3 office chairs",
            "10 warehouse shelves",
        ]
