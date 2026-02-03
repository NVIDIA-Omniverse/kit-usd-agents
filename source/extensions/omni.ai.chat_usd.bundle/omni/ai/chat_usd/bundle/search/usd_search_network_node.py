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
        ] = """Agent to search and Import Assets using text or images.
Connect to the USD Search NIM to find USD assets based on natural language queries or similar images.
Drag and drop discovered assets directly into your scene for seamless integration"""

        self.metadata["examples"] = [
            "What can you do?",
            "Find 3 traffic cones and 2 Boxes",
            "I need 3 office chairs",
            "10 warehouse shelves",
            "Find assets similar to /path/to/reference/image.png",
            "Search using this image: C:/Users/example.jpg",
        ]
