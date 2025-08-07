## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os

import omni.ext
import omni.kit.app
from lc_agent import get_node_factory

from .nodes import (
    ChatUSDNavigationNetworkNode,
    ChatUSDNavigationSupervisorNode,
    NavigationGenNode,
    NavigationNetworkNode,
)


class NavigationExtension(omni.ext.IExt):
    """
    Extension for Scene Navigation Agent.

    This extension provides natural language navigation capabilities for USD scenes,
    allowing users to:
    - List all points of interest (POIs) in a scene
    - Navigate to specific POIs by setting the camera transform
    - Save current camera positions as new POIs

    The extension integrates with Chat USD to provide a comprehensive
    USD development and navigation experience.
    """

    def on_startup(self, ext_id):
        """
        Called when the extension is started.

        Args:
            ext_id: The extension id
        """
        self._ext_id = ext_id
        self._register_navigation_agent()
        self._add_usdcode_custom_functions()

    def on_shutdown(self):
        """Called when the extension is shut down."""
        self._unregister_navigation_agent()

    def _register_navigation_agent(self):
        """Register the navigation agent components."""
        get_node_factory().register(NavigationGenNode, name="NavigationGenNode", hidden=True)
        get_node_factory().register(NavigationNetworkNode, name="ChatUSD_Navigation", hidden=True)
        get_node_factory().register(
            ChatUSDNavigationSupervisorNode, name="ChatUSDNavigationSupervisorNode", hidden=True
        )
        get_node_factory().register(ChatUSDNavigationNetworkNode, name="ChatUSD with navigation")

    def _unregister_navigation_agent(self):
        """Unregister the navigation agent components."""
        get_node_factory().unregister(NavigationGenNode)
        get_node_factory().unregister(NavigationNetworkNode)
        get_node_factory().unregister(ChatUSDNavigationSupervisorNode)
        get_node_factory().unregister(ChatUSDNavigationNetworkNode)

    def _add_usdcode_custom_functions(self):
        """Add custom functions to USDCode."""
        import usdcode

        helper_functions_path = os.path.join(os.path.dirname(__file__), "helpers", "helper_functions.py")
        camera_utils_path = os.path.join(os.path.dirname(__file__), "helpers", "camera_utils.py")

        usdcode.setup.add_functions_from_file(helper_functions_path)
        usdcode.setup.add_functions_from_file(camera_utils_path)
