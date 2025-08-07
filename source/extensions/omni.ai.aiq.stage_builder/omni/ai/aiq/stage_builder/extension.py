## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
from pathlib import Path

import carb
import omni.ext
import omni.kit.app
from aiq.cli.cli_utils.config_override import load_and_override_config
from aiq.runtime.loader import PluginTypes, discover_and_register_plugins
from lc_agent import get_node_factory

from .utils.config_utils import replace_md_file_references

try:
    from lc_agent_aiq import RunnableAIQNode
except ImportError:
    RunnableAIQNode = None


class StageBuilderExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        """
        Called when the extension is started.
        """
        self._registered = False

        if RunnableAIQNode is None:
            carb.log_warn("LangChain agent is not installed, skipping stage builder function registration")
            return

        # Import required functions from chat_usd
        from omni.ai.aiq.agent.chat_usd import (
            chat_usd_code_function,
            chat_usd_scene_info_function,
            chat_usd_search_function,
        )
        from omni_aiq_planning.register import planning_function

        # Discover and register plugins
        discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

        # Get extension path and workflow path
        extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module(__name__)
        extension_path = Path(extension_path)
        workflow_path = extension_path / "data" / "workflow.yaml"

        # Check if workflow file exists
        if not workflow_path.exists():
            carb.log_error(f"Stage Builder workflow file not found at: {workflow_path}")
            carb.log_error("Extension cannot start without workflow configuration")
            return

        try:
            # Load the AIQ configuration
            aiq_config = load_and_override_config(f"{workflow_path}", None)
            # Process the config to replace markdown file references with their contents
            aiq_config = replace_md_file_references(aiq_config, extension_path)
        except Exception as e:
            carb.log_error(f"Failed to load Stage Builder workflow configuration from: {workflow_path}")
            carb.log_error(f"Error details: {type(e).__name__}: {str(e)}")
            return

        # Register the stage builder agent with the node factory
        get_node_factory().register(RunnableAIQNode, name="Stage Builder AIQ", aiq_config=aiq_config)
        self._registered = True

        carb.log_info("Stage Builder extension started successfully")

    def on_shutdown(self):
        """
        Called when the extension is shut down.
        """
        if hasattr(self, "_registered") and self._registered:
            get_node_factory().unregister("Stage Builder AIQ")
            self._registered = False
        carb.log_info("Stage Builder extension shut down")
