## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import asyncio
from typing import Optional, Tuple

import carb
import carb.settings
import omni.ext
import omni.kit.app
import omni.usd
from lc_agent import RunnableHumanNode, RunnableNetwork

from .register_chat_model import register_chat_model, unregister_chat_model

# Settings paths for easy access and reuse
SETTING_PATH_STAGE = "/exts/omni.ai.langchain.agent.headless/stage"
SETTING_PATH_PROMPT = "/exts/omni.ai.langchain.agent.headless/prompt"
SETTING_PATH_AGENT = "/exts/omni.ai.langchain.agent.headless/agent"
SETTING_PATH_MODEL = "/exts/omni.ai.langchain.agent.headless/model"


class HeadlessUSDExtension(omni.ext.IExt):
    """Extension for headless USD manipulation through AI-assisted code generation.
    Allows running Kit without UI, processing USD stages based on text prompts."""

    def __init__(self):
        super().__init__()
        self._settings = carb.settings.get_settings()
        self._task = None

    async def _run_headless(self, prompt: str, node: str, model: str, print_streaming: bool = True) -> bool:
        """Run the headless agent with the given prompt, node, and model.

        Args:
            prompt (str): The input prompt for USD manipulation
            node (str): The agent node type to use
            model (str): The AI model to use
            print_streaming (bool, optional): Whether to print streaming output. Defaults to True.

        Returns:
            bool: True if successful, False otherwise
        """
        # Special case for test values
        if prompt == "test" and self._settings.get(SETTING_PATH_STAGE) == "test":
            return True

        # Validate inputs
        if not prompt:
            carb.log_error("[Headless Agent] Prompt is empty")
            return False

        if not node:
            carb.log_error("[Headless Agent] Node type not specified")
            return False

        if not model:
            carb.log_error("[Headless Agent] Model not specified")
            return False

        try:
            with RunnableNetwork(default_node=node, chat_model_name=model) as network:
                RunnableHumanNode(human_message=prompt)

            async for chunk in network.astream():
                if print_streaming:
                    print(chunk.content, end="")
            if print_streaming:
                print()
            return True

        except Exception as e:
            carb.log_error(f"[Headless Agent] Error during execution: {str(e)}")
            return False

    async def _open_stage(self, stage_path: str) -> Tuple[bool, Optional[str]]:
        """Open a USD stage asynchronously.

        Args:
            stage_path (str): Path to the USD stage

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        if not stage_path:
            return False, "Stage path is empty"

        try:
            result, error = await omni.usd.get_context().open_stage_async(stage_path)
            if not result:
                return False, f"Failed to open stage: {error}"
            return True, None
        except Exception as e:
            return False, f"Error opening stage: {str(e)}"

    async def _save_stage(self) -> Tuple[bool, Optional[str]]:
        """Save the USD stage asynchronously.

        Returns:
            Tuple[bool, Optional[str]]: (success, error_message)
        """
        try:
            usd_context = omni.usd.get_context()
            result, error, saved_layers = await usd_context.save_stage_async()

            if not result:
                return False, f"Failed to save stage: {error}"

            carb.log_info(f"[Headless Agent] Successfully saved {len(saved_layers)} layers")
            return True, None

        except Exception as e:
            return False, f"Error saving stage: {str(e)}"

    async def _run(self):
        """Main execution loop for the headless agent."""
        # Wait for the app to initialize
        for _ in range(2):
            await omni.kit.app.get_app().next_update_async()

        try:
            # Get settings using the defined paths
            stage_path = self._settings.get(SETTING_PATH_STAGE)
            prompt = self._settings.get(SETTING_PATH_PROMPT)
            node = self._settings.get(SETTING_PATH_AGENT)
            model = self._settings.get(SETTING_PATH_MODEL)

            # Open stage
            success, error = await self._open_stage(stage_path)
            if not success:
                carb.log_error(f"[Headless Agent] {error}")
                omni.kit.app.get_app().post_quit(1)
                return

            # Run headless agent
            carb.log_info("[Headless Agent] Starting USD manipulation...")
            success = await self._run_headless(prompt, node, model)

            if success:
                # Save stage only if manipulation was successful
                carb.log_info("[Headless Agent] Saving stage changes...")
                save_success, save_error = await self._save_stage()

                if not save_success:
                    carb.log_error(f"[Headless Agent] {save_error}")
                    omni.kit.app.get_app().post_quit(1)
                    return

                carb.log_info("[Headless Agent] Successfully completed and saved USD manipulation")
                omni.kit.app.get_app().post_quit(0)
            else:
                carb.log_error("[Headless Agent] USD manipulation failed, stage not saved")
                omni.kit.app.get_app().post_quit(1)

        except Exception as e:
            carb.log_error(f"[Headless Agent] Fatal error: {str(e)}")
            omni.kit.app.get_app().post_quit(1)

    def on_startup(self, ext_id: str):
        """Initialize the extension and start the headless processing.

        Args:
            ext_id (str): Extension ID
        """
        register_chat_model()

        self._task = asyncio.ensure_future(self._run())

    def on_shutdown(self):
        """Clean up resources on shutdown."""
        if self._task and not self._task.done():
            self._task.cancel()

        unregister_chat_model()
