## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import weakref

import carb.settings
import omni.ui as ui

from .chat_widget import ChatWidget
from .style import chat_window_style


def sanitize_filename(filename: str) -> str:
    """Sanitize the filename by removing invalid characters."""
    import re

    s = re.sub(r"[^\w\s-]", "", filename)  # Remove non-alphanumeric characters except whitespace and hyphen
    s = re.sub(r"\s+", "_", s)  # Replace whitespace with underscore
    return s


class ChatWindow(ui.Window):
    def __init__(self, window_name="AI Agent", width=1200, height=800):
        # Initialize the window with a title and size
        win_name = carb.settings.get_settings().get("/exts/omni.ai.langchain.widget.core/window_name")
        if win_name:
            window_name = win_name
        super().__init__(window_name, width=width, height=height)

        self.frame.style = chat_window_style

        # Subscribe the network to the process and event of the modifier

        use_redis = carb.settings.get_settings().get("/exts/omni.ai.langchain.widget.core/redis")
        if use_redis:
            from lc_agent.network_lists.redis_network_list import RedisNetworkList

            redis_name = carb.settings.get_settings().get("/exts/omni.ai.langchain.widget.core/redis_name")
            self._network_list = RedisNetworkList(redis_name)
        else:
            from lc_agent import JsonNetworkList

            self._network_list = JsonNetworkList()

        self._chat_widget = None

        self.frame.set_build_fn(lambda s=weakref.proxy(self): s.build_ui())

    def build_ui(self):
        if self._network_list is not None:
            # Autoload
            self._network_list.load()
            self._chat_widget = ChatWidget(self._network_list)

    def destroy(self):
        if self._network_list is not None:
            # Autosave
            self._network_list.save()
            self._network_list = None

        self._network = None
        if self._chat_widget:
            self._chat_widget.destroy()
            self._chat_widget = None
        super().destroy()

    async def new_chat_invoke_async(self, prompt=None, default_node=None, chat_model_name=None, **kwargs):
        if not self._chat_widget:
            return

        await self._chat_widget.new_chat_invoke_async(prompt, default_node, chat_model_name, **kwargs)

    def add_network(self, network):
        if not self._chat_widget:
            return

        self._chat_widget.add_network(network)

    def __del__(self):
        print("ChatWindow is deleted")
