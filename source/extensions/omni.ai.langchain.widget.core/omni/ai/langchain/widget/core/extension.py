## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import carb.settings
import omni.ext
import omni.kit.menu.utils
import omni.ui as ui
from lc_agent import RunnableAINode, RunnableHumanImageNode, RunnableHumanNode, RunnableNode, get_node_factory

from .agent_delegate import DefaultDelegate
from .assistant_agent import AssistantAgent
from .chat_view import ChatView
from .chat_window import ChatWindow
from .network_node_delegate import NetworkNodeDelegate

_widget_core_extension = None


def turn_off_background_renderer_load():
    """
    This function turns off the viewport loading.

    It ensures the compatibility with different Kit by setting relevant
    settings.

    If these settings were defined in TOML, this would lead to key duplication
    and parsing failure due to TOML handling of duplicate keys.

    Specifically, it sets "/exts/omni.app.setup/backgroundRendererLoad" for 105.0,
    and "/exts/omni.app.setup/backgroundRendererLoad/renderer" for 105.1.
    """
    setting_path = "/exts/omni.ai.langchain.widget.core/turn_off_background_renderer_load"

    settings = carb.settings.get_settings()
    if not settings.get(setting_path):
        return

    # Set the 'backgroundRendererLoad' to False for version 105.0
    settings.set("/exts/omni.app.setup/backgroundRendererLoad", False)
    # Set the 'backgroundRendererLoad/renderer' to False for version 105.1
    settings.set("/exts/omni.app.setup/backgroundRendererLoad/renderer", False)


class WidgetCoreExtension(omni.ext.IExt):

    def on_startup(self, ext_id):
        global _widget_core_extension
        _widget_core_extension = self

        self.__ext_id = omni.ext.get_extension_name(ext_id)
        turn_off_background_renderer_load()

        get_node_factory().register(AssistantAgent, name="Assistant", hidden=True)
        get_node_factory().register(RunnableHumanNode, name="UserAgent", hidden=True)
        get_node_factory().register(RunnableHumanImageNode, name="UserAgentImage", hidden=True)
        get_node_factory().register(RunnableNode, hidden=True)
        get_node_factory().register(RunnableAINode, hidden=True)

        ChatView.add_delegate("RunnableNode", DefaultDelegate())
        ChatView.add_delegate("NetworkNode", NetworkNodeDelegate())
        ChatView.add_delegate("RunnableAIQNode", NetworkNodeDelegate())

        self._window_name = (
            carb.settings.get_settings().get("/exts/omni.ai.langchain.widget.core/window_name") or "AI Agent"
        )
        self._menu_path = f"Window/{self._window_name}"

        self._window = None
        ui.Workspace.set_show_window_fn(
            self._window_name,
            self._show_window,  # pylint: disable=unnecessary-lambda
        )

        # show the window on activation
        settings = carb.settings.get_settings()
        self._show_window(settings.get("/exts/omni.ai.langchain.widget.core/show_window_on_startup"))

        self._register_menuitem()

    def _show_window(self, visible) -> None:
        if visible:
            if self._window is None:
                self._window = ChatWindow()
                self._window.set_visibility_changed_fn(self._on_visibility_changed)
            else:
                self._window.visible = True
        else:
            if self._window:
                self._window.visible = False

    def _register_menuitem(self):
        self._window_entry = [
            omni.kit.menu.utils.MenuItemDescription(
                name=self._window_name,
                ticked=True,
                ticked_fn=self._is_visible,
                onclick_fn=self._toggle_window,
            )
        ]

        omni.kit.menu.utils.add_menu_items(self._window_entry, "Window")

    def _de_register_menuitem(self):
        self._window_entry = [
            omni.kit.menu.utils.MenuItemDescription(
                name=self._window_name,
                ticked=True,
                ticked_fn=self._is_visible,
                onclick_fn=self._toggle_window,
            )
        ]

        omni.kit.menu.utils.remove_menu_items(self._window_entry, "Window")

    def _toggle_window(self):
        self._show_window(not self._is_visible())

    def _is_visible(self):
        if self._window:
            return self._window.visible
        else:
            return False

    def _on_visibility_changed(self, visible):
        omni.kit.menu.utils.refresh_menu_items("Window")

    def on_shutdown(self):
        global _widget_core_extension
        _widget_core_extension = None

        if self._window:
            self._window.destroy()

        self._window = None
        ChatView.remove_delegate("RunnableNode")
        ChatView.remove_delegate("NetworkNode")
        ChatView.remove_delegate("RunnableAIQNode")

        self._de_register_menuitem()

        get_node_factory().unregister(AssistantAgent)
        get_node_factory().unregister(RunnableHumanNode)
        get_node_factory().unregister(RunnableHumanImageNode)
        get_node_factory().unregister(RunnableNode)
        get_node_factory().unregister(RunnableAINode)


async def new_chat_invoke_async(prompt=None, default_node=None, chat_model_name=None, **kwargs):
    """
    Starts a new chat with the given prompt and default node and runs it.
    """
    if not _widget_core_extension:
        return

    if not _widget_core_extension._window:
        return

    await _widget_core_extension._window.new_chat_invoke_async(prompt, default_node, chat_model_name, **kwargs)


def add_network(network):
    """
    Adds a new network to the chat window.
    """
    if not _widget_core_extension:
        return

    if not _widget_core_extension._window:
        return

    _widget_core_extension._window.add_network(network)
