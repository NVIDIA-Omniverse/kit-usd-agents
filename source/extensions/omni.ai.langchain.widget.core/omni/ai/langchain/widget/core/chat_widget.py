## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import asyncio
import weakref

import omni.ui as ui
from lc_agent import NetworkList, NetworkModifier, RunnableHumanNode, RunnableNetwork

from .agent_combobox import AgentComboBox
from .chat_history_widget import ChatHistoryWidget
from .chat_view import ChatView


class _SaveNetworkModifier(NetworkModifier):
    def __init__(self, network_list: NetworkList):
        super().__init__()
        self._network_list = network_list

    async def on_end_invoke_async(self, network: "RunnableNetwork"):
        # Save every time the network is processed
        await self._network_list.save_async()


class ChatWidget:
    """
    The ChatWidget class combines ChatHistoryWidget and ChatView,
    managing their interactions. This widget can easily be embedded
    into existing applications.
    """

    def __init__(self, network_list: NetworkList, **kwargs):
        weakself = self.__weak()
        self._frame = ui.Frame(build_fn=lambda s=weakself: s.build_widget(), **kwargs)
        self._network_list = network_list
        self._modifier_id = None

    def destroy(self):
        self._network_list = None
        self._frame = None
        if self._chat_history_widget:
            self._chat_history_widget.destroy()
            self._chat_history_widget = None
        if self._chat_view:
            self._chat_view.destroy()
            self._chat_view = None

    def build_widget(self):
        """
        Builds the entire ChatWidget, including its header, body, and footer.
        """
        with ui.VStack():
            with ui.Frame(height=0):
                self.build_header()
            with ui.Frame():
                self.build_body()
            with ui.Frame(height=0):
                self.build_footer()

        self.new()

    def build_header(self):
        """
        Constructs the header of the ChatWidget, which typically includes toolbar.
        """
        pass

    def build_body(self):
        """
        Constructs the body of the ChatWidget, which includes the ChatHistoryWidget
        (catalog) and ChatView separated by a splitter.
        """
        with ui.VStack():
            with ui.ZStack(height=32):
                with ui.HStack():
                    header_background = ui.Rectangle(width=260, name="history")
                    ui.Rectangle(name="header_view_background")
                with ui.HStack():
                    list_view_toggle = ui.ToolButton(width=32, name="list_view_toggle")
                    header_spacer = ui.Spacer(width=260 - 64)
                    ui.Button(width=32, name="new_chat", clicked_fn=self.new)

                    self.agent_combo = AgentComboBox(
                        height=38,
                        width=210,
                        selection_changed_fn=self._on_agent_selection_changed,
                    )
            with ui.HStack():
                self._chat_history_frame = ui.Frame(width=0)
                with self._chat_history_frame:
                    self._chat_history_widget = self.build_history()
                with ui.Frame():
                    self._chat_view = self.build_view()

        def toggle_history():
            header_background.visible = not header_background.visible
            header_spacer.visible = not header_spacer.visible
            self._chat_history_frame.visible = not self._chat_history_frame.visible

        list_view_toggle.set_clicked_fn(toggle_history)

    def _on_agent_selection_changed(self, selection):
        if self._chat_view.agent_selection_changed_fn:
            self._chat_view.agent_selection_changed_fn(selection)

    def build_history(self) -> ChatHistoryWidget:
        """
        Builds the ChatHistoryWidget (catalog) which is displayed on the left side
        of the body. It allows users to navigate between different conversations.
        """
        chat_history_widget = ChatHistoryWidget(self._network_list, width=260)
        chat_history_widget.set_selection_changed_fn(
            lambda selection, s=self: s._on_network_list_selection_changed(selection)
        )

        return chat_history_widget

    def build_view(self) -> ChatView:
        """
        Builds the ChatView which is displayed on the right side of the body.
        It shows the conversation of the currently selected RunnableNetwork.
        """
        return ChatView()

    def build_footer(self):
        """
        Constructs the footer of the ChatWidget.
        """
        pass

    def add_network(self, network: RunnableNetwork):
        """
        Adds a new network to the ChatWidget.
        """
        self._network_list.append(network)
        self._set_network(network)

    def new(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "Main"

        network = RunnableNetwork(**kwargs)
        if "name" not in network.metadata:
            network.metadata["name"] = "New"
        self.add_network(network)
        return network

    async def new_chat_invoke_async(self, prompt=None, default_node=None, chat_model_name=None, **kwargs):
        """
        Starts a new chat with the given prompt and default node.
        """
        network = self.new(**kwargs)
        if default_node:
            network.default_node = default_node
        if chat_model_name:
            network.chat_model_name = chat_model_name
        if prompt:
            with network:
                RunnableHumanNode(prompt)

        stop_astream_event = self.view_widget.delegate._stop_astream_event
        stop_astream_event.clear()

        async_generator = network.astream()
        try:
            async for _ in async_generator:
                if stop_astream_event.is_set():
                    await async_generator.aclose()
        except asyncio.CancelledError:
            # We always cancel the task. It's not a problem.
            pass
        except BaseException:
            raise
        finally:
            for node in network.get_leaf_nodes():
                network._event_callback(RunnableNetwork.Event.NODE_INVOKED, {"node": node, "network": network})

    def _set_network(self, network):
        old_network = self._chat_view.network
        if old_network:
            old_modifier_id = old_network.get_modifier_id(_SaveNetworkModifier)
            if old_modifier_id is not None:
                old_network.remove_modifier(old_modifier_id)

        self._chat_view.network = network
        self.agent_combo.set_selection(network.default_node)

        if network.get_modifier_id(_SaveNetworkModifier) is None and self._network_list:
            network.add_modifier(_SaveNetworkModifier(self._network_list))

    @property
    def history_widget(self) -> ChatHistoryWidget:
        """
        Returns the ChatHistoryWidget (catalog) of the ChatWidget,
        allowing direct access to its properties and methods.
        """
        return self._chat_history_widget

    @property
    def view_widget(self) -> ChatView:
        """
        Returns the ChatView of the ChatWidget, allowing direct access
        to its properties and methods.
        """
        return self._chat_view

    def _on_network_list_selection_changed(self, selection):
        if selection:
            self._set_network(selection[0])
        else:
            self._set_network(self._network_list[-1])

    def __weak(self):
        return weakref.proxy(self)
