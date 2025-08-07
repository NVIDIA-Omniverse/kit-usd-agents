## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

__all__ = ["ChatHistoryWidget"]

import tempfile
import weakref
import webbrowser
from functools import partial
from typing import Callable, List, Optional

import omni.ui as ui
from lc_agent import JsonNetworkList, NetworkList, RunnableNetwork

# Try to import the new profiling API
try:
    from lc_agent.utils.profiling_html import create_profiling_html

    HAS_PROFILING_HTML = True
except ImportError:
    HAS_PROFILING_HTML = False


class _ChatHistoryItem(ui.AbstractItem):
    def __init__(self, network):
        super().__init__()
        self._network = network
        self._model = ui.SimpleStringModel(str(self))
        self._event_id = self._network.set_event_fn(self._on_metadata_event, RunnableNetwork.Event.METADATA_CHANGED)

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._network is not None and self._event_id is not None:
            self._network.remove_event_fn(self._event_id)
        self._network = None
        self._model = None
        self._event_id = None

    @property
    def name_model(self):
        name = str(self)
        self._model.as_string = name
        return self._model

    def _on_metadata_event(self, event_type, payload):
        if payload["key"] == "name":
            name = payload["value"]
            if name:
                self._model.as_string = name

    def __repr__(self):
        return self._network.metadata.get("name", "Unknown Network")


class _ChatHistoryModel(ui.AbstractItemModel):
    def __init__(self, network_list: Optional[NetworkList] = None):
        super().__init__()
        self.__network_list = None
        self.__network_list_event_id = None
        self.__items = []
        self._network_list = network_list

    def get_item_children(self, item) -> List[_ChatHistoryItem]:
        """
        AbstractItemModel
        """
        if item is not None:
            return []
        return self.__items

    def get_item_value_model_count(self, item) -> int:
        """
        AbstractItemModel
        """
        return 1

    def get_item_value_model(self, item, column_id) -> ui.AbstractItemModel:
        """
        AbstractItemModel
        """
        if item:
            return item.name_model

    @property
    def _network_list(self) -> Optional[NetworkList]:
        return self.__network_list

    @_network_list.setter
    def _network_list(self, value):
        if self.__network_list is not None and self.__network_list_event_id is not None:
            self.__network_list.remove_event_fn(self.__network_list_event_id)

        if value is None:
            self.__network_list = None
            self.__network_list_event_id = None
        else:
            self.__network_list = value
            self.__network_list_event_id = self.__network_list.set_event_fn(
                lambda e, p, s=self.__weak(): s.__on_network_changed(e, p)
            )

    def __on_network_changed(self, event, payload):
        # TODO: Replace, not rebuild
        self.__rebuild_all()

    def __rebuild_all(self):
        for item in self.__items:
            item.destroy()

        if not self.__network_list:
            self.__items = []
        else:
            self.__items = [_ChatHistoryItem(n) for n in reversed(self.__network_list)]
        self._item_changed(None)

    def __weak(self):
        return weakref.proxy(self)


class _ChatHistoryDelegate(ui.AbstractItemDelegate):
    def build_branch(self, model, item, column_id, level, expanded):
        """Create a branch widget that opens or closes subtree"""
        pass

    def build_widget(self, model: _ChatHistoryModel, item, column_id, level, expanded):
        """Create a widget per column per item"""
        if item is None:
            return

        weak_model = weakref.proxy(model)
        weak_item = weakref.proxy(item)
        item_model: ui.SimpleStringModel = model.get_item_value_model(item, column_id)

        stack = ui.ZStack(height=40)
        with stack:
            highlight_background = ui.Rectangle(name="chat-history-highlight")
            highlight_stack = ui.HStack(visible=False)
            with highlight_stack:
                ui.Rectangle(name="chat-history-indicator", width=5)
                ui.Spacer()
                button = ui.Button(
                    name="chat-history-delete",
                    width=30,
                    clicked_fn=lambda item=weak_item, model=weak_model: remove_and_save(model, item._network),
                )
                ui.Spacer(width=10)
            with ui.HStack():
                ui.Spacer(width=10)
                with ui.VStack():
                    ui.Spacer()
                    ui.StringField(
                        item_model,
                        enabled=False,
                        style_type_name_override="Label",
                        height=0,
                        tooltip=item_model.as_string,
                        name="history_title",
                    )
                    ui.Spacer()
                ui.Spacer(width=40)

            def remove_and_save(model: _ChatHistoryModel, network: NetworkList):
                if isinstance(model._network_list, JsonNetworkList):
                    model._network_list.remove(network)
                    model._network_list.save()
                else:
                    model._network_list.delete(network)

            ui.Spacer(width=10)

        def on_hovered(hovered):
            highlight_stack.visible = hovered
            highlight_background.checked = hovered

        stack.set_mouse_hovered_fn(lambda hovered, button=button: on_hovered(hovered))


class ChatHistoryWidget:
    """
    ChatHistoryWidget behaves like ui.Widget.
    The main component is ui.TreeView which is presented as a list.
    It provides a way to navigate between different conversations.
    """

    def __init__(self, network_list=None, **kwargs):
        self.__selection_callbacks = {}
        self._frame = ui.Frame(build_fn=lambda s=self.__weak(): s.build_history(), **kwargs)
        self._model = _ChatHistoryModel(network_list)
        self._delegate = _ChatHistoryDelegate()

        self.__network_graph_window = None
        self._context_menu = None

    def destroy(self):
        if self.__network_graph_window:
            self.__network_graph_window.destroy()
            self.__network_graph_window = None

    def build_history(self):
        """
        Method to construct the entire ChatHistoryWidget UI.
        This includes building the header, body, and footer.
        """
        with ui.ZStack():
            ui.Rectangle(name="history")
            with ui.VStack():
                with ui.Frame(height=0):
                    self.build_history_header()
                self.build_history_body()
                with ui.Frame(height=0):
                    self.build_history_footer()

    def build_history_header(self):
        """
        Method to construct the header component of the ChatHistoryWidget UI.
        The header may contain controls for interaction with the chat history.
        """
        pass

    def build_history_body(self):
        """
        Method to construct the body component of the ChatHistoryWidget UI.
        The body contains the ui.TreeView, which represents the list of conversations.
        """
        with ui.ScrollingFrame(
            name="history",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        ):
            self._network_list_tree = ui.TreeView(
                self._model, delegate=self._delegate, root_visible=False, header_visible=False, name="history"
            )
            self._network_list_tree.set_selection_changed_fn(self.__on_model_selection_changed)
            self._network_list_tree.set_mouse_double_clicked_fn(
                partial(self.__on_tree_double_click, self._network_list_tree)
            )
            self._network_list_tree.set_mouse_pressed_fn(partial(self.__on_tree_context_menu, self._network_list_tree))

    def build_history_footer(self):
        """
        Method to construct the footer component of the ChatHistoryWidget UI.
        The footer may contain additional controls or information.
        """
        pass

    @property
    def network_list(self) -> NetworkList:
        """
        Accessor for the NetworkListModel instance used by this widget.
        """
        if self._model:
            return self._model._network_list

    @network_list.setter
    def network_list(self, value):
        if self._model:
            self._model._network_list = value
        else:
            self._model = _ChatHistoryModel(value)

    @property
    def selection(self) -> Optional[List[RunnableNetwork]]:
        """
        Accessor for the currently selected RunnableNetwork(s) in the ui.TreeView.
        """
        if self._network_list_tree:
            selected_networks = [item._network for item in self._network_list_tree.selection]
            return selected_networks

    def on_selection_changed(self, selection: List[RunnableNetwork]):
        """
        Event handler for changes in the selected conversation(s) in the ui.TreeView.
        """
        pass

    def set_selection_changed_fn(self, callable: Callable[[List[RunnableNetwork]], None]) -> int:
        """
        Method to add a callback to be executed when the selection changes.
        """
        callback_id = len(self.__selection_callbacks)
        self.__selection_callbacks[callback_id] = callable
        return callback_id

    def remove_selection_changed_fn(self, callback_id):
        """
        Removes the on_selection_changed_fn.

        Args:
            callback_id (int): The id from set_selection_changed_fn.
        """
        self.__selection_callbacks[callback_id] = None

    def __on_model_selection_changed(self, selection: List[_ChatHistoryItem]):
        selected_networks = [item._network for item in selection]
        self.__call_selection_changed(selected_networks)

        global network
        if selected_networks:
            network = selected_networks[0]

    def __call_selection_changed(self, selection: List[RunnableNetwork]):
        self.on_selection_changed(selection)
        for i, c in self.__selection_callbacks.items():
            if c:
                c(selection)

    def __on_tree_double_click(self, tree_view: ui.TreeView, x, y, button, modifiers):
        if button != 0:
            return

        selection = tree_view.selection
        if not selection:
            return

        self.__show_network_in_graph(selection[0]._network)

    def __show_network_in_graph(self, network: RunnableNetwork):
        """Opens the selected network in the graph window."""
        try:
            from .network_graph_window import NetworkGraphWindow
        except ImportError:
            return

        if self.__network_graph_window:
            self.__network_graph_window.destroy()
            self.__network_graph_window = None

        self.__network_graph_window = NetworkGraphWindow(network)

    def __open_profiling_in_browser(self, network: RunnableNetwork):
        """Opens the network profiling in the browser."""
        if not HAS_PROFILING_HTML:
            print("Profiling HTML creation not available - missing lc_agent.utils.profiling_html")
            return

        if not hasattr(network, "profiling") or not network.profiling:
            print(f"No profiling data available for network: {network.metadata.get('name', 'Unknown')}")
            return

        try:
            # Create a temporary HTML file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp_file:
                html_path = tmp_file.name

            # Generate the profiling HTML
            html_path = create_profiling_html(network, html_path)

            # Open in browser
            webbrowser.open(f"file://{html_path}")
            print(f"Opened profiling for network: {network.metadata.get('name', 'Unknown')}")

        except Exception as e:
            print(f"Error creating profiling HTML: {e}")

    def __has_profiling_data(self, network: RunnableNetwork) -> bool:
        """Check if profiling data is available for the network."""
        if not HAS_PROFILING_HTML:
            return False
        return hasattr(network, "profiling") and network.profiling

    def __get_selected_network(self):
        """Get the currently selected network from the tree view."""
        if hasattr(self, "_network_list_tree") and self._network_list_tree:
            selection = self._network_list_tree.selection
            if selection:
                return selection[0]._network
        return None

    def __on_show_in_graph_triggered(self):
        """Handler for 'View in Network Graph' menu item."""
        network = self.__get_selected_network()
        if network:
            self.__show_network_in_graph(network)

    def __on_open_profiling_triggered(self):
        """Handler for 'Open Performance Profiling' menu item."""
        network = self.__get_selected_network()
        if network:
            self.__open_profiling_in_browser(network)

    def __on_tree_context_menu(self, tree_view: ui.TreeView, x, y, button, modifiers):
        """Shows context menu on right mouse button click."""
        # Only show context menu on right mouse button click
        if button != 1:
            return

        # Check if there's a selection
        selection = tree_view.selection
        if not selection:
            return

        # Get the selected network for checking profiling availability
        selected_network = selection[0]._network

        # Clear and rebuild the context menu
        if self._context_menu is None:
            self._context_menu = ui.Menu()

        self._context_menu.clear()
        with self._context_menu:
            # Menu item to show network in graph
            show_in_graph_item = ui.MenuItem("View in Network Graph", triggered_fn=self.__on_show_in_graph_triggered)

            # Menu item to open profiling
            # Gray out if profiling is not available
            has_profiling = self.__has_profiling_data(selected_network)
            if has_profiling:
                profiling_item = ui.MenuItem(
                    "Open Performance Profiling",
                    triggered_fn=self.__on_open_profiling_triggered,
                )

            # You can easily add more menu items here in the future
            # ui.Separator()
            # ui.MenuItem("Export Network", enabled=False)  # Example of disabled item

        # Show the context menu at mouse position
        self._context_menu.show()

    def __weak(self):
        return weakref.proxy(self)
