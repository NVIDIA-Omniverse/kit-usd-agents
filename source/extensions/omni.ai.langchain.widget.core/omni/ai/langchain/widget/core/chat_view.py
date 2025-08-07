## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import asyncio
import inspect
import weakref
from typing import Any, Dict, List, Optional

import carb.settings
import omni.ui as ui
from lc_agent import NetworkNode, RunnableNetwork, RunnableNode, get_chat_model_registry, get_node_factory

from .agent_delegate import AgentDelegate, DefaultDelegate

CHAT_MODEL_SETTING = "/persistent/exts/omni.ai.langchain.widget.core/chat_model"
NODE_NAME_SETTING = "/persistent/exts/omni.ai.langchain.widget.core/node_name"


def get_node_names():
    all_names = get_node_factory().get_registered_node_names(hidden=False)

    result = []
    for name in all_names:
        # Let's not expose the UserAgent
        if name == "UserAgent":
            continue
        agent_type = get_node_factory().get_registered_node_type(name)
        if issubclass(agent_type, RunnableNode):
            result.append(name)
    return result


def get_persistent_node_name():
    return carb.settings.get_settings().get(NODE_NAME_SETTING)


def get_persistent_chat_model_name():
    return carb.settings.get_settings().get(CHAT_MODEL_SETTING)


class _ChatItem(ui.AbstractItem):
    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        self._name_model = ui.SimpleStringModel()

    @property
    def name_model(self):
        self._name_model.as_string = str(self._agent.outputs.content)
        return self._name_model


class _ChatModel(ui.AbstractItemModel):
    def __init__(self, network):
        super().__init__()
        self._network = network
        if self._network:
            self._event_id = self._network.set_event_fn(self._on_network_event)
        else:
            self._event_id = None
        self.__items = []
        self._refresh_items(self._network, self.__items)

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self._event_id is not None and self._network is not None:
            self._network.remove_event_fn(self._event_id)
            self._network = None
            self._event_id = None

    def get_item_children(self, item) -> List[_ChatItem]:
        if item is not None:
            return []
        return self.__items

    def get_item_value_model_count(self, item) -> int:
        return 1

    def get_item_value_model(self, item, column_id) -> "Model":
        if item:
            return item.name_model

    def _on_network_event(self, event_type, payload):
        """
        React to events triggered in the agent network. This method
        specifically listens for the addition or removal of agents
        and updates the internal items list accordingly.

        Args:
            event_type (enum): Type of the event triggered.
            payload (dict): Additional data related to the event.
        """

        if event_type in [
            RunnableNetwork.Event.NODE_ADDED,
            RunnableNetwork.Event.CONNECTION_REMOVED,
            RunnableNetwork.Event.CONNECTION_ADDED,
        ]:
            # Refresh the items list when a new agent is added
            self._refresh_items(self._network, self.__items)
            self._item_changed(None)

        elif event_type == RunnableNetwork.Event.NODE_REMOVED:
            # Remove the item associated with the removed agent
            self._refresh_items(self._network, self.__items)
            self._item_changed(None)

    def _refresh_items(self, network: RunnableNetwork, items):
        """
        Update the items list based on the agents present in the network.

        Args:
            payload (dict): Information about the newly added agent and the network.
        """

        # Create a mapping of existing agents to items
        old_items = {item._agent: item for item in items}

        # Refresh the items list
        new_items_list = []
        if network is None:
            return

        for agent in network.get_sorted_nodes():
            if agent not in network:
                continue

            if agent.metadata.get("contribute_to_ui", True) is False:
                continue

            # Get the item for the agent or create a new one if it doesn't exist
            item = old_items.get(agent) or _ChatItem(agent)
            new_items_list.append(item)

        items[:] = new_items_list

    def _find_item(self, agent):
        # Recursively search for the item associated with the agent
        return next((i for i in self.__items if i._agent is agent), None)


class _ChatTreeViewDelegate(ui.AbstractItemDelegate):
    def __init__(self, view: "ChatView"):
        super().__init__()
        self._chat_view = view

    def build_widget(self, model, item, column_id, level, expanded):
        if item is None:
            return

        self._chat_view.build_agent_widget(item._agent)
        self._chat_view.build_agent_thread(item._agent)

    def build_branch(self, model, item, column_id, level, expanded):
        pass


class ChatView:
    _registered_delegates = {}

    def __init__(self, network=None, **kwargs):
        """
        Initializes ChatView, behaving like a ui.Widget.
        """
        weakself = self.__weak()
        self._frame = ui.Frame(build_fn=lambda s=weakself: s.build_view(), **kwargs)
        self._frame_stack = None
        self._tree_model = None
        self._tree_model_subscription = None
        self._network = None
        self._delegate = _ChatTreeViewDelegate(weakself)
        self._event_id = None
        self._agent_widget_data: Dict[str, Any] = {}
        self._is_startup: Optional[bool] = None
        self.delegate = None
        self._agent_selection_changed_fn = None

        if network:
            self.network = network

    def destroy(self):
        """
        Destroys the ChatView instance and cleans up resources.
        """
        self.network = None
        if self._frame:
            self._frame.destroy()
        self._frame = None
        if self._tree_model:
            self._tree_model.destroy()
            self._tree_model = None
        self._tree_model_subscription = None
        self._agent_widget_data = {}
        self.delegate = None
        self._agent_selection_changed_fn = None

    @property
    def network(self):
        """
        RW. Returns the RunnableNetwork associated with this ChatView.
        """
        return self._network

    @network.setter
    def network(self, value):
        if self._network is not None and self._event_id is not None:
            self._network.remove_event_fn(self._event_id)
            self._event_id = None

        self._network = value

        if self._network:
            self._event_id = self._network.set_event_fn(self._on_network_event)

        if self._frame:
            self._frame.rebuild()

    @classmethod
    def add_delegate(cls, name: str, delegate: AgentDelegate):
        """
        Registers a delegate for a specific agent type. It's a class level to
        let register when the UI is not created. It considers inheritance.

        If the delegate is registered in ChatView, it will be available to all
        the classes inherited from ChatView. It's also possible to call it in
        the class that is inherited. This class will be able to decide if it
        heeds the delegates from the base class. So the delegate is available
        acoss the application.

        Args:
            name (str): The agent type name.
            delegate (AgentDelegate): The delegate to handle the agent type.
        """
        if name in cls._registered_delegates:
            cls._registered_delegates[name].append(delegate)
        else:
            cls._registered_delegates[name] = [delegate]

    @classmethod
    def remove_delegate(cls, name: str):
        """
        Removes the delegate.

        Args:
            name (str): The agent type name.
        """
        if name in cls._registered_delegates and cls._registered_delegates[name]:
            cls._registered_delegates[name].pop()
            if not cls._registered_delegates[name]:
                cls._registered_delegates.pop(name)

    @property
    def current_agent_name(self) -> str:
        """
        Returns the name of the currently active agent. Alias for network.
        """
        pass

    @property
    def agent_selection_changed_fn(self):
        return self._agent_selection_changed_fn

    def build_view(self):
        """
        Called to build the ChatView UI, including header, body, and footer.
        """
        if self._frame_stack:
            self._frame_stack.destroy()
        self._frame_stack = ui.ZStack()

        with self._frame_stack:
            with ui.VStack():
                with ui.Frame(height=0):
                    self.build_view_header()
                with ui.Frame():
                    self.build_view_body()
                with ui.Frame(height=0):
                    self.build_view_footer()

    def build_view_header(self):
        """
        Called to build the ChatView header UI component.
        """
        pass

    def build_view_body(self):
        """
        Called to build the ChatView body UI component, including agent messages.
        """
        self._is_startup = not self._network.nodes
        if self._is_startup:
            with ui.ScrollingFrame():
                self.build_startup()
            return

        if self._tree_model:
            self._tree_model.destroy()

        self._tree_model = self._create_tree_model()

        with ui.VStack():
            ui.Spacer(height=10)
            with ui.ScrollingFrame():
                self._tree_view = ui.TreeView(
                    self._tree_model, delegate=self._delegate, root_visible=False, header_visible=False, name="chat"
                )

    def build_view_footer(self):
        """
        Called to build the ChatView footer UI component, including the request field.
        """
        # Request is in the footer
        self.build_request_widget()

    def build_agent_widget(self, agent: RunnableNode):
        """
        Called to build the UI for a specific agent. It should call the
        appropriate delegate. The developer can overrider it for customization.

        Args:
            agent (RunnableNode): The agent for which to build the UI.
        """
        delegate = self._find_delegate(agent)
        if delegate:
            data = delegate.build_agent_widget(self._network, agent)
            # Keep it
            self._agent_widget_data[agent.uuid()] = data

    def build_agent_thread(self, agent: RunnableNode):
        """
        Called to build the UI for the thread line between the agents.

        Args:
            agent (RunnableNode): The agent for which to build the thread line.
        """
        pass

    def build_request_widget(self):
        """
        Called to build the request UI component (string field on the bottom)
        using the delegate for the current agent.
        """
        if not self._network:
            return

        if not self.delegate:
            self.delegate = DefaultDelegate()
        self.delegate.build_request_widget(self._network)

    def build_startup(self):
        """
        Called to build the startup screen when the model is empty.
        """
        agent_names = get_node_names()
        agent_cores = get_chat_model_registry().get_registered_names()

        default_node_name = get_persistent_node_name()
        default_chat_model_name = get_persistent_chat_model_name()

        default_node_name = next((i for i, x in enumerate(agent_names) if x == default_node_name), 0)
        default_chat_model_name = next((i for i, x in enumerate(agent_cores) if x == default_chat_model_name), 0)

        def build_examples(examples):
            # TODO: Looks like it's for a delegate
            if len(examples) == 0:
                with ui.VStack():
                    ui.Spacer(height=30)
                    ui.Label("This RunnableNode doesn't have examples", height=0, name="new_conversation")
                    ui.Spacer(height=30)
            else:
                # building the example conversation stack
                example_conversation = ui.VStack()
                with example_conversation:
                    ui.Spacer(height=30)
                    ui.Label("Examples", height=0, name="new_conversation")
                    ui.Spacer(height=30)

                    with ui.VStack():
                        for example in examples:
                            with ui.HStack(height=46):
                                ui.Spacer()
                                ui.Button(
                                    example,
                                    height=40,
                                    width=0,
                                    name="example_conversation",
                                    clicked_fn=lambda example=example: self._example_conversation_clicked(example),
                                )
                                ui.Spacer()
                            ui.Spacer(height=10)

        def agent_changed(node_name, description_frame, examples_frame, network):
            dummy = get_node_factory().create_node(node_name)
            description = dummy.metadata.get("description", "")
            examples = dummy.metadata.get("examples", [])
            network.default_node = node_name
            description_frame.set_build_fn(lambda d=description: ui.Label(d, name="agent_desp", word_wrap=True))
            examples_frame.set_build_fn(lambda e=examples: build_examples(examples))
            # Save persistent
            carb.settings.get_settings().set(NODE_NAME_SETTING, node_name)

        def core_changed(chat_model_name, network):
            network.chat_model_name = chat_model_name
            for delegate_name in self._registered_delegates:
                for delegate in self._registered_delegates[delegate_name]:
                    delegate.on_core_changed(chat_model_name, network)
            # Save persistent
            carb.settings.get_settings().set(CHAT_MODEL_SETTING, chat_model_name)

        with ui.VStack():
            ui.Spacer(height=50)

            ui.Label("NEW CHAT", height=0, name="new_conversation")
            ui.Spacer(height=32)

            with ui.VStack(height=0):
                agent_desciption = ui.Frame(height=0)

                ui.Spacer(height=28)
                with ui.HStack():
                    ui.Spacer()
                    with ui.HStack(width=350):
                        ui.Label("Chat Model:", width=100, name="model_name")
                        combo_box_items = agent_cores or ["No Chat Model Loaded"]
                        enabled = bool(agent_cores)
                        with ui.ZStack():
                            model_cores = ui.ComboBox(
                                default_chat_model_name, *combo_box_items, name="model-combo", enabled=enabled
                            ).model
                            with ui.VStack():
                                ui.Spacer(height=3)
                                with ui.HStack():
                                    ui.Spacer()
                                    with ui.ZStack(width=20, height=20):
                                        ui.Rectangle()  # hide the combo arrow button
                                        ui.Image(name="combo_button")
                                ui.Spacer(height=3)
                    ui.Spacer()
                ui.Spacer(height=10)

            example_conversation_frame = ui.Frame(height=0)

        self._agent_selection_changed_fn = lambda name, d=agent_desciption, e=example_conversation_frame, agent_names=agent_names, s=self: agent_changed(
            name, d, e, s._network
        )
        model_cores.add_item_changed_fn(
            lambda m, i, agent_cores=agent_cores, s=self: core_changed(
                agent_cores[m.get_item_value_model().as_int], s._network
            )
        )
        if agent_names:
            agent_changed(agent_names[default_node_name], agent_desciption, example_conversation_frame, self._network)
        if agent_cores:
            core_changed(agent_cores[default_chat_model_name], self._network)

    def _create_tree_model(self):
        return _ChatModel(self._network)

    def _get_delegate(self, agent_name: str):
        return self._registered_delegates.get(agent_name, [None])[-1]

    def _find_delegate(self, agent: RunnableNode):
        for agent_type in inspect.getmro(agent.__class__):
            delegate = self._get_delegate(agent_type.__name__)
            if delegate:
                return delegate

            if agent_type is RunnableNode:
                # We reached the base class
                # TODO: Exception
                break

    def _find_parent_node(self, node: RunnableNode, network: RunnableNetwork):
        for current_node in network.nodes:
            if current_node.uuid() == node.uuid():
                return current_node
            if isinstance(current_node, NetworkNode):
                subnode = self._find_parent_node(node, current_node)
                if subnode:
                    # Found the node in the subnetwork
                    return current_node

    def _find_data(self, node: RunnableNode, network: RunnableNetwork):
        data = self._agent_widget_data.get(node.uuid(), None)
        if not data:
            node = self._find_parent_node(node, network)

            # If the node is not found, we will update the leaf node
            if not node:
                node = network.get_leaf_node()

            if node:
                data = self._agent_widget_data.get(node.uuid(), None)
        return data, node

    def _on_network_event(self, event_type, payload):
        # React to an event in the network, such as an node being added
        if event_type == RunnableNetwork.Event.NODE_ADDED:
            self._on_agent_added()
        elif event_type == RunnableNetwork.Event.NODE_REMOVED:
            node = payload["node"]
            if node:
                self._agent_widget_data.pop(node.uuid(), None)
        elif event_type == RunnableNetwork.Event.NODE_INVOKED:
            # This is the way to update the node widgets without rebuilding
            node: RunnableNode = payload["node"]
            network: RunnableNetwork = payload.get("network") or self._network
            data, node = self._find_data(node, network)
            delegate = self._find_delegate(node)
            if delegate and delegate.need_rebuild_agent_widget(self._network, node, data):
                if self._tree_model:
                    item = self._tree_model._find_item(node)
                    self._tree_model._item_changed(item)

    def _on_agent_added(self):
        """
        It's called when startup screen is on and an agent is added to the
        network
        """
        if self._is_startup:
            self._frame.rebuild()

            if len(self._network.nodes) == 1:
                asyncio.ensure_future(self._name_network(self._network))

    async def _name_network(self, network):
        # Get the first agent
        agent = network.nodes[0]
        # Get the request to ask AI how to name it
        request = (
            "Summerize a short name for the question in 3 words \n\n"
            + agent.outputs.content
            + "\n\n The answer just contains the short name"
        )

        # Create a temporary network to get the name of the conversation
        temporary_network = RunnableNetwork(
            default_node="Assistant", chat_model_name=network.chat_model_name, name="Name Network"
        )
        temporary_network.add_node(get_node_factory().create_node("UserAgent", request))

        network_name = ""
        async for chunk in temporary_network.astream():
            if len(network_name) < 50:
                network_name += chunk.content
            else:
                break

        network.metadata["name"] = network_name
        network._event_callback(RunnableNetwork.Event.METADATA_CHANGED, {"key": "name", "value": network_name})

    def _example_conversation_clicked(self, example):
        if not self._network:
            return
        if not self.delegate:
            self.delegate = DefaultDelegate()
        self.delegate._submit_prompt(self._network, example)

    def __weak(self):
        return weakref.proxy(self)
