## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .default_modifier import DefaultModifier
from .network_modifier import NetworkModifier
from .runnable_node import AINodeMessageChunk
from .runnable_node import RunnableNode
from .utils.profiling_utils import ProfilingData, Profiler
from .utils.pydantic import PrivateAttr
from .uuid_utils import UUIDMixin
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.utils import Input, Output
from pydantic import model_serializer
from pydantic import model_validator
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type, Union
from typing_extensions import Self
import contextvars
import enum
import time


_active_networks_var = contextvars.ContextVar("_active_networks")


class RunnableNetwork(RunnableSerializable[Input, Output], UUIDMixin):
    nodes: List[RunnableNode] = []
    modifiers: Dict = {}
    callbacks: Dict = {}

    default_node: str = ""
    chat_model_name: Optional[str] = None
    metadata: Dict[str, Any] = {}
    profiling: Optional[ProfilingData] = None

    verbose: bool = False

    _current_modifier_name: Optional[str] = PrivateAttr(None)
    _node_set: set = PrivateAttr(set())

    class ParentMode(enum.Enum):
        NONE = 0
        LEAF = 1

    class Event(enum.Enum):
        ALL = 0
        NODE_ADDED = 1
        NODE_INVOKED = 2
        NODE_REMOVED = 3
        CONNECTION_ADDED = 4
        CONNECTION_REMOVED = 5
        METADATA_CHANGED = 6
        # ETC...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.uuid()
        self.add_modifier(DefaultModifier())

    def _iter(self, *args, **kwargs):
        """Pydantic serialization method"""
        # No modifiers and callbacks
        kwargs["exclude"] = (kwargs.get("exclude", None) or set()) | {
            "modifiers",
            "callbacks",
        }

        # Call super
        yield from super()._iter(*args, **kwargs)

        # Add extra data about connections
        yield "__connections__", {
            i: [self._get_node_id(parent) for parent in node.parents if self._get_node_id(parent) is not None]
            for i, node in enumerate(self.nodes)
        }

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except modifiers and callbacks
        result = {}
        for field_name, field_value in self:
            if field_name not in ["modifiers", "callbacks", "parents"]:
                result[field_name] = field_value

        # Add connections information
        result["__connections__"] = {
            i: [self._get_node_id(parent) for parent in node.parents if self._get_node_id(parent) is not None]
            for i, node in enumerate(self.nodes)
        }

        return result

    def __restore_node_set(self):
        self._node_set = set(self.nodes)

    def __contains__(self, node: RunnableNode):
        return node in self._node_set

    @classmethod
    def parse_obj(cls: Type["RunnableNetwork"], obj: Any) -> "RunnableNetwork":
        """Pydantic deserialization method"""
        connections = obj.pop("__connections__", {})
        nodes = obj.pop("nodes", [])

        network = super(RunnableNetwork, cls).parse_obj(obj)

        # Restore nodes
        nodes = [RunnableNode.parse_obj(node) for node in nodes]

        # Restore connections
        for node_id, parent_ids in connections.items():
            if isinstance(node_id, str):
                node_id = int(node_id)
            node = nodes[node_id]

            parents = []
            for parent_id in parent_ids:
                parent = nodes[parent_id]
                parents.append(parent)

            network.add_node(node, parents)

        return network

    @model_validator(mode="wrap")
    @classmethod
    def deserialize(cls, data: Any, handler) -> Self:
        """Pydantic v2 deserialization method using model_validator with wrap mode"""
        # Handle dictionary input
        if isinstance(data, dict):
            # Make a copy to avoid modifying the original
            data_copy = data.copy()

            # Extract connections and nodes
            connections = data_copy.pop("__connections__", {})
            nodes = data_copy.pop("nodes", [])

            # Create the network instance
            network = handler(data_copy)

            # Restore nodes using deserialize
            deserialized_nodes = [RunnableNode.model_validate(node) for node in nodes]

            # Restore connections
            for node_id, parent_ids in connections.items():
                if isinstance(node_id, str):
                    node_id = int(node_id)
                node = deserialized_nodes[node_id]

                parents = []
                for parent_id in parent_ids:
                    parent = deserialized_nodes[parent_id]
                    parents.append(parent)

                network.add_node(node, parents)

            return network

        # Handle case where obj is already a RunnableNetwork instance
        if isinstance(data, RunnableNetwork):
            return data

        # For other types, use the default handler
        return handler(data)

    @classmethod
    def get_active_network(cls):
        stack = _active_networks_var.get(None)
        if stack:
            return stack[-1]
        return None

    @classmethod
    def get_active_networks(cls):
        """Yields active networks from most recent to oldest."""
        stack = _active_networks_var.get(None)
        if stack:
            for network in reversed(stack):
                yield network

    def invoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        if len(self.nodes) != len(self._node_set):
            self.__restore_node_set()

        with self:
            return self._invoke(input, config, **kwargs)

    async def ainvoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        if len(self.nodes) != len(self._node_set):
            self.__restore_node_set()

        with self:
            return await self._ainvoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        if len(self.nodes) != len(self._node_set):
            self.__restore_node_set()

        async with self:
            async for item in self._astream(input, config, **kwargs):
                yield item

    def add_node(
        self,
        node: RunnableNode,
        parent: Optional[Union[RunnableNode, List[RunnableNode], ParentMode]] = ParentMode.LEAF,
    ):
        """
        Adds an node to the network with an optional parent.

        node1 = Mynode("Hello, how can I help you?")
        node2 = Mynode("What's the weather like today?")
        node3 = Mynode("You have access to the internet.")
        network.add_node(node1)
        network.add_node(node2, parent=node1)
        network.add_node(node3, parent=node2)

        Args:
            node (node): The node to add.
            parent (Union[node, ParentMode]): The parent node. Default is the
                       first available leaf. It's possible to specify the node.
        """
        if parent is RunnableNetwork.ParentMode.LEAF:
            leafs = self.get_leaf_nodes()
            parent = leafs[-1] if leafs else None

        # Save the information about the modifier that added the node
        if self._current_modifier_name:
            if "modifier_info" not in node.metadata:
                node.metadata["modifier_info"] = {}
            node.metadata["modifier_info"]["added_by"] = self._current_modifier_name

        node.on_before_node_added(self)

        node._clear_parents()

        self.nodes.append(node)
        self._node_set.add(node)
        if len(self.nodes) != len(self._node_set):
            self.__restore_node_set()

        if isinstance(parent, RunnableNode):
            node._add_parent(parent)
        elif isinstance(parent, list):
            for p in parent:
                node._add_parent(p)

        self._event_callback(RunnableNetwork.Event.NODE_ADDED, {"node": node, "network": self})

        node.on_node_added(self)

        return node

    def remove_node(self, node: RunnableNode):
        """
        Removes a node from the network.

        Args:
            node (RunnableNode): The node to remove.
        """
        parents = self.get_parents(node)
        children = self.get_children(node)
        for child in children:
            child_parents = child.parents[:]
            child._clear_parents()
            child_parents.remove(node)
            child_parents.extend(parents)
            for new_parent in child_parents:
                child._add_parent(new_parent)

        node.on_before_node_removed(self)

        self.nodes.remove(node)
        self._node_set.remove(node)
        if len(self.nodes) != len(self._node_set):
            self.__restore_node_set()

        node.on_node_removed(self)
        self._event_callback(RunnableNetwork.Event.NODE_REMOVED, {"node": node, "network": self})

    def get_parents(self, node: RunnableNode) -> List[RunnableNode]:
        return node.parents[:]

    def get_children(self, node: RunnableNode) -> List[RunnableNode]:
        result = []

        for a in self.nodes:
            if a is node:
                continue

            if node in a.parents:
                result.append(a)

        return result

    def get_all_parents(self, node: RunnableNode) -> List[RunnableNode]:
        all_parents = []
        to_visit = [node]

        # While there are still nodes to visit
        while to_visit:
            # Get the next node to visit
            next_node = to_visit.pop()

            # Get the parents of the current node
            parents = self.get_parents(next_node)

            # Add the parents to the list of parents to return
            all_parents.extend(parents)

            # Add the parents to the list of nodes to visit next
            to_visit.extend(parents)

        return all_parents

    def get_all_children(self, node: RunnableNode) -> List[RunnableNode]:
        all_children = []
        visited_nodes = set()

        def explore_children(current_node):
            # Check if we've already visited this node to prevent cycles
            if current_node in visited_nodes:
                return
            visited_nodes.add(current_node)

            for child in self.get_children(current_node):
                all_children.append(child)
                explore_children(child)

        explore_children(node)
        return all_children

    def get_root_nodes(self) -> List[RunnableNode]:
        """
        Gets the root nodes in the network. Root nodes are those with no parents.

        Returns:
            List[node]: The root nodes in the network.
        """
        return [node for node in self.nodes if not self.get_parents(node)]

    def get_leaf_nodes(self, unevaluated_only: bool = False) -> List[RunnableNode]:
        """
        Gets the leaf nodes in the network.

        Args:
            unevaluated_only: When True, it returns only unevaluated nodes

        Returns:
            List[RunnableNode]: The leaf nodes in the network.
        """
        parent_nodes = set()

        # Populate the parent_nodes set
        for node in self.nodes:
            for parent in node.parents:
                parent_nodes.add(parent)

        # Leaf nodes are nodes not in parent_nodes, preserving the order
        leaf_nodes = [
            node for node in self.nodes if node not in parent_nodes and (not unevaluated_only or not node.is_evaluated)
        ]

        return leaf_nodes

    def get_leaf_node(self) -> RunnableNode:
        """
        First available node. It throws an exception if several nodes are
        leafs.
        """
        leafs = self.get_leaf_nodes()
        if not leafs:
            return None
        elif len(leafs) == 0:
            return leafs[0]

        return leafs[0]

    def get_sorted_nodes(self) -> List[RunnableNode]:
        """
        Get nodes sorted such that every node is placed after its parent.

        Returns:
            List[node]: The sorted list of nodes.
        """
        visited = set()
        sorted_nodes = []

        def dfs(node: RunnableNode):
            """Depth First Search to get nodes in order."""
            if node in visited:
                return

            visited.add(node)
            for child in self.get_parents(node):
                # we only visit nodes that are in the network
                if child in self:
                    dfs(child)

            sorted_nodes.append(node)

        leafs = self.get_leaf_nodes()
        for node in leafs:
            dfs(node)

        return sorted_nodes

    def set_event_fn(
        self,
        callable: Callable[["Event", "Payload"], None],
        event: "Event" = Event.ALL,
        priority: int = 100,
    ) -> int:
        """
        Adds a callback to the network event like: node added/removed,
        connection added/removed, etc...

        Args:
            callable: The callable that will be called on event.
            event: The event to subscribe.
            priority (int): Used to order the process.

        Returns:
            int: id to be able to remove it
        """
        event_id = len(self.callbacks)
        self.callbacks[event_id] = (callable, event, priority)
        return event_id

    def remove_event_fn(self, event_id: int):
        """
        Removes the callback.

        Args:
            event_id (int): The id from set_event_fn.
        """
        self.callbacks[event_id] = None

    def _event_callback(self, event: "RunnableNetwork.Event", payload: Dict[str, Any]):
        for _, callback_data in self.callbacks.items():
            if callback_data:
                callable, callable_event, callable_priority = callback_data
                if callable_event == event or callable_event == RunnableNetwork.Event.ALL:
                    callable(event, payload)

    def add_modifier(self, modifier: NetworkModifier, once: bool = False, priority: Optional[int] = None) -> int:
        """
        Ads a modifier and subscribes it to invokeing.

        Args:
            modifier (NetworkModifier): The modifier to add.
            priority (int): Used to order the invoke.

        Returns:
            int: id to be able to remove it
        """
        if once:
            for modifier_id, modifier_to_check in self.modifiers.items():
                if type(modifier) is type(modifier_to_check):
                    return modifier_id

        modifier_id = len(self.modifiers) if priority is None else priority
        self.modifiers[modifier_id] = modifier
        return modifier_id

    def remove_modifier(self, modifier_id: int):
        """
        Removes the callback.

        Args:
            modifier_id (int): The id from add_modifier.
        """
        self.modifiers[modifier_id] = None

    def get_modifier_id(self, modifier: Union[Type, NetworkModifier]) -> Optional[int]:
        """
        Checks if the network has modifier registered and returns its id

        Args:
            modifier: Modifier to check
        """

        if isinstance(modifier, type):
            for modifier_id, modifier_to_check in self.modifiers.items():
                if type(modifier_to_check) is modifier:
                    return modifier_id
        else:
            for modifier_id, modifier_to_check in self.modifiers.items():
                if modifier_to_check == modifier:
                    return modifier_id

    def _invoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        result = None
        node = None

        self._modifier_begin_invoke()

        while True:
            nodes = self.get_sorted_nodes()
            invoked = False
            for node in nodes:
                if node.invoked:
                    continue

                self._modifier_pre_invoke(node)

                # On the case the node is removed
                if node not in self:
                    continue

                result = node.invoke(input, config, **kwargs)

                self._modifier_post_invoke(node)

                invoked = True
                self._event_callback(RunnableNetwork.Event.NODE_INVOKED, {"node": node, "network": self})

            if not invoked:
                if not result and node:
                    result = node.outputs
                break

        self._modifier_end_invoke()

        return result

    async def _ainvoke(
        self,
        input: Dict[str, Any] = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ):
        result = None
        node = None

        await self._modifier_begin_invoke_async()

        while True:
            nodes = self.get_sorted_nodes()
            invoked = False
            for node in nodes:
                if node.invoked:
                    continue

                await self._modifier_pre_invoke_async(node)

                # On the case the node is removed
                if node not in self:
                    continue

                result = await node.ainvoke(input, config, **kwargs)

                await self._modifier_post_invoke_async(node)

                invoked = True
                self._event_callback(RunnableNetwork.Event.NODE_INVOKED, {"node": node, "network": self})

            if not invoked:
                if not result and node:
                    result = node.outputs
                break

        await self._modifier_end_invoke_async()

        return result

    async def _astream(
        self,
        input: Input = {},
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        with Profiler("network_astream", "network", network=self, network_id=self.uuid()):
            result = None
            node = None

            await self._modifier_begin_invoke_async()

            while True:
                nodes = self.get_sorted_nodes()
                invoked = False
                for node in nodes:
                    if node.invoked:
                        continue

                    with Profiler(
                        f"node_stream_{node.__class__.__name__}",
                        "node",
                        network=self,
                        node_id=node.uuid(),
                        node_name=node.name or node.__class__.__name__,
                    ):
                        await self._modifier_pre_invoke_async(node)

                        # On the case the node is removed
                        if node not in self:
                            continue

                        # Determine if we should profile chunks (skip for NetworkNode)
                        from .network_node import NetworkNode

                        should_profile_chunks = not isinstance(node, NetworkNode)

                        current_node = None
                        chunk_count = 0
                        chunk_profiler = None

                        try:
                            # Start chunk profiling if needed
                            if should_profile_chunks:
                                chunk_profiler = Profiler(
                                    f"chunk_{chunk_count}",
                                    "chunk",
                                    network=self,
                                    chunk_index=chunk_count,
                                    node_id=node.uuid(),
                                )
                                chunk_profiler.start()

                            async for result in node.astream(input, config, **kwargs):
                                # Handle chunk profiling
                                if should_profile_chunks and chunk_profiler:
                                    # Stop current chunk timer
                                    chunk_profiler.stop()

                                    # Update metadata with chunk content
                                    if isinstance(result, BaseMessage):
                                        chunk_profiler.update_metadata(content=str(result.content))

                                    # Start next chunk timer
                                    chunk_count += 1
                                    chunk_profiler = Profiler(
                                        f"chunk_{chunk_count}",
                                        "chunk",
                                        network=self,
                                        chunk_index=chunk_count,
                                        node_id=node.uuid(),
                                    )
                                    chunk_profiler.start()

                                # Handle result
                                if isinstance(result, AINodeMessageChunk):
                                    current_node = result.node
                                else:
                                    current_node = node
                                self._event_callback(
                                    RunnableNetwork.Event.NODE_INVOKED,
                                    {"node": current_node, "network": self},
                                )
                                yield result
                        finally:
                            # Always stop final chunk profiler if needed
                            if should_profile_chunks and chunk_profiler:
                                chunk_profiler.stop()

                        await self._modifier_post_invoke_async(node)

                        if current_node is not None:
                            self._event_callback(
                                RunnableNetwork.Event.NODE_INVOKED,
                                {"node": current_node, "network": self},
                            )

                    invoked = True

                if not invoked:
                    if not result and node:
                        result = node.outputs
                    break

            await self._modifier_end_invoke_async()

    def _get_node_id(self, node: RunnableNode):
        """
        Gets the index of the node in the network using identity comparison.
        """
        for i, n in enumerate(self.nodes):
            if n is node:
                return i
        return None

    def _iterate_modifiers(self):
        """
        Yields each modifier in the _modifiers dictionary.

        If new modifiers are added to _modifiers during iteration, they will also
        be yielded in subsequent loops.
        """

        visited = set()

        while True:
            # Collect the keys that haven't been invokeed yet
            uninvoked_keys = [key for key in sorted(self.modifiers.keys()) if key not in visited]

            # If no more uninvokeed keys are found, stop the generator
            if not uninvoked_keys:
                # Using return in a generator will raise StopIteration, ending
                # the iteration
                return

            # Mark these keys as visited
            visited.update(uninvoked_keys)

            # Yield each uninvokeed modifier
            for key in uninvoked_keys:
                modifier = self.modifiers.get(key)
                if modifier:
                    yield modifier

    def _modifier_begin_invoke(self):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            modifier.on_begin_invoke(self)

            self._current_modifier_name = None

    def _modifier_pre_invoke(self, node: RunnableNode):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            with Profiler(
                f"pre_invoke_{modifier.__class__.__name__}",
                "modifier",
                network=self,
                node_id=node.uuid(),
                node_name=node.name or node.__class__.__name__,
                modifier_name=modifier.__class__.__name__,
            ):
                start_time = time.time()

                modifier.on_pre_invoke(self, node)

                elapsed_time = time.time() - start_time
                if elapsed_time > 0.001:
                    self._set_modifier_info_metadata(
                        node, str(self._current_modifier_name), "on_pre_invoke_time", elapsed_time
                    )

            self._current_modifier_name = None

    def _modifier_post_invoke(self, node: RunnableNode):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            with Profiler(
                f"post_invoke_{modifier.__class__.__name__}",
                "modifier",
                network=self,
                node_id=node.uuid(),
                node_name=node.name or node.__class__.__name__,
                modifier_name=modifier.__class__.__name__,
            ):
                start_time = time.time()

                modifier.on_post_invoke(self, node)

                elapsed_time = time.time() - start_time
                if elapsed_time > 0.001:
                    self._set_modifier_info_metadata(
                        node, str(self._current_modifier_name), "on_post_invoke_time", elapsed_time
                    )

            self._current_modifier_name = None

    def _modifier_end_invoke(self):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            modifier.on_end_invoke(self)

            self._current_modifier_name = None

    async def _modifier_begin_invoke_async(self):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            await modifier.on_begin_invoke_async(self)

            self._current_modifier_name = None

    async def _modifier_pre_invoke_async(self, node: RunnableNode):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            with Profiler(
                f"pre_invoke_{modifier.__class__.__name__}",
                "modifier",
                network=self,
                node_id=node.uuid(),
                node_name=node.name or node.__class__.__name__,
                modifier_name=modifier.__class__.__name__,
            ):
                start_time = time.time()

                await modifier.on_pre_invoke_async(self, node)

                elapsed_time = time.time() - start_time
                if elapsed_time > 0.001:
                    self._set_modifier_info_metadata(
                        node, str(self._current_modifier_name), "on_pre_invoke_time", elapsed_time
                    )

            self._current_modifier_name = None

    async def _modifier_post_invoke_async(self, node: RunnableNode):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            with Profiler(
                f"post_invoke_{modifier.__class__.__name__}",
                "modifier",
                network=self,
                node_id=node.uuid(),
                node_name=node.name or node.__class__.__name__,
                modifier_name=modifier.__class__.__name__,
            ):
                start_time = time.time()

                await modifier.on_post_invoke_async(self, node)

                elapsed_time = time.time() - start_time
                if elapsed_time > 0.001:
                    self._set_modifier_info_metadata(
                        node, str(self._current_modifier_name), "on_post_invoke_time", elapsed_time
                    )

            self._current_modifier_name = None

    async def _modifier_end_invoke_async(self):
        for modifier in self._iterate_modifiers():
            self._current_modifier_name = modifier.__class__.__name__

            await modifier.on_end_invoke_async(self)

            self._current_modifier_name = None

    def _set_modifier_info_metadata(self, node: RunnableNode, name: str, key: str, value):
        if "modifier_info" not in node.metadata:
            node.metadata["modifier_info"] = {}

        modifier_info = node.metadata["modifier_info"]
        if name not in modifier_info:
            modifier_info[name] = {}

        modifier_info[name][key] = value

    def __enter__(self):
        # Get the current stack if it exists, or initialize a new one
        stack = _active_networks_var.get(None)
        if stack is None:
            stack = []
        stack.append(self)
        # Set the new stack for the current context
        _active_networks_var.set(stack)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack = _active_networks_var.get(None)
        if stack is not None:
            stack.pop()
            # Set the possibly modified stack back, if there are still items,
            # otherwise remove the variable
            if stack:
                _active_networks_var.set(stack)
            else:
                # This effectively clears the context variable for this context
                _active_networks_var.set(None)

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.__exit__(exc_type, exc_val, exc_tb)
