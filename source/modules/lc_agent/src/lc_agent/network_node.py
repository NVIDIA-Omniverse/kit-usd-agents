## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .network_modifier import NetworkModifier
from .node_factory import get_node_factory
from .runnable_network import RunnableNetwork
from .runnable_node import RunnableNode
from pydantic import model_serializer, model_validator
from typing import Dict, Any, TypeVar, Type, cast

# Create a TypeVar as a replacement for Self
T = TypeVar('T', bound='NetworkNode')


class NetworkNodeModifier(NetworkModifier):
    """
    A class that helps to connect subnetwork to the parent network.
    """

    def on_begin_invoke(self, network: "NetworkNode"):
        """
        Create a default node if the network is empty.
        """
        if not network.nodes:
            default_node = network.default_node
            if default_node:
                with network:
                    node = get_node_factory().create_node(default_node)

    def on_pre_invoke(self, network: "NetworkNode", node: "RunnableNode"):
        """
        Connect the root nodes to the parents of the agent node.
        """
        if not network.get_parents(node):
            network.parents >> node


class NetworkNode(RunnableNode, RunnableNetwork):
    """
    Represents a subnetwork.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_modifier(NetworkNodeModifier())

    @model_serializer
    def serialize_model(self) -> Dict[str, Any]:
        """Pydantic 2 serialization method using model_serializer"""
        # Create a base dictionary with all fields except parents
        result = {}
        for field_name, field_value in self:
            if field_name not in ["modifiers", "callbacks", "parents"]:
                result[field_name] = field_value

        # Add type information
        result["__node_type__"] = self.__class__.__name__

        # Add connections information
        result["__connections__"] = {
            i: [self._get_node_id(parent) for parent in node.parents if self._get_node_id(parent) is not None]
            for i, node in enumerate(self.nodes)
        }

        return result

    @model_validator(mode="wrap")
    @classmethod
    def deserialize(cls: Type[T], data: Any, handler) -> T:
        """Pydantic v2 deserialization method using model_validator with wrap mode"""
        # Handle dictionary input
        if isinstance(data, dict):
            # Make a copy to avoid modifying the original
            data_copy = data.copy()

            # Extract connections and nodes
            connections = data_copy.pop("__connections__", {})
            nodes = data_copy.pop("nodes", [])

            outputs = None
            # Handle outputs field specially to avoid message deserialization issues
            if "outputs" in data_copy:
                outputs = cls._deserialize_outputs(data_copy.pop("outputs"))

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

            if outputs is not None:
                network.outputs = outputs

            return cast(T, network)

        # Handle case where obj is already a NetworkNode instance
        if isinstance(data, NetworkNode):
            return cast(T, data)

        # For other types, use the default handler
        return handler(data)

    def _pre_invoke_network(self):
        """Called before invoking the network."""
        parent_network = RunnableNetwork.get_active_network()
        if parent_network and not self.chat_model_name:
            self.chat_model_name = parent_network.chat_model_name

    def _post_invoke_network(self):
        pass

    async def _ainvoke_chat_model(self, chat_model, chat_model_input, invoke_input, config, **kwargs):
        self._pre_invoke_network()

        result = await RunnableNetwork.ainvoke(self, invoke_input, config, **kwargs)

        self._post_invoke_network()

        return result

    def _invoke_chat_model(self, chat_model, chat_model_input, invoke_input, config, **kwargs):
        self._pre_invoke_network()

        result = RunnableNetwork.invoke(self, invoke_input, config, **kwargs)

        self._post_invoke_network()

        return result

    async def _astream_chat_model(self, chat_model, chat_model_input, invoke_input, config, **kwargs):
        self._pre_invoke_network()

        async for item in RunnableNetwork.astream(self, invoke_input, config, **kwargs):
            yield item

        self._post_invoke_network()
