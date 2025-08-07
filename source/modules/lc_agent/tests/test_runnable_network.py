## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
import asyncio
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_node import RunnableNode
from lc_agent.network_modifier import NetworkModifier
from langchain_core.messages import AIMessage
from typing import Dict, Any, AsyncIterator
from lc_agent.default_modifier import DefaultModifier

class DummyNode(RunnableNode):
    def invoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content="Dummy response")
        self.invoked = True
        return self.outputs

    async def ainvoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content="Dummy response")
        self.invoked = True
        return self.outputs

    async def astream(self, input: Dict[str, Any] = {}, config=None, **kwargs) -> AsyncIterator[AIMessage]:
        if not self.invoked:
            self.outputs = AIMessage(content="Dummy response")
            self.invoked = True
            yield self.outputs
        else:
            yield self.outputs

    def __eq__(self, other):
        return isinstance(other, DummyNode) and id(self) == id(other)

    def __hash__(self):
        return hash(id(self))

    def _get_chat_model_name(self, chat_model_input, invoke_input, config):
        from lc_agent.runnable_network import RunnableNetwork
        network = RunnableNetwork.get_active_network()
        if network and network.chat_model_name:
            return network.chat_model_name
        return super()._get_chat_model_name(chat_model_input, invoke_input, config)

    @classmethod
    def parse_obj(cls, obj):
        instance = cls()
        instance.__dict__.update(obj)
        return instance

class DummyModifier(NetworkModifier):
    def on_begin_invoke(self, network):
        network.metadata["modifier_called"] = True

@pytest.fixture
def simple_network():
    with RunnableNetwork() as network:
        DummyNode()
        DummyNode()
    
    # Reset invoked flag for all nodes
    for node in network.nodes:
        node.invoked = False
    
    return network

def test_add_node():
    with RunnableNetwork() as network:
        node = DummyNode()
    assert node in network.nodes
    assert network.get_leaf_nodes() == [node]

def test_remove_node():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    network.remove_node(node1)
    assert node1 not in network.nodes
    assert node2 in network.nodes
    assert network.get_leaf_nodes() == [node2]
    assert network.get_root_nodes() == [node2]
    assert node2.parents == []  # Ensure node2's parent is removed

def test_get_parents():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    assert network.get_parents(node2) == [node1]

def test_get_children():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    assert network.get_children(node1) == [node2]
    assert network.get_children(node2) == []  # Ensure node2 has no children

def test_get_root_nodes():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    root_nodes = network.get_root_nodes()
    assert len(root_nodes) == 1
    assert node1 in root_nodes
    assert node2 not in root_nodes

def test_get_leaf_nodes():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
        node3 = DummyNode()
    assert set(network.get_leaf_nodes()) == {node3}

def test_get_sorted_nodes():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
        node3 = DummyNode()
    assert network.get_sorted_nodes() == [node1, node2, node3]

def test_add_modifier():
    network = RunnableNetwork()
    modifier = DummyModifier()
    modifier_id = network.add_modifier(modifier)
    assert modifier_id in network.modifiers
    assert network.modifiers[modifier_id] == modifier

def test_remove_modifier():
    network = RunnableNetwork()
    modifier = DummyModifier()
    modifier_id = network.add_modifier(modifier)
    network.remove_modifier(modifier_id)
    assert modifier_id not in network.modifiers or network.modifiers[modifier_id] is None

def test_invoke(simple_network):
    result = simple_network.invoke()
    assert isinstance(result, AIMessage)
    assert result.content == "Dummy response"

@pytest.mark.asyncio
async def test_ainvoke(simple_network):
    result = await simple_network.ainvoke()
    assert isinstance(result, AIMessage)
    assert result.content == "Dummy response"

@pytest.mark.asyncio
async def test_astream(simple_network):
    chunks = []
    async for chunk in simple_network.astream():
        chunks.append(chunk)
    assert len(chunks) > 0
    assert all(isinstance(chunk, AIMessage) for chunk in chunks)
    assert all(chunk.content == "Dummy response" for chunk in chunks)

def test_modifier_execution(simple_network):
    modifier = DummyModifier()
    simple_network.add_modifier(modifier)
    simple_network.invoke()
    assert simple_network.metadata.get("modifier_called") == True

def test_event_callback():
    network = RunnableNetwork()
    events = []
    
    def event_handler(event, payload):
        events.append((event, payload))
    
    network.set_event_fn(event_handler)
    with network:
        node = DummyNode()
    
    assert len(events) == 1
    assert events[0][0] == RunnableNetwork.Event.NODE_ADDED

def test_serialization():
    with RunnableNetwork() as network:
        DummyNode()
        DummyNode()
    
    serialized = network.dict()
    deserialized = RunnableNetwork.parse_obj(serialized)
    
    assert len(deserialized.nodes) == 2
    assert isinstance(deserialized.nodes[0], RunnableNode)
    assert isinstance(deserialized.nodes[1], RunnableNode)
    assert deserialized.get_parents(deserialized.nodes[1]) == [deserialized.nodes[0]]

def test_context_manager():
    with RunnableNetwork() as network:
        assert RunnableNetwork.get_active_network() == network
    assert RunnableNetwork.get_active_network() is None

def test_chat_model_name():
    with RunnableNetwork(chat_model_name="test_model") as network:
        node = DummyNode()
    with network:
        assert node._get_chat_model_name(None, None, None) == "test_model"

def test_default_modifier():
    network = RunnableNetwork()
    assert len(network.modifiers) == 1
    assert isinstance(list(network.modifiers.values())[0], DefaultModifier)

def test_add_node_with_parent():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    assert node1 in network.nodes
    assert node2 in network.nodes
    assert network.get_parents(node2) == [node1]
    assert network.get_children(node1) == [node2]
    assert network.get_root_nodes() == [node1]
    assert network.get_leaf_nodes() == [node2]

def test_multiple_root_nodes():
    with RunnableNetwork() as network:
        None >> DummyNode()
        None >> DummyNode()
    assert len(network.get_root_nodes()) == 2
    assert len(network.get_leaf_nodes()) == 2

def test_complex_network():
    with RunnableNetwork() as network:
        a = DummyNode()
        b = DummyNode()
        c = DummyNode()
        None >> DummyNode()  # Create a separate root node
        d = DummyNode()
        e = DummyNode()
    
    assert len(network.get_root_nodes()) == 2  # a and the separate root node
    assert len(network.get_leaf_nodes()) == 2  # c and e
    assert network.get_children(a) == [b]
    assert network.get_children(b) == [c]
    assert network.get_children(d) == [e]
    assert network.get_parents(e) == [d]

def test_automatic_node_connection():
    with RunnableNetwork() as network:
        node_a = DummyNode()
        node_b = DummyNode()
        node_c = DummyNode()
    
    assert network.get_root_nodes() == [node_a]
    assert network.get_leaf_nodes() == [node_c]
    assert network.get_parents(node_b) == [node_a]
    assert network.get_parents(node_c) == [node_b]
    assert network.get_children(node_a) == [node_b]
    assert network.get_children(node_b) == [node_c]

def test_mixed_connection_methods():
    with RunnableNetwork() as network:
        a = DummyNode()
        b = DummyNode()
        c = DummyNode()
        d = DummyNode()
        a >> d
        e = DummyNode()  # e will be automatically connected to d
    
    assert network.get_root_nodes() == [a]
    assert network.get_leaf_nodes() == [c, e]
    assert network.get_children(a) == [b, d]
    assert network.get_children(b) == [c]
    assert network.get_children(d) == [e]

def test_get_all_parents():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
        node3 = DummyNode()
    assert network.get_all_parents(node3) == [node2, node1]
    assert network.get_all_parents(node2) == [node1]
    assert network.get_all_parents(node1) == []

def test_get_all_children():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
        node3 = DummyNode()
    assert network.get_all_children(node1) == [node2, node3]
    assert network.get_all_children(node2) == [node3]
    assert network.get_all_children(node3) == []

def test_restore_node_set():
    network = RunnableNetwork()
    node1 = DummyNode()
    node2 = DummyNode()
    network.nodes = [node1, node2]
    network._node_set = set()
    network._RunnableNetwork__restore_node_set()
    assert network._node_set == set(network.nodes)

def test_get_leaf_node():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    assert network.get_leaf_node() == node2
    
    # Test when there are no leaf nodes
    network.nodes = []
    assert network.get_leaf_node() is None

def test_remove_event_fn():
    network = RunnableNetwork()
    def dummy_handler(event, payload):
        pass
    event_id = network.set_event_fn(dummy_handler)
    network.remove_event_fn(event_id)
    assert network.callbacks[event_id] is None

def test_get_modifier_id():
    network = RunnableNetwork()
    modifier = DummyModifier()
    modifier_id = network.add_modifier(modifier)
    assert network.get_modifier_id(modifier) == modifier_id
    assert network.get_modifier_id(DummyModifier) == modifier_id

def test_set_modifier_info_metadata():
    network = RunnableNetwork()
    node = DummyNode()
    network._set_modifier_info_metadata(node, "TestModifier", "on_pre_invoke_time", 0.5)
    network._set_modifier_info_metadata(node, "TestModifier", "on_post_invoke_time", 0.7)
    
    assert "modifier_info" in node.metadata
    assert "TestModifier" in node.metadata["modifier_info"]
    assert node.metadata["modifier_info"]["TestModifier"]["on_pre_invoke_time"] == 0.5
    assert node.metadata["modifier_info"]["TestModifier"]["on_post_invoke_time"] == 0.7

def test_add_node_with_modifier():
    class TestModifier(NetworkModifier):
        def on_begin_invoke(self, network):
            network._current_modifier_name = "TestModifier"
            node = DummyNode()
            network.add_node(node)
            network._current_modifier_name = None

    network = RunnableNetwork()
    modifier = TestModifier()
    network.add_modifier(modifier)
    initial_node_count = len(network.nodes)
    network.invoke()
    
    assert len(network.nodes) == initial_node_count + 2  # One node added by modifier, one by invoke
    added_node = network.nodes[-2]  # The node added by the modifier
    assert "modifier_info" in added_node.metadata
    assert added_node.metadata["modifier_info"]["added_by"] == "TestModifier"

def test_add_modifier_once():
    network = RunnableNetwork()
    modifier1 = DummyModifier()
    modifier2 = DummyModifier()
    
    id1 = network.add_modifier(modifier1, once=True)
    id2 = network.add_modifier(modifier2, once=True)
    
    assert id1 == id2
    assert len(network.modifiers) == 2  # Including the default modifier

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])