## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.default_modifier import DefaultModifier
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_node import RunnableNode
from lc_agent.node_factory import get_node_factory
from langchain_core.messages import HumanMessage, AIMessage

class MockNode(RunnableNode):
    def __init__(self, message_content, **kwargs):
        super().__init__(**kwargs)
        self.outputs = HumanMessage(content=message_content)
        self.invoked = True

class MockDefaultNode(RunnableNode):
    pass

def test_on_post_invoke_with_human_message():
    get_node_factory().register(MockDefaultNode)
    
    with RunnableNetwork(default_node="MockDefaultNode") as network:
        node = MockNode("Test message")
    
        network.modifiers[0].on_post_invoke(network, node)
    
    children = network.get_children(node)
    
    assert len(children) == 1
    assert isinstance(children[0], MockDefaultNode)
    
    get_node_factory().unregister("MockDefaultNode")

def test_on_post_invoke_with_ai_message():
    get_node_factory().register(MockDefaultNode)
    
    with RunnableNetwork(default_node="MockDefaultNode") as network:
        node = RunnableNode()
        node.outputs = AIMessage(content="AI response")
        node.invoked = True
    
    network.modifiers[0].on_post_invoke(network, node)
    
    children = network.get_children(node)
    assert len(children) == 0
    
    get_node_factory().unregister("MockDefaultNode")

def test_on_post_invoke_with_existing_children():
    get_node_factory().register(MockDefaultNode)
    
    with RunnableNetwork(default_node="MockDefaultNode") as network:
        parent_node = MockNode("Parent message")
        child_node = MockNode("Child message")
    
        network.modifiers[0].on_post_invoke(network, parent_node)
    
    children = network.get_children(parent_node)
    assert len(children) == 1
    assert children[0] == child_node
    
    get_node_factory().unregister("MockDefaultNode")

def test_on_post_invoke_without_default_node():
    get_node_factory().register(MockDefaultNode)
    
    with RunnableNetwork() as network:
        node = MockNode("Test message")
    
    network.modifiers[0].on_post_invoke(network, node)
    
    children = network.get_children(node)
    assert len(children) == 0
    
    get_node_factory().unregister("MockDefaultNode")

def test_on_post_invoke_not_invoked_node():
    get_node_factory().register(MockDefaultNode)
    
    with RunnableNetwork(default_node="MockDefaultNode") as network:
        node = MockNode("Test message")
        node.invoked = False
    
    network.modifiers[0].on_post_invoke(network, node)
    
    children = network.get_children(node)
    assert len(children) == 0
    
    get_node_factory().unregister("MockDefaultNode")

if __name__ == "__main__":
    pytest.main(["-v", "test_default_modifier.py"])