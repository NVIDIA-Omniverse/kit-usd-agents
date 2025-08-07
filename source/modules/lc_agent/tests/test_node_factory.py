## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.node_factory import NodeFactory, get_node_factory
from lc_agent.runnable_node import RunnableNode
from lc_agent.runnable_utils import RunnableHumanNode, RunnableAINode
from pydantic import Field
import inspect

class CustomNode(RunnableNode):
    custom_arg: str = Field(default=None)

    def __init__(self, custom_arg=None, **kwargs):
        super().__init__(custom_arg=custom_arg, **kwargs)

def test_node_factory_singleton():
    factory1 = get_node_factory()
    factory2 = get_node_factory()
    assert factory1 is factory2

def test_register_and_create_node():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNode")
    
    node = factory.create_node("CustomNode", custom_arg="test")
    assert isinstance(node, CustomNode)
    assert node.custom_arg == "test"

def test_register_existing_node():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNode")
    factory.register(CustomNode, name="CustomNode")  # This should not raise an error

def test_create_unregistered_node():
    factory = NodeFactory()
    
    node = factory.create_node("UnregisteredNode")
    assert node is None

def test_get_registered_node_type():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNode")
    
    node_type = factory.get_registered_node_type("CustomNode")
    assert node_type == CustomNode

def test_get_unregistered_node_type():
    factory = NodeFactory()
    
    node_type = factory.get_registered_node_type("UnregisteredNode")
    assert node_type is None

def test_unregister_node():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNode")
    factory.unregister("CustomNode")
    
    node = factory.create_node("CustomNode")
    assert node is None

def test_unregister_nonexistent_node():
    factory = NodeFactory()
    factory.unregister("NonexistentNode")  # This should not raise an error

def test_default_nodes():
    factory = get_node_factory()
    
    # First, ensure these nodes are not registered
    factory.unregister("RunnableHumanNode")
    factory.unregister("RunnableAINode")
    
    # Now test that they are not available
    assert factory.create_node("RunnableHumanNode", human_message="Hello") is None
    assert factory.create_node("RunnableAINode", ai_message="Response") is None
    
    # Register the nodes
    factory.register(RunnableHumanNode)
    factory.register(RunnableAINode)
    
    # Now test that they are available and work correctly
    human_node = factory.create_node("RunnableHumanNode", human_message="Hello")
    assert isinstance(human_node, RunnableHumanNode)
    assert human_node.human_message == "Hello"
    
    ai_node = factory.create_node("RunnableAINode", ai_message="Response")
    assert isinstance(ai_node, RunnableAINode)
    assert ai_node.ai_message == "Response"
    
    # Clean up
    factory.unregister("RunnableHumanNode")
    factory.unregister("RunnableAINode")

def test_register_with_args():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNodeWithArgs", custom_arg="default")
    
    node = factory.create_node("CustomNodeWithArgs")
    assert isinstance(node, CustomNode)
    assert node.custom_arg == "default"
    
    node_override = factory.create_node("CustomNodeWithArgs", custom_arg="override")
    assert node_override.custom_arg == "override"

def test_register_multiple_nodes():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNode1")
    factory.register(CustomNode, name="CustomNode2")
    
    node1 = factory.create_node("CustomNode1", custom_arg="test1")
    node2 = factory.create_node("CustomNode2", custom_arg="test2")
    
    assert isinstance(node1, CustomNode)
    assert isinstance(node2, CustomNode)
    assert node1.custom_arg == "test1"
    assert node2.custom_arg == "test2"

def test_get_base_node_names():
    factory = NodeFactory()

    class BaseNode(RunnableNode):
        pass

    class ChildNode(BaseNode):
        pass

    class GrandchildNode(ChildNode):
        pass

    factory.register(BaseNode, name="BaseNode")
    factory.register(ChildNode, name="ChildNode")
    factory.register(GrandchildNode, name="GrandchildNode")

    base_names = factory.get_base_node_names("GrandchildNode")
    assert "BaseNode" in base_names
    assert "ChildNode" in base_names
    assert "GrandchildNode" in base_names

def test_find_by_name_or_type():
    factory = NodeFactory()
    factory.register(CustomNode, name="CustomNode")

    assert factory._find_by_name_or_type("CustomNode") == "CustomNode"
    assert factory._find_by_name_or_type("NonexistentNode") is None
    assert factory._find_by_name_or_type("CustomNode") == "CustomNode"

if __name__ == "__main__":
    pytest.main(["-v", "test_node_factory.py"])