## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from unittest.mock import MagicMock, patch
from lc_agent.network_node import NetworkNode, NetworkNodeModifier
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_node import RunnableNode
from lc_agent.node_factory import get_node_factory
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def network_node():
    return NetworkNode()

def test_network_node_initialization(network_node):
    assert isinstance(network_node, NetworkNode)
    assert isinstance(network_node, RunnableNode)
    assert isinstance(network_node, RunnableNetwork)
    assert any(isinstance(modifier, NetworkNodeModifier) for modifier in network_node.modifiers.values())

def test_network_node_modifier_on_begin_invoke():
    # No default node - no new node should be added
    with NetworkNode() as network:
        RunnableNode()
        network.modifiers[0].on_begin_invoke(network)
    assert len(network.nodes) == 1

def test_network_node_modifier_on_pre_invoke():
    network = NetworkNode()
    modifier = NetworkNodeModifier()
    node = RunnableNode()
    network.add_node(node)
    
    # Test when node has no parents
    modifier.on_pre_invoke(network, node)
    assert network.get_parents(node) == network.parents

    # Test when node already has parents
    parent_node = RunnableNode()
    network.add_node(parent_node)
    node._add_parent(parent_node)
    modifier.on_pre_invoke(network, node)
    assert network.get_parents(node) == [parent_node]

@pytest.mark.asyncio
async def test_network_node_ainvoke():
    network = NetworkNode()
    input_data = {"input": "test"}
    
    with patch.object(RunnableNetwork, 'ainvoke') as mock_ainvoke:
        mock_ainvoke.return_value = AIMessage(content="Test response")
        
        result = await network._ainvoke_chat_model(None, None, input_data, None)
        
        assert result.content == "Test response"
        mock_ainvoke.assert_called_once_with(network, input_data, None)

@pytest.mark.asyncio
async def test_network_node_astream():
    network = NetworkNode()
    input_data = {"input": "test"}
    
    async def mock_astream(*args, **kwargs):
        yield AIMessage(content="Test")
        yield AIMessage(content="response")
    
    with patch.object(RunnableNetwork, 'astream', new=mock_astream):
        result = []
        async for item in network._astream_chat_model(None, None, input_data, None):
            result.append(item)
        
        assert len(result) == 2
        assert result[0].content == "Test"
        assert result[1].content == "response"

@pytest.mark.asyncio
async def test_network_node_invoke():
    network = NetworkNode()
    input_data = {"input": "test"}
    
    with patch.object(RunnableNetwork, 'invoke') as mock_invoke:
        mock_invoke.return_value = AIMessage(content="Test response")
        
        result = network._invoke_chat_model(None, None, input_data, None)
        
        assert result.content == "Test response"
        mock_invoke.assert_called_once_with(network, input_data, None)

def test_network_node_pre_invoke_network():
    network = NetworkNode()
    parent_network = RunnableNetwork(chat_model_name="test_model")
    
    with patch.object(RunnableNetwork, 'get_active_network', return_value=parent_network):
        network._pre_invoke_network()
        
        assert network.chat_model_name == "test_model"

def test_network_node_pre_invoke_network_no_parent():
    network = NetworkNode(chat_model_name="own_model")
    
    with patch.object(RunnableNetwork, 'get_active_network', return_value=None):
        network._pre_invoke_network()
        
        assert network.chat_model_name == "own_model"

if __name__ == "__main__":
    pytest.main(["-v", "test_network_node.py"])