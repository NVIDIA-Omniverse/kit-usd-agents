## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.network_lists.network_list import NetworkList
from lc_agent.runnable_network import RunnableNetwork

class TestNetworkList(NetworkList):
    def save(self, network=None):
        pass

    def load(self):
        pass

    def delete(self, network):
        pass

    async def save_async(self, network=None):
        pass

    async def load_async(self):
        pass

    async def delete_async(self, network):
        pass

@pytest.fixture
def network_list():
    return TestNetworkList()

@pytest.fixture
def sample_networks():
    network1 = RunnableNetwork()
    network1.metadata = {"name": "Network1", "type": "A"}
    
    network2 = RunnableNetwork()
    network2.metadata = {"name": "Network2", "type": "B"}
    
    network3 = RunnableNetwork()
    network3.metadata = {"name": "Network3", "type": "A"}
    
    return [network1, network2, network3]

def test_find_network(network_list, sample_networks):
    for network in sample_networks:
        network_list.append(network)
    
    result = network_list.find_network(lambda n: n.metadata["type"] == "A")
    assert len(result) == 2
    assert result[0].metadata["name"] == "Network1"
    assert result[1].metadata["name"] == "Network3"

def test_find_network_by_metadata(network_list, sample_networks):
    for network in sample_networks:
        network_list.append(network)
    
    result = network_list.find_network_by_metadata({"type": "B"})
    assert len(result) == 1
    assert result[0].metadata["name"] == "Network2"

def test_set_event_fn(network_list):
    events = []
    
    def event_handler(event, payload):
        events.append((event, payload))
    
    network_list.set_event_fn(event_handler)
    
    network = RunnableNetwork()
    network_list.append(network)
    
    assert len(events) == 1
    assert events[0][0] == NetworkList.Event.NETWORK_ADDED
    assert events[0][1]["network"] == network

def test_remove_event_fn(network_list):
    events = []
    
    def event_handler(event, payload):
        events.append((event, payload))
    
    event_id = network_list.set_event_fn(event_handler)
    network_list.remove_event_fn(event_id)
    
    network = RunnableNetwork()
    network_list.append(network)
    
    assert len(events) == 0

def test_multiple_event_handlers(network_list):
    events1 = []
    events2 = []
    
    def event_handler1(event, payload):
        events1.append((event, payload))
    
    def event_handler2(event, payload):
        events2.append((event, payload))
    
    network_list.set_event_fn(event_handler1)
    network_list.set_event_fn(event_handler2)
    
    network = RunnableNetwork()
    network_list.append(network)
    
    assert len(events1) == 1
    assert len(events2) == 1
    assert events1[0][0] == NetworkList.Event.NETWORK_ADDED
    assert events2[0][0] == NetworkList.Event.NETWORK_ADDED

def test_find_network_empty_list(network_list):
    result = network_list.find_network(lambda n: True)
    assert len(result) == 0

def test_find_network_by_metadata_no_match(network_list, sample_networks):
    for network in sample_networks:
        network_list.append(network)
    
    result = network_list.find_network_by_metadata({"type": "C"})
    assert len(result) == 0

if __name__ == "__main__":
    pytest.main(["-v", "test_network_list.py"])