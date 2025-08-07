## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent.network_lists.json_network_list import JsonNetworkList
from lc_agent.network_lists.json_network_list import _sanitize_filename
from lc_agent.node_factory import get_node_factory
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_node import RunnableNode
from lc_agent.runnable_utils import RunnableHumanNode
import os
import pytest
import shutil
import tempfile

get_node_factory().register(RunnableNode)
get_node_factory().register(RunnableHumanNode)


@pytest.fixture
def temp_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def json_network_list(temp_dir):
    JsonNetworkList.SAVE_PATH = temp_dir + "/networks/test"
    return JsonNetworkList()


def create_sample_network():
    network = RunnableNetwork(metadata={"name": "Test Network"})
    node1 = RunnableHumanNode("Hello, how can I assist you today?")
    node2 = RunnableHumanNode("What's the weather like?")
    node3 = RunnableHumanNode(
        "I'm sorry, I don't have access to real-time weather information."
    )
    network.add_node(node1)
    network.add_node(node2, parent=node1)
    network.add_node(node3, parent=node2)
    return network


def test_name(json_network_list):
    get_node_factory().register(
        RunnableHumanNode, name="tmp", human_message="human_message"
    )

    with RunnableNetwork() as network:
        get_node_factory().create_node("tmp")

    json_network_list.save(network)

    loaded_network_list = JsonNetworkList()
    loaded_network_list.load()

    get_node_factory().unregister("tmp")

    assert len(loaded_network_list) == 1
    network = loaded_network_list[0]
    assert len(network.nodes) == 1
    assert network.nodes[0].outputs.content == "human_message"
    assert network.nodes[0].name == "tmp"


def test_save_and_load(json_network_list):
    network1 = create_sample_network()
    network2 = create_sample_network()

    json_network_list.append(network1)
    json_network_list.append(network2)

    json_network_list.save()

    loaded_network_list = JsonNetworkList()
    loaded_network_list.load()

    assert len(loaded_network_list) == 2
    assert loaded_network_list[0].metadata == network1.metadata
    assert loaded_network_list[1].metadata == network2.metadata


def test_save_single_network(json_network_list):
    network = create_sample_network()

    json_network_list.save(network)

    loaded_network_list = JsonNetworkList()
    loaded_network_list.load()

    assert len(loaded_network_list) == 1
    assert loaded_network_list[0].metadata == network.metadata


def test_remove_old_files(json_network_list):
    network1 = create_sample_network()
    network2 = create_sample_network()

    json_network_list.append(network1)
    json_network_list.append(network2)

    json_network_list.save()

    json_network_list.remove(network1)
    json_network_list.save()

    loaded_network_list = JsonNetworkList()
    loaded_network_list.load()

    assert len(loaded_network_list) == 1
    assert loaded_network_list[0].metadata == network2.metadata


def test_load_non_existent_directory(json_network_list):
    loaded_network_list = JsonNetworkList()
    loaded_network_list.load()

    assert len(loaded_network_list) == 0


def test_load_invalid_file(json_network_list, temp_dir):
    invalid_file_path = os.path.join(temp_dir, "networks", "invalid.json")
    os.makedirs(os.path.dirname(invalid_file_path), exist_ok=True)
    with open(invalid_file_path, "w") as file:
        file.write("invalid json")

    loaded_network_list = JsonNetworkList()
    loaded_network_list.load()

    assert len(loaded_network_list) == 0


def test_sanitize_filename():
    assert _sanitize_filename("Test Network") == "Test_Network"
    assert _sanitize_filename("Test!@#$%^&*()Network") == "TestNetwork"
    assert _sanitize_filename("Test Network 123") == "Test_Network_123"
