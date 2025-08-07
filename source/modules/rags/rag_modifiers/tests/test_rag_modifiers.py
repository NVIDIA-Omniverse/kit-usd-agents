## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from lc_agent import (
    FromRunnableNode,
    RunnableHumanNode,
    RunnableNetwork,
    RunnableNode,
    get_chat_model_registry,
    get_node_factory,
)
from lc_agent_rag_modifiers import RetrieverMessage, SystemRagModifier, HumanRagModifier

# Register fake chat model
get_chat_model_registry().register(
    "Fake",
    FakeListChatModel(name="Fake", responses=["I am happy", "who are you", "me too"]),
)

# Register RunnableNode to the node factory
get_node_factory().register(RunnableNode)


@pytest.fixture
def network_with_system_rag_modifier():
    with RunnableNetwork(
        default_node="RunnableNode", chat_model_name="Fake"
    ) as network:
        network.add_modifier(SystemRagModifier(retriever_name="embedqa"))
        RunnableHumanNode("How to create a cube? Answer code only.")
        yield network


@pytest.fixture
def network_with_human_rag_modifier():
    with RunnableNetwork(
        default_node="RunnableNode", chat_model_name="Fake"
    ) as network:
        network.add_modifier(HumanRagModifier(retriever_name="embedqa"))
        RunnableHumanNode("How to create a cube? Answer code only.")
        yield network


def test_system_rag_modifier(network_with_system_rag_modifier):
    result = network_with_system_rag_modifier.invoke()

    nodes = network_with_system_rag_modifier.get_sorted_nodes()

    # Check it has created the correct number of nodes
    assert len(nodes) == 2

    assert type(nodes[0]) == RunnableHumanNode
    assert type(nodes[1]) == RunnableNode

    retriever_message = nodes[1].inputs[0]
    assert (
        retriever_message.question == "How to create a cube? Answer code only."
    ), f"Unexpected question: {retriever_message.question}"

    assert result.content == "I am happy", f"Unexpected content: {result.content}"


def test_human_rag_modifier(network_with_human_rag_modifier):
    result = network_with_human_rag_modifier.invoke()

    nodes = network_with_human_rag_modifier.get_sorted_nodes()

    # Check it has created the correct number of nodes
    assert len(nodes) == 3

    assert type(nodes[0]) == FromRunnableNode
    assert type(nodes[1]) == RunnableHumanNode
    assert type(nodes[2]) == RunnableNode

    retriever_message = nodes[0].runnable
    assert (
        retriever_message.question == "How to create a cube? Answer code only."
    ), f"Unexpected question: {retriever_message.question}"

    assert result.content == "who are you", f"Unexpected content: {result.content}"
