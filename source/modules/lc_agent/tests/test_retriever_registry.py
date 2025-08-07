## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.retriever_registry import RetrieverRegistry, get_retriever_registry


class DummyRetriever:
    def __init__(self, name):
        self.name = name


def test_register_retriever():
    registry = RetrieverRegistry()
    retriever = DummyRetriever("test_retriever")

    registry.register("test", retriever)
    assert "test" in registry.get_registered_names()
    assert registry.get_retriever("test") == retriever


def test_unregister_retriever():
    registry = RetrieverRegistry()
    retriever = DummyRetriever("test_retriever")

    registry.register("test", retriever)
    registry.unregister("test")
    assert "test" not in registry.get_registered_names()
    assert registry.get_retriever("test") is None


def test_get_retriever_default():
    registry = RetrieverRegistry()
    retriever1 = DummyRetriever("retriever1")
    retriever2 = DummyRetriever("retriever2")

    registry.register("retriever1", retriever1)
    registry.register("retriever2", retriever2)

    # Should return the first registered retriever if no name is provided
    assert registry.get_retriever("") == retriever1


def test_unregister_nonexistent_retriever():
    registry = RetrieverRegistry()
    retriever = DummyRetriever("test_retriever")

    registry.register("test", retriever)
    try:
        raised = False
        registry.unregister("nonexistent")
    except ValueError:
        raised = True

    assert raised
    assert "test" in registry.get_registered_names()
    assert registry.get_retriever("test") == retriever


def test_get_nonexistent_retriever():
    registry = RetrieverRegistry()
    retriever = DummyRetriever("test_retriever")

    registry.register("test", retriever)
    assert registry.get_retriever("nonexistent") is None


def test_global_retriever_registry():
    global_registry = get_retriever_registry()
    retriever = DummyRetriever("global_retriever")

    global_registry.register("global", retriever)
    assert "global" in global_registry.get_registered_names()
    assert global_registry.get_retriever("global") == retriever

    global_registry.unregister("global")
    assert "global" not in global_registry.get_registered_names()
    assert global_registry.get_retriever("global") is None


if __name__ == "__main__":
    pytest.main()
