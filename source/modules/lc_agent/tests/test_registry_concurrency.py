## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Tests for concurrent access to NodeFactory and ChatModelRegistry.

These tests verify that the NodeFactory and ChatModelRegistry work correctly
when multiple concurrent requests use the same node/model names. This addresses
the race condition bug where one request's unregister() call removes a node
that another concurrent request still needs.

See: lc_agent_concurrency_bug_report.md for details on the original issue.

The issue occurs in this scenario:
    Request 1: pre_invoke()  -> register("SharedNode")
    Request 2: pre_invoke()  -> register("SharedNode")  # same name
    Request 1: post_invoke() -> unregister("SharedNode")  # removes it!
    Request 2: create_node("SharedNode") -> returns None (BUG!)
"""

import asyncio
import threading
import pytest
from typing import Dict, Any, AsyncIterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from lc_agent.node_factory import NodeFactory, get_node_factory
from lc_agent.chat_model_registry import ChatModelRegistry, get_chat_model_registry
from lc_agent.retriever_registry import RetrieverRegistry, get_retriever_registry
from lc_agent.runnable_node import RunnableNode
from langchain_core.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage
from pydantic import Field


class DummyNode(RunnableNode):
    """A simple node for testing that doesn't require a chat model."""

    def invoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content=f"Response from {self.name or 'unnamed'}")
        self.invoked = True
        return self.outputs

    async def ainvoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content=f"Response from {self.name or 'unnamed'}")
        self.invoked = True
        return self.outputs

    def __hash__(self):
        return hash(id(self))


class DummyChatModel(BaseChatModel):
    """A dummy chat model for testing."""
    model_name: str = Field(...)

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs
    ) -> ChatResult:
        return ChatResult(generations=[])

    @property
    def _llm_type(self) -> str:
        return "dummy"


# =============================================================================
# NodeFactory Concurrency Tests
# =============================================================================

class TestNodeFactoryConcurrency:
    """Tests for NodeFactory thread safety and reference counting."""

    def test_concurrent_register_unregister_race_condition(self):
        """Test that simulates the race condition in concurrent request handling.

        This test replicates the bug from the bug report:
        - Request 1 registers "SharedNode"
        - Request 2 registers "SharedNode" (same name)
        - Request 1 finishes and calls unregister("SharedNode")
        - Request 2 tries to use the node but it's gone!

        With reference counting, the node should remain available until
        ALL requests have unregistered.
        """
        factory = NodeFactory()
        errors = []
        node_creation_results = []

        def simulate_request(request_id: int, delay_before_unregister: float):
            """Simulate a request that registers, uses, and unregisters a node."""
            node_name = "SharedNode"

            # Pre-invoke: register the node
            factory.register(DummyNode, name=node_name)

            # Small delay to let other requests start
            import time
            time.sleep(0.01)

            # Simulate some work
            time.sleep(delay_before_unregister)

            # Try to create and use the node
            node = factory.create_node(node_name)
            if node is None:
                errors.append(f"Request {request_id}: create_node returned None!")
                node_creation_results.append((request_id, False))
            else:
                node_creation_results.append((request_id, True))

            # Post-invoke: unregister the node
            factory.unregister(node_name)

        # Run concurrent requests with different delays
        # Request 1 will unregister quickly while Request 2 is still using the node
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(simulate_request, 1, 0.0),   # Quick finish
                executor.submit(simulate_request, 2, 0.05),  # Slower
                executor.submit(simulate_request, 3, 0.03),  # Medium
                executor.submit(simulate_request, 4, 0.02),  # Medium-quick
            ]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        # With the bug, some requests will have create_node return None
        # With the fix, all requests should succeed
        failed_requests = [r for r in node_creation_results if not r[1]]

        # This assertion describes what SHOULD happen after the fix
        # Before the fix, this test should FAIL
        assert len(errors) == 0, f"Race condition errors: {errors}"
        assert len(failed_requests) == 0, f"Failed requests: {failed_requests}"

    @pytest.mark.asyncio
    async def test_async_concurrent_register_unregister(self):
        """Async version of the race condition test."""
        factory = NodeFactory()
        errors = []
        results = []

        async def simulate_async_request(request_id: int, delay: float):
            """Simulate an async request that registers, uses, and unregisters a node."""
            node_name = "AsyncSharedNode"

            # Pre-invoke: register the node
            factory.register(DummyNode, name=node_name)

            # Simulate some async work
            await asyncio.sleep(delay)

            # Try to create and use the node
            node = factory.create_node(node_name)
            if node is None:
                errors.append(f"Async Request {request_id}: create_node returned None!")
                results.append((request_id, False))
            else:
                results.append((request_id, True))

            # Post-invoke: unregister the node
            factory.unregister(node_name)

        # Run concurrent async requests
        await asyncio.gather(
            simulate_async_request(1, 0.0),    # Quick finish
            simulate_async_request(2, 0.05),   # Slower
            simulate_async_request(3, 0.02),   # Medium
            simulate_async_request(4, 0.03),   # Medium
        )

        failed = [r for r in results if not r[1]]
        assert len(errors) == 0, f"Async race condition errors: {errors}"
        assert len(failed) == 0, f"Failed async requests: {failed}"

    def test_reference_counting_basic(self):
        """Test that reference counting prevents premature unregistration."""
        factory = NodeFactory()
        node_name = "RefCountedNode"

        # Simulate 3 requests registering the same node
        factory.register(DummyNode, name=node_name)
        factory.register(DummyNode, name=node_name)
        factory.register(DummyNode, name=node_name)

        # First two unregisters should NOT remove the node
        factory.unregister(node_name)
        assert factory.has_registered(node_name), "Node was removed too early (after 1st unregister)"

        factory.unregister(node_name)
        assert factory.has_registered(node_name), "Node was removed too early (after 2nd unregister)"

        # Third unregister should remove the node
        factory.unregister(node_name)
        assert not factory.has_registered(node_name), "Node should be removed after all unregisters"

    def test_unregister_without_register(self):
        """Test that unregistering a non-existent node is safe."""
        factory = NodeFactory()
        # Should not raise an exception
        factory.unregister("NonExistentNode")

    def test_thread_safety_stress(self):
        """Stress test with many concurrent operations."""
        factory = NodeFactory()
        errors = []
        num_threads = 20
        operations_per_thread = 50

        def thread_worker(thread_id: int):
            node_name = "StressTestNode"
            for i in range(operations_per_thread):
                try:
                    factory.register(DummyNode, name=node_name)
                    # Verify registration succeeded
                    if not factory.has_registered(node_name):
                        errors.append(f"Thread {thread_id}, op {i}: Node not registered after register()")
                    factory.unregister(node_name)
                except Exception as e:
                    errors.append(f"Thread {thread_id}, op {i}: Exception: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Thread safety errors: {errors[:10]}..."  # Show first 10


# =============================================================================
# ChatModelRegistry Concurrency Tests
# =============================================================================

class TestChatModelRegistryConcurrency:
    """Tests for ChatModelRegistry thread safety and reference counting."""

    def test_concurrent_register_unregister_race_condition(self):
        """Test that simulates the race condition in concurrent request handling."""
        registry = ChatModelRegistry()
        errors = []
        model_retrieval_results = []

        def simulate_request(request_id: int, delay_before_unregister: float):
            """Simulate a request that registers, uses, and unregisters a model."""
            model_name = "SharedModel"

            # Pre-invoke: register the model
            chat_model = DummyChatModel(model_name=f"model_{request_id}")
            registry.register(model_name, chat_model)

            # Small delay to let other requests start
            import time
            time.sleep(0.01)

            # Simulate some work
            time.sleep(delay_before_unregister)

            # Try to retrieve and use the model
            model = registry.get_model(model_name)
            if model is None:
                errors.append(f"Request {request_id}: get_model returned None!")
                model_retrieval_results.append((request_id, False))
            else:
                model_retrieval_results.append((request_id, True))

            # Post-invoke: unregister the model
            registry.unregister(model_name)

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(simulate_request, 1, 0.0),
                executor.submit(simulate_request, 2, 0.05),
                executor.submit(simulate_request, 3, 0.03),
                executor.submit(simulate_request, 4, 0.02),
            ]
            for future in as_completed(futures):
                future.result()

        failed_requests = [r for r in model_retrieval_results if not r[1]]

        assert len(errors) == 0, f"Race condition errors: {errors}"
        assert len(failed_requests) == 0, f"Failed requests: {failed_requests}"

    @pytest.mark.asyncio
    async def test_async_concurrent_register_unregister(self):
        """Async version of the race condition test for ChatModelRegistry."""
        registry = ChatModelRegistry()
        errors = []
        results = []

        async def simulate_async_request(request_id: int, delay: float):
            """Simulate an async request."""
            model_name = "AsyncSharedModel"

            chat_model = DummyChatModel(model_name=f"model_{request_id}")
            registry.register(model_name, chat_model)

            await asyncio.sleep(delay)

            model = registry.get_model(model_name)
            if model is None:
                errors.append(f"Async Request {request_id}: get_model returned None!")
                results.append((request_id, False))
            else:
                results.append((request_id, True))

            registry.unregister(model_name)

        await asyncio.gather(
            simulate_async_request(1, 0.0),
            simulate_async_request(2, 0.05),
            simulate_async_request(3, 0.02),
            simulate_async_request(4, 0.03),
        )

        failed = [r for r in results if not r[1]]
        assert len(errors) == 0, f"Async race condition errors: {errors}"
        assert len(failed) == 0, f"Failed async requests: {failed}"

    def test_reference_counting_basic(self):
        """Test that reference counting prevents premature unregistration."""
        registry = ChatModelRegistry()
        model_name = "RefCountedModel"

        # Simulate 3 requests registering the same model
        chat_model = DummyChatModel(model_name="test_model")
        registry.register(model_name, chat_model)
        registry.register(model_name, chat_model)
        registry.register(model_name, chat_model)

        # First two unregisters should NOT remove the model
        registry.unregister(model_name)
        assert registry.get_model(model_name) is not None, "Model was removed too early (after 1st unregister)"

        registry.unregister(model_name)
        assert registry.get_model(model_name) is not None, "Model was removed too early (after 2nd unregister)"

        # Third unregister should remove the model
        registry.unregister(model_name)
        assert registry.get_model(model_name) is None, "Model should be removed after all unregisters"

    def test_registered_names_list_handling(self):
        """Test that registered_names list is properly handled with reference counting.

        The original ChatModelRegistry appends to registered_names on every register call,
        which causes the list to grow unbounded. With reference counting, this should be fixed.
        """
        registry = ChatModelRegistry()
        model_name = "DuplicateTestModel"
        chat_model = DummyChatModel(model_name="test")

        # Register the same model multiple times
        registry.register(model_name, chat_model)
        registry.register(model_name, chat_model)
        registry.register(model_name, chat_model)

        # Count how many times the name appears in registered_names
        name_count = registry.registered_names.count(model_name)

        # With reference counting, the name should only appear once in the list
        # Before the fix, it would appear 3 times
        assert name_count == 1, f"Name appears {name_count} times in registered_names (should be 1)"

    def test_thread_safety_stress(self):
        """Stress test with many concurrent operations."""
        registry = ChatModelRegistry()
        errors = []
        num_threads = 20
        operations_per_thread = 50

        def thread_worker(thread_id: int):
            model_name = "StressTestModel"
            for i in range(operations_per_thread):
                try:
                    chat_model = DummyChatModel(model_name=f"model_{thread_id}_{i}")
                    registry.register(model_name, chat_model)
                    # Verify registration succeeded
                    if registry.get_model(model_name) is None:
                        errors.append(f"Thread {thread_id}, op {i}: Model not available after register()")
                    registry.unregister(model_name)
                except Exception as e:
                    errors.append(f"Thread {thread_id}, op {i}: Exception: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Thread safety errors: {errors[:10]}..."


# =============================================================================
# Integration Tests (Global Singletons)
# =============================================================================

class TestGlobalSingletonConcurrency:
    """Tests for the global singleton instances under concurrent access."""

    def test_global_node_factory_concurrent_access(self):
        """Test that the global NodeFactory handles concurrent access correctly."""
        factory = get_node_factory()
        errors = []
        unique_node_name = "GlobalFactoryTestNode_" + str(id(self))

        def worker(worker_id: int):
            try:
                factory.register(DummyNode, name=unique_node_name)
                import time
                time.sleep(0.01 * (worker_id % 3))
                node = factory.create_node(unique_node_name)
                if node is None:
                    errors.append(f"Worker {worker_id}: create_node returned None")
                factory.unregister(unique_node_name)
            except Exception as e:
                errors.append(f"Worker {worker_id}: Exception: {e}")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Global factory errors: {errors}"

    def test_global_chat_model_registry_concurrent_access(self):
        """Test that the global ChatModelRegistry handles concurrent access correctly."""
        registry = get_chat_model_registry()
        errors = []
        unique_model_name = "GlobalRegistryTestModel_" + str(id(self))

        def worker(worker_id: int):
            try:
                chat_model = DummyChatModel(model_name=f"model_{worker_id}")
                registry.register(unique_model_name, chat_model)
                import time
                time.sleep(0.01 * (worker_id % 3))
                model = registry.get_model(unique_model_name)
                if model is None:
                    errors.append(f"Worker {worker_id}: get_model returned None")
                registry.unregister(unique_model_name)
            except Exception as e:
                errors.append(f"Worker {worker_id}: Exception: {e}")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Global registry errors: {errors}"


# =============================================================================
# RetrieverRegistry Concurrency Tests
# =============================================================================

class DummyRetriever:
    """A dummy retriever for testing."""

    def __init__(self, name: str = "dummy"):
        self.name = name

    def retrieve(self, query: str):
        return [f"Result for {query} from {self.name}"]


class TestRetrieverRegistryConcurrency:
    """Tests for RetrieverRegistry thread safety and reference counting."""

    def test_concurrent_register_unregister_race_condition(self):
        """Test that simulates the race condition in concurrent request handling."""
        registry = RetrieverRegistry()
        errors = []
        retrieval_results = []

        def simulate_request(request_id: int, delay_before_unregister: float):
            """Simulate a request that registers, uses, and unregisters a retriever."""
            retriever_name = "SharedRetriever"

            # Pre-invoke: register the retriever
            retriever = DummyRetriever(name=f"retriever_{request_id}")
            registry.register(retriever_name, retriever)

            # Small delay to let other requests start
            import time
            time.sleep(0.01)

            # Simulate some work
            time.sleep(delay_before_unregister)

            # Try to retrieve and use the retriever
            retrieved = registry.get_retriever(retriever_name)
            if retrieved is None:
                errors.append(f"Request {request_id}: get_retriever returned None!")
                retrieval_results.append((request_id, False))
            else:
                retrieval_results.append((request_id, True))

            # Post-invoke: unregister the retriever
            registry.unregister(retriever_name)

        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(simulate_request, 1, 0.0),
                executor.submit(simulate_request, 2, 0.05),
                executor.submit(simulate_request, 3, 0.03),
                executor.submit(simulate_request, 4, 0.02),
            ]
            for future in as_completed(futures):
                future.result()

        failed_requests = [r for r in retrieval_results if not r[1]]

        assert len(errors) == 0, f"Race condition errors: {errors}"
        assert len(failed_requests) == 0, f"Failed requests: {failed_requests}"

    @pytest.mark.asyncio
    async def test_async_concurrent_register_unregister(self):
        """Async version of the race condition test for RetrieverRegistry."""
        registry = RetrieverRegistry()
        errors = []
        results = []

        async def simulate_async_request(request_id: int, delay: float):
            """Simulate an async request."""
            retriever_name = "AsyncSharedRetriever"

            retriever = DummyRetriever(name=f"retriever_{request_id}")
            registry.register(retriever_name, retriever)

            await asyncio.sleep(delay)

            retrieved = registry.get_retriever(retriever_name)
            if retrieved is None:
                errors.append(f"Async Request {request_id}: get_retriever returned None!")
                results.append((request_id, False))
            else:
                results.append((request_id, True))

            registry.unregister(retriever_name)

        await asyncio.gather(
            simulate_async_request(1, 0.0),
            simulate_async_request(2, 0.05),
            simulate_async_request(3, 0.02),
            simulate_async_request(4, 0.03),
        )

        failed = [r for r in results if not r[1]]
        assert len(errors) == 0, f"Async race condition errors: {errors}"
        assert len(failed) == 0, f"Failed async requests: {failed}"

    def test_reference_counting_basic(self):
        """Test that reference counting prevents premature unregistration."""
        registry = RetrieverRegistry()
        retriever_name = "RefCountedRetriever"

        # Simulate 3 requests registering the same retriever
        retriever = DummyRetriever(name="test")
        registry.register(retriever_name, retriever)
        registry.register(retriever_name, retriever)
        registry.register(retriever_name, retriever)

        # First two unregisters should NOT remove the retriever
        registry.unregister(retriever_name)
        assert registry.get_retriever(retriever_name) is not None, "Retriever was removed too early (after 1st unregister)"

        registry.unregister(retriever_name)
        assert registry.get_retriever(retriever_name) is not None, "Retriever was removed too early (after 2nd unregister)"

        # Third unregister should remove the retriever
        registry.unregister(retriever_name)
        assert registry.get_retriever(retriever_name) is None, "Retriever should be removed after all unregisters"

    def test_registered_names_list_handling(self):
        """Test that registered_names list is properly handled with reference counting."""
        registry = RetrieverRegistry()
        retriever_name = "DuplicateTestRetriever"
        retriever = DummyRetriever(name="test")

        # Register the same retriever multiple times
        registry.register(retriever_name, retriever)
        registry.register(retriever_name, retriever)
        registry.register(retriever_name, retriever)

        # Count how many times the name appears in registered_names
        name_count = registry.registered_names.count(retriever_name)

        # With reference counting, the name should only appear once in the list
        assert name_count == 1, f"Name appears {name_count} times in registered_names (should be 1)"

    def test_thread_safety_stress(self):
        """Stress test with many concurrent operations."""
        registry = RetrieverRegistry()
        errors = []
        num_threads = 20
        operations_per_thread = 50

        def thread_worker(thread_id: int):
            retriever_name = "StressTestRetriever"
            for i in range(operations_per_thread):
                try:
                    retriever = DummyRetriever(name=f"retriever_{thread_id}_{i}")
                    registry.register(retriever_name, retriever)
                    # Verify registration succeeded
                    if registry.get_retriever(retriever_name) is None:
                        errors.append(f"Thread {thread_id}, op {i}: Retriever not available after register()")
                    registry.unregister(retriever_name)
                except Exception as e:
                    errors.append(f"Thread {thread_id}, op {i}: Exception: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Thread safety errors: {errors[:10]}..."


class TestGlobalRetrieverRegistryConcurrency:
    """Tests for the global RetrieverRegistry singleton under concurrent access."""

    def test_global_retriever_registry_concurrent_access(self):
        """Test that the global RetrieverRegistry handles concurrent access correctly."""
        registry = get_retriever_registry()
        errors = []
        unique_retriever_name = "GlobalRetrieverTest_" + str(id(self))

        def worker(worker_id: int):
            try:
                retriever = DummyRetriever(name=f"retriever_{worker_id}")
                registry.register(unique_retriever_name, retriever)
                import time
                time.sleep(0.01 * (worker_id % 3))
                retrieved = registry.get_retriever(unique_retriever_name)
                if retrieved is None:
                    errors.append(f"Worker {worker_id}: get_retriever returned None")
                registry.unregister(unique_retriever_name)
            except Exception as e:
                errors.append(f"Worker {worker_id}: Exception: {e}")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(worker, i) for i in range(6)]
            for future in as_completed(futures):
                future.result()

        assert len(errors) == 0, f"Global retriever registry errors: {errors}"


if __name__ == "__main__":
    pytest.main(["-v", "--tb=short", __file__])
