## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Tests for concurrent network isolation.

These tests verify that the RunnableNetwork context management works correctly
when multiple networks are used concurrently in async operations. This addresses
the race condition bug where mutable list operations in ContextVar could cause
concurrent workflows to interfere with each other.

See: BUG_REPORT_LC_AGENT_CONCURRENCY.md for details on the original issue.
"""

import asyncio
import pytest
from typing import Dict, Any, AsyncIterator
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_node import RunnableNode
from langchain_core.messages import AIMessage


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

    async def astream(self, input: Dict[str, Any] = {}, config=None, **kwargs) -> AsyncIterator[AIMessage]:
        if not self.invoked:
            self.outputs = AIMessage(content=f"Response from {self.name or 'unnamed'}")
            self.invoked = True
        yield self.outputs

    def __hash__(self):
        return hash(id(self))


@pytest.mark.asyncio
async def test_concurrent_network_isolation():
    """Verify that concurrent networks don't interfere with each other.

    This is the primary test for the concurrency bug fix. It verifies that
    when multiple async workflows run concurrently, each network maintains
    its own context and doesn't get corrupted by other networks.
    """
    results = []
    errors = []

    async def workflow(network_id: int, delay: float):
        network = RunnableNetwork()

        with network:
            # Verify correct network is active at start
            active = RunnableNetwork.get_active_network()
            if active is not network:
                errors.append(f"Wrong network active for {network_id} at start")
                return

            DummyNode(name=f"Node_{network_id}")

            await asyncio.sleep(delay)

            # Verify still correct after delay - this is the critical check
            # The bug would cause a different network to be active here
            active_after = RunnableNetwork.get_active_network()
            if active_after is not network:
                errors.append(
                    f"Network changed during execution for {network_id}. "
                    f"Expected {id(network)}, got {id(active_after) if active_after else 'None'}"
                )
                return

            results.append(network_id)

        # Verify network removed after exit (may still have other networks active in parent contexts)

    # Run multiple concurrent workflows with different delays
    # This ensures they overlap and complete in different orders
    await asyncio.gather(
        workflow(1, 0.3),
        workflow(2, 0.1),
        workflow(3, 0.2),
    )

    assert not errors, f"Concurrent isolation errors: {errors}"
    assert sorted(results) == [1, 2, 3], f"Not all workflows completed: {results}"


@pytest.mark.asyncio
async def test_nested_network_context():
    """Verify nested network contexts work correctly."""
    outer = RunnableNetwork()
    inner = RunnableNetwork()

    with outer:
        assert RunnableNetwork.get_active_network() is outer

        with inner:
            assert RunnableNetwork.get_active_network() is inner

            # Both should be in the stack
            networks = list(RunnableNetwork.get_active_networks())
            assert inner in networks
            assert outer in networks

        # After inner exits, outer should be active again
        assert RunnableNetwork.get_active_network() is outer

    # After both exit, no network should be active
    assert RunnableNetwork.get_active_network() is None


@pytest.mark.asyncio
async def test_node_registration_isolation():
    """Verify nodes are registered to correct network in concurrent contexts.

    This tests that when nodes are created during concurrent async operations,
    each node is added to the correct network, not to a network from a
    different concurrent context.
    """
    errors = []

    async def create_nodes(network_id: int, delay: float):
        network = RunnableNetwork()

        with network:
            node1 = DummyNode(name=f"Node1_{network_id}")
            await asyncio.sleep(delay)
            node2 = DummyNode(name=f"Node2_{network_id}")

            # Verify both nodes are in THIS network
            if node1 not in network.nodes:
                errors.append(f"Node1 not in network {network_id}")
            if node2 not in network.nodes:
                errors.append(f"Node2 not in network {network_id}")

            return len(network.nodes)

    counts = await asyncio.gather(
        create_nodes(1, 0.2),
        create_nodes(2, 0.1),
    )

    assert not errors, f"Node registration errors: {errors}"
    # Each network should have exactly 2 nodes
    assert all(c == 2 for c in counts), f"Wrong node counts: {counts}, expected [2, 2]"


@pytest.mark.asyncio
async def test_async_context_manager():
    """Verify async context manager works correctly for concurrent operations."""
    results = []

    async def workflow(network_id: int, delay: float):
        network = RunnableNetwork()

        async with network:
            active = RunnableNetwork.get_active_network()
            assert active is network, f"Wrong network active for {network_id}"

            await asyncio.sleep(delay)

            active_after = RunnableNetwork.get_active_network()
            assert active_after is network, f"Network changed for {network_id}"

            results.append(network_id)

    await asyncio.gather(
        workflow(1, 0.15),
        workflow(2, 0.05),
        workflow(3, 0.10),
    )

    assert sorted(results) == [1, 2, 3]


@pytest.mark.asyncio
async def test_get_active_networks_isolation():
    """Verify get_active_networks returns correct networks for each context."""
    errors = []

    async def workflow(network_id: int, delay: float):
        outer_network = RunnableNetwork()
        inner_network = RunnableNetwork()

        with outer_network:
            with inner_network:
                await asyncio.sleep(delay)

                # Get all active networks for this context
                networks = list(RunnableNetwork.get_active_networks())

                # Should have exactly inner and outer (in that order - most recent first)
                if len(networks) < 2:
                    errors.append(f"Network {network_id}: Expected at least 2 networks, got {len(networks)}")
                elif networks[0] is not inner_network:
                    errors.append(f"Network {network_id}: Inner network not at top of stack")
                elif networks[1] is not outer_network:
                    errors.append(f"Network {network_id}: Outer network not in stack")

    await asyncio.gather(
        workflow(1, 0.2),
        workflow(2, 0.1),
    )

    assert not errors, f"Active networks isolation errors: {errors}"


@pytest.mark.asyncio
async def test_many_concurrent_networks():
    """Stress test with many concurrent networks to verify isolation at scale."""
    num_workflows = 20
    results = []
    errors = []

    async def workflow(network_id: int):
        network = RunnableNetwork()
        delay = (network_id % 5) * 0.02  # Variable delays

        with network:
            active = RunnableNetwork.get_active_network()
            if active is not network:
                errors.append(f"Wrong network at start for {network_id}")
                return

            DummyNode(name=f"Node_{network_id}")
            await asyncio.sleep(delay)

            active_after = RunnableNetwork.get_active_network()
            if active_after is not network:
                errors.append(f"Network changed for {network_id}")
                return

            results.append(network_id)

    await asyncio.gather(*[workflow(i) for i in range(num_workflows)])

    assert not errors, f"Errors in stress test: {errors}"
    assert len(results) == num_workflows, f"Only {len(results)} of {num_workflows} workflows completed"


@pytest.mark.asyncio
async def test_exception_handling_in_concurrent_context():
    """Verify context is properly cleaned up even when exceptions occur."""
    cleanup_verified = []

    async def failing_workflow():
        network = RunnableNetwork()
        with network:
            active_before = RunnableNetwork.get_active_network()
            cleanup_verified.append(("failing_start", active_before is network))
            await asyncio.sleep(0.01)
            raise ValueError("Intentional error")

    async def successful_workflow():
        network = RunnableNetwork()
        with network:
            active_before = RunnableNetwork.get_active_network()
            cleanup_verified.append(("successful_start", active_before is network))
            await asyncio.sleep(0.02)
            active_after = RunnableNetwork.get_active_network()
            cleanup_verified.append(("successful_end", active_after is network))
            return True

    # Run both concurrently - one will fail
    results = await asyncio.gather(
        failing_workflow(),
        successful_workflow(),
        return_exceptions=True
    )

    # First task should have raised ValueError
    assert isinstance(results[0], ValueError)
    # Second task should have succeeded
    assert results[1] is True

    # Verify all context checks passed
    for name, result in cleanup_verified:
        assert result, f"Context check failed for {name}"

    # No network should be active after both complete
    assert RunnableNetwork.get_active_network() is None


def test_sync_context_still_works():
    """Verify synchronous context management still works correctly."""
    network1 = RunnableNetwork()
    network2 = RunnableNetwork()

    assert RunnableNetwork.get_active_network() is None

    with network1:
        assert RunnableNetwork.get_active_network() is network1

        with network2:
            assert RunnableNetwork.get_active_network() is network2
            networks = list(RunnableNetwork.get_active_networks())
            assert networks == [network2, network1]

        assert RunnableNetwork.get_active_network() is network1

    assert RunnableNetwork.get_active_network() is None


if __name__ == "__main__":
    pytest.main(["-v", "--tb=short", __file__])
