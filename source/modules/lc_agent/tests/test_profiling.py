## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from lc_agent import (
    RunnableNetwork,
    RunnableNode,
    NetworkNode,
    Profiler,
    ProfilingFrame,
    ProfilingData,
    enable_profiling,
    disable_profiling,
    is_profiling_enabled,
    format_profiling_tree,
    create_profiling_html,
    get_chat_model_registry,
)
from lc_agent.utils.profiling_utils import _profiling_stacks
from langchain_community.chat_models.fake import FakeListChatModel

# Register fake chat model for testing
get_chat_model_registry().register(
    "Fake",
    FakeListChatModel(name="Fake", responses=["I am happy", "who are you", "me too"]),
)


class TestProfilingBasics:
    """Test basic profiling functionality."""

    def setup_method(self):
        """Reset profiling state before each test."""
        disable_profiling()
        # Clear any existing profiling stacks
        _profiling_stacks.set({})

    def teardown_method(self):
        """Clean up after each test."""
        disable_profiling()
        _profiling_stacks.set({})

    def test_profiling_enabled_disabled(self):
        """Test enabling and disabling profiling."""
        assert not is_profiling_enabled()
        
        enable_profiling()
        assert is_profiling_enabled()
        
        disable_profiling()
        assert not is_profiling_enabled()

    def test_profiling_environment_variable(self):
        """Test profiling controlled by environment variable."""
        # Save original value
        original = os.environ.get("LC_AGENT_PROFILING")
        
        try:
            # Test enabling via env var
            os.environ["LC_AGENT_PROFILING"] = "1"
            # Need to reload the module to pick up env var
            import importlib
            import lc_agent.utils.profiling_utils
            importlib.reload(lc_agent.utils.profiling_utils)
            from lc_agent.utils.profiling_utils import is_profiling_enabled as check_enabled
            assert check_enabled()
            
            # Test disabling via env var
            os.environ["LC_AGENT_PROFILING"] = "0"
            importlib.reload(lc_agent.utils.profiling_utils)
            from lc_agent.utils.profiling_utils import is_profiling_enabled as check_disabled
            assert not check_disabled()
        finally:
            # Restore original value
            if original is not None:
                os.environ["LC_AGENT_PROFILING"] = original
            elif "LC_AGENT_PROFILING" in os.environ:
                del os.environ["LC_AGENT_PROFILING"]
            # Reload module one more time to restore state
            import importlib
            import lc_agent.utils.profiling_utils
            importlib.reload(lc_agent.utils.profiling_utils)

    def test_profiling_frame_creation(self):
        """Test ProfilingFrame creation and methods."""
        frame = ProfilingFrame(
            name="test_operation",
            frame_type="custom",
            start_time=1000.0,
            metadata={"key": "value"}
        )
        
        assert frame.name == "test_operation"
        assert frame.frame_type == "custom"
        assert frame.start_time == 1000.0
        assert frame.end_time is None
        assert frame.duration is None
        assert frame.metadata == {"key": "value"}
        assert frame.children == []
        
        # Test closing frame
        frame.close()
        assert frame.end_time is not None
        assert frame.duration is not None
        assert frame.duration == frame.end_time - frame.start_time

    def test_profiling_frame_durations(self):
        """Test frame duration calculations."""
        parent = ProfilingFrame(
            name="parent",
            frame_type="custom",
            start_time=1000.0
        )
        
        child1 = ProfilingFrame(
            name="child1",
            frame_type="custom",
            start_time=1001.0
        )
        child1.end_time = 1002.0
        child1.duration = 1.0
        
        child2 = ProfilingFrame(
            name="child2",
            frame_type="custom",
            start_time=1002.0
        )
        child2.end_time = 1004.0
        child2.duration = 2.0
        
        parent.children = [child1, child2]
        parent.end_time = 1005.0
        parent.duration = 5.0
        
        assert parent.get_total_duration() == 5.0
        assert parent.get_self_duration() == 2.0  # 5.0 - (1.0 + 2.0)

    def test_profiling_data_container(self):
        """Test ProfilingData container."""
        data = ProfilingData()
        assert data.enabled
        assert data.frames == []
        assert data.total_duration is None
        
        # Add frames
        frame1 = ProfilingFrame(
            name="frame1",
            frame_type="custom",
            start_time=1000.0,
            end_time=1002.0,
            duration=2.0
        )
        frame2 = ProfilingFrame(
            name="frame2",
            frame_type="custom",
            start_time=1001.0,
            end_time=1003.0,
            duration=2.0
        )
        
        data.add_frame(frame1)
        data.add_frame(frame2)
        assert len(data.frames) == 2
        
        # Calculate total duration
        data.calculate_total_duration()
        assert data.total_duration == 3.0  # 1003.0 - 1000.0


class TestProfilerContextManager:
    """Test Profiler context manager functionality."""

    def setup_method(self):
        """Reset profiling state before each test."""
        enable_profiling()
        _profiling_stacks.set({})
        
    def teardown_method(self):
        """Clean up after each test."""
        disable_profiling()
        _profiling_stacks.set({})

    def test_profiler_context_manager(self):
        """Test Profiler as context manager."""
        network = RunnableNetwork()
        
        with Profiler("test_op", "custom", network=network) as frame:
            assert frame is not None
            assert frame.name == "test_op"
            assert frame.frame_type == "custom"
            assert frame.start_time is not None
            assert frame.end_time is None
            time.sleep(0.01)  # Small delay to ensure measurable duration
        
        # After context exit
        assert frame.end_time is not None
        assert frame.duration is not None
        assert frame.duration > 0
        
        # Check frame was added to network
        assert network.profiling is not None
        assert len(network.profiling.frames) == 1
        assert network.profiling.frames[0] == frame

    def test_profiler_auto_stop(self):
        """Test Profiler auto-stop on destruction."""
        network = RunnableNetwork()
        
        def create_profiler():
            p = Profiler("auto_stop_test", "custom", network=network)
            assert p._started
            return p.frame
        
        frame = create_profiler()
        # Profiler should auto-stop when it goes out of scope
        assert frame.end_time is not None
        assert frame.duration is not None

    def test_profiler_manual_control(self):
        """Test manual start/stop of Profiler."""
        network = RunnableNetwork()
        
        p = Profiler("manual_test", "custom", network=network, auto_start=False)
        assert not p._started
        assert p.frame is None
        
        # Start manually
        frame = p.start()
        assert p._started
        assert frame is not None
        assert frame.start_time is not None
        assert frame.end_time is None
        
        time.sleep(0.01)
        
        # Stop manually
        p.stop()
        assert not p._started
        assert frame.end_time is not None
        assert frame.duration is not None
        assert frame.duration > 0

    def test_profiler_metadata_update(self):
        """Test updating profiler metadata."""
        network = RunnableNetwork()
        
        with Profiler("metadata_test", "custom", network=network, initial_key="initial") as frame:
            assert frame.metadata["initial_key"] == "initial"
            
            # Update metadata
            p = Profiler("dummy", "custom", network=network, auto_start=False)
            p.frame = frame  # Set frame manually for testing
            p.update_metadata(new_key="new_value", count=42)
            
            assert frame.metadata["new_key"] == "new_value"
            assert frame.metadata["count"] == 42
            assert frame.metadata["initial_key"] == "initial"

    def test_profiler_nested_frames(self):
        """Test nested profiling frames."""
        network = RunnableNetwork()
        
        with Profiler("outer", "custom", network=network) as outer_frame:
            assert len(network.profiling.frames) == 1
            
            with Profiler("inner1", "custom", network=network) as inner1_frame:
                assert len(outer_frame.children) == 1
                assert outer_frame.children[0] == inner1_frame
                
                with Profiler("deep", "custom", network=network) as deep_frame:
                    assert len(inner1_frame.children) == 1
                    assert inner1_frame.children[0] == deep_frame
            
            with Profiler("inner2", "custom", network=network) as inner2_frame:
                assert len(outer_frame.children) == 2
                assert outer_frame.children[1] == inner2_frame

    def test_profiler_idempotent_start(self):
        """Test that Profiler.start() is idempotent."""
        network = RunnableNetwork()
        
        p = Profiler("idempotent_test", "custom", network=network, auto_start=True)
        frame1 = p.frame
        assert frame1 is not None
        
        # Start again should return same frame
        frame2 = p.start()
        assert frame2 is frame1
        
        # Using with context manager after auto_start should work
        with p as frame3:
            assert frame3 is frame1

    @pytest.mark.asyncio
    async def test_profiler_async_context_manager(self):
        """Test Profiler as async context manager."""
        network = RunnableNetwork()
        
        async with Profiler("async_test", "custom", network=network) as frame:
            assert frame is not None
            await asyncio.sleep(0.01)
        
        assert frame.end_time is not None
        assert frame.duration > 0


class TestNetworkProfiling:
    """Test profiling integration with RunnableNetwork."""

    def setup_method(self):
        """Reset profiling state before each test."""
        enable_profiling()
        _profiling_stacks.set({})

    def teardown_method(self):
        """Clean up after each test."""
        disable_profiling()
        _profiling_stacks.set({})

    @pytest.mark.asyncio
    async def test_network_astream_profiling(self):
        """Test profiling of network astream execution."""
        network = RunnableNetwork(chat_model_name="Fake")
        
        # Since we're using a fake model, the node will stream the model's response
        # The fake model returns "I am happy" which will be streamed character by character
        node = RunnableNode()
        network.add_node(node)
        
        # Execute with profiling
        chunks = []
        async for chunk in network.astream({}):
            chunks.append(chunk)
        
        # The fake model response "I am happy" should produce multiple chunks
        assert len(chunks) > 0
        
        # Check profiling data
        assert network.profiling is not None
        assert len(network.profiling.frames) == 1
        
        # Check network frame
        network_frame = network.profiling.frames[0]
        assert network_frame.name == "network_astream"
        assert network_frame.frame_type == "network"
        assert network_frame.duration > 0
        
        # Check node frame
        assert len(network_frame.children) > 0
        node_frames = [f for f in network_frame.children if f.frame_type == "node"]
        assert len(node_frames) >= 1  # At least the RunnableNode
        
        # Check chunk frames exist
        all_chunk_frames = []
        for node_frame in node_frames:
            chunk_frames = [f for f in node_frame.children if f.frame_type == "chunk"]
            all_chunk_frames.extend(chunk_frames)
        assert len(all_chunk_frames) >= 3  # At least some chunk frames

    @pytest.mark.asyncio
    async def test_modifier_profiling(self):
        """Test profiling of network modifiers."""
        network = RunnableNetwork(chat_model_name="Fake")
        
        # Track modifier calls
        modifier_calls = []
        
        class TestModifier:
            async def on_begin_invoke_async(self, network):
                pass
            
            async def on_pre_invoke_async(self, network, node):
                modifier_calls.append("pre")
                await asyncio.sleep(0.01)
            
            async def on_post_invoke_async(self, network, node):
                modifier_calls.append("post")
                await asyncio.sleep(0.01)
            
            async def on_end_invoke_async(self, network):
                pass
        
        network.add_modifier(TestModifier())
        
        node = RunnableNode()
        network.add_node(node)
        
        # Execute
        async for _ in network.astream({}):
            pass
        
        assert modifier_calls == ["pre", "post"]
        
        # Check profiling
        network_frame = network.profiling.frames[0]
        node_frame = next(f for f in network_frame.children if f.frame_type == "node")
        
        modifier_frames = [f for f in node_frame.children if f.frame_type == "modifier"]
        assert len(modifier_frames) >= 2
        
        pre_frames = [f for f in modifier_frames if "pre_invoke" in f.name]
        post_frames = [f for f in modifier_frames if "post_invoke" in f.name]
        
        assert len(pre_frames) > 0
        assert len(post_frames) > 0
        assert all(f.duration > 0 for f in modifier_frames)

    def test_per_network_profiling_stacks(self):
        """Test that each network maintains its own profiling stack."""
        # This tests the fix for the issue where frames from nested networks
        # would incorrectly go to the parent network
        
        # Create two independent networks
        network1 = RunnableNetwork(chat_model_name="Fake")
        network2 = RunnableNetwork(chat_model_name="Fake")
        
        # Start profiling in network1
        with Profiler("network1_op", "custom", network=network1) as frame1:
            assert network1.profiling is not None
            assert len(network1.profiling.frames) == 1
            assert network1.profiling.frames[0] == frame1
            
            # Start profiling in network2 - should be independent
            with Profiler("network2_op", "custom", network=network2) as frame2:
                assert network2.profiling is not None
                assert len(network2.profiling.frames) == 1
                assert network2.profiling.frames[0] == frame2
                
                # Nested profiling in network1 should go to network1, not network2
                with Profiler("network1_nested", "custom", network=network1) as nested1:
                    assert len(frame1.children) == 1
                    assert frame1.children[0] == nested1
                    
                    # network2's frame should have no children
                    assert len(frame2.children) == 0
                
                # Nested profiling in network2 should go to network2, not network1
                with Profiler("network2_nested", "custom", network=network2) as nested2:
                    assert len(frame2.children) == 1
                    assert frame2.children[0] == nested2
                    
                    # network1's frame should still have only its own child
                    assert len(frame1.children) == 1
        
        # Verify final state
        assert len(network1.profiling.frames) == 1
        assert len(network1.profiling.frames[0].children) == 1
        assert network1.profiling.frames[0].children[0].name == "network1_nested"
        
        assert len(network2.profiling.frames) == 1
        assert len(network2.profiling.frames[0].children) == 1
        assert network2.profiling.frames[0].children[0].name == "network2_nested"


class TestProfilingTreeFormatter:
    """Test profiling tree formatting."""

    def setup_method(self):
        """Reset profiling state before each test."""
        enable_profiling()
        _profiling_stacks.set({})

    def teardown_method(self):
        """Clean up after each test."""
        disable_profiling()
        _profiling_stacks.set({})

    def test_format_profiling_tree_empty(self):
        """Test formatting empty profiling data."""
        network = RunnableNetwork()
        result = format_profiling_tree(network)
        assert result == "No profiling data available"

    def test_format_profiling_tree_simple(self):
        """Test formatting simple profiling tree."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        frame = ProfilingFrame(
            name="test_operation",
            frame_type="custom",
            start_time=1000.0,
            end_time=1002.0,
            duration=2.0,
            metadata={"key": "value"}
        )
        network.profiling.add_frame(frame)
        
        result = format_profiling_tree(network)
        assert "test_operation [custom] - 2.000s" in result
        assert "Profiling Results:" in result
        assert "-" * 50 in result

    def test_format_profiling_tree_nested(self):
        """Test formatting nested profiling tree."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        parent = ProfilingFrame(
            name="parent_op",
            frame_type="network",
            start_time=1000.0,
            end_time=1003.0,
            duration=3.0
        )
        
        child1 = ProfilingFrame(
            name="child1_op",
            frame_type="node",
            start_time=1000.5,
            end_time=1001.5,
            duration=1.0,
            metadata={"node_id": "123", "node_name": "TestNode"}
        )
        
        child2 = ProfilingFrame(
            name="child2_op",
            frame_type="modifier",
            start_time=1001.5,
            end_time=1002.5,
            duration=1.0,
            metadata={"modifier_name": "TestModifier"}
        )
        
        parent.children = [child1, child2]
        network.profiling.add_frame(parent)
        
        result = format_profiling_tree(network)
        lines = result.split("\n")
        
        # Check structure
        assert any("parent_op [network] - 3.000s" in line for line in lines)
        assert any("child1_op [node] - 1.000s" in line and "node_name=TestNode" in line for line in lines)
        assert any("child2_op [modifier] - 1.000s" in line and "modifier_name=TestModifier" in line for line in lines)
        
        # Check indentation
        parent_line_idx = next(i for i, line in enumerate(lines) if "parent_op" in line)
        child1_line_idx = next(i for i, line in enumerate(lines) if "child1_op" in line)
        child2_line_idx = next(i for i, line in enumerate(lines) if "child2_op" in line)
        
        assert child1_line_idx > parent_line_idx
        assert child2_line_idx > parent_line_idx
        assert lines[child1_line_idx].index("child1_op") > lines[parent_line_idx].index("parent_op")


class TestProfilingHTMLGeneration:
    """Test HTML profiling visualization generation."""

    def setup_method(self):
        """Reset profiling state before each test."""
        enable_profiling()
        _profiling_stacks.set({})

    def teardown_method(self):
        """Clean up after each test."""
        disable_profiling()
        _profiling_stacks.set({})

    def test_create_profiling_html_no_data(self):
        """Test HTML generation with no profiling data."""
        network = RunnableNetwork()
        
        with pytest.raises(ValueError, match="No profiling data available"):
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                create_profiling_html(network, f.name)

    def test_create_profiling_html_basic(self):
        """Test basic HTML generation."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        frame = ProfilingFrame(
            name="test_op",
            frame_type="custom",
            start_time=1000.0,
            end_time=1002.0,
            duration=2.0
        )
        network.profiling.add_frame(frame)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network, f.name)
            
            assert Path(output_path).exists()
            content = Path(output_path).read_text()
            
            # Check basic HTML structure
            assert "<html" in content
            assert "LC Agent Profiling" in content
            assert "test_op" in content
            
            # Check JavaScript data
            assert "frames = " in content
            assert '"name": "test_op"' in content
            assert '"type": "custom"' in content
            
            # Clean up
            Path(output_path).unlink()

    def test_create_profiling_html_from_dict(self):
        """Test HTML generation from dictionary (JSON) data."""
        network_dict = {
            "profiling": {
                "enabled": True,
                "frames": [
                    {
                        "name": "dict_test",
                        "frame_type": "network",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "duration": 1.0,
                        "children": []
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network_dict, f.name)
            
            assert Path(output_path).exists()
            content = Path(output_path).read_text()
            assert "dict_test" in content
            
            # Clean up
            Path(output_path).unlink()

    def test_create_profiling_html_complex(self):
        """Test HTML generation with complex nested data."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        # Create complex frame hierarchy
        root = ProfilingFrame(
            name="root",
            frame_type="network",
            start_time=0.0,
            end_time=10.0,
            duration=10.0
        )
        
        for i in range(3):
            node_frame = ProfilingFrame(
                name=f"node_{i}",
                frame_type="node",
                start_time=i * 3.0,
                end_time=(i + 1) * 3.0,
                duration=3.0,
                metadata={"node_id": f"id_{i}"}
            )
            
            for j in range(2):
                chunk_frame = ProfilingFrame(
                    name=f"chunk_{j}",
                    frame_type="chunk",
                    start_time=i * 3.0 + j * 1.0,
                    end_time=i * 3.0 + (j + 1) * 1.0,
                    duration=1.0,
                    metadata={"content": f"chunk content {j}"}
                )
                node_frame.children.append(chunk_frame)
            
            root.children.append(node_frame)
        
        network.profiling.add_frame(root)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network, f.name)
            
            content = Path(output_path).read_text()
            
            # Check all frames are present
            assert "root" in content
            for i in range(3):
                assert f"node_{i}" in content
            assert "chunk_0" in content
            assert "chunk_1" in content
            
            # Check frame colors are defined
            assert "frame-network" in content
            assert "frame-node" in content
            assert "frame-chunk" in content
            
            # Clean up
            Path(output_path).unlink()


class TestProfilingSerialization:
    """Test profiling data serialization/deserialization."""

    def setup_method(self):
        """Reset profiling state before each test."""
        enable_profiling()
        _profiling_stacks.set({})

    def teardown_method(self):
        """Clean up after each test."""
        disable_profiling()
        _profiling_stacks.set({})

    def test_profiling_serialization(self):
        """Test that profiling data is properly serialized with network."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        frame = ProfilingFrame(
            name="test_frame",
            frame_type="custom",
            start_time=1000.0,
            end_time=1002.0,
            duration=2.0,
            metadata={"key": "value"}
        )
        network.profiling.add_frame(frame)
        
        # Serialize
        json_str = network.model_dump_json()
        data = json.loads(json_str)
        
        # Check profiling data is included
        assert "profiling" in data
        assert data["profiling"]["enabled"] is True
        assert len(data["profiling"]["frames"]) == 1
        assert data["profiling"]["frames"][0]["name"] == "test_frame"
        
        # Deserialize
        loaded_network = RunnableNetwork.model_validate_json(json_str)
        
        # Check profiling data is restored
        assert loaded_network.profiling is not None
        assert loaded_network.profiling.enabled is True
        assert len(loaded_network.profiling.frames) == 1
        assert loaded_network.profiling.frames[0].name == "test_frame"
        assert loaded_network.profiling.frames[0].duration == 2.0


class TestProfilingWithDisabled:
    """Test behavior when profiling is disabled."""

    def setup_method(self):
        """Ensure profiling is disabled."""
        disable_profiling()
        _profiling_stacks.set({})

    def teardown_method(self):
        """Clean up."""
        _profiling_stacks.set({})

    def test_profiler_when_disabled(self):
        """Test Profiler behavior when profiling is disabled."""
        network = RunnableNetwork()
        
        with Profiler("test", "custom", network=network) as frame:
            assert frame is None
        
        # Network should not have profiling data
        assert network.profiling is None

    @pytest.mark.asyncio
    async def test_network_execution_when_disabled(self):
        """Test network execution with profiling disabled."""
        network = RunnableNetwork(chat_model_name="Fake")
        node = RunnableNode()
        network.add_node(node)
        
        # Execute
        async for _ in network.astream({}):
            pass
        
        # No profiling data should be created
        assert network.profiling is None