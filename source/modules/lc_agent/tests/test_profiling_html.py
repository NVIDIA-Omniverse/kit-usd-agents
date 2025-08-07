## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from lc_agent import (
    RunnableNetwork,
    RunnableNode,
    NetworkNode,
    ProfilingFrame,
    ProfilingData,
    enable_profiling,
    create_profiling_html,
)
# Note: Internal functions are tested indirectly through create_profiling_html


class TestHTMLGeneration:
    """Test HTML generation functionality."""

    def setup_method(self):
        """Enable profiling for tests."""
        enable_profiling()

    def test_html_from_network_object(self):
        """Test generating HTML from RunnableNetwork object."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        frame = ProfilingFrame(
            name="test_operation",
            frame_type="network",
            start_time=0.0,
            end_time=1.0,
            duration=1.0
        )
        network.profiling.add_frame(frame)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network, f.name)
            
            # Check file was created
            assert Path(output_path).exists()
            
            # Check content
            content = Path(output_path).read_text()
            assert "<!DOCTYPE html>" in content
            assert "LC Agent Profiling" in content
            assert "test_operation" in content
            assert "Total Duration:" in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_from_dict(self):
        """Test generating HTML from dictionary."""
        network_data = {
            "__node_type__": "TestNetwork",
            "profiling": {
                "enabled": True,
                "frames": [
                    {
                        "name": "main_operation",
                        "frame_type": "network",
                        "start_time": 100.0,
                        "end_time": 105.0,
                        "duration": 5.0,
                        "metadata": {"network_id": "test123"},
                        "children": [
                            {
                                "name": "sub_operation",
                                "frame_type": "node",
                                "start_time": 101.0,
                                "end_time": 103.0,
                                "duration": 2.0,
                                "metadata": {"node_name": "TestNode"},
                                "children": []
                            }
                        ]
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network_data, f.name)
            
            content = Path(output_path).read_text()
            
            # Check frame data is properly embedded
            assert "main_operation" in content
            assert "sub_operation" in content
            assert '"type": "network"' in content
            assert '"type": "node"' in content
            
            # Check metadata is included
            assert "network_id" in content
            assert "node_name" in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_error_no_profiling_data(self):
        """Test error when no profiling data is available."""
        network = RunnableNetwork()
        
        with pytest.raises(ValueError, match="No profiling data available"):
            create_profiling_html(network, "test.html")

    def test_html_error_empty_frames(self):
        """Test error when profiling has no frames."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        with pytest.raises(ValueError, match="No profiling data available"):
            create_profiling_html(network, "test.html")

    def test_html_with_incomplete_frames(self):
        """Test HTML generation with incomplete frames (no end_time)."""
        network_data = {
            "profiling": {
                "frames": [
                    {
                        "name": "complete_frame",
                        "frame_type": "network",
                        "start_time": 0.0,
                        "end_time": 5.0,
                        "duration": 5.0,
                        "children": [
                            {
                                "name": "incomplete_frame",
                                "frame_type": "node",
                                "start_time": 1.0,
                                # No end_time - incomplete
                                "children": []
                            }
                        ]
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network_data, f.name)
            
            content = Path(output_path).read_text()
            
            # Complete frame should be included
            assert "complete_frame" in content
            
            # Incomplete frame should be skipped
            assert "incomplete_frame" not in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_with_nested_networks(self):
        """Test HTML generation with nested networks."""
        network_data = {
            "profiling": {
                "frames": [
                    {
                        "name": "parent_network",
                        "frame_type": "network",
                        "start_time": 0.0,
                        "end_time": 10.0,
                        "duration": 10.0,
                        "children": []
                    }
                ]
            },
            "nodes": [
                {
                    "name": "SubNetwork",
                    "nodes": [  # This node is also a network
                        {"name": "InnerNode"}
                    ],
                    "profiling": {
                        "frames": [
                            {
                                "name": "sub_network_op",
                                "frame_type": "network",
                                "start_time": 2.0,
                                "end_time": 7.0,
                                "duration": 5.0,
                                "children": []
                            }
                        ]
                    }
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network_data, f.name)
            
            content = Path(output_path).read_text()
            
            # Both network frames should be included
            assert "parent_network" in content
            assert "sub_network_op" in content
            
            # Separator should be added for nested network
            assert "=== SubNetwork ===" in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_javascript_features(self):
        """Test that JavaScript features are properly included."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        frame = ProfilingFrame(
            name="test",
            frame_type="custom",
            start_time=0.0,
            end_time=1.0,
            duration=1.0
        )
        network.profiling.add_frame(frame)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network, f.name)
            
            content = Path(output_path).read_text()
            
            # Check JavaScript functions
            assert "function zoom(" in content
            assert "function resetZoom(" in content
            assert "function createTimeRuler(" in content
            assert "function formatTime(" in content
            assert "function renderFrames(" in content
            assert "function showTooltip(" in content
            assert "function updateSeparatorTextPositions(" in content
            
            # Check event handlers
            assert "addEventListener('mousedown'" in content
            assert "addEventListener('mousemove'" in content
            assert "addEventListener('mouseup'" in content
            assert "addEventListener('wheel'" in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_css_frame_types(self):
        """Test that all frame type CSS classes are included."""
        network = RunnableNetwork()
        network.profiling = ProfilingData()
        
        # Add one frame of each type
        frame_types = [
            "network", "node", "modifier", "chunk", "custom",
            "process_parents", "combine_inputs", "retriever"
        ]
        
        for i, frame_type in enumerate(frame_types):
            frame = ProfilingFrame(
                name=f"test_{frame_type}",
                frame_type=frame_type,
                start_time=float(i),
                end_time=float(i + 0.5),
                duration=0.5
            )
            network.profiling.add_frame(frame)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network, f.name)
            
            content = Path(output_path).read_text()
            
            # Check CSS classes for all frame types
            for frame_type in frame_types:
                assert f".frame-{frame_type}" in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_duration_formatting(self):
        """Test duration formatting in HTML."""
        network_data = {
            "profiling": {
                "frames": [
                    {
                        "name": "microseconds",
                        "frame_type": "custom",
                        "start_time": 0.0,
                        "end_time": 0.0000005,
                        "duration": 0.0000005,  # 0.5 μs
                        "children": []
                    },
                    {
                        "name": "milliseconds",
                        "frame_type": "custom",
                        "start_time": 1.0,
                        "end_time": 1.0123,
                        "duration": 0.0123,  # 12.3 ms
                        "children": []
                    },
                    {
                        "name": "seconds",
                        "frame_type": "custom",
                        "start_time": 2.0,
                        "end_time": 4.567,
                        "duration": 2.567,  # 2.567 s
                        "children": []
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network_data, f.name)
            
            content = Path(output_path).read_text()
            
            # Duration strings should be properly formatted
            # The duration strings are embedded in JavaScript, so check the frames array
            import re
            
            # Find the frames array in the JavaScript
            frames_match = re.search(r'const frames = (\[.*?\]);', content, re.DOTALL)
            assert frames_match, "Could not find frames data in HTML"
            
            frames_json = frames_match.group(1)
            # Check for microseconds - the μ character might be encoded
            assert ('0.5' in frames_json and ('μs' in frames_json or '\u03bc' in frames_json or 'us' in frames_json))
            # Check for milliseconds
            assert '12.3ms' in frames_json
            # Check for seconds
            assert '2.567s' in frames_json
            
            # Clean up
            Path(output_path).unlink()


class TestHTMLVisualizationIntegration:
    """Test integration of HTML visualization with real network execution."""

    def setup_method(self):
        """Enable profiling for tests."""
        enable_profiling()

    @pytest.mark.asyncio
    async def test_html_from_real_network_execution(self):
        """Test generating HTML from actual network execution."""
        import asyncio
        from lc_agent import get_chat_model_registry
        from langchain_community.chat_models.fake import FakeListChatModel
        
        # Register fake chat model
        get_chat_model_registry().register(
            "Fake",
            FakeListChatModel(name="Fake", responses=["I am happy", "who are you", "me too"]),
        )
        
        network = RunnableNetwork(chat_model_name="Fake")
        
        # Create a node that produces chunks
        class StreamingNode(RunnableNode):
            async def _astream(self, *args, **kwargs):
                for i in range(3):
                    yield f"chunk_{i}"
                    await asyncio.sleep(0.001)
        
        node = StreamingNode()
        network.add_node(node)
        
        # Execute network
        chunks = []
        async for chunk in network.astream({}):
            chunks.append(chunk)
        
        # Generate HTML
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            output_path = create_profiling_html(network, f.name)
            
            content = Path(output_path).read_text()
            
            # Check that actual profiling data is included
            assert "network_astream" in content
            assert "StreamingNode" in content
            assert "chunk_" in content
            
            # Check that timing data is reasonable
            assert '"duration":' in content
            assert not '"duration": null' in content
            
            # Clean up
            Path(output_path).unlink()

    def test_html_debug_mode(self):
        """Test HTML generation with debug mode enabled."""
        network_data = {
            "profiling": {
                "frames": [{
                    "name": "test",
                    "frame_type": "network",
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "duration": 1.0,
                    "children": []
                }]
            },
            "nodes": [
                {"name": "Node1"},
                {"name": "Node2", "subnetwork": {
                    "profiling": {"frames": []}
                }}
            ]
        }
        
        # Capture debug output
        import io
        import sys
        captured_output = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
                output_path = create_profiling_html(network_data, f.name, debug=True)
                
                # Check debug output
                debug_output = captured_output.getvalue()
                assert "Processing network:" in debug_output
                assert "Found 1 profiling frames" in debug_output
                assert "Checking 2 nodes for sub-networks" in debug_output
                assert "Debug Summary:" in debug_output
                
                # Clean up
                Path(output_path).unlink()
        finally:
            sys.stdout = old_stdout