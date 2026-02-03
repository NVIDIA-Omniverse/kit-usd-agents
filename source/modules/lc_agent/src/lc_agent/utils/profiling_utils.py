## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Profiling utilities for LC Agent networks and nodes.

This module provides a profiling system that integrates with RunnableNetwork
to capture timing and execution data. Profiling data is stored directly in
the network's `profiling` field and is automatically serialized/deserialized
with the network.
"""

from functools import wraps
from langsmith import traceable as langsmith_traceable
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, TYPE_CHECKING, Callable
import contextvars
import inspect
import os
import time

if TYPE_CHECKING:
    from ..runnable_network import RunnableNetwork


def _force_single_task_langsmith_streaming() -> None:
    """Ensure LangSmith does not use asyncio.create_task(context=...) during streaming.

    LangSmith's run_helpers uses langsmith._internal._aiter.asyncio_accepts_context()
    to decide whether to call asyncio.create_task for each generator step.
    For compatibility with anyio-based clients (e.g., NAT MCP), we force this
    probe to return False so streaming stays within a single task.
    """
    try:
        from langsmith._internal import _aiter as _ls_aiter  # type: ignore

        def _no_context() -> bool:
            return False

        _ls_aiter.asyncio_accepts_context = _no_context  # type: ignore[attr-defined]
    except Exception:
        pass


# Apply the patch at import time so all decorators pick it up
_force_single_task_langsmith_streaming()

# Thread-local storage for profiling stacks per network
# Note: We don't use a default dict here because it would be shared across contexts.
# Instead, we'll create a new dict on first access in each context.
_profiling_stacks = contextvars.ContextVar("profiling_stacks")

# Global profiling state (can be controlled via environment variable)
_PROFILING_ENABLED = os.environ.get("LC_AGENT_PROFILING", "").lower() in ("1", "true", "yes", "on")


class ProfilingFrame(BaseModel):
    """
    Represents a single profiling measurement with timing data and metadata.

    Attributes:
        name: Descriptive name for this measurement
        frame_type: Type of frame (e.g., "network", "modifier", "node", "chunk")
        start_time: Time when measurement started (from time.perf_counter())
        end_time: Time when measurement ended
        duration: Calculated duration in seconds
        metadata: Additional context data
        children: Nested profiling frames
    """

    name: str
    frame_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    children: List["ProfilingFrame"] = Field(default_factory=list)

    def close(self) -> None:
        """Mark frame as complete and calculate duration."""
        if self.end_time is None:
            self.end_time = time.perf_counter()
            self.duration = self.end_time - self.start_time

    def get_total_duration(self) -> float:
        """Get total duration including all children."""
        if self.duration is None:
            return 0.0
        return self.duration

    def get_self_duration(self) -> float:
        """Get duration excluding children."""
        if self.duration is None:
            return 0.0
        children_duration = sum(child.get_total_duration() for child in self.children)
        return max(0.0, self.duration - children_duration)

    class Config:
        # For Pydantic v1 compatibility
        json_encoders = {float: lambda v: round(v, 6) if v is not None else None}


# Update forward refs for recursive model
ProfilingFrame.model_rebuild()


class ProfilingData(BaseModel):
    """
    Container for all profiling information in a network.

    Attributes:
        enabled: Whether profiling was enabled during execution
        frames: Root-level profiling frames
        total_duration: Total execution time
    """

    enabled: bool = True
    frames: List[ProfilingFrame] = Field(default_factory=list)
    total_duration: Optional[float] = None

    def add_frame(self, frame: ProfilingFrame) -> None:
        """Add a root-level frame."""
        self.frames.append(frame)

    def calculate_total_duration(self) -> None:
        """Calculate total duration from all root frames."""
        if self.frames:
            start_time = min(frame.start_time for frame in self.frames)
            end_time = max(frame.end_time for frame in self.frames if frame.end_time is not None)
            self.total_duration = end_time - start_time if end_time else None


class Profiler:
    """
    Context manager for automatic profiling of code blocks.

    Usage:
        # As context manager:
        with Profiler("operation_name", "operation_type", metadata_key="value"):
            # Code to profile
            pass

        # Auto-stop on destruction (default):
        def foo():
            p = Profiler("operation_name", "operation_type")
            # Code to profile
            # p automatically stops when destroyed at function exit

        # Manual start/stop:
        p = Profiler("operation_name", "operation_type", auto_start=False)
        p.start()
        # Code to profile
        p.stop()

    The profiling data is automatically attached to the active RunnableNetwork.
    """

    def __init__(
        self,
        name: str,
        frame_type: str = "custom",
        network: Optional["RunnableNetwork"] = None,
        auto_start: bool = True,
        **metadata,
    ):
        """
        Initialize profiler.

        Args:
            name: Descriptive name for this measurement
            frame_type: Type of frame (e.g., "modifier", "chunk", "node", "network")
            network: The network to attach profiling to (if None, uses current context)
            auto_start: If True, automatically start profiling on creation
            **metadata: Additional metadata to store with the frame
        """
        self.name = name
        self.frame_type = frame_type
        self.network = network
        self.metadata = metadata
        self.frame: Optional[ProfilingFrame] = None
        self._started = False

        if auto_start:
            self.start()

    def start(self) -> Optional[ProfilingFrame]:
        """Start profiling and return the frame."""
        if self._started:
            return self.frame

        if not is_profiling_enabled():
            return None

        # Import here to avoid circular dependency
        from ..runnable_network import RunnableNetwork

        # Use provided network or get from context (for backward compatibility)
        if not self.network:
            # This is only for backward compatibility - new code should pass network explicitly
            self.network = RunnableNetwork.get_active_network()
            if not self.network:
                return None

        # Initialize profiling if needed
        if self.network.profiling is None:
            self.network.profiling = ProfilingData(enabled=True)

        # Create frame
        self.frame = ProfilingFrame(
            name=self.name, frame_type=self.frame_type, start_time=time.perf_counter(), metadata=self.metadata
        )

        # Get all stacks (create new dict if none exists in this context)
        try:
            all_stacks = _profiling_stacks.get()
        except LookupError:
            all_stacks = {}
            _profiling_stacks.set(all_stacks)

        # Get stack for THIS network only (using object id as key)
        network_id = self.network.uuid()
        stack = all_stacks.get(network_id, [])

        if stack:
            # Add as child to current frame IN THIS NETWORK
            parent_frame = stack[-1]
            parent_frame.children.append(self.frame)
        else:
            # Add as root frame to THIS network
            self.network.profiling.add_frame(self.frame)

        # Update stack for this network
        new_stack = stack + [self.frame]
        all_stacks = dict(all_stacks)  # Create a copy
        all_stacks[network_id] = new_stack
        _profiling_stacks.set(all_stacks)

        self._started = True
        return self.frame

    def stop(self) -> None:
        """Stop profiling and finalize the frame."""
        if not self._started:
            return

        if not self.frame or not self.network:
            return

        # Close frame
        self.frame.close()

        # Pop from THIS network's stack
        try:
            all_stacks = _profiling_stacks.get()
        except LookupError:
            # No stacks were ever created in this context, nothing to do
            return

        network_id = self.network.uuid()
        stack = all_stacks.get(network_id, [])

        if stack and self.frame in stack:
            # Find and remove this frame from the stack
            all_stacks = dict(all_stacks)  # Create a copy
            new_stack = []
            for frame in stack:
                if frame is not self.frame:
                    new_stack.append(frame)
                else:
                    # Found our frame, stop adding to new_stack
                    break

            all_stacks[network_id] = new_stack
            # Clean up empty stacks
            if not all_stacks[network_id]:
                del all_stacks[network_id]
            _profiling_stacks.set(all_stacks)

        self._started = False

    def update_metadata(self, **metadata) -> None:
        """Update frame metadata after creation."""
        if self.frame:
            self.frame.metadata.update(metadata)

    def __enter__(self) -> Optional[ProfilingFrame]:
        """Start profiling and return the frame."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop profiling and finalize the frame."""
        self.stop()

    async def __aenter__(self) -> Optional[ProfilingFrame]:
        """Async context manager support."""
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager support."""
        self.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self) -> None:
        """Automatically stop profiling when object is destroyed."""
        if self._started:
            self.stop()


def enable_profiling() -> None:
    """Enable profiling globally."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = True


def disable_profiling() -> None:
    """Disable profiling globally."""
    global _PROFILING_ENABLED
    _PROFILING_ENABLED = False


def is_profiling_enabled() -> bool:
    """Check if profiling is enabled."""
    return _PROFILING_ENABLED


def create_langsmith_traceable(
    get_name: Callable[[Any], str],
    get_metadata: Callable[[Any], Dict[str, Any]],
    get_process_inputs: Callable[[Any], Callable],
    get_process_outputs: Callable[[Any], Callable],
) -> Callable:
    """
    Create a langsmith traceable decorator for both sync and async functions.

    This factory creates a decorator that wraps functions with langsmith tracing,
    using callbacks to get runtime information from the instance (self).

    Args:
        get_name: Function that takes self and returns the trace name
        get_metadata: Function that takes self and returns metadata dict
        get_process_inputs: Function that takes self and returns process_inputs callable
        get_process_outputs: Function that takes self and returns process_outputs callable

    Returns:
        Decorator function that can wrap both sync and async methods
    """

    def decorator(func: Callable) -> Callable:
        # Check for async generator first (before iscoroutinefunction which also matches generators)
        if inspect.isasyncgenfunction(func):

            @wraps(func)
            async def async_gen_wrapper(self, *args, **kwargs):
                # Apply traceable with closures at runtime
                traced = langsmith_traceable(
                    name=get_name(self),
                    metadata=get_metadata(self),
                    process_inputs=get_process_inputs(self),
                    process_outputs=get_process_outputs(self),
                )(func)
                # For async generators, iterate and yield (don't await!)
                async for item in traced(self, *args, **kwargs):
                    yield item

            return async_gen_wrapper

        elif inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                # Apply traceable with closures at runtime
                traced = langsmith_traceable(
                    name=get_name(self),
                    metadata=get_metadata(self),
                    process_inputs=get_process_inputs(self),
                    process_outputs=get_process_outputs(self),
                )(func)
                return await traced(self, *args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(self, *args, **kwargs):
                # Apply traceable with closures at runtime
                traced = langsmith_traceable(
                    name=get_name(self),
                    metadata=get_metadata(self),
                    process_inputs=get_process_inputs(self),
                    process_outputs=get_process_outputs(self),
                )(func)
                return traced(self, *args, **kwargs)

            return sync_wrapper

    return decorator


def format_profiling_tree(network: "RunnableNetwork", indent: int = 2) -> str:
    """
    Format profiling data as a tree for display.

    Args:
        network: Network with profiling data
        indent: Number of spaces per indent level

    Returns:
        Formatted string representation of profiling tree
    """
    if not network.profiling or not network.profiling.frames:
        return "No profiling data available"

    # Import here to avoid circular dependency
    from ..network_node import NetworkNode

    def format_frame(
        frame: ProfilingFrame, level: int = 0, network_for_node_lookup: "RunnableNetwork" = None
    ) -> List[str]:
        lines = []
        prefix = " " * (level * indent)
        duration_str = f"{frame.duration:.3f}s" if frame.duration is not None else "pending"

        # Format main line
        line = f"{prefix}{frame.name} [{frame.frame_type}] - {duration_str}"
        if frame.metadata:
            # Add selected metadata
            meta_parts = []
            for key, value in frame.metadata.items():
                if key in ["node_id", "node_name", "modifier_name", "chunk_index"]:
                    meta_parts.append(f"{key}={value}")
            if meta_parts:
                line += f" ({', '.join(meta_parts)})"

        # Add chunk content on a separate line if present
        if frame.frame_type == "chunk" and "content" in frame.metadata:
            content = frame.metadata["content"]
            if content:
                # Clean up the content for display
                content = content.replace("\n", " ").strip()
                line += f" Content: {content}"

        lines.append(line)

        # Format children
        for child in frame.children:
            lines.extend(format_frame(child, level + 1, network_for_node_lookup))

        # Check if this frame represents a NetworkNode and has its own profiling
        if frame.frame_type == "node" and network_for_node_lookup:
            node_id = frame.metadata.get("node_id")
            if node_id:
                # Find the node in the network
                for node in network_for_node_lookup.nodes:
                    if node.uuid() == node_id and isinstance(node, NetworkNode):
                        # This node is a NetworkNode, check if it has profiling data
                        if node.profiling and node.profiling.frames:
                            lines.append(f"{prefix}  === Nested Network Profiling ===")
                            for nested_frame in node.profiling.frames:
                                lines.extend(format_frame(nested_frame, level + 1, node))
                        break

        return lines

    lines = ["Profiling Results:"]
    lines.append("-" * 50)

    for frame in network.profiling.frames:
        lines.extend(format_frame(frame, 0, network))

    if network.profiling.total_duration is not None:
        lines.append("-" * 50)
        lines.append(f"Total Duration: {network.profiling.total_duration:.3f}s")

    return "\n".join(lines)
