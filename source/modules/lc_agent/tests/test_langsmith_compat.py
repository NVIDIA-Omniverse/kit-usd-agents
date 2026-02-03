## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
Test for LangSmith 0.6 compatibility fix in runnable_network._to_dict.

Before the fix, invoking a network with message outputs would crash:
    ValidationError: 1 validation error for RunTree inputs
    Input should be a valid dictionary

The issue: LangSmith's traceable decorator calls process_inputs(inputs) and passes
the result to RunTree(inputs=...). LangSmith 0.6 uses Pydantic v2 which strictly
validates that inputs must be a dict.

The fix: _to_dict in runnable_network.py converts messages to dict before
passing to RunTree.
"""

import pytest
from langchain_core.messages import AIMessage
from langsmith import tracing_context

from lc_agent.runnable_network import RunnableNetwork
from lc_agent.runnable_node import RunnableNode


class MessageNode(RunnableNode):
    """Node that outputs an AIMessage."""

    def invoke(self, input=None, config=None, **kwargs):
        self.outputs = AIMessage(content="test")
        self.invoked = True
        return self.outputs


def test_network_invoke_with_langsmith_tracing_enabled():
    """
    Test that invoking a network with message output works when LangSmith
    tracing is enabled.

    With tracing enabled, LangSmith:
    1. Calls process_inputs = get_process_io(network)
    2. Calls inputs = process_inputs(inputs)  -> this calls _to_dict
    3. Creates RunTree(inputs=inputs)  -> ValidationError if not dict

    Before the fix, this would crash with:
        ValidationError: 1 validation error for RunTree inputs
        Input should be a valid dictionary
    """
    with RunnableNetwork() as network:
        MessageNode()

    # Enable tracing so RunTree is actually created
    # Without this, LangSmith skips RunTree creation entirely
    with tracing_context(enabled=True):
        # This calls invoke() which is decorated with @_traceable
        # -> calls get_process_io(network) to get process_io
        # -> LangSmith calls process_inputs(inputs) which calls _to_dict
        # -> RunTree(inputs=dict) is created
        # If _to_dict is broken, ValidationError is raised here
        result = network.invoke()

    assert isinstance(result, AIMessage)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
