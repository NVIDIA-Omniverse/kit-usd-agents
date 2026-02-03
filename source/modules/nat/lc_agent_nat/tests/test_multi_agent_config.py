## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent_nat.multi_agent_register import MultiAgentConfig, MultiAgentConfigBase


def test_multi_agent_config_base_defaults():
    config = MultiAgentConfigBase()
    assert config.name is None
    assert config.description == "A function that supports multi-agent workflow with routing capabilities"


def test_multi_agent_config_defaults():
    config = MultiAgentConfig()
    assert config.description == "Multi-agent router that directs requests to specialized agents"
    assert config.llm_name is None
    assert config.tool_names == []
    assert config.system_message == ""
    assert config.multishot is True
    assert config.function_calling is False
    assert config.classification_node is True
    assert config.generate_prompt_per_agent is True
    assert config.first_routing_instruction == ""
    assert config.subsequent_routing_instruction == ""
    assert config.output_mode == "default"


def test_multi_agent_config_output_mode_default():
    config = MultiAgentConfig()
    assert config.output_mode == "default"


def test_multi_agent_config_output_mode_raw():
    config = MultiAgentConfig(output_mode="raw")
    assert config.output_mode == "raw"


def test_multi_agent_config_custom_values():
    config = MultiAgentConfig(
        name="test_agent",
        description="Test description",
        system_message="You are a helpful assistant",
        multishot=False,
        function_calling=True,
        output_mode="raw",
    )
    assert config.name == "test_agent"
    assert config.description == "Test description"
    assert config.system_message == "You are a helpful assistant"
    assert config.multishot is False
    assert config.function_calling is True
    assert config.output_mode == "raw"

