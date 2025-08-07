## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.chat_model_registry import ChatModelRegistry
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage
from pydantic import Field
from dataclasses import dataclass


@dataclass
class ChatModelConfig:
    model_name: str
    api_key: str


class DummyChatModel(BaseChatModel):
    model_name: str = Field(...)
    api_key: str = Field(...)

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager=None, **kwargs
    ) -> ChatResult:
        return ChatResult(generations=[])

    @property
    def _llm_type(self) -> str:
        return "dummy"


def test_register_chat_model():
    registry = ChatModelRegistry()
    config = ChatModelConfig(model_name="test_model", api_key="test_key")
    chat_model = DummyChatModel(model_name=config.model_name, api_key=config.api_key)
    registry.register("test", chat_model)
    assert "test" in registry.registered_names
    assert registry.chat_models["test"].chat_model == chat_model


def test_get_model():
    registry = ChatModelRegistry()
    config = ChatModelConfig(model_name="test_model", api_key="test_key")
    chat_model = DummyChatModel(model_name=config.model_name, api_key=config.api_key)
    registry.register("test", chat_model)
    model = registry.get_model("test")
    assert model == chat_model
    assert model.model_name == "test_model"
    assert model.api_key == "test_key"


def test_get_nonexistent_model():
    registry = ChatModelRegistry()
    assert registry.get_model("nonexistent") is None


def test_register_duplicate_chat_model():
    registry = ChatModelRegistry()
    config1 = ChatModelConfig(model_name="test_model", api_key="test_key1")
    config2 = ChatModelConfig(model_name="test_model", api_key="test_key2")
    chat_model1 = DummyChatModel(model_name=config1.model_name, api_key=config1.api_key)
    chat_model2 = DummyChatModel(model_name=config2.model_name, api_key=config2.api_key)
    registry.register("test", chat_model1)
    registry.register("test", chat_model2)
    model = registry.get_model("test")
    assert model == chat_model2
    assert model.api_key == "test_key2"  # The second registration should overwrite the first


def test_get_registered_names():
    registry = ChatModelRegistry()
    config1 = ChatModelConfig(model_name="model1", api_key="key1")
    config2 = ChatModelConfig(model_name="model2", api_key="key2")
    chat_model1 = DummyChatModel(model_name=config1.model_name, api_key=config1.api_key)
    chat_model2 = DummyChatModel(model_name=config2.model_name, api_key=config2.api_key)
    registry.register("test1", chat_model1)
    registry.register("test2", chat_model2, hidden=True)
    assert registry.get_registered_names() == ["test1"]


def test_get_tokenizer():
    registry = ChatModelRegistry()
    config = ChatModelConfig(model_name="test_model", api_key="test_key")
    chat_model = DummyChatModel(model_name=config.model_name, api_key=config.api_key)
    tokenizer = lambda x: x.split()
    registry.register("test", chat_model, tokenizer=tokenizer)
    assert registry.get_tokenizer("test") == tokenizer


def test_get_max_tokens():
    registry = ChatModelRegistry()
    config = ChatModelConfig(model_name="test_model", api_key="test_key")
    chat_model = DummyChatModel(model_name=config.model_name, api_key=config.api_key)
    registry.register("test", chat_model, max_tokens=100)
    assert registry.get_max_tokens("test") == 100
