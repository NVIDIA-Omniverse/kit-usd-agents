## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from unittest.mock import patch, MagicMock
from lc_agent.chat_models.chat_nvcf import ChatNVCF, NvcfCallSync, NvcfCallAsync
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

@pytest.fixture
def chat_nvcf():
    return ChatNVCF()

def test_chat_nvcf_initialization(chat_nvcf):
    assert isinstance(chat_nvcf, ChatNVCF)
    assert chat_nvcf.model is None
    assert chat_nvcf.max_tokens == 1024
    assert chat_nvcf.temperature == 0.1
    assert chat_nvcf.top_p == 1.0
    assert chat_nvcf.top_k == 1.0
    assert chat_nvcf.invoke_url == "https://api.nvcf.nvidia.com/v2/nvcf/pexec"
    assert chat_nvcf.api_token is None

def test_chat_nvcf_get_messages(chat_nvcf):
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello, how are you?"),
        AIMessage(content="I'm doing well, thank you for asking!")
    ]
    converted_messages = chat_nvcf._ChatNVCF__get_messages(messages)
    assert len(converted_messages) == 3
    assert converted_messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert converted_messages[1] == {"role": "user", "content": "Hello, how are you?"}
    assert converted_messages[2] == {"role": "assistant", "content": "I'm doing well, thank you for asking!"}

def test_chat_nvcf_invoke_url(chat_nvcf):
    assert chat_nvcf._invoke_url == "https://api.nvcf.nvidia.com/v2/nvcf/pexec"
    chat_nvcf.model = "test_model"
    assert chat_nvcf._invoke_url == "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/test_model"

@patch.dict('os.environ', {'NVIDIA_API_KEY': 'test_api_key'})
def test_chat_nvcf_api_token(chat_nvcf):
    assert chat_nvcf._api_token == 'test_api_key'
    chat_nvcf.api_token = 'custom_api_key'
    assert chat_nvcf._api_token == 'custom_api_key'

def test_chat_nvcf_header(chat_nvcf):
    header = chat_nvcf._header
    assert header['Accept'] == 'application/json'
    assert 'Authorization' not in header

    chat_nvcf.api_token = 'test_token'
    header = chat_nvcf._header
    assert header['Authorization'] == 'Bearer test_token'

def test_chat_nvcf_llm_type(chat_nvcf):
    assert chat_nvcf._llm_type == "echoing-chat-model-advanced"

def test_chat_nvcf_identifying_params(chat_nvcf):
    params = chat_nvcf._identifying_params
    assert params['model'] is None
    assert params['max_tokens'] == 1024
    assert params['temperature'] == 0.1
    assert params['top_p'] == 1.0
    assert params['invoke_url'] == "https://api.nvcf.nvidia.com/v2/nvcf/pexec"
    assert params['api_token'] is None

@pytest.mark.asyncio
async def test_nvcf_call_async_initialization():
    payload = {"test": "payload"}
    headers = {"test": "header"}
    invoke_url = "https://test.url"
    call = NvcfCallAsync(payload, headers, invoke_url)
    assert call._payload == payload
    assert call._headers == headers
    assert call._invoke_url == invoke_url
    assert not call._started
    assert not call._finished

def test_nvcf_call_sync_initialization():
    payload = {"test": "payload"}
    headers = {"test": "header"}
    invoke_url = "https://test.url"
    call = NvcfCallSync(payload, headers, invoke_url)
    assert call._payload == payload
    assert call._headers == headers
    assert call._invoke_url == invoke_url
    assert not call._started
    assert not call._finished

if __name__ == "__main__":
    pytest.main(["-v", "test_chat_nvcf.py"])