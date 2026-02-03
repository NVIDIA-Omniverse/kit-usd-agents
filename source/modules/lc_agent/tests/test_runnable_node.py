## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.runnable_node import RunnableNode, AINodeMessageChunk, _is_message
from lc_agent.runnable_network import RunnableNetwork
from lc_agent.network_modifier import NetworkModifier
from lc_agent.chat_model_registry import get_chat_model_registry
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, AIMessageChunk, ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.runnables import RunnableConfig
from typing import List, Dict, Any, AsyncIterator
import asyncio
from langchain_core.runnables import Runnable
from lc_agent.from_runnable_node import FromRunnableNode

class DummyChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy"

    def _generate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Dummy response"))])

    async def _agenerate(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Async dummy response"))])

    async def _astream(self, messages: List[Any], stop: List[str] | None = None, run_manager = None, **kwargs) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Streaming ", type="AIMessageChunk"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="dummy ", type="AIMessageChunk"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="response", type="AIMessageChunk"))

class TestRunnableNode(RunnableNode):
    def _get_chat_model(self, chat_model_name, chat_model_input, invoke_input, config):
        return DummyChatModel()

class DummyNode(TestRunnableNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invoked = False

    def invoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content="Dummy response")
        self.invoked = True
        return self.outputs

    async def ainvoke(self, input: Dict[str, Any] = {}, config=None, **kwargs):
        if self.invoked:
            return self.outputs
        self.outputs = AIMessage(content="Async dummy response")
        self.invoked = True
        return self.outputs

    async def astream(self, input: Dict[str, Any] = {}, config=None, **kwargs) -> AsyncIterator[AINodeMessageChunk]:
        if not self.invoked:
            self.outputs = AIMessage(content="Streaming dummy response")
            self.invoked = True
            yield AINodeMessageChunk(content="Streaming ", node=self)
            yield AINodeMessageChunk(content="dummy ", node=self)
            yield AINodeMessageChunk(content="response", node=self)
        else:
            yield AINodeMessageChunk(content=self.outputs.content, node=self)

@pytest.fixture
def dummy_chat_model():
    return DummyChatModel()

def test_runnable_node_creation():
    node = TestRunnableNode()
    assert isinstance(node, RunnableNode)
    assert node.parents == []
    assert node.inputs == []
    assert node.outputs is None
    assert 'uuid' in node.metadata
    assert len(node.metadata) == 1
    assert not node.invoked

def test_runnable_node_invoke():
    node = TestRunnableNode()
    result = node.invoke()
    assert isinstance(result, AIMessage)
    assert result.content == "Dummy response"
    assert node.invoked

@pytest.mark.asyncio
async def test_runnable_node_ainvoke():
    node = TestRunnableNode()
    result = await node.ainvoke()
    assert isinstance(result, AIMessage)
    assert result.content == "Async dummy response"
    assert node.invoked

@pytest.mark.asyncio
async def test_runnable_node_astream():
    node = TestRunnableNode()
    chunks = []
    async for chunk in node.astream():
        chunks.append(chunk)
    # langchain may add an empty final chunk, so we check for at least 3 chunks
    assert len(chunks) >= 3
    assert all(isinstance(chunk, AINodeMessageChunk) for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == "Streaming dummy response"
    assert node.invoked

def test_runnable_node_add_parent():
    parent = DummyNode()
    child = DummyNode()
    child._add_parent(parent)
    assert parent in child.parents

def test_runnable_node_clear_parents():
    parent1 = DummyNode()
    parent2 = DummyNode()
    child = DummyNode()
    child._add_parent(parent1)
    child._add_parent(parent2)
    child._clear_parents()
    assert child.parents == []

def test_runnable_node_rshift():
    node1 = DummyNode()
    node2 = DummyNode()
    result = node1 >> node2
    assert result == node2
    assert node1 in node2.parents

def test_runnable_node_lshift():
    node1 = DummyNode()
    node2 = DummyNode()
    result = node2 << node1
    assert result == node1
    assert node1 in node2.parents

def test_runnable_node_rrshift():
    node1 = DummyNode()
    node2 = DummyNode()
    result = None >> node1 >> node2
    assert result == node2
    assert node1 in node2.parents
    assert node1.parents == []

def test_runnable_node_combine_inputs():
    node = TestRunnableNode()
    parents_result = [
        SystemMessage(content="System message"),
        SystemMessage(content="Other system message"),
        HumanMessage(content="Human message"),
    ]
    result = node._combine_inputs({}, None, parents_result)
    assert len(result) == 2
    assert isinstance(result[0], SystemMessage)
    assert isinstance(result[1], HumanMessage)
    assert "System message" in result[0].content
    assert "Other system message" in result[0].content
    assert result[1].content == "Human message"

def test_runnable_node_combine_inputs_with_chat_prompt_value():
    node = TestRunnableNode()
    parents_result = [HumanMessage(content="Human message")]
    input_result = ChatPromptValue(messages=[AIMessage(content="AI message")])
    result = node._combine_inputs({"input": input_result}, None, parents_result)
    assert len(result) == 1
    assert isinstance(result[0], HumanMessage)
    assert result[0].content == "Human message"

def test_runnable_node_in_network():
    with RunnableNetwork() as network:
        node1 = DummyNode()
        node2 = DummyNode()
    
    assert node1 in network.nodes
    assert node2 in network.nodes
    assert network.get_parents(node2) == [node1]

def test_runnable_node_metadata():
    node = TestRunnableNode()
    node.metadata["key"] = "value"
    assert node.metadata["key"] == "value"

def test_runnable_node_chat_model_name():
    node = TestRunnableNode(chat_model_name="test_model")
    assert node.chat_model_name == "test_model"

def test_runnable_node_verbose():
    node = TestRunnableNode(verbose=True)
    assert node.verbose

class DummyModifier(NetworkModifier):
    def on_begin_invoke(self, network):
        network.metadata["modifier_called"] = True

def test_runnable_node_with_modifier():
    with RunnableNetwork() as network:
        node = DummyNode()
    
    modifier = DummyModifier()
    network.add_modifier(modifier)
    
    network.invoke()
    assert network.metadata.get("modifier_called") == True

@pytest.mark.asyncio
async def test_runnable_node_process_parents():
    class TestNode(TestRunnableNode):
        async def ainvoke(self, input, config=None, **kwargs):
            return AIMessage(content="Test")

    with RunnableNetwork() as network:
        node1 = TestNode()
        node2 = TestNode()
        node3 = TestRunnableNode()

    result = await node3._aprocess_parents({}, None)
    assert len(result) == 2
    assert all(isinstance(r, AIMessage) for r in result)
    assert [r.content for r in result] == ["Test", "Test"]

class DummyRunnable(Runnable):
    def invoke(self, input, config=None):
        return input

def test_lshift_with_runnable():
    node = TestRunnableNode()
    runnable = DummyRunnable()
    result = node << runnable
    assert isinstance(node.parents[0], FromRunnableNode)

def test_rshift_with_runnable():
    node = TestRunnableNode()
    runnable = DummyRunnable()
    result = node >> runnable
    assert isinstance(result, FromRunnableNode)
    assert result.parents[0] == node

def test_rrshift_with_runnable():
    runnable = DummyRunnable()
    node = TestRunnableNode()
    result = runnable >> node
    assert isinstance(result, TestRunnableNode)
    assert isinstance(result.parents[0], FromRunnableNode)

def test_add_parent_existing():
    node = TestRunnableNode()
    parent = TestRunnableNode()
    node._add_parent(parent)
    node._add_parent(parent)  # Adding the same parent again
    assert len(node.parents) == 1
    assert node.parents[0] == parent

def test_add_parent_with_index():
    node = TestRunnableNode()
    parent1 = TestRunnableNode()
    parent2 = TestRunnableNode()
    node._add_parent(parent1)
    node._add_parent(parent2, parent_index=0)
    assert node.parents == [parent2, parent1]

@pytest.mark.asyncio
async def test_astream_with_ainodemessagechunk():
    class TestStreamNode(TestRunnableNode):
        async def _astream_chat_model(self, *args, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content="Test"))
            yield ChatGenerationChunk(message=AIMessageChunk(content="NodeChunk"))

        async def astream(self, input: Dict[str, Any] = {}, config=None, **kwargs) -> AsyncIterator[AINodeMessageChunk]:
            async for item in self._astream_chat_model():
                if isinstance(item, ChatGenerationChunk):
                    yield AINodeMessageChunk(content=item.message.content, node=self)
                else:
                    yield item

    node = TestStreamNode()
    chunks = []
    async for chunk in node.astream():
        chunks.append(chunk)
    
    assert len(chunks) == 2
    assert all(isinstance(chunk, AINodeMessageChunk) for chunk in chunks)
    assert chunks[0].content == "Test"
    assert chunks[1].content == "NodeChunk"

def test_is_message_with_various_types():
    assert _is_message(HumanMessage(content="Test"))
    assert _is_message(AIMessage(content="Test"))
    assert _is_message(SystemMessage(content="Test"))
    assert _is_message(ChatPromptValue(messages=[HumanMessage(content="Test")]))
    assert _is_message({"role": "user", "content": "Test"})
    assert _is_message("Test")
    assert _is_message(("user", "Test"))
    assert not _is_message(123)

def test_repr_with_different_states():
    node = TestRunnableNode(name="TestNode")
    assert "TestNode" in repr(node)
    assert "(no outputs)" in repr(node)

    node.outputs = AIMessage(content="Test output")
    repr_string = repr(node)
    assert "TestNode" in repr_string
    assert "Test output" in repr_string

def test_parse_and_extract_channel_metadata_multiple_channels():
    """Test parsing multiple consecutive channel sections."""
    node = TestRunnableNode()
    content = '<|channel|>commentary<|message|>We need to use mcp_kit.search_extensions again with query "extension manager load unload extensions".<|end|><|channel|>commentary<|message|>Let\'s call search_extensions.<|end|><|channel|>commentary<|message|>We need to output a tool call.<|end|><|channel|>commentary<|message|>We need to output a single line with the tool call. Use mcp_kit.search_extensions with query "extension manager load unload extensions".<|end|>mcp_kit.search_extensions {"query": "extension manager load unload extensions", "top_k": 10}'
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    # Check that only the final message remains
    assert cleaned_content == 'mcp_kit.search_extensions {"query": "extension manager load unload extensions", "top_k": 10}'
    
    # Check that all channel messages are in metadata
    assert "channel" in updated_metadata
    assert "commentary" in updated_metadata["channel"]
    
    # Check that all commentary messages are concatenated
    commentary = updated_metadata["channel"]["commentary"]
    assert 'We need to use mcp_kit.search_extensions again with query "extension manager load unload extensions".' in commentary
    assert "Let's call search_extensions." in commentary
    assert "We need to output a tool call." in commentary
    assert 'We need to output a single line with the tool call. Use mcp_kit.search_extensions with query "extension manager load unload extensions".' in commentary

def test_parse_and_extract_channel_metadata_single_channel():
    """Test parsing a single channel section."""
    node = TestRunnableNode()
    content = '<|channel|>commentary<|message|>This is a comment.<|end|>Actual message content'
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    assert cleaned_content == 'Actual message content'
    assert "channel" in updated_metadata
    assert "commentary" in updated_metadata["channel"]
    assert updated_metadata["channel"]["commentary"] == "This is a comment."

def test_parse_and_extract_channel_metadata_think_tag():
    """Test parsing <think> tags."""
    node = TestRunnableNode()
    content = '<think>This is my reasoning process</think>Final response'
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    assert cleaned_content == 'Final response'
    assert "think" in updated_metadata
    assert updated_metadata["think"] == "This is my reasoning process"

def test_parse_and_extract_channel_metadata_thinking_tag():
    """Test parsing <thinking> tags."""
    node = TestRunnableNode()
    content = '<thinking>Deep thought process here</thinking>Final answer'
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    assert cleaned_content == 'Final answer'
    assert "thinking" in updated_metadata
    assert updated_metadata["thinking"] == "Deep thought process here"

def test_parse_and_extract_channel_metadata_no_special_tags():
    """Test that content without special tags is returned unchanged."""
    node = TestRunnableNode()
    content = 'Just a regular message'
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    assert cleaned_content == 'Just a regular message'
    assert updated_metadata == {}

def test_parse_and_extract_channel_metadata_non_string_content():
    """Test that non-string content is returned unchanged."""
    node = TestRunnableNode()
    content = ['list', 'content']
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    assert cleaned_content == ['list', 'content']
    assert updated_metadata == {}

def test_parse_and_extract_channel_metadata_different_channels():
    """Test parsing multiple different channel types."""
    node = TestRunnableNode()
    content = '<|channel|>commentary<|message|>First comment<|end|><|channel|>debug<|message|>Debug info<|end|>Main message'
    metadata = {}
    
    cleaned_content, updated_metadata = node._parse_and_extract_channel_metadata(content, metadata)
    
    assert cleaned_content == 'Main message'
    assert "channel" in updated_metadata
    assert "commentary" in updated_metadata["channel"]
    assert "debug" in updated_metadata["channel"]
    assert updated_metadata["channel"]["commentary"] == "First comment"
    assert updated_metadata["channel"]["debug"] == "Debug info"

def test_reorder_tool_messages_valid():
    """Test _reorder_tool_messages with valid tool call/response pairs."""
    node = TestRunnableNode()
    
    # Create messages with tool calls and responses
    ai_msg = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool", "args": {}}])
    tool_msg = ToolMessage(content="result", tool_call_id="call_1")
    human_msg = HumanMessage(content="test")
    
    messages = [ai_msg, tool_msg, human_msg]
    reordered = node._reorder_tool_messages(messages)
    
    # Should keep the order and all messages
    assert len(reordered) == 3
    assert reordered[0] == ai_msg
    assert reordered[1] == tool_msg
    assert reordered[2] == human_msg

def test_reorder_tool_messages_unmatched():
    """Test _reorder_tool_messages removes unmatched tool messages."""
    node = TestRunnableNode()
    
    # AI message with tool call but no corresponding ToolMessage
    ai_msg_unmatched = AIMessage(content="", tool_calls=[{"id": "call_missing", "name": "tool", "args": {}}])
    # ToolMessage without corresponding AI message
    tool_msg_unmatched = ToolMessage(content="orphan", tool_call_id="call_orphan")
    # Valid pair
    ai_msg = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool", "args": {}}])
    tool_msg = ToolMessage(content="result", tool_call_id="call_1")
    human_msg = HumanMessage(content="test")
    
    messages = [ai_msg_unmatched, ai_msg, tool_msg, tool_msg_unmatched, human_msg]
    reordered = node._reorder_tool_messages(messages)
    
    # Only valid pair and human message should remain
    assert len(reordered) == 3
    assert ai_msg in reordered
    assert tool_msg in reordered
    assert human_msg in reordered
    assert ai_msg_unmatched not in reordered
    assert tool_msg_unmatched not in reordered

def test_sanitize_messages_for_chat_model_with_tools():
    """Test that messages are not sanitized when model has tools bound."""
    from langchain_core.runnables.base import RunnableBinding
    from langchain_core.language_models.chat_models import BaseChatModel
    node = TestRunnableNode()
    
    # Create a mock chat model
    class MockChatModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "mock"
        
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.outputs import ChatResult, ChatGeneration
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="test"))])
    
    # Bind tools to it
    mock_model = RunnableBinding(bound=MockChatModel(), kwargs={"tools": [{"name": "test_tool"}]})
    
    tool_msg = ToolMessage(content="result", tool_call_id="call_1")
    ai_msg = AIMessage(content="", tool_calls=[{"id": "call_1", "name": "tool", "args": {}}])
    
    messages = [tool_msg, ai_msg]
    result = node._sanitize_messages_for_chat_model(messages, "test_model", mock_model)
    
    # Messages should be unchanged
    assert len(result) == 2
    assert isinstance(result[0], ToolMessage)
    assert isinstance(result[1], AIMessage)

def test_sanitize_messages_for_chat_model_without_tools():
    """Test that messages are sanitized when model doesn't have tools."""
    node = TestRunnableNode()
    
    tool_msg = ToolMessage(content="result", tool_call_id="call_1")
    ai_msg_with_content = AIMessage(content="text", tool_calls=[{"id": "call_1", "name": "tool", "args": {}}])
    ai_msg_empty = AIMessage(content="", tool_calls=[{"id": "call_2", "name": "tool", "args": {}}])
    human_msg = HumanMessage(content="test")
    
    messages = [tool_msg, ai_msg_with_content, ai_msg_empty, human_msg]
    
    # Mock chat model without tools
    class MockModel:
        pass
    
    result = node._sanitize_messages_for_chat_model(messages, "test_model", MockModel())
    
    # ToolMessage should be converted to HumanMessage
    # AI message with content but tool_calls should have tool_calls removed
    # Empty AI message with tool_calls should be removed
    # Human message should remain
    assert len(result) == 3
    assert isinstance(result[0], HumanMessage)  # Converted from ToolMessage
    assert result[0].content == "result"
    assert isinstance(result[1], AIMessage)  # AI with content, tool_calls removed
    assert result[1].content == "text"
    assert not hasattr(result[1], 'tool_calls') or not result[1].tool_calls
    assert isinstance(result[2], HumanMessage)  # Original human message

def test_deserialize_message_types():
    """Test _deserialize_message for different message types."""
    # Test HumanMessage
    human_dict = {"type": "human", "content": "Hello"}
    result = RunnableNode._deserialize_message(human_dict)
    assert isinstance(result, HumanMessage)
    assert result.content == "Hello"
    
    # Test AIMessage
    ai_dict = {"type": "ai", "content": "Hi there"}
    result = RunnableNode._deserialize_message(ai_dict)
    assert isinstance(result, AIMessage)
    assert result.content == "Hi there"
    
    # Test SystemMessage
    system_dict = {"type": "system", "content": "You are helpful"}
    result = RunnableNode._deserialize_message(system_dict)
    assert isinstance(result, SystemMessage)
    assert result.content == "You are helpful"
    
    # Test ToolMessage
    tool_dict = {"type": "tool", "content": "Result", "tool_call_id": "call_1"}
    result = RunnableNode._deserialize_message(tool_dict)
    assert isinstance(result, ToolMessage)
    assert result.content == "Result"
    assert result.tool_call_id == "call_1"

def test_deserialize_message_already_message():
    """Test that BaseMessage instances are returned as-is."""
    msg = HumanMessage(content="test")
    result = RunnableNode._deserialize_message(msg)
    assert result is msg

def test_deserialize_outputs_single_and_list():
    """Test _deserialize_outputs with single message and list."""
    # Single message
    msg_dict = {"type": "human", "content": "Test"}
    result = RunnableNode._deserialize_outputs(msg_dict)
    assert isinstance(result, HumanMessage)
    
    # List of messages
    msg_list = [
        {"type": "human", "content": "Test1"},
        {"type": "ai", "content": "Test2"}
    ]
    result = RunnableNode._deserialize_outputs(msg_list)
    assert isinstance(result, list)
    assert len(result) == 2
    assert isinstance(result[0], HumanMessage)
    assert isinstance(result[1], AIMessage)
    
    # None
    result = RunnableNode._deserialize_outputs(None)
    assert result is None

def test_find_metadata_in_node():
    """Test find_metadata finds value in node's own metadata."""
    node = TestRunnableNode()
    node.metadata["test_key"] = "test_value"
    
    result = node.find_metadata("test_key")
    assert result == "test_value"

def test_find_metadata_in_network():
    """Test find_metadata finds value in active network."""
    with RunnableNetwork() as network:
        network.metadata["network_key"] = "network_value"
        node = TestRunnableNode()
        
        result = node.find_metadata("network_key")
        assert result == "network_value"

def test_find_metadata_not_found():
    """Test find_metadata returns None when key not found."""
    node = TestRunnableNode()
    result = node.find_metadata("nonexistent_key")
    assert result is None

def test_serialize_model():
    """Test serialize_model excludes parents and adds node type."""
    node = TestRunnableNode(name="test_node")
    parent = TestRunnableNode()
    node._add_parent(parent)
    
    serialized = node.serialize_model()
    
    # Should not include parents
    assert "parents" not in serialized
    # Should include node type
    assert "__node_type__" in serialized
    assert serialized["__node_type__"] == "TestRunnableNode"
    # Should include name
    assert "name" in serialized
    assert serialized["name"] == "test_node"

def test_hash_consistency():
    """Test that node hash is based on object id."""
    node1 = TestRunnableNode()
    node2 = TestRunnableNode()
    
    # Different nodes have different hashes
    assert hash(node1) != hash(node2)
    
    # Same node has consistent hash
    hash1 = hash(node1)
    hash2 = hash(node1)
    assert hash1 == hash2

def test_rrshift_with_list():
    """Test __rrshift__ with list of nodes."""
    node1 = TestRunnableNode()
    node2 = TestRunnableNode()
    child = TestRunnableNode()
    
    result = [node1, node2] >> child
    
    assert result is child
    assert len(child.parents) == 2
    assert node1 in child.parents
    assert node2 in child.parents

def test_lshift_invalid_type():
    """Test __lshift__ raises error with invalid type."""
    node = TestRunnableNode()
    
    with pytest.raises(ValueError, match="Invalid parent type"):
        node << 123

def test_rshift_invalid_type():
    """Test __rshift__ raises error with invalid type."""
    node = TestRunnableNode()
    
    with pytest.raises(ValueError, match="Invalid child type"):
        node >> "invalid"

def test_rrshift_invalid_type():
    """Test __rrshift__ raises error with invalid type."""
    node = TestRunnableNode()
    
    with pytest.raises(ValueError, match="Invalid parent type"):
        123 >> node

if __name__ == "__main__":
    pytest.main(["-v", "--tb=short"])