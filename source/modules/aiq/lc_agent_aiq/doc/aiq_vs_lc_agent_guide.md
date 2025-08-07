# AIQ Functions vs LC Agent Networks: When to Use Which

This guide explains when to use regular AIQ functions versus LC Agent networks in the `lc_agent_aiq` module. Understanding these differences is crucial for choosing the right approach for your agent development.

## Overview

The `lc_agent_aiq` module provides a bridge between AIQ (AgentIQ) and LC Agent, offering two main approaches for creating AI agents:

1. **Regular AIQ Functions** - Simple, stateless function-based agents
2. **LC Agent Networks** - Complex, stateful network-based agents using `LCAgentFunction`

## Regular AIQ Functions

### When to Use

Use regular AIQ functions when you need:

1. **Simple, stateless operations**
   - File operations (read, write, list)
   - API calls
   - Data transformations
   - Mathematical calculations

2. **No conversation history**
   - Each invocation is independent
   - No need to remember previous interactions
   - Stateless processing

3. **Direct input/output mapping**
   - Clear input parameters
   - Predictable output format
   - No complex decision trees

4. **Quick tool integration**
   - External services
   - System commands
   - Database queries

5. **Multiple typed arguments**
   - Functions with specific parameter types (e.g., `calculate(s: int, f: float, k: dict)`)
   - Structured data inputs
   - Type-safe parameter validation
   - Clear function signatures

### Example: GitLab Pipeline Function

```python
@register_function(config_type=GitLabPipelinesConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def register_gitlab_pipelines(config: GitLabPipelinesConfig, builder: Builder):
    """Simple function that fetches GitLab pipelines."""
    
    async def fetch_pipelines(project_path: str) -> str:
        # Direct API call, no state management
        result = await get_gitlab_pipelines(project_path)
        return result
    
    yield FunctionInfo.create(
        single_fn=fetch_pipelines,
        description="Fetches pipelines from GitLab"
    )
```

### Characteristics
- No LLM interaction within the function
- Simple async/await pattern
- Direct parameter passing
- Returns formatted strings or data
- No complex state management
- Well-defined typed parameters

## LC Agent Networks

### When to Use

Use LC Agent networks (via `LCAgentFunction`) when you need:

1. **Subnetworks and node hierarchies**
   - Multiple processing stages
   - Parent-child node relationships
   - Complex message routing

2. **LLM integration and processing**
   - Dynamic prompt generation
   - Multi-turn conversations
   - Context-aware responses

3. **Dynamic system messages**
   - Conditional prompts based on context
   - Runtime message modification
   - Adaptive behavior

4. **Stateful processing**
   - Conversation history
   - Context preservation
   - Multi-step workflows

5. **Complex analysis of inputs/outputs**
   - Pre-processing user inputs
   - Post-processing LLM outputs
   - Result synthesis

6. **Network modifiers**
   - Dynamic network structure changes
   - Conditional node creation
   - Runtime behavior modification

7. **Natural language or network-based input**
   - When the input is natural language that needs interpretation
   - When the conversation history (network state) itself is the primary input
   - When context from previous interactions is essential
   - When the agent needs to understand and respond to human language

### Example: Summary Agent Network

```python
@register_function(config_type=SummaryNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def summary_node_function(config: SummaryNodeConfig, builder: Builder):
    """Complex agent using LC Agent networks."""
    
    # Uses LCAgentFunction for network-based processing
    yield LCAgentFunction(
        config=config,
        builder=builder,
        lc_agent_node_type=SummaryNode,  # NetworkNode subclass
        lc_agent_node_gen_type=SummaryNodeGenerator,  # Optional generator
        max_length=config.max_length,
    )
```

## Decision Matrix

| Requirement | Regular AIQ Function | LC Agent Network |
|------------|---------------------|------------------|
| Simple tool calling | ✅ | ❌ (overkill) |
| Stateless operations | ✅ | ❌ |
| Multiple typed parameters | ✅ | ❌ |
| Natural language input | ❌ | ✅ |
| Network/history as input | ❌ | ✅ |
| LLM interaction | ❌ | ✅ |
| Conversation history | ❌ | ✅ |
| Dynamic prompts | ❌ | ✅ |
| Complex workflows | ❌ | ✅ |
| Runtime modifications | ❌ | ✅ |
| Subnetworks | ❌ | ✅ |
| File/API operations | ✅ | ❌ (unnecessary) |

## Best Practices

### For Regular AIQ Functions:
1. Keep functions focused and single-purpose
2. Use clear parameter schemas (Pydantic)
3. Return structured data or formatted strings
4. Handle errors gracefully
5. Avoid state management
6. Define clear types for all parameters

### For LC Agent Networks:
1. Use NetworkNode for encapsulation
2. Implement modifiers for dynamic behavior
3. Leverage system message composition
4. Design clear node hierarchies
5. Consider streaming for long responses
6. Process natural language inputs through LLM nodes
