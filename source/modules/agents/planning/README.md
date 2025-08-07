# AIQ Planning Agent

A Planning Agent plugin for AgentIQ that creates detailed, step-by-step plans for any kind of task and guides their execution.

## Overview

The Planning Agent is a sophisticated component that:
- Breaks down user requests into comprehensive, actionable roadmaps
- Creates tool-aware plans that leverage available system capabilities
- Guides plan execution by injecting step-by-step instructions to supervisors
- Supports both detailed and concise plan formats
- Generates implementation details dynamically during execution

## Features

### Tool-Aware Planning
The planning agent automatically discovers available tools from the multi-agent network and incorporates them into its plans. This ensures that generated plans are executable using the actual tools available in the system.

### Guided Plan Execution
Once a plan is generated, the Planning Agent doesn't just hand it off - it actively guides execution by:
- Injecting a planning modifier into the MultiAgent network
- Presenting each step to the supervisor at the appropriate time
- Tracking execution progress and status for each step
- Ensuring the plan is followed systematically

### Flexible Plan Formats

#### Detailed Plans (Default)
By default, the agent generates comprehensive plans with full implementation details:

```
PLAN: Deploy Web Application

Step 1: Set up infrastructure
- Create cloud instances for web and database servers
- Configure networking and security groups
- Expected outcome: Infrastructure ready for deployment

Step 2: Deploy database
- Install PostgreSQL on database server
- Configure database security and access
- Expected outcome: Database server operational
```

#### Short Plans
With `short_plan: true`, the agent generates concise plans without implementation details:

```
PLAN: Exit the Room

Step 1: Locate the door

Step 2: Check if the door is locked

Step 3: Unlock the door if necessary

Step 4: Open the door

Step 5: Walk through the doorway
```

### Dynamic Detail Generation
With `add_details: true`, the agent generates implementation details on-demand during execution. This provides several benefits:
- Details are contextually relevant to the current execution state
- Reduces initial plan generation time
- Allows for adaptive planning based on execution results
- Provides focused, just-in-time instructions

Example of dynamically generated details:
```
To locate the door, look around the room for an opening that leads outside. 
Typically, it's a rectangular structure with a handle. Check walls for 
something that looks like an exit.
```

## Configuration

The Planning Agent supports the following configuration options:

```yaml
functions:
  planning:
    _type: planning
    system_message: "{path/to/custom_system_message.md}"  # Optional custom instructions
    short_plan: false     # Generate concise plans (default: false)
    add_details: false    # Enable dynamic detail generation (default: false)
    llm_name: nim_llm    # Specify LLM for planning (optional)
```

## Usage Examples

### Basic Planning
```yaml
# Standard detailed planning
planning:
  _type: planning
```

### Quick Task Planning
```yaml
# For simple tasks that don't need detailed steps
planning:
  _type: planning
  short_plan: true
```

### Combined Configuration
```yaml
# Start with short plan, add details during execution
planning:
  _type: planning
  short_plan: true
  add_details: true
```

## Architecture

The Planning Agent consists of several key components:

1. **PlanningGenNode**: Generates plans based on user requests and available tools
2. **PlanningNetworkNode**: Coordinates the planning process
3. **PlanningModifier**: Manages plan execution and guides the supervisor
4. **System Prompts**: Configurable instructions for plan generation and detail creation

## How It Works

1. **Plan Generation Phase**:
   - User request is received by the planning agent
   - Available tools are discovered from the multi-agent network
   - A plan is generated based on the request and available tools
   - Plan format depends on `short_plan` configuration

2. **Plan Sharing Phase**:
   - Generated plan is validated and structured
   - Plan is shared with the MultiAgent network
   - Planning modifier is injected into the network
   - Execution status tracking is initialized

3. **Guided Execution Phase**:
   - Planning modifier presents each step to the supervisor
   - If `add_details` is enabled, details are generated just before each step
   - Execution progress is tracked
   - Supervisor follows the plan systematically

## Best Practices

1. **Use short plans for simple tasks** - When the steps are self-explanatory
2. **Enable dynamic details for complex workflows** - When steps need context-aware instructions
3. **Provide custom system messages** - To specialize planning for specific domains
4. **Monitor plan execution** - Check the plan_status metadata for progress tracking

## Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history and changes.
