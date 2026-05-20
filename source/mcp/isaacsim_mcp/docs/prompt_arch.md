

Okay, so we are going to talk about the Isaac Sim MCP.

So it's going to be a developer MCP, so this one is going to be very focused on robotics and simulation development workflow with NVIDIA Omniverse Isaac Sim.

There are a few things that are key that we're going to do:

1. It's a hierarchical data structure. So first, there is an overview set of instructions that may be part of Cursor rules or agentic workflow. Effectively, there is an initial, kind of large system prompt about what is Isaac Sim and how to use it for robotics and simulation development.

2. Then it's a hierarchical set of tools in order to retrieve data and information about it into a scaffolding way.

1) There is a tool for search extensions. This tool will enable you to find extensions about certain topics, so it will use some RAG and some good prompting techniques in order to search into the Isaac Sim extensions that we have. And then it will sort of return to the caller, to the agent, the list of extensions that are related and relevant for the task that the user is asking. This is especially useful for finding robotics-related extensions, sensor extensions, and physics simulation tools.

2) A tool will be that once I have extensions, I might want to go a bit deeper into what actually do they do. So I will be able to pass these extensions to another MCP tool that will collect detailed information. So then I take all of these extension IDs and pass them to the extension detail MCP that will then collect some shortened documentation about key features, key objectives, how to use it, API availability, and this type of things. And I will be able to retrieve that and get a general sense of what that does. This is particularly valuable for understanding robotics frameworks like `omni.isaac.core`, sensor extensions like `omni.isaac.sensor`, and manipulation tools.

3) Then there is a tool for code examples. The tool will enable you to get code examples about certain topics related to Isaac Sim and robotics development. You'll be able to ask for something like "robot articulation setup" or "sensor data collection" and it will return you some files or some extensions that are implementing the type of methods or workflows that you're trying to go after. This uses semantic search over a comprehensive database of Isaac Sim code examples extracted from the codebase. Get code examples will be a pretty clear one that would be really useful, especially for learning how to create robots, set up physics, integrate sensors, or work with USD in Isaac Sim context.

4) Finally, there's a settings search tool. Isaac Sim has a comprehensive settings system for configuring everything from physics parameters to rendering options to robot simulation behaviors. You'll be able to search for settings using natural language queries, and the tool will return relevant settings with their paths, types, default values, and documentation. This is particularly useful for understanding how to configure physics simulation, rendering options, and Isaac Sim-specific behaviors. You can filter by setting prefix (like `/physics/`, `/isaac/`, `/rtx/`) or by type (bool, int, float, string).

5) Another important tool is to get instructions and documentation. There are **49 instruction sets** covering the full breadth of Isaac Sim development. Key examples include:
   - `isaacsim_system`: Introduction to Isaac Sim — design, simulation, sensors, digital twins, and core capabilities
   - `robot_setup`: Robot Wizard, editors, assembler, and robot building tutorials
   - `robot_simulation`: Wheeled robots, manipulators, policy control, motion generation
   - `sensors`: Cameras, depth, RTX sensors, physics-based sensors, calibration
   - `physics`: PhysX, Newton, USD schemas, simulation flow
   - `ros_2`: ROS 2 bridge, installation, workspaces, tutorials
   - `python_scripting_and_tutorials`: Python scripting (standalone/interactive), core API, tutorials
   - `omnigraph`: Visual programming for Replicators, ROS 2, sensors, controllers
   - `isaac_lab`: Robot learning framework with RL/imitation learning
   - And many more — call the tool without parameters to list all 49 available instruction sets.

These instruction sets provide comprehensive guidance for developers working with Isaac Sim.

Based on this I have defined the following API:

# Isaac Sim MCP Tools

## Documentation

- get_isaac_sim_instructions(instruction_sets: [str]) - Gather System Instructions about Isaac Sim
    - 49 instruction sets covering all aspects of Isaac Sim development
    - Key sets: `isaacsim_system`, `robot_setup`, `robot_simulation`, `sensors`, `physics`, `ros_2`, `python_scripting_and_tutorials`, `omnigraph`, `isaac_lab`, and more
    - Call without parameters to list all available instruction sets

## Extension Discovery

- search_isaac_sim_extensions(query: str, top_k: int) - Find relevant Isaac Sim extensions by topic/functionality
- get_isaac_sim_extension_details(extension_ids: [str]) - Get detailed information about extensions including features, dependencies, and API availability

## Code & Examples

- search_isaac_sim_code_examples(query: str, top_k: int) - Search for relevant Isaac Sim code examples using semantic search
  - Finds robot setup examples, sensor integration patterns, physics simulation code
  - Returns complete source code with file paths and line numbers
  - Includes relevance scoring and tags

## Settings & Configuration

- search_isaac_sim_settings(query: str, top_k: int, prefix_filter: str, type_filter: str) - Search Isaac Sim configuration settings
  - Natural language search across all Isaac Sim settings
  - Filter by prefix: `exts`, `app`, `persistent`, `rtx`
  - Filter by type: bool, int, float, string, array, object
  - Returns setting paths, types, defaults, documentation, and usage information

## Key Differences from Kit MCP

The Isaac Sim MCP is specialized for robotics and simulation development:

1. **Focused Scope**: Only Isaac Sim extensions, not all Kit extensions
2. **Robotics Emphasis**: Code examples and documentation focus on robot creation, articulation, sensors, and physics
3. **Specialized Instructions**: Includes robotics-specific development patterns
4. **Physics Settings**: Enhanced focus on physics and simulation configuration settings
5. **No Runtime Tools**: Currently focused on documentation and discovery (no live instance interaction yet)

## Data Pipeline

The Isaac Sim MCP relies on an automated data collection pipeline that processes Isaac Sim extensions:

1. **Extension Data Collection**: Processes extension.toml files, generates Code Atlas for APIs
2. **Code Examples Extraction**: Analyzes Code Atlas to find interesting robotics and simulation code patterns
3. **Settings Discovery**: Extracts settings from TOML files and source code, tracking usage
4. **Embeddings Generation**: Generates semantic embeddings using NVIDIA models (nv-embedqa-e5-v5)
5. **FAISS Database Creation**: Builds vector databases for fast semantic search

The pipeline outputs versioned data structures organized by Isaac Sim version, allowing support for multiple versions.

## Use Cases

The Isaac Sim MCP is particularly valuable for:

- **Learning Isaac Sim**: Discover extensions and understand their purposes
- **Robot Development**: Find examples for articulation setup, sensor integration, control
- **Physics Configuration**: Understand and configure physics simulation parameters
- **Extension Development**: Learn patterns for creating Isaac Sim extensions
- **Debugging**: Find relevant settings and understand their impact
- **Migration**: Understand differences between Isaac Sim versions

## Integration

The Isaac Sim MCP integrates with:
- **Cursor IDE**: Via MCP protocol on port 9904
- **NAT Framework**: Built using NAT (NeMo Agent Toolkit) for robust function registration
- **LangChain**: For embedding and vector store integration
- **FAISS**: For high-performance semantic search
- **Redis** (optional): For usage telemetry and analytics

The server can run alongside other MCP servers like Kit MCP (port 9902), allowing developers to access both general Kit documentation and specialized Isaac Sim tools simultaneously.
