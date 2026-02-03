# OmniUI MCP Server - Comprehensive Documentation

## Project Overview

The OmniUI MCP Server is a Model Context Protocol (MCP) server implementation that provides stub classes for OmniUI development tools. Built using the AIQ toolkit framework, it follows the same architectural patterns as the USD Code MCP server and serves as a demonstration of how to create simple MCP servers with the AIQ infrastructure.

## Purpose and Functionality

This MCP server demonstrates:
- Implementation of a minimal MCP server using AIQ toolkit
- Stub implementations for two conceptual classes: ClaudeCode and OmniUI
- Integration with usage logging and analytics
- Docker containerization for deployment
- Development and production configurations

### Core Functionality

The server exposes a single tool function:
- **`list_ui_classes`**: Returns information about available stub classes (ClaudeCode and OmniUI)
  - No parameters required
  - Returns JSON with class names, count, and description
  - Includes usage logging and error handling

## Architecture and Components

### Directory Structure

```
source/mcp/omni_ui_mcp/
├── examples/
│   └── config.yaml                     # Configuration file for the MCP server
├── src/
│   └── omni_ui_mcp/
│       ├── __init__.py                 # Package initialization with AIQ patching
│       ├── __main__.py                 # Entry point for running the server
│       ├── config.py                   # Configuration constants and utilities
│       ├── register_get_classes.py     # AIQ function registration wrapper
│       ├── functions/
│       │   ├── __init__.py            # Functions module placeholder
│       │   └── list_ui_classes.py         # Core function implementation
│       ├── models/
│       │   └── __init__.py            # Models module placeholder
│       └── utils/
│           ├── __init__.py            # Utils module placeholder
│           ├── usage_logging.py       # Usage logging implementation
│           └── usage_logging_decorator.py  # Logging decorator utilities
├── Dockerfile                          # Docker container definition
├── pyproject.toml                      # Poetry configuration and dependencies
├── README.md                           # User-facing documentation
├── build-docker.bat                    # Windows Docker build script
├── build-docker.sh                     # Unix Docker build script
├── run.bat                       # Windows local run script
├── run.sh                        # Unix local run script
├── setup-dev.bat                       # Windows development setup script
├── setup-dev.sh                        # Unix development setup script
└── start-server.bat                    # Simple Windows server start script
```

### Core Components Analysis

#### 1. Package Initialization (`__init__.py`)
- **Version**: 0.1.0
- **Key Feature**: Patches AIQConfig to allow extra fields in YAML configuration
- **Usage Logging**: Initializes global usage logger on package import
- **Environment Control**: Respects `OMNI_UI_DISABLE_USAGE_LOGGING` environment variable

#### 2. Main Entry Point (`__main__.py`)
- **Purpose**: CLI entry point for the MCP server
- **Configuration Detection**: 
  - Searches multiple paths for config files
  - Supports both local development and production configs
  - Auto-detects Docker environment
- **Port Configuration**: Uses `MCP_PORT` environment variable (default: 9901)
- **Execution**: Runs AIQ MCP server via subprocess

#### 3. Configuration Module (`config.py`)
- **Default Values**:
  - MCP Port: 9901
  - Timeout: 30.0 seconds
  - Usage Logging: Enabled by default
- **OpenSearch Integration**: Points to AWS OpenSearch for analytics
- **Utility Functions**:
  - `get_env_bool()`: Parse boolean environment variables
  - `get_env_int()`: Parse integer environment variables
  - `get_env_float()`: Parse float environment variables

#### 4. Function Registration (`register_get_classes.py`)
- **AIQ Integration**: Uses AIQ's `@register_function` decorator
- **Input Schema**: Defines `GetClassesInput` as empty Pydantic model (zero-argument function)
- **Wrapper Function**: Implements async wrapper with usage logging
- **Error Handling**: Comprehensive try-catch with error reporting
- **Tool Description**: Detailed multi-line description for MCP exposure

#### 5. Core Function (`list_ui_classes.py`)
- **Stub Classes Data**:
  - **ClaudeCode**: 4 methods (generate_code, explain_code, review_code)
  - **OmniUI**: 6 methods (create_window, create_button, create_label, etc.)
- **Return Format**: Simplified JSON with class names and count
- **Internal Function**: `get_detailed_classes()` for future expansion

#### 6. Usage Logging (`usage_logging.py`)
- **Global Logger**: Singleton pattern for usage tracking
- **Log Format**: JSON structured logging with timestamps
- **Tracked Metrics**:
  - Tool name
  - Parameters
  - Success/failure status
  - Execution time in milliseconds
  - Error messages (if any)

#### 7. Logging Decorator (`usage_logging_decorator.py`)
- **Decorator Pattern**: `@log_tool_usage` for automatic logging
- **Async Support**: Handles both sync and async functions
- **Parameter Extraction**: Automatically extracts Pydantic model parameters
- **Error Resilience**: Logging failures don't break main functionality

## Configuration Details

### YAML Configuration Structure (`config.yaml`)

```yaml
llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    temperature: 0.0
    max_tokens: 16384

functions:
  list_ui_classes:
    _type: list_ui_classes
    verbose: false

workflow:
  _type: react_agent
  llm_name: nim_llm
  tool_names:
    - list_ui_classes
```

### Package Configuration (`pyproject.toml`)

- **Package Name**: omni-ui-aiq
- **Version**: 0.1.0
- **Python Requirements**: >=3.11, <3.13
- **Key Dependencies**:
  - aiqtoolkit: 1.2.1
  - aiqtoolkit-langchain: 1.2.1
  - langchain-community: >=0.0.10
  - langchain-nvidia-ai-endpoints: >=0.0.4
  - faiss-cpu: >=1.7.0
  - aiohttp-sse: ^2.2.0

- **Entry Points**:
  - Console Script: `omni-ui-aiq`
  - AIQ Plugin: `omni_ui_mcp_get_classes`

## Build and Deployment

### Development Setup (Windows)
1. **setup-dev.bat**:
   - Checks Python installation
   - Installs Poetry if missing
   - Configures Poetry for local virtual environments
   - Installs all dependencies
   - Creates necessary directories

2. **run.bat**:
   - Verifies Poetry environment
   - Sets development environment variables
   - Runs server with local configuration
   - Port: 9901 (configurable)

### Docker Deployment
1. **Dockerfile**:
   - Base Image: python:3.11-slim
   - Installs from wheel file
   - Copies configuration examples
   - Sets environment variables
   - Exposes port 9901

2. **build-docker.bat**:
   - Cleans previous builds
   - Builds Python package with Poetry
   - Creates Docker image tagged as `omni-ui-mcp:latest`

## Stub Classes Documentation

### ClaudeCode Class (Stub)
**Description**: AI-powered code assistance interface

**Methods**:
1. `__init__()` - Initialize ClaudeCode instance
2. `generate_code(prompt: str) -> str` - Generate code from natural language
3. `explain_code(code: str) -> str` - Explain code functionality
4. `review_code(code: str) -> dict` - Review and suggest improvements

### OmniUI Class (Stub)
**Description**: User interface framework for building UIs

**Methods**:
1. `__init__()` - Initialize OmniUI instance
2. `create_window(title: str, width: int, height: int) -> Window` - Create UI window
3. `create_button(text: str, callback: callable) -> Button` - Create button widget
4. `create_label(text: str) -> Label` - Create text label
5. `create_text_field(placeholder: str) -> TextField` - Create text input
6. `apply_style(element: UIElement, style_dict: dict) -> None` - Apply styling

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_PORT` | 9901 | Server port number |
| `OMNI_UI_DISABLE_USAGE_LOGGING` | false | Disable usage analytics |
| `PYTHONPATH` | /app | Python module search path (Docker) |

## Development Workflow

### Local Development Steps
1. Run `setup-dev.bat` once to initialize environment
2. Run `run.bat` to start development server
3. Server available at `http://localhost:9901`
4. Modify code - changes reflected after restart
5. Use `poetry shell` for virtual environment access

### Adding New Functionality
1. Create new function in `src/omni_ui_mcp/functions/`
2. Create registration module `register_[function_name].py`
3. Add entry point to `pyproject.toml` under `[tool.poetry.plugins."aiq.components"]`
4. Update configuration in `examples/config.yaml`
5. Add function to workflow tool_names list

### Testing and Validation
- Run: `poetry run pytest` for unit tests
- Build: `poetry build` for package distribution
- Validate: `poetry run python validate-setup.py` (if available)

## Integration with IDEs

### Cursor IDE Configuration
```json
{
  "mcpServers": {
    "omni-ui-mcp": {
      "type": "mcp",
      "url": "http://localhost:9901/sse"
    }
  }
}
```

## Technical Implementation Notes

### AIQ Framework Integration
- Uses AIQ's builder pattern for function registration
- Leverages AIQ's MCP server implementation
- Follows AIQ's configuration schema
- Integrates with AIQ's workflow system

### Key Design Patterns
1. **Singleton Pattern**: Global usage logger instance
2. **Decorator Pattern**: Usage logging decorator
3. **Wrapper Pattern**: Function registration wrappers
4. **Configuration Pattern**: Environment-based configuration

### Error Handling Strategy
- Function-level try-catch blocks
- Graceful degradation for logging failures
- Detailed error messages in responses
- Success/failure tracking in usage logs

## Troubleshooting Guide

### Common Issues and Solutions

1. **Poetry Not Found**
   - Install from: https://python-poetry.org/docs/#installation
   - Add `~/.local/bin` to PATH (Unix)
   - Restart terminal after installation

2. **AIQ Command Not Found**
   - Run: `poetry install`
   - Activate environment: `poetry shell`
   - Verify: `poetry show aiqtoolkit`

3. **Port Already in Use**
   - Check port: `netstat -an | grep 9901`
   - Change port: Set `MCP_PORT` environment variable
   - Modify `examples/config.yaml` directly

4. **Virtual Environment Issues**
   - Delete `.venv` directory
   - Re-run setup script
   - Check Poetry config: `poetry config --list`

## Summary

The OmniUI MCP Server is a well-structured demonstration project that showcases:
- Clean implementation of an MCP server using AIQ toolkit
- Proper separation of concerns with modular architecture
- Comprehensive logging and error handling
- Docker containerization for deployment
- Development-friendly setup scripts
- Stub implementations for conceptual UI classes

The server serves as an excellent template for building custom MCP servers with the AIQ framework, providing patterns for function registration, usage tracking, and configuration management.