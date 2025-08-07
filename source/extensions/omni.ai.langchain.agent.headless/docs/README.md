## Headless USD Agent (Preview)

This extension enables headless USD manipulation in Omniverse Kit without requiring a graphical interface. It provides AI-assisted USD code generation and manipulation capabilities specifically designed for automated and server-side operations.

Key features:
- Headless USD stage manipulation
- Command-line interface support
- Batch processing capabilities
- Pipeline integration support
- Automated USD modifications through natural language prompts

## Overview

The `omni.ai.langchain.agent.headless` extension is designed for automated, headless manipulation of USD files in Omniverse Kit. This extension enables batch processing and automation of USD modifications without requiring a graphical interface or viewport, making it ideal for server-side operations and automated workflows.

### Use Cases

1. **Automated USD Processing**: Process and modify USD files through natural language commands in headless environments.
2. **Batch Operations**: Perform bulk modifications on multiple USD files without requiring GUI interaction.
3. **Server-Side USD Manipulation**: Enable USD modifications in server environments where graphical interfaces are unavailable.
4. **Pipeline Integration**: Seamlessly integrate USD modifications into automated content pipelines.
5. **Command-Line Operations**: Execute USD modifications through command-line interfaces and scripts.

## Setup

### Building the Kit App

To use headless mode, you need to compile the Kit app. A simple example app is provided in the `app` folder of the extension. The app's main purpose is to include the required dependencies:

```toml
[dependencies]
"omni.ai.langchain.agent.headless" = {}
"omni.ai.langchain.agent.usd_code" = {}
```

### API Key Configuration

To use the LLM capabilities, you need to set up your NVIDIA API key. There are several ways to configure this:

1. Environment variable:
```
set NVIDIA_API_KEY=your_api_key_here
```

2. Extension settings in `extension.toml`:
```toml
[settings.exts."omni.ai.langchain.agent.headless"]
nvidia_api_key = "your_api_key_here"
```

### Custom Chat Models

You can configure custom models in the extension settings:

```toml
[settings.exts."omni.ai.langchain.agent.headless"]
custom_chat_model = "http://127.0.0.1:9901/"
```

The extension supports using any LangChain-compatible chat model. See the `register_chat_model.py` file for examples of custom model configurations and supported parameters.

## Usage

To use the headless application, you can run it from the command line with the following parameters:

### Command Line Parameters

- `--/exts/omni.ai.langchain.agent.headless/stage`: Path to the USD file to manipulate
- `--/exts/omni.ai.langchain.agent.headless/prompt`: Natural language prompt describing the desired modifications
- `--/exts/omni.ai.langchain.agent.headless/agent`: Agent type to use (default: "USD Code Interactive")
- `--/exts/omni.ai.langchain.agent.headless/model`: AI model to use (default: "meta/llama-3.1-70b-instruct")

### Example

```
_build\windows-x86_64\release\omni.app.chat_usd.headless.bat ^
    --/exts/omni.ai.langchain.agent.headless/stage=C:\usd\cone.usda ^
    --/exts/omni.ai.langchain.agent.headless/prompt="Find the cone in the scene. It can have any name. Move it 1 unit up."
```

This command will:
1. Open the specified USD file (cone.usda)
2. Process the natural language prompt using the specified agent and model
3. Execute the requested modifications (finding and moving the cone)
4. Save the changes back to the file
5. Exit automatically upon completion

Note: The `agent` and `model` parameters are optional and will use the default values if not specified.

The extension is particularly valuable for automation workflows, continuous integration pipelines, and server-side applications where visual feedback is not required or available.
