# Extension Integration

This document explains how Chat USD integrates with Omniverse Kit as an extension, including the extension lifecycle, configuration, and integration with other extensions.

## Overview

Chat USD is implemented as an Omniverse Kit extension, which allows it to integrate seamlessly with the Omniverse ecosystem. The extension is defined in the `ChatUSDBundleExtension` class, which extends `omni.ext.IExt` and implements the required lifecycle methods.

## Extension Structure

The Chat USD extension follows the standard Omniverse Kit extension structure:

```text
omni.ai.chat_usd.bundle/
├── config/
│   └── extension.toml
├── docs/
│   └── ...
├── omni/
│   └── ai/
│       └── chat_usd/
│           └── bundle/
│               ├── __init__.py
│               ├── extension.py
│               ├── register_chat_model.py
│               ├── tokenizer.py
│               ├── chat/
│               │   └── ...
│               ├── search/
│               │   └── ...
│               └── ...
└── ...
```

This structure follows the Omniverse Kit extension conventions, with the main extension code in the `omni/ai/chat_usd/bundle` directory and configuration in the `config` directory.

## Extension Registration

The extension is registered with Omniverse Kit through the `extension.toml` file, which defines the extension's metadata and dependencies:

```toml
[package]
# Semantic Versioning is used: https://semver.org/
version = "1.0.0"
category = "AI"
title = "Chat USD"
description = "Chat USD is a specialized AI assistant for USD development."
authors = ["NVIDIA"]
repository = ""
keywords = ["ai", "usd", "chat", "assistant"]
changelog = "docs/CHANGELOG.md"
readme = "docs/README.md"
preview_image = "data/preview.png"
icon = "data/icon.png"

# Dependencies
[dependencies]
"omni.kit.uiapp" = {}
"omni.ui" = {}
"omni.usd" = {}
"omni.ai.langchain.agent.usd_code" = {}
"omni.ai.langchain.agent.usd_search" = {}
```

This file specifies the extension's version, category, title, description, and dependencies, allowing Omniverse Kit to load the extension and its dependencies correctly.

## Extension Lifecycle

The `ChatUSDBundleExtension` class implements the extension lifecycle methods:

```python
class ChatUSDBundleExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        # Initialization code
        register_chat_model(
            register_all_lc_agent_models=carb.settings.get_settings().get(REGISTER_ALL_CHAT_MODELS_SETTING)
        )

        # Register components
        get_node_factory().register(USDSearchNetworkNode, name="USD Search")
        get_node_factory().register(USDSearchNode, hidden=True)

        # More initialization code...

    def on_shutdown(self):
        # Cleanup code
        unregister_chat_model(
            unregister_all_lc_agent_models=carb.settings.get_settings().get(REGISTER_ALL_CHAT_MODELS_SETTING)
        )

        # Unregister components
        get_node_factory().unregister(USDSearchNetworkNode)
        get_node_factory().unregister(USDSearchNode)

        # More cleanup code...
```

These methods are called by Omniverse Kit when the extension is started and shut down, allowing the extension to initialize and clean up its resources.

## Component Registration

During startup, the extension registers its components with the node factory:

```python
# Register Chat USD Network Node only if the setting is on, it is Off by default
register_chat_usd_agent = carb.settings.get_settings().get(REGISTER_CHAT_USD_SETTING)
if register_chat_usd_agent:
    from .chat.chat_usd_network_node import ChatUSDNetworkNode, ChatUSDSupervisorNode

    get_node_factory().register(ChatUSDSupervisorNode, hidden=True)
    get_node_factory().register(ChatUSDNetworkNode, name="Chat USD", multishot=chat_usd_multishot)

    # Register Chat USD tools
    get_node_factory().register(
        USDCodeInteractiveNetworkNode,
        name="ChatUSD_USDCodeInteractive",
        scene_info=False,
        enable_code_interpreter=enable_code_interpreter,
        code_interpreter_hide_items=code_interpreter_hide_items,
        enable_code_atlas=need_rags,
        enable_metafunctions=need_rags,
        enable_interpreter_undo_stack=True,
        max_retries=1,
        enable_code_promoting=True,
        hidden=True,
    )

    # More component registration...
```

This registration makes the components available for use by other extensions and the Omniverse Kit UI.

## Configuration Settings

The extension uses Carb settings to configure its behavior:

```python
REGISTER_CHAT_USD_SETTING = "/exts/omni.ai.chat_usd.bundle/register_chat_usd_agent"
REGISTER_USD_TUTOR_SETTING = "/exts/omni.ai.chat_usd.bundle/register_usd_tutor_agent"
REGISTER_ALL_CHAT_MODELS_SETTING = "/exts/omni.ai.chat_usd.bundle/register_all_chat_models"
REGISTER_CHAT_USD_OMNI_UI_SETTING = "/exts/omni.ai.chat_usd.bundle/chat_usd_with_omni_ui"
```

These settings can be configured through the Omniverse Kit settings UI or programmatically, allowing users to customize the extension's behavior.

## UI Integration

The extension integrates with the Omniverse Kit UI through the `ChatView` class, which provides a chat interface for interacting with Chat USD:

```python
try:
    from omni.ai.langchain.widget.core import ChatView

    from .chat.chat_usd_network_node_delegate import ChatUSDNetworkNodeDelegate
    from .chat.multi_agent_delegate import SupervisorNodeDelegate, ToolNodeDelegate

    ChatView.add_delegate("ChatUSDNetworkNode", ChatUSDNetworkNodeDelegate())
    ChatView.add_delegate("RunnableSupervisorNode", SupervisorNodeDelegate())
    ChatView.add_delegate("RunnableToolNode", ToolNodeDelegate())

except ImportError:
    # this extension is not available in the current environment
    pass
```

This integration allows users to interact with Chat USD through a user-friendly chat interface.

## Dependency Management

The extension manages its dependencies through the `extension.toml` file and runtime checks:

```python
try:
    from omni.ai.langchain.widget.core import ChatView

    from .search.usd_search_delegate import USDSearchImageDelegate

    ChatView.add_delegate("USDSearchNode", USDSearchImageDelegate())
    ChatView.add_delegate("USDSearchNetworkNode", USDSearchImageDelegate())
except ImportError:
    # this extension is not available in the current environment
    # print("ChatView not available")
    pass
```

These checks ensure that the extension works correctly even if some dependencies are not available.

## Chat Model Registration

The extension registers chat models through the `register_chat_model` function:

```python
def register_chat_model(register_all_lc_agent_models=False):
    """Register chat models for the extension."""
    # Register the chat model
    if register_all_lc_agent_models:
        # Register all LC Agent models
        register_all_models()
    else:
        # Register only the models needed for this extension
        register_model("nvidia/usdcode-llama3-70b-instruct")
        register_model("nvidia/usdcode-llama3-70b-instruct-interactive")
```

This function registers the chat models used by Chat USD, making them available for use by the extension.

## Extension Variants

The extension supports different variants, such as the omni.ui variant, which adds UI generation capabilities:

```python
if carb.settings.get_settings().get(REGISTER_CHAT_USD_OMNI_UI_SETTING):
    try:
        from omni.ai.langchain.agent.omni_ui.nodes import OmniUICodeNetworkNode

        from .chat.chat_usd_network_node import (
            ChatUSDWithOmniUINetworkNode,
            ChatUSDWithOmniUISupervisorNode,
        )

        omni_ui_imported = True
    except ImportError:
        # this extension is not available in the current environment
        omni_ui_imported = False

    if omni_ui_imported:
        get_node_factory().register(ChatUSDWithOmniUISupervisorNode, hidden=True)
        get_node_factory().register(
            OmniUICodeNetworkNode, name="OmniUI_Code", hidden=True, rag=False, code_interpreter=False
        )
        get_node_factory().register(
            ChatUSDWithOmniUINetworkNode, name="Chat USD with omni.ui", multishot=chat_usd_multishot
        )
```

This variant adds the `OmniUI_Code` agent to the route nodes, allowing Chat USD to generate UI code in addition to USD code.

## Integration with Other Extensions

Chat USD integrates with other extensions, such as `omni.ai.langchain.agent.usd_code` and `omni.ai.langchain.agent.usd_search`:

```python
from omni.ai.langchain.agent.usd_code import SceneInfoNetworkNode, USDCodeInteractiveNetworkNode
from omni.ai.langchain.agent.usd_code.extension import USDCodeExtension
```

This integration allows Chat USD to leverage the capabilities of these extensions to provide a comprehensive USD development assistant.
