## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

"""
This file is duplicated from:
source/extensions/omni.ai.chat_usd.bundle/omni/ai/chat_usd/bundle/register_chat_model.py

We had two possible solutions for chat model registration:

1. Create omni.ai.langchain.chat_models:
   - Chat models are registered in omni.ai.chat_usd.bundle
   - omni.ai.chat_usd.bundle depends on the widget
   - We can't load omni.ai.chat_usd.bundle in headless mode
   - Making widget dependency optional isn't viable as users test Chat USD by loading
     omni.ai.chat_usd.bundle and expect to see the window

   This approach is problematic because API keys would need to be specified in different
   extension.toml files, while we already have key specification documented for
   omni.ai.chat_usd.bundle.

2. Duplicate register_chat_model.py:
   - Duplicating the file to omni.ai.langchain.agent.headless
   - This requires specifying API keys in both omni.ai.chat_usd.bundle and
     omni.ai.langchain.agent.headless

We chose the second solution despite the duplication, as it provides better user
experience and maintains compatibility with existing documentation.
"""

from pathlib import Path

import carb
from langchain_openai import ChatOpenAI
from lc_agent import get_chat_model_registry

from .chat_models.chat_nvnim import ChatNVNIM

# Settings paths for easy access and reuse
SETTING_PATH_API_KEY = "/exts/omni.ai.langchain.agent.headless/nvidia_api_key"
SETTING_PATH_FUNC_ID = "/exts/omni.ai.langchain.agent.headless/nvidia_func_id"
SETTING_PATH_CUSTOM_MODEL = "/exts/omni.ai.langchain.agent.headless/custom_chat_model"

# Model configurations
MODELS = {
    "nvidia/usdcode-llama3-70b-instruct": (
        {
            "model": "nvidia/usdcode-llama3-70b-instruct",
            "temperature": 0.1,
            "max_tokens": 1024,
            "extra_body": {
                "rag_top_k": 5,
                "rag_max_tokens": 2000,
            },
        },
        6192,  # 8192 - rag_max_tokens
        False,
        None,
    ),
    "nvidia/usdcode-llama3-70b-instruct-interactive": (
        {
            "model": "nvidia/usdcode-llama3-70b-instruct",
            "temperature": 0.1,
            "max_tokens": 1024,
            "extra_body": {
                "rag_type": "none",
                "rag_top_k": 0,
                "rag_max_tokens": 0,
            },
        },
        8192,
        True,
        None,
    ),
    "stg/nvidia/usdcode-llama-3.1-70b-instruct": (
        {
            "model": "stg/nvidia/usdcode-llama-3.1-70b-instruct",
            "temperature": 0.1,
            "max_tokens": 4 * 1024,
        },
        128 * 1024,
        False,
        "ChatNVNIM",
    ),
    "127.0.0.1:7000": (
        {
            "model": "meta/llama-3.1-70b-instruct",
            "temperature": 0.1,
            "max_tokens": 4 * 1024,
            "base_url": "http://127.0.0.1:7000/code/completion",
        },
        128 * 1024,
        False,
        "ChatNVNIM",
    ),
    "meta/llama-3.1-70b-instruct": (
        {
            "model": "meta/llama-3.1-70b-instruct",
            "temperature": 0.1,
            "max_tokens": 4 * 1024,
            "base_url": "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/{func_id}",
        },
        128 * 1024,
        False,
        None,
    ),
    "meta/llama-3.1-405b-instruct": (
        {
            "model": "meta/llama-3.1-405b-instruct",
            "temperature": 0.1,
            "max_tokens": 4 * 1024,
        },
        128 * 1024,
        False,
        None,
    ),
}


def register_chat_model(model_names=None, api_key=None):
    # Fallback URL
    base_url = "https://integrate.api.nvidia.com/v1"

    # API Key
    if api_key is None:
        settings = carb.settings.get_settings().get(SETTING_PATH_API_KEY)
        if settings:
            api_key = settings
        else:
            # Get from NVIDIA_API_KEY environment variable
            import os

            api_key = os.environ.get("NVIDIA_API_KEY")
            if api_key is None:
                carb.log_warn(f"NVIDIA_API_KEY is required for {list(MODELS.keys())[0]} model")

    # Func ID
    func_id = carb.settings.get_settings().get(SETTING_PATH_FUNC_ID)
    if not func_id:
        import os

        func_id = os.environ.get("NVIDIA_FUNC_ID")

    model_registry = get_chat_model_registry()

    # Add custom models from settings
    settings = carb.settings.get_settings()
    custom_models = settings.get(SETTING_PATH_CUSTOM_MODEL)

    if not custom_models and not api_key:
        carb.log_warn("No custom models or api_key provided, skipping usdcode models registration")
        return

    if custom_models:
        # Convert string to dict with base_url
        if isinstance(custom_models, str):
            custom_models = {"base_url": custom_models}

        # Convert single dict to list
        if isinstance(custom_models, dict):
            custom_models = [custom_models]

        # Process each custom model
        for idx, model_config in enumerate(custom_models):
            # Extract and set defaults for special fields
            nice_name = model_config.pop("nice_name", None)
            context_window_size = model_config.pop("context_window_size", 128 * 1024)
            hidden = model_config.pop("hidden", False)

            # Generate model name if not provided
            if not nice_name:
                if "model" in model_config:
                    nice_name = model_config["model"]
                else:
                    nice_name = f"Custom Model {idx:02d}" if idx > 0 else "Custom Model"

            # Skip if model_names is specified and this model isn't in it
            if model_names is not None and nice_name not in model_names:
                continue

            # Create model instance
            model_args = model_config.copy()
            if api_key and "api_key" not in model_args:
                model_args["api_key"] = api_key

            # Handle custom URL
            custom_url = model_args.pop("custom_url", None)
            if custom_url:
                model = ChatNVNIM(**model_args)

                # Patching to use exact URL and streaming support
                async def edit_url_and_post_async(
                    *args, original_post=model.async_client._post, url=custom_url, **kwargs
                ):
                    kwargs["options"]["headers"] = {"Accept": "text/event-stream"}
                    args = (url,) + args[1:]
                    return await original_post(*args, **kwargs)

                async def edit_url_and_post(*args, original_post=model.client._post, url=custom_url, **kwargs):
                    kwargs["options"]["headers"] = {"Accept": "text/event-stream"}
                    args = (url,) + args[1:]
                    return await original_post(*args, **kwargs)

                model.async_client._post = edit_url_and_post_async
                model.client._post = edit_url_and_post
            else:
                model = ChatNVNIM(**model_args)

            # Register the model
            max_tokens = context_window_size - model_args.get("max_tokens", 1024)
            model_registry.register(
                nice_name,
                model,
                None,  # No tokenizer
                max_tokens,
                hidden,
            )

    # Continue with built-in models
    models = MODELS
    for name, config in models.items():
        if model_names is None or name in model_names:
            args, max_tokens, hidden, chat_model_class = config

            args = args.copy()
            custom_url = args.pop("base_url", None)
            custom_func_id = args.pop("func_id", None) or func_id
            if custom_func_id and custom_url and "{func_id}" in custom_url:
                url = custom_url.replace("{func_id}", custom_func_id)
            elif custom_url:
                url = custom_url
            else:
                url = base_url

            if chat_model_class and chat_model_class == "ChatNVNIM":
                model = ChatNVNIM(api_key=api_key, base_url=url, **args)
            else:
                model = ChatOpenAI(api_key=api_key, base_url=url, **args)

            if custom_url:
                # Patching to use exact URL and streaming support
                async def edit_url_and_post(*args, original_post=model.async_client._post, url=url, **kwargs):
                    kwargs["options"]["headers"] = {"Accept": "text/event-stream"}
                    args = (url,) + args[1:]
                    return await original_post(*args, **kwargs)

                model.async_client._post = edit_url_and_post

            model_registry.register(
                name,
                model,
                None,  # No tokenizer
                max_tokens - args["max_tokens"],
                hidden,
            )


def unregister_chat_model(model_names=None):
    model_registry = get_chat_model_registry()
    models = MODELS.keys()
    for name in models:
        if model_names is None or name in model_names:
            model_registry.unregister(name)
