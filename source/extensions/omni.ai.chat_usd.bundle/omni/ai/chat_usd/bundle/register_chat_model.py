## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
from pathlib import Path
from typing import Any, Dict, Sequence

import carb.settings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from lc_agent import get_chat_model_registry

from .chat_models.chat_nvnim import ChatNVNIM
from .tokenizer import Tokenizer


def get_custom_payload_fn(custom_payload):
    """Return a function that adds custom payload to the payload of the model."""

    def payload_fn(payload):
        for key, value in custom_payload.items():
            payload[key] = value
        return payload

    return payload_fn


class NoThinkChatNVIDIA(ChatNVIDIA):
    """Custom ChatNVIDIA class that prepends the /no_think directive to system messages.

    This is used for models like Nemotron that support the /no_think directive,
    which instructs the model to skip internal reasoning and provide direct responses.
    """

    def _get_payload(self, inputs: Sequence[Dict], **kwargs: Any) -> dict:
        payload = super()._get_payload(inputs, **kwargs)

        messages = payload.get("messages", [])

        # Prepend /no_think directive to the first system message
        if messages:
            first_message = messages[0]
            if first_message.get("role") == "system":
                first_message["content"] = "/no_think\n" + first_message["content"]

        return payload


MODELS = {
    "meta/llama-4-maverick-17b-128e-instruct": (
        {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "max_tokens": 4 * 1024,
            "temperature": 0.0,
            "top_p": 0.95,
        },
        256 * 1024,
        False,
        None,
        True,
    ),
    # Nemotron model with no_think mode enabled for faster direct responses
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": (
        {
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1.5",
            "max_tokens": 4 * 1024,
            "no_think": True,  # Enables /no_think directive for this model
        },
        128 * 1024,  # context window size
        False,  # not hidden
        None,  # use default chat model class
        True,  # devmode_only
    ),
    "openai/gpt-oss-120b": (
        {
            "model": "openai/gpt-oss-120b",
            "max_tokens": 4 * 1024,
        },
        128 * 1024,
        False,
        None,
        True,
    ),
    "openai/gpt-4o": (
        {
            "model": "gpt-4o",
            "max_tokens": 4 * 1024,
        },
        128 * 1024,
        False,
        "ChatOpenAI",
        True,
    ),
}

TOKENIZER_PATH = Path(__file__).parent.joinpath("../../../../data/Llama3-70B-tokenizer.model")


def register_chat_model(model_names=None, api_key=None, register_all_lc_agent_models=False):
    already_registered = False
    if register_all_lc_agent_models:
        try:
            from lc_agent_chat_models import register_all

            imported_successfully = True
        except ImportError:
            print("ERROR: Failed to import lc_agent_chat_models")
            imported_successfully = False

        if imported_successfully:
            register_all()
            already_registered = True

    if already_registered:
        return

    chat_usd_developer_mode = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/chat_usd_developer_mode")
    chat_usd_developer_mode = chat_usd_developer_mode or os.environ.get("USD_AGENT_DEV_MODE")

    # Fallback URL
    base_url = "https://integrate.api.nvidia.com/v1"

    # API Key
    if api_key is None:
        settings = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/nvidia_api_key")
        if settings:
            api_key = settings
        else:
            # Get from NVIDIA_API_KEY environment variable
            api_key = os.environ.get("NVIDIA_API_KEY")
            if api_key is None:
                carb.log_warn(f"NVIDIA_API_KEY is required for {list(MODELS.keys())[0]} model")

    # OpenAI API Key for native OpenAI models
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/openai_api_key")

    # Func ID
    func_id = carb.settings.get_settings().get("/exts/omni.ai.chat_usd.bundle/nvidia_func_id")
    if not func_id:
        func_id = os.environ.get("NVIDIA_FUNC_ID")

    model_registry = get_chat_model_registry()

    # Add custom models from settings
    settings = carb.settings.get_settings()
    custom_models = settings.get("/exts/omni.ai.chat_usd.bundle/custom_chat_model")

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

            if "temperature" not in model_args:
                model_args["temperature"] = 0.1

            if "max_tokens" not in model_args:
                model_args["max_tokens"] = 4096

            if "top_p" not in model_args:
                model_args["top_p"] = 0.95

            # Handle custom URL
            custom_url = model_args.pop("custom_url", None)

            # Create a factory function for lazy initialization
            def create_custom_model(model_args=model_args, custom_url=custom_url):
                # Make a fresh copy of args at call time to avoid mutations
                model_args = model_args.copy()
                # Extract no_think flag - if True, use NoThinkChatNVIDIA instead of regular model
                no_think = model_args.pop("no_think", False)

                if custom_url:
                    # Use NoThinkChatNVIDIA for models that require /no_think directive
                    if no_think:
                        model = NoThinkChatNVIDIA(**model_args)
                    else:
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
                    # For non-custom URLs, still check if no_think mode is needed
                    if no_think:
                        model = NoThinkChatNVIDIA(**model_args)
                    else:
                        model = ChatNVNIM(**model_args)

                return model

            tokenizer = Tokenizer(model_path=f"{TOKENIZER_PATH}")

            # Register the model factory function
            max_tokens = context_window_size - model_args.get("max_tokens", 1024)
            model_registry.register(
                nice_name,
                create_custom_model,
                tokenizer,
                max_tokens,
                hidden,
            )

    # Continue with built-in models
    models = MODELS
    for name, config in models.items():
        if model_names is None or name in model_names:
            args, max_tokens, hidden, chat_model_class, devmode_only = config

            # Skip OpenAI models if no OpenAI API key is available
            if chat_model_class == "ChatOpenAI" and not openai_api_key:
                carb.log_warn(
                    f"Skipping {name} registration: OPENAI_API_KEY or /exts/omni.ai.chat_usd.bundle/openai_api_key not specified"
                )
                continue

            hidden = hidden or (not chat_usd_developer_mode and devmode_only)

            args = args.copy()
            custom_url = args.pop("base_url", None)
            custom_func_id = args.pop("func_id", None) or func_id
            if custom_func_id and custom_url and "{func_id}" in custom_url:
                url = custom_url.replace("{func_id}", custom_func_id)
            else:
                url = base_url

            tokenizer = Tokenizer(model_path=f"{TOKENIZER_PATH}")

            # Create a factory function for lazy initialization
            def create_builtin_model(
                api_key=api_key,
                url=url,
                args=args,
                chat_model_class=chat_model_class,
                custom_url=custom_url,
                openai_api_key=openai_api_key,
            ):
                # Make a fresh copy of args at call time to avoid mutations
                args = args.copy()

                # Extract no_think flag for models that support /no_think directive
                no_think = args.pop("no_think", False)

                if chat_model_class == "ChatOpenAI":
                    # Native OpenAI model - no base_url, use OpenAI API key
                    model = ChatOpenAI(api_key=openai_api_key, **args)
                elif chat_model_class == "ChatNVNIM":
                    model = ChatNVNIM(api_key=api_key, base_url=url, **args)
                elif custom_url:
                    model = ChatOpenAI(api_key=api_key, base_url=url, **args)
                elif no_think:
                    # Use NoThinkChatNVIDIA for models like Nemotron that benefit from /no_think
                    model = NoThinkChatNVIDIA(api_key=api_key, base_url=url, **args)
                else:
                    model = ChatNVIDIA(api_key=api_key, base_url=url, **args)

                if custom_url:
                    # Patching to use exact URL and streaming support
                    async def edit_url_and_post(*args, original_post=model.async_client._post, url=url, **kwargs):
                        kwargs["options"]["headers"] = {"Accept": "text/event-stream"}
                        args = (url,) + args[1:]
                        return await original_post(*args, **kwargs)

                    model.async_client._post = edit_url_and_post

                return model

            model_registry.register(
                name,
                create_builtin_model,
                tokenizer,
                max_tokens - args["max_tokens"],
                hidden,
            )


def unregister_chat_model(model_names=None, unregister_all_lc_agent_models=False):
    already_unregistered = False
    if unregister_all_lc_agent_models:
        try:
            from lc_agent_chat_models import unregister_all

            imported_successfully = True
        except ImportError:
            imported_successfully = False

        if imported_successfully:
            unregister_all()
            already_unregistered = True

    if already_unregistered:
        return

    model_registry = get_chat_model_registry()
    models = MODELS.keys()
    for name in models:
        if model_names is None or name in model_names:
            model_registry.unregister(name)
