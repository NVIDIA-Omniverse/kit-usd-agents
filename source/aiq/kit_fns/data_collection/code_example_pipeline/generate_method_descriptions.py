# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import openai
import requests
import tiktoken
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_openai import AzureChatOpenAI
from lc_agent import (
    NetworkModifier,
    RunnableHumanNode,
    RunnableNetwork,
    RunnableNode,
    RunnableSystemAppend,
    get_chat_model_registry,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


_METHOD_DESCRIPTION_SYSTEM = """
You are an assistant that helps with writing descriptions for Python methods.
The user will provide you with the source code of the methods.

- You will return only the a sentence of description, no explanation, arguments or code.
- Don't write anything which has not been done in the provided code.
- Don't write sentences that don't provide specific, actionable information about the input code. Be concise. Avoid generic statements that could apply to anything. For example:
    - Avoid: "it provides a seamless integration point that enables users to view and interact with ACL data directly within the file browser."
    - Instead, focus on specific functionality, go straight to the point: "it enables users to view and interact with ACL data directly within the file browser."
- Focus on the primary purpose and effect from the user's perspective, not implementation details:
    - Avoid describing internal logic, error handling, or fallback mechanisms unless they are the main purpose
    - Focus on what the function accomplishes for the user, not how it accomplishes it
    - Be concise. Example: Instead of "Updates the stage by setting the scene template URL. If the property does not exist, creates an attribute to store the URL." use "Updates the stage by setting the scene template URL."
- Never replace the extension name or public API names with generic terms. For example:
    - Use "The omni.kit.ui.actions extension" instead of "The UI Actions and Hotkeys extension".
    - Use "omni.localcache" instead of "Local Cache" when referring to the specific extension name.
    - Use "USDCodeInteractiveNetworkNode" instead of "Code Node" when referencing specific class.
- Always preserve precise technical terminology from the input code rather than substituting with generic terms. Always prioritize technical accuracy over readability when the terms represent specific named entities in the codebase. Because precise names help users and developers locate the correct resources.
- Focus on what is the code serves for in the input context instead of just writing what the code does.
   - for example, if the code is a function that returns a list of strings in a file browser widget, don't say "it returns a list of strings", say "it returns a list of filenames".
   - Use consistent terminology across related methods. When one method establishes that `editing state` controls edit widget visibility, other methods should reference the same concept consistently (e.g., "`editing state` controls hover widget visibility" rather than "edit widget visibility controls hover widget visibility").
- There is no need to mention things that are obvious from the function signature.
    - for example, no need to state that a function is "using async stat call" or "asynchronously", users can easily tell from the async keyword.
- when the method is using acronym, try to derive the full name from the code. But keep the acronym if you can't find any, never hallucinate terminology which does not exist in the code.
So for example if the method is named "get_ptmb_fso", we want to explain "ptmb" and "fso" using "Path Trace Motion Blur" and "Frame Shutter Open" respectively in the description.

The output is a json payload with the following format:

{{
    "method_name": "description",
    "method_name2": "description2",
    ...
}}

In the json payload, use the actual new line or tab in the output. Never keep '\\n' or '\\t' in the output, and ensure that it is properly enclosed by '{{' and '}}'.
Only return the json payload, no other text or comments.
Please check the number of method descriptions in the json payload is equal to {num_methods}.
"""


def retry_model(_class):
    """
    A wrapper that adds retry to a model.
    """
    retry_decorator = create_base_retry_decorator(
        error_types=[
            openai.APIError,
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ],
        max_retries=3,
        run_manager=None,
    )

    class _RetryModel(_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def invoke(self, *args, **kwargs):
            @retry_decorator
            def _invoke(self, *args, **kwargs):
                return super(_class, self).invoke(*args, **kwargs)

            return _invoke(self, *args, **kwargs)

        async def ainvoke(self, *args, **kwargs):
            @retry_decorator
            async def _ainvoke(self, *args, **kwargs):
                return await super(_class, self).ainvoke(*args, **kwargs)

            return await _ainvoke(self, *args, **kwargs)

    return _RetryModel


def register_model(model_names=None, url=None):
    url = (
        url
        or "https://gitlab-master.nvidia.com/omniverse/kit-extensions/kit-ai-agent/uploads/f7bd6430e4eb4c47c551339d7aa36585/creds-perflab.json"
    )
    response = requests.get(url, timeout=180)
    if response.status_code == 200:
        data = response.json()
        api_key = data["api_key"]
    else:
        api_key = None

    model_registry = get_chat_model_registry()
    models = {
        "nvidia-llama-3.1-nemotron-70b-instruct": {
            "model_name": "nvidia-llama-3.1-nemotron-70b-instruct",
            "api_version": "2024-12-01-preview",
        },
        "nvidia-llama-3.1-nemotron-ultra-253b-v1": {
            "model_name": "nvidia-llama-3.1-nemotron-ultra-253b-v1",
            "api_version": "2024-12-01-preview",
        },
        "nvidia-llama-3.3-nemotron-super-49b-v1": {
            "model_name": "nvidia-llama-3.3-nemotron-super-49b-v1",
            "api_version": "2024-12-01-preview",
        },
    }

    _AzureChatOpenAIWithRetry = retry_model(AzureChatOpenAI)

    for name, config in models.items():
        if model_names is None or name in model_names:
            model_registry.register(
                name,
                _AzureChatOpenAIWithRetry(
                    api_key=api_key,
                    azure_endpoint="https://llm-proxy.perflab.nvidia.com",
                    timeout=120,
                    **config,
                ),
            )


def unregister_model(model_names=None):
    model_registry = get_chat_model_registry()
    models = [
        "nvidia-llama-3.1-nemotron-70b-instruct",
        "nvidia-llama-3.3-nemotron-super-49b-v1",
        "nvidia-llama-3.1-nemotron-ultra-253b-v1",
    ]
    for name in models:
        if model_names is None or name in model_names:
            model_registry.unregister(name)


class MethodDescriptionAgent(RunnableNode):
    def __init__(self, num_methods: int, **kwargs):
        """Initializes the MethodDescriptionAgent with a system prompt for method description generation."""
        super().__init__(**kwargs)
        self.inputs.append(
            RunnableSystemAppend(system_message=_METHOD_DESCRIPTION_SYSTEM.format(num_methods=num_methods))
        )


class MethodDescriptionModifier(NetworkModifier):

    def __init__(
        self,
        method_names: Set[str],
        method_descriptions: Dict[str, str],
        retry_times: int = 3,
    ) -> None:
        super().__init__()
        self.method_names = method_names
        self.method_descriptions = method_descriptions
        self.output_format_time = 0
        self.json_format_time = 0
        self.update_descriptions_time = 0
        self.retry_times = retry_times

    def _retry_on_error(
        self,
        error_str: str,
        fix_str: str,
        network: "RunnableNetwork",
        node: "RunnableNode",
    ):
        """
        we experienced an error, we will add the node again to retry after inserting
        a Human feedback node with the error and guidance on the fix
        """
        fix_error_prompt = (
            f"""We experienced an error while trying to generate method descriptions.\n\n{error_str}\n\n{fix_str}"""
        )

        # Create a new HumanNode with the error message and a docAgent to retry
        (node >> RunnableHumanNode(fix_error_prompt) >> type(node)(num_methods=len(self.method_names)))

        # we are logging that we inserted new nodes to fix the issue, maybe we only retry certain number of times
        logger.debug(f"Inserted nodes to fix errors")

    async def on_post_invoke_async(self, network: "RunnableNetwork", node: "RunnableNode"):
        """
        Called after a node's invocation is completed.
        parse the output and update the method descriptions.

        Args:
            network (RunnableNetwork): The network in which the node is run.
            node (RunnableNode): The node that has been invoked.
        """

        # Ensure the node has output to process
        if not node.outputs:
            return

        content = node.outputs.content
        if not content:
            return

        if isinstance(node, RunnableHumanNode):
            return

        error_msg = f"Retry time of generating method descriptions for {self.method_names} exceeds the RETRY_TIMES: {self.retry_times}."

        try:
            method_descriptions = json.loads(content)
        except Exception as e:
            logger.warning(f'Error {e} happpens. Generated method descriptions "{content}" is not in json format')
            if self.json_format_time > self.retry_times:
                logger.error(error_msg)
            else:
                logger.warning("We will retry.")
                self.json_format_time += 1
                error = f"""
                We got the following error: {e}
                The output is not a valid json payload.
                """
                fix = "Please regenerate the result as ONLY a raw JSON payload and ensure it is properly enclosed by '{' and '}'. Do not include any other text."
                self._retry_on_error(error, fix, network, node)
            return

        for method_name, description in method_descriptions.items():
            if method_name not in self.method_names:
                logger.warning(
                    f"The output json payload has {method_name} which is not in the list of {self.method_names}."
                )
                continue
            self.method_descriptions[method_name] = description
            self.method_names.remove(method_name)

        if len(self.method_names) > 0:
            logger.warning(
                f"The output json payload is missing the method descriptions for {self.method_names} in {method_descriptions}."
            )
            if self.update_descriptions_time > self.retry_times:
                logger.error(error_msg)
            else:
                logger.warning("We will retry.")
                self.update_descriptions_time += 1
                error = f"We've encountered errors.\nIn the output json payload, the descriptions for method names {self.method_names} are missing."
                fix = "Please generate ONLY the missing method descriptions in a json payload with nothing else."
                self._retry_on_error(error, fix, network, node)
            return


class MethodDescriptionGenerator:
    def __init__(
        self,
        model: str = "nvidia-llama-3.3-nemotron-super-49b-v1",
        concurrency: int = 20,
        batch_size: int = 10,
    ):
        self.model = model
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(concurrency)
        register_model()

    def __del__(self):
        unregister_model()

    async def generate(self, method_code_snippets: Dict[str, str]):
        """Generate method descriptions for a single example file."""

        async def process_batch(batch_code_snippets):
            """Process a single batch of method code snippets."""
            method_descriptions = {}

            async with self.semaphore:
                logger.debug(f"Generating method descriptions for {batch_code_snippets.keys()}")
                num_methods = len(batch_code_snippets)
                with RunnableNetwork(chat_model_name=self.model) as network:
                    prompt = f"We want to generate descriptions for the following {num_methods} methods:\n\n"
                    for method_name, source_code in batch_code_snippets.items():
                        prompt += (
                            f'For method "{method_name}", the output methodName key is "{method_name}".Source code:\n'
                        )
                        prompt += f"{source_code}\n\n"
                    network.add_modifier(
                        MethodDescriptionModifier(set(batch_code_snippets.keys()), method_descriptions)
                    )
                    RunnableHumanNode(prompt)
                    MethodDescriptionAgent(num_methods=num_methods)

                result = await network.ainvoke()

            return method_descriptions

        # Process all batches concurrently, limited by semaphore
        tasks = [process_batch(batch) for batch in self._chunked(method_code_snippets, self.batch_size)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        method_descriptions = {}

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception {result} happens.")
                continue
            method_descriptions.update(result)

        return method_descriptions

    def _chunked(self, d, n):
        dl = list(d.items())
        start = 0
        for _ in range(len(dl) // n):
            yield dict(dl[start : start + n])
            start += n
        if start < len(dl):
            yield dict(dl[start:])
