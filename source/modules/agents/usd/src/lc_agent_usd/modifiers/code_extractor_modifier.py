## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from ..nodes.usd_code_gen_node import USDCodeGenNode
from lc_agent import NetworkModifier
from lc_agent import RunnableAINode
from lc_agent import RunnableHumanNode
from lc_agent import RunnableNetwork
from lc_agent import RunnableNode
from lc_agent import RunnableSystemAppend
from typing import Optional
import re
import time

# pyright: reportUnusedExpression=false


CODE_VERIFICATION_PROMPT = """
You are an AI assistant tasked with determining if a given piece of Python code in an answer needs to be verified. Based on the question and answer provided, you should decide if the code should be executed to check its correctness or output. You should only answer "yes" if the code needs to be verified or "no" if it does not. Use the following guidelines to make your decision:

# When the code does **not** need verification:

1. The code is written as pseudocode.
2. The code is intentionally not executable (e.g., it contains placeholders or is meant to illustrate a concept).
3. The code is part of an explanation about how a specific part of the code works or how a concept is applied.
4. The question explicitly states that the code is illustrative or conceptual.
5. The answer includes annotations or comments that explain the steps without requiring execution.
6. The code is a snippet demonstrating syntax or usage without a requirement to produce a specific output.
7. The question and answer focus on theoretical aspects or design patterns rather than actual execution.

# When the code **does** need verification:

1. The question asks for a complete and executable piece of code (e.g., "Write a Python function that does X").
2. The answer provides code that claims to produce a specific output or result.
3. The question involves solving a problem or implementing a feature, and the answer includes the full implementation.
4. The code is intended to perform a specific task or calculation, and its correctness can be verified by execution.
5. The question and answer are part of a coding exercise or assessment where correct execution is required.

Based on these guidelines, determine if the Python code in the answer needs to be verified. Respond with "yes" if it does, or "no" if it does not.
"""


class VerifyNode(RunnableNode):
    system_message: Optional[str] = CODE_VERIFICATION_PROMPT

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.system_message:
            self.inputs.append(RunnableSystemAppend(system_message=self.system_message))


def extract_code_snippet(text: str) -> tuple:
    """
    Extracts the longest code snippet and its language tag from a text block that's enclosed within triple backticks.

    Parameters:
    - text (str): The text containing code snippets.

    Returns:
    - tuple: A tuple containing the extracted longest code snippet and its language tag, or (None, None) if not found.
    """
    if text and text.startswith("<|python_tag|>"):
        return text[len("<|python_tag|>") :], "python"

    # Define pattern for code block enclosed within triple backticks with an optional language tag
    pattern = r"```(\w[\w+-]*)?\s*([^`]+)```"

    # Search for all non-greedy matches of the pattern
    matches = re.findall(pattern, text, re.DOTALL)

    # Initialize variables to store the longest code snippet and its language tag
    longest_code = None
    language_tag = None

    # Iterate through all matches to find the longest code snippet
    for lang_tag, match in matches:
        code_snippet = match.strip()
        lang_tag = lang_tag.strip() if lang_tag else None

        # Check if the current code snippet is longer than the previously stored one
        if longest_code is None or len(code_snippet) > len(longest_code):
            longest_code = code_snippet
            language_tag = lang_tag

    # If no code snippet found within backticks, use the secondary logic
    if longest_code is None:
        code_snippet = text.split("```")[-1].strip()

        if code_snippet.startswith("python"):
            code_snippet = code_snippet[len("python") :].strip()
            language_tag = "python"
        elif code_snippet.startswith("py"):
            code_snippet = code_snippet[len("py") :].strip()
            language_tag = "py"

        if code_snippet.startswith("import "):
            longest_code = code_snippet
        elif code_snippet.startswith("from "):
            longest_code = code_snippet

    return longest_code, language_tag


async def verify_to_run(parent_outputs: str, node_outputs: str, network) -> bool:
    prompt = (
        f"Question:\n{parent_outputs}\n\nAnswer:\n{node_outputs}\n\n"
        "Determine if the Python code in the answer needs to be verified. "
        'Respond with "yes" if it does, or "no" if it does not.'
    )

    # start_time = time.time()

    with RunnableNetwork(chat_model_name=network.chat_model_name) as network:
        RunnableHumanNode(human_message=prompt)
        VerifyNode()

    result = await network.ainvoke()
    result = result.content.strip().lower()

    # print(f"verify_to_run took {time.time() - start_time} seconds, result: {result}")

    if result == "no":
        return False

    return True


class CodeExtractorModifier(NetworkModifier):
    def __init__(self, snippet_verification=False, shippet_language_check=False):
        super().__init__()
        self._snippet_verification = snippet_verification
        self._shippet_language_check = shippet_language_check

    async def on_post_invoke_async(
        self, network: "RunnableNetwork", node: "RunnableNode"
    ):
        if (
            # If it's a final result
            node.invoked
            and type(node).__name__ == network.default_node
            and not network.get_children(node)
        ):
            parents = network.get_parents(node)
            parent = parents[-1] if parents else None
            if self._snippet_verification and parent:
                if not await verify_to_run(
                    parent.outputs.content, node.outputs.content, network
                ):
                    return

            code_snippet, language_tag = extract_code_snippet(node.outputs.content)
            if not code_snippet:
                # TODO: Critical error, no code snippet found.
                return

            if self._shippet_language_check and language_tag != "python":
                return

            # Mute this node, add a new one with output
            node.metadata["contribute_to_history"] = False
            node >> RunnableAINode(
                ai_message=f"```python\n{code_snippet}\n```",
                name=str(type(self).__name__),
                metadata={"contribute_to_ui": False},
            )
