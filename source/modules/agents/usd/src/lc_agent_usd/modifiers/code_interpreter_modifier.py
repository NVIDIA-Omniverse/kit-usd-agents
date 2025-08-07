## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from lc_agent import NetworkModifier
from lc_agent import RunnableAINode
from lc_agent import RunnableHumanNode
from lc_agent.code_atlas import CodeInterpreterTool
from typing import Optional
from typing import List
import re

# pyright: reportUnusedExpression=false


def remove_brackets(s):
    # Pattern to remove double brackets
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        return match.group(1).strip()
    return s.strip()


class CodeInterpreterModifier(NetworkModifier):

    def __init__(
        self,
        show_stdout=True,
        error_message: Optional[str] = None,
        success_message: Optional[str] = None,
        hide_items: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._show_stdout = show_stdout
        self._error_message = error_message
        self._success_message = success_message
        self._hide_items = hide_items

    def _fix_before_run(self, code):
        """Fixes the code before running it."""
        return code

    def _run(self, code):
        """Run the code."""
        code_interpreter_tool = CodeInterpreterTool(hide_items=self._hide_items)
        execution_result = code_interpreter_tool._run(code)
        return execution_result

    async def _run_async(self, code):
        """Run the code asynchronously."""
        code_interpreter_tool = CodeInterpreterTool(hide_items=self._hide_items)
        execution_result = await code_interpreter_tool._arun(code)
        return execution_result

    async def on_post_invoke_async(self, network: "RunnableNetwork", node: "RunnableNode"):
        if (
            # If it's a final result
            node.invoked
            and isinstance(node, RunnableAINode)
            and not network.get_children(node)
            and "interpreter_code" not in node.metadata
            and "stop_reason" not in node.metadata
        ):
            # Remove the code block if exists
            code_snippet_original = node.outputs.content
            code_snippet_no_brackets = remove_brackets(code_snippet_original)
            code_snippet_run = self._fix_before_run(code_snippet_no_brackets)

            # Execute the code
            execution_result = await self._run_async(code_snippet_run)

            if "error:" in execution_result.lower():
                # Error occurred
                error_message_metadata = {
                    "interpreter_code": code_snippet_no_brackets,
                    "interpreter_error": execution_result,
                    "interpreter_result": execution_result,
                }

                # Here we need some mechanism to integrate RAG of common issues
                custom_message = ""
                if "Path must be an absolute path" in execution_result:
                    custom_message = "This error is generally because the path start with a number, it should aways start with a letter"
                error_message_node = RunnableHumanNode(
                    human_message=(
                        (self._error_message or "There was an error while executing the code.\n")
                        + f"{execution_result}\n\n"
                        f"{custom_message}\n\n"
                        "Please fix the code."
                    ),
                    metadata=error_message_metadata,
                )
                node >> error_message_node
            else:
                # No error
                success_metadata = {
                    "interpreter_code": code_snippet_no_brackets,
                    "interpreter_ok": True,
                    "interpreter_result": execution_result,
                }

                if not self._show_stdout:
                    message = self._success_message or "The code executed successfully!"
                else:
                    message = (
                        self._success_message or "The code executed successfully with the following output.\n"
                    ) + f"```\n{execution_result}\n```\n"
                success_node = RunnableAINode(
                    ai_message=message,
                    metadata=success_metadata,
                )
                node >> success_node
