# Copyright (c) 2018-2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from io import StringIO
from langchain_core.tools import ArgsSchema, BaseTool
from pydantic import BaseModel, Field, SkipValidation
from typing import List, Optional, Annotated
import sys
import types
import inspect
import ast
import asyncio


CODE_EXECUTION_NONE_MESSAGE = "Script is successfully executed with no errors. " "Nothing is printed. " "The result is unknown."


def disabled_function(*args, **kwargs):
    raise RuntimeError("This function is disabled for security reasons")


def disable_items(items):
    originals = {}
    for item in items:
        if "." in item:
            # Handle specific functions or methods
            module_name, func_name = item.rsplit(".", 1)
            module = sys.modules.get(module_name)
            if module:
                original_func = getattr(module, func_name, None)
                if original_func:
                    # Store the original function
                    originals[item] = original_func
                    setattr(module, func_name, disabled_function)
        else:
            # Handle entire modules
            originals[item] = sys.modules.get(item)
            sys.modules[item] = None
    return originals


def restore_items(originals):
    for item, original in originals.items():
        if "." in item:
            module_name, func_name = item.rsplit(".", 1)
            module = sys.modules.get(module_name)
            if module:
                setattr(module, func_name, original)
        else:
            if original is not None:
                sys.modules[item] = original
            else:
                del sys.modules[item]


class ExecutionContext:
    def __init__(self, hide_items=None):
        self._hide_items = hide_items
        self._captured_stdout = StringIO()
        self._output = None
        self._eval_lineno = None
        self._eval_result = None
        # Use a dictionary to store global variables, including imported modules, these variables must be hold
        # after exec() is called because we might still need to access them, such as UI models and delegates
        self._global_vars = {}

    def __enter__(self):
        if self._hide_items:
            self._originals = disable_items(self._hide_items)
        else:
            self._originals = {}

        # Prepare a string buffer to capture print statements
        self._old_stdout = sys.stdout
        sys.stdout = self._captured_stdout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the original stdout
        sys.stdout = self._old_stdout

        if exc_type:
            import traceback
            tb = exc_tb.tb_next.tb_next
            if self._eval_lineno is not None:
                # Fix traceback line number as eval is done separately
                if tb.tb_frame.f_code.co_filename == "<string>":
                    tb = types.TracebackType(tb.tb_next, tb.tb_frame, tb.tb_lasti, self._eval_lineno)
            modified_exc_tuple = exc_type, exc_val, tb
            traceback_list = traceback.format_exception(*modified_exc_tuple)
            exec_error = ""
            for tb_line in traceback_list:
                exec_error += tb_line
            self._output = f"Error: {exec_error}"
        else:
            output = self._captured_stdout.getvalue()
            # Combine the printed output and the result if the last line is not indented
            if output:
                self._output = output.strip() + ("\n" + str(self._eval_result).strip() if self._eval_result is not None else "")
            else:
                self._output = str(self._eval_result).strip() if self._eval_result is not None else None

        # Close the StringIO object to release the buffer
        self._captured_stdout.close()
        # Restore the original modules and functions
        restore_items(self._originals)

        return True

    def execute(self, code: str):
        # Split the code by lines to inspect them
        lines = code.strip().split("\n")
        last_line = lines[-1]

        # Check if the last line is indented
        if (
            not last_line
            or not last_line.strip()
            or last_line.startswith(" ")
            or last_line.startswith("\t")
            or last_line.strip().startswith("#")
            or last_line.strip().startswith(")")
            or last_line.strip().startswith("}")
            or last_line.strip().startswith("]")
            or last_line.strip().startswith("from")
            or last_line.strip().startswith("import")
            or "=" in last_line
        ):
            # If the last line is indented, execute everything including the last line
            exec(code, self._global_vars)
        else:
            # Execute all but the last line, then evaluate the last line separately
            exec("\n".join(lines[:-1]), self._global_vars)
            self._eval_lineno = len(lines)
            self._eval_result = eval(last_line, self._global_vars)

    async def async_execute(self, code: str):
        # Split the code by lines to inspect them
        lines = code.strip().split("\n")
        last_line = lines[-1] if lines else ""
        
        # Check if we need to evaluate the last line separately
        should_eval_last = (
            last_line
            and last_line.strip()
            and not last_line.startswith(" ")
            and not last_line.startswith("\t")
            and not last_line.strip().startswith("#")
            and not last_line.strip().startswith(")")
            and not last_line.strip().startswith("}")
            and not last_line.strip().startswith("]")
            and not last_line.strip().startswith("from")
            and not last_line.strip().startswith("import")
            and "=" not in last_line
        )
        
        # Check if code contains await statements
        has_await = "await " in code
        
        if has_await:
            try:
                if should_eval_last and len(lines) > 1:
                    # Execute all but the last line, then evaluate the last line
                    exec_code = "\n".join(lines[:-1])
                    if exec_code.strip():
                        compiled_exec = compile(exec_code, '<string>', 'exec', 
                                              flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                        coro = eval(compiled_exec, self._global_vars)
                        if asyncio.iscoroutine(coro):
                            await coro
                    
                    # Evaluate the last line
                    self._eval_lineno = len(lines)
                    compiled_eval = compile(last_line, '<string>', 'eval', 
                                          flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    result = eval(compiled_eval, self._global_vars)
                    if asyncio.iscoroutine(result):
                        self._eval_result = await result
                    else:
                        self._eval_result = result
                else:
                    # Execute everything
                    compiled_code = compile(code, '<string>', 'exec', 
                                          flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
                    coro = eval(compiled_code, self._global_vars)
                    if asyncio.iscoroutine(coro):
                        await coro
            except SyntaxError:
                # If async compilation fails, try regular execution
                self.execute(code)
        else:
            # For non-async code, use the regular execute method
            self.execute(code)

    @property
    def output(self):
        return self._output


def execute_python_code(code: str, hide_items=None):
    """
    Executes a given Python code snippet, capturing any printed outputs.
    Temporarily disables specified modules or functions/methods during execution for safety.
    If the last line is not part of an indentation block (suggesting it's standalone),
    it evaluates that independent of any print outputs.

    Args:
        code (str): The Python code snippet to be executed.
        hide_items (Optional[List[str]]): List of module names or functions/methods to temporarily hide during execution.
        wait_fn (Optional[types.FunctionType]): Function to wait for the code execution is complete.
    Returns:
        str: A string containing any printed output and/or the result of the
             last line of code if it's not indented. If an exception occurs,
             it returns an error message.
    """
    context = ExecutionContext(hide_items=hide_items)
    try:
        with context:
            context.execute(code)
    except Exception:
        pass
    return context.output


async def aexecute_python_code(code: str, hide_items=None, wait_fn=None):
    context = ExecutionContext(hide_items=hide_items)
    try:
        with context:
            await context.async_execute(code)
            if wait_fn:
                if inspect.iscoroutinefunction(wait_fn):
                    await wait_fn()
                else:
                    wait_fn()
    except Exception:
        pass
    return context.output


class CodeInterpreterToolInput(BaseModel):
    code: str = Field(description="the code to execute")


class CodeInterpreterTool(BaseTool):
    name: str = "CodeInterpreter"
    description: str = (
        "Executes Python code snippets. "
        "It has Omniverse environment with USD and omni.ui. "
        "Use multiple different tools to gather all the necessary information before using CodeInterpreter. "
        "Always call the callbacks as a test in the code to make sure there are no errors and callbacks can be executed. "
    )
    args_schema = CodeInterpreterToolInput
    ask_human_input: bool = False
    hide_items: Optional[List[str]] = None
    wait_fn: Optional[types.FunctionType] = None

    def _run(self, code: str) -> str:
        """
        Asynchronously interprets and executes a Python code snippet.

        Args:
            code (str): The code snippet to be executed.

        Returns:
            str: The output of the executed code or an error message.
        """
        # Logic to execute code based on the specified programming language.
        # This is a placeholder for the actual code execution logic.
        # For example, if the language is Python, it might use an eval-like function.
        result = execute_python_code(code, hide_items=self.hide_items)

        if result is None:
            return CODE_EXECUTION_NONE_MESSAGE

        return str(result)

    async def _arun(self, code: str) -> str:
        result = await aexecute_python_code(code, hide_items=self.hide_items, wait_fn=self.wait_fn)

        if result is None:
            return CODE_EXECUTION_NONE_MESSAGE

        return str(result)