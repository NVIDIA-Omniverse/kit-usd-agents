## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import ast
from typing import List, Optional

import omni.kit.commands
import omni.kit.undo
import omni.usd
from lc_agent.code_atlas import CodeInterpreterTool
from lc_agent_usd.modifiers.code_interpreter_modifier import CodeInterpreterModifier
from lc_agent_usd.modifiers.usd_code_gen_patcher_modifier import fix_pxr_import, fix_typing_import

from .double_run_commands import DoubleRunUSDCodeGenCommand
from .double_run_edit_target_manager import EditTargetManager
from .double_run_utils import format_numbers_in_string

INMEMORY_STAGE_CODE = """
import usdcode
try:
    import uicode
except ImportError:
    pass
from pxr import Sdf, Usd, UsdGeom
from typing import List
import omni.usd


# Start with the real selection
__MOCK_SELECTION = [str(s) for s in omni.usd.get_context().get_selection().get_selected_prim_paths()]


def __get_selection() -> List[str]:
    return __MOCK_SELECTION[:]


def __set_selection(selected_prim_paths: List[str]):
    __MOCK_SELECTION[:] = [str(s) for s in selected_prim_paths]


def __create_inmemory_stage_with_sublayers_and_session_layer():
    original_stage = omni.usd.get_context().get_stage()

    # Create a new in-memory stage
    new_stage = Usd.Stage.CreateInMemory()

    # Get the root layer of the original stage
    original_root_layer = original_stage.GetRootLayer()

    # Add the original root layer to the new root layer
    new_root_layer = new_stage.GetRootLayer()
    new_root_layer.subLayerPaths.append(original_root_layer.identifier)

    # Copy other properties from the original stage
    new_stage.SetStartTimeCode(original_stage.GetStartTimeCode())
    new_stage.SetEndTimeCode(original_stage.GetEndTimeCode())

    # Copy the up axis from the original stage
    up_axis_token = UsdGeom.GetStageUpAxis(original_stage)
    if up_axis_token:
        UsdGeom.SetStageUpAxis(new_stage, up_axis_token)

    # Copy the session layer if it exists
    original_session_layer = original_stage.GetSessionLayer()
    if original_session_layer:
        # Create a new in-memory session layer
        new_session_layer = Sdf.Layer.CreateAnonymous()

        # Copy the contents of the original session layer to the new session layer
        new_session_layer.TransferContent(original_session_layer)

        # Set the new session layer to the new stage
        new_stage.GetSessionLayer().TransferContent(new_session_layer)

    return new_stage

stage = __create_inmemory_stage_with_sublayers_and_session_layer()

"""

# It's a separate function because LLM tends to break down
# omni.usd.get_context().get_selection().get_selected_prim_paths()
# into multiple lines. And it complicates replacement of this method for mocking
# when running the snippet the first time.
SELECTION_CODE = """
import usdcode
try:
    import uicode
except ImportError:
    pass
from typing import List
import omni.usd

stage = omni.usd.get_context().get_stage()

"""


def format_python_code(input_code):
    import keyword

    # Split the input code by semicolons to handle multiple statements on a single line
    statements = input_code.split("; ")
    formatted_statements = []
    for statement in statements:
        try:
            # Parse each statement into an AST
            tree = ast.parse(statement)
            # Convert the AST back to a formatted string
            formatted_statement = ast.unparse(tree).strip()
            formatted_statements.append(formatted_statement)
        except SyntaxError as e:
            print(f"Syntax error in statement: {statement}\n{e}")

    # Join the formatted statements with newlines
    formatted_code = "\n".join(formatted_statements)

    # Add proper indentation
    lines = formatted_code.split("\n")
    indented_code = []
    indent_level = 0
    keywords = set(keyword.kwlist)
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.split()[0] in keywords:
            indent_level = 0
        indented_code.append("    " * indent_level + stripped_line)
        if stripped_line.endswith(":"):
            indent_level += 1

    return "\n".join(indented_code)


class DoubleRunUSDCodeGenInterpreterModifier(CodeInterpreterModifier):
    def __init__(
        self,
        show_stdout=True,
        error_message: Optional[str] = None,
        success_message: Optional[str] = None,
        hide_items: Optional[List[str]] = None,
        undo_stack=True,
        first_run=True,
        second_run=True,
        **kwargs,
    ):
        super().__init__(
            show_stdout=show_stdout,
            error_message=error_message,
            success_message=success_message,
            hide_items=hide_items,
            **kwargs,
        )
        self._first_run = first_run
        self._second_run = second_run
        self._undo_stack = undo_stack

    def _run(self, code):
        # Run the code twice to ensure that the code is deterministic.
        code_interpreter_tool = CodeInterpreterTool(hide_items=self._hide_items)

        if code.startswith("python"):
            code = code[len("python") :].strip()
        if code.startswith("\npython"):
            code = code[len("\npython") :].strip()

        # fix the issue when the output is like this:
        # """import omni.ui as ui; import uicode; def on_slider_value_changed(value):..."""
        if len(code.split("\n")) == 1 and code.count(";") > 1:
            code = format_python_code(code)

        code = fix_pxr_import(code)
        code = fix_typing_import(code)

        empty_stage_code = INMEMORY_STAGE_CODE
        # Just in case it really wants to do something with the stage
        empty_stage_code += (
            code.replace(
                "omni.usd.get_context().get_stage()", "__create_inmemory_stage_with_sublayers_and_session_layer()"
            )
            .replace("usdcode.get_selection()", "__get_selection()")
            .replace("usdcode.set_selection(", "__set_selection(")
        )

        # First run to check for errors
        if self._first_run:
            execution_result = code_interpreter_tool._run(empty_stage_code)
            if "error:" in execution_result.lower():
                return execution_result

            print("First run of the code is successful.", execution_result)

            if not self._second_run:
                return execution_result

        if self._undo_stack:
            edit_target_manager = EditTargetManager()
            edit_target_manager.set_edit_target()

        try:
            # Second run with the actual stage
            result = super()._run(SELECTION_CODE + code)
        finally:
            if self._undo_stack:
                print("Restoring the original edit target.")
                edit_target_manager.restore_edit_target()

                if edit_target_manager._new_layer and not edit_target_manager._new_layer.empty:
                    # Mark the layer for merging
                    layer = edit_target_manager._new_layer
                    custom_layer_data = layer.customLayerData
                    custom_layer_data["DoubleRunUSDCodeGenCommand"] = "yes"
                    layer.customLayerData = custom_layer_data

                    # Execute the command to handle merging on save
                    omni.kit.commands.execute("DoubleRunUSDCodeGenCommand", layer=layer)

        result = format_numbers_in_string(result)

        if "error:" in result.lower():
            return result

        lines = result.splitlines()
        max_lines = 500
        half_lines = max_lines // 2 - 3
        lines_count = len(lines)
        if lines_count > max_lines:
            begin = lines[:half_lines]
            end = lines[-half_lines:]
            begin = "\n".join(begin)
            end = "\n".join(end)
            result = begin + "\n\n... (skipped) ...\n\n" + end
            return f"error: Code printed too many lines ({lines_count}). Please reduce the output:\n{result}"

        return result
