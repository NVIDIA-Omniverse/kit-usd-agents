## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Union
from typing import get_origin, get_args
import ast
import inspect
import types


def extract_function_signatures(filename: str) -> str:
    with open(filename, "r") as file:
        code = file.read()

    # Parse the code into an AST
    parsed_code = ast.parse(code)

    # Extract the function definitions
    functions = [node for node in parsed_code.body if isinstance(node, ast.FunctionDef)]

    # Generate the desired output
    output = ""
    for func in functions:
        func_name = func.name
        args = [arg.arg + ": " + ast.unparse(arg.annotation) for arg in func.args.args]
        return_annotation = ast.unparse(func.returns) if func.returns else "None"
        docstring = ast.get_docstring(func)

        output += f"def {func_name}({', '.join(args)}) -> {return_annotation}:\n"
        if docstring:
            output += f'    """{docstring}"""\n'
        output += "    ...\n\n"

    return output.strip()


def format_type_annotation(annotation) -> str:
    """Format a type annotation into a clean string representation.

    Args:
        annotation: A type annotation object

    Returns:
        str: A clean string representation of the type
    """
    if annotation == inspect.Parameter.empty:
        return "Any"

    # Handle Union types
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        # Format each argument
        formatted_args = [format_type_annotation(arg) for arg in args]
        # Special case for Union - use | for Python 3.10+ syntax
        if origin == Union:
            return " | ".join(formatted_args)
        # For other types (List, Dict, etc)
        return f"{origin.__name__}[{', '.join(formatted_args)}]"

    # Handle simple types
    if hasattr(annotation, "__name__"):
        # Use __module__ and __qualname__ to get the full type name.
        module = getattr(annotation, "__module__", "")
        qualname = getattr(annotation, "__qualname__", annotation.__name__)
        if module and module != "builtins":
            full_name = f"{module}.{qualname}"
        else:
            full_name = qualname

        if full_name.startswith("typing."):
            full_name = full_name[len("typing.") :]

        if full_name.startswith("pxr."):
            full_name = full_name[len("pxr.") :]

        return full_name

    # Fallback for any other case
    return str(annotation)


def extract_module_functions(module) -> str:
    """Extract function signatures from a module object.

    Args:
        module: A Python module object to extract functions from.

    Returns:
        str: A string containing all function signatures with their docstrings,
             but without function bodies.

    Example:
        >>> import usdcode
        >>> signatures = extract_module_functions(usdcode)
        >>> print(signatures)
        def function1(arg1: str, arg2: int) -> None:
            '''Function docstring'''
            ...
    """
    output = ""

    # Get all module attributes
    for name, obj in inspect.getmembers(module):
        # Skip if it's not a function or is a builtin/special method
        if not isinstance(obj, types.FunctionType) or name.startswith("_"):
            continue

        # Get function signature
        sig = inspect.signature(obj)

        # Format parameters with their type annotations
        params = []
        for param_name, param in sig.parameters.items():
            annotation_str = format_type_annotation(param.annotation)
            params.append(f"{param_name}: {annotation_str}")

        # Get return type annotation
        return_annotation_str = format_type_annotation(sig.return_annotation)

        # Format function definition
        output += f"def {name}({', '.join(params)}) -> {return_annotation_str}:\n"

        # Add docstring if it exists
        if obj.__doc__:
            output += f'    """{obj.__doc__}"""\n'

        output += "\n"

    return output.strip()
