## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import types
import inspect
import importlib.util
import os


def add_function(function_code: str):
    """Add a function to the usdcode module.

    Args:
        function_code (str): The function code to add as a string.

    Example:
        >>> import usdcode
        >>> usdcode.setup.add_function('''
        ... def print_hello():
        ...     \"\"\"prints hello\"\"\"
        ...     print("Hello")
        ... ''')
        >>> usdcode.print_hello()
        Hello
    """
    # Create a new module namespace
    module_namespace = {}

    # Execute the function code in the new namespace
    exec(function_code, module_namespace)

    # Get the function object from the namespace
    # We expect only one function to be defined
    function_name = None
    function_obj = None
    for name, obj in module_namespace.items():
        if isinstance(obj, types.FunctionType):
            function_name = name
            function_obj = obj
            break

    if not function_name or not function_obj:
        raise ValueError("No function found in the provided code")

    # Add the function to the usdcode module
    import usdcode

    setattr(usdcode, function_name, function_obj)


def add_functions_from_file(file_path: str):
    """Add all functions from a Python file to the usdcode module.

    Args:
        file_path (str): Path to the Python file containing functions to add.

    Example:
        >>> import usdcode
        >>> usdcode.setup.add_functions_from_file('my_functions.py')
    """
    # Get absolute path
    abs_path = os.path.abspath(file_path)

    # Create a unique module name based on the file name
    module_name = f"usdcode_dynamic_{os.path.splitext(os.path.basename(file_path))[0]}"

    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get all functions from the module
    import usdcode

    for name, obj in inspect.getmembers(module):
        if isinstance(obj, types.FunctionType) and not name.startswith("_"):
            # Add each function to usdcode
            setattr(usdcode, name, obj)
