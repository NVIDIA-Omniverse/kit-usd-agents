## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import glob
import sys

import libcst as cst
from libcst.metadata import MetadataWrapper, ParentNodeProvider

# Get the pattern from the command line arguments or default to 'MF*.py'
pattern = sys.argv[1] if len(sys.argv) > 1 else "MF*.py"
file_list = glob.glob(pattern)

# Mapping of method names to function names
method_to_function = {
    "AddTranslateOp": "add_translate_op",
    "AddRotateXYZOp": "add_rotate_xyz_op",
    "AddScaleOp": "add_scale_op",
    "AddOrientOp": "add_orient_op",
}


class MethodToFunctionTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def leave_Call(self, original_node, updated_node):
        # Check if the function being called is an attribute access
        if isinstance(updated_node.func, cst.Attribute):
            attr = updated_node.func
            method_name = attr.attr.value

            if method_name in method_to_function:
                function_name = method_to_function[method_name]
                obj = attr.value

                # Create new function call
                new_args = [cst.Arg(value=obj)] + list(updated_node.args)
                new_call = updated_node.with_changes(func=cst.Name(value=function_name), args=new_args)
                return new_call

        return updated_node


# Process each file matching the pattern
for filename in file_list:
    # Read the source code from the file
    with open(filename, "r") as f:
        source_code = f.read()

    # Parse the code
    module = cst.parse_module(source_code)

    # Wrap the module with metadata
    wrapper = MetadataWrapper(module)

    # Apply the transformer
    transformer = MethodToFunctionTransformer()
    modified_module = wrapper.visit(transformer)

    # Write the modified code back to the same file (overwrite)
    with open(filename, "w") as f:
        f.write(modified_module.code)
