## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import re

from pxr import Sdf


def merge_layer_content(src_layer, dst_layer):
    """
    Merges content from src_layer into dst_layer without flattening.
    This function merges all prims, properties, and specs recursively,
    preserving references and other composition arcs.
    """

    visited_paths = set()

    def merge_prim(src_prim_path):
        # Prevent infinite recursion by tracking visited paths
        if src_prim_path in visited_paths:
            return
        visited_paths.add(src_prim_path)

        src_prim_spec = src_layer.GetPrimAtPath(src_prim_path)
        dst_prim_spec = dst_layer.GetPrimAtPath(src_prim_path)

        # If the source prim spec does not exist, return early
        if src_prim_spec is None:
            return

        # Determine the specifier for the destination prim
        if dst_prim_spec is None:
            # If the destination prim does not exist, create it
            # If the source prim is an 'over', we create a 'def' in the destination
            dst_specifier = (
                Sdf.SpecifierDef if src_prim_spec.specifier == Sdf.SpecifierOver else src_prim_spec.specifier
            )
            parent_path = src_prim_path.GetParentPath()
            if not parent_path.isEmpty:
                # Ensure parent prims exist
                merge_prim(parent_path)
                parent_prim_spec = dst_layer.GetPrimAtPath(parent_path)
                dst_prim_spec = Sdf.PrimSpec(
                    parent_prim_spec,
                    src_prim_spec.name,
                    dst_specifier,
                    src_prim_spec.typeName,
                )
            else:
                # Root prim; parent is the layer
                dst_prim_spec = Sdf.PrimSpec(
                    dst_layer,
                    src_prim_spec.name,
                    dst_specifier,
                    src_prim_spec.typeName,
                )
        else:
            # Destination prim exists
            # Do not change the specifier to 'over' if it's already 'def'
            if src_prim_spec.specifier != Sdf.SpecifierOver:
                dst_prim_spec.specifier = src_prim_spec.specifier
            # Update the typeName if necessary
            if src_prim_spec.typeName and dst_prim_spec.typeName != src_prim_spec.typeName:
                dst_prim_spec.typeName = src_prim_spec.typeName

        # Copy fields from source prim spec to destination prim spec
        for field in src_prim_spec.ListInfoKeys():
            if field != "name" and field != "typeName" and field != "specifier":
                value = src_prim_spec.GetInfo(field)
                dst_prim_spec.SetInfo(field, value)

        # Copy properties
        for src_prop_spec in src_prim_spec.properties:
            prop_name = src_prop_spec.name
            if isinstance(src_prop_spec, Sdf.AttributeSpec):
                dst_prop_spec = dst_prim_spec.attributes.get(prop_name)
            elif isinstance(src_prop_spec, Sdf.RelationshipSpec):
                dst_prop_spec = dst_prim_spec.relationships.get(prop_name)
            else:
                continue  # Skip unknown property types

            if dst_prop_spec is None:
                # Create new property spec
                if isinstance(src_prop_spec, Sdf.AttributeSpec):
                    dst_prop_spec = Sdf.AttributeSpec(
                        dst_prim_spec,
                        prop_name,
                        src_prop_spec.typeName,
                        src_prop_spec.variability,
                    )
                elif isinstance(src_prop_spec, Sdf.RelationshipSpec):
                    dst_prop_spec = Sdf.RelationshipSpec(
                        dst_prim_spec,
                        prop_name,
                        custom=src_prop_spec.custom,
                        variability=src_prop_spec.variability,
                    )

            # Copy fields
            for field in src_prop_spec.ListInfoKeys():
                if field == "targetPaths" and isinstance(src_prop_spec, Sdf.RelationshipSpec):
                    # Handle targetPaths separately for relationships
                    dst_prop_spec.targetPathList.explicitItems = []
                    for target in src_prop_spec.targetPathList.explicitItems:
                        dst_prop_spec.targetPathList.explicitItems.append(target)
                else:
                    value = src_prop_spec.GetInfo(field)
                    dst_prop_spec.SetInfo(field, value)

        # Recurse to child prims
        for child_spec in src_prim_spec.nameChildren:
            child_name = child_spec.name
            child_path = src_prim_path.AppendChild(child_name)
            merge_prim(child_path)

    # Start merging from the root prims in the source layer
    for root_prim_spec in src_layer.rootPrims:
        merge_prim(root_prim_spec.path)


def format_numbers_in_string(s, precision=2):
    """
    Formats all numbers in the given string to the specified precision.

    Parameters:
    - s (str): The input string containing numbers.
    - precision (int): The number of decimal places to round to.

    Returns:
    - str: The string with all numbers formatted to the given precision.
    """
    # Regular expression to match numbers, including integers and floats
    number_pattern = re.compile(
        r"""
        (?<![\w.])      # Negative lookbehind to ensure the previous char is not a word character or dot
        [+-]?           # Optional sign
        (?:
            \d+\.\d+    # Numbers with a decimal point
            |
            \d+         # Integer numbers
            |
            \.\d+       # Numbers that start with a dot (e.g., .5)
        )
        (?:[eE][+-]?\d+)?  # Optional exponent
        (?![\w.])       # Negative lookahead to ensure the next char is not a word character or dot
    """,
        re.VERBOSE,
    )

    def replace_number(match):
        num_str = match.group(0)
        try:
            num = float(num_str)
            rounded_num = round(num, precision)
            # Convert to integer if there's no decimal part after rounding
            if rounded_num.is_integer():
                formatted_num = str(int(rounded_num))
            else:
                formatted_num = f"{rounded_num:.{precision}f}".rstrip("0").rstrip(".")
            return formatted_num
        except ValueError:
            # If parsing fails, return the original string
            return num_str

    # Substitute all numbers in the string using the replace_number function
    return number_pattern.sub(replace_number, s)
