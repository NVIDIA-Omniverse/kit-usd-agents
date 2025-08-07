## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from .code_patcher_modifier import CodePatcherModifier
from lc_agent.code_atlas import USDAtlasTool
import ast
import re
import typing


def fix_pxr_import(code):
    """
    Fixes the `import pxr` and replaces it to `from pxr import` in the given
    code snippet.
    """
    # Find all the occurrences of "pxr." followed by a word
    matches = re.findall(r"\bpxr\.(\w+)", code)

    # Get the unique module names
    modules = sorted(set(matches))

    # Create the import statements
    import_statements = [f"from pxr import {module}" for module in modules]
    if import_statements:
        import_statements_str = "\n".join(import_statements) + "\n"
    else:
        import_statements_str = ""

    # Replace "import pxr" with an empty string to remove it
    code = code.replace("import pxr\n", "")

    # Replace "pxr." with an empty string in the code
    fixed_code = re.sub(r"\bpxr\.", "", code)

    # Combine the import statements and the fixed code if pxr is used
    if modules:
        result = import_statements_str + fixed_code
    else:
        result = fixed_code.strip()

    return result


def collect_usd_types():
    code_atlas_tool = USDAtlasTool()

    # Update class reordering logic
    all_modules = code_atlas_tool.cache._modules

    return set([c.split(".")[-1] for c in all_modules.keys()])


def fix_typing_import(code: str) -> str:
    """Adds imports for used types."""

    usd_types = collect_usd_types()
    typing_types = [t for t in dir(typing) if not t.startswith("__")]

    usd_types_set = set(usd_types)
    typing_types_set = set(typing_types)

    class ImportCollector(ast.NodeVisitor):
        def __init__(self):
            self.usd_imports = set()
            self.typing_imports = set()
            self.existing_usd_imports = set()
            self.existing_typing_imports = set()

        def visit_Name(self, node):
            if node.id in usd_types_set:
                self.usd_imports.add(node.id)
            elif node.id in typing_types_set:
                self.typing_imports.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Attribute):
                self.visit_Attribute(node.value)
            elif isinstance(node.value, ast.Name) and node.value.id in usd_types_set:
                if f"{node.value.id}.{node.attr}" in usd_types_set:
                    self.usd_imports.add(f"{node.value.id}.{node.attr}")
                else:
                    self.usd_imports.add(node.value.id)
            elif isinstance(node.value, ast.Name) and node.value.id in typing_types_set:
                self.typing_imports.add(node.value.id)
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            if node.module == "pxr":
                for alias in node.names:
                    self.existing_usd_imports.add(alias.name)
            elif node.module == "typing":
                for alias in node.names:
                    self.existing_typing_imports.add(alias.name)
            self.generic_visit(node)

    tree = ast.parse(code)
    collector = ImportCollector()
    collector.visit(tree)

    # Filter out already imported types
    usd_imports = sorted(collector.usd_imports - collector.existing_usd_imports)
    typing_imports = sorted(collector.typing_imports - collector.existing_typing_imports)

    import_lines = []
    for usd_type in usd_imports:
        import_lines.append(f"from pxr import {usd_type.split('.')[-1]}")

    for typing_type in typing_imports:
        import_lines.append(f"from typing import {typing_type}")

    if import_lines:
        import_lines.append("")  # Add a blank line to separate imports from code

    return "\n".join(import_lines) + code


class USDCodeGenPatcherModifier(CodePatcherModifier):
    def _patch_code(self, code):
        """Patches the given code snippet."""
        code = super()._patch_code(code)
        code = fix_pxr_import(code)

        # this can fail on wrongly formatted code
        try:
            code = fix_typing_import(code)
        except Exception as e:
            print("Failed to fix typing imports:", e)

        return code
