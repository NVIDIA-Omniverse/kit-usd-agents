# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import ast
import re
from typing import Dict, List, Optional
from .codeatlas_module_info import (
    CodeAtlasModuleInfo,
    CodeAtlasClassInfo,
    CodeAtlasMethodInfo,
    CodeAtlasArgumentInfo,
)

def _get_attribute_text(attr):
    parts = []

    a = attr
    while isinstance(a, ast.Attribute):
        parts.append(a.attr)
        a = a.value

    if not isinstance(a, ast.Name):
        return None
    
    parts.append(a.id)

    return ".".join(reversed(parts))

class CodeAtlasCollector(ast.NodeVisitor):
    def __init__(self, module_name, source_lines):
        self.collected_modules: List[str] = []
        self.wildcarts_modules: List[str] = []
        self.equivalent_modules: List[str] = []
        self.classes: List[CodeAtlasClassInfo] = []

        self.module_name = module_name
        self.source_lines = source_lines
        self.scope_stack = []
        self.imports = {}
        self.methods: List[CodeAtlasMethodInfo] = []
        self.method_stack: List[List[str]] = [[]]

        self.scope_stack = []

    def enter_scope(self, name):
        self.scope_stack.append(name)

    def exit_scope(self):
        self.scope_stack.pop()

    def current_scope(self) -> str:
        return ".".join(self.scope_stack)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports[alias.asname or alias.name] = alias.name

    def visit_ImportFrom(self, node):
        """Visit 'from ... import ...' statements and record the module being imported."""
        level = node.level
        module = node.module if node.module else ""
        if level > 0:
            # Only handle relative imports

            modules = [name.name for name in node.names if not name.name.startswith("_")] if module == "" else [module]

            for module in modules:
                module = "." * level + module

                if len(node.names) == 1 and node.names[0].name == "*":
                    self.wildcarts_modules.append(module)
                else:
                    self.collected_modules.append(module)
        else:
            # Handle absolute imports. We only deal with wildcard imports here, since that is the equivalent modules of the current modules
            if len(node.names) == 1 and node.names[0].name == "*":
                self.equivalent_modules.append(module)

        level = "." * level
        for alias in node.names:
            if alias.name == "*":
                continue  # We'll ignore wildcard imports
            self.imports[alias.asname or alias.name] = f"{level}{module}.{alias.name}"

    def visit_ClassDef(self, node: ast.ClassDef):
        self.enter_scope(node.name)

        # Determine the full name based on the current scope
        full_name = f"{self.module_name}{'.'.join([''] + self.scope_stack)}"

        class_info = CodeAtlasClassInfo(
            name=node.name,
            full_name=full_name,
            docstring=ast.get_docstring(node) if ast.get_docstring(node) else None,
            line_number=node.lineno,
            module_name=self.module_name,
            parent_classes=[base.id for base in node.bases if isinstance(base, ast.Name)],  # Parent class names
            class_variables=[],  # This requires further processing to find class variables
            decorators=[d.id for d in node.decorator_list if isinstance(d, ast.Name)],  # Decorator names
        )
        self.method_stack.append([])

        self.generic_visit(node)

        class_info.methods = self.method_stack.pop()
        self.classes.append(class_info)

        self.exit_scope()
    
    def get_method_identifier(self, node: ast.FunctionDef):
        # Create unique name for methods sharing the same name
        identifier = node.name
        count = 1
        while identifier in self.method_stack[-1]:
            identifier = f"{node.name}_{count}"
            count += 1
        return identifier

    def visit_FunctionDef(self, node: ast.FunctionDef):
        identifier = self.get_method_identifier(node)

        # Push the current function name onto the scope stack
        self.enter_scope(identifier)

        # Determine the full name based on the current scope
        full_name = f"{self.module_name}{'.'.join([''] + self.scope_stack)}"

        # Determine the return type annotation, if present
        return_type = self.get_type_annotation_str(node.returns) if node.returns else None

        docstring = ast.get_docstring(node) or None

        # Call extract_function_body to get the dedented source code of the function body
        source_code = self.extract_function_body(node, self.source_lines)

        # Process the function body for class usages
        class_usages = self.find_class_usages_in_body(node)
        class_usages = [self.resolve_name(usage) for usage in class_usages if usage[0] in self.imports]

        decorators = []
        for d in node.decorator_list:
            text = _get_attribute_text(d)
            if text:
                decorators.append(text)

        method_info = CodeAtlasMethodInfo(
            name=node.name,
            full_name=full_name,
            return_type=return_type,
            docstring=docstring,
            line_number=node.lineno,
            module_name=self.module_name,
            is_class_method=any(isinstance(d, ast.Name) and d.id == "classmethod" for d in node.decorator_list),
            is_static_method=any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in node.decorator_list),
            is_async_method=isinstance(node, ast.AsyncFunctionDef),
            decorators=decorators,  # Decorator names
            source_code=source_code,
            class_usages=class_usages,
        )
        self.methods.append(method_info)
        self.method_stack[-1].append(identifier)

        docstring_types = self.extract_types_from_docstring(docstring)

        # Visit the arguments to collect their detailed information
        default_start_index = len(node.args.args) - len(node.args.defaults)
        for i, arg in enumerate(node.args.args):
            arg_info = self.visit(arg)
            # If type annotation is not in the AST, try to fetch it from the docstring
            if arg_info.type_annotation is None and arg_info.name in docstring_types:
                arg_info.type_annotation = docstring_types[arg_info.name]
            if i >= default_start_index:
                default = node.args.defaults[i - default_start_index]
                value = self.extract_code(self.source_lines, default.lineno, default.end_lineno, default.col_offset, default.end_col_offset, exclude_docstring=False)
                arg_info.default_value = value
            method_info.arguments.append(arg_info)
        if node.args.vararg:
            arg_info = self.visit(node.args.vararg)
            arg_info.name = "*" + arg_info.name
            arg_info.is_variadic = True
            method_info.arguments.append(arg_info)
        if node.args.kwarg:
            arg_info = self.visit(node.args.kwarg)
            arg_info.name = "**" + arg_info.name
            arg_info.is_variadic = True
            method_info.arguments.append(arg_info)

        # Visit any potential nested functions or classes
        self.generic_visit(node)

        # Pop the function name from the scope stack as we leave the function scope
        self.exit_scope()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self.visit_FunctionDef(node)

    def visit_arg(self, node: ast.arg):
        # Capture argument information
        default_value = None  # This will need further handling to obtain the correct default value string
        type_annotation = None if node.annotation is None else self.get_type_annotation_str(node.annotation)

        if self.module_name:
            parent_method = self.module_name + "." + self.current_scope()
        else:
            parent_method = self.current_scope()

        arg_info = CodeAtlasArgumentInfo(
            name=node.arg,
            type_annotation=type_annotation,
            default_value=default_value,
            # is_variadic would be determined by checking for *args or **kwargs in the AST
            parent_method=parent_method,
        )
        return arg_info

    def get_type_annotation_str(self, annotation_node):
        """
        Function to extract the string representation of the type annotation.
        """
        if isinstance(annotation_node, ast.Name):
            return annotation_node.id
        elif isinstance(annotation_node, ast.Attribute):
            # Recursively get the string for the left part of the attribute access
            value_str = self.get_type_annotation_str(annotation_node.value)
            if value_str is None:
                return None  # In case the left part of the attribute could not be resolved
            return value_str + "." + annotation_node.attr
        elif isinstance(annotation_node, ast.Subscript):
            # Recursively get the string for the base type
            base_str = self.get_type_annotation_str(annotation_node.value)
            if base_str is None:
                return None  # In case the base type could not be resolved

            # Recursively get the string for the subscript part (index)
            if isinstance(annotation_node.slice, (ast.Index, ast.Slice, ast.Tuple)):
                # ast.Index is deprecated since Python 3.9, but you might still need to support it
                index_node = (
                    annotation_node.slice.value
                    if isinstance(annotation_node.slice, ast.Index)
                    else annotation_node.slice
                )
                index_str = self.get_type_annotation_str(index_node)
                if index_str is None:
                    return None  # In case the index part could not be resolved
                return f"{base_str}{index_str}" if isinstance(annotation_node.slice, ast.Tuple) else f"{base_str}[{index_str}]"          
            # Handle more complex cases as needed
        elif isinstance(annotation_node, ast.Constant):
            # Handle string annotations (mostly for Python 3.7 and earlier)
            if annotation_node.value is None:
                return "None"
            else:
                return annotation_node.s
        elif isinstance(annotation_node, (ast.Tuple, ast.List)):
            child_strs = []
            for child in annotation_node.elts:
                child_str = self.get_type_annotation_str(child)
                if child_str is None:
                    return None
                child_strs.append(child_str)
            return "[" + ", ".join(child_strs) + "]"
        # More cases might need to be handled depending on the complexity of type annotations
        return None

    def extract_types_from_docstring(self, docstring):
        """
        Parse the docstring and extract parameter type information.
        """
        types = {}
        if not docstring:
            return types

        # Use re.findall to look for parameter lines directly in the docstring
        param_matches = re.findall(r"^\s*(\w+)\s*:\s*(.+)", docstring, flags=re.MULTILINE)

        # Process matches and populate the types dictionary
        for param_name, param_type in param_matches:
            param_name = param_name.strip()
            param_type = param_type.strip()
            types[param_name] = param_type

        return types

    def extract_function_body(self, node, source_lines):
        if ast.get_docstring(node) and len(node.body) > 1:
            # If a docstring is present, body starts from the second element of node.body
            start_line = node.body[1].lineno
            start_line_col_offset = getattr(node.body[1], "col_offset", 0)
        else:
            # Otherwise, body starts from the first element of node.body
            start_line = node.body[0].lineno
            start_line_col_offset = getattr(node.body[0], "col_offset", 0)

        # Get the end line of the function body
        end_line = node.body[-1].end_lineno
        return self.extract_code(source_lines, start_line, end_line, start_line_col_offset, exclude_docstring=True)
    
    def extract_code(self, source_lines, start_line, end_line, start_line_col_offset=0, end_line_col_offset=None, exclude_docstring=True):
        # Subtract 1 because lineno is 1-based, unlike list indexing
        body_lines = source_lines[start_line - 1 : end_line]

        # Also check for col offsets
        if start_line_col_offset > 0:
            body_lines[0] = body_lines[0][start_line_col_offset:]
        if end_line_col_offset:
            end = end_line_col_offset
            if len(body_lines) == 1 and start_line_col_offset > 0:
                end -= start_line_col_offset
            body_lines[-1] = body_lines[-1][:end]

        # Determine the minimum indentation (ignoring empty lines and start line if not extracting the full line)
        min_indentation = min((len(line) - len(line.lstrip()) for line in body_lines[1 if start_line_col_offset > 0 else 0:] if line.strip()), default=0)

        # Dedent the body lines and return the joined string
        dedented_body_lines = [line[min_indentation:] if line.strip() else line for line in body_lines]
        result = "".join(dedented_body_lines).rstrip()

        if exclude_docstring:
            if result.startswith('"""') and result.endswith('"""') and result.count('"""') == 2:
                return None

            if result.startswith("'''") and result.endswith("'''") and result.count("'''") == 2:
                return None

        return result

    def find_class_usages_in_body(self, node):
        class ClassUsageFinder(ast.NodeVisitor):
            def __init__(self, imports: Dict[str, str]):
                self.usages = []
                self.imports = imports

            def visit_Call(self, call_node):
                # Handles cases like `ui.Window()` where `ui` is an imported module or alias
                if isinstance(call_node.func, ast.Attribute) and isinstance(call_node.func.value, ast.Name):
                    module_alias = call_node.func.value.id
                    attr_name = call_node.func.attr
                    if module_alias in self.imports:
                        self.usages.append((module_alias, attr_name))
                # Handles cases where a function is called without .attribute syntax
                elif isinstance(call_node.func, ast.Name):
                    func_name = call_node.func.id
                    if func_name in self.imports:
                        # Record usages that exactly match the import aliases
                        self.usages.append((None, func_name))
                # Continue traversal to cover nested expressions and call arguments
                self.generic_visit(call_node)

        usage_finder = ClassUsageFinder(self.imports)
        usage_finder.visit(node)
        return usage_finder.usages

    def resolve_name(self, module_alias_attribute_tuple):
        module_alias, attribute = module_alias_attribute_tuple
        if module_alias in self.imports:
            # Resolve alias to full module name and append the attribute if it's there
            if attribute:
                return f"{self.imports[module_alias]}.{attribute}"
            # If there's no attribute, just return the resolved alias (module level import)
            return self.imports[module_alias]
        return None
