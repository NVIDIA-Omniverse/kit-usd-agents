# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from .codeatlas_module_info import CodeAtlasClassInfo
from .codeatlas_module_info import CodeAtlasMethodInfo
from .codeatlas_module_info import CodeAtlasModuleInfo
from typing import Dict
from typing import Optional


INDENT = "    "


def indent(text, indent_count=1):
    if indent_count == 0:
        return text
    indent_sequence = INDENT * indent_count  # Prepare the indent sequence to be added
    lines = text.split("\n")  # Split the text into lines
    # Add the indent sequence to each non-empty line and join them back into a single string
    indented_text = "\n".join(indent_sequence + line if line.strip() else "" for line in lines)
    return indented_text


class CodeAtlasLookup:
    def __init__(
        self,
        modules: Dict[str, CodeAtlasModuleInfo],
        classes: Dict[str, CodeAtlasClassInfo],
        methods: Dict[str, CodeAtlasMethodInfo],
    ):
        self._modules = modules
        self._classes = classes
        self._methods = methods

    def lookup_module(
        self,
        module_name: str,
        classes: bool = True,
        methods: bool = True,
        functions: bool = True,
        method_bodies: bool = True,
        docs: bool = True,
    ) -> Optional[str]:
        # Collect all modules that match by full name, module name, or suffix
        matching_modules = []

        # Try to find the module by its full name in the dictionary
        module_info = self._modules.get(module_name)
        if module_info:
            matching_modules.append(module_info)
        else:
            # Search for the module with a matching name attribute
            for full_name, mod in self._modules.items():
                if mod.name == module_name:
                    matching_modules.append(mod)
            # Try to find by suffix if full_name or name didn't match
            module_name_with_dot = "." + module_name
            for full_name, mod in self._modules.items():
                if mod.full_name and mod.full_name.endswith(module_name_with_dot):
                    matching_modules.append(mod)

        # If no modules match, return None
        if not matching_modules:
            return None

        # Combine the restored information for all matching modules and return
        results = [
            self._restore_module(module_info, classes, methods, method_bodies, docs) for module_info in matching_modules
        ]

        # restore the functions in the module if needed
        if functions:
            for module_info in matching_modules:
                for function_name in module_info.function_names:
                    function_full_name = '.'.join([module_info.full_name, function_name])

                    function_info = self._methods.get(function_full_name)
                    if not function_info:
                        continue

                    results.append(self._restore_method(function_info, method_bodies, docs))

        return "\n".join(results)

    def lookup_class(
        self,
        class_name: str,
        methods: bool = True,
        method_bodies: bool = True,
        docs: bool = True,
        pass_in_body: bool = True,
    ) -> Optional[str]:
        # Collect all classes that match by full name, class name, or suffix
        matching_classes = []

        # Try to find the class by its full name in the dictionary
        class_info = self._classes.get(class_name)
        if class_info:
            matching_classes.append(class_info)
        else:
            # Search for the class with a matching name attribute
            for full_name, cls in self._classes.items():
                if cls.name == class_name:
                    matching_classes.append(cls)

            # If still not found by name, try to find by suffix
            if not matching_classes:
                class_name_with_dot = "." + class_name
                for full_name, cls in self._classes.items():
                    if cls.full_name and cls.full_name.endswith(class_name_with_dot):
                        matching_classes.append(cls)

        if not matching_classes:
            return None

        # Combine the restored information for all matching classes and return
        results = [
            self._restore_class(class_info, methods, method_bodies, docs, pass_in_body)
            for class_info in matching_classes
        ]

        return "\n".join(results)

    def lookup_method(
        self,
        method_name: str,
        method_bodies: bool = True,
        docs: bool = True,
        pass_in_body: bool = True,
    ) -> Optional[str]:
        # Collect all methods or functions that match by full name, method/function name, or suffix
        matching_methods = []

        # Try to find the method/function by its full name in the dictionary
        method_info = self._methods.get(method_name)
        if method_info:
            matching_methods.append(method_info)
        else:
            # Search for the method/function with a matching name attribute
            for full_name, method in self._methods.items():
                if method.name == method_name:
                    matching_methods.append(method)

            # If still not found by name, try to find by suffix
            if not matching_methods:
                method_name_with_dot = "." + method_name
                for full_name, method in self._methods.items():
                    if method.full_name and method.full_name.endswith(method_name_with_dot):
                        matching_methods.append(method)

        if not matching_methods:
            return None
        
        # Combine the restored information for all matching methods and return
        results = [
            self._restore_method(method_info, method_bodies, docs, pass_in_body)
            for method_info in matching_methods
        ]

        return "\n".join(results)

    def _restore_module(
        self, module_info: CodeAtlasModuleInfo, classes: bool, methods: bool, method_bodies: bool, docs: bool
    ) -> str:
        text = f"# {module_info.full_name}\n"
        text += f"# {module_info.file_path}\n"

        # Include module docstring
        if docs and module_info.docstring:
            text += f'"""\n{module_info.docstring}\n"""\n\n'

        # Iterate over the classes in the module
        if classes:
            for class_name in module_info.class_names:
                if module_info.full_name:
                    class_full_name = module_info.full_name + "." + class_name
                else:
                    class_full_name = class_name

                class_info = self._classes.get(class_full_name)
                if not class_info:
                    # TODO: Placeholder?
                    continue

                text += self._restore_class(class_info, methods, method_bodies, docs)
                text += "\n\n"  # Add some space between classes

        return text

    def _restore_class(
        self,
        class_info: CodeAtlasClassInfo,
        methods: bool,
        method_bodies: bool,
        docs: bool,
        pass_in_body: bool = True,
    ) -> str:
        definition = ""

        if class_info.full_name:
            definition += f"# {class_info.full_name}\n"

        definition += f"class {class_info.name}"

        # Include parent classes
        if class_info.parent_classes:
            parents = ", ".join(class_info.parent_classes)
            definition += f"({parents})"

        definition += ":\n"

        # Include class docstring
        if docs and class_info.docstring:
            definition += indent(f'"""\n{class_info.docstring}\n"""\n\n')

        code = ""

        # Iterate over the methods in the class
        if methods:
            for method_name in class_info.methods:
                if class_info.full_name:
                    method_full_name = class_info.full_name + "." + method_name
                else:
                    method_full_name = method_name

                method_info = self._methods.get(method_full_name)
                if not method_info:
                    # TODO: Placeholder?
                    continue

                method_text = indent(
                    self._restore_method(method_info, method_bodies, docs, pass_in_body)
                )
                code += method_text
                code += "\n"  # Add a newline after each method

        if not code and pass_in_body:
            code = indent("pass\n")

        return definition + code

    def _restore_method(
        self,
        method_info: CodeAtlasMethodInfo,
        method_bodies: bool,
        docs: bool,
        pass_in_body: bool = True,
    ) -> str:
        # Include decorators if any
        decorators = "\n".join(f"@{decorator}" for decorator in method_info.decorators)
        if decorators:
            decorators += "\n"

        # Include arguments
        args = ", ".join(
            [a.name + ((": " + a.type_annotation) if a.type_annotation else "") + ((" = " + a.default_value) if a.default_value else "") for a in method_info.arguments]
        )

        # Include return type if available
        return_type = method_info.return_type
        if return_type == "NONE" or return_type == "None":
            return_type = None
        return_annotation = f" -> {return_type}" if return_type else ""

        definition = ""

        # if method_info.full_name:
        #     definition += f"# {method_info.full_name}\n"

        # Construct method definition with return type
        definition += f"{'async ' if method_info.is_async_method else ''}def {method_info.name}({args}){return_annotation}:\n"

        if docs and method_info.docstring:
            definition += indent(f'"""\n{method_info.docstring}\n"""\n')

        if method_bodies and method_info.source_code:
            code = indent(method_info.source_code)
        else:
            if pass_in_body:
                code = indent("pass\n")
            else:
                code = ""

        return decorators + definition + code
