# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from .codeatlas_collector import CodeAtlasCollector
from .codeatlas_module_info import CodeAtlasClassInfo
from .codeatlas_module_info import CodeAtlasMethodInfo
from .codeatlas_module_info import CodeAtlasModuleInfo
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
import ast
import glob
import os
import copy
import toml
from collections import namedtuple
import re

_ModuleObject = namedtuple('_ModuleObject', ['module_name', 'object_name'])

def _replace_colons_outside_brackets(source):
    processed_lines = []
    for line in source.splitlines():
        result = []
        inside_brackets = False  # Track whether we're inside square brackets
        chars = iter(enumerate(line))  # Create an iterator to go through line with index

        for index, char in chars:
            if char == "[":
                inside_brackets = True
            elif char == "]":
                inside_brackets = False

            # Check for '::' outside of square brackets
            if not inside_brackets and char == ":" and (index + 1) < len(line) and line[index + 1] == ":":
                result.append(".")
                next(chars, None)  # Skip the next character as it's part of '::'
            else:
                result.append(char)

        processed_lines.append("".join(result))

    return "\n".join(processed_lines)

def _process_equivalent_module(existing_module: CodeAtlasModuleInfo, new_module_name: str, modules: Dict[str, CodeAtlasModuleInfo], classes: Dict[str, CodeAtlasClassInfo], methods: Dict[str, CodeAtlasMethodInfo]):
    """Creates and saves a deep copy of a single module with the module name changed to the new module name into the modules dictionary and processes the classes in the module as well."""
    if not existing_module:
        return None

    new_module = modules.get(new_module_name)
    if new_module is None:
        new_module = existing_module.model_copy(deep=True)
        new_module.full_name = new_module_name
        modules[new_module.full_name] = new_module
    else:
        # If the module already exists, only copy classes
        new_module.class_names += existing_module.class_names

    # update the classes in the existing module as well, which will also update the methods in the classes
    for existing_class_name in existing_module.class_names:
        full_class_name = '.'.join(filter(None, [existing_module.full_name, existing_class_name]))
        if full_class_name in classes.keys():
            existing_class = classes[full_class_name]
            _process_equivalent_class(existing_class, new_module.full_name, existing_class_name, classes, methods)

def _process_equivalent_class(existing_class: CodeAtlasClassInfo, new_module_name: str, new_class_name: str, classes: Dict[str, CodeAtlasClassInfo], methods: Dict[str, CodeAtlasMethodInfo]):
    """Creates and saves a deep copy of a single class into the classes dictionary and processes the methods in the class as well."""
    if not existing_class:
        return None

    new_class = existing_class.model_copy(deep=True)
    new_class.module_name = new_module_name
    new_class.full_name = '.'.join(filter(None, [new_module_name, new_class_name]))

    classes[new_class.full_name] = new_class

    # update the methods in the class as well
    for method_name in existing_class.methods:
        full_method_name = '.'.join(filter(None, [existing_class.full_name, method_name]))
        if full_method_name in methods.keys():
            existing_method = methods[full_method_name]
            _process_equivalent_method(existing_method, new_module_name, new_class_name, method_name, methods)

def _process_equivalent_method(existing_method: CodeAtlasMethodInfo, new_module_name: str, new_class_name: str, new_method_name: str, methods: Dict[str, CodeAtlasMethodInfo]):
    """Creates and saves a deep copy of a single method into the methods dictionary."""
    if not existing_method:
        return None

    new_method = existing_method.model_copy(deep=True)
    new_method.module_name = new_module_name
    new_method.full_name = '.'.join(filter(None, [new_module_name, new_class_name, new_method_name]))

    for arg in new_method.arguments:
        arg.parent_method = new_method.full_name

    methods[new_method.full_name] = new_method

class ModuleResolver:
    """Utility class for resolving module names and paths."""

    @staticmethod
    def get_full_module_name(import_name: str, parent_full_name: str, is_root: bool) -> str:
        """Resolve the full module name from an import statement."""
        # Absolute import
        if not import_name.startswith("."):
            return import_name

        base_parts = parent_full_name.split(".")
        depth = len(import_name) - len(import_name.lstrip(".")) - int(is_root)
        relative_part = import_name.lstrip(".")
        base_full_name = ".".join(base_parts[:-depth] if depth > 0 else base_parts) if depth < len(base_parts) else ""
        return f"{base_full_name}.{relative_part}".strip(".")

    @staticmethod
    def get_module_path(import_name: str, parent_module_path: str, is_root: bool) -> Optional[str]:
        """
        Resolve the absolute path of a module from an import statement and the
        path of the parent module.
        """
        # Determine the base directory of the parent module
        parent_dir = os.path.dirname(parent_module_path)

        if import_name.startswith("."):
            # Relative import: navigate up the path by the number of leading dots.
            depth = len(import_name) - len(import_name.lstrip(".")) - 1
            module_relative_path = import_name.lstrip(".").replace(".", os.sep)
            # Ascend to the correct parent directory
            module_dir = parent_dir
            for _ in range(depth):
                module_dir = os.path.dirname(module_dir)
            # Construct potential paths
            potential_paths = [
                os.path.join(module_dir, *module_relative_path.split("/"), "__init__.py"),  # Package
                os.path.join(module_dir, f"{module_relative_path}.py"),  # Module
                os.path.join(module_dir, f"{module_relative_path}.pyi"),  # Module
            ]
        else:
            # Absolute import: build the path from the package root.
            package_root = os.path.dirname(parent_dir)
            module_path = import_name.replace(".", os.sep)
            potential_paths = [
                os.path.join(package_root, *module_path.split("/"), "__init__.py"),  # Package
                os.path.join(package_root, f"{module_path}.py"),  # Module
                os.path.join(package_root, f"{module_path}.pyi"),  # Module
            ]

        # Return the first existing path
        for path in potential_paths:
            if os.path.exists(path):
                return os.path.normpath(path)
        # Module not found, return None
        return None


class ModuleAnalyzer:
    """Analyzes a given directory to collect all Python modules present."""

    def __init__(self, starting_directory: str, visited_modules=None, excluded_modules=None):
        self.starting_directory = Path(starting_directory)
        if visited_modules is None:
            visited_modules = {}
        self.visited_modules: Dict[str, CodeAtlasModuleInfo] = copy.copy(visited_modules)
        self.excluded_modules: Optional[List[str]] = excluded_modules
        self.found_modules: List[CodeAtlasModuleInfo] = []
        self.found_classes: List[CodeAtlasClassInfo] = []
        self.found_methods: List[CodeAtlasMethodInfo] = []
        self.root_modules: List[Tuple[str, Path]] = []
        # maps an object to the list of objects that reference it
        self.object_references: Dict[_ModuleObject, List[_ModuleObject]] = {}
        # maps a module to the list of objects imported into it
        self.module_objects: Dict[str, List[str]] = {}

    def analyze(self) -> List[CodeAtlasModuleInfo]:
        """Kick-starts the module analysis process and returns a list of found modules."""
        # Handling the case when the path includes a wildcard (*)
        starting_directories = glob.glob(str(self.starting_directory))
        for starting_directory in starting_directories:
            print("Scan", starting_directory)
            for root, dirs, files in os.walk(starting_directory, followlinks=True):
                if self.excluded_modules and any(Path(root).relative_to(starting_directory).is_relative_to(Path(excluded_module)) for excluded_module in self.excluded_modules):
                    continue
                module_name = None
                files_set = set(files)
                for file in files:
                    # Process each '__init__.py' or '__init__.pyi' file to identify modules
                    if file in ("__init__.py", "__init__.pyi"):
                        # Prefer .pyi files over .py if both are present
                        if file == "__init__.py" and "__init__.pyi" in files_set:
                            continue
                        module_name = self.process_init_file(root, file)
                        break

                for file in files_set:
                    if not any(file.endswith(ext) for ext in [".py", ".pyi"]) or file == "__init__.py":
                        continue
                    submodule_name = (module_name if module_name is not None else self.module_name_from_path(root)) + "." + file.split(".")[0]
                    full_path = os.path.join(root, file)
                    if submodule_name not in self.visited_modules and not any(m.file_path == Path(full_path).relative_to(starting_directory).as_posix() for m in self.visited_modules.values()):
                        self.process_module(full_path, submodule_name, is_root=False)

        self._promote_publicly_exposed()

    def _promote_publicly_exposed(self):
        """Promote classes and methods that are publicly exposed to the higher-level extension module."""
        classes = {c.full_name: c for c in self.found_classes}
        methods = {m.full_name: m for m in self.found_methods}

        for object_name in self.object_references.keys():
            full_object_name = ".".join(object_name)
            is_class = full_object_name in classes.keys()
            is_method = full_object_name in methods.keys()
            # go through objects that might need to be promoted
            if not (is_class or is_method):
                continue
            # get the list of all objects that are referenced by the current object
            stack = [object_name]
            visited = set()
            while len(stack) > 0:
                current_object_name = stack.pop()
                if current_object_name in visited:
                    continue
                visited.add(current_object_name)
                stack += self.object_references.get(current_object_name, [])
            visited.remove(object_name)
            if len(visited) == 0:
                continue
            references = list(sorted(visited))
            is_ancestor = object_name.module_name.startswith(references[0].module_name)

            if is_ancestor:
                new_name = f"{references[0].module_name}.{object_name.object_name}"
                if is_class:
                    existing_class = classes[full_object_name]
                    # Update the full name and module name of all classes that are nested in the existing class
                    for class_info in self.found_classes:
                        if class_info.full_name.startswith(f"{existing_class.full_name}."):
                            class_info.full_name = class_info.full_name.replace(existing_class.full_name, new_name)
                            class_info.module_name = references[0].module_name
                    # Update the full name and module name of all methods that are nested in the existing class
                    for method_info in self.found_methods:
                        if method_info.full_name.startswith(f"{existing_class.full_name}."):
                            method_info.full_name = method_info.full_name.replace(existing_class.full_name, new_name)
                            method_info.module_name = references[0].module_name
                            for arg in method_info.arguments:
                                arg.parent_method = method_info.full_name

                    existing_class.full_name = new_name
                    existing_class.module_name = references[0].module_name
                elif is_method:
                    existing_method = methods[full_object_name]
                    existing_method.full_name = new_name
                    existing_method.module_name = references[0].module_name
                    for arg in existing_method.arguments:
                        arg.parent_method = existing_method.full_name

            for ref_module, _ in references:
                module_info = self.visited_modules.get(ref_module)
                if is_class:
                    module_info.class_names.append(object_name.object_name)
                elif is_method:
                    module_info.function_names.append(object_name.object_name)

    def module_name_from_path(self, directory_path: str) -> str:
        """Generate a module's fully qualified name from its directory path."""
        relpath = os.path.relpath(directory_path, self.starting_directory)
        return relpath.replace(os.sep, ".") or self.starting_directory.name

    def process_init_file(self, root: str, init_file: str):
        """Processes a __init__.py or __init__.pyi file to collect module information."""
        full_module_name = self.module_name_from_path(root)
        full_path = os.path.join(root, init_file)
        self.process_module(full_path, full_module_name, is_root=True)
        return full_module_name

    def _detect_file_encoding(self, file_path: str) -> str:
        """
        Detects the file encoding by examining the first few bytes for BOM markers.

        Args:
            file_path: Path to the file to analyze

        Returns:
            encoding_name
        """
        with open(file_path, 'rb') as file:
            # Read first 4 bytes to check for BOM
            first_bytes = file.read(4)

        # Check for various BOM signatures
        if first_bytes.startswith(b'\xef\xbb\xbf'):
            # UTF-8 BOM
            return 'utf-8-sig'
        elif first_bytes.startswith(b'\xff\xfe\x00\x00'):
            # UTF-32 LE BOM
            return 'utf-32-le'
        elif first_bytes.startswith(b'\x00\x00\xfe\xff'):
            # UTF-32 BE BOM
            return 'utf-32-be'
        elif first_bytes.startswith(b'\xff\xfe'):
            # UTF-16 LE BOM
            return 'utf-16-le'
        elif first_bytes.startswith(b'\xfe\xff'):
            # UTF-16 BE BOM
            return 'utf-16-be'
        else:
            # No BOM detected, assume UTF-8
            return 'utf-8'

    def process_module(self, full_path: str, full_module_name: str, is_root: bool = True):
        """Processes a single Python module to collect its information and any sub-module."""
        # Avoid processing the same module twice
        if full_module_name in self.visited_modules:
            return

        # Record current module's information
        relative_path = Path(full_path).relative_to(self.starting_directory)
        root_module_name = next((name for name, path in self.root_modules if relative_path.is_relative_to(path)), None)
        is_root_module = is_root and root_module_name is None

        module_info = CodeAtlasModuleInfo(
            name=full_module_name.split(".")[-1],
            full_name=full_module_name,
            file_path=relative_path.as_posix()
        )

        if is_root_module:
            self.root_modules.append((full_module_name, relative_path.parent))
            path = Path(full_path)
            parts = full_module_name.split(".")
            if len(path.parent.parts) >= len(parts):
                extension_root = Path(*path.parent.parts[:-len(parts)])
                toml_path = extension_root / "config" / "extension.toml"
                if toml_path.exists():
                    config = toml.load(toml_path)
                    if any(python_module.get("name") == full_module_name for python_module in config.get("python", {}).get("module", [])):
                        # Remove the version at the end if it has one
                        module_info.extension_name = extension_root.parts[-1].split("-")[0]


        self.visited_modules[full_module_name] = module_info
        self.found_modules.append(module_info)

        # Read module's source code with proper encoding detection
        encoding = self._detect_file_encoding(full_path)
        with open(full_path, "r", encoding=encoding, errors="replace") as file:
            source = file.read()
        # Remove placeholder that interferes with AST parsing
        source = source.replace("None = 'none'", "")
        source = source.replace("None:", "NONE:")
        source = source.replace("${ext_name}Extension", "ExtNameExtension")
        source = source.replace("${python_module}", "python_module")
        source = _replace_colons_outside_brackets(source)
        try:
            parsed_source = ast.parse(source)
        except SyntaxError as e:
            print(f"Syntax error in {full_path}: {e}")
            return

        collector = CodeAtlasCollector(full_module_name, source.splitlines(keepends=True))
        collector.visit(parsed_source)

        if collector.equivalent_modules:
            module_info.equivalent_modules += collector.equivalent_modules

        is_init_file = full_path.endswith("__init__.py") or full_path.endswith("__init__.pyi")

        def _process_collected_module(import_name: str, is_root: bool):
            resolver = ModuleResolver()
            # Fix for subpackage imports not being resolved correctly. Example:
            # omni.foo
            # __init__.py
            # impl\
            #   __init__.py
            #   bar.py
            # omni.foo.impl.bar should be resolved as omni.foo.impl.bar, not omni.foo.bar.
            resolved_name = resolver.get_full_module_name(import_name, full_module_name, is_init_file)
            resolved_path = resolver.get_module_path(import_name, full_path, is_root)
            # Process further if the import corresponds to a module that has a found path
            if resolved_path:
                self.process_module(resolved_path, resolved_name, is_root=False)
                return resolved_name
            return None

        for wildcard_import in collector.wildcarts_modules:
            resolved_name = _process_collected_module(wildcard_import, is_root)
            if resolved_name:
                module_info.equivalent_modules.append(resolved_name)

        for import_name in collector.collected_modules:
            _process_collected_module(import_name, is_root)

        module_name_length = len(module_info.full_name.split("."))

        # Updated to include the class and method names in the module info, exclude classes and methods that are nested in other classes
        module_info.class_names = [class_info.name for class_info in collector.classes if len(class_info.full_name.split(".")) == module_name_length + 1]
        module_info.function_names = [method_info.name for method_info in collector.methods if len(method_info.full_name.split(".")) == module_name_length + 1]

        # Store classes found in this module
        self.found_classes.extend(collector.classes)
        # Store methods found in this module
        self.found_methods.extend(collector.methods)

        self._collect_object_references(collector, module_info, is_init_file)

    def _collect_object_references(self, collector: CodeAtlasCollector, module_info: CodeAtlasModuleInfo, is_init_file: bool):
        """Collects the object references of the current module."""
        module_object_map = {}

        for equiv_module_name in module_info.equivalent_modules:
            equiv_module_info = self.visited_modules.get(equiv_module_name)
            if equiv_module_info is None:
                continue

            for class_name in equiv_module_info.class_names:
                module_object_map[class_name] = _ModuleObject(equiv_module_name, class_name)
            for function_name in equiv_module_info.function_names:
                module_object_map[function_name] = _ModuleObject(equiv_module_name, function_name)

            for object_name in self.module_objects.get(equiv_module_name, []):
                module_object_map[object_name] = _ModuleObject(equiv_module_name, object_name)

        for import_alias, import_full_name in collector.imports.items():
            if import_full_name.startswith("."):
                import_module_name, import_object_name = import_full_name.rsplit(".", 1)
                resolved_module_name = ModuleResolver.get_full_module_name(import_module_name, module_info.full_name, is_init_file)
                module_object_map[import_alias] = _ModuleObject(resolved_module_name, import_object_name)
            #else:
            #    module_object_map[(module_info.full_name, import_alias)] = (import_full_name, import_full_name)

        if collector.all_attribute_list is not None:
            new_module_object_map = {
                object_name: mapped_object_name
                for object_name, mapped_object_name in module_object_map.items()
                if object_name in collector.all_attribute_list
            }
            module_object_map = new_module_object_map

        for object_name, mapped_object_name in module_object_map.items():
            object_list = self.object_references.get(mapped_object_name)
            if object_list is not None:
                object_list.append(_ModuleObject(module_info.full_name, object_name))
            else:
                self.object_references[mapped_object_name] = [_ModuleObject(module_info.full_name, object_name)]
        self.module_objects[module_info.full_name] = list(module_object_map.keys())
