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

    def __init__(self, starting_directory: str, visited_modules=None):
        self.starting_directory = Path(starting_directory)
        if visited_modules is None:
            visited_modules = {}
        self.visited_modules: Dict[str, CodeAtlasModuleInfo] = copy.copy(visited_modules)
        self.found_modules: List[CodeAtlasModuleInfo] = []
        self.found_classes: List[CodeAtlasClassInfo] = []
        self.found_methods: List[CodeAtlasMethodInfo] = []
        self.root_modules: List[Tuple[str, Path]] = []

    def analyze(self) -> List[CodeAtlasModuleInfo]:
        """Kick-starts the module analysis process and returns a list of found modules."""
        # Handling the case when the path includes a wildcard (*)
        starting_directories = glob.glob(str(self.starting_directory))

        for starting_directory in starting_directories:
            print("Scan", starting_directory)
            for root, dirs, files in os.walk(starting_directory):
                is_module = False
                files_set = set(files)
                for file in files:
                    # Process each '__init__.py' or '__init__.pyi' file to identify modules
                    if file in ("__init__.py", "__init__.pyi"):
                        # Prefer .pyi files over .py if both are present
                        if file == "__init__.py" and "__init__.pyi" in files_set:
                            continue
                        is_module = True
                        module_name = self.process_init_file(root, file)
                        break

                if is_module:
                    for file in files_set:
                        if not file.endswith(".py") or file == "__init__.py":
                            continue
                        submodule_name = module_name + "." + file.split(".")[0]
                        if submodule_name not in self.visited_modules:
                            self.process_module(os.path.join(root, file), submodule_name, is_root=False)

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

    def process_module(self, full_path: str, full_module_name: str, is_root: bool = True):
        """Processes a single Python module to collect its information and any sub-module."""
        # Avoid processing the same module twice
        if full_module_name in self.visited_modules:
            return
        
        # Record current module's information
        root_module_name, root_module_path = next(((name, path) for name, path in self.root_modules if Path(full_path).is_relative_to(path)), (None, None))
        is_root_module = is_root and root_module_name is None
        
        module_info = CodeAtlasModuleInfo(
            name=full_module_name.split(".")[-1],
            full_name=full_module_name,
            file_path=
                Path(full_module_name).joinpath(os.path.basename(full_path)).as_posix()
                if is_root_module
                else Path(root_module_name).joinpath(Path(full_path).relative_to(root_module_path)).as_posix(),
        )

        if is_root_module:
            self.root_modules.append((full_module_name, Path(full_path).parent))
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

        # Read module's source code
        with open(full_path, "r", encoding="utf-8", errors="replace") as file:
            source = file.read()
        # Remove placeholder that interferes with AST parsing
        source = source.replace("None = 'none'", "")
        source = source.replace("None:", "NONE:")
        source = source.replace("${ext_name}Extension", "ExtNameExtension")
        source = source.replace("${python_module}", "python_module")
        source = _replace_colons_outside_brackets(source)
        parsed_source = ast.parse(source)

        collector = CodeAtlasCollector(full_module_name, source.splitlines(keepends=True))
        collector.visit(parsed_source)

        if collector.equivalent_modules:
            module_info.equivelant_modules = collector.equivalent_modules

        for wildcard_import in collector.wildcarts_modules:
            # resolved_name = ModuleResolver.get_full_module_name(wildcard_import, full_module_name, is_root)
            resolved_path = ModuleResolver.get_module_path(wildcard_import, full_path, is_root)
            if resolved_path:
                # TODO: It should be a recursive call to process_module
                with open(resolved_path, "r", encoding="utf-8", errors="replace") as file:
                    source = file.read()

                # Remove placeholder that interferes with AST parsing
                source = source.replace("None = 'none'", "")
                source = source.replace("None:", "NONE:")
                source = source.replace("${ext_name}Extension", "ExtNameExtension")
                source = source.replace("${python_module}", "python_module")
                source = _replace_colons_outside_brackets(source)
                parsed_source = ast.parse(source)

                sub_collector = CodeAtlasCollector(full_module_name, source.splitlines(keepends=True))
                sub_collector.visit(parsed_source)

                collector.classes += sub_collector.classes
                collector.methods += sub_collector.methods

        resolver = ModuleResolver()
        # Resolve full module names and paths for each collected import
        for import_name in collector.collected_modules:
            resolved_name = resolver.get_full_module_name(import_name, full_module_name, is_root)
            resolved_path = resolver.get_module_path(import_name, full_path, is_root)
            # Process further if the import corresponds to a module that has a found path
            if resolved_path:
                self.process_module(resolved_path, resolved_name, is_root=False)

        # Updated to include the class names in the module info
        module_info.class_names = [class_info.name for class_info in collector.classes]

        # Store classes found in this module
        self.found_classes.extend(collector.classes)
        # Store methods found in this module
        self.found_methods.extend(collector.methods)

        # process all publicly exposed classes and methods in the module if the module is a root module
        if is_root_module:
            self.process_publicly_exposed(parsed_source, full_module_name, module_info, collector.imports.keys())

    def process_publicly_exposed(self, parsed_source: ast.Module, full_module_name: str, module_info: CodeAtlasModuleInfo, imports: List[str]):
        '''Processes all publicly exposed classes and methods in the module via imports and the __all__ variable, and creates copies of them in the higher-level extension module''' 

        # Find elements assigned to the __all__ variable in the module
        all_var_elements = None
        for node in ast.walk(parsed_source):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "__all__":
                all_var_elements = [ast.literal_eval(node) for node in node.value.elts]
                break
        
        if all_var_elements:
            elements = all_var_elements
        elif imports:
            elements = imports
        else:
            return

        # create dictionaries of self.found_modules, self.found_classes, and self.found_methods to be more easily searchable for helper functions
        current_modules = {module.full_name: module for module in self.found_modules}
        current_classes = {class_info.full_name: class_info for class_info in self.found_classes}
        current_methods = {method.full_name: method for method in self.found_methods}
 
        # list to hold the new classes and functions created from the __all__ variable in __init__.py
        new_direct_class_names = []
        new_direct_function_names = []
    
        # iterate through the __all__ variable's values, and for each value, find the corresponding method/class/module, and create a copy and add it in this higher-level extension module
        for val in elements:
            if not isinstance(val, str):
                continue
            existing_class = next((c for c in self.found_classes if val == c.name), None)
            if existing_class:
                new_direct_class_names.append(val)
                _process_equivalent_class(existing_class, full_module_name, val, current_classes, current_methods)
            else:
                existing_method = next((m for m in self.found_methods if val == m.name), None)
                if existing_method:
                    new_direct_function_names.append(val)
                    _process_equivalent_method(existing_method, full_module_name, None, val, current_methods)
                else:
                    existing_module = next((m for m in self.found_modules if val == m.name), None)
                    if existing_module:
                        _process_equivalent_module(existing_module, f"{full_module_name}.{val}", current_modules, current_classes, current_methods)
                    else:
                        print(f"value in variable __all__ ({val}) not found in found_classes, found_methods, or found_modules of {full_module_name}")
        
        # add the new_direct_classes to the module_info
        module_info.class_names += [class_name for class_name in new_direct_class_names if class_name not in module_info.class_names]
        module_info.function_names += [function_name for function_name in new_direct_function_names if function_name not in module_info.function_names]
        current_modules[full_module_name] = module_info

        # update our self.found_* Lists with the new values created from the __all__ variable
        self.found_modules = list(current_modules.values())
        self.found_classes = list(current_classes.values())
        self.found_methods = list(current_methods.values())
