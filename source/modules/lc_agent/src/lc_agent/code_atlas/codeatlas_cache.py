# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from .module_analyzer import ModuleAnalyzer, _process_equivalent_module
from .codeatlas_module_info import CodeAtlasClassInfo
from .codeatlas_module_info import CodeAtlasMethodInfo
from .codeatlas_module_info import CodeAtlasModuleInfo
from .codeatlas_lookup import CodeAtlasLookup
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
# import carb.tokens
import copy
import json


class CodeAtlasCache(CodeAtlasLookup):
    def __init__(self):
        self._modules: Dict[str, CodeAtlasModuleInfo] = {}
        self._classes: Dict[str, CodeAtlasClassInfo] = {}
        self._methods: Dict[str, CodeAtlasMethodInfo] = {}

        self._used_classes: Dict[str, List[str]] = {}

        super().__init__(self._modules, self._classes, self._methods)

    def clear(self):
        self._modules.clear()
        self._classes.clear()
        self._methods.clear()

        self._used_classes.clear()

    def empty(self):
        return not (self._modules or self._classes or self._methods or self._used_classes)

    def scan(self, module_path: str, overwrite=True):
        # token = carb.tokens.get_tokens_interface()
        # module_path = token.resolve(module_path)

        # Primary library we are interested in
        module_analyzer = ModuleAnalyzer(module_path, self._modules if not overwrite else None)
        module_analyzer.analyze()

        # All the classes in module analyzer directory
        classes = set([c.full_name for c in module_analyzer.found_classes])

        # Find methods that use classes from module_analyzer directory
        used_classes = defaultdict(list)
        for f in module_analyzer.found_methods:
            for used_class in f.class_usages:
                if used_class in classes:
                    used_classes[used_class].append(f.full_name)

        # Store the results
        self._modules.update({m.full_name: m for m in module_analyzer.found_modules})
        self._classes.update({c.full_name: c for c in module_analyzer.found_classes})
        self._methods.update({m.full_name: m for m in module_analyzer.found_methods})
        self._used_classes.update(used_classes)

    def scan_used_with(self, module_path: str):
        # token = carb.tokens.get_tokens_interface()
        # module_path = token.resolve(module_path)

        # Primary library we are interested in
        module_analyzer = ModuleAnalyzer(module_path)
        module_analyzer.analyze()

        # All the classes in module analyzer directory
        classes = set(self._classes.keys())

        # Find methods that use classes from module_analyzer directory
        used_classes = defaultdict(list)
        methods = set()
        for f in module_analyzer.found_methods:
            for used_class in f.class_usages:
                if used_class in classes:
                    used_classes[used_class].append(f.full_name)
                    methods.add(f.full_name)

        # Store the results
        self._methods.update({m.full_name: m for m in module_analyzer.found_methods if m.full_name in methods})
        self._used_classes.update(used_classes)

    def load(self, path: str, expand_equivalent_modules=True):
        self.clear()

        # token = carb.tokens.get_tokens_interface()
        # path = token.resolve(path)

        with open(path, "r") as f:
            json_data = json.load(f)

            # equivelant modules map
            module_maps = {}

            # Reconstruct modules
            if "modules" in json_data:
                for k, v in json_data["modules"].items():
                    self._modules[k] = CodeAtlasModuleInfo(**v)
                    if expand_equivalent_modules and self._modules[k].equivelant_modules:
                        module_maps[k] = self._modules[k].equivelant_modules
                
            # Reconstruct classes
            if "classes" in json_data:
                self._classes.update({k: CodeAtlasClassInfo(**v) for k, v in json_data["classes"].items()})

            # Reconstruct methods
            if "methods" in json_data:
                self._methods.update({k: CodeAtlasMethodInfo(**v) for k, v in json_data["methods"].items()})

            # Reconstruct used_classes
            if "used_classes" in json_data:
                self._used_classes.update(json_data["used_classes"])

            # if there are equivelant modules, map the modules, classes and methods
            if module_maps:
                for k, v in module_maps.items():
                    for m in v:
                        if m in self._modules and k in self._modules:
                            k_full_name = self._modules[k].full_name
                            _process_equivalent_module(self._modules[m], k_full_name, self._modules, self._classes, self._methods)


    def save(self, path: str):
        # token = carb.tokens.get_tokens_interface()
        # path = token.resolve(path)     

        with open(path, "w") as f:
            json_data = {
                "modules": {k: self._modules[k].model_dump(by_alias=True, exclude_defaults=True) for k in sorted(self._modules.keys())},
                "classes": {k: self._classes[k].model_dump(by_alias=True, exclude_defaults=True) for k in sorted(self._classes.keys())},
                "methods": {k: self._methods[k].model_dump(by_alias=True, exclude_defaults=True) for k in sorted(self._methods.keys())},
                "used_classes": {k: self._used_classes[k] for k in sorted(self._used_classes.keys())},
            }
            # Convert Python objects to JSON string
            json_string = json.dumps(json_data, indent=4)
            # Write JSON string to file
            f.write(json_string)

    def lookup_used_with(
        self, class_name: str, method_bodies: bool = True, docs: bool = True, as_list: bool = False
    ) -> Optional[Union[List[str], str]]:
        method_names = self._used_classes.get(class_name)
        if not method_names and "." not in class_name:
            method_names = []
            for k, v in self._used_classes.items():
                if k.split(".")[-1] == class_name:
                    method_names.extend(v)
        if not method_names:
            class_name = "." + class_name
            method_names = []
            for k, v in self._used_classes.items():
                if k.endswith(class_name):
                    method_names.extend(v)
        if not method_names:
            return None

        results = []
        for m in method_names:
            method_info = self._methods.get(m)
            if not method_info:
                continue

            results.append(self._restore_method(method_info, method_bodies, docs))

        if as_list:
            return results

        return "\n".join(results)
