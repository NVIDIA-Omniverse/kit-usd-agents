## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from unittest.mock import patch, mock_open
from lc_agent.code_atlas.module_analyzer import ModuleAnalyzer, ModuleResolver, _replace_colons_outside_brackets
from lc_agent.code_atlas.codeatlas_module_info import CodeAtlasModuleInfo, CodeAtlasClassInfo, CodeAtlasMethodInfo
import ast
import os

@pytest.fixture
def sample_directory_structure(tmp_path):
    root = tmp_path / "test_package"
    root.mkdir()
    (root / "__init__.py").write_text("# Root init")
    (root / "module1.py").write_text("# Module 1 content")
    sub_pkg = root / "sub_package"
    sub_pkg.mkdir()
    (sub_pkg / "__init__.py").write_text("# Sub-package init")
    (sub_pkg / "module2.py").write_text("# Module 2 content")
    root2 = tmp_path / "test_package2"
    root2.mkdir()
    (root2 / "__init__.py").write_text("# Root2 init")
    return tmp_path

def test_module_resolver_get_full_module_name():
    resolver = ModuleResolver()
    assert resolver.get_full_module_name("module", "package", True) == "module"
    assert resolver.get_full_module_name(".module", "package", True) == "package.module"
    assert resolver.get_full_module_name("..module", "package.sub", False) == "module"

@patch('os.path.exists')
def test_module_resolver_get_module_path(mock_exists, sample_directory_structure):
    mock_exists.return_value = True
    resolver = ModuleResolver()
    root = sample_directory_structure / "test_package"
    root_init = str(root / "__init__.py")
    sub_init = str(root / "sub_package" / "__init__.py")
    
    expected_module1_path = os.path.join(os.path.dirname(os.path.dirname(root_init)), "module1", "py", "__init__.py")
    expected_sub_package_path = os.path.join(os.path.dirname(root_init), "sub_package", "__init__.py")
    expected_parent_module1_path = os.path.join(os.path.dirname(os.path.dirname(sub_init)), "module1", "py", "__init__.py")
    
    assert resolver.get_module_path("module1.py", root_init, True) == expected_module1_path
    assert resolver.get_module_path(".sub_package", root_init, True) == expected_sub_package_path
    assert resolver.get_module_path("..module1.py", sub_init, False) == expected_parent_module1_path

def test_replace_colons_outside_brackets():
    assert _replace_colons_outside_brackets("a::b") == "a.b"
    assert _replace_colons_outside_brackets("a[::b]") == "a[::b]"
    assert _replace_colons_outside_brackets("a::b[c::d]e::f") == "a.b[c::d]e.f"

@patch('builtins.open', new_callable=mock_open, read_data="class TestClass:\n    pass")
def test_module_analyzer_process_module(mock_file, tmp_path):
    analyzer = ModuleAnalyzer(str(tmp_path))
    analyzer.process_module("/fake/path/module.py", "test_module")
    
    assert len(analyzer.found_modules) == 1
    assert analyzer.found_modules[0].name == "test_module"
    assert len(analyzer.found_classes) == 1
    assert analyzer.found_classes[0].name == "TestClass"

@patch('os.walk')
@patch('glob.glob')
@patch('builtins.open', new_callable=mock_open, read_data="# Module content")
def test_module_analyzer_analyze(mock_file, mock_glob, mock_walk, tmp_path):
    mock_glob.return_value = [str(tmp_path)]
    mock_walk.return_value = [
        (str(tmp_path), [], ['__init__.py', 'module1.py']),
        (str(tmp_path / 'subpackage'), [], ['__init__.py', 'module2.py'])
    ]
    
    analyzer = ModuleAnalyzer(str(tmp_path))
    analyzer.analyze()
    
    assert len(analyzer.found_modules) == 4  # .,  .subpackage, .module1 and .subpackage.module2
    module_names = set(m.name for m in analyzer.found_modules)
    assert '' in module_names or tmp_path.name in module_names
    assert 'subpackage' in module_names

def test_module_analyzer_module_name_from_path(tmp_path):
    analyzer = ModuleAnalyzer(str(tmp_path))
    assert analyzer.module_name_from_path(str(tmp_path / 'subpackage')) == 'subpackage'
    assert analyzer.module_name_from_path(str(tmp_path / 'deep' / 'nested')) == 'deep.nested'

def test_module_analyzer_process_publicly_exposed(tmp_path):
    analyzer = ModuleAnalyzer(str(tmp_path))
    module_info = CodeAtlasModuleInfo(name="test_module", file_path="/fake/path/module.py")
    full_module_name = "test_module"
    imports = ["imported_class1", "imported_function1"]
    all_variable = """__all__ = ["all_variable_class1", "all_variable_class2"]"""
    mock_module = ast.parse(all_variable)

    analyzer.found_classes = [
        CodeAtlasClassInfo(name="imported_class1", module_name="test_module", full_name="test_module.imported_class1"),
        CodeAtlasClassInfo(name="all_variable_class1", module_name="test_module", full_name="test_module.all_variable_class1"), 
        CodeAtlasClassInfo(name="all_variable_class2", module_name="test_module", full_name="test_module.all_variable_class2")
    ]

    analyzer.found_methods = [
        CodeAtlasMethodInfo(name="imported_function1", module_name="test_module", full_name="test_module.imported_function1")
    ]

    analyzer.process_publicly_exposed(mock_module, full_module_name, module_info, imports)

    assert len(analyzer.found_classes) == 3
    assert len(analyzer.found_methods) == 1
    assert analyzer.found_classes[0].name == "imported_class1"
    assert analyzer.found_classes[1].name == "all_variable_class1"
    assert analyzer.found_classes[2].name == "all_variable_class2"
    assert analyzer.found_methods[0].name == "imported_function1"
    
def test_module_analyzer_multiple_roots(sample_directory_structure):
    analyzer = ModuleAnalyzer(str(sample_directory_structure))
    analyzer.analyze()
    assert len(analyzer.root_modules) == 2

if __name__ == "__main__":
    pytest.main(["-v", "test_module_analyzer.py"])