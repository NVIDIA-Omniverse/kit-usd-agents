## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from lc_agent.code_atlas.codeatlas_lookup import CodeAtlasLookup, indent
from lc_agent.code_atlas.codeatlas_module_info import CodeAtlasModuleInfo, CodeAtlasClassInfo, CodeAtlasMethodInfo, CodeAtlasArgumentInfo

@pytest.fixture
def sample_lookup():
    modules = {
        "test_module": CodeAtlasModuleInfo(
            name="test_module",
            full_name="test_module",
            file_path="/path/to/test_module.py",
            class_names=["TestClass"],
            function_names=["test_function"]
        )
    }
    classes = {
        "test_module.TestClass": CodeAtlasClassInfo(
            name="TestClass",
            full_name="test_module.TestClass",
            module_name="test_module",
            methods=["test_method"]
        )
    }
    methods = {
        "test_module.TestClass.test_method": CodeAtlasMethodInfo(
            name="test_method",
            full_name="test_module.TestClass.test_method",
            module_name="test_module",
            parent_class="TestClass",
            arguments=[CodeAtlasArgumentInfo(name="arg1", type_annotation="int", default_value="10")]
        ),
        "test_module.test_function": CodeAtlasMethodInfo(
            name="test_function",
            full_name="test_module.test_function",
            module_name="test_module",
            arguments=[CodeAtlasArgumentInfo(name="arg1", type_annotation="int", default_value="15")]
        )
    }
    return CodeAtlasLookup(modules, classes, methods)

def test_indent():
    assert indent("test") == "    test"
    assert indent("test", 2) == "        test"
    assert indent("test\ntest") == "    test\n    test"

def test_lookup_module(sample_lookup):
    result = sample_lookup.lookup_module("test_module")
    assert result is not None
    assert "test_module" in result
    assert "TestClass" in result
    assert "test_function" in result

def test_lookup_class(sample_lookup):
    result = sample_lookup.lookup_class("test_module.TestClass")
    assert result is not None
    assert "class TestClass" in result
    assert "def test_method" in result

def test_lookup_method(sample_lookup):
    result = sample_lookup._restore_method(sample_lookup._methods["test_module.TestClass.test_method"], True, True)
    assert result is not None
    assert "def test_method" in result
    assert "arg1: int = 10" in result

def test_lookup_nonexistent_module(sample_lookup):
    assert sample_lookup.lookup_module("nonexistent_module") is None

def test_lookup_nonexistent_class(sample_lookup):
    assert sample_lookup.lookup_class("nonexistent_class") is None

def test_lookup_module_with_classes(sample_lookup):
    result = sample_lookup.lookup_module("test_module", classes=True)
    assert "class TestClass" in result

def test_lookup_module_without_classes(sample_lookup):
    result = sample_lookup.lookup_module("test_module", classes=False)
    assert "class TestClass" not in result

def test_lookup_module_with_functions(sample_lookup):
    result = sample_lookup.lookup_module("test_module", functions=True)
    assert "def test_function" in result

def test_lookup_module_without_functions(sample_lookup):
    result = sample_lookup.lookup_module("test_module", functions=False)
    assert "def test_function" not in result

def test_lookup_class_with_methods(sample_lookup):
    result = sample_lookup.lookup_class("test_module.TestClass", methods=True)
    assert "def test_method" in result

def test_lookup_class_without_methods(sample_lookup):
    result = sample_lookup.lookup_class("test_module.TestClass", methods=False)
    assert "def test_method" not in result

def test_lookup_method(sample_lookup):
    result = sample_lookup.lookup_method("test_module.TestClass.test_method", method_bodies=True)
    assert "def test_method" in result
    assert "arg1: int = 10" in result

def test_lookup_function(sample_lookup):
    result = sample_lookup.lookup_method("test_module.test_function", method_bodies=True)
    assert "def test_function" in result
    assert "arg1: int = 15" in result

def test_lookup_nonexistent_method(sample_lookup):
    assert sample_lookup.lookup_method("nonexistent_method") is None

if __name__ == "__main__":
    pytest.main(["-v", "test_codeatlas_lookup.py"])