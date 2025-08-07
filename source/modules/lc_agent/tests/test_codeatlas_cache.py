## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open
from lc_agent.code_atlas.codeatlas_cache import CodeAtlasCache
from lc_agent.code_atlas.codeatlas_module_info import CodeAtlasModuleInfo, CodeAtlasClassInfo, CodeAtlasMethodInfo, CodeAtlasArgumentInfo

@pytest.fixture
def sample_cache():
    cache = CodeAtlasCache()
    module_info = CodeAtlasModuleInfo(name="test_module", full_name="test_module", file_path="/path/to/test_module.py", class_names=["TestClass"])
    class_info = CodeAtlasClassInfo(name="TestClass", full_name="test_module.TestClass", module_name="test_module")
    method_info = CodeAtlasMethodInfo(name="test_method", full_name="test_module.TestClass.test_method", module_name="test_module", parent_class="TestClass")
    
    module_info2 = CodeAtlasModuleInfo(name="test_module2", full_name="test_module2", file_path="/path/to/test_module2.py", class_names=["TestClass2"])
    class_info2 = CodeAtlasClassInfo(name="TestClass2", full_name="test_module.TestClass2", module_name="test_module2")

    module_info3 = CodeAtlasModuleInfo(name="test_module3", full_name="test_module3", file_path="/path/to/test_module3.py", equivelant_modules=["test_module", "test_module2"])

    cache._modules["test_module"] = module_info
    cache._classes["test_module.TestClass"] = class_info
    cache._methods["test_module.TestClass.test_method"] = method_info

    cache._modules["test_module2"] = module_info2
    cache._classes["test_module2.TestClass2"] = class_info2

    cache._modules["test_module3"] = module_info3
    
    cache._used_classes["test_module.TestClass"] = ["test_method"]
    
    return cache

def test_lookup_module(sample_cache):
    result = sample_cache.lookup_module("test_module")
    assert result is not None
    assert "test_module" in result

def test_lookup_class(sample_cache):
    result = sample_cache.lookup_class("test_module.TestClass")
    assert result is not None
    assert "TestClass" in result

def test_lookup_nonexistent_module(sample_cache):
    assert sample_cache.lookup_module("nonexistent_module") is None

def test_lookup_nonexistent_class(sample_cache):
    assert sample_cache.lookup_class("nonexistent_class") is None

def test_lookup_module_with_classes(sample_cache):
    result = sample_cache.lookup_module("test_module", classes=True)
    assert "test_module" in result
    assert "test_module.TestClass" in sample_cache._classes

def test_lookup_module_without_classes(sample_cache):
    result = sample_cache.lookup_module("test_module", classes=False)
    assert "test_module" in result
    assert "TestClass" not in result

def test_lookup_class_with_methods(sample_cache):
    result = sample_cache.lookup_class("test_module.TestClass", methods=True)
    assert "TestClass" in result
    assert "test_module.TestClass.test_method" in sample_cache._methods

def test_lookup_class_without_methods(sample_cache):
    result = sample_cache.lookup_class("test_module.TestClass", methods=False)
    assert "TestClass" in result
    assert "test_method" not in result

def test_restore_method(sample_cache):
    method_info = CodeAtlasMethodInfo(name="test_method", full_name="test_module.TestClass.test_method", module_name="test_module", parent_class="TestClass")
    result = sample_cache._restore_method(method_info, True, True)
    assert result is not None
    assert "test_method" in result

def test_empty(sample_cache):
    # Initially, the cache should not be empty
    assert sample_cache.empty() == False
    
    # Clear the cache manually
    sample_cache._modules.clear()
    sample_cache._classes.clear()
    sample_cache._methods.clear()
    sample_cache._used_classes.clear()
    
    # Now the cache should be empty
    assert sample_cache.empty() == True

    # Add some data back
    sample_cache._modules["test"] = CodeAtlasModuleInfo(name="test", full_name="test", file_path="/path/to/test.py")
    
    # Cache should no longer be empty
    assert sample_cache.empty() == False

@patch('lc_agent.code_atlas.module_analyzer.ModuleAnalyzer')
def test_scan(mock_module_analyzer, sample_cache):
    mock_analyzer = mock_module_analyzer.return_value
    mock_analyzer.found_modules = [CodeAtlasModuleInfo(name="new_module", full_name="new_module", file_path="/path/to/new_module.py")]
    mock_analyzer.found_classes = [CodeAtlasClassInfo(name="NewClass", full_name="new_module.NewClass", module_name="new_module")]
    mock_analyzer.found_methods = [CodeAtlasMethodInfo(name="new_method", full_name="new_module.NewClass.new_method", module_name="new_module", parent_class="NewClass")]
    mock_analyzer.analyze = lambda: None

    sample_cache.scan(str(Path(__file__).parent.parent))
    sample_cache.scan_used_with(str(Path(__file__).parent.parent))

    assert "src.lc_agent" in sample_cache._modules
    assert "src.lc_agent.RunnableNode" in sample_cache._classes
    assert "src.lc_agent.RunnableNode.invoke" in sample_cache._methods

    assert "test_module" in sample_cache._modules
    assert "test_module.TestClass" in sample_cache._classes
    assert "test_module.TestClass.test_method" in sample_cache._methods

def test_save(sample_cache):
    sample_cache.scan(str(Path(__file__).parent.parent))

    assert sample_cache._modules
    assert sample_cache._classes
    assert sample_cache._methods
    
    # Get temp directory
    file = tempfile.NamedTemporaryFile().name + ".json"
    sample_cache.save(file)
    # Check the file exists
    assert os.path.exists(file)
    assert os.path.getsize(file) > 0
    
    # Check equivalent modules
    sample_cache.load(file)
    info = sample_cache._modules["test_module3"]
    assert "TestClass" in info.class_names
    assert "TestClass2" in info.class_names
    assert "test_module3.TestClass" in sample_cache._classes
    assert "test_module3.TestClass2" in sample_cache._classes

def test_lookup_used_with(sample_cache):
    sample_cache.scan(str(Path(__file__).parent.parent))
    sample_cache.scan_used_with(str(Path(__file__).parent.parent))

    assert sample_cache.lookup_used_with("nonexistent_class") is None

if __name__ == "__main__":
    pytest.main(["-v", "test_codeatlas_cache.py"])