## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
import ast
from lc_agent.code_atlas.codeatlas_collector import CodeAtlasCollector
from lc_agent.code_atlas.codeatlas_module_info import CodeAtlasClassInfo, CodeAtlasMethodInfo, CodeAtlasArgumentInfo

@pytest.fixture
def sample_code():
    return """
import module1
from module2 import Class2
from .module3 import *

class TestClass:
    class_var = 1

    def __init__(self, arg1: int, arg2: str = "default", *args, **kwargs):
        '''Docstring for __init__'''
        self.arg1 = arg1
        self.arg2 = arg2

    @typing.overload
    def __init__(self):
        pass

    @classmethod
    def class_method(cls):
        return cls.class_var

    @staticmethod
    def static_method():
        return "static"

    def method_with_annotation(self, arg1: typing.Callable[[int], None]) -> str:
        return "annotated"

    async def async_method():
        pass
        
    @property
    def value(self):
        return self.arg1

    @value.setter
    def value(self, value):
        self.arg1 = value

    class TestInnerClass:
        def inner_method(self):
            pass
            
    def single_line_method(self): ...

def global_function(param1, param2: list):
    '''Docstring for global_function'''
    Class2().method()
    return param1, param2

"""

@pytest.fixture
def collector(sample_code):
    tree = ast.parse(sample_code)
    collector = CodeAtlasCollector("test_module", sample_code.splitlines())
    collector.visit(tree)
    return collector

def test_import_collection(collector):
    assert "module1" in collector.imports
    assert "Class2" in collector.imports
    assert ".module3" in collector.wildcarts_modules

def test_class_collection(collector):
    assert len(collector.classes) == 2
    test_inner_class = collector.classes[0]
    assert isinstance(test_inner_class, CodeAtlasClassInfo)
    assert test_inner_class.name == "TestInnerClass"
    assert test_inner_class.full_name == "test_module.TestClass.TestInnerClass"
    assert len(test_inner_class.methods) == 1
    test_class = collector.classes[1]
    assert isinstance(test_class, CodeAtlasClassInfo)
    assert test_class.name == "TestClass"
    assert test_class.full_name == "test_module.TestClass"
    assert len(test_class.methods) == 9

def test_method_collection(collector):
    assert len(collector.methods) == 11  # Including global function
    init_method = next(m for m in collector.methods if m.name == "__init__")
    assert isinstance(init_method, CodeAtlasMethodInfo)
    assert init_method.docstring == "Docstring for __init__"
    assert len(init_method.arguments) == 5  # self, arg1, arg2, *args, **kwargs


def test_argument_collection(collector):
    init_method = next(m for m in collector.methods if m.name == "__init__")
    arg1 = next(a for a in init_method.arguments if a.name == "arg1")
    assert isinstance(arg1, CodeAtlasArgumentInfo)
    assert arg1.type_annotation == "int"
    arg2 = next(a for a in init_method.arguments if a.name == "arg2")
    assert arg2.default_value == "\"default\""
    args = next(a for a in init_method.arguments if a.name == "*args")
    assert isinstance(args, CodeAtlasArgumentInfo) 
    kwargs = next(a for a in init_method.arguments if a.name == "**kwargs")
    assert isinstance(kwargs, CodeAtlasArgumentInfo)

def test_decorator_collection(collector):
    class_method = next(m for m in collector.methods if m.name == "class_method")
    assert "classmethod" in class_method.decorators

def test_type_annotation(collector):
    annotated_method = next(m for m in collector.methods if m.name == "method_with_annotation")
    assert annotated_method.return_type == "str"
    arg1 = next(a for a in annotated_method.arguments if a.name == "arg1")
    assert isinstance(arg1, CodeAtlasArgumentInfo)
    assert arg1.type_annotation == "typing.Callable[[int], None]"

def test_async_collection(collector):
    async_method = next(m for m in collector.methods if m.name == "async_method")
    assert async_method.is_async_method

def test_property_collection(collector):
    value_methods = [m for m in collector.methods if m.name == "value"]
    assert len(value_methods) == 2
    assert any("property" in m.decorators for m in value_methods)
    assert any("value.setter" in m.decorators for m in value_methods)

def test_overload_collection(collector):
    init_methods = [m for m in collector.methods if m.name == "__init__"]
    assert len(init_methods) == 2
    assert any("typing.overload" in m.decorators for m in init_methods)
    
def test_single_method_collection(collector):
    single_line_method = next(m for m in collector.methods if m.name == "single_line_method")
    assert single_line_method.source_code == "..."

def test_global_function_collection(collector):
    global_func = next(m for m in collector.methods if m.name == "global_function")
    assert global_func.docstring == "Docstring for global_function"
    assert len(global_func.arguments) == 2

def test_class_usage_collection(collector):
    global_func = next(m for m in collector.methods if m.name == "global_function")
    assert len(global_func.class_usages) == 0  # Assuming class_usages is an empty list

def test_get_type_annotation_str():
    collector = CodeAtlasCollector("test", [])
    assert collector.get_type_annotation_str(ast.Name(id="int")) == "int"
    assert collector.get_type_annotation_str(ast.Attribute(value=ast.Name(id="typing"), attr="List")) == "typing.List"
    assert collector.get_type_annotation_str(ast.Constant(value="str")) == "str"

def test_extract_types_from_docstring():
    collector = CodeAtlasCollector("test", [])
    docstring = """
    Args:
        param1 (int): Description of param1
        param2 (str): Description of param2
    """
    types = collector.extract_types_from_docstring(docstring)
    assert types == {"Args": "param1 (int): Description of param1"}

def test_extract_function_body():
    collector = CodeAtlasCollector("test", [])
    node = ast.FunctionDef(
        name="test_func",
        args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[
            ast.Expr(value=ast.Constant(value="Docstring")),
            ast.Expr(value=ast.Constant(value="Body")),
        ],
        decorator_list=[]
    )
    node.body[0].lineno = 2
    node.body[1].lineno = 3
    node.body[1].end_lineno = 3
    source_lines = ["def test_func():", '    """Docstring"""', '    "Body"']
    result = collector.extract_function_body(node, source_lines)
    assert result == '"Body"'

def test_find_class_usages_in_body():
    collector = CodeAtlasCollector("test", [])
    collector.imports = {"module": "full.path.module"}
    node = ast.FunctionDef(
        name="test_func",
        args=ast.arguments(args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[
            ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id="module"), attr="Class"),
                args=[],
                keywords=[]
            ))
        ],
        decorator_list=[]
    )
    usages = collector.find_class_usages_in_body(node)
    assert ("module", "Class") in usages

def test_resolve_name():
    collector = CodeAtlasCollector("test", [])
    collector.imports = {"module": "full.path.module"}
    assert collector.resolve_name(("module", "Class")) == "full.path.module.Class"
    assert collector.resolve_name(("unknown", "Class")) is None

if __name__ == "__main__":
    pytest.main(["-v", "test_codeatlas_collector.py"])