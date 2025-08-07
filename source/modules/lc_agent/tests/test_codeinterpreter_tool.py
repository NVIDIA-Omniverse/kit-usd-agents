## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import pytest
from unittest.mock import patch, MagicMock
from lc_agent.code_atlas.codeinterpreter_tool import CodeInterpreterTool, execute_python_code, disable_items, restore_items, disabled_function

@pytest.fixture
def code_interpreter_tool():
    return CodeInterpreterTool()

def test_code_interpreter_tool_initialization(code_interpreter_tool):
    assert code_interpreter_tool.name == "CodeInterpreter"
    assert "Executes Python code snippets" in code_interpreter_tool.description
    assert code_interpreter_tool.ask_human_input == False

def test_code_interpreter_tool_run(code_interpreter_tool):
    code = "print('Hello, World!')"
    result = code_interpreter_tool._run(code)
    assert result == "Hello, World!"

@pytest.mark.asyncio
async def test_code_interpreter_tool_async_run():
    code = """
import asyncio
async def test():
    await asyncio.sleep(1)
    print("Hello world!")
asyncio.ensure_future(test())
"""
    async def wait():
        import asyncio
        await asyncio.sleep(2)
    interpreter = CodeInterpreterTool(wait_fn=wait)
    result = await interpreter._arun(code)
    assert "Hello world!" in result

def test_execute_python_code_print():
    code = "print('Test output')"
    result = execute_python_code(code)
    assert result == "Test output"

def test_execute_python_code_return():
    code = "2 + 2"
    result = execute_python_code(code)
    assert result == "4"

def test_execute_python_code_error():
    code = "1 / 0"
    result = execute_python_code(code)
    assert "Error: Traceback" in result
    assert "ZeroDivisionError" in result

def test_execute_python_code_with_hidden_items():
    code = "import os\nprint(os.getcwd())"
    result = execute_python_code(code, hide_items=['os'])
    assert "Error: Traceback" in result
    assert "ModuleNotFoundError" in result

def test_execute_python_code_eval_error():
    code = "import os\nos.foo()"
    result = execute_python_code(code)
    assert "Error: Traceback" in result
    assert "AttributeError: module 'os' has no attribute 'foo'" in result

@patch('lc_agent.code_atlas.codeinterpreter_tool.sys.modules', new_callable=dict)
def test_disable_items(mock_sys_modules):
    mock_module = MagicMock()
    mock_sys_modules['test_module'] = mock_module
    mock_sys_modules['test_module.func'] = lambda: None

    originals = disable_items(['test_module', 'test_module.func'])
    
    assert 'test_module' in originals
    assert 'test_module.func' not in originals
    assert mock_sys_modules['test_module'] is None
    assert 'test_module.func' in mock_sys_modules

@patch('lc_agent.code_atlas.codeinterpreter_tool.sys.modules', new_callable=dict)
def test_restore_items(mock_sys_modules):
    mock_module = MagicMock()
    original_func = lambda: None
    originals = {
        'test_module': mock_module,
        'test_module.func': original_func
    }

    restore_items(originals)

    assert mock_sys_modules['test_module'] is mock_module
    assert 'test_module.func' not in mock_sys_modules  # This line should remain unchanged

if __name__ == "__main__":
    pytest.main(["-v", "test_codeinterpreter_tool.py"])