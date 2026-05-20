# Test Writing Best Practices and Framework Usage

## Kit Testing Framework Comprehensive Guide

This document provides complete guidance for writing effective tests in Kit applications using the built-in testing frameworks.

## Testing Architecture

### Test Categories
1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **UI Tests**: Test user interface behavior
4. **Performance Tests**: Measure and validate performance
5. **End-to-End Tests**: Test complete user workflows

## Kit Test Framework

### Base Test Classes
```python
import omni.kit.test

# Synchronous tests
class TestMyComponent(omni.kit.test.TestCase):
    def test_basic_functionality(self):
        # Synchronous test code
        pass

# Asynchronous tests (recommended)
class TestAsyncComponent(omni.kit.test.AsyncTestCase):
    async def test_async_functionality(self):
        # Asynchronous test code
        pass
```

### Test Lifecycle
```python
async def setUp(self):
    """Called before each test method."""
    # Initialize test environment
    pass
    
async def tearDown(self):
    """Called after each test method."""
    # Clean up test environment
    pass
```

## UI Testing Patterns

### Window and Widget Testing
```python
import omni.ui as ui

class TestUIComponents(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.window = ui.Window("Test Window", width=300, height=200)
        
    async def tearDown(self):
        if self.window:
            self.window.destroy()
            self.window = None
            
    async def test_button_interaction(self):
        clicked = False
        
        def on_click():
            nonlocal clicked
            clicked = True
            
        with self.window.frame:
            button = ui.Button("Test Button", clicked_fn=on_click)
            
        # Simulate button click
        button.call_clicked_fn()
        self.assertTrue(clicked)
```

### Layout Testing
```python
async def test_layout_structure(self):
    with self.window.frame:
        with ui.VStack() as vstack:
            ui.Label("Label 1")
            ui.Label("Label 2")
            
    # Verify layout structure
    self.assertEqual(len(vstack.children), 2)
    self.assertIsInstance(vstack.children[0], ui.Label)
```

## USD Testing Patterns

### Stage Testing
```python
import omni.usd
from pxr import Usd, UsdGeom

class TestUSDOperations(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # Create new stage for each test
        await omni.usd.get_context().new_stage_async()
        
    async def test_prim_creation(self):
        context = omni.usd.get_context()
        stage = context.get_stage()
        
        # Create a cube prim
        cube_path = "/World/TestCube"
        cube_prim = UsdGeom.Cube.Define(stage, cube_path)
        
        # Validate prim creation
        self.assertTrue(stage.GetPrimAtPath(cube_path).IsValid())
        self.assertEqual(cube_prim.GetPrim().GetTypeName(), "Cube")
        
    async def test_attribute_operations(self):
        context = omni.usd.get_context()
        stage = context.get_stage()
        
        # Create sphere with custom size
        sphere_path = "/World/TestSphere"
        sphere_prim = UsdGeom.Sphere.Define(stage, sphere_path)
        
        # Set and test radius
        radius_attr = sphere_prim.CreateRadiusAttr()
        radius_attr.Set(5.0)
        
        self.assertEqual(radius_attr.Get(), 5.0)
```

## Extension Testing

### Extension Lifecycle Testing
```python
import omni.kit.app

class TestExtensionLifecycle(omni.kit.test.AsyncTestCase):
    async def test_extension_enable_disable(self):
        ext_manager = omni.kit.app.get_app().get_extension_manager()
        ext_id = "omni.my.test.extension"
        
        # Test enabling
        enabled = ext_manager.set_extension_enabled_immediate(ext_id, True)
        self.assertTrue(enabled)
        self.assertTrue(ext_manager.is_extension_enabled(ext_id))
        
        # Test disabling
        disabled = ext_manager.set_extension_enabled_immediate(ext_id, False)
        self.assertTrue(disabled)
        self.assertFalse(ext_manager.is_extension_enabled(ext_id))
```

## Mocking and Test Doubles

### Service Mocking
```python
from unittest.mock import Mock, patch

class TestWithMocks(omni.kit.test.AsyncTestCase):
    @patch('omni.kit.app.get_app')
    async def test_with_mocked_app(self, mock_get_app):
        # Setup mock
        mock_app = Mock()
        mock_get_app.return_value = mock_app
        
        # Test code that uses the app
        # Assertions
        mock_get_app.assert_called_once()
```

## Performance Testing

### Timing Tests
```python
import time

async def test_performance_requirement(self):
    start_time = time.perf_counter()
    
    # Code to test
    result = await expensive_operation()
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    # Assert performance requirement
    self.assertLess(execution_time, 1.0)  # Must complete within 1 second
```

## Test Data Management

### Temporary File Handling
```python
import tempfile
import os

async def test_file_operations(self):
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test.usd")
        
        # Create test file
        stage = Usd.Stage.CreateNew(test_file)
        
        # Test operations
        self.assertTrue(os.path.exists(test_file))
        
        # Cleanup automatic with context manager
```

## Test Organization

### Test Discovery
```python
# File naming: test_*.py or *_test.py
# Class naming: Test*
# Method naming: test_*
```

### Test Categories with Decorators
```python
import omni.kit.test

@omni.kit.test.slow_test
async def test_slow_operation(self):
    # Test that takes significant time
    pass

@omni.kit.test.skip("Not implemented yet")
async def test_future_feature(self):
    # Test for future functionality
    pass
```

## Debugging Tests

### Test Output and Logging
```python
import carb

async def test_with_logging(self):
    carb.log_info("Test starting")
    
    # Test logic
    result = some_operation()
    
    carb.log_info(f"Test result: {result}")
    self.assertIsNotNone(result)
```

## Continuous Integration

### Test Configuration
- Configure test runners
- Set up environment variables  
- Manage test dependencies
- Handle test artifacts

This comprehensive testing guide ensures robust, reliable Kit applications.