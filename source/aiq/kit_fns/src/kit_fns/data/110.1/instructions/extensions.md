# Extension Development Guidelines and Patterns

## Extension Development Best Practices for Kit

This document covers comprehensive extension development patterns, architecture decisions, and best practices for creating robust Kit extensions.

## Extension Architecture Patterns

### Single Responsibility Extensions
Each extension should have a clear, focused purpose:
- UI extensions provide specific interface components
- Service extensions offer backend functionality
- Integration extensions bridge external systems

### Extension Dependencies
- Minimize dependencies to reduce coupling
- Use optional dependencies for enhanced features
- Respect dependency ordering and lifecycle

## Extension Configuration

### extension.toml Structure
```toml
[package]
name = "omni.my.extension"
version = "1.0.0"
title = "My Extension"
description = "Description of functionality"

[dependencies]
"omni.ui" = {}
"omni.kit.window" = {}

[settings]
persistent = true
```

### Configuration Management
- Use settings API for persistent configuration
- Provide reasonable defaults
- Validate configuration values
- Support runtime configuration changes

## Extension Lifecycle Management

### Proper Startup Sequence
```python
class MyExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        # 1. Initialize services first
        self._setup_services()
        # 2. Create UI components
        self._setup_ui()  
        # 3. Register callbacks and subscriptions
        self._register_callbacks()
```

### Clean Shutdown
```python
def on_shutdown(self):
    # 1. Unregister callbacks first
    self._unregister_callbacks()
    # 2. Clean up UI
    self._cleanup_ui()
    # 3. Shutdown services last
    self._cleanup_services()
```

## UI Extension Patterns

### Window Management
- Create windows in startup, show/hide as needed
- Implement proper window state management
- Handle window destruction gracefully

### Widget Organization
- Group related widgets in frames
- Use consistent layout patterns
- Implement responsive design principles

### Event Handling
- Use weak references for callbacks when possible
- Unregister all callbacks in shutdown
- Handle edge cases and exceptions

## Service Extensions

### Service Registration
- Register services early in startup
- Provide clear service interfaces
- Handle service dependencies properly

### API Design
- Design for extensibility
- Use dependency injection patterns
- Provide both sync and async APIs where appropriate

## Testing Patterns

### Unit Testing
```python
class TestMyExtension(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        # Setup test environment
        pass
        
    async def tearDown(self):
        # Clean up test environment
        pass
```

### Integration Testing
- Test extension loading/unloading
- Verify service interactions
- Test UI behavior and responsiveness

## Performance Considerations

### Lazy Loading
- Defer expensive operations until needed
- Load resources on-demand
- Cache frequently accessed data

### Memory Management
- Clean up resources in shutdown
- Use weak references for callbacks
- Monitor memory usage in development

### UI Performance
- Minimize UI updates
- Use efficient layout strategies
- Implement virtual scrolling for large datasets

## Extension Distribution

### Packaging
- Include all necessary files
- Provide clear documentation
- Version dependencies appropriately

### Registry Integration
- Follow naming conventions
- Provide rich metadata
- Include screenshots and examples

## Common Patterns

### Settings Management
```python
import carb.settings

def get_setting(path: str, default=None):
    settings = carb.settings.get_settings()
    return settings.get(path) or default
```

### Event Subscription
```python
import omni.kit.app

def subscribe_to_updates(self):
    self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
        self._on_update, name="my_extension_update"
    )
```

This comprehensive guide ensures robust, maintainable extension development.