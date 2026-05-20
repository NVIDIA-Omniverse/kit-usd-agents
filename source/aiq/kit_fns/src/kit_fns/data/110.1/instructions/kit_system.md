# Kit System Instructions

## Core Kit Framework Fundamentals and Architecture

This document provides comprehensive guidance for developing with NVIDIA Omniverse Kit framework.

## Kit Architecture Overview

Kit is built on a modular extension system that allows developers to:

- Create reusable, composable extensions
- Integrate with USD (Universal Scene Description) 
- Build rich UI applications with omni.ui
- Develop 3D applications with scene management
- Leverage Carbonite (carb) services for low-level functionality

## Extension System

### Extension Structure
```
my_extension/
├── config/
│   └── extension.toml
├── omni/
│   └── my_extension/
│       ├── __init__.py
│       ├── extension.py
│       └── scripts/
└── docs/
```

### Extension Lifecycle
1. **Startup**: `on_startup(ext_id)` called when extension loads
2. **Runtime**: Extension provides services and UI
3. **Shutdown**: `on_shutdown()` called when extension unloads

### Key Extension APIs
- `omni.ext.IExt`: Base interface for all extensions
- Extension manager for loading/unloading extensions
- Dependency management through extension.toml

## USD Integration

Kit applications are built around USD (Universal Scene Description):

### Core USD Operations
- Stage management and context handling
- Prim creation and manipulation
- Layer composition and editing
- Attribute and relationship management

### USD Context API
```python
import omni.usd
context = omni.usd.get_context()
stage = context.get_stage()
```

## UI Development with omni.ui

### Widget Hierarchy
- **Windows**: Top-level containers
- **Layouts**: VStack, HStack, ZStack for organization
- **Widgets**: Button, Label, Field, Slider for interaction
- **Styling**: CSS-like styling system

### Event Handling
- Callback functions for user interactions
- Subscription system for data changes
- Model-View patterns for data binding

## 3D Scene Development

### Viewport Integration
- Multiple viewport support
- Camera control and manipulation
- Selection and interaction handling

### 3D UI with omni.ui.scene
- 3D shapes and primitives
- Transform manipulation
- Scene-based user interfaces

## Services and APIs

### Carbonite Services
- File I/O and asset management
- Settings and configuration
- Logging and profiling
- Plugin system

### Application Services
- Window management
- Menu and action systems
- Tool and mode management
- Event broadcasting

## Best Practices

### Performance
- Lazy loading of heavy resources
- Efficient USD operations
- UI update optimization
- Memory management

### Code Organization
- Clear separation of UI and business logic
- Proper extension dependencies
- Consistent naming conventions
- Documentation and testing

### User Experience
- Responsive UI design
- Consistent interaction patterns
- Error handling and user feedback
- Accessibility considerations

## Development Workflow

1. **Setup**: Install Kit SDK and tools
2. **Create**: Generate extension structure
3. **Develop**: Implement functionality
4. **Test**: Unit and integration testing
5. **Package**: Build and distribute extension
6. **Deploy**: Install in Kit applications

This foundational knowledge enables effective Kit application development.