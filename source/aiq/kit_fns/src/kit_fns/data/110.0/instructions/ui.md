# UI Development with Kit Widgets and Layouts

## Kit UI Development Comprehensive Guide

This document provides complete guidance for creating user interfaces in Kit applications using omni.ui framework, including widgets, layouts, styling, and advanced UI patterns.

## UI Framework Architecture

### Core Components
- **Windows**: Top-level containers for UI content
- **Layouts**: Organizational structures (VStack, HStack, ZStack, Grid)
- **Widgets**: Interactive elements (Button, Label, Field, Slider)
- **Styling**: CSS-like styling system for appearance
- **Models**: Data binding and state management

## Window Management

### Basic Window Creation
```python
import omni.ui as ui

# Simple window
window = ui.Window("My Window", width=400, height=300)

# Window with advanced options
window = ui.Window(
    "Advanced Window",
    width=500, 
    height=400,
    dockPreference=ui.DockPreference.DISABLED,
    flags=ui.WINDOW_FLAGS_NO_RESIZE
)
```

### Window Properties and Methods
```python
# Window state
window.visible = True
window.title = "Updated Title"

# Window positioning
window.position_x = 100
window.position_y = 200

# Window callbacks
window.set_visibility_changed_fn(lambda visible: print(f"Window visible: {visible}"))
```

## Layout Systems

### Vertical Stack (VStack)
```python
with ui.VStack(spacing=10):
    ui.Label("Header")
    ui.Button("Button 1")
    ui.Button("Button 2")
    ui.Label("Footer")
```

### Horizontal Stack (HStack)
```python
with ui.HStack():
    ui.Button("Left", width=100)
    ui.Spacer()  # Flexible space
    ui.Button("Center", width=100)
    ui.Spacer()
    ui.Button("Right", width=100)
```

### Z-Stack (Overlay)
```python
with ui.ZStack(height=100):
    ui.Rectangle(style={"background_color": 0xFF0000FF})
    ui.Label("Overlaid Text", alignment=ui.Alignment.CENTER)
    ui.Button("Top Button", width=80, height=30)
```

### Grid Layout
```python
with ui.VGrid(column_count=3, row_height=30):
    for i in range(9):
        ui.Button(f"Button {i+1}")
```

## Widgets and Controls

### Text and Labels
```python
# Basic label
label = ui.Label("Static Text")

# Dynamic label with model
text_model = ui.SimpleStringModel("Initial Text")
dynamic_label = ui.Label("", model=text_model)

# Update text
text_model.as_string = "Updated Text"
```

### Buttons and Interactions
```python
# Button with callback
def on_button_click():
    print("Button clicked!")

button = ui.Button("Click Me", clicked_fn=on_button_click)

# Button with lambda
button = ui.Button("Quick Action", clicked_fn=lambda: print("Quick!"))

# Button states
button.enabled = False  # Disable button
button.text = "Disabled Button"
```

### Input Fields and Forms
```python
# String input
string_model = ui.SimpleStringModel("default value")
string_field = ui.StringField(model=string_model)

# Numeric inputs
float_model = ui.SimpleFloatModel(1.0)
float_field = ui.FloatDrag(model=float_model, min=0.0, max=10.0)

int_model = ui.SimpleIntModel(5)
int_field = ui.IntSlider(model=int_model, min=0, max=100)
```

### Advanced Widgets
```python
# ComboBox
options = ["Option 1", "Option 2", "Option 3"]
combo_model = ui.SimpleIntModel(0)
combo_box = ui.ComboBox(combo_model, *options)

# TreeView for hierarchical data
tree_model = ui.SimpleTreeModel()
tree_view = ui.TreeView(tree_model, root_visible=False, header_visible=True)

# Plot widget for data visualization
plot_data = [1, 2, 3, 4, 5]
plot = ui.Plot(ui.Type.LINE, 0, 1, *plot_data)
```

## Styling System

### Basic Styling
```python
# Inline styling
button = ui.Button("Styled Button", style={
    "background_color": 0xFF0088FF,
    "font_size": 16,
    "border_radius": 5
})

# Style classes
button_style = {
    "Button": {
        "background_color": 0xFF0088FF,
        "font_size": 16
    }
}

with ui.Frame(style=button_style):
    ui.Button("Styled Button")
```

### Advanced Styling
```python
# Responsive styling with selectors
responsive_style = {
    "Button": {
        "background_color": 0xFF0088FF,
        "font_size": 16
    },
    "Button:hovered": {
        "background_color": 0xFF00AAFF
    },
    "Button:pressed": {
        "background_color": 0xFF0066FF
    }
}
```

## Model-View Patterns

### Data Models
```python
# Simple value models
bool_model = ui.SimpleBoolModel(True)
float_model = ui.SimpleFloatModel(3.14)
string_model = ui.SimpleStringModel("Hello")

# Custom models
class CustomModel(ui.AbstractValueModel):
    def __init__(self):
        super().__init__()
        self._value = 0
        
    def get_value_as_float(self):
        return self._value
        
    def set_value(self, value):
        if value != self._value:
            self._value = value
            self._value_changed()
```

### Model Callbacks and Subscriptions
```python
def on_value_changed(model):
    new_value = model.as_float
    print(f"Value changed to: {new_value}")

# Subscribe to model changes
subscription = float_model.subscribe_value_changed_fn(on_value_changed)

# Unsubscribe when done
subscription = None
```

## Advanced UI Patterns

### Custom Widgets
```python
class CustomSliderWidget:
    def __init__(self, label_text, min_val=0.0, max_val=1.0):
        self.model = ui.SimpleFloatModel(min_val)
        
        with ui.HStack():
            ui.Label(label_text, width=100)
            ui.FloatSlider(self.model, min=min_val, max=max_val)
            ui.FloatField(self.model, width=60)
            
    @property
    def value(self):
        return self.model.as_float
```

### Dialog Windows
```python
class ConfirmDialog:
    def __init__(self, message, callback):
        self.callback = callback
        self.window = ui.Window("Confirm", width=300, height=150)
        
        with self.window.frame:
            with ui.VStack():
                ui.Label(message)
                ui.Spacer(height=20)
                with ui.HStack():
                    ui.Button("OK", clicked_fn=self._on_ok)
                    ui.Button("Cancel", clicked_fn=self._on_cancel)
                    
    def _on_ok(self):
        self.callback(True)
        self.window.visible = False
        
    def _on_cancel(self):
        self.callback(False) 
        self.window.visible = False
```

## 3D UI with omni.ui.scene

### Scene View Integration
```python
import omni.ui.scene as scene

# Create scene view
scene_view = scene.SceneView()

with scene_view.scene:
    # 3D shapes
    scene.Line([0, 0, 0], [100, 100, 0], color=0xFF00FF00, thickness=2)
    scene.Rectangle(0, 0, width=50, height=50, color=0xFFFF0000)
    scene.Circle(25, 25, radius=20, color=0xFF0000FF)
    
    # Transform grouping
    with scene.Transform(look_at=[0, 0, 0]):
        scene.Arc(radius=30, color=0xFFFFFF00, thickness=3)
```

### Interactive 3D Elements
```python
# Clickable 3D shapes
def on_shape_click():
    print("3D shape clicked!")

clickable_rect = scene.Rectangle(
    0, 0, 
    width=100, 
    height=100,
    gesture=scene.ClickGesture(on_click=on_shape_click)
)

# Draggable manipulator
manipulator = scene.DefaultManipulator(
    model=transform_model,
    transform=scene.Transform()
)
```

## Event Handling and Callbacks

### Widget Events
```python
# Mouse events
def on_mouse_pressed():
    print("Mouse pressed")
    
def on_mouse_released():
    print("Mouse released")

widget.set_mouse_pressed_fn(on_mouse_pressed)
widget.set_mouse_released_fn(on_mouse_released)

# Keyboard events
def on_key_pressed(key, modifier, pressed):
    if pressed and key == carb.input.KeyboardInput.SPACE:
        print("Space key pressed")

window.set_key_pressed_fn(on_key_pressed)
```

### Drag and Drop
```python
# Drag source
def on_drag_start():
    return "dragged_data"

widget.set_drag_fn(on_drag_start)

# Drop target
def on_drop(data):
    print(f"Dropped: {data}")
    return True

widget.set_drop_fn(on_drop)
```

## Performance and Best Practices

### UI Updates and Refresh
```python
# Batch UI updates
with ui.Frame():
    # Multiple UI changes here
    # Will be batched for efficiency
    pass

# Manual refresh control
ui.Application.get().get_main_dpi_window().refresh()
```

### Memory Management
```python
# Proper cleanup
def cleanup_ui():
    if hasattr(self, '_subscriptions'):
        for sub in self._subscriptions:
            sub.unsubscribe()
        self._subscriptions.clear()
        
    if hasattr(self, '_window'):
        self._window.destroy()
        self._window = None
```

This comprehensive UI guide enables creation of sophisticated, responsive user interfaces in Kit applications.