# OMNIVERSE KIT UI STYLE BEST PRACTICE

## Overview

This guide outlines best practices for implementing centralized styling in omni.ui, providing examples for building Kit-based applications with custom themes from the start and serving as a reference for future development.

**Why Centralized Styling Matters:**
Centralized styling using `ui.style.default` ensures consistent UI design across your application. By managing styles in one location, you gain:
- **consistency** - new UI elements automatically use your defined colors and spacing;
- **maintainability** - updating a value in one place changes it everywhere;
- **clarity** - style definitions stay separate from business logic;
- **flexibility** - switching themes or color schemes becomes straightforward.
This pattern scales well as your application grows and makes it easier for teams to collaborate on UI development.

In this guideline, you will learn about the styling best practices, as well as the best practices to build window and widgets:

**Styling Best Practices**
- ✅ **Use global styling with ui.style.default for application-wide styles**
- ✅ **Each extension should have its own style.py for extension-specific styles**
- ✅ **Use styling constants over magic numbers**
- ✅ **Use single centralized STYLES dictionary per extension and establish clear styles for different selectors**
- ✅ **Use name and style_type_name_override attribute instead of inline style parameter**

**ui.Window Best Practices**
- ✅ Always derive from ui.Window when creating new window objects
- ✅ **Use **kwargs** to allow users to pass styling and configuration parameters
- ✅ **Use frame.set_build_fn() or ui.Frame(build_fn=self.build) to build window content** - enables lazy construction
- ✅ Use frame's style or frame.set_style() to apply window-specific styling
- ✅ Create the window in its own dedicated file

**Widget Development Best Practices**
- ✅ **For reusable widgets, encapsulate in classes** - better state management and APIs
- ✅ **Use **kwargs** to allow users to pass parameters to the underlying Frame
- ✅ **Use frame.set_build_fn() or ui.Frame(build_fn=self.build)** pattern for custom widgets
---

## Table of Contents

1. [Best Practice Criteria with Examples](#best-practice-criteria-with-examples)
    - [Styling Best Practices](#1-styling-best-practices)
    - [ui.Window Best Practices](#2-uiwindow-best-practices)
    - [Widget Development Best Practices](#3-widget-development-best-practices)
2. [Quick Reference for Developers](#quick-reference-for-developers)
3. [Current Limitation](#current-limitations)


---


## Best Practice with Examples

### 1. Styling Best Practices

**Criterion:** Use centralized styling with proper organization - global default styles for the application and extension-specific styles for individual extensions.

#### 1.1 Two Levels of Styling

1. **Global Application Styles** - Set once for the entire app via `ui.style.default`
   use
   ```python
   import omni.ui as ui

   style = ui.style.default
   style.update(STYLES)
   ui.style.default = style
   ```
2. **Extension-Specific Styles**
   - if not defined globally, organize styles in a separate style.py file within each extension
   - Each extension has its own `style.py` for extension-specific styling

**Why Two Levels:**
- ✅ **Global styles** ensure consistency across the entire application
- ✅ **Extension-specific styles** allow customization without polluting global namespace
- ✅ Not everything needs to be in the global default - extensions define their own unique styling
- ✅ Clear separation between app-level and extension-level styling

**In your .kit file:**
```toml
[dependencies]
"app.style" = {}  # Load FIRST - sets global styles for the entire app
"my.extension" = {}  # Extension with its own style.py for extension-specific styles
```

**Best Practice for Global Application Styles:**
```python
# app_style_extension/styles.py
from pathlib import Path
from omni.ui import color as cl
from omni.ui import constant as fl
from omni.ui import url
import omni.ui as ui
import carb

# Define global constants
cl.app_highlight = cl("0285ED")
cl.app_background = cl("1a1a1a")
fl.border_radius = 5
url.my_icon = f"${my.extension}/icons/icon.svg"

# Global STYLES dictionary - for application-wide styling
STYLES = {
   "Window": {"background_color": cl.app_background},
   "Button": {"border_radius": fl.border_radius},
   "Button:hovered": {"color": cl.app_highlight},
   "Image::icon": {"image_url": url.my_icon},
}

# Apply as global default - this controls the ENTIRE app!
style = ui.style.default
style.update(STYLES)
ui.style.default = style
```


**Best Practice for Extension-Specific Styles:**
```python
# my_extension/style.py
from omni.ui import color as cl
from omni.ui import constant as fl
import omni.ui as ui

# Extension-specific constants (not in global default)
cl.my_ext_primary = cl("FF5722")
fl.my_ext_spacing = 12

# Extension-specific STYLES - only for this extension
STYLES = {
   "Window::my_extension": {"background_color": cl.my_ext_primary},
   "Button::my_action": {"background_color": cl.my_ext_primary},
   "Label::my_header": {"font_size": 18, "color": cl.my_ext_primary},
}

# In your window.py
class MyWindow(ui.Window):
   def __init__(self):
       super().__init__("My Window")
       # Apply extension-specific styles
       self.frame.set_style(STYLES)
       self.frame.set_build_fn(self._build_ui)

   def _build_ui(self):
       ...
```
----

#### 1.2 Use styling constants
Use constants defined from omni.ui instead of magic numbers
```python
from omni.ui import color as cl     # for colors
from omni.ui import constant as fl  # for float numbers, e.g. width, height etc
from omni.ui import url             # for image url
```

**❌ WRONG:**
```python
import omni.ui as ui

# Anti-pattern: Magic numbers everywhere
ui.Label("Text", style={"color": 0xFFC0C0C0, "font_size": 14})
ui.Rectangle(style={"background_color": 0xFF343432, "border_radius": 5})
```

**✅ CORRECT:**
```python
from omni.ui import color as cl
from omni.ui import constant as fl

# Define constants with meaningful names
cl.my_text_color = cl("C0C0C0")
cl.my_container_bg = cl("343432")
fl.my_font_size_medium = 14
fl.my_border_radius = 5

STYLES = {
   "Label": {
       "color": cl.my_text_color,
       "font_size": fl.my_font_size_medium
   },
   "Rectangle::container": {
       "background_color": cl.my_container_bg,
       "border_radius": fl.my_border_radius
   }
}
```

**Why This Is Wrong:**
- No semantic meaning for values
- Impossible to maintain consistency
- Can't implement theming or branding changes
- Hard to understand what colors represent


**⚠️ IMPORTANT: These are Global Singletons**

The `omni.ui.color`, `omni.ui.constant`, and `omni.ui.url` modules are **singletons** shared across the entire application. This means:
- All extensions share the same `omni.ui.color`, `omni.ui.constant`, and `omni.ui.url` singleton instances
- Any attribute you set (e.g., `cl.my_color = cl("FF0000")`) is visible to **all extensions**
- Name collisions between extensions will cause conflicts and overwrites
- **Use unique, extension-specific prefixes** to avoid clashes

**Best Practice for Naming:**
```python
# ✅ GOOD - Extension-specific prefixes prevent conflicts
# These set attributes on the global omni.ui.color/constant/url singletons
cl.myext_primary_color = cl("FF5722")
cl.myext_text_color = cl("C0C0C0")
fl.myext_spacing_large = 16
url.myext_icon_add = f"{ICON_PATH}/add.svg"

# ❌ BAD - Generic names that could conflict with other extensions
cl.primary = cl("FF5722")        # Another extension might overwrite this!
cl.text_color = cl("C0C0C0")     # Very likely to conflict
fl.spacing = 16                   # Too generic - will be overwritten
url.icon = f"{ICON_PATH}/add.svg" # Will definitely conflict
```

1.2.1 Color Constants

There are several ways of representing colors in the styling system, such as `0xFF23211F` and `cl("1F2123")`. We recommend using `cl("1F2123")` or `cl("CCCCCCCC")` (if Alpha channel is involved) for better readability and consistency.

Note that `0xFF23211F` uses ABGR format (Alpha-Blue-Green-Red), where the red and blue color channels are reversed. When you use `cl("1F2123")`, it follows the more intuitive ARGB format (Alpha-Red-Green-Blue), making it easier to work with standard hex color representations.

1.2.2 Float Constants

Float constants in the styling system are not actual numeric values—they are special objects designed for the style system. While they work perfectly within style dictionaries, they cannot be used in mathematical calculations or with functions that expect real numbers.

**✅ WORKS - Regular Python constants:**
```python
IMAGE_SIZE = 16
ui.Button("", image_width=IMAGE_SIZE, image_height=IMAGE_SIZE)

FONT_SIZE = 16
height = ui.Pixel(3 * FONT_SIZE)  # Works fine
```

**❌ DOES NOT WORK - Float constants in calculations:**
```python
from omni.ui import constant as fl
fl.image_size = 16.0
ui.Button("", image_width=fl.image_size, image_height=fl.image_size) # Will fail!

fl.font_size = 16.0
height = ui.Pixel(3 * fl.font_size)  # Will fail!
```

When you try to use float constants in calculations, you'll get an error because the system cannot convert the style constant object into the actual numeric value needed for the math operation.

**Key takeaway:** If you need to perform calculations with values (e.g., `ui.Pixel(3 * FONT_SIZE)`), use regular Python numeric constants instead of float constants.

1.2.3 URL Constants

When referencing resources like icons or images in your styles, use the extension path syntax instead of manually constructing file paths. This approach is more maintainable and resilient to directory structure changes.

**✅ RECOMMENDED:**
```python
"image_url": "${omni.kit.markup.core}/icons/Close.svg"
```

**❌ NOT RECOMMENDED:**
```python
ICON_PATH = Path(__file__).parent.parent.parent.parent.parent.joinpath("icons")
"image_url": f"{ICON_PATH}/Close.svg"
```

----

#### 1.3 No Scattered styles
Use a single centralized STYLES dictionary per extension and establish clear styles for different selectors. We keep all styles in one place and no scattered Style Variables

**❌ WRONG:**
```python
import omni.ui as ui

# Anti-pattern: Multiple scattered style dictionaries
objects_container = {
   "Rectangle":{
       "background_color": editable_text_field_color,
       "border_radius": 5,
   }
}

object_button_style = {
   "Button":{
       "background_color": button_color,
       "hovered_background_color": button_color,
   },
   "Button::attach_object:hovered": {
       "background_color": cl.button_color,
   },
}

# Later in code
with ui.ZStack():
   ui.Rectangle(style=objects_container)
   self.attach_object_button = ui.Button(
       "Attach 3D Object",
       image_url=f"{EXTENSION_FOLDER_PATH}/data/image_add.svg",
       style=object_button_style
   )
```

**✅ CORRECT:**
```python
# In centralized styles.py
from omni.ui import color as cl
from omni.ui import constant as fl
from omni.ui import url

cl.myext_editable_text_field = cl("343432")
cl.myext_button = cl("0285ED")
fl.myext_container_border_radius = 5

url.myext_image_add = f"{ICON_PATH}/image_add.svg"

STYLES = {
   "Rectangle::objects_container": {
       "background_color": cl.myext_editable_text_field,
       "border_radius": fl.myext_container_border_radius,
   },
   "Button::attach_object": {
       "background_color": cl.myext_button,
       "image_url": url.myext_image_add,
   },
   "Button::attach_object:hovered": {
       "background_color": cl.myext_button,
   },
}

# In code.py - clean and simple
frame.set_style(STYLES)
with frame:
   with ui.ZStack():
       ui.Rectangle(name="objects_container")
       self.attach_object_button = ui.Button(
           "Attach 3D Object",
           name="attach_object"
       )
```

**Why This Is Wrong:**
- Styles scattered across multiple variables
- Hard to maintain consistency
- Difficult to implement themes or dark/light mode
- Style logic mixed with widget creation logic

----

#### 1.4 Use Hierarchical Selector Pattern (Type::Name:State)

In omni.ui, styles are defined with three types of selectors: type Selector, name Selector and state Selector. They are structured as:


Type Selector :: Name Selector : State Selector


e.g.,Button::okButton:hovered


where `Button` is the type selector, which gets the default Button styles, `okButton` is the name selector, whose style overrides the default `Button` type's style. `hovered` is the sate selector. 


## Example

```python
style = {
    # Type selector - applies to all Buttons
    "Button": {"border_width": 0.5, "margin": 5.0},

    # Type + Name selector - specific button instances
    "Button::primary": {
        "background_color": cl("#097eff"),
        "border_color": cl("#1d76fd"),
    },
    "Button::secondary": {
        "background_color": cl.white,
        "border_color": cl("#B1B1B1")
    },

    # Type + Name + State selector - interactive states
    "Button::primary:hovered": {
        "background_color": cl("#006eff")
    },
    "Button::primary:pressed": {
        "background_color": cl("#6db2fa")
    },
}

with ui.HStack(style=style):
    ui.Button("Save", name="primary")
    ui.Button("Cancel", name="secondary")

```


#### 1.5 No inline styles - use `name` and `style_type_name_override`

**`name`** of a widget is used to select an optional named variant that overrides only some values while inheriting the rest from the base style.

What if the user has a customized widget which is not a standard omni.ui one, or I just simply have too many `okButton` (taken previous example of `Button::okButton:hovered`) which I want them to have different styles. How to define that Type Selector?


**`style_type_name_override`** allows users to give more meaningful type names for example "Rectangle.DragAndDrop" instead of "Rectangle", which brings much more ease for debugging and the cleanliness to the lengthy style dictionary.


This enables a common pattern:


- **`style_type_name_override` selects the base style “type”** to use for a widget
- **`name` selects an optional named variant** (a more specific style key) that defines *variant styles* that override only specific properties.
- Use `style_type_name_override` to pick the base, and `name` to apply a variant on top.


## Example


```python
STYLES = {
"RedRect": {"background_color": cl.red},
"RedRect::round_border": {"border_radius": fl.border_radius},
}


class MyWindow(ui.Window):
   def __init__(self):
       super().__init__("My Window")
       self.frame.set_style(STYLES)
       self.frame.set_build_fn(self._build_ui)

   def _build_ui(self):
       with ui.HStack():
           ui.Rectangle(style_type_name_override="RedRect")
           ui.Rectangle(style_type_name_override="RedRect", name="round_border")
```

What this does
- `ui.Rectangle(style_type_name_override="RedRect")`
Uses the style entry keyed by `"RedRect"` → sets `background_color` to red.
- `ui.Rectangle(style_type_name_override="RedRect", name="round_border")`
Uses `"RedRect"` as the base style *and* applies the named variant `"round_border"` on top → it keeps the red background from `"RedRect"` and additionally applies `border_radius`.

**Note** that `::` or `:` is not allowed in the attribute of `style_type_name_override`, we can define `style_type_name_override=Button.Playbar`, but not `style_type_name_override=Button::Playbar`. If a style is defined like `Button::Playbar`, it is really a button without style_type_name_override, but with a name `Player`.

## Example

```python
import omni.ui as ui
from omni.ui import color as cl     # for colors

STYLES = {
"Button.Playbar": {"background_color": cl.red},
"Button::Playbar": {"background_color":cl.green},
}

class MyWindow(ui.Window):
   def __init__(self):
       super().__init__("My Window", width=800, height=600)
       self.frame.set_style(STYLES)
       self.frame.set_build_fn(self._build_ui)


   def _build_ui(self):
       with self.frame:
           with ui.HStack():
                ui.Button("Green button", name="Playbar")  # green button
                ui.Button("Red button", style_type_name_override="Button.Playbar") # red button

```

We should use `name` and `style_type_name_override` attribute instead of inline style parameters.

**❌ WRONG:**
```python
import omni.ui as ui
from omni.ui import color as cl

str_field = ui.StringField(model, multiline=False, style={"color": cl.green})
str_field = ui.StringField(model, multiline=False, style={"color": cl.red})
str_field = ui.StringField(model, multiline=True, style={"color": cl.darkgreen})
```

**✅ CORRECT:**
```python
import omni.ui as ui
from omni.ui import color as cl
# Define in centralized styles.py
STYLES = {
   "StringField::green_field": {
       "color": cl.green,
   },
   "StringField::red_field": {
       "color": cl.red,
   },
   "StringField.Multiline::green_field": {
       "color": cl.darkgreen,
   },
}

# Use in code
str_field = ui.StringField(model, multiline=False, name="green_field")
str_field = ui.StringField(model, multiline=False, name="red_field")
str_field = ui.StringField(model, multiline=False, name="green_field", style_type_name_override="StringField.Multiline")
```

**Why This Is Wrong:**
- **Entangled with business logic, hard to debug** - Style logic should live in style.py, not business logic
- **Duplication and inconsistency** - Same color values repeated multiple times, easy to miss updates
- **No semantic meaning** - `cl.green` doesn't explain what the field represents (status? category?)
- **Hard to maintain** - Style definitions scattered throughout functional code, Changing green fields requires finding and updating every instance
- **Can't reuse styles** - Each widget redeclares the same styling
- **Breaks theming** - Can't easily switch color schemes or implement dark/light modes
----

#### 1.6 Don't miss type selector
We always want the type selector defined e.g. "Button" or "Label" for the widget. Otherwise, the style attributes behavior could be undefined.

**❌ WRONG:**
```python
ui.Label("Example 1", style={
"color" : ui.color.yellow,
"Tooltip": {"color": ui.color.black}
}, tooltip="This is a tooltip")
```

```python
objects_container = {
       "background_color": editable_text_field_color,   # no type selector
       "border_radius": 5,
   }

with ui.VStack():
   with ui.ZStack():
       ui.Rectangle(style=objects_container)
```


**✅ CORRECT:**
```python
STYLES = {
   "Label": {
       "color" : ui.color.yellow
   },
   "Tooltip":{
       "color": ui.color.black
   },
   ...
}

ui.Label("Example 1", style=STYLES, tooltip="This is a tooltip")
```

```python
STYLES = {
   "Rectangle::objects_container": {
       "background_color": editable_text_field_color,   # no type selector
       "border_radius": 5,
   },
   ...
}

with ui.VStack(style=STYLES):
   with ui.ZStack():
       ui.Rectangle(name="objects_container")
```
---

### 2. ui.Window Best Practices

**Criterion:**
- Derive from `ui.Window` and use `frame.set_build_fn()` for lazy UI construction.
- Use kwargs for ui.Window to allow customization

**Best Practice:**
```python
# window.py
import omni.ui as ui

class MyWindow(ui.Window):
   WINDOW_TITLE = "My Window"

   def __init__(self, width=800, height=800, **kwargs):
       # Pass **kwargs to allow users to customize window parameters
       # (width, height, flags, dockPreference, etc.)
       super().__init__(self.WINDOW_TITLE, width=width, height=height, **kwargs)

       # Set build function for lazy construction
       self.frame.set_build_fn(self._build_ui)

   def _build_ui(self):
       """Build window content - called when window becomes visible"""
       with self.frame:
           with ui.VStack(spacing=15):
               self._build_header()
               self._build_content()

   def _build_header(self):
       """Separate method for logical sections"""
       ...


   def _build_content(self):
       """Main content area"""
       ...
```

**Usage:**
```python
# Users can pass any ui.Window parameters
window = MyWindow()  # Uses defaults
window = MyWindow(width=1200, height=900)  # Custom size
window = MyWindow(flags=ui.WINDOW_FLAGS_NO_RESIZE)  # Custom flags
window = MyWindow(dockPreference=ui.DockPreference.LEFT_BOTTOM)  # Custom dock
```

**Why This Pattern:**
- ✅ **Flexible** - Users can customize window parameters without modifying the class
- ✅ Lazy construction - UI only built when window is visible
- ✅ Proper cleanup - build_fn handles destruction
- ✅ Clean separation of build logic

---

###  3. Widget Development Best Practices

**Criterion:**
- For reusable custom widgets, encapsulate them in classes using the `ui.Frame(build_fn=self.build)` pattern. Classes provide better state management, lifecycle control, and reusability.
- Use kwargs for ui.Frame to allow customization

**Recommendation:** While functions can work for simple cases, classes are recommended for:
- Widgets that need to maintain state
- Widgets that need callbacks or public APIs
- Widgets that will be reused across multiple places
- Complex widgets with multiple internal components

**Best Practice (Recommended for reusable widgets):**
```python
# custom_widget.py
import omni.ui as ui

class CustomComboBox:
   """Reusable combo box widget with custom styling"""

   def __init__(self, label="", items=None, **kwargs):
       self._label = label
       self._items = items or []
       self._combo_box = None

       # Pass **kwargs to Frame - allows users to control Frame parameters
       # (visible, width, height, style, etc.)
       self.frame = ui.Frame(build_fn=self.build, **kwargs)

   def build(self):
       """Build the widget UI"""
       with ui.HStack():
           if self._label:
               ui.Label(self._label)
           self._combo_box = ui.ComboBox(
               0, *self._items,
               style_type_name_override="CustomComboBox"
           )

   def get_current_value(self):
       """Public API - can't do this with functions"""
       return self._combo_box.model.get_item_value_model().as_int
```

**Usage:**
```python
# Users can pass Frame parameters via **kwargs
widget = CustomComboBox("Select:", ["A", "B", "C"])  # Default
widget = CustomComboBox("Select:", ["A", "B", "C"], width=200)  # Custom width
widget = CustomComboBox("Select:", ["A", "B", "C"], visible=False)  # Hidden initially
widget = CustomComboBox("Select:", ["A", "B", "C"], height=50, width=300)  # Custom size
```

**Why Classes Are Better:**
- ✅ **Flexible** - **kwargs allows users to customize Frame parameters
- ✅ State management - store widget references, data
- ✅ Public API - expose methods for interaction
- ✅ Lifecycle control - build_fn for lazy construction
- ✅ Easier to test and maintain
- ✅ Can add callbacks and event handlers

**Acceptable for simple cases:**
```python
# Simple helper function is OK for one-off UI patterns
def build_header(title):
   """Simple helper - no state needed"""
   with ui.HStack(height=30):
       ui.Label(title, style_type_name_override="Label.Header")
```

---

## Quick Reference for Developers

### Starting a New Application?

**Set up centralized styling**
Create an extension for your global application style, it could contain different themes. e.g. "my_app.style"

**In your .kit file:**
```toml
[dependencies]
"my_app.style" = {}  # Load this FIRST - it sets global styles for everything!
```

**The Magic:** Once loaded, `ui.style.default` controls styling for your ENTIRE application. Every window and widget automatically uses these styles.


### Starting a New Window Extension?
**Step 1: Create window**
```python
# my_extension/window.py
import omni.ui as ui

class MyWindow(ui.Window):
   def __init__(self):
       super().__init__("My Window", width=800, height=600)
       self.frame.set_build_fn(self._build_ui)


   def _build_ui(self):
       with self.frame:
           with ui.VStack():
               ui.Label("Hello", style_type_name_override="Label.Header")
```

**Step 2: Create custom widgets**
```python
# my_extension/widgets/custom_widget.py
import omni.ui as ui

class CustomWidget:
   def __init__(self, label="", **kwargs):
       self._label = label
       self.frame = ui.Frame(build_fn=self.build, **kwargs)

   def build(self):
       with ui.VStack():
           ui.Label(self._label)
           ui.Button("Click Me", style_type_name_override="Button.primary")
```

**Step 3: Customize the window style**
Create a style.py module for the extension specific styles. And apply the STYLES dict to the window's frame.

### Checklist Before Committing

- All styles in the centralized STYLES dictionary? globally and extension wise
   - Applied styles with `ui.style.default = style`?
   - Included this global style extension in your kit file?
- Using style.py for all the style definition for extensions
- In style.py, we have one STYLES dictionary containing all the styles.
- Using `color`, `float`, `url` constants instead of magic numbers?
- No inline style dictionaries (`style={...}`)?
- Using `name` and `style_type_name_override` instead of inline styles?
- Windows use `frame.set_build_fn()` pattern?
- Reusable custom widgets use class encapsulation with Frame?

---

## Current limitations
In kit 109, not all extensions comply with this best practice yet. We are working on updating our extensions to comply with these guidelines. If you have extensions which are forked from Kit with local modification, here is the “how-to” guide for extensions that are not yet following the best practices.

### Workarounds for Extensions Not Yet Following Best Practices
While migrating to centralized styling, you may encounter extensions that haven't adopted the pattern yet. Here are practical workarounds:

#### 1. Override Window Background with frame.set_style()
When an extension's window doesn't respect the global default styles, override it directly on the window's frame:


```python
import omni.ui as ui
from omni.ui import color as cl

STYLES = {
   "Window": {"background_color": cl.black},
   ...
}

# In your window __init__ or after super().__init__
class MyWindow(ui.Window):
   def __init__(self):
       super().__init__("My Window")

       # Override window-specific styles that aren't in global default
       self.frame.set_style(STYLES)

       self.frame.set_build_fn(self._build_ui)
```

#### 2. Add Missing Styles to Extension-Specific style.py
When widgets in third-party extensions don't have proper styles defined, create your own extension-specific style dictionary:

```python
# my_extension/style.py
from omni.ui import color as cl
from omni.ui import constant as fl

# Define your color constants
cl.my_black = cl.black
cl.my_border_default = cl("C1C1C1")
cl.my_border_inactive = cl("BBBBBB")
cl.my_primary = cl("90035F")

fl.border_radius_small = 3
fl.border_radius_medium = 4
fl.border_radius_large = 6
fl.border_width = 1

# Define missing styles for widgets used in your extension
MY_EXTENSION_STYLES = {
   # Override third-party widget styles
   "TreeView.ScrollingFrame": {"background_color": cl.my_black},
   "SearchField.Frame": {
       "background_color": cl.my_black,
       "border_radius": fl.border_radius_small,
       "border_color": cl.my_border_default,
       "border_width": fl.border_width,
   },
   "RadioButton": {
       "border_radius": fl.border_radius_large,
       "background_color": cl.transparent,
       "border_width": fl.border_width,
       "border_color": cl.my_border_inactive,
   },
   "RadioButton:checked": {
       "background_color": cl.my_primary,
       "border_color": cl.transparent,
   },
   "ComboBox::renderer_choice": {
       "background_color": cl.my_black,
       "border_radius": fl.border_radius_medium,
       "border_width": fl.border_width,
       "border_color": cl.my_border_inactive,
   },
}

# Apply in your window
self.frame.set_style(MY_EXTENSION_STYLES)
```

#### 3. Replace or Update Icons
If extensions use old icon assets that don't match your design system, provide updated SVG icons:

```python
# Update icon paths in your style dictionary
STYLES = {
   "Image::plus_icon": {"image_url": f"{YOUR_ICON_PATH}/Plus.svg"},
   "Image::minus_icon": {"image_url": f"{YOUR_ICON_PATH}/Minus.svg"},
}
```

#### 4. Adjust Widget Sizes When Needed
Some widgets may need size adjustments to work with your custom styles:

```python
# If default widget sizes don't work with your styling
self._option_button = OptionsButton(option_items, width=25, height=25)  # was 20x20
```

#### 5. Define Custom Style Constants
For consistent theming across workarounds, define your own style constants instead of using inline style assignments:

```python
# style.py
from omni.ui import color as cl

# Define your app's color palette
cl.app_background = cl("000000")
cl.app_border_default = cl("C1C1C1")
cl.app_border_active = cl("FFFFFF")
# Use consistently in all style overrides
STYLES = {
   "Window": {"background_color": cl.app_background},
   "SearchField.Frame": {"border_color": cl.app_border_default},
   "SearchField.Frame:selected": {"border_color": cl.app_border_active},
}
```

#### 6. Update widgets to fix the customized design
Remove or replace the widgets that conflict with the desired design. If a built-in  widget is hard to restyle to the desired design, update the widget. For example, remove the filter button from the search field when not needed.

### Migration Strategy

When working with non-compliant extensions:

1. **Start with global defaults** - Set up your `app.style` extension first
2. **Identify gaps** - Note which widgets don't respect global styles
3. **Create extension-specific overrides** - Use `frame.set_style()` for missing styles
4. **Refactor widgets** - Refactor the widgets if they do not fit in the design.

This approach lets you maintain consistent styling across your app while waiting for all extensions to adopt the centralized styling pattern.
