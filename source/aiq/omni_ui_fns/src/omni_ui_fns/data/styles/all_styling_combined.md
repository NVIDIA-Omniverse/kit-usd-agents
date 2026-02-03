# Omni Kit UI Style Documentation - Complete Reference

*Generated on: 2026-01-16 12:58:31*

This document combines all UI styling documentation for Omni Kit into a single reference.

## Table of Contents

1. [Overview](#overview)
2. [Styling](#styling)
3. [Ui Style Best Practice](#ui_style_best_practice)
4. [Units](#units)
5. [Fonts](#fonts)
6. [Shades](#shades)
7. [Window](#window)
8. [Containers](#containers)
9. [Widgets](#widgets)
10. [Buttons](#buttons)
11. [Sliders](#sliders)
12. [Shapes](#shapes)
13. [Line](#line)

---

<a name="overview"></a>

================================================================================

# Overview Section

*Source: overview.md*

================================================================================

# Overview

OmniUI style allows users to build customized widgets, make these widgets visually pleasant and functionally indicative with user interactions.

Each widget has its own style to be tweaked with based on their use cases and behaviors, while they also follow the same syntax rules. The container widgets provide a customized style for the widgets layout, providing flexibility for the arrangement of elements.

Each omni ui item has its own style to be tweaked with based on their use cases and behaviors, while they also follow the same syntax rules for the style definition.

Shades are used to have different themes for the entire ui, e.g. dark themed ui and light themed ui. Omni.ui also supports different font styles and sizes. Different length units allows users to define the widgets accurate to exact pixel or proportional to the parent widget or siblings.

Shapes are the most basic elements in the ui, which allows users to create stylish ui shapes, rectangles, circles, triangles, line and curve. Freeshapes are the extended shapes, which allows users to control some of the attributes dynamically through bounded widgets.

Widgets are mostly a combination of shapes, images or texts, which are created to be stepping stones for the entire ui window. Each of the widget has its own style to be characterized.

The container widgets provide a customized style for the widgets layout, providing flexibility for the arrangement of elements and possibility of creating more complicated and customized widgets.



<a name="styling"></a>

================================================================================

# Styling Section

*Source: styling.md*

================================================================================

# The Style Sheet Syntax

omni.ui Style Sheet rules are almost identical to those of HTML CSS. It applies to the style of all omni ui elements.

Style sheets consist of a sequence of style rules. A style rule is made up of a selector and a declaration. The selector specifies which widgets are affected by the rule. The declaration specifies which properties should be set on the widget. For example:

```execute 200
## Double comment means hide from shippet
from omni.ui import color as cl
##
with ui.VStack(width=0, style={"Button": {"background_color": cl("#097eff")}}):
    ui.Button("Style Example")
```
In the above style rule, Button is the selector, and {"background_color": cl("#097eff")} is the declaration. The rule specifies that Button should use blue as its background color.

## Selector
There are three types of selectors, type Selector, name Selector and state Selector They are structured as:

Type Selector :: Name Selector : State Selector

e.g.,Button::okButton:hovered

### Type Selector
where `Button` is the type selector, which matches the ui.Button's type.

### Name Selector
`okButton` is the name selector, which selects all Button instances whose object name is okButton. It separates from the type selector with `::`.

### State Selector
`hovered` is the state selector, which by itself matches all Button instances whose state is hovered. It separates from the other selectors with `:`.

When type, name and state selector are used together, it defines the style of all Button typed instances named as `okButton` and in hovered, while `Button:hovered` defines the style of all Button typed instances which are in hovered states.

These are the states recognized by omni.ui:
* hovered : the mouse in the widget area
* pressed : the mouse is pressing in the widget area
* selected : the widget is selected
* disabled : the widget is disabled
* checked : the widget is checked
* drop : the rectangle is accepting a drop. For example,
style = {"Rectangle:drop" :  {"background_color": cl.blue}} meaning if the drop is acceptable, the rectangle is blue.

## Style Override
### Omit the selector
It's possible to omit the selector and override the property in all the widget types.

In this example, the style is set to VStack. The style will be propagated to all the widgets in VStack including VStack itself. Since only `background_color` is in the style, only the widgets which have `background_color` as the style will have the background color set. For VStack and Label which don't have `background_color`, the style is ignored. Button and FloatField get the blue background color.

```execute 200
from omni.ui import color as cl
with ui.VStack(width=400, style={"background_color": cl("#097eff")}, spacing=5):
    ui.Button("One")
    ui.Button("Two")
    ui.FloatField()
    ui.Label("Label doesn't have background_color style")
```

### Style overridden with name and state selector
In this example, we set the "Button" style for all the buttons, then override different buttons with name selector style, e.g. "Button::one" and "Button::two". Furthermore, the we also set different style for Button::one when pressed or hovered, e.g. "Button::one:hovered" and "Button::one:pressed", which will override "Button::one" style when button is pressed or hovered.

```execute 200
from omni.ui import color as cl
style1 = {
    "Button": {"border_width": 0.5, "border_radius": 0.0, "margin": 5.0, "padding": 5.0},
    "Button::one": {
        "background_color": cl("#097eff"),
        "background_gradient_color": cl("#6db2fa"),
        "border_color": cl("#1d76fd"),
    },
    "Button.Label::one": {"color": cl.white},
    "Button::one:hovered": {"background_color": cl("#006eff"), "background_gradient_color": cl("#5aaeff")},
    "Button::one:pressed": {"background_color": cl("#6db2fa"), "background_gradient_color": cl("#097eff")},
    "Button::two": {"background_color": cl.white, "border_color": cl("#B1B1B1")},
    "Button.Label::two": {"color": cl("#272727")},
    "Button::three:hovered": {
        "background_color": cl("#006eff"),
        "background_gradient_color": cl("#5aaeff"),
        "border_color": cl("#1d76fd"),
    },
    "Button::four:pressed": {
        "background_color": cl("#6db2fa"),
        "background_gradient_color": cl("#097eff"),
        "border_color": cl("#1d76fd"),
    },
}

with ui.HStack(style=style1):
    ui.Button("One", name="one")
    ui.Button("Two", name="two")
    ui.Button("Three", name="three")
    ui.Button("Four", name="four")
    ui.Button("Five", name="five")
```

### Style override to different levels of the widgets
It's possible to assign any style override to any level of the widgets. It can be assigned to both parents and children at the same time.

In this example, we have style_system which will be propagated to all buttons, but buttons with its own style  will override the style_system.

```execute 200
from omni.ui import color as cl
style_system = {
    "Button": {
        "background_color": cl("#E1E1E1"),
        "border_color": cl("#ADADAD"),
        "border_width": 0.5,
        "border_radius": 3.0,
        "margin": 5.0,
        "padding": 5.0,
    },
    "Button.Label": {
        "color": cl.black,
    },
    "Button:hovered": {
        "background_color": cl("#e5f1fb"),
        "border_color": cl("#0078d7"),
    },
    "Button:pressed": {
        "background_color": cl("#cce4f7"),
        "border_color": cl("#005499"),
        "border_width": 1.0
    },
}

with ui.HStack(style=style_system):
    ui.Button("One")
    ui.Button("Two", style={"color": cl("#AAAAAA")})
    ui.Button("Three", style={"background_color": cl("#097eff"), "background_gradient_color": cl("#6db2fa")})
    ui.Button(
        "Four", style={":hovered": {"background_color": cl("#006eff"), "background_gradient_color": cl("#5aaeff")}}
    )
    ui.Button(
        "Five",
        style={"Button:pressed": {"background_color": cl("#6db2fa"), "background_gradient_color": cl("#097eff")}},
    )
```

### Customize the selector type using style_type_name_override
What if the user has a customized widget which is not a standard omni.ui one. How to define that Type Selector? In this case, We can use `style_type_name_override` to override the type name. `name` attribute is the Name Selector and State Selector can be added as usual.

Another use case is when we have a giant list of the same typed widgets, for example `Button`, but some of the Buttons are in the main window, and some of the Buttons are in the pop-up window, which we want to differentiate for easy look-up. Instead of calling all of them the same Type Selector as `Button` and only have different Name Selectors, we can override the type name for the main window buttons as `WindowButton` and the pop-up window buttons as `PopupButton`. This groups the style-sheet into categories and makes the change of the look or debug much easier.

Here is an example where we use `style_type_name_override` to override the style type name.

```execute 200
from omni.ui import color as cl
style={
    "WindowButton::one": {"background_color": cl("#006eff")},
    "WindowButton::one:hovered": {"background_color": cl("#006eff"), "background_gradient_color": cl("#FFAEFF")},
    "PopupButton::two": {"background_color": cl("#6db2fa")},
    "PopupButton::two:hovered": {"background_color": cl("#6db2fa"), "background_gradient_color": cl("#097eff")},
    }

with ui.HStack(width=400, style=style, spacing=5):
    ui.Button("Open", style_type_name_override="WindowButton", name="one")
    ui.Button("Save", style_type_name_override="PopupButton", name="two")
```

### Default style override
From the above examples, we know that if we want to propagate the style to all children, we just need to set the style to the parent widget, but this rule doesn't apply to windows. The style set to the window will not propagate to its widgets. If we want to propagate the style to ui.Window and their widgets, we should set the default style with `ui.style.default`.

```python
from omni.ui import color as cl
ui.style.default = {
    "background_color": cl.blue,
    "border_radius": 10,
    "border_width": 5,
    "border_color": cl.red,
}
```

## Debug Color
All shapes or widgets can be styled to use a debug color that enables you to visualize their frame. It is very useful when debugging complicated ui layout with overlaps.

Here we use red as the debug_color to indicate the label widget:

```execute 200
from omni.ui import color as cl
style = {"background_color": cl("#DDDD00"), "color": cl.white, "debug_color":cl("#FF000055")}

ui.Label("Label with Debug", width=200, style=style)
```

If several widgets are adjacent, we can use the `debug_color` in the `hovered` state to differentiate the widget with others.

```execute 200
from omni.ui import color as cl
style = {
    "Label": {"padding": 3, "background_color": cl("#DDDD00"),"color": cl.white},
    "Label:hovered": {"debug_color": cl("#00FFFF55")},}

with ui.HStack(width=500, style=style):
    ui.Label("Label 1", width=50)
    ui.Label("Label 2")
    ui.Label("Label 3", width=100, alignment=ui.Alignment.CENTER)
    ui.Spacer()
    ui.Label("Label 3", width=50)
```

## Visibility
This property holds whether the shape or widget is visible. Invisible shape or widget is not rendered, and it doesn't take part in the layout. The layout skips it.

In the following example, click the button from one to five to hide itself. The `Visible all` button brings them all back.
```execute 200
def invisible(button):
    button.visible = False

def visible(buttons):
    for button in buttons:
        button.visible = True

buttons = []
with ui.HStack():
    for n in ["One", "Two", "Three", "Four", "Five"]:
        button = ui.Button(n, width=0)
        button.set_clicked_fn(lambda b=button: invisible(b))
        buttons.append(button)

    ui.Spacer()
    button = ui.Button("Visible all", width=0)
    button.set_clicked_fn(lambda b=buttons: visible(b))
```


<a name="ui_style_best_practice"></a>

================================================================================

# Ui Style Best Practice Section

*Source: ui_style_best_practice.md*

================================================================================

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



<a name="units"></a>

================================================================================

# Units Section

*Source: units.md*

================================================================================

# Length Units
The Framework UI offers several different units for expressing length: Pixel, Percent and Fraction. There is no restriction on where certain units should be used.

## Pixel
Pixel is the size in pixels and scaled with the HiDPI scale factor. Pixel is the default unit. If a number is not specified to be a certain unit, it is Pixel. e.g. `width=100` meaning `width=ui.Pixel(100)`.

```execute 200
with ui.HStack():
    ui.Button("40px", width=ui.Pixel(40))
    ui.Button("60px", width=ui.Pixel(60))
    ui.Button("100px", width=100)
    ui.Button("120px", width=120)
    ui.Button("150px", width=150)
```

## Percent
Percent and Fraction units make it possible to specify sizes relative to the parent size. 1 Percent is 1/100 of the parent size.

```execute 200
with ui.HStack():
    ui.Button("5%", width=ui.Percent(5))
    ui.Button("10%", width=ui.Percent(10))
    ui.Button("15%", width=ui.Percent(15))
    ui.Button("20%", width=ui.Percent(20))
    ui.Button("25%", width=ui.Percent(25))
```

## Fraction
Fraction length is made to take the available space of the parent widget and then divide it among all the child widgets with Fraction length in proportion to their Fraction factor.

```execute 200
with ui.HStack():
    ui.Button("One", width=ui.Fraction(1))
    ui.Button("Two", width=ui.Fraction(2))
    ui.Button("Three", width=ui.Fraction(3))
    ui.Button("Four", width=ui.Fraction(4))
    ui.Button("Five", width=ui.Fraction(5))
```



<a name="fonts"></a>

================================================================================

# Fonts Section

*Source: fonts.md*

================================================================================

# Fonts

## Font style
It's possible to set different font types with the style. The style key 'font' should point to the font file, which allows packaging of the font to the extension. We support both TTF and OTF formats. All text-based widgets support custom fonts.

```execute 200
with ui.VStack():
    ui.Label("Omniverse", style={"font":"${fonts}/OpenSans-SemiBold.ttf", "font_size": 40.0})
    ui.Label("Omniverse", style={"font":"${fonts}/roboto_medium.ttf", "font_size": 40.0})
```

## Font size
It's possible to set the font size with the style.

Drag the following slider to change the size of the text.

```execute 200
## Double comment means hide from snippet
from functools import partial
##
def value_changed(label, value):
    label.style = {"color": ui.color(0), "font_size": value.as_float}

slider = ui.FloatSlider(min=1.0, max=150.0)
slider.model.as_float = 10.0
label = ui.Label("Omniverse", style={"color": ui.color(0), "font_size": 7.0})
slider.model.add_value_changed_fn(partial(value_changed, label))
## Double comment means hide from snippet
ui.Spacer(height=30)
##
```


<a name="shades"></a>

================================================================================

# Shades Section

*Source: shades.md*

================================================================================

# Shades
Shades are used to have multiple named color palettes with the ability for runtime switch. For example, one App could have several ui themes users can switch during using the App.

The shade can be defined with the following code:

```python
    cl.shade(cl("#FF6600"), red=cl("#0000FF"), green=cl("#66FF00"))
```

It can be assigned to the color style. It's possible to switch the color with the following command globally:

```python
    cl.set_shade("red")
```

## Example
```execute 200
from omni.ui import color as cl
from omni.ui import constant as fl
from functools import partial

def set_color(color):
    cl.example_color = color

def set_width(value):
    fl.example_width = value

cl.example_color = cl.green
fl.example_width = 1.0

with ui.HStack(height=100, spacing=5):
    with ui.ZStack():
        ui.Rectangle(
            style={
                "background_color": cl.shade(
                    "aqua",
                    orange=cl.orange,
                    another=cl.example_color,
                    transparent=cl(0, 0, 0, 0),
                    black=cl.black,
                ),
                "border_width": fl.shade(1, orange=4, another=8),
                "border_radius": fl.one,
                "border_color": cl.black,
            },
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.Rectangle(\n"
                "\tstyle={\n"
                '\t\t"background_color":\n'
                "\t\t\tcl.shade(\n"
                '\t\t\t\t"aqua",\n'
                "\t\t\t\torange=cl(1, 0.5, 0),\n"
                "\t\t\t\tanother=cl.example_color),\n"
                '\t\t"border_width":\n'
                "\t\t\tfl.shade(1, orange=4, another=8)})",
                style={"color": cl.black, "margin": 15},
                width=0,
            )
            ui.Spacer()

    with ui.ZStack():
        ui.Rectangle(
            style={
                "background_color": cl.example_color,
                "border_width": fl.example_width,
                "border_radius": fl.one,
                "border_color": cl.black,
            }
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.Rectangle(\n"
                "\tstyle={\n"
                '\t\t"background_color": cl.example_color,\n'
                '\t\t"border_width": fl.example_width)})',
                style={"color": cl.black, "margin": 15},
                width=0,
            )
            ui.Spacer()

with ui.VStack(style={"Button": {"background_color": cl("097EFF")}}):
    ui.Label("Click the following buttons to change the shader of the left rectangle")
    with ui.HStack():
        ui.Button("cl.set_shade()", clicked_fn=partial(cl.set_shade, ""))
        ui.Button('cl.set_shade("orange")', clicked_fn=partial(cl.set_shade, "orange"))
        ui.Button('cl.set_shade("another")', clicked_fn=partial(cl.set_shade, "another"))
    ui.Label("Click the following buttons to change the border width of the right rectangle")
    with ui.HStack():
        ui.Button("fl.example_width = 1", clicked_fn=partial(set_width, 1))
        ui.Button("fl.example_width = 4", clicked_fn=partial(set_width, 4))
    ui.Label("Click the following buttons to change the background color of both rectangles")
    with ui.HStack():
        ui.Button('cl.example_color = "green"', clicked_fn=partial(set_color, "green"))
        ui.Button("cl.example_color = cl(0.8)", clicked_fn=partial(set_color, cl(0.8)))
    ## Double comment means hide from shippet
    ui.Spacer(height=15)
    ##
```

## URL Shades Example
It's also possible to use shades for specifying shortcuts to the images and style-based paths.

```execute 200
from omni.ui import color as cl
from omni.ui.url_utils import url
from functools import partial

def set_url(url_path: str):
    url.example_url = url_path

walk = "resources/icons/Nav_Walkmode.png"
fly = "resources/icons/Nav_Flymode.png"

url.example_url = walk

with ui.HStack(height=100, spacing=5):
    with ui.ZStack():
        ui.Image(height=100, style={"image_url": url.example_url})
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                'ui.Image(\n\tstyle={"image_url": cl.example_url})\n',
                style={"color": cl.black, "font_size": 12, "margin": 15},
                width=0,
            )
            ui.Spacer()
    with ui.ZStack():
        ui.ImageWithProvider(
            height=100,
            style={
                "image_url": url.shade(
                    "resources/icons/Move_local_64.png",
                    another="resources/icons/Move_64.png",
                    orange="resources/icons/Rotate_local_64.png",
                )
            }
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.ImageWithProvider(\n"
                "\tstyle={\n"
                '\t\t"image_url":\n'
                "\t\t\tst.shade(\n"
                '\t\t\t\t"Move_local_64.png",\n'
                '\t\t\t\tanother="Move_64.png")})\n',
                style={"color": cl.black, "font_size": 12, "margin": 15},
                width=0,
            )
            ui.Spacer()

with ui.HStack():
    # buttons to change the url for the image
    with ui.VStack():
        ui.Button("url.example_url = Nav_Walkmode.png", clicked_fn=partial(set_url, walk))
        ui.Button("url.example_url = Nav_Flymode.png", clicked_fn=partial(set_url, fly))
    # buttons to switch between shades to a different image
    with ui.VStack():
        ui.Button("ui.set_shade()", clicked_fn=partial(ui.set_shade, ""))
        ui.Button('ui.set_shade("another")', clicked_fn=partial(ui.set_shade, "another"))
```



<a name="window"></a>

================================================================================

# Window Section

*Source: window.md*

================================================================================

# Window Widgets

## MainWindow
The MainWindow represents the main window for an application. There should only be one MainWindow in each application.

Here is a list of styles you can customize on MainWindow:

> background_color (color): the background color of the main window.
> margin_height (float): the height distance between the window content and the window border.
> margin_width (float): the width distance between the window content and the window border.

Here is an example of a main window with style. Click the button to show the main window. Since the example is running within a MainWindow already, creating a new MainWindow will not run correctly in this example, but it demonstrates how to set the style of the `MainWindow`. And note the style of MainWindow is not propagated to other windows.

```execute 200
from omni.ui import color as cl

self._main_window = None
self._window1 = None
self._window2 = None
def create_main_window():
    if not self._main_window:
        self._main_window = ui.MainWindow()
        self._main_window.main_frame.set_style({
            "MainWindow": {
                "background_color": cl.purple,
                "margin_height": 20,
                "margin_width": 10
            }})
        self._window1 = ui.Window("window 1", width=300, height=300)
        self._window2 = ui.Window("window 2", width=300, height=300)
        main_dockspace = ui.Workspace.get_window("DockSpace")
        self._window1.dock_in(main_dockspace, ui.DockPosition.SAME)
        self._window2.dock_in(main_dockspace, ui.DockPosition.SAME)
        self._window2.focus()
    self._window2.visible = True

ui.Button("click for Main Window", width=180, clicked_fn=create_main_window)
```

## Window
The window is a child window of the MainWindow. And it can be docked. You can have any type of widgets as the window content widgets.

Here is a list of styles you can customize on Window:
> background_color (color): the background color of the window.
> border_color (color): the border color if the window has a border.
> border_radius (float): the radius of the corner angle if the user wants to round the window.
> border_width (float): the border width if the window has a border.

Here is an example of a window with style. Click the button to show the window.
```execute 200
from omni.ui import color as cl

self._style_window_example = None
def create_styled_window():
    if not self._style_window_example:
        self._style_window_example = ui.Window("Styled Window Example", width=300, height=300)
        self._style_window_example.frame.set_style({
            "Window": {
                "background_color": cl.blue,
                "border_radius": 10,
                "border_width": 5,
                "border_color": cl.red,
            }})
        self._style_window_example.visible = True

ui.Button("click for Styled Window", width=180, clicked_fn=create_styled_window)
```

Note that a window's style is set from its frame since ui.Window itself is not a widget. We can't set style to it like other widgets. ui.Window's frame is a normal ui.Frame widget which itself doesn't have styles like `background_color` or `border_radius` (see `Container Widgets`->`Frame`). We specifically interpret the input ui.Window's frame style as the window style here. Therefore, the window style is not propagated to the content widget either just like the MainWindow.

If you want to set up a default style for the entire window. You should use `ui.style.default`. More details in `The Style Sheet Syntax` -> `Style Override` -> `Default style override`.

## Menu
The Menu class provides a menu widget for use in menu bars, context menus, and other popup menus. It can be either a pull-down menu in a menu bar or a standalone context menu. Pull-down menus are shown by the menu bar when the user clicks on the respective item. Context menus are usually invoked by some special keyboard key or by right-clicking.

Here is a list of styles you can customize on Menu:
> color (color): the color of the menu text
> background_color (color): the background color of sub menu window
> background_selected_color (color): the background color when the current menu is selected
> border_color (color): the border color of the sub menu window if it has a border
> border_width (float): the border width of the sub menu window if it has a border
> border_radius (float): the border radius of the sub menu window if user wants to round the sub menu window
> padding (float): the padding size of the sub menu window

Here is a list of styles you can customize on MenuItem:
> color (color): the color of the menu Item text
> background_selected_color (color): the background color when the current menu is selected

Right click for the context menu with customized menu style:
```execute 200
from omni.ui import color as cl
self.context_menu = None
def show_context_menu(x, y, button, modifier, widget):
    if button != 1:
        return
    self.context_menu = ui.Menu("Context menu",
        style={
                "Menu": {
                    "background_color": cl.blue,
                    "color": cl.pink,
                    "background_selected_color": cl.green,
                    "border_radius": 5,
                    "border_width": 2,
                    "border_color": cl.yellow,
                    "padding": 15
                    },
                "MenuItem": {
                    "color": cl.white,
                    "background_selected_color": cl.cyan},
                "Separator": {
                    "color": cl.red},
            },)
    with self.context_menu:
        ui.MenuItem("Delete Shot")
        ui.Separator()
        ui.MenuItem("Attach Selected Camera")
        with ui.Menu("Sub-menu"):
            ui.MenuItem("One")
            ui.MenuItem("Two")
            ui.MenuItem("Three")
            ui.Separator()
            ui.MenuItem("Four")
            with ui.Menu("Five"):
                ui.MenuItem("Six")
                ui.MenuItem("Seven")

    self.context_menu.show()

with ui.VStack():
    button = ui.Button("Right click to context menu", height=0, width=0)
    button.set_mouse_pressed_fn(lambda x, y, b, m, widget=button: show_context_menu(x, y, b, m, widget))
```

Left click for the push button menu with default menu style:
```execute 200
self.pushed_menu = None
def show_pushed_menu(x, y, button, modifier, widget):
    self.pushed_menu = ui.Menu("Pushed menu")
    with self.pushed_menu:
        ui.MenuItem("Camera 1")
        ui.MenuItem("Camera 2")
        ui.MenuItem("Camera 3")
        ui.Separator()
        with ui.Menu("More Cameras"):
            ui.MenuItem("This Menu is Pushed")
            ui.MenuItem("and Aligned with a widget")

    self.pushed_menu.show_at(
            (int)(widget.screen_position_x), (int)(widget.screen_position_y + widget.computed_content_height)
        )

with ui.VStack():
    button = ui.Button("Pushed Button Menu", height=0, width=0)
    button.set_mouse_pressed_fn(lambda x, y, b, m, widget=button: show_pushed_menu(x, y, b, m, widget))
```

### Separator
Separator is a type of MenuItem which creates a separator line in the UI elements.

From the above example, you can see the use of Separator in Menu.
Here is a list of styles you can customize on Separator:
> color (color): the color of the Separator

## MenuBar
All the Windows in Omni.UI can have a MenuBar. To add a MenuBar to your window add this flag to your constructor: omni.ui.Window(flags=ui.WINDOW_FLAGS_MENU_BAR). The MenuBar object can then be accessed through the menu_bar read-only property on your window.

A MenuBar is a container so it is built like a Frame or Stack but only takes Menu objects as children. You can leverage the 'priority' property on the Menu to order them. They will automatically be sorted when they are added, but if you change the priority of an item then you need to explicitly call sort().

MenuBar has exactly the same style list you can customize as Menu.

Here is an example of MenuBar with style for the Window:

```execute 200
from omni.ui import color as cl
style={"MenuBar": {
            "background_color": cl.blue,
            "color": cl.pink,
            "background_selected_color": cl.green,
            "border_radius": 2,
            "border_width": 1,
            "border_color": cl.yellow,
            "padding": 2}}
self._window_menu_example = None
def create_and_show_window_with_menu():
    if not self._window_menu_example:
        self._window_menu_example = ui.Window(
            "Window Menu Example",
            width=300,
            height=300,
            flags=ui.WINDOW_FLAGS_MENU_BAR | ui.WINDOW_FLAGS_NO_BACKGROUND,
        )
        menu_bar = self._window_menu_example.menu_bar
        menu_bar.style = style
        with menu_bar:
            with ui.Menu("File"):
                ui.MenuItem("Load")
                ui.MenuItem("Save")
                ui.MenuItem("Export")
            with ui.Menu("Window"):
                ui.MenuItem("Hide")

        with self._window_menu_example.frame:
            with ui.VStack():
                ui.Button("This Window has a Menu")

                def show_hide_menu(menubar):
                    menubar.visible = not menubar.visible

                ui.Button("Click here to show/hide Menu", clicked_fn=lambda m=menu_bar: show_hide_menu(m))

                def add_menu(menubar):
                    with menubar:
                        with ui.Menu("New Menu"):
                            ui.MenuItem("I don't do anything")

                ui.Button("Add New Menu", clicked_fn=lambda m=menu_bar: add_menu(m))

    self._window_menu_example.visible = True

with ui.HStack(width=0):
    ui.Button("window with MenuBar Example", width=180, clicked_fn=create_and_show_window_with_menu)
    ui.Label("this populates the menuBar", name="text", width=180, style={"margin_width": 10})
```



<a name="containers"></a>

================================================================================

# Containers Section

*Source: containers.md*

================================================================================

# Container widgets
Container widgets are used for grouping items. It's possible to add children to the container with Python's `with` statement. It's not possible to reparent items. Instead, it's necessary to remove the item and recreate a similar item under another parent.

## Stack
We have three main components: VStack, HStack, and ZStack.

Here is a list of styles you can customize on Stack:
> margin (float): the distance between the stack items and the parent widget defined boundary
> margin_width (float): the width distance between the stack items and the parent widget defined boundary
> margin_height (float): the height distance between the stack items and the parent widget defined boundary

It's possible to determine the direction of a stack with the property `direction`. Here is an example of a stack which is able to change its direction dynamically by clicking the button `Change`.

```execute 200
def rotate(dirs, stack, label):
    dirs[0] = (dirs[0] + 1) % len(dirs[1])
    stack.direction = dirs[1][dirs[0]]
    label.text = str(stack.direction)

dirs = [
    0,
    [
        ui.Direction.LEFT_TO_RIGHT,
        ui.Direction.RIGHT_TO_LEFT,
        ui.Direction.TOP_TO_BOTTOM,
        ui.Direction.BOTTOM_TO_TOP,
    ],
]
stack = ui.Stack(ui.Direction.LEFT_TO_RIGHT, width=0, height=0, style={"margin_height": 5, "margin_width": 10})
with stack:
    for name in ["One", "Two", "Three", "Four"]:
        ui.Button(name)
ui.Spacer(height=100)
with ui.HStack():
    ui.Label("Current direction is ", name="text", width=0)
    label = ui.Label("", name="text")

    button = ui.Button("Change")
    button.set_clicked_fn(lambda d=dirs, s=stack, l=label: rotate(d, s, l))
    rotate(dirs, stack, label)
```

### HStack
This class is used to construct horizontal layout objects.

The simplest use of the class is like this:
```execute 200
with ui.HStack(style={"margin": 10}):
    ui.Button("One")
    ui.Button("Two")
    ui.Button("Three")
    ui.Button("Four")
    ui.Button("Five")
```
### VStack
The VStack class lines up widgets vertically.
```execute 200
with ui.VStack(width=100.0, style={"margin": 5}):
    with ui.VStack():
        ui.Button("One")
        ui.Button("Two")
        ui.Button("Three")
        ui.Button("Four")
        ui.Button("Five")
```

### ZStack
ZStack is a view that overlays its children, aligning them on top of each other. The later one is on top of the previous ones.
```execute 200
with ui.VStack(width=100.0, style={"margin": 5}):
    with ui.ZStack():
        ui.Button("Very Long Text to See How Big it Can Be", height=0)
        ui.Button("Another\nMultiline\nButton", width=0)
```

### Layout
Here is an example of using combined HStack and VStack:
```execute 200
with ui.VStack():
    for i in range(2):
        with ui.HStack():
            ui.Spacer(width=50)

            with ui.VStack(height=0):
                ui.Button("Left {}".format(i), height=0)
                ui.Button("Vertical {}".format(i), height=50)

            with ui.HStack(width=ui.Fraction(2)):
                ui.Button("Right {}".format(i))
                ui.Button("Horizontal {}".format(i), width=ui.Fraction(2))

            ui.Spacer(width=50)
```

### Spacing
Spacing is a property of Stack. It defines the non-stretchable space in pixels between child items of the layout.

Here is an example that you can change the HStack spacing by a slider
```execute 200
from omni.ui import color as cl
SPACING = 5

def set_spacing(stack, spacing):
    stack.spacing = spacing

ui.Spacer(height=SPACING)
spacing_stack = ui.HStack(style={"margin": 0})
with spacing_stack:
    for name in ["One", "Two", "Three", "Four"]:
        ui.Button(name)

ui.Spacer(height=SPACING)
with ui.HStack(spacing=SPACING):
    with ui.HStack(width=100):
        ui.Spacer()
        ui.Label("spacing", width=0, name="text")
    with ui.HStack(width=ui.Percent(20)):
        field = ui.FloatField(width=50)
        slider = ui.FloatSlider(min=0, max=50, style={"color": cl.transparent})
        # Link them together
        slider.model = field.model
        slider.model.add_value_changed_fn(
            lambda m, s=spacing_stack: set_spacing(s, m.get_value_as_float()))
```

## Frame
Frame is a container that can keep only one child. Each child added to Frame overrides the previous one. This feature is used for creating dynamic layouts. The whole layout can be easily recreated with a simple callback.

Here is a list of styles you can customize on Frame:
> padding (float): the distance between the child widgets and the border of the button

In the following example, you can drag the IntDrag to change the slider value. The buttons are recreated each time the slider changes.
```execute 200
self._recreate_ui = ui.Frame(height=40, style={"Frame":{"padding": 5}})

def changed(model, recreate_ui=self._recreate_ui):
    with recreate_ui:
        with ui.HStack():
            for i in range(model.get_value_as_int()):
                ui.Button(f"Button #{i}")

model = ui.IntDrag(min=0, max=10).model
self._sub_recreate = model.subscribe_value_changed_fn(changed)
```

Another feature of Frame is the ability to clip its child. When the content of Frame is bigger than Frame itself, the exceeding part is not drawn if the clipping is on. There are two clipping types: `horizontal_clipping` and `vertical_clipping`.

Here is an example of vertical clipping.
```execute 200
with ui.Frame(vertical_clipping=True, height=20):
    ui.Label("This should be clipped vertically. " * 10, word_wrap=True)
```

## CanvasFrame
CanvasFrame is the widget that allows the user to pan and zoom its children with a mouse. It has a layout that can be infinitely moved in any direction.

Here is a list of styles you can customize on CanvasFrame:
> background_color (color): the main color of the rectangle

Here is an example of a CanvasFrame, you can scroll the middle mouse to zoom the canvas and middle mouse move to pan in it (press CTRL to avoid scrolling the docs).

```execute 200
from omni.ui import color as cl
TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
)

IMAGE = "resources/icons/ov_logo_square.png"
with ui.CanvasFrame(height=256, style={"CanvasFrame":{"background_color": cl("#aa4444")}}):
    with ui.VStack(height=0, spacing=10):
        ui.Label(TEXT, name="text", word_wrap=True)
        ui.Button("Button")
        ui.Image(IMAGE, width=128, height=128)
```

## ScrollingFrame
The ScrollingFrame class provides the ability to scroll onto other widgets. ScrollingFrame is used to display the contents of children widgets within a frame. If the widget exceeds the size of the frame, the frame can provide scroll bars so that the entire area of the child widget can be viewed by scrolling.

Here is a list of styles you can customize on ScrollingFrame:
> scrollbar_size (float): the width of the scroll bar
> secondary_color (color): the color the scroll bar
> background_color (color): the background color the scroll frame

Here is an example of a ScrollingFrame, you can scroll the middle mouse to scroll the frame.

```execute 200
from omni.ui import color as cl
with ui.HStack():
    left_frame = ui.ScrollingFrame(
        height=250,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        style={"ScrollingFrame":{
            "scrollbar_size":10,
            "secondary_color": cl.red,
            "background_color": cl("#4444dd")}}
    )
    with left_frame:
        with ui.VStack(height=0):
            for i in range(20):
                ui.Button(f"Button Left {i}")

    right_frame = ui.ScrollingFrame(
        height=250,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        style={"ScrollingFrame":{
            "scrollbar_size":30,
            "secondary_color": cl.blue,
            "background_color": cl("#44dd44")}}
    )
    with right_frame:
        with ui.VStack(height=0):
            for i in range(20):
                ui.Button(f"Button Right {i}")

# Synchronize the scroll position of two frames
def set_scroll_y(frame, y):
    frame.scroll_y = y

left_frame.set_scroll_y_changed_fn(lambda y, frame=right_frame: set_scroll_y(frame, y))
right_frame.set_scroll_y_changed_fn(lambda y, frame=left_frame: set_scroll_y(frame, y))
```

## CollapsableFrame
CollapsableFrame is a frame widget that can hide or show its content. It has two states: expanded and collapsed. When it's collapsed, it looks like a button. If it's expanded, it looks like a button and a frame with the content. It's handy to group properties, and temporarily hide them to get more space for something else.

Here is a list of styles you can customize on Image:
> background_color (color): the background color of the CollapsableFrame widget
> secondary_color (color): the background color of the CollapsableFrame's header
> border_radius (float): the border radius if user wants to round the CollapsableFrame
> border_color (color): the border color if the CollapsableFrame has a border
> border_width (float): the border width if the CollapsableFrame has a border
> padding (float): the distance between the header or the content to the border of the CollapsableFrame
> margin (float): the distance between the CollapsableFrame and other widgets

Here is a default `CollapsableFrame` example:
```execute 200
with ui.CollapsableFrame("Header"):
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")
```

It's possible to use a custom header.
```execute 200
from omni.ui import color as cl
def custom_header(collapsed, title):
    with ui.HStack():
        with ui.ZStack(width=30):
            ui.Circle(name="title")
            with ui.HStack():
                ui.Spacer()
                align = ui.Alignment.V_CENTER
                ui.Line(name="title", width=6, alignment=align)
                ui.Spacer()
            if collapsed:
                with ui.VStack():
                    ui.Spacer()
                    align = ui.Alignment.H_CENTER
                    ui.Line(name="title", height=6, alignment=align)
                    ui.Spacer()

        ui.Label(title, name="title")

style = {
    "CollapsableFrame": {
        "background_color": cl(0.5),
        "secondary_color": cl("#CC211B"),
        "border_radius": 10,
        "border_color": cl.blue,
        "border_width": 2,
    },
    "CollapsableFrame:hovered": {"secondary_color": cl("#FF4321")},
    "CollapsableFrame:pressed": {"secondary_color": cl.red},
    "Label::title": {"color": cl.white},
    "Circle::title": {
        "color": cl.yellow,
        "background_color": cl.transparent,
        "border_color": cl(0.9),
        "border_width": 0.75,
    },
    "Line::title": {"color": cl(0.9), "border_width": 1},
}

ui.Spacer(height=5)
with ui.HStack():
    ui.Spacer(width=5)
    with ui.CollapsableFrame("Header", build_header_fn=custom_header, style=style):
        with ui.VStack(height=0):
            ui.Button("Hello World")
            ui.Button("Hello World")
    ui.Spacer(width=5)
ui.Spacer(height=5)
```

This example demonstrates how padding and margin work in the collapsable frame.
```execute 200
from omni.ui import color as cl
style = {
    "CollapsableFrame": {
        "border_color": cl("#005B96"),
        "border_radius": 4,
        "border_width": 2,
        "padding": 0,
        "margin": 0,
    }
}
frame = ui.CollapsableFrame("Header", style=style)
with frame:
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")

def set_style(field, model, style=style, frame=frame):
    frame_style = style["CollapsableFrame"]
    frame_style[field] = model.get_value_as_float()
    frame.set_style(style)

with ui.HStack():
    ui.Label("Padding:", width=ui.Percent(10), name="text")
    model = ui.FloatSlider(min=0, max=50).model
model.add_value_changed_fn(lambda m: set_style("padding", m))

with ui.HStack():
    ui.Label("Margin:", width=ui.Percent(10), name="text")
    model = ui.FloatSlider(min=0, max=50).model
model.add_value_changed_fn(lambda m: set_style("margin", m))
```

## Order in Stack and use of content_clipping
Due to Imgui, ScrollingFrame and CanvasFrame will create a new window, meaning if we have them in a ZStack, they don't respect the Stack order. To fix that we need to create a separate window, with the widget wrapped in a `ui.Frame(separate_window=True)` will fix the order issue. And if we also want the mouse input in the new separate window, we use `ui.HStack(content_clipping=True)` for that.

In the following example, you won't see the red rectangle.

```execute 200
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

With the use of `separate_window=True` or `content_clipping=True`, you will see the red rectangle.

```execute 200
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    with ui.Frame(separate_window=True):
        ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

```execute 200
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    with ui.HStack(content_clipping=True):
        ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

In the following example, you will see the button click action is captured on Button 1.
```execute 200
from functools import partial

def clicked(name):
	print(f'clicked {name}')

with ui.ZStack():
    b1 = ui.Button('Button 1')
    b1.set_clicked_fn(partial(clicked, b1.text))
    b2 = ui.Button('Button 2')
    b2.set_clicked_fn(partial(clicked, b2.text))
```

With the use of `content_clipping=True`, you will see the button click action is now fixed and captured on Button 2.
```execute 200
from functools import partial

def clicked(name):
	print(f'clicked {name}')

with ui.ZStack():
    b1 = ui.Button('Button 1')
    b1.set_clicked_fn(partial(clicked, b1.text))
    with ui.VStack(content_clipping=1):
        b2 = ui.Button('Button 2')
        b2.set_clicked_fn(partial(clicked, b2.text))
```

## Grid
Grid is a container that arranges its child views in a grid. Depends on the direction the grid size grows with creating more children, we call it VGrid (grow in vertical direction) and HGrid (grow in horizontal direction)

There is currently no style you can customize on Grid.

### VGrid
VGrid has two modes for cell width:
 - If the user sets column_count, the column width is computed from the grid width.
 - If the user sets column_width, the column count is computed from the grid width.

VGrid also has two modes for height:
 - If the user sets row_height, VGrid uses it to set the height for all the cells. It's the fast mode because it's considered that the cell height never changes. VGrid easily predicts which cells are visible.

- If the user sets nothing, VGrid computes the size of the children. This mode is slower than the previous one, but the advantage is that all the rows can be different custom sizes. VGrid still only draws visible items, but to predict it, it uses cache, which can be big if VGrid has hundreds of thousands of items.

Here is an example of VGrid:
```execute 200
from omni.ui import color as cl
with ui.ScrollingFrame(
    height=250,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    with ui.VGrid(column_width=100, row_height=100):
        for i in range(100):
            with ui.ZStack():
                ui.Rectangle(
                    style={
                        "border_color": cl.red,
                        "background_color": cl.white,
                        "border_width": 1,
                        "margin": 0,
                    }
                )
                ui.Label(f"{i}", style={"margin": 5})
```

### HGrid
HGrid works exactly like VGrid, but with swapped width and height.
```execute 200
from omni.ui import color as cl
with ui.ScrollingFrame(
    height=250,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
):
    with ui.HGrid(column_width=100, row_height=100):
        for i in range(100):
            with ui.ZStack():
                ui.Rectangle(
                    style={
                        "border_color": cl.red,
                        "background_color": cl.white,
                        "border_width": 1,
                        "margin": 0,
                    }
                )
                ui.Label(f"{i}", style={"margin": 5})
```

## Placer
Placer enables you to place a widget precisely with offset. Placer's property `draggable` allows changing the position of the child widget by dragging it with the mouse.

There is currently no style you can customize on Placer.

Here is an example of 4 Placers. Two of them have fixed positions, each with a ui.Button as the child. You can see the buttons are moved to the exact place by the parent Placer, one at (100, 10) and the other at (200, 50). The third one is `draggable`, which has a Circle as the child, so that you can move the circle freely with mouse drag in the frame. The fourth one is also `draggable`, which has a ZStack as the child. The ZStack is composed of Rectangle and HStack and Label. This Placer is only draggable on the Y-axis, defined by `drag_axis=ui.Axis.Y`, so that you can only move the ZStack on the y-axis.


```execute 200
from omni.ui import color as cl
with ui.ScrollingFrame(
    height=170,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
):
    with ui.ZStack():
        with ui.HStack():
            for index in range(60):
                ui.Line(width=10, style={"color": cl.black, "border_width": 0.5}, alignment=ui.Alignment.LEFT)
        with ui.VStack():
            ui.Line(
                height=10,
                width=600,
                style={"color": cl.black, "border_width": 0.5},
                alignment=ui.Alignment.TOP,
            )
            for index in range(15):
                ui.Line(
                    height=10,
                    width=600,
                    style={"color": cl.black, "border_width": 0.5},
                    alignment=ui.Alignment.TOP,
                )
            ui.Line(
                height=10,
                width=600,
                style={"color": cl.black, "border_width": 0.5},
                alignment=ui.Alignment.TOP,
            )

        with ui.Placer(offset_x=100, offset_y=10):
            ui.Button("moved 100px in X, and 10px in Y", width=0, height=20, name="placed")

        with ui.Placer(offset_x=200, offset_y=50):
            ui.Button("moved 200px X , and 50 Y", width=0, height=0)

        def set_text(widget, text):
            widget.text = text

        with ui.Placer(draggable=True, offset_x=300, offset_y=100):
            ui.Circle(radius=50, width=50, height=50, size_policy=ui.CircleSizePolicy.STRETCH, name="placed")

        placer = ui.Placer(draggable=True, drag_axis=ui.Axis.Y, offset_x=400, offset_y=120)

        with placer:
            with ui.ZStack(width=180, height=40):
                ui.Rectangle(name="placed")
                with ui.HStack(spacing=5):
                    ui.Circle(
                        radius=3,
                        width=15,
                        size_policy=ui.CircleSizePolicy.FIXED,
                        style={"background_color": cl.white},
                    )
                    ui.Label("UP / Down", style={"color": cl.white, "font_size": 16.0})
                    offset_label = ui.Label("120", style={"color": cl.white})

        placer.set_offset_y_changed_fn(lambda o: set_text(offset_label, str(o)))
```

The following example shows the way to interact between three Placers to create a resizable rectangle's body, left handle and right handle. The rectangle can be moved on X-axis and can be resized with small orange handles.

When multiple widgets fire the callbacks simultaneously, it's possible to collect the event data and process them one frame later using asyncio.

```execute 200
import asyncio
import omni.kit.app
from omni.ui import color as cl

def placer_track(self, id):
    # Initial size
    BEGIN = 50 + 100 * id
    END = 120 + 100 * id

    HANDLE_WIDTH = 10

    class EditScope:
        """The class to avoid circular event calling"""

        def __init__(self):
            self.active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, type, value, traceback):
            self.active = False

        def __bool__(self):
            return not self.active

    class DoLater:
        """A helper to collect data and process it one frame later"""

        def __init__(self):
            self.__task = None
            self.__data = []

        def do(self, data):
            # Collect data
            self.__data.append(data)

            # Update in the next frame. We need it because we want to accumulate the affected prims
            if self.__task is None or self.__task.done():
                self.__task = asyncio.ensure_future(self.__delayed_do())

        async def __delayed_do(self):
            # Wait one frame
            await omni.kit.app.get_app().next_update_async()

            print(f"In the previous frame the user clicked the rectangles: {self.__data}")
            self.__data.clear()

    self.edit = EditScope()
    self.dolater = DoLater()

    def start_moved(start, body, end):
        if not self.edit:
            # Something already edits it
            return

        with self.edit:
            body.offset_x = start.offset_x
            rect.width = ui.Pixel(end.offset_x - start.offset_x + HANDLE_WIDTH)

    def body_moved(start, body, end, rect):
        if not self.edit:
            # Something already edits it
            return

        with self.edit:
            start.offset_x = body.offset_x
            end.offset_x = body.offset_x + rect.width.value - HANDLE_WIDTH

    def end_moved(start, body, end, rect):
        if not self.edit:
            # Something already edits it
            return

        with self.edit:
            body.offset_x = start.offset_x
            rect.width = ui.Pixel(end.offset_x - start.offset_x + HANDLE_WIDTH)

    with ui.ZStack(height=30):
        # Body
        body = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=BEGIN)
        with body:
            rect = ui.Rectangle(width=END - BEGIN + HANDLE_WIDTH)
            rect.set_mouse_pressed_fn(lambda x, y, b, m, id=id: self.dolater.do(id))
        # Left handle
        start = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=BEGIN)
        with start:
            ui.Rectangle(width=HANDLE_WIDTH, style={"background_color": cl("#FF660099")})
        # Right handle
        end = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=END)
        with end:
            ui.Rectangle(width=HANDLE_WIDTH, style={"background_color": cl("#FF660099")})

    # Connect them together
    start.set_offset_x_changed_fn(lambda _, s=start, b=body, e=end: start_moved(s, b, e))
    body.set_offset_x_changed_fn(lambda _, s=start, b=body, e=end, r=rect: body_moved(s, b, e, r))
    end.set_offset_x_changed_fn(lambda _, s=start, b=body, e=end, r=rect: end_moved(s, b, e, r))

ui.Spacer(height=5)
with ui.ZStack():
    placer_track(self, 0)
    placer_track(self, 1)
ui.Spacer(height=5)
```

It's possible to set `offset_x` and `offset_y` in percentages. It allows stacking the children to the proportions of the parent widget. If the parent size is changed, then the offset is updated accordingly.
```execute 200
from omni.ui import color as cl

# The size of the rectangle
SIZE = 20.0

with ui.ZStack(height=200):
    # Background
    ui.Rectangle(style={"background_color": cl(0.6)})

    # Small rectangle
    p = ui.Percent(50)
    placer = ui.Placer(draggable=True, offset_x=p, offset_y=p)
    with placer:
        ui.Rectangle(width=SIZE, height=SIZE)

def clamp_x(offset):
    if offset.value < 0:
        placer.offset_x = ui.Percent(0)
    max_per = 100.0 - SIZE / placer.computed_width * 100.0
    if offset.value > max_per:
        placer.offset_x = ui.Percent(max_per)

def clamp_y(offset):
    if offset.value < 0:
        placer.offset_y = ui.Percent(0)
    max_per = 100.0 - SIZE / placer.computed_height * 100.0
    if offset.value > max_per:
        placer.offset_y = ui.Percent(max_per)

# Callbacks
placer.set_offset_x_changed_fn(clamp_x)
placer.set_offset_y_changed_fn(clamp_y)
```



<a name="widgets"></a>

================================================================================

# Widgets Section

*Source: widgets.md*

================================================================================

# Widgets

## Label
Labels are used everywhere in omni.ui. They are text only objects.

Here is a list of styles you can customize on Label:
> color (color): the color of the text
> font_size (float): the size of the text
> margin (float): the distance between the label and the parent widget defined boundary
> margin_width (float): the width distance between the label and the parent widget defined boundary
> margin_height (float): the height distance between the label and the parent widget defined boundary
> alignment (enum): defines how the label is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Here are a few examples of labels:

```execute 200
from omni.ui import color as cl
ui.Label("this is a simple label", style={"color":cl.red, "margin": 5})
```

```execute 200
from omni.ui import color as cl
ui.Label("label with alignment", style={"color":cl.green, "margin": 5}, alignment=ui.Alignment.CENTER)
```

Notice that alignment could be either a property or a style.
```execute 200
from omni.ui import color as cl
label_style = {
    "Label": {"font_size": 20, "color": cl.blue, "alignment":ui.Alignment.RIGHT, "margin_height": 20}
    }
ui.Label("Label with style", style=label_style)
```

When the text of the Label is too long, it can be elided by `...`:
```execute 200
from omni.ui import color as cl
ui.Label(
            "Label can be elided: Lorem ipsum dolor "
            "sit amet, consectetur adipiscing elit, sed do "
            "eiusmod tempor incididunt ut labore et dolore "
            "magna aliqua. Ut enim ad minim veniam, quis "
            "nostrud exercitation ullamco laboris nisi ut "
            "aliquip ex ea commodo consequat. Duis aute irure "
            "dolor in reprehenderit in voluptate velit esse "
            "cillum dolore eu fugiat nulla pariatur. Excepteur "
            "sint occaecat cupidatat non proident, sunt in "
            "culpa qui officia deserunt mollit anim id est "
            "laborum.",
            style={"color":cl.white},
            elided_text=True,
        )
```

## CheckBox
A CheckBox is an option button that can be switched on (checked) or off (unchecked). Checkboxes are typically used to represent features in an application that can be enabled or disabled without affecting others.

The checkbox is implemented using the model-delegate-view pattern. The model is the central component of this system. It is the application's dynamic data structure independent of the widget. It directly manages the data, logic and rules of the checkbox. If the model is not specified, the simple one is created automatically when the object is constructed.

Here is a list of styles you can customize on Line:
> color (color): the color of the tick
> background_color (color): the background color of the check box
> font_size: the size of the tick
> border_radius (float): the radius of the corner angle if the user wants  to round the check box.
> border_width (float): the size of the check box border
> secondary_background_color (color): the color of the check box border

Default checkbox
```execute 200
with ui.HStack(width=0, spacing=5):
    ui.CheckBox().model.set_value(True)
    ui.CheckBox()
    ui.Label("Default")
```

Disabled checkbox:
```execute 200
with ui.HStack(width=0, spacing=5):
    ui.CheckBox(enabled=False).model.set_value(True)
    ui.CheckBox(enabled=False)
    ui.Label("Disabled")
```

In the following example, the models of two checkboxes are connected, and if one checkbox is changed, it makes another checkbox change as well.

```execute 200
from omni.ui import color as cl
with ui.HStack(width=0, spacing=5):
    # Create two checkboxes
    style = {"CheckBox":{
        "color": cl.white, "border_radius": 0, "background_color": cl("#ff5555"), "font_size": 30}}
    first = ui.CheckBox(style=style)
    second = ui.CheckBox(style=style)

    # Connect one to another
    first.model.add_value_changed_fn(lambda a, b=second: b.model.set_value(not a.get_value_as_bool()))
    second.model.add_value_changed_fn(lambda a, b=first: b.model.set_value(not a.get_value_as_bool()))

    # Set the first one to True
    first.model.set_value(True)

    ui.Label("One of two")
```

In the following example, that is a bit more complicated, only one checkbox can be enabled.
```execute 200
from omni.ui import color as cl
style = {"CheckBox":{
    "color": cl("#ff5555"), "border_radius": 5, "background_color": cl(0.35), "font_size": 20}}
with ui.HStack(width=0, spacing=5):
    # Create two checkboxes
    first = ui.CheckBox(style=style)
    second = ui.CheckBox(style=style)
    third = ui.CheckBox(style=style)

    def like_radio(model, first, second):
        """Turn on the model and turn off two checkboxes"""
        if model.get_value_as_bool():
            model.set_value(True)
            first.model.set_value(False)
            second.model.set_value(False)

    # Connect one to another
    first.model.add_value_changed_fn(lambda a, b=second, c=third: like_radio(a, b, c))
    second.model.add_value_changed_fn(lambda a, b=first, c=third: like_radio(a, b, c))
    third.model.add_value_changed_fn(lambda a, b=first, c=second: like_radio(a, b, c))

    # Set the first one to True
    first.model.set_value(True)

    ui.Label("Almost like radio box")
```

## ComboBox
The ComboBox widget is a combination of a button and a drop-down list. A ComboBox is a selection widget that displays the current item and can pop up a list of selectable items.

Here is a list of styles you can customize on ComboBox:
> color (color): the color of the combo box text and the arrow of the drop-down button
> background_color (color): the background color of the combo box
> secondary_color (color): the color of the drop-down button's background
> selected_color (color): the selected highlight color of option items
> secondary_selected_color (color): the color of the option item text
> font_size (float): the size of the text
> border_radius (float): the border radius if the user wants  to round the ComboBox
> padding (float): the overall padding of the ComboBox. If padding is defined, padding_height and padding_width will have no effects.
> padding_height (float): the width padding of the drop-down list
> padding_width (float): the height padding of the drop-down list
> secondary_padding (float): the height padding between the ComboBox and options

Default ComboBox:

```execute 200
ui.ComboBox(1, "Option 1", "Option 2", "Option 3")
```

ComboBox with style
```execute 200
from omni.ui import color as cl
style={"ComboBox":{
    "color": cl.red,
    "background_color": cl(0.15),
    "secondary_color": cl("#1111aa"),
    "selected_color": cl.green,
    "secondary_selected_color": cl.white,
    "font_size": 15,
    "border_radius": 20,
    "padding_height": 2,
    "padding_width": 20,
    "secondary_padding": 30,
}}
with ui.VStack():
    ui.ComboBox(1, "Option 1", "Option 2", "Option 3", style=style)
    ui.Spacer(height=20)
```


The following example demonstrates how to add items to the ComboBox.
```execute 200
editable_combo = ui.ComboBox()
ui.Button(
    "Add item to combo",
    clicked_fn=lambda m=editable_combo.model: m.append_child_item(
        None, ui.SimpleStringModel("Hello World")),
)
```

The minimal model implementation to have more flexibility of the data. It requires holding the value models and reimplementing two methods: `get_item_children` and `get_item_value_model`.
```execute 200
class MinimalItem(ui.AbstractItem):
    def __init__(self, text):
        super().__init__()
        self.model = ui.SimpleStringModel(text)

class MinimalModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()

        self._current_index = ui.SimpleIntModel()
        self._current_index.add_value_changed_fn(
            lambda a: self._item_changed(None))

        self._items = [
            MinimalItem(text)
            for text in ["Option 1", "Option 2"]
        ]

    def get_item_children(self, item):
        return self._items

    def get_item_value_model(self, item, column_id):
        if item is None:
            return self._current_index
        return item.model

self._minimal_model = MinimalModel()
with ui.VStack():
    ui.ComboBox(self._minimal_model, style={"font_size": 22})
    ui.Spacer(height=10)
```

The example of communication between widgets. Type anything in the field and it will appear in the combo box.
```execute 200
editable_combo = None

class StringModel(ui.SimpleStringModel):
    '''
    String Model activated when editing is finished.
    Adds item to combo box.
    '''
    def __init__(self):
        super().__init__("")

    def end_edit(self):
        combo_model = editable_combo.model
        # Get all the options ad list of strings
        all_options = [
            combo_model.get_item_value_model(child).as_string
            for child in combo_model.get_item_children()
        ]

        # Get the current string of this model
        fieldString = self.as_string
        if fieldString:
            if fieldString in all_options:
                index = all_options.index(fieldString)
            else:
                # It's a new string in the combo box
                combo_model.append_child_item(
                    None,
                    ui.SimpleStringModel(fieldString)
                )
                index = len(all_options)

            combo_model.get_item_value_model().set_value(index)

self._field_model = StringModel()

def combo_changed(combo_model, item):
    all_options = [
        combo_model.get_item_value_model(child).as_string
        for child in combo_model.get_item_children()
    ]
    current_index = combo_model.get_item_value_model().as_int
    self._field_model.as_string = all_options[current_index]

with ui.HStack():
    ui.StringField(self._field_model)
    editable_combo = ui.ComboBox(width=0, arrow_only=True)
    editable_combo.model.add_item_changed_fn(combo_changed)
```

## TreeView
TreeView is a widget that presents a hierarchical view of information. Each item can have a number of subitems. An indentation often visualizes this in a list. An item can be expanded to reveal subitems, if any exist, and collapsed to hide subitems.

TreeView can be used in file manager applications, where it allows the user to navigate the file system directories. They are also used to present hierarchical data, such as the scene object hierarchy.

TreeView uses a model-delegate-view pattern to manage the relationship between data and the way it is presented. The separation of functionality gives developers greater flexibility to customize the presentation of items and provides a standard interface to allow a wide range of data sources to be used with other widgets.

Here is a list of styles you can customize on TreeView:
> background_color (color): specifically used when Treeview item is selected. It indicates the background color of the TreeView item when selected.
> background_selected_color (color): the hover color of the TreeView selected item. The actual selected color of the TreeView selected item should be defined by the "background_color" of ":selected".
> secondary_color (color): if the TreeView has more than one column, this is the color of the line which divides the columns.
> secondary_selected_color (color): if the TreeView has more than one column and if the column is resizable, this is the color of the line which divides the columns when hovered over the divider.
> border_color (color): the border color of the TreeView item when hovered. During drag and drop of the Treeview item, it is also the border color of the Treeview item border which indicates where the dragged item targets to drop.
> border_width (float): specifically used when Treeview item drag and dropped. Thickness of the Treeview item border which indicates where the dragged item targets to drop.

Here is a list of styles you can customize on TreeView.Item:
> margin (float): the margin between TreeView items. This will be overridden by the value of margin_width or margin_height
> margin_width (float): the margin width between TreeView items
> margin_height (float): the margin height between TreeView items
> color (color): the text color of the TreeView items
> font_size (float): the text size of the TreeView items

The following example demonstrates how to make a single level tree appear like a simple list.
```execute 200
import omni.ui as ui
from omni.ui import color as cl
style = {
    "TreeView":
    {
        "background_selected_color": cl("#55FF9033"),
        "secondary_color": cl.green,
        "secondary_selected_color": cl.purple,
        "border_color": cl.red,
    },
    "TreeView:selected": {"background_color": cl("#888888")},
    "TreeView.Item":
    {
        "margin": 4,
        "margin_width": 10,
        "color": cl("#AAAAAA"),
        "font_size": 13,
    },
    "TreeView.Item:selected": {"color": cl.pink},
}
class CommandItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

class CommandModel(ui.AbstractItemModel):
    """
    Represents the list of commands registered in Kit.
    It is used to make a single level tree appear like a simple list.
    """

    def __init__(self):
        super().__init__()

        self._commands = []
        try:
            import omni.kit.commands
        except ModuleNotFoundError:
            return

        omni.kit.commands.subscribe_on_change(self._commands_changed)
        self._commands_changed()

    def _commands_changed(self):
        """Called by subscribe_on_change"""
        self._commands = []
        import omni.kit.commands

        for cmd_list in omni.kit.commands.get_commands().values():
            for k in cmd_list.values():
                self._commands.append(CommandItem(k.__name__))

        self._item_changed(None)

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._commands

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 2

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel.
        """
        if item and isinstance(item, CommandItem):
            return item.name_model

with ui.ScrollingFrame(
    height=400,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    self._command_model = CommandModel()
    tree_view = ui.TreeView(
        self._command_model,
        root_visible=False,
        header_visible=False,
        columns_resizable=True,
        column_widths=[350, 350],
        style_type_name_override="TreeView",
        style=style,
    )
```

The following example demonstrates reordering with drag and drop. You can drag one item of the TreeView and move it to the position where you want to insert the item.
```execute 200
from omni.ui import color as cl

style = {
    "TreeView":
    {
        "border_color": cl.red,
        "border_width": 2,
    },
    "TreeView.Item": {"margin": 4},
}

class ListItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

    def __repr__(self):
        return f'"{self.name_model.as_string}"'

class ListModel(ui.AbstractItemModel):
    """
    Represents the model for lists. It's very easy to initialize it
    with any string list:
        string_list = ["Hello", "World"]
        model = ListModel(*string_list)
        ui.TreeView(model)
    """

    def __init__(self, *args):
        super().__init__()
        self._children = [ListItem(t) for t in args]

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 1

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel.
        """
        return item.name_model

class ListModelWithReordering(ListModel):
    """
    Represents the model for the list with the ability to reorder the
    list with drag and drop.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def get_drag_mime_data(self, item):
        """Returns Multipurpose Internet Mail Extensions (MIME) data for be able to drop this item somewhere"""
        # As we don't do Drag and Drop to the operating system, we return the string.
        return item.name_model.as_string

    def drop_accepted(self, target_item, source, drop_location=-1):
        """Reimplemented from AbstractItemModel. Called to highlight target when drag and drop."""
        # If target_item is None, it's the drop to root. Since it's
        # list model, we support reorganization of root only and we
        # don't want to create a tree.
        return not target_item and drop_location >= 0

    def drop(self, target_item, source, drop_location=-1):
        """Reimplemented from AbstractItemModel. Called when dropping something to the item."""
        try:
            source_id = self._children.index(source)
        except ValueError:
            # Not in the list. This is the source from another model.
            return

        if source_id == drop_location:
            # Nothing to do
            return

        self._children.remove(source)

        if drop_location > len(self._children):
            # Drop it to the end
            self._children.append(source)
        else:
            if source_id < drop_location:
                # Because when we removed source, the array became shorter
                drop_location = drop_location - 1

            self._children.insert(drop_location, source)

        self._item_changed(None)

with ui.ScrollingFrame(
    height=150,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    self._list_model = ListModelWithReordering("Simplest", "List", "Model", "With", "Reordering")
    tree_view = ui.TreeView(
        self._list_model,
        root_visible=False,
        header_visible=False,
        style_type_name_override="TreeView",
        style=style,
        drop_between_items=True,
    )

```

The following example demonstrates the ability to edit TreeView items.
```execute 200
from omni.ui import color as cl
class FloatModel(ui.AbstractValueModel):
    """An example of custom float model that can be used for formatted string output"""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def get_value_as_float(self):
        """Reimplemented get float"""
        return self._value or 0.0

    def get_value_as_string(self):
        """Reimplemented get string"""
        # This string goes to the field.
        if self._value is None:
            return ""

        # General format. This prints the number as a fixed-point
        # number, unless the number is too large, in which case it
        # switches to 'e' exponent notation.
        return "{0:g}".format(self._value)

    def set_value(self, value):
        """Reimplemented set"""
        try:
            value = float(value)
        except ValueError:
            value = None
        if value != self._value:
            # Tell the widget that the model is changed
            self._value = value
            self._value_changed()

class NameValueItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text, value):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)
        self.value_model = FloatModel(value)

    def __repr__(self):
        return f'"{self.name_model.as_string} {self.value_model.as_string}"'

class NameValueModel(ui.AbstractItemModel):
    """
    Represents the model for name-value tables. It's very easy to initialize it
    with any string-float list:
        my_list = ["Hello", 1.0, "World", 2.0]
        model = NameValueModel(*my_list)
        ui.TreeView(model)
    """

    def __init__(self, *args):
        super().__init__()
        # ["Hello", 1.0, "World", 2.0"] -> [("Hello", 1.0), ("World", 2.0)]
        regrouped = zip(*(iter(args),) * 2)
        self._children = [NameValueItem(*t) for t in regrouped]

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 2

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel for the first column
        and SimpleFloatModel for the second column.
        """
        return item.value_model if column_id == 1 else item.name_model

class EditableDelegate(ui.AbstractItemDelegate):
    """
    Delegate is the representation layer. TreeView calls the methods
    of the delegate to create custom widgets for each item.
    """

    def __init__(self):
        super().__init__()
        self.subscription = None

    def build_branch(self, model, item, column_id, level, expanded):
        """Create a branch widget that opens or closes subtree"""
        pass

    def build_widget(self, model, item, column_id, level, expanded):
        """Create a widget per column per item"""
        stack = ui.ZStack(height=20)
        with stack:
            value_model = model.get_item_value_model(item, column_id)
            label = ui.Label(value_model.as_string)
            if column_id == 1:
                field = ui.FloatField(value_model, visible=False)
            else:
                field = ui.StringField(value_model, visible=False)

        # Start editing when double clicked
        stack.set_mouse_double_clicked_fn(lambda x, y, b, m, f=field, l=label: self.on_double_click(b, f, l))

    def on_double_click(self, button, field, label):
        """Called when the user double-clicked the item in TreeView"""
        if button != 0:
            return

        # Make Field visible when double clicked
        field.visible = True
        field.focus_keyboard()
        # When editing is finished (enter pressed of mouse clicked outside of the viewport)
        self.subscription = field.model.subscribe_end_edit_fn(
            lambda m, f=field, l=label: self.on_end_edit(m, f, l)
        )

    def on_end_edit(self, model, field, label):
        """Called when the user is editing the item and pressed Enter or clicked outside of the item"""
        field.visible = False
        label.text = model.as_string
        self.subscription = None

with ui.ScrollingFrame(
    height=100,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    self._name_value_model = NameValueModel("First", 0.2, "Second", 0.3, "Last", 0.4)
    self._name_value_delegate = EditableDelegate()
    tree_view = ui.TreeView(
        self._name_value_model,
        delegate=self._name_value_delegate,
        root_visible=False,
        header_visible=False,
        style_type_name_override="TreeView",
        style={"TreeView.Item": {"margin": 4}},
    )
```

This is an example of async filling the TreeView model. It's collecting only as many as it's possible of USD prims for 0.016s and waits for the next frame, so the UI is not locked even if the USD Stage is extremely big.
To play with it, create several materials in the stage or open a stage which contains materials, click "Traverse All" or "Stop Traversing".

```execute 200
import asyncio
import time
from omni.ui import color as cl
class ListItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

    def __repr__(self):
        return f'"{self.name_model.as_string}"'

class ListModel(ui.AbstractItemModel):
    """
    Represents the model for lists. It's very easy to initialize it
    with any string list:
        string_list = ["Hello", "World"]
        model = ListModel(*string_list)
        ui.TreeView(model)
    """

    def __init__(self, *args):
        super().__init__()
        self._children = [ListItem(t) for t in args]

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 1

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel.
        """
        return item.name_model

class AsyncQueryModel(ListModel):
    """
    This is an example of async filling the TreeView model. It's
    collecting only as many as it's possible of USD prims for 0.016s
    and waits for the next frame, so the UI is not locked even if the
    USD Stage is extremely big.
    """

    def __init__(self):
        super().__init__()
        self._stop_event = None

    def destroy(self):
        self.stop()

    def stop(self):
        """Stop traversing the stage"""
        if self._stop_event:
            self._stop_event.set()

    def reset(self):
        """Traverse the stage and keep materials"""
        self.stop()
        self._stop_event = asyncio.Event()

        self._children.clear()
        self._item_changed(None)

        asyncio.ensure_future(self.__get_all(self._stop_event))

    def __push_collected(self, collected):
        """Add given array to the model"""
        for c in collected:
            self._children.append(c)
        self._item_changed(None)

    async def __get_all(self, stop_event):
        """Traverse the stage portion at time, so it doesn't freeze"""
        stop_event.clear()

        start_time = time.time()
        # The widget will be updated not faster than 60 times a second
        update_every = 1.0 / 60.0

        import omni.usd
        from pxr import Usd
        from pxr import UsdShade

        context = omni.usd.get_context()
        stage = context.get_stage()
        if not stage:
            return

        # Buffer to keep the portion of the items before sending to the
        # widget
        collected = []

        for p in stage.Traverse(
            Usd.TraverseInstanceProxies(Usd.PrimIsActive and Usd.PrimIsDefined and Usd.PrimIsLoaded)
        ):
            if stop_event.is_set():
                break

            if p.IsA(UsdShade.Material):
                # Collect materials only
                collected.append(ListItem(str(p.GetPath())))

            elapsed_time = time.time()

            # Loop some amount of time so fps will be about 60FPS
            if elapsed_time - start_time > update_every:
                start_time = elapsed_time

                # Append the portion and update the widget
                if collected:
                    self.__push_collected(collected)
                    collected = []

                # Wait one frame to let other tasks go
                await omni.kit.app.get_app().next_update_async()

        self.__push_collected(collected)

try:
    import omni.usd
    from pxr import Usd

    usd_available = True
except ModuleNotFoundError:
    usd_available = False

if usd_available:
    with ui.ScrollingFrame(
        height=200,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    ):
        self._async_query_model = AsyncQueryModel()
        ui.TreeView(
            self._async_query_model,
            root_visible=False,
            header_visible=False,
            style_type_name_override="TreeView",
            style={"TreeView.Item": {"margin": 4}},
        )

    _loaded_label = ui.Label("Press Button to Load Materials", name="text")

    with ui.HStack():
        ui.Button("Traverse All", clicked_fn=self._async_query_model.reset)
        ui.Button("Stop Traversing", clicked_fn=self._async_query_model.stop)

    def _item_changed(model, item):
        if item is None:
            count = len(model._children)
            _loaded_label.text = f"{count} Materials Traversed"

    self._async_query_sub = self._async_query_model.subscribe_item_changed_fn(_item_changed)
```


<a name="buttons"></a>

================================================================================

# Buttons Section

*Source: buttons.md*

================================================================================

# Buttons and Images

## Common Styling for Buttons and Images
Here is a list of common style you can customize on Buttons and Images:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary

## Button
The Button widget provides a command button. Click a button to execute a command. The command button is perhaps the most commonly used widget in any graphical user interface. It is rectangular and typically displays a text label or image describing its action.

Except the common style for Buttons and Images, here is a list of styles you can customize on Button:
> background_color (color): the background color of the button
> padding (float): the distance between the content widgets (e.g. Image or Label) and the border of the button
> stack_direction (enum): defines how the content widgets (e.g. Image or Label) on the button are placed.

There are 6 types of stack_directions supported
* ui.Direction.TOP_TO_BOTTOM : layout from top to bottom
* ui.Direction.BOTTOM_TO_TOP : layout from bottom to top
* ui.Direction.LEFT_TO_RIGHT : layout from left to right
* ui.Direction.RIGHT_TO_LEFT : layout from right to left
* ui.Direction.BACK_TO_FRONT : layout from back to front
* ui.Direction.FRONT_TO_BACK : layout from front to back

To control the style of the button content, you can customize `Button.Image` when image on button and `Button.Label` when text on button.

Here is an example showing a list of buttons with different types of the stack directions:
```execute 200
from omni.ui import color as cl
direction_flags = {
    "ui.Direction.TOP_TO_BOTTOM": ui.Direction.TOP_TO_BOTTOM,
    "ui.Direction.BOTTOM_TO_TOP": ui.Direction.BOTTOM_TO_TOP,
    "ui.Direction.LEFT_TO_RIGHT": ui.Direction.LEFT_TO_RIGHT,
    "ui.Direction.RIGHT_TO_LEFT": ui.Direction.RIGHT_TO_LEFT,
    "ui.Direction.BACK_TO_FRONT": ui.Direction.BACK_TO_FRONT,
    "ui.Direction.FRONT_TO_BACK": ui.Direction.FRONT_TO_BACK,
}

with ui.ScrollingFrame(
    height=50,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in direction_flags.items():
            button_style = {"Button": {"stack_direction": value}}
            ui_button = ui.Button(
                                key,
                                image_url="resources/icons/Nav_Flymode.png",
                                image_width=24,
                                height=40,
                                style=button_style
                            )
```

Here is an example of two buttons. Pressing the second button makes the name of the first button longer. And press the first button makes the name of itself shorter:
```execute 200
from omni.ui import color as cl
style_system = {
    "Button": {
        "background_color": cl(0.85),
        "border_color": cl.yellow,
        "border_width": 2,
        "border_radius": 5,
        "padding": 5,
    },
    "Button.Label": {"color": cl.red, "font_size": 17},
    "Button:hovered": {"background_color": cl("#E5F1FB"), "border_color": cl("#0078D7"), "border_width": 2.0},
    "Button:pressed": {"background_color": cl("#CCE4F7"), "border_color": cl("#005499"), "border_width": 2.0},
}

def make_longer_text(button):
    """Set the text of the button longer"""
    button.text = "Longer " + button.text

def make_shorter_text(button):
    """Set the text of the button shorter"""
    splitted = button.text.split(" ", 1)
    button.text = splitted[1] if len(splitted) > 1 else splitted[0]

with ui.HStack(style=style_system):
    btn_with_text = ui.Button("Text", width=0)
    ui.Button("Press me", width=0, clicked_fn=lambda b=btn_with_text: make_longer_text(b))
    btn_with_text.set_clicked_fn(lambda b=btn_with_text: make_shorter_text(b))
```

Here is an example where you can tweak most of the Button's style and see the results:
```execute 200
from omni.ui import color as cl
style = {
    "Button": {"stack_direction": ui.Direction.TOP_TO_BOTTOM},
    "Button.Image": {
        "color": cl("#99CCFF"),
        "image_url": "resources/icons/Learn_128.png",
        "alignment": ui.Alignment.CENTER,
    },
    "Button.Label": {"alignment": ui.Alignment.CENTER},
}

def direction(model, button, style=style):
    value = model.get_item_value_model().get_value_as_int()
    direction = (
        ui.Direction.TOP_TO_BOTTOM,
        ui.Direction.BOTTOM_TO_TOP,
        ui.Direction.LEFT_TO_RIGHT,
        ui.Direction.RIGHT_TO_LEFT,
        ui.Direction.BACK_TO_FRONT,
        ui.Direction.FRONT_TO_BACK,
    )[value]
    style["Button"]["stack_direction"] = direction
    button.set_style(style)

def align(model, button, image, style=style):
    value = model.get_item_value_model().get_value_as_int()
    alignment = (
        ui.Alignment.LEFT_TOP,
        ui.Alignment.LEFT_CENTER,
        ui.Alignment.LEFT_BOTTOM,
        ui.Alignment.CENTER_TOP,
        ui.Alignment.CENTER,
        ui.Alignment.CENTER_BOTTOM,
        ui.Alignment.RIGHT_TOP,
        ui.Alignment.RIGHT_CENTER,
        ui.Alignment.RIGHT_BOTTOM,
    )[value]
    if image:
        style["Button.Image"]["alignment"] = alignment
    else:
        style["Button.Label"]["alignment"] = alignment
    button.set_style(style)

def layout(model, button, padding, style=style):
    if padding == 0:
        padding = "padding"
    elif padding == 1:
        padding = "margin"
    elif padding == 2:
        padding = "margin_width"
    else:
        padding = "margin_height"

    style["Button"][padding] = model.get_value_as_float()
    button.set_style(style)

def spacing(model, button):
    button.spacing = model.get_value_as_float()

button = ui.Button("Label", style=style, width=64, height=64)

with ui.HStack(width=ui.Percent(50)):
    ui.Label('"Button": {"stack_direction"}', name="text")
    options = (
        0,
        "TOP_TO_BOTTOM",
        "BOTTOM_TO_TOP",
        "LEFT_TO_RIGHT",
        "RIGHT_TO_LEFT",
        "BACK_TO_FRONT",
        "FRONT_TO_BACK",
    )
    model = ui.ComboBox(*options).model
    model.add_item_changed_fn(lambda m, i, b=button: direction(m, b))

alignment = (
    4,
    "LEFT_TOP",
    "LEFT_CENTER",
    "LEFT_BOTTOM",
    "CENTER_TOP",
    "CENTER",
    "CENTER_BOTTOM",
    "RIGHT_TOP",
    "RIGHT_CENTER",
    "RIGHT_BOTTOM",
)
with ui.HStack(width=ui.Percent(50)):
    ui.Label('"Button.Image": {"alignment"}', name="text")
    model = ui.ComboBox(*alignment).model
    model.add_item_changed_fn(lambda m, i, b=button: align(m, b, 1))

with ui.HStack(width=ui.Percent(50)):
    ui.Label('"Button.Label": {"alignment"}', name="text")
    model = ui.ComboBox(*alignment).model
    model.add_item_changed_fn(lambda m, i, b=button: align(m, b, 0))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("padding", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 0))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("margin", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 1))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("margin_width", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 2))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("margin_height", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 3))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("Button.spacing", name="text")
    model = ui.FloatSlider(min=0, max=50).model
    model.add_value_changed_fn(lambda m, b=button: spacing(m, b))
```

## Radio Button
RadioButton is the widget that allows the user to choose only one from a predefined set of mutually exclusive options.

RadioButtons are arranged in collections of two or more buttons within a RadioCollection, which is the central component of the system and controls the behavior of all the RadioButtons in the collection.

Except the common style for Buttons and Images, here is a list of styles you can customize on RadioButton:
> background_color (color): the background color of the RadioButton
> padding (float): the distance between the the RadioButton content widget (e.g. Image) and the RadioButton border

To control the style of the button image, you can customize `RadioButton.Image`. For example RadioButton.Image's image_url defines the image when it's not checked. You can define the image for checked status with `RadioButton.Image:checked` style.

Here is an example of RadioCollection which contains 5 RadioButtons with style. Also there is an IntSlider which shares the model with the RadioCollection, so that when RadioButton value or the IntSlider value changes, the other one will update too.

```execute 200
from omni.ui import color as cl
style = {
            "RadioButton": {
                "background_color": cl.cyan,
                "margin_width": 2,
                "padding": 1,
                "border_radius": 0,
                "border_color": cl.white,
                "border_width": 1.0},
            "RadioButton.Image": {
                "image_url": f"../exts/omni.kit.documentation.ui.style/icons/radio_off.svg",
            },
            "RadioButton.Image:checked": {
                "image_url": f"../exts/omni.kit.documentation.ui.style/icons/radio_on.svg"},
        }

collection = ui.RadioCollection()
for i in range(5):
    with ui.HStack(style=style):
        ui.RadioButton(radio_collection=collection, width=30, height=30)
        ui.Label(f"Option {i}", name="text")

ui.IntSlider(collection.model, min=0, max=4)
```

## ToolButton
ToolButton is functionally similar to Button, but provides a model that determines if the button is checked. This button toggles between checked (on) and unchecked (off) when the user clicks it.

Here is an example of a ToolButton:

```execute 200
def update_label(model, label):
    checked = model.get_value_as_bool()
    label.text = f"The check status button is {checked}"

with ui.VStack(spacing=5):
    model = ui.ToolButton(text="click", name="toolbutton", width=100).model
    checked = model.get_value_as_bool()
    label = ui.Label(f"The check status button is {checked}")
    model.add_value_changed_fn(lambda m, l=label: update_label(m, l))

```

## ColorWidget
The ColorWidget is a button that displays the color from the item model and can open a picker window. The color dialog's function is to allow users to choose color.

Except the common style for Buttons and Images, here is a list of styles you can customize on ColorWidget:
> background_color (color): the background color of the tooltip widget when hover over onto the ColorWidget
> color (color): the text color of the tooltip widget when hover over onto the ColorWidget

Here is an example of a ColorWidget with three FloatFields. The ColorWidget model is shared with the FloatFields so that users can click and edit the field value to change the ColorWidget's color, and the value change of the ColorWidget will also reflect in the value change of the FloatFields.

```execute 200
from omni.ui import color as cl
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0, style={"ColorWidget":{
        "border_width": 2,
        "border_color": cl.white,
        "border_radius": 4,
        "color": cl.pink,
        "margin": 2
    }}).model
    for item in color_model.get_item_children():
        component = color_model.get_item_value_model(item)
        ui.FloatField(component)
```

Here is an example of a ColorWidget with three FloatDrags. The ColorWidget model is shared with the FloatDrags so that users can drag the field value to change the color, and the value change of the ColorWidget will also reflect in the value change of the FloatDrags.

```execute 200
from omni.ui import color as cl
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(0.125, 0.25, 0.5, width=0, height=0, style={
        "background_color": cl.pink
    }).model
    for item in color_model.get_item_children():
        component = color_model.get_item_value_model(item)
        ui.FloatDrag(component, min=0, max=1)
```

Here is an example of a ColorWidget with a ComboBox. The ColorWidget model is shared with the ComboBox. Only the value change of the ColorWidget will reflect in the value change of the ComboBox.

```execute 200
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0).model
    ui.ComboBox(color_model)
```

Here is an interactive example with USD. You can create a Mesh in the Stage. Choose `Pixar Storm` as the render. Select the mesh and use this ColorWidget to change the color of the mesh. You can use `Ctrl+z` for undoing and `Ctrl+y` for redoing.

```execute 200
import weakref
import omni.kit.commands
from omni.usd.commands import UsdStageHelper
from pxr import UsdGeom
from pxr import Gf
import omni.usd
from carb.eventdispatcher import get_eventdispatcher

class SetDisplayColorCommand(omni.kit.commands.Command, UsdStageHelper):
    """
    Change prim display color undoable **Command**. Unlike ChangePropertyCommand, it can undo property creation.

    Args:
        gprim (Gprim): Prim to change display color on.
        value: Value to change to.
        value: Value to undo to.
    """

    def __init__(self, gprim: UsdGeom.Gprim, color: any, prev: any):
        self._gprim = gprim
        self._color = color
        self._prev = prev

    def do(self):
        color_attr = self._gprim.CreateDisplayColorAttr()
        color_attr.Set([self._color])

    def undo(self):
        color_attr = self._gprim.GetDisplayColorAttr()
        if self._prev is None:
            color_attr.Clear()
        else:
            color_attr.Set([self._prev])

omni.kit.commands.register(SetDisplayColorCommand)

class FloatModel(ui.SimpleFloatModel):
    def __init__(self, parent):
        super().__init__()
        self._parent = weakref.ref(parent)

    def begin_edit(self):
        parent = self._parent()
        parent.begin_edit(None)

    def end_edit(self):
        parent = self._parent()
        parent.end_edit(None)


class USDColorItem(ui.AbstractItem):
    def __init__(self, model):
        super().__init__()
        self.model = model


class USDColorModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()

        # Create root model
        self._root_model = ui.SimpleIntModel()
        self._root_model.add_value_changed_fn(lambda a: self._item_changed(None))

        # Create three models per component
        self._items = [USDColorItem(FloatModel(self)) for i in range(3)]
        for item in self._items:
            item.model.add_value_changed_fn(lambda a, item=item: self._on_value_changed(item))

        # Omniverse contexts
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage_event_sub = get_eventdispatcher().observe_event(
            observer_name="omni.example.ui observer",
            event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.SELECTION_CHANGED),
            on_event=lambda _: self._on_selection_changed()
        )

        # Privates
        self._subscription = None
        self._gprim = None
        self._prev_color = None
        self._edit_mode_counter = 0

    def _on_selection_changed(self):
        """Called when the user changes the selection"""
        selection = self._selection.get_selected_prim_paths()
        stage = self._usd_context.get_stage()
        self._subscription = None
        self._gprim = None

        # When TC runs tests, it's possible that stage is None
        if selection and stage:
            self._gprim = UsdGeom.Gprim.Get(stage, selection[0])
            if self._gprim:
                color_attr = self._gprim.GetDisplayColorAttr()
                usd_watcher = omni.usd.get_watcher()
                self._subscription = usd_watcher.subscribe_to_change_info_path(
                    color_attr.GetPath(), self._on_usd_changed
                )

        # Change the widget color
        self._on_usd_changed()

    def _on_value_changed(self, item):
        """Called when the submodel is changed"""
        if not self._gprim:
            return

        if self._edit_mode_counter > 0:
            # Change USD only if we are in edit mode.
            color_attr = self._gprim.CreateDisplayColorAttr()

            color = Gf.Vec3f(
                self._items[0].model.get_value_as_float(),
                self._items[1].model.get_value_as_float(),
                self._items[2].model.get_value_as_float(),
            )

            color_attr.Set([color])

        self._item_changed(item)

    def _on_usd_changed(self, path=None):
        """Called with UsdWatcher when something in USD is changed"""
        color = self._get_current_color() or Gf.Vec3f(0.0)

        for i in range(len(self._items)):
            self._items[i].model.set_value(color[i])

    def _get_current_color(self):
        """Returns color of the current object"""
        if self._gprim:
            color_attr = self._gprim.GetDisplayColorAttr()
            if color_attr:
                color_array = color_attr.Get()
                if color_array:
                    return color_array[0]

    def get_item_children(self, item):
        """Reimplemented from the base class"""
        return self._items

    def get_item_value_model(self, item, column_id):
        """Reimplemented from the base class"""
        if item is None:
            return self._root_model
        return item.model

    def begin_edit(self, item):
        """
        Reimplemented from the base class.
        Called when the user starts editing.
        """
        if self._edit_mode_counter == 0:
            self._prev_color = self._get_current_color()

        self._edit_mode_counter += 1

    def end_edit(self, item):
        """
        Reimplemented from the base class.
        Called when the user finishes editing.
        """
        self._edit_mode_counter -= 1

        if not self._gprim or self._edit_mode_counter > 0:
            return

        color = Gf.Vec3f(
            self._items[0].model.get_value_as_float(),
            self._items[1].model.get_value_as_float(),
            self._items[2].model.get_value_as_float(),
        )

        omni.kit.commands.execute("SetDisplayColor", gprim=self._gprim, color=color, prev=self._prev_color)


with ui.HStack(spacing=5):
    ui.ColorWidget(USDColorModel(), width=0)
    ui.Label("Interactive ColorWidget with USD", name="text")
```

## Image
The Image type displays an image. The source of the image is specified as a URL using the source property. By default, specifying the width and height of the item makes the image to be scaled to fit that size. This behavior can be changed by setting the `fill_policy` property, allowing the image to be stretched or scaled instead. The property alignment controls how the scaled image is aligned in the parent defined space.

Except the common style for Buttons and Images, here is a list of styles you can customize on Image:
> image_url (str): the url path of the image source
> color (color): the overlay color of the image
> corner_flag (enum): defines which corner or corners to be rounded. The supported corner flags are the same as Rectangle since Image is eventually an image on top of a rectangle under the hood.
> fill_policy (enum): defines how the Image fills the rectangle.
There are three types of fill_policy
* ui.FillPolicy.STRETCH: stretch the image to fill the entire rectangle.
* ui.FillPolicy.PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.FillPolicy.PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary
> alignment (enum): defines how the image is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Default Image is scaled uniformly to fit without stretching or cropping (ui.FillPolicy.PRESERVE_ASPECT_FIT), and aligned to ui.Alignment.CENTER:
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source)
```

The image is stretched to fit and aligned to the left
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.STRETCH, alignment=ui.Alignment.LEFT_CENTER)
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the top
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_TOP)
```

The image is scaled uniformly to fit without cropping and aligned to the right. Notice the fill_policy and alignment are defined in style.
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, style={
        "Image": {
            "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
            "alignment": ui.Alignment.RIGHT_CENTER,
            "margin": 5}})
```

The image has rounded corners and an overlayed color. Note image_url is in the style dictionary.
```execute 200
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(style={"image_url": source, "border_radius": 10, "color": cl("#5eb3ff")})
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the bottom, with a blue border.
```execute 200
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(
        source,
        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_BOTTOM,
        style={"Image":{
            "border_width": 5,
            "border_color": cl("#1ab3ff"),
            "corner_flag": ui.CornerFlag.TOP,
            "border_radius": 15}})
```

The image is arranged in a HStack with different margin styles defined. Note image_url is in the style dict.
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(height=100):
    with ui.HStack(spacing =5, style={"Image":{'image_url': source}}):
        ui.Image()
        ui.Image(style={"Image":{"margin_height": 15}})
        ui.Image()
        ui.Image(style={"Image":{"margin_width": 20}})
        ui.Image()
        ui.Image(style={"Image":{"margin": 10}})
        ui.Image()
```

It's possible to set a different image per style state. And switch them depending on the mouse hovering, selection state, etc.
```execute 200

styles = [
    {
        "": {"image_url": "resources/icons/Nav_Walkmode.png"},
        ":hovered": {"image_url": "resources/icons/Nav_Flymode.png"},
    },
    {
        "": {"image_url": "resources/icons/Move_local_64.png"},
        ":hovered": {"image_url": "resources/icons/Move_64.png"},
    },
    {
        "": {"image_url": "resources/icons/Rotate_local_64.png"},
        ":hovered": {"image_url": "resources/icons/Rotate_global.png"},
    },
]

def set_image(model, image):
    value = model.get_item_value_model().get_value_as_int()
    image.set_style(styles[value])

with ui.Frame(height=80):
    with ui.VStack():
        image = ui.Image(width=64, height=64, style=styles[0])
        with ui.HStack(width=ui.Percent(50)):
            ui.Label("Select a texture to display", name="text")
            model = ui.ComboBox(0, "Navigation", "Move", "Rotate").model
            model.add_item_changed_fn(lambda m, i, im=image: set_image(m, im))
```

## ImageWithProvider
ImageWithProvider also displays an image just like Image. It is a much more advanced image widget. ImageWithProvider blocks until the image is loaded, Image doesn't block. Sometimes Image blinks because when the first frame is created, the image is not loaded. Users are recommended to use ImageWithProvider if the UI is updated pretty often. Because it doesn't blink when recreating.

It has the almost the same style list as Image, except the fill_policy has different enum values.
> fill_policy (enum): defines how the Image fills the rectangle.
There are three types of fill_policy
* ui.IwpFillPolicy.IWP_STRETCH: stretch the image to fill the entire rectangle.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary

The image source comes from `ImageProvider` which could be `ByteImageProvider`, `RasterImageProvider` or `VectorImageProvider`.

`RasterImageProvider` and `VectorImageProvider` are using image urls like Image. Here is an example taken from Image. Notice the fill_policy value difference.
```execute 200
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.ImageWithProvider(
        source,
        style={
            "ImageWithProvider": {
            "border_width": 5,
            "border_color": cl("#1ab3ff"),
            "corner_flag": ui.CornerFlag.TOP,
            "border_radius": 15,
            "fill_policy": ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP,
            "alignment": ui.Alignment.CENTER_BOTTOM}})
```

`ByteImageProvider` is really useful to create gradient images. Here is an example:
```execute 200
self._byte_provider = ui.ByteImageProvider()
self._byte_provider.set_bytes_data([
    255, 0, 0, 255,    # red
    255, 255, 0, 255,  # yellow
    0,  255, 0, 255,   # green
    0, 255, 255, 255,  # cyan
    0, 0, 255, 255],   # blue
    [5, 1])            # size
with ui.Frame(height=20):
    ui.ImageWithProvider(self._byte_provider,fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
```

## Plot
The Plot class displays a line or histogram image. The data of the image is specified as a data array or a provider function.

Except the common style for Buttons and Images, here is a list of styles you can customize on Plot:
> color (color): the color of the plot, line color in the line typed plot or rectangle bar color in the histogram typed plot
> selected_color (color): the selected color of the plot, dot in the line typed plot and rectangle bar in the histogram typed plot
> background_color (color): the background color of the plot
> secondary_color (color): the color of the text and the border of the text box which shows the plot selection value
> background_selected_color (color): the background color of the text box which shows the plot selection value

Here are couple of examples of Plots:
```execute 200
import math
from omni.ui import color as cl
data = []
for i in range(360):
    data.append(math.cos(math.radians(i)))

def on_data_provider(index):
    return math.sin(math.radians(index))

with ui.Frame(height=20):
    with ui.HStack():
        plot_1 = ui.Plot(ui.Type.LINE, -1.0, 1.0, *data, width=360, height=100,
                style={"Plot":{
                    "color": cl.red,
                    "background_color": cl(0.08),
                    "secondary_color": cl("#aa1111"),
                    "selected_color": cl.green,
                    "background_selected_color": cl.white,
                    "border_width":5,
                    "border_color": cl.blue,
                    "border_radius": 20
                    }})
        ui.Spacer(width = 20)
        plot_2 = ui.Plot(ui.Type.HISTOGRAM, -1.0, 1.0, on_data_provider, 360, width=360, height=100,
                style={"Plot":{
                    "color": cl.blue,
                    "background_color": cl("#551111"),
                    "secondary_color": cl("#11AA11"),
                    "selected_color": cl(0.67),
                    "margin_height": 10,
                    }})
        plot_2.value_stride = 6
```


<a name="sliders"></a>

================================================================================

# Sliders Section

*Source: sliders.md*

================================================================================

# Fields and Sliders

## Common Styling for Fields and Sliders
Here is a list of common style you can customize on Fields and Sliders:
> background_color (color): the background color of the field or slider
> border_color (color): the border color if the field or slider background has a border
> border_radius (float): the border radius if the user wants to round the field or slider
> border_width (float): the border width if the field or slider background has a border
> padding (float): the distance between the text and the border of the field or slider
> font_size (float): the size of the text in the field or slider

## Field
There are fields for string, float and int models.

Except the common style for Fields and Sliders, here is a list of styles you can customize on Field:
> color (color): the color of the text
> background_selected_color (color): the background color of the selected text

### StringField
The StringField widget is a one-line text editor. A field allows the user to enter and edit a single line of plain text. It's implemented using the model-delegate-view pattern and uses AbstractValueModel as the central component of the system.

The following example demonstrates how to connect a StringField and a Label. You can type anything into the StringField.

```execute 200
from omni.ui import color as cl
field_style = {
    "Field": {
        "background_color": cl(0.8),
        "border_color": cl.blue,
        "background_selected_color": cl.yellow,
        "border_radius": 5,
        "border_width": 1,
        "color": cl.red,
        "font_size": 20.0,
        "padding": 5,
    },
    "Field:pressed": {"background_color": cl.white, "border_color": cl.green, "border_width": 2, "padding": 8},
}

def setText(label, text):
    """Sets text on the label"""
    # This function exists because lambda cannot contain assignment
    label.text = f"You wrote '{text}'"

with ui.HStack():
    field = ui.StringField(style=field_style)
    ui.Spacer(width=5)
    label = ui.Label("", name="text")
    field.model.add_value_changed_fn(lambda m, label=label: setText(label, m.get_value_as_string()))
    ui.Spacer(width=10)
```

The following example demonstrates that the CheckBox's model decides the content of the Field. Click to edit and update the string field value also updates the value of the CheckBox. The field can only have one of the two options, either 'True' or 'False', because the model only supports those two possibilities.

```execute 200
from omni.ui import color as cl
with ui.HStack():
    field = ui.StringField(width=100, style={"background_color": cl.black})
    checkbox = ui.CheckBox(width=0)
    field.model = checkbox.model
```

In this example, the field can have anything because the model accepts any string. The model returns bool for checkbox, and the checkbox is unchecked when the string is empty or 'False'.

```execute 200
from omni.ui import color as cl
with ui.HStack():
    field = ui.StringField(width=100, style={"background_color": cl.black})
    checkbox = ui.CheckBox(width=0)
    checkbox.model = field.model
```

The Field widget doesn't keep the data due to the model-delegate-view pattern. However, there are two ways to track the state of the widget. It's possible to re-implement the AbstractValueModel. The second way is using the callbacks of the model. Here is a minimal example of callbacks. When you start editing the field, you will see "Editing is started", and when you finish editing by press `enter`, you will see "Editing is finished".

```execute 200
def on_value(label):
    label.text = "Value is changed"

def on_begin(label):
    label.text = "Editing is started"

def on_end(label):
    label.text = "Editing is finished"

label = ui.Label("Nothing happened", name="text")
model = ui.StringField().model
model.add_value_changed_fn(lambda m, l=label: on_value(l))
model.add_begin_edit_fn(lambda m, l=label: on_begin(l))
model.add_end_edit_fn(lambda m, l=label: on_end(l))
```

### Multiline StringField
Property `multiline` of `StringField` allows users to press enter and create a new line. It's possible to finish editing with Ctrl-Enter.
```execute 200
from omni.ui import color as cl
import inspect

field_style = {
    "Field": {
        "background_color": cl(0.8),
        "color": cl.black,
    },
    "Field:pressed": {"background_color": cl(0.8)},
}

field_callbacks = lambda: field_callbacks()
with ui.Frame(style=field_style, height=200):
    model = ui.SimpleStringModel("hello \nworld \n")
    field = ui.StringField(model, multiline=True)
```

### FloatField and IntField
The following example shows how string field, float field and int field interact with each other. All three fields share the same default FloatModel:
```execute 200
with ui.HStack(spacing=5):
    ui.Label("FloatField")
    ui.Label("IntField")
    ui.Label("StringField")
with ui.HStack(spacing=5):
    left = ui.FloatField()
    center = ui.IntField()
    right = ui.StringField()
    center.model = left.model
    right.model = left.model
ui.Spacer(height=5)
```

## MultiField
MultiField widget groups the widgets that have multiple similar widgets to represent each item in the model. It's handy to use them for arrays and multi-component data like float3, matrix, and color.

MultiField is using `Field` as the Type Selector. Therefore, the list of styless we can customize on MultiField is the same as Field

### MultiIntField
Each of the field value could be changed by editing
```execute 200
from omni.ui import color as cl
field_style = {
    "Field": {
        "background_color": cl(0.8),
        "border_color": cl.blue,
        "border_radius": 5,
        "border_width": 1,
        "color": cl.red,
        "font_size": 20.0,
        "padding": 5,
    },
    "Field:pressed": {"background_color": cl.white, "border_color": cl.green, "border_width": 2, "padding": 8},
}

ui.MultiIntField(0, 0, 0, 0, style=field_style)
```

### MultiFloatField
Use MultiFloatField to construct a matrix field:
```execute 200
args = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
ui.MultiFloatField(*args, width=ui.Percent(50), h_spacing=5, v_spacing=2)
```

### MultiFloatDragField
Each of the field value could be changed by dragging
```execute 200
ui.MultiFloatDragField(0.0, 0.0, 0.0, 0.0)
```

## Sliders
The Sliders are more like a traditional slider that can be dragged and snapped where you click. The value of the slider can be shown on the slider or not, but can not be edited directly by clicking.

Except the common style for Fields and Sliders, here is a list of styles you can customize on ProgressBar:
> color (color): the color of the text
> secondary_color (color): the color of the handle in `ui.SliderDrawMode.HANDLE` draw_mode or the background color of the left portion of the slider in `ui.SliderDrawMode.DRAG` draw_mode
> secondary_selected_color (color): the color of the handle when selected, not useful when the draw_mode is FILLED since there is no handle drawn.
> draw_mode (enum): defines how the slider handle is drawn. There are three types of draw_mode.
* ui.SliderDrawMode.HANDLE: draw the handle as a knob at the slider position
* ui.SliderDrawMode.DRAG: the same as `ui.SliderDrawMode.HANDLE` for now
* ui.SliderDrawMode.FILLED: the handle is eventually the boundary between the `secondary_color` and `background_color`

Sliders with different draw_mode:
```execute 200
from omni.ui import color as cl
with ui.VStack(spacing=5):
    ui.FloatSlider(style={"background_color": cl(0.8),
                           "secondary_color": cl(0.6),
                           "color": cl(0.1),
                           "draw_mode": ui.SliderDrawMode.HANDLE}
                    ).model.set_value(0.5)
    ui.FloatSlider(style={"background_color": cl(0.8),
                           "secondary_color": cl(0.6),
                           "color": cl(0.1),
                           "draw_mode": ui.SliderDrawMode.DRAG}
                    ).model.set_value(0.5)
    ui.FloatSlider(style={"background_color": cl(0.8),
                           "secondary_color": cl(0.6),
                           "color": cl(0.1),
                           "draw_mode": ui.SliderDrawMode.FILLED}
                    ).model.set_value(0.5)
```

### FloatSlider
Default slider whose range is between 0 to 1:
```execute 200
ui.FloatSlider()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.FloatSlider(min=0, max=10)
```

With defined Min/Max from the model. Notice the model allows the value range between 0 to 100, but the FloatSlider has a more strict range between 0 to 10.
```execute 200
model = ui.SimpleFloatModel(1.0, min=0, max=100)
ui.FloatSlider(model, min=0, max=10)
```

With styles and rounded slider:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatSlider(
                        min=-180,
                        max=180,
                        style={
                            "color": cl.blue,
                            "background_color": cl(0.8),
                            "draw_mode": ui.SliderDrawMode.HANDLE,
                            "secondary_color": cl.red,
                            "secondary_selected_color": cl.green,
                            "font_size": 20,
                            "border_width": 3,
                            "border_color": cl.black,
                            "border_radius": 10,
                            "padding": 10,
                        }
                    )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

Filled mode slider with style:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatSlider(
                        min=-180,
                        max=180,
                        style={
                            "color": cl.blue,
                            "background_color": cl(0.8),
                            "draw_mode": ui.SliderDrawMode.FILLED,
                            "secondary_color": cl.red,
                            "font_size": 20,
                            "border_radius": 10,
                            "padding": 10,
                        }
                    )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

Transparent background:
```execute 200
from omni.ui import color as cl
with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatSlider(
                        min=-180,
                        max=180,
                        style={
                            "draw_mode": ui.SliderDrawMode.HANDLE,
                            "background_color": cl.transparent,
                            "color": cl.red,
                            "border_width": 1,
                            "border_color": cl.white,
                        }
                    )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

Slider with transparent value. Notice the use of `step` attribute
```execute 200
from omni.ui import color as cl
with ui.HStack():
    # a separate float field
    field = ui.FloatField(height=15, width=50)
    # a slider using field's model
    ui.FloatSlider(
        min=0,
        max=20,
        step=0.25,
        model=field.model,
        style={
            "color":cl.transparent,
            "background_color": cl(0.3),
            "draw_mode": ui.SliderDrawMode.HANDLE}
    )
    # default value
    field.model.set_value(12.0)
```

### IntSlider
Default slider whose range is between 0 to 100:
```execute 200
ui.IntSlider()
```

With defined Min/Max whose range is between min to max. Note that the handle width is much wider.
```execute 200
ui.IntSlider(min=0, max=20)
```

With style:
```execute 200
from omni.ui import color as cl
with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.IntSlider(
                    min=0,
                    max=20,
                    style={
                        "background_color": cl("#BBFFBB"),
                        "color": cl.purple,
                        "draw_mode": ui.SliderDrawMode.HANDLE,
                        "secondary_color": cl.green,
                        "secondary_selected_color": cl.red,
                        "font_size": 14.0,
                        "border_width": 3,
                        "border_color": cl.green,
                        "padding": 5,
                    }
                ).model.set_value(4)
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

## Drags
The Drags are very similar to Sliders, but more like Field in the way that they behave. You can double click to edit the value but they also have a mean to be 'Dragged' to increase or decrease the value.

Except the common style for Fields and Sliders, here is a list of styles you can customize on ProgressBar:
> color (color): the color of the text
> secondary_color (color): the left portion of the slider in `ui.SliderDrawMode.DRAG` draw_mode

### FloatDrag
Default float drag whose range is -inf and +inf
```execute 200
ui.FloatDrag()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.FloatDrag(min=-10, max=10, step=0.1)
```

With styles and rounded shape:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatDrag(
                        min=-180,
                        max=180,
                        style={
                            "color": cl.blue,
                            "background_color": cl(0.8),
                            "secondary_color": cl.red,
                            "font_size": 20,
                            "border_width": 3,
                            "border_color": cl.black,
                            "border_radius": 10,
                            "padding": 10,
                        }
                    )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

### IntDrag
Default int drag whose range is -inf and +inf
```execute 200
ui.IntDrag()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.IntDrag(min=-10, max=10)
```

With styles and rounded slider:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.IntDrag(
                        min=-180,
                        max=180,
                        style={
                            "color": cl.blue,
                            "background_color": cl(0.8),
                            "secondary_color": cl.purple,
                            "font_size": 20,
                            "border_width": 4,
                            "border_color": cl.black,
                            "border_radius": 20,
                            "padding": 5,
                        }
                    )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

## ProgressBar
A ProgressBar is a widget that indicates the progress of an operation.

Except the common style for Fields and Sliders, here is a list of styles you can customize on ProgressBar:
> color (color): the color of the progress bar indicating the progress value of the progress bar in the portion of the overall value
> secondary_color (color): the color of the text indicating the progress value

In the following example, it shows how to use ProgressBar and override the style of the overlay text.
```execute 200
from omni.ui import color as cl
class CustomProgressValueModel(ui.AbstractValueModel):
    """An example of custom float model that can be used for progress bar"""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def set_value(self, value):
        """Reimplemented set"""
        try:
            value = float(value)
        except ValueError:
            value = None
        if value != self._value:
            # Tell the widget that the model is changed
            self._value = value
            self._value_changed()

    def get_value_as_float(self):
        return self._value

    def get_value_as_string(self):
        return "Custom Overlay"

with ui.VStack(spacing=5):
    # Create ProgressBar
    first = ui.ProgressBar()
    # Range is [0.0, 1.0]
    first.model.set_value(0.5)

    second = ui.ProgressBar()
    second.model.set_value(1.0)

    # Overrides the overlay of ProgressBar
    model = CustomProgressValueModel(0.8)
    third = ui.ProgressBar(model)
    third.model.set_value(0.1)

    # Styling its color
    fourth = ui.ProgressBar(style={"color": cl("#0000dd")})
    fourth.model.set_value(0.3)

    # Styling its border width
    ui.ProgressBar(style={"border_width": 2, "border_color": cl("#dd0000"), "color": cl("#0000dd")}).model.set_value(0.7)

    # Styling its border radius
    ui.ProgressBar(style={"border_radius": 100, "color": cl("#0000dd")}).model.set_value(0.6)

    # Styling its background color
    ui.ProgressBar(style={"border_radius": 10, "background_color": cl("#0000dd")}).model.set_value(0.6)

    # Styling the text color
    ui.ProgressBar(style={"ProgressBar":{"border_radius": 30, "secondary_color": cl("#00dddd"), "font_size": 20}}).model.set_value(0.6)

    # Two progress bars in a row with padding
    with ui.HStack():
        ui.ProgressBar(style={"color": cl("#0000dd"), "padding": 100}).model.set_value(1.0)
        ui.ProgressBar().model.set_value(0.0)
```

## Tooltip
All Widget can be augmented with a tooltip. It can take 2 forms, either a simple ui.Label or a callback when using the callback of `tooltip_fn=` or `widget.set_tooltip_fn()`. You can create the tooltip for any widget.

Except the common style for Fields and Sliders, here is a list of styles you can customize on Line:
> color (color): the color of the text of the tooltip.
> margin_width (float): the width distance between the tooltip content and the parent widget defined boundary
> margin_height (float): the height distance between the tooltip content and the parent widget defined boundary

Here is a simple label tooltip with style when you hover over a button:
```execute 200
from omni.ui import color as cl
tooltip_style = {
    "Tooltip": {
        "background_color": cl("#DDDD00"),
        "color": cl(0.2),
        "padding": 10,
        "border_width": 3,
        "border_color": cl.red,
        "font_size": 20,
        "border_radius": 10}}

ui.Button("Simple Label Tooltip", name="tooltip", width=200, tooltip="I am a text ToolTip", style=tooltip_style)
```

You can create a callback function as the tooltip where you can create any types of widgets you like in the tooltip and layout them. Make the tooltip very illustrative to have Image or Field or Label etc.
```execute 200
from omni.ui import color as cl
def create_tooltip():
    with ui.VStack(width=200, style=tooltip_style):
        with ui.HStack():
            ui.Label("Fancy tooltip", width=150)
            ui.IntField().model.set_value(12)
        ui.Line(height=2, style={"color":cl.white})
        with ui.HStack():
            ui.Label("Anything is possible", width=150)
            ui.StringField().model.set_value("you bet")
        image_source = "resources/desktop-icons/omniverse_512.png"
        ui.Image(
            image_source,
            width=200,
            height=200,
            alignment=ui.Alignment.CENTER,
            style={"margin": 0},
        )
tooltip_style = {
    "Tooltip": {
        "background_color": cl(0.2),
        "border_width": 2,
        "border_radius": 5,
        "margin_width": 5,
        "margin_height": 10
        },
    }
ui.Button("Callback function Tooltip", width=200, style=tooltip_style, tooltip_fn=create_tooltip)
```

You can define a fixed position for tooltip:
```execute 200
ui.Button("Fixed-position Tooltip", width=200, tooltip="Hello World", tooltip_offset_y=22)
```

You can also define a random position for tooltip:
```execute 200
import random
button = ui.Button("Random-position Tooltip", width=200, tooltip_offset_y=22)

def create_tooltip(button=button):
    button.tooltip_offset_x = random.randint(0, 200)
    ui.Label("Hello World")

button.set_tooltip_fn(create_tooltip)
```


<a name="shapes"></a>

================================================================================

# Shapes Section

*Source: shapes.md*

================================================================================

# Shapes
Shapes enable you to build custom widgets with specific looks. There are many shapes you can stylize: Rectangle, Circle, Ellipse, Triangle and FreeShapes of FreeRectangle, FreeCircle, FreeEllipse, FreeTriangle. In most cases those shapes will fit into the widget size which is defined by the parent widget they are in.

The FreeShapes are the shapes that are independent of the layout. It means it can be stuck to other shapes. It means it is possible to stick the freeshape to the layout's widgets, and the freeshape will follow the changes of the layout automatically.

## Common Style of shapes
Here is a list of common style you can customize on all the Shapes:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

## Rectangle
Rectangle is a shape with four sides and four corners. You can use Rectangle to draw rectangle shapes, or mix it with other controls e.g. using ZStack to create an advanced look.

Except the common style for shapes, here is a list of styles you can customize on Rectangle:
> background_gradient_color (color): the gradient color on the top part of the rectangle
> border_radius (float): default rectangle has 4 right corner angles, border_radius defines the radius of the corner angle if the user wants to round the rectangle corner. We only support one border_radius across all the corners, but users can choose which corner to be rounded.
> corner_flag (enum): defines which corner or corners to be rounded

Here is a list of the supported corner flags:
```execute 200
from omni.ui import color as cl
corner_flags = {
    "ui.CornerFlag.NONE": ui.CornerFlag.NONE,
    "ui.CornerFlag.TOP_LEFT": ui.CornerFlag.TOP_LEFT,
    "ui.CornerFlag.TOP_RIGHT": ui.CornerFlag.TOP_RIGHT,
    "ui.CornerFlag.BOTTOM_LEFT": ui.CornerFlag.BOTTOM_LEFT,
    "ui.CornerFlag.BOTTOM_RIGHT": ui.CornerFlag.BOTTOM_RIGHT,
    "ui.CornerFlag.TOP": ui.CornerFlag.TOP,
    "ui.CornerFlag.BOTTOM": ui.CornerFlag.BOTTOM,
    "ui.CornerFlag.LEFT": ui.CornerFlag.LEFT,
    "ui.CornerFlag.RIGHT": ui.CornerFlag.RIGHT,
    "ui.CornerFlag.ALL": ui.CornerFlag.ALL,
}

with ui.ScrollingFrame(
    height=100,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in corner_flags.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}):
                    ui.Rectangle(
                        style={"background_color": cl("#aa4444"), "border_radius": 20.0, "corner_flag": value}
                    )
                    ui.Spacer(height=10)
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
```
Here are a few examples of Rectangle using different selections of styles:

Default rectangle which is scaled to fit:
```execute 200
with ui.Frame(height=20):
    ui.Rectangle(name="default")
```

This rectangle uses its own style to control colors and shape. Notice how three colors "background_color", "border_color" and "border_color" are affecting the look of the rectangle:
```execute 200
from omni.ui import color as cl
with ui.Frame(height=40):
    ui.Rectangle(style={"Rectangle":{
        "background_color":cl("#aa4444"),
        "border_color":cl("#22FF22"),
        "background_gradient_color": cl("#4444aa"),
        "border_width": 2.0,
        "border_radius": 5.0}})
```

This rectangle uses fixed width and height. Notice the `border_color` is not doing anything if `border_width` is not defined.
```execute 200
from omni.ui import color as cl
with ui.Frame(height=20):
    ui.Rectangle(width=40, height=10, style={"background_color":cl(0.6), "border_color":cl("#ff2222")})
```

Compose with ZStack for an advanced look
```execute 200
from omni.ui import color as cl
with ui.Frame(height=20):
    with ui.ZStack(height=20):
        ui.Rectangle(width=150,
            style={"background_color":cl(0.6),
                    "border_color":cl(0.1),
                    "border_width": 1.0,
                    "border_radius": 8.0} )
        with ui.HStack():
            ui.Spacer(width=10)
            ui.Image("resources/icons/Cloud.png", width=20, height=20 )
            ui.Label( "Search Field", style={"color":cl(0.875)})
```

## FreeRectangle
FreeRectangle is a rectangle whose width and height will be determined by other widgets. The supported style list is the same as Rectangle.

Here is an example of a FreeRectangle with style following two draggable circles:
```execute 200
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Circle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=150):
            control2 = ui.Circle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeRectangle(control1, control2, style={
                    "background_color":cl(0.6),
                    "border_color":cl(0.1),
                    "border_width": 1.0,
                    "border_radius": 8.0})
```

## Circle
You can use Circle to draw a circular shape. Circle doesn't have any other style except the common style for shapes.

Here is some of the properties you can customize on Circle:
> size_policy (enum): there are two types of the size_policy, fixed and stretch.
    * ui.CircleSizePolicy.FIXED: the size of the circle is defined by the radius and is fixed without being affected by the parent scaling.
    * ui.CircleSizePolicy.STRETCH: the size of the circle is defined by the parent and will be stretched if the parent widget size changed.
> alignment (enum): the position of the circle in the parent defined space
> arc (enum): this property defines the way to draw a half or a quarter of the circle.

Here is a list of the supported Alignment and Arc value for the Circle:

```execute 200
from omni.ui import color as cl
alignments = {
    "ui.Alignment.CENTER": ui.Alignment.CENTER,
    "ui.Alignment.LEFT_TOP": ui.Alignment.LEFT_TOP,
    "ui.Alignment.LEFT_CENTER": ui.Alignment.LEFT_CENTER,
    "ui.Alignment.LEFT_BOTTOM": ui.Alignment.LEFT_BOTTOM,
    "ui.Alignment.CENTER_TOP": ui.Alignment.CENTER_TOP,
    "ui.Alignment.CENTER_BOTTOM": ui.Alignment.CENTER_BOTTOM,
    "ui.Alignment.RIGHT_TOP": ui.Alignment.RIGHT_TOP,
    "ui.Alignment.RIGHT_CENTER": ui.Alignment.RIGHT_CENTER,
    "ui.Alignment.RIGHT_BOTTOM": ui.Alignment.RIGHT_BOTTOM,
}
ui.Label("Alignment: ")
with ui.ScrollingFrame(
    height=150,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    with ui.ZStack():
                        ui.Rectangle(name="table", style={"border_color":cl.white, "border_width": 1.0})
                        ui.Circle(
                            radius=10,
                            size_policy=ui.CircleSizePolicy.FIXED,
                            name="orientation",
                            alignment=value,
                            style={"background_color": cl("#aa4444")},
                        )
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
ui.Spacer(height=10)
ui.Label("Arc: ")
with ui.ScrollingFrame(
    height=150,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    with ui.ZStack():
                        ui.Rectangle(name="table", style={"border_color":cl.white, "border_width": 1.0})
                        ui.Circle(
                            radius=10,
                            size_policy=ui.CircleSizePolicy.FIXED,
                            name="orientation",
                            arc=value,
                            style={
                                "background_color": cl("#aa4444"),
                                "border_color": cl.blue,
                                "border_width": 2,
                                },
                        )
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
```

Default circle which is scaled to fit, the alignment is centered:
```execute 200
with ui.Frame(height=20):
    ui.Circle(name="default")
```

This circle is scaled to fit with 100 height:
```execute 200
with ui.Frame(height=100):
    ui.Circle(name="default")
```

This circle has a fixed radius of 20, the alignment is LEFT_CENTER:
```execute 200
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#1111ff"), "border_color": cl("#cc0000"), "border_width": 4}}
with ui.Frame(height=100, style=style):
    with ui.HStack():
        ui.Rectangle(width=40, style={"background_color": cl.white})
        ui.Circle(radius=20, size_policy=ui.CircleSizePolicy.FIXED, alignment=ui.Alignment.LEFT_CENTER)
```

This circle has a fixed radius of 10, the alignment is RIGHT_CENTER
```execute 200
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#ff1111"), "border_color": cl.blue, "border_width": 2}}
with ui.Frame(height=100, width=200, style=style):
    with ui.ZStack():
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Circle(radius=10, size_policy=ui.CircleSizePolicy.FIXED, alignment=ui.Alignment.RIGHT_CENTER)
```

This circle has a fixed radius of 10, it has all the same style as the previous one, except its size_policy is `ui.CircleSizePolicy.STRETCH`
```execute 200
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#ff1111"), "border_color": cl.blue, "border_width": 2}}
with ui.Frame(height=100, width=200, style=style):
    with ui.ZStack():
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Circle(radius=10, size_policy=ui.CircleSizePolicy.STRETCH, alignment=ui.Alignment.RIGHT_CENTER)
```

## FreeCircle
FreeCircle is a circle whose radius will be determined by other widgets. The supported style list is the same as Circle.

Here is an example of a FreeCircle with style following two draggable rectangles:
```execute 200
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Rectangle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=150):
            control2 = ui.Rectangle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeCircle(control1, control2, style={
                    "background_color":cl.transparent,
                    "border_color":cl.red,
                    "border_width": 2.0})
```

## Ellipse
Ellipse is drawn in a rectangle bounding box, and It is always scaled to fit the rectangle's width and height. Ellipse doesn't have any other style except the common style for shapes.

Default ellipse is scaled to fit:
```execute 200
with ui.Frame(height=20, width=150):
    ui.Ellipse(name="default")
```

Stylish ellipse with border and colors:
```execute 200
from omni.ui import color as cl
style = {"Ellipse": {"background_color": cl("#1111ff"), "border_color": cl("#cc0000"), "border_width": 4}}
with ui.Frame(height=100, width=50):
    ui.Ellipse(style=style)
```

## FreeEllipse
FreeEllipse is an ellipse whose width and height will be determined by other widgets. The supported style list is the same as Ellipse.

Here is an example of a FreeEllipse with style following two draggable circles:

```execute 200
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Circle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=200):
            control2 = ui.Circle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeEllipse(control1, control2, style={
                    "background_color":cl.purple})
```

## Triangle
You can use Triangle to draw Triangle shape. Triangle doesn't have any other style except the common style for shapes.

Here is some of the properties you can customize on Triangle:
> alignment (enum): the alignment defines where the tip of the triangle is, base will be at the opposite side

Here is a list of the supported alignment value for the triangle:

```execute 200
from omni.ui import color as cl
alignments = {
    "ui.Alignment.LEFT_TOP": ui.Alignment.LEFT_TOP,
    "ui.Alignment.LEFT_CENTER": ui.Alignment.LEFT_CENTER,
    "ui.Alignment.LEFT_BOTTOM": ui.Alignment.LEFT_BOTTOM,
    "ui.Alignment.CENTER_TOP": ui.Alignment.CENTER_TOP,
    "ui.Alignment.CENTER_BOTTOM": ui.Alignment.CENTER_BOTTOM,
    "ui.Alignment.RIGHT_TOP": ui.Alignment.RIGHT_TOP,
    "ui.Alignment.RIGHT_CENTER": ui.Alignment.RIGHT_CENTER,
    "ui.Alignment.RIGHT_BOTTOM": ui.Alignment.RIGHT_BOTTOM,
}
colors = [cl.red, cl.yellow, cl.purple, cl("#ff0ff0"), cl.green, cl("#f00fff"), cl("#fff000"), cl("#aa3333")]
index = 0
with ui.ScrollingFrame(
    height=160,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}):
                    color = colors[index]
                    index = index + 1
                    ui.Triangle(alignment=value, style={"Triangle":{"background_color": color}})
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER, height=20)
```

Here are a few examples of Triangle using different selections of styles:

The triangle is scaled to fit, base on the left and tip on the center right. Users can define the border_color and border_width but without background_color to make the triangle look like it's drawn in wireframe style.
```execute 200
from omni.ui import color as cl
style = {
    "Triangle::default":
    {
        "background_color": cl.green,
        "border_color": cl.white,
        "border_width": 1
    },
    "Triangle::transparent":
    {
        "border_color": cl.purple,
        "border_width": 4,
    },
}
with ui.Frame(height=100, width=200, style=style):
    with ui.HStack(spacing=10, style={"margin": 5}):
        ui.Triangle(name="default")
        ui.Triangle(name="transparent", alignment=ui.Alignment.CENTER_TOP)
```

## FreeTriangle
FreeTriangle is a triangle whose width and height will be determined by other widgets. The supported style list is the same as Triangle.

Here is an example of a FreeTriangle with style following two draggable rectangles. The default alignment is `ui.Alignment.RIGHT_CENTER`. We make the alignment as `ui.Alignment.CENTER_BOTTOM`.

```execute 200
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Rectangle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=200):
            control2 = ui.Rectangle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeTriangle(control1, control2, alignment=ui.Alignment.CENTER_BOTTOM, style={
                    "background_color":cl.blue,
                    "border_color":cl.red,
                    "border_width": 2.0})
```


<a name="line"></a>

================================================================================

# Line Section

*Source: line.md*

================================================================================

# Lines and Curves

## Common Style of Lines and Curves
Here is a list of common styles you can customize on all the Lines and Curves:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve

## Line
Line is the simplest shape that represents a straight line. It has two points, color and thickness. You can use Line to draw line shapes. Line doesn't have any other style besides the common styles for Lines and Curves.

Here are some of the properties you can customize on Line:
> alignment (enum): the Alignment defines where the line is in parent defined space. It is always scaled to fit.

Here is a list of the supported Alignment value for the line:
```execute 200
from omni.ui import color as cl
style ={
    "Rectangle::table": {"background_color": cl.transparent, "border_color": cl(0.8), "border_width": 0.25},
    "Line::demo": {"color": cl("#007777"), "border_width": 3},
    "ScrollingFrame": {"background_color": cl.transparent},
}
alignments = {
    "ui.Alignment.LEFT": ui.Alignment.LEFT,
    "ui.Alignment.RIGHT": ui.Alignment.RIGHT,
    "ui.Alignment.H_CENTER": ui.Alignment.H_CENTER,
    "ui.Alignment.TOP": ui.Alignment.TOP,
    "ui.Alignment.BOTTOM": ui.Alignment.BOTTOM,
    "ui.Alignment.V_CENTER": ui.Alignment.V_CENTER,
}
with ui.ScrollingFrame(
    height=100,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style=style,
):
    with ui.HStack(height=100):
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    ui.Line(name="demo", alignment=value)
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)

```

By default, the line is scaled to fit.
```execute 200
from omni.ui import color as cl
style = {"Line::default": {"color": cl.red, "border_width": 1}}
with ui.Frame(height=50, style=style):
    ui.Line(name="default")
```

Users can define the color and border_width to make customized lines.
```execute 200
from omni.ui import color as cl
with ui.Frame(height=50):
    with ui.ZStack(width=200):
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Line(alignment=ui.Alignment.H_CENTER, style={"border_width":5, "color": cl("#880088")})
```

## FreeLine
FreeLine is a line whose length will be determined by other widgets. The supported style list is the same as Line.

Here is an example of a FreeLine with style, driven by two draggable circles. Notice the control widgets are not the start and end points of the line. By default, the alignment of the line is `ui.Alighment.V_CENTER`, and the line direction won't be changed by the control widgets.

```execute 200
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Circle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=200):
            control2 = ui.Circle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeLine(control1, control2, style={"color":cl.yellow})
```

## BezierCurve
BezierCurve is a smooth mathematical curve defined by a set of control points, used to create curves and shapes that can be scaled indefinitely. BezierCurve doesn't have any other style except the common styles for Lines and Curves.

Here is a BezierCurve with style:
```execute 200
from omni.ui import color as cl
style = {"BezierCurve": {"color": cl.red, "border_width": 2}}
ui.Spacer(height=2)
with ui.Frame(height=50, style=style):
    ui.BezierCurve()
ui.Spacer(height=2)
```

## FreeBezierCurve
FreeBezierCurve uses two widgets to get the position of the curve endpoints. This is super useful to build graph connections. The supported style list is the same as BezierCurve.

Here is an example of a FreeBezierCurve which is controlled by 4 control points.

```execute 200
from omni.ui import color as cl
with ui.ZStack(height=400):
    # The Bezier tangents
    tangents = [(50, 50), (-50, -50)]

    # Four draggable rectangles that represent the control points
    placer1 = ui.Placer(draggable=True, offset_x=0, offset_y=0)
    with placer1:
        rect1 = ui.Rectangle(width=20, height=20)
    placer2 = ui.Placer(draggable=True, offset_x=50, offset_y=50)
    with placer2:
        rect2 = ui.Rectangle(width=20, height=20)
    placer3 = ui.Placer(draggable=True, offset_x=100, offset_y=100)
    with placer3:
        rect3 = ui.Rectangle(width=20, height=20)
    placer4 = ui.Placer(draggable=True, offset_x=150, offset_y=150)
    with placer4:
        rect4 = ui.Rectangle(width=20, height=20)

    # The bezier curve
    curve = ui.FreeBezierCurve(rect1, rect4, style={"color": cl.red, "border_width": 5})
    curve.start_tangent_width = ui.Pixel(tangents[0][0])
    curve.start_tangent_height = ui.Pixel(tangents[0][1])
    curve.end_tangent_width = ui.Pixel(tangents[1][0])
    curve.end_tangent_height = ui.Pixel(tangents[1][1])

    # The logic of moving the control points
    def left_moved(_):
        x = placer1.offset_x
        y = placer1.offset_y
        tangent = tangents[0]
        placer2.offset_x = x + tangent[0]
        placer2.offset_y = y + tangent[1]

    def right_moved(_):
        x = placer4.offset_x
        y = placer4.offset_y
        tangent = tangents[1]
        placer3.offset_x = x + tangent[0]
        placer3.offset_y = y + tangent[1]

    def left_tangent_moved(_):
        x1 = placer1.offset_x
        y1 = placer1.offset_y
        x2 = placer2.offset_x
        y2 = placer2.offset_y
        tangent = (x2 - x1, y2 - y1)
        tangents[0] = tangent
        curve.start_tangent_width = ui.Pixel(tangent[0])
        curve.start_tangent_height = ui.Pixel(tangent[1])

    def right_tangent_moved(_):
        x1 = placer4.offset_x
        y1 = placer4.offset_y
        x2 = placer3.offset_x
        y2 = placer3.offset_y
        tangent = (x2 - x1, y2 - y1)
        tangents[1] = tangent
        curve.end_tangent_width = ui.Pixel(tangent[0])
        curve.end_tangent_height = ui.Pixel(tangent[1])

    # Callback for moving the control points
    placer1.set_offset_x_changed_fn(left_moved)
    placer1.set_offset_y_changed_fn(left_moved)
    placer2.set_offset_x_changed_fn(left_tangent_moved)
    placer2.set_offset_y_changed_fn(left_tangent_moved)
    placer3.set_offset_x_changed_fn(right_tangent_moved)
    placer3.set_offset_y_changed_fn(right_tangent_moved)
    placer4.set_offset_x_changed_fn(right_moved)
    placer4.set_offset_y_changed_fn(right_moved)
```

## Curve Anchors
Curve Anchors and Line Anchors allow for decorations to be placed on a curve or line, such that when the shape is moved, the decoration will stay attached to it at the same parametric position.  The anchor has 2 properties for its alignment and position (0-1), and an anchor_fn to supply a callback function which draws the decoration that will be attached to the curve.

Here is an example of an Anchor on a FreeBezierCurve.  The decoration can be dragged along the curve with the left mouse button.
```execute 200
from functools import partial
import asyncio

params = [None, None, None, None]

def moved(x, y, b, m):
    x1 = params[0].screen_position_x + params[0].computed_width / 2
    x2 = params[1].screen_position_x + params[1].computed_width / 2
    anchor_position = (x - x1) / (x2 - x1)
    anchor_position = max(min(anchor_position, 1), 0)
    params[2].anchor_position = anchor_position
    params[3].text = f"{params[2].anchor_position:.1f}"

def bound(curve=None):
    with ui.ZStack(content_clipping=1):
        params[3] = ui.Label(f"{params[2].anchor_position:.1f}", mouse_moved_fn=moved)

with ui.ZStack():
    with ui.Placer(draggable=1):
        r1 = ui.Rectangle(width=10, height=10, style={"background_color": ui.color.blue})
    with ui.Placer(draggable=1, offset_x=100, offset_y=100):
        r2 = ui.Rectangle(width=10, height=10, style={"background_color": ui.color.green})
    with ui.Frame(separate_window=True):
        curve = ui.FreeBezierCurve(r1, r2, anchor_position=0.25)
    curve.set_anchor_fn(partial(bound, curve))

params[0] = r1
params[1] = r2
params[2] = curve
```


---

## End of Documentation

This combined documentation was automatically generated from the individual documentation files.
Total sections included: 13