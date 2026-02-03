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