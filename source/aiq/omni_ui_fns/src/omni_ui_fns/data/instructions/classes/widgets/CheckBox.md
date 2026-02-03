# omni.ui.CheckBox

A CheckBox is an option button that can be switched on (checked) or off (unchecked). Checkboxes are typically used to represent features in an application that can be enabled or disabled without affecting others.

The checkbox is implemented using the model-delegate-view pattern. The model is the central component of this system. It is the application's dynamic data structure independent of the widget. It directly manages the data, logic and rules of the checkbox. If the model is not specified, the simple one is created automatically when the object is constructed.

Here is a list of styles you can customize on CheckBox:
> color (color): the color of the tick
> background_color (color): the background color of the check box
> font_size: the size of the tick
> border_radius (float): the radius of the corner angle if the user wants  to round the check box.
> border_width (float): the size of the check box border
> secondary_background_color (color): the color of the check box border

Default checkbox
```
with ui.HStack(width=0, spacing=5):
    ui.CheckBox().model.set_value(True)
    ui.CheckBox()
    ui.Label("Default")
```

Disabled checkbox:
```
with ui.HStack(width=0, spacing=5):
    ui.CheckBox(enabled=False).model.set_value(True)
    ui.CheckBox(enabled=False)
    ui.Label("Disabled")
```

In the following example, the models of two checkboxes are connected, and if one checkbox is changed, it makes another checkbox change as well.

```
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
```
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

