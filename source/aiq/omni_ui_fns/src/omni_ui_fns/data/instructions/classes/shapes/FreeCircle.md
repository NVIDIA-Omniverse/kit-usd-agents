# omni.ui.FreeCircle

FreeCircle is a circle whose radius will be determined by other widgets.

Here is a list of styles you can customize on FreeCircle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is an example of a FreeCircle with style following two draggable rectangles:
```
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


