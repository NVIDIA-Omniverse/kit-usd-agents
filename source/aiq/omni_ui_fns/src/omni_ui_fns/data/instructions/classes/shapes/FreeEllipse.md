# omni.ui.FreeEllipse

FreeEllipse is an ellipse whose width and height will be determined by other widgets.

Here is a list of styles you can customize on FreeEllipse:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is an example of a FreeEllipse with style following two draggable circles:

```
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

