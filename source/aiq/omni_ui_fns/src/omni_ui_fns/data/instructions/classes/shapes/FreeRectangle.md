# omni.ui.FreeRectangle

FreeRectangle is a rectangle whose width and height will be determined by other widgets.

Here is a list of styles you can customize on FreeRectangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border
> background_gradient_color (color): the gradient color on the top part of the rectangle
> border_radius (float): default rectangle has 4 right corner angles, border_radius defines the radius of the corner angle if the user wants to round the rectangle corner. We only support one border_radius across all the corners, but users can choose which corner to be rounded.
> corner_flag (enum): defines which corner or corners to be rounded

Here is an example of a FreeRectangle with style following two draggable circles:
```
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

