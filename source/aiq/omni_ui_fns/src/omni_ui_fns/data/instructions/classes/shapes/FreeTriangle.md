# omni.ui.FreeTriangle

FreeTriangle is a triangle whose width and height will be determined by other widgets.

Here is a list of styles you can customize on Triangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is an example of a FreeTriangle with style following two draggable rectangles. The default alignment is `ui.Alignment.RIGHT_CENTER`. We make the alignment as `ui.Alignment.CENTER_BOTTOM`.

```
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

