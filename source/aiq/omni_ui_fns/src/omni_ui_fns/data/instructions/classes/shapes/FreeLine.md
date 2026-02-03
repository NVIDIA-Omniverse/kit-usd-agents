# omni.ui.FreeLine

FreeLine is a line whose length will be determined by other widgets.

Here is a list of common styles you can customize on Line:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here is an example of a FreeLine with style, driven by two draggable circles. Notice the control widgets are not the start and end points of the line. By default, the alignment of the line is `ui.Alighment.V_CENTER`, and the line direction won't be changed by the control widgets.

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
        ui.FreeLine(control1, control2, style={"color":cl.yellow})
```

