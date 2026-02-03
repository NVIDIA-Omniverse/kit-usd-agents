# omni.ui.BezierCurve

BezierCurve is a smooth mathematical curve defined by a set of control points, used to create curves and shapes that can be scaled indefinitely.

Here is a list of common styles you can customize on BezierCurve:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here is a BezierCurve with style:
```
from omni.ui import color as cl
style = {"BezierCurve": {"color": cl.red, "border_width": 2}}
ui.Spacer(height=2)
with ui.Frame(height=50, style=style):
    ui.BezierCurve()
ui.Spacer(height=2)
```

