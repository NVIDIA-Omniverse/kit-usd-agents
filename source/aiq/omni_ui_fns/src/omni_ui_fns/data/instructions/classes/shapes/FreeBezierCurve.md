# omni.ui.FreeBezierCurve

FreeBezierCurve uses two widgets to get the position of the curve endpoints. This is super useful to build graph connections.

Here is a list of common styles you can customize on BezierCurve:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here is an example of a FreeBezierCurve which is controlled by 4 control points.

```
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

Curve Anchors and Line Anchors allow for decorations to be placed on a curve or line, such that when the shape is moved, the decoration will stay attached to it at the same parametric position.  The anchor has 2 properties for its alignment and position (0-1), and an anchor_fn to supply a callback function which draws the decoration that will be attached to the curve.

Here is an example of an Anchor on a FreeBezierCurve.  The decoration can be dragged along the curve with the left mouse button.
```
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

