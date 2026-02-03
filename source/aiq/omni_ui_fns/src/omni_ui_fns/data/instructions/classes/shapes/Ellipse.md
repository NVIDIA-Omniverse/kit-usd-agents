# omni.ui.Ellipse

Ellipse is drawn in a rectangle bounding box, and It is always scaled to fit the rectangle's width and height.

Here is a list of styles you can customize on Ellipse:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Default ellipse is scaled to fit:
```
with ui.Frame(height=20, width=150):
    ui.Ellipse(name="default")
```

Stylish ellipse with border and colors:
```
from omni.ui import color as cl
style = {"Ellipse": {"background_color": cl("#1111ff"), "border_color": cl("#cc0000"), "border_width": 4}}
with ui.Frame(height=100, width=50):
    ui.Ellipse(style=style)
```

