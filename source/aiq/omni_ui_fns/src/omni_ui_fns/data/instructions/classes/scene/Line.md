# omni.ui.Line

Line is the simplest shape that represents a straight line. It has two points, color and thickness. You can use Line to draw line shapes.

Here is a list of common styles you can customize on Line:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here are some of the properties you can customize on Line:
> alignment (enum): the Alignment defines where the line is in parent defined space. It is always scaled to fit.

Here is a list of the supported Alignment value for the line:
```
from omni.ui import color as cl
style ={
    "Rectangle::table": {"background_color": cl.transparent, "border_color": cl(0.8), "border_width": 0.25},
    "Line::demo": {"color": cl("#007777"), "border_width": 3},
    "ScrollingFrame": {"background_color": cl.transparent},
}
alignments = {
    "ui.Alignment.LEFT": ui.Alignment.LEFT,
    "ui.Alignment.RIGHT": ui.Alignment.RIGHT,
    "ui.Alignment.H_CENTER": ui.Alignment.H_CENTER,
    "ui.Alignment.TOP": ui.Alignment.TOP,
    "ui.Alignment.BOTTOM": ui.Alignment.BOTTOM,
    "ui.Alignment.V_CENTER": ui.Alignment.V_CENTER,
}
with ui.ScrollingFrame(
    height=100,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style=style,
):
    with ui.HStack(height=100):
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    ui.Line(name="demo", alignment=value)
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)

```

By default, the line is scaled to fit.
```
from omni.ui import color as cl
style = {"Line::default": {"color": cl.red, "border_width": 1}}
with ui.Frame(height=50, style=style):
    ui.Line(name="default")
```

Users can define the color and border_width to make customized lines.
```
from omni.ui import color as cl
with ui.Frame(height=50):
    with ui.ZStack(width=200):
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Line(alignment=ui.Alignment.H_CENTER, style={"border_width":5, "color": cl("#880088")})
```

