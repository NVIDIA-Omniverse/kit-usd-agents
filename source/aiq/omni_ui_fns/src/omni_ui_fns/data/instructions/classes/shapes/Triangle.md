# omni.ui.Triangle

You can use Triangle to draw Triangle shape.

Here is a list of styles you can customize on Triangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is some of the properties you can customize on Triangle:
> alignment (enum): the alignment defines where the tip of the triangle is, base will be at the opposite side

Here is a list of the supported alignment value for the triangle:

```
from omni.ui import color as cl
alignments = {
    "ui.Alignment.LEFT_TOP": ui.Alignment.LEFT_TOP,
    "ui.Alignment.LEFT_CENTER": ui.Alignment.LEFT_CENTER,
    "ui.Alignment.LEFT_BOTTOM": ui.Alignment.LEFT_BOTTOM,
    "ui.Alignment.CENTER_TOP": ui.Alignment.CENTER_TOP,
    "ui.Alignment.CENTER_BOTTOM": ui.Alignment.CENTER_BOTTOM,
    "ui.Alignment.RIGHT_TOP": ui.Alignment.RIGHT_TOP,
    "ui.Alignment.RIGHT_CENTER": ui.Alignment.RIGHT_CENTER,
    "ui.Alignment.RIGHT_BOTTOM": ui.Alignment.RIGHT_BOTTOM,
}
colors = [cl.red, cl.yellow, cl.purple, cl("#ff0ff0"), cl.green, cl("#f00fff"), cl("#fff000"), cl("#aa3333")]
index = 0
with ui.ScrollingFrame(
    height=160,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}):
                    color = colors[index]
                    index = index + 1
                    ui.Triangle(alignment=value, style={"Triangle":{"background_color": color}})
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER, height=20)
```

Here are a few examples of Triangle using different selections of styles:

The triangle is scaled to fit, base on the left and tip on the center right. Users can define the border_color and border_width but without background_color to make the triangle look like it's drawn in wireframe style.
```
from omni.ui import color as cl
style = {
    "Triangle::default":
    {
        "background_color": cl.green,
        "border_color": cl.white,
        "border_width": 1
    },
    "Triangle::transparent":
    {
        "border_color": cl.purple,
        "border_width": 4,
    },
}
with ui.Frame(height=100, width=200, style=style):
    with ui.HStack(spacing=10, style={"margin": 5}):
        ui.Triangle(name="default")
        ui.Triangle(name="transparent", alignment=ui.Alignment.CENTER_TOP)
```

