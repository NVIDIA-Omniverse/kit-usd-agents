# omni.ui.Circle

You can use Circle to draw a circular shape.

Here is a list of styles you can customize on Circle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is some of the properties you can customize on Circle:
> size_policy (enum): there are two types of the size_policy, fixed and stretch.
    * ui.CircleSizePolicy.FIXED: the size of the circle is defined by the radius and is fixed without being affected by the parent scaling.
    * ui.CircleSizePolicy.STRETCH: the size of the circle is defined by the parent and will be stretched if the parent widget size changed.
> alignment (enum): the position of the circle in the parent defined space
> arc (enum): this property defines the way to draw a half or a quarter of the circle.

Here is a list of the supported Alignment and Arc value for the Circle:

```
from omni.ui import color as cl
alignments = {
    "ui.Alignment.CENTER": ui.Alignment.CENTER,
    "ui.Alignment.LEFT_TOP": ui.Alignment.LEFT_TOP,
    "ui.Alignment.LEFT_CENTER": ui.Alignment.LEFT_CENTER,
    "ui.Alignment.LEFT_BOTTOM": ui.Alignment.LEFT_BOTTOM,
    "ui.Alignment.CENTER_TOP": ui.Alignment.CENTER_TOP,
    "ui.Alignment.CENTER_BOTTOM": ui.Alignment.CENTER_BOTTOM,
    "ui.Alignment.RIGHT_TOP": ui.Alignment.RIGHT_TOP,
    "ui.Alignment.RIGHT_CENTER": ui.Alignment.RIGHT_CENTER,
    "ui.Alignment.RIGHT_BOTTOM": ui.Alignment.RIGHT_BOTTOM,
}
ui.Label("Alignment: ")
with ui.ScrollingFrame(
    height=150,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    with ui.ZStack():
                        ui.Rectangle(name="table", style={"border_color":cl.white, "border_width": 1.0})
                        ui.Circle(
                            radius=10,
                            size_policy=ui.CircleSizePolicy.FIXED,
                            name="orientation",
                            alignment=value,
                            style={"background_color": cl("#aa4444")},
                        )
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
ui.Spacer(height=10)
ui.Label("Arc: ")
with ui.ScrollingFrame(
    height=150,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    with ui.ZStack():
                        ui.Rectangle(name="table", style={"border_color":cl.white, "border_width": 1.0})
                        ui.Circle(
                            radius=10,
                            size_policy=ui.CircleSizePolicy.FIXED,
                            name="orientation",
                            arc=value,
                            style={
                                "background_color": cl("#aa4444"),
                                "border_color": cl.blue,
                                "border_width": 2,
                                },
                        )
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
```

Default circle which is scaled to fit, the alignment is centered:
```
with ui.Frame(height=20):
    ui.Circle(name="default")
```

This circle is scaled to fit with 100 height:
```
with ui.Frame(height=100):
    ui.Circle(name="default")
```

This circle has a fixed radius of 20, the alignment is LEFT_CENTER:
```
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#1111ff"), "border_color": cl("#cc0000"), "border_width": 4}}
with ui.Frame(height=100, style=style):
    with ui.HStack():
        ui.Rectangle(width=40, style={"background_color": cl.white})
        ui.Circle(radius=20, size_policy=ui.CircleSizePolicy.FIXED, alignment=ui.Alignment.LEFT_CENTER)
```

This circle has a fixed radius of 10, the alignment is RIGHT_CENTER
```
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#ff1111"), "border_color": cl.blue, "border_width": 2}}
with ui.Frame(height=100, width=200, style=style):
    with ui.ZStack():
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Circle(radius=10, size_policy=ui.CircleSizePolicy.FIXED, alignment=ui.Alignment.RIGHT_CENTER)
```

This circle has a fixed radius of 10, it has all the same style as the previous one, except its size_policy is `ui.CircleSizePolicy.STRETCH`
```
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#ff1111"), "border_color": cl.blue, "border_width": 2}}
with ui.Frame(height=100, width=200, style=style):
    with ui.ZStack():
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Circle(radius=10, size_policy=ui.CircleSizePolicy.STRETCH, alignment=ui.Alignment.RIGHT_CENTER)
```

