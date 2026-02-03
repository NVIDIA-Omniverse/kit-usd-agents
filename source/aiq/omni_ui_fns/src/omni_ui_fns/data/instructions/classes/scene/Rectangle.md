# omni.ui.Rectangle

Rectangle is a shape with four sides and four corners. You can use Rectangle to draw rectangle shapes, or mix it with other controls e.g. using ZStack to create an advanced look.

Here is a list of styles you can customize on Rectangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border
> background_gradient_color (color): the gradient color on the top part of the rectangle
> border_radius (float): default rectangle has 4 right corner angles, border_radius defines the radius of the corner angle if the user wants to round the rectangle corner. We only support one border_radius across all the corners, but users can choose which corner to be rounded.
> corner_flag (enum): defines which corner or corners to be rounded

Here is a list of the supported corner flags:
```
from omni.ui import color as cl
corner_flags = {
    "ui.CornerFlag.NONE": ui.CornerFlag.NONE,
    "ui.CornerFlag.TOP_LEFT": ui.CornerFlag.TOP_LEFT,
    "ui.CornerFlag.TOP_RIGHT": ui.CornerFlag.TOP_RIGHT,
    "ui.CornerFlag.BOTTOM_LEFT": ui.CornerFlag.BOTTOM_LEFT,
    "ui.CornerFlag.BOTTOM_RIGHT": ui.CornerFlag.BOTTOM_RIGHT,
    "ui.CornerFlag.TOP": ui.CornerFlag.TOP,
    "ui.CornerFlag.BOTTOM": ui.CornerFlag.BOTTOM,
    "ui.CornerFlag.LEFT": ui.CornerFlag.LEFT,
    "ui.CornerFlag.RIGHT": ui.CornerFlag.RIGHT,
    "ui.CornerFlag.ALL": ui.CornerFlag.ALL,
}

with ui.ScrollingFrame(
    height=100,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in corner_flags.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}):
                    ui.Rectangle(
                        style={"background_color": cl("#aa4444"), "border_radius": 20.0, "corner_flag": value}
                    )
                    ui.Spacer(height=10)
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
```
Here are a few examples of Rectangle using different selections of styles:

Default rectangle which is scaled to fit:
```
with ui.Frame(height=20):
    ui.Rectangle(name="default")
```

This rectangle uses its own style to control colors and shape. Notice how three colors "background_color", "border_color" and "border_color" are affecting the look of the rectangle:
```
from omni.ui import color as cl
with ui.Frame(height=40):
    ui.Rectangle(style={"Rectangle":{
        "background_color":cl("#aa4444"),
        "border_color":cl("#22FF22"),
        "background_gradient_color": cl("#4444aa"),
        "border_width": 2.0,
        "border_radius": 5.0}})
```

This rectangle uses fixed width and height. Notice the `border_color` is not doing anything if `border_width` is not defined.
```
from omni.ui import color as cl
with ui.Frame(height=20):
    ui.Rectangle(width=40, height=10, style={"background_color":cl(0.6), "border_color":cl("#ff2222")})
```

Compose with ZStack for an advanced look
```
from omni.ui import color as cl
with ui.Frame(height=20):
    with ui.ZStack(height=20):
        ui.Rectangle(width=150,
            style={"background_color":cl(0.6),
                    "border_color":cl(0.1),
                    "border_width": 1.0,
                    "border_radius": 8.0} )
        with ui.HStack():
            ui.Spacer(width=10)
            ui.Image("resources/icons/Cloud.png", width=20, height=20 )
            ui.Label( "Search Field", style={"color":cl(0.875)})
```

