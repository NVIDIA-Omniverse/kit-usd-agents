# Shades
Shades are used to have multiple named color palettes with the ability for runtime switch. For example, one App could have several ui themes users can switch during using the App.

The shade can be defined with the following code:

```python
    cl.shade(cl("#FF6600"), red=cl("#0000FF"), green=cl("#66FF00"))
```

It can be assigned to the color style. It's possible to switch the color with the following command globally:

```python
    cl.set_shade("red")
```

## Example
```execute 200
from omni.ui import color as cl
from omni.ui import constant as fl
from functools import partial

def set_color(color):
    cl.example_color = color

def set_width(value):
    fl.example_width = value

cl.example_color = cl.green
fl.example_width = 1.0

with ui.HStack(height=100, spacing=5):
    with ui.ZStack():
        ui.Rectangle(
            style={
                "background_color": cl.shade(
                    "aqua",
                    orange=cl.orange,
                    another=cl.example_color,
                    transparent=cl(0, 0, 0, 0),
                    black=cl.black,
                ),
                "border_width": fl.shade(1, orange=4, another=8),
                "border_radius": fl.one,
                "border_color": cl.black,
            },
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.Rectangle(\n"
                "\tstyle={\n"
                '\t\t"background_color":\n'
                "\t\t\tcl.shade(\n"
                '\t\t\t\t"aqua",\n'
                "\t\t\t\torange=cl(1, 0.5, 0),\n"
                "\t\t\t\tanother=cl.example_color),\n"
                '\t\t"border_width":\n'
                "\t\t\tfl.shade(1, orange=4, another=8)})",
                style={"color": cl.black, "margin": 15},
                width=0,
            )
            ui.Spacer()

    with ui.ZStack():
        ui.Rectangle(
            style={
                "background_color": cl.example_color,
                "border_width": fl.example_width,
                "border_radius": fl.one,
                "border_color": cl.black,
            }
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.Rectangle(\n"
                "\tstyle={\n"
                '\t\t"background_color": cl.example_color,\n'
                '\t\t"border_width": fl.example_width)})',
                style={"color": cl.black, "margin": 15},
                width=0,
            )
            ui.Spacer()

with ui.VStack(style={"Button": {"background_color": cl("097EFF")}}):
    ui.Label("Click the following buttons to change the shader of the left rectangle")
    with ui.HStack():
        ui.Button("cl.set_shade()", clicked_fn=partial(cl.set_shade, ""))
        ui.Button('cl.set_shade("orange")', clicked_fn=partial(cl.set_shade, "orange"))
        ui.Button('cl.set_shade("another")', clicked_fn=partial(cl.set_shade, "another"))
    ui.Label("Click the following buttons to change the border width of the right rectangle")
    with ui.HStack():
        ui.Button("fl.example_width = 1", clicked_fn=partial(set_width, 1))
        ui.Button("fl.example_width = 4", clicked_fn=partial(set_width, 4))
    ui.Label("Click the following buttons to change the background color of both rectangles")
    with ui.HStack():
        ui.Button('cl.example_color = "green"', clicked_fn=partial(set_color, "green"))
        ui.Button("cl.example_color = cl(0.8)", clicked_fn=partial(set_color, cl(0.8)))
    ## Double comment means hide from shippet
    ui.Spacer(height=15)
    ##
```

## URL Shades Example
It's also possible to use shades for specifying shortcuts to the images and style-based paths.

```execute 200
from omni.ui import color as cl
from omni.ui.url_utils import url
from functools import partial

def set_url(url_path: str):
    url.example_url = url_path

walk = "resources/icons/Nav_Walkmode.png"
fly = "resources/icons/Nav_Flymode.png"

url.example_url = walk

with ui.HStack(height=100, spacing=5):
    with ui.ZStack():
        ui.Image(height=100, style={"image_url": url.example_url})
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                'ui.Image(\n\tstyle={"image_url": cl.example_url})\n',
                style={"color": cl.black, "font_size": 12, "margin": 15},
                width=0,
            )
            ui.Spacer()
    with ui.ZStack():
        ui.ImageWithProvider(
            height=100,
            style={
                "image_url": url.shade(
                    "resources/icons/Move_local_64.png",
                    another="resources/icons/Move_64.png",
                    orange="resources/icons/Rotate_local_64.png",
                )
            }
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.ImageWithProvider(\n"
                "\tstyle={\n"
                '\t\t"image_url":\n'
                "\t\t\tst.shade(\n"
                '\t\t\t\t"Move_local_64.png",\n'
                '\t\t\t\tanother="Move_64.png")})\n',
                style={"color": cl.black, "font_size": 12, "margin": 15},
                width=0,
            )
            ui.Spacer()

with ui.HStack():
    # buttons to change the url for the image
    with ui.VStack():
        ui.Button("url.example_url = Nav_Walkmode.png", clicked_fn=partial(set_url, walk))
        ui.Button("url.example_url = Nav_Flymode.png", clicked_fn=partial(set_url, fly))
    # buttons to switch between shades to a different image
    with ui.VStack():
        ui.Button("ui.set_shade()", clicked_fn=partial(ui.set_shade, ""))
        ui.Button('ui.set_shade("another")', clicked_fn=partial(ui.set_shade, "another"))
```
