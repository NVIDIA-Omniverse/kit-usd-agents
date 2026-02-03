# omni.ui.CollapsableFrame

CollapsableFrame is a frame widget that can hide or show its content. It has two states: expanded and collapsed. When it's collapsed, it looks like a button. If it's expanded, it looks like a button and a frame with the content. It's handy to group properties, and temporarily hide them to get more space for something else.

Here is a list of styles you can customize on CollapsableFrame:
> background_color (color): the background color of the CollapsableFrame widget
> secondary_color (color): the background color of the CollapsableFrame's header
> border_radius (float): the border radius if user wants to round the CollapsableFrame
> border_color (color): the border color if the CollapsableFrame has a border
> border_width (float): the border width if the CollapsableFrame has a border
> padding (float): the distance between the header or the content to the border of the CollapsableFrame
> margin (float): the distance between the CollapsableFrame and other widgets

Here is a default `CollapsableFrame` example:
```
with ui.CollapsableFrame("Header"):
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")
```

It's possible to use a custom header.
```
from omni.ui import color as cl
def custom_header(collapsed, title):
    with ui.HStack():
        with ui.ZStack(width=30):
            ui.Circle(name="title")
            with ui.HStack():
                ui.Spacer()
                align = ui.Alignment.V_CENTER
                ui.Line(name="title", width=6, alignment=align)
                ui.Spacer()
            if collapsed:
                with ui.VStack():
                    ui.Spacer()
                    align = ui.Alignment.H_CENTER
                    ui.Line(name="title", height=6, alignment=align)
                    ui.Spacer()

        ui.Label(title, name="title")

style = {
    "CollapsableFrame": {
        "background_color": cl(0.5),
        "secondary_color": cl.red, # with a red header
        "border_radius": 10,
        "border_color": cl.blue,
        "border_width": 2,
    },
    "CollapsableFrame:hovered": {"secondary_color": cl.green}, # header becomes green when hovered
    "CollapsableFrame:pressed": {"secondary_color": cl.yellow}, # header becomes yellow when pressed
    "Label::title": {"color": cl.white},
    "Circle::title": {
        "color": cl.yellow,
        "background_color": cl.transparent,
        "border_color": cl(0.9),
        "border_width": 0.75,
    },
    "Line::title": {"color": cl(0.9), "border_width": 1},
}

ui.Spacer(height=5)
with ui.HStack():
    ui.Spacer(width=5)
    with ui.CollapsableFrame("Header", build_header_fn=custom_header, style=style):
        with ui.VStack(height=0):
            ui.Button("Hello World")
            ui.Button("Hello World")
    ui.Spacer(width=5)
ui.Spacer(height=5)
```

This example demonstrates how padding and margin work in the collapsable frame.
```
from omni.ui import color as cl
style = {
    "CollapsableFrame": {
        "border_color": cl("#005B96"),
        "border_radius": 4,
        "border_width": 2,
        "padding": 0,
        "margin": 0,
    }
}
frame = ui.CollapsableFrame("Header", style=style)
with frame:
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")

def set_style(field, model, style=style, frame=frame):
    frame_style = style["CollapsableFrame"]
    frame_style[field] = model.get_value_as_float()
    frame.set_style(style)

with ui.HStack():
    ui.Label("Padding:", width=ui.Percent(10), name="text")
    model = ui.FloatSlider(min=0, max=50).model
model.add_value_changed_fn(lambda m: set_style("padding", m))

with ui.HStack():
    ui.Label("Margin:", width=ui.Percent(10), name="text")
    model = ui.FloatSlider(min=0, max=50).model
model.add_value_changed_fn(lambda m: set_style("margin", m))
```


