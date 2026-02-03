# omni.ui.MultiIntField

Each of the field value could be changed by editing
```execute 200
from omni.ui import color as cl
field_style = {
    "Field": {
        "background_color": cl(0.8),
        "border_color": cl.blue,
        "border_radius": 5,
        "border_width": 1,
        "color": cl.red,
        "font_size": 20.0,
        "padding": 5,
    },
    "Field:pressed": {"background_color": cl.white, "border_color": cl.green, "border_width": 2, "padding": 8},
}

ui.MultiIntField(0, 0, 0, 0, style=field_style)
```

