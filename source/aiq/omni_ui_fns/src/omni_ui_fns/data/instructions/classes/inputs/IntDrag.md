# omni.ui.IntDrag

Default int drag whose range is -inf and +inf
```execute 200
ui.IntDrag()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.IntDrag(min=-10, max=10)
```

With styles and rounded slider:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.IntDrag(
            min=-180,
            max=180,
            style={
                "color": cl.blue, # text color
                "background_color": cl(0.8), # background color of the unfilled part of the drag
                "secondary_color": cl.purple, # background color of the filled part of the drag
                "font_size": 20,
                "border_width": 4,
                "border_color": cl.black,
                "border_radius": 20,
                "padding": 5,
            }
        )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```


