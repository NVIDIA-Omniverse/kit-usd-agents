# omni.ui.IntSlider

Default slider whose range is between 0 to 100:
```execute 200
ui.IntSlider()
```

With defined Min/Max whose range is between min to max. Note that the handle width is much wider.
```execute 200
ui.IntSlider(min=0, max=20)
```

With style:
```execute 200
from omni.ui import color as cl
with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.IntSlider(
            min=0,
            max=20,
            style={
                "background_color": cl("#BBFFBB"),
                "color": cl.purple,
                "draw_mode": ui.SliderDrawMode.HANDLE,
                "secondary_color": cl.green, # green slider handle
                "secondary_selected_color": cl.red, # slider handle becomes red when selected
                "font_size": 14.0,
                "border_width": 3,
                "border_color": cl.green,
                "padding": 5,
            }
        ).model.set_value(4)
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

