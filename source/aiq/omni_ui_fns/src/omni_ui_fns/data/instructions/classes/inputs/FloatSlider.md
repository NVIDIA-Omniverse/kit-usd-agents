# omni.ui.FloatSlider

Default slider whose range is between 0 to 1:
```execute 200
ui.FloatSlider()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.FloatSlider(min=0, max=10)
```

With defined Min/Max from the model. Notice the model allows the value range between 0 to 100, but the FloatSlider has a more strict range between 0 to 10.
```execute 200
model = ui.SimpleFloatModel(1.0, min=0, max=100)
ui.FloatSlider(model, min=0, max=10)
```

With styles and rounded slider:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatSlider(
            min=-180,
            max=180,
            style={
                "color": cl.blue,
                "background_color": cl(0.8),
                "draw_mode": ui.SliderDrawMode.HANDLE,
                "secondary_color": cl.red,   # red slider handle
                "secondary_selected_color": cl.green, # slider handle becomes green when selected
                "font_size": 20,
                "border_width": 3,
                "border_color": cl.black,
                "border_radius": 10,
                "padding": 10,
            }
        )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

Filled mode slider with style:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatSlider(
            min=-180,
            max=180,
            style={
                "color": cl.blue,
                "background_color": cl(0.8),
                "draw_mode": ui.SliderDrawMode.FILLED,
                "secondary_color": cl.red, # background color of slider filled part
                "font_size": 20,
                "border_radius": 10,
                "padding": 10,
            }
        )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

Transparent background:
```execute 200
from omni.ui import color as cl
with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatSlider(
                        min=-180,
                        max=180,
                        style={
                            "draw_mode": ui.SliderDrawMode.HANDLE,
                            "background_color": cl.transparent,
                            "color": cl.red,
                            "border_width": 1,
                            "border_color": cl.white,
                        }
                    )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```

Slider with transparent value. Notice the use of `step` attribute
```execute 200
from omni.ui import color as cl
with ui.HStack():
    # a separate float field
    field = ui.FloatField(height=15, width=50)
    # a slider using field's model
    ui.FloatSlider(
        min=0,
        max=20,
        step=0.25,
        model=field.model,
        style={
            "color":cl.transparent,
            "background_color": cl(0.3),
            "draw_mode": ui.SliderDrawMode.HANDLE}
    )
    # default value
    field.model.set_value(12.0)
```

