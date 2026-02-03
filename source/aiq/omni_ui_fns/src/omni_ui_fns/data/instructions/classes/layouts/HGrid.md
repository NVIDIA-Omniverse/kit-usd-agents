# omni.ui.HGrid

HGrid works exactly like VGrid, but with swapped width and height.
```execute 200
from omni.ui import color as cl
with ui.ScrollingFrame(
    height=250,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
):
    with ui.HGrid(column_width=100, row_height=100):
        for i in range(100):
            with ui.ZStack():
                ui.Rectangle(
                    style={
                        "border_color": cl.red,
                        "background_color": cl.white,
                        "border_width": 1,
                        "margin": 0,
                    }
                )
                ui.Label(f"{i}", style={"margin": 5})
```

