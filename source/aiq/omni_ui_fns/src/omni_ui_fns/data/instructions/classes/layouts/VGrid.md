# omni.ui.VGrid

VGrid has two modes for cell width:
 - If the user sets column_count, the column width is computed from the grid width.
 - If the user sets column_width, the column count is computed from the grid width.

VGrid also has two modes for height:
 - If the user sets row_height, VGrid uses it to set the height for all the cells. It's the fast mode because it's considered that the cell height never changes. VGrid easily predicts which cells are visible.

- If the user sets nothing, VGrid computes the size of the children. This mode is slower than the previous one, but the advantage is that all the rows can be different custom sizes. VGrid still only draws visible items, but to predict it, it uses cache, which can be big if VGrid has hundreds of thousands of items.

Here is an example of VGrid:
```execute 200
from omni.ui import color as cl
with ui.ScrollingFrame(
    height=250,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    with ui.VGrid(column_width=100, row_height=100):
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

