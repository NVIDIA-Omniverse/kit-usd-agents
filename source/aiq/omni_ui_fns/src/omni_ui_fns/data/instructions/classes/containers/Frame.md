# omni.ui.Frame

Frame is a container that can keep only one child. Each child added to Frame overrides the previous one. This feature is used for creating dynamic layouts. The whole layout can be easily recreated with a simple callback.

Here is a list of styles you can customize on Frame:
> padding (float): the distance between the child widgets and the border of the button

In the following example, you can drag the IntDrag to change the slider value. The buttons are recreated each time the slider changes.
```
self._recreate_ui = ui.Frame(height=40, style={"Frame":{"padding": 5}})

def changed(model, recreate_ui=self._recreate_ui):
    with recreate_ui:
        with ui.HStack():
            for i in range(model.get_value_as_int()):
                ui.Button(f"Button #{i}")

model = ui.IntDrag(min=0, max=10).model
self._sub_recreate = model.subscribe_value_changed_fn(changed)
```

Another feature of Frame is the ability to clip its child. When the content of Frame is bigger than Frame itself, the exceeding part is not drawn if the clipping is on. There are two clipping types: `horizontal_clipping` and `vertical_clipping`.

Here is an example of vertical clipping.
```
with ui.Frame(vertical_clipping=True, height=20):
    ui.Label("This should be clipped vertically. " * 10, word_wrap=True)
```

