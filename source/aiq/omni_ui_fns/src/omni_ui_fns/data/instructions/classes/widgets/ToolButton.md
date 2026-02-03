# omni.ui.ToolButton

ToolButton is functionally similar to Button, but provides a model that determines if the button is checked. This button toggles between checked (on) and unchecked (off) when the user clicks it.

Here is an example of a ToolButton:

```
def update_label(model, label):
    checked = model.get_value_as_bool()
    label.text = f"The check status button is {checked}"

with ui.VStack(spacing=5):
    model = ui.ToolButton(text="click", name="toolbutton", width=100).model
    checked = model.get_value_as_bool()
    label = ui.Label(f"The check status button is {checked}")
    model.add_value_changed_fn(lambda m, l=label: update_label(m, l))
```

