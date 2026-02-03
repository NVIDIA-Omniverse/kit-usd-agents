# omni.ui.ProgressBar

A ProgressBar is a widget that indicates the progress of an operation.

Here is a list of styles you can customize on ProgressBar:
> background_color (color): the background color of the field or slider
> border_color (color): the border color if the field or slider background has a border
> border_radius (float): the border radius if the user wants to round the field or slider
> border_width (float): the border width if the field or slider background has a border
> padding (float): the distance between the text and the border of the field or slider
> font_size (float): the size of the text in the field or slider
> color (color): the color of the progress bar indicating the progress value of the progress bar in the portion of the overall value
> secondary_color (color): the color of the text indicating the progress value

In the following example, it shows how to use ProgressBar and override the style of the overlay text.
```execute 200
from omni.ui import color as cl
class CustomProgressValueModel(ui.AbstractValueModel):
    """An example of custom float model that can be used for progress bar"""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def set_value(self, value):
        """Reimplemented set"""
        try:
            value = float(value)
        except ValueError:
            value = None
        if value != self._value:
            # Tell the widget that the model is changed
            self._value = value
            self._value_changed()

    def get_value_as_float(self):
        return self._value

    def get_value_as_string(self):
        return "Custom Overlay"

with ui.VStack(spacing=5):
    # Create ProgressBar
    first = ui.ProgressBar()
    # Range is [0.0, 1.0]
    first.model.set_value(0.5)

    second = ui.ProgressBar()
    second.model.set_value(1.0)

    # Overrides the overlay of ProgressBar
    model = CustomProgressValueModel(0.8)
    third = ui.ProgressBar(model)
    third.model.set_value(0.1)

    # Styling its color
    fourth = ui.ProgressBar(style={"color": cl("#0000dd")})
    fourth.model.set_value(0.3)

    # Styling its border width
    ui.ProgressBar(style={"border_width": 2, "border_color": cl("#dd0000"), "color": cl("#0000dd")}).model.set_value(0.7)

    # Styling its border radius
    ui.ProgressBar(style={"border_radius": 100, "color": cl("#0000dd")}).model.set_value(0.6)

    # Styling its background color
    ui.ProgressBar(style={"border_radius": 10, "background_color": cl("#0000dd")}).model.set_value(0.6)

    # Styling the text color
    ui.ProgressBar(style={"ProgressBar":{"border_radius": 30, "secondary_color": cl("#00dddd"), "font_size": 20}}).model.set_value(0.6)

    # Two progress bars in a row with padding
    with ui.HStack():
        ui.ProgressBar(style={"color": cl("#0000dd"), "padding": 100}).model.set_value(1.0)
        ui.ProgressBar().model.set_value(0.0)
```

