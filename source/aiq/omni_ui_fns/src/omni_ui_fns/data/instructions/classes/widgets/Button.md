# omni.ui.Button

The Button widget provides a command button. Click a button to execute a command. The command button is perhaps the most commonly used widget in any graphical user interface. It is rectangular and typically displays a text label or image describing its action.

Here is a list of styles you can customize on Button:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> background_color (color): the background color of the button
> padding (float): the distance between the content widgets (e.g. Image or Label) and the border of the button
> stack_direction (enum): defines how the content widgets (e.g. Image or Label) on the button are placed.

There are 6 types of stack_directions supported
* ui.Direction.TOP_TO_BOTTOM : layout from top to bottom
* ui.Direction.BOTTOM_TO_TOP : layout from bottom to top
* ui.Direction.LEFT_TO_RIGHT : layout from left to right
* ui.Direction.RIGHT_TO_LEFT : layout from right to left
* ui.Direction.BACK_TO_FRONT : layout from back to front
* ui.Direction.FRONT_TO_BACK : layout from front to back

To control the style of the button content, you can customize `Button.Image` when image on button and `Button.Label` when text on button.

Here is an example showing a list of buttons with different types of the stack directions:
```
from omni.ui import color as cl
direction_flags = {
    "ui.Direction.TOP_TO_BOTTOM": ui.Direction.TOP_TO_BOTTOM,
    "ui.Direction.BOTTOM_TO_TOP": ui.Direction.BOTTOM_TO_TOP,
    "ui.Direction.LEFT_TO_RIGHT": ui.Direction.LEFT_TO_RIGHT,
    "ui.Direction.RIGHT_TO_LEFT": ui.Direction.RIGHT_TO_LEFT,
    "ui.Direction.BACK_TO_FRONT": ui.Direction.BACK_TO_FRONT,
    "ui.Direction.FRONT_TO_BACK": ui.Direction.FRONT_TO_BACK,
}

with ui.ScrollingFrame(
    height=50,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in direction_flags.items():
            button_style = {"Button": {"stack_direction": value}}
            ui_button = ui.Button(
                                key,
                                image_url="resources/icons/Nav_Flymode.png",
                                image_width=24,
                                height=40,
                                style=button_style
                            )
```

Here is an example of two buttons. Pressing the second button makes the name of the first button longer. And press the first button makes the name of itself shorter:
```
from omni.ui import color as cl
style_system = {
    "Button": {
        "background_color": cl(0.85),
        "border_color": cl.yellow,
        "border_width": 2,
        "border_radius": 5,
        "padding": 5,
    },
    "Button.Label": {"color": cl.red, "font_size": 17},
    "Button:hovered": {"background_color": cl("#E5F1FB"), "border_color": cl("#0078D7"), "border_width": 2.0},
    "Button:pressed": {"background_color": cl("#CCE4F7"), "border_color": cl("#005499"), "border_width": 2.0},
}

def make_longer_text(button):
    """Set the text of the button longer"""
    button.text = "Longer " + button.text

def make_shorter_text(button):
    """Set the text of the button shorter"""
    splitted = button.text.split(" ", 1)
    button.text = splitted[1] if len(splitted) > 1 else splitted[0]

with ui.HStack(style=style_system):
    btn_with_text = ui.Button("Text", width=0)
    ui.Button("Press me", width=0, clicked_fn=lambda b=btn_with_text: make_longer_text(b))
    btn_with_text.set_clicked_fn(lambda b=btn_with_text: make_shorter_text(b))
```

Here is an example where you can tweak most of the Button's style and see the results:
```
from omni.ui import color as cl
style = {
    "Button": {"stack_direction": ui.Direction.TOP_TO_BOTTOM},
    "Button.Image": {
        "color": cl("#99CCFF"),
        "image_url": "resources/icons/Learn_128.png",
        "alignment": ui.Alignment.CENTER,
    },
    "Button.Label": {"alignment": ui.Alignment.CENTER},
}

def direction(model, button, style=style):
    value = model.get_item_value_model().get_value_as_int()
    direction = (
        ui.Direction.TOP_TO_BOTTOM,
        ui.Direction.BOTTOM_TO_TOP,
        ui.Direction.LEFT_TO_RIGHT,
        ui.Direction.RIGHT_TO_LEFT,
        ui.Direction.BACK_TO_FRONT,
        ui.Direction.FRONT_TO_BACK,
    )[value]
    style["Button"]["stack_direction"] = direction
    button.set_style(style)

def align(model, button, image, style=style):
    value = model.get_item_value_model().get_value_as_int()
    alignment = (
        ui.Alignment.LEFT_TOP,
        ui.Alignment.LEFT_CENTER,
        ui.Alignment.LEFT_BOTTOM,
        ui.Alignment.CENTER_TOP,
        ui.Alignment.CENTER,
        ui.Alignment.CENTER_BOTTOM,
        ui.Alignment.RIGHT_TOP,
        ui.Alignment.RIGHT_CENTER,
        ui.Alignment.RIGHT_BOTTOM,
    )[value]
    if image:
        style["Button.Image"]["alignment"] = alignment
    else:
        style["Button.Label"]["alignment"] = alignment
    button.set_style(style)

def layout(model, button, padding, style=style):
    if padding == 0:
        padding = "padding"
    elif padding == 1:
        padding = "margin"
    elif padding == 2:
        padding = "margin_width"
    else:
        padding = "margin_height"

    style["Button"][padding] = model.get_value_as_float()
    button.set_style(style)

def spacing(model, button):
    button.spacing = model.get_value_as_float()

button = ui.Button("Label", style=style, width=64, height=64)

with ui.HStack(width=ui.Percent(50)):
    ui.Label('"Button": {"stack_direction"}', name="text")
    options = (
        0,
        "TOP_TO_BOTTOM",
        "BOTTOM_TO_TOP",
        "LEFT_TO_RIGHT",
        "RIGHT_TO_LEFT",
        "BACK_TO_FRONT",
        "FRONT_TO_BACK",
    )
    model = ui.ComboBox(*options).model
    model.add_item_changed_fn(lambda m, i, b=button: direction(m, b))

alignment = (
    4,
    "LEFT_TOP",
    "LEFT_CENTER",
    "LEFT_BOTTOM",
    "CENTER_TOP",
    "CENTER",
    "CENTER_BOTTOM",
    "RIGHT_TOP",
    "RIGHT_CENTER",
    "RIGHT_BOTTOM",
)
with ui.HStack(width=ui.Percent(50)):
    ui.Label('"Button.Image": {"alignment"}', name="text")
    model = ui.ComboBox(*alignment).model
    model.add_item_changed_fn(lambda m, i, b=button: align(m, b, 1))

with ui.HStack(width=ui.Percent(50)):
    ui.Label('"Button.Label": {"alignment"}', name="text")
    model = ui.ComboBox(*alignment).model
    model.add_item_changed_fn(lambda m, i, b=button: align(m, b, 0))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("padding", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 0))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("margin", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 1))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("margin_width", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 2))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("margin_height", name="text")
    model = ui.FloatSlider(min=0, max=500).model
    model.add_value_changed_fn(lambda m, b=button: layout(m, b, 3))

with ui.HStack(width=ui.Percent(50)):
    ui.Label("Button.spacing", name="text")
    model = ui.FloatSlider(min=0, max=50).model
    model.add_value_changed_fn(lambda m, b=button: spacing(m, b))
```

