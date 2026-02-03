# omni.ui.Tooltip

All Widget can be augmented with a tooltip. It can take 2 forms, either a simple ui.Label or a callback when using the callback of `tooltip_fn=` or `widget.set_tooltip_fn()`. You can create the tooltip for any widget.

Except the common style for Fields and Sliders, here is a list of styles you can customize on Line:
> color (color): the color of the text of the tooltip.
> margin_width (float): the width distance between the tooltip content and the parent widget defined boundary
> margin_height (float): the height distance between the tooltip content and the parent widget defined boundary

Here is a simple label tooltip with style when you hover over a button:
```execute 200
from omni.ui import color as cl
tooltip_style = {
    "Tooltip": {
        "background_color": cl("#DDDD00"),
        "color": cl(0.2),
        "padding": 10,
        "border_width": 3,
        "border_color": cl.red,
        "font_size": 20,
        "border_radius": 10}}

ui.Button("Simple Label Tooltip", name="tooltip", width=200, tooltip="I am a text ToolTip", style=tooltip_style)
```

You can create a callback function as the tooltip where you can create any types of widgets you like in the tooltip and layout them. Make the tooltip very illustrative to have Image or Field or Label etc.
```execute 200
from omni.ui import color as cl
def create_tooltip():
    with ui.VStack(width=200, style=tooltip_style):
        with ui.HStack():
            ui.Label("Fancy tooltip", width=150)
            ui.IntField().model.set_value(12)
        ui.Line(height=2, style={"color":cl.white})
        with ui.HStack():
            ui.Label("Anything is possible", width=150)
            ui.StringField().model.set_value("you bet")
        image_source = "resources/desktop-icons/omniverse_512.png"
        ui.Image(
            image_source,
            width=200,
            height=200,
            alignment=ui.Alignment.CENTER,
            style={"margin": 0},
        )
tooltip_style = {
    "Tooltip": {
        "background_color": cl(0.2),
        "border_width": 2,
        "border_radius": 5,
        "margin_width": 5,
        "margin_height": 10
        },
    }
ui.Button("Callback function Tooltip", width=200, style=tooltip_style, tooltip_fn=create_tooltip)
```

You can define a fixed position for tooltip:
```execute 200
ui.Button("Fixed-position Tooltip", width=200, tooltip="Hello World", tooltip_offset_y=22)
```

You can also define a random position for tooltip:
```execute 200
import random
button = ui.Button("Random-position Tooltip", width=200, tooltip_offset_y=22)

def create_tooltip(button=button):
    button.tooltip_offset_x = random.randint(0, 200)
    ui.Label("Hello World")

button.set_tooltip_fn(create_tooltip)
```

### omni.ui.StringField
The StringField widget is a one-line text editor. A field allows the user to enter and edit a single line of plain text. It's implemented using the model-delegate-view pattern and uses AbstractValueModel as the central component of the system.

Here is a list of common style you can customize on Fields:
> background_color (color): the background color of the field or slider
> border_color (color): the border color if the field or slider background has a border
> border_radius (float): the border radius if the user wants to round the field or slider
> border_width (float): the border width if the field or slider background has a border
> padding (float): the distance between the text and the border of the field or slider
> font_size (float): the size of the text in the field or slider
> color (color): the color of the text
> background_selected_color (color): the background color of the selected text

The following example demonstrates how to connect a StringField and a Label. You can type anything into the StringField.

```execute 200
from omni.ui import color as cl
field_style = {
    "Field": {
        "background_color": cl(0.8),
        "border_color": cl.blue,
        "background_selected_color": cl.yellow,
        "border_radius": 5,
        "border_width": 1,
        "color": cl.red,
        "font_size": 20.0,
        "padding": 5,
    },
    "Field:pressed": {"background_color": cl.white, "border_color": cl.green, "border_width": 2, "padding": 8},
}

def setText(label, text):
    """Sets text on the label"""
    # This function exists because lambda cannot contain assignment
    label.text = f"You wrote '{text}'"

with ui.HStack():
    field = ui.StringField(style=field_style)
    ui.Spacer(width=5)
    label = ui.Label("", name="text")
    field.model.add_value_changed_fn(lambda m, label=label: setText(label, m.get_value_as_string()))
    ui.Spacer(width=10)
```

The following example demonstrates that the CheckBox's model decides the content of the Field. Click to edit and update the string field value also updates the value of the CheckBox. The field can only have one of the two options, either 'True' or 'False', because the model only supports those two possibilities.

```execute 200
from omni.ui import color as cl
with ui.HStack():
    field = ui.StringField(width=100, style={"background_color": cl.black})
    checkbox = ui.CheckBox(width=0)
    field.model = checkbox.model
```

In this example, the field can have anything because the model accepts any string. The model returns bool for checkbox, and the checkbox is unchecked when the string is empty or 'False'.

```execute 200
from omni.ui import color as cl
with ui.HStack():
    field = ui.StringField(width=100, style={"background_color": cl.black})
    checkbox = ui.CheckBox(width=0)
    checkbox.model = field.model
```

**Multiline StringField**
Property `multiline` of `StringField` allows users to press enter and create a new line. It's possible to finish editing with Ctrl-Enter.
```execute 200
from omni.ui import color as cl
import inspect

field_style = {
    "Field": {
        "background_color": cl(0.8),
        "color": cl.black,
    },
    "Field:pressed": {"background_color": cl(0.8)},
}

field_callbacks = lambda: field_callbacks()
with ui.Frame(style=field_style, height=200):
    model = ui.SimpleStringModel("hello \nworld \n")
    field = ui.StringField(model, multiline=True)
```

