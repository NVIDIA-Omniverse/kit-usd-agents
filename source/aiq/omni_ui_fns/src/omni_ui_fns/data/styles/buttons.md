# Buttons and Images

## Common Styling for Buttons and Images
Here is a list of common style you can customize on Buttons and Images:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary

## Button
The Button widget provides a command button. Click a button to execute a command. The command button is perhaps the most commonly used widget in any graphical user interface. It is rectangular and typically displays a text label or image describing its action.

Except the common style for Buttons and Images, here is a list of styles you can customize on Button:
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
```execute 200
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
```execute 200
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
```execute 200
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

## Radio Button
RadioButton is the widget that allows the user to choose only one from a predefined set of mutually exclusive options.

RadioButtons are arranged in collections of two or more buttons within a RadioCollection, which is the central component of the system and controls the behavior of all the RadioButtons in the collection.

Except the common style for Buttons and Images, here is a list of styles you can customize on RadioButton:
> background_color (color): the background color of the RadioButton
> padding (float): the distance between the the RadioButton content widget (e.g. Image) and the RadioButton border

To control the style of the button image, you can customize `RadioButton.Image`. For example RadioButton.Image's image_url defines the image when it's not checked. You can define the image for checked status with `RadioButton.Image:checked` style.

Here is an example of RadioCollection which contains 5 RadioButtons with style. Also there is an IntSlider which shares the model with the RadioCollection, so that when RadioButton value or the IntSlider value changes, the other one will update too.

```execute 200
from omni.ui import color as cl
style = {
            "RadioButton": {
                "background_color": cl.cyan,
                "margin_width": 2,
                "padding": 1,
                "border_radius": 0,
                "border_color": cl.white,
                "border_width": 1.0},
            "RadioButton.Image": {
                "image_url": f"../exts/omni.kit.documentation.ui.style/icons/radio_off.svg",
            },
            "RadioButton.Image:checked": {
                "image_url": f"../exts/omni.kit.documentation.ui.style/icons/radio_on.svg"},
        }

collection = ui.RadioCollection()
for i in range(5):
    with ui.HStack(style=style):
        ui.RadioButton(radio_collection=collection, width=30, height=30)
        ui.Label(f"Option {i}", name="text")

ui.IntSlider(collection.model, min=0, max=4)
```

## ToolButton
ToolButton is functionally similar to Button, but provides a model that determines if the button is checked. This button toggles between checked (on) and unchecked (off) when the user clicks it.

Here is an example of a ToolButton:

```execute 200
def update_label(model, label):
    checked = model.get_value_as_bool()
    label.text = f"The check status button is {checked}"

with ui.VStack(spacing=5):
    model = ui.ToolButton(text="click", name="toolbutton", width=100).model
    checked = model.get_value_as_bool()
    label = ui.Label(f"The check status button is {checked}")
    model.add_value_changed_fn(lambda m, l=label: update_label(m, l))

```

## ColorWidget
The ColorWidget is a button that displays the color from the item model and can open a picker window. The color dialog's function is to allow users to choose color.

Except the common style for Buttons and Images, here is a list of styles you can customize on ColorWidget:
> background_color (color): the background color of the tooltip widget when hover over onto the ColorWidget
> color (color): the text color of the tooltip widget when hover over onto the ColorWidget

Here is an example of a ColorWidget with three FloatFields. The ColorWidget model is shared with the FloatFields so that users can click and edit the field value to change the ColorWidget's color, and the value change of the ColorWidget will also reflect in the value change of the FloatFields.

```execute 200
from omni.ui import color as cl
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0, style={"ColorWidget":{
        "border_width": 2,
        "border_color": cl.white,
        "border_radius": 4,
        "color": cl.pink,
        "margin": 2
    }}).model
    for item in color_model.get_item_children():
        component = color_model.get_item_value_model(item)
        ui.FloatField(component)
```

Here is an example of a ColorWidget with three FloatDrags. The ColorWidget model is shared with the FloatDrags so that users can drag the field value to change the color, and the value change of the ColorWidget will also reflect in the value change of the FloatDrags.

```execute 200
from omni.ui import color as cl
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(0.125, 0.25, 0.5, width=0, height=0, style={
        "background_color": cl.pink
    }).model
    for item in color_model.get_item_children():
        component = color_model.get_item_value_model(item)
        ui.FloatDrag(component, min=0, max=1)
```

Here is an example of a ColorWidget with a ComboBox. The ColorWidget model is shared with the ComboBox. Only the value change of the ColorWidget will reflect in the value change of the ComboBox.

```execute 200
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0).model
    ui.ComboBox(color_model)
```

Here is an interactive example with USD. You can create a Mesh in the Stage. Choose `Pixar Storm` as the render. Select the mesh and use this ColorWidget to change the color of the mesh. You can use `Ctrl+z` for undoing and `Ctrl+y` for redoing.

```execute 200
import weakref
import omni.kit.commands
from omni.usd.commands import UsdStageHelper
from pxr import UsdGeom
from pxr import Gf
import omni.usd
from carb.eventdispatcher import get_eventdispatcher

class SetDisplayColorCommand(omni.kit.commands.Command, UsdStageHelper):
    """
    Change prim display color undoable **Command**. Unlike ChangePropertyCommand, it can undo property creation.

    Args:
        gprim (Gprim): Prim to change display color on.
        value: Value to change to.
        value: Value to undo to.
    """

    def __init__(self, gprim: UsdGeom.Gprim, color: any, prev: any):
        self._gprim = gprim
        self._color = color
        self._prev = prev

    def do(self):
        color_attr = self._gprim.CreateDisplayColorAttr()
        color_attr.Set([self._color])

    def undo(self):
        color_attr = self._gprim.GetDisplayColorAttr()
        if self._prev is None:
            color_attr.Clear()
        else:
            color_attr.Set([self._prev])

omni.kit.commands.register(SetDisplayColorCommand)

class FloatModel(ui.SimpleFloatModel):
    def __init__(self, parent):
        super().__init__()
        self._parent = weakref.ref(parent)

    def begin_edit(self):
        parent = self._parent()
        parent.begin_edit(None)

    def end_edit(self):
        parent = self._parent()
        parent.end_edit(None)


class USDColorItem(ui.AbstractItem):
    def __init__(self, model):
        super().__init__()
        self.model = model


class USDColorModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()

        # Create root model
        self._root_model = ui.SimpleIntModel()
        self._root_model.add_value_changed_fn(lambda a: self._item_changed(None))

        # Create three models per component
        self._items = [USDColorItem(FloatModel(self)) for i in range(3)]
        for item in self._items:
            item.model.add_value_changed_fn(lambda a, item=item: self._on_value_changed(item))

        # Omniverse contexts
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._stage_event_sub = get_eventdispatcher().observe_event(
            observer_name="omni.example.ui observer",
            event_name=self._usd_context.stage_event_name(omni.usd.StageEventType.SELECTION_CHANGED),
            on_event=lambda _: self._on_selection_changed()
        )

        # Privates
        self._subscription = None
        self._gprim = None
        self._prev_color = None
        self._edit_mode_counter = 0

    def _on_selection_changed(self):
        """Called when the user changes the selection"""
        selection = self._selection.get_selected_prim_paths()
        stage = self._usd_context.get_stage()
        self._subscription = None
        self._gprim = None

        # When TC runs tests, it's possible that stage is None
        if selection and stage:
            self._gprim = UsdGeom.Gprim.Get(stage, selection[0])
            if self._gprim:
                color_attr = self._gprim.GetDisplayColorAttr()
                usd_watcher = omni.usd.get_watcher()
                self._subscription = usd_watcher.subscribe_to_change_info_path(
                    color_attr.GetPath(), self._on_usd_changed
                )

        # Change the widget color
        self._on_usd_changed()

    def _on_value_changed(self, item):
        """Called when the submodel is changed"""
        if not self._gprim:
            return

        if self._edit_mode_counter > 0:
            # Change USD only if we are in edit mode.
            color_attr = self._gprim.CreateDisplayColorAttr()

            color = Gf.Vec3f(
                self._items[0].model.get_value_as_float(),
                self._items[1].model.get_value_as_float(),
                self._items[2].model.get_value_as_float(),
            )

            color_attr.Set([color])

        self._item_changed(item)

    def _on_usd_changed(self, path=None):
        """Called with UsdWatcher when something in USD is changed"""
        color = self._get_current_color() or Gf.Vec3f(0.0)

        for i in range(len(self._items)):
            self._items[i].model.set_value(color[i])

    def _get_current_color(self):
        """Returns color of the current object"""
        if self._gprim:
            color_attr = self._gprim.GetDisplayColorAttr()
            if color_attr:
                color_array = color_attr.Get()
                if color_array:
                    return color_array[0]

    def get_item_children(self, item):
        """Reimplemented from the base class"""
        return self._items

    def get_item_value_model(self, item, column_id):
        """Reimplemented from the base class"""
        if item is None:
            return self._root_model
        return item.model

    def begin_edit(self, item):
        """
        Reimplemented from the base class.
        Called when the user starts editing.
        """
        if self._edit_mode_counter == 0:
            self._prev_color = self._get_current_color()

        self._edit_mode_counter += 1

    def end_edit(self, item):
        """
        Reimplemented from the base class.
        Called when the user finishes editing.
        """
        self._edit_mode_counter -= 1

        if not self._gprim or self._edit_mode_counter > 0:
            return

        color = Gf.Vec3f(
            self._items[0].model.get_value_as_float(),
            self._items[1].model.get_value_as_float(),
            self._items[2].model.get_value_as_float(),
        )

        omni.kit.commands.execute("SetDisplayColor", gprim=self._gprim, color=color, prev=self._prev_color)


with ui.HStack(spacing=5):
    ui.ColorWidget(USDColorModel(), width=0)
    ui.Label("Interactive ColorWidget with USD", name="text")
```

## Image
The Image type displays an image. The source of the image is specified as a URL using the source property. By default, specifying the width and height of the item makes the image to be scaled to fit that size. This behavior can be changed by setting the `fill_policy` property, allowing the image to be stretched or scaled instead. The property alignment controls how the scaled image is aligned in the parent defined space.

Except the common style for Buttons and Images, here is a list of styles you can customize on Image:
> image_url (str): the url path of the image source
> color (color): the overlay color of the image
> corner_flag (enum): defines which corner or corners to be rounded. The supported corner flags are the same as Rectangle since Image is eventually an image on top of a rectangle under the hood.
> fill_policy (enum): defines how the Image fills the rectangle.
There are three types of fill_policy
* ui.FillPolicy.STRETCH: stretch the image to fill the entire rectangle.
* ui.FillPolicy.PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.FillPolicy.PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary
> alignment (enum): defines how the image is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Default Image is scaled uniformly to fit without stretching or cropping (ui.FillPolicy.PRESERVE_ASPECT_FIT), and aligned to ui.Alignment.CENTER:
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source)
```

The image is stretched to fit and aligned to the left
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.STRETCH, alignment=ui.Alignment.LEFT_CENTER)
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the top
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_TOP)
```

The image is scaled uniformly to fit without cropping and aligned to the right. Notice the fill_policy and alignment are defined in style.
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, style={
        "Image": {
            "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
            "alignment": ui.Alignment.RIGHT_CENTER,
            "margin": 5}})
```

The image has rounded corners and an overlayed color. Note image_url is in the style dictionary.
```execute 200
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(style={"image_url": source, "border_radius": 10, "color": cl("#5eb3ff")})
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the bottom, with a blue border.
```execute 200
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(
        source,
        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_BOTTOM,
        style={"Image":{
            "border_width": 5,
            "border_color": cl("#1ab3ff"),
            "corner_flag": ui.CornerFlag.TOP,
            "border_radius": 15}})
```

The image is arranged in a HStack with different margin styles defined. Note image_url is in the style dict.
```execute 200
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(height=100):
    with ui.HStack(spacing =5, style={"Image":{'image_url': source}}):
        ui.Image()
        ui.Image(style={"Image":{"margin_height": 15}})
        ui.Image()
        ui.Image(style={"Image":{"margin_width": 20}})
        ui.Image()
        ui.Image(style={"Image":{"margin": 10}})
        ui.Image()
```

It's possible to set a different image per style state. And switch them depending on the mouse hovering, selection state, etc.
```execute 200

styles = [
    {
        "": {"image_url": "resources/icons/Nav_Walkmode.png"},
        ":hovered": {"image_url": "resources/icons/Nav_Flymode.png"},
    },
    {
        "": {"image_url": "resources/icons/Move_local_64.png"},
        ":hovered": {"image_url": "resources/icons/Move_64.png"},
    },
    {
        "": {"image_url": "resources/icons/Rotate_local_64.png"},
        ":hovered": {"image_url": "resources/icons/Rotate_global.png"},
    },
]

def set_image(model, image):
    value = model.get_item_value_model().get_value_as_int()
    image.set_style(styles[value])

with ui.Frame(height=80):
    with ui.VStack():
        image = ui.Image(width=64, height=64, style=styles[0])
        with ui.HStack(width=ui.Percent(50)):
            ui.Label("Select a texture to display", name="text")
            model = ui.ComboBox(0, "Navigation", "Move", "Rotate").model
            model.add_item_changed_fn(lambda m, i, im=image: set_image(m, im))
```

## ImageWithProvider
ImageWithProvider also displays an image just like Image. It is a much more advanced image widget. ImageWithProvider blocks until the image is loaded, Image doesn't block. Sometimes Image blinks because when the first frame is created, the image is not loaded. Users are recommended to use ImageWithProvider if the UI is updated pretty often. Because it doesn't blink when recreating.

It has the almost the same style list as Image, except the fill_policy has different enum values.
> fill_policy (enum): defines how the Image fills the rectangle.
There are three types of fill_policy
* ui.IwpFillPolicy.IWP_STRETCH: stretch the image to fill the entire rectangle.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary

The image source comes from `ImageProvider` which could be `ByteImageProvider`, `RasterImageProvider` or `VectorImageProvider`.

`RasterImageProvider` and `VectorImageProvider` are using image urls like Image. Here is an example taken from Image. Notice the fill_policy value difference.
```execute 200
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.ImageWithProvider(
        source,
        style={
            "ImageWithProvider": {
            "border_width": 5,
            "border_color": cl("#1ab3ff"),
            "corner_flag": ui.CornerFlag.TOP,
            "border_radius": 15,
            "fill_policy": ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP,
            "alignment": ui.Alignment.CENTER_BOTTOM}})
```

`ByteImageProvider` is really useful to create gradient images. Here is an example:
```execute 200
self._byte_provider = ui.ByteImageProvider()
self._byte_provider.set_bytes_data([
    255, 0, 0, 255,    # red
    255, 255, 0, 255,  # yellow
    0,  255, 0, 255,   # green
    0, 255, 255, 255,  # cyan
    0, 0, 255, 255],   # blue
    [5, 1])            # size
with ui.Frame(height=20):
    ui.ImageWithProvider(self._byte_provider,fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
```

## Plot
The Plot class displays a line or histogram image. The data of the image is specified as a data array or a provider function.

Except the common style for Buttons and Images, here is a list of styles you can customize on Plot:
> color (color): the color of the plot, line color in the line typed plot or rectangle bar color in the histogram typed plot
> selected_color (color): the selected color of the plot, dot in the line typed plot and rectangle bar in the histogram typed plot
> background_color (color): the background color of the plot
> secondary_color (color): the color of the text and the border of the text box which shows the plot selection value
> background_selected_color (color): the background color of the text box which shows the plot selection value

Here are couple of examples of Plots:
```execute 200
import math
from omni.ui import color as cl
data = []
for i in range(360):
    data.append(math.cos(math.radians(i)))

def on_data_provider(index):
    return math.sin(math.radians(index))

with ui.Frame(height=20):
    with ui.HStack():
        plot_1 = ui.Plot(ui.Type.LINE, -1.0, 1.0, *data, width=360, height=100,
                style={"Plot":{
                    "color": cl.red,
                    "background_color": cl(0.08),
                    "secondary_color": cl("#aa1111"),
                    "selected_color": cl.green,
                    "background_selected_color": cl.white,
                    "border_width":5,
                    "border_color": cl.blue,
                    "border_radius": 20
                    }})
        ui.Spacer(width = 20)
        plot_2 = ui.Plot(ui.Type.HISTOGRAM, -1.0, 1.0, on_data_provider, 360, width=360, height=100,
                style={"Plot":{
                    "color": cl.blue,
                    "background_color": cl("#551111"),
                    "secondary_color": cl("#11AA11"),
                    "selected_color": cl(0.67),
                    "margin_height": 10,
                    }})
        plot_2.value_stride = 6
```