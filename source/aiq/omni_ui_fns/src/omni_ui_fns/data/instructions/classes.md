## omni.ui.AbstractValueModel

There are several predefined models:
- SimpleStringModel
- SimpleBoolModel
- SimpleFloatModel
- SimpleIntModel

To get and set the value of the model it's possible to use `set_value` method:

```
print(model.as_string)
model.as_float = 0.0
model.as_int = int(model.as_bool)
model.set_value(0.0)
```

There are also 4 properties: as_string, as_float, as_int, as_bool, you can use to get or set the value with different types

Also note that widget using the model must always use the model to get or set the value, as data is always being managed by the model, such as:
```
field = ui.IntField()
field.model.set_value(10)
```
instead of getting or setting the value through the widget:
```
field = ui.IntField(default_value=10)
```

It's possible to create the callback when the model is changed:
```
first_checkbox = ui.CheckBox()
second_checkbox = ui.CheckBox()

def on_first_changed(model):
    second_checkbox.model.as_bool = model.as_bool

first_checkbox.model.add_value_changed_fn(on_first_changed)
```

This is a custom implementation of the model.

```
class FloatModel(ui.AbstractValueModel):
    '''An example of custom float model that can be used for formatted string output'''

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def get_value_as_float(self):
        '''Reimplemented get float'''
        return self._value or 0.0

    def get_value_as_string(self):
        '''Reimplemented get string'''
        # This string goes to the field.
        if self._value is None:
            return ""

        # General format. This prints the number as a fixed-point
        # number, unless the number is too large, in which case it
        # switches to 'e' exponent notation.
        return "{0:g}".format(self._value)

    def set_value(self, value):
        '''Reimplemented set'''
        try:
            value = float(value)
        except ValueError:
            value = None
        if value != self._value:
            # Tell the widget that the model is changed
            self._value = value
            # This updates the widget. Without this line the widget will
            # not be updated.
            self._value_changed()

model = FloatModel()
my_window = ui.Window("Example Window", width=300, height=300)
with my_window.frame:
    with ui.VStack():
        ui.FloatSlider(model, min=-100, max=100)
```

## omni.ui.AbstractItemModel
The item model doesn't hold the data itself. It's using APIs to manage the data.
The `get_item_children` of `AbstractItemModel` returns a list of items
(inheriting from `omni.ui.AbstractItem`), which each item is being managed by
the value model (inheriting from `omni.ui.AbstractValueModel`) that can contain
any data type and supports callbacks. Thus, the model client can track the changes
in both the item model and any value it holds.

The item model can get both the value model and the nested items from any item.
Therefore, the model is flexible to represent anything from color to complicated
tree-table construction.

```
# child items owned by the model
items = model.get_item_children()

# child items of the first item in the model
items = model.get_item_children(items[0])

# get the value model from the first column of the second item in the model
model.get_item_value_model(items[1], 0)
```

To clear all items in the default model from the widget class, you could use `remove_item`
combined with `get_item_children`.
```
cb = ui.ComboBox(0, "Table", "Chair", "Sofa")
for item in cb.model.get_item_children():
    cb.model.remove_item(item)
```

### Item

Item is the object that is associated with the data entity of the model. It must
inherit from `ui.AbstractItem`.

Each item should be created and stored by the model implementation. And can
contain any data in it. Another option would be to use it as a raw pointer to
the data. In any case, it's the choice of the model how to manage this class.

### Hierarchial Model

Usually, the model is a hierarchical system where the item can have any number
of child items. The model is only populated at the moment the user expands the
item to save resources. The following example demonstrates that the model can be
infinitely long.

```execute 200
class Item(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text, value):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)
        self.value_model = ui.SimpleIntModel(value)
        self.children = None

class Model(ui.AbstractItemModel):
    def __init__(self, *args):
        super().__init__()
        self._children = [Item(t) for t in args]

    def get_item_children(self, item):
        """
        Returns all the children when the widget asks it.
        Ensure you handle the case where item is None, which should return the child items of the model.
        """
        if item is not None:
            if not item.children:
                item.children = [Item(f"Child #{i}") for i in range(5)]
            return item.children

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 2

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value of the column.
        In our case we use ui.SimpleStringModel.
        """
        if column_id == 0:
            return item.name_model
        else:
            return item.value_model

with ui.ScrollingFrame(
    height=200,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    style_type_name_override="TreeView",
):
    self._model = Model("Root", "Items")
    ui.TreeView(self._model, root_visible=False, style={"margin": 0.5})
```

### Nested Model

Since the model doesn't keep any data and serves as an API protocol, sometimes
it's very helpful to merge multiple models into one single model. The parent
model should redirect the calls to the children.

In the following example, three different models are merged into one.

```execute 200
class Item(ui.AbstractItem):
    def __init__(self, text, name, d=5):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)
        self.children = [Item(f"Child {name}{i}", name, d - 1) for i in range(d)]

class Model(ui.AbstractItemModel):
    def __init__(self, name):
        super().__init__()
        self._children = [Item(f"Model {name}", name)]

    def get_item_children(self, item):
        return item.children if item else self._children

    def get_item_value_model_count(self, item):
        return 1

    def get_item_value_model(self, item, column_id):
        return item.name_model

class NestedItem(ui.AbstractItem):
    def __init__(self, source_item, source_model):
        super().__init__()
        self.source = source_item
        self.model = source_model
        self.children = None

class NestedModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        models = [Model("A"), Model("B"), Model("C")]
        self.children = [
            NestedItem(i, m) for m in models for i in m.get_item_children(None)]

    def get_item_children(self, item):
        if item is None:
            return self.children

        if item.children is None:
            m = item.model
            item.children = [
                NestedItem(i, m) for i in m.get_item_children(item.source)]

        return item.children

    def get_item_value_model_count(self, item):
        return 1

    def get_item_value_model(self, item, column_id):
        return item.model.get_item_value_model(item.source, column_id)

with ui.ScrollingFrame(
    height=200,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    style_type_name_override="TreeView",
):
    self._model = NestedModel()
    ui.TreeView(self._model, root_visible=False, style={"margin": 0.5})
```

## omni.ui.AbstractItemDelegate

There are some complex widgets (e.g. omni.ui.TreeView, omni.ui.Combobox) for which we can define our customized delegates.

For example, we can create a Delegate for a TreeView widget that displays the items in branches.

```
class Delegate(ui.AbstractItemDelegate):
    """
    Delegate is the representation layer. TreeView calls the methods
    of the delegate to create custom widgets for each item.
    """

    def build_branch(self, model, item, column_id, level, expanded):
        """Create a branch widget that opens or closes subtree"""
        # Offset depents on level
        text = "     " * (level + 1)
        # > and v symbols depending on the expanded state
        if expanded:
            text += "v    "
        else:
            text += ">    "
        ui.Label(text, height=22, alignment=ui.Alignment.CENTER, tooltip="Branch")

    def build_widget(self, model, item, column_id, level, expanded):
        """Create a widget per column per item"""
        ui.Label(
            model.get_item_value_model(item, column_id).as_string,
            tooltip="Widget"
        )

    def build_header(self, column_id):
        """Build the header"""
        ui.Label("Header", tooltip="Header", height=25)

self._delegate = Delegate()
ui.TreeView(delegate=self._delegate, root_visible=False, header_visible=True)
```

## omni.ui.Rectangle
Rectangle is a shape with four sides and four corners. You can use Rectangle to draw rectangle shapes, or mix it with other controls e.g. using ZStack to create an advanced look.

Here is a list of styles you can customize on Rectangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border
> background_gradient_color (color): the gradient color on the top part of the rectangle
> border_radius (float): default rectangle has 4 right corner angles, border_radius defines the radius of the corner angle if the user wants to round the rectangle corner. We only support one border_radius across all the corners, but users can choose which corner to be rounded.
> corner_flag (enum): defines which corner or corners to be rounded

Here is a list of the supported corner flags:
```
from omni.ui import color as cl
corner_flags = {
    "ui.CornerFlag.NONE": ui.CornerFlag.NONE,
    "ui.CornerFlag.TOP_LEFT": ui.CornerFlag.TOP_LEFT,
    "ui.CornerFlag.TOP_RIGHT": ui.CornerFlag.TOP_RIGHT,
    "ui.CornerFlag.BOTTOM_LEFT": ui.CornerFlag.BOTTOM_LEFT,
    "ui.CornerFlag.BOTTOM_RIGHT": ui.CornerFlag.BOTTOM_RIGHT,
    "ui.CornerFlag.TOP": ui.CornerFlag.TOP,
    "ui.CornerFlag.BOTTOM": ui.CornerFlag.BOTTOM,
    "ui.CornerFlag.LEFT": ui.CornerFlag.LEFT,
    "ui.CornerFlag.RIGHT": ui.CornerFlag.RIGHT,
    "ui.CornerFlag.ALL": ui.CornerFlag.ALL,
}

with ui.ScrollingFrame(
    height=100,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in corner_flags.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}):
                    ui.Rectangle(
                        style={"background_color": cl("#aa4444"), "border_radius": 20.0, "corner_flag": value}
                    )
                    ui.Spacer(height=10)
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
```
Here are a few examples of Rectangle using different selections of styles:

Default rectangle which is scaled to fit:
```
with ui.Frame(height=20):
    ui.Rectangle(name="default")
```

This rectangle uses its own style to control colors and shape. Notice how three colors "background_color", "border_color" and "border_color" are affecting the look of the rectangle:
```
from omni.ui import color as cl
with ui.Frame(height=40):
    ui.Rectangle(style={"Rectangle":{
        "background_color":cl("#aa4444"),
        "border_color":cl("#22FF22"),
        "background_gradient_color": cl("#4444aa"),
        "border_width": 2.0,
        "border_radius": 5.0}})
```

This rectangle uses fixed width and height. Notice the `border_color` is not doing anything if `border_width` is not defined.
```
from omni.ui import color as cl
with ui.Frame(height=20):
    ui.Rectangle(width=40, height=10, style={"background_color":cl(0.6), "border_color":cl("#ff2222")})
```

Compose with ZStack for an advanced look
```
from omni.ui import color as cl
with ui.Frame(height=20):
    with ui.ZStack(height=20):
        ui.Rectangle(width=150,
            style={"background_color":cl(0.6),
                    "border_color":cl(0.1),
                    "border_width": 1.0,
                    "border_radius": 8.0} )
        with ui.HStack():
            ui.Spacer(width=10)
            ui.Image("resources/icons/Cloud.png", width=20, height=20 )
            ui.Label( "Search Field", style={"color":cl(0.875)})
```

## omni.ui.FreeRectangle
FreeRectangle is a rectangle whose width and height will be determined by other widgets.

Here is a list of styles you can customize on FreeRectangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border
> background_gradient_color (color): the gradient color on the top part of the rectangle
> border_radius (float): default rectangle has 4 right corner angles, border_radius defines the radius of the corner angle if the user wants to round the rectangle corner. We only support one border_radius across all the corners, but users can choose which corner to be rounded.
> corner_flag (enum): defines which corner or corners to be rounded

Here is an example of a FreeRectangle with style following two draggable circles:
```
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Circle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=150):
            control2 = ui.Circle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeRectangle(control1, control2, style={
                    "background_color":cl(0.6),
                    "border_color":cl(0.1),
                    "border_width": 1.0,
                    "border_radius": 8.0})
```

## omni.ui.Circle
You can use Circle to draw a circular shape.

Here is a list of styles you can customize on Circle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is some of the properties you can customize on Circle:
> size_policy (enum): there are two types of the size_policy, fixed and stretch.
    * ui.CircleSizePolicy.FIXED: the size of the circle is defined by the radius and is fixed without being affected by the parent scaling.
    * ui.CircleSizePolicy.STRETCH: the size of the circle is defined by the parent and will be stretched if the parent widget size changed.
> alignment (enum): the position of the circle in the parent defined space
> arc (enum): this property defines the way to draw a half or a quarter of the circle.

Here is a list of the supported Alignment and Arc value for the Circle:

```
from omni.ui import color as cl
alignments = {
    "ui.Alignment.CENTER": ui.Alignment.CENTER,
    "ui.Alignment.LEFT_TOP": ui.Alignment.LEFT_TOP,
    "ui.Alignment.LEFT_CENTER": ui.Alignment.LEFT_CENTER,
    "ui.Alignment.LEFT_BOTTOM": ui.Alignment.LEFT_BOTTOM,
    "ui.Alignment.CENTER_TOP": ui.Alignment.CENTER_TOP,
    "ui.Alignment.CENTER_BOTTOM": ui.Alignment.CENTER_BOTTOM,
    "ui.Alignment.RIGHT_TOP": ui.Alignment.RIGHT_TOP,
    "ui.Alignment.RIGHT_CENTER": ui.Alignment.RIGHT_CENTER,
    "ui.Alignment.RIGHT_BOTTOM": ui.Alignment.RIGHT_BOTTOM,
}
ui.Label("Alignment: ")
with ui.ScrollingFrame(
    height=150,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    with ui.ZStack():
                        ui.Rectangle(name="table", style={"border_color":cl.white, "border_width": 1.0})
                        ui.Circle(
                            radius=10,
                            size_policy=ui.CircleSizePolicy.FIXED,
                            name="orientation",
                            alignment=value,
                            style={"background_color": cl("#aa4444")},
                        )
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
ui.Spacer(height=10)
ui.Label("Arc: ")
with ui.ScrollingFrame(
    height=150,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    with ui.ZStack():
                        ui.Rectangle(name="table", style={"border_color":cl.white, "border_width": 1.0})
                        ui.Circle(
                            radius=10,
                            size_policy=ui.CircleSizePolicy.FIXED,
                            name="orientation",
                            arc=value,
                            style={
                                "background_color": cl("#aa4444"),
                                "border_color": cl.blue,
                                "border_width": 2,
                                },
                        )
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)
```

Default circle which is scaled to fit, the alignment is centered:
```
with ui.Frame(height=20):
    ui.Circle(name="default")
```

This circle is scaled to fit with 100 height:
```
with ui.Frame(height=100):
    ui.Circle(name="default")
```

This circle has a fixed radius of 20, the alignment is LEFT_CENTER:
```
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#1111ff"), "border_color": cl("#cc0000"), "border_width": 4}}
with ui.Frame(height=100, style=style):
    with ui.HStack():
        ui.Rectangle(width=40, style={"background_color": cl.white})
        ui.Circle(radius=20, size_policy=ui.CircleSizePolicy.FIXED, alignment=ui.Alignment.LEFT_CENTER)
```

This circle has a fixed radius of 10, the alignment is RIGHT_CENTER
```
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#ff1111"), "border_color": cl.blue, "border_width": 2}}
with ui.Frame(height=100, width=200, style=style):
    with ui.ZStack():
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Circle(radius=10, size_policy=ui.CircleSizePolicy.FIXED, alignment=ui.Alignment.RIGHT_CENTER)
```

This circle has a fixed radius of 10, it has all the same style as the previous one, except its size_policy is `ui.CircleSizePolicy.STRETCH`
```
from omni.ui import color as cl
style = {"Circle": {"background_color": cl("#ff1111"), "border_color": cl.blue, "border_width": 2}}
with ui.Frame(height=100, width=200, style=style):
    with ui.ZStack():
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Circle(radius=10, size_policy=ui.CircleSizePolicy.STRETCH, alignment=ui.Alignment.RIGHT_CENTER)
```

## omni.ui.FreeCircle
FreeCircle is a circle whose radius will be determined by other widgets.

Here is a list of styles you can customize on FreeCircle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is an example of a FreeCircle with style following two draggable rectangles:
```
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Rectangle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=150):
            control2 = ui.Rectangle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeCircle(control1, control2, style={
                    "background_color":cl.transparent,
                    "border_color":cl.red,
                    "border_width": 2.0})
```


## omni.ui.Ellipse
Ellipse is drawn in a rectangle bounding box, and It is always scaled to fit the rectangle's width and height.

Here is a list of styles you can customize on Ellipse:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Default ellipse is scaled to fit:
```
with ui.Frame(height=20, width=150):
    ui.Ellipse(name="default")
```

Stylish ellipse with border and colors:
```
from omni.ui import color as cl
style = {"Ellipse": {"background_color": cl("#1111ff"), "border_color": cl("#cc0000"), "border_width": 4}}
with ui.Frame(height=100, width=50):
    ui.Ellipse(style=style)
```

## omni.ui.FreeEllipse
FreeEllipse is an ellipse whose width and height will be determined by other widgets.

Here is a list of styles you can customize on FreeEllipse:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is an example of a FreeEllipse with style following two draggable circles:

```
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Circle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=200):
            control2 = ui.Circle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeEllipse(control1, control2, style={
                    "background_color":cl.purple})
```

## omni.ui.Triangle
You can use Triangle to draw Triangle shape.

Here is a list of styles you can customize on Triangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is some of the properties you can customize on Triangle:
> alignment (enum): the alignment defines where the tip of the triangle is, base will be at the opposite side

Here is a list of the supported alignment value for the triangle:

```
from omni.ui import color as cl
alignments = {
    "ui.Alignment.LEFT_TOP": ui.Alignment.LEFT_TOP,
    "ui.Alignment.LEFT_CENTER": ui.Alignment.LEFT_CENTER,
    "ui.Alignment.LEFT_BOTTOM": ui.Alignment.LEFT_BOTTOM,
    "ui.Alignment.CENTER_TOP": ui.Alignment.CENTER_TOP,
    "ui.Alignment.CENTER_BOTTOM": ui.Alignment.CENTER_BOTTOM,
    "ui.Alignment.RIGHT_TOP": ui.Alignment.RIGHT_TOP,
    "ui.Alignment.RIGHT_CENTER": ui.Alignment.RIGHT_CENTER,
    "ui.Alignment.RIGHT_BOTTOM": ui.Alignment.RIGHT_BOTTOM,
}
colors = [cl.red, cl.yellow, cl.purple, cl("#ff0ff0"), cl.green, cl("#f00fff"), cl("#fff000"), cl("#aa3333")]
index = 0
with ui.ScrollingFrame(
    height=160,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style={"ScrollingFrame": {"background_color": cl.transparent}},
):
    with ui.HStack():
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}):
                    color = colors[index]
                    index = index + 1
                    ui.Triangle(alignment=value, style={"Triangle":{"background_color": color}})
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER, height=20)
```

Here are a few examples of Triangle using different selections of styles:

The triangle is scaled to fit, base on the left and tip on the center right. Users can define the border_color and border_width but without background_color to make the triangle look like it's drawn in wireframe style.
```
from omni.ui import color as cl
style = {
    "Triangle::default":
    {
        "background_color": cl.green,
        "border_color": cl.white,
        "border_width": 1
    },
    "Triangle::transparent":
    {
        "border_color": cl.purple,
        "border_width": 4,
    },
}
with ui.Frame(height=100, width=200, style=style):
    with ui.HStack(spacing=10, style={"margin": 5}):
        ui.Triangle(name="default")
        ui.Triangle(name="transparent", alignment=ui.Alignment.CENTER_TOP)
```

## omni.ui.FreeTriangle
FreeTriangle is a triangle whose width and height will be determined by other widgets.

Here is a list of styles you can customize on Triangle:
> background_color (color): the background color of the shape
> border_width (float): the border width if the shape has a border
> border_color (color): the border color if the shape has a border

Here is an example of a FreeTriangle with style following two draggable rectangles. The default alignment is `ui.Alignment.RIGHT_CENTER`. We make the alignment as `ui.Alignment.CENTER_BOTTOM`.

```
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Rectangle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=200):
            control2 = ui.Rectangle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeTriangle(control1, control2, alignment=ui.Alignment.CENTER_BOTTOM, style={
                    "background_color":cl.blue,
                    "border_color":cl.red,
                    "border_width": 2.0})
```

## omni.ui.Line
Line is the simplest shape that represents a straight line. It has two points, color and thickness. You can use Line to draw line shapes.

Here is a list of common styles you can customize on Line:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here are some of the properties you can customize on Line:
> alignment (enum): the Alignment defines where the line is in parent defined space. It is always scaled to fit.

Here is a list of the supported Alignment value for the line:
```
from omni.ui import color as cl
style ={
    "Rectangle::table": {"background_color": cl.transparent, "border_color": cl(0.8), "border_width": 0.25},
    "Line::demo": {"color": cl("#007777"), "border_width": 3},
    "ScrollingFrame": {"background_color": cl.transparent},
}
alignments = {
    "ui.Alignment.LEFT": ui.Alignment.LEFT,
    "ui.Alignment.RIGHT": ui.Alignment.RIGHT,
    "ui.Alignment.H_CENTER": ui.Alignment.H_CENTER,
    "ui.Alignment.TOP": ui.Alignment.TOP,
    "ui.Alignment.BOTTOM": ui.Alignment.BOTTOM,
    "ui.Alignment.V_CENTER": ui.Alignment.V_CENTER,
}
with ui.ScrollingFrame(
    height=100,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    style=style,
):
    with ui.HStack(height=100):
        for key, value in alignments.items():
            with ui.ZStack():
                ui.Rectangle(name="table")
                with ui.VStack(style={"VStack": {"margin": 10}}, spacing=10):
                    ui.Line(name="demo", alignment=value)
                    ui.Label(key, style={"color": cl.white, "font_size": 12}, alignment=ui.Alignment.CENTER)

```

By default, the line is scaled to fit.
```
from omni.ui import color as cl
style = {"Line::default": {"color": cl.red, "border_width": 1}}
with ui.Frame(height=50, style=style):
    ui.Line(name="default")
```

Users can define the color and border_width to make customized lines.
```
from omni.ui import color as cl
with ui.Frame(height=50):
    with ui.ZStack(width=200):
        ui.Rectangle(style={"background_color": cl(0.4)})
        ui.Line(alignment=ui.Alignment.H_CENTER, style={"border_width":5, "color": cl("#880088")})
```

## omni.ui.FreeLine
FreeLine is a line whose length will be determined by other widgets.

Here is a list of common styles you can customize on Line:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here is an example of a FreeLine with style, driven by two draggable circles. Notice the control widgets are not the start and end points of the line. By default, the alignment of the line is `ui.Alighment.V_CENTER`, and the line direction won't be changed by the control widgets.

```
from omni.ui import color as cl
with ui.Frame(height=200):
    with ui.ZStack():
        # Four draggable rectangles that represent the control points
        with ui.Placer(draggable=True, offset_x=0, offset_y=0):
            control1 = ui.Circle(width=10, height=10)
        with ui.Placer(draggable=True, offset_x=150, offset_y=200):
            control2 = ui.Circle(width=10, height=10)

        # The rectangle that fits to the control points
        ui.FreeLine(control1, control2, style={"color":cl.yellow})
```

## omni.ui.Button
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

## omni.ui.RadioButton
RadioButton is the widget that allows the user to choose only one from a predefined set of mutually exclusive options.

RadioButtons are arranged in collections of two or more buttons within a RadioCollection, which is the central component of the system and controls the behavior of all the RadioButtons in the collection.

Here is a list of styles you can customize on RadioButton:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> background_color (color): the background color of the RadioButton
> padding (float): the distance between the the RadioButton content widget (e.g. Image) and the RadioButton border

To control the style of the button image, you can customize `RadioButton.Image`. For example RadioButton.Image's image_url defines the image when it's not checked. You can define the image for checked status with `RadioButton.Image:checked` style.

Here is an example of RadioCollection which contains 5 RadioButtons with style. Also there is an IntSlider which shares the model with the RadioCollection, so that when RadioButton value or the IntSlider value changes, the other one will update too.

```
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

## omni.ui.ToolButton
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

## omni.ui.ColorWidget
The ColorWidget is a button that displays the color from the item model and can open a picker window. The color dialog's function is to allow users to choose color.

Here is a list of styles you can customize on ColorWidget:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary

> background_color (color): the background color of the tooltip widget when hover over onto the ColorWidget
> color (color): the text color of the tooltip widget when hover over onto the ColorWidget

ColorWidget's model (omni.ui.SimpleListModel) can contain 4 or 3 items with each representing the corresponding RGBA or RGB color channel, to change the color to green, you can do:
```
import omni.ui as ui

cw = ui.ColorWidget()
for item, value in zip(cw.model.get_item_children(), (0, 1, 0, 0)):
    cw.model.get_item_value_model(item).set_value(value)
```

Here is an example of a ColorWidget with three FloatFields. The ColorWidget model is shared with the FloatFields so that users can click and edit the field value to change the ColorWidget's color, and the value change of the ColorWidget will also reflect in the value change of the FloatFields.

```
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

```
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

```
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0).model
    ui.ComboBox(color_model)
```

Here is an interactive example with USD. You can create a Mesh in the Stage. Choose `Pixar Storm` as the render. Select the mesh and use this ColorWidget to change the color of the mesh. You can use `Ctrl+z` for undoing and `Ctrl+y` for redoing.

```
import weakref
import omni.kit.commands
from omni.usd.commands import UsdStageHelper
from pxr import UsdGeom
from pxr import Gf
import omni.usd

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
        self._events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = self._events.create_subscription_to_pop(
            self._on_stage_event, name="omni.example.ui ColorWidget stage update"
        )

        # Privates
        self._subscription = None
        self._gprim = None
        self._prev_color = None
        self._edit_mode_counter = 0

    def _on_stage_event(self, event):
        """Called with subscription to pop"""
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self._on_selection_changed()

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

## omni.ui.Image
The Image type displays an image. The source of the image is specified as a URL using the source property. By default, specifying the width and height of the item makes the image to be scaled to fit that size. This behavior can be changed by setting the `fill_policy` property, allowing the image to be stretched or scaled instead. The property alignment controls how the scaled image is aligned in the parent defined space.

Here is a list of styles you can customize on Image:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
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
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source)
```

The image is stretched to fit and aligned to the left
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.STRETCH, alignment=ui.Alignment.LEFT_CENTER)
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the top
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_TOP)
```

The image is scaled uniformly to fit without cropping and aligned to the right. Notice the fill_policy and alignment are defined in style.
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, style={
        "Image": {
            "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
            "alignment": ui.Alignment.RIGHT_CENTER,
            "margin": 5}})
```

The image has rounded corners and an overlayed color. Note image_url is in the style dictionary.
```
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(style={"image_url": source, "border_radius": 10, "color": cl("#5eb3ff")})
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the bottom, with a blue border.
```
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
```
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
```

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

## omni.ui.ImageWithProvider
ImageWithProvider also displays an image just like Image. It is a much more advanced image widget. ImageWithProvider blocks until the image is loaded, Image doesn't block. Sometimes Image blinks because when the first frame is created, the image is not loaded. Users are recommended to use ImageWithProvider if the UI is updated pretty often. Because it doesn't blink when recreating.

Here is a list of styles you can customize on ImageWithProvider:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> image_url (str): the url path of the image source
> color (color): the overlay color of the image
> corner_flag (enum): defines which corner or corners to be rounded. The supported corner flags are the same as Rectangle since Image is eventually an image on top of a rectangle under the hood.
> fill_policy (enum): defines how the Image fills the rectangle.
There are three types of fill_policy
* ui.IwpFillPolicy.IWP_STRETCH: stretch the image to fill the entire rectangle.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary
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

The image source comes from `ImageProvider` which could be `ByteImageProvider`, `RasterImageProvider` or `VectorImageProvider`.

`RasterImageProvider` and `VectorImageProvider` are using image urls like Image. Here is an example taken from Image. Notice the fill_policy value difference.
```
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
```
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

## omni.ui.Plot
The Plot class displays a line or histogram image. The data of the image is specified as a data array or a provider function.

Here is a list of styles you can customize on Plot:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> color (color): the color of the plot, line color in the line typed plot or rectangle bar color in the histogram typed plot
> selected_color (color): the selected color of the plot, dot in the line typed plot and rectangle bar in the histogram typed plot
> background_color (color): the background color of the plot
> secondary_color (color): the color of the text and the border of the text box which shows the plot selection value
> background_selected_color (color): the background color of the text box which shows the plot selection value

Here are couple of examples of Plots:
```
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
                    "secondary_color": cl("#aa1111"), #the color of the text and the border of the text box which shows the plot selection value
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
                    "secondary_color": cl("#11AA11"), the color of the text and the border of the text box which shows the plot selection value
                    "selected_color": cl(0.67),
                    "margin_height": 10,
                    }})
        plot_2.value_stride = 6
```

## omni.ui.Label
Labels are used everywhere in omni.ui. They are text only objects.

Here is a list of styles you can customize on Label:
> color (color): the color of the text
> font_size (float): the size of the text
> margin (float): the distance between the label and the parent widget defined boundary
> margin_width (float): the width distance between the label and the parent widget defined boundary
> margin_height (float): the height distance between the label and the parent widget defined boundary
> alignment (enum): defines how the label is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Here are a few examples of labels:

```
from omni.ui import color as cl
ui.Label("this is a simple label", style={"color":cl.red, "margin": 5})
```

```
from omni.ui import color as cl
ui.Label("label with alignment", style={"color":cl.green, "margin": 5}, alignment=ui.Alignment.CENTER)
```

Notice that alignment could be either a property or a style.
```
from omni.ui import color as cl
label_style = {
    "Label": {"font_size": 20, "color": cl.blue, "alignment":ui.Alignment.RIGHT, "margin_height": 20}
    }
ui.Label("Label with style", style=label_style)
```

When the text of the Label is too long, it can be elided by `...`:
```
from omni.ui import color as cl
ui.Label(
            "Label can be elided: Lorem ipsum dolor "
            "sit amet, consectetur adipiscing elit, sed do "
            "eiusmod tempor incididunt ut labore et dolore "
            "magna aliqua. Ut enim ad minim veniam, quis "
            "nostrud exercitation ullamco laboris nisi ut "
            "aliquip ex ea commodo consequat. Duis aute irure "
            "dolor in reprehenderit in voluptate velit esse "
            "cillum dolore eu fugiat nulla pariatur. Excepteur "
            "sint occaecat cupidatat non proident, sunt in "
            "culpa qui officia deserunt mollit anim id est "
            "laborum.",
            style={"color":cl.white},
            elided_text=True,
        )
```

## omni.ui.CheckBox
A CheckBox is an option button that can be switched on (checked) or off (unchecked). Checkboxes are typically used to represent features in an application that can be enabled or disabled without affecting others.

The checkbox is implemented using the model-delegate-view pattern. The model is the central component of this system. It is the application's dynamic data structure independent of the widget. It directly manages the data, logic and rules of the checkbox. If the model is not specified, the simple one is created automatically when the object is constructed.

Here is a list of styles you can customize on CheckBox:
> color (color): the color of the tick
> background_color (color): the background color of the check box
> font_size: the size of the tick
> border_radius (float): the radius of the corner angle if the user wants  to round the check box.
> border_width (float): the size of the check box border
> secondary_background_color (color): the color of the check box border

Default checkbox
```
with ui.HStack(width=0, spacing=5):
    ui.CheckBox().model.set_value(True)
    ui.CheckBox()
    ui.Label("Default")
```

Disabled checkbox:
```
with ui.HStack(width=0, spacing=5):
    ui.CheckBox(enabled=False).model.set_value(True)
    ui.CheckBox(enabled=False)
    ui.Label("Disabled")
```

In the following example, the models of two checkboxes are connected, and if one checkbox is changed, it makes another checkbox change as well.

```
from omni.ui import color as cl
with ui.HStack(width=0, spacing=5):
    # Create two checkboxes
    style = {"CheckBox":{
        "color": cl.white, "border_radius": 0, "background_color": cl("#ff5555"), "font_size": 30}}
    first = ui.CheckBox(style=style)
    second = ui.CheckBox(style=style)

    # Connect one to another
    first.model.add_value_changed_fn(lambda a, b=second: b.model.set_value(not a.get_value_as_bool()))
    second.model.add_value_changed_fn(lambda a, b=first: b.model.set_value(not a.get_value_as_bool()))

    # Set the first one to True
    first.model.set_value(True)

    ui.Label("One of two")
```

In the following example, that is a bit more complicated, only one checkbox can be enabled.
```
from omni.ui import color as cl
style = {"CheckBox":{
    "color": cl("#ff5555"), "border_radius": 5, "background_color": cl(0.35), "font_size": 20}}
with ui.HStack(width=0, spacing=5):
    # Create two checkboxes
    first = ui.CheckBox(style=style)
    second = ui.CheckBox(style=style)
    third = ui.CheckBox(style=style)

    def like_radio(model, first, second):
        """Turn on the model and turn off two checkboxes"""
        if model.get_value_as_bool():
            model.set_value(True)
            first.model.set_value(False)
            second.model.set_value(False)

    # Connect one to another
    first.model.add_value_changed_fn(lambda a, b=second, c=third: like_radio(a, b, c))
    second.model.add_value_changed_fn(lambda a, b=first, c=third: like_radio(a, b, c))
    third.model.add_value_changed_fn(lambda a, b=first, c=second: like_radio(a, b, c))

    # Set the first one to True
    first.model.set_value(True)

    ui.Label("Almost like radio box")
```

## omni.ui.ComboBox
The ComboBox widget is a combination of a button and a drop-down list. A ComboBox is a selection widget that displays the current item and can pop up a list of selectable items.

Here is a list of styles you can customize on ComboBox:
> color (color): the color of the combo box text and the arrow of the drop-down button
> background_color (color): the background color of the combo box
> secondary_color (color): the color of the drop-down button's background
> selected_color (color): the selected highlight color of option items
> secondary_selected_color (color): the color of the option item text
> font_size (float): the size of the text
> border_radius (float): the border radius if the user wants  to round the ComboBox
> padding (float): the overall padding of the ComboBox. If padding is defined, padding_height and padding_width will have no effects.
> padding_height (float): the width padding of the drop-down list
> padding_width (float): the height padding of the drop-down list
> secondary_padding (float): the height padding between the ComboBox and options

Default ComboBox with a default model, note that each item in the model saves the index number instead of the
value string:

```
# create a comboBox with three options 1, 2 and 3, and defaults to index 1 which is "Option 2"
combo_box = ui.ComboBox(1, "Option 1", "Option 2", "Option 3")
# set the combo box to index 2 which is "Option 3"
combo_box.model.get_item_value_model().set_value(2)
```

ComboBox with style
```
from omni.ui import color as cl
style={"ComboBox":{
    "color": cl.red,
    "background_color": cl(0.15),
    "secondary_color": cl("#1111aa"), # the drop-down button's background color is blue
    "selected_color": cl.green,
    "secondary_selected_color": cl.white,
    "font_size": 15,
    "border_radius": 20,
    "padding_height": 2,
    "padding_width": 20,
    "secondary_padding": 30,
}}
with ui.VStack():
    ui.ComboBox(1, "Option 1", "Option 2", "Option 3", style=style)
    ui.Spacer(height=20)
```


The following example demonstrates how to add items to the ComboBox.
```
editable_combo = ui.ComboBox()
ui.Button(
    "Add item to combo",
    clicked_fn=lambda m=editable_combo.model: m.append_child_item(
        None, ui.SimpleStringModel("Hello World")),
)
```

The minimal model implementation to have more flexibility of the data. It requires holding the value models and reimplementing two methods: `get_item_children` and `get_item_value_model`.
```
class MinimalItem(ui.AbstractItem):
    def __init__(self, text):
        super().__init__()
        self.model = ui.SimpleStringModel(text)

class MinimalModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()

        self._current_index = ui.SimpleIntModel()
        self._current_index.add_value_changed_fn(
            lambda a: self._item_changed(None))

        self._items = [
            MinimalItem(text)
            for text in ["Option 1", "Option 2"]
        ]

    def get_item_children(self, item):
        return self._items

    def get_item_value_model(self, item, column_id):
        if item is None:
            return self._current_index
        return item.model

self._minimal_model = MinimalModel()
with ui.VStack():
    ui.ComboBox(self._minimal_model, style={"font_size": 22})
    ui.Spacer(height=10)
```

The example of communication between widgets. Type anything in the field and it will appear in the combo box.
```
editable_combo = None

class StringModel(ui.SimpleStringModel):
    '''
    String Model activated when editing is finished.
    Adds item to combo box.
    '''
    def __init__(self):
        super().__init__("")

    def end_edit(self):
        combo_model = editable_combo.model
        # Get all the options ad list of strings
        all_options = [
            combo_model.get_item_value_model(child).as_string
            for child in combo_model.get_item_children()
        ]

        # Get the current string of this model
        fieldString = self.as_string
        if fieldString:
            if fieldString in all_options:
                index = all_options.index(fieldString)
            else:
                # It's a new string in the combo box
                combo_model.append_child_item(
                    None,
                    ui.SimpleStringModel(fieldString)
                )
                index = len(all_options)

            combo_model.get_item_value_model().set_value(index)

self._field_model = StringModel()

def combo_changed(combo_model, item):
    all_options = [
        combo_model.get_item_value_model(child).as_string
        for child in combo_model.get_item_children()
    ]
    current_index = combo_model.get_item_value_model().as_int
    self._field_model.as_string = all_options[current_index]

with ui.HStack():
    ui.StringField(self._field_model)
    editable_combo = ui.ComboBox(width=0, arrow_only=True)
    editable_combo.model.add_item_changed_fn(combo_changed)
```


## omni.ui.TreeView
TreeView is a widget that presents a hierarchical view of information. Each item can have a number of subitems. An indentation often visualizes this in a list. An item can be expanded to reveal subitems, if any exist, and collapsed to hide subitems.

TreeView can be used in file manager applications, where it allows the user to navigate the file system directories. They are also used to present hierarchical data, such as the scene object hierarchy.

TreeView uses a model-delegate-view pattern to manage the relationship between data and the way it is presented. The separation of functionality gives developers greater flexibility to customize the presentation of items and provides a standard interface to allow a wide range of data sources to be used with other widgets. Note that the model and delegate instances MUST be stored in a variable, and the variable is then passed into the TreeView constructor. This ensures we are keeping an actual reference to the model.

Here is a list of styles you can customize on TreeView:
> background_color (color): specifically used when Treeview item is selected. It indicates the background color of the TreeView item when selected.
> background_selected_color (color): the hover color of the TreeView selected item. The actual selected color of the TreeView selected item should be defined by the "background_color" of ":selected".
> secondary_color (color): if the TreeView has more than one column, this is the color of the line which divides the columns.
> secondary_selected_color (color): if the TreeView has more than one column and if the column is resizable, this is the color of the line which divides the columns when hovered over the divider.
> border_color (color): the border color of the TreeView item when hovered. During drag and drop of the Treeview item, it is also the border color of the Treeview item border which indicates where the dragged item targets to drop.
> border_width (float): specifically used when Treeview item drag and dropped. Thickness of the Treeview item border which indicates where the dragged item targets to drop.

Here is a list of styles you can customize on TreeView.Item:
> margin (float): the margin between TreeView items. This will be overridden by the value of margin_width or margin_height
> margin_width (float): the margin width between TreeView items
> margin_height (float): the margin height between TreeView items
> color (color): the text color of the TreeView items
> font_size (float): the text size of the TreeView items

The following example demonstrates how to make a single level tree appear like a simple list.
```
import omni.ui as ui
from omni.ui import color as cl
style = {
    "TreeView":
    {
        "background_selected_color": cl("#55FF9033"),
        "secondary_color": cl.green,  # green column resizer color
        "secondary_selected_color": cl.purple,
        "border_color": cl.red,
    },
    "TreeView:selected": {"background_color": cl("#888888")},
    "TreeView.Item":
    {
        "margin": 4,
        "margin_width": 10,
        "color": cl("#AAAAAA"),
        "font_size": 13,
    },
    "TreeView.Item:selected": {"color": cl.pink},
}
class CommandItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

class CommandModel(ui.AbstractItemModel):
    """
    Represents the list of commands registered in Kit.
    It is used to make a single level tree appear like a simple list.
    """

    def __init__(self):
        super().__init__()

        self._commands = []
        try:
            import omni.kit.commands
        except ModuleNotFoundError:
            return

        omni.kit.commands.subscribe_on_change(self._commands_changed)
        self._commands_changed()

    def _commands_changed(self):
        """Called by subscribe_on_change"""
        self._commands = []
        import omni.kit.commands

        for cmd_list in omni.kit.commands.get_commands().values():
            for k in cmd_list.values():
                self._commands.append(CommandItem(k.__name__))

        self._item_changed(None)

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._commands

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 2

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel.
        """
        if item and isinstance(item, CommandItem):
            return item.name_model

with ui.ScrollingFrame(
    height=400,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    self._command_model = CommandModel()
    tree_view = ui.TreeView(
        self._command_model,
        root_visible=False,
        header_visible=False,
        columns_resizable=True,
        column_widths=[350, 350],
        style_type_name_override="TreeView",
        style=style,
    )
```

The following example demonstrates reordering with drag and drop. You can drag one item of the TreeView and move it to the position where you want to insert the item.
```
from omni.ui import color as cl

style = {
    "TreeView":
    {
        "border_color": cl.red,
        "border_width": 2,
    },
    "TreeView.Item": {"margin": 4},
}

class ListItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

    def __repr__(self):
        return f'"{self.name_model.as_string}"'

class ListModel(ui.AbstractItemModel):
    """
    Represents the model for lists. It's very easy to initialize it
    with any string list:
        string_list = ["Hello", "World"]
        model = ListModel(*string_list)
        ui.TreeView(model)
    """

    def __init__(self, *args):
        super().__init__()
        self._children = [ListItem(t) for t in args]

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 1

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel.
        """
        return item.name_model

class ListModelWithReordering(ListModel):
    """
    Represents the model for the list with the ability to reorder the
    list with drag and drop.
    """

    def __init__(self, *args):
        super().__init__(*args)

    def get_drag_mime_data(self, item):
        """Returns Multipurpose Internet Mail Extensions (MIME) data for be able to drop this item somewhere"""
        # As we don't do Drag and Drop to the operating system, we return the string.
        return item.name_model.as_string

    def drop_accepted(self, target_item, source, drop_location=-1):
        """Reimplemented from AbstractItemModel. Called to highlight target when drag and drop."""
        # If target_item is None, it's the drop to root. Since it's
        # list model, we support reorganization of root only and we
        # don't want to create a tree.
        return not target_item and drop_location >= 0

    def drop(self, target_item, source, drop_location=-1):
        """Reimplemented from AbstractItemModel. Called when dropping something to the item."""
        try:
            source_id = self._children.index(source)
        except ValueError:
            # Not in the list. This is the source from another model.
            return

        if source_id == drop_location:
            # Nothing to do
            return

        self._children.remove(source)

        if drop_location > len(self._children):
            # Drop it to the end
            self._children.append(source)
        else:
            if source_id < drop_location:
                # Because when we removed source, the array became shorter
                drop_location = drop_location - 1

            self._children.insert(drop_location, source)

        self._item_changed(None)

with ui.ScrollingFrame(
    height=150,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    self._list_model = ListModelWithReordering("Simplest", "List", "Model", "With", "Reordering")
    tree_view = ui.TreeView(
        self._list_model,
        root_visible=False,
        header_visible=False,
        style_type_name_override="TreeView",
        style=style,
        drop_between_items=True,
    )

```

The following example demonstrates the ability to edit TreeView items.
```
from omni.ui import color as cl
class FloatModel(ui.AbstractValueModel):
    """An example of custom float model that can be used for formatted string output"""

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def get_value_as_float(self):
        """Reimplemented get float"""
        return self._value or 0.0

    def get_value_as_string(self):
        """Reimplemented get string"""
        # This string goes to the field.
        if self._value is None:
            return ""

        # General format. This prints the number as a fixed-point
        # number, unless the number is too large, in which case it
        # switches to 'e' exponent notation.
        return "{0:g}".format(self._value)

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

class NameValueItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text, value):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)
        self.value_model = FloatModel(value)

    def __repr__(self):
        return f'"{self.name_model.as_string} {self.value_model.as_string}"'

class NameValueModel(ui.AbstractItemModel):
    """
    Represents the model for name-value tables. It's very easy to initialize it
    with any string-float list:
        my_list = ["Hello", 1.0, "World", 2.0]
        model = NameValueModel(*my_list)
        ui.TreeView(model)
    """

    def __init__(self, *args):
        super().__init__()
        # ["Hello", 1.0, "World", 2.0"] -> [("Hello", 1.0), ("World", 2.0)]
        regrouped = zip(*(iter(args),) * 2)
        self._children = [NameValueItem(*t) for t in regrouped]

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 2

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel for the first column
        and SimpleFloatModel for the second column.
        """
        return item.value_model if column_id == 1 else item.name_model

class EditableDelegate(ui.AbstractItemDelegate):
    """
    Delegate is the representation layer. TreeView calls the methods
    of the delegate to create custom widgets for each item.
    """

    def __init__(self):
        super().__init__()
        self.subscription = None

    def build_branch(self, model, item, column_id, level, expanded):
        """Create a branch widget that opens or closes subtree"""
        pass

    def build_widget(self, model, item, column_id, level, expanded):
        """Create a widget per column per item"""
        stack = ui.ZStack(height=20)
        with stack:
            value_model = model.get_item_value_model(item, column_id)
            label = ui.Label(value_model.as_string)
            if column_id == 1:
                field = ui.FloatField(value_model, visible=False)
            else:
                field = ui.StringField(value_model, visible=False)

        # Start editing when double clicked
        stack.set_mouse_double_clicked_fn(lambda x, y, b, m, f=field, l=label: self.on_double_click(b, f, l))

    def on_double_click(self, button, field, label):
        """Called when the user double-clicked the item in TreeView"""
        if button != 0:
            return

        # Make Field visible when double clicked
        field.visible = True
        field.focus_keyboard()
        # When editing is finished (enter pressed of mouse clicked outside of the viewport)
        self.subscription = field.model.subscribe_end_edit_fn(
            lambda m, f=field, l=label: self.on_end_edit(m, f, l)
        )

    def on_end_edit(self, model, field, label):
        """Called when the user is editing the item and pressed Enter or clicked outside of the item"""
        field.visible = False
        label.text = model.as_string
        self.subscription = None

with ui.ScrollingFrame(
    height=100,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
):
    self._name_value_model = NameValueModel("First", 0.2, "Second", 0.3, "Last", 0.4)
    self._name_value_delegate = EditableDelegate()
    tree_view = ui.TreeView(
        self._name_value_model,
        delegate=self._name_value_delegate,
        root_visible=False,
        header_visible=False,
        style_type_name_override="TreeView",
        style={"TreeView.Item": {"margin": 4}},
    )
```

This is an example of async filling the TreeView model. It's collecting only as many as it's possible of USD prims for 0.016s and waits for the next frame, so the UI is not locked even if the USD Stage is extremely big.
To play with it, create several materials in the stage or open a stage which contains materials, click "Traverse All" or "Stop Traversing".

```
import asyncio
import time
from omni.ui import color as cl
class ListItem(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)

    def __repr__(self):
        return f'"{self.name_model.as_string}"'

class ListModel(ui.AbstractItemModel):
    """
    Represents the model for lists. It's very easy to initialize it
    with any string list:
        string_list = ["Hello", "World"]
        model = ListModel(*string_list)
        ui.TreeView(model)
    """

    def __init__(self, *args):
        super().__init__()
        self._children = [ListItem(t) for t in args]

    def get_item_children(self, item):
        """Returns all the children when the widget asks it."""
        if item is not None:
            # Since we are doing a flat list, we return the children of root only.
            # If it's not root we return.
            return []

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 1

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value.
        In our case we use ui.SimpleStringModel.
        """
        return item.name_model

class AsyncQueryModel(ListModel):
    """
    This is an example of async filling the TreeView model. It's
    collecting only as many as it's possible of USD prims for 0.016s
    and waits for the next frame, so the UI is not locked even if the
    USD Stage is extremely big.
    """

    def __init__(self):
        super().__init__()
        self._stop_event = None

    def destroy(self):
        self.stop()

    def stop(self):
        """Stop traversing the stage"""
        if self._stop_event:
            self._stop_event.set()

    def reset(self):
        """Traverse the stage and keep materials"""
        self.stop()
        self._stop_event = asyncio.Event()

        self._children.clear()
        self._item_changed(None)

        asyncio.ensure_future(self.__get_all(self._stop_event))

    def __push_collected(self, collected):
        """Add given array to the model"""
        for c in collected:
            self._children.append(c)
        self._item_changed(None)

    async def __get_all(self, stop_event):
        """Traverse the stage portion at time, so it doesn't freeze"""
        stop_event.clear()

        start_time = time.time()
        # The widget will be updated not faster than 60 times a second
        update_every = 1.0 / 60.0

        import omni.usd
        from pxr import Usd
        from pxr import UsdShade

        context = omni.usd.get_context()
        stage = context.get_stage()
        if not stage:
            return

        # Buffer to keep the portion of the items before sending to the
        # widget
        collected = []

        for p in stage.Traverse(
            Usd.TraverseInstanceProxies(Usd.PrimIsActive and Usd.PrimIsDefined and Usd.PrimIsLoaded)
        ):
            if stop_event.is_set():
                break

            if p.IsA(UsdShade.Material):
                # Collect materials only
                collected.append(ListItem(str(p.GetPath())))

            elapsed_time = time.time()

            # Loop some amount of time so fps will be about 60FPS
            if elapsed_time - start_time > update_every:
                start_time = elapsed_time

                # Append the portion and update the widget
                if collected:
                    self.__push_collected(collected)
                    collected = []

                # Wait one frame to let other tasks go
                await omni.kit.app.get_app().next_update_async()

        self.__push_collected(collected)

try:
    import omni.usd
    from pxr import Usd

    usd_available = True
except ModuleNotFoundError:
    usd_available = False

if usd_available:
    with ui.ScrollingFrame(
        height=200,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    ):
        self._async_query_model = AsyncQueryModel()
        ui.TreeView(
            self._async_query_model,
            root_visible=False,
            header_visible=False,
            style_type_name_override="TreeView",
            style={"TreeView.Item": {"margin": 4}},
        )

    _loaded_label = ui.Label("Press Button to Load Materials", name="text")

    with ui.HStack():
        ui.Button("Traverse All", clicked_fn=self._async_query_model.reset)
        ui.Button("Stop Traversing", clicked_fn=self._async_query_model.stop)

    def _item_changed(model, item):
        if item is None:
            count = len(model._children)
            _loaded_label.text = f"{count} Materials Traversed"

    self._async_query_sub = self._async_query_model.subscribe_item_changed_fn(_item_changed)
```

## omni.ui.HStack
```
with ui.HStack(style={"margin": 10}):
    ui.Button("One")
    ui.Button("Two")
    ui.Button("Three")
    ui.Button("Four")
    ui.Button("Five")
```

## omni.ui.VStack
```
with ui.VStack(width=100.0, style={"margin": 5}):
    with ui.VStack():
        ui.Button("One")
        ui.Button("Two")
        ui.Button("Three")
        ui.Button("Four")
        ui.Button("Five")
```

## omni.ui.ZStack
```
with ui.VStack(width=100.0, style={"margin": 5}):
    with ui.ZStack():
        ui.Button("Very Long Text to See How Big it Can Be", height=0)
        ui.Button("Another\nMultiline\nButton", width=0)
```

## omni.ui.Frame
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

## omni.ui.CanvasFrame
CanvasFrame is the widget that allows the user to pan and zoom its children with a mouse. It has a layout that can be infinitely moved in any direction.

Here is a list of styles you can customize on CanvasFrame:
> background_color (color): the main color of the rectangle

Here is an example of a CanvasFrame, you can scroll the middle mouse to zoom the canvas and middle mouse move to pan in it (press CTRL to avoid scrolling the docs).

```
from omni.ui import color as cl
TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
)

IMAGE = "resources/icons/ov_logo_square.png"
with ui.CanvasFrame(height=256, style={"CanvasFrame":{"background_color": cl("#aa4444")}}):
    with ui.VStack(height=0, spacing=10):
        ui.Label(TEXT, name="text", word_wrap=True)
        ui.Button("Button")
        ui.Image(IMAGE, width=128, height=128)
```

## omni.ui.ScrollingFrame
The ScrollingFrame class provides the ability to scroll onto other widgets. ScrollingFrame is used to display the contents of children widgets within a frame. If the widget exceeds the size of the frame, the frame can provide scroll bars so that the entire area of the child widget can be viewed by scrolling.

Here is a list of styles you can customize on ScrollingFrame:
> scrollbar_size (float): the width of the scroll bar
> secondary_color (color): the color the scroll bar
> background_color (color): the background color the scroll frame

Here is an example of a ScrollingFrame, you can scroll the middle mouse to scroll the frame.

```
from omni.ui import color as cl
with ui.HStack():
    left_frame = ui.ScrollingFrame(
        height=250,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        style={"ScrollingFrame":{
            "scrollbar_size":10,
            "secondary_color": cl.red,  # red scroll bar
            "background_color": cl("#4444dd")}}
    )
    with left_frame:
        with ui.VStack(height=0):
            for i in range(20):
                ui.Button(f"Button Left {i}")

    right_frame = ui.ScrollingFrame(
        height=250,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        style={"ScrollingFrame":{
            "scrollbar_size":30,
            "secondary_color": cl.blue, # blue scroll bar
            "background_color": cl("#44dd44")}}
    )
    with right_frame:
        with ui.VStack(height=0):
            for i in range(20):
                ui.Button(f"Button Right {i}")

# Synchronize the scroll position of two frames
def set_scroll_y(frame, y):
    frame.scroll_y = y

left_frame.set_scroll_y_changed_fn(lambda y, frame=right_frame: set_scroll_y(frame, y))
right_frame.set_scroll_y_changed_fn(lambda y, frame=left_frame: set_scroll_y(frame, y))
```

## omni.ui.CollapsableFrame
CollapsableFrame is a frame widget that can hide or show its content. It has two states: expanded and collapsed. When it's collapsed, it looks like a button. If it's expanded, it looks like a button and a frame with the content. It's handy to group properties, and temporarily hide them to get more space for something else.

Here is a list of styles you can customize on CollapsableFrame:
> background_color (color): the background color of the CollapsableFrame widget
> secondary_color (color): the background color of the CollapsableFrame's header
> border_radius (float): the border radius if user wants to round the CollapsableFrame
> border_color (color): the border color if the CollapsableFrame has a border
> border_width (float): the border width if the CollapsableFrame has a border
> padding (float): the distance between the header or the content to the border of the CollapsableFrame
> margin (float): the distance between the CollapsableFrame and other widgets

Here is a default `CollapsableFrame` example:
```
with ui.CollapsableFrame("Header"):
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")
```

It's possible to use a custom header.
```
from omni.ui import color as cl
def custom_header(collapsed, title):
    with ui.HStack():
        with ui.ZStack(width=30):
            ui.Circle(name="title")
            with ui.HStack():
                ui.Spacer()
                align = ui.Alignment.V_CENTER
                ui.Line(name="title", width=6, alignment=align)
                ui.Spacer()
            if collapsed:
                with ui.VStack():
                    ui.Spacer()
                    align = ui.Alignment.H_CENTER
                    ui.Line(name="title", height=6, alignment=align)
                    ui.Spacer()

        ui.Label(title, name="title")

style = {
    "CollapsableFrame": {
        "background_color": cl(0.5),
        "secondary_color": cl.red, # with a red header
        "border_radius": 10,
        "border_color": cl.blue,
        "border_width": 2,
    },
    "CollapsableFrame:hovered": {"secondary_color": cl.green}, # header becomes green when hovered
    "CollapsableFrame:pressed": {"secondary_color": cl.yellow}, # header becomes yellow when pressed
    "Label::title": {"color": cl.white},
    "Circle::title": {
        "color": cl.yellow,
        "background_color": cl.transparent,
        "border_color": cl(0.9),
        "border_width": 0.75,
    },
    "Line::title": {"color": cl(0.9), "border_width": 1},
}

ui.Spacer(height=5)
with ui.HStack():
    ui.Spacer(width=5)
    with ui.CollapsableFrame("Header", build_header_fn=custom_header, style=style):
        with ui.VStack(height=0):
            ui.Button("Hello World")
            ui.Button("Hello World")
    ui.Spacer(width=5)
ui.Spacer(height=5)
```

This example demonstrates how padding and margin work in the collapsable frame.
```
from omni.ui import color as cl
style = {
    "CollapsableFrame": {
        "border_color": cl("#005B96"),
        "border_radius": 4,
        "border_width": 2,
        "padding": 0,
        "margin": 0,
    }
}
frame = ui.CollapsableFrame("Header", style=style)
with frame:
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")

def set_style(field, model, style=style, frame=frame):
    frame_style = style["CollapsableFrame"]
    frame_style[field] = model.get_value_as_float()
    frame.set_style(style)

with ui.HStack():
    ui.Label("Padding:", width=ui.Percent(10), name="text")
    model = ui.FloatSlider(min=0, max=50).model
model.add_value_changed_fn(lambda m: set_style("padding", m))

with ui.HStack():
    ui.Label("Margin:", width=ui.Percent(10), name="text")
    model = ui.FloatSlider(min=0, max=50).model
model.add_value_changed_fn(lambda m: set_style("margin", m))
```


## omni.ui.BezierCurve
BezierCurve is a smooth mathematical curve defined by a set of control points, used to create curves and shapes that can be scaled indefinitely.

Here is a list of common styles you can customize on BezierCurve:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here is a BezierCurve with style:
```
from omni.ui import color as cl
style = {"BezierCurve": {"color": cl.red, "border_width": 2}}
ui.Spacer(height=2)
with ui.Frame(height=50, style=style):
    ui.BezierCurve()
ui.Spacer(height=2)
```

## omni.ui.FreeBezierCurve
FreeBezierCurve uses two widgets to get the position of the curve endpoints. This is super useful to build graph connections.

Here is a list of common styles you can customize on BezierCurve:
> color (color): the color of the line or curve
> border_width (float): the thickness of the line or curve.

Here is an example of a FreeBezierCurve which is controlled by 4 control points.

```
from omni.ui import color as cl
with ui.ZStack(height=400):
    # The Bezier tangents
    tangents = [(50, 50), (-50, -50)]

    # Four draggable rectangles that represent the control points
    placer1 = ui.Placer(draggable=True, offset_x=0, offset_y=0)
    with placer1:
        rect1 = ui.Rectangle(width=20, height=20)
    placer2 = ui.Placer(draggable=True, offset_x=50, offset_y=50)
    with placer2:
        rect2 = ui.Rectangle(width=20, height=20)
    placer3 = ui.Placer(draggable=True, offset_x=100, offset_y=100)
    with placer3:
        rect3 = ui.Rectangle(width=20, height=20)
    placer4 = ui.Placer(draggable=True, offset_x=150, offset_y=150)
    with placer4:
        rect4 = ui.Rectangle(width=20, height=20)

    # The bezier curve
    curve = ui.FreeBezierCurve(rect1, rect4, style={"color": cl.red, "border_width": 5})
    curve.start_tangent_width = ui.Pixel(tangents[0][0])
    curve.start_tangent_height = ui.Pixel(tangents[0][1])
    curve.end_tangent_width = ui.Pixel(tangents[1][0])
    curve.end_tangent_height = ui.Pixel(tangents[1][1])

    # The logic of moving the control points
    def left_moved(_):
        x = placer1.offset_x
        y = placer1.offset_y
        tangent = tangents[0]
        placer2.offset_x = x + tangent[0]
        placer2.offset_y = y + tangent[1]

    def right_moved(_):
        x = placer4.offset_x
        y = placer4.offset_y
        tangent = tangents[1]
        placer3.offset_x = x + tangent[0]
        placer3.offset_y = y + tangent[1]

    def left_tangent_moved(_):
        x1 = placer1.offset_x
        y1 = placer1.offset_y
        x2 = placer2.offset_x
        y2 = placer2.offset_y
        tangent = (x2 - x1, y2 - y1)
        tangents[0] = tangent
        curve.start_tangent_width = ui.Pixel(tangent[0])
        curve.start_tangent_height = ui.Pixel(tangent[1])

    def right_tangent_moved(_):
        x1 = placer4.offset_x
        y1 = placer4.offset_y
        x2 = placer3.offset_x
        y2 = placer3.offset_y
        tangent = (x2 - x1, y2 - y1)
        tangents[1] = tangent
        curve.end_tangent_width = ui.Pixel(tangent[0])
        curve.end_tangent_height = ui.Pixel(tangent[1])

    # Callback for moving the control points
    placer1.set_offset_x_changed_fn(left_moved)
    placer1.set_offset_y_changed_fn(left_moved)
    placer2.set_offset_x_changed_fn(left_tangent_moved)
    placer2.set_offset_y_changed_fn(left_tangent_moved)
    placer3.set_offset_x_changed_fn(right_tangent_moved)
    placer3.set_offset_y_changed_fn(right_tangent_moved)
    placer4.set_offset_x_changed_fn(right_moved)
    placer4.set_offset_y_changed_fn(right_moved)
```

Curve Anchors and Line Anchors allow for decorations to be placed on a curve or line, such that when the shape is moved, the decoration will stay attached to it at the same parametric position.  The anchor has 2 properties for its alignment and position (0-1), and an anchor_fn to supply a callback function which draws the decoration that will be attached to the curve.

Here is an example of an Anchor on a FreeBezierCurve.  The decoration can be dragged along the curve with the left mouse button.
```
from functools import partial
import asyncio

params = [None, None, None, None]

def moved(x, y, b, m):
    x1 = params[0].screen_position_x + params[0].computed_width / 2
    x2 = params[1].screen_position_x + params[1].computed_width / 2
    anchor_position = (x - x1) / (x2 - x1)
    anchor_position = max(min(anchor_position, 1), 0)
    params[2].anchor_position = anchor_position
    params[3].text = f"{params[2].anchor_position:.1f}"

def bound(curve=None):
    with ui.ZStack(content_clipping=1):
        params[3] = ui.Label(f"{params[2].anchor_position:.1f}", mouse_moved_fn=moved)

with ui.ZStack():
    with ui.Placer(draggable=1):
        r1 = ui.Rectangle(width=10, height=10, style={"background_color": ui.color.blue})
    with ui.Placer(draggable=1, offset_x=100, offset_y=100):
        r2 = ui.Rectangle(width=10, height=10, style={"background_color": ui.color.green})
    with ui.Frame(separate_window=True):
        curve = ui.FreeBezierCurve(r1, r2, anchor_position=0.25)
    curve.set_anchor_fn(partial(bound, curve))

params[0] = r1
params[1] = r2
params[2] = curve
```

## omni.ui.VGrid
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

## omni.ui.HGrid
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

## omni.ui.Tooltip
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

## omni.ui.AbstractMultiField
AbstractMultiField is a widget with multiple AbstractFields implemented using the model-delegate-view pattern and uses `omni.ui.AbstractItemModel` as the central component of the system. The item model has multiple AbstractItems each managed by its respective AbstractField's value model.

To access each field's value model, you could get its corresponding item via `get_item_children` before the item's value model with `get_item_value_model`.
```
import omni.ui as ui
# Create a multi float field widget with 3 fields with default set to 0.0, 5.0, 10.0 respectively
multi_field = ui.MultiFloatField(0.0, 5.0, 10.0)
# Get all items of the multi field
items = multi_field.model.get_item_children()
# Get the second item's value model at the first column
multi_field.model.get_item_value_model(items[1], 0)
```

## omni.ui.MultiIntField
Each of the field value could be changed by editing
```execute 200
from omni.ui import color as cl
field_style = {
    "Field": {
        "background_color": cl(0.8),
        "border_color": cl.blue,
        "border_radius": 5,
        "border_width": 1,
        "color": cl.red,
        "font_size": 20.0,
        "padding": 5,
    },
    "Field:pressed": {"background_color": cl.white, "border_color": cl.green, "border_width": 2, "padding": 8},
}

ui.MultiIntField(0, 0, 0, 0, style=field_style)
```

## omni.ui.MultiFloatField
Use MultiFloatField to construct a matrix field:
```execute 200
args = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
ui.MultiFloatField(*args, width=ui.Percent(50), h_spacing=5, v_spacing=2)
```

### omni.ui.MultiFloatDragField
Each of the field value could be changed by dragging
```execute 200
ui.MultiFloatDragField(0.0, 0.0, 0.0, 0.0)
```

## omni.ui.FloatSlider
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

## omni.ui.IntSlider
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

## omni.ui.FloatDrag
Default float drag whose range is -inf and +inf
```execute 200
ui.FloatDrag()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.FloatDrag(min=-10, max=10, step=0.1)
```

With styles and rounded shape:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.FloatDrag(
            min=-180,
            max=180,
            style={
                "color": cl.blue,  # text color
                "background_color": cl(0.8), # background color of the unfilled part of the drag
                "secondary_color": cl.red, # background color of the filled part of the drag
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

## omni.ui.IntDrag
Default int drag whose range is -inf and +inf
```execute 200
ui.IntDrag()
```

With defined Min/Max whose range is between min to max:
```execute 200
ui.IntDrag(min=-10, max=10)
```

With styles and rounded slider:
```execute 200
from omni.ui import color as cl

with ui.HStack(width=200):
    ui.Spacer(width=20)
    with ui.VStack():
        ui.Spacer(height=5)
        ui.IntDrag(
            min=-180,
            max=180,
            style={
                "color": cl.blue, # text color
                "background_color": cl(0.8), # background color of the unfilled part of the drag
                "secondary_color": cl.purple, # background color of the filled part of the drag
                "font_size": 20,
                "border_width": 4,
                "border_color": cl.black,
                "border_radius": 20,
                "padding": 5,
            }
        )
        ui.Spacer(height=5)
    ui.Spacer(width=20)
```


## omni.ui.Pixel
Pixel is the size in pixels and scaled with the HiDPI scale factor. Pixel is the default unit. If a number is not specified to be a certain unit, it is Pixel. e.g. `width=100` meaning `width=ui.Pixel(100)`.

```execute 200
with ui.HStack():
    ui.Button("40px", width=ui.Pixel(40))
    ui.Button("60px", width=ui.Pixel(60))
    ui.Button("100px", width=100)
    ui.Button("120px", width=120)
    ui.Button("150px", width=150)
```

## omni.ui.Percent
Percent and Fraction units make it possible to specify sizes relative to the parent size. 1 Percent is 1/100 of the parent size.

```execute 200
with ui.HStack():
    ui.Button("5%", width=ui.Percent(5))
    ui.Button("10%", width=ui.Percent(10))
    ui.Button("15%", width=ui.Percent(15))
    ui.Button("20%", width=ui.Percent(20))
    ui.Button("25%", width=ui.Percent(25))
```

## omni.ui.Fraction
Fraction length is made to take the available space of the parent widget and then divide it among all the child widgets with Fraction length in proportion to their Fraction factor.

```execute 200
with ui.HStack():
    ui.Button("One", width=ui.Fraction(1))
    ui.Button("Two", width=ui.Fraction(2))
    ui.Button("Three", width=ui.Fraction(3))
    ui.Button("Four", width=ui.Fraction(4))
    ui.Button("Five", width=ui.Fraction(5))
```


## omni.ui.ProgressBar
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

## omni.ui.Placer
Placer enables you to place a widget precisely with offset. Placer's property `draggable` allows changing the position of the child widget by dragging it with the mouse.

There is currently no style you can customize on Placer.

Here is an example of 4 Placers. Two of them have fixed positions, each with a ui.Button as the child. You can see the buttons are moved to the exact place by the parent Placer, one at (100, 10) and the other at (200, 50). The third one is `draggable`, which has a Circle as the child, so that you can move the circle freely with mouse drag in the frame. The fourth one is also `draggable`, which has a ZStack as the child. The ZStack is composed of Rectangle and HStack and Label. This Placer is only draggable on the Y-axis, defined by `drag_axis=ui.Axis.Y`, so that you can only move the ZStack on the y-axis.


```execute 200
from omni.ui import color as cl
with ui.ScrollingFrame(
    height=170,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
):
    with ui.ZStack():
        with ui.HStack():
            for index in range(60):
                ui.Line(width=10, style={"color": cl.black, "border_width": 0.5}, alignment=ui.Alignment.LEFT)
        with ui.VStack():
            ui.Line(
                height=10,
                width=600,
                style={"color": cl.black, "border_width": 0.5},
                alignment=ui.Alignment.TOP,
            )
            for index in range(15):
                ui.Line(
                    height=10,
                    width=600,
                    style={"color": cl.black, "border_width": 0.5},
                    alignment=ui.Alignment.TOP,
                )
            ui.Line(
                height=10,
                width=600,
                style={"color": cl.black, "border_width": 0.5},
                alignment=ui.Alignment.TOP,
            )

        with ui.Placer(offset_x=100, offset_y=10):
            ui.Button("moved 100px in X, and 10px in Y", width=0, height=20, name="placed")

        with ui.Placer(offset_x=200, offset_y=50):
            ui.Button("moved 200px X , and 50 Y", width=0, height=0)

        def set_text(widget, text):
            widget.text = text

        with ui.Placer(draggable=True, offset_x=300, offset_y=100):
            ui.Circle(radius=50, width=50, height=50, size_policy=ui.CircleSizePolicy.STRETCH, name="placed")

        placer = ui.Placer(draggable=True, drag_axis=ui.Axis.Y, offset_x=400, offset_y=120)

        with placer:
            with ui.ZStack(width=180, height=40):
                ui.Rectangle(name="placed")
                with ui.HStack(spacing=5):
                    ui.Circle(
                        radius=3,
                        width=15,
                        size_policy=ui.CircleSizePolicy.FIXED,
                        style={"background_color": cl.white},
                    )
                    ui.Label("UP / Down", style={"color": cl.white, "font_size": 16.0})
                    offset_label = ui.Label("120", style={"color": cl.white})

        placer.set_offset_y_changed_fn(lambda o: set_text(offset_label, str(o)))
```

The following example shows the way to interact between three Placers to create a resizable rectangle's body, left handle and right handle. The rectangle can be moved on X-axis and can be resized with small orange handles.

When multiple widgets fire the callbacks simultaneously, it's possible to collect the event data and process them one frame later using asyncio.

```execute 200
import asyncio
import omni.kit.app
from omni.ui import color as cl

def placer_track(self, id):
    # Initial size
    BEGIN = 50 + 100 * id
    END = 120 + 100 * id

    HANDLE_WIDTH = 10

    class EditScope:
        """The class to avoid circular event calling"""

        def __init__(self):
            self.active = False

        def __enter__(self):
            self.active = True

        def __exit__(self, type, value, traceback):
            self.active = False

        def __bool__(self):
            return not self.active

    class DoLater:
        """A helper to collect data and process it one frame later"""

        def __init__(self):
            self.__task = None
            self.__data = []

        def do(self, data):
            # Collect data
            self.__data.append(data)

            # Update in the next frame. We need it because we want to accumulate the affected prims
            if self.__task is None or self.__task.done():
                self.__task = asyncio.ensure_future(self.__delayed_do())

        async def __delayed_do(self):
            # Wait one frame
            await omni.kit.app.get_app().next_update_async()

            print(f"In the previous frame the user clicked the rectangles: {self.__data}")
            self.__data.clear()

    self.edit = EditScope()
    self.dolater = DoLater()

    def start_moved(start, body, end):
        if not self.edit:
            # Something already edits it
            return

        with self.edit:
            body.offset_x = start.offset_x
            rect.width = ui.Pixel(end.offset_x - start.offset_x + HANDLE_WIDTH)

    def body_moved(start, body, end, rect):
        if not self.edit:
            # Something already edits it
            return

        with self.edit:
            start.offset_x = body.offset_x
            end.offset_x = body.offset_x + rect.width.value - HANDLE_WIDTH

    def end_moved(start, body, end, rect):
        if not self.edit:
            # Something already edits it
            return

        with self.edit:
            body.offset_x = start.offset_x
            rect.width = ui.Pixel(end.offset_x - start.offset_x + HANDLE_WIDTH)

    with ui.ZStack(height=30):
        # Body
        body = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=BEGIN)
        with body:
            rect = ui.Rectangle(width=END - BEGIN + HANDLE_WIDTH)
            rect.set_mouse_pressed_fn(lambda x, y, b, m, id=id: self.dolater.do(id))
        # Left handle
        start = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=BEGIN)
        with start:
            ui.Rectangle(width=HANDLE_WIDTH, style={"background_color": cl("#FF660099")})
        # Right handle
        end = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=END)
        with end:
            ui.Rectangle(width=HANDLE_WIDTH, style={"background_color": cl("#FF660099")})

    # Connect them together
    start.set_offset_x_changed_fn(lambda _, s=start, b=body, e=end: start_moved(s, b, e))
    body.set_offset_x_changed_fn(lambda _, s=start, b=body, e=end, r=rect: body_moved(s, b, e, r))
    end.set_offset_x_changed_fn(lambda _, s=start, b=body, e=end, r=rect: end_moved(s, b, e, r))

ui.Spacer(height=5)
with ui.ZStack():
    placer_track(self, 0)
    placer_track(self, 1)
ui.Spacer(height=5)
```

It's possible to set `offset_x` and `offset_y` in percentages. It allows stacking the children to the proportions of the parent widget. If the parent size is changed, then the offset is updated accordingly.
```execute 200
from omni.ui import color as cl

# The size of the rectangle
SIZE = 20.0

with ui.ZStack(height=200):
    # Background
    ui.Rectangle(style={"background_color": cl(0.6)})

    # Small rectangle
    p = ui.Percent(50)
    placer = ui.Placer(draggable=True, offset_x=p, offset_y=p)
    with placer:
        ui.Rectangle(width=SIZE, height=SIZE)

def clamp_x(offset):
    if offset.value < 0:
        placer.offset_x = ui.Percent(0)
    max_per = 100.0 - SIZE / placer.computed_width * 100.0
    if offset.value > max_per:
        placer.offset_x = ui.Percent(max_per)

def clamp_y(offset):
    if offset.value < 0:
        placer.offset_y = ui.Percent(0)
    max_per = 100.0 - SIZE / placer.computed_height * 100.0
    if offset.value > max_per:
        placer.offset_y = ui.Percent(max_per)

# Callbacks
placer.set_offset_x_changed_fn(clamp_x)
placer.set_offset_y_changed_fn(clamp_y)
```


## omni.ui.Window
The window is a child window of the MainWindow. And it can be docked. You can have any type of widgets as the window content widgets.

Here is a list of styles you can customize on Window:
> background_color (color): the background color of the window.
> border_color (color): the border color if the window has a border.
> border_radius (float): the radius of the corner angle if the user wants to round the window.
> border_width (float): the border width if the window has a border.

Here is an example of a window with style. Click the button to show the window.
```execute 200
from omni.ui import color as cl

self._style_window_example = None
def create_styled_window():
    if not self._style_window_example:
        self._style_window_example = ui.Window("Styled Window Example", width=300, height=300)
        self._style_window_example.frame.set_style({
            "Window": {
                "background_color": cl.blue,
                "border_radius": 10,
                "border_width": 5,
                "border_color": cl.red,
            }})
        self._style_window_example.visible = True

ui.Button("click for Styled Window", width=180, clicked_fn=create_styled_window)
```

Note that a window's style is set from its frame since ui.Window itself is not a widget. We can't set style to it like other widgets. ui.Window's frame is a normal ui.Frame widget which itself doesn't have styles like `background_color` or `border_radius` (see `Container Widgets`->`Frame`). We specifically interpret the input ui.Window's frame style as the window style here. Therefore, the window style is not propagated to the content widget either just like the MainWindow.

If you want to set up a default style for the entire window. You should use `ui.style.default`. More details in `The Style Sheet Syntax` -> `Style Override` -> `Default style override`.


## omni.ui.MainWindow
The MainWindow represents the main window for an application. There should only be one MainWindow in each application.

Here is a list of styles you can customize on MainWindow:

> background_color (color): the background color of the main window.
> margin_height (float): the height distance between the window content and the window border.
> margin_width (float): the width distance between the window content and the window border.

Here is an example of a main window with style. Click the button to show the main window. Since the example is running within a MainWindow already, creating a new MainWindow will not run correctly in this example, but it demonstrates how to set the style of the `MainWindow`. And note the style of MainWindow is not propagated to other windows.

```execute 200
from omni.ui import color as cl

self._main_window = None
self._window1 = None
self._window2 = None
def create_main_window():
    if not self._main_window:
        self._main_window = ui.MainWindow()
        self._main_window.main_frame.set_style({
            "MainWindow": {
                "background_color": cl.purple,
                "margin_height": 20,
                "margin_width": 10
            }})
        self._window1 = ui.Window("window 1", width=300, height=300)
        self._window2 = ui.Window("window 2", width=300, height=300)
        main_dockspace = ui.Workspace.get_window("DockSpace")
        self._window1.dock_in(main_dockspace, ui.DockPosition.SAME)
        self._window2.dock_in(main_dockspace, ui.DockPosition.SAME)
        self._window2.focus()
    self._window2.visible = True

ui.Button("click for Main Window", width=180, clicked_fn=create_main_window)
```


## omni.ui.Menu
The Menu class provides a menu widget for use in menu bars, context menus, and other popup menus. It can be either a pull-down menu in a menu bar or a standalone context menu. Pull-down menus are shown by the menu bar when the user clicks on the respective item. Context menus are usually invoked by some special keyboard key or by right-clicking.

Here is a list of styles you can customize on Menu:
> color (color): the color of the menu text
> background_color (color): the background color of sub menu window
> background_selected_color (color): the background color when the current menu is selected
> border_color (color): the border color of the sub menu window if it has a border
> border_width (float): the border width of the sub menu window if it has a border
> border_radius (float): the border radius of the sub menu window if user wants to round the sub menu window
> padding (float): the padding size of the sub menu window

Here is a list of styles you can customize on MenuItem:
> color (color): the color of the menu Item text
> background_selected_color (color): the background color when the current menu is selected

Right click for the context menu with customized menu style:
```execute 200
from omni.ui import color as cl
self.context_menu = None
def show_context_menu(x, y, button, modifier, widget):
    if button != 1:
        return
    self.context_menu = ui.Menu("Context menu",
        style={
                "Menu": {
                    "background_color": cl.blue,
                    "color": cl.pink,
                    "background_selected_color": cl.green,
                    "border_radius": 5,
                    "border_width": 2,
                    "border_color": cl.yellow,
                    "padding": 15
                    },
                "MenuItem": {
                    "color": cl.white,
                    "background_selected_color": cl.cyan},
                "Separator": {
                    "color": cl.red},
            },)
    with self.context_menu:
        ui.MenuItem("Delete Shot")
        ui.Separator()
        ui.MenuItem("Attach Selected Camera")
        with ui.Menu("Sub-menu"):
            ui.MenuItem("One")
            ui.MenuItem("Two")
            ui.MenuItem("Three")
            ui.Separator()
            ui.MenuItem("Four")
            with ui.Menu("Five"):
                ui.MenuItem("Six")
                ui.MenuItem("Seven")

    self.context_menu.show()

with ui.VStack():
    button = ui.Button("Right click to context menu", height=0, width=0)
    button.set_mouse_pressed_fn(lambda x, y, b, m, widget=button: show_context_menu(x, y, b, m, widget))
```

Left click for the push button menu with default menu style:
```execute 200
self.pushed_menu = None
def show_pushed_menu(x, y, button, modifier, widget):
    self.pushed_menu = ui.Menu("Pushed menu")
    with self.pushed_menu:
        ui.MenuItem("Camera 1")
        ui.MenuItem("Camera 2")
        ui.MenuItem("Camera 3")
        ui.Separator()
        with ui.Menu("More Cameras"):
            ui.MenuItem("This Menu is Pushed")
            ui.MenuItem("and Aligned with a widget")

    self.pushed_menu.show_at(
            (int)(widget.screen_position_x), (int)(widget.screen_position_y + widget.computed_content_height)
        )

with ui.VStack():
    button = ui.Button("Pushed Button Menu", height=0, width=0)
    button.set_mouse_pressed_fn(lambda x, y, b, m, widget=button: show_pushed_menu(x, y, b, m, widget))
```

## omni.ui.Separator
Separator is a widget type which draws a omni.ui.Line. It could be used to
as a MenuItem which creates a separator line in the UI elements.

We can also define a text for separator, and the text will be rendered before the line in the horizontal layout.

```
ui.Separator("Section 1")
```

Here is a list of styles you can customize on Separator:
> color (color): the color of the Separator

## omni.ui.MenuBar
All the Windows in Omni.UI can have a MenuBar. To add a MenuBar to your window add this flag to your constructor: omni.ui.Window(flags=ui.WINDOW_FLAGS_MENU_BAR). The MenuBar object can then be accessed through the menu_bar read-only property on your window.

A MenuBar is a container so it is built like a Frame or Stack but only takes Menu objects as children. You can leverage the 'priority' property on the Menu to order them. They will automatically be sorted when they are added, but if you change the priority of an item then you need to explicitly call sort().

MenuBar has exactly the same style list you can customize as Menu.

Here is an example of MenuBar with style for the Window:

```execute 200
from omni.ui import color as cl
style={"MenuBar": {
            "background_color": cl.blue,
            "color": cl.pink,
            "background_selected_color": cl.green,
            "border_radius": 2,
            "border_width": 1,
            "border_color": cl.yellow,
            "padding": 2}}
self._window_menu_example = None
def create_and_show_window_with_menu():
    if not self._window_menu_example:
        self._window_menu_example = ui.Window(
            "Window Menu Example",
            width=300,
            height=300,
            flags=ui.WINDOW_FLAGS_MENU_BAR | ui.WINDOW_FLAGS_NO_BACKGROUND,
        )
        menu_bar = self._window_menu_example.menu_bar
        menu_bar.style = style
        with menu_bar:
            with ui.Menu("File"):
                ui.MenuItem("Load")
                ui.MenuItem("Save")
                ui.MenuItem("Export")
            with ui.Menu("Window"):
                ui.MenuItem("Hide")

        with self._window_menu_example.frame:
            with ui.VStack():
                ui.Button("This Window has a Menu")

                def show_hide_menu(menubar):
                    menubar.visible = not menubar.visible

                ui.Button("Click here to show/hide Menu", clicked_fn=lambda m=menu_bar: show_hide_menu(m))

                def add_menu(menubar):
                    with menubar:
                        with ui.Menu("New Menu"):
                            ui.MenuItem("I don't do anything")

                ui.Button("Add New Menu", clicked_fn=lambda m=menu_bar: add_menu(m))

    self._window_menu_example.visible = True

with ui.HStack(width=0):
    ui.Button("window with MenuBar Example", width=180, clicked_fn=create_and_show_window_with_menu)
    ui.Label("this populates the menuBar", name="text", width=180, style={"margin_width": 10})
```

## omni.ui.Style
Each type of widget has a list of styles you can customize. There are no other style properties exist except the ones listed for each widget. Please don't make up style properties.

Make sure you read the style property explanation before using them. For example, `secondary_color` for CollapsableFrame defines the background color of the CollapsableFrame's header, while `secondary_color` for TreeView defines the color of the resizer between columns.

## omni.ui.scene.Line
Line is the simplest shape that represents a straight line. It has two points,
color, and thickness.

```execute
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Line([-0.5,-0.5,0], [0.5, 0.5, 0], color=cl.green, thickness=5)
```

## omni.ui.scene.Curve
Curve is a shape drawn with multiple lines which has a bent or turns in it. There are two supported cuve types: linear and cubic. `sc.Curve` is default to draw cubic curve and can be switched to linear with `curve_type=sc.Curve.CurveType.LINEAR`. `tessellation` controls the smoothness of the curve. The higher the value is, the smoother the curve is, with the higher computational cost. The curve also has positions, colors, thicknesses which are all array typed properties, which means per vertex property is supported. This feature needs further development with ImGui support.

```execute
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)
##

with scene_view.scene:
    # linear curve
    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(-4, 0, 0)):
        sc.Curve(
            [[0.5, -0.7, 0], [0.1, 0.6, 0], [2.0, 0.6, 0], [3.5, -0.7, 0]],
            thicknesses=[1.0],
            colors=[cl.red],
            curve_type=sc.Curve.CurveType.LINEAR,
        )
    # corresponding cubic curve
    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0, 0, 0)):
        sc.Curve(
            [[0.5, -0.7, 0], [0.1, 0.6, 0], [2.0, 0.6, 0], [3.5, -0.7, 0]],
            thicknesses=[3.0],
            colors=[cl.blue],
            tessellation=9,
        )
```

## omni.ui.scene.Rectangle
Rectangle is a shape with four sides and four corners. The corners are all right angles.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Rectangle(color=cl.green)
```

It's also possible to draw Rectangle with lines with enabling property `wireframe`:

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Rectangle(2, 1, thickness=5, wireframe=True)
```

## omni.ui.scene.Arc
Two radii of a circle and the arc between them. It also can be a wireframe.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Arc(1, begin=0, end=1, thickness=5, wireframe=True)
```

## omni.ui.scene.Image
A rectangle with an image on it. It can read raster and vector graphics format
and supports `http://` and `omniverse://` paths.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
from pathlib import Path

EXT_PATH = f"{Path(__file__).parent.parent.parent.parent.parent}"

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    filename = f"{EXT_PATH}/data/main_ov_logo_square.png"
    sc.Image(filename)
```

## omni.ui.scene.Points
The list of points in 3d space. Points may have different sizes and different colors.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
import math

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    point_count = 36
    points = []
    sizes = []
    colors = []
    for i in range(point_count):
        weight = i / point_count
        angle = 2.0 * math.pi * weight
        points.append(
            [math.cos(angle), math.sin(angle), 0]
        )
        colors.append([weight, 1 - weight, 1, 1])
        sizes.append(6 * (weight + 1.0 / point_count))
    sc.Points(points, colors=colors, sizes=sizes)
```

## omni.ui.scene.PolygonMesh

Encodes a mesh. Meshes are defined as points connected to edges and faces. Each
face is defined by a list of face vertices `vertex_indices` using indices into
the point `positions` array. `vertex_counts` provides the number of points at
each face. This is the minimal requirement to construct the mesh.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
import math

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    point_count = 36

    # Form the mesh data
    points = []
    vertex_indices = []
    sizes = []
    colors = []
    for i in range(point_count):
        weight = i / point_count
        angle = 2.0 * math.pi * weight
        vertex_indices.append(i)
        points.append(
            [
                math.cos(angle) * weight,
                -math.sin(angle) * weight,
                0
            ]
        )
        colors.append([weight, 1 - weight, 1, 1])
        sizes.append(6 * (weight + 1.0 / point_count))

    # Draw the mesh
    sc.PolygonMesh(
        points, colors, [point_count], vertex_indices
    )
```

## omni.ui.scene.TexturedMesh
Encodes a polygonal mesh with free-form textures. Meshes are defined the same as PolygonMesh. It supports both ImageProvider and URL. Basically it's PolygonMesh with the ability to use images. Users can provide either sourceUrl or imageProvider, just as sc.Image as the source of the texture. And `uvs` provides how the texture is applied to the mesh.

NOTE: in Kit 105 UVs are specified with V-coordinate flipped, while Kit 106 will move to specifying UVs in same "space" as USD.
In 105.1 there is a transitional property "legacy_flipped_v" that can be provided to the TexturedMesh constructor to internally handle the conversion, but specifying UV cordinates with legacy_flipped_v=True has a negative performance impact.

```execute 150
from omni.ui import scene as sc
from pathlib import Path

EXT_PATH = f"{Path(__file__).parent.parent.parent.parent.parent}"

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    point_count = 4
    # Form the mesh data
    points = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0)]
    vertex_indices = [0, 2, 3, 1]
    colors = [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]
    uvs = [(0, 0), (0, 1), (1, 1), (1, 0)]
    # Draw the mesh
    filename = f"{EXT_PATH}/data/main_ov_logo_square.png"
    sc.TexturedMesh(filename, uvs, points, colors, [point_count], vertex_indices, legacy_flipped_v=False)
```

## omni.ui.scene.Label

Defines a standard label for user interface items. The text size is always in
the screen space and oriented to the camera. It supports `omni.ui` alignment.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
import math

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Label(
        "NVIDIA Omniverse",
        alignment=ui.Alignment.CENTER,
        color=cl("#76b900"),
        size=50
    )
```

## omni.kit.widget.filter.FilterButton

Allows users to access and toggle a set of predefined filter options presented in a dropdown menu. The button reflects the state of the filters, such as when there are unsaved changes, and provides callbacks for mouse interactions, making it interactive and responsive to user input.

To use a FilterButton, you must include ONLY ONE parameter: option_items, which is a list of OptionItems. No other types of options are allowed, this list is required, and no other parameters should be passed in.

The following example creates a UI Window with a FilterButton that displays a list of options of OptionItem type that includes filtering by audio, materials, scripts, textures, and usd:

```execute 150
import omni.ui as ui
from omni.kit.widget.options_menu import OptionItem
from omni.kit.widget.filter import FilterButton

# Create filter option items
option_items = [
    OptionItem("audio", text="Audio"),
    OptionItem("materials", text="Materials"),
    OptionItem("scripts", text="Scripts"),
    OptionItem("textures", text="Textures"),
    OptionItem("usd", text="USD"),
]

# Display the filter button in a UI window
window = ui.Window("Filter Example Window", width=200, height=200)
with window.frame:
    with ui.VStack():
        # Create a filter button with the given option items
        filter_button = FilterButton(option_items)

# Set first filter item on
model = filter_button.model
model.get_item_children()[0].value = True
```

## omni.kit.widget.option_menu.OptionsMenu

Represents a menu to show various options with a header and list of menu items. This is an example of creating an OptionsMenu with various items such as OptionItems, OptionRadios. OptionsSeparator, and OptionSeparator.

```execute 150
from omni.kit.widget.options_menu import OptionsMenu, OptionsModel, OptionItem, OptionSeparator, OptionRadios, OptionCustom
import omni.ui as ui

# Define a custom build function for the custom menu item
def custom_build_function():
    return ui.Label("Custom Option")

# Define radio options
radios = [
    ("Radios", None),
    "First",
    ("Second", "Second Radio"),
    "Third",
]

# Create an options model with items
options_model = OptionsModel(
    "Options",
    [
        OptionItem("audio", text="Audio"),
        OptionItem("materials", text="Materials"),
        OptionItem("scripts", text="Scripts"),
        OptionItem("textures", text="Textures"),
        OptionItem("usd", text="USD"),
        OptionRadios(radios, default="First"),
        OptionSeparator(),
        OptionCustom(build_fn=custom_build_function),
        OptionSeparator(),
        OptionItem("Disabled", enabled=False),
        OptionItem("Disabled and checked", default=True, enabled=False),
    ]
)

# Instantiate and display the options menu
options_menu = OptionsMenu(model=options_model)
options_menu.show_at(100, 100)  # Position where the menu will appear
```

## omni.kit.widget.option_menu.RadioMenu

Represents a menu specifically for radio button groups. This is an example of a RadioMenu that allows you to sort by Name, Date, Price, in ascending, descending, or random order. Note that RadioMenu uses RadioModel parameters of field_model and order_model.

```execute 150
from omni.kit.widget.options_menu import RadioMenu, RadioModel

field_model = RadioModel(["Name", "Date", "Price"], default_index=1)
order_model = RadioModel(["Ascending", "Descending", "Random"])

menu = RadioMenu("Sort By", [field_model, order_model])
menu.show_at(100, 100)
```

## omni.kit.widget.option_menu.OptionsModel

Represents a model for managing a collection of option items within a menu. This is an example of initializing an OptionsModel with various items like OptionItems for audio, materials, and scripts, as well as an OptionSeparator and OptionCustom.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionItem, OptionSeparator

model = OptionsModel(
    "Options",
    [
        OptionItem("Audio", text="Audio"),
        OptionItem("Materials", text="Materials"),
        OptionItem("Scripts", text="Scripts"),
        OptionSeparator(),
    ]
)
```

## omni.kit.widget.option_menu.RadioModel

Represents a model for managing a collection of option items within a menu. This example shows a RadioModel storing name, date, and price that can later be used in a RadioMenu.

```execute 150
from omni.kit.widget.options_menu import RadioModel

field_model = RadioModel(["Name", "Date", "Price"], default_index=1)
menu = RadioMenu("Sort By", [field_model])
menu.show_at(100, 100)
```

## omni.kit.widget.option_menu.OptionCustom

Represents a custom option item with a build function and optional model. This example shows creating a OptionCustom via a function `build_custom_menu_item` passed in as a parameter. This OptionCustom can be included in an OptionsModel.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionCustom

# Function to build custom menu items
def build_custom_menu_item():
    ui.Label("Custom Menu Item")

# Initialize the options model with various items
model = OptionsModel(
    "Custom Option",
    [
        OptionCustom(build_custom_menu_item)  # Using OptionCustom for custom build function
    ]
)
```

## omni.kit.widget.option_menu.OptionItem

Represents a general item for options menus, supporting features like checkability and visibility toggling.

The following are examples of how to initialize a list of OptionItem, where each OptionItem is any string name, followed by the text, default display, and whether the option is enabled or not.

```execute 150
from omni.kit.widget.options_menu import OptionItem

option_items = [
    OptionItem("audio", text="Audio"),
    OptionItem("materials", text="Materials"),
    OptionItem("scripts", text="Scripts"),
    OptionItem("textures", text="Textures"),
    OptionItem("usd", text="USD"),
    OptionItem("Disabled", enabled=False),
    OptionItem("Disabled and checked", default=True, enabled=False),
]
```

## omni.kit.widget.option_menu.OptionRadios

Represents a list of radio options within a menu. This is an example of creating an OptionsModel with an OptionRadios that is defined by the list `radios` and contains various radio field options.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionRadios

# Define radio options
radios = [
    ("Radios", None),
    "First",
    ("Second", "Second Radio"),
    "Third",
]

# Create an options model with items
options_model = OptionsModel(
    "Options with Radio",
    [
        OptionRadios(radios, default="First"),
    ]
)
```

## omni.kit.widget.option_menu.OptionSeparator

Represents a separator in menu items, optionally with a title. This is an example of using OptionSeparator to separate two OptionItem in an OptionsModel.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionItem, OptionSeparator

model = OptionsModel(
    "Options",
    [
        OptionItem("Audio", text="Audio"),
        OptionSeparator(),
        OptionItem("Materials", text="Materials"),
    ]
)
```

## omni.kit.widget.option_menu.OptionLabelMenuItemDelegate

A delegate for a normal menu item with additional spacing. This shows two use cases of OptionLabelMenuItemDelegate, where we pass a delegate for an OptionCustom of a ui.MenuItem.

```execute 150
from omni.kit.widget.options_menu import OptionCustom, OptionSeparator, OptionLabelMenuItemDelegate

option_items = [
    OptionCustom(build_fn=lambda: ui.MenuItem("Reload Outdated Primitives", delegate=OptionLabelMenuItemDelegate(), triggered_fn=self._on_reload_all_prims)),
    OptionSeparator(),
    OptionCustom(build_fn=lambda: ui.MenuItem("Reset", delegate=OptionLabelMenuItemDelegate(), triggered_fn=self._on_reset)),
]
```

## omni.kit.widget.searchfield.SearchField

A widget that represents a search field where users can input search words and receive suggestions.

This is an example of creating a general search field with a callback

```python
from typing import List, Optional
from omni.kit.widget.searchfield import SearchField
from omni import ui

def on_search(search_words: Optional[List[str]]):
    if search_words is not None:
        print(f"Now searching {search_words}")

search_field = SearchField(on_search_fn=on_search)
```

This is an example of creating a Search Field with Suggestions

```python
from omni.kit.widget.searchfield import SearchField
from omni import ui

search_field = SearchField(suggestions=["apple", "banana", "cherry"])
```

This is an example of creating a Search Field with and without Tokens.

The search field can be set to show or not show tokens. When `show_token` is set to `True`, each search term is rendered as a SearchWordButton, where you can click to remove some of the tokens from the search. When `show_token` is set to `False`, the search terms are rendered as plain text without tokenization. `show_token` is set to `True` by default.

```python
from omni.kit.widget.searchfield import SearchField
from omni import ui

search_field = SearchField(show_tokens=True)
search_field.search_words = ["show", "tokens"]

search_field_no_tokens = SearchField(show_tokens=False)
search_field_no_tokens.text = "do not show tokens"
```

## omni.kit.widget.searchable_combobox.SearchModel

A model to support searching functionality in UI components.

This class provides methods to set and retrieve search strings, and to determine if a given string is part of the current search string. It is designed to be used within UI components that require search capabilities, such as searchable combo boxes or lists, and takes a `modified_fn` callable function that's called when the search string is modified.

A simple example of using `SearchModel` to create the base model of a `SearchWidget`, setting and getting the search string, and checking if a string is part of the search string:

```python
from omni.kit.widget.searchable_combobox import SearchModel

def on_search_modified(search_string):
    print(f"Search string modified: {search_string}")

search_model = SearchModel(modified_fn=on_search_modified)

search_model.set_search_string("search string")
search_string = search_model.get_search_string()

is_search_string = search_model.is_in_string("search")
```

## omni.kit.widget.searchable_combobox.SearchWidget

A widget that provides a searchable combobox with customizable styles.

This is an example of creating a SearchWidget with a theme and icon path, getting the text, setting the text, and destroying the SearchWidget. `SearchWidget` can also take an optional `modified_fn` parameter, which is a callback function for when search input is modified. This example also uses the `build_ui_popup` function inside `SearchWidget` to create a popup with a list of items.

```python
from omni.kit.widget.searchable_combobox import SearchWidget

search_widget = SearchWidget(theme=theme, icon_path=None)
name_field, listbox_button = search_widget.build_ui_popup(
    search_size=widget_height,
    default_value=default_value,
    popup_text=combo_list[combo_index] if combo_index >= 0 else default_value,
    index=combo_index,
    update_fn=combo_click_fn,
)

widget_name = search_widget.get_text()
search_widget.set_text("new text for search widget")
search_widget.destroy()
```

## omni.kit.widget.searchable_combobox.build_searchable_combo_widget

A public function that easily helps create a searchable combo widget. This function is accessible by writing `from omni.kit.widget.searchable_combobox import build_searchable_combo_widget`, and returns a SearchWidget object (an instance of the `SearchWidget` with an attached searchable combo box).

Here is an example of how to use the `build_searchable_combo_widget` function to build a searchable combo widget with a list of showroom components:

```python
from omni.kit.widget.searchable_combobox import build_searchable_combo_widget

def on_combo_click_fn(model):
    component = model.get_value_as_string()
    print(f"{component} selected")

component_list = ['3d Reconstruction', 'AEC Experience', 'AI Framework', 'AI Toybox']
component_index = -1
self._component_combo = build_searchable_combo_widget(component_list, component_index, on_combo_click_fn, widget_height=18, default_value="Kit")
```

Cleanup:

```
self._component_combo.destroy()
self._component_combo = None
```

Here is an example of creating a general searchable combobox with `build_searchable_combo_widget`:

```python
import omni.kit.app
import omni.ui as ui
import asyncio
from omni.kit.widget.searchable_combobox import build_searchable_combo_widget, ComboBoxListDelegate

# Create a searchable combo box widget
def create_searchable_combo_box():
    # Define a callback function to be called when an item is selected
    def on_combo_click_fn(model):
        selected_item = model.get_value_as_string()
        print(f"Selected item: {selected_item}")

    # Define the list of items for the combo box
    item_list = ["Item 1", "Item 2", "Item 3"]

    # Create the searchable combo box with the specified items and callback
    searchable_combo_widget = build_searchable_combo_widget(
        combo_list=item_list,
        combo_index=-1,  # Start with no item selected
        combo_click_fn=on_combo_click_fn,
        widget_height=18,
        default_value="Select an item",  # Placeholder text when no item is selected
        window_id="SearchableComboBoxWindow",
        delegate=ComboBoxListDelegate()  # Use the default delegate for item rendering
    )

    # Return the created widget
    return searchable_combo_widget

searchable_combo_box = create_searchable_combo_box()
```

## omni.kit.widget.searchable_combobox.ComboBoxListItem

A single item of the model, with a text string to be displayed by the item.
This is an example of creating a ComboBoxListItem with the text "Apple".

```python
from omni.kit.widget.searchable_combobox import ComboBoxListItem

item = ComboBoxListItem("Apple")
```

## omni.kit.widget.searchable_combobox.ComboBoxListModel

A model for managing a list of items in a ComboBox. This is an example of creating a ComboBoxListModel with a list of fruits.

```python
from omni.kit.widget.searchable_combobox import ComboBoxListModel

fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
model = ComboBoxListModel(fruits)
```

## omni.kit.widget.searchable_combobox.ComboBoxListDelegate

A delegate for custom widget representation in a TreeView widget. This is an example of creating a ComboBoxListDelegate with a custom build function to render a custom widget in a TreeView:

```python
from omni.kit.widget.searchable_combobox import ComboBoxListDelegate
from omni.ui import color as cl

def build_custom_widget():
    return ui.Label("Custom Widget", style={"color": cl.red})

delegate = ComboBoxListDelegate(build_fn=build_custom_widget)
```

## omni.kit.widget.searchable_combobox.ComboListBoxWidget

A widget that combines a search field with a list box for item selection.

This widget facilitates the selection of items from a list that can be filtered through a search interface. Users can type in the search widget to filter the items displayed in the list box below it. It supports customization through themes and delegates for item display.

This is an example of creating a ComboListBoxWidget with a search widget, a list of items, a window ID, a delegate, and a theme:
We also set the parent of the listbox widget to the name field and build the UI by calling the ComboListBoxWidget's `set_parent` and `build_ui` methods.

```python
from omni.kit.widget.searchable_combobox import ComboListBoxWidget

listbox_widget = ComboListBoxWidget(
    search_widget=search_widget, item_list=combo_list, window_id=window_id, delegate=delegate, theme=theme
)
listbox_widget.set_parent(name_field)
listbox_widget.build_ui()
```

## uicode.EditableTreeWidget

A two column treeview with editable fields. The first column is a string and the second column could be a string, float or an int. With double click, the clicked field becomes editable.

Example: Create a table of prim names and corresponding numbers of them in the scene

```python
import omni.ui as ui
import uicode
window = ui.Window("Capital & Population table", height=500, width=500)
with window.frame:
    with ui.ScrollingFrame():
        data = {"Sphere": 4, "Cube": 3, "Torus": 1}
        treeview=uicode.EditableTreeWidget(data)
```