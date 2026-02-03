# omni.ui
There are 3 main classes of omni.ui Objects, the Shapes, the Widgets, and the Containers

## Shapes
Shapes are the most basic elements in the ui. These are the defined shape types: Rectangle, Circle, Ellipse, Triangle. In most cases those shapes will fit into the widget size which is defined by the parent widget they are in.

Freeshapes are the extended shapes of FreeRectangle, FreeCircle, FreeEllipse, FreeTriangle, which allow users to control some of the attributes dynamically through bounded widgets. It means it is possible to change the freeshape's shape by moving the bounded widgets.

## Widgets
Widgets are mostly a combination of shapes, images or texts, which are created to be stepping stones for the entire ui window. Each of the widget has its own style to be characterized.

They have width and height properties in pixel, and they don't have children
Here is a list of the main ones and some of their properties:

### Label:
Show some text using the text property, but the constructor requires first argument to be text.
The label has an alignment property that can be ui.Alignment.CENTER, ui.Alignment.LEFT or ui.Alignment.RIGHT.

### Button:
The Button also has a text property.
The Button also has a source_url property to display an image; it is a valid URL for a jpg or png
The Button can be created with clicked_fn passed in the constructor body
clicked_fn is triggered when the user clicks
clicked_fn is NOT a property of Button
Here is an example of using clicked_fn
ui.Button("a", clicked_fn=lambda:carb.log_warn('clicked'))

### Rectangle:
It is used to give things some background color using the styling

### Field, Sliders and Drags
Here is a common list of styles you can customize on Field, Sliders and Drags:
> background_color (color): the background color of the field or slider
> border_color (color): the border color if the field or slider background has a border
> border_radius (float): the border radius if the user wants to round the field or slider
> border_width (float): the border width if the field or slider background has a border
> padding (float): the distance between the text and the border of the field or slider
> font_size (float): the size of the text in the field or slider

There are fields for string, float and int models.

> color (color): the color of the text
> background_selected_color (color): the background color of the selected text


#### FloatField and IntField:
They are input for Float and Int respectively.
They are model-based. The optional model can be specified as the first argument in the constructor.
If the model is not specified, it's created automatically.
To change the value of the slider, it's necessary to change the value of the model.

#### StringField:
They are used to show or enter text input/output. Model based.
Here is how String Field works:
    They have a model : StringField.model
    you can set the value of the string field using mode.set_value('some string')```

The following example shows how string field, float field and int field interact with each other. All three fields share the same default FloatModel:
```
with ui.HStack(spacing=5):
    ui.Label("FloatField")
    ui.Label("IntField")
    ui.Label("StringField")
with ui.HStack(spacing=5):
    left = ui.FloatField()
    center = ui.IntField()
    right = ui.StringField()
    center.model = left.model
    right.model = left.model
ui.Spacer(height=5)
```

### MultiField
MultiField widget groups the widgets that have multiple similar widgets to represent each item in the model. It's handy to use them for arrays and multi-component data like float3, matrix, and color.

MultiField is using `Field` as the Type Selector. Therefore, the list of styless we can customize on MultiField is the same as Field


### Sliders
The Sliders are more like a traditional slider that can be dragged and snapped where you click. The value of the slider can be shown on the slider or not, but can not be edited directly by clicking.

Here is a list of styles you can customize on Sliders:
> background_color (color): the background color of the field or slider
> border_color (color): the border color if the field or slider background has a border
> border_radius (float): the border radius if the user wants to round the field or slider
> border_width (float): the border width if the field or slider background has a border
> padding (float): the distance between the text and the border of the field or slider
> font_size (float): the size of the text in the field or slider
> color (color): the color of the text
> secondary_color (color): the color of the handle in `ui.SliderDrawMode.HANDLE` draw_mode or the background color of the left portion of the slider in `ui.SliderDrawMode.DRAG` draw_mode
> secondary_selected_color (color): the color of the handle when selected, not useful when the draw_mode is FILLED since there is no handle drawn.
> draw_mode (enum): defines how the slider handle is drawn. There are three types of draw_mode.
* ui.SliderDrawMode.HANDLE: draw the handle as a knob at the slider position
* ui.SliderDrawMode.DRAG: the same as `ui.SliderDrawMode.HANDLE` for now
* ui.SliderDrawMode.FILLED: the handle is eventually the boundary between the `secondary_color` and `background_color`

### FloatSlider and IntSlider:
They show a slider that Float and Int respectively. Model based.

Sliders with different draw_mode:
```
from omni.ui import color as cl
with ui.VStack(spacing=5):
    ui.FloatSlider(style={"background_color": cl(0.8),
                           "secondary_color": cl(0.6),
                           "color": cl(0.1),
                           "draw_mode": ui.SliderDrawMode.HANDLE}
                    ).model.set_value(0.5)
    ui.FloatSlider(style={"background_color": cl(0.8),
                           "secondary_color": cl(0.6),
                           "color": cl(0.1),
                           "draw_mode": ui.SliderDrawMode.DRAG}
                    ).model.set_value(0.5)
    ui.FloatSlider(style={"background_color": cl(0.8),
                           "secondary_color": cl(0.6),
                           "color": cl(0.1),
                           "draw_mode": ui.SliderDrawMode.FILLED}
                    ).model.set_value(0.5)
```

### Drags
The Drags are very similar to Sliders, but more like Field in the way that they behave. You can double click to edit the value but they also have a mean to be 'Dragged' to increase or decrease the value.

Except the common style for Fields and Sliders, here is a list of styles you can customize on Drags:
> color (color): the color of the text
> secondary_color (color): the left portion of the slider in `ui.SliderDrawMode.DRAG` draw_mode

### ComboBox
To set value for ComboBox, we need to do
```combo_box.model.get_item_value_model().set_value(index)```
There is no
```combo_box.model.set_value(index)```

### Spacer
This is used to create empty space, if width and height is not set it will use all the space available

### Image
This is used to display images. The image URL is on the source_url property, and the image URL should be passed to the constructor: `ui.Image("c:\\tmp\\image.png")`

## Containers
The Third Class of Objects are Containers, the Containers can have children. The container is using the "with" Python syntax to define their children. It's not possible to reparent items. Instead, it's necessary to remove the item and recreate a similar item under another parent.
They derive from Widget and have the same property as the Widget Abstract class (including width and height).
It provides support on the widgets layout, providing flexibility for the arrangement of elements and possibility of creating more complicated and customized widgets.

They also have a spacing property that add some space inbetween children.

### Frame:
it can have only one child usually another container

### Layout
We have three main components:
VStack: to construct horizontal layout objects.
HStack: to line up widgets vertically.
ZStack: ZStack is a view that overlays its children, aligning them on top of each other. The later one is on top of the previous ones.

Here is a list of styles you can customize on Stack:
> margin (float): the distance between the stack items and the parent widget defined boundary
> margin_width (float): the width distance between the stack items and the parent widget defined boundary
> margin_height (float): the height distance between the stack items and the parent widget defined boundary

It's possible to determine the direction of a stack with the property `direction`. Here is an example of a stack which is able to change its direction dynamically by clicking the button `Change`.

```execute 200
def rotate(dirs, stack, label):
    dirs[0] = (dirs[0] + 1) % len(dirs[1])
    stack.direction = dirs[1][dirs[0]]
    label.text = str(stack.direction)

dirs = [
    0,
    [
        ui.Direction.LEFT_TO_RIGHT,
        ui.Direction.RIGHT_TO_LEFT,
        ui.Direction.TOP_TO_BOTTOM,
        ui.Direction.BOTTOM_TO_TOP,
    ],
]
stack = ui.Stack(ui.Direction.LEFT_TO_RIGHT, width=0, height=0, style={"margin_height": 5, "margin_width": 10})
with stack:
    for name in ["One", "Two", "Three", "Four"]:
        ui.Button(name)
ui.Spacer(height=100)
with ui.HStack():
    ui.Label("Current direction is ", name="text", width=0)
    label = ui.Label("", name="text")

    button = ui.Button("Change")
    button.set_clicked_fn(lambda d=dirs, s=stack, l=label: rotate(d, s, l))
    rotate(dirs, stack, label)
```

Here is an example of using combined HStack and VStack:
```execute 200
with ui.VStack():
    for i in range(2):
        with ui.HStack():
            ui.Spacer(width=50)

            with ui.VStack(height=0):
                ui.Button("Left {}".format(i), height=0)
                ui.Button("Vertical {}".format(i), height=50)

            with ui.HStack(width=ui.Fraction(2)):
                ui.Button("Right {}".format(i))
                ui.Button("Horizontal {}".format(i), width=ui.Fraction(2))

            ui.Spacer(width=50)
```


#### HStack:
it is a row of Widgets or Containers.

Here is an example that you can change the HStack spacing by a slider
```
from omni.ui import color as cl
SPACING = 5

def set_spacing(stack, spacing):
    stack.spacing = spacing

ui.Spacer(height=SPACING)
spacing_stack = ui.HStack(style={"margin": 0})
with spacing_stack:
    for name in ["One", "Two", "Three", "Four"]:
        ui.Button(name)

ui.Spacer(height=SPACING)
with ui.HStack(spacing=SPACING):
    with ui.HStack(width=100):
        ui.Spacer()
        ui.Label("spacing", width=0, name="text")
    with ui.HStack(width=ui.Percent(20)):
        field = ui.FloatField(width=50)
        slider = ui.FloatSlider(min=0, max=50, style={"color": cl.transparent})
        # Link them together
        slider.model = field.model
        slider.model.add_value_changed_fn(
            lambda m, s=spacing_stack: set_spacing(s, m.get_value_as_float()))
```

#### VStack:
it is a column of Widgets or Containers use a default spacing of 5

#### ZStack:
it is used to build Widgets or Containers on top of each other, for example to add a background to a Label

### Order in Stack and use of content_clipping
Due to Imgui, ScrollingFrame and CanvasFrame will create a new window, meaning if we have them in a ZStack, they don't respect the Stack order. To fix that we need to create a separate window, with the widget wrapped in a `ui.Frame(separate_window=True)` will fix the order issue. And if we also want the mouse input in the new separate window, we use `ui.HStack(content_clipping=True)` for that.

In the following example, you won't see the red rectangle.

```
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

With the use of `separate_window=True` or `content_clipping=True`, you will see the red rectangle.

```
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    with ui.Frame(separate_window=True):
        ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

```
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    with ui.HStack(content_clipping=True):
        ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

In the following example, you will see the button click action is captured on Button 1.
```
from functools import partial

def clicked(name):
	print(f'clicked {name}')

with ui.ZStack():
    b1 = ui.Button('Button 1')
    b1.set_clicked_fn(partial(clicked, b1.text))
    b2 = ui.Button('Button 2')
    b2.set_clicked_fn(partial(clicked, b2.text))
```

With the use of `content_clipping=True`, you will see the button click action is now fixed and captured on Button 2.
```
from functools import partial

def clicked(name):
	print(f'clicked {name}')

with ui.ZStack():
    b1 = ui.Button('Button 1')
    b1.set_clicked_fn(partial(clicked, b1.text))
    with ui.VStack(content_clipping=1):
        b2 = ui.Button('Button 2')
        b2.set_clicked_fn(partial(clicked, b2.text))
```

### Grid
Grid is a container that arranges its child views in a grid. Depends on the direction the grid size grows with creating more children, we call it `VGrid` (grow in vertical direction) and `HGrid` (grow in horizontal direction) There is no ui.GridLayout.

### Window
Finally there is Window, it take a Title in the constructor as the first argument, and width and height.

Here is one examples of small UI Window using Widget and Containers

```
import omni.ui as ui

# creating the window
my_window = ui.Window("example", width=300, height=300)
with my_window.frame:
    with ui.VStack():
        ui.Label("Simple window example")
        ui.Spacer()
        with ui.HStack():
            ui.Button("Cancel")
            ui.Button("Apply")
```

You can also inherit from ui.Window to create a customized window and use `set_build_fn` to draw the content

```
class SimpleWindow(ui.Window):
    def __init__(self, title: str, **kwargs):
        super().__init__(title, **kwargs)
        self.frame.set_build_fn(self.build_ui)

    def build_ui(self):
        with ui.VStack():
            ui.Label("Simple window example")
```

For the output code, try not to create a window in a function.

When asked questions about Widgets or container just write the relevant code, don't create window or full stacks, focus on answering just the questions asked

for example

Questions: how to create a slider:
Answers:
you can create a FloatSlider like that:

```python
model = ui.SimpleFloatModel(0.5)
ui.FloatSlider(model, min=0.0, max=1.0)
```

Questions: show me how to use the Grid
Answers:
you can create a VGrid like that:

```python
with ui.VGrid(..):
    for i in range(10):
        ui.Button(f"{i}")
```

there is no need to "show" the window after it will be automatic

This is very important. The output code should never include window.show()

The Widgets added into the Window will always go to ui.Window().frame(). Never directly add to ui.Window().
write code like this

```python
window = ui.Window("name of the window")
with window.frame:  # never write `with window:`
    ui.Button("hello")
```

#### Show Window
Always instance the class with the window you generated. The window is visible until its instance is alive. Without class instance created, the window is not shown.

```python
import omni.ui as ui

class AwesomeWindow:
    def __init__(self):
        self._window = ui.Window("Awesome", width=300, height=100)

# Make sure the windows is created and shown
awesome = AwesomeWindow()
```

## Visibility
This property holds whether the shape, widget or container is visible. Invisible shape, widget or container is not rendered, and it doesn't take part in the layout. The layout skips it.

In the following example, click the button from one to five to hide itself. The `Visible all` button brings them all back.
```
def invisible(button):
    button.visible = False

def visible(buttons):
    for button in buttons:
        button.visible = True

buttons = []
with ui.HStack():
    for n in ["One", "Two", "Three", "Four", "Five"]:
        button = ui.Button(n, width=0)
        button.set_clicked_fn(lambda b=button: invisible(b))
        buttons.append(button)

    ui.Spacer()
    button = ui.Button("Visible all", width=0)
    button.set_clicked_fn(lambda b=buttons: visible(b))
```

## Length Units
The Framework UI offers several different units for expressing length: Pixel, Percent and Fraction. There is no restriction on where certain units should be used.
Different length units allows users to define the widgets accurate to exact pixel or proportional to the parent widget or siblings.

## Style
Styles defines the look. Each widget has its own style to be tweaked with based on their use cases and behaviors, while they also follow the same syntax rules.

omni.ui Style Sheet rules are almost identical to those of HTML CSS. It applies to the style of all omni ui elements.

Style sheets consist of a sequence of style rules. A style rule is made up of a selector and a declaration. The selector specifies which widgets are affected by the rule. The declaration specifies which style properties should be set on the widget. For example:

```
## Double comment means hide from shippet
from omni.ui import color as cl
##
with ui.VStack(width=0, style={"Button": {"background_color": cl("#097eff")}}):
    ui.Button("Style Example")
```
In the above style rule, `Button` is the `selector`, and `{"background_color": cl("#097eff")}` is the `declaration`. The rule specifies that Button should use blue as its background color.

There are three types of selectors: Type Selector, Name Selector and State Selector They are structured as:

** Type Selector :: Name Selector : State Selector **

e.g., `Button::okButton:hovered`, where `Button` is the `Type Selector`, which matches the ui.Button's type, meaning we are setting style for ui.Button type. `okButton` is the` Name Selector`, which means we are setting style for1 all Button instances whose name is `okButton`. They are separated using the type selector with `::`. `hovered` is the `State Selector`, which means we are setting style for buttons whose state are hovered. It separates from the other selectors with `:`.

When type, name and state selector are used together, it defines the style of all Button typed instances named as `okButton` and in hovered, while `Button:hovered` defines the style of all Button typed instances which are in hovered states.

These are the only `State Selector` recognized by omni.ui:
* hovered : the mouse in the widget area
* pressed : the mouse is pressing in the widget area
* selected : the widget is selected
* disabled : the widget is disabled
* checked : the widget is checked
* drop : the rectangle is accepting a drop. For example,
style = {"Rectangle:drop" :  {"background_color": cl.blue}} meaning if the drop is acceptable, the rectangle is blue.
Please don't make up other `State Selector` except the above ones. For example, "hover" doesn't exist, use "hovered" instead.

Here is an example that a red circle becomes blue when hovered.
```
import omni.ui as ui
ui.Circle(
    width=100,
    height=100,
    style={"Circle": {"background_color": cl.red}, "Circle:hovered": {"background_color": cl.blue}}
```

### Omit the selector
It's possible to omit the selector and override the property in all the widget types.

In this example, the style is set to VStack. The style will be propagated to all the widgets in VStack including VStack itself. Since only `background_color` is in the style, only the widgets which have `background_color` in the style properties will have the background color set. For VStack and Label which don't have `background_color`, the style is ignored. Button and FloatField get the blue background color.

```
from omni.ui import color as cl
with ui.VStack(width=400, style={"background_color": cl("#097eff")}, spacing=5):
    ui.Button("One")
    ui.Button("Two")
    ui.FloatField()
    ui.Label("Label doesn't have background_color style")
```

### Style overridden with name and state selector
In the following example, we set the "Button" style for all the buttons, then override different buttons with name selector style, e.g. "Button::one" and "Button::two". Furthermore, we also set different style for Button::one when pressed or hovered, e.g. "Button::one:hovered" and "Button::one:pressed", which will override the style of the Buttons which are named `one` when they are pressed or hovered.

```
from omni.ui import color as cl
style1 = {
    "Button": {"border_width": 0.5, "border_radius": 0.0, "margin": 5.0, "padding": 5.0},
    "Button::one": {
        "background_color": cl("#097eff"),
        "background_gradient_color": cl("#6db2fa"),
        "border_color": cl("#1d76fd"),
    },
    "Button.Label::one": {"color": cl.white},
    "Button::one:hovered": {"background_color": cl("#006eff"), "background_gradient_color": cl("#5aaeff")},
    "Button::one:pressed": {"background_color": cl("#6db2fa"), "background_gradient_color": cl("#097eff")},
    "Button::two": {"background_color": cl.white, "border_color": cl("#B1B1B1")},
    "Button.Label::two": {"color": cl("#272727")},
    "Button::three:hovered": {
        "background_color": cl("#006eff"),
        "background_gradient_color": cl("#5aaeff"),
        "border_color": cl("#1d76fd"),
    },
    "Button::four:pressed": {
        "background_color": cl("#6db2fa"),
        "background_gradient_color": cl("#097eff"),
        "border_color": cl("#1d76fd"),
    },
}

with ui.HStack(style=style1):
    ui.Button("One", name="one")
    ui.Button("Two", name="two")
    ui.Button("Three", name="three")
    ui.Button("Four", name="four")
    ui.Button("Five", name="five")
```

### Style override to different levels of the widgets
It's possible to assign any style override to any level of the widgets. It can be assigned to both parents and children at the same time.

In this example, we have style_system which will be propagated to all buttons, but buttons with its own style will override the style_system.

```
from omni.ui import color as cl
style_system = {
    "Button": {
        "background_color": cl("#E1E1E1"),
        "border_color": cl("#ADADAD"),
        "border_width": 0.5,
        "border_radius": 3.0,
        "margin": 5.0,
        "padding": 5.0,
    },
    "Button.Label": {
        "color": cl.black,
    },
    "Button:hovered": {
        "background_color": cl("#e5f1fb"),
        "border_color": cl("#0078d7"),
    },
    "Button:pressed": {
        "background_color": cl("#cce4f7"),
        "border_color": cl("#005499"),
        "border_width": 1.0
    },
}

with ui.HStack(style=style_system):
    ui.Button("One")
    ui.Button("Two", style={"color": cl("#AAAAAA")})
    ui.Button("Three", style={"background_color": cl("#097eff"), "background_gradient_color": cl("#6db2fa")})
    ui.Button(
        "Four", style={":hovered": {"background_color": cl("#006eff"), "background_gradient_color": cl("#5aaeff")}}
    )
    ui.Button(
        "Five",
        style={"Button:pressed": {"background_color": cl("#6db2fa"), "background_gradient_color": cl("#097eff")}},
    )
```

### Customize the selector type using style_type_name_override
What if the user has a customized widget which is not a standard omni.ui one. How to define that Type Selector? In this case, We can use `style_type_name_override` to override the type name. `name` attribute is the Name Selector and State Selector can be added as usual.

Another use case is when we have a giant list of the same typed widgets, for example `Button`, but some of the Buttons are in the main window, and some of the Buttons are in the pop-up window, which we want to differentiate for easy look-up. Instead of calling all of them the same Type Selector as `Button` and only have different Name Selectors, we can override the type name for the main window buttons as `WindowButton` and the pop-up window buttons as `PopupButton`. This groups the style-sheet into categories and makes the change of the look or debug much easier.

Here is an example where we use `style_type_name_override` to override the style type name.

```
from omni.ui import color as cl
style={
    "WindowButton::one": {"background_color": cl("#006eff")},
    "WindowButton::one:hovered": {"background_color": cl("#006eff"), "background_gradient_color": cl("#FFAEFF")},
    "PopupButton::two": {"background_color": cl("#6db2fa")},
    "PopupButton::two:hovered": {"background_color": cl("#6db2fa"), "background_gradient_color": cl("#097eff")},
    }

with ui.HStack(width=400, style=style, spacing=5):
    ui.Button("Open", style_type_name_override="WindowButton", name="one")
    ui.Button("Save", style_type_name_override="PopupButton", name="two")
```

### Default style override
From the above examples, we know that if we want to propagate the style to all children, we just need to set the style to the parent widget, but this rule doesn't apply to windows. The style set to the window will not propagate to its widgets. If we want to propagate the style to ui.Window and their widgets, we should set the default style with `ui.style.default`.

```python
from omni.ui import color as cl
ui.style.default = {
    "background_color": cl.blue,
    "border_radius": 10,
    "border_width": 5,
    "border_color": cl.red,
}
```

## Shades
Shades are used to have multiple named color palettes with the ability for runtime switch. For example, one App could have several ui themes users can switch during using the App, e.g. dark themed ui and light themed ui.

The shade can be defined with the following code:

```python
    from omni.ui import color as cl
    cl.shade(cl("#FF6600"), red=cl("#0000FF"), green=cl("#66FF00"))
```

It can be assigned to the color style. It's possible to switch the color with the following command globally:

```python
    from omni.ui import color as cl
    cl.set_shade("red")
```

### Example
```
from omni.ui import color as cl
from omni.ui import constant as fl
from functools import partial

def set_color(color):
    cl.example_color = color

def set_width(value):
    fl.example_width = value

cl.example_color = cl.green
fl.example_width = 1.0

with ui.HStack(height=100, spacing=5):
    with ui.ZStack():
        ui.Rectangle(
            style={
                "background_color": cl.shade(
                    "aqua",
                    orange=cl.orange,
                    another=cl.example_color,
                    transparent=cl(0, 0, 0, 0),
                    black=cl.black,
                ),
                "border_width": fl.shade(1, orange=4, another=8),
                "border_radius": fl.one,
                "border_color": cl.black,
            },
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.Rectangle(\n"
                "\tstyle={\n"
                '\t\t"background_color":\n'
                "\t\t\tcl.shade(\n"
                '\t\t\t\t"aqua",\n'
                "\t\t\t\torange=cl(1, 0.5, 0),\n"
                "\t\t\t\tanother=cl.example_color),\n"
                '\t\t"border_width":\n'
                "\t\t\tfl.shade(1, orange=4, another=8)})",
                style={"color": cl.black, "margin": 15},
                width=0,
            )
            ui.Spacer()

    with ui.ZStack():
        ui.Rectangle(
            style={
                "background_color": cl.example_color,
                "border_width": fl.example_width,
                "border_radius": fl.one,
                "border_color": cl.black,
            }
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.Rectangle(\n"
                "\tstyle={\n"
                '\t\t"background_color": cl.example_color,\n'
                '\t\t"border_width": fl.example_width)})',
                style={"color": cl.black, "margin": 15},
                width=0,
            )
            ui.Spacer()

with ui.VStack(style={"Button": {"background_color": cl("097EFF")}}):
    ui.Label("Click the following buttons to change the shader of the left rectangle")
    with ui.HStack():
        ui.Button("cl.set_shade()", clicked_fn=partial(cl.set_shade, ""))
        ui.Button('cl.set_shade("orange")', clicked_fn=partial(cl.set_shade, "orange"))
        ui.Button('cl.set_shade("another")', clicked_fn=partial(cl.set_shade, "another"))
    ui.Label("Click the following buttons to change the border width of the right rectangle")
    with ui.HStack():
        ui.Button("fl.example_width = 1", clicked_fn=partial(set_width, 1))
        ui.Button("fl.example_width = 4", clicked_fn=partial(set_width, 4))
    ui.Label("Click the following buttons to change the background color of both rectangles")
    with ui.HStack():
        ui.Button('cl.example_color = "green"', clicked_fn=partial(set_color, "green"))
        ui.Button("cl.example_color = cl(0.8)", clicked_fn=partial(set_color, cl(0.8)))
    ## Double comment means hide from shippet
    ui.Spacer(height=15)
    ##
```

### URL Shades Example
It's also possible to use shades for specifying shortcuts to the images and style-based paths.

```
from omni.ui import color as cl
from omni.ui.url_utils import url
from functools import partial

def set_url(url_path: str):
    url.example_url = url_path

walk = "resources/icons/Nav_Walkmode.png"
fly = "resources/icons/Nav_Flymode.png"

url.example_url = walk

with ui.HStack(height=100, spacing=5):
    with ui.ZStack():
        ui.Image(height=100, style={"image_url": url.example_url})
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                'ui.Image(\n\tstyle={"image_url": cl.example_url})\n',
                style={"color": cl.black, "font_size": 12, "margin": 15},
                width=0,
            )
            ui.Spacer()
    with ui.ZStack():
        ui.ImageWithProvider(
            height=100,
            style={
                "image_url": url.shade(
                    "resources/icons/Move_local_64.png",
                    another="resources/icons/Move_64.png",
                    orange="resources/icons/Rotate_local_64.png",
                )
            }
        )
        with ui.HStack():
            ui.Spacer()
            ui.Label(
                "ui.ImageWithProvider(\n"
                "\tstyle={\n"
                '\t\t"image_url":\n'
                "\t\t\tst.shade(\n"
                '\t\t\t\t"Move_local_64.png",\n'
                '\t\t\t\tanother="Move_64.png")})\n',
                style={"color": cl.black, "font_size": 12, "margin": 15},
                width=0,
            )
            ui.Spacer()

with ui.HStack():
    # buttons to change the url for the image
    with ui.VStack():
        ui.Button("url.example_url = Nav_Walkmode.png", clicked_fn=partial(set_url, walk))
        ui.Button("url.example_url = Nav_Flymode.png", clicked_fn=partial(set_url, fly))
    # buttons to switch between shades to a different image
    with ui.VStack():
        ui.Button("ui.set_shade()", clicked_fn=partial(ui.set_shade, ""))
        ui.Button('ui.set_shade("another")', clicked_fn=partial(ui.set_shade, "another"))
```


### Fonts
Omni.ui also supports different font styles and sizes.
#### Font style
It's possible to set different font types with the style. The style key 'font' should point to the font file, which allows packaging of the font to the extension. We support both TTF and OTF formats. All text-based widgets support custom fonts.

```
with ui.VStack():
    ui.Label("Omniverse", style={"font":"${fonts}/OpenSans-SemiBold.ttf", "font_size": 40.0})
    ui.Label("Omniverse", style={"font":"${fonts}/roboto_medium.ttf", "font_size": 40.0})
```

#### Font size
It's possible to set the font size with the style.

Drag the following slider to change the size of the text.

```
## Double comment means hide from snippet
from functools import partial
##
def value_changed(label, value):
    label.style = {"color": ui.color(0), "font_size": value.as_float}

slider = ui.FloatSlider(min=1.0, max=150.0)
slider.model.as_float = 10.0
label = ui.Label("Omniverse", style={"color": ui.color(0), "font_size": 7.0})
slider.model.add_value_changed_fn(partial(value_changed, label))
## Double comment means hide from snippet
ui.Spacer(height=30)
##
```

## Model
MDV (Model-Delegate-View) is a pattern commonly used in `omni.ui` to implement
user interfaces with data-controlling logic. It highlights a separation between
the data and the display logic. This separation of concepts provides for better
maintenance of the code.

It closely follows the MVC pattern with a slightly different separation of
components. Unlike MVC, the View component of MDV takes responsibility of
Controller, takes control of the layout, and routes the Delegate component that
controls the look.

The three parts of the MDV software-design pattern can be described as follows:

 1. Model: The central component of the system. Manages data and logic. It
creates items that are used as pointers to the specific parts of the data
layer. The omni.ui defined omni.ui.Widget classes are inherited from either
ValueModelHelper or ItemModelHelper to get/set the model for the widgets.
The models of are inherited from abstract classes, either AbstractValueModel
or AbstractItemModel respectively.

For AbstractItemModel, its item children are a list of items inheriting from
`omni.ui.AbstractItem`, with each item's data being managed by `omni.ui.AbstractValueModel`.

Some of the omni.ui.Widget have built-in models. For example, omni.ui.ToolButton, omni.ui.CheckBox, omni.ui.ProgressBar, all the Fields (e.g. omni.ui.StringField), Sliders (e.g. omni.ui.IntSlider) and Drags (e.g. omni.ui.FloatDrag), they have model based on omni.ui.AbstractValueModel; while omni.ui.ColorWidget, omni.ui.ComboBox, omni.ui.TreeView, all the MultiField and MultiDrag (e.g. omni.ui.MultiFloatDragField), they have model based on omni.ui.AbstractItemModel. However, the widget itself only keep a weak reference to the model instance. Therefore, if we want to pass a customized model to these widgets, we need to retain the model from the widget. Otherwise the model will be garbage collected and the widget will not work.
This is a common code issue that appears as a "Tried to call pure virtual function" error. This is not because the model is implemented incorrectly, but rather the model has not been retained. The solution is to store the model in a variable.

For example, to fix the following code:
```
tree_view = ui.TreeView(model=CustomModel(), delegate=CustomDelegate())
```

we want to do something like this:

```
model = CustomModel()
delegate = CustomDelegate()
tree_view = ui.TreeView(model=model, delegate=delegate)
```

Notice the above example, it is the same issue for the delegate. The delegate should also be retained in a variable.

 2. Delegate: Creates widgets and defines the look.

 3. View: Handles layout, holds the widgets, and coordinates Model and Delegate.

Widget classes using the model-delegate-view pattern don't keep the data. However,
you can use the callbacks of the model to track or modify the state of the widget.

Here is a minimal example of callbacks.
When you start editing the field or slider, you will see "Editing is started",
and when you finish editing by press `enter`, you will see "Editing is finished".

```execute 200
def on_value(label):
    label.text = "Value is changed"

def on_begin(label):
    label.text = "Editing is started"

def on_end(label):
    label.text = "Editing is finished"

label = ui.Label("Nothing happened", name="text")
model = ui.StringField().model
model.add_value_changed_fn(lambda m, l=label: on_value(l))
model.add_begin_edit_fn(lambda m, l=label: on_begin(l))
model.add_end_edit_fn(lambda m, l=label: on_end(l))
```

Note that apart from field widgets, ColorWidget could also take the begin/end edit functions to trigger a callback when opened or closed.

For item model, there is no value changed callback, but instead there is item changed callback. In the example, we
initialize a combo box with default as index 1 which is "Cat", and also be aware that the item changed callback takes
model and item as arguments.

```
model = ui.ComboBox(1, "Dog", "Cat", "Pig").model
def combo_changed(model, item):
    print("Selected Index:", model.get_item_value_model(item).get_value_as_int())
model.add_item_changed_fn(combo_changed)
```

There are 2 ways to maintain the lifetime of model callbacks:

Add and Remove the callback. The add_xxx_fn method returns the callback ID, which can be passed to remove_xxx_fn method when
the callback is no longer needed
```
model = ui.IntDrag(width=50, min=0, max=1000).model
value_changed_cb = model.add_value_changed_fn(lambda m: print("Value changed"))

# Do something...

# Cleanup
model.remove_value_changed_fn(value_changed_cb)
```

Here is a list of add and remove callbacks for omni.ui.AbstractValueModel:
- add_value_changed_fn and remove_value_changed_fn
- add_begin_edit_fn and remove_begin_edit_fn
- add_end_edit_fn and remove_end_edit_fn

For omni.ui.AbstractItemModel the callbacks are:
- add_item_changed_fn and remove_item_changed_fn
- add_begin_edit_fn and remove_begin_edit_fn
- add_end_edit_fn and remove_end_edit_fn

The other way is subscription. The subscribe_xxx_fn method will return a subscription object. When the callback is not needed anymore,
clean up the subscription.
```
model = ui.IntDrag(width=50, min=0, max=1000).model
value_changed_sub = model.subscsribe_value_changed_fn(lambda m: print("Value changed"))

# Do something...

# Cleanup
value_changed_sub = None
```

Here is a list of subscription callbacks for omni.ui.AbstractValueModel:
- subscribe_value_changed_fn
- subscribe_begin_edit_fn
- subscribe_end_edit_fn

For omni.ui.AbstractItemModel the callbacks are:
- subscribe_item_changed_fn
- subscribe_begin_edit_fn
- subscribe_end_edit_fn

## Callbacks
When writing the code with callbacks, always call the callbacks as a test in the code to make sure there are no errors and callbacks can be executed.

```
import omni.ui as ui
import carb

def clicked():
    carb.log_warn('a was clicked')

window = ui.Window("some name", width=300, height=300)
with window.frame:
    ui.Button("a", clicked_fn=clicked)

# execute callback to know early it can be executed and it doesn't have errors
clicked()
```

## Drag & Drop

A drag and drop operation consists of three parts: the drag, accept the drop, and the drop. For drag,
Widget has a callback `drag_fn`. By adding this function to a widget, you set the callback attached
to the drag operation. The function should return the string data to drag.

To accept the drop Widget has a callback `accept_drop_fn`. It's called when the mouse enters the widget
area. It takes the drag string data and should return either the widget can accept this data for the
drop.

For the drop part, Widget has a callback `drop_fn` with a drop event argument `omni.ui.WidgetMouseDropEvent`
which the dropped string and x, y position could be found in properties `mime_data`, `x` and `y` respectively.
The callback is called when the user drops the string data to the widget.

```
def drag_area(url):
    """Create a draggable image."""
    def drag():
        """Called for the drag operation. Returns drag data."""
        return url

    image = ui.Image(url, width=64, height=64)
    image.set_drag_fn(lambda: drag(url))

def drop_area():
    """A drop area that shows image when drop."""
    with ui.ZStack(width=64, height=64):
        ui.Rectangle()
        image = ui.Image()
        label = ui.Label("path/to/your/image")

    def drop_accept(url):
        """Called to check the drag data is acceptable."""
        return True

    def drop(event):
        """Called when dropping the data."""
        nonlocal label
        label.text = event.mime_data

    image.set_accept_drop_fn(drop_accept)
    image.set_drop_fn(drop)

with ui.HStack():
    drag_area("resources/icons/Nav_Flymode.png")
    drag_area("resources/icons/Move_64.png")
    ui.Spacer(width=64)
    drop_area()
```

### Styling and Tooltips

It's possible to customize the drag tooltip with creating widgets in drag_fn.

To show the drop widget accepts drops visually, there is a style 'drop', and it's propagated to the
children widgets.
```
def drag_area(url):
    def drag():
        # Draw the image and the image name in the drag tooltip
        with ui.VStack():
            ui.Image(url, width=32, height=32)
            ui.Label(url)
        return url

    image = ui.Image(url, width=64, height=64)
    image.set_drag_fn(lambda: drag(url))

def drop_area(ext):
    # If drop is acceptable, the rectangle is blue
    style = {}
    style["Rectangle"] = {"background_color": cl("#999999")}
    style["Rectangle:drop"] = {"background_color": cl("#004499")}

    stack = ui.ZStack(width=64, height=64)
    with stack:
        ui.Rectangle(style=style)
        ui.Label(f"Accepts {ext}")
        image = ui.Image(style={"margin": 2})
        label = ui.Label("path/to/your/image")

    def drop_accept(url):
        # Accept drops of specific extension only
        return url.endswith(ext)

    def drop(event):
        nonlocal label
        label.text = event.mime_data

    stack.set_accept_drop_fn(lambda: drop_accept(ext))
    stack.set_drop_fn(drop)

with ui.HStack():
    drag_area("resources/icons/sound.png")
    drag_area("resources/icons/stage.png")
    drag_area("resources/glyphs/menu_audio.svg")
    drag_area("resources/glyphs/menu_camera.svg")
    ui.Spacer(width=64)
    drop_area(".png")
    ui.Spacer(width=64)
    drop_area(".svg")
```