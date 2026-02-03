# Container widgets
Container widgets are used for grouping items. It's possible to add children to the container with Python's `with` statement. It's not possible to reparent items. Instead, it's necessary to remove the item and recreate a similar item under another parent.

## Stack
We have three main components: VStack, HStack, and ZStack.

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

### HStack
This class is used to construct horizontal layout objects.

The simplest use of the class is like this:
```execute 200
with ui.HStack(style={"margin": 10}):
    ui.Button("One")
    ui.Button("Two")
    ui.Button("Three")
    ui.Button("Four")
    ui.Button("Five")
```
### VStack
The VStack class lines up widgets vertically.
```execute 200
with ui.VStack(width=100.0, style={"margin": 5}):
    with ui.VStack():
        ui.Button("One")
        ui.Button("Two")
        ui.Button("Three")
        ui.Button("Four")
        ui.Button("Five")
```

### ZStack
ZStack is a view that overlays its children, aligning them on top of each other. The later one is on top of the previous ones.
```execute 200
with ui.VStack(width=100.0, style={"margin": 5}):
    with ui.ZStack():
        ui.Button("Very Long Text to See How Big it Can Be", height=0)
        ui.Button("Another\nMultiline\nButton", width=0)
```

### Layout
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

### Spacing
Spacing is a property of Stack. It defines the non-stretchable space in pixels between child items of the layout.

Here is an example that you can change the HStack spacing by a slider
```execute 200
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

## Frame
Frame is a container that can keep only one child. Each child added to Frame overrides the previous one. This feature is used for creating dynamic layouts. The whole layout can be easily recreated with a simple callback.

Here is a list of styles you can customize on Frame:
> padding (float): the distance between the child widgets and the border of the button

In the following example, you can drag the IntDrag to change the slider value. The buttons are recreated each time the slider changes.
```execute 200
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
```execute 200
with ui.Frame(vertical_clipping=True, height=20):
    ui.Label("This should be clipped vertically. " * 10, word_wrap=True)
```

## CanvasFrame
CanvasFrame is the widget that allows the user to pan and zoom its children with a mouse. It has a layout that can be infinitely moved in any direction.

Here is a list of styles you can customize on CanvasFrame:
> background_color (color): the main color of the rectangle

Here is an example of a CanvasFrame, you can scroll the middle mouse to zoom the canvas and middle mouse move to pan in it (press CTRL to avoid scrolling the docs).

```execute 200
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

## ScrollingFrame
The ScrollingFrame class provides the ability to scroll onto other widgets. ScrollingFrame is used to display the contents of children widgets within a frame. If the widget exceeds the size of the frame, the frame can provide scroll bars so that the entire area of the child widget can be viewed by scrolling.

Here is a list of styles you can customize on ScrollingFrame:
> scrollbar_size (float): the width of the scroll bar
> secondary_color (color): the color the scroll bar
> background_color (color): the background color the scroll frame

Here is an example of a ScrollingFrame, you can scroll the middle mouse to scroll the frame.

```execute 200
from omni.ui import color as cl
with ui.HStack():
    left_frame = ui.ScrollingFrame(
        height=250,
        horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
        vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        style={"ScrollingFrame":{
            "scrollbar_size":10,
            "secondary_color": cl.red,
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
            "secondary_color": cl.blue,
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

## CollapsableFrame
CollapsableFrame is a frame widget that can hide or show its content. It has two states: expanded and collapsed. When it's collapsed, it looks like a button. If it's expanded, it looks like a button and a frame with the content. It's handy to group properties, and temporarily hide them to get more space for something else.

Here is a list of styles you can customize on Image:
> background_color (color): the background color of the CollapsableFrame widget
> secondary_color (color): the background color of the CollapsableFrame's header
> border_radius (float): the border radius if user wants to round the CollapsableFrame
> border_color (color): the border color if the CollapsableFrame has a border
> border_width (float): the border width if the CollapsableFrame has a border
> padding (float): the distance between the header or the content to the border of the CollapsableFrame
> margin (float): the distance between the CollapsableFrame and other widgets

Here is a default `CollapsableFrame` example:
```execute 200
with ui.CollapsableFrame("Header"):
    with ui.VStack(height=0):
        ui.Button("Hello World")
        ui.Button("Hello World")
```

It's possible to use a custom header.
```execute 200
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
        "secondary_color": cl("#CC211B"),
        "border_radius": 10,
        "border_color": cl.blue,
        "border_width": 2,
    },
    "CollapsableFrame:hovered": {"secondary_color": cl("#FF4321")},
    "CollapsableFrame:pressed": {"secondary_color": cl.red},
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
```execute 200
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

## Order in Stack and use of content_clipping
Due to Imgui, ScrollingFrame and CanvasFrame will create a new window, meaning if we have them in a ZStack, they don't respect the Stack order. To fix that we need to create a separate window, with the widget wrapped in a `ui.Frame(separate_window=True)` will fix the order issue. And if we also want the mouse input in the new separate window, we use `ui.HStack(content_clipping=True)` for that.

In the following example, you won't see the red rectangle.

```execute 200
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

With the use of `separate_window=True` or `content_clipping=True`, you will see the red rectangle.

```execute 200
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    with ui.Frame(separate_window=True):
        ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

```execute 200
from omni.ui import color as cl

with ui.ZStack():
    ui.Rectangle(width=200, height=200, style={'background_color':cl.green})
    with ui.CanvasFrame(width=150, height=150):
        ui.Rectangle(style={'background_color':cl.blue})
    with ui.HStack(content_clipping=True):
        ui.Rectangle(width=100, height=100, style={'background_color':cl.red})
```

In the following example, you will see the button click action is captured on Button 1.
```execute 200
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
```execute 200
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

## Grid
Grid is a container that arranges its child views in a grid. Depends on the direction the grid size grows with creating more children, we call it VGrid (grow in vertical direction) and HGrid (grow in horizontal direction)

There is currently no style you can customize on Grid.

### VGrid
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

### HGrid
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

## Placer
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
