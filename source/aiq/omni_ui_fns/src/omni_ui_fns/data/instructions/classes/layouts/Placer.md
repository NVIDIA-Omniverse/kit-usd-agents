# omni.ui.Placer

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


