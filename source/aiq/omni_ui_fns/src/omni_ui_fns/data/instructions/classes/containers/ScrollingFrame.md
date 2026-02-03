# omni.ui.ScrollingFrame

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

