# omni.ui.Window

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


