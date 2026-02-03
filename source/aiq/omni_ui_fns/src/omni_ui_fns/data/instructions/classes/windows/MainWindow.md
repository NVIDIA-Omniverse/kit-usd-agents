# omni.ui.MainWindow

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


