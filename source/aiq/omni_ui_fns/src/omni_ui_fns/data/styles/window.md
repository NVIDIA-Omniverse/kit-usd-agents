# Window Widgets

## MainWindow
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

## Window
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

## Menu
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

### Separator
Separator is a type of MenuItem which creates a separator line in the UI elements.

From the above example, you can see the use of Separator in Menu.
Here is a list of styles you can customize on Separator:
> color (color): the color of the Separator

## MenuBar
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
