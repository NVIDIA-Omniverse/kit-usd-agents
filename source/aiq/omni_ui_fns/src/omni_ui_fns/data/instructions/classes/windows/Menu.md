# omni.ui.Menu

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

