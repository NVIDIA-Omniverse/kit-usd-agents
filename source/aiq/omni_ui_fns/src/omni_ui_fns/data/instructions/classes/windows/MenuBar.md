# omni.ui.MenuBar

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

