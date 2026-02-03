# omni.ui.TreeView

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

