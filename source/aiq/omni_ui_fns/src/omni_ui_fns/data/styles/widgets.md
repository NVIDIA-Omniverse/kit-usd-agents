# Widgets

## Label
Labels are used everywhere in omni.ui. They are text only objects.

Here is a list of styles you can customize on Label:
> color (color): the color of the text
> font_size (float): the size of the text
> margin (float): the distance between the label and the parent widget defined boundary
> margin_width (float): the width distance between the label and the parent widget defined boundary
> margin_height (float): the height distance between the label and the parent widget defined boundary
> alignment (enum): defines how the label is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Here are a few examples of labels:

```execute 200
from omni.ui import color as cl
ui.Label("this is a simple label", style={"color":cl.red, "margin": 5})
```

```execute 200
from omni.ui import color as cl
ui.Label("label with alignment", style={"color":cl.green, "margin": 5}, alignment=ui.Alignment.CENTER)
```

Notice that alignment could be either a property or a style.
```execute 200
from omni.ui import color as cl
label_style = {
    "Label": {"font_size": 20, "color": cl.blue, "alignment":ui.Alignment.RIGHT, "margin_height": 20}
    }
ui.Label("Label with style", style=label_style)
```

When the text of the Label is too long, it can be elided by `...`:
```execute 200
from omni.ui import color as cl
ui.Label(
            "Label can be elided: Lorem ipsum dolor "
            "sit amet, consectetur adipiscing elit, sed do "
            "eiusmod tempor incididunt ut labore et dolore "
            "magna aliqua. Ut enim ad minim veniam, quis "
            "nostrud exercitation ullamco laboris nisi ut "
            "aliquip ex ea commodo consequat. Duis aute irure "
            "dolor in reprehenderit in voluptate velit esse "
            "cillum dolore eu fugiat nulla pariatur. Excepteur "
            "sint occaecat cupidatat non proident, sunt in "
            "culpa qui officia deserunt mollit anim id est "
            "laborum.",
            style={"color":cl.white},
            elided_text=True,
        )
```

## CheckBox
A CheckBox is an option button that can be switched on (checked) or off (unchecked). Checkboxes are typically used to represent features in an application that can be enabled or disabled without affecting others.

The checkbox is implemented using the model-delegate-view pattern. The model is the central component of this system. It is the application's dynamic data structure independent of the widget. It directly manages the data, logic and rules of the checkbox. If the model is not specified, the simple one is created automatically when the object is constructed.

Here is a list of styles you can customize on Line:
> color (color): the color of the tick
> background_color (color): the background color of the check box
> font_size: the size of the tick
> border_radius (float): the radius of the corner angle if the user wants  to round the check box.
> border_width (float): the size of the check box border
> secondary_background_color (color): the color of the check box border

Default checkbox
```execute 200
with ui.HStack(width=0, spacing=5):
    ui.CheckBox().model.set_value(True)
    ui.CheckBox()
    ui.Label("Default")
```

Disabled checkbox:
```execute 200
with ui.HStack(width=0, spacing=5):
    ui.CheckBox(enabled=False).model.set_value(True)
    ui.CheckBox(enabled=False)
    ui.Label("Disabled")
```

In the following example, the models of two checkboxes are connected, and if one checkbox is changed, it makes another checkbox change as well.

```execute 200
from omni.ui import color as cl
with ui.HStack(width=0, spacing=5):
    # Create two checkboxes
    style = {"CheckBox":{
        "color": cl.white, "border_radius": 0, "background_color": cl("#ff5555"), "font_size": 30}}
    first = ui.CheckBox(style=style)
    second = ui.CheckBox(style=style)

    # Connect one to another
    first.model.add_value_changed_fn(lambda a, b=second: b.model.set_value(not a.get_value_as_bool()))
    second.model.add_value_changed_fn(lambda a, b=first: b.model.set_value(not a.get_value_as_bool()))

    # Set the first one to True
    first.model.set_value(True)

    ui.Label("One of two")
```

In the following example, that is a bit more complicated, only one checkbox can be enabled.
```execute 200
from omni.ui import color as cl
style = {"CheckBox":{
    "color": cl("#ff5555"), "border_radius": 5, "background_color": cl(0.35), "font_size": 20}}
with ui.HStack(width=0, spacing=5):
    # Create two checkboxes
    first = ui.CheckBox(style=style)
    second = ui.CheckBox(style=style)
    third = ui.CheckBox(style=style)

    def like_radio(model, first, second):
        """Turn on the model and turn off two checkboxes"""
        if model.get_value_as_bool():
            model.set_value(True)
            first.model.set_value(False)
            second.model.set_value(False)

    # Connect one to another
    first.model.add_value_changed_fn(lambda a, b=second, c=third: like_radio(a, b, c))
    second.model.add_value_changed_fn(lambda a, b=first, c=third: like_radio(a, b, c))
    third.model.add_value_changed_fn(lambda a, b=first, c=second: like_radio(a, b, c))

    # Set the first one to True
    first.model.set_value(True)

    ui.Label("Almost like radio box")
```

## ComboBox
The ComboBox widget is a combination of a button and a drop-down list. A ComboBox is a selection widget that displays the current item and can pop up a list of selectable items.

Here is a list of styles you can customize on ComboBox:
> color (color): the color of the combo box text and the arrow of the drop-down button
> background_color (color): the background color of the combo box
> secondary_color (color): the color of the drop-down button's background
> selected_color (color): the selected highlight color of option items
> secondary_selected_color (color): the color of the option item text
> font_size (float): the size of the text
> border_radius (float): the border radius if the user wants  to round the ComboBox
> padding (float): the overall padding of the ComboBox. If padding is defined, padding_height and padding_width will have no effects.
> padding_height (float): the width padding of the drop-down list
> padding_width (float): the height padding of the drop-down list
> secondary_padding (float): the height padding between the ComboBox and options

Default ComboBox:

```execute 200
ui.ComboBox(1, "Option 1", "Option 2", "Option 3")
```

ComboBox with style
```execute 200
from omni.ui import color as cl
style={"ComboBox":{
    "color": cl.red,
    "background_color": cl(0.15),
    "secondary_color": cl("#1111aa"),
    "selected_color": cl.green,
    "secondary_selected_color": cl.white,
    "font_size": 15,
    "border_radius": 20,
    "padding_height": 2,
    "padding_width": 20,
    "secondary_padding": 30,
}}
with ui.VStack():
    ui.ComboBox(1, "Option 1", "Option 2", "Option 3", style=style)
    ui.Spacer(height=20)
```


The following example demonstrates how to add items to the ComboBox.
```execute 200
editable_combo = ui.ComboBox()
ui.Button(
    "Add item to combo",
    clicked_fn=lambda m=editable_combo.model: m.append_child_item(
        None, ui.SimpleStringModel("Hello World")),
)
```

The minimal model implementation to have more flexibility of the data. It requires holding the value models and reimplementing two methods: `get_item_children` and `get_item_value_model`.
```execute 200
class MinimalItem(ui.AbstractItem):
    def __init__(self, text):
        super().__init__()
        self.model = ui.SimpleStringModel(text)

class MinimalModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()

        self._current_index = ui.SimpleIntModel()
        self._current_index.add_value_changed_fn(
            lambda a: self._item_changed(None))

        self._items = [
            MinimalItem(text)
            for text in ["Option 1", "Option 2"]
        ]

    def get_item_children(self, item):
        return self._items

    def get_item_value_model(self, item, column_id):
        if item is None:
            return self._current_index
        return item.model

self._minimal_model = MinimalModel()
with ui.VStack():
    ui.ComboBox(self._minimal_model, style={"font_size": 22})
    ui.Spacer(height=10)
```

The example of communication between widgets. Type anything in the field and it will appear in the combo box.
```execute 200
editable_combo = None

class StringModel(ui.SimpleStringModel):
    '''
    String Model activated when editing is finished.
    Adds item to combo box.
    '''
    def __init__(self):
        super().__init__("")

    def end_edit(self):
        combo_model = editable_combo.model
        # Get all the options ad list of strings
        all_options = [
            combo_model.get_item_value_model(child).as_string
            for child in combo_model.get_item_children()
        ]

        # Get the current string of this model
        fieldString = self.as_string
        if fieldString:
            if fieldString in all_options:
                index = all_options.index(fieldString)
            else:
                # It's a new string in the combo box
                combo_model.append_child_item(
                    None,
                    ui.SimpleStringModel(fieldString)
                )
                index = len(all_options)

            combo_model.get_item_value_model().set_value(index)

self._field_model = StringModel()

def combo_changed(combo_model, item):
    all_options = [
        combo_model.get_item_value_model(child).as_string
        for child in combo_model.get_item_children()
    ]
    current_index = combo_model.get_item_value_model().as_int
    self._field_model.as_string = all_options[current_index]

with ui.HStack():
    ui.StringField(self._field_model)
    editable_combo = ui.ComboBox(width=0, arrow_only=True)
    editable_combo.model.add_item_changed_fn(combo_changed)
```

## TreeView
TreeView is a widget that presents a hierarchical view of information. Each item can have a number of subitems. An indentation often visualizes this in a list. An item can be expanded to reveal subitems, if any exist, and collapsed to hide subitems.

TreeView can be used in file manager applications, where it allows the user to navigate the file system directories. They are also used to present hierarchical data, such as the scene object hierarchy.

TreeView uses a model-delegate-view pattern to manage the relationship between data and the way it is presented. The separation of functionality gives developers greater flexibility to customize the presentation of items and provides a standard interface to allow a wide range of data sources to be used with other widgets.

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
```execute 200
import omni.ui as ui
from omni.ui import color as cl
style = {
    "TreeView":
    {
        "background_selected_color": cl("#55FF9033"),
        "secondary_color": cl.green,
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
```execute 200
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
```execute 200
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

```execute 200
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