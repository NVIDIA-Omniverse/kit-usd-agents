# omni.ui.ColorWidget

The ColorWidget is a button that displays the color from the item model and can open a picker window. The color dialog's function is to allow users to choose color.

Here is a list of styles you can customize on ColorWidget:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary

> background_color (color): the background color of the tooltip widget when hover over onto the ColorWidget
> color (color): the text color of the tooltip widget when hover over onto the ColorWidget

ColorWidget's model (omni.ui.SimpleListModel) can contain 4 or 3 items with each representing the corresponding RGBA or RGB color channel, to change the color to green, you can do:
```
import omni.ui as ui

cw = ui.ColorWidget()
for item, value in zip(cw.model.get_item_children(), (0, 1, 0, 0)):
    cw.model.get_item_value_model(item).set_value(value)
```

Here is an example of a ColorWidget with three FloatFields. The ColorWidget model is shared with the FloatFields so that users can click and edit the field value to change the ColorWidget's color, and the value change of the ColorWidget will also reflect in the value change of the FloatFields.

```
from omni.ui import color as cl
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0, style={"ColorWidget":{
        "border_width": 2,
        "border_color": cl.white,
        "border_radius": 4,
        "color": cl.pink,
        "margin": 2
    }}).model
    for item in color_model.get_item_children():
        component = color_model.get_item_value_model(item)
        ui.FloatField(component)
```

Here is an example of a ColorWidget with three FloatDrags. The ColorWidget model is shared with the FloatDrags so that users can drag the field value to change the color, and the value change of the ColorWidget will also reflect in the value change of the FloatDrags.

```
from omni.ui import color as cl
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(0.125, 0.25, 0.5, width=0, height=0, style={
        "background_color": cl.pink
    }).model
    for item in color_model.get_item_children():
        component = color_model.get_item_value_model(item)
        ui.FloatDrag(component, min=0, max=1)
```

Here is an example of a ColorWidget with a ComboBox. The ColorWidget model is shared with the ComboBox. Only the value change of the ColorWidget will reflect in the value change of the ComboBox.

```
with ui.HStack(spacing=5):
    color_model = ui.ColorWidget(width=0, height=0).model
    ui.ComboBox(color_model)
```

Here is an interactive example with USD. You can create a Mesh in the Stage. Choose `Pixar Storm` as the render. Select the mesh and use this ColorWidget to change the color of the mesh. You can use `Ctrl+z` for undoing and `Ctrl+y` for redoing.

```
import weakref
import omni.kit.commands
from omni.usd.commands import UsdStageHelper
from pxr import UsdGeom
from pxr import Gf
import omni.usd

class SetDisplayColorCommand(omni.kit.commands.Command, UsdStageHelper):
    """
    Change prim display color undoable **Command**. Unlike ChangePropertyCommand, it can undo property creation.

    Args:
        gprim (Gprim): Prim to change display color on.
        value: Value to change to.
        value: Value to undo to.
    """

    def __init__(self, gprim: UsdGeom.Gprim, color: any, prev: any):
        self._gprim = gprim
        self._color = color
        self._prev = prev

    def do(self):
        color_attr = self._gprim.CreateDisplayColorAttr()
        color_attr.Set([self._color])

    def undo(self):
        color_attr = self._gprim.GetDisplayColorAttr()
        if self._prev is None:
            color_attr.Clear()
        else:
            color_attr.Set([self._prev])

omni.kit.commands.register(SetDisplayColorCommand)

class FloatModel(ui.SimpleFloatModel):
    def __init__(self, parent):
        super().__init__()
        self._parent = weakref.ref(parent)

    def begin_edit(self):
        parent = self._parent()
        parent.begin_edit(None)

    def end_edit(self):
        parent = self._parent()
        parent.end_edit(None)


class USDColorItem(ui.AbstractItem):
    def __init__(self, model):
        super().__init__()
        self.model = model


class USDColorModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()

        # Create root model
        self._root_model = ui.SimpleIntModel()
        self._root_model.add_value_changed_fn(lambda a: self._item_changed(None))

        # Create three models per component
        self._items = [USDColorItem(FloatModel(self)) for i in range(3)]
        for item in self._items:
            item.model.add_value_changed_fn(lambda a, item=item: self._on_value_changed(item))

        # Omniverse contexts
        self._usd_context = omni.usd.get_context()
        self._selection = self._usd_context.get_selection()
        self._events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = self._events.create_subscription_to_pop(
            self._on_stage_event, name="omni.example.ui ColorWidget stage update"
        )

        # Privates
        self._subscription = None
        self._gprim = None
        self._prev_color = None
        self._edit_mode_counter = 0

    def _on_stage_event(self, event):
        """Called with subscription to pop"""
        if event.type == int(omni.usd.StageEventType.SELECTION_CHANGED):
            self._on_selection_changed()

    def _on_selection_changed(self):
        """Called when the user changes the selection"""
        selection = self._selection.get_selected_prim_paths()
        stage = self._usd_context.get_stage()
        self._subscription = None
        self._gprim = None

        # When TC runs tests, it's possible that stage is None
        if selection and stage:
            self._gprim = UsdGeom.Gprim.Get(stage, selection[0])
            if self._gprim:
                color_attr = self._gprim.GetDisplayColorAttr()
                usd_watcher = omni.usd.get_watcher()
                self._subscription = usd_watcher.subscribe_to_change_info_path(
                    color_attr.GetPath(), self._on_usd_changed
                )

        # Change the widget color
        self._on_usd_changed()

    def _on_value_changed(self, item):
        """Called when the submodel is changed"""
        if not self._gprim:
            return

        if self._edit_mode_counter > 0:
            # Change USD only if we are in edit mode.
            color_attr = self._gprim.CreateDisplayColorAttr()

            color = Gf.Vec3f(
                self._items[0].model.get_value_as_float(),
                self._items[1].model.get_value_as_float(),
                self._items[2].model.get_value_as_float(),
            )

            color_attr.Set([color])

        self._item_changed(item)

    def _on_usd_changed(self, path=None):
        """Called with UsdWatcher when something in USD is changed"""
        color = self._get_current_color() or Gf.Vec3f(0.0)

        for i in range(len(self._items)):
            self._items[i].model.set_value(color[i])

    def _get_current_color(self):
        """Returns color of the current object"""
        if self._gprim:
            color_attr = self._gprim.GetDisplayColorAttr()
            if color_attr:
                color_array = color_attr.Get()
                if color_array:
                    return color_array[0]

    def get_item_children(self, item):
        """Reimplemented from the base class"""
        return self._items

    def get_item_value_model(self, item, column_id):
        """Reimplemented from the base class"""
        if item is None:
            return self._root_model
        return item.model

    def begin_edit(self, item):
        """
        Reimplemented from the base class.
        Called when the user starts editing.
        """
        if self._edit_mode_counter == 0:
            self._prev_color = self._get_current_color()

        self._edit_mode_counter += 1

    def end_edit(self, item):
        """
        Reimplemented from the base class.
        Called when the user finishes editing.
        """
        self._edit_mode_counter -= 1

        if not self._gprim or self._edit_mode_counter > 0:
            return

        color = Gf.Vec3f(
            self._items[0].model.get_value_as_float(),
            self._items[1].model.get_value_as_float(),
            self._items[2].model.get_value_as_float(),
        )

        omni.kit.commands.execute("SetDisplayColor", gprim=self._gprim, color=color, prev=self._prev_color)


with ui.HStack(spacing=5):
    ui.ColorWidget(USDColorModel(), width=0)
    ui.Label("Interactive ColorWidget with USD", name="text")
```

