# omni.ui.ComboBox

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

Default ComboBox with a default model, note that each item in the model saves the index number instead of the
value string:

```
# create a comboBox with three options 1, 2 and 3, and defaults to index 1 which is "Option 2"
combo_box = ui.ComboBox(1, "Option 1", "Option 2", "Option 3")
# set the combo box to index 2 which is "Option 3"
combo_box.model.get_item_value_model().set_value(2)
```

ComboBox with style
```
from omni.ui import color as cl
style={"ComboBox":{
    "color": cl.red,
    "background_color": cl(0.15),
    "secondary_color": cl("#1111aa"), # the drop-down button's background color is blue
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
```
editable_combo = ui.ComboBox()
ui.Button(
    "Add item to combo",
    clicked_fn=lambda m=editable_combo.model: m.append_child_item(
        None, ui.SimpleStringModel("Hello World")),
)
```

The minimal model implementation to have more flexibility of the data. It requires holding the value models and reimplementing two methods: `get_item_children` and `get_item_value_model`.
```
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
```
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


