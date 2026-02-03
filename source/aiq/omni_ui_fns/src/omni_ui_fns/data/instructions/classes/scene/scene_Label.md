# omni.ui.scene.Label

Defines a standard label for user interface items. The text size is always in
the screen space and oriented to the camera. It supports `omni.ui` alignment.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
import math

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Label(
        "NVIDIA Omniverse",
        alignment=ui.Alignment.CENTER,
        color=cl("#76b900"),
        size=50
    )
```

## omni.kit.widget.filter.FilterButton

Allows users to access and toggle a set of predefined filter options presented in a dropdown menu. The button reflects the state of the filters, such as when there are unsaved changes, and provides callbacks for mouse interactions, making it interactive and responsive to user input.

To use a FilterButton, you must include ONLY ONE parameter: option_items, which is a list of OptionItems. No other types of options are allowed, this list is required, and no other parameters should be passed in.

The following example creates a UI Window with a FilterButton that displays a list of options of OptionItem type that includes filtering by audio, materials, scripts, textures, and usd:

```execute 150
import omni.ui as ui
from omni.kit.widget.options_menu import OptionItem
from omni.kit.widget.filter import FilterButton

# Create filter option items
option_items = [
    OptionItem("audio", text="Audio"),
    OptionItem("materials", text="Materials"),
    OptionItem("scripts", text="Scripts"),
    OptionItem("textures", text="Textures"),
    OptionItem("usd", text="USD"),
]

# Display the filter button in a UI window
window = ui.Window("Filter Example Window", width=200, height=200)
with window.frame:
    with ui.VStack():
        # Create a filter button with the given option items
        filter_button = FilterButton(option_items)

# Set first filter item on
model = filter_button.model
model.get_item_children()[0].value = True
```

## omni.kit.widget.option_menu.OptionsMenu

Represents a menu to show various options with a header and list of menu items. This is an example of creating an OptionsMenu with various items such as OptionItems, OptionRadios. OptionsSeparator, and OptionSeparator.

```execute 150
from omni.kit.widget.options_menu import OptionsMenu, OptionsModel, OptionItem, OptionSeparator, OptionRadios, OptionCustom
import omni.ui as ui

# Define a custom build function for the custom menu item
def custom_build_function():
    return ui.Label("Custom Option")

# Define radio options
radios = [
    ("Radios", None),
    "First",
    ("Second", "Second Radio"),
    "Third",
]

# Create an options model with items
options_model = OptionsModel(
    "Options",
    [
        OptionItem("audio", text="Audio"),
        OptionItem("materials", text="Materials"),
        OptionItem("scripts", text="Scripts"),
        OptionItem("textures", text="Textures"),
        OptionItem("usd", text="USD"),
        OptionRadios(radios, default="First"),
        OptionSeparator(),
        OptionCustom(build_fn=custom_build_function),
        OptionSeparator(),
        OptionItem("Disabled", enabled=False),
        OptionItem("Disabled and checked", default=True, enabled=False),
    ]
)

# Instantiate and display the options menu
options_menu = OptionsMenu(model=options_model)
options_menu.show_at(100, 100)  # Position where the menu will appear
```

## omni.kit.widget.option_menu.RadioMenu

Represents a menu specifically for radio button groups. This is an example of a RadioMenu that allows you to sort by Name, Date, Price, in ascending, descending, or random order. Note that RadioMenu uses RadioModel parameters of field_model and order_model.

```execute 150
from omni.kit.widget.options_menu import RadioMenu, RadioModel

field_model = RadioModel(["Name", "Date", "Price"], default_index=1)
order_model = RadioModel(["Ascending", "Descending", "Random"])

menu = RadioMenu("Sort By", [field_model, order_model])
menu.show_at(100, 100)
```

## omni.kit.widget.option_menu.OptionsModel

Represents a model for managing a collection of option items within a menu. This is an example of initializing an OptionsModel with various items like OptionItems for audio, materials, and scripts, as well as an OptionSeparator and OptionCustom.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionItem, OptionSeparator

model = OptionsModel(
    "Options",
    [
        OptionItem("Audio", text="Audio"),
        OptionItem("Materials", text="Materials"),
        OptionItem("Scripts", text="Scripts"),
        OptionSeparator(),
    ]
)
```

## omni.kit.widget.option_menu.RadioModel

Represents a model for managing a collection of option items within a menu. This example shows a RadioModel storing name, date, and price that can later be used in a RadioMenu.

```execute 150
from omni.kit.widget.options_menu import RadioModel

field_model = RadioModel(["Name", "Date", "Price"], default_index=1)
menu = RadioMenu("Sort By", [field_model])
menu.show_at(100, 100)
```

## omni.kit.widget.option_menu.OptionCustom

Represents a custom option item with a build function and optional model. This example shows creating a OptionCustom via a function `build_custom_menu_item` passed in as a parameter. This OptionCustom can be included in an OptionsModel.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionCustom

# Function to build custom menu items
def build_custom_menu_item():
    ui.Label("Custom Menu Item")

# Initialize the options model with various items
model = OptionsModel(
    "Custom Option",
    [
        OptionCustom(build_custom_menu_item)  # Using OptionCustom for custom build function
    ]
)
```

## omni.kit.widget.option_menu.OptionItem

Represents a general item for options menus, supporting features like checkability and visibility toggling.

The following are examples of how to initialize a list of OptionItem, where each OptionItem is any string name, followed by the text, default display, and whether the option is enabled or not.

```execute 150
from omni.kit.widget.options_menu import OptionItem

option_items = [
    OptionItem("audio", text="Audio"),
    OptionItem("materials", text="Materials"),
    OptionItem("scripts", text="Scripts"),
    OptionItem("textures", text="Textures"),
    OptionItem("usd", text="USD"),
    OptionItem("Disabled", enabled=False),
    OptionItem("Disabled and checked", default=True, enabled=False),
]
```

## omni.kit.widget.option_menu.OptionRadios

Represents a list of radio options within a menu. This is an example of creating an OptionsModel with an OptionRadios that is defined by the list `radios` and contains various radio field options.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionRadios

# Define radio options
radios = [
    ("Radios", None),
    "First",
    ("Second", "Second Radio"),
    "Third",
]

# Create an options model with items
options_model = OptionsModel(
    "Options with Radio",
    [
        OptionRadios(radios, default="First"),
    ]
)
```

## omni.kit.widget.option_menu.OptionSeparator

Represents a separator in menu items, optionally with a title. This is an example of using OptionSeparator to separate two OptionItem in an OptionsModel.

```execute 150
from omni.kit.widget.options_menu import OptionsModel, OptionItem, OptionSeparator

model = OptionsModel(
    "Options",
    [
        OptionItem("Audio", text="Audio"),
        OptionSeparator(),
        OptionItem("Materials", text="Materials"),
    ]
)
```

## omni.kit.widget.option_menu.OptionLabelMenuItemDelegate

A delegate for a normal menu item with additional spacing. This shows two use cases of OptionLabelMenuItemDelegate, where we pass a delegate for an OptionCustom of a ui.MenuItem.

```execute 150
from omni.kit.widget.options_menu import OptionCustom, OptionSeparator, OptionLabelMenuItemDelegate

option_items = [
    OptionCustom(build_fn=lambda: ui.MenuItem("Reload Outdated Primitives", delegate=OptionLabelMenuItemDelegate(), triggered_fn=self._on_reload_all_prims)),
    OptionSeparator(),
    OptionCustom(build_fn=lambda: ui.MenuItem("Reset", delegate=OptionLabelMenuItemDelegate(), triggered_fn=self._on_reset)),
]
```

## omni.kit.widget.searchfield.SearchField

A widget that represents a search field where users can input search words and receive suggestions.

This is an example of creating a general search field with a callback

```python
from typing import List, Optional
from omni.kit.widget.searchfield import SearchField
from omni import ui

def on_search(search_words: Optional[List[str]]):
    if search_words is not None:
        print(f"Now searching {search_words}")

search_field = SearchField(on_search_fn=on_search)
```

This is an example of creating a Search Field with Suggestions

```python
from omni.kit.widget.searchfield import SearchField
from omni import ui

search_field = SearchField(suggestions=["apple", "banana", "cherry"])
```

This is an example of creating a Search Field with and without Tokens.

The search field can be set to show or not show tokens. When `show_token` is set to `True`, each search term is rendered as a SearchWordButton, where you can click to remove some of the tokens from the search. When `show_token` is set to `False`, the search terms are rendered as plain text without tokenization. `show_token` is set to `True` by default.

```python
from omni.kit.widget.searchfield import SearchField
from omni import ui

search_field = SearchField(show_tokens=True)
search_field.search_words = ["show", "tokens"]

search_field_no_tokens = SearchField(show_tokens=False)
search_field_no_tokens.text = "do not show tokens"
```

## omni.kit.widget.searchable_combobox.SearchModel

A model to support searching functionality in UI components.

This class provides methods to set and retrieve search strings, and to determine if a given string is part of the current search string. It is designed to be used within UI components that require search capabilities, such as searchable combo boxes or lists, and takes a `modified_fn` callable function that's called when the search string is modified.

A simple example of using `SearchModel` to create the base model of a `SearchWidget`, setting and getting the search string, and checking if a string is part of the search string:

```python
from omni.kit.widget.searchable_combobox import SearchModel

def on_search_modified(search_string):
    print(f"Search string modified: {search_string}")

search_model = SearchModel(modified_fn=on_search_modified)

search_model.set_search_string("search string")
search_string = search_model.get_search_string()

is_search_string = search_model.is_in_string("search")
```

## omni.kit.widget.searchable_combobox.SearchWidget

A widget that provides a searchable combobox with customizable styles.

This is an example of creating a SearchWidget with a theme and icon path, getting the text, setting the text, and destroying the SearchWidget. `SearchWidget` can also take an optional `modified_fn` parameter, which is a callback function for when search input is modified. This example also uses the `build_ui_popup` function inside `SearchWidget` to create a popup with a list of items.

```python
from omni.kit.widget.searchable_combobox import SearchWidget

search_widget = SearchWidget(theme=theme, icon_path=None)
name_field, listbox_button = search_widget.build_ui_popup(
    search_size=widget_height,
    default_value=default_value,
    popup_text=combo_list[combo_index] if combo_index >= 0 else default_value,
    index=combo_index,
    update_fn=combo_click_fn,
)

widget_name = search_widget.get_text()
search_widget.set_text("new text for search widget")
search_widget.destroy()
```

## omni.kit.widget.searchable_combobox.build_searchable_combo_widget

A public function that easily helps create a searchable combo widget. This function is accessible by writing `from omni.kit.widget.searchable_combobox import build_searchable_combo_widget`, and returns a SearchWidget object (an instance of the `SearchWidget` with an attached searchable combo box).

Here is an example of how to use the `build_searchable_combo_widget` function to build a searchable combo widget with a list of showroom components:

```python
from omni.kit.widget.searchable_combobox import build_searchable_combo_widget

def on_combo_click_fn(model):
    component = model.get_value_as_string()
    print(f"{component} selected")

component_list = ['3d Reconstruction', 'AEC Experience', 'AI Framework', 'AI Toybox']
component_index = -1
self._component_combo = build_searchable_combo_widget(component_list, component_index, on_combo_click_fn, widget_height=18, default_value="Kit")
```

Cleanup:

```
self._component_combo.destroy()
self._component_combo = None
```

Here is an example of creating a general searchable combobox with `build_searchable_combo_widget`:

```python
import omni.kit.app
import omni.ui as ui
import asyncio
from omni.kit.widget.searchable_combobox import build_searchable_combo_widget, ComboBoxListDelegate

# Create a searchable combo box widget
def create_searchable_combo_box():
    # Define a callback function to be called when an item is selected
    def on_combo_click_fn(model):
        selected_item = model.get_value_as_string()
        print(f"Selected item: {selected_item}")

    # Define the list of items for the combo box
    item_list = ["Item 1", "Item 2", "Item 3"]

    # Create the searchable combo box with the specified items and callback
    searchable_combo_widget = build_searchable_combo_widget(
        combo_list=item_list,
        combo_index=-1,  # Start with no item selected
        combo_click_fn=on_combo_click_fn,
        widget_height=18,
        default_value="Select an item",  # Placeholder text when no item is selected
        window_id="SearchableComboBoxWindow",
        delegate=ComboBoxListDelegate()  # Use the default delegate for item rendering
    )

    # Return the created widget
    return searchable_combo_widget

searchable_combo_box = create_searchable_combo_box()
```

## omni.kit.widget.searchable_combobox.ComboBoxListItem

A single item of the model, with a text string to be displayed by the item.
This is an example of creating a ComboBoxListItem with the text "Apple".

```python
from omni.kit.widget.searchable_combobox import ComboBoxListItem

item = ComboBoxListItem("Apple")
```

## omni.kit.widget.searchable_combobox.ComboBoxListModel

A model for managing a list of items in a ComboBox. This is an example of creating a ComboBoxListModel with a list of fruits.

```python
from omni.kit.widget.searchable_combobox import ComboBoxListModel

fruits = ["Apple", "Banana", "Cherry", "Date", "Elderberry"]
model = ComboBoxListModel(fruits)
```

## omni.kit.widget.searchable_combobox.ComboBoxListDelegate

A delegate for custom widget representation in a TreeView widget. This is an example of creating a ComboBoxListDelegate with a custom build function to render a custom widget in a TreeView:

```python
from omni.kit.widget.searchable_combobox import ComboBoxListDelegate
from omni.ui import color as cl

def build_custom_widget():
    return ui.Label("Custom Widget", style={"color": cl.red})

delegate = ComboBoxListDelegate(build_fn=build_custom_widget)
```

## omni.kit.widget.searchable_combobox.ComboListBoxWidget

A widget that combines a search field with a list box for item selection.

This widget facilitates the selection of items from a list that can be filtered through a search interface. Users can type in the search widget to filter the items displayed in the list box below it. It supports customization through themes and delegates for item display.

This is an example of creating a ComboListBoxWidget with a search widget, a list of items, a window ID, a delegate, and a theme:
We also set the parent of the listbox widget to the name field and build the UI by calling the ComboListBoxWidget's `set_parent` and `build_ui` methods.

```python
from omni.kit.widget.searchable_combobox import ComboListBoxWidget

listbox_widget = ComboListBoxWidget(
    search_widget=search_widget, item_list=combo_list, window_id=window_id, delegate=delegate, theme=theme
)
listbox_widget.set_parent(name_field)
listbox_widget.build_ui()
```

## uicode.EditableTreeWidget

A two column treeview with editable fields. The first column is a string and the second column could be a string, float or an int. With double click, the clicked field becomes editable.

Example: Create a table of prim names and corresponding numbers of them in the scene

```python
import omni.ui as ui
import uicode
window = ui.Window("Capital & Population table", height=500, width=500)
with window.frame:
    with ui.ScrollingFrame():
        data = {"Sphere": 4, "Cube": 3, "Torus": 1}
        treeview=uicode.EditableTreeWidget(data)
```