# omni.ui.AbstractItemDelegate

There are some complex widgets (e.g. omni.ui.TreeView, omni.ui.Combobox) for which we can define our customized delegates.

For example, we can create a Delegate for a TreeView widget that displays the items in branches.

```
class Delegate(ui.AbstractItemDelegate):
    """
    Delegate is the representation layer. TreeView calls the methods
    of the delegate to create custom widgets for each item.
    """

    def build_branch(self, model, item, column_id, level, expanded):
        """Create a branch widget that opens or closes subtree"""
        # Offset depents on level
        text = "     " * (level + 1)
        # > and v symbols depending on the expanded state
        if expanded:
            text += "v    "
        else:
            text += ">    "
        ui.Label(text, height=22, alignment=ui.Alignment.CENTER, tooltip="Branch")

    def build_widget(self, model, item, column_id, level, expanded):
        """Create a widget per column per item"""
        ui.Label(
            model.get_item_value_model(item, column_id).as_string,
            tooltip="Widget"
        )

    def build_header(self, column_id):
        """Build the header"""
        ui.Label("Header", tooltip="Header", height=25)

self._delegate = Delegate()
ui.TreeView(delegate=self._delegate, root_visible=False, header_visible=True)
```

