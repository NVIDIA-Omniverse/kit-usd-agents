# omni.ui.AbstractItemModel

The item model doesn't hold the data itself. It's using APIs to manage the data.
The `get_item_children` of `AbstractItemModel` returns a list of items
(inheriting from `omni.ui.AbstractItem`), which each item is being managed by
the value model (inheriting from `omni.ui.AbstractValueModel`) that can contain
any data type and supports callbacks. Thus, the model client can track the changes
in both the item model and any value it holds.

The item model can get both the value model and the nested items from any item.
Therefore, the model is flexible to represent anything from color to complicated
tree-table construction.

```
# child items owned by the model
items = model.get_item_children()

# child items of the first item in the model
items = model.get_item_children(items[0])

# get the value model from the first column of the second item in the model
model.get_item_value_model(items[1], 0)
```

To clear all items in the default model from the widget class, you could use `remove_item`
combined with `get_item_children`.
```
cb = ui.ComboBox(0, "Table", "Chair", "Sofa")
for item in cb.model.get_item_children():
    cb.model.remove_item(item)
```

### Item

Item is the object that is associated with the data entity of the model. It must
inherit from `ui.AbstractItem`.

Each item should be created and stored by the model implementation. And can
contain any data in it. Another option would be to use it as a raw pointer to
the data. In any case, it's the choice of the model how to manage this class.

### Hierarchial Model

Usually, the model is a hierarchical system where the item can have any number
of child items. The model is only populated at the moment the user expands the
item to save resources. The following example demonstrates that the model can be
infinitely long.

```execute 200
class Item(ui.AbstractItem):
    """Single item of the model"""

    def __init__(self, text, value):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)
        self.value_model = ui.SimpleIntModel(value)
        self.children = None

class Model(ui.AbstractItemModel):
    def __init__(self, *args):
        super().__init__()
        self._children = [Item(t) for t in args]

    def get_item_children(self, item):
        """
        Returns all the children when the widget asks it.
        Ensure you handle the case where item is None, which should return the child items of the model.
        """
        if item is not None:
            if not item.children:
                item.children = [Item(f"Child #{i}") for i in range(5)]
            return item.children

        return self._children

    def get_item_value_model_count(self, item):
        """The number of columns"""
        return 2

    def get_item_value_model(self, item, column_id):
        """
        Return value model.
        It's the object that tracks the specific value of the column.
        In our case we use ui.SimpleStringModel.
        """
        if column_id == 0:
            return item.name_model
        else:
            return item.value_model

with ui.ScrollingFrame(
    height=200,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    style_type_name_override="TreeView",
):
    self._model = Model("Root", "Items")
    ui.TreeView(self._model, root_visible=False, style={"margin": 0.5})
```

### Nested Model

Since the model doesn't keep any data and serves as an API protocol, sometimes
it's very helpful to merge multiple models into one single model. The parent
model should redirect the calls to the children.

In the following example, three different models are merged into one.

```execute 200
class Item(ui.AbstractItem):
    def __init__(self, text, name, d=5):
        super().__init__()
        self.name_model = ui.SimpleStringModel(text)
        self.children = [Item(f"Child {name}{i}", name, d - 1) for i in range(d)]

class Model(ui.AbstractItemModel):
    def __init__(self, name):
        super().__init__()
        self._children = [Item(f"Model {name}", name)]

    def get_item_children(self, item):
        return item.children if item else self._children

    def get_item_value_model_count(self, item):
        return 1

    def get_item_value_model(self, item, column_id):
        return item.name_model

class NestedItem(ui.AbstractItem):
    def __init__(self, source_item, source_model):
        super().__init__()
        self.source = source_item
        self.model = source_model
        self.children = None

class NestedModel(ui.AbstractItemModel):
    def __init__(self):
        super().__init__()
        models = [Model("A"), Model("B"), Model("C")]
        self.children = [
            NestedItem(i, m) for m in models for i in m.get_item_children(None)]

    def get_item_children(self, item):
        if item is None:
            return self.children

        if item.children is None:
            m = item.model
            item.children = [
                NestedItem(i, m) for i in m.get_item_children(item.source)]

        return item.children

    def get_item_value_model_count(self, item):
        return 1

    def get_item_value_model(self, item, column_id):
        return item.model.get_item_value_model(item.source, column_id)

with ui.ScrollingFrame(
    height=200,
    horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
    vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
    style_type_name_override="TreeView",
):
    self._model = NestedModel()
    ui.TreeView(self._model, root_visible=False, style={"margin": 0.5})
```

