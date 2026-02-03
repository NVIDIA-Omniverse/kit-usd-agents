# omni.ui.AbstractMultiField

AbstractMultiField is a widget with multiple AbstractFields implemented using the model-delegate-view pattern and uses `omni.ui.AbstractItemModel` as the central component of the system. The item model has multiple AbstractItems each managed by its respective AbstractField's value model.

To access each field's value model, you could get its corresponding item via `get_item_children` before the item's value model with `get_item_value_model`.
```
import omni.ui as ui
# Create a multi float field widget with 3 fields with default set to 0.0, 5.0, 10.0 respectively
multi_field = ui.MultiFloatField(0.0, 5.0, 10.0)
# Get all items of the multi field
items = multi_field.model.get_item_children()
# Get the second item's value model at the first column
multi_field.model.get_item_value_model(items[1], 0)
```

