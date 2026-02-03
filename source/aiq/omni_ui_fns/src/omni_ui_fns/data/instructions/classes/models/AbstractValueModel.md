# omni.ui.AbstractValueModel

There are several predefined models:
- SimpleStringModel
- SimpleBoolModel
- SimpleFloatModel
- SimpleIntModel

To get and set the value of the model it's possible to use `set_value` method:

```
print(model.as_string)
model.as_float = 0.0
model.as_int = int(model.as_bool)
model.set_value(0.0)
```

There are also 4 properties: as_string, as_float, as_int, as_bool, you can use to get or set the value with different types

Also note that widget using the model must always use the model to get or set the value, as data is always being managed by the model, such as:
```
field = ui.IntField()
field.model.set_value(10)
```
instead of getting or setting the value through the widget:
```
field = ui.IntField(default_value=10)
```

It's possible to create the callback when the model is changed:
```
first_checkbox = ui.CheckBox()
second_checkbox = ui.CheckBox()

def on_first_changed(model):
    second_checkbox.model.as_bool = model.as_bool

first_checkbox.model.add_value_changed_fn(on_first_changed)
```

This is a custom implementation of the model.

```
class FloatModel(ui.AbstractValueModel):
    '''An example of custom float model that can be used for formatted string output'''

    def __init__(self, value: float):
        super().__init__()
        self._value = value

    def get_value_as_float(self):
        '''Reimplemented get float'''
        return self._value or 0.0

    def get_value_as_string(self):
        '''Reimplemented get string'''
        # This string goes to the field.
        if self._value is None:
            return ""

        # General format. This prints the number as a fixed-point
        # number, unless the number is too large, in which case it
        # switches to 'e' exponent notation.
        return "{0:g}".format(self._value)

    def set_value(self, value):
        '''Reimplemented set'''
        try:
            value = float(value)
        except ValueError:
            value = None
        if value != self._value:
            # Tell the widget that the model is changed
            self._value = value
            # This updates the widget. Without this line the widget will
            # not be updated.
            self._value_changed()

model = FloatModel()
my_window = ui.Window("Example Window", width=300, height=300)
with my_window.frame:
    with ui.VStack():
        ui.FloatSlider(model, min=-100, max=100)
```

