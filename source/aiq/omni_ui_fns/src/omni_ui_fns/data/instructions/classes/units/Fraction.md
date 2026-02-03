# omni.ui.Fraction

Fraction length is made to take the available space of the parent widget and then divide it among all the child widgets with Fraction length in proportion to their Fraction factor.

```execute 200
with ui.HStack():
    ui.Button("One", width=ui.Fraction(1))
    ui.Button("Two", width=ui.Fraction(2))
    ui.Button("Three", width=ui.Fraction(3))
    ui.Button("Four", width=ui.Fraction(4))
    ui.Button("Five", width=ui.Fraction(5))
```


