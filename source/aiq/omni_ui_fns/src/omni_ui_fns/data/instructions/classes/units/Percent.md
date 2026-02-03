# omni.ui.Percent

Percent and Fraction units make it possible to specify sizes relative to the parent size. 1 Percent is 1/100 of the parent size.

```execute 200
with ui.HStack():
    ui.Button("5%", width=ui.Percent(5))
    ui.Button("10%", width=ui.Percent(10))
    ui.Button("15%", width=ui.Percent(15))
    ui.Button("20%", width=ui.Percent(20))
    ui.Button("25%", width=ui.Percent(25))
```

