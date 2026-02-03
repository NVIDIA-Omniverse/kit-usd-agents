# Length Units
The Framework UI offers several different units for expressing length: Pixel, Percent and Fraction. There is no restriction on where certain units should be used.

## Pixel
Pixel is the size in pixels and scaled with the HiDPI scale factor. Pixel is the default unit. If a number is not specified to be a certain unit, it is Pixel. e.g. `width=100` meaning `width=ui.Pixel(100)`.

```execute 200
with ui.HStack():
    ui.Button("40px", width=ui.Pixel(40))
    ui.Button("60px", width=ui.Pixel(60))
    ui.Button("100px", width=100)
    ui.Button("120px", width=120)
    ui.Button("150px", width=150)
```

## Percent
Percent and Fraction units make it possible to specify sizes relative to the parent size. 1 Percent is 1/100 of the parent size.

```execute 200
with ui.HStack():
    ui.Button("5%", width=ui.Percent(5))
    ui.Button("10%", width=ui.Percent(10))
    ui.Button("15%", width=ui.Percent(15))
    ui.Button("20%", width=ui.Percent(20))
    ui.Button("25%", width=ui.Percent(25))
```

## Fraction
Fraction length is made to take the available space of the parent widget and then divide it among all the child widgets with Fraction length in proportion to their Fraction factor.

```execute 200
with ui.HStack():
    ui.Button("One", width=ui.Fraction(1))
    ui.Button("Two", width=ui.Fraction(2))
    ui.Button("Three", width=ui.Fraction(3))
    ui.Button("Four", width=ui.Fraction(4))
    ui.Button("Five", width=ui.Fraction(5))
```
