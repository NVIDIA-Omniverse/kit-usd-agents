# omni.ui.Pixel

Pixel is the size in pixels and scaled with the HiDPI scale factor. Pixel is the default unit. If a number is not specified to be a certain unit, it is Pixel. e.g. `width=100` meaning `width=ui.Pixel(100)`.

```execute 200
with ui.HStack():
    ui.Button("40px", width=ui.Pixel(40))
    ui.Button("60px", width=ui.Pixel(60))
    ui.Button("100px", width=100)
    ui.Button("120px", width=120)
    ui.Button("150px", width=150)
```

