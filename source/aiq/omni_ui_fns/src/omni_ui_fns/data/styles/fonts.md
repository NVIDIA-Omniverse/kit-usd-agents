# Fonts

## Font style
It's possible to set different font types with the style. The style key 'font' should point to the font file, which allows packaging of the font to the extension. We support both TTF and OTF formats. All text-based widgets support custom fonts.

```execute 200
with ui.VStack():
    ui.Label("Omniverse", style={"font":"${fonts}/OpenSans-SemiBold.ttf", "font_size": 40.0})
    ui.Label("Omniverse", style={"font":"${fonts}/roboto_medium.ttf", "font_size": 40.0})
```

## Font size
It's possible to set the font size with the style.

Drag the following slider to change the size of the text.

```execute 200
## Double comment means hide from snippet
from functools import partial
##
def value_changed(label, value):
    label.style = {"color": ui.color(0), "font_size": value.as_float}

slider = ui.FloatSlider(min=1.0, max=150.0)
slider.model.as_float = 10.0
label = ui.Label("Omniverse", style={"color": ui.color(0), "font_size": 7.0})
slider.model.add_value_changed_fn(partial(value_changed, label))
## Double comment means hide from snippet
ui.Spacer(height=30)
##
```