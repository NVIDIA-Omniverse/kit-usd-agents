# omni.ui.RadioButton

RadioButton is the widget that allows the user to choose only one from a predefined set of mutually exclusive options.

RadioButtons are arranged in collections of two or more buttons within a RadioCollection, which is the central component of the system and controls the behavior of all the RadioButtons in the collection.

Here is a list of styles you can customize on RadioButton:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> background_color (color): the background color of the RadioButton
> padding (float): the distance between the the RadioButton content widget (e.g. Image) and the RadioButton border

To control the style of the button image, you can customize `RadioButton.Image`. For example RadioButton.Image's image_url defines the image when it's not checked. You can define the image for checked status with `RadioButton.Image:checked` style.

Here is an example of RadioCollection which contains 5 RadioButtons with style. Also there is an IntSlider which shares the model with the RadioCollection, so that when RadioButton value or the IntSlider value changes, the other one will update too.

```
from omni.ui import color as cl
style = {
            "RadioButton": {
                "background_color": cl.cyan,
                "margin_width": 2,
                "padding": 1,
                "border_radius": 0,
                "border_color": cl.white,
                "border_width": 1.0},
            "RadioButton.Image": {
                "image_url": f"../exts/omni.kit.documentation.ui.style/icons/radio_off.svg",
            },
            "RadioButton.Image:checked": {
                "image_url": f"../exts/omni.kit.documentation.ui.style/icons/radio_on.svg"},
        }

collection = ui.RadioCollection()
for i in range(5):
    with ui.HStack(style=style):
        ui.RadioButton(radio_collection=collection, width=30, height=30)
        ui.Label(f"Option {i}", name="text")

ui.IntSlider(collection.model, min=0, max=4)
```

