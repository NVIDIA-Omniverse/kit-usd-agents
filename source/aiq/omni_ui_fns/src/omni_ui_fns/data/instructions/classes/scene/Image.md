# omni.ui.Image

The Image type displays an image. The source of the image is specified as a URL using the source property. By default, specifying the width and height of the item makes the image to be scaled to fit that size. This behavior can be changed by setting the `fill_policy` property, allowing the image to be stretched or scaled instead. The property alignment controls how the scaled image is aligned in the parent defined space.

Here is a list of styles you can customize on Image:
> border_color (color): the border color if the button or image background has a border
> border_radius (float): the border radius if the user wants to round the button or image
> border_width (float): the border width if the button or image or image background has a border
> margin (float): the distance between the widget content and the parent widget defined boundary
> margin_width (float): the width distance between the widget content and the parent widget defined boundary
> margin_height (float): the height distance between the widget content and the parent widget defined boundary
> image_url (str): the url path of the image source
> color (color): the overlay color of the image
> corner_flag (enum): defines which corner or corners to be rounded. The supported corner flags are the same as Rectangle since Image is eventually an image on top of a rectangle under the hood.
> fill_policy (enum): defines how the Image fills the rectangle.
There are three types of fill_policy
* ui.FillPolicy.STRETCH: stretch the image to fill the entire rectangle.
* ui.FillPolicy.PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.FillPolicy.PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary
> alignment (enum): defines how the image is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Default Image is scaled uniformly to fit without stretching or cropping (ui.FillPolicy.PRESERVE_ASPECT_FIT), and aligned to ui.Alignment.CENTER:
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source)
```

The image is stretched to fit and aligned to the left
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.STRETCH, alignment=ui.Alignment.LEFT_CENTER)
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the top
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_TOP)
```

The image is scaled uniformly to fit without cropping and aligned to the right. Notice the fill_policy and alignment are defined in style.
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(source, style={
        "Image": {
            "fill_policy": ui.FillPolicy.PRESERVE_ASPECT_FIT,
            "alignment": ui.Alignment.RIGHT_CENTER,
            "margin": 5}})
```

The image has rounded corners and an overlayed color. Note image_url is in the style dictionary.
```
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(style={"image_url": source, "border_radius": 10, "color": cl("#5eb3ff")})
```

The image is scaled uniformly to fill, cropping if necessary and aligned to the bottom, with a blue border.
```
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.Image(
        source,
        fill_policy=ui.FillPolicy.PRESERVE_ASPECT_CROP,
        alignment=ui.Alignment.CENTER_BOTTOM,
        style={"Image":{
            "border_width": 5,
            "border_color": cl("#1ab3ff"),
            "corner_flag": ui.CornerFlag.TOP,
            "border_radius": 15}})
```

The image is arranged in a HStack with different margin styles defined. Note image_url is in the style dict.
```
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(height=100):
    with ui.HStack(spacing =5, style={"Image":{'image_url': source}}):
        ui.Image()
        ui.Image(style={"Image":{"margin_height": 15}})
        ui.Image()
        ui.Image(style={"Image":{"margin_width": 20}})
        ui.Image()
        ui.Image(style={"Image":{"margin": 10}})
        ui.Image()
```

It's possible to set a different image per style state. And switch them depending on the mouse hovering, selection state, etc.
```

styles = [
    {
        "": {"image_url": "resources/icons/Nav_Walkmode.png"},
        ":hovered": {"image_url": "resources/icons/Nav_Flymode.png"},
    },
    {
        "": {"image_url": "resources/icons/Move_local_64.png"},
        ":hovered": {"image_url": "resources/icons/Move_64.png"},
    },
    {
        "": {"image_url": "resources/icons/Rotate_local_64.png"},
        ":hovered": {"image_url": "resources/icons/Rotate_global.png"},
    },
]

def set_image(model, image):
    value = model.get_item_value_model().get_value_as_int()
    image.set_style(styles[value])

with ui.Frame(height=80):
    with ui.VStack():
        image = ui.Image(width=64, height=64, style=styles[0])
        with ui.HStack(width=ui.Percent(50)):
            ui.Label("Select a texture to display", name="text")
            model = ui.ComboBox(0, "Navigation", "Move", "Rotate").model
            model.add_item_changed_fn(lambda m, i, im=image: set_image(m, im))
```

