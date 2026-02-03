# omni.ui.ImageWithProvider

ImageWithProvider also displays an image just like Image. It is a much more advanced image widget. ImageWithProvider blocks until the image is loaded, Image doesn't block. Sometimes Image blinks because when the first frame is created, the image is not loaded. Users are recommended to use ImageWithProvider if the UI is updated pretty often. Because it doesn't blink when recreating.

Here is a list of styles you can customize on ImageWithProvider:
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
* ui.IwpFillPolicy.IWP_STRETCH: stretch the image to fill the entire rectangle.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT: uniformly to fit the image without stretching or cropping.
* ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP: scaled uniformly to fill, cropping if necessary
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

The image source comes from `ImageProvider` which could be `ByteImageProvider`, `RasterImageProvider` or `VectorImageProvider`.

`RasterImageProvider` and `VectorImageProvider` are using image urls like Image. Here is an example taken from Image. Notice the fill_policy value difference.
```
from omni.ui import color as cl
source = "resources/desktop-icons/omniverse_512.png"
with ui.Frame(width=200, height=100):
    ui.ImageWithProvider(
        source,
        style={
            "ImageWithProvider": {
            "border_width": 5,
            "border_color": cl("#1ab3ff"),
            "corner_flag": ui.CornerFlag.TOP,
            "border_radius": 15,
            "fill_policy": ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_CROP,
            "alignment": ui.Alignment.CENTER_BOTTOM}})
```

`ByteImageProvider` is really useful to create gradient images. Here is an example:
```
self._byte_provider = ui.ByteImageProvider()
self._byte_provider.set_bytes_data([
    255, 0, 0, 255,    # red
    255, 255, 0, 255,  # yellow
    0,  255, 0, 255,   # green
    0, 255, 255, 255,  # cyan
    0, 0, 255, 255],   # blue
    [5, 1])            # size
with ui.Frame(height=20):
    ui.ImageWithProvider(self._byte_provider,fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
```

