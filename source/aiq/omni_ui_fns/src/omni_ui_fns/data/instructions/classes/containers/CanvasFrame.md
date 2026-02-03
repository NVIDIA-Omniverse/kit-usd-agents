# omni.ui.CanvasFrame

CanvasFrame is the widget that allows the user to pan and zoom its children with a mouse. It has a layout that can be infinitely moved in any direction.

Here is a list of styles you can customize on CanvasFrame:
> background_color (color): the main color of the rectangle

Here is an example of a CanvasFrame, you can scroll the middle mouse to zoom the canvas and middle mouse move to pan in it (press CTRL to avoid scrolling the docs).

```
from omni.ui import color as cl
TEXT = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
)

IMAGE = "resources/icons/ov_logo_square.png"
with ui.CanvasFrame(height=256, style={"CanvasFrame":{"background_color": cl("#aa4444")}}):
    with ui.VStack(height=0, spacing=10):
        ui.Label(TEXT, name="text", word_wrap=True)
        ui.Button("Button")
        ui.Image(IMAGE, width=128, height=128)
```

