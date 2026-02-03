# omni.ui.Label

Labels are used everywhere in omni.ui. They are text only objects.

Here is a list of styles you can customize on Label:
> color (color): the color of the text
> font_size (float): the size of the text
> margin (float): the distance between the label and the parent widget defined boundary
> margin_width (float): the width distance between the label and the parent widget defined boundary
> margin_height (float): the height distance between the label and the parent widget defined boundary
> alignment (enum): defines how the label is positioned in the parent defined space. There are 9 alignments supported which are quite self-explanatory.
* ui.Alignment.LEFT_CENTER
* ui.Alignment.LEFT_TOP
* ui.Alignment.LEFT_BOTTOM
* ui.Alignment.RIGHT_CENTER
* ui.Alignment.RIGHT_TOP
* ui.Alignment.RIGHT_BOTTOM
* ui.Alignment.CENTER
* ui.Alignment.CENTER_TOP
* ui.Alignment.CENTER_BOTTOM

Here are a few examples of labels:

```
from omni.ui import color as cl
ui.Label("this is a simple label", style={"color":cl.red, "margin": 5})
```

```
from omni.ui import color as cl
ui.Label("label with alignment", style={"color":cl.green, "margin": 5}, alignment=ui.Alignment.CENTER)
```

Notice that alignment could be either a property or a style.
```
from omni.ui import color as cl
label_style = {
    "Label": {"font_size": 20, "color": cl.blue, "alignment":ui.Alignment.RIGHT, "margin_height": 20}
    }
ui.Label("Label with style", style=label_style)
```

When the text of the Label is too long, it can be elided by `...`:
```
from omni.ui import color as cl
ui.Label(
            "Label can be elided: Lorem ipsum dolor "
            "sit amet, consectetur adipiscing elit, sed do "
            "eiusmod tempor incididunt ut labore et dolore "
            "magna aliqua. Ut enim ad minim veniam, quis "
            "nostrud exercitation ullamco laboris nisi ut "
            "aliquip ex ea commodo consequat. Duis aute irure "
            "dolor in reprehenderit in voluptate velit esse "
            "cillum dolore eu fugiat nulla pariatur. Excepteur "
            "sint occaecat cupidatat non proident, sunt in "
            "culpa qui officia deserunt mollit anim id est "
            "laborum.",
            style={"color":cl.white},
            elided_text=True,
        )
```

