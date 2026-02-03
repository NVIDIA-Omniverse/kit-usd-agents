# omni.ui.scene.Rectangle

Rectangle is a shape with four sides and four corners. The corners are all right angles.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Rectangle(color=cl.green)
```

It's also possible to draw Rectangle with lines with enabling property `wireframe`:

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Rectangle(2, 1, thickness=5, wireframe=True)
```

