# omni.ui.scene.Line

Line is the simplest shape that represents a straight line. It has two points,
color, and thickness.

```execute
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    sc.Line([-0.5,-0.5,0], [0.5, 0.5, 0], color=cl.green, thickness=5)
```

