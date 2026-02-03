# omni.ui.scene.Arc

Two radii of a circle and the arc between them. It also can be a wireframe.

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
    sc.Arc(1, begin=0, end=1, thickness=5, wireframe=True)
```

