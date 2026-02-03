# omni.ui.scene.Points

The list of points in 3d space. Points may have different sizes and different colors.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
import math

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    point_count = 36
    points = []
    sizes = []
    colors = []
    for i in range(point_count):
        weight = i / point_count
        angle = 2.0 * math.pi * weight
        points.append(
            [math.cos(angle), math.sin(angle), 0]
        )
        colors.append([weight, 1 - weight, 1, 1])
        sizes.append(6 * (weight + 1.0 / point_count))
    sc.Points(points, colors=colors, sizes=sizes)
```

