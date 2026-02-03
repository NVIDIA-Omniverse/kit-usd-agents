# omni.ui.scene.PolygonMesh

Encodes a mesh. Meshes are defined as points connected to edges and faces. Each
face is defined by a list of face vertices `vertex_indices` using indices into
the point `positions` array. `vertex_counts` provides the number of points at
each face. This is the minimal requirement to construct the mesh.

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

    # Form the mesh data
    points = []
    vertex_indices = []
    sizes = []
    colors = []
    for i in range(point_count):
        weight = i / point_count
        angle = 2.0 * math.pi * weight
        vertex_indices.append(i)
        points.append(
            [
                math.cos(angle) * weight,
                -math.sin(angle) * weight,
                0
            ]
        )
        colors.append([weight, 1 - weight, 1, 1])
        sizes.append(6 * (weight + 1.0 / point_count))

    # Draw the mesh
    sc.PolygonMesh(
        points, colors, [point_count], vertex_indices
    )
```

