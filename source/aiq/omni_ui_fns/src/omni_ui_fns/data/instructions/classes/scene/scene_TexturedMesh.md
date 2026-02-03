# omni.ui.scene.TexturedMesh

Encodes a polygonal mesh with free-form textures. Meshes are defined the same as PolygonMesh. It supports both ImageProvider and URL. Basically it's PolygonMesh with the ability to use images. Users can provide either sourceUrl or imageProvider, just as sc.Image as the source of the texture. And `uvs` provides how the texture is applied to the mesh.

NOTE: in Kit 105 UVs are specified with V-coordinate flipped, while Kit 106 will move to specifying UVs in same "space" as USD.
In 105.1 there is a transitional property "legacy_flipped_v" that can be provided to the TexturedMesh constructor to internally handle the conversion, but specifying UV cordinates with legacy_flipped_v=True has a negative performance impact.

```execute 150
from omni.ui import scene as sc
from pathlib import Path

EXT_PATH = f"{Path(__file__).parent.parent.parent.parent.parent}"

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    point_count = 4
    # Form the mesh data
    points = [(-1, -1, 0), (1, -1, 0), (-1, 1, 0), (1, 1, 0)]
    vertex_indices = [0, 2, 3, 1]
    colors = [[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]]
    uvs = [(0, 0), (0, 1), (1, 1), (1, 0)]
    # Draw the mesh
    filename = f"{EXT_PATH}/data/main_ov_logo_square.png"
    sc.TexturedMesh(filename, uvs, points, colors, [point_count], vertex_indices, legacy_flipped_v=False)
```

