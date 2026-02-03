# omni.ui.scene.Curve

Curve is a shape drawn with multiple lines which has a bent or turns in it. There are two supported cuve types: linear and cubic. `sc.Curve` is default to draw cubic curve and can be switched to linear with `curve_type=sc.Curve.CurveType.LINEAR`. `tessellation` controls the smoothness of the curve. The higher the value is, the smoother the curve is, with the higher computational cost. The curve also has positions, colors, thicknesses which are all array typed properties, which means per vertex property is supported. This feature needs further development with ImGui support.

```execute
##
from omni.ui import scene as sc
from omni.ui import color as cl

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)
##

with scene_view.scene:
    # linear curve
    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(-4, 0, 0)):
        sc.Curve(
            [[0.5, -0.7, 0], [0.1, 0.6, 0], [2.0, 0.6, 0], [3.5, -0.7, 0]],
            thicknesses=[1.0],
            colors=[cl.red],
            curve_type=sc.Curve.CurveType.LINEAR,
        )
    # corresponding cubic curve
    with sc.Transform(transform=sc.Matrix44.get_translation_matrix(0, 0, 0)):
        sc.Curve(
            [[0.5, -0.7, 0], [0.1, 0.6, 0], [2.0, 0.6, 0], [3.5, -0.7, 0]],
            thicknesses=[3.0],
            colors=[cl.blue],
            tessellation=9,
        )
```

