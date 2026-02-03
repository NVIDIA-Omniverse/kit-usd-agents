# omni.ui.scene.Image

A rectangle with an image on it. It can read raster and vector graphics format
and supports `http://` and `omniverse://` paths.

```execute 150
##
from omni.ui import scene as sc
from omni.ui import color as cl
from pathlib import Path

EXT_PATH = f"{Path(__file__).parent.parent.parent.parent.parent}"

scene_view = sc.SceneView(
    aspect_ratio_policy=sc.AspectRatioPolicy.PRESERVE_ASPECT_FIT,
    height=150,
)

with scene_view.scene:
    ##
    filename = f"{EXT_PATH}/data/main_ov_logo_square.png"
    sc.Image(filename)
```

