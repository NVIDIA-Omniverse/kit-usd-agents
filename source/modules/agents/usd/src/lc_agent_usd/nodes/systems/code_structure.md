General Template:

from pxr import <module>, <module>, <module>

<code>
Dont use if __name__ == "__main__": in your code


Here are some examples of coding answer:

Questions: show me how to create an orthographic camera
Answer:
```python
from pxr import Sdf, Usd, UsdGeom

stage: Usd.Stage = Usd.Stage.CreateInMemory()

# Create an orthographic camera
cam_path = "/World/MyOrthoCam"
camera_path = Sdf.Path(prim_path)

usd_camera = UsdGeom.Camera.Define(stage, camera_path)
usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
```

Questions: how to create attributes on a Prim 
```python
from pxr import Gf, Sdf, Usd, UsdGeom

stage: Usd.Stage = Usd.Stage.CreateInMemory()

# Create a prim named /World (type Xform) and make it the default prim.
prim_path = "/World"
xform: UsdGeom.Xform = UsdGeom.Xform.Define(stage, prim_path)
prim: Usd.Prim = xform.GetPrim()

# create float attribute
float_attr: Usd.Attribute = prim.CreateAttribute("my_float_attr", Sdf.ValueTypeNames.Float)

# create vector attribute
vector_attr: Usd.Attribute = prim.CreateAttribute("my_vector_attr", Sdf.ValueTypeNames.Float3)
```


