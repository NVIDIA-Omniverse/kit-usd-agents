Here is the structure:

from pxr import <module>, <module>, <module>

def <function_name>(<parameters>) -> <return_type>:
    \"\"\"<short description>\"\"\"
    <code>

#############
# Full Usage
#############
<some example usage of the functions>

<some tests to check the functions work as expected>

Dont use if __name__ == "__main__": in your code

Here are some examples of snippets:

from pxr import Sdf, Usd, UsdGeom

def create_orthographic_camera(stage: Usd.Stage, prim_path: str="/World/MyOrthoCam") -> UsdGeom.Camera:
    \"\"\"Create an orthographic camera

    Args:
        stage (Usd.Stage): A USD Stage to create the camera on.
        prim_path (str, optional): The prim path for where to create the camera. Defaults to "/World/MyOrthoCam".
    \"\"\"
    camera_path = Sdf.Path(prim_path)
    usd_camera = UsdGeom.Camera.Define(stage, camera_path)
    usd_camera.CreateProjectionAttr().Set(UsdGeom.Tokens.orthographic)
    return usd_camera

# Full Usage
cam_path = "/World/MyOrthoCam"
stage: Usd.Stage = Usd.Stage.CreateInMemory()
root_prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
stage.SetDefaultPrim(root_prim.GetPrim())

camera = create_orthographic_camera(stage, cam_path)

usda = stage.GetRootLayer().ExportToString()
print(usda)


-----------------------  End of example -----------------------

New Example:

from pxr import Usd

def add_inherit(stage: Usd.Stage, prim: Usd.Prim, class_prim: Usd.Prim):
    inherits: Usd.Inherits = prim.GetInherits()
    inherits.AddInherit(class_prim.GetPath())

# Full Usage

from pxr import Sdf, UsdGeom

# Create an in-memory Stage with /World Xform prim as the default prim
stage: Usd.Stage = Usd.Stage.CreateInMemory()
default_prim: Usd.Prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
stage.SetDefaultPrim(default_prim)

# The base prim typically uses the "class" Specifier to designate that it
# is meant to be inherited and skipped in standard stage traversals
tree_class: Usd.Prim = stage.CreateClassPrim("/_class_Tree")
tree_prim: Usd.Prim = UsdGeom.Mesh.Define(stage, default_prim.GetPath().AppendPath("TreeA")).GetPrim()

add_inherit(stage, tree_prim, tree_class)

usda = stage.GetRootLayer().ExportToString()
print(usda)


-----------------------  End of example -----------------------

New Example:
from pxr import Gf, Sdf, Usd, UsdGeom


def create_float_attribute(prim: Usd.Prim, attribute_name: str) -> Usd.Attribute:
    \"\"\"Creates attribute for a prim that holds a float.
    See: https://graphics.pixar.com/usd/release/api/class_usd_prim.html
    Args:
        prim (Usd.Prim): A Prim for holding the attribute.
        attribute_name (str): The name of the attribute to create.
    Returns:
        Usd.Attribute: An attribute created at specific prim.
    \"\"\"
    attr: Usd.Attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float)
    return attr


def create_vector_attribute(prim: Usd.Prim, attribute_name: str) -> Usd.Attribute:
    \"\"\"Creates attribute for a prim that holds a vector.
    See: https://graphics.pixar.com/usd/release/api/class_usd_prim.html
    Args:
        prim (Usd.Prim): A Prim for holding the attribute.
        attribute_name (str): The name of the attribute to create.
    Returns:
        Usd.Attribute: An attribute created at specific prim.
    \"\"\"
    attr: Usd.Attribute = prim.CreateAttribute(
        attribute_name, Sdf.ValueTypeNames.Float3
    )
    return attr


# Full Usage
# Create an in-memory Stage
stage: Usd.Stage = Usd.Stage.CreateInMemory()

# Create a prim named /World (type Xform) and make it the default prim.
prim_path = "/World"
xform: UsdGeom.Xform = UsdGeom.Xform.Define(stage, prim_path)
prim: Usd.Prim = xform.GetPrim()
stage.SetDefaultPrim(prim)

# Create a float attribute on /World
float_attr: Usd.Attribute = create_float_attribute(prim, "my_float_attr")

# Create a vector attribute on /World
vector_attr: Usd.Attribute = create_vector_attribute(prim, "my_vector_attr")

# Set and query values
print(float_attr.Get())
float_attr.Set(0.1)
print(float_attr.Get())

vector_value: Gf.Vec3f = Gf.Vec3f(0.1, 0.2, 0.3)
print(vector_attr.Get())
vector_attr.Set(vector_value)
print(vector_attr.Get())

# We always printout the stage at the end
print(stage.GetRootLayer().ExportToString())
