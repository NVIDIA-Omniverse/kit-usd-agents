# Omniverse and USD

## USD

- OpenUSD Fundamentals
- Working with USD
- USD Tools
- Robot Schema
- Composing Robots
- Applying the Robot Schema
- Parsing Robot Structure
- Asset Structure

## Omniverse

- Commands
- Registered Actions

---

# OpenUSD Fundamentals

The language used in Isaac Sim to describe the robot and its environment is the [Universal Scene Description (USD)](https://openusd.org/release/index.html).

## Why USD?

USD enables seamless interchange of 3D content among diverse content creation apps with its rich, extensible language. With concepts of layering and variants, it’s a powerful tool that enables live collaboration on the same asset and scene. And when properly used, it permits working on assets without overwriting and erasing someone else’s work.

USD provides a text-based format for direct editing (*.usda). For higher performance and space optimization, there is a binary-encoded format (*.usd). All aspects of USD can be accessed through coding in C++ or Python.

APIs are available for you to set up a scene or tune a robot directly in USD, but, typically it is not necessary to use them.

## Hello World

Let’s start by creating a basic USD file from code:

```python
from pxr import Usd, UsdGeom

stage = Usd.Stage.CreateNew("/path/to/HelloWorld.usda")
xformPrim = UsdGeom.Xform.Define(stage, "/hello")
spherePrim = UsdGeom.Sphere.Define(stage, "/hello/world")
# generic_spherePrim = stage.DefinePrim('/hello/world_generic', 'Sphere')
stage.GetRootLayer().Save()
```

Replacing `/path/to/` with the desired save folder. You can execute this code in the script editor (**Window > Script Editor**) in Isaac Sim, and it yields the following USD file:

```python
#usda 1.0

def Xform "hello"
{
    def Sphere "world"
    {
    }
}
```

This example contains a couple of powerful things we can take away from it:

* **Type**: Elements in USD (called *Prims*) have a defined type. In the case of `hello`, it is of type `Xform`, a type used everywhere, and it defines elements that contain a transform in the world. `World` is of type *Sphere*, which represents a primitive geometry.
* **Composition**: Prims can have *nested prims*. These nested prims are, for all effects, fully defined elements, with their own attributes.
* **Introspection**: If uncommented, the line `generic_spherePrim = stage.DefinePrim('/hello/world_generic', 'Sphere')` would yield a sphere just like the `/hello/world`. Prim types can be defined directly through their schema name.
* **Namespaces**: Both *Xform* and *Sphere* are part of the standard pxr namespace *UsdGeom*, a set of types that represent geometry elements in the scene.

You can open this USD file in Isaac Sim in the script editor window with:

```python
import omni

omni.usd.get_context().open_stage("/path/to/HelloWorld.usda")
```

### Inspecting and Authoring Properties

With a basic scene, you can start making modifications to the elements. Start by opening and getting the elements from the scene:

```python
from pxr import Usd, Vt

stage = Usd.Stage.Open("/path/to/HelloWorld.usda")
xform = stage.GetPrimAtPath("/hello")
sphere = stage.GetPrimAtPath("/hello/world")
print(xform.GetPropertyNames())
print(sphere.GetPropertyNames())
```

The output for the code above is:

```python
['proxyPrim', 'purpose', 'visibility', 'xformOpOrder']
['doubleSided', 'extent', 'orientation', 'primvars:displayColor' 'primvars:displayOpacity', 'proxyPrim', 'purpose', 'radius', 'visibility', 'xformOpOrder']
```

USD offers polymorphism. If you review both lists you can see the common attributes. By having a common `XFormable` ancestor, Xforms and Spheres share a subset of properties, while sphere contains some unique elements that only make sense for its specialization (for example, *radius*).

To update these attributes, you can append the following to the code above:

```python
from pxr import Usd, Vt

stage = Usd.Stage.Open("/path/to/HelloWorld.usda")
xform = stage.GetPrimAtPath("/hello")
sphere = stage.GetPrimAtPath("/hello/world")
print(xform.GetPropertyNames())
print(sphere.GetPropertyNames())
```

Because the stage was still open from the previous sample, you’ll see the sphere reducing from radius 1.0 to 0.5, but it also prints these values in the console.

To move the sphere to a new position use `xformOpOrder`, which is common to `Xform` and `Sphere`. Many different transforms can be applied to a prim, each from potentially different layers. The `xformOpOrder` tracks and manages the different transforms, it is like a list of `Xform` operations, applied in the order specified from first to last.

Our sphere doesn’t have its own, so to create a new one:

```python
from pxr import Usd, Vt

stage = Usd.Stage.Open("/path/to/HelloWorld.usda")
xform = stage.GetPrimAtPath("/hello")
sphere = stage.GetPrimAtPath("/hello/world")
print(xform.GetPropertyNames())
print(sphere.GetPropertyNames())
```

Notice that the sphere has jumped to a new position along the X-axis. Alternatively, you could apply the translation to the parent `xform` instead.

```python
from pxr import Usd, Vt

stage = Usd.Stage.Open("/path/to/HelloWorld.usda")
xform = stage.GetPrimAtPath("/hello")
sphere = stage.GetPrimAtPath("/hello/world")
print(xform.GetPropertyNames())
print(sphere.GetPropertyNames())
```

Verify that you see the sphere jump to a new location, which is the composition of both the parent and child transforms.

A consequence of the universal nature of USD is that when you fetch a prim by path, it is always of type `prim` and needs to be cast appropriately before performing operations with or on it.

To create and bind a material to the prim to change its color, first create it:

```python
from pxr import Usd, Vt

stage = Usd.Stage.Open("/path/to/HelloWorld.usda")
xform = stage.GetPrimAtPath("/hello")
sphere = stage.GetPrimAtPath("/hello/world")
print(xform.GetPropertyNames())
print(sphere.GetPropertyNames())
```

Material color shading is complicated. After creating the prim and appropriate attributes, you must link those attributes and properties together to form a `shader graph` that is processed to produce the desired material effect. After it’s created, the material can then be bound to the prim, thus changing its apparent color in the viewport.

```python
# bind the material
material = UsdShade.Material(material_prim)
binding_api = UsdShade.MaterialBindingAPI.Apply(sphere)
binding_api.Bind(material)
```

If you save the stage and examine the USDA file, you can see the material.

```python
#usda 1.0

def Material "material"
{
    token outputs:mdl:displacement.connect = </hello/material/Shader.outputs:out>
    token outputs:mdl:surface.connect = </hello/material/Shader.outputs:out>
    token outputs:mdl:volume.connect = </hello/material/Shader.outputs:out>

    def Shader "Shader"
    {
        uniform token info:implementationSource = "sourceAsset"
        uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
        uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
        color3f inputs:diffuse_color_constant = (1, 0, 0) (
            customData = {
                float3 default = (0.2, 0.2, 0.2)
            }
            displayGroup = "Albedo"
            displayName = "Albedo Color"
            doc = "This is the albedo base color"
            hidden = false
            renderType = "color"
        )
        color3f inputs:emissive_color = (1, 0, 0) (
            customData = {
                float3 default = (1, 0.1, 0.1)
            }
            displayGroup = "Emissive"
            displayName = "Emissive Color"
            doc = "The emission color"
            hidden = false
            renderType = "color"
        )
        token outputs:out
    }
}
```

and specifically, the `diffuse_color_constant` attribute type. To directly modify this attribute to change the color of our sphere:

```python
import omni

omni.usd.get_context().open_stage("/path/to/HelloWorld.usda")
```

Of course, this level of direct manipulation of USD can become tedious. For situations like this, there are a set of predefined commands through the kit API, which dramatically simplifies working with USD in code. For example, you could have done the following instead:

```python
import omni

omni.usd.get_context().open_stage("/path/to/HelloWorld.usda")
```

## Further Reading

For a complete tutorial on USD, see the [openUSD tutorials](https://openusd.org/release/tut_usd_tutorials.html). With a few tweaks, as shown on the basic examples above, these tutorials can be run from the Script editor or in the [Isaac Python shell](Python_Scripting_and_Tutorials.md).

For more in-depth content, see [guided learning](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/guided-learning.html#openusd-guided-learning) content or the [independent learning](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent-learning.html).

## Units in USD

By default, Isaac Sim USD uses the following default units:

| Unit | Default |
| --- | --- |
| Distance | meters (m) |
| Time | seconds (s) |
| Mass | Kilogram (kg) |
| Angle | Degrees |

For more Isaac Sim conventions, see [Isaac Sim Conventions](Isaac_Sim_Conventions.md).

There are cases when assets coming from different apps follow a different standard. By default, Isaac Sim has enabled the [Metrics Assembler](https://docs.omniverse.nvidia.com/extensions/latest/ext_metrics_assembler.html "(in Omniverse Extensions)"), which automatically converts the asset scale for the distance unit, mass unit, and Up Axis.

For more details about how USD handles units, see [Units in USD](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/independent/units.html).

## Useful USD Snippets

Here are some useful snippets that can be useful when dealing with USD in code. These snippets assume that `stage` and `prim`: are respectively pxr.UsdStage and pxr.UsdPrim types, and if any additional type is used, the necessary imports are included in the snippet.

### Traversing Stage or Prim

```python
# For stage traversal there's a built-in method:
for a in stage.Traverse():
    do_something(a)

# For prim, it's not the same method though
from pxr import Usd

prim = stage.GetDefaultPrim()
for a in Usd.PrimRange(prim):
    do_something(a)
```

### Working with Multiple Layers

```python
from pxr import Sdf

# Get References to all layers
root_layer = stage.GetRootLayer()
session_layer = stage.GetSessionLayer()

# Add a SubLayer to the Root Layer
additional_layer = layer = Sdf.Layer.FindOrOpen("my_layer.usd")
root_layer.subLayerPaths.append(additional_layer.identifier)

# Set Edit Layer
# Method 1
with Usd.EditContext(stage, root_layer):
    do_something()

# Method 2
stage.SetEditTarget(additional_layer)

# Make non-persistent changes to the stage (won't be saved regardless if you call stage.Save)

with Usd.EditContext(stage, session_layer):
    do_something()
```

### Converting Transform Pose in Position, Orient, Scale

Note

You can use this to create a set\_pose method that receives a transform and applies to the prim.

```python
from pxr import Gf, Usd, UsdGeom

def convert_ops_from_transform(prim: pxr.UsdPrim):

    # Get the Xformable from prim
    xform = UsdGeom.Xformable(prim)

    # Gets local transform matrix - used to convert the Xform Ops.
    pose = omni.usd.get_local_transform_matrix(prim)

    # Compute Scale
    x_scale = Gf.Vec3d(pose[0][0], pose[0][1], pose[0][2]).GetLength()
    y_scale = Gf.Vec3d(pose[1][0], pose[1][1], pose[1][2]).GetLength()
    z_scale = Gf.Vec3d(pose[2][0], pose[2][1], pose[2][2]).GetLength()

    # Clear Transforms from xform.
    xform.ClearXformOpOrder()

    # Add the Transform, orient, scale set
    xform_op_t = xform.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_r = xform.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op_s = xform.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")

    xform_op_t.Set(pose.ExtractTranslation())
    xform_op_r.Set(pose.ExtractRotationQuat().GetNormalized())
    xform_op_s.Set(Gf.Vec3d(x_scale, y_scale, z_scale))
```

---

# Working with USD

## Learning Objectives

This tutorial covers how to:

* save USD stages
* load and reference existing USD stages
* Organize stage tree hierarchy

## Saving Options

* **Save**: To save the current USD stage, go to the Menu Bar and click *Files > Save* or *Files > Save As ..* to save as a new file.
* **Save Flattened As**: Saves the current USD file while merging all components to one mesh.
* **as .usda files**: You have the option to save as `.usda` file instead of `.usd` file. `.usda` file is a human-readable text file format for the given USD stage.
* **Collect Assets**: If your current stage used many reference USD stages, materials, and textures from other folders and servers, you must *Collect Assets* to make sure all the external references that are used in your stage get collected in one folder. To do so, save the current USD locally, then find it in the *Content* tab, right-click on it, and select *Collect Asset*.

## Loading Options

* **Open**: To load a USD stage, go to Menu Bar and click *Files > Open*. This opens the USD stage for direct editing.
* **Add Reference**: *Files > Add Reference* adds a USD file as a reference. Or find the file in the *Content* Tab and drag it into the viewport. You can not edit the referenced USD.

### Set the Stage for a Reference

To demonstrate adding a file as reference, save the current stage with the cube and cylinders as a mock robot.
First, you must rearrange the rigid bodies on the stage into a hierarchical structure with meaningful names.
Put all the rigid body parts of the robot under a single [Prim](Glossary.md).

1. Right click inside the *Stage* tab, select *Create > Xform*.
2. Rename the newly added [Xform](https://docs.omniverse.nvidia.com/utilities/latest/common/glossary-of-terms.html#term-XForm "(in Omniverse Utilities)") to *mock\_robot*. The Prim appears under the *World* prim.
3. Drag and drop the Cube, both Cylinders, Physics Material, and Looks folder under *mock\_robot*.
4. Rename the Cube and Cylinders to the body, wheel\_left, and wheel\_right.
5. Save the stage as an USD file.
6. Open a new stage.
7. Load the USD file as a reference, either *Files > Add Reference* or drag the file from *Content* on to the stage. It loads the referenced USD under a Prim withe the same name as the USD filename.
8. Validate that it loaded everything under the original `World(defaultPrim)`, including `PhysicsScene`, `defaultLight`, and `GroundPlane`. This may not be optimal if you are loading multiple USD references that all have their own version of PhysicsScenes and defaultLights. You cannot delete them on the new stage because they are loaded by reference, but deleting them in the original USD would make it difficult to work within those USD stages.

To have the necessary environment set up in the USD stages but not export them when they are being referenced, you need to move non-referenced items out of the default Prim:

* Select the robot’s parent prim on stage, in this tutorial /mock\_robot.
* Open the menu *Edit* while the prim is selected, and click on *unparent*.
* Validate that instead of being under World, mock\_robot is parallel to World.
* Right-click on the robot prim again on stage, and *Set as a Default Prim*. Save.
* Open a new stage and load the same file again as a reference, verify that only the robot is imported.

## Summary

In this tutorial, you learned how to save and open USD files.

### Further Readings

More on [File Menu](https://docs.omniverse.nvidia.com/composer/latest/menu_file.html "(in Omniverse USD Composer)"), [Collect Assets](https://docs.omniverse.nvidia.com/extensions/latest/ext_collect.html "(in Omniverse Extensions)"), and others in [Composer](https://docs.omniverse.nvidia.com/composer/latest/index.html "(in Omniverse USD Composer)").

---

# USD Tools

## USD Paths

The USD contains paths to all of these assets which can become invalid for many reasons. For example, moving the files around in the local file system during reorganization.

The USD Paths tool lets you edit these paths and has find and replace functionality.

A video tutorial can be found in the Omniverse tutorial [USD Paths](https://docs.omniverse.nvidia.com/extensions/latest/ext_usd-paths.html)

## Variant Presenter

The Variant Presenter provides a menu to view and manage variants and variant groups. More details can be found in the Omniverse tutorial [Variant Presenter](https://docs.omniverse.nvidia.com/extensions/latest/ext_variant-presenter.html)

## Variant Editor

The Variant Editor provides an interface for adding, removing, and modifying variants in USD. More details can be found in the Omniverse tutorial [Variant Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_variant-editor.html)

---

# Robot Schema

The Robot Schema extends OpenUSD definitions to define robotic structures. While currently experimental, it provides a standardized way to represent robots by building upon USD Common definitions and the Physics Schema for kinematic tree definitions.

The schema defines four fundamental Structure Types:

In this second revision, we introduce new utilities to the schema to auto-populate the Links and Joints lists, based on the physics of the robot. Additionally, we deprecated the indexed attributes for the DOF offsets, and replaced them with a single attribute list stating the degrees of freedom order. Also, ReferencePointAPI was renamed to SiteAPI.
We also added metadata to the robot to describe its type, what License it is under, the source of the original asset, and a version control with changelog, in the form of attributes to the Robot API.

## Robot API

The `IsaacRobotAPI` serves as the root definition for a robot, describing its complete composition. Applied to the robot’s root prim, it contains:

1. Description: Metadata describing the robot’s purpose and capabilities
2. Namespace: Unique identifier namespace for robot component messaging
3. Links: Ordered list of constituent links, starting with the base link
4. Joints: Ordered list of connecting joints
5. Type: The type of robot, such as “Manipulator”, “Humanoid”, “Moving base”, “etc.”
6. License: The license under which the robot is distributed.
7. Source: The source of the original asset (e.g., the link to the original asset, or the company/website of the original author)
8. Version: The version of the robot asset. This should be updated whenever the robot asset is updated, and should be a semantic versioning number.
9. Changelog: A changelog of the robot asset, with the changes made to the asset over time. This should be updated whenever the robot asset is updated, and should be a list of changes made to the asset.

Note

The Links and Joints lists need only contain elements relevant for reporting. The full kinematic tree may contain additional unlisted elements.

## Link API

The Link API describes a single link in the robot and serves as a flag to indicate that the link should be included in the robot composition. This schema should be applied to the bodies of the robot. It contains the following attributes:

1. Name Override: By default, Isaac Sim will use the prim name as the link name when reporting the robot state. This attribute allows for a custom name to be used.

Links are not limited to Rigid bodies, and could be applied to other types of simulation, such as deformable bodies. Care must be taken when using links on deformable bodies, as it would require an equivalent way to compute the robot state if needed.

All Links used by the robot must have an `IsaacLinkAPI` applied, regardless of whether they are included in the `IsaacRobotAPI` Links list or not.

## Joint API

The Joint API describes a single joint in the robot and serves as a flag to indicate that the joint should be included in the robot composition. This schema should be applied to the joints of the robot. It contains the following attributes:

1. Name Override: By default, Isaac Sim will use the prim name as the joint name when reporting the robot state. This attribute allows for a custom name to be used.
2. DOF (Degree of Freedom) Offset: For each degree of freedom, we introduce an index offset to the reported state, so we can report all DOF stats as a single flat list. This is useful for composing robots that have multiple degrees of freedom but share a common root joint. There is one attribute per degree of freedom axis. The default value is 0-6, depending on the axis. If the joint represents a single degree of freedom, this attribute can be ignored.

All Joints used by the robot must have an `IsaacJointAPI` applied, regardless of whether they are included in the `IsaacRobotAPI` Joints list or not.

## Site API (Formerly Reference Point API)

The `IsaacSiteAPI` describes points of interest on the robot, for example, attachment points for tools or sensors. This schema should be applied to the points of interest of the robot. It contains the following attributes:

1. Description: A description of the reference point, for example “Tool Attachment Point” or “Sensor Location”.
2. Forward Axis: The axis that is considered to be the forward direction of the reference point (X, Y, Z).

Note

The Site API replaces the Reference Point API. The Reference Point API is deprecated and not available to be applied to new robots. Robots with Reference Point API applied will still work in this release, but will issue a depreaction warning, and need to be updated to use the Site API.

# Composing Robots

Robot compositions can be created by applying the Robot API to each sub-robot’s root prim. The final assembly is achieved by either:
- Adding a sub-robot’s root prim to the parent robot’s joints and links lists.
- Selecting specific links and joints from sub-robots

# Applying the Robot Schema

All robots in Isaac Sim’s library and imported through [URDF Importer Extension](Importers_and_Exporters.md) and [MJCF Importer Extension](Importers_and_Exporters.md) will have the Robot Schema applied to them. For robots imported in prior versions of Isaac Sim, the schema will need to be applied manually. To do so, select the root prim of the robot, and in the right panel under the Properties tab, check the + Add button, and select `Isaac -> Robot Schema -> Robot API`. This will apply the robot schema to the root prim, and will automatically apply the Link API and Joint API to the child prims.

Properties for the Robot Schema will be displayed in the right panel under the Properties tab, in their appropriate API section in purple.

If the robot is updated over time, there are two options to update the schema: manually add the Link API to new bodies and the Joint API to new joints. Alternatively, apply the schema again to the root prim, which will automatically apply the Link API and Joint API to the child prims.

Note

When applying the schema to the robot, if your asset follows the [Asset Structure](Robot_Setup.md) guidelines, be sure to apply it either in the base layer of the robot asset or in a separate robot schema layer, and not directly in the interface layer. The auto-population will require the physics to be authored, so you can temporarily add physics as a sublayer to the base layer, and remove it after the schema is applied and before saving the asset.

## Applying the Robot Schema through code

The following snippet shows how to apply the robot schema through code in existing assets that do not currently have it. Following the [Asset Structure](Robot_Setup.md) guidelines, we recommend applying the schema in the base layer, or through a layer, so it remains separate from other payloads and is easier to update as the schema evolves. To use this script, open the asset you desire to add the schema to through the interface layer.

```python
import omni.usd
import pxr
import usd.schema.isaac.robot_schema as rs
from pxr import Sdf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
robot_asset_path = "/".join(stage.GetRootLayer().identifier.split("/")[:-1])  # Get the asset path from the stage
robot_asset = ".".join(
    stage.GetRootLayer().identifier.split("/")[-1].split(".")[:-1]
)  # Get the asset name from the stage
schema_asset = f"configuration/{robot_asset}_robot_schema.usda"
edit_layer = Sdf.Layer.FindOrOpen(f"{robot_asset_path}/{schema_asset}")
if not edit_layer:
    edit_layer = Sdf.Layer.CreateNew(f"{robot_asset_path}/{schema_asset}")
# Add sublayer to the stage, but as a relative path, only if not already present
if schema_asset not in stage.GetRootLayer().subLayerPaths:
    stage.GetRootLayer().subLayerPaths.append(schema_asset)
# Make all edits in the edit layer
with pxr.Usd.EditContext(stage, edit_layer):

    default_prim = stage.GetDefaultPrim()

    # Apply the Robot API to the default prim, and auto-populate the Links and Joints lists
    rs.ApplyRobotAPI(default_prim)

edit_layer.Save()
stage.Save()
```

# Parsing Robot Structure

The robot structure relies on the Physics Schema to define the robot kinematic tree. The robot schema extends the Physics Schema to include the robot composition information. To parse the robot structure, we need to first collect the Links and Joints that make up the robot, and then, from the Robot API, we start to build the robot structure from the first link on the Links list, iterating over the joints based on their connection to the next links. The robot structure should always be a tree. Loops in the hierarchy need to be flagged with the “Exclude from Articulation” attribute in the joints; otherwise, they will be arbitrarily broken during parsing, based on the depth-first search of the hierarchy. In the extension `isaacsim.robot.schema` we provide a set of utility scripts that parse the robot structure and output the Robot Kinematic tree based on the USD data.

## Example

1. In the Content Browser, drag and drop a UR10e robot Robots/UniversalRobots/ur10e/ur10e.usd into the stage.
2. On the Variant selection menu at the properties panel, select the Robotiq 2f-140 gripper variant.

1. Open the Script Editor in Window > Script Editor, and run the following script:

   ```python
   import omni.usd
   from pxr import Usd, UsdGeom

   # For legacy reasons, we need to import the schema from the usd.schema.isaac package
   from usd.schema.isaac import robot_schema

   stage = omni.usd.get_context().get_stage()
   prim = stage.GetPrimAtPath("/World/ur10e")

   robot_tree = robot_schema.utils.GenerateRobotLinkTree(stage, prim)

   robot_schema.utils.PrintRobotTree(robot_tree)
   ```

On the console, you should see the following output:

> 

```python
base_link
   shoulder_link
      upper_arm_link
         forearm_link
            wrist_1_link
               wrist_2_link
                  wrist_3_link
                     robotiq_base_link
                        left_outer_knuckle
                           left_outer_finger
                           left_inner_finger
                              left_inner_knuckle
                        right_outer_knuckle
                           right_outer_finger
                           right_inner_finger
                              right_inner_knuckle
```

Note how the gripper is included in the robot structure, even though it is not part of the UR10e robot. Select the UR10e prim on the stage, and check how the Robot Lists have `ee_link` listed.

## Robot Schema Utility Functions

1. `GetAllRobotJoints(stage, robot_link_prim, parse_nested_robots)`: Returns all joints of a robot.
2. `GetAllRobotLinks(stage, robot_link_prim, include_reference_points)`: Returns all links of a robot.
3. `GetJointBodyRelationship(joint_prim, bodyIndex)`: Gets the target link for joint’s body connection, by index.
4. `GetJointPose(robot_prim, joint_prim)`: Returns joint pose in robot’s coordinate system.
5. `GetLinksFromJoint(root, joint_prim)`: Returns lists of links before/after specified joint.
6. `GenerateRobotLinkTree(stage, robot_link_prim)`: Generates tree structure of robot links.
7. `PrintRobotTree(root, indent)`: Prints visual representation of robot link tree.

# Asset Structure

Following the guidelines for [Asset Structure](Robot_Setup.md), it is recommended to apply the schema on a separate layer and load it as a sublayer on the robot asset.

---

# Commands

You can run many of the UI commands through `omni.kit.commands.execute("CommandName", args)`. To find a list of available commands, and what args to use, open **Window > Commands**, then click on `Search Commands`. On the window that appears, you will find an extensive list of all the commands available, and their respective documentation. Each command comes from a source Extension, and enabling/disabling extensions will change the list of available commands.

More information can be found [Command History](https://docs.omniverse.nvidia.com/extensions/latest/ext_command-history.html)

# Registered Actions

An action is a pre-defined sequence of API and/or UI commands. Open the **Utilities > Registered Actions** to see a list of all the registered actions. The actions are registered by the extensions, and enabling/disabling extensions will change the list of available actions. Double clicking on the action name will execute the action.

You can also call these functions from Python scripts when using the `onclick_action` variable.

You can create your own actions using [Kit Action API](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/extensions_api.html#actions-api):

```python
import omni.kit.actions.core

action_registry = omni.kit.actions.core.get_action_registry()
action_registry.register_action(
    extension_id,
    action_name,
    action_function_callable,
)

# deregistered action at extension shutdown
action_registry.deregister_action(
    extension_id,
    action_name,
)
```