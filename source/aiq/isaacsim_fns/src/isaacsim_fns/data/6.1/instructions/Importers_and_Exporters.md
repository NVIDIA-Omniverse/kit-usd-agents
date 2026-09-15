# Importers and Exporters

## URDF

- URDF Importer Extension
- USD to URDF Exporter Extension

## MJCF

- MJCF Importer Extension

## CAD

- Onshape importer
- CAD Converter

## Shapenet (Deprecated)

- ShapeNet Importer

## Formats

- Formats

## Tutorials

- Importer and Exporter Tutorials Series

---

# URDF Importer Extension

Note

Starting from the Isaac Sim 2023.1.0 release, the URDF importer has been open-sourced.
Source code and information for contributing can be found at [our Github repository](https://github.com/isaac-sim/IsaacSim/tree/main/source/extensions/isaacsim.asset.importer.urdf).
As of Isaac sim 5.0, the former dedicated repository has been deprecated, and the code has been moved to the Isaac Sim repository.

The [URDF Importer Extension](#isaac-sim-urdf-importer) is used to import URDF representations of robots.
[Unified Robot Description Format (URDF)](http://wiki.ros.org/urdf/XML/model), is an XML format for representing a robot model in ROS.

To Import URDF files, go to the top menu bar and click **File > Import**.

This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)")
by searching for `isaacsim.asset.importer.urdf`.

Import results are logged in the **Output Log**, accessible from the bottom of the screen.
The **Output Log** will display any errors or warnings that occur during the import process. For more detailed log information, open Isaac Sim’s log file, change the console to Info mode, or start Isaac Sim with the parameter `--verbose` to display results in the terminal output.

Note

The Imported model follows the [Isaac Sim Asset Structure](Robot_Setup.md) convention, and the meshes are already instantiable to optimize performance.

## Conventions

Note

To comply with USD prim name conventions, special characters in link, joint, mesh names, and all other reference asset filenames are not supported and will be replaced with an underscore. In the event that the name starts with an underscore due to the replacement, an a is pre-pended. It is recommended to make these name changes in the URDF directly.

Refer to the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of NVIDIA Isaac Sim conventions.

## Import Options

**Model**: Provides the Options to Import in Stage, or add as a referenced model. If **Create in Stage** is selected. Choose the options to set as the default prim, and **Clear Stage on Import**. By default both are left unchecked.

**Links**: Choose between a **Moveable base** (for example, a wheeled robot) or a **Static base** (for example, a 6DoF robotic arm). If the robot is a moveable base, the base link will be set to moveable. If the robot is a static base, the base link will be fixed in place with a `root_joint`.

The **Default Density** is used for links that do not have a mass specified in the URDF. If the density is set to `0`, the physics engine will automatically compute the density with its default value.

## Joints and Drives

Provides an interface to configure individual joints and is loaded with the default values.

**Ignore Mimic**: If checked, the Mimic tag will be ignored on import. Otherwise joints with the mimic tag will receive the PhysX Mimic API, allowing it to work in tandem with the primary joint that is defined in its setup.

**Joint Configuration**: Choose between configuring the joints directly through stiffness or with natural frequency. Saved values will always be in stiffness.

> * **Stiffness**: Edit the joint drive stiffness and damping directly.
>
>   > The stiffness value is used to control the strength of the position drive. A combination of setting stiffness and damping on a drive will result in both targets being applied, this can be useful in position control to reduce vibrations.
> * **Natural Frequency**: Computes the joint drive stiffness and Damping ratio based on the desired natural frequency using the formula:
>
>   > \[Kp = m \omega\_n^2, Kd = 2 m \zeta \omega\_n\]
>   >
>   > where \(\omega\_n\) is the natural frequency, \(\zeta\) is the damping ratio, and \(m\) is the total equivalent inertia at the joint.
>   > The damping ratio is such that \(\zeta = 1.0\) is a critically damped system, \(\zeta < 1.0\) is underdamped, and \(\zeta > 1.0\) is overdamped.
> * **Multi-Edit Edit**: To Edit multiple joints at the same time, you can ctrl+click at their names, to select individual joints, or shift+click to select a range of joints. After selected, the values will be applied to all selected joints.

**Drive Type**: The drive type can be chosen between **Acceleration** and **Force**. Acceleration drives normalize the inertia before applying the effort, making it invariant to changes in robot mass (payload not included), equivalent to ideal damped actuator. In force drives, the effort is applied directly to the joint, equivalent to a spring-damper system.

**Target**: Can be chosen between **None**, **Position**, and **Velocity**. If the drive type is set to position, the target will be the position in radians for revolute joints, or distance units for prismatic. For velocity drives, it’s the unit per second. When the joint is configured as **Mimic** you cannot change the **Target Type**.

**Colliders**:

> * **Collision From Visuals**: If checked, the collision objects will be created from the visual meshes when a collision object is not provided. Otherwise, no collision will be created for that link.
> * **Collider Type**: Select between:
>   :   + **Convex Hull** will create a single convex hull around the collision mesh.
>       + **Convex Decomposition** will create multiple convex hulls around the collision mesh to better match the visual asset.
> * **Allow self-collision**: Enables self collision between adjacent links. It may cause instability if the collision meshes are intersecting at the joint.
> * **Replace Cylinders with Capsules**: When selected, cylinder colliders will be replaced with capsule primitives.
>
> Note
>
> * It is recommended that you set Self Collision to false unless you are certain that links on the robot are not self colliding.
> * You must have write access to the output directory used for import, it will default to the current open stage, change this as necessary.

## Importing URDF from a ROS 2 Node

Enable the extension `isaacsim.ros2.urdf` to enable this feature. This will open a standalone URDF importer UI that allows to define a ROS 2 Node containing a robot description.

To select the appropriate node, type in the name of the node in the `Node` text box. If changes were made to the import settings, or to the published node hit Refresh. If the node name is in

Note

This feature is only available when the ROS 2 bridge is enabled.

For more on how to use the ROS 2 URDF Importer, refer to the [Import from ROS 2 Node](Importers_and_Exporters.md) Tutorial.

## Robot Properties

There might be many properties you want to tune on your robot.
These properties can be spread across many different schemas and APIs.

The general steps of getting and setting a parameter are:

1. Find which API is the parameter under. Most common ones can be found in the [Pixar USD API](https://docs.omniverse.nvidia.com/usd/latest/index.html).
2. Get the prim handle that the API is applied to. For example, articulation and drive APIs are applied to joints, and MassAPIs are applied to the rigid bodies.
3. Get the handle to the API. From there on, you can Get or Set the attributes associated with that API.

For example, if you want to set the wheel’s drive velocity and the actuators’ stiffness, find the DriveAPI:

```python
# get handle to the Drive API for both wheels
left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/left_wheel"), "angular")
right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/right_wheel"), "angular")

# Set the velocity drive target in degrees/second
left_wheel_drive.GetTargetVelocityAttr().Set(150)
right_wheel_drive.GetTargetVelocityAttr().Set(150)

# Set the drive damping, which controls the strength of the velocity drive
left_wheel_drive.GetDampingAttr().Set(15000)
right_wheel_drive.GetDampingAttr().Set(15000)

# Set the drive stiffness, which controls the strength of the position drive
# In this case because we want to do velocity control this should be set to zero
left_wheel_drive.GetStiffnessAttr().Set(0)
right_wheel_drive.GetStiffnessAttr().Set(0)
```

Alternatively you can use the [Omniverse Commands Tool Extension](Debugging_Profiling.md) to change a value in the UI and get the associated Omniverse command that changes the property.

Note

* The drive stiffness parameter should be set when using position control on a joint drive.
* The drive damping parameter should be set when using velocity control on a joint drive.
* A combination of setting stiffness and damping on a drive will result in both targets being applied, this can be useful in position control to reduce vibrations.

## Custom Isaac Sim URDF Attributes and Tags

### sensor.isaac\_sim\_config

This this attribute is used in the sensor tag to provide Isaac Sim configuration for Sensors. There are two possible uses:

> * preconfigured Lidars that are shipped with Isaac Sim
> * user-defined configurations. When it’s used with a user-defined configuration, the location of the configuration JSON must be provided, otherwise provide the configuration name for a preconfigured Lidar. A sample configuration file is provided in the tests provided with the URDF Importer in `data/lidar_sensor_template`.
>
>   > Note
>   >
>   > When using a custom Lidar configuration, the importer will try to create a symlink to the configuration in the isaacsim.sensors.rtx` folder. If you get Error Code: 1314 on Windows try running Isaac Sim with Administrator Priviledges, or manually create the Symbolic Link post-import. Alternatively, add the imported asset path into the lookup folders for isaacsim.sensors.rtx. If you get Error Code: 183 on Windows, the symbolic link already exists, double check and replace manually if necessary.

#### Example

```python
 1<robot>
 2    <link name="root_link"/>
 3    <joint name="root_to_base" type="fixed">
 4        <parent link="root_link"/>
 5        <child link="link_1"/>
 6    </joint>
 7    <link name="link_1"/>
 8
 9    <sensor name="custom_lidar" type="ray" update_rate="30" isaac_sim_config="../lidar_sensor_template/lidar_template.json">
10        <parent link="link_1"/>
11        <origin xyz="0.5 0.5 0" rpy="0 0 0"/>
12    </sensor>
13
14    <sensor name="preconfigured_lidar" type="ray" update_rate="30" isaac_sim_config="Velodyne_VLS128">
15        <parent link="link_1"/>
16        <origin xyz="0.5 1.5 0" rpy="0 0 0"/>
17    </sensor>
18</robot>
```

### loop\_joint

Defines a joint to close kinematic chain loops. This is useful for robots with closed kinematic chains, such as a quadruped robot with a loop joint at the hip. The loop joint is defined in the URDF as follows:

```python
1<loop_joint name="loop_joint_name" type="spherical">
2    <link1 link="link_1" rpy="0 0 0" xyz="0 0 0"/>
3    <link1 link="link_2" rpy="0 0 0" xyz="0 0 0"/>
4</loop_joint>
```

### fixed\_frame

Fixed frames are used to define a reference point attached to a link. This is useful to define reference points (for example, sensor placements or end-effector offset) without using the link tag. The fixed frame is defined in the URDF as follows:

```python
1<fixed_frame name="frame_0">
2    <parent link="link_1"/>
3    <origin rpy="0.0 0.0 0.0" xyz="1.00 -0.020 0.10"/>
4</fixed_frame>
```

Fixed frames must have an exclusive name and parent link pair.

## References

Refer to the [Asset Structure](Robot_Setup.md) for more information about the asset structure.

## Examples

For usage examples, refer to the [Tutorial: Import URDF](Importers_and_Exporters.md) .

---

# USD to URDF Exporter Extension

## Overview

The USD to URDF Exporter is a tool to convert a USD stage or file to a [URDF](http://wiki.ros.org/urdf/XML/model) file.
A user just needs to open a stage with the desired USD they want to export, and provide the path to directory where the new URDF file will be saved, or the path to the new URDF file directly.

To enable this extension, open the **Extension Manager** window by navigating to **Window** > **Extensions**, and enable `isaacsim.asset.exporter.urdf`.

Once enabled, the USD to Exporter is accessed by going to the top Menu Bar and clicking **File** > **Export to URDF**. Mesh files are saved by default to a meshes directory, which is placed in the same directory as the new URDF file.
Additional options are available that allow customization of some parts of the conversion.

## Parameters

**Output File/Directory**

The file path for the new URDF file.
The file path conventionally ends with the extension `.urdf`.

Or a directory path where new URDF file will be saved.
The new URDF file will have the same name as the USD.

**Mesh Directory Path**

The directory where the meshes will be saved for the URDF (defaults to the same directory as the as where the URDF file is saved).

**Mesh Path Prefix**

A prefix to apply to each mesh filename.
For example, to set the mesh file paths to valid URI with the file scheme, set the mesh path prefix to `file://`.

**Root Prim Path**

The root prim within the USD stage of the kinematic tree to be exported to URDF. The default is the default prim of the USD file.

**Visualize Collisions**

If set, the collision meshes are included as visual meshes in the resulting URDF.

## Notes

* URDF files do not support kinematic loops. If the USD file has a kinematic loop, the converter will fail. Try manually break the loop first.
* URDF files require a parent link and a child link when defining a joint, they do not allow having a joint with a single link. If there are such joints in the USD file, for example, an unattached end-effector, the converter will likely produce inconsistent joint transforms and fail.

---

# MJCF Importer Extension

Note

Starting from the Isaac Sim 2023.1.0 release, the MJCF importer has been open-sourced.
Source code and information for contributing can be found at [our Github repository](https://github.com/isaac-sim/IsaacSim/tree/main/source/extensions/isaacsim.asset.importer.mjcf).
As of Isaac sim 5.0, the former dedicated repository has been deprecated, and the code has been moved to the Isaac Sim repository.

The [MJCF Importer Extension](#isaac-sim-mjcf-importer) Extension is used to import MuJoCo representations of robots.
[MuJoCo Modeling XML File (MJCF)](https://mujoco.readthedocs.io/en/latest/modeling.html), is an XML format for representing a robot model in the MuJoCo simulator.

To access this extension, go to the top menu bar and click **File > Import**.

This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)")
by searching for `isaacsim.asset.importer.mjcf`.

## Conventions

Note

Special characters in link or joint names are not supported and are replaced with an underscore. In the event that the name starts with an underscore due to the replacement, an a is pre-pended. It is recommended to make these name changes in the MJCF directly.

Refer to the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of NVIDIA Isaac Sim conventions.

### User Interface

## Import Options

**Model**: Provides the Options to Import in Stage, or add as a referenced model. If Create in Stage is selected. Choose the options to Set as the default prim, and Clear Stage on Import. By default both are left unchecked.

**Links**: Choose:

> * **Moveable base** (for example, a wheeled robot) the base link will be set to moveable.
> * **Static base** (for example, a 6DoF robotic arm) the base link will be fixed in place with a `root_joint`.

The **Default Density** is used for links that do not have a mass specified in the URDF. If the density is set to `0`, the physics engine will automatically compute the density with its default value.

**Colliders**:

> * **Visualize Collision Geometry**: When selected, the collision geometry will be visible in the viewport.
> * **Allow self-collision**: Enables self collision between adjacent links. It might cause instability if the collision meshes are intersecting at the joint.
>
> Note
>
> * It is recommended that you set Self Collision to false unless you are certain that links on the robot are not self colliding
> * You must have write access to the output directory used for import, it will default to the current open stage, change this as necessary.

### Robot Properties

There might be many properties you want to tune on your robot.
These properties can be spread across many different Schemas and APIs.

The general steps of getting and setting a parameter are:

1. Find which API the parameter is under. Most common ones can be found in the [Pixar USD API](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/104.0/api/pxr_index.html).
2. Get the prim handle that the API is applied to. For example, Articulation and Drive APIs are applied to joints, and MassAPIs are applied to the rigid bodies.
3. Get the handle to the API. From there on, you can Get or Set the attributes associated with that API.

For example, if you want to set the wheel’s drive velocity and the actuators’ stiffness, you must find the DriveAPI:

```python
# get handle to the Drive API for both wheels
left_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/left_wheel"), "angular")
right_wheel_drive = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/carter/chassis_link/right_wheel"), "angular")

# Set the velocity drive target in degrees/second
left_wheel_drive.GetTargetVelocityAttr().Set(150)
right_wheel_drive.GetTargetVelocityAttr().Set(150)

# Set the drive damping, which controls the strength of the velocity drive
left_wheel_drive.GetDampingAttr().Set(15000)
right_wheel_drive.GetDampingAttr().Set(15000)

# Set the drive stiffness, which controls the strength of the position drive
# In this case because we want to do velocity control this should be set to zero
left_wheel_drive.GetStiffnessAttr().Set(0)
right_wheel_drive.GetStiffnessAttr().Set(0)
```

Alternatively you can use the [Omniverse Commands Tool Extension](Debugging_Profiling.md) to change a value in the UI and get the associated Omniverse command that changes the property.

Note

* The drive stiffness parameter should be set when using position control on a joint drive.
* The drive damping parameter should be set when using velocity control on a joint drive.
* A combination of setting stiffness and damping on a drive will result in both targets being applied, this can be useful in position control to reduce vibrations.

### References

Refer to the [Asset Structure](Robot_Setup.md) for more information about the asset structure.

### Tutorial

Review [Tutorial: Import MJCF](Importers_and_Exporters.md).

---

# ShapeNet Importer

Warning

[DEPRECATED]: omni.isaac.shapenet is deprecated. You can import ShapeNet models as you would any other OBJ file.

The [ShapeNet Importer](#shapenet) Extension allowed import and conversion of 3D models from the [ShapeNet V2 database](https://shapenet.org/) into Omniverse as [USD](Glossary.md) files.

---

# Formats

The standard format used in Omniverse is USD for scenes and MDL for materials. You need to convert your content to be usable in Omniverse, if coming from external applications. Omniverse offers several ways to manage such content.

## Asset Converter

Apps in Omniverse are loaded with the Asset Converter extension. With it, you can convert models into USD using the [Asset Converter](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html "(in Omniverse Extensions)") service. Below is a list of formats it can convert to USD.

| Format | Name | Description |
| --- | --- | --- |
| `.fbx` | Autodesk FBX Interchange File | Common 3D model saved in the Autodesk Filmbox format |
| `.obj` | Object File Format | Common 3D Model format |
| `.gltf` | GL Transmission Format File | Common 3D Scene Description |

## Materials

NVIDIA has developed a custom schema in USD to represent material assignments and specify material parameters. In Omniverse, these specialized USD’s get an extension change to `.MDL` signifying that it is represented in NVIDIA’s open-source MDL (Material Definition Language).

---

# Importer and Exporter Tutorials Series

This series of tutorials walks you through the process of importing and exporting assets in Isaac Sim.

- Tutorial: Import URDF
- Tutorial: Export URDF
- Tutorial: Import MJCF
- Tutorial: ShapeNet Importer

---

# Tutorial: Import URDF

## Learning Objectives

This tutorial shows how to import a URDF and convert it to a USD in NVIDIA Isaac Sim.
After this tutorial, you can use URDF files in your pipeline while using NVIDIA Isaac Sim.

*10-15 Minutes Tutorial*

## Getting Started

**Prerequisites**

* Review the [Quick Tutorials](Quick_Tutorials.md) prior to beginning this tutorial.
* Check the [URDF Importer Extension](Importers_and_Exporters.md) for more details on the extension.

Direct Import

To import a Franka Panda URDF from the *Built in URDF* files that come with the extension:

1. Enable the isaacsim.asset.importer.urdf extension in NVIDIA Isaac Sim if it is not automatically loaded by going to **Window > Extensions** and enable isaacsim.asset.importer.urdf.

   > * In this example, import the panda\_arm\_hand.urdf that is included in the URDF importer extension. To find it:
   >   :   + Click on the file icon beside *AUTOLOAD* to find the *isaacsim.asset.importer.urdf* extension.
   >       + Navigate to `/data/urdf/robots/franka_description/robots` and find `panda_arm_hand.urdf`, and copy this path.
2. Accesses the URDF extension by going to the **File > Import**, and select an URDF file you want to import. In this case, paste the path above to the navigation bar and left-click on **panda\_arm\_hand.urdf**.

   > 
3. Specify the settings you want to use to import Franka with:

   > * Set USD Ouptut to your desired output location for the USD.
   > * Select **Static Base** and leave **Default Density** empty.
   > * Refer to [urdf importer Robot Properties](Importers_and_Exporters.md) for joints and drive instructions. In this tutorial, increase the natural frequencies of the joints to reduce oscillations during movement.
   > * Select **Allow Self-Collision** for the Colliders section and leave everything else as default.
   >
   > Note
   >
   > You must have write access to the output directory used for import, it will default to the same directory as your URDF.
4. Click the **Import** button to add the robot to the stage.

   > 

#. Visualize the collision meshes, not all the rigid bodies need to have collision properties, and collision meshes are often a simplified mesh compared to the visual ones. Therefore you might want to visualize the collision mesh for inspection.
To visualize collision in any viewport:

> * **Select**: the eye icon in the upper left corner of the viewport.
> * **Select**: Show by type.
> * **Select**: Physics.
> * **Select**: Colliders.
> * **Check**: All.
>
> 

Note

If you are importing a mobile robot, you might need to change the following settings:

* Select [Moveable Base](Importers_and_Exporters.md).
* Set the joint drive type to **Velocity** drive for the velocity controlled joints (that is, wheels), and **Position** for the position controlled joints (that is, steering joint).
* Set the **Joint Drive Strength** to the desired level. This will be imported as the joint’s damping parameter. Joint stiffness are always set to zero in velocity drive mode.

Note

If you are importing a torque controlled mobile robot such as a quadruped:

* Select [Moveable Base](Importers_and_Exporters.md).
* Set the joint drive type to **None** drive for the torque controlled joints (that is, legs), and **Position** or **Velocity** for the position or velocity controlled joints.
* Set the **Joint Drive Strength** to the desired level. For the torque controlled drives, stiffness and damping have no effect and will be imported as zero.

UI Integration Examples

Activate **Windows** > **Examples** > **Robotics Examples**, which will open the **Robotics Examples** tab at the bottom dock.

Note

For these examples, wait for materials to get loaded.
You can track progress on the bottom right corner of the UI.

There are Four examples available in the **Import Robots** section:

* **Nova Carter URDF**
* **Franka URDF**
* **Kaya URDF**
* **UR10 URDF**

Each one of them contains an individual import configuration and post import setup in code, but overall the usage is similar:

1. Go to the **Robotics Examples** tab and navigate to **Import Robots > <Robot> URDF**.
2. Press the **Load Robot** button to import the URDF into the stage, add a ground plane, add a light, and a physics scene.
3. Press the **Configure Drives** button to configure the joint drives. This sets each drive stiffness and damping value.
4. Press the **Open Source Code** button to view the source code. The source code illustrates how to import and integrate the robot using the Python API.
5. Press the **PLAY** button to begin simulating.
6. Press the **Move to Pose** button to make the robot move to a home or rest position.

Python Script

Use Python scripting to do what can be done through the Import window. Then use the imported robot with one of
the tasks defined under **isaacsim.robot.manipulators.examples.franka** extension to follow a target in the stage.

1. Open the **Hello World** example.
   :   * Go to the top Menu Bar and click **Window > Examples > Robotics Examples**.
       * In the **Robotics Examples** tab at the bottom, select **General > Hello World**.
2. Validate that the window for the *Hello World* example extension is in the workspace.
3. Click the **Open Source Code** button to launch the source code for editing in [Visual Studio Code](https://code.visualstudio.com/download).
4. Edit the `hello_world.py` file as shown below:

```python
import omni.kit.commands
import omni.usd
from isaacsim.asset.importer.urdf import _urdf
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.examples.interactive.base_sample import BaseSample
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.examples.franka.tasks import FollowTarget

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        # Get the world object to set up the simulation environment
        world = self.get_world()

        # Add a default ground plane to the scene for the robot to interact with
        world.scene.add_default_ground_plane()

        # Acquire the URDF extension interface for parsing and importing URDF files
        urdf_interface = _urdf.acquire_urdf_interface()

        # Configure the settings for importing the URDF file
        import_config = _urdf.ImportConfig()
        import_config.convex_decomp = False  # Disable convex decomposition for simplicity
        import_config.fix_base = True  # Fix the base of the robot to the ground
        import_config.make_default_prim = True  # Make the robot the default prim in the scene
        import_config.self_collision = False  # Disable self-collision for performance
        import_config.distance_scale = 1  # Set distance scale for the robot
        import_config.density = 0.0  # Set density to 0 (use default values)

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = extension_path + "/data/urdf/robots/franka_description/robots"
        file_name = "panda_arm_hand.urdf"

        # Parse the robot's URDF file to generate a robot model
        result, robot_model = omni.kit.commands.execute(
            "URDFParseFile", urdf_path="{}/{}".format(root_path, file_name), import_config=import_config
        )

        # Update the joint drive parameters for better stiffness and damping
        for joint in robot_model.joints:
            robot_model.joints[joint].drive.strength = 1047.19751  # High stiffness value
            robot_model.joints[joint].drive.damping = 52.35988  # Moderate damping value

        # Import the robot onto the current stage and retrieve its prim path
        result, prim_path = omni.kit.commands.execute(
            "URDFImportRobot",
            urdf_robot=robot_model,
            import_config=import_config,
        )

        # Optionally, import the robot onto a new stage and reference it in the current stage
        # (Useful for assets with textures to ensure textures load correctly)
        # dest_path = "/path/to/dest.usd"
        # result, prim_path = omni.kit.commands.execute(
        #     "URDFParseAndImportFile",
        #     urdf_path="{}/{}".format(root_path, file_name),
        #     import_config=import_config,
        #     dest_path=dest_path
        # )
        # prim_path = omni.usd.get_stage_next_free_path(
        #     self.world.scene.stage, str(current_stage.GetDefaultPrim().GetPath()) + prim_path, False
        # )
        # robot_prim = self.world.scene.stage.OverridePrim(prim_path)
        # robot_prim.GetReferences().AddReference(dest_path)

        # Initialize a predefined task for the robot (for example, following a target)
        my_task = FollowTarget(
            name="follow_target_task",
            franka_prim_path=prim_path,  # Path to the robot's prim in the scene
            franka_robot_name="fancy_franka",  # Name for the robot instance
            target_name="target",  # Name of the target object the robot should follow
        )

        # Add the task to the simulation world
        world.add_task(my_task)
        return

    async def setup_post_load(self):
        # Set up post-load configurations, such as controllers
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")

        # Initialize the RMPFlow controller for the robot
        self._controller = RMPFlowController(name="target_follower_controller", robot_articulation=self._franka)

        # Add a physics callback for simulation steps
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        # Reset the controller to its initial state
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        # Perform a simulation step and compute actions for the robot
        world = self.get_world()
        observations = world.get_observations()

        # Compute actions for the robot to follow the target's position and orientation
        actions = self._controller.forward(
            target_end_effector_position=observations["target"]["position"],
            target_end_effector_orientation=observations["target"]["orientation"],
        )

        # Apply the computed actions to the robot
        self._franka.apply_action(actions)
        return
```

1. Press `Ctrl+S` to save the code and hot-reload NVIDIA Isaac Sim.
2. Click the **File > New From Stage Template > Empty** to create a new stage. Click **Don’t Save** if the simulator is prompting you to save the stage.
3. Open the menu again and load the example.
4. Click the **LOAD** button and move the target prim around to observe the robot follow it.

   > 

Import from ROS 2 Node

Importing a URDF through a ROS 2 node is a powerful way to integrate NVIDIA Isaac Sim with your existing ROS 2 workflow. This allows you to import a URDF from a ROS 2 node and use it in NVIDIA Isaac Sim, also indirectly enabling importing XACRO definitions without explicit conversion to URDF.

Note

This tutorial is only supported on Linux and for Isaac Sim (while it may be possible to run in other Omniverse Applications, it is not covered by this tutorial and the extension may not work as expected).

**Prerequisites**

* [ROS 2](ROS_2.md)
* A ROS 2 workspace with a robot description (for example [Universal Robots ROS 2 Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) ).
* Follow the tutorials on how to [set up a ROS 2 workspace (Humble)](https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Creating-A-Workspace/Creating-A-Workspace.html) and include a robot description like the one in this example, along with all its dependencies.

**Steps**

* Terminal 1
  :   + Source ROS 2
      + Launch a transform publisher for the robot description node (for example `ros2 launch ur_description view_ur.launch.py ur_type:=ur10e`).
* Terminal 2
  :   + Source ROS 2
      + Pick the ROS 2 node name for the node just created with `ros2 node list`. For example, `robot_state_publisher`.
* Terminal 3
  :   + Source ROS 2
      + Start Isaac Sim
      + Enable the extension `isaacsim.ros2.urdf`
      + Open the URDF Importer using the **File > Import from ROS 2 URDF Node** menu
      + Put the node name in the text box
      + Define an output directory
      + Import

**Extra steps to try:**

* Terminal 1
  :   + Stop the publisher, change it to another robot and start service again (for example, `ros2 launch ur_description view_ur.launch.py ur_type:=ur3`)
* Terminal 3
  :   + Click the **Refresh** button
      + Change the output directory
      + Import

The robot is now imported into the stage. You can now use it in your simulation. You can perform additional changes to the asset after it’s imported, such as adding sensors, changing materials, and updating the joint drives and configuration to achieve a more stable simulation. Robots are mapped as Articulations in the simulation, and for a complete guide in tuning articulations, refer to [Articulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html).

## Summary

This tutorial covered the following topics:

1. Importing URDF file using GUI
2. Importing URDF file using Python
3. Importing URDF file using a ROS Node
4. Using the imported URDF in a Task
5. Visualizing collision meshes
6. Setting up importing a robot with the UI through the built-in examples

### Further Learning

Checkout [URDF Importer Extension](Importers_and_Exporters.md) to learn more about the different configuration settings to import a URDF in NVIDIA Isaac Sim.

---

# Tutorial: Export URDF

## Learning Objectives

This tutorial explores exporting a URDF file from USD in NVIDIA Isaac Sim.
After this tutorial, you will be able to convert robot USD files to URDF files using NVIDIA Isaac Sim.

*10-20 Minute Tutorial*

## Getting Started

**Prerequisites**

* Review the [Quick Tutorials](Quick_Tutorials.md) prior to beginning this tutorial.

## Exporting A Robot

To convert a robot USD file to a URDF file and cover some advanced options:

### Enable the Exporter Extension

To enable the exporter extension:

1. Navigate to **Windows > Extensions** and type `urdf` in the search bar, then enable the USD to URDF exporter extension.

   This will add the **File > Export to URDF** menu option.
2. Select it to open the extension, verify that the user interface is similar to:

   > 
3. Open the USD for the Franka robot, which is found in the Isaac asset root path at `/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`.

   > After the USD finishes loading:
4. Open the **File > Export to URDF** menu option. A select file dialog will appear.
5. Select the destination file and folder.
6. Click **Export**.

   > 
7. Open the output folder to view the resulting files.
8. Verify that a `franka.urdf` file and a `meshes` directory is present. The `meshes` directory contains the mesh files for the robot.

> To check the results:
>
> > * The URDF can be imported back to USD and opened in Isaac Sim. Refer to the [Import URDF](Importers_and_Exporters.md) tutorial for the steps to do that.
> > * Review the results with this [URDF Viewer Example](https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/index.html) website. Drag the output directory directly into the site to view the URDF file, and examine the joints.

### Import Options

#### Mesh Folder Name

The folder name for where the mesh `.obj` files are saved. Defaults to the name `meshes`, and is placed in the same directory as where the URDF file is saved.

> 

#### Mesh Path Prefix

There are three options for the mesh path prefix:

1. Absolute path (default), defined by the prefix `file://`
2. package path, defined by the prefix `package://`
3. Relative path, defined by the prefix `./`

When using the `package://` prefix, the package name needs to be specified in the `Package Name` field. If left blank, the package name will be the name of the urdf file.

#### Root Prim Path

If you are exporting a robot directly from its asset file, the default prim would be the root prim for it, but if exporting from a scene that contains a robot and multiple other objects, you can elect to export only the robot by specifying which prim represents it.

#### Collision Objects

In a URDF file, a link often has two separate meshes associated with it:

* a visual mesh
* a collision mesh

In USD there is no distinction between a visual mesh and a collision mesh.
USD prims can have the `PhysicsCollisionAPI` attached to them, which tells the physics engine to resolve the motion of the body as it touches other bodies.
Additionally, prims can be set to be visible or invisible.
The USD to URDF exporter creates visual meshes and collision meshes for each link based on if it has the `PhysicsCollisionAPI` applied to it and if it is visible.

To explore how geometry prims map to visual and collision meshes in the URDF, add a geometry prim to the Franka robot and export it in different ways to verify what is created with each of the resulting URDF files.

1. Open the USD for the Franka robot (found at `/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd`).
2. Right click the `panda_hand` Xform prim, and from the contextual menu select **Create > Mesh > Sphere**.
3. Select the new `Sphere` Mesh prim, and change the scaling for the x, y, and z components to all be `0.3`.
4. Verify that the Franka is similar to:

   > 
5. Export your current stage by following the steps discussed above and outlined below (there is no need to save your changes).

   > * Open the **USD to URDF Exporter** menu.
   > * Select an output directory.
   > * Press the **EXPORT** button.
6. Drag your output directory into the [URDF Viewer Example](https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/index.html) website to view the results.
7. Verify that your results are similar to:

   > 
8. Enable the **Show Collision** option, which visualizes all the collisions meshes and highlights them with a gold color.
9. Observe how the sphere is not highlighted with the gold color, that is because in the URDF it is not a collision mesh it is a visual mesh.
10. Back in the Franka USD, add the collision API to the sphere.

    This can be done by selecting the `Sphere` prim and clicking the **+Add** button in the prim’s property menu.
11. Select **Physics > Colliders Preset**.
12. After adding the collision API to the sphere, re-export the USD stage to URDF and drag the output directory into the URDF viewer again. You might need to refresh the viewer’s webpage before dragging in the new URDF.
13. Verify that your Franka is similar to:

    > 
14. Enable the **Show Collision** option.
15. Observe that this time the sphere is highlighted with the gold color.

    > That is because the sphere is both a collision mesh and a visual mesh in the URDF file.
16. Back in the Franka USD, make the sphere invisible by disabling the “eye” icon next to the `Sphere` prim.

    > 
17. After making the sphere invisible, re-export the USD stage to URDF and drag the output directory into the URDF viewer again.

    > Initially the sphere is not there, but after enabling **Show Collision**, the sphere is highlighted with the gold color.
    > This is because the sphere is a collision mesh, but not a visual mesh in the URDF file.
18. Verify that you have something similar to:

> 

To export link collision meshes correctly to URDF, they must have the collision API and must be set to invisible.
To make all collision API prims into visual meshes, regardless of the visibility state of the prim, enable the `Visualize Collisions` option under the advanced options of the USD to URDF Exporter.

## Limitations

The USD format offers much greater expressiveness and provides more capabilities compared to URDF.
The set of all scenes and robots that can be described using USD is a superset to those that can be described with URDF.
Meaning all scenes and robots that can be described by a URDF file can also be described by a USD file, but not vice versa.
Therefore, there is no direct one-to-one mapping between USDs and URDFs.
Consequently, when converting a USD file to a URDF file, several assumptions are made and constraints are imposed.

Here is list of constraints for the USD in order for the USD to URDF exporter to succeed.

* The kinematic structure of the robot must be a tree structure
* Scaling on sphere shapes must be the same for every axis
* Scaling on cylinder shapes must be the same for radius axes (that is, the non-height axes)
* The coordinates for each body frame of a joint must be co-located and aligned
* Parent link prims should be `Body 0`, and child link prims should be `Body 1` of the joint
* Joint prims must be either `prismatic`, `revolute`, or `fixed`
* Link prims must be `Xform`.
* Sensor prims must be either `Camera` or `IsaacImuSensor`
* Geometry prims must be either `Cube`, `Sphere`, `Cylinder`, or `Mesh`
* Geometry prims must be “leafs” in the kinematic tree

If your USD violates one of these constraints an error is thrown.

Note

Depending on the robot structure, some body names might be overridden because of the merging of different frames. Review the output and verify that it is accurate.

## Summary

This tutorial covered the following topics:

1. Exporting URDF files using the exporter GUI
2. Validating the URDF result by viewing in a viewer
3. Understanding how collision and visual meshes in the URDF are controlled from the USD
4. Outline the limitation of the USD to URDF exporter

### Further Learning

Review [USD to URDF Exporter Extension](Importers_and_Exporters.md) to learn more about other configuration options.

---

# Tutorial: Import MJCF

## Learning Objectives

This tutorial shows how to import a MJCF model and convert it to a USD in NVIDIA Isaac Sim.
After this tutorial, you can use MJCF files in your pipeline while using NVIDIA Isaac Sim.

*5-10 Minute Tutorial*

## Getting Started

**Prerequisites**

* Review the [Quick Tutorials](Quick_Tutorials.md) prior to beginning this tutorial.

## Using the MJCF Importer

Begin by importing an Ant MJCF from the *Built in MJCF* files that come with the extension.

1. Load the MJCF Importer extension, which should be automatically loaded when NVIDIA Isaac Sim opens and can be accessed from the **File** > **Import** menu. If not MJCF files are not listed in the import formats, go to **Window** > **Extensions** and enable `isaacsim.asset.importer.mjcf`.
2. In the file selection dialog box, navigate to the desired folder, and select the desired MJCF file. For this example, use the Humanoid `nv_humanoid.xml` file that comes with this extension, included in the extension assets. To find it:

   > * Click on the folder icon beside *AUTOLOAD* to find the **isaacsim.asset.importer.mjcf** extension.
   > * Navigate to `/data/mjcf` and find `nv_humanoid.xml`.
3. Change the import options according to the your needs. Check [Import Options](Importers_and_Exporters.md) for more information on the import options.

   > 
4. Click the **Import** button to add the robot to the stage.

   > 

The robot is now imported into the stage. You can now use it in your simulation. You can perform additional changes to the asset after it’s imported, such as adding sensors, changing materials, and updating the joint drives and configuration to achieve a more stable simulation. Robots are mapped as articulations in the simulation, and for a complete guide in tuning articulations, refer to [Articulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html).

## Importing MJCF Using Python

Do the exact same thing with Python scripting instead.

1. Open the **Script Editor**. Go to the top Menu Bar and click **Window > Script Editor**.
2. The window for the **Script Editor** is visible in the workspace.
3. Copy the following code into the **Script Editor** window.

   > ```python
   > import omni.kit.commands
   > from pxr import Gf, PhysicsSchemaTools, Sdf, UsdLux, UsdPhysics
   >
   > # create new stage
   > omni.usd.get_context().new_stage()
   >
   > # setting up import configuration:
   > status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
   > import_config.set_fix_base(False)
   > import_config.set_make_default_prim(False)
   >
   > # Get path to extension data:
   > ext_manager = omni.kit.app.get_app().get_extension_manager()
   > ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.mjcf")
   > extension_path = ext_manager.get_extension_path(ext_id)
   >
   > # import MJCF
   > omni.kit.commands.execute(
   >     "MJCFCreateAsset", mjcf_path=extension_path + "/data/mjcf/nv_ant.xml", import_config=import_config, prim_path="/ant"
   > )
   >
   > # get stage handle
   > stage = omni.usd.get_context().get_stage()
   >
   > # enable physics
   > scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
   >
   > # set gravity
   > scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
   > scene.CreateGravityMagnitudeAttr().Set(981.0)
   >
   > # add lighting
   > distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
   > distantLight.CreateIntensityAttr(500)
   > ```
4. Click the **Run (Ctrl + Enter)** button to import the Ant robot.

## Summary

This tutorial covered the following topics:

1. Importing MJCF file using GUI
2. Importing MJCF file using Python
3. Create a Ground Plane

### Further Learning

Review [MJCF Importer Extension](Importers_and_Exporters.md) to learn more about the different configuration settings to import a MJCF in NVIDIA Isaac Sim.

---

# Tutorial: ShapeNet Importer

Warning

[DEPRECATED]: omni.isaac.shapenet is deprecated. You can import ShapeNet models as you would any other OBJ file.