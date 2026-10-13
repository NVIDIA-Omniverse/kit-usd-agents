# Robot Setup

These tools and tutorials details the different importer and exporter for converting assets to and from the USD (Universal Scene Description) format and how you can build custom robots in the simulator.

## Tools

- Robot Wizard [Beta]
- Robot Wizard Tutorial
- Editor Tools
- Merge Mesh Utility
- Gain Tuner Extension
- Robot Assembler
- Inspector Tools
- Physics Inspector
- Simulation Data Visualizer

## Asset Structure

- Asset Structure

## Tutorials

- Robot Setup Tutorials Series

## Troubleshooting

- Robot Setup Troubleshooting

## Validation

- Asset Validation
- IsaacSim.PhysicsRules
- IsaacSim.RobotRules
- IsaacSim.SimReadyAssetRules
- Running the Validation Rules

---

# Robot Wizard [Beta]

The Robot Wizard is designed to speed up the process of setting up a robot in Isaac Sim. It allows you to define the robot’s hierarchy, organize the meshes, add colliders, joints and joint drives. It will automatically apply relevant Schemas and APIs without needing to manually edit the USD files. It separates the robot into different configurations based on the desired structure described in [Asset Structure](Robot_Setup.md). This is particularly useful if you are not familiar with the complexities of USD or the specific requirements for Sim-ready robots in Isaac Sim.

To enable the Robot Wizard, go to the **Window > Extensions** and enable **Isaac Sim Robot Wizard**. The Robot Wizard window can be toggled on or off inside the **Window > Robot Wizard** menu.

The Robot Wizard is in Beta and not fully functional for all use cases. It is recommended for CAD imports for basic robots with relatively few links and joints.

The following sections explain the UI and functions behind each step in the wizard. To observe the wizard in action, refer to [Robot Wizard Tutorial](Robot_Setup.md).

## Overview

The Robot Wizard guides you through the following steps:

^ **File Preparation**: Load the robot model and allocate folders and files for the final robot files.

* **Organize Link Hierarchy**: Define the robot’s hierarchy and link the parent-mesh child relationships.
* **Colliders**: Examine colliders to the robot’s links.
* **Joints and Drives**: Add joints to the robot’s links and drives and configure their properties.

The resulting files of the Robot Wizard are all placed in a folder, which you will have a chance to indicate in the wizard. The folder contains the following files:

* <robot\_root\_folder>/configurations:
  :   + The folder that contains the robot’s configurations USD files. The configurations are the different variants of the robot, such as a robot with or without physics, with sensors, and different end-effectors.
* <robot\_root\_folder>/configurations/<robot-name>\_base.usd
  :   + The configuration file that contains the robot’s base mesh and hierarchy.
* <robot\_root\_folder>/configurations/<robot-name>\_physics.usd
  :   + The configuration file that contains the robot’s physics setup in a sublayer. This includes the rigid body definition, colliders, joints, and drives.
* <robot\_root\_folder>/configurations/<robot-name>\_robot.usd
  :   + The configuration file that contains the robot schema, labeling the robot and its components.
* <robot\_root\_folder>/<robot-name>.usd:
  :   + The file that contains all the variants of the robot. In this case, the option of a robot with or without physics.

## Wizard Steps

### Page Orientation

| Ref # | Panel Name | Description |
| --- | --- | --- |
| 1 | Wizard Steps | The Wizard Steps panel shows your progress. You can click on each step to navigate to it. It will also advance itself as you go through the wizard. The names of the steps will change to green when it’s completed. |
| 2 | Additional Tools | You may open other tools for robot setup here. |
| 3 | Step Pages | Each step in the wizard has a page. You can navigate through the pages by clicking on the step name in the Wizard Steps panel. |
| 4 | Next Button | The Next button will advance you to the next step in the wizard. |
| 5 | Start Over | The Button will reset the wizard to the first step. |
| 6 | Launch On Startup | When checked, the wizard will launch automatically when Isaac Sim is started. It’s defaulted to not start. |
| 7 | Help | Open the documentation page for the wizard in your browser. |

### Add Robot

This page allows you the select the starting point of the robot configuration. For the current iteration, the robot wizard only supports configuring robots that are already loaded in the stage. If you are starting with a URDF or MJCF file, go to **File > Import** can use the importer instead.

**Steps:**

1. Select **Configure a Robot on Stage**.
2. Indicate the type of the robot you are configuring from the dropdown menu. Pick **custom** if your robot does not fit into the other categories. This will automatically populate the links that are frequently used in the selected robot types in [Robot Hierarchy](#isaac-sim-app-tutorial-wizard-hierarchy).
3. Give your robot a name. You can change it later.
4. Select the parent link of the robot from stage.
5. Click **Prepare Files** to advance to the next step.

### Prepare Files

This page allows you to indicate the folder where the robot files will be saved. The resulting files are described in [Overview](#isaac-sim-app-robot-wizard-file-structure). While no files are created at this step, they will be created in the subsequent steps.

**Steps:**

1. The folder will be created in the format of `<Root Folder>/configurations/<robot-name>_base.usd`. You can change the name and the root folder.
2. The stage that is currently open will not be the final robot file. You may choose to save a copy of it in the `<Root Folder>/stage_copy.usd`. If it has unsaved changes, you will also have the choice to save it and overwrite the existing path.
3. The **Robot Files Allocated** displays the filepaths that will be created in the folder. If filepaths text turned purple in color, it means that the file already exists, proceeding without changing the filepath will overwrite the existing file. If it turns red, it means you do not have permission to write to the folder.
4. In the case where you are examining a robot that’s loaded to the stage as a reference or payload, the **Additional Information** section contains the path to the original file that the robot is loaded from.
5. Click **Robot Hierarchy** to advance to the next step.

### Robot Hierarchy

Assets in Isaac Sim are organized based on how the robot moves. All the components that move as a single link are grouped together under a single parent. This page allows you to organize your robot components accordingly.

| Ref # | Panel Name | Description |
| --- | --- | --- |
| 1 | New Link Structure | This section displays the new structure of the robot. This structure is based on the links. It might have existing links populated for you if you have chosen a robot type in [Add Robot](#isaac-sim-app-tutorial-wizard-add-robot). You can always add or remove links by using the buttons in the lower right hand corner. |
| 2 | Current Link Structure | This section displays the current structure of the robot that’s on stage. |
| 3 | Parent | The Parent button will be enabled after you’ve selected a target link from the window above and source links from the window below. Clicking on the button will parent the source link to the target link. |
| 4 | Unparent | The Unparent button will be enabled after you’ve selected a link from the window above. Clicking on the button will unparent the link from its parent. |
| 5 | Add/Remove Links | The Add/Remove buttons will add/remove links from the new link structure. |
| 6 | Clear All/Copy All | The Clear All button will clear all the links in the new link structure. The Copy All button will copy all the links in the current link structure to the new link structure. |
| 7 | Instructions | Expand to observe the instructions for the current step. |
| 8 | Add Colliders | The Next button will advance you to the add collider step. |

#### Notes

* The reorganization is focused on grouping different mesh components under a single parent when they belong to the same link. If you have robots where the mesh is nested under many layers of Xforms, choose only the mesh prim and move that to the top window, and delete (right click > delete) any leftover empty parent prims in the bottom window (old stage) where the mesh has been moved.
* It will also ignore any non-mesh prims, such as materials, joints, and textures. Those will be directly copied over to the new file under relevant parent prims.
* Any mesh that is not parented at the end will also get automatically copied over to the new file, unless explicitly deleted.
* The position of the links is set to align with a “reference child” prim. The reference child prim can be indicated by right clicking on the link in the top window, and selecting **Mark as Reference Child**. If no reference child is indicated through the stage, the link’s origin will be positioned at the origin of the first child. Consequently, the transform of all the child meshes will be recalculated to be relative to the parent link’s location.
* No actual prims are created or modified while on this page. All changes are implemented when clicking the **Add Colliders** button to move on to the next step.

### Add Colliders

This page allows you to examine and add the colliders of the robot.

At this point of the process, all the meshes are purely for visualization purposes. There are no colliders or rigid body physics applied to any of the meshes.

The table displays the existing meshes of the robot in the first column. The second column displays the collision approximation method that will be applied to the mesh after you complete the page and move on by clicking the **Add Joints & Drives** button. You can modify the approximation method using the dropdown menu.

For this iteration of the wizard, no new colliders can be created on this page. However, you can always manually add additional meshes and apply the necessary [Physics in USD Schemas](Physics.md) directly in the USD file.

### Add Joints and Drives

This page allows you to add the joints and drives to the robot.

To add a joint, click on the **Create New Joint** button. A popup will appear.

Give the joint a name and select the type of the joint from the dropdown menu. Select the parent and child links the joint will connect, the axis that the joint will move along, and the driver type from the dropdown menu. Then **Create** or **Create & Close** to add the joint to the table on the main page.

To modify the settings for a particular joint, click on the joint name in the table. Two additional sections will appear. The first section allows you to modify the joint properties. The second section allows you to configure the drive. Selecting a different joint or moving to the next page will automatically save the settings for the previously edit joint.

No USD changes are made while on this page. All changes are implemented when clicking the **Save Robot** button to move on to the next step.

### Save Robot

This page finishes the process of creating the robot and creates the final robot files.

You must indicate the link or joint to be the “Articulation Root”. Think of this as the start of the joint chain. For fixed based robots, this is usually the fixed joint. For mobile robots, this is usually the chassis.

You can also choose to add a minimal environment to the main robot USD file. This can be a ground, a default light, and a PhysicsScene. These will be added outside of the Default Prim of the file, so that they will only show up when the original robot file is opened directly on stage, but not when the robot is added as a reference or payload into another scene. They are particularly useful for debugging purposes.

Click on the **Save Robot** button to finish the process.

## Tutorials

[Robot Wizard Tutorial](Robot_Setup.md)

---

# Robot Wizard Tutorial

This tutorial is a step-by-step guide for using the Robot Wizard to create a mock robot that contains a fixed base, a prismatic joint, and a revolute joint.
Load the prepared file for this tutorial onto stage. Go to `Isaac/Samples/Rigging/RobotWizard/` and open the original file for `raw_blocks.usd`.

For more in-depth explanation about the Wizard, refer to [Robot Wizard [Beta]](robot_wizard.html#isaac-sim-app-robot-wizard).

If the Wizard window is not already open, open it from **Window > Robot Wizard**. If you don’t observe the wizard under the Window menu, go to the **Window > Extensions** and enable **Isaac Sim Robot Wizard**.

## Instructions

### Add Robot Page

1. Select **Configure a Robot on Stage**.
2. Select **custom** in the dropdown list for Robot Type.
3. Give your robot a name. The name will be used as the robot’s parent prim name on stage.
4. If the robot prim is not already populated in the **Select Robot Parent Xform** field, click on the dropper and select **World** from the stage popup.
5. Click **Prepare Files** to complete this page.

### Prepare Files

1. Indicate the root folder to save the robot files. You can also modify the robot name here, if needed, for creating a new folder.
2. Check the **Save a Copy in Robot Root Folder** for the current stage and keep the default filepath when that field appears.
3. Click **Next** to move to the next page.

### Robot Hierarchy

1. Add a new link inside the **New Links Structure** window and name it “<robot\_name>/link3”.
2. Put Cube and Cone under `link1`, Cylinder under `link2`, and `Cylinder_01` and `Cube_01` under `link3`.
3. Click **Add Colliders** to finish this page.

Note

You are no longer looking at the original `raw_blocks.usd` file on stage. Instead, a new stage is opened with a robot that’s organized in the link structure you organized in the **Robot Hierarchy** page. This might look a little strange in the viewport.

Take a look at the **Stage** window, verify that in addition to the new robot with the new structure, there are also three new “Scopes” added to the stage:

* **meshes** scope contains the original meshes from the `raw_blocks.usd` file. Each link is a separate mesh, and each mesh has a (0,0,0) origin, that is why there are two copies of each shape, some of them appearing to be clustered in the center of the grid.
* **visuals** scope contains the visual meshes for each link. They are references pointing towards meshes inside the “meshes” scope that are being used for visual purposes.
* **colliders** scope contains the collision meshes for each link. They are also references, but pointing to the meshes inside the “meshes” scope that are used for collision detections. For basic shapes, the visual and collider meshes are often the same. For complex shapes, it is computationally performant to use an approximated version of the visual mesh, such as the bounding volume or convex hull. This allows for faster physics computation while retaining the visual accuracy.

Verify that the main robot prim contains the links as its immediate children, and that each link prims contains the visual and collider meshes as its immediate children. Additional scopes (folders) and a placeholder folder for the joints are created to organize the materials.

To observe what the new robot looks like without the original meshes distraction, hide the “meshes”, “visuals”, and “colliders” scopes.

### Add Colliders

1. For this particular asset, where the shapes are basic, there’s nothing to do on this page. If you are configuring a robot with more complex shapes, you can modify the collision approximation on the right hand column for level of accuracy.
2. Click **Add Joints & Drives** to move on to the next page.

Note

Everything prior to adding the colliders are considered fundamental to the definition of the robot, therefore are saved in the base layer. Rigid Body APIs, Collision APIs, and Joint and Drive APIs are specifically adding properties and setting for physics simulation, and therefore are applied to the physics sublayer of the robot. To inspect the layers, click on the **Layers** tab next to the **Stage** tab in the main Isaac Sim window. Verify that you are editing the physics layer, which has the base layer included as a sublayer.

### Add Joints and Drives

1. Click on the **Create New Joint** button to add a new joint.
2. Make three joints for this robot. Select **Create** to add the first two, and **Create & Close** to add the third.

| Joint Name | Joint Type | Axis | Parent Link | Child Link | Driver Type |
| --- | --- | --- | --- | --- | --- |
| fixed\_joint | Fixed | — | — | <robot\_name>/link1 | (not used) |
| slider\_joint | Prismatic | X | <robot\_name>/link1 | <robot\_name>/link2 | force |
| rotate\_joint | Revolute | Z | <robot\_name>/link2 | <robot\_name>/link3 | force |

3. Click on the joint name in the table to open the joint settings for the joint. Update the joint parameters for the `slider_joint` with the Prismatic Joint settings below, and for the `rotate_joint` with the Revolute Joint settings below.

**Prismatic joint**

> * set the joint limit to 0 to 3
> * set the target position to 1
> * set the stiffness to 1e5 and damping to 2e4

**Revolute joint**

> * uncheck the **Joint Range is Limited** checkbox so that this joint can rotate perpetually
> * set the target velocity to 100
> * set the stiffness to 0 to enable pure velocity drive

4. Click **Save Robot** to save the robot and move on to the next page.

Verify that the Joints folder is populated with the three created joints. You can confirm the settings for each joint by clicking on the joint name in the stage tree. Then use the property panel to validate the joint and drive parameters.

### Save Robot

1. Select the “fixed\_joint” as the “Articulation Root”.
2. Select to add a light and physics scene to the main robot file. The ground is optional because the robot has a fixed joint to the world.
3. Click **Save Robot** to finish the process.

After saving the robot, all the appropriate USD files are created in the robot root folder. Verify that the viewport has the robot from the main robot file with physics variant applied.

Click play to observe the joints move. Validate that the slider joint moves to a target position and that the revolute joint rotates perpetually.

## Final Product Summary

Here is summary of all the things that were done by the wizard. There is a final product for you to compare against your own results in the `Isaac/Samples/Rigging/RobotWizard/final/` folder.

**Robot Hierarchy**:

There is still a vestigial `/World` prim on stage, for now, you can manually move the content into the robot prim’s corresponding folders.

**Files and Folders created**:

```python
robot_root_folder/
├── configurations/
│   └── <robot-name>_robot.usd
│   └── <robot-name>_physics.usd
│   └── <robot-name>_robot.usd
└── <robot-name>.usd
└── stage_copy.usd
```

**APIs applied**:

The APIs applied can be found by selecting the prim on stage and examining the properties panel.

```python
robot_prim (RobotAPI)
├── link1 (RigidBodyAPI, LinkAPI)
│   └── visual
│   ├── collider
│       └── <mesh>  (ColliderAPI)
├── link2 (RigidBodyAPI, LinkAPI)
│   └── visual
│   ├── collider
│       └── <mesh>  (ColliderAPI)
├── link3 (RigidBodyAPI, LinkAPI)
│   └── visual
│   ├── collider
│       └── <mesh>  (ColliderAPI)
├── Joints
│   └── fixed_joint (JointAPI, ArticulationRootAPI)
│   └── slider_joint (JointAPI, DriveAPI, JointStateAPI)
│   └── rotate_joint (JointAPI, DriveAPI, JointStateAPI)
```

## Next Steps

* [Gain Tuner Extension](Robot_Setup.md)
* [Robot Assembler](Robot_Setup.md)
* [Tutorial: Export URDF](Importers_and_Exporters.md)

---

# Editor Tools

- Merge Mesh Utility
- Gain Tuner Extension
- Robot Assembler

# Inspector Tools

- Physics Inspector
- Simulation Data Visualizer

---

# Merge Mesh Utility

## About

The [Merge Mesh Utility](#isaac-merge-mesh) Extension is used to merge multiple prims into a single mesh.
Geometry subsets are used when there are multiple materials on the set of meshes being merged.

To access this Extension, go to the top menu bar and click **Tools**> **Robotics** > **Asset Editors** > **Mesh Merge Tool**.
This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)")
by searching for `isaacsim.util.merge_mesh`.

## User Interface

### Configuration Options

Under the input section you will observe:

* **Source Prim**: This text box shows the prims selected to be merged. TThis gets automatically populated by the selection in the scene. The first item selected will be used as base and origin for the merged asset.
* **Submeshes**: The number of meshes the selected prim contains.
* **Geometry Subsets**: The number of subsets the selected prim contains.
* **Materials**: The number of unique materials used by the selected prim.

Note

To change the mesh origin, you can first select an empty Xform at the desired origin pose, then select all meshes you want to merge afterwards.

Under the output section you will observe:

* **Destination Prim**: The auto-generated output path for the merged mesh.
* **Geometry Subsets**: The number of geometry subsets created after merge. Each unique material will generate a subset on the final merged mesh.

The options when merging are:

* **Clear Parent Transform**: When selected, the merged mesh transform will be at world origin, otherwise it will be the same as the source prim.
* **Deactivate source assets**: When selected, the prims that were selected for merging will be set to “inactive” (effectively deleted, but can be reactivated later).
* **Combine Materials**: When selected, provide the Prim for the Looks folder, it will combine all meshes that use the same material (checked by material name) into a single geomsubset, and move that material to the provided Looks Folder. This is useful for Onshape and Cad imported assets, that contain internal Looks Scopes that are sublayers to the a materials USD layer.

## Tutorials and Examples

The following example showcases how to best use this extension:

---

# Gain Tuner Extension

The [Gain Tuner Extension](#isaac-gain-tuner) is used to tune the stiffness and damping gains of a selected Articulation. This extension is useful when importing any new robot or when needing to fine tune the gains of an existing robot.

This page provides the explanation of the function behind the Gain Tuner. For more detail about the step-by-step usage of the Gain Tuner, refer to [Tutorial 11: Tuning Joint Drive Gains](Robot_Setup.md) tutorial page.

## Overview

The purpose of the Gain Tuner is to find a pair of stiffness and damping gains for each robot joint so that the robot is able to follow commanded trajectories according to the robot’s expected behavior.

The Gain Tuner offers a set of tests that allow you to quickly assess the quality of the current set of gains and a utility for tuning gains manually.

* Tuning Gains: A utility for tuning the gains for the robot.
* Gains Test: A utility for testing the robot’s behavior with a continuous sinusoidal or Step Function trajectory for each joint within the Robot’s Limits, maximum velocities, and accelerations.
* Test Results: A plot of the results of the gains tests on the tracked Joint Positions and Velocities, compared against the commanded trajectory.

### Understanding Joint Drives

Joint Drives are dual-proportional controllers used to set a joint to a given target. One proportional gain is moderating the error in position, while the other gain is moderating the error in velocity. For historical reasons, these gains are called Stiffness and Damping, respectively.

Note

These Joint drives are *implicit* - meaning the position and velocity constraints are imposed by the drive with respect to the current time-step. In engineering this is typically done where it uses a closed loop control with readings of the previous time-step of position and velocity and reacting to it for future control. Refer to [Articulation Joint Drives](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/docs/Articulations.html#articulation-joint-drives).

**Stiffness** is similar to a spring stiffness constant multiplying the error in position, as if the spring was stretched by that amount. **Damping** comes from the effect of targeting zero velocity and therefore any movement would result in a reaction that attempts to stop it. You can actually have it track a velocity that is different than zero and the effect is the same as stiffness would be in position.

\[\tau = \text{stiffness} \* (q - q\_{\text{target}}) + \text{damping} \* (\dot{q} - \dot{q}\_{\text{target}})\]

where \(q\) and \(\dot{q}\) are the joint position and velocity, respectively. When \(\dot{q}\_{target} = 0\), the system reduces to a conventional PD controller on the joint position.

This formula applies for both revolute and prismatic joints.

The joint max force will act as a clamp for \(\tau\), and finally, the drive type will dictate if the effort will be applied directly as a torque or force, or if it will be converted into an acceleration applied to the bodies connected to the joint.

#### Drive Modes

This dual-proportional controller provides two main ways to control the robot:

> * **Position target** - used for controlled joints that are driven by defining a target distance/angle that the connected bodies should be.
> * **Velocity target** - usually done for wheels or other free-spinning objects.

To have a position-controlled joint: set Stiffness to something greater than zero and Damping can be any value.
To have a velocity-controlled joint: set Stiffness to zero and Damping to any value greater than zero.

## Tools

### Tuning Gains

The Joint Gains are a pair of Stiffness and Damping values that are used to drive the joint. They are applied to the joint in the form of a drive that applies an effort (Force/Torque) to the joint, based on the error between the desired position and velocity or both. This Effort is computed as:

\[Effort = K\_p \* (Position\_{Desired} - Position\_{Current}) + K\_d \* (Velocity\_{Desired} - Velocity\_{Current})\]

Where \(K\_p\) is the Stiffness and \(K\_d\) is the Damping.

From this formula, you can describe the different modes of the joint drive:

* Position Drive: When the joint drive is in position mode, the desired position is the target position. This requires the stiffness to be greater than `0`, and the damping to be any value.
* Velocity Drive: When the joint drive is in velocity mode, the desired position is the current position, and the desired velocity is the target velocity. This requires the stiffness to be `0` and the damping to be any value.
* None: When the joint drive is in none mode, the joint drive is not active. The joint can still be controlled by applying a direct effort. This requires the stiffness to be `0` and the damping to be `0`.
* Mimic: When the joint drive is in mimic mode, the joint drive is driven by the mimic joint. This means that the joint drive will not be active, but the mimic joint’s attributes of Natural Frequency and Damping Ratio can still be configured through the Tuner.

This Dampener-Spring model can also be described in terms of the natural frequency and damping ratio:

\[ \begin{align}\begin{aligned}\omega\_n = \sqrt{\frac{K\_p}{m}}\\\zeta = \frac{K\_d}{2 m \omega\_n}\end{aligned}\end{align} \]

Where \(\omega\_n\) is the natural frequency and \(\zeta\) is the damping ratio, and \(m\) is the computed joint inertia based on the mass of the robot at both sides of the joint. The damping ratio is such that \(\zeta = 1.0\) is a critically damped system, \(\zeta < 1.0\) is underdamped, and \(\zeta > 1.0\) is overdamped.

From the above formula, observe that there are two ways to Tune Gains:

* Directly editing Stiffness and Damping values: On the joints table, you can directly edit the Stiffness and Damping values for each joint.
* Natural Frequency: The Gain tuner can also automatically compute the Stiffness and Damping values for each joint based on the desired natural frequency and damping ratio.

Note

Because the robot is a structure that is made of multiple links and moving joints, the natural frequency of each joint is dependent on the robot’s configuration. To establish a standard, the natural frequency of the robot at its home configuration is used.

#### Tuning Options

In the Tuning Options, you can select the tuning mode between Stiffness and Natural Frequency. On the joints table, observe the following options:

* **Mode**: The mode of the joint drive (Position, Velocity, None, Mimic)
* **Type**: The type of the joint drive (Force, Acceleration). In Force, the effort is applied directly to the joint. In Acceleration, the effort is Normalized by the joint’s mass, and is thus invariant to the robot’s configuration, behaving as an ideal actuator.
* **Stiffness** (Stiffness Mode): The stiffness of the joint drive. Changing this will lead to a change in the natural frequency of the joint.
* **Damping** (Stiffness Mode): The damping of the joint drive. Changing this will lead to a change in the damping ratio of the joint.
* **Natural Frequency** (Natural Frequency Mode): The natural frequency of the joint drive.
* **Damping Ratio** (Natural Frequency Mode): The damping ratio of the joint drive.

The configurable Degrees of Freedom (DOF) of the robot are displayed in accordance with what is defined in the Robot’s Joints list.

### Gains Tests

The Gains Tests are a set of tests that allow the user to test the robot’s behavior with a continuous sinusoidal or Step Function trajectory for each joint within the Robot’s Limits, maximum velocities, and accelerations.

The test is divided by sequences and each sequence is a group of joints to be tested together. The sequence is defined per joint and is an index of the order in which the test will be run. For each sequence, the robot resets to the initial configuration, and then the test is run for the provided duration. In addition to that, each joint can be configured to have an individual test setting, which contains the following parameters:

#### Common Test Settings

* **Test**: Check to run the test.
* **Period**: The period of the waveform.
* **Phase**: The phase of the waveform.

#### Sinusoidal

* **Amplitude**: The amplitude of the waveform, from 0 to 100%.
* **Offset**: The offset of the waveform, from 0 to 100%.

#### Step Function

* **Step Minimum**: The minimum value of the waveform, in the joint value units of measurement.
* **Step Maximum**: The maximum value of the waveform, in the joint value units of measurement.

The tests send only Position commands for Position drives, and velocity commands for velocity Drives. In position commands, the target velocities are always zero, such that the joint damping is properly evaluated. In a real control scenario, a proper trajectory command must be ideally sent, where the velocity command is equivalent to the integrated positions of the designated trajectory.

## Further Learning

* The [Tutorial 11: Tuning Joint Drive Gains](Robot_Setup.md) tutorial for detailed instructions on how to use the Gain Tuner.

---

# Robot Assembler

## Learning Objectives

This tutorial shows how to use the isaacsim.robot\_setup.assembler extension to assemble two USD assets into a single rigid body.
This tutorial will primarily demonstrate the use of the Robot Assembler UI tool. By the end of this tutorial, you will understand
the physical mechanics of assembled bodies, when to use the Robot Assembler,
and the current limitations with assembling rigid bodies in NVIDIA Isaac Sim.

*5-10 Minutes Tutorial*

## Getting Started

To find this tutorial of use, you must have two USD assets to assemble into one. This can include:

* A robot arm that needs to be attached to a gripper.
* A robot that needs to be fixed to a moving base.

The use of the word ‘robot’ here indicates a USD asset that Contains the [Robot Schema](Omniverse_and_USD.md) Applied.

## Understanding the Mechanics of Assembled Bodies

The Robot Assembler tool allows you to combine two USD assets together by a physically simulated fixed joint. The result is a USD
asset that can be saved and loaded without needing to use the Robot Assembler each time. The fact that the fixed joint is physically
simulated is key in understanding proper application of the Robot Assembler extension. In Omniverse, physics is only simulated
while the timeline is playing. When physics is not active, the fixed joint will not have any effect. Only use the Robot Assembler to combine USD assets that are going to be moving while the timeline is playing. For example, a robot that is fixed in place
on a static table does not need to have a fixed joint connecting it to the table; you can place both the robot and the table
independently of each other and they will stay in place after the timeline is played.

Additionally, because two assembled assets are attached using a physically simulated fixed joint, the position of one asset relative to another
is resolved by a physics solver. This solution is easy if the assets are already placed correctly relative to each other while the timeline is
stopped, but you might experience instability if, on a stopped timeline, you move one part of an assembled asset far away from the other and
start the timeline.

## Using the Robot Assembler Tool

### Assembling Robots

The Robot Assembler UI tool can be found in the NVIDIA Isaac Sim toolbar by under **Tools > Robotics > Asset Editors > Robot Assembler**.

To use the Robot Assembler, start by loading the assets you want to assemble on the USD stage. There are two editing modes for the Robot Assembler. The workflow is the same for both modes, but the final result will be slightly different:

-**Direct Asset editing**: Open the robot that will serve as a base of the assembly directly, and add a reference to the components to be assembled. This will configure the attached component as a configuration option in the original asset.
-**Stage Editing**: Add both components to be assembled together as a reference to the stage. This will connect both components together at the current stage and will not modify the original assets.

With the **Robot Assembler** window open and both Robots available in the current stage, you can select a **Base Robot** and an **Attach Robot**.

Each robot has an “Attach Point” frame that can be used to specify the point on the robot that will be attached to the other robot. This attach point can be either a [Robot Link](Omniverse_and_USD.md) or a [Reference Point](Omniverse_and_USD.md).

The Assembler also expects an assembly namespace, which defaults to “Gripper”, but can be changed to any string. This namespace is used to identify the attachment point on the base robot when making the assembly directly on the base robot asset.

After selections are made, click on the **Begin Assembly** button to begin the assembly process. This will move the “Attach Robot” to the “Attach Point” of the “Base Robot”, and let you make any final adjustments to the transform. For convenience, a set of Buttons will be shown to allow you to rotate the “Attach Robot” around the X, Y, and Z axes, by increments of 90 degrees. You can also move it through the viewport gizmos however you choose. If you de-select the “attach robot”, the **Select Attach Point Prim** button will re-select it so you can manually move it to the desired position.

After you are happy with the transform, you can click on the **Assemble and Simulate** button to verify the assembly and check if the resulting robot is stable.

At any point, you can click on the **Cancel Assemble** button to undo the assembly and start over.

After assembly and simulation, you can click on the **End Simulation and Finish** button to save the assembly.

If the assembly is performed on the Base asset stage, the resulting assembly is saved as a configuration option under `configuration/<robot_name>_<assembly_namespace>_<attach_robot_name>.usd`, and the `<assembly_namespace>` will be used to create a Variant set on the robot interface layer, such that the new attachment can be selected for use wherever the base robot is used. While the configuration file is automatically saved, you must save the stage to keep the changes.

If the base robot is loaded as a reference, the attachment will be available on the open stage directly, without configuration through variants, and you can save the stage to keep the changes.

With the robot assembled, you might need to execute additional tests to verify simulation stability, given that the articulation system is changed. For a complete guide in tuning articulations, refer to [Articulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html).

## Robot Assembler API

The Robot Assembler can also be accessed by a Python API, where the assembly settings can be specified programmatically.

```python
import omni
from isaacsim.robot_setup.assembler import RobotAssembler

# Prerequisites: Have both the base robot and the attach robot loaded in the stage at the paths specified below (or change the paths to where the assets are loaded in your stage)

# Prim path to the base robot
robot_base = "/World/BaseRobot"
# Prim path to the mount point of the base robot
robot_base_mount = "/World/BaseRobot/Mount"
# Prim path to the attach robot
robot_attach = "/World/AttachRobot"
# Prim path to the mount point of the attach robot
robot_attach_mount = "/World/AttachRobot/Mount"
# Assembly namespace
assembly_namespace = "Gripper"
variant_name = "my_assembled_robot"

stage = omni.usd.get_context().get_stage()
assembler = RobotAssembler()

# Begin the Assembly process - Creates a session layer and attach it to the current stage, where all the modifications necessary for the assembly will be made.
assembler.begin_assembly(
    stage, robot_base, robot_base_mount, robot_attach, robot_attach_mount, assembly_namespace, variant_name
)

# Perform any Additional transformations on the Attach robot pose here directly through USD.

assembler.assemble()

# That's where the Robot Assembler will create the fixed joint between the two robots.
# It will also remove Physic's Articualtion Root from the attached robot, and disable the root joint that attaches 	robot to the world, if it exists.
# If you need to perform any physics simulation test - this is the time to do it.
# If the assembly is successful, and you are ready to finish the assembly, you can call the following function.
# Otherwise at any point you can call the `assembler.cancel_assemble()` function to cancel the assembly process.
# It will remove the session layer from the stage, undoing any changes made to the stage.

assembler.finish_assemble()

# This function will finish the assembly process by adding the attachment link to the parent robot joint and link lists, and then either merge the session layer into the current stage, or save a configuration file, and remove the session layer from the stage.
# If modifing a robot asset directly, it will also create the variant set to load the configuration for the assembled component through a payload.
```

## Summary

In this tutorial, you learned that:

1. The Robot Assembler tool exists to attach two robots or rigid bodies using a user-specified fixed joint.
2. The Robot Assembler creates a fixed joint that is physically simulated, and so it will only be active while the timeline is playing.
3. The Robot Assembler is only needed to attach Robot components together.
4. The Robot Assembler can also be accessed by a Python API that is demonstrated on the example code above.

---

# Asset Structure

The Isaac Sim Imported assets are organized in a specific structure to make it easier to manage, reuse, and simulate them. Each asset is broken down into multiple components, which can be categorized as follows:

For an example of an asset following these guidelines check Nova Carter at `Robots/NVIDIA/Carter/nova_carter/` in Isaac Sim assets.

## Asset Source

Assets in this stage represent their raw form as imported from their original file format. They are typically organized into:

1. **Base Asset (** `asset_base.usd` **):** Contains the full structural hierarchy of the asset, such as robot assemblies.
2. **Parts (** `parts.usd` **):** Includes individual components, with one USD file per mesh. This modular breakdown ensures easy access and management.
3. **Materials (** `materials.usd` **)**: A collection of Physically Based Rendering (PBR) materials used by the asset.

### Guidelines

* The source assets must remain unchanged to ensure that they can be re-imported seamlessly without losing downstream modifications.
* Consistency is critical. The structural hierarchy, naming conventions, and part assemblies must remain intact.

### Transformation

This stage prepares the asset for simulation by reorganizing and optimizing it. This transformation is necessary when the source asset contains nested rigid bodies or a complex structure that doesn’t meet the requirements of simulation. The structure must be flattened with rigid bodies organized into a basic list, and meshes must be simplified to minimize their total count. The transformation process includes:

1. **Reorganizing Structure**:
   - Create the simulation structure (for example, separating visuals and colliders as needed).
   - Adjust the hierarchy to fit simulation requirements.
2. **Optimizing Meshes**:
   - Merge meshes that will function as a single rigid body.
   - Simplify the material count into a single visual material list.
   - Clean and format meshes as instantiable references to enhance performance.

Note

If the **asset source** is already in a format suitable for simulation, this step or parts of it can be skipped.

## Features

Simulation features are added in this stage and each feature is defined as a separate lightweight layer that builds on top of the transformed asset. These features include, but are not limited to, physics setups, sensor configurations, and control graphs.

### Workflow for Adding and Modifying Features

1. Create a new empty stage or open the existing feature stage.
2. Add the **optimized asset** (`asset_sim_optimized.usd`) as a sub-layer.
3. Modify the root layer to add/modify the feature.
4. Remove or disable the sub-layer (optimized asset) from the stage composition before saving.
5. Add the feature to the final asset as a **payload**. Optionally, a Variant set can be configured to enable quick switching between different feature sets by selecting them on a list.

### Example Features

* **Physics (** `asset_physics.usd` **)**: Adds rigid bodies, colliders, joints, and articulations.
* **Sensors (** `asset_sensors.usd` **)**: Defines sensor specifications.
* **Control Graphs (** `asset_control.usd` **)**: Adds control features for simulations.
* **ROS Integration (** `asset_ros.usd` **)**: Configures ROS Omnigraph functionalities.

Note

The Physics feature is an exception and is added as a reference to the default prim, while other features are added as payloads and it maintains the layer connection to the optimized asset.

## Composition of Final Asset

The final composed asset is represented in the `asset.usd` file, which integrates all the necessary components for simulation. This is achieved through the following composition process:

* **Sublayers**:
  :   + The base or optimized asset (`asset_sim_optimized.usd`) is included as a sublayer to provide the core structural and visual elements.
* **Payloads**:
  :   + Features such as sensors (`asset_sensors.usd`) and control graphs (`asset_control.usd`) are dynamically added as payloads. This allows for flexible and efficient loading of components.
* **References**:
  :   + The physics setup (`asset_physics.usd`) is added as a reference to the default prim, ensuring a consistent simulation-ready configuration.
* **Variants**:
  :   + Variants can be configured in the `asset.usd` file to enable different feature sets, such as alternative sensor configurations or control setups, without duplicating the asset.

This modular approach ensures that the final asset file is both lightweight and highly flexible, making it easy to adapt to different simulation scenarios.

To keep assets organized and maintainable, it is recommended that you follow the structure and guidelines outlined above. This will help streamline the asset creation process and improve overall simulation performance.

It is also suggested that you keep the assets organized in folders, with the source assets in their own folder, and all features in a features folder, while the final asset is saved in the root folder. By default Isaac Sim importers for robots follow this structure.

### Robot Schema

The [Robot Schema](Omniverse_and_USD.md) provides a way to describe the robot structure agnostic of the simulation asset structure. The robot schema must be included as a sublayer on the robot asset.

## Key Definitions and Notes

* **Add-ons**:
  - Features that have the simulation asset as a temporary sublayer used during feature creation. It is called the Add-on. The sublayer connection is broken before saving the feature asset.
* **Payloads**: Dynamically loadable components that reduce memory overhead and improve modularity.

---

# Robot Setup Tutorials Series

The GUI tutorials walk you through setting up your virtual world and building robot digital twins with various NVIDIA Isaac Sim features. In the process, you will learn where to find frequently used properties, settings, and tools, and familiarize yourself with the toolbars, icons, and OpenUSD standards.

**Important:** These tutorials are designed as a progressive learning path from beginner to advanced. We recommend starting with the *Setup a Wheeled Robot* section, as it covers essential beginner concepts like environment setup, basic robot assembly, and fundamental rigging techniques that are required for all robot types.

## **Beginner Level** - Setup a Wheeled Robot

Start here to learn fundamental concepts that apply to all robot types:

- Tutorial 1: Stage Setup
- Tutorial 2: Assemble a Simple Robot
- Tutorial 3: Articulate a Basic Robot
- Tutorial 4: Add Camera and Sensors to a Robot
- Tutorial 5: Rig a Mobile Robot

## **Intermediate Level** - Setup a Manipulator

Build upon the foundational knowledge to work with more complex robot structures:

- Tutorial 6: Setup a Manipulator
- Tutorial 7: Configure a Manipulator
- Tutorial 8: Generate Robot Configuration File
- Tutorial 9: Pick and Place Example

## **Advanced Level** - Asset Tuning and Optimization

Master advanced techniques for complex robot configurations:

- Tutorial 10: Rig Closed-Loop Structures
- Tutorial 11: Tuning Joint Drive Gains
- Tutorial 12: Asset Optimization
- Tutorial 13: Rigging a Legged Robot for Locomotion Policy

---

# Tutorial 1: Stage Setup

Isaac Sim is built on [NVIDIA Omniverse](https://docs.omniverse.nvidia.com/) using tools provided in [Omniverse Kit](https://docs.omniverse.nvidia.com/dev-guide/latest/index.html "(in Omniverse Developer Guide)"). Omniverse Kit comes with a default UI that
allows you to edit a USD stage with ease. In this tutorial, you learn the basic steps for setting up an environment, adding and editing simple objects and their properties on a USD stage,
rigging rigid bodies with joints and articulations, and adding cameras and sensors.
The goal is to build your basic skills in navigating Isaac Sim, becoming familiar with frequently used terms, and using the GUI to build an environment and set up your robots.

## Learning Objectives

This tutorial teaches you to build a physics-enabled virtual world using the tools provided in the Isaac Sim GUI, including:

* Setup global stage properties
* Setup global physics properties
* Add ground plane
* Add lighting

## Prerequisites

To start with a clean Isaac Sim stage, go to the File menu and click on **New**.
The stage provided has a default `World` [Xform](https://docs.omniverse.nvidia.com/utilities/latest/common/glossary-of-terms.html#term-XForm "(in Omniverse Utilities)"), and a `defaultLight`. Both can be found on the stage tree on the right of the viewport.

## Setting up Stage Properties

Before anything is added onto the stage, verify that the current stage property setup matches the your expected conventions.

1. Go to **Edit > Preferences** to open up the Preference panel.
2. Browse the many types of settings inside Omniverse Kit grouped into categories in the column on the left of the panel.
3. Select **Stage** from the left column and review the properties such as:

   * The axis that determines *Up*. The default in Isaac Sim is Z. If your asset is created in a program with a different up-axis, it causes your assets to be imported rotated.
   * Stage units. Isaac Sim versions prior to 2022.1 have stage units in centimeters, but the default is now meters. However, the default units for Omniverse Kit is still in centimeters. Keep that in mind if you see USD units that are seemingly off by 100x.
   * Default rotation order. The default is set to execute rotation in Z, then Y, and last X.

## Creating the Physics Scene

To add a **Physics Scene** to simulate real world physics, including gravity and physics time steps:

1. Go to the Menu Bar and click **Create > Physics > Physics Scene**.
2. Validate that a **PhysicsScene** is added to the stage tree.
3. Click on it to examine its properties.
   You can see that gravity is set to the magnitude of `Earth Gravity`, or `9.8` meters per second squared. Remember that the default unit of length is meters.
4. Unless you are simulating hundreds of rigid bodies and robots, it is more efficient to use CPU physics
   :   * Open Physics Scene’s **Property** tab
       * Uncheck **Enable GPU dynamics**
       * Set the **Broadphase** type to **MBP**.

## Adding a Ground Plane

The ground plane prevents any physics-enabled objects from falling below it.
The ground plane’s collision property extends indefinitely even though the plane is only visible up to 25 meters in each direction.

To add a ground plane to the virtual environment:

1. Go to the top Menu Bar and click **Create > Physics > Ground Plane**.
2. Turn on the grid by clicking on  and selecting **Grid** to make the ground plane easier to see.

## Lighting

Every new [Stage](Glossary.md) is pre-populated with a `defaultLight`, otherwise you wouldn’t see anything. This default light is a child of the `Environment` Xform in the stage and can be found in the stage context tree.

To create additional spotlights:

1. Add a ground plane, if there isn’t already one, so we can see the reflection of the light. **Create > Physics > Ground Plane**.
2. Go to **Create > Light > Sphere Light**.
3. Pose the light on the stage.
   - In the **Stage** tab on the top right, select the newly created light in the stage tree.
   - In the **Property** tab on the bottom , in the **Transform** section use the **Translate** tool to move it to a position above the ground plane, such as `(0, 0, 7)`.
   - In the **Property** tab, in the **Transform** section, use the **Orient** tool to set the rotation to `(0, 0, 0)`.
4. Modify light color, brightness, and scope properties:
   - Inside the **Property** tab, change its color in **Main > Color** by clicking on the color bar and pick a color of your choice. For example a light green color `(RGB: 0.5, 1.0, 0.5)`.
   - Also inside the **Property** tab, change its intensity **Main > Intensity** to **1e6**; **Main > Radius** to **0.05**
   - In the **Shaping** section, change the **cone:angle** to **45** degrees and **cone:softness** to **0.05**.
5. To make the new spotlight easier to see, we will reduce the intensity of the default light by going to its **Property** tab and set **Main > Intensity** to **300**.

## Summary

This tutorial begins the necessary steps to create a virtual world suitable for physics simulation and testing Isaac Sim.
The following topics were covered:

* Adding a ground plane, lighting, and physics scene.

### Next Steps

Continue on to [Tutorial 2: Assemble a Simple Robot](Robot_Setup.md) to learn how to add simple objects to Isaac Sim and edit their properties.

### Further Learning

For more in-depth and creative world-building tools, refer to our sister Omniverse tool [Composer](https://docs.omniverse.nvidia.com/composer/latest/index.html "(in Omniverse USD Composer)").

---

# Tutorial 2: Assemble a Simple Robot

This tutorial guides you through the basic GUI functions that add objects to the stage. It also introduces inspecting and modifying their physics and material properties.

## Learning Objectives

This tutorial covers how to:

* Add and manipulate basic shapes
* Enable physics properties in objects
* Examine collision properties
* Edit physics properties such as friction
* Edit material properties such as color and reflectivity

## Prerequisites

* Complete [Tutorial 1: Stage Setup](Robot_Setup.md) prior to beginning this tutorial.

## Adding Objects to the Scene

There are many ways to “add objects” to the stage, but all of them fundamentally do the same thing, which is to define a USD primitive in the stage context tree. The goal is to create a basic, two wheeled robot. Start by creating some basic shapes and modifying their properties. For the body, use a cube and for the wheels use cylinders.

To create the body of the robot:

1. Create an Xform by right clicking on the stage, selecting **Create > Xform**.
2. Rename it to **body** by right clicking on it and selecting **Rename**.
3. Fix the translation of the Xform to `(0, 0, 1)` by clicking on the **Translate** section in the property panel and setting the **X** to `0`, **Y** to `0`, and **Z** to `1`.
4. Create a cube clicking **Create > Shape > Cube** in the top menu bar. You should see the cube and the **Move** **gizmo** (the red, blue, and green arrows) appear in the viewport window
5. Click and drag on the blue arrow to raise the cube above the ground plane.
6. On the left side of the app, click the Scale icon (or press the R key while the cube is selected) to activate the scale widget.
7. Click and drag on the red part of the widget to scale the cube in the x direction
8. Place the cube in a specific location. Navigate to **Transform > Scale** in the property pane, and set the scale to `(1, 2, 0.5)`.
9. Drag the cube to the **Body** Xform.

To create the wheels of the robot:

1. Create a Xform by right clicking on the stage, selecting **Create > Xform**. Set the **Translate** to `(1.5, 0, 1)` and the **Orient** to `90, 0, 0` to rotate the wheel Xform 90 degrees around the x axis.
2. Rename it to **wheel\_left** by right clicking on it and selecting **Rename**.
3. Create a cylinder by clicking **Create > Shape > Cylinder** in the top menu bar.
4. In the property panel on the bottom right corner, scroll down to the **Geometry** section. Change its **Radius** to `0.5` and **Height** to `1.0`.
5. Drag the cylinder to the **wheel\_left** Xform.
6. Rename the cylinder to **wheel\_left** by right clicking on it and selecting **Rename**.
7. Duplicate the `wheel_left` by right clicking the `wheel_left` Xform on the stage tree, select **Duplicate**, and move it to `x = -1.5` while keeping all other parameters the same.
8. Rename the duplicated Xform to **wheel\_right** by right clicking on it and selecting **Rename**.
9. Rename the duplicated cylinder to **wheel\_right** by right clicking on it and selecting **Rename**.

## Adding Physics Properties

The cubes and cylinders added so far are strictly visual prims, with no physics or collision properties attached to them.
When you start the simulation by pressing **Play** and gravity is applied, these objects do not move because they are unaffected by physics.

To make the robot have physics, turn it into a rigid body with collision properties:

1. Select the Cube and both Cylinders on the stage tree by clicking while holding down the `Ctrl + Shift` key to select each object, or just `Shift` if they are consecutively listed on the tree.
2. In the **Property** tab, click on the `+ Add` button.
3. Select **Physics > Rigid Body with Colliders Preset**.
4. Press **Play** and verify that all three objects fall to the ground.

**Rigid Body with Colliders Preset** automatically adds the Rigid Body API and the Collision API to the objects.
These two APIs can be applied separately because you can have objects that:

* have mass and are affected by gravity, but have no collision properties so you can pass through them
* can be run into but hang in the air and are not affected by gravity

To validate, add, or remove APIs assigned to the selected object:

1. Go to its **Property** tab, and scroll down to find sections labeled **Rigid Body** and **Collider**.
2. To add the APIs separately, find them under the same **+ Add** button.
3. To remove APIs, click on the `X` to delete the section.

Hint

Dynamic objects can only select from Convex Hull, Convex Decomposition, Sphere Approximation, SDF mesh (GPU backend only) for collision shapes.
Triangle mesh collision shapes are only available for static objects.

### Examine Collision Meshes

To visually examine the outlines of collision meshes for the objects:

1. Find the eye icon on top of the viewport.
2. Click **Show By Type > Physics > Colliders > All**.
3. Verify that purple outlines show up surrounding any objects that have collision APIs applied. For example, verify that it is the cuboid, the cylinders, and the ground plane.

### Adding Contact and Friction Parameters

For modifying frictional properties, you must create a different physics material and then assign it to the desired object.

1. Go to the Menu Bar and click **Create > Physics > Physics Material**.
2. Select **Rigid Body Material** in the popup box. A new `PhysicsMaterial` appears on the stage tree.
3. Tune the parameters such as friction coefficients and restitution in its property tab.

To apply the assigned physics material to an object:

1. Select the object in the stage tree.
2. Find the menu item **Materials on Selected Model** in the **Property** tab.
3. Select the desired material in the drop-down menu.

## Material Properties

The objects may reflect the color of the spotlight added earlier, but it doesn’t actually have any colors assigned. You can confirm this by turning off the spotlight.

To change the color of the object, create a different material and then assign it to the objects, just like with the physics materials.
For example, create two different materials, one for the body of the car and one for the wheels.

1. Click **Create > Materials > OmniPBR** twice.
2. Right-click on the newly added materials on the stage tree and rename them to **body** and **wheel**.
3. Assign the corresponding rigid bodies to the newly created materials by going to the **Materials on selected models** item in its **Property** tab, and select the matching material from the dropdown.
4. Change the property of the new materials. Select one of them on the stage tree, change its base color in *Material and Shader/Albedo* and play with its reflectivity roughness and whatever else you find interesting.
5. Verify that you see the color of the corresponding parts on the car change accordingly.

## Summary

By the end of this tutorial, you should have a robot with a body and two wheels, similar to the `mock_robot_no_joints` asset, located in the **Samples > Rigging > MockRobot** folder.

This tutorial explained how to add and manipulate object properties in the GUI, including:

> 1. Adding primitive shapes onto the [Stage](Glossary.md).
> 2. Editing material properties, physics properties, and collision properties.

### Next Steps

* Continue to [Working with USD](Omniverse_and_USD.md) to learn how to save your world and load assets in USD format inside Isaac Sim.
* Go to [Tutorial 3: Articulate a Basic Robot](Robot_Setup.md) to learn how to turn these geometries into a moving car.

---

# Tutorial 3: Articulate a Basic Robot

NVIDIA Isaac Sim’s GUI interface features are the same ones used in NVIDIA Omniverse™ USD Composer, an application dedicated to world-building. This tutorial focuses on the GUI functions that are most relevant to robotic uses. For more sophisticated general world creation, see [Omniverse Composer](https://docs.omniverse.nvidia.com/composer/latest/index.html "(in Omniverse USD Composer)").

You will rig a basic “robot” with three links and two revolute joints to introduce the concepts of joints and articulations. You take the objects that were added to the stage in [Tutorial 2: Assemble a Simple Robot](Robot_Setup.md) and turn them into a mock mobile robot with a rectangular body and two cylindrical wheels.

This is not needed for robots that are imported from [Importing your Onshape Document](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html#isaac-onshape-importer-tutorials-importing "(in Omniverse Extensions)") or [URDF Importer Extension](Importers_and_Exporters.md), these are important concepts to understand for tuning your robots and assembling objects with articulations.

## Learning Objectives

This tutorial details how to rig a two-wheel mobile robot and covers how to:

* Organize stage tree hierarchy
* Add joints between two rigid bodies
* Add joint drives and joint properties
* Add articulations
* Move the robot via a Articulation Velocity Controller

## Prerequisites

* Complete [Tutorial 2: Assemble a Simple Robot](Robot_Setup.md).
* Or load the checkpoint asset provided in the Content Browser at `Isaac Sim/Samples/Rigging/MockRobot/mock_robot_no_joints`. Do not load it as a reference because you must make permanent modifications to the file.

## Add Joints

1. If you are continuing from the GUI Tutorials and have your own `mock_robot.usd` saved, open it using **File > Open**. Otherwise, load the asset provided in the Content Browser at `Isaac Sim/Samples/Rigging/MockRobot/mock_robot_no_joints`. Do not load it as a reference because you must make permanent modifications to the file.
2. For organization, create a Scope to store the joints by right clicking **Create > Scope**, rename it to **Joints**.
3. To add a joint between two bodies, you must first select them both. Begin by clicking on the body and wheel parent transforms in the context tree window. For our mock robot, select the the cube object `body`, then while holding `Ctrl`, select the cylinder object `wheel_left`.
4. With both bodies highlighted, right-click and select **Create > Physics > Joints > Revolute Joint**. `RevoluteJoint` appears under `wheel_left` on the stage tree. Rename it to `wheel_joint_left`.
5. Verify in the **Property** tab that **body0** is `/mock_robot/body/body` (the cube) and **body1** is `/mock_robot/wheel_left/wheel_left` (the cylinder).
6. Set the X axis of the joint to **Local Rotation 0** to `0.0` and **Local Rotation 1** to `-90.0` to account for the transformation between the body and the cylinder. This is because the cylinder is rotated 90 degrees in the X axis compared to the body.
7. Change the **Axis** of the joint to **Y**. Because there is no local rotation `0` for the robot, the joint is in the same pose as the body.
8. For organization, drag the joint you just created into the **Joints** scope.
9. Repeat the previous five steps with the right wheel joint.

Before the joints were added, the three rigid bodies fell to the ground separately after pressing **Play**. Now that there are joints attached, the bodies fall as if they are connected.
To see that they move together like they are connected with revolute joints, you can drag the robot around by holding down the `Shift` key and clicking and dragging on any part of the robot in the viewport.

## Add a Joint Drive

Adding the joint adds the mechanical connection. To be able to control and drive the joints, you must add a joint drive API.
Select both joints and click the `+ Add` button in the **Property** tab, and select **Physics > Angular Drive** to add drive to both joints simultaneously.

* **Position Control:** For position controlled joints, set a high stiffness and relatively low or zero damping.
* **Velocity Control:** For velocity controller joints, set a high damping and zero stiffness.

For joints on a wheel, it makes more sense to be velocity controlled, so set both wheels’ **Damping** to **1e4** and **Target Velocity** to **200** **rad/s**.
If you are working with joints with limited range, those can be set in the **Property** tab, under the **Raw USD Properties > Lower (Upper) Limit**.
Press **Play** to see the mock mobile robot drive off.

## Add Articulation

Even though directly driving the joints can move the robot, it is not the most computationally efficient way. Making things into *articulations* can achieve higher simulation fidelity, fewer joint errors, and can handle larger mass ratios between the jointed bodies. For more information on the physics simulation behind it, see [Physics Core: Articulation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html "(in Omni Physics)").

To turn a series of connected rigid bodies and joints into articulation, set an *articulation root* to anchor the articulation tree. According to instructions on defining articulation trees in [Physics Core: Articulation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html "(in Omni Physics)"):

> > > For a fixed-base articulation, add the Articulation Root Component either to:
> >
> > * the fixed joint that connects the articulation base to the world.
> > * an ancestor of the fixed joint in the USD hierarchy. This allows creating multiple articulations from a single root component added to the scene.
>
> Each descendant fixed joint defines an articulation base link.
>
> > > For a floating-base articulation, add the Articulation Root Component to either:
> >
> > * the root rigid-body link
> > * an ancestor of the root link in the USD hierarchy

For this tutorial, add the articulation root to the robot:

1. Select `mock_robot` on the tree.
2. Open **+ Add** in the **Property** tab.
3. Add **Physics > Articulation Root**.

Validate that the resulting robot matches the asset that is provided in the Content Browser at `Isaac Sim/Samples/Rigging/MockRobot/mock_robot_rigged`.

## Add Controller

After the joints are part of an articulation, you can use tools to test the robot’s movement.

1. Create another scope by right clicking **Create > Scope**, rename it to **Graphs**. This will be used to store the ActionGraphs.
2. Drag the **Graphs** scope under the `mock_robot` Xform in the stage tree.
3. Go to **Tools > Robotics > Omnigraph Controllers > Joint Velocity** to add a velocity controller graph to the stage. This graph will allow you to control the robot’s movement by setting the target velocity for each joint.
4. Click the **Add** button for “Robot Prim” and select the prim with the Articulation Root API, in this case, it’s `/mock_robot`.
5. For Graph Path, write `mock_robot/Graphs/Velocity_Controller` to place the ActionGraph in the **Graphs** scope above.
6. Click **OK** to create the graph.
7. To move the robot, press **Play** to start the simulation. If you have any default position or velocity targets set, the robot starts moving towards those targets immediately. To change the joint commands, select the `JointCommandArray` on the stage tree under **/Graphs/velocity\_controller**, and change the parameters `input0` and `input1` in the properties window.

Note

The articulation controllers use **radians**, the default USD properties you find under Drive API when you select the individual joints on the stage tree are in **degrees**.

For this particular robot, it can also be controlled using a Differential Controller. For more information about Omnigraph Controller shortcuts, go to [Commonly Used Omnigraph Shortcuts](Omnigraph.md).

## Summary

In this tutorial, you learned to connect rigid bodies using joints, add a joint drive to control the joints, turn a chain of joints into an articulation, and control the robot using an Articulation Velocity Controller.

By the end of this tutorial, you have a robot with a body and two wheels, similar to the `mock_robot_rigged` asset, located in the `Samples/Rigging/MockRobot` folder.

### Next Steps

* Continue on to [Tutorial 4: Add Camera and Sensors to a Robot](Robot_Setup.md) to learn how to add a camera to the car.

### Further Reading

[Physics Core](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html "(in Omni Physics)") for more details regarding joints and articulations.

---

# Tutorial 4: Add Camera and Sensors to a Robot

Isaac Sim provides a variety of sensors that can be used to sense the environment and robot’s state.
This tutorial guides you through attaching a camera sensor to a mock robot, a process that can be generalized to other sensors.
Details regarding the camera and other types of sensors can be found in our Advanced Tutorials and Sensor Extensions.

## Learning Objectives

This tutorial details how to:

* Add cameras
* Attach cameras to geometries

## Prerequisites

* Complete [Tutorial 3: Articulate a Basic Robot](Robot_Setup.md).
* Review the [introduction to camera frames and axes](Isaac_Sim_Conventions.md).

Start this tutorial using the `Isaac Sim/Samples/Rigging/MockRobot/mock_robot_rigged.usd` file provided, to have a standardized setup.

## Adding a Camera

To add a camera:

1. Go to the Menu Bar and select **Create > Camera**. A camera appears on the stage tree, and a grey wireframe representing the camera’s view appears on the stage.
2. You can move and rotate the camera’s transform just like any other objects on the stage.

Note

The camera icon is hidden by default in the viewport. To see the camera icon, go to the **eye** menu on the top edge of the viewport, and select **Show By Type > Cameras**. The camera icon appears in the viewport.

You can also add a camera by moving the current view in the viewport to a view of your choosing, and then go to the **Camera** button on the upper left hand corner of the viewport display, and select **Camera > Create from View**.
A new camera appears on the Stage tree, and the list of cameras that can be selected in the **Camera** button is provided.

## Inspect the Camera

Use the [Camera Inspector Extension](Sensors.md) to inspect the camera image and modify the camera’s states as needed.

1. Select **Tools > Robotics > Camera Inspector**.
2. Verify that you can see the camera in the dropdown. Click the **Refresh** button to find new cameras.
3. Select the camera you want to inspect. Create new viewports if necessary, and get and set camera poses as needed.

## Attach a Camera to Robot

1. Rename the newly added camera to `car_camera`.
2. It is easier to place the camera if you can see the desired camera input stream and where it is relative to the robot from an outside camera.
   Open up a second viewport window by going to the Menu Bar and click **Window > Viewports > Viewport 2**. A new viewport appears. Dock it wherever you’d like.
3. Keep one of the viewports in **Perspective** camera view, and change the other one to *car\_camera* view. Find the **Cameras** menu on the top edge of the viewport, and switch to **Camera > car\_camera**.
4. Validate that you have a view of the onboard camera and an overview of the scene.
5. Attach the camera to the robot’s body by dragging the prim under `body`. The camera moves together with the body. You may need to switch the camera view for the viewport again.
6. Point the camera slightly down and make it face forward so you can see the car and the ground. Set the camera transform translation to `x=-6,y=0,z=2.2`, orientation to `x=0,y=-80,z=-90`, and scale to `x=1,y=1,z=1`.
7. Verify that you see the viewport showing the onboard camera view splitting the window between the robot’s body and the ground and the relative position and orientation of the camera to the robot in the *Perspective* camera viewport.
8. Press **Play**. The camera onboard the robot moves with the robot.

A similar strategy is used to apply other onboard sensors.

Note

If the view of the camera is moved while displaying, it changes the camera’s properties. Instead, affix a prim to the parent with the correct offset and affix the camera to that new prim. Then, if the camera position is accidentally moved, it can be reset by zeroing all its position and orientation parameters relative to the prim, which cannot be easily changed.

## Summary

In this tutorial, you learned how to use the Camera Inspector Extension. Additionally, you also learned how to add a camera to the robot.

### Next Steps

* Continue on to [Omniverse Script Editor](Development_Tools.md) to learn how to run Python APIs inside the GUI.
* For rigging a more complex robot, go to [Tutorial 5: Rig a Mobile Robot](Robot_Setup.md).

---

# Tutorial 5: Rig a Mobile Robot

If you built a robot inside Omniverse USD Composer or used importers that do not carry over joint information, you’ll need to rig the robot before it can move like an articulated robot and be controlled by Isaac Sim APIs. This involves defining the types of joints between the body parts and setting the parameters that governs the joints’ behavior, such as stiffness and damping. This tutorial covers step-by-step instruction on how to rig a forklift.

## Learning Objectives

In this tutorial, an unrigged forklift USD asset is turned into a forklift that can move and be driven by Isaac Sim commands.

*30 Minutes Tutorial*

## Getting Started

**Prerequisite**

* Complete the [Quick Tutorials](Quick_Tutorials.md) series to learn the basic core concepts of how to navigate inside Isaac Sim.
* Complete the [Assemble a Simple Robots](Robot_Setup.md) and [Adding Sensors and Cameras](Robot_Setup.md) tutorials to learn the concepts of rigid body API, collision API, joints, drives, and articulations.

**Reference USDs**

We provide USD assets relating to this tutorial in [Isaac Sim Assets](Isaac_Sim_Assets.md), and can be found in the Content Browser.

* Unrigged Forklift: `Isaac Sim/Samples/Rigging/Forklift/forklift_b_unrigged_cm.usd`
* Rigged Forklift: `Isaac Sim/Samples/Rigging/Forklift/forklift_b_rigged_cm.usd`

This tutorial guides you through the steps for going from file to file. The rigged assets serve as a reference for the final goal.

## Rigging the Robot

### Identify the Joints

Before making any modifications to the asset, the first step of rigging a robot is to identify the joints on the robot, both actuated and unactuated ones. The joints govern how all the mesh components are organized, and identifying the type and their degrees of freedom (DOF) are key in making sure the robot moves as expected once rigged.

For the forklift, there are seven DOF in total:

* There are four smaller roller wheels at the front. They have unactuated, revolute joints, and each has one degree of freedom for rotation about a single axis.
* The fork has linear motion relative to the main body of the forklift as it moves up and down to pick up objects stacked on the pellet, which means there is one actuated, prismatic joint between the fork and the body.
* The bigger wheel at the rear end is responsible for propelling the forklift and turning it. There are two actuated joints related to this wheel:

  > + A revolute joint that spins the wheel around its central axis to provide the forward and backward movement.
  > + A revolute joint between the rear wheelbase and the forklift body that provides the pivot to turn the forklift.

### Organize the Hierarchy

Open the unrigged forklift asset from the Content Browser: `Isaac Sim/Samples/Rigging/Forklift/forklift_b_unrigged_cm.usd`.
Depending on the importer used and the original asset’s setup, the unrigged structure of the USD could have no hierarchy in terms of how parts are organized. It could have every single component listed independently on the stage tree. This makes it difficult to read and navigate, but more importantly, it does not define which objects are moving as a group and how these groups are related to each other.

> 

All meshes that are children of a parent prim are expected to move together when the parent prim moves. For example, the sticker and chains on the meshes are a part of the forklift body, and the entire body, no matter how many screws or blocks are used to make up the body, can be considered as a single link of this robot. Organize them all under a single parent ‘body’ prim. This ensures that when the ‘body’ moves, that all child parts that make up the body are moving together.

To organize prims for the forklift:

1. Create two XForms called `body` and `lift`.
2. Move all the meshes that make up the forklift body under the `body` Xform, and the operator cab meshes under the `lift` Xform. For ease of use, the meshes provided in the USD file are sorted according to their hierarchy. All meshes above `Looks` are a part of the `lift` XForm. Meshes below `Looks` (Right Chain Wheel to Body Glass) are a part of the `body` XForm. Remaining are for the wheelbase and wheels.
3. Create new Xforms for the `back wheel`, `back wheel swivel`, and separate prims for each of the front roller supports.
4. Create a new Xform for each of the four front roller wheels. Name them `roller_front_left`, `roller_front_right`, `roller_back_left`, and `roller_back_right`. Move the correct lead wheel mesh and cylinder collider under them.
5. Ensure that all the Xforms mentioned above have physics set to rigid body by clicking **Add** > **Physics > Rigid Body**.
   :   Note

       Rigid body prims cannot have children that are also rigid bodies.
6. It is easier to set the joints if they align the frame of the Xform to the frames of the respective wheels. To do so, for each wheel, select the mesh, and in its property tab under **Transform**, there are two components `Translate` and `Translate:pivot`. The newly created Xform’s transform must be the sum of those two components. For example, if `translate` is at \(X=x\_1, Y=y\_1, Z=z\_1\), and `translate:pivot` is at \(X=x\_p, Y=y\_p, Z=z\_p\), then the transform of the newly created Xform must be set to: \(X = x\_1+ x\_p , Y = y\_1 + y\_p , Z = z\_1 + z\_p\).
7. `Translate` of the wheel mesh needs to be set to the inverse of the `Translate:pivot` property of the corresponding mesh. For example, if `Translate` is \(X, Y, Z\) and `Translate:pivot` is \(X\_p, Y\_p, Z\_p\), so now, set the translate to \(-X\_p, -Y\_p, -Z\_p\).
8. Move the corresponding mesh under the XForm, this will define the parent-child relationship between them.

Verify that the resultant hierarchy looks like this:

> 

Note

If you got stuck in this this section, review the Rigged Forklift from the Content Browser, `Isaac Sim/Samples/Rigging/Forklift/forklift_b_rigged_cm.usd`, for reference.

### Assign Collision Meshes

To ensure that the collision properties are set correctly for the meshes. If no collision properties are set, then as the robot moves, it can self penetrate depending on the joint configuration.

**The correct collision meshes for the body and the lift are already set for the USD provided, so you do not need to set them up manually.** But for reference, the steps to set the collision for the `SM_Forklift_Body_B01_01` are:

1. Select the `SM_Forklift_OperatorCab_B01_01` mesh under the `lift` Xform, right click and **Add > Physics > Collider Preset**. The default collision approximation is through Convex Hull, which can be found when you scroll under the property tab for the mesh selected and find the collision section.
2. To visualize the colliders, click on the **eye** icon near the top right of the Viewport, select Show By **Type > Physics > Colliders > Selected**. Verify that you can see a Pink outline when you select the mesh that was just added to the collision. This approximation is not suitable because the collision region covers large areas that are not part of the fork and are regions that are necessary to allow other objects to exist.
3. Different approximations can be used to define different collision meshes. To see this, select one of the meshes with a collision and navigate to the colliders section of its property pane. Select the **Convex Decomposition** approximation. Update the visualization for the collision mesh. Verify that the mesh generated, this time, covers more of the collidable surface because it has a tighter approximation. Try other approximations and to see what works best for you.

Follow the same process for other meshes that interact with each other using joints. Set the **Convex Decomposition** approximation for the `SM_Forklift_BackWheelbase_B01_01` mesh that is a part of the swivel.

> 
>
> 

The process for the wheels is a little different, any collision approximation that is not smooth and captures the exact shape and curvature of the wheel causes bumpy motion when attempting to drive the wheel. This can be avoided by using a cylinder to approximate the collision mesh.

1. Go to **Create > Shape > Cylinder**.
2. Set the scale to `X=0.16`, `Y=0.16`, ``` Z=0.08`,` and Orient along ``Y=90 ```.
3. Right click and create four duplicates of this cylinder, one for each of the four front roller wheels.
4. Drag the cylinders under the respective wheel’s Xform and change their transform about all axes to `0`. This aligns the cylinder axis and the Xform axis completely.
5. Right click on the cylinder and **Add > Physics > Collider**.
6. Following the same process for the back wheel, modify the cylinder scale to `X=0.3`, `Y=0.3`, `Z=0.1`, orient along `Y=90` because of its bigger size.

All the appropriate collision meshes and properties are set up and you can move on to adding the joints.

> 

### Add Joints and Drives

In this step, add appropriate joints for the Forklift.

**Prismatic Joint**

The first joint is the joint between the forklift body and the fork. It needs linear motion between the two bodies, and the fork must move up and down relative to the body of the forklift.

1. Select the `lift` Xform and while holding the **Ctrl** key select the `body` Xform. While the two prims are highlighted, right click and **Create > Physics > Joints > Prismatic Joint**.
2. Find the newly created prismatic joint, select it. Under the properties tab, set the axis to **Z** axis, this denotes that the linear motion between the two bodies is in along the Z-axis.
3. Set the lower and upper limits for the joint in the **Property > Physics > Prismatic Joint** tab, for now set it to `-15` and `200`.
4. Add a Linear Drive for this joint by left clicking on the joint, and selecting **Add > Physics > Linear Drive**.
5. In the **Property > Physics > Drive > Linear** tab, set target position to `-15` so that the fork can start its initial position close to the ground, and set the Damping to `10000` and Stiffness to `100000`.
6. Create a Scope by right clicking on the stage and name it `lift_joint`. Drag the prismatic joint under the scope.

   > 

**Revolute Joints**

For all the roller support wheels, create revolute joints:

1. Select the `body` XForm, holding the **Ctrl** key select any of the roller wheel XForms. Right click **Create > Physics > Joint > Revolute Joint**. Verify that you see a Revolute joint added under the Xform for the wheel.
2. Verify that the joints appear in the expected location. If not, make sure that the location of the joint matches the with the rotation axis of the wheel, and make sure to set the rotation axis to “X”.
3. Follow the same process for the three remaining roller supports of the forklift.
4. Create a Scope by right clicking on the stage and name it `roller_joints`. Drag the roller joints under the scope.

Next, add the last two joints, which are responsible for driving and turning the forklift:

1. Select the `back_wheel_swivel` and `back_wheel` XForms and add a revolute joint between them. The location of this joint must match with the center of the back wheel.
2. Add an angular drive to this joint with the following properties: `Damping=10000`, `stiffness = 100`.
3. Select the `body` and `back_wheel_swivel` XForm and add a revolute joint between them. Make sure the axis of rotation is set to `Z`.
4. Change the axis of the joint to Z axis and lower with upper limits as `-60` and `60`, because this joint enables turning of the forklift. This is the range of the angles in degrees that the wheelbase would rotate.
5. Add an angular drive with the following properties: Damping = 100, stiffness = 100000.
6. Go to **Create > Scope**, name it `back_wheel_joints` and drag the rear wheel joints under the scope.
7. Remember to add a Physics Scene and Ground Plane before pressing **Play**.

   > 

### Add Articulations

The last step is adding articulation to the Forklift and putting all the joints into a single articulation chain, which makes it easier for the physics solver when solving for articulated objects such as a robot. **This has already been added for the prim in the reference USD assets**. But if not, to put select and right click on the ‘SMV\_Forklift\_B01\_01’ Xform and **Add > Physics > Articulation Root**. Under properties, disable the **Self collision** check box.

There are a few caveats for the placement of the articulation root.

If you place the articulation root on the root Xform prim of the asset, which is the standard for all Isaac Sim assets, then the simulation automatically assigns the articulation root to a rigid body in the robot, which minimizes the depth of the articulation tree.

However, if you want to manually determine the location of the articulation root, assign it to a rigid body component of the robot. It is recommended that you place the articulation root on the base or the chassis of a mobile robot or the fixed joint on a robotics arm.

Verify that the asset you have is similar to the Rigged Forklift asset provided.

### Converting Asset to a Different Unit

The original asset is in centimeters. The asset is automatically converted to meters when it is added into a scene that is in meters (see [Metrics Assembler](https://docs.omniverse.nvidia.com/extensions/latest/ext_metrics_assembler.html "(in Omniverse Extensions)")). When the asset is added to a stage, it must match the Rigged Forklift in Meters asset provided.

You can now try the Forklift, set the back wheel velocity to `-200` in the Angular Drive section for the joint. After pressing **play**, verify that you can see the forklift move forward.

## Summary

In this tutorial, you took an unrigged forklift USD asset:

* organized its structure
* added collision, joints, and drives
* turned it into a forklift that can move and driven by Isaac Sim commands

### Troubleshooting Tips

If when playing the simulation or after some movements, your robot explodes, check if any of the collision meshes are colliding with each other.

---

# Tutorial 6: Setup a Manipulator

## Learning Objectives

This is the first manipulator tutorial in a series of four tutorials. This tutorial shows how to import the UR10e robot from Universal Robots and the 2F-140 gripper from Robotiq into NVIDIA Isaac Sim from URDF files and connect them together under one articulation.

*30 Minutes Tutorial*

## Prerequisites

* If you are new to NVIDIA Isaac Sim, complete the [Wheeled Robot Set Up Tutorials](Robot_Setup.md) tutorial prior to beginning this tutorial.
* Review the ROS 2 installations [ROS 2 Installation (Default)](ROS_2.md) prior to beginning this tutorial.
* Review the URDF importer [URDF Importer Extension](Importers_and_Exporters.md) tutorial.
* In a ROS sourced terminal, install xacro using the following command (Linux only):

  ```python
  sudo apt install ros-$ROS_DISTRO-xacro
  ```
* Locate the `import_manipulator` folder in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/`.

Note

The ROS URDF import steps are tested on Linux only, it may not work on Windows. If you are using Windows, you can skip the ROS import steps and use the USD files provided in the content browser.

## Build and Install the UR Description Package (Linux only)

Isaac Sim requires Python 3.10 on Ubuntu 22.04 and Python 3.12 on Ubuntu 24.04, which is not natively supported by the ROS 2 UR description package, so we need to build the package from source.

Note

See [Isaac Sim ROS Workspaces](ROS_2.md) for more information on setting up your custom ROS 2 package in your ROS workspace.

### Clone the UR Description Package

1. Clone the UR description package from the [Universal Robots ROS 2 Description repository](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description).

   ```python
   git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Description.git
   ```
2. Switch to the branch that matches your ROS distribution.

   ROS 2 Humble

   ```python
   git checkout humble
   ```

   ROS 2 Jazzy

   ```python
   git checkout jazzy
   ```
3. Copy the repository into your Isaac Sim ROS Workspace `src` folder.

### Build the UR Description Package Using Python 3.11

1. Go to the Isaac Sim ROS Workspace, and run the following command to build the UR description package using Python 3.11.

   ```python
   ./build_ros.sh
   ```
2. Source the Python 3.11 ROS environment and launch Isaac Sim. Replace `<ROS distro>` with your ROS distribution (for example, `humble` or `jazzy`).

   ```python
   source build_ws/<ROS distro>/<ROS distro>_ws/install/local_setup.bash
   source build_ws/<ROS distro>/isaac_sim_ros_ws/install/local_setup.bash
   ./path/to/isaac-sim.sh
   ```

### Build the UR Description Package Using System ROS

1. Source your system ROS environment. Refer to [Setup ROS 2 Workspaces](ROS_2.md) for more information on setting up your ROS 2 workspace.
2. Navigate to your Isaac Sim ROS Workspace and run the following commands to build it:

   ```python
   rosdep install -i --from-path src --rosdistro $ROS_DISTRO -y
   colcon build
   source install/setup.sh
   ```

## Import the UR10e Robot (Linux only)

### Enable the ROS 2 Robot Description URDF Importer Extension

1. Go to `Window` > `Extensions`.
2. Type `URDF` in the search box, and find the `ROS 2 Robot Description URDF Importer Extension`.
3. If you can’t find it, remove the `@feature` filter from the search box.
4. If you still can’t find it, make sure Isaac Sim was launched from the same terminal where ROS was sourced.
5. Enable the extension by clicking the toggle button labeled `ENABLE`.
6. Check the box for `AUTOLOAD`, just to the right of `ENABLE`.

### Launch the URDF Publisher Topic

1. In the system ROS sourced terminal that you created earlier, launch the UR10e description by running:

   ```python
   ros2 launch ur_description view_ur.launch.py ur_type:=ur10e
   ```
2. Verify that you see a window similar to the image below:

   
3. Set up one more terminal for `rqt_graph`, to see ROS nodes and topics being published:

   ```python
   rqt_graph
   ```
4. Verify that you see a window similar to the image below:

   

Hint

If the nodes are not showing up in `rqt_graph`, press the refresh button next to the drop down menu.

### Import the UR10e Robot into Isaac Sim

1. Go to Isaac Sim.
2. Navigate to **File** > **Import from the ROS 2 URDF Node**.

   * In the **Node** field, type `robot_state_publisher`, click **Refresh**.
   * In the **Model** field, select the desired output (for example, `~/Desktop`).
   * Select **Natural Frequency** for joint configuration.
   * Select all the joints listed below, then set the **Natural Frequency** to `300` to ensure the joints are sufficiently stiff.
3. Click **Import**.

For reference, the resulting USD file is available in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/ur10e/ur/ur.usd`.

## Import the Robotiq 2F-140 Gripper (Linux only)

Use the URDF file provided by [ros-industrial-attic](https://github.com/ros-industrial-attic/robotiq/tree/kinetic-devel).
Even though this package is built for ROS1 and is deprecated, you can still adopt the URDF files and import the gripper for NVIDIA Isaac Sim.

### Convert XACRO to URDF

1. Download the Repository from [here](https://github.com/ros-industrial-attic/robotiq/tree/kinetic-devel).

   ```python
   git clone https://github.com/ros-industrial-attic/robotiq.git
   ```
2. Navigate to the `robotiq/robotiq_2f_140_gripper_visualization/urdf` folder, open each xacro file.

   * Replace `$(find robotiq_2f_140_gripper_visualization)` with the relative path to the target file (for example, `robotiq_arg2f_transimission.xacro`) from the current xacro file.

     > For example, in `robotiq_arg2f_140_model.xacro`, replace:
     >
     > ```python
     > <xacro:include filename="$(find robotiq_2f_140_gripper_visualization)/urdf/robotiq_arg2f_transmission.xacro" />
     > ```
     >
     > With:
     >
     > ```python
     > <xacro:include filename="./robotiq_arg2f_transmission.xacro" />
     > ```
   * Replace `package://` with the relative path to the target file (for example, `robotiq_arg2f_${stroke}_inner_finger.stl`) from the current xacro file.

     > For example, in `robotiq_arg2f_140_model.xacro`, replace:
     >
     > > ```python
     > > <mesh filename="package://robotiq_2f_140_gripper_visualization/meshes/visual/robotiq_arg2f_${stroke}_inner_finger.stl" />
     > > ```
     >
     > With:
     >
     > ```python
     > <mesh filename="../meshes/visual/robotiq_arg2f_${stroke}_inner_finger.stl" />
     > ```
3. Convert the xacro files to URDF format:

   ```python
   xacro robotiq_arg2f_140_model.xacro > robotiq_2f_140.urdf
   ```

   If you encounter the error `xacro: command not found`, you need to install xacro.

   * Install xacro using the following command:
   > ```python
   > sudo apt install ros-$ROS_DISTRO-xacro
   > ```

For reference, the resulting URDF files is available in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/robotiq_2f_140_urdf/urdf/robotiq_2f_140.urdf`.

### Import Robotiq 2F-140 Gripper into Isaac Sim

1. Go to Isaac Sim.
2. Let’s create a new stage by going to **File** > **New**.
3. Navigate to **File** > **Import**.
4. Select the `robotiq_2f_140.urdf` file that you imported from the previous step.
5. In the import settings:

   * For USD Output, navigate to your desktop using file browser and select **Desktop** this will be the output location of the gripper USD.
   * For `finger_joint`, set the Natural Frequency to `300`.
   * For the other joints of target `Mimic`, set the Natural Frequency to `2500`.
6. Click `Import` to complete the process.

   

For reference, the resulting USD file is available in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/robotiq_2f_140/robotiq_2f_140.usd`.

#### Expected Parameters for Finger and Knuckle Joints

| Joint Name | Lower Limit | Upper Limit | Gearing | Stiffness | Damping | Max Force |
| --- | --- | --- | --- | --- | --- | --- |
| Finger Joint | 0 | 40.107 | N/A | 37.51957 | 0.00125 | 1000 |
| Left inner Finger | -8.021 | 48.128 | -1 | N/A | N/A | N/A |
| Left Inner Knuckle | -48.128 | 8.021 | 1 | N/A | N/A | N/A |
| Right inner Knuckle | -48.128 | 8.021 | 1 | N/A | N/A | N/A |
| Right outer knuckle | -48.128 | 8.021 | 1 | N/A | N/A | N/A |
| Right inner Finger | -8.021 | 48.128 | -1 | N/A | N/A | N/A |

#### Expected Parameters for Mimic Joints

* Reference Joint: `/robotiq_arg2f_140_model/joints/finger_joint`
* Reference Joint Axis: `rotX`
* Natural Frequency: `2500`
* Damping Ratio: `0.005`

## Connect the UR10e Robot with the Robotiq 2F-140 Gripper

Much like a real robot can have its tools changed for different tasks, simulated robots benefit from the same capability. This section outlines two methods to connect the UR10e robot with the Robotiq 2F-140 gripper:

* **Option 1**, shows how to connect the gripper to the robot directly using a fixed joint with a shared articulation.
* **Option 2**, shows how to use the robot assembler and variant to connect the end effectors to the robot. Depending on the variant selected, the gripper will be added as a payload, which allows us to load or unload the different end effectors depending on which variant is enabled.

### Option 1: Connect the UR10e with the Robotiq 2F-140 Gripper using the GUI

1. Open the UR10e USD file created from the last activity (`ur.usd`).
2. Drag and drop the `robotiq_2f_140.usd` file, we created earlier, into the stage.
3. Rename the `robotiq_2f_140.usd` prim to `ee_link`.
4. Set the `ee_link` xform to the position and orientation of `wrist_3_link`.

   ```python
   Translate (1.18425, 0.2907, 0.06085)
   Orient (-90, 0, -90)
   ```
5. Select `ee_link/root_joint`.
6. Go to the `Physics Articulation Root` section in the Property Editor, remove the `Articulation Root`.

   > Only select a single articulation for the robot.
7. Go down to the `Joints` section in the Property Editor.
8. Set `Body0` to `/ur/wrist_3_link`, to joint the end effector to the robot.

   

Nest the UR10e robot schema into the 2F-140 gripper’s robot schema:

1. Select the `ur` prim.
2. Go down to the `IsaacRobotAPI` section in the Property Editor, and add `/ur/ee_link` to both the `isaac:physics:robotjoints` and `isaac:physics:robotLinks` fields, to make sure that the UR10e robot’s robot schema includes the 2F-140 gripper’s robot schema.

Your robot is now connected to the gripper, and you can test your robot in [Tutorial 6: Setup a Manipulator](#).

For reference, we also provide the resulting USD file in Content Browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/ur10e/ur/ur_gripper_manual.usd`.

### Option 2: Connect the UR10e with the Robotiq 2F-140 Gripper using the Robot Assembler

Alternatively, you can use the Robot Assembler to connect the UR10e with the Robotiq 2F-140 gripper. The robot assembler will add the gripper as a variant to a sublayer of the base robot,
giving you greater flexibility to switch between different end effectors.

1. Open the UR10e USD file created from the last activity (`ur.usd`).
2. Drag and drop the `robotiq_2f_140.usd` file we created earlier into the stage.
3. Rename the `robotiq_2f_140` prim to `ee_link`.
4. Open the robot assembler by going to **Tools** > **Robotics** > **Asset Editor** > **Robot Assembler**.

   * In **Base Robot**, set **Select Base Robot** to `/ur`, **Attach Point** to `wrist_3_link`.
   * In **Attach Robot**, set **Select Attach Robot** to `/ur/ee_link`, **Attach Point** to `robotiq_arg2f_base_link`.
   * Set **Assembly Namespace** to `ee_link`.
5. Click **Begin Assembling Process** to start the process.

   
6. Adjust the attachment point orientation to make sure the end effector is attached to the gripper correctly. Rotate the gripper 90 degrees around the z-axis by clicking **Z +90**.

   
7. Click **Assemble and Simulate** to test the process.
8. Click **End Simulation And Finish** to complete the process.

#### Run the Simulation

1. In the Stage panel, select the **ur** prim.
2. In the Property Editor at the bottom right, find the **Variants** section.
3. Beside **ee\_link**, select **None** and the gripper will be removed from the robot.
4. Beside **ee\_link**, select **robotiq\_2f\_140** and the gripper will be added to the robot.
5. Save the asset by going to **File** > **Save** or press **Ctrl+S**.

Note

The completed robotics arm asset is available in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/ur10e/ur/ur_gripper.usd`.

## Summary

In this tutorial, you took the UR10e robot from Universal Robots and the 2F-140 gripper from Robotiq and imported them into NVIDIA Isaac Sim from URDF files and connected them together under one articulation using the GUI and Robot Assembler.

---

# Tutorial 7: Configure a Manipulator

## Learning Objectives

This is the second manipulator tutorial in a series of four tutorials. This tutorial shows how to configure physics, joint effort limits, and gains for the UR10e robot from Universal Robots and the 2F-140 gripper from Robotiq.

*30 Minutes Tutorial*

## Prerequisites

* Review [Tutorial 6: Setup a Manipulator](Robot_Setup.md) tutorial prior to beginning this tutorial. The steps here continue from the asset built in the previous tutorial.

Note

If you have not completed the previous tutorial, you can find the prebuilt asset in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/Import_Manipulator/ur10e/ur/ur_gripper.usd`.

## Adjust the Articulation for Manipulation Tasks

Adjust the articulation for the UR10e robot to make it more stable and accurate for manipulation tasks. Open the physics layer for the UR10e robot.
The physics layer is located in the `configuration` folder with subfix `_physics`.

1. In the Stage panel, select the **ur/root\_joint** prim.
2. In the Property Editor at the bottom right, scroll down to the **Physics/Articulation** section.
3. Select **Articulation Enabled**.
4. Increase the **Solver Position Iterations Count** to `64`.
5. Increase the **Solver Velocity Iterations Count** to `4`.

   Note

   The **Solver Position Iterations Count** and **Solver Velocity Iterations Count** are used to control the accuracy of the simulation.

   For a complex robot with many degrees of freedoms and mimic joints, increasing these values will make the simulation more accurate at the cost of performance.
   See [articulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html#articulation-drive-stability) for more information.
6. Decrease **Sleep Threshold** to `0.00005`, this lowers the threshold for the robot to go to sleep when it is not moving. see [rigid body dynamics documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/RigidBodyDynamics.html#sleeping) for more information.
7. Decrease the **Stabilization Threshold** to `0.00001`, this lowers the threshold for the robot to start stabilizing itself when it is not moving. see [articulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html#articulation-drive-stability) for more information.
8. **Ctrl + S** to save the changes.

   

Note

See [PhysX Best Practice Guide](https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/BestPractices.html#jointed-objects-are-unstable) for tuning the articulation for manipulation tasks.

## Add Physics Materials

Add physics materials to the robot gripper to make it more realistic and stable for manipulation tasks.

1. Open the physics layer from the 2F-140 gripper asset from the last tutorial. It is located in the `configuration` folder with suffix `_physics`.

   Note

   If you have not completed the previous tutorial, you can find the prebuilt asset in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/import_manipulator/ robotiq_2f_140/configuration/robotiq_2f_140_physics.usd`.
2. Right click on the **robotiq\_arg2f\_140\_model** prim and select **Create** > **Physics** > **Physics Material**, select **Rigid Body Material**. This will add a physics material attribute to the gripper.
3. Drag the physics material to the **robotiq\_arg2f\_140\_model/Looks** folder.
4. In the properties panel, scroll down to the **Physics/Rigid Body Material** section and set the **static friction** to **1.0** and **dynamic friction** to **1.0**. For your robot, match the friction values to the robot’s surface friction coefficients.
5. Apply the physics material to the gripper finger tip.
   - Select the `colliders/left_inner_finger/mesh_1/box` and in the properties panel, scroll down to the **Physics/Physics material on selected Material** section.
   - Select the **Physics Material** you just created at `/World/robotiq_arg2f_140_model/Looks/finger`.
6. Repeat the same process for the `colliders/right_inner_finger/mesh_1/box` prim.

Note

See [Adding Props](Python_Scripting_and_Tutorials.md) for more information on how to add physics materials to the robot.

## Configure Joint Effort Limits

In the physics layer of the robotiq\_arg2f\_140\_model asset from the previous step, let’s configure the joint effort limits for the gripper.

1. In the **Stage** panel, select the `robotiq_arg2f_140_model/joints/finger_joint` prim. This is the joint that controls the gripper fingers, all other gripper joints are `Mimic` joints.
2. In the **Property Editor** at the bottom right, scroll down to the `Drive/Angular/Max Force` section.
3. Set the **Max Force** to `200`. This is the maximum force that can be applied to the gripper fingers. For your robot, match the max force to the robot’s joint torque limits.
4. **Ctrl + S** to save the changes.

Note

When the max force is very high, you might need to increase the physics step frequency (`Time Step per Second`) to avoid penetration and instabilities.

## Inspect the Robot Articulation

Let’s inspect the robot articulation to verify the joint effort limits are applied correctly. Open the top level `ur` asset that you built in the previous tutorial.
This asset references the physics layers that you modified, so all the changes you made to the physics layer will be reflected in this asset.

Note

You can find the prebuilt asset in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd`.

1. Open the **Physics Inspector** through **Tools** > **Physics** > **Physics Inspector**.
2. Select the UR articulation in the stage, click on the circular arrow icon to refresh the articulation.
3. Try changing the target position with the blue slider and verify that the DOF position reaches the target specified.

   
4. Close the **Physics Inspector** window/panel (discarding any changes authored by this tool, if prompted).

   Warning

   Since the Physics Inspector partially initializes `omni.physx`, it is expected for general simulations to not behave properly when the tool is opened.

## Tune Gains Using the Gain Tuner

Use the [Gain Tuner Extension](Robot_Setup.md) to verify the gains for the UR robot and the gripper fingers.
To critically damp the robot gains, set the `Nat. Freq.` to `0.5` and the `Damping Ratio` to `1.0`.

1. Go to **Tools** > **Robotics** > **Asset Editors** > **Gain Tuner**.
2. On the **Gain Tuner** window, on the **Select Robot** dropdown, select the **ur** articulation in the stage.
3. In the **Tune Gains** panel, you can adjust the gains for the robot and the gripper fingers joints. Test it with the **Test Gains Settings** panel.

Hint

We recommend determining the gains for a small group of joints first, if it is difficult to tune the gains for the whole robot. Below are some tips for tuning the gains:

* If the resulting plot shows the robot is undershooting the target position, you can increase the `Nat. Freq.` slightly.
* If the resulting plot shows the robot is overshooting the target position, you can decrease the `Nat. Freq.` slightly and increase the `Damping Ratio`.
* Disabling gravity can help you see the gains more clearly.
* Only gain test the joints that are expected to be moving together, the gain test order can be selected by the **Sequence** dropdown.
* Reduce the maximum speed of a joint that you are tuning, if it is not expected to be commanded to move that fast in practice. The default values in the Gains Test are the maximum velocity written into the USD.

Note

See [Gain Tuner Extension](Robot_Setup.md) for more information on the Gain Tuner.

See [Tutorial 11: Tuning Joint Drive Gains](Robot_Setup.md) for more information on how to tune the gains for the robot.

The complete asset for this tutorial can be found in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd`.

## Summary

In this tutorial, you learned how to configure the physics, joint effort limits, and gains for the UR10e robot from Universal Robots and the 2F-140 gripper from Robotiq using the Gain Tuner.
You added physics materials to the robot gripper to make it more realistic and stable for manipulation tasks.
You inspected the robot articulation and tuned the gains for the robot and the gripper fingers joints using the Physics Inspector.

---

# Tutorial 8: Generate Robot Configuration File

## Learning Objectives

This is the third manipulator tutorial in a series of four tutorials. This tutorial will show you how to generate the robot configuration file for the UR10e robot from Universal Robots and the 2F-140 gripper from Robotiq.
These robot configuration files provide information about the robot’s kinematics, dynamics, and other properties that are used in RMPFlow, CuMotion, and Lula kinematics solvers.

*30 Minutes Tutorial*

## Prerequisites

* Review [Tutorial 7: Configure a Manipulator](Robot_Setup.md) tutorial prior to beginning this tutorial, continue the following steps from the asset built in the previous tutorial.

Note

If you have not completed the previous tutorial, you can find the prebuilt asset in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/Configure_Manipulator/ur10e/ur/ur_gripper.usd`.

## Generate Robot URDF

Generate the robot URDF file from the UR10e robot and the 2F-140 gripper.

### Enable the Isaac Sim USD to URDF Exporter Extension

1. Go to **Window** > **Extensions**.
2. Type **URDF** in the search box, and find the **Isaac Sim USD to URDF Exporter Extension**.
3. If you can’t find it, remove the **@feature** filter from the search box.
4. Enable the extension by clicking the toggle button labeled **ENABLE**.
5. Check the box for **AUTOLOAD**, just to the right of **ENABLE**.

### Export the URDF File

1. Open the `ur_gripper.usd` asset you made in the previous tutorial, or use the completed asset provided above.
2. Click **File** > **Export URDF**.
3. In File name on the bottom left corner, save the file name to `ur_gripper.urdf`.
4. In the **Mesh Directory Path** field, select the correct folder path to save the URDF meshes.
5. Click **Export**.

Note

Learn more about the USD to URDF Exporter Extension in the [USD to URDF Exporter Extension](Importers_and_Exporters.md) manual.

## Generate Lula Robot Description Files and Collision Spheres

Generate the Lula robot description files and collision spheres for the UR10e robot and the 2F-140 gripper.

### Enable the Isaac Sim Lula Extension

1. Go to **Window** > **Extensions**.
2. Type **Lula** in the search box, and find the **Isaac Sim Lula** Extension.
3. If you can’t find it, remove the **@feature** filter from the search box.
4. Enable the extension by clicking the toggle button labeled **ENABLE**.
5. Check the box for **AUTOLOAD**, just to the right of **ENABLE**.

### Prepare the Robot Asset for Lula

The Lula robot description editor does not support instantiable meshes. You must prepare the robot asset for Lula by removing the instantiable meshes.

1. Open the `ur_gripper.usd` asset you made in the previous tutorial, or use the completed asset provided above.
2. Select all `visuals` and `collisions` prims on the stage.
3. On the property editor, uncheck the **Instantiable** field.

   Hint

   You can use the search feature to find the `visuals` and `collisions` prims by searching for `visuals` and `collisions` respectively.

The completed asset for this tutorial can be found in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper_lula.usd`.

### Configure Joints in Lula Robot Description Editor

1. Press **PLAY** to start the simulation.
2. Click **Tools** > **Robotics > \*\*Lula Robot Description Editor**.
3. In the **Selection Panel**, select the **ur** articulation.
4. Go down to the **Set Joint Properties** section.
5. For each of the Universal Robots joints, set the **Joint Status** to **Active Joint**, keep the other settings as default.
6. Keep the Robotiq 2F-140 gripper joints as **Fixed Joint**, so the robotics controller will not attempt to move the gripper joints to optimize for the robot position.

Hint

The gripper and arm usually are controlled separately. Because the Lula framework does not actually control the gripper during collision checking, the cspace does not need to include the gripper joints.

Important

**Do not stop the simulation**, you will need it to generate the collision spheres.

Pay attention to the default values of the joints in `cspace_to_urdf_rules`.
They must be the same positions with the initial pose in the manipulator USD, or you need to reset the robot joint positions to these initial positions during task initialization.

### Generate Collision Spheres

1. **Do not stop the simulation**, or exit the Lula Robot Description Editor, or you will need to redo the previous steps.
2. Go down to the **Link Sphere editor** section.
3. For each of the robot links that you want ot generate collision spheres for, in the **Selection Panel/Select link**, select the link. Use **upper\_arm\_link** as an example.
4. In the **Link Sphere editor/Generate Spheres/Select Mesh** dropdown menu, select the mesh that the collision spheres are based on. For example, select `/collisions/upperarm/mesh`.
5. Set the **Radius Offset** to `0.03`. This is the offset between the mesh radius and the collision sphere radius.
6. Set the **Number of Spheres** to `8`. This is the number of collision spheres to generate. Validate that you see eight red spheres on the **upper\_arm\_link**.
7. Optionally, adjust the **Sphere Position** by left clicking on the spheres and dragging them around.
8. Click **Generate Spheres**, the sphere will turn a cyan color to indicate that the collision spheres have been generated.
9. Repeat the same steps for all the other links in the **ur** articulation, including the gripper links.

   

   Important

   **Do not stop the simulation**, you will need it to generate the robot configuration file.
10. Verify that the completed asset looks like the following image:

    

The following suggestions can help you tune the collision spheres:

> 1. In general, make the collision spheres large enough to encompass the link, but not too large to cause solver issues.
> 2. When choosing the size and number of collision spheres, the more collision spheres the more accurate the collision detection will be, but too many collision spheres will slow down the solver.
> 3. Unless you have specified collider meshes, there’s no restrictions to generate collision spheres on the collision meshes of the links only. If the visual mesh give you better collision mesh approximation, you can generate the collision spheres on the visual mesh.
> 4. For longer arm links, it is generally easier to use the method above to only generate collision spheres on the ends of the link, then use `Link Sphere editor/Generate Spheres/Add Spheres` to add the collision spheres to the entire link evenly.
> 5. If the sphere sizes are too small or too large, you can use `Link Sphere editor/Generate Spheres/Scale Spheres in Link` to scale the sphere sizes.
> 6. The generate spheres utility is not guaranteed to work for all meshes. It only works for water-tight triangle meshes. If the automatic generator doesn’t work for a link, add the spheres and connect them by hand.

### Export the Lula Robot Description File

1. **Do not stop the simulation or save the file**, you need it to export the robot configuration file.
2. In the **Lula Robot Description Editor**, go to the very bottom and find the **Export To File** section.
3. Expand **Export to Lula Robot Description File**, click the file icon and specify the file name to `ur10e.yaml`.
4. Click **Save** to export the robot configuration file:

   
5. You can also export the cuMotion XRDF file by going to **Export To File** > **Export to cuMotion XRDF** and specify the file name to `ur10e.xrdf`.
6. Stop the simulation after the robot configuration files are exported.

See [Lula Robot Description and XRDF Editor](Robot_Simulation.md) for more information on the robot description files.

## Summary

In this tutorial, you have learned how to generate the robot configuration file for the UR10e robot and the 2F-140 gripper using the [Lula Robot Description and XRDF Editor](Robot_Simulation.md)
and the [USD to URDF Exporter Extension](Importers_and_Exporters.md) extensions.

---

# Tutorial 9: Pick and Place Example

## Learning Objectives

This is the final manipulator tutorial in a series of four tutorials. This tutorial tie everything together by showing how to use the UR10e robot and the 2F-140 gripper to follow a target and pick up a block.
We will be using the robot imported in [Tutorial 6: Setup a Manipulator](Robot_Setup.md), tuned in [Tutorial 7: Configure a Manipulator](Robot_Setup.md), and the robot configuration file generated in [Tutorial 8: Generate Robot Configuration File](Robot_Setup.md).

In this tutorial, we will be using the Lula kinematics solver to follow a target and the RMPFlow to pick up a block.

Note

All the files created in this tutorial are available at `standalone_examples/api/isaacsim.robot.manipulators/ur10e` for verification.

*30 Minutes Tutorial*

## Prerequisites

* Review [Tutorial 8: Generate Robot Configuration File](Robot_Setup.md) tutorial prior to beginning this tutorial, continue the following steps from the asset built in the previous tutorial.

Note

If you have not completed the previous tutorial, you can find the prebuilt asset in the content browser at `Isaac Sim/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd`.

## Gripper Control Example

The script below uses the `Parallel Gripper` class to control the gripper joints and the `Manipulator` class to control the robot joints.
Steps 0 to 400 close the gripper slowly. Steps 400 to 800, open the gripper slowly, and then reset the gripper position to 0.

Note

The provided script can be run using:

```python
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/gripper_control.py
```

gripper\_control.py

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.storage.native import get_assets_root_path

parser = argparse.ArgumentParser()
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode")
args, unknown = parser.parse_known_args()

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise Exception("Could not find Isaac Sim assets folder")
asset_path = assets_root_path + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/ur")
# define the gripper
gripper = ParallelGripper(
    # We chose the following values while inspecting the articulation
    end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
    joint_prim_names=["finger_joint"],
    joint_opened_positions=np.array([0]),
    joint_closed_positions=np.array([40]),
    action_deltas=np.array([-40]),
    use_mimic_joints=True,
)
# define the manipulator
my_ur10 = my_world.scene.add(
    SingleManipulator(
        prim_path="/ur",
        name="ur10_robot",
        end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
        gripper=gripper,
    )
)

my_world.scene.add_default_ground_plane()
my_world.reset()

i = 0
reset_needed = False
while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
        i += 1
        gripper_positions = my_ur10.gripper.get_joint_positions()
        if i < 400:
            # close the gripper slowly
            my_ur10.gripper.apply_action(ArticulationAction(joint_positions=[gripper_positions[0] + 0.1]))
        if i > 400:
            # open the gripper slowly
            my_ur10.gripper.apply_action(ArticulationAction(joint_positions=[gripper_positions[0] - 0.1]))
        if i == 800:
            i = 0
    if args.test is True:
        break

simulation_app.close()
```

## Follow Target Example using Lula Kinematics Solver

Create a follow target task using the Lula kinematics solver, where you can specify the target position using a cube and the robot will move its end effector to the target position.
The inverse kinematics solver will use the Lula robot descriptor created in the [Tutorial 8: Generate Robot Configuration File](Robot_Setup.md) tutorial.

The generated robot descriptor file is available at `source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/robot_descriptor.yaml`.

Note

The provided script can be run using:

```python
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/follow_target_example.py
```

Move the cube to the target location and run the script to see the robot move its end effector to the target location.

The `ik_solver.py` script initializes the `KinematicsSolver` class and the `LulaKinematicsSolver` class.

controllers/ik\_solver.py

```python
import os
from typing import Optional

from isaacsim.core.prims import Articulation
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver

class KinematicsSolver(ArticulationKinematicsSolver):
    def __init__(self, robot_articulation: Articulation, end_effector_frame_name: Optional[str] = None) -> None:
        # TODO: change the config path
        self._kinematics = LulaKinematicsSolver(
            robot_description_path=os.path.join(os.path.dirname(__file__), "../rmpflow/robot_descriptor.yaml"),
            urdf_path=os.path.join(os.path.dirname(__file__), "../rmpflow/ur10e.urdf"),
        )
        if end_effector_frame_name is None:
            end_effector_frame_name = "ee_link_robotiq_arg2f_base_link"
        ArticulationKinematicsSolver.__init__(self, robot_articulation, self._kinematics, end_effector_frame_name)
        return
```

The `follow_target.py` script initializes the `FollowTarget` class and sets up the `manipulator` and `parallel_gripper` objects.

tasks/follow\_target.py

```python
from typing import Optional

import isaacsim.core.api.tasks as tasks
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path

# Inheriting from the base class Follow Target
class FollowTarget(tasks.FollowTarget):
    def __init__(
        self,
        name: str = "ur10e_follow_target",
        target_prim_path: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        tasks.FollowTarget.__init__(
            self,
            name=name,
            target_prim_path=target_prim_path,
            target_name=target_name,
            target_position=target_position,
            target_orientation=target_orientation,
            offset=offset,
        )
        return

    def set_robot(self) -> SingleManipulator:

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise Exception("Could not find Isaac Sim assets folder")
        asset_path = (
            assets_root_path + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
        )
        add_reference_to_stage(usd_path=asset_path, prim_path="/ur")
        # define the gripper
        gripper = ParallelGripper(
            # We chose the following values while inspecting the articulation
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0]),
            joint_closed_positions=np.array([40]),
            action_deltas=np.array([-40]),
            use_mimic_joints=True,
        )
        # define the manipulator
        manipulator = SingleManipulator(
            prim_path="/ur",
            name="ur10_robot",
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            gripper=gripper,
        )
        return manipulator
```

The `follow_target_example.py` script initializes the `FollowTarget` task and the `KinematicsSolver` created in the previous step with a target location for the cube to be followed by the end effector.

follow\_target\_example.py

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run in test mode.")
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from controller.ik_solver import KinematicsSolver
from isaacsim.core.api import World
from tasks.follow_target import FollowTarget

my_world = World(stage_units_in_meters=1.0)
# Initialize the Follow Target task with a target location for the cube to be followed by the end effector
my_task = FollowTarget(name="ur10e_follow_target", target_position=np.array([0.5, 0, 0.5]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("ur10e_follow_target").get_params()
target_name = task_params["target_name"]["value"]
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)

# initialize the ik solver
ik_solver = KinematicsSolver(my_ur10e)
articulation_controller = my_ur10e.get_articulation_controller()

# run the simulation
i = 0
while simulation_app.is_running() and (not args.test or i < 100):
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

        observations = my_world.get_observations()
        actions, succ = ik_solver.compute_inverse_kinematics(
            target_position=observations[target_name]["position"],
            target_orientation=observations[target_name]["orientation"],
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            print("IK did not converge to a solution.  No action is being taken.")
    i += 1
simulation_app.close()
```

## RMP Flow Configuration

Use RMPFlow to control the end effector. See [RMPflow](Robot_Simulation.md) for more details about RMPFlow.

The `ur10e_rmpflow_common.yaml` file is available at `source/standalone_examples/api/isaacsim.robot.manipulators/ur10e/rmpflow/ur10e_rmpflow_common.yaml`,
it specifies various parameters for the RMPFlow controller.

rmpflow/ur10e\_rmpflow\_common.yaml

```python
 1joint_limit_buffers: [.01, .01, .01, .01, .01, .01]
 2rmp_params:
 3    cspace_target_rmp:
 4        metric_scalar: 50.
 5        position_gain: 100.
 6        damping_gain: 50.
 7        robust_position_term_thresh: .5
 8        inertia: 1.
 9    cspace_trajectory_rmp:
10        p_gain: 100.
11        d_gain: 10.
12        ff_gain: .25
13        weight: 50.
14    cspace_affine_rmp:
15        final_handover_time_std_dev: .25
16        weight: 2000.
17    joint_limit_rmp:
18        metric_scalar: 1000.
19        metric_length_scale: .01
20        metric_exploder_eps: 1e-3
21        metric_velocity_gate_length_scale: .01
22        accel_damper_gain: 200.
23        accel_potential_gain: 1.
24        accel_potential_exploder_length_scale: .1
25        accel_potential_exploder_eps: 1e-2
26    joint_velocity_cap_rmp:
27        max_velocity: 1.
28        velocity_damping_region: .3
29        damping_gain: 1000.0
30        metric_weight: 100.
31    target_rmp:
32        accel_p_gain: 30.
33        accel_d_gain: 85.
34        accel_norm_eps: .075
35        metric_alpha_length_scale: .05
36        min_metric_alpha: .01
37        max_metric_scalar: 10000
38        min_metric_scalar: 2500
39        proximity_metric_boost_scalar: 20.
40        proximity_metric_boost_length_scale: .02
41        xi_estimator_gate_std_dev: 20000.
42        accept_user_weights: false
43    axis_target_rmp:
44        accel_p_gain: 210.
45        accel_d_gain: 60.
46        metric_scalar: 10
47        proximity_metric_boost_scalar: 3000.
48        proximity_metric_boost_length_scale: .08
49        xi_estimator_gate_std_dev: 20000.
50        accept_user_weights: false
51    collision_rmp:
52        damping_gain: 50.
53        damping_std_dev: .04
54        damping_robustness_eps: 1e-2
55        damping_velocity_gate_length_scale: .01
56        repulsion_gain: 800.
57        repulsion_std_dev: .01
58        metric_modulation_radius: .5
59        metric_scalar: 10000.
60        metric_exploder_std_dev: .02
61        metric_exploder_eps: .001
62    damping_rmp:
63        accel_d_gain: 30.
64        metric_scalar: 50.
65        inertia: 100.
66canonical_resolve:
67    max_acceleration_norm: 50.
68    projection_tolerance: .01
69    verbose: false
70body_cylinders:
71    - name: base
72    pt1: [0,0,.10]
73    pt2: [0,0,0.]
74    radius: .2
75body_collision_controllers:
76    - name: ee_link_robotiq_arg2f_base_link
77    radius: .05
```

## Follow Target Example using RMP Flow

Create an RMP flow controller to move the robot end effector to the target location.

Note

The provided script can be run using:

```python
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/follow_target_example_rmpflow.py
```

The `rmpflow.py` initializes Lula motion generation policy using the `ur10e_rmpflow_common.yaml` file above, the `ur10e.urdf` and the robot descriptor file created in [Tutorial 8: Generate Robot Configuration File](Robot_Setup.md).

controllers/rmpflow.py

```python
import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import Articulation

class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(os.path.dirname(__file__), "../rmpflow/robot_descriptor.yaml"),
            rmpflow_config_path=os.path.join(os.path.dirname(__file__), "../rmpflow/ur10e_rmpflow_common.yaml"),
            urdf_path=os.path.join(os.path.dirname(__file__), "../rmpflow/ur10e.urdf"),
            end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
```

The `follow_target_example_rmpflow.py` script initializes the `FollowTarget` task and the `RMPFlowController` created in the previous step with a target location for the cube to be followed by the end effector.

follow\_target\_example\_rmpflow.py

```python
import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import Articulation

class RMPFlowController(mg.MotionPolicyController):
    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:

        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(os.path.dirname(__file__), "../rmpflow/robot_descriptor.yaml"),
            rmpflow_config_path=os.path.join(os.path.dirname(__file__), "../rmpflow/ur10e_rmpflow_common.yaml"),
            urdf_path=os.path.join(os.path.dirname(__file__), "../rmpflow/ur10e.urdf"),
            end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        mg.MotionPolicyController.__init__(self, name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
        return

    def reset(self):
        mg.MotionPolicyController.reset(self)
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
```

## Basic Pick and Place Task using RMP Flow

Use the RMPFlow controller to pick up a block and place it in a target location.

Note

The provided script can be run using:

```python
./python.sh standalone_examples/api/isaacsim.robot.manipulators/ur10e/pick_up_example.py
```

The `controllers/pick_place.py` script creates a `PickPlace` controller that will pick up a block and place it in a target location.

controllers/pick\_place.py

```python
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers import ParallelGripper

from .rmpflow import RMPFlowController

class PickPlaceController(manipulators_controllers.PickPlaceController):
    def __init__(
        self, name: str, gripper: ParallelGripper, robot_articulation: SingleArticulation, events_dt=None
    ) -> None:
        if events_dt is None:
            events_dt = [0.005, 0.002, 1, 0.05, 0.0008, 0.005, 0.0008, 0.1, 0.0008, 0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.6,
        )
        return
```

The `tasks/pick_place.py` script creates a `PickPlace` task that sets up the UR10e manipulator and the gripper to pick up a block and place it in a target location.

tasks/pick\_place.py

```python
import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.grippers import ParallelGripper

from .rmpflow import RMPFlowController

class PickPlaceController(manipulators_controllers.PickPlaceController):
    def __init__(
        self, name: str, gripper: ParallelGripper, robot_articulation: SingleArticulation, events_dt=None
    ) -> None:
        if events_dt is None:
            events_dt = [0.005, 0.002, 1, 0.05, 0.0008, 0.005, 0.0008, 0.1, 0.0008, 0.008]
        manipulators_controllers.PickPlaceController.__init__(
            self,
            name=name,
            cspace_controller=RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
            ),
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.6,
        )
        return
```

The `pick_place_example.py` script puts everything together and runs the simulation.

Important

Make sure to tune the `end_effector_offset` parameter to get the best results, this is the offset between the end effector link on the robot and optimal grasp position for the claw.

pick\_place\_example.py

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="Run in test mode.")
args, unknown = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from controller.ik_solver import KinematicsSolver
from isaacsim.core.api import World
from tasks.follow_target import FollowTarget

my_world = World(stage_units_in_meters=1.0)
# Initialize the Follow Target task with a target location for the cube to be followed by the end effector
my_task = FollowTarget(name="ur10e_follow_target", target_position=np.array([0.5, 0, 0.5]))
my_world.add_task(my_task)
my_world.reset()
task_params = my_world.get_task("ur10e_follow_target").get_params()
target_name = task_params["target_name"]["value"]
ur10e_name = task_params["robot_name"]["value"]
my_ur10e = my_world.scene.get_object(ur10e_name)

# initialize the ik solver
ik_solver = KinematicsSolver(my_ur10e)
articulation_controller = my_ur10e.get_articulation_controller()

# run the simulation
i = 0
while simulation_app.is_running() and (not args.test or i < 100):
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()

        observations = my_world.get_observations()
        actions, succ = ik_solver.compute_inverse_kinematics(
            target_position=observations[target_name]["position"],
            target_orientation=observations[target_name]["orientation"],
        )
        if succ:
            articulation_controller.apply_action(actions)
        else:
            print("IK did not converge to a solution.  No action is being taken.")
    i += 1
simulation_app.close()
```

## Advanced Pick and Place Task using RMPFlow and Foundation Pose

In the pick and place example above, you used the RMPFlow controller to pick up a block and place it in a target location. However there are some limitations to this approach.

* The robot gets the cube pose directly from the simulator observation, which does not translate to the real world.
* The class set up is limited to the cube, and in real life different objects have different shapes and sizes, and different grasping strategies.

To address these limitations, see [Isaac Manipulator](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_manipulator/index.html) tutorials for more advanced pick and place tasks.

## Summary

In this tutorial, you learned how to use the Lula kinematics solver to follow a target and the RMPFlow to pick up a block.

---

# Tutorial 10: Rig Closed-Loop Structures

Some models are challenging to represent. Robots and grippers still have unique features and structures that are uncommon. In this document you learn some techniques to model these unique features and learn a general approach for managing these unique configurations.

## Learning Objectives

In this tutorial, you will:

* Use USD Layers to edit and test assets
* Add materials and adjust joints post CAD import
* Break a closed loop articulation chain
* Add joint drives, including mimic joints
* Adjust collision shapes
* Test grippers by building a test setup and using a gripper controller Omnigraph

*30 Minutes Tutorial*

Start with a [Robotiq 2F-85 Parallel Gripper](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) STP file imported into an [Onshape document](https://cad.onshape.com/documents/02712153b53a69118b4e5c99/w/e4160a7cfa8bb14f2585a92f/e/6d63d85251b40eee71da6b56) and with joints modeled. This tutorial does not directly cover tuning the joints. Instead, tuned parameters are provided when configuring the asset. To learn more about gains tuning see [Tutorial 11: Tuning Joint Drive Gains](Robot_Setup.md) and [Gain Tuner Extension](Robot_Setup.md).

## Getting Started

**Prerequisite**

* Complete the [Quick Tutorials](Quick_Tutorials.md) series to learn the basic core concepts of how to navigate inside NVIDIA Isaac Sim.
* Complete the [Assemble a Simple Robot](Robot_Setup.md) and [Adding Sensors and Cameras](Robot_Setup.md) tutorials to learn the concepts of rigid body API, collision API, joints, drives, and articulations.
* Read [Onshape importer](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html "(in Omniverse Extensions)") and watch the videos on rigging the robot in Onshape.
* Have a version of the Robotiq 2F-85 Gripper imported in Onshape and model the joints that connect the fingers together and to the body.

Note

The Onshape document used in this tutorial is publicly available. The imported USD asset is located at `Samples/Rigging/Gripper/Robotiq 2F-85` to get started.

## Rigging the Robot

### Using Layers to Edit and Test an Asset

All the rigid body, masses, and joint definition are done in [Onshape](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html#configuring-mates-for-physics). After they are imported to Isaac Sim, the asset contains basic joint information and rigid bodies setup. You must complete a few additional steps to make the asset fully functional.

Instead of opening the original asset, edit the asset using **layers**. Layers allow for building a scene on top of a root asset and saving it without changing the underlying root layer assets. For example, you can add a ground plane and objects used to test the gripper, save the testing setup in the layers, while keeping the original gripper asset free of any extraneous items used for testing.

1. Create a new stage without the reference added during import.
2. Save this stage with the name `Robotiq_2F_85_config.usd` at the same folder as the imported assets (you can locate the source file in the Reference or Payload section on the Property panel, and click the “Locate file” icon).
3. Open the layer tab and drag the `Robotiq_2F_85_edit.usd` in the **Root Layer**.

There is also a file named `Robotiq_2F_85_base.usd` in the source folder. This is the clean stage post import from Onshape and must not be directly edited to facilitate updates when the asset is re-imported from Onshape.

The *Authoring layer* is where changes are saved. To switch between layers, double click on the choice.

If changes are made in the wrong authoring layer you can drag the prims with the delta between layers to merge them into the receiving layer. Use this to your benefit by first authoring everything in the Root layer. After you are satisfied, you can drag your updates to the `Robotiq_2F_85_edit.usd` layer.

This is how the joints were named for this asset:

Note

Remember to combine parts that make rigid bodies on Group Mates before importing, to simplify the rigid bodies on stage (also useful for renaming the fingers to `left_finger_...` and `right_finger_...`).

## Adjusting Joints Post Import

Sometimes a limitation with the Onshape Client API causes the joints to become flipped 180 degrees from the drawing. To fix that, select the joints that are flipped, and apply an equal 180 degrees offset in Rotation 0 and Rotation 1 X axis. With the asset you imported, this was the case on the four joints.

The joints `[left, right]_outer_finger_joint` require limits [0,180] and `[finger_joint, right_outer_knuckle_joint]` require limits [0, 75]. Leave all other joints unconstrained.

Add fingertip physics material to increase the friction contact:

1. Open the Menu **Create** > **Physics** > **Physics Material**.
2. Select **Rigid Body Material**.
3. Rename the material to `fingertip_material`.
4. Set both friction coefficients to 0.8 (default rubber) and friction **Combine Mode** `Max`.
5. Select `right_inner_finger` and `left_inner_finger`. Scroll down to **Physics**, in Physics materials on selected models pick the created material.

Note

you may need to de-select instanceable for the two xforms in `right/left_inner_finger`, and set the physics materials on the mesh `Defeatured_2F_85_PAD_OPEN_fingertipsstep` directly.

## Breaking the Articulation Loop

If you try to simulate this asset now, you’ll get two big warnings on the screen:

For more information see [Physics Simulation Fundamentals](Physics.md). Articulations must be kinematic trees, but there is no need to delete any joints. To eliminate those warnings you must choose one joint to exclude from the Articulation and have it be treated as a maximal coordinate joint. Because maximal coordinate joints are treated with a lower priority by the solver, it is the joint that accumulates the most error in simulation.

In terms of simulation efficiency, the best choice of joint to exclude from articulation is the one that minimizes the length of articulations. However, you must also consider utility. The best joint to remove is the one that interferes the least with the robot functionality. In an ideal scenario, the joint to exclude from articulation only serves as a spatial constraint. Identify a joint with no limits, no resistance, and no drive. If there are no joints that fit this criteria, transfer these attributes to the adjacent joints before removing it from articulation.

In the case of this gripper, the best option to remove from the articulation are the joints that connect the inner shafts to the gripper body - the `inner_knuckle_joint` - highlighted in orange in the image.

1. To remove the joint from the articulation, select the `left_inner_knuckle_joint` prim.
2. In the Joint section under physics, select **Exclude From Articulation**.
3. Repeat for the `right_inner_knuckle_joint`.

Note

The fully completed asset is located in the `Samples/Rigging/Gripper/Robotiq 2F-85_complete` folder.

## Preparing For Tests

Because the gripper is not connected to anything to move it and test its physical properties, add a structure to later help us test the stability of the gripper:

1. Create two Xforms and add the Rigid Body API to them.
2. Add a fixed joint from world to the first Xform.
3. Add a Prismatic Joint from the first Xform to the second Xform.
4. Add a second prismatic joint from the second Xform to base\_link.
5. Add a Joint drive to the prismatic joints so that you can lift and move forward with a position command.
6. In the drives set the following:

   > * In the Advanced properties for the joint, set a maximum joint velocity of 5.0.
   > * Set the joint limits to [0, 1].
   > * In the joint drive, set the following:
   >
   >   > + Damping: 10000.0
   >   > + Stiffness: 10000.0

Make sure to move all joints that were just created outside of the Robotiq\_2f\_85 prim.

To assist in checking the grip:

1. Create a Cylinder and scale it to `[0.05, 0.05, 0.2]`.
2. Place the cylinder at `X=0.12`.
3. Set the cylinder collider to `Convex Hull`.
4. Create a ground plane and move it to `Z=-0.1`.

To assist in creating these prims, use the following script. You can run them by opening a Script Editor (**Window > Script Editor**) and pasting the code below.

```python
import omni.usd
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()

# Create Xform nodes
xform = UsdGeom.Xform.Define(stage, "/World/Xform")
xform_1 = UsdGeom.Xform.Define(stage, "/World/Xform_1")

# Add Physics Rigid Body API to Xform nodes
for node in [xform, xform_1]:
    UsdPhysics.RigidBodyAPI.Apply(node.GetPrim())

# Create Fixed Joint from Xform to Xform_1
fixed_joint = UsdPhysics.FixedJoint.Define(stage, xform.GetPath().AppendChild("fixed_joint"))
fixed_joint.CreateBody1Rel().SetTargets([str(xform.GetPath())])

# Create Prismatic Joints
prismatic_joint_1 = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joint_Z")
prismatic_joint_1.CreateAxisAttr("Z")
prismatic_joint_1.CreateLowerLimitAttr(0.0)
prismatic_joint_1.CreateUpperLimitAttr(1.0)
prismatic_joint_1.CreateBody0Rel().SetTargets([str(xform.GetPath())])
prismatic_joint_1.CreateBody1Rel().SetTargets([str(xform_1.GetPath())])

prismatic_joint_2 = UsdPhysics.PrismaticJoint.Define(stage, "/World/Joint_X")
prismatic_joint_2.CreateAxisAttr("X")
prismatic_joint_2.CreateLowerLimitAttr(0.0)
prismatic_joint_2.CreateUpperLimitAttr(1.0)
prismatic_joint_2.CreateBody0Rel().SetTargets([str(xform_1.GetPath())])
prismatic_joint_2.CreateBody1Rel().SetTargets(
    ["/World/Robotiq_2F_85/base_link"]
)  # update this to match your robot's base_link prim path

# Add Prismatic Joint Drive with damping and stiffness
for joint in [prismatic_joint_1, prismatic_joint_2]:
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateDampingAttr(10000)
    drive.CreateStiffnessAttr(10000)
    px_joint = PhysxSchema.PhysxJointAPI.Get(stage, str(joint.GetPath()))
    px_joint.CreateMaxJointVelocityAttr().Set(5.0)

# Add Ground Plane
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 100, Gf.Vec3f(0, 0, -0.1), Gf.Vec3f(1.0))

# Create cylinder mesh
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cylinder")
# Get the prim
cylinder_prim = stage.GetPrimAtPath(path)
cylinder_prim.GetAttribute("xformOp:scale").Set(
    (0.05, 0.05, 0.2)
)  # if your gripper is oriented differently, you may need to update the position and orientation of this cylinder or gripper accordingly to align them.  You can also do this post-creation.
cylinder_prim.GetAttribute("xformOp:translate").Set((0.12, 0, 0))

# Add Rigid Body and Mass API to cylinder
cylinder_body = UsdPhysics.RigidBodyAPI.Apply(cylinder_prim)
UsdPhysics.CollisionAPI.Apply(cylinder_prim)
massAPI = UsdPhysics.MassAPI.Apply(cylinder_body.GetPrim())
massAPI.CreateMassAttr(0.20)

# Create a Physics Scene
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
# This is a Small test scene, no need for GPU Dynamics
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
```

1. Set the target position for Joint X to 1 in the property panel, by going to the Joint Drive section and setting the target position to 1.
2. Set the target position for Joint Z to 1 in the property panel, by going to the Joint Drive section and setting the target position to 1.
3. Verify that you see the fingers ragdoll on the screen. It’s still necessary to Tune the Joint Drives for the fingers.

You can see in the video below that the gripper will move forward and lift up.

Until this point, if you start simulation you will see the fingers rotate freely, and also you will notice collision clipping between the fingers. This is because the fingers do not have drivers that tell them how to move, and because the finger components are connected with joints, there is a natural collision filter between them. This is normal and expected, and you fix it in the next sections.

## Adding Joint Drives

Add the Joint Drive API to all joints:

1. Select all joints on the gripper, then, in the Properties panel, **Add** > **Physics** > **Angular Drive** ( or **Linear Drive** for prismatic joints).

   > * In this gripper, the joints that drive the fingers are `finger_joint` and `right_outer_knuckle_joint`.
   > * Additionally, you have to flip the direction of `finger_joint` and `right_outer_knuckle_joint`, by setting lower limit to -75, and upper limit to 0
2. Select all the joints on the gripper, then, in the Properties panel, **Add** > **Physics** > **Joint State** ( or **Joint State Linear** for prismatic joints).
3. Model this gripper as a force-driven grasp. For that, position control must be disabled. Select `finger_joint` and `right_outer_knuckle_joint`, then set **Stiffness** to 10. The **Damping** is set to 0.1.
4. To control how much pressure is applied when the grippers close, set the `Max Force` to 16.5 (N).

   > * These grippers also have a limit speed at which they can operate. Converting from the data sheet to angular speed at the fingertips, the angular limit speed is 130 degrees per second.
5. In the joint section, under the **Advanced** tab, set the **Maximum Joint Velocity** to 130.0 (deg/s).

Summarizing the changes:

> * Maximum Joint Velocity: 130
> * Max Force: 16.5 (N)
> * Damping: 0.1
> * Stiffness: 10

When trying to control the fingers now, notice that they instantly bulge inwards instead of moving parallel. The system still needs stability to maintain the parallel motion when closing without resistance.

The Robotiq hand has a spring mechanism at the outer knuckle to keep the fingers parallel until an object is grasped.

1. Set the stiffness of `[left, right]_inner_finger_joint` to 0.0002, damping to 0.00001 and max force to 0.5 (N) to achieve this behavior.

## Adding Mimic Joint

This gripper is controlled with a single input command that moves both fingers concurrently. This is achieved by combining the drive joints together with a Mimic Joint specification.

1. Select `right_outer_knuckle_joint`.
2. Remove or set all values to zero in the joint drive we just added.
3. On the Properties Panel, click on **Add** > **Physics** > **Mimic Joint**.

   > Note
   >
   > Because this is a single degree of freedom revolute joint, the schema axis is not relevant. The UI will show rotX as the default axis, despite the joint being defined in the Z axis.
4. In the Mimic settings, set gearing to -1.0 to make it act in the opposite direction of the reference joint.
5. Set the reference Joint to `finger_joint`.

   > All drive features are copied over from the reference joint and having an authored joint drive would negatively impact the drive outcome.
   >
   > Note
   >
   > The Rotation Axis for the mimic joint only makes a difference, if the joint where mimic is applied contains multiple Degrees of Freedom (for Example Spherical Joint). For Prismatic and Revolute joints any selection will work just the same. It is still recommended to maintain it aligned with the DOF axis.
6. Run the simulation again.

## Collision Meshes

The default setting for collision meshes at import is Convex Hull. This is a good balance between performance and accuracy. However, for grippers, you often want the fingertips to have a collision mesh that closely follows the contour of fingertips’ geometry, so that there won’t be any gaps between the fingertips and the objects being grasped.

To visualize the collision meshes:

1. Find the eye icon on top of the viewport, and click **Show By Type** > **Physics** > **Colliders** > **All**.
2. Verify that outlines show up surrounding any objects that have collision meshes.
3. Optionally, to change any collision meshes, select the part of the object associated with that mesh by clicking on it in the viewport, and then in the Physics section of the Property panel, change the Collider Approximation type to Convex Decomposition, or any other type that’s appropriate for your use case.
4. If you don’t see a Physics or Collider section, then you might need to go down or up the stage tree from the selected item.
5. The collision API can be applied to a nested child Xform, or the parent of the selected object.

### Self-Collision

During your tests you may notice that the fingers are not colliding against each other. This is the default behavior when importing from Onshape. To disable that:

1. Select `/World/Robotiq_2F85`.
2. Check **Self-Collision Enabled** in the Articulation Root Options.

Note

For more details on how to tune the articulation, refer to [Joint Parameter Tuning Example: 2F-85](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/gripper_tuning_example.html).

## Saving Results

After you are satisfied with the configuration, push the changes to the original asset:

1. Open the **Layer** tab.
2. Select the `Robotiq_2F_85` prim, and all children prims in it.
3. Drag the selection into `Robotiq_2F_85_edit.usd`.
4. Click the **Save Layer** button on both Layers.

Note

The fully completed asset is located in the `Samples/Rigging/Gripper/Robotiq 2F-85_complete` folder.

## Test the Gripper

Now we can test the gripping by lifting the gripper and moving it forward, while closing the gripper to grasp the cylinder.

1. Set the target position for
   :   * Joint X to 0.1 in the property panel, by going to the Joint Drive section and setting the target position to 0.1.
       * Joint Z to 0.1 in the property panel, by going to the Joint Drive section and setting the target position to 0.1.
       * Finger joints to -40 degrees in the property panel, by going to the Joint Drive section and setting the target position to -40.

You can see in the video below that the gripper will move forward and lift up.

## Control the Gripper with Omnigraph

We can also use an Omnigraph to control the gripper, by writting the target position of the finger joints directly in the graph.

We have already prepared the graph in the `Samples/Rigging/Gripper/Robotiq 2F-85/Robotiq_2F_85_complete/Robotiq_2F_75_controller.usd` file, insert it as a layer to your Robotiq\_2F\_85\_config.usd layer.

1. Open the **Layer** tab.
2. Select the Insert Sub-Layer layer.
3. Find the `Robotiq_2F_75_controller.usd` file in the `Samples/Rigging/Gripper/Robotiq 2F-85/Robotiq_2F_85_complete` folder, and click `Open`.

Explaining the graph:

In this graph, the read upper and lower limit of the finger joints are used to calculate the range of motion of the gripper to map the input signal to the joint target position in degrees. The target position is set to the prim using `Write Prim Attribute` (Write Target) node.

Variables:

> * `input_signal`: A input signal (float) where 1 means open the gripper and 0 means close the gripper.

Nodes:
:   * `Read Upper Limit` / `Read Lower Limit`: A node that reads the upper and lower limit of the finger joint joint.
    * `Isaac Read Simulation Time`: A node that reads the simulation time, with reset on stop enabled.
    * `On Playback Tick`: A node that ticks the graph on every frame.
    * `Write Prim Attribute`: A node that writes the target position to the finger joint prim.

Set the input signal to 0.5 and press the **Play** button to start the simulation. You should see the gripper move forward and lift up.

Note

The fully completed asset is located in the `Samples/Rigging/Gripper/Robotiq 2F-85_complete` folder.

## Summary

In this tutorial, you experienced a comprehensive workflow for importing assets from a rigged Onshape document, performed post-processing adjustments to enable correct simulation hierarchy, and configured effort drives with Mimic Joints. You conducted validation and troubleshooting to address simulation behavior issues, optimizing performance. Additionally, you utilized layered editing to prepare a ready-to-use asset while retaining a test environment for validating gripper functionality.

---

# Tutorial 11: Tuning Joint Drive Gains

## Learning Objectives

In this tutorial, you learn how to use the Gain Tuner to tune joints on a robot so that it behaves as expected. For a more detailed explanation of how the Gain Tuner works and the physics behind it, see [Gain Tuner Extension](Robot_Setup.md).

### Prerequisite

* If the robot is in URDF format, follow [Tutorial: Import URDF](Importers_and_Exporters.md) to import a URDF file into Isaac Sim.
* The Gain Tuner extension is designed to be used on Robot assets, which are USD assets that contain the [Robot Schema](Omniverse_and_USD.md) applied.
* We also encourage you to setup your robot based on recommended [Asset Structure](Robot_Setup.md).
* This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.robot_setup.gain_tuner`.
* To access this Extension, go to the top menu bar and click **Tools** > **Robotics** > **Asset Editors** > **Gain Tuner**. The robots that are available for tuning will automatically populate under the Gain Tuner **Select Robot** Dropdown menu.

You can import any robot on the library and work on the joint drive parameters. For a more isolated test, you can also author a simple prismatic joint connected to a fixed base and model gains based on a rigid body with a given mass that moves along this prismatic joint. Remember you need to apply :[Robot Schema](Omniverse_and_USD.md) to the robot before the Gain Tuner can recognize the relevant joints and links.

## Gain Tuning

Tuning the joint drive gains is a process of finding the optimal values that balances the trade-off between stability and responsiveness. For example, low damping and stiffness may not be able to overcome the robot’s inertia, and the measured value will be offset from the target value, and too high of a stiffness may cause the robot to overshoot and oscillate around the target. Here we provide some tips in tuning position and velocity driven robots.

Note

The specific tuning process may vary based on the characteristics of the robot and its control system.

### Position Drive

For each joint of the robot:

1. Start by setting the damping to zero and only tuning the stiffness. This will help you establish a stable response without the influence of the derivative term.
2. Increase the stiffness until the joint is able to converge near the target position.
3. Reduce the stiffness by one order of magnitude.
4. After setting the stiffness, add damping with one order of magnitude lower than stiffness. This will be your baseline for the parameters and in general should not overshoot. If you want a faster response, reduce damping further.
5. Fine-tune both gains around this established baseline to achieve the desired performance, considering factors such as stability, response time, and overshoot.
6. If you want to emulate a control that includes gravity compensation, select all rigid bodies of the robot and check Disable Gravity in the properties panel.

#### Velocity Limit and Industrial Robots

Many robots, including the majority of Industrial robots, come with pre-tuned PD control for their joint drives and can be set up to have perfect position control response, always driving at the given joint velocity limit. To reproduce this behavior, we can increase the joint stiffness from the previous tuning heuristic by a factor of two and define the maximum joint velocity in the **Joint** > **Advanced** > **Maximum Joint velocity** in the **Properties** panel. Run the simulation to verify the joint velocity is meeting the specification and fine-tune the stiffness until the joint max velocity limit is within tolerance. If stiffness is too high, the max velocity may still be violated, so it is not advised to just add infinite stiffness to the joint, and instead operate with stiffness similar to the ones calibrated without a max joint velocity.

### Velocity Drive

For each joint of the robot:

1. Start by setting the **Stiffness** to zero and only tuning the damping.
2. Increase the damping until the joint is able to converge near the target velocity.
3. If the robot may carry additional load, slightly increase the damping (for example, add 10% extra) to account for the extra load.
4. You can limit the joint’s output by either setting the max joint velocity, or restrict the max joint force to impose a maximum joint load effort.

## Saving Gains to the Asset

Following the NVIDIA Isaac Sim [Asset Structure](Robot_Setup.md), Joint gains would be a physics configuration, and should ideally be saved on the physics configuration layer. To facilitate this, The `Save Gains to Physics Layer` button on the UI searches for the Asset’s physics layer where the joint is defined, and applies the updated gains to that layer. If you don’t want or don’t have permission to save on that file, you can just save the currently open stage instead to author an override to the joint target values locally.

## Visualize Results

The results of the tests are visualized in the form of a plot, where the tracked Joint Positions and Velocities are compared against the commanded trajectory. Select the desired joint to visualize the results on the left panel, and their respective test results will be displayed on the plots. The test results are color-coded by joint, with the measured values being a faded version of the commanded trajectory’s color.

Even if the joint is not listed on the Robot Schema, it will still be visualized in the plots, if it’s part of the physical robot.

To select more than one joint, users can hold down the control key and click on the desired joint, or select the first joint and then hold down the shift key and click on the last desired joint, and all joints between them will be selected.

Note

The visualization results are only available when the tests are finished running, so depending on the configuration of the tests, it may take some time to get the results.

## Tips

* A reasonable goal is to find a set of gains that is able to ramp to position but keep overshoot within 1% of the target.
* Disable Gravity if your robot has built-in gravity compensation or you have a separate gravity compensation controller.
* Group the joints that are expected to move together, and tune the gains for each group individually first, then combine them for the final test. For example, for a humanoid robot, you may want to separate the legs and arms because they are not expected to be moving at the same time with high tracking accuracy.
* Reduce the maximum speed of a joint that you are tuning if it is not expected to be commanded to move that fast in practice. Most of the default maximum velocities written inside the USD are likely impractically high.

## Further Learning

* Read [Gain Tuner Extension](Robot_Setup.md) for more details on the physical mechanics relating joint gains to derived motions, and how the Gain Tuner works.

---

# Tutorial 12: Asset Optimization

## Learning Objectives

This tutorial details how to make robot assets more performant and where to find tradeoffs to achieve a faster simulation or rendering time.

*30 Minutes Tutorial*

## Getting Started

**Prerequisites**

* Complete the [Quick Tutorials](Quick_Tutorials.md) series to learn the basic core concepts of how to navigate inside NVIDIA Isaac Sim.
* Complete the [Assemble a Simple Robots](Robot_Setup.md) tutorial to learn the concepts of rigid body API, collision API, joints, drives, and articulations.
* Read [Onshape importer](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html "(in Omniverse Extensions)") and watch the videos on rigging the robot in Onshape.
* Familiarity with [Mesh Merge Tool](Robot_Setup.md).

**Loading the Robot**

This tutorial explores the NVIDIA Jetbot Robot asset which improve performance.
If you import the asset from a different source, for example from custom CAD, you might end up with numerous meshes per rigid body and this can severely impact performance.

From the recording of this Jetbot asset imported from CAD that on the right side we have an unoptimized asset, and it’s achieving 40 FPS, while the asset on the left was optimized, and now achieves 64 FPS.

## Asset Structure Optimization

In this activity, you use a workflow with the multi-layered asset structure introduced in an earlier module,
and create an optimized version of an asset.
Use the Jetbot robot as a starting place. This model was imported from a CAD model made in Onshape.
Although the physics layer is already in place, the bodies contain a significant number of meshes, which leads to suboptimal simulation performance.
Begin with an empty stage to learn several useful tricks for asset authoring.
By the end of this activity, you transform the initial Jetbot model into a well-structured, optimized asset ready for efficient simulation.

### Set Up Reparenting and Layers

1. In Isaac Sim, go to **Edit** > **Preferences** to open the Preferences panel.
2. Under **Stage** > **Authoring**, ensure that **Inherit Parent Transform** is checked.

   > 
3. Open the Jetbot located at `Isaac Sim/Samples/Rigging/Jetbot/Jetbot_Optimized/Jetbot_optimized.usd`, verify that you have an empty USD.
4. Select the **Layers** panel, click the **Insert Sublayer** button at the bottom of the tab, select `Isaac Sim/Samples/Rigging/Jetbot/Jetbot_Base/Jetbot_base.usd`, and click **Open**.

   > 

### Create Asset Structure

The Jetbot asset is already close to the final goal, but to work on a retargeting of the structure to get the merged meshes,
create a new prim to be set as default.

1. On the right side menu of the Stage panel, select **Show Root**.

   > 
2. Create a new Xform called `Jetbot_Sim` and drag it onto Root.
3. Right click on `Jetbot_Sim` and choose **Set as Default Prim**.
4. Right click and choose **Create** > **Scope** and name it `Visuals`.
5. Drag this scope onto Root so it’s unparented from `Jetbot_Sim`.
6. Select the prims under `Jetbot` and drag them onto `Jetbot_Sim`.

   > 
   >
   > Note
   >
   > To select multiple prims, use shift-select or control-select standards. For example: select one prim, then hold shift and another to choose all prims listed between them.
7. Verify that instead of being deleted from Jetbot, they were instead deactivated.
8. Select them all, then right-click and choose **Activate**.

   > 
9. Delete the contents inside the prims in `Jetbot_Sim`.

   > 

### Merge Meshes

With the stage ready, you can begin merging the meshes.

1. Open the Mesh Merge Tool by going to **Tools** > **Robotics** > **Asset Editors** > **Mesh Merge Tool**.
2. Select `Jetbot/left_wheel` prim.
3. Check the **Combine Materials** box, insert `Jetbot_Sim/Looks` to save the material in the Jetbot Sim xform.

   > 
4. Click on **Merge**.
5. Select the resulting mesh on `/Merged/left_wheel` and clear the transform on the properties panel.

   > 
6. Right-click on the **Visuals** scope, create an xform called `left_wheel` and drag the resulting mesh into it. Remove the `/Merged` xform from the stage.

   > 
7. To create an internal reference to the wheel, create a **Visuals** Xform inside `left_wheel`, then right-click it and choose **Add** > **Reference**.

   > 
8. Select `Isaac Sim/Samples/Rigging/Jetbot/Jetbot_Base/Jetbot_base.usd` in the dialog.

   > 
9. For `prim_path`, type in `/Visuals/left_wheel`.

   > 
10. Back in the **Stage** panel, select the `/Visuals/left_wheel` prim, which you just added a reference onto. Then in the **Property** panel, scroll down to the **References** section. The prim path is in red, select the Asset Path entry and **clear** it.
11. This will make the reference point to the internal `/Visuals/left_wheel` prim. The mesh for `left_wheel` shows as a child. Verify that a **Looks** scope was created in `Jetbot_Sim`, with the materials for this mesh.

    > 
12. Verify that the wheel is referenced correctly in place, along with the base mesh that is at the origin. You can hide the Visuals scope so base meshes won’t be visible.

    > 
13. Save the file with CTRL+S.
14. To complete the mesh optimization, repeat the previous steps for other bodies.

Note

The finished USD with all mesh merges is available for you at `Isaac Sim/Samples/Rigging/Jetbot/Jetbot_Optimized/Jetbot_optimized_post_merge.usd`.

## Scenegraph Instancing

Scenegraph instancing enables sharable, composed representations of subgraphs of prims. It is a directive that instructs the scene composer that a certain component of the scene is a repeatable pattern. While this allows for a leaner overall scene, it does require a few rules to be followed.

Any children of an instance cannot have attributes modified, because they all inherit from the same asset in memory.
Instances must be applied on Referenced assets, so that the scenegraph composer knows that from the reference and downwards, things are expected to remain the same and it needs to create a pointer to the asset data to be used anywhere it’s referenced.

1. Start by opening the USD file `Isaac Sim/Samples/Rigging/Jetbot/Jetbot_Optimized/Jetbot_optimized_post_merge.usd`, if you have not merged all the meshes.
2. The left and right wheel meshes are identical. Further simplify the asset by having left and right wheel reference the same mesh. Select `Visuals/left_wheel` and rename it to `Visuals/wheel`.
3. Delete `Visuals/right_wheel`. Verify that the Jetbot wheel disappears.
4. Select `Jetbot_Sim/right_wheel/Visuals`.
5. Under the References section of the **Property** panel, replace the reference Prim Path from `/Visuals/right_wheel` to `/Visuals/wheel`.

   > 

   At this point, all meshes are still considered unique elements because the assets are only defined as a reference.
6. To leverage memory savings, shift-select all Visuals prims under `/Jetbot_Sim` and check **Instanceable** in the Property panel.
7. On the Visuals prims, notice the reference icon now has a blue “I” on it. This indicates they are instantiable meshes and effectively applying any memory savings.

   > 
8. Save the file with CTRL+S.

Note

The finished USD with all mesh merges and scenegraph instancing is available for you at `Isaac Sim/Samples/Rigging/Jetbot/Jetbot_Optimized/Jetbot_optimized_final.usd`.

## Other Considerations

* **Minimize Number of Lights**: Each light negatively impacts the performance of the rendering. By default, if the scene has more than 10 lights, the rendering reverts to sample-based lighting to avoid severe slowdown in performance.
* **Reduce Translucent Materials**: Each translucent material generates a larger performance bottleneck than the default OmniPBR material.
* **Optimize Physics Performance**: Search for simulation aspects that you can modify to reduce computational cost. Typically, colliders have high computational costs. The more basic that you can make a collision shape, the more performant the simulation behaves. Reducing the number of contact points can also bring huge performance benefits. Tuning this can take several experiments to achieve the best precision versus performance point for your situation.
* **Approximate Wheel Colliders**: If you have a wheel collider, consider using a simple cylinder or sphere collider instead of a mesh collider. This can significantly improve performance and allows the robot to drive smoothly over terrains.

---

# Tutorial 13: Rigging a Legged Robot for Locomotion Policy

The objective of this tutorial is to explain the process of rigging a legged robot to match the configuration specified by the locomotion policy.
The Isaac Sim [Policy Controller Class](Isaac_Lab.md) for inference in Isaac Sim is already handling the process of rigging the robot at run time,
so this tutorial is only relevant if you want to run the robot policy with an external process like ROS.

## Learning Objectives

In this tutorial, you will walk through the process of rigging a H1 humanoid robot to match the configuration specified by the H1 flat terrain locomotion policy.

1. Setting initial robot position
2. Setting joint configuration
3. Verifying joint configuration

Note

The H1 flat terrain policy environment definition file is available [here](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/H1_Policies/h1_env.yaml).

## Setting Initial Robot Position

The initial joint position of the robot is specified under robot:init\_state:joint\_pos section of the environment definition file. The joint names are specified using the .\* wildcard.

```python
 1robot:
 2  init_state:
 3    joint_pos:
 4      .*_hip_yaw: 0.0
 5      .*_hip_roll: 0.0
 6      .*_hip_pitch: -0.28
 7      .*_knee: 0.79
 8      .*_ankle: -0.52
 9      torso: 0.0
10      .*_shoulder_pitch: 0.28
11      .*_shoulder_roll: 0.0
12      .*_shoulder_yaw: 0.0
13      .*_elbow: 0.52
14    joint_vel:
15      .*: 0.0
```

Note

The joint positions are specified in radians, where as in USD, the joint positions are specified in degrees.

To store the initial state of the robot:

1. Open the `h1.usd` file from the content browser present at `Isaac Sim/Robots/Unitree/H1`.
2. Create a joint state api for reporting the robot joint position and velocity.
3. On the top right corner of the stage, select the `funnel` icon and click `Physics Joints` to filter the joint list.

> 

1. Left click on the first joint (`left_hip_yaw`), shift left click on the last joint (`right_elbow`) to select all the joints.
2. Right click on any selected joint, and click **Add** > **Physics** > **Joint State Angular** to create a joint state API attribute to the joints
3. Right click on any selected joint, and click **Add** > **Physics** > **Angular drive** to create a joint drive API attribute to the joints

> Note
>
> The `Joint State Angular` API is used to report the joint position and velocity, and the `Angular drive` API is used to drive the joint. If the joint already has a `Joint State Angular` API or `Angular drive` API, you can skip the above steps.

1. Go to each joint and set the `Target Position` attribute in the joint drive API to the value specified in the environment definition file above on the `joint_pos` attribute.
2. Similarly, set the `Target Velocity` attribute in the joint drive API to the value specified in the environment definition file above on the `joint_vel` attribute.
3. Make sure you convert the joint positions and velocities from radians to degrees.
4. Left click on the joint you are changing.
5. In the property panel below, scroll down to the `Target Position` attribute.
6. Set the `Target Position` attribute to the value specified in the environment definition file above on the `joint_pos` attribute
7. Repeat the same for the `Target Velocity` attribute
8. Press play.
9. Verify that you see the robot moving to the initial position specified in the environment definition file. To make the robot start in the initial position when the simulation starts, store the data in the joint state API.
10. To prevent the robot from falling infinitely, you can add a Fixed Joint between the robot and the world by right clicking on the `/h1/torso_link` and click **Create** > **Physics** > **Joint** > **Fixed Joint**.

To prevent the joint state API values from resetting, you need to change the simulation setting to not reset the robot state on stop.

1. On the top left corner of the stage, click on the **Edit** and click **Preferences**.
2. Select the **Preferences** window at the bottom, on the left side, click on the **Physics** tab.
3. Uncheck **Reset Simulation on Stop**.

> 
>
> Now you can play the simulation, and when you stop the simulation, the robot will remain in the last state. When you play the simulation again, the robot will start from the last state.

1. Delete the Fixed Joint between the robot and the world.
2. Press **Ctrl+S** to save the USD file.
3. Check **Reset Simulation on Stop** again.

## Setting Joint Configuration

Set the joint configuration to match the policy’s robot configuration, this maybe different from the value stored in the USD file.
The joint drive configuration is specified under `scene:robot:actuators` section of the environment definition file.

The snippet below shows the actuator configuration for the H1 robot legs.

```python
 1actuators:k
 2  legs:
 3    class_type: omni.isaac.lab.actuators.actuator_pd:ImplicitActuator
 4    joint_names_expr:
 5    - .*_hip_yaw
 6    - .*_hip_roll
 7    - .*_hip_pitch
 8    - .*_knee
 9    - torso
10    effort_limit: 300
11    velocity_limit: 100.0
12    stiffness:
13      .*_hip_yaw: 150.0
14      .*_hip_roll: 150.0
15      .*_hip_pitch: 200.0
16      .*_knee: 200.0
17      torso: 200.0
18    damping:
19      .*_hip_yaw: 5.0
20      .*_hip_roll: 5.0
21      .*_hip_pitch: 5.0
22      .*_knee: 5.0
23      torso: 5.0
24    armature: null
25    friction: null
```

The `joint_names_expr` is a list of joint names to be controlled by the actuator. The `class_type` is the type of the actuator to be used.
The `effort_limit` is the maximum effort that can be applied to the joint. The `velocity_limit` is the maximum velocity that can be applied to the joint.
The `stiffness` is the stiffness of the joint. The `damping` is the damping of the joint. The `armature` is the armature of the joint. The `friction` is the friction of the joint.

To set the joint configurations:

1. Left click on a joint such as `left_hip_yaw` for example.
2. In the property panel, scroll down to `joint drive` attribute, and set the `stiffness`, `damping` to the values specified in the environment definition file.

Note

Remember to convert stiffness and damping to degrees.

The USD file stiffness is in \(\frac{Kg \cdot m^2}{Deg \cdot s^2}\) and the damping is in \(\frac{Kg \cdot m^2}{Deg \cdot s}\).
To convert them to radians, you can use the following formulas:

\[S\_{deg} = S\_{rad} \times \frac{\pi}{180}\]

\[D\_{deg} = D\_{rad} \times \frac{\pi}{180}\]

The `effort_limit` is the maximum effort that can be applied to the joint, set that value to the `Max Force` attribute of the joint drive API.

Scroll down to **Raw USD Properties** under the **Advanced** tab, set the **Armature**, **Joint Friction** attribute to the value specified in the environment definition file.

For the **Maximum Joint Velocity** attribute, set it to the **velocity\_limit** value specified in the environment definition file, remember to convert it to degrees.

\[\omega\_{deg} = \omega\_{rad} \times \frac{180}{\pi}\]

Note

Remember to set the joint configurations for all active joints in the robot. For example, arms and legs.

## Verify Joint Configuration

To verify the joint configuration, you can play the simulation and run the following snippet in script editor to print the joint configuration.

1. Play the simulation.
2. Open the script editor by clicking on **Window** > **Script Editor**.
3. Copy and paste the following snippet into the script editor.
4. Run the snippet by clicking on the **Run** button.

> ```python
> from isaacsim.core.prims import SingleArticulation
>
> prim_path = "/h1"
> prim = SingleArticulation(prim_path=prim_path, name="h1")
> print(prim.dof_names)
> print(prim.dof_properties)
> ```

1. Verify that you see the console output like the following:

```python
['left_hip_yaw', 'right_hip_yaw', 'torso', 'left_hip_roll', 'right_hip_roll', 'left_shoulder_pitch', 'right_shoulder_pitch', 'left_hip_pitch', 'right_hip_pitch', 'left_shoulder_roll', 'right_shoulder_roll', 'left_knee', 'right_knee', 'left_shoulder_yaw', 'right_shoulder_yaw', 'left_ankle', 'right_ankle', 'left_elbow', 'right_elbow']
[(0,  True, -0.42999998, 0.42999998, 1, 100.00003815, 300., 149.54197693,  5.00000191)
(0,  True, -0.42999998, 0.42999998, 1, 100.00003815, 300., 149.54197693,  5.00000191)
(0,  True, -2.34999967, 2.34999967, 1, 100.00003815, 300., 200.00009155,  4.98473263)
(0,  True, -0.42999998, 0.42999998, 1, 100.00003815, 300., 149.54197693,  5.00000191)
(0,  True, -0.42999998, 0.42999998, 1, 100.00003815, 300., 149.54197693,  5.00000191)
(0,  True, -2.86999965, 2.86999965, 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -2.86999965, 2.86999965, 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -3.13999987, 2.52999973, 1, 100.00003815, 300., 199.96228027,  4.99619198)
(0,  True, -3.13999987, 2.52999973, 1, 100.00003815, 300., 199.96228027,  4.99619198)
(0,  True, -0.33999997, 3.1099999 , 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -3.1099999 , 0.33999997, 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -0.25999996, 2.04999971, 1, 100.00003815, 300., 200.00009155,  4.98473263)
(0,  True, -0.25999996, 2.04999971, 1, 100.00003815, 300., 200.00009155,  4.98473263)
(0,  True, -1.29999983, 4.44999933, 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -4.44999933, 1.29999983, 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -0.86999995, 0.51999992, 1, 100.00003815, 100.,  19.99622726,  4.00000191)
(0,  True, -0.86999995, 0.51999992, 1, 100.00003815, 100.,  19.99622726,  4.00000191)
(0,  True, -1.24999988, 2.6099999 , 1, 100.00003815, 300.,  40.00001526, 10.00000381)
(0,  True, -1.24999988, 2.6099999 , 1, 100.00003815, 300.,  40.00001526, 10.00000381)]
```

The values in the console output are already in radians. Each row is for a joint listed in the same order as the first list.
Verify the last four values in each row, which are the `maxVelocity`, `maxEffort`, `stiffness`, `damping` respectively. Verify that the values match the values specified in the environment definition file.

For example, for the `left_hip_yaw`, the max velocity is `100.0`, the max effort is `300.0`, the stiffness is `150.0`, and the damping is `5.0`.

Note

The rigged H1 robot is available in the content browser at `Isaac/Samples/Rigging/H1/h1_rigged.usd`.

## Summary

This tutorial covered the following topics:

* Setting initial robot position
* Setting joint configuration
* Verifying joint configuration

---

# Robot Setup Troubleshooting

This page consolidates troubleshooting information for robot setup and simulation in Isaac Sim.

## Reparenting Assets

You can change how reparenting behaves under **Edit > Preferences**, and on the **Stage Panel**, scroll down to authoring. The checkbox **Keep Prim world Transform when reparenting**, lets you decide when reparenting if the objects remain in place or if they get moved to the parent’s frame of reference. You can use this to your advantage to apply offsets or change the parent’s origin without impacting the children elements.

## Robot Rigging Issues

If your robot “explodes” during simulation or after some movements, check if any of the collision meshes are colliding with each other.

Common rigging issues and their solutions:

1. Colliding collision geometries - Ensure that collision geometries do not intersect or overlap, especially at joint pivot points
2. Joint limit violations - Verify that joint limits are set appropriately and not being exceeded during simulation
3. Incorrect joint ordering - Make sure that joint orderings in articulation chains are correct
4. Physics instabilities - Adjust physics timestep or solver iteration counts if experiencing vibrations or instabilities

Physics Inspector “failed to find internal joint” errors for robots with mimic joints does not affect the functionality of the mimic joints and can be ignored:

```python
[Error] [omni.physx.plugin] Usd Physics: failed to find internal joint object for PhysxMimicJointAPI at /Franka/panda_hand/panda_finger_joint2. Please ensure that the prim is a supported joint type and is part of an articulation.
```

## Robot Controller Issues

1. Gains produced by the gain turner may not perfectly track the robot’s commanded movements (for example, as seen in the Cobotta Pro robot). Manual tuning of gains may be necessary for optimal performance.
2. Some grippers with parallel mechanism (that is, Robotiq 2F-85 and 2F-C2) have links that do not move with rest of the gripper. This is a known issue and may require manual adjustment of the gripper joints.
3. When working with differential drive robots, make sure that wheel friction is appropriate. Too little friction can result in wheel slippage, while too much friction can cause erratic movement.

## Robot Import Issues

USD to URDF Exporter issues:

* The Collider meshes may be improperly included in the visuals. They can be manually removed from the URDF file.
* The Body and Joints are authored in the URDF file in alphabetical order. They can be manually reordered in the URDF file.
* Depending on the robot structure, some body names may be overridden due to the merging of different frames. Review the output and verify that it’s accurate.
* The URDF exporter adds joint effort and velocity limits as inf when unbounded. This may make the URDF not import correctly if the URDF parser does not support inf values in Float.

When importing a URDF:

1. If more than one asset in URDF contains the same material name, only one material is created regardless if the parameters in the material are different. For example, if two meshes have materials with the name “material”, one is blue and the other is red, both meshes will be either red or blue. This also applies for textured materials.
2. MJCF importer does not show the built-in bookmark in the file picker dialog. The bookmark is still available in the content pane and can be copy-pasted into the file picker dialog.

## Closed Loop Structure Issues

For robots with closed-loop kinematic chains:

1. Make sure that the constraints are properly defined and initialized
2. Check that all joints in the closed loop have appropriate drive settings
3. Consider simulating the closed loop as separate articulations with constraints rather than with a single complex closed-loop structure
4. Adjust solver settings for better convergence if experiencing stability issues

## Robot Importing tips

1. Sometimes the robot may have non-zero target positions. When the target position does not match the initial position, the robot will move to the target position on the first frame. To prevent this, either set the target position to zero or set the initial position to the target position.
2. Max forces may be high or low in the URDF, set them to a more reasonable value in the USD.
3. If the stiffness and damping values are too high, the robot may oscillate. If it’s too low, the robot may not move to the desired position. Use the gain tuner to test the stiffness and damping.
4. If the robot have overlapping collision meshes, use a filtered pair to ignore collisions between specific meshes.

## Common Issues

| Observation | Solution |
| --- | --- |
| Robot meshes are penetrating each other after importing | Verify the source file (MJCF or URDF) have the correct transforms for the meshes. Adjust the transforms in the source file or in the USD after importing. |
| Robot joints are not moving at all | Check the joint limits and ensure they are set correctly. Adjust the limits in the source file or in the USD after importing. Verify that the joint gains are non zero. If you have mimic joints, make sure the gear ratio and direction are set correctly. One suggestion is to disable all the joints first, and then add them back one by one to isolate the issue. |
| Robot joints are moving in the wrong direction | Check the joint axis and ensure they are set correctly. Adjust the joint axis in the source file or in the USD after importing. For mimic joints, verify that the direction is set correctly. |
| Robot shakes uncontrollably starting from the first frame | Usually, conflicting collisions can generate adnormal amount of force which cause the robot to behave incorrectly. Check for self overlapping collision geometries. Uncheck self collision enabled in Articulation Root if self collision is not needed. If self collision is required, apply contact filter to specific pairs of colliders that should not collide. |
| Robot shakes uncontrollably after some movements | This usually happens when the robot gains are too high and generating adnormal amount of torque. Try increasing the physics substeps and solver iteration counts in the Physics Settings window. You can also try reducing the robot’s maximum velocity and force limits to prevent extreme movements. |
| Robot experiences physX transform errors | This usually happens when the robot is under extreme forces or torques similar to the previous scenario and it can be induced by conflicting joint transformations. First disable all the joints and see if the issue persists. If the issue is resolved, re-enable the joints one by one to isolate the problematic joints. Check for conflicting joint limits or positions. |
| The robot is penetrating the ground or other objects on the first frame | Check the initial position of the robot and ensure it is above the ground plane and not intersecting with any meshes. Verify that the collision geometries are correctly defined and not intersecting with other objects at the start of the simulation. |
| The robot is penetrating the ground or other objects during simulation | Adjust the physics timestep and solver iteration counts to improve stability, modify the contact offset of the colliders to ensure proper collision detection, and verify that the robot’s mass and inertia properties are realistic. |
| The simulation performacne is slow at run time | Reduce the number of collision meshes and simplify their geometry by using simliar colliders, and adjust the physics timestep and solver settings for better performance. |
| The robot joints are not following the commanded positions accurately | Tune the joint gains using the [Gain Tuner Extension](Robot_Setup.md), ensure that the maximum velocity and force limits are set appropriately, and verify that there are no conflicting forces acting on the robot. |

---

# Asset Validation

Isaac Sim comes with the [isaacsim.asset.validation](#isaac-sim-app-reference-asset-validation) extension that provides a set of validation rules to ensure that USD assets are properly configured for use in Isaac Sim.

While some of the rules are related to recommended guidelines, such as the [Asset Structure](Robot_Setup.md), many are fundamental for the asset to work properly in Isaac Sim.

This document provides a comprehensive overview of all validation rules available in the Isaac Sim Asset Validation extension. The rules are organized by their registration categories and help ensure that USD assets are properly configured for use in Isaac Sim.

The Isaac Sim asset validation comes enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.asset.validation`.

To open the **Asset Validation** window, navigate to **Window > Asset Validator**. For more information on the Asset Validation window, refer to [Asset Validator](https://docs.omniverse.nvidia.com/kit/docs/asset-validator/latest/index.html).

There are many validation rules available in the Asset validation window. You can choose to run all validation rules, but specifically for Isaac Sim, there are three categories of rules to review.

The Isaac Simvalidation rules are grouped into the following categories:

* IsaacSim.PhysicsRules
  :   + Fundamental rules related to physics simulation
* IsaacSim.RobotRules
  :   + Rules related to robot assets
* IsaacSim.SimReadyAssetRules
  :   + Rules related to sim ready assets

This document will go through each of the rules and provide a detailed explanation of what it checks.

# IsaacSim.PhysicsRules

Physics Validation Rules

| Rule Name | Description and Checks |
| --- | --- |
| **PhysicsJointHasDriveOrMimicAPI** | Validates that joints have a drive or mimic API.   * Non-fixed joints must have either a drive API or mimic API * Joints excluded from articulation are exempt from this requirement * When both drive and mimic APIs are present, drive stiffness and damping must be 0.0 |
| **PhysicsJointMaxVelocity** | Validates that joints have a positive max velocity set.   * Max joint velocity attribute is defined on joints with PhysxJointAPI * Max joint velocity value is greater than zero |
| **PhysicsDriveAndJointState** | Validates that joint drives have proper force limits and matching state values.   * Drive max force is defined and positive (not zero or infinite) * Drive target positions match joint state positions within tolerance (1e-2) * Drive target velocities match joint state velocities within tolerance (1e-2) |
| **DriveJointValueReasonable** | Validates that joint drive stiffness values are within reasonable ranges.   * Drive stiffness is within range (0.0 to 1,000,000.0) * Mimic joints have stiffness and damping set to 0.0 * Non-mimic joints have stiffness values defined * Maximum natural frequency warning threshold: 500.0 Hz |
| **JointHasCorrectTransformAndState** | Validates that joint transforms and states are consistent with the connected bodies.   * Joint position consistency between connected bodies * Joint orientation consistency between connected bodies * Joint state values match the robot pose configuration * Applies to revolute and prismatic joints |
| **JointHasJointStateAPI** | Validates that joints have the JointStateAPI applied.   * Prismatic joints have JointStateAPI with “linear” type * Revolute joints have JointStateAPI with “angular” type * Provides automatic fix suggestion to apply missing APIs |
| **MimicAPICheck** | Validates proper configuration of mimic joint APIs.   * Reference joint relationship has exactly one target * Gear ratio, natural frequency, and damping ratio are defined and non-zero * Joint limits are properly configured relative to reference joint limits * Limit compatibility based on gear ratio sign (positive/negative) |
| **RigidBodyHasMassAPI** | Validates that rigid bodies have properly configured mass properties.   * Rigid bodies have MassAPI applied * Mass attribute is authored and non-zero * Diagonal inertia is authored and non-zero * Principal axes are authored and normalized |
| **RigidBodyHasCollider** | Validates that enabled rigid bodies have collision geometry.   * Enabled rigid bodies have collision geometry in their hierarchy * Searches through prim range including instance proxies |
| **NonAdjacentCollisionMeshesDoNotClash** | Validates that non-adjacent collision meshes don’t intersect.   * Performs physics simulation to detect colliding pairs * Verifies that colliding bodies are connected by joints * Reports errors for non-adjacent colliding meshes |
| **InvisibleCollisionMeshHasPurposeGuide** | Validates that invisible collision meshes have purpose set to ‘guide’.   * Collision meshes with invisible visibility * Purpose attribute is set to ‘guide’ for invisible collision meshes |
| **HasArticulationRoot** | Validates that at least one prim in the stage has the ArticulationRootAPI.   * At least one prim in the stage has ArticulationRootAPI applied |

# IsaacSim.RobotRules

Robot Validation Rules

| Rule Name | Description and Checks |
| --- | --- |
| RobotNaming | Validates that robot assets follow the standard naming convention.   * Minimum folder nesting depth (at least 3 levels) * Folder name matches robot filename * Supports versioned folder structure: <Manufacturer>/<robot>/<robot.usd> or <Manufacturer>/<robot>/<version>/<robot.usd> |
| **CleanFolder** | Validates that robot asset folders don’t contain unexpected files.   * Robot asset folders only contain expected files * Warns about unexpected files in the asset directory |
| **NoOverrides** | Validates that prims don’t have overridden attributes.   * Prims don’t have overridden attributes (excluding /Render paths) * Detects attributes with authored values in layer stack * Only applies for the open stage |
| **RobotSchema** | Validates that robot assets have the required RobotAPI and relationships.   * Default prim is set on the stage * Default prim has RobotAPI applied * robotLinks relationship exists and has targets * robotJoints relationship exists and has targets |
| **JointsExist** | Validates that robot assets contain at least one joint.   * At least one prim in the stage has JointAPI applied |
| **LinksExist** | Validates that robot assets contain at least one link.   * At least one prim in the stage has LinkAPI applied |
| **ThumbnailExists** | Validates that robot assets have a thumbnail image.   * Thumbnail image exists at expected path: `<folder>/.thumbs/256x256/<filename>.png` |
| **CheckRobotRelationships** | Validates that robot relationships are properly defined and prepended.   * robotLinks and robotJoints relationships exist * Relationships are prepended for proper USD composition * Provides automatic fix suggestions for missing or non-prepended relationships |
| **VerifyRobotPhysicsAttributesSourceLayer** | Validates that physics attributes are authored in the physics layer.   * Physics attributes (starting with “physics:”) are authored in \_physics.usd layer * Warns when physics attributes are found in other layers |
| **VerifyRobotPhysicsSchemaSourceLayer** | Validates that physics schemas are applied in the physics layer.   * Physics schemas (starting with “Physx” or “Physics”) are applied in \_physics.usd layer * Warns when physics schemas are found in other layers |

# IsaacSim.SimReadyAssetRules

Sim Ready Asset Validation Rules

| Rule Name | Description and Checks |
| --- | --- |
| **NoNestedMaterials** | Validates that materials don’t contain nested materials.   * Material prims don’t contain child materials in their hierarchy * Warns about nested material configurations |
| **MaterialsOnTopLevelOnly** | Validates that materials are only defined in the top-level Looks prim.   * All materials are children of the top-level Looks prim * Materials are not scattered throughout the stage hierarchy * Skips materials in referenced/payload content |

# Running the Validation Rules

See video below for a demonstration of running the validation rules. We navigate to **Window > Asset Validator**, then select the **Isaac Sim** category rules. We can then select individual rules to run, but we chose to select all rules for each category.