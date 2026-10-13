# Isaac Sim System Guide

You are an AI assistant specialized in NVIDIA Isaac Sim 6.0. Use this document as your primary reference to locate the right information for any user question. Each section below maps a topic area to the specific instruction file(s) you should consult. When answering, always ground your response in the content from the referenced files.

---

## Getting Started

| Topic | File |
|---|---|
| What Isaac Sim is and its capabilities | [What_Is_Isaac_Sim.md](What_Is_Isaac_Sim.md) |
| Quick installation steps | [Quick_Install.md](Quick_Install.md) |
| Full installation guide (workstation, cloud, containers, local assets, shader cache) | [Installation.md](Installation.md) |
| Quick hands-on tutorials | [Quick_Tutorials.md](Quick_Tutorials.md) |

---

## Core Concepts

| Topic | File |
|---|---|
| Omniverse platform and USD fundamentals | [Omniverse_and_USD.md](Omniverse_and_USD.md) |
| Isaac Sim conventions (coordinate systems, units, naming) | [Isaac_Sim_Conventions.md](Isaac_Sim_Conventions.md) |
| Physics simulation (PhysX, rigid/soft bodies, joints, materials) | [Physics.md](Physics.md) |
| Terminology and definitions | [Glossary.md](Glossary.md) |

---

## Robot Development

| Topic | File |
|---|---|
| Robot setup (URDF/MJCF import, articulation, gains, tuning) | [Robot_Setup.md](Robot_Setup.md) |
| Robot simulation (controllers, motion planning, grasping) | [Robot_Simulation.md](Robot_Simulation.md) |

---

## User Interface

| Topic | File |
|---|---|
| Full UI reference (windows, panels, menus) | [User_Interface_Reference.md](User_Interface_Reference.md) |
| GUI reference details | [GUI_Reference.md](GUI_Reference.md) |
| Content and asset browsers | [Browsers.md](Browsers.md) |
| Keyboard shortcuts | [Keyboard_Shortcuts_Reference.md](Keyboard_Shortcuts_Reference.md) |

---

## Workflows, Tools & Scripting

| Topic | File |
|---|---|
| End-to-end workflows | [Workflows.md](Workflows.md) |
| Project templates | [Templates.md](Templates.md) |
| Application template | [Application_Template.md](Application_Template.md) |
| OmniGraph visual scripting | [Omnigraph.md](Omnigraph.md) |
| Python scripting and tutorials | [Python_Scripting_and_Tutorials.md](Python_Scripting_and_Tutorials.md) |
| Development tools | [Development_Tools.md](Development_Tools.md) |
| Debugging and profiling (Tracy, logging) | [Debugging_Profiling.md](Debugging_Profiling.md) |

---

## Sensors & Synthetic Data

| Topic | File |
|---|---|
| Sensor types and configuration | [Sensors.md](Sensors.md) |
| Synthetic data generation (Replicator, annotators, writers) | [Synthetic_Data_Generation.md](Synthetic_Data_Generation.md) |
| Data collection and usage telemetry | [Data_Collection_Usage.md](Data_Collection_Usage.md) |

---

## Integrations

| Topic | File |
|---|---|
| ROS 2 bridge, topics, services, OmniGraph nodes | [ROS_2.md](ROS_2.md) |
| Isaac Lab (RL training, environments, tasks) | [Isaac_Lab.md](Isaac_Lab.md) |
| Digital Twin workflows | [Digital_Twin.md](Digital_Twin.md) |

---

## Assets & Import/Export

| Topic | File |
|---|---|
| Isaac Sim asset library | [Isaac_Sim_Assets.md](Isaac_Sim_Assets.md) |
| Asset structure and organization | [Asset_Structure.md](Asset_Structure.md) |
| Importers and exporters (URDF, MJCF, CAD, glTF) | [Importers_and_Exporters.md](Importers_and_Exporters.md) |

---

## Extensions

| Topic | File |
|---|---|
| Adding and updating extensions | [Adding_and_Updating_Extensions_Guide.md](Adding_and_Updating_Extensions_Guide.md) |
| Extension renaming in Isaac Sim 4.5 | [Renaming_Extensions_in_Isaac_Sim_4_5.md](Renaming_Extensions_in_Isaac_Sim_4_5.md) |

---

## Performance & Architecture

| Topic | File |
|---|---|
| Performance optimization handbook | [Isaac_Sim_Performance_Optimization_Handbook.md](Isaac_Sim_Performance_Optimization_Handbook.md) |
| Benchmarks | [Isaac_Sim_Benchmarks.md](Isaac_Sim_Benchmarks.md) |
| Reference architecture and task groupings | [Reference_Architecture_and_Task_Groupings.md](Reference_Architecture_and_Task_Groupings.md) |

---

## Examples & Community

| Topic | File |
|---|---|
| Example scenes and scripts | [Examples.md](Examples.md) |
| Community project highlights | [Community_Project_Highlights.md](Community_Project_Highlights.md) |

---

## API & Documentation

| Topic | File |
|---|---|
| API documentation reference | [API_Documentation.md](API_Documentation.md) |

---

## Release Information & Legal

| Topic | File |
|---|---|
| Release notes and known issues | [Release_Notes.md](Release_Notes.md) |
| Licenses | [Licenses.md](Licenses.md) |

---

## Help, FAQ & Troubleshooting

For all support, FAQ, and troubleshooting questions, consult [Help_FAQ.md](Help_FAQ.md). The complete contents are reproduced below so you never miss any detail.

### Isaac Sim Developer Resources

#### Discord

**[Link to Isaac Sim Discord Channel](https://discord.gg/4ZsTFksGh8)**

Within the **NVIDIA Omniverse** Discord server, search for the **isaac-sim** channel.

#### Forum

**[Link To NVIDIA Forums](https://forums.developer.nvidia.com/c/omniverse/simulation/69)**

This link takes you to the NVIDIA Developer Forums. You must have a user name and password to access the site.

---

### Omniverse Feedback and Forums

Get in touch with Omniverse by a variety of methods. See also [Omniverse_Feedback_and_Forums.md](Omniverse_Feedback_and_Forums.md).

#### Discussion Forums

[Discussion Forums](https://forums.developer.nvidia.com/c/omniverse/300) are the general audience help mechanism for Omniverse, the [forums](https://forums.developer.nvidia.com/c/omniverse/300) are a great place to ask questions and get answers from the community and the Omniverse team.

#### Discord

[Discord](https://discord.com/invite/nvidiaomniverse) offers a variety of channels to meet and interact with the Omniverse Community. It is a great place to stay up to date with the latest information and keep track of our events.

#### Forms

Forms are a great way to offer feedback in a quick way. Suggestions, Praise or Feedback all welcome. We monitor the forms and use the information to help us improve our documentation and products. We also use the information to help us prioritize our work.

* [General Feedback Form](Omniverse_Feedback_and_Forums.md) allows for an easy way to quickly communicate your sentiment.
* [Bug Report Form](Omniverse_Feedback_and_Forums.md) allows you to call out issues on our documentation.

---

### FAQ

#### General

**Is there a way to use Isaac Sim without an internet connection?**

Yes. Download the [Latest Release](Installation.md) of Isaac Sim Assets Packs. See [Local Assets Packs](Installation.md) to run local assets.

**Is there a way to use Isaac Sim without downloading it locally?**

Yes. See [Cloud Deployment](Installation.md)

#### Performance

**Isaac Sim is loading very slowly.**

The first time you open NVIDIA Isaac Sim, it may take a while for the materials to compile.

If a large asset is loading very slowly, it is likely that the materials are compiling. This should happen only on first load of the asset.

See [Location for Isaac Sim shader cache](Installation.md) for location of the local shader cache.

For more information, see [Workstation Installation](Installation.md) for how to install.

**Isaac Sim is running very slowly.**

To speed up the simulation, you can reduce the complexity of the scene and robot, such as reducing the number of joints and links, simplify collision geometry and texture. You can also modify simulation step settings. For more information, see [Isaac Sim Performance Optimization Handbook](Isaac_Sim_Performance_Optimization_Handbook.md).

#### Additional FAQ Pages

Many sections have dedicated FAQ and troubleshooting pages, check them for more targeted information.

* [Installation FAQ](Installation.md)
* [Troubleshooting](#troubleshooting)
* [Robot Simulation Tips](Robot_Simulation.md)

---

### Troubleshooting

This page serves as a central hub for troubleshooting information across Isaac Sim components. For detailed troubleshooting guidance on specific components, follow the links below.

#### Isaac Sim Issues

##### Isaac Lab

- Isaac Lab Troubleshooting — see [Isaac Lab section](#isaac-lab-troubleshooting) below

##### ROS 2 Troubleshooting

- ROS 2 Troubleshooting — see [ROS 2 section](#ros-2-troubleshooting) below

##### Replicator

- Replicator Troubleshooting — see [Replicator section](#replicator-troubleshooting) below

##### Robot Setup

- Robot Setup Troubleshooting — see [Robot Setup section](#robot-setup-troubleshooting) below

##### Digital Twin

- Digital Twin Troubleshooting — see [Digital Twin section](#digital-twin-troubleshooting) below

#### Common Issues

##### Installation Issues

###### Linux Driver Installation

* See [Linux Troubleshooting](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html) to resolve driver installation issues on Linux.
* We recommend installing the **Latest Production Branch Version drivers** from the [Unix Driver Archive](https://www.nvidia.com/Download/Find.aspx?lang=en-us) using the `.run` installer on Linux, if you are on a new GPU or experiencing issues with current drivers.
* NVIDIA driver version **535.216.01** or later is recommended when upgrading to **Ubuntu 22.04.5 kernel 6.8.0-48-generic** or later.

##### Performance Issues

###### Tracy Profiler

For performance troubleshooting, you can use the Tracy profiler to gauge the performance of various components of the application.
See [Profiling Performance Using Tracy](Debugging_Profiling.md) for details on using Tracy for performance profiling.

###### Simulation Frame Rates

If you observe publish rates that differ from the target simulation frame rate, try:

1. Running Isaac Sim with factory settings to clear any persistent simulation frame rate settings:

   ```
   ./isaac-sim.sh --reset-user
   ```
2. Check your computer's CPU usage to identify bottlenecks. If Isaac Sim is exhibiting incredibly high usage, try running with *Fabric* enabled.

###### Reducing Log Output

There can be many warnings and other messages when running Isaac Sim. You can reduce log output using command line arguments:

```
--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
```

##### UI Issues

When the frame rate is low, the UI response may be sluggish. This can be resolved using "Ctrl + click" instead of the standard "click" to select objects.

##### Windows-Specific Issues

###### Thread Cleanup

When running standalone examples in Windows, threads may not be properly cleaned up when the application is closed. This can usually be ignored because the application will still successfully close. As a workaround, add multiple `standalone_app.update()` calls before calling `standalone_app.close()`.

###### File Path Length

The `omni.kit.telemetry` extension startup error with code `(error = 206)` on Windows is caused by a file path exceeding the length limit. Verify that the file path of `omni.telemetry.transmitter.exe` does not exceed 260 characters.

###### GPU Overclocking

If you encounter the error message `Windows fatal exception: int divide by zero` once the app is started, it could be due to GPU overclocking software such as MSI Afterburner. Try disabling such software to resolve the issue.

##### Python Issues

Python errors related to `tkinter` indicate the user is attempting to use `tkinter` with the Python distribution shipped with Isaac Sim. This is not supported.

#### Additional Resources

For a comprehensive list of known issues and their workarounds, see [Known Issues](Release_Notes.md).

---

### Isaac Lab Troubleshooting

This section consolidates troubleshooting information for Isaac Lab components in Isaac Sim. See also [Isaac_Lab.md](Isaac_Lab.md).

#### Common Issues

##### Installation Issues

* Make sure you have the correct Python version (3.9+) when setting up Isaac Lab
* If encountering ModuleNotFoundError, ensure all dependencies are installed via `pip install -e .` in the Isaac Lab repository
* For GPU compatibility issues, verify that your CUDA version matches the requirements for Isaac Lab

##### Performance Issues

* For slow training performance, try reducing the number of environments or the complexity of the scene
* Memory issues can be resolved by setting smaller batch sizes or reducing environment complexity
* To improve frame rates during training, consider using fewer sensors or reducing their resolution

##### Environment Setup Issues

* If robots fail to initialize, check that the URDF/USD files are correctly specified
* For task initialization failures, ensure your task configuration files are properly formatted
* Make sure reward terms and observations are correctly defined in your environment configurations

##### Policy Deployment Issues

* When deploying trained policies, verify the observation space matches the training environment
* For poor policy performance after deployment, check if the simulation parameters match the training settings
* If policy files fail to load, verify they are in the correct format supported by Isaac Lab

---

### ROS 2 Troubleshooting

This section consolidates troubleshooting information for ROS 2 components in Isaac Sim. See also [ROS_2.md](ROS_2.md).

#### ROS 2 Multi-Navigation Issues

The ROS 2 Multi-Navigation tutorial has high CPU usage. If you observe instances of robots colliding or experiencing localization issues, it's likely because the Nav2 stack is unable to properly synchronize with sensor data, resulting in missed controller commands.

To improve Nav2 performance:

1. Try enabling the **Publish Full Scan** checkbox accessible through the publish\_front\_3d\_lidar\_scan OmniGraph node found in the ros\_lidars action graph under each robot.
2. If the previous step still results in issues, try running Isaac Sim from the terminal using the following command:

   ```
   ./isaac-sim.sh --/app/asyncRendering=true --/app/renderFrameTimeout=60 --/app/asyncPhysics=true
   ```

#### ROS 2 Camera Issues

If your depth image only shows black and white sections, it is likely due to somewhere in the field of view having "infinite" depth, which skews the contrast. Adjust your field of view so that the depth range in the image is limited.

If your RGB camera images appear distorted or have incorrect coloring, check the following:

1. Ensure proper camera parameters are set in the ROS 2 camera publisher node
2. Verify that the render product resolution matches your expected output
3. Check if anti-aliasing settings are affecting the image quality

#### MoveIt Integration Issues

If your Rviz window is showing a black screen where the robot should be, you can update your mesa driver. Add the following commands to `moveit2_tutorials/doc/how_to_guides/isaac_panda/.docker/Dockerfile` after line 17:

```
# update mesa driver
RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:kisak/kisak-mesa && apt install -y mesa-utils
RUN apt -y upgrade
```

#### ROS 2 TurtleBot Movement Issues

For TurtleBot movement issues, make sure your robot is on the ground. The table has different properties, making it difficult for the robot to move on it.

Potential solutions:
1. Change the properties of either the ground or the wheels
2. Adjust the friction coefficients on the robot's wheels
3. Verify that the correct controller parameters are being used

#### ROS 2 Publish Rate Issues

If you observe publish rates that differ from the target simulation frame rate, try:

1. Running Isaac Sim with factory settings to clear any persistent simulation frame rate settings:

   ```
   ./isaac-sim.sh --reset-user
   ```
2. Check your computer's CPU usage to identify bottlenecks. If Isaac Sim is exhibiting incredibly high usage, try running with *Fabric* enabled.

If you observe that the */camera\_1/rgb/image\_raw* topic is publishing at a slower rate than anticipated, it can be because the large size of each image message is causing bottlenecks in network traffic or DDS queue management. To improve the publish rate, try reducing the dimensions of the render product resolution by modifying the dimensions in the render product node attached to the image publisher.

#### ROS 2 QoS Profile Issues

The ROS 2 QoS Profile OmniGraph node is unable to save custom profiles unless you manually change the createProfile input to "Custom" first before updating the other fields.

When using sensor data with RViz, be aware that all sensors and images in Isaac Sim are being published with [Sensor Data QoS](https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Quality-of-Service-Settings.html#qos-profiles). If you wish to visualize the images in RViz, expand the image tab, navigate to **Topic > Reliability Policy** and change the policy to Best Effort.

#### General ROS 2 Issues

1. In certain instances, prolonged execution of the ROS 2 `carter_warehouse_navigation.usd` sample scene or the ROS 2 Joint State publisher with the `franka_alt_fingers.usd` asset can lead to a memory leak.
2. When using OmniGraph nodes with ROS 2, make sure to save your scene after setting up the nodes and before hitting play to ensure all values are correctly set.
3. The ROS 2 Auto Namespace feature can not correctly apply to all nodes in complex hierarchies. Review your node namespaces in ROS 2 command line tools to ensure they're behaving as expected.

---

### Replicator Troubleshooting

This section consolidates troubleshooting information for Replicator components in Isaac Sim. See also [Synthetic_Data_Generation.md](Synthetic_Data_Generation.md).

#### Replicator Rendering Issues

If there is unwanted noise in simulated depth images, disable anti-aliasing under the **Render Settings > Ray Tracing > Anti-Aliasing** tab by setting the `Algorithm` to `None`.

If randomized materials are not loaded on time for synthetic data generation, the [rt\_subframes](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples) must be set to be at least `2`.

The replicator Scatter3D OmniGraph node breaks physics when called on a stage using world. Avoid using these together or use alternative methods for object placement.

If ghosting artifacts are observed in the captured data, especially for scenes with moving objects or significant changes in lighting conditions, increase the `rt_subframes` value when capturing the data to a value until the renderer is able to remove the artifacts. For more information see [RT Subframes Parameter](Synthetic_Data_Generation.md) and [subframes examples](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples).

If the captured images are written as black, try starting Isaac Sim once with the `--reset-user` to clear any previous user settings.

#### Async Rendering and Frame Skipping

When using Replicator, frames may be skipped due to the `isaacsim.core.throttling` extension toggling `/app/asyncRendering=True` by default when the timeline is stopped. Since Replicator remains in STARTED mode, it does not re-initialize and toggle the setting back to False, leading to frames being skipped during capture.

**Solution:** Launch Isaac Sim with the following flag to disable async rendering toggling from the throttling extension:

```
--/exts/isaacsim.core.throttling/enable_async=false
```

This occurs because when the timeline is stopped, the throttling extension enables async rendering for performance. However, when Replicator schedules frames for capture before the timeline starts playing again, those frames may be skipped due to async rendering being enabled. The flag above prevents the throttling extension from toggling async rendering, ensuring all scheduled frames are captured properly.

#### Replicator Data Storage Issues

Using Replicator to write to S3 buckets with the built-in backend in Windows may require setting the credentials in the environment variables instead of the AWS config files. This is because of a possible path parsing error in Boto3 on Windows.

When working with large datasets or high-resolution images, you may experience storage bottlenecks. Consider:
1. Using a faster storage device
2. Reducing the image resolution or compression level
3. Using batch processing with smaller batches

#### Replicator Layers and Randomization

Using [replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/basic_functionalities.html)'s `rep.new_layer()` functionality, which creates a new layer in which to place and randomize assets, may lead to issues in simulation scenarios where these assets are used. In such cases the use of `rep.new_layer()` can be omitted.

When using multiple randomizers, be aware that they may conflict with each other. Test your randomization settings carefully to ensure they produce the expected results.

#### Replicator Performance Issues

For complex scenes with many objects and randomizers, you may experience performance issues. Consider:
1. Reducing the number of objects in the scene
2. Simplifying the randomization parameters
3. Using fewer sensors or lower resolution sensors
4. Running with headless mode for improved performance during data generation

#### Replicator API Changes

If you are encountering any issues regarding the dependencies on `omni.replicator.character` or `omni.replicator.agent`, the extension is now renamed to `isaacsim.replicator.agent`. Revise your code accordingly.

#### Getting Started Scripts Issues

Common issues and solutions for the Getting Started Scripts:

1. **Data not being captured**
   - Ensure the capture-on-play flag is properly set
   - Check if the render products are correctly attached to writers
   - Verify the output directory has write permissions
2. **Rendering artifacts**
   - Try increasing RTSubframes value
   - Check if materials are fully loaded before capture
   - Ensure proper lighting setup
3. **Performance issues**
   - Reduce resolution or number of cameras
   - Use headless mode for faster processing
   - Optimize scene complexity
4. **Memory issues**
   - Reduce batch size
   - Clear unused resources with `destroy()`
   - Monitor GPU memory usage

#### First Frame Missing in Windows Standalone Mode

On Windows, when running SDG pipelines with Replicator in standalone mode, the first frame may be skipped by writers or data may be missing from annotators.

##### Workaround

Call a few "warmup" steps to advance the simulation before the first capture to avoid missing the initial frame. For example:

```python
# Warmup the simulation
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(2):
    standalone_app.update()
```

Alternative (depending on your Replicator control flow):

```python
import omni.replicator.core as rep
# [..] initialize writer [..]
rep.orchestrator.step()
# [..] start SDG pipeline [..]
```

---

### Robot Setup Troubleshooting

This section consolidates troubleshooting information for robot setup and simulation in Isaac Sim. See also [Robot_Setup.md](Robot_Setup.md).

#### Reparenting Assets

You can change how reparenting behaves under **Edit > Preferences**, and on the **Stage Panel**, scroll down to authoring. The checkbox **Keep Prim world Transform when reparenting**, lets you decide when reparenting if the objects remain in place or if they get moved to the parent's frame of reference. You can use this to your advantage to apply offsets or change the parent's origin without impacting the children elements.

#### Robot Rigging Issues

If your robot "explodes" during simulation or after some movements, check if any of the collision meshes are colliding with each other.

Common rigging issues and their solutions:

1. Colliding collision geometries — Ensure that collision geometries do not intersect or overlap, especially at joint pivot points
2. Joint limit violations — Verify that joint limits are set appropriately and not being exceeded during simulation
3. Incorrect joint ordering — Make sure that joint orderings in articulation chains are correct
4. Physics instabilities — Adjust physics timestep or solver iteration counts if experiencing vibrations or instabilities

Physics Inspector "failed to find internal joint" errors for robots with mimic joints does not affect the functionality of the mimic joints and can be ignored:

```
[Error] [omni.physx.plugin] Usd Physics: failed to find internal joint object for PhysxMimicJointAPI at /Franka/panda_hand/panda_finger_joint2. Please ensure that the prim is a supported joint type and is part of an articulation.
```

#### Robot Controller Issues

1. Gains produced by the gain turner may not perfectly track the robot's commanded movements (for example, as seen in the Cobotta Pro robot). Manual tuning of gains may be necessary for optimal performance.
2. Some grippers with parallel mechanism (that is, Robotiq 2F-85 and 2F-C2) have links that do not move with rest of the gripper. This is a known issue and may require manual adjustment of the gripper joints.
3. When working with differential drive robots, make sure that wheel friction is appropriate. Too little friction can result in wheel slippage, while too much friction can cause erratic movement.

#### Robot Import Issues

USD to URDF Exporter issues:

* The Collider meshes may be improperly included in the visuals. They can be manually removed from the URDF file.
* The Body and Joints are authored in the URDF file in alphabetical order. They can be manually reordered in the URDF file.
* Depending on the robot structure, some body names may be overridden due to the merging of different frames. Review the output and verify that it's accurate.
* The URDF exporter adds joint effort and velocity limits as inf when unbounded. This may make the URDF not import correctly if the URDF parser does not support inf values in Float.

When importing a URDF:

1. If more than one asset in URDF contains the same material name, only one material is created regardless if the parameters in the material are different. For example, if two meshes have materials with the name "material", one is blue and the other is red, both meshes will be either red or blue. This also applies for textured materials.
2. MJCF importer does not show the built-in bookmark in the file picker dialog. The bookmark is still available in the content pane and can be copy-pasted into the file picker dialog.

#### Closed Loop Structure Issues

For robots with closed-loop kinematic chains:

1. Make sure that the constraints are properly defined and initialized
2. Check that all joints in the closed loop have appropriate drive settings
3. Consider simulating the closed loop as separate articulations with constraints rather than with a single complex closed-loop structure
4. Adjust solver settings for better convergence if experiencing stability issues

#### Robot Importing Tips

1. Sometimes the robot may have non-zero target positions. When the target position does not match the initial position, the robot will move to the target position on the first frame. To prevent this, either set the target position to zero or set the initial position to the target position.
2. Max forces may be high or low in the URDF, set them to a more reasonable value in the USD.
3. If the stiffness and damping values are too high, the robot may oscillate. If it's too low, the robot may not move to the desired position. Use the gain tuner to test the stiffness and damping.
4. If the robot have overlapping collision meshes, use a filtered pair to ignore collisions between specific meshes.

#### Common Robot Issues

| Observation | Solution |
|---|---|
| Robot meshes are penetrating each other after importing | Verify the source file (MJCF or URDF) have the correct transforms for the meshes. Adjust the transforms in the source file or in the USD after importing. |
| Robot joints are not moving at all | Check the joint limits and ensure they are set correctly. Adjust the limits in the source file or in the USD after importing. Verify that the joint gains are non zero. If you have mimic joints, make sure the gear ratio and direction are set correctly. One suggestion is to disable all the joints first, and then add them back one by one to isolate the issue. |
| Robot joints are moving in the wrong direction | Check the joint axis and ensure they are set correctly. Adjust the joint axis in the source file or in the USD after importing. For mimic joints, verify that the direction is set correctly. |
| Robot shakes uncontrollably starting from the first frame | Usually, conflicting collisions can generate abnormal amount of force which cause the robot to behave incorrectly. Check for self overlapping collision geometries. Uncheck self collision enabled in Articulation Root if self collision is not needed. If self collision is required, apply contact filter to specific pairs of colliders that should not collide. |
| Robot shakes uncontrollably after some movements | This usually happens when the robot gains are too high and generating abnormal amount of torque. Try increasing the physics substeps and solver iteration counts in the Physics Settings window. You can also try reducing the robot's maximum velocity and force limits to prevent extreme movements. |
| Robot experiences physX transform errors | This usually happens when the robot is under extreme forces or torques similar to the previous scenario and it can be induced by conflicting joint transformations. First disable all the joints and see if the issue persists. If the issue is resolved, re-enable the joints one by one to isolate the problematic joints. Check for conflicting joint limits or positions. |
| The robot is penetrating the ground or other objects on the first frame | Check the initial position of the robot and ensure it is above the ground plane and not intersecting with any meshes. Verify that the collision geometries are correctly defined and not intersecting with other objects at the start of the simulation. |
| The robot is penetrating the ground or other objects during simulation | Adjust the physics timestep and solver iteration counts to improve stability, modify the contact offset of the colliders to ensure proper collision detection, and verify that the robot's mass and inertia properties are realistic. |
| The simulation performance is slow at run time | Reduce the number of collision meshes and simplify their geometry by using similar colliders, and adjust the physics timestep and solver settings for better performance. |
| The robot joints are not following the commanded positions accurately | Tune the joint gains using the [Gain Tuner Extension](Robot_Setup.md), ensure that the maximum velocity and force limits are set appropriately, and verify that there are no conflicting forces acting on the robot. |

---

### Digital Twin Troubleshooting

This section consolidates troubleshooting information for Digital Twin components in Isaac Sim. See also [Digital_Twin.md](Digital_Twin.md).

#### Warehouse Logistics Issues

##### Warehouse Creator Issues

* If warehouse components don't appear after generation, check for errors in the console logs
* For layout issues, ensure the grid dimensions and spacing are properly configured
* If textures appear incorrect, verify your material settings and check GPU compatibility

##### Conveyor Belt Issues

* For non-functioning conveyors, ensure the physics settings are correctly applied
* If objects fall through conveyors, adjust collision settings and physics parameters
* Animation speed issues can be resolved by checking the conveyor speed settings

#### Cortex Issues

##### Decider Network Issues

* If decision networks fail to initialize, check that all required extensions are enabled
* For unexpected behavior, review your network configurations and connections
* Debug flows by enabling verbose logging and tracing through decisions step by step

##### Asset Loading Issues

* Missing assets can be resolved by checking file paths and ensuring assets are available
* For slow loading of complex assets, consider using simpler versions for testing
* USD file compatibility issues may require updating to the latest USD schema

#### Mapping Issues

##### Occupancy Map Issues

* If occupancy maps fail to generate, ensure the scene has proper collision geometry
* For inaccurate maps, adjust the resolution and sensor parameters
* Missing areas in the map may indicate occlusion issues or raycast failures
