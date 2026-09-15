# Reference Architecture and Task Groupings

Isaac Sim is typically installed and used as one part of a larger solution. Depending on your use case and requirements, this document provides a reference architecture. Almost all use-cases involve some commonalities, which are highlighted as task groupings in the diagrams below. Within each task grouping your product architecture could include one or more of the products or components that are listed.

Regardless of your product components, most Isaac Sim use cases involve the following high level task groupings that occur in roughly the same order:

1. Geometry Authoring

* [SimReady Assets](https://developer.nvidia.com/omniverse/simready-assets)
* [Isaac Sim Assets](Isaac_Sim_Assets.md)

2. Importing Assets

* [Importers and Exporters](Importers_and_Exporters.md)
* [USD Tools](Omniverse_and_USD.md)

3. Scene Setup

* [Robot Setup Tools and Standards](Robot_Setup.md)
* [Robot Setup Tutorials](Robot_Setup.md)
* [Synthetic Sensors](Sensors.md)

4. Interaction with the Digital Twin

* [Robot Simulation and Controllers](Robot_Simulation.md)
* [Robot Articulation and Physics Tools](Physics.md)
* [Motion Generation](Robot_Simulation.md)
* [Omnigraph](Omnigraph.md)

5. Use Cases

* [Isaac Lab](Isaac_Lab.md)
* [Synthetic Data Generation](Synthetic_Data_Generation.md)
* [ROS 2 Bridge](ROS_2.md)
* [Isaac ROS](ROS_2.md)

Typical use-cases are summarized in [Use Cases](#isaac-sim-ra-consumption).

## Geometry Authoring

The simulation environment (scene) is composed of various components including robots,
static, and dynamic objects. The mechanical and geometrical design for these components
is usually done with CAD software like Solidworks, Pro-E, Catia, AutoCad, or Creo. Parts and
components of varying complexity can be designed and assembled.

Developers can also leverage existing 3D asset libraries, which provide a vast collection of
existing 3D assets. Omniverse and Isaac Sim leverage a file format called [Universal Scene
Description (OpenUSD)](https://www.nvidia.com/en-us/omniverse/usd/).

All assets need to be converted to OpenUSD before they can be used with Isaac Sim, and the default unit for Isaac Sim is meters.

NVIDIA provides a vast collection of OpenUSD ‘SimReady’ assets. [SimReady](https://developer.nvidia.com/omniverse/simready-assets), or
simulation-ready, assets are physically accurate 3D objects that have accurate
physical properties, behavior, and connected data streams that are used to represent the real world in
simulated digital worlds. Developers can use these building blocks to construct scenes and
generate data per their requirements. The Warehouse asset collection includes over
800 3D assets of commonly available tools, equipment, and items in a warehouse including
forklifts, pallets, racks, and shelves.

## Importing Assets

There are extensions that enable importing CAD (Computer Aided Design) files into Isaac Sim
that handle conversion to OpenUSD. Extensions are core building blocks that
interact with and add or extend the functionality of Isaac Sim.

### Importing and Creating Environments

The [asset importer](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-importer.html#asset-importer)
can be leveraged for importing OBJ, FBX, and glTF formats. The [CAD converter](https://docs.omniverse.nvidia.com/extensions/latest/ext_cad-converter.html#cad-converter)
extension supports a variety of popular CAD files from applications including
Catia, Solidworks, AutoCad, and Creo. This enables you to quickly convert and
import your environment into Isaac Sim.

OpenUSD Connections and Data Exchange, formerly Omniverse Connect, is a collection of
[importers](https://docs.omniverse.nvidia.com/connect/latest/catalog.html#importers-exporters),
[exporters](https://docs.omniverse.nvidia.com/connect/latest/catalog.html#exporters),
[converters](https://docs.omniverse.nvidia.com/connect/latest/catalog.html#converters), and
[USD file format](https://docs.omniverse.nvidia.com/connect/latest/catalog.html#file-format-plugins) plug-ins. They enable various 3D applications, products, and file formats to exchange data using OpenUSD.

Some CAD applications have connectors with Omniverse, which allows them to bring over more relevant
and contextual information when converting to USD. For example, PTC Creo, Autodesk Revit, or Autodesk Alias have corresponding connectors. The files generated
from their CAD converters will have all the visual meshes represented in OpenUSD.

### Importing Robots

Isaac Sim comes with a variety of robots already imported. The pre-imported robots can be found on the [Robot Assets](Isaac_Sim_Assets.md) page. Isaac Sim also provides advanced options for importing other robots.

If your robot is in [URDF](https://wiki.ros.org/urdf), you can use the [URDF Importer Extension](Importers_and_Exporters.md) extension from the GUI or from Python. This extension will import the visual meshes and the prim hierarchies (child-parent relationships), along with extra information about how the collision meshes, joints, and sensors are encoded.

You could also use the [Onshape Importer](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html) and [MJCF Importer Extension](Importers_and_Exporters.md). With these importers, you will have to add the joint drives in
and may have to tune them. The [Gain Tuner Extension](Robot_Setup.md) allows you to visualize and tune the joints.

## Scene Setup

You can set up the scene after all the necessary assets are converted to OpenUSD and
imported into Isaac Sim.

To properly simulate real world situations, you must have physics characteristics defined. For example, physics characteristics define if an object is subject to gravity or how solid it is.

### Adding Physics

After importing the required assets into Isaac Sim, make sure they have appropriate
Physics for accurate simulations. Some asset importers like the URDF and Onshape Importer carry over
most Physics parameters and configurations, for the rest of the imported assets adding
physics before proceeding would be necessary. The [NVIDIA Omniverse™ Physics simulation
extension](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics.html#physics-core)
is powered by the NVIDIA PhysX SDK. It supports Rigid Body Simulation,
Character Control, Deformable Body Simulation, Particle Simulation, and Articulations. The
important steps for adding Physics to your scene are:

> 1. Creating the physics scene
> 2. Assigning collision settings
> 3. Adding joints and drives.

#### Creating the Physics Scene

The first step is to create a Physics Scene and ensure that the default parameters
for it are acceptable. For example, verify the direction and magnitude of gravity in
the scene. If the imported scene does not contain a ground plane, make sure to add
one before proceeding. It will prevent any physics-enabled objects from falling
below it. Unless you are simulating hundreds of rigid bodies and robots, it is more
efficient to use the CPU solver instead of the GPU solver.
Refer to the [Tutorial 1: Stage Setup](Robot_Setup.md) tutorial

#### Assigning Collision Settings

[Collision](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html#create-a-rigid-body-with-a-collider)
enables rigid bodies to interact with each other in an environment. The
geometry of the object can be approximated by convex hull, convex decomposition,
bounding sphere, bounding box, and SDF collision meshes. Each of them
approximates the geometry using different methods and may be better suited for
specific use cases. PhysX supports exact representations for Cube, Capsule, and
Sphere shapes. Cones and Cylinders are supported through the custom geometry
flag and are particularly useful when setting collision approximations for wheels of
robots. [Rigid-body physics materials](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html#configure-rigid-body-s-material-properties)
provide friction, restitution (a.k.a. ‘bounciness’), and material density properties

#### Adding Joints and Drives

After adding the appropriate collision meshes to prims in the scene, we need to
ensure that they interact correctly with one another. We can do this by defining
appropriate joints between prims connected to each other. Joints give you the
ability to connect physics objects by defining how the objects may move relative to
each other. There are various [joints types](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html)
to select from including revolute, prismatic, spherical, fixed, etc.

### Adding Sensors

Isaac Sim sensor simulation extensions can simulate ground truth perception and
physics-based sensors, and has a library of realistic sensor models. You can simulate
camera, lidar, radar, and physics-based sensors. It is possible to use camera calibration
parameters obtained from OpenCV or ROS by converting them to Isaac Sim units, refer to the [Camera Sensors](Sensors.md) page.
RTX Lidar and Radar sensors are simulated at render time on the GPU with RTX hardware. A variety of physics-based sensors
like contact sensors, IMUs sensor, force sensor, effort sensor, and proximity sensor are also included. These sensors can be added
at the appropriate locations in the stage hierarchy (for example, a camera or lidar might be
added near the front/top of the robot). The [Camera and Depth Sensors](Sensors.md) and [Non-Visual Sensors](Sensors.md) pages
highlight all the available physical sensor assets available with Isaac Sim

### Import and Create Materials

[Materials](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/materials.html#omniverse-materials) in Isaac Sim
are supported using [NVIDIA Material Definition Language (MDL)](https://www.nvidia.com/en-us/design-visualization/technologies/material-definition-language/) ,
a shading language designed for defining and describing the appearance
of materials in computer graphics. It allows artists and developers to create highly realistic
materials by specifying their physical properties, surface characteristics, and how they
interact with light. Omniverse comes with several template materials, including a physically
based glass; several general purpose multi-lobed materials useful for dielectric and
non-dielectric materials, skin, hair, liquids and other materials requiring subsurface
scattering or transmissive effects; and USD’s UsdPreviewSurface.

## Interaction with Digital Twin

Once the assets have been imported and the scene has been set up, there are various ways
to interact with the simulated environment, which are summarized below.

### GUI

The GUI provides intuitive controls for scene management, object manipulation, and
real-time monitoring, providing a streamlined interface for developing and testing robotic
systems. Pre-packaged examples, robots and environments can easily be accessed and
added to the scene via the GUI. Create tools make it easy to assemble, illuminate, simulate,
and render scenes large and small, therefore making it the ideal place to build your virtual
worlds, assemble robots, and examine physics. Refer to the [GUI Reference](GUI_Reference.md) for
getting started with the GUI tutorials.

### Standalone Python

Isaac Sim provides a built-in [Python Environment](Python_Scripting_and_Tutorials.md) that packages can use, like a
system-level Python install. This is the recommended environment for running Python
Scripts with Isaac Sim. All Isaac Sim libraries and dependencies can be imported and
accessed through this Python environment. It also allows users to script and run their
entire worflkow headlessly. For using libraries and tools which are not a part of Isaac Sim,
ensure that they work with Python 3.11 first. A collection of standalone python examples is
provided with Isaac Sim and serves as a good starting point to understand the overall
steps involved. Jupyter notebook and Visual Studio Code support is also available. Workflows from the GUI
can be completely scripted in Python and can be run in headless mode too.

### Extensions

Extensions enable developers to add functionality and integrate other tools for Isaac Sim.
They are individually built application modules. All the tools used in Isaac Sim are built as
extensions. Various extensions enable easier interaction with sensors, robots and prims in
the scene. The ROS 2 Bridge extension can be used to connect your ROS packages and
code to Isaac Sim. Developers can write their own extensions in C++, Python or a
combination of both.

### OmniGraph

[OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html) is the visual scripting language
for Omniverse. It is not a single type of graph, but a composition of many different graph systems
under a single framework. Many Isaac Sim extensions provide nodes for building graphs for common
use cases. Core, sensor, and ROS extensions are a few examples that contain such OmniGraph Nodes.

## Use Cases

### Synthetic Data Generation (SDG)

Developers can generate physically accurate synthetic data that can enhance the training
and performance of AI perception networks used for robotics using
[Omniverse Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html).
Replicator is a collection of extensions, Python APIs, workflows, and tools that enable
synthetic data generation tasks.

Once the scene has been set up, Replicator can be used to modify and randomize various
features like position, rotation, lighting, size, and textures of assets in the scene. A wide
range of annotations are supported including 2D bounding boxes, 3D bounding boxes,
semantic and instance segmentation masks, normals, depth, pointclouds, and more, with
data being written in common formats like COCO and KITTI formats. Custom annotators
and writers can also be implemented for advanced use cases like pose estimation. This
enables developers to seamlessly integrate the generated data with their training pipelines

To get started, developers can leverage the Python API provided by Omniverse Replicator
for generating synthetic data. The same scripts can be used to generate data headlessly in
the cloud through the Isaac Sim docker container (instructions [Container Installation](Installation.md))
on a developer’s preferred CSP (AWS, Alibaba, Azure, GCP) with the [Cloud Deployment](Installation.md) guide.
[Replicator YAML](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/yaml_workflow.html#replicator-yaml)
can be used for low-code situations where scripts can easily be edited by non-technical experts.
They offer a high level of portability and care suitable for cloud use cases.

### Software in the Loop (SIL)

Once the simulation scenario has been set up, developers can tune and test various
aspects of their robotics software stack. The insights gained after varying parameters and
configurations of the software stack and the simulated robot enable an easier and
accurate transition to the physical real-world robot. A few common use cases are
highlighted below:

#### Single and Multi-Robot Navigation

The navigation stack for a robot can be easily tested in various scenarios. The assets in the
environment can be randomized to come up with these scenarios. For running navigation
with multiple robots, a multi-GPU setup can be leveraged. Isaac Sim supports the ROS 2
Nav2 stack via the ROS 2 Bridge.

#### AI Model Evaluation

Evaluating models is easy in simulation because of direct access to ground truth which
could be through Physics, the state of a robot, or reading of a sensor. These can then be
compared with the model predictions to obtain the evaluation metrics.

For example, in computer vision tasks, the rendered image from the simulated camera can
be passed through the model for obtaining predictions. This can then be compared with
the ground truth (available directly from simulation or via Replicator) to obtain evaluation
metrics. This can also be done for other sensors like Lidars and can be easily extended to
multimodal applications

#### Perception

[Isaac Perceptor](https://developer.nvidia.com/isaac/perceptor) is a reference workflow of
NVIDIA-accelerated libraries and AI models that helps you quickly build robust autonomous
mobile robots (AMRs) to perceive, localize, and operate in unstructured environments like
warehouses or factories. It works with inputs from simulated environments in Isaac Sim

#### Manipulation

[Isaac Manipulator](https://nvidia-isaac-ros.github.io/reference_workflows/isaac_manipulator/index.html)
can be leveraged for manipulation tasks and verified in simulation. It is a
collection of GPU-accelerated packages for perception driven manipulation, providing
capabilities such as object detection and pose estimation. Time optimal collision-free
motion can be generated with cuMotion. Nvblox can be used for local 3D reconstruction
and obstacle detection. MoveIt is also supported via the ROS Bridge in Isaac Sim.

#### Reinforcement Learning

[Isaac Lab](Isaac_Lab.md) is a united and modular framework for robot learning that aims to simplify
common workflows in robotics research (such as RL, learning from demonstrations, and
motion planning). It is built upon NVIDIA Isaac Sim.

### Hardware in the Loop (HIL)

Hardware in the loop testing and evaluation can be done with Isaac Sim. The target
deployment device will receive all the data from the simulated robot and sensors which can
be fed to the needed software stacks/algorithms. ROS 2 can be leveraged as the
middleware which handles sending and receiving all the data from the simulation computer
to the target device. For example, a simulated camera from Isaac Sim will stream over all
the image data via the ROS 2 bridge from Isaac Sim to an
[NVIDIA Jetson Orin](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/), the
embedded device on the robot which will run the computer vision application. This is
particularly useful when selecting the target deployment device to verify it can run the
software stack needed before physically setting up the robot.

### CI/CD

#### OSMO

[OSMO](https://developer.nvidia.com/osmo)
is a cloud-native workow orchestration platform that lets you easily scale your
workloads across distributed environments — from on-premises to private and public
cloud. You can now apply for early access.

#### Sizing Calculator

Please refer to the [Isaac Sim Benchmarks](Isaac_Sim_Benchmarks.md) page for Isaac Sim performance benchmarks
across multiple consumer and enterprise hardware configurations.

## Further Reading

Follow the relevant tutorials for a deeper dive into a corresponding section.