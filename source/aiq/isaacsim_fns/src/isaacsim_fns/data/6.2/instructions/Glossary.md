# Glossary

This section provides an explanation of the terms used throughout NVIDIA Isaac Sim and replicates several of the terms defined in the Omniverse Glossary.

## [Omniverse](#id3)

### [Application](#id4)

An Omniverse App is built upon a specific set of Extensions to provide a desired functionality. An App gives the user a customized experience by implementing the UI’s of its Extensions with a custom layout. You can quickly and easily create customized Apps comprised of any number of Extensions developed by you, the Omniverse Community or NVIDIA. An App can be as simple as a 3D viewer or as complex as an AI suite. This modular approach to building Apps makes it easy to create a customized workflow or a global scale cloud application

### [Apps](#id5)

An Omniverse App is built upon a specific set of Extensions to provide a desired functionality. An App gives the user a customized experience by implementing the UI’s of its Extensions with a custom layout. You can quickly and easily create customized Apps comprised of any number of Extensions developed by you, the Omniverse Community or NVIDIA. An App can be as simple as a 3D viewer or as complex as an AI suite. This modular approach to building Apps makes it easy to create a customized workflow or a global scale cloud application

### [Connectors](#id6)

An Omniverse Connector is middleware with which Omniverse and other software applications communicate with each other. They enable the import/export 3D assets, data, and models between different tools and workflows. It’s important to note that this means using USD as the “go between” format to convert 3D data.

### [Omniverse Nucleus](#id7)

Omniverse Nucleus offers a set of fundamental services that allow a variety of client applications, renderers, and microservices to share and modify representations of virtual worlds.

Nucleus operates under a publish/subscribe model. Subject to access controls, Omniverse clients can publish modifications to digital assets and virtual worlds to the Nucleus Database (DB) or subscribe to their changes. Changes are transmitted in real-time between connected applications. Digital assets can include geometry, lights, materials, textures and other data that describe virtual worlds and their evolution through time.

This allows a variety of Omniverse-enabled client applications ( Apps, Connectors, and others) to share and modify authoritative representations of virtual worlds.

* See [Nucleus overview](https://docs.omniverse.nvidia.com/nucleus/latest/overview/overview.html "(in Omniverse Nucleus)") for a more in-depth look at Nucleus’s data model, architecture, and distribution platforms.

### [Hub Workstation Cache](#id8)

Hub Workstation Cache is a service that helps speed up USD workflows on your local workstation. This is a stand-alone service that runs on your local workstation and benefits Kit-based applications or Client Library tools.

Hub Workstation Cache has been performance optimized and supports storage-derived data from newer versions of Kit-based applications.

* See [Hub Workstation Cache overview](https://docs.omniverse.nvidia.com/utilities/latest/cache/hub-workstation.html "(in Omniverse Utilities)") for more details.
* See the [Workstation Installation](Installation.md) for how to install

### [Live Sync](#id9)

Live Sync mode enables real-time “live” editing of shared files on a Nucleus Server. The Live Sync button is on the top-right corner of the Workspace.

### [Omniverse Kit](#id10)

NVIDIA Omniverse™ Kit is a toolkit for building native Omniverse applications and microservices. It is built on a base framework known as Carbonite that provides a wide variety of functionality through a set of light-weight plugins. Carbonite plugins are all authored with C interfaces for persistent ABI compatibility. A Python interpreter is provided for scripting and customization.

NVIDIA Omniverse™ Kit exposes much of its functionality through Python bindings. This provides an API that can be used to write new extensions to Omniverse Kit or new experiences for Omniverse.

* For a more in-depth look at developing in Kit, see the [Kit Programming Manual](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/kit_overview.html "(in Omniverse Kit)").

### [Omniverse Launcher](#id11)

The NVIDIA Omniverse Launcher is your first step into the Omniverse. It provides immediate access to all the apps, connectors and other downloads within the Omniverse.

* See the [Launcher overview](https://docs.omniverse.nvidia.com/launcher/latest/index.html "(in Omniverse Launcher)") for more details.

### [Omniverse USD Composer](#id12)

NVIDIA Omniverse™ USD Composer was an Omniverse app for world-building that allows users to assemble, light, simulate and render large scale scenes. It is built using NVIDIA Omniverse™ Kit. The Scene Description and in-memory model is based on Pixar’s USD. USD Composer takes advantage of the advanced workflows of USD like Layers, Variants, Instancing and much more.

### [Carbonite (carb)](#id13)

The Carbonite SDK provides the core functionality of all Omniverse apps. This is a C++ based SDK with Python bindings that provides features such as plugin management, input handling, file access, asset loading and management, thread and task management, and much more.

### [RTX - Real-Time mode](#id14)

High quality real-time rendering mode.

### [RTX – Interactive (Path Tracing) mode](#id15)

The highest quality, physically accurate rendering mode.

### [Extensions](#id16)

Extensions are plug-ins to Omniverse Kit that extend its capabilities. They are offered with complete source code to help developers easily create, add, and modify the tools and workflows they need to be productive. Extensions are the core building blocks of Omniverse Kit based applications.

* See [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") for more details.

### [Omniverse Connect](#id17)

Connectors are extensions and additional software layers on top of the open-source USD distribution that allow DCC tools and compute services to communicate easily with each other through the Omniverse Nucleus DB. Those extensions and additions are collectively known as NVIDIA Omniverse™ Connect.

## [USD](#id18)

### [USD](#id19)

Universal Scene Description (USD) is an easily extensible, open-source 3D scene description file format developed by Pixar for content creation and interchange among different tools. As a result of its power and versatility, it’s being widely adopted, not only in the visual effects community, but also in architecture, design, robotics, manufacturing, and other disciplines.

* For a more in-depth look at USD in Omniverse, see NVIDIA’s USD primer [What is USD?](https://developer.nvidia.com/usd/).
* See the [USD API](https://graphics.pixar.com/usd/release/index.html) docs for more details.
* See the [USD Glossary of Terms & Concepts](https://graphics.pixar.com/usd/release/glossary.html) for more details.
* See [NVIDIA’s USD tutorials](https://developer.nvidia.com/usd/tutorials)

### [MDL](#id20)

Material Definition Language (MDL) is a NVIDIA-developed USD schema that represents material assignments and specifies material parameters.

### [Stage](#id21)

The Omniverse Stage window allows you to see all the assets in your current USD Scene. The [USD Stage](https://graphics.pixar.com/usd/release/glossary.html#usdglossary-stage) is the USD abstraction for a scenegraph derived from a root USD file, and all of the referenced/layered files it composes. Listed in a hierarchical (parent/child) order the Stage offers convenient access and is typically used to navigate large scenes.

* See the [Stage](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_stage.html "(in Omniverse Extensions)") docs for more details.
* See the [USD Glossary of Terms & Concepts](https://graphics.pixar.com/usd/release/glossary.html) for more details.

### [Prim](#id22)

A [Prim](https://graphics.pixar.com/usd/release/glossary.html#usdglossary-prim) is the primary container object in USD: prims can contain (and order) other prims, creating a “namespace hierarchy” on a Stage,
and prims can also contain (and order) properties that hold meaningful data. Prims, along with their associated, computed indices, are
the only persistent scenegraph objects that a Stage retains in memory, and the API for interacting with prims is provided by the UsdPrim class.

* See the [USD Glossary of Terms & Concepts](https://graphics.pixar.com/usd/release/glossary.html) for more details.

### [Mesh](#id23)

A mesh is a subdividable primitive that consists of points, edges, and faces that define its shape. In USD, a mesh is encoded in a [UseGeomMesh](https://graphics.pixar.com/usd/release/api/class_usd_geom_mesh.html) class.

### [Shape](#id24)

A Shape is a geometric primitive that maps to one of USD’s five “intrinsic” `UsdGeomGprim` classes:

> * [UsdGeomCapsule](https://graphics.pixar.com/usd/release/api/class_usd_geom_capsule.html)
> * [UsdGeomCone](https://graphics.pixar.com/usd/release/api/class_usd_geom_cone.html)
> * [UsdGeomCube](https://graphics.pixar.com/usd/release/api/class_usd_geom_cube.html)
> * [UsdGeomCylinder](https://graphics.pixar.com/usd/release/api/class_usd_geom_cylinder.html)
> * [UsdGeomSphere](https://graphics.pixar.com/usd/release/api/class_usd_geom_sphere.html)

Shapes are not [meshes](https://docs.omniverse.nvidia.com/utilities/latest/common/glossary-of-terms.html#term-Mesh "(in Omniverse Utilities)"), in that they are not defined by a collection of points, edges, and faces. Instead, they are defined by their shape and volume.

Pixar describes their use cases for these prims in their [UsdGeomGprim schema documentation](https://graphics.pixar.com/usd/release/api/usd_geom_page_front.html).

### [Reference vs Payload vs Instance](#id25)

Everything in USD is a primitive (prim) with attributes. Some of these primitives are defined in your current layer (the active stage), while others are defined in other layers (other USD files).

A primitive that is included from some other layer is a **Reference** to that prim, and are indicated by the **orange arrow** on the associated Xform icon in the context tree of Isaac Sim. References are designed to be lightweight, and carry with them an implicit assumption that the child prims of a reference will not be modified.

If the contents of a reference need to be modified during simulation, then it must be converted into a **Payload**. A payload is indicated by the **blue arrow** on the associated Xform in the context tree of Isaac Sim. Payloads are references that have all of their data actively loaded by the sim so that it can be modified at runtime.

**Instances** are indicated by a **blue “I”**, and can be either references or payloads. They carry additional assumptions about the structure of the asset for more efficient vectorization (scaled up).

For example, suppose you want to collect synthetic data with a robot. If you aren’t going to modify the structure of the robot, it can exist as a reference on the stage (the asset is defined in some other file). If, during data collection, you want to be able to swap the robot out for a different one, those meshes need to be held in active memory. This means that the asset first needs to be converted from a reference to a payload. If you wanted to collect data with a 1000 robots at once, and they are all the same, you might use instantiable references. Whereas, if you wanted to collect data with a 1000 randomly sampled robots (different arms with the same number of joints for example), you would use instance payloads.

### [Y-Up / Z-Up](#id26)

The axis of orientation of a given scene/prim. Y-Up refers to the Positive Y Axis is pointing up. Z-Up refers to the Positive Z Axis is pointing up. This orientation setting is generally set by the application of the scene/prims origination.

### [Layer](#id27)

A component of the collaborative nature of USD. Each layer in USD signifies a user’s “opinion” on assets inside a stage. Layers can override other layers.

### [Instance](#id28)

A light-weight and less manipulable copy of a prim.

### [Checkpoint](#id29)

Immutable historical file versions. Checkpoints are used for version control and allow you to look at and restore the stage to a previous state.

## [PhysX](#id30)

NVIDIA PhysX is a scalable multi-platform physics simulation solution.
The NVIDIA Omniverse™ Physics simulation extension is powered by the NVIDIA PhysX SDK, and includes
Rigid Body Simulation, Articulations, Deformable-Body Simulation, and Character Controller.

* See [Physics Core](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html "(in Omni Physics)") for more details.

## [Isaac Sim](#id31)

### [ROS / ROS 2](#id32)

The [Robot Operating System (ROS)](https://www.ros.org/) is a set of software libraries and tools that help you build robot applications.
NVIDIA Isaac Sim provides many extensions, examples, and APIs for connecting to ROS and ROS 2 workflows.

### [Dynamic Control](#id33)

The Dynamic Control extension a set of utilities to control physics objects. It provides opaque handles for different physics objects that remain valid between PhysX scene resets, which occur whenever play or stop is pressed.

* See the [API Documentation](../py/source/deprecated/omni.isaac.dynamic_control/docs/index.html) documentation for full usage examples and API details.

Note

omni.isaac.dynamic\_control is deprecated.

### [Core API](#id34)

Important

Isaac Sim 5.0.0 has introduced the [Core Experimental API](../py/docs/overview/experimental.html): a rewritten implementation of the current Core API
designed to be more robust, flexible, and powerful, yet still maintain the core utilities and wrapper concepts.

Going forward, it will become the base API used in all Isaac Sim source code.
The current Core API will be deprecated and removed in future releases.

Therefore, **we strongly encourage early adoption and use of the Core Experimental API**.

The Isaac Core Extension in Isaac Sim provides high-level interfaces to PhysX and raw USD APIs. It abstracts away default parameters to simplify creation and manipulation of a simulated world
and scenarios encountered in robotics simulators. Specifically, the extension allows for

> 1. easy creation, manipulation, and management of the world, all its time-related events, and related physical and numerical parameters
> 2. creation of various robotic tasks and controllers
> 3. vectorized manipulation of reinforcement learning environments through various view classes such as `ArticulationView`, `RigidPrimView`, `XFormPrimView`, which provide high-level functionalities to manipulate in parallel sets of articulations, rigid prims, and xforms, respectively.

* See the [API Documentation](../py/source/extensions/isaacsim.core.api/docs/index.html) documentation for full usage examples and API details.

### [Riemannian Motion Policy (RMP)](#id35)

Riemannian Motion Policy (RMP) is a set of motion generation tools that underlies most of our manipulator controls inside Omniverse Isaac Sim. It creates smooth trajectories for the robots with intelligent collision avoidance.

* See the [Motion Generation](Robot_Simulation.md) documentation for more details and examples.

### [World](#id36)

World is the core class that enables you to interact with the simulator in an easy and modular way. It takes care of many time-related events such as adding callbacks, stepping physics, resetting the scene, adding tasks, etc. The World class is a Singleton which means only one World can exist while running Omniverse Isaac Sim. Query the World for information about the simulation from different extensions.

### [Scene](#id37)

A world contains an instance of a Scene, think about it as a scene management class that manages the assets of interest in the USD stage. It provides an easy API to add, manipulate and inspect different USD assets in the stage as well as setting its default reset states. Many of the object classes available which could be added to a Scene usually takes an already existing USD prim in stage or creates a new USD prim, thus providing an easy way to set/ get its common properties.

### [Task](#id38)

The Task class in `isaacsim.core.api` provides a way to modularize the scene creation, information retrieval, calculating metrics and creating more complex scenes with more involved logic.

### [Articulation](#id39)

An articulated robot is a robot with rotary joints (e.g: a legged robot, a manipulator or a wheeled robot). In `isaacsim.core.api` extension in NVIDIA Isaac Sim there exists an Articulation class which enables the interaction with articulations that exists in a USD stage in an easy way.

### [Replicator](#id40)

Replicator is a Synthetic Data Generation tool for creating parameterizable offline datasets in NVIDIA Isaac Sim.

* See the [omni.replicator extension documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") for additional usage information.

### [Synthetic Data Generation](#id41)

NVIDIA Isaac Sim supports Synthetic Data Generations workflows. See [Replicator](#isaac-sim-glossary-replicator) for more details.

### [Ground Truth](#id42)

NVIDIA Isaac Sim can be used to generate ground truth data that is very similar to real-life analogs. See [Replicator](#isaac-sim-glossary-replicator) for more details.