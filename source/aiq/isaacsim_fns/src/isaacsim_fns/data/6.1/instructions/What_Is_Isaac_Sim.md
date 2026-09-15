

# What Is Isaac Sim?

NVIDIA Isaac Sim™ is a reference application built on NVIDIA Omniverse that enables developers to develop, simulate, and test AI-driven robots in physically-based virtual environments.

## Design

Isaac Sim comes with a collection of workflows for importing and tuning mechanical
systems designed in the most common formats including [Onshape](https://docs.omniverse.nvidia.com/extensions/latest/ext_onshape.html#isaac-onshape-importer), the [Unified Robotics Description Format (URDF)](Importers_and_Exporters.md), and the [MuJoCo XML Format (MJCF)](Importers_and_Exporters.md).
This is made possible through the use of the [Universal Scene Description (USD)](https://openusd.org/release/index.html), an easily extensible, open source 3D scene description API that
serves as the unifying data interchange format at the heart of Isaac Sim.

## Tune and Train

The core functionality of Isaac Sim is the simulation itself: a high-fidelity GPU-based [PhysX engine](https://developer.nvidia.com/physx-sdk), capable of
supporting [multi-sensor RTX rendering](Sensors.md) at an industrial scale. Isaac Sim’s direct access to the GPU enables the platform to support the simulation of various
kinds of sensors including [cameras](Sensors.md), [Lidars](Sensors.md), and [contact sensors](Sensors.md). This
in turn facilitates the simulation of digital twins, allowing your end-to-end pipelines to run before ever needing to turn on a real robot. Isaac Sim provides a suite of tools for collecting synthetic
data with [Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html), orchestrating simulated environments through [Omnigraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html), tuning [PhysX simulation](https://docs.omniverse.nvidia.com/extensions/latest/ext_simulation.html) parameters to match reality, and finally training control agents
through various methods like Reinforcement Learning (RL) with [Isaac Lab](Isaac_Lab.md).

## Deploy

Isaac Sim comes pre-equipped with all of the components necessary to not only deploy agents to real robots, but also build applications that are fully integrable with
such systems. [Omniverse](https://docs.omniverse.nvidia.com/dev-guide/latest/index.html) provides [APIs](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/kit_overview.html) for app infrastructure including GUI creation and file management.
The Isaac Sim platform also provides bridge APIs to [ROS 2](ROS_2.md), for direct communication between live robots
and the simulation, as well as [NVIDIA Isaac ROS](https://nvidia-isaac-ros.github.io/), a collection of performant, hardware-accelerated ROS 2 packages for making autonomous robots.

## Getting Started

* [Quick Install](Quick_Install.md): The quick install to get you started in under an hour.
* [Isaac Sim Basic Usage Tutorial](Quick_Tutorials.md): The tutorial to get your feet wet with NVIDIA Isaac Sim.
* [Workstation Installation](Installation.md): Installation guide for a local workstation.
* [Container Installation](Installation.md): Installation guide for a remote headless server.
* [Development Tools](Development_Tools.md): The tools and environments for debugging and development.
* [Python Scripting](Python_Scripting_and_Tutorials.md): Tools and tutorials for building environments, robots, and tasks using NVIDIA Isaac Sim Core Python APIs.
* [GUI Reference](GUI_Reference.md): The fundamental concepts of robotics in NVIDIA Isaac Sim via GUI.
* [Importer and Exporter](Importers_and_Exporters.md): Tools for importing and exporting robots and assets from various file formats.
* [Robot setup](Robot_Setup.md): Isaac Sim tools for modifying robots.
* [Robot Setup Tutorials Series](Robot_Setup.md): Tutorial series for using the robot setup tools and workflows.
* [Robot simulation](Robot_Simulation.md): Controllers, motion generation tools for simulating robots.
* [ROS 2](ROS_2.md): ROS 2 bridges and interfaces.
* [Isaac Lab](Isaac_Lab.md): Reinforcement learning framework and Cloner APIs.
* [Synthetic Data Generation](Synthetic_Data_Generation.md): Collection of tools and workflows for generating synthetic data.
* [Digital Twin](Digital_Twin.md): Tools for building and operating digital twins, such as [Warehouse logistics](Digital_Twin.md), [Cortex](Digital_Twin.md), and [Mapping](Digital_Twin.md).

## System Architecture

The purpose of Isaac Sim is to support the creation of new robotics tools and empower the ones that already exist. The platform provides a flexible API for both C++ and Python and can be integrated into a project to varying degrees depending on your needs. The goal of the platform is not to compete with current or existing software, but to collaborate with and enhance it. To this end, many components of Isaac Sim are open source and freely available for independent use. You may want to design your robot in OnShape, simulate its sensors with Isaac Sim, and control the stage through ROS or some other messaging system. Likewise, it is also possible to build a complete, standalone application entirely on the platform provided by Isaac Sim!

## Omniverse Kit

Isaac Sim uses the Omniverse™ Kit, a toolkit for building native Omniverse applications and microservices. Omniverse Kit provides a wide variety of functionality through a set of lightweight plugins. Plugins are authored with C interfaces for persistent API compatibility; however, a Python interpreter is also provided for accessible scripting and customization.

The Python API can be used to write new extensions to Omniverse Kit or new experiences for Omniverse.

## Development Workflows

Isaac Sim is built on C++ and Python, and operates most commonly through the use of compiled plugins and bindings respectively. This means the platform is capable of supporting a wide variety of workflows for building and interacting with projects that make use of Isaac Sim. Isaac Sim comes with a full, standalone Omniverse application for interacting with and simulating robots, and while this is the most common way users interact with the platform, it is by no means the only method. Isaac Sim also provides direct Python development support in the form of extensions for VS Code and Jupyter Notebooks. Isaac Sim is not limited to synchronous operation either, and can operate with hardware in the loop through ROS 2, facilitating sim-to-real transfer and digital twins.

## USD

NVIDIA Isaac Sim uses the USD interchange file format to represent scenes. Universal Scene Description
(USD) is an easily extensible, open-source 3D scene description file format developed by Pixar
for content creation and interchange among different tools. Because of its power and versatility,
USD is being adopted widely, not only in the visual effects community, but also in architecture,
design, robotics, manufacturing, and other disciplines.

* For a more in-depth look at USD in Omniverse, see the NVIDIA USD primer [What is USD?](https://developer.nvidia.com/usd).
* See the [USD API](https://graphics.pixar.com/usd/release/index.html) docs for more details.
* See the [NVIDIA USD API](https://docs.omniverse.nvidia.com/kit/docs/pxr-usd-api/latest/pxr.html) for our Python wrappers around USD.
* See the [USD Glossary of Terms & Concepts](https://graphics.pixar.com/usd/release/glossary.html) for more details.
* See the [NVIDIA USD tutorials](https://developer.nvidia.com/usd/tutorials) for a step-by-step introduction to USD.