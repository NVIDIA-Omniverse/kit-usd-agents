# Examples

Isaac Sim provides a library of examples and demos that serves as a showcase of Isaac Sim capabilities and a learning resource for developing your own projects. Some are [Interactive Examples](#isaac-sim-app-intro-examples-interactive) that can be opened while the simulator is open, some are [Standalone Examples](#isaac-sim-app-intro-examples-standalone) for running Isaac Sim from the command line using the [Standalone workflow](Workflows.md).

- Interactive Examples Reference Table
- Standalone Examples Reference List

## Interactive Examples

Interactive examples can be found by going to the top Menu Bar and clicking **Window > Examples > Robotics Examples**. A browser should appear on the bottom left of the screen, in the same space as the **Content** browser. Click on the **Robotics Examples** tab to bring the browser into view.

The examples are organized into categories shown on the left hand panel. Click through the categories and subcateogries to see the examples inside each. Click on the example thumbnail in the main browser window to load the interactive GUI on the right hand panel of the browser. Expand the **Information** tab to reveal the summary and instructions for the example.

| Ref # | Function | Action |
| --- | --- | --- |
| 1 | Category Menu | Click on the category to see the included examples |
| 2 | Example | Click on the example to open the interactive GUI on the left hand panel, click again to refresh and reload the example |
| 3 | Information Window | Expand to see the summary and instructions for the example |
| 4 | Controls | Expand to see the buttons for running the example |
| 5 | Links | Quick Access to source code, source folder, and documentation |

## Standalone Examples

In addition to the interactive examples, Isaac Sim also provides standalone examples that can be run from the command line. These examples are located in the **<isaac\_sim\_root\_dir>/standalone\_examples** directory.

To run an example:

1. Navigate to your <isaac\_sim\_root\_dir>.
2. Run the example script using `./python.sh` for Linux or `python.bat` for Windows.

For example, to run the `hello_world` example, navigate to the `<isaac_sim_root_dir>` and run the following command:

Linux

```python
./python.sh standalone_examples/api/isaacsim.simulation_app/hello_world.py
```

Windows

```python
python.bat standalone_examples\api\isaacsim.simulation_app\hello_world.py
```

## Physics Examples

* [PhysX examples](https://docs.omniverse.nvidia.com/extensions/latest/ext_physics.html#explore-physics-demos)
* [Warp examples](https://nvidia.github.io/warp/)

---

# Interactive Examples Reference Table

| Menu Items | Action |
| --- | --- |
| Sensors | Examples showing different ways of sensing the environment. |
| * LIDAR | See [PhysX SDK Lidar Sensor Example](Sensors.md) docs for more details. |
| * Custom Pattern Range Sensor | See [PhysX SDK Generic Sensor Example](Sensors.md) docs for more details. |
| * IMU | See [IMU Sensor](Sensors.md) docs for more details. |
| * Contact | See [Contact Sensor](Sensors.md) docs for more details. |
| Input Devices | Examples using different HIDs. |
| * Kaya Gamepad | Connect to Gamepad using OmniGraph to control a Kaya robot. |
| * Omnigraph Keyboard | Connect to Keyboard using OmniGraph. |
| Manipulation | Examples showing different manipulation tools in Isaac Sim. |
| * Follow Target | Example showing a FrankaPanda robot arm following a target and avoid obstacles using RMPFlow controllers. |
| * Path Planning | An extension version of the standalone example in [Lula RRT](Robot_Simulation.md) utilizing a FrankaPanda arm. |
| * Bin Filling | Example showing UR10 filling bins using suction grippers. |
| * Replay Follow Target | Example of saving and replaying joint trajectories using a FrankaPanda arm. |
| * Surface Gripper | See the [Surface Gripper Extension](Robot_Simulation.md) docs for more details. |
| * Pick and Place | See the [Franka Pick and Place Example](../examples/manipulation_franka_pick_place.html#isaac-franka-pick-place) docs for more details. |
| * UR10 Follow Target | Example showing a UR10 robot arm following a target using damped least square, pseudoinverse, transpose, and single value decomposition (SVD) inverse kinematics solvers. |
| Multi-Robot | Examples showing heterogeneous robot scenes. |
| * Robo Party | A demonstration of running multiple different robots (Kaya, FrankaPanda, UR10, Jetbot) with the same controller. |
| * Robo Factory | A demonstration of running multiple FrankaPanda robots with different controllers. |
| General | Examples showing general use cases. |
| * Hello World | Base Sample that can be used as template extension. |
| Policy | Examples showing deploying learned policies on robots. |
| * Quadruped | See the [Reinforcement Learning Policies Examples in Isaac Sim](Robot_Simulation.md) for more details. Uses the Bostondynamics Spot. |
| * Humanoid | See the [Reinforcement Learning Policies Examples in Isaac Sim](Robot_Simulation.md) for more details. Uses the Unitree H1. |
| * Franka | See the [Reinforcement Learning Policies Examples in Isaac Sim](Robot_Simulation.md) for more details. Uses the Franka Panda robot. |
| Cortex | Examples of Cortex |
| * UR10 Palletizing | See the [Randomization in Simulation – UR10 Palletizing](Synthetic_Data_Generation.md) docs for more details. |
| * Franka Cortex Examples | See the [Walkthrough: Franka Block Stacking](Digital_Twin.md) docs for more details. |
| Import Robots | Examples showing different ways of importing robots and assemblies. |
| * Carter URDF | Example of Carter robot imported from URDF. |
| * Franka URDF | Example of FrankaPanda robot imported from URDF. |
| * Kaya URDF | Example of Kaya robot imported from URDF. |
| * UR10 URDF | Example of UR10 robot imported from URDF. |
| Tutorials | Examples following the Quickstart Series |
| * Part I: Basics | Following the steps found in [Isaac Sim Basic Usage Tutorial](Quick_Tutorials.md). |
| * Part II: Robot | Following the steps found in [Basic Robot Tutorial](Quick_Tutorials.md). |

---

# Standalone Examples Reference List

This page provides a list of [standalone examples](Examples.md) available in the Isaac Sim.

These examples are designed to demonstrate various features and functionalities of the platform, allowing users to explore and control Isaac Sim through Python.

* [Standalone Examples](Examples.md)