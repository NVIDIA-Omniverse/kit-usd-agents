# Robot Simulation

The Robot Simulation section provides information on tools that you will need to move a robot. The lowest level of control is joint control. For the next level up, we separated the controllers by the robot types, for they represent the three types of controllers we provide in Isaac Sim:

* **Wheeled Robots**: use controllers that are based on universal formulas and require very few robot-specific parameters as inputs.
* **Manipulators**: use controllers that are based on complex optimization, therefore the same robot performing the same task could use many variety of controllers, each with a different optimization method. They often require the robot models in the optimization process.
* **Policy Controlled Robots**: uses controllers that are trained using reinforcement learning. They also has a much looser definition “controllers”, for they can have task and path planners embedded as well.

## Joint Level Control

- Articulation Controller

## Wheeled Robots

- Mobile Robot Controllers

## Manipulators

- Motion Generation
- Surface Gripper Extension
- Grasp Editor

## Policy Controlled Robots

- Reinforcement Learning Policies Examples in Isaac Sim

### Tips and Deep Dives

- Robot Simulation Tips
- Useful Links

---

# Articulation Controller

## Overview

Articulation controller is the low level controller that controls joint position, joint velocity, and joint effort in Isaac Sim. The articulation controller can be interfaced using Python and Omnigraph.

Note

Angular units are expressed in radians while angles in USD are expressed in degrees and will be adjusted accordingly by the articulation controller.

## Python Interface

### Create the articulation controller

There are several ways to create the articulation controller. The articulation controller is usually created implicitly by applying articulation on a robot prim through the `SingleArticulation` class.
However, the articulation controller can be created directly by importing the controller class before the simulation starts, but this approach will require you to create or pass in the `Articulation` during initialization.

Single Articulation

> The snippet below will load and apply articulation on a franka robot.
>
> ```python
> import isaacsim.core.utils.stage as stage_utils
> from isaacsim.core.prims import SingleArticulation
>
> usd_path = "/Path/To/Robots/FrankaRobotics/FrankaPanda/franka.usd"
> prim_path = "/World/envs/env_0/panda"
>
> # load the Franka Panda robot USD file
> stage_utils.add_reference_to_stage(usd_path, prim_path)
> # wrap the prim as an articulation
> prim = SingleArticulation(prim_path=prim_path, name="franka_panda")
> ```

Articulation Controller

> ```python
> import isaacsim.core.utils.stage as stage_utils
> from isaacsim.core.api.controllers.articulation_controller import ArticulationController
>
> usd_path = "/Path/To/Robots/FrankaRobotics/FrankaPanda/franka.usd"
> prim_path = "/World/envs/env_0/panda"
>
> # load the Franka Panda robot USD file
> stage_utils.add_reference_to_stage(usd_path, prim_path)
> # Create the articulation controller
> articulation_controller = ArticulationController()
> ```

### Initialize the controller

After the simulation is started, the robot articulation must be initialized before any commands can be passed to the robot.

Single Articulation

> The more common approach is by initializing the single articulation object that you have created earlier, this will initialize the articulation controller and articulation view stored in the SingleArticulation object
>
> ```python
> prim.initialize()
> ```

Articulation Controller

> After the simulation starts, the articulation controller must be initialzied with an articulation view. Articulation view is the backend for selecting the joints and applying joint actions.
>
> For example, the code snippet below creates an articulation view with the Franka robot and initializes the articulation controller.
>
> ```python
> prim.initialize()
> ```

### Articulation Action

Joint controls commands are packaged in `ArticulationAction` objects first, before sending them to the articulation controller. The articulation controller allows you to specify the command joint postion, velocity and effort, as well as joint indicies of the joints actuated.

If the joint indice is empty, the articulation action will assume the command will apply to all joints of the robot, and if any of the command is 0, articulation action will assume it is unactuated.

For example, the snippet below creates the command that closes the franka robot fingers: panda\_finger\_joint1 (7) and panda\_finger\_joint2 (8) to 0.0

```python
import numpy as np
from isaacsim.core.utils.types import ArticulationAction

action = ArticulationAction(joint_positions=np.array([0.0, 0.0]), joint_indices=np.array([7, 8]))
```

This snippet creates the command that moves all the robot joints to the indicated position

```python
import numpy as np
from isaacsim.core.utils.types import ArticulationAction

action = ArticulationAction(joint_positions=np.array([0.0, 0.0]), joint_indices=np.array([7, 8]))
```

Important

Make sure the joint commands matches the order and the number of joint indices passed in to the articulation action. If joint indice is not passed in, make sure the command matches the number of joints in the robot.

Note

A joint can only be controlled by one control method. For example a joint cannot be controlled by both desired position and desired torque

### Apply Action

The `apply_action` function in both `SingleArticulation` and `ArticulationController` classes will apply the `ArticulationAction` you created earlier to the robot.

Single Articulation

> ```python
> prim.apply_action(action)
> ```

Articulation Controller

> ```python
> prim.apply_action(action)
> ```

### Script Editor Example

You can try out basic articulation controller examples by running the following code snippets in the Script Editor. For more advanced usage, it is recommended to follow the [Core API Tutorial Series](Python_Scripting_and_Tutorials.md).

Single Articulation

> ```python
> import asyncio
>
> import numpy as np
> from isaacsim.core.api.world import World
> from isaacsim.core.prims import SingleArticulation
> from isaacsim.core.utils.stage import add_reference_to_stage
> from isaacsim.core.utils.types import ArticulationAction
> from isaacsim.storage.native import get_assets_root_path
>
>
> async def robot_control_example():
>     if World.instance():
>         World.instance().clear_instance()
>     world = World()
>     await world.initialize_simulation_context_async()
>     world.scene.add_default_ground_plane()
>
>     # Load the robot USD file
>     usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
>     prim_path = "/World/envs/env_0/panda"
>     add_reference_to_stage(usd_path, prim_path)
>
>     # Create SingleArticulation wrapper (automatically creates articulation controller)
>     robot = SingleArticulation(prim_path=prim_path, name="franka_panda")
>     await world.reset_async()
>
>     # Initialize the robot (initializes articulation controller internally)
>     robot.initialize()
>
>     # Run simulation
>     await world.play_async()
>
>     # Get current joint positions
>     current_positions = robot.get_joint_positions()
>     print(f"Current joint positions: {current_positions}")
>
>     # Create target positions
>     target_positions = np.array([0.0, -1.5, 0.0, -2.8, 0.0, 2.8, 1.2, 0.04, 0.04])
>
>     # Create and apply articulation action
>     action = ArticulationAction(joint_positions=target_positions)
>     robot.apply_action(action)
>
>     await asyncio.sleep(5.0)  # Run for 5 seconds to reach target positions
>
>     # Get current joint positions
>     current_positions = robot.get_joint_positions()
>     print(f"Current joint positions: {current_positions}")
>
>     world.pause()
>
>
> # Run the example
> asyncio.ensure_future(robot_control_example())
> ```

Articulation Controller

> ```python
> import asyncio
>
> import numpy as np
> from isaacsim.core.api.controllers.articulation_controller import ArticulationController
> from isaacsim.core.api.world import World
> from isaacsim.core.prims import Articulation
> from isaacsim.core.utils.stage import add_reference_to_stage
> from isaacsim.core.utils.types import ArticulationAction
> from isaacsim.storage.native import get_assets_root_path
>
>
> async def robot_control_example():
>     if World.instance():
>         World.instance().clear_instance()
>     world = World()
>     await world.initialize_simulation_context_async()
>     world.scene.add_default_ground_plane()
>
>     # Load the robot USD file
>     usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
>     prim_path = "/World/envs/env_0/panda"
>     add_reference_to_stage(usd_path, prim_path)
>
>     # Create Articulation view for the robot
>     robot_view = Articulation(prim_paths_expr=prim_path, name="franka_panda_view")
>
>     # Create and initialize the articulation controller with the articulation view
>     articulation_controller = ArticulationController()
>     articulation_controller.initialize(robot_view)
>
>     # Run simulation
>     await world.play_async()
>
>     # Get current joint positions
>     current_positions = robot_view.get_joint_positions()
>     print(f"Current joint positions: {current_positions}")
>
>     # Create target positions
>     target_positions = np.array([0.0, -1.5, 0.0, -2.8, 0.0, 2.8, 1.2, 0.04, 0.04])
>
>     # Create and apply articulation action
>     action = ArticulationAction(joint_positions=target_positions)
>     articulation_controller.apply_action(action)
>
>     await asyncio.sleep(5.0)  # Run for 5 seconds to reach target positions
>
>     # Get current joint positions
>     current_positions = robot_view.get_joint_positions()
>     print(f"Current joint positions: {current_positions}")
>
>     world.pause()
>
>
> # Run the example
> asyncio.ensure_future(robot_control_example())
> ```

## Omnigraph Interface

The articulation controller can also be accessed through Omnigraph nodes, providing a visual, node-based approach to robot control.

### Input Parameters

The articulation controller Omnigraph node accepts the following input parameters:

Articulation Controller Omnigraph Inputs

| Input Parameter | Description |
| --- | --- |
| **execIn** | Input execution trigger - connects to other nodes to control when the articulation controller runs |
| **targetPrim** | The prim containing the robot articulation root. Leave empty if using robotPath |
| **robotPath** | String path to the robot articulation root. Leave empty if using targetPrim |
| **jointIndices** | Array of joint indices to control. Leave empty to control all joints or use jointNames |
| **jointNames** | Array of joint names to control. Leave empty to control all joints or use jointIndices |
| **positionCommand** | Desired joint positions. Leave empty if not using position control |
| **velocityCommand** | Desired joint velocities. Leave empty if not using velocity control |
| **effortCommand** | Desired joint efforts/torques. Leave empty if not using effort control |

### Usage Guidelines

Important

**Parameter Validation**: Ensure joint commands match the order and number of joint indices or joint names. If neither joint indices nor joint names are specified, the command must match the total number of joints in the robot.

Note

**Control Method Limitation**: A joint can only be controlled by one method at a time. For example, a joint cannot be controlled by both position and effort commands simultaneously.

### Example Usage

For a complete example of the articulation controller Omnigraph node in action, see the `mock_robot_rigged` asset in the Content Browser at **Isaac Sim > Samples > Rigging > MockRobot > mock\_robot\_rigged.usd**.

---

# Mobile Robot Controllers

## Differential controller

The differential controller uses the speed differential between the left and right wheels to control the robot’s linear and angular velocity. The differential robot enables the robot to turn in place and is used in the NVIDIA Nova Carter robot.

### The Math

\[ \begin{align}\begin{aligned}\omega\_R &= \frac{1}{2r}(2V + \omega l\_{tw})\\\omega\_L &= \frac{1}{2r}(2V - \omega l\_{tw})\end{aligned}\end{align} \]

where \(\omega\) is the desired angular velocity, \(V\) is the desired linear velocity, \(r\) is the radius of the wheels, and \(l\_{tw}\) is the distance between them.
\(\omega\_R\) is the desired right wheel angular velocity and \(\omega\_L\) is the desired left wheel angular velocity.

### OmniGraph Node

Differential Controller Omnigraph Inputs

| Input Commands | description |
| --- | --- |
| execIn | Input execution |
| wheelRadius | Radius of the wheels in meters |
| wheelDistance | Distance between the wheels in meters |
| dt | Delta time in seconds |
| maxAcceleration | Max linear acceleration for moving forward and reverse in m/s^2, 0.0 means not set |
| maxDeceleration | Max linear breaking of the robot in m/s^2, 0.0 means not set |
| maxAngularAcceleration | Max angular acceleration of the robot in rad/s^2, 0.0 means not set |
| maxLinearSpeed | Max linear speed allowed for the robot in m/s, 0.0 means not set |
| maxAngularSpeed | Max angular speed allowed for the robot in rad/s, 0.0 means not set |
| maxWheelSpeed | Max wheel speed in rad/s |
| Desired Linear Velocity | Desired linear velocity in m/s |
| Desired Angular Velocity | Desired angular velocity in rad/s |

Differential Controller Omnigraph Outputs

| Output Commands | description |
| --- | --- |
| VelocityCommand | Velocity command for the left and right wheel in m/s and rad/s |

### Python

The code snippet below setups the differential controller for a NVIDIA Jetbot with a wheel radius of 3 cm and a base of 11.25cm, with a linear speed of 0.3m/s and angular speed of 1.0rad/s.

```python
import asyncio

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path

async def differential_controller_example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()

    world.scene.add_default_ground_plane()
    assets_root_path = get_assets_root_path()
    jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    my_jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="my_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=jetbot_asset_path,
            position=np.array([-1.5, -1.5, 0]),
        )
    )
    await world.reset_async()
    await world.play_async()

    wheel_radius = 0.03
    wheel_base = 0.1125
    controller = DifferentialController("test_controller", wheel_radius, wheel_base)
    linear_speed = 0.3
    angular_speed = 1.0

    command = [linear_speed, angular_speed]
    my_jetbot.apply_wheel_actions(controller.forward(command))

    await asyncio.sleep(5.0)  # Run for 5 seconds
    world.pause()

# Run the example
asyncio.ensure_future(differential_controller_example())
```

## Holonomic Controller

The holonomic controller computes the joint drive commands required on omni-directional robots to produce the commanded forward, lateral, and yaw speeds of the robot. An example of a holonomic robot is the NVIDIA Kaya robot.
The problem is framed as a quadratic program to minimize the residual “net force” acting on the center of mass.

Note

The wheel joints of the robot prim must have additional attributes to definine the roller angles and radii of the mecanum wheels.

```python
stage = omni.usd.get_context().get_stage()
joint_prim = stage.GetPrimAtPath("/path/to/wheel_joint")
joint_prim.CreateAttribute("isaacmecanumwheel:radius", Sdf.ValueTypeNames.Float).Set(0.12)
joint_prim.CreateAttribute("isaacmecanumwheel:angle", Sdf.ValueTypeNames.Float).Set(10.3)
```

The `HolonomicRobotUsdSetup` class automates this process.

### The Math

The cost funciton is defined as the control input to the robot joints. By minimizing the control inputs, excess acceleration and be reduced.

\[J = min(X^T \cdot X)\]

The equality constrains are set by the linear and angular target velocity Inputs:

\[ \begin{align}\begin{aligned}v\_{input} &= V^T \cdot X\\w\_{input} &= (V \times D\_{wheel dist to COM}) \cdot X\end{aligned}\end{align} \]

### OmniGraph Node

Holonomic Controller Omnigraph Inputs

| Input Commands | description |
| --- | --- |
| execIn | Input execution |
| wheelRadius | Array of wheel radius in meters |
| wheelPositions | Position of the wheel with respect to chassis’ center of mass in meters |
| wheelOrientations | Orientation of the wheel with respect to chassis’ center of mass frame |
| mecanumAngles | Angles of the mecanum wheels with respect to wheel’s rotation axis in radians |
| wheelAxis | The rotation axis of the wheels |
| upAxis | The up axis (default to z axis) |
| Velocity Commands for the vehicle | Velocity in x and y (m/s) and rotation (rad/s) |
| maxLinearSpeed | Maximum speed allowed for the vehicle in m/s |
| maxAngularSpeed | Maximum angular rotation speed allowed for the vehicles in rad/s |
| maxWheelSpeed | Maximum rotation speed allowed for the wheel joints in rad/s |
| linearGain | Gain for the linear velocity input |
| angularGain | Gain for the angular input |

Holonomic Controller Omnigraph Outputs

| Output Commands | description |
| --- | --- |
| jointVelocityCommand | Velocity command for the wheel joints in rad/s |

### Python

The code snippet below computes the joint velocity output for a three wheeled NVIDIA Kaya holonomic robot with command velocity of [1.0, 1.0, 0.1]

```python
import asyncio

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path

async def differential_controller_example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()

    world.scene.add_default_ground_plane()
    assets_root_path = get_assets_root_path()
    jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    my_jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="my_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=jetbot_asset_path,
            position=np.array([-1.5, -1.5, 0]),
        )
    )
    await world.reset_async()
    await world.play_async()

    wheel_radius = 0.03
    wheel_base = 0.1125
    controller = DifferentialController("test_controller", wheel_radius, wheel_base)
    linear_speed = 0.3
    angular_speed = 1.0

    command = [linear_speed, angular_speed]
    my_jetbot.apply_wheel_actions(controller.forward(command))

    await asyncio.sleep(5.0)  # Run for 5 seconds
    world.pause()

# Run the example
asyncio.ensure_future(differential_controller_example())
```

## Ackermann Controller

The ackeramnn controller is commonly used for robots with steerable wheels, an example of steerable robot is the NVIDIA leatherback robot.
The ackeramnn controller in Isaac Sim will assume the desired steering angle and linear velocity is provided, and based on the robot geometry

### The Math

Compute the steering angle offset between the left and right steering wheels:

\[ \begin{align}\begin{aligned}R\_{icr} &= \frac{l\_{wb}}{tan(\theta\_{steer})}\\\theta\_L &= \arctan[\frac{l\_{wb}}{R\_{icr} - 0.5 \* l\_{tw}}]\\\theta\_R &= \arctan[\frac{l\_{wb}}{R\_{icr} + 0.5 \* l\_{tw}}]\end{aligned}\end{align} \]

where \(R\_{icr}\) is the radius to the instantaneous center of rotation, \(\theta\_{steer}\) is the desired steering angle, \(l\_{wb}\) is the distance between rear and front axles (wheel base), \(l\_{tw}\) is the track width

Compute the individual wheel velocities (Forward steering case):

First step is to find the distance between the wheels and the instantaneous center of rotation.

\[ \begin{align}\begin{aligned}D\_{front} &= \sqrt{ (R\_{icr} \pm 0.5 l\_{tw})^2 + (l\_{wb})^2 }\\D\_{rear} &= R\_{icr} \pm 0.5 l\_{tw}\end{aligned}\end{align} \]

Note

for \(\pm\), use \(-\) for the wheel closer to the \(R\_{icr}\) and \(+\) for the wheel further to the \(R\_{icr}\)

Then desired wheel velocity can be computed

\[ \begin{align}\begin{aligned}\omega\_{front} &= \frac{V\_{desired}}{R\_{icr}} \cdot \frac{D\_{front}}{r\_{front}}\\\omega\_{rear} &= \frac{V\_{desired}}{R\_{icr}} \cdot \frac{D\_{rear}}{r\_{rear}}\end{aligned}\end{align} \]

Where \(V\_{desired}\) is the desired linear velocity, \(r\_{front}\) is the desired front wheel radius, and \(r\_{rear}\) is the desired rear wheel radius.

### OmniGraph Node

Ackermann Controller Omnigraph Inputs

| Input Commands | description |
| --- | --- |
| execIn | Input execution |
| acceleration | Desired forward acceleration for the robot in m/s^2 |
| speed | Desired forward speed in m/s |
| steeringAngle | Desired steering angle in radians, by default it is positive for turning left for a front wheel drive |
| currentLinearVelocity | Current linear velocity of the robot in m/s |
| wheelBase | Distance between the front and rear axles of the robot in meters |
| trackWidth | Distance between the left and right rear wheels of the robot in meters |
| turningWheelRadius | Radius of the front wheels of the robot in meters |
| maxWheelVelocity | Maximum angular velocity of the robot wheel in rad/s |
| invertSteeringAngle | Flips the sign of the steering angle, Set to true for rear wheel steering |
| useAcceleration | Use acceleration as an input, Set to false to use speed as input instead |
| maxWheelRotation | Maximum angle of rotation for the front wheels in radians |
| dt | Delta time for the simulation step |

Ackermann Controller Omnigraph Outputs

| Output Commands | description |
| --- | --- |
| execOut | Output execution |
| leftWheelAngle | Angle for the left turning wheel in radians |
| rightWheelAngle | Angle for the right turning wheel in radians |
| wheelRotationVelocity | Angular velocity for the turning wheels in rad/s |

### Python

The python snippet below creates an ackerrmann controller for a NVIDIA Leatherback robot with a wheel base of 1.65m, trackwidth of 1.25m, and wheel radius of 0.25m, sending it a desired forward velocity of 1.1 rad/s and steering angle of 0.1rad.

```python
import asyncio

import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.storage.native import get_assets_root_path

async def differential_controller_example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()

    world.scene.add_default_ground_plane()
    assets_root_path = get_assets_root_path()
    jetbot_asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
    my_jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="my_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=jetbot_asset_path,
            position=np.array([-1.5, -1.5, 0]),
        )
    )
    await world.reset_async()
    await world.play_async()

    wheel_radius = 0.03
    wheel_base = 0.1125
    controller = DifferentialController("test_controller", wheel_radius, wheel_base)
    linear_speed = 0.3
    angular_speed = 1.0

    command = [linear_speed, angular_speed]
    my_jetbot.apply_wheel_actions(controller.forward(command))

    await asyncio.sleep(5.0)  # Run for 5 seconds
    world.pause()

# Run the example
asyncio.ensure_future(differential_controller_example())
```

---

# Motion Generation

Lula is a high-performance motion generation library for robotic manipulation. RMPflow provides
real-time, reactive local policies to guide a robot manipulator to a task space target while
avoiding dynamic obstacles. A suite of Rapidly-exploring Random Tree (RRT) algorithms,
including RRT-Connect and JT-RRT, deliver global planning solutions in static environments.
Additionally, the trajectory generation tools in Lula provide time-optimal trajectories for
paths described as a series of c-space and task-space moves. Finally, Lula provides interfaces
to the performant forward and inverse kinematic solvers underpinning the higher-level motion
generation tools.

NVIDIA Isaac Sim also interfaces with [cuRobo](https://curobo.org), a high-performance, GPU-accelerated robotics motion
generation library that adds additional features to NVIDIA Isaac Sim such as batched collision-free inverse kinematics,
collision-free motion planning, and reactive control in the presence of obstacles represented as meshes or Nvblox maps.
For more information, see the [cuRobo tutorial](Robot_Simulation.md).

- Motion Generation
- Lula Robot Description and XRDF Editor
- Lula RMPflow
- Lula RRT
- Lula Kinematics Solver
- Lula Trajectory Generator
- Configuring RMPflow for a New Manipulator
- cuRobo and cuMotion

## Examples

**Interactive Examples**

To locate the interactive examples, go to **Windows** > **Examples** > **Robotics Examples** and open the **Robotics Examples** tab if it’s not already. Select one of the following examples from the browser, read the **Information** tab on the right hand side of the browser window for instructions on how to run it.

* Follow Target Example: **Manipulation > Follow Target**
* RoboFactory Example: **Multi-Robot > RoboFactory**
* RoboParty Example: **Multi-Robot > RoboParty**

Note

Pressing **STOP**, then **PLAY** in this workflow might not reset the world properly. Use
the **RESET** button instead.

**Standalone Examples**

To run a standalone example, navigate to your `<isaac_sim_root_dir>`, then use `./python.sh` for Linux or `python.bat` for Windows to run the example scripts listed here.

* Follow Target with RMPflow: `standalone_examples/api/isaacsim.robot.manipulators/franka/follow_target_with_rmpflow.py`
* Follow Target with IK: `standalone_examples/api/isaacsim.robot.manipulators/franka/follow_target_with_ik.py`

---

# Motion Generation

The [Motion Generation](#isaac-sim-motion-generation) provides an API that you can use to control objects within Isaac Sim.
The API is made up of abstract interfaces for adding motion control algorithms to Isaac Sim.
The interfaces in the [Motion Generation](#isaac-sim-motion-generation) provide two basic utilities:

> * Simplify the integration of new robotics algorithms into NVIDIA Isaac Sim.
> * Provide a standard structure with which to compare similar robotics algorithms.

For example, if you have a robot that has not previously been described to Isaac Sim, you can use these APIs to define that robot and how it moves.

> * Simplify the integration of new robotics algorithms into NVIDIA Isaac Sim.
> * Provide a standard structure with which to compare similar robotics algorithms.

For example, if you have a robot that has not previously been described to Isaac Sim, you can use these APIs to define that robot and how it moves.

Three interfaces are provided in the Motion Generation Extension:

* [Motion Policy Algorithm](Robot_Simulation.md)
* [Path Planner](Robot_Simulation.md)
* [Kinematics Solvers](Robot_Simulation.md)

In Isaac Sim, the robot is specified using a USD file that gets added to the stage. However, we expect that robotics algorithms will have their
own way of specifying the robot’s kinematic structure and custom parameters. To avoid interfering with any particular robot description format, the interfaces
in the Motion Generation Extension include functions that facilitate the translation between the USD robot and a specific algorithm. Specifically,
an algorithm can specify which joints in the robot it cares about, and the order in which it expects those joints to be listed. The helper classes provided in this extension,
[Articulation Motion Policy](Robot_Simulation.md), [Path Planner Visualizer](Robot_Simulation.md), and [Articulation Kinematics Solver](Robot_Simulation.md), use the interface
functions to appropriately map robot joint states between the USD robot articulation and an interface implementation.

In Isaac Sim, we use the word [Articulation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html "(in Omni Physics)") to refer to the simulated robot represented through USD.
The word “Articulation” is used as a prefix in the
Motion Generation Extension to indicate utility classes that handle interfacing an algorithm with the simulated robot.

In addition, the **Motion Generation extension** includes a handful of special-purpose
controllers that do not leverage MotionPolicy or PathPlanner.

- Motion Generation Extension API Documentation
- Kinematics Solvers
- Trajectory Generation
- Path Planner Algorithm
- Lula RRT
- Motion Policy Algorithm
- RMPflow
- RMPflow Tuning Guide

## References

---

# Motion Generation Extension API Documentation

See the Isaac Sim Motion Generation Extension [API Documentation](../../py/source/extensions/isaacsim.robot_motion.motion_generation/docs/index.html) for usage information. This API content is part of the broader Omniverse API documentation.

---

# Kinematics Solvers

Like a [Motion Policy Algorithm](Robot_Simulation.md), a [Kinematics Solvers](#isaac-sim-kinematics-solver) is an interface class with a single provided implementation. A KinematicsSolver
is able to compute forward and inverse kinematics. A single implementation is provided using the NVIDIA-developed **Lula** library. (see [Lula Kinematics Solver](#isaac-sim-lula-kinematics-solver))

includes:

* Kinematics Solver
* Articulation Kinematics Solver
* Lula Kinematics Solver

## Kinematics Solver

The KinematicsSolver interface specifies functions for computing both forward and inverse kinematics at any available frame in the robot. Like a [Motion Policy Algorithm](Robot_Simulation.md),
an instance of the KinematicsSolver class is not expected to use the same USD robot representation as NVIDIA Isaac Sim. A KinematicsSolver can have its own internal
representation of the robot, and there are necessary interface functions for performing the mapping between the internal robot representation and the robot
Articulation.

### Joint Names

An instance of the KinematicsSolver class must fulfill a function KinematicsSolver.get\_joint\_names() that specifies the joints of interest to the solver, and the order in which it
expects them. Think of a robot arm mounted on a moving base. A KinematicsSolver can use only the URDF for the robot arm without knowing about the robot base. In this case, many of
the joints in the robot Articulation would not be recognized by the KinematicsSolver.

When computing forward kinematics, the joint positions that are passed to the solver must correspond to the output of KinematicsSolver.get\_joint\_names(). Likewise, the output of
inverse kinematics will have the same shape as KinematicsSolver.get\_joint\_names(). A mapping layer between the robot Articulation and the KinematicsSolver is provided in the
[Articulation Kinematics Solver](#isaac-sim-articulation-kinematics-solver) class.

### Frame Names

An instance of the KinematicsSolver class must fulfill a function KinematicsSolver.get\_all\_frame\_names() to provide a list of frames in the robot’s kinematics chain that can have their positions
referenced by name when solving either forward or inverse kinematics. The frame names returned by a KinematicsSolver do not have to match the frames present in the robot Articulation. Like joint names,
the frame names come from the individual solver’s config file structure.

### Robot Base Pose

As with a [Motion Policy Algorithm](Robot_Simulation.md), a the KinematicsSolver interface includes a function set\_robot\_base\_pose() that allows the caller to specify the location of the robot base. If this function has been called,
the KinematicsSolver must apply appropriate transformations when computing forward and inverse kinematics.
A KinematicsSolver operates in world coordinates. The solution to the forward kinematics will be translated and rotated according to the robot base pose to return the position of the end effector relative to the world frame,
and the input to the inverse kinematics will be provided in the world coordinates and transformed so that it is relative to the robot base frame. If you prefers that the solver inputs are relative to the robot base frame,
they can simply set the robot base pose to the origin.

### Collision Awareness

Implementations of the KinematicsSolver class do not need to be collision aware with external objects, but they have the option. A function KinematicsSolver.supports\_collision\_avoidance() -> bool must be implemented
to indicate whether a particular KinematicsSolver supports collision avoidance. If a KinematicsSolver supports collision avoidance, it can fulfill the same set of world functions as a MotionPolicy ([Inputs: World State](Robot_Simulation.md)).
If a solver is collision aware, it is especially important to specify the robot base pose correctly, as the positions of objects can only be queried relative to the world frame, and it is up to the solver to compute the positions of obstacles
relative to the robot.

## Articulation Kinematics Solver

The ArticulationKinematicsSolver class exists to handle the mapping between the robot Articulation and an implementation of a [Kinematics Solvers](#isaac-sim-kinematics-solver).

### Forward Kinematics

ArticulationKinematicsSolver wraps the forward kinematics function of a KinematicsSolver to query the joint positions of the robot Articulation and pass the appropriate joint positions to the KinematicsSolver in the order
specified by KinematicsSolver.get\_joint\_names(). This allows the current position of the simulated robot end effector to be queried easily.

### Inverse Kinematics

ArticulationKinematicsSolver wraps the inverse kinematics to return the resulting joint positions as an ArticulationAction that can be directly applied to the robot Articulation.
The current robot Articulation joint positions at the time this method is called are automatically used as a warm start in the IK calculation.

## Lula Kinematics Solver

The LulaKinematicsSolver implements the [Kinematics Solvers](#isaac-sim-kinematics-solver) interface. The solver does not support collision avoidance with objects in the world. In addition to the functions in the
KinematicsSolver interface, the LulaKinematicsSolver includes getters and setters for changing internal settings such as LulaKinematicsSolver.set\_max\_iterations() to set the maximum number
of iterations before the IK computation returns a failure.

### Lula Kinematics Solver Configuration

Two files are necessary to configure Lula Kinematics for use with a new robot:

> 1. A URDF (universal robot description file), used for specifying robot kinematics as well as joint and link names. Position limits for each joint are also required. Other properties in the URDF are ignored and can be omitted; these include masses, moments of inertia, visual and collision meshes.
> 2. A supplemental robot description file in YAML format. In addition to enumerating the list of actuated joints that define the configuration space (c-space) for the robot, this file also includes sections for specifying the default c-space configuration. This file can also be used to specify fixed positions for unactuated joints.

---

# Trajectory Generation

In the Motion Generation extension, a workflow is provided for defining c-space and task-space trajectories. An interface is provided for a [Trajectory Interface](#isaac-sim-trajectory) class:

* Trajectory Interface
* Articulation Trajectory
* Lula Trajectory Generator

## Trajectory Interface

An interface is provided in the Motion Generation extension for defining a robot trajectory.
An instance of the Trajectory interface must return robot c-space position as a continuous function of time within a specified time horizon. A Trajectory has four basic accessors:

* **start\_time**: The earliest time at which this Trajectory will return a robot c-space position.
* **end\_time**: The latest time at which this Trajectory will return a robot c-space position.
* **active\_joints**: The names of the joints that this Trajectory is intended to control corresponding to the order the joint targets are returned.
* **joint\_targets(time)**: Joint position/velocity targets as a function of time between start\_time and end\_time.

An instance of the Trajectory class can be used to directly control a robot by using it to initialize an [Articulation Trajectory](#isaac-sim-articulation-trajectory).

## Articulation Trajectory

The ArticulationTrajectory class is initialized using a robot Articulation and an instance of the Trajectory class.
This class handles the mapping from a defined Trajectory to controlling a simulated robot Articulation. The ArticulationTrajectory class has two main functions:

* **get\_action\_at\_time(time)**: Return an ArticulationAction at a time that is within the time horizon of the provided Trajectory object.
* **get\_action\_sequence(timestep)**: Return a list of ArticulationAction that interpolates between the provided Trajectory start\_time and end\_time by the specified timestep. This is a convenience method for when the timestep of the physics simulator is known to be fixed.

As a Trajectory only defines the robot behavior within the provided time horizon, it is necessary to bring the robot Articulation to the initial state of the Trajectory before attempting to follow a sequence of generated ArticultionAction.

## Lula Trajectory Generator

We provide a **Lula** implementation of a trajectory generator that can generate a Trajectory given c-space or task-space waypoints. Two classes are provided:

* LulaCSpaceTrajectoryGenerator
* LulaTaskSpaceTrajectoryGenerator

Both classes share the same required configuration information.

To configure Lula Trajectory Generators for a specific robot you must have the following files:

> * A URDF (universal robot description file), used for specifying robot kinematics as well as joint and link names. Position limits for each joint are also required. Other properties in the URDF are ignored and can be omitted; these include masses, moments of inertia, visual and collision meshes.
> * A supplemental robot description file in YAML format. In addition to enumerating the list of actuated joints that define the configuration space (c-space) for the robot, this file also includes sections for specifying the default c-space configuration, acceleration limits, or jerk limits. This file can also be used to specify fixed positions for unactuated joints.

### Lula C-Space Trajectory Generator

The `LulaCSpaceTrajectoryGenerator` class takes in a series of c-space waypoints that correspond to the c-space coordinates listed in the required robot description YAML file.
The generator will use spline-based interpolation to connect the waypoints with an initial and final velocity of 0.
The trajectory is time-optimal – that is, either the velocity, acceleration, or jerk limits are saturated at any given time to produce a trajectory with as short a duration as possible.
The generator will return an instance of the Trajectory interface.

### Lula Task-Space Trajectory Generator

The `LulaTaskSpaceTrajectoryGenerator` class takes in a sequence of task-space targets and an end effector frame name (which must be a valid frame in the provided URDF file), and it returns an instance of the Trajectory interface if possible.

Task-space trajectories can be defined as a series of position and orientation targets. In this case, the generated trajectory will linearly interpolate in task-space between the provided targets.

Task-space trajectories can also be defined using the `lula.TaskSpacePathSpec` class, which provides a set of useful primitives to connect task-space waypoints such as creating an arc, pure rotation, pure translation.

Internally, a task-space trajectory is converted to a c-space trajectory using the
[Lula Kinematics Solver](Robot_Simulation.md), and is then passed through the `LulaCSpaceTrajectoryGenerator`. For this reason, the `LulaTaskSpaceTrajectoryGenerator` class shares the same set of parameters as the `LulaCSpaceTrajectoryGenerator` class, with added parameters that affect how the task-space trajectory is converted to c-space.

---

# Path Planner Algorithm

A [Path Planner](#isaac-sim-path-planner) is an algorithm that outputs a series of configuration space waypoints, which
when linearly interpolated, produce a collision-free path from a starting c-space pose to a c-space or task-space target pose.
The PathPlanner class provides an interface that specifies the necessary functions that must be fulfilled to specify a path planning algorithm that can interface with NVIDIA Isaac Sim.

An implementation is provided using the NVIDIA-developed **Lula** library (see [Lula RRT](Robot_Simulation.md)).

## Path Planner

The PathPlanner interface specifies functions for computing a series of configuration space waypoints, which
when linearly interpolated, produce a collision-free path from a starting c-space pose to a c-space or task-space target pose.
A PathPlanner uses the same set of functions to interface with the USD world as a [Motion Policy Algorithm](Robot_Simulation.md).
Like a [Motion Policy Algorithm](Robot_Simulation.md),
an instance of the PathPlanner class is not expected to use the same USD robot representation as NVIDIA Isaac Sim. A PathPlanner can have its own internal
representation of the robot, and there are necessary interface functions for performing the mapping between the internal robot representation and the robot
Articulation.

### Active and Watched Joints

The robot Articulation in Isaac Sim comes from a loaded USD file. This robot specification is not expected to perfectly match the specification used internally by a PathPlanner.

To perform the appropriate mapping, a PathPlanner has two functions it must fulfill:

* `PathPlanner.get_active_joints()`: joints that the PathPlanner is going to directly control to achieve the desired end effector target.
* `PathPlanner.get_watched_joints()`: joints that the PathPlanner observes to plan motions, but will not actively control. These are assumed to remain constant when generating a path.

Both functions return a list of joint names in the order that the PathPlanner expects to receive them.

For example, the Franka robot has nine degrees of freedom (DOFs):

* seven revolute joints for controlling the arm
* two prismatic joints for controlling its gripper

The robot Articulation exposes all nine degrees of
freedom, but [Lula RRT](Robot_Simulation.md) only cares about the seven revolute joints when navigating the robot to a position target. It is not appropriate for RRT to take
control of the gripper DOFs, because those DOFs can be controlled separately when performing a task such as pick-and-place. `RRT.get_active_joints()` returns the names of the seven revolute joints
in the Franka robot. `RRT.get_watched_joints()` returns an empty list because the joint states of the gripper DOFs are irrelevant when navigating the Franka’s hand to a target position.
Every time RRT returns joint targets for the Franka, it is returning arrays of length seven. When RRT is passed an argument such as `active_joint_positions`,
it is expecting a vector of seven numbers that describe the joint positions of the Franka robot in the order specified by `RRT.get_active_joints()`.

### Inputs: World State

NVIDIA Isaac Sim provides a set of objects in `isaacsim.core.api.objects` that are intended to fully describe the simulated world. Only object primitives such as sphere and cone
are supported. More advanced objects defined by meshes and point clouds will be added in a future release. A PathPlanner has an for each type of object that exists in
`isaacsim.core.api.objects` for example:

\[PathPlanner.add\_sphere(sphere: isaacsim.core.api.objects.sphere.\*)\]

Objects in isaacsim.core.api.objects wrap objects that exist on the USD stage.
As objects move around on the stage, their location can be retrieved on each frame using the representative object from `isaacsim.core.api.objects`. This means that after a
PathPlanner has been passed an object, it can internally query the position of that object on the stage over time as needed. A PathPlanner queries all relevant obstacle positions
from the `isaacsim.core.api.objects` that have been passed in when `PathPlanner.update_world()` is called, and passes the information to its internal world state.

It is not required that a specific PathPlanner actually implement an adder for every type of object that exists in `isaacsim.core.api.objects`. When a class inherits from PathPlanner,
any unimplemented adder functions will throw warnings. For example, [Lula RRT](Robot_Simulation.md) supports spheres, capsules, and cuboids in its world representation.
In environments with cones, RRT will ignore the cone objects, and a warning will be printed for each cone object that gets added.

### Inputs: Robot State

There are two methods for specifying robot state in a PathPlanner:

> * The base pose of the robot can be specified to a PathPlanner using `PathPlanner.set_robot_base_pose()`. If this function is never called, the policy implementation can make a reasonable assumption about the position of the robot. [Lula RRT](Robot_Simulation.md) assumes that the robot is at the origin of the stage until it is told otherwise.
> * `PathPlanner.compute_path(active_joint_positions, watched_joint_positions)` expects robot joint positions and velocities to be passed in using the order specified by `PathPlanner.get_active_joints()` and `PathPlanner.get_watched_joints()`.

### Outputs: Path

`PathPlanner.compute_path(active_joint_positions, watched_joint_positions)` returns a set of configuration space waypoints that can be linearly interpolated to produce a collision free trajectory to reach a target-pose. The c-space configurations output by a PathPlanner will correspond only to the active joints returned by `PathPlanner.get_active_joints()`. The path output by a PathPlanner is difficult to use on its own; a linearly interpolated path will have sharp corners in c-space. But, a PathPlanner can be a useful component in generating a high-quality trajectory through difficult environments.

A helper class is provided with the PathPlanner interface to enable easy visualization of planned paths connected by linear interpolation in the [Path Planner Visualizer](#isaac-sim-path-planner-visualizer) class.

## Path Planner Visualizer

The PathPlannerVisualizer class is provided to make it easy to visualize the path output by a PathPlanner. This class handles the mapping between controllable DOFs in the robot Articulation and the active joints considered by the PathPlanner.

The main function of the class is `PathPlannerVisualizer.compute_plan_as_articulation_actions(max_c-space_dist)`. Calling this function queries the robot state from the robot Articulation, extracts and arranges the appropriate joints from the joint state
to use the `PathPlanner.compute_path()` function, linearly interpolates the result, and then creates a valid list of ArticulationAction that can be passed to the robot Articulation one by one to produce the planned path. The max\_c-space\_dist function determines the density of the linear interpolation such that the L2 norm between any two c-space positions in the output is less than or equal to max\_c-space\_dist.

---

# Lula RRT

We provide a **Lula** implementation of the classic Randomly-Exploring Random Tree (RRT) algorithm to fulfill the PathPlanner interface. Specifically, the c-space RRT is using RRT-Connect based on [[2]](#id3), and the task-space RRT is using Jacobian transpose RRT based on [[3]](#id4). The RRT implementation does not support orientation targets.

## Lula RRT Configuration

Three files are necessary to configure Lula RRT for use with a new robot:

> 1. A URDF (universal robot description file), used for specifying robot kinematics as well as joint and link names. Position limits for each joint are also required. Other properties in the URDF are ignored and can be omitted; these include masses, moments of inertia, visual and collision meshes.
> 2. A supplemental robot description file in YAML format. In addition to enumerating the list of actuated joints that define the configuration space (c-space) for the robot, this file also includes sections for specifying the default c-space configuration. This file can also be used to specify fixed positions for unactuated joints.
> 3. A configuration file in the YAML format specifying parameters for the RRT algorithm such as termination conditions, exploration weights, and step size. These parameters can be modified programmatically with the RRT.set\_param() function.

## References

[[2](#id1)]

J. J. Kuffner and S. M. LaValle, “RRT-connect: An efficient approach to single-query path planning,” Proceedings 2000 ICRA. Millennium Conference. IEEE International
Conference on Robotics and Automation. Symposia Proceedings (Cat. No.00CH37065), 2000, pp. 995-1001 vol.2, doi: 10.1109/ROBOT.2000.844730.

[[3](#id2)]

M. Vande Weghe, D. Ferguson and S. S. Srinivasa, “Randomized path planning for redundant manipulators without inverse kinematics,” 2007 7th IEEE-RAS International Conference
on Humanoid Robots, 2007, pp. 477-482, doi: 10.1109/ICHR.2007.4813913.

---

# Motion Policy Algorithm

An Isaac Sim motion policy is a collision aware algorithm that outputs actions on each frame to navigate a single robot to a single task-space target.
The MotionPolicy class is an interface that is designed to be basic to fulfill,
but complete enough that an implementation of a MotionPolicy can be used alongside the [Articulation Motion Policy](#isaac-sim-articulation-motion-policy) class
to start moving the simulated robot around with just a few lines of code.

A single flexible MotionPolicy is provided based on the implementation of **RMPflow**
in the NVIDIA-developed **Lula** library (see [RMPflow](Robot_Simulation.md)).

Broadly defined, a *motion policy* is a mathematical function that takes the current
state of a robot (that is, position and velocity in generalized coordinates) and returns
a quantity representing a desired change in that state. Such a policy can depend
implicitly on variables representing one or more objectives or constraints, the state of
the environment. The MotionPolicy interface has two forms of state as input:

* [Inputs: World State](#isaac-sim-motion-policy-world-state)
* [Inputs: Robot State](#isaac-sim-motion-policy-robot-state)

The main output of a MotionPolicy are position and velocity targets for the robot on the next frame. A MotionPolicy is
expected to be able to perform an internal world update and compute joint targets in real time (a few ms per frame).

## Active and Watched Joints

The robot Articulation in Isaac Sim comes from a loaded USD file. This robot specification is not expected to perfectly match the specification used internally by a MotionPolicy.
To perform the appropriate mapping, a MotionPolicy has two functions it must fulfill:

* `MotionPolicy.get_active_joints()`: joints that the MotionPolicy is going to directly control to achieve the desired end effector target.
* `MotionPolicy.get_watched_joints()`: joints that the MotionPolicy observes to plan motions, but will not actively control.

Both functions return a list of joint names in the order that the MotionPolicy expects to receive them.

For example, the Franka robot has nine degrees of freedom (DOFs):

* seven revolute joints for controlling the arm
* two prismatic joints for controlling its gripper

The robot Articulation exposes all nine degrees of freedom, but [RMPflow](Robot_Simulation.md) only cares about the seven revolute joints when navigating the robot to a position target. It is not appropriate for RMPflow to take control of the gripper DOFs, because those DOFs can be controlled separately when performing a task such as pick-and-place. `RmpFlow.get_active_joints()` returns the names of the seven revolute joints
in the Franka robot. `RmpFlow.get_watched_joints()` returns an empty list because the joint states of the gripper DOFs are irrelevant when navigating the Franka’s hand to a target position.

Every time RmpFlow returns joint targets for the Franka, it is returning arrays of length seven. When RmpFlow is passed an argument such as active\_joint\_positions, it is expecting a vector of seven numbers that describe the joint positions of the Franka robot in the order specified by `RmpFlow.get_active_joints()`.

## Inputs: World State

NVIDIA Isaac Sim provides a set of objects in `isaacsim.core.api.objects` that are intended to fully describe the simulated world. Only object primitives such as sphere and cone
are supported. A MotionPolicy has an adder for each type of object that exists in
`isaacsim.core.api.objects` for example, `MotionPolicy.add_sphere(sphere: isaacsim.core.api.objects.sphere.*)`. Objects in `isaacsim.core.api.objects` wrap objects that exist on the USD stage.
As objects move around on the stage, their location can be retrieved on each frame using the representative object from `isaacsim.core.api.objects`. This means that after a
MotionPolicy has been passed an object, it can internally query the position of that object on the stage over time as needed. A MotionPolicy queries all relevant obstacle positions
from the `isaacsim.core.api.objects` that have been passed in when `MotionPolicy.update_world()` is called, and passes the information to its internal world state.

It is not required that a specific MotionPolicy actually implement an adder for every type of object that exists in `isaacsim.core.api.objects`. When a class inherits from MotionPolicy,
any unimplemented adder functions will throw warnings. For example, [RMPflow](Robot_Simulation.md) supports spheres, capsules, and cuboids in its world representation.
In environments with cones, RMPflow will ignore the cone objects, and a warning will be printed for each cone object that gets added.

## Inputs: Robot State

There are two methods for specifying robot state in a MotionPolicy:

* The base pose of the robot can be specified to a MotionPolicy using `MotionPolicy.set_robot_base_pose()`. If this function is never called, the policy implementation can make a reasonable assumption about the position of the robot. [RMPflow](Robot_Simulation.md) assumes that the robot is at the origin of the stage until it is told otherwise.
* `MotionPolicy.compute_joint_targets(active_joint_positions, active_joints_velocities, watched_joint_positions, watched_joint_velocities,...)` expects robot joint positions and velocities to be passed in using the order specified by `MotionPolicy.get_active_joints()` and `MotionPolicy.get_watched_joints()`.

## Outputs: Robot Joint Targets

`MotionPolicy.compute_joints_targets(active_joint_positions, active_joints_velocities, watched_joint_positions, watched_joint_velocities,...)` returns position and velocity targets for the robot Articulation
on the next frame. The joint targets are for the active joints, and so they will have the same shape as the active\_joint\_positions argument. By passing a MotionPolicy
to the [Articulation Motion Policy](#isaac-sim-articulation-motion-policy) helper class, the work of translating the robot state between the robot Articulation and the MotionPolicy is done automatically using the outputs of
`MotionPolicy.get_active_joints()` and `MotionPolicy.get_watched_joints()`. A MotionPolicy might expect joint targets to be used in a standard PD controller:

\[kp\*(joint\_position\_targets-joint\_positions) + kd\*(joint\_velocity\_targets-joint\_velocities)\]

Both position and velocity targets must always be returned by a MotionPolicy. NVIDIA Isaac Sim supports providing only position targets or only velocity targets.
To match the default behavior of the Isaac Sim controller when only one target is set, you can set the joint\_velocity\_targets to zero for pure damping,
and it can set the joint\_position\_targets to be equal to the current joint\_positions to effectively remove the position term from the PD equation.

### Articulation Motion Policy

An ArticulationMotionPolicy is initialized using a robot Articulation object that represents the simulated robot, and a MotionPolicy. The purpose of this class is to handle the mapping of joints
between the robot articulation and the policy automatically by using the outputs of `MotionPolicy.get_active_joints()` and `MotionPolicy.get_watched_joints()`. There is a single important function in
this class: `ArticulationMotionPolicy.get_next_articulation_action()`. Calling this function queries the robot state from the robot Articulation, extracts and arranges the appropriate joints from the joint state
to use the `MotionPolicy.compute_joint_targets()` function, and then creates a valid ArticulationAction that can be passed to the robot Articulation to generate motions.

In the Franka example discussed in [Active and Watched Joints](#isaac-sim-motion-policy-active-joints), the robot Articulation that represents the Franka expects nine DOF joint targets. RmpFlow only controls seven of the DOFs. The appropriate
seven DOFs are passed to RmpFlow, and seven DOF joint position and velocity targets are returned. This 7-vector is mapped to a 9-vector, padding with None when no action is supposed to be taken for a particular joint. The
ArticulationAction object that is returned contains a 9-vector for position and velocity targets, and this can be applied to the robot Articulation using `Articulation.get_articulation_controller().apply_action(articulation_action)`.

### Motion Policy Controller

The MotionPolicyController class wraps a motion policy into an instance of `isaacsim.core.api.controllers.BaseController`. Extensions representing individual robots such as `isaacsim.robot.manipulators.franka` have an instance of
a BaseController for moving the robot around. The Franka robot can be moved to a target by importing `isaacsim.robot.manipulators.franka.controllers.RMPFlowController` and using the forward function.

---

# RMPflow

[Riemannian Motion Policy (RMP)](Glossary.md) is a set of motion generation tools that underlies most Isaac Sim manipulator controls.
It creates smooth trajectories for the robots with intelligent collision avoidance.

A **Riemannian Motion Policy**, or *RMP*, is an acceleration policy
accompanied by a matrix \(M(q, \dot{q})\) that is sometimes called an inertia matrix,
borrowing terminology from classical mechanics, but is also closely related to the concept
of a Riemannian metric.

Leveraging the machinery of Riemannian geometry, *RMPflow* is a
framework for combining RMPs representing multiple (possibly competing) objectives and
constraints into a single global acceleration policy. Within this framework, the local RMPs
can be defined on any number of intermediate task spaces (including the operational space of
the end effector, generalizing operational space control). For details, refer to
[\*RMPflow: A computational graph for automatic motion policy generation\*](https://arxiv.org/abs/1811.07049).

Broadly defined, a *motion policy* is a mathematical function that takes the current
state of a robot (for example, position and velocity in generalized coordinates) and returns
a quantity representing a desired change in that state. Such a policy can depend
implicitly on variables representing one or more objectives or constraints, the state of
the environment. An *acceleration policy* is a motion policy where the output is
desired acceleration, \(\ddot q = \pi(q, \dot{q})\), resulting in a second-order
differential equation.

For the purpose of controlling a robot by position or velocity control, typically motion policies are used
where the output is position or velocity. Such
policies can be produced from an acceleration policy using a numerical
integration scheme such as Euler integration.

The [RMPflow Debugging Features](#isaac-sim-motion-generation-rmpflow-debugging-features) section reviews functions belonging to RMPflow that are not part of the MotionPolicy interface.
You can interact with RMPflow to control a robot that is already supported. If you are interested in the internal
mechanics of RMPflow or want to configure RMPflow for an unsupported robot, continue reading the RMPflow documentation.

After reviewing the basics here, also see the [RMPflow Tuning Guide](Robot_Simulation.md) for practical advice on configuring RMPflow for a new robot.

## RMPflow Debugging Features

By directly interacting with an RmpFlow instance, you can access features that are not available in other MotionPolicy implementations.
It is common for developers to want to decouple a [Motion Policy Algorithm](Robot_Simulation.md) from the simulated robot Articulation in NVIDIA Isaac Sim.
For example, when the simulated robot is moving sluggishly, it is important to determine whether the MotionPolicy or the PD gains have been improperly tuned,
but this can be difficult when both the PD gains and the MotionPolicy play a role in driving the robot joints (see [Outputs: Robot Joint Targets](Robot_Simulation.md)).

RMPflow provides visualization functions to clearly show the internal state of the algorithm as part of the stage. RMPflow uses collision spheres internally to
avoid hitting obstacles in the world. These spheres can be visualized over time by calling `RmpFlow.visualize_collision_spheres()`. The visualization will stop when
`RmpFlow.stop_visualizing_collision_spheres()` is called. The nominal end effector position can likewise be visualized with
`RmpFlow.visualize_end_effector_position()` and `RmpFlow.stop_visualizing_end_effector()`.

On their own, the visualization functions can be used to make sure that RMPflow’s internal representation of the robot is reasonable, but it does not help to decouple the
simulated robot from the RmpFlow internal representation of the robot.

On each frame when `RmpFlow.compute_joint_targets(active_joint_positions,...)` is called,
the visualization is updated to use the `active_joint_positions`. This behavior can be turned off using `RmpFlow.set_ignore_state_updates(True)`. When RmpFlow
is “ignoring state updates”, it starts ignoring the `active_joint_positions` argument, and instead begins internally tracking the believed state of the robot by assuming
that is completely independent of the physical simulation of the robot. When RmpFlow is set to ignore state updates from the simulator, and the visualization functions are used,
it becomes simple to determine if an undesirable robot behavior
comes from RmpFlow or from the robot Articulation and its PD gains.

## RMPflow Configuration

Three files are necessary to configure RMPflow for use with a new robot:

> * A **URDF** (universal robot description file), used for specifying robot kinematics
>   :   as well as joint and link names. Position limits for each joint are also required.
>       Other properties in the URDF are ignored and can be omitted; these include masses,
>       moments of inertia, visual, and collision meshes.
> * A **supplementary robot description file** in YAML format. In addition to enumerating
>   :   the list of actuated joints that define the configuration space (c-space) for the robot,
>       this file includes sections for specifying the default c-space configuration
>       and sets of collision spheres used for collision avoidance. This file can also
>       be used to specify fixed positions for unactuated joints.
> * A **RMPflow configuration file** in YAML format, containing parameters for all enabled RMPs.

As a general mathematical framework, RMPflow does not prescribe the form that individual RMPs
must take. The particular implementation of RMPflow in Lula (and by extension NVIDIA Isaac Sim) does
however expose a pre-specified set of RMPs that have been constructed and empirically found
to produce smooth reactive behaviors for a variety of manipulation tasks.

## C-Space Target RMP (c-space\_target\_rmp)

**Purpose:** Specifies a default c-space configuration for the robot, used for redundancy resolution.

**Definition:** Acceleration for this RMP is given by an equation similar to a PD controller, with a
position gain and damping gain, but the magnitude of the position term is capped when the C-space
distance exceeds a threshold. This cap avoids excessive forces when the configuration is far away
from the target. Defining \(q\) to be the full configuration vector:

\[\ddot q = k\_p r(q\_0 - q) - k\_d \dot q\,,\]

where the “robust capping function” \(r(p)\) is given by:

\[\begin{split}r(p) = \left \{ \begin{array}{cl}
p, & ||p|| < \theta \\
\theta\, p / ||p|| & \textrm{otherwise.}
\end{array} \right.\end{split}\]

The inertia matrix is proportional to the identity:

\[M = \mu I\]

The c-space\_target\_rmp section of the RMPflow configuration file contains an additional
inertia parameter \(m\). When this parameter is nonzero, it results in the introduction of
a conceptually separate RMP corresponding to zero c-space acceleration, \(\ddot q = 0\), with inertia
matrix given by \(M = mI\).

**Parameters:**

Units assume revolute joints where \(q\) is expressed in radians. If joints are instead prismatic,
robust\_position\_term\_thresh will have units of meters.

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| metric\_scalar | \(\mu\) | - | Priority weight relative to other RMPs |
| position\_gain | \(k\_p\) | s-2 | Position gain, determining how strongly configuration is pulled toward target |
| damping\_gain | \(k\_d\) | s-1 | Damping gain, determining amount of “drag” |
| robust\_position\_term\_thresh | \(\theta\) | rad | Distance in c-space at which the position correction vector is capped |
| inertia | \(m\) | - | Additional c-space inertia |

## Target RMP (target\_rmp)

**Purpose:** Drives end effector toward specified position target.

**Definition:** Similar to the c-space target RMP, acceleration for this RMP resembles a PD
controller, albeit with a slightly different strategy for capping the magnitude of the position
correction vector.

\[\ddot x = k\_p (x\_0 - x) / (||x\_0-x|| + \epsilon) - k\_d \dot x\]

The inertia matrix blends between a rank-deficient metric \(S = n n^T`\), where \(n\) is
the direction vector toward the target, and the identity \(I\).

Intuitively, \(S\) cares
only about the direction toward the target (letting other RMPs such as the obstacle avoidance RMP
control the orthogonal directions).

\(I\) cares about all directions.

The contribution of
\(S\) is larger farther from the goal, allowing obstacles to push the system more effectively,
while \(I\) dominates near the goal, encouraging faster convergence.

Blending is
controlled by a radial basis function, specifically a Gaussian, that transitions from a minimum
constant value far from the target to 1 near the target.

Near the target, an additional nonlinear “proximity boost” multiplier turns on. This
factor takes the form of a Gaussian:

\[M = \left[\beta(x) b + (1-\beta(x))\right] \left[\alpha(x) M\_\textrm{near} + (1-\alpha(x)) M\_\textrm{far} \right]\]

where

\[\begin{split}\begin{array}{l}
\alpha(x) = (1-\alpha\_\textrm{min})\exp \left(\frac{-||x\_0-x||^2}{2 \sigma\_a^2}\right) + \alpha\_\textrm{min} \\
\beta(x) = \exp \left(-\frac{||x\_0 - x||^2}{2 \sigma\_b^2}\right) \\
M\_\textrm{near} = \mu\_\textrm{near} I \\
M\_\textrm{far} = \mu\_\textrm{far} S = \frac{\mu\_\textrm{far}}{||x\_0-x||^2} (x\_0-x)(x\_0-x)^T\,.
\end{array}\end{split}\]

**Parameters:**

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| accel\_p\_gain | \(k\_p\) | m/s2 | Position gain |
| accel\_d\_gain | \(k\_d\) | s-1 | Damping gain |
| accel\_norm\_eps | \(\epsilon\) | m | Length scale controlling transition between constant acceleration region far from target and linear region near target |
| metric\_alpha\_length\_scale | \(\sigma\_a\) | m | Length scale of the Gaussian controlling blending between \(S\) and \(I\) |
| min\_metric\_alpha | \(\alpha\_\textrm{min}\) | - | Controls the minimum contribution of the isotropic \(M\_\textrm{near}\) term to the metric (inertia matrix) |
| max\_metric\_scalar | \(\mu\_\textrm{near}\) | - | Metric scalar for the isotropic \(M\_\textrm{near}\) contribution to the metric (inertia matrix) |
| min\_metric\_scalar | \(\mu\_\textrm{far}\) | - | Metric scalar for the directional \(M\_\textrm{far}\) contribution to the metric (inertia matrix) |
| proximity\_metric\_boost\_scalar | \(b\) | - | Scale factor controlling the strength of boosting near the target |
| proximity\_metric\_boost\_length\_scale | \(\sigma\_b\) | m | Length scale of the Gaussian controlling boosting near the target |
| xi\_estimator\_gate\_std\_dev | - | - | Unused parameter (to be removed in a future release) |

## Axis Target RMP (axis\_target\_rmp)

**Purpose:** Drives x-, y-, or z-axis of end effector frame toward target orientation. This
RMP is used for general orientation targets (where an axis target RMP is added for each of the
three axes) as well as for “partial pose” targets where only alignment of a single axis is
desired.

Note

Partial pose targets are not supported by the Motion Generation extension.

**Definition:**

Similar to the (position) target RMP, the axis target RMP supports “proximity boosting,”
but only when a target RMP is active at the same time. In this case, it’s the distance to
the position target (\(||x\_0-x||\)) that controls the strength of boosting.

The current and desired axis orientations are represented by unit vectors, denoted
by \(n\) and \(n\_0\) respectively. Acceleration is given by:

\[\ddot n = k\_p (n\_0 - n) - k\_d \dot n\,\]

If a position target (that is, target RMP) is active, the metric has the form:

\[M\_\textrm{boosted} = \left[\beta(x) b + (1-\beta(x))\right] \mu I\,\]

where:

\[\beta(x) = \exp \left(-\frac{||x\_0 - x||^2}{2 \sigma\_b^2}\right)\,\]

When no position target is active, this simplifies to:

\[M = \mu I\,.\]

**Parameters:**

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| accel\_p\_gain | \(k\_p\) | s-2 | Position gain |
| accel\_d\_gain | \(k\_d\) | s-1 | Damping gain |
| metric\_scalar | \(\mu\) | - | Priority weight relative to other RMPs |
| proximity\_metric\_boost\_scalar | \(b\) | - | Scale factor controlling the strength of boosting near the position target |
| proximity\_metric\_boost\_length\_scale | \(\sigma\_b\) | m | Length scale of the Gaussian controlling boosting near the position target |

## Joint Limit RMP (joint\_limit\_rmp)

**Purpose:** Avoids joint limits.

**Definition:** This is a one-dimensional RMP that depends on a single
c-space coordinate (joint) and a corresponding upper or lower joint limit as specified in
the URDF for the robot. If a robot has \(N\) joints, it follows that a total of \(2N\)
joint limit RMPs will be introduced. The joint limits specified in the URDF can be padded
(that is, made more conservative) by entering positive padding values in the joint\_limit\_buffers
array in the RMPflow configuration file. For a given joint, the same padding value is used
for both upper and lower limits.

The task space for this RMP consists of a shifted and scaled c-space coordinate, measuring
the scaled distance to either the upper or lower joint limit. Without loss of generality,
we consider a lower joint limit RMP. If \(q\) is the c-space coordinate for a given
joint, and \(q\_\textrm{upper}\) and \(q\_\textrm{lower}\) are the upper and lower
limits for that joint, respectively, we define:

\[x = \frac{q - q\_\textrm{lower}}{q\_\textrm{upper} - q\_\textrm{lower}}\,\]

The acceleration for that coordinate is then given by:

\[\ddot x = \frac{k\_p}{x^2/\ell\_p^2 + \epsilon\_p} - k\_d \dot x\,\]

The metric (inertia matrix) is a scalar given by:

\[m = \left(1 - \frac{1}{1+\exp(-\dot x/v\_m)}\right) \frac{\mu}{x/\ell\_m + \epsilon\_m}\,\]

**Parameters:**

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| metric\_scalar | \(\mu\) | - | Overall priority weight relative to other RMPs |
| metric\_length\_scale | \(\ell\_m\) | - | Length scale controlling ramp-up of metric as joint limit is approached |
| metric\_exploder\_eps | \(\epsilon\_m\) | - | Offset determining \(x\) value at which metric diverges |
| metric\_velocity\_gate\_length\_scale | \(v\_m\) | s-1 | Scale determining rate at which metric increases with velocity in direction of barrier |
| accel\_damper\_gain | \(k\_d\) | s-1 | Damping gain |
| accel\_potential\_gain | \(k\_p\) | s-2 | Gain multiplying position barrier term |
| accel\_potential\_exploder\_length\_scale | \(\ell\_p\) | - | Length scale controlling steepness of position barrier |
| accel\_potential\_exploder\_eps | \(\epsilon\_p\) | - | Offset limiting divergence of position barrier strength |

## Joint Velocity Limit RMP (joint\_velocity\_cap\_rmp)

**Purpose:** Limits maximum joint velocity.

**Definition:** This RMP applies damping when the magnitude of the velocity of a given joint
approaches the specified limit.

This is a one-dimensional RMP with acceleration given by:

\[\ddot q = -k\_d\,\textrm{sgn}(\dot q) \left(|\dot q| - (v\_\textrm{max} - v\_r)\right)\,\]

The metric (inertia matrix) is a scalar given by:

\[\begin{split}m = \left \{ \begin{array}{cl}
0, & |\dot q| < (v\_\textrm{max} - v\_r) \\
\frac{\mu}{1 - \left(|\dot q| - (v\_\textrm{max} - v\_r)\right)^2 / v\_r^2} & \textrm{otherwise.}
\end{array} \right\end{split}\]

The metric is zero outside you-specified damping region, thereby disabling this RMP.
In addition, clipping is applied to avoid divergence of the metric as \(\dot q\) approaches \(v\_\textrm{max}\).

**Parameters:**

Units assume revolute joints where \(q\) is expressed in radians. If joints are instead prismatic,
max\_velocity and velocity\_damping\_region will have units of m/s.

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| max\_velocity | \(v\_\textrm{max}\) | rad/s | Maximum allowed velocity magnitude |
| velocity\_damping\_region | \(v\_r\) | rad/s | Defines width of velocity region affect by damping |
| damping\_gain | \(k\_d\) | s-1 | Damping gain |
| metric\_weight | \(\mu\) | - | Overall priority weight relative to other RMPs |

## Collision Avoidance RMP (collision\_rmp)

**Purpose:** Avoids collision with obstacles in the environment.

**Definition:** This is a one-dimensional RMP where the task space consists of a single
coordinate measuring distance from a given collision sphere on the robot (specified in the
robot description YAML file) to an obstacle in the environment. Denoting that coordinate
as \(x\), the acceleration is given by:

\[\ddot x = k\_p \exp(-x / \ell\_p) - k\_d \left[1 - \frac{1}{1 + \exp(-\dot x/v\_d)} \right] \frac{\dot x}{x/\ell\_d + \epsilon\_d}\,\]

The metric (inertia matrix) is a scalar given by:

\[m = \left[1 - \frac{1}{1 + \exp(-\dot x/v\_d)} \right] g(x) \frac{\mu}{x / \ell\_m + \epsilon\_m}\,\]

where \(g(x)\) is a piecewise polynomial that varies smoothly from 1 to 0 as \(x\) varies from 0 to \(r\)

\[\begin{split}g(x) = \left \{ \begin{array}{cl}
x^2/r^2 -2s/r + 1, & x\le r \\
0, & x\gt r
\end{array} \right.\end{split}\]

**Parameters:**

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| damping\_gain | \(k\_d\) | s-1 | Damping gain |
| damping\_std\_dev | \(\ell\_d\) | m | Length scale controlling increase in acceleration as obstacle is approached |
| damping\_robustness\_eps | \(\epsilon\_d\) | - | Offset determining \(x\) value at which acceleration diverges (before clipping) |
| damping\_velocity\_gate\_length\_scale | \(v\_d\) | m/s | Scale determining velocity dependence of “velocity gating” function |
| repulsion\_gain | \(k\_p\) | m/s2 | Gain for position repulsion term |
| repulsion\_std\_dev | \(\ell\_p\) | m | Length scale controlling distance dependence of repulsion |
| metric\_modulation\_radius | \(r\) | m | Length scale determining distance from obstacle at which RMP is disabled completely |
| metric\_scalar | \(\mu\) | - | Overall priority weight relative to other RMPs |
| metric\_exploder\_std\_dev | \(\ell\_m\) | m | Length scale controlling increase in metric as obstacle is approached |
| metric\_exploder\_eps | \(\epsilon\_m\) | - | Offset determining \(x\) value at which metric diverges (before clipping) |

## Damping RMP (damping\_rmp)

**Purpose:** Contributes additional nonlinear damping based on control frame
(for example, end effector) velocity relative to target.

**Definition:** This is a one-dimensional RMP where the task space consists of a single coordinate
\(x\) measuring distance from the origin of the control frame to the target.
The acceleration is given by:

\[\ddot x = -k\_d |\dot x|\dot x\]

and the metric by:

\[M = \mu |\dot x| I\,\]

The damping\_rmp section of the RMPflow configuration file contains an additional
inertia parameter \(m\). When this parameter is nonzero, it results in the introduction of
a conceptually separate RMP corresponding to zero acceleration, \(\ddot x = 0\), with inertia
matrix given by \(M = mI\).

**Parameters:**

| Name | Symbol | Units | Meaning |
| --- | --- | --- | --- |
| accel\_d\_gain | \(k\_d\) | m-1 | Nonlinear damping gain |
| metric\_scalar | \(\mu\) | (m/s)-1 | Priority weight relative to other RMPs |
| inertia | \(m\) | - | Additional inertia |

## Further Reading

Refer to the [RMPflow Tuning Guide](Robot_Simulation.md) for practical advice on configuring RMPflow for a new robot.

---

# RMPflow Tuning Guide

Given the number of parameters involved in fully specifying a complete set of RMPs,
tuning an RMPflow-based motion policy for a new robot or task can be intimidating.
In practice, however, parameters that work well for one robot are likely to work well
for other robots with similar morphology. Furthermore, for a given robot, it is
generally possible to choose a set of parameters that work well for a wide variety
of tasks.

To review RMPflow and its features see, [RMPflow](Robot_Simulation.md).

NVIDIA Isaac Sim includes example RMPflow configuration files for multiple robot arms, including
the 7-DOF Franka Emika Panda and the 6-DOF Universal Robots UR10. When tuning RMPflow for a
new manipulator, it’s usually best to start with one of these two files. If the new robot
is significantly larger or smaller than the one used as a reference, it might be necessary
to rescale any parameters that have units of length. If the number of joints differ, the
c-space\_target\_rmp/robust\_position\_term\_thresh parameter might also have to be adjusted.
Often, these steps are sufficient to produce a working motion policy.

If adapting an existing RMPflow configuration fails to produce acceptable results, use
the following procedure to tune a new policy from scratch:

Hint

It can helpful to play with parameter values for an existing robot (for example, the Franka).

1. Turn off all RMPs.
2. Each RMP has a parameter called either metric\_weight or metric\_scalar. Setting this parameter to zero will disable the corresponding RMP. For the target RMP, set the parameters min\_metric\_scalar, max\_metric\_scalar, and min\_metric\_alpha all to zero.
3. Set all inertia terms to zero (that is, c-space\_target\_rmp/inertia and damping\_rmp/inertia).
4. Re-enable RMPs one at a time, in the following suggested order:

   1. **c-space\_target\_rmp:** To get the robot moving to a configuration in c-space robustly.
      The magnitude of the metric scalar should be kept relatively small (for example, in the range 1 to 100), because
      this sets the global scale of all RMPs.
      Remember to set the default configuration in the robot description file (YAML) to a reasonable natural
      “ready” posture. This will be the default posture that the robot will favor while moving from place to place.
   2. **target\_rmp:** To get the end effector moving to a target robustly while continuing
      to use the c-space target RMP for redundancy resolution.

      1. Set target\_rmp/min\_metric\_alpha to zero and target\_rmp/metric\_alpha\_length\_scale
         to a large value relative to the size of the robot (in meters), such as 100,000. This effectively turns
         off the directional \(S\) term in the metric, reducing \(M\) to a simpler isotropic metric.
      2. Set target\_rmp/proximity\_metric\_boost\_length\_scalar to 1 to turn off priority boosting.
      3. Set target\_rmp/max\_metric\_scalar to a large value relative to c-space\_target\_rmp/metric\_scalar
         so it dominates. This will effectively make the c-space target RMP operate purely in the
         nullspace of the target RMP.
      4. Tune target\_rmp/accel\_p\_gain, target\_rmp/accel\_d\_gain, and target\_rmp/accel\_norm\_eps until
         good attractor behavior for the end effector has been achieved.
      5. Experiment with reducing target\_rmp/max\_metric\_scalar to ensure that it’s not too large. As
         max\_metric\_scalar is increased toward a suitable value, convergence accuracy should progressively
         improve. If convergence accuracy saturates at small constant error before the chosen max\_metric\_scalar
         value is reached, then it is probably set too high. This will be relevant when re-enabling the directional
         term in the target RMP metric below, ensuring that it makes a difference when the metric scalar decreases.
   3. **collision\_rmp:** Enable the collision avoidance RMP by setting collision\_rmp/metric\_scalar to a value
      comparable to target\_rmp/max\_metric\_scalar. It can be useful to plot the formulas for the acceleration
      and metric to gain some understanding of the roles of the various parameters.
   4. **target\_rmp (redux):** After the collision RMP is enabled, the system will probably drag near obstacles
      more slowly than it usually moves because the target RMP is fighting with the collision RMPs.
      Turning on the directional term in the metric will correct that effect.

      1. Plot the target RMP metric (as a function of distance from target)
         to build understanding. Try this first without the boosting term, noting how the metric transitions
         from the reduced-rank far metric to the full-rank near metric.
      2. Set target\_rmp/min\_metric\_alpha to a non-zero value and reduce the value of
         target\_rmp/metric\_alpha\_length\_scale until good behavior is achieved.
   5. **axis\_target\_rmp:** If an orientation target is set, the axis target RMP will be used to bring
      the orientation of the control frame (for example, end effector) into alignment with the target orientation.
      This RMP includes a “priority boosting” factor that depends on distance to the current
      position target, if one is set. This allows the robot to make progress toward the position
      target before zeroing in on the desired orientation.
   6. **joint\_limit\_rmp:** When properly tuned, behavior should be unchanged, except that joint
      limits will be avoided.
   7. **damping\_rmp:** Enable the damping RMP as well as target\_rmp/inertia to reduce jerk as necessary.

Throughout this process, referring to an existing RMPflow configuration file is helpful.

---

# Lula Robot Description and XRDF Editor

## Learning Objectives

This tutorial shows how to use the **Robot Description Editor** UI tool to generate a configuration file that supplements
the information available about a robot in its URDF. Two motion generation packages leverage the
**Robot Description Editor** to specify necessary configuration information:

* [cuMotion](https://nvidia-isaac-ros.github.io/concepts/manipulation/index.html#nvidia-cumotion)
* Lula

This tutorial describes the motivation for needing specific config files for `Lula` and `cuMotion` algorithms, and goes over the minimal set of data that needs to be written into a robot description file for each available Lula algorithm.

This tutorial then shows how to use the **Robot Description Editor** UI tool to automatically write the appropriate information into (or edit) a `robot_description.yaml` file for Lula or an
[XRDF](https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html) file for `cuMotion`.

The **Robot Description Editor** is used on a stage that already has an Articulation on it. To follow along with the steps in the tutorial, it is best to open a single asset by reference. That is,
drag and drop a USD file onto an empty stage rather than clicking on the USD file to open it directly.

## What is in a Robot Description File?

A robot description file is the main configuration file that is required along with the robot URDF to use all Lula algorithms. Creating a `robot_description.yaml` file is the first and most time-consuming step that a user must take when hoping to use Lula algorithms on a new robot.

### Defining the Robot C-Space: Active and Fixed Joints

A key aspect of a robot description file is defining the robot c-space. For example, suppose we have a seven DOF robot manipulator such as the Franka arm with an attached two DOF gripper. In the robot URDF file, there are a total of nine non-fixed joints that could be considered controllable. However, the set of Lula algorithms ([RMPflow](Robot_Simulation.md), [Lula RRT](Robot_Simulation.md), [Lula Trajectory Generator](Robot_Simulation.md)) are designed to move the robot into position but not to control the end effector. In a typical use case, you might use `RmpFlow` to move the robot end effector into position above a block and then separately open and close the gripper.

A robot description file must distinguish each joint as:

* Active Joint
* Fixed Joint

Anything marked as an Active Joint will be directly controlled, while anything marked as a Fixed Joint will be assumed to be fixed from the perspective of Lula algorithms. In the case of using `RmpFlow` on the Franka robot, the seven joints in the Franka’s arm are marked as Active Joints, and the gripper joints are marked as Fixed Joints.

In the **Robot Description Editor**, positions must be selected for both active and fixed joints. The positions of Fixed Joints are taken to be default positions. When RmpFlow is not given any target, it will move the robot towards the default position. And when it is given a target, it will use the default positions of the Fixed Joints to resolve null-space behavior; that is, there are many ways for a seven DOF robot to reach a single target, and RmpFlow will be biased towards a c-space position that is close to the default position.

There is no way of telling RmpFlow that the Fixed Joints are in any other position than the position written into the robot description file, and as such it is important to choose a reasonable value for the positions of fixed joints. In the Franka example, the gripper joint positions are given a fixed value corresponding to the gripper being open, as this best facilitates RmpFlow avoiding collisions between the gripper and obstacles no matter the gripper state (when closed, the gripper fingers are inside the convex hull of an open gripper).

### Collision Spheres

Lula algorithms use a custom configuration to perform efficient collision avoidance. For a given robot, a set of collision spheres must be defined that roughly cover the surface of the robot. Lula algorithms will not allow any collision sphere defined in the robot description file to intersect any obstacle in the USD world. The **Robot Description Editor** provides multiple tools that allow you to quickly define a complete set of collision spheres for any robot.

### What is the Difference between a Robot Description File and an XRDF file?

An [XRDF](https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html) file is the main configuration file that is required by cuMotion for a specific robot,
and it contains a superset of the data in a Lula Robot Description File.
The **Robot Description Editor** can be used to generate an XRDF file that contains the minimal data required to start using cuMotion.
The use of the **Robot Description Editor** need not change in any way when configuring a robot for use with cuMotion versus Lula.
In the future, Lula will fully support XRDF files and deprecate Robot Description Files.

**As of Isaac Sim 4.0.0, the Robot Description Editor was modified to support XRDF files and some parts of this tutorial reference UI components that have changed.**

## What Information is Required for Each Lula Algorithm?

Different Lula algorithms require different levels of completion of the robot description file.
Every algorithm requires you to appropriately choose active and fixed joints.
However, collision spheres are only necessary to configure when using algorithms that perform collision avoidance with external obstacles.
For example, the [Lula Kinematics Solver](Robot_Simulation.md) is purely kinematic, and it does not interact with the outside world.
As such, the collision sphere representation can be omitted from the robot description file.
[RMPflow](Robot_Simulation.md) can function without any collision spheres being defined, but it will not be able to avoid obstacles.

## Using the Robot Description Editor

This section of the tutorial includes brief text descriptions of the different panels in the **Robot Description Editor** UI tool. A more step-by-step tutorial can be found in the [Generate Lula Robot Description Files and Collision Spheres](Robot_Setup.md) tutorial.

Note

The **Robot Description Editor** is not compatible with [Instanceable Assets](Isaac_Lab.md), but a robot description file generated
for an asset that was later converted to an instanceable asset will still work on the instanceable asset.

### Getting Started

The **Robot Description Editor** can be found from the tool bar under **Tools > Robotics > Lula Robot Description Editor**. To get started, open the USD file of your chosen robot and click the **Play** button on the left-hand side.

In the **Selection Panel**, after a robot is on the stage and the stage is playing, a drop-down menu will populate where your robot can be selected. Select the prim path of your robot Articulation from the **Select Articulation** field. After this is done, another drop-down labeled **Select Link** will populate with the names of each link in our robot. This will be needed later as we use the tool.

We have done everything we need to do to start making our robot description file. Other panels will populate with robot-specific information, and we can move on to the **Set Joint Properties**.

### Set Joint Properties

As of Isaac Sim 4.0.0, **Command Panel** was renamed to **Set Joint Properties**, and fields were added to each joint for jerk and acceleration limits.

After the robot **Articulation** has been selected from the **Select Articulation** menu, the **Set Joint Properties** will expand and populate. The **Set Joint Properties** requires you to supply critical information for the robot description file to be properly generated. You can refer to [Defining the Robot C-Space: Active and Fixed Joints](#isaac-sim-tutorial-robot-description-editor-active-vs-fixed) for details.

In the **Set Joint Properties** select a **Joint Position** and a **Joint Status** for each joint in the robot Articulation. Keep in mind the following:

> * Joints are marked as Fixed Joints if and only if you intend for that joint to be directly controlled by Lula algorithms. Typically this involves marking each joint in the robot arm as active while leaving the joints in the manipulator attached to the arm as Fixed Joints. **At least one joint must be marked as an `Active Joint`**.
> * The joint positions of Fixed Joints can matter, depending on the use case and are worth some thought. The positions of Fixed Joints will be assumed by Lula to be truly fixed; that is, there is no way override the positions at runtime.
> * The positions of Fixed Joints are considered to be the default configuration of the robot. This default configuration is used by a subset of Lula algorithms, with the main case being `RmpFlow`. A default configuration should be chosen that is in front of the robot (along the +X axis by convention in Isaac Sim) and is not near any joint limits.

### Adding Collision Spheres

Collision spheres are added to the robot one link at a time. You can select the link of interest from the “Select Link” field of the **Selection Panel**. The **Link Sphere Editor** panel contains functions that are within the scope of the selected link such as adding spheres, scaling spheres, and clearing spheres only within the link. The **Editor Tools** panel contains functions that are outside the scope of the selected link such as **Undo** and **Redo** buttons, changing the color of collision spheres, and toggling the visibility of the robot.

When spheres are added to a link, they are added to the USD stage as a prim that is nested under the selected link. You can click on and modify any sphere by moving it around on the stage or changing its radius. The position of a sphere relative to the origin of the link that contains it is written as a fixed value into the robot description file.

There are three main ways to add a sphere to a link:

> * **Add Sphere:** Add a single sphere with a specified relative translation from the origin of the link. This translation can be easily changed after creation by modifying the sphere prim.
> * **Connect Spheres:** Select two spheres that have already been created under a link and connect them with a specified number of spheres in between. The locations and sizes of the connecting spheres are interpolated to best fill the volume of the cone-section defined by the two spheres being connected.
> * **Generate Spheres:** Select a mesh that defines the volume of the link, and automatically generate a set of N spheres that best fill the volume of the mesh. When a number of generated spheres is specified, a preview of the generated spheres will automatically appear, which can be finalized by clicking the “Generate Spheres” button. Any visible robot must will at least one mesh defining its link. When there are more than one mesh, it is best to try each of them to figure out the minimal set of spheres that can be generated for good coverage. It is typically better to “Connect Spheres” by hand for links with simple cylindrical shapes. This utility is not guaranteed to work for all meshes. It only works for water-tight triangle meshes. If the automatic generator doesn’t work for a link, add the spheres and connect them to the links by hand.

### Exporting Configuration Files

#### Lula Robot Description File

After completing the **Set Joint Properties** and creating a collision sphere representation of the robot, the robot description file can be exported under
**Export To File > Export to Lula Robot Description File**.
A file path to your local machine must be selected with a file name ending in `.yaml`.
The **Save** button will become enabled when a valid file path has been typed.

#### XRDF File

After completing the **Set Joint Properties** and creating a collision sphere representation of the robot, an [XRDF](https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html)
file can be generated under **Export to File > Export to cuMotion XRDF.** The file path must end in `.yaml` or `.xrdf`.
The **Save** button will become enabled when a valid file path has been typed.
When exporting an XRDF file, the **Robot Description Editor** has the following behavior:

* Create a single collision group that is used for both `collision` and `self_collision` that uses the spheres created in the editor.
* Under `self_collision`, set each link to ignore both its parent and other links that have the same parent.
* Do not write Tool Frames.
* Do not write Modifiers.

Because XRDF files can contain more information than is represented in the **Robot Description Editor**, it is possible to merge the data in the **Robot Description Editor** with
an existing XRDF file. By selecting a file path to an XRDF file that already exists, an option will appear to **Merge With Existing XRDF**.
When merging with an existing XRDF file, the **Robot Description Editor** has the following behavior:

* Copy Tool Frames from the existing file.
* Copy Modifiers from the existing file.
* Copy self\_collision > ignore from the existing file if self\_collision > geometry matches collision > geometry.
* Copy collision spheres from the existing file for any frames that were not represented in the **Robot Description Editor**.

### Importing Configuration Files

#### Lula Robot Description File

A pre-existing robot description file can be imported into the editor under **Import From File > Import Lula Robot Description File**.
Importing will overwrite all information in the **Robot Description Editor**.

#### XRDF File

A pre-existing [XRDF](https://nvidia-isaac-ros.github.io/concepts/manipulation/xrdf.html) file can be imported under **Import From File > Import XRDF File**.
The **Robot Description Editor** imports XRDF files with the following behavior:

* The format version is assumed to be compatible with version 1.0.
* Only the collision group spheres are imported.
* Modifiers are not used.
* Tool Frames are not used.
* The `self_collision` group is not used.

Importing will overwrite all information in the **Robot Description Editor**.

## Summary

This tutorial shows how to use the Lula **Robot Description Editor** to efficiently generate a Lula robot description file. This covers most of the configuration information required for different Lula algorithms.

The **Robot Description Editor** also supports XRDF files for use with `cuMotion`.

### Further Learning

To get the robot moving around with Lula algorithms, review the following tutorials:

> [Configuring RMPflow for a New Manipulator](Robot_Simulation.md)
>
> [Lula Kinematics Solver](Robot_Simulation.md)
>
> [Lula Trajectory Generator](Robot_Simulation.md)

---

# Lula RMPflow

This tutorial shows how the [RMPflow](Robot_Simulation.md) class in the [Motion Generation](concepts/index.html#isaac-sim-motion-generation) can be used to
generate smooth motions to reach task-space targets while avoiding dynamic obstacles. This tutorial demonstrates how:

* `RmpFlow` can be directly instantiated and used to generate motions using a custom robot description file
* `RmpFlow` can be loaded and used on supported robots
* built-in debugging features can improve easy of use and integration

## Getting Started

**Prerequisites**

* Complete the [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md) tutorial prior to beginning this tutorial.
* Review the [Loaded Scenario Extension Template](Templates.md) to understand how this tutorial is structured and run.

To follow along with the tutorial, you can search and enable the **Motion Generation Tutorials** extension within your running Isaac Sim 6.0 instance.
Within the isaacsim.robot\_motion.motion\_generation.tutorials extension, there is a fully functional example of RMPflow including following a target, world awareness,
and a debugging option. The sections of this tutorial build up the file `scenario.py` from basic functionality to the completed code.

## Generating Motions with an RMPflow Instance

[RMPflow](Robot_Simulation.md) is used heavily throughout NVIDIA Isaac Sim for controlling robot manipulators. As documented
in [RMPflow Configuration](Robot_Simulation.md), there are three configuration files needed to directly instantiate the `RmpFlow` class directly.
After these configuration files are loaded and an end effector target has been specified, actions can be computed to move the robot to the desired target.

```python
import os

import numpy as np
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.storage.native import get_assets_root_path

class FrankaRmpFlowExample:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rmpflow_config_path=rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml",
            end_effector_frame_name="right_gripper",
            maximum_substep_size=0.00334,
        )

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))

    def update(self, step: float):
        # Step is the time elapsed on this frame
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(target_position, target_orientation)

        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self):
        # Rmpflow is stateless unless it is explicitly told not to be

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))
```

`RMPflow` is an implementation of the [Motion Policy Algorithm](Robot_Simulation.md) interface.
Any MotionPolicy can be passed to an [Articulation Motion Policy](Robot_Simulation.md)
to start moving a robot on the USD stage. On line 43, an instance of `RmpFlow` is instantiated
with the required configuration information. The `ArticulationMotionPolicy` created on line 52 acts as a
translational layer between `RmpFlow` and the simulated Franka robot `Articulation`. You can interact with
`RmpFlow` directly to communicate the world state, set an end effector target, or modify internal settings.
On each frame, an end effector target is passed directly to the `RmpFlow` object (line 60).
The `ArticulationMotionPolicy` is used on line 64 to compute an action that can be directly consumed by the
Franka `Articulation`.

Note

The RMPflow algorithm takes in consideration the robot structure provided by the configuration URDF file. If working on a robot with assembled components (for example, a UR10 with a gripper attached), the URDF file should be updated to reflect the correct robot structure and contain the offset of the gripper at the end effector frame, or additional control joints. The final assembly URDF can be exported with the [USD to URDF Exporter](Importers_and_Exporters.md). When modifying the source URDF file, it is recommended to review and update the [Robot Description file](Robot_Simulation.md) to ensure that the correct supplemental file is being used.

### World State

As a [Motion Policy Algorithm](Robot_Simulation.md), `RmpFlow` is capable of dynamic collision avoidance
while navigating the end effector to a target. The world state can be
changing over time while `RmpFlow` is navigating to its target. Objects created with the `isaacsim.core.api.objects` package
(see [Inputs: World State](Robot_Simulation.md)) can be registered with `RmpFlow` and the policy will automatically avoid collisions with these obstacles.
`RmpFlow` is triggered to query the current state of all tracked objects whenever `RmpFlow.update_world()` is called.

`RmpFlow` can also be informed about a change in the robot base pose on a given frame by calling `RmpFlow.set_robot_base_pose()`.
As object positions are queried in world coordinates, it is critical to use this function, if the base of the robot is moved
within the USD stage.

```python
import os

import numpy as np
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
from isaacsim.storage.native import get_assets_root_path

class FrankaRmpFlowExample:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])

        self._obstacle = FixedCuboid(
            "/World/obstacle", size=0.05, position=np.array([0.4, 0.0, 0.65]), color=np.array([0.0, 0.0, 1.0])
        )

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, self._obstacle

    def setup(self):
        # RMPflow config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(
            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
            rmpflow_config_path=rmp_config_dir + "/franka/rmpflow/franka_rmpflow_common.yaml",
            end_effector_frame_name="right_gripper",
            maximum_substep_size=0.00334,
        )
        self._rmpflow.add_obstacle(self._obstacle)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))

    def update(self, step: float):
        # Step is the time elapsed on this frame
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(target_position, target_orientation)

        # Track any movements of the cube obstacle
        self._rmpflow.update_world()

        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self):
        # Rmpflow is stateless unless it is explicitly told not to be

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))
```

On lines 22, an obstacle is added to the stage, and on line 40, it is registered as an obstacle with `RmpFlow`.
On each frame, `RmpFlow.update_world()` is called (line 56). This triggers `RmpFlow` to query the current position of the cube to account for any movement.

On lines 59-60, the current position of the robot base is queried and passed to `RmpFlow`.
This step is separated from other world state because
it is often unnecessary (when the robot base never moves from the origin), or this step might require extra consideration
(for example, `RmpFlow` is controlling an arm that is mounted on a moving base).

## Loading RMPflow for Supported Robots

In the previous sections, observe that `RmpFlow` requires five arguments to be initialized. Three of these arguments are file paths to required configuration files.
The `end_effector_frame_name` argument specifies what frame on the robot (from the frames found in the referenced URDF file) should be considered the end effector.
The `maximum_substep_size` argument specifies a maximum step-size when internally performing the Euler Integration.

For manipulators in the NVIDIA Isaac Sim library, appropriate config information for loading RmpFlow can be found in the `isaacsim.robot_motion.motion_generation` extension.
This information is indexed by robot name and can be accessed simply.
The following change shows how loading configs for supported robots can be simplified.

```python
import os

import numpy as np
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
from isaacsim.storage.native import get_assets_root_path

class FrankaRmpFlowExample:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])

        self._obstacle = FixedCuboid(
            "/World/obstacle", size=0.05, position=np.array([0.4, 0.0, 0.65]), color=np.array([0.0, 0.0, 1.0])
        )

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, self._obstacle

    def setup(self):
        # Loading RMPflow can be done quickly for supported robots
        print("Supported Robots with a Provided RMPflow Config:", list(get_supported_robot_policy_pairs().keys()))
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)
        self._rmpflow.add_obstacle(self._obstacle)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))

    def update(self, step: float):
        # Step is the time elapsed on this frame
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(target_position, target_orientation)

        # Track any movements of the cube obstacle
        self._rmpflow.update_world()

        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self):
        # Rmpflow is stateless unless it is explicitly told not to be

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))
```

A supported set of robots can have their RMPflow configs loaded by name.
Line 34 prints the names of every supported robot with a provided RMPflow config (at the time of writing this tutorial):

> [‘Franka’, ‘UR3’, ‘UR3e’, ‘UR5’, ‘UR5e’, ‘UR10’, ‘UR10e’, ‘UR16e’, ‘Rizon4’, ‘Cobotta\_Pro\_900’, ‘Cobotta\_Pro\_1300’, ‘RS007L’, ‘RS007N’, ‘RS013N’, ‘RS025N’, ‘RS080N’, ‘FestoCobot’, ‘Techman\_TM12’, ‘Kuka\_KR210’, ‘Fanuc\_CRX10IAL’]

On lines 35,38, the RmpFlow class initializer is simplified to unpacking a dictionary of loaded keyword arguments.
The `load_supported_motion_policy_config()` function is the simplest way to load supported robots.

## Debugging Features

The `RmpFlow` class has contains debugging features that are not generally available in the [Motion Policy Algorithm](Robot_Simulation.md) interface.
These debugging features allow decoupling of the simulator from the RmpFlow algorithm to help diagnose any undesirable behaviors
that are encountered ([RMPflow Debugging Features](Robot_Simulation.md)).

`RmpFlow` uses collision spheres internally to avoid collisions with external objects. These spheres can be visualized with
the `RmpFlow.visualize_collision_spheres()` function. This helps to determine whether `RmpFlow` has a reasonable representation
of the simulated robot.

The visualization can be used alongside a flag `RmpFlow.set_ignore_state_updates(True)` to ignore state updates from the robot
`Articulation` and instead assume that robot joint targets returned by `RmpFlow` are always perfectly achieved. This causes `RmpFlow`
to compute a robot path over time that is independent of the simulated robot `Articulation`. At each timestep, `RmpFlow` returns joint
targets that are passed to the robot `Articulation`.

```python
import os

import numpy as np
from isaacsim.core.api.objects.cuboid import FixedCuboid
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy, RmpFlow
from isaacsim.robot_motion.motion_generation.interface_config_loader import (
    get_supported_robot_policy_pairs,
    load_supported_motion_policy_config,
)
from isaacsim.storage.native import get_assets_root_path

class FrankaRmpFlowExample:
    def __init__(self):
        self._rmpflow = None
        self._articulation_rmpflow = None

        self._articulation = None
        self._target = None

        self._dbg_mode = True

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])

        self._obstacle = FixedCuboid(
            "/World/obstacle", size=0.05, position=np.array([0.4, 0.0, 0.65]), color=np.array([0.0, 0.0, 1.0])
        )

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target, self._obstacle

    def setup(self):
        # Loading RMPflow can be done quickly for supported robots
        print("Supported Robots with a Provided RMPflow Config:", list(get_supported_robot_policy_pairs().keys()))
        rmp_config = load_supported_motion_policy_config("Franka", "RMPflow")

        # Initialize an RmpFlow object
        self._rmpflow = RmpFlow(**rmp_config)
        self._rmpflow.add_obstacle(self._obstacle)

        if self._dbg_mode:
            self._rmpflow.set_ignore_state_updates(True)
            self._rmpflow.visualize_collision_spheres()

            # Set the robot gains to be deliberately poor
            bad_proportional_gains = self._articulation.get_articulation_controller().get_gains()[0] / 50
            self._articulation.get_articulation_controller().set_gains(kps=bad_proportional_gains)

        # Use the ArticulationMotionPolicy wrapper object to connect rmpflow to the Franka robot articulation.
        self._articulation_rmpflow = ArticulationMotionPolicy(self._articulation, self._rmpflow)

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))

    def update(self, step: float):
        # Step is the time elapsed on this frame
        target_position, target_orientation = self._target.get_world_pose()

        self._rmpflow.set_end_effector_target(target_position, target_orientation)

        # Track any movements of the cube obstacle
        self._rmpflow.update_world()

        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._rmpflow.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action = self._articulation_rmpflow.get_next_articulation_action(step)
        self._articulation.apply_action(action)

    def reset(self):
        # Rmpflow is stateless unless it is explicitly told not to be
        if self._dbg_mode:
            # RMPflow was set to roll out robot state internally, assuming that all returned joint targets were hit exactly.
            self._rmpflow.reset()
            self._rmpflow.visualize_collision_spheres()

        self._target.set_world_pose(np.array([0.5, 0, 0.7]), euler_angles_to_quats([0, np.pi, 0]))
```

The collision sphere visualization can be very helpful to distinguish between behaviors that are coming from the simulator, and behaviors that are coming from `RmpFlow`.
In the image below, the Franka robot is given weak proportional gains (lines 43-44). Using the debugging visualization, it is easy to
determine that RmpFlow is producing reasonable motions, but the simulated robot is simply not able to follow the motions. When RMPflow moves the robot quickly,
the Franka robot `Articulation` lags significantly behind the commanded position.

## Summary

This tutorial reviews using the `RmpFlow` class to generate reactive motions in response to a dynamic environment. The `RmpFlow`
class can be used to generate motions directly alongside an [Articulation Motion Policy](Robot_Simulation.md).

This tutorial reviewed four of the main features of `RmpFlow`:

> 1. Navigating the robot through an environment to a target position and orientation.
> 2. Adapting to a dynamic world on every frame.
> 3. Adapting to a change in the robot’s position on the USD stage.
> 4. Using visualization to decouple the simulated robot `Articulation` from the RMPflow algorithm for quick and easy debugging.

### Further Learning

To learn how to configure RMPflow for a new robot, review the
[basic formalism](Robot_Simulation.md), and then read the
[RMPflow tuning guide](Robot_Simulation.md) for practical advice.

To understand the motivation behind the structure and usage of `RmpFlow` in NVIDIA Isaac Sim, reference
the [Motion Generation](concepts/index.html#isaac-sim-motion-generation) page.

---

# Lula RRT

This tutorial shows how the [Lula RRT](Robot_Simulation.md) class in the [Motion Generation](concepts/index.html#isaac-sim-motion-generation) extension can be used to
produce a collision free path from a starting configuration space (c-space) position to a c-space or task-space target.

## Getting Started

**Prerequisites**

* Complete the [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md) tutorial prior to beginning this tutorial.
* You can reference the Lula Robot Description Editor to understand how to generate your own robot\_description.yaml file to be able to use RRT on unsupported robots.
* Review the [Loaded Scenario Extension Template](Templates.md) to understand how this tutorial is structured and run.

To follow along with the tutorial, you can search and enable the **Motion Generation Tutorials** extension within your running Isaac Sim 6.0 instance.
Within the isaacsim.robot\_motion.motion\_generation.tutorials extension, there is a fully functional example of RRT being used to plan to a task-space target.
The sections of this tutorial build up the file `scenario.py` from basic functionality to the completed code.

## Generating a Path Using an RRT Instance

### Required Configuration Files

[Lula RRT](Robot_Simulation.md) requires three configuration files to identify a specific robot in
[Lula RRT Configuration](Robot_Simulation.md). Paths to these configuration files are used to initialize the `RRT`
class along with an end effector name matching a frame in the robot URDF.

One of the required files contains parameters for the RRT algorithm specifically, and is not shared with any other Lula algorithms.
This tutorial loads the following RRT config file for the Franka robot:

```python
 1seed: 123456
 2step_size: 0.05
 3max_iterations: 50000
 4max_sampling: 10000
 5distance_metric_weights: [3.0, 2.0, 2.0, 1.5, 1.5, 1.0, 1.0]
 6task_space_frame_name: "panda_rightfingertip"
 7task_space_limits: [[0.0, 0.7], [-0.6, 0.6], [0.0, 0.8]]
 8cuda_tree_params:
 9    max_num_nodes: 10000
10    max_buffer_size: 30
11    num_nodes_cpu_gpu_crossover: 3000
12c_space_planning_params:
13    exploration_fraction: 0.5
14task_space_planning_params:
15    translation_target_zone_tolerance: 0.05
16    orientation_target_zone_tolerance: 0.09
17    translation_target_final_tolerance: 1e-4
18    orientation_target_final_tolerance: 0.005
19    translation_gradient_weight: 1.0
20    orientation_gradient_weight: 0.125
21    nn_translation_distance_weight: 1.0
22    nn_orientation_distance_weight: 0.125
23    task_space_exploitation_fraction: 0.4
24    task_space_exploration_fraction: 0.1
25    max_extension_substeps_away_from_target: 6
26    max_extension_substeps_near_target: 50
27    extension_substep_target_region_scale_factor: 2.0
28    unexploited_nodes_culling_scalar: 1.0
29    gradient_substep_size: 0.025
```

You can reference the `docstring` to the function `RRT.set_param()` in our [API Documentation](../py/source/extensions/isaacsim.robot_motion.motion_generation/docs/index.html) for a description of each parameter.

### RRT Example

The file `/RRT_Example_python/scenario.py` loads the Franka robot and uses `RRT` to move it around obstacles to a target.
Every 60 frames, the planner replans to move to the current target position (if possible). In this example, the planner does
not attempt to plan to the same target multiple times if a failure is encountered. The returned plan will be `None` and no actions will be taken.

```python
  1import os
  2
  3import numpy as np
  4from isaacsim.core.api.objects.cuboid import VisualCuboid
  5from isaacsim.core.prims import SingleArticulation as Articulation
  6from isaacsim.core.prims import SingleXFormPrim as XFormPrim
  7from isaacsim.core.utils.distance_metrics import rotational_distance_angle
  8from isaacsim.core.utils.extensions import get_extension_path_from_name
  9from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, quats_to_rot_matrices
 10from isaacsim.core.utils.stage import add_reference_to_stage
 11from isaacsim.robot_motion.motion_generation import PathPlannerVisualizer, interface_config_loader
 12from isaacsim.robot_motion.motion_generation.lula import RRT
 13from isaacsim.storage.native import get_assets_root_path
 14
 15
 16class FrankaRrtExample:
 17    def __init__(self):
 18        self._rrt = None
 19        self._path_planner_visualizer = None
 20        self._plan = []
 21
 22        self._articulation = None
 23        self._target = None
 24        self._target_position = None
 25
 26        self._frame_counter = 0
 27
 28    def load_example_assets(self):
 29        # Add the Franka and target to the stage
 30        # The position in which things are loaded is also the position in which they
 31
 32        robot_prim_path = "/panda"
 33        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
 34
 35        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
 36        self._articulation = Articulation(robot_prim_path)
 37
 38        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
 39        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
 40        self._target.set_default_state(np.array([0.45, 0.5, 0.7]), euler_angles_to_quats([3 * np.pi / 4, 0, np.pi]))
 41
 42        self._obstacle = VisualCuboid(
 43            "/World/Wall", position=np.array([0.3, 0.6, 0.6]), size=1.0, scale=np.array([0.1, 0.4, 0.4])
 44        )
 45
 46        # Return assets that were added to the stage so that they can be registered with the core.World
 47        return self._articulation, self._target
 48
 49    def setup(self):
 50        # Lula config files for supported robots are stored in the motion_generation extension under
 51        # "/path_planner_configs" and "/motion_policy_configs"
 52        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
 53        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
 54        rrt_config_dir = os.path.join(mg_extension_path, "path_planner_configs")
 55
 56        # Initialize an RRT object
 57        self._rrt = RRT(
 58            robot_description_path=rmp_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
 59            urdf_path=rmp_config_dir + "/franka/lula_franka_gen.urdf",
 60            rrt_config_path=rrt_config_dir + "/franka/rrt/franka_planner_config.yaml",
 61            end_effector_frame_name="right_gripper",
 62        )
 63
 64        # RRT for supported robots can also be loaded with a simpler equivalent:
 65        # rrt_config = interface_config_loader.load_supported_path_planner_config("Franka", "RRT")
 66        # self._rrt = RRT(**rrt_confg)
 67
 68        self._rrt.add_obstacle(self._obstacle)
 69
 70        # Set the maximum number of iterations of RRT to prevent it from blocking Isaac Sim for
 71        # too long.
 72        self._rrt.set_max_iterations(5000)
 73
 74        # Use the PathPlannerVisualizer wrapper to generate a trajectory of ArticulationActions
 75        self._path_planner_visualizer = PathPlannerVisualizer(self._articulation, self._rrt)
 76
 77        self.reset()
 78
 79    def update(self, step: float):
 80        current_target_translation, current_target_orientation = self._target.get_world_pose()
 81        current_target_rotation = quats_to_rot_matrices(current_target_orientation)
 82
 83        translation_distance = np.linalg.norm(self._target_translation - current_target_translation)
 84        rotation_distance = rotational_distance_angle(current_target_rotation, self._target_rotation)
 85        target_moved = translation_distance > 0.01 or rotation_distance > 0.01
 86
 87        if self._frame_counter % 60 == 0 and target_moved:
 88            # Replan every 60 frames if the target has moved
 89            self._rrt.set_end_effector_target(current_target_translation, current_target_orientation)
 90            self._rrt.update_world()
 91            self._plan = self._path_planner_visualizer.compute_plan_as_articulation_actions(max_cspace_dist=0.01)
 92
 93            self._target_translation = current_target_translation
 94            self._target_rotation = current_target_rotation
 95
 96        if self._plan:
 97            action = self._plan.pop(0)
 98            self._articulation.apply_action(action)
 99
100        self._frame_counter += 1
101
102    def reset(self):
103        self._target_translation = np.zeros(3)
104        self._target_rotation = np.eye(3)
105        self._frame_counter = 0
106        self._plan = []
```

The `RRT` class is initialized on lines 51-61. For supported robots, this can be simplified as on lines 63-65. `RRT` is made
aware of the obstacle it needs to watch on line 67. Any time `RRT.update_world()` is called (line 89), it will query the current position
of watched obstacles.

`RRT` outputs sparse plans that, when linearly interpolated, form a collision-free path to the goal position.
As an instance of the `PathPlanner` interface, `RRT` can be passed to a [Path Planner Visualizer](Robot_Simulation.md) to convert its output
to a form that is directly usable by the robot `Articulation` (line 74).

In this example, `RRT` replans every second if the target has been moved. The replanning is performed on lines 88-90.

* First, `RRT` is informed of the new target position.
* Then it is told to query the position of watched obstacles.
* Finally, the `path_planner_visualizer` wrapping `RRT` is used to generate a plan in the form of a list of `ArticulationAction`.

The `max_cspace_dist` argument passed to the `path_planner_visualizer` interpolates the sparse output with a maximum l2 norm of `.01`
between any two commanded robot positions. On every frame, one of the actions in the plan is removed from the plan and sent to the
robot (lines 92-93).

## Current Limitations

### Following a Plan with Exactness

The `PathPlannerVisualizer` class is called a “Visualizer” because it is only meant to give a visualization of an output plan, but it is not likely to be useful
beyond this. By densely linearly interpolating an `RRT` plan, the resulting trajectory is far from time-optimal or smooth. To follow a plan in a
more theoretically sound way, the output of `RRT` can be combined with the `LulaTrajectoryGenerator`. This is demonstrated in the NVIDIA Isaac Sim Path Planning Example
in the **Robotics Examples** tab. You can activate **Robotics Examples** tab from **Windows** > **Examples** > **Robotics Examples**.

## Summary

This tutorial reviews using the `RRT` class to generate a collision-free path through an environment from a starting position to a task-space target.

### Further Learning

To understand the motivation behind the structure and usage of `RRT` in NVIDIA Isaac Sim, reference the [Motion Generation](concepts/index.html#isaac-sim-motion-generation)
page.

---

# Lula Kinematics Solver

This tutorial shows how the [Lula Kinematics Solver](Robot_Simulation.md) class is used to compute forward and inverse kinematics on a robot in NVIDIA Isaac Sim.

## Getting Started

**Prerequisites**

* Complete the [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md) tutorial prior to beginning this tutorial.
* You can reference the [Lula Robot Description and XRDF Editor](Robot_Simulation.md) to understand how to generate your own robot\_description.yaml file to be able to use `LulaKinematicsSolver` on unsupported robots.
* Review the [Loaded Scenario Extension Template](Templates.md) to understand how this tutorial is structured and run.

To follow along with the tutorial, you can search and enable the **Motion Generation Tutorials** extension within your running Isaac Sim 6.0 instance.
Within the isaacsim.robot\_motion.motion\_generation.tutorials extension, there is a fully functional example using a `LulaKinematicsSolver` to track a task-space target.
The sections of this tutorial build up the file `scenario.py` from basic functionality to the completed code.

## Using the LulaKinematicsSolver to Compute Forward and Inverse Kinematics

The [Lula Kinematics Solver](Robot_Simulation.md) is able to calculate forward and inverse kinematics for a robot that is defined
by two configuration files (see [Lula Kinematics Solver Configuration](Robot_Simulation.md)). The `LulaKinematicsSolver` can be paired with
an [Articulation Kinematics Solver](Robot_Simulation.md) to compute kinematics in a way that can be directly applied to the robot `Articulation`.

The file `/Lula_Kinematics_python/scenario.py` uses the `LulaKinematicsSolver` to generate inverse kinematic solutions to move the robot to a target.

```python
import os

import carb
import numpy as np
from isaacsim.core.prims import SingleArticulation as Articulation
from isaacsim.core.prims import SingleXFormPrim as XFormPrim
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
    interface_config_loader,
)
from isaacsim.storage.native import get_assets_root_path

class FrankaKinematicsExample:
    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None

        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Add the Franka and target to the stage

        robot_prim_path = "/panda"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        add_reference_to_stage(get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        self._target.set_default_state(np.array([0.3, 0, 0.5]), euler_angles_to_quats([0, np.pi, 0]))

        # Return assets that were added to the stage so that they can be registered with the core.World
        return self._articulation, self._target

    def setup(self):
        # Load a URDF and Lula Robot Description File for this robot:
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=kinematics_config_dir + "/franka/rmpflow/robot_descriptor.yaml",
            urdf_path=kinematics_config_dir + "/franka/lula_franka_gen.urdf",
        )

        # Kinematics for supported robots can be loaded with a simpler equivalent
        # print("Supported Robots with a Lula Kinematics Config:", interface_config_loader.get_supported_robots_with_lula_kinematics())
        # kinematics_config = interface_config_loader.load_supported_lula_kinematics_solver_config("Franka")
        # self._kinematics_solver = LulaKinematicsSolver(**kinematics_config)

        print("Valid frame names at which to compute kinematics:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = "right_gripper"
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, self._kinematics_solver, end_effector_name
        )

    def update(self, step: float):
        target_position, target_orientation = self._target.get_world_pose()

        # Track any movements of the robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if success:
            self._articulation.apply_action(action)
        else:
            carb.log_warn("IK did not converge to a solution.  No action is being taken")

        # Unused Forward Kinematics:
        # ee_position,ee_rot_mat = articulation_kinematics_solver.compute_end_effector_pose()

    def reset(self):
        # Kinematics is stateless
        pass
```

The `LulaKinematicsSolver` is instantiated on lines 41-47 using file paths to the appropriate configuration files. The
`LulaKinematicsSolver` uses the same robot description files as the Lula-based [RMPflow](Robot_Simulation.md) [Motion Policy Algorithm](Robot_Simulation.md).
The `LulaKinematicsSolver` can solve forward and inverse kinematics at any frame that exists in the robot URDF file.
On line 54, the complete list of recognized frames in the Franka robot is printed:

```python
Valid frame names at which to compute kinematics:
['base_link', 'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_forearm_end_pt', 'panda_forearm_mid_pt',
 'panda_forearm_mid_pt_shifted', 'panda_link5', 'panda_forearm_distal', 'panda_link6', 'panda_link7', 'panda_link8', 'panda_hand',
 'camera_bottom_screw_frame', 'camera_link', 'camera_depth_frame', 'camera_color_frame', 'camera_color_optical_frame', 'camera_depth_optical_frame',
 'camera_left_ir_frame', 'camera_left_ir_optical_frame', 'camera_right_ir_frame', 'camera_right_ir_optical_frame', 'panda_face_back_left',
 'panda_face_back_right', 'panda_face_left', 'panda_face_right', 'panda_leftfinger', 'panda_leftfingertip', 'panda_rightfinger', 'panda_rightfingertip', 'right_gripper', 'panda_wrist_end_pt']
```

Supported robots can be loaded directly by name as on lines 50-52. This is equivalent to lines 41-47.

On line 57, an [Articulation Kinematics Solver](Robot_Simulation.md) is instantiated with the Franka robot `Articulation`, the `LulaKinematicsSolver` instance,
and the end effector name. The `ArticulationKinematicsSolver` class allows you to
compute the end effector position and orientation for the robot `Articulation` in a single line (line 75).

The `ArticulationKinematicsSolver` also allows you to compute inverse kinematics.
The current position of the robot `Articulation` is used as a warm start in the IK calculation,
and the result is returned as an `ArticulationAction` that can be consumed by the robot `Articulation`
to move the specified end effector frame to a target position (lines 67 and 70).

The `LulaKinematicsSolver` returns a flag marking the success or failure of the inverse kinematics computation. On line
67, the script applies the inverse kinematics solution to the robot `Articulation` only if the kinematics converged
successfully to a solution, otherwise no new action is sent to the robot,
and a warning is thrown. The `LulaKinematicsSolver` exposes
settings that allow you to specify how quickly it terminates its search. These settings are outside the
scope of this tutorial.

The `LulaKinematicsSolver` assumes that the robot base is positioned at the origin unless another location is specified. On lines 64-65,
the `LulaKinematicsSolver` is given the current position of the robot base on every frame. This allows the forward
and inverse kinematics to operate using world coordinates. For example, the position of the target is queried in world
coordinates and passed to the `LulaKinematicsSolver`, which internally performs the necessary transformation to compute
accurate inverse kinematics.

The `LulaKinematicsSolver` can be used on its own to compute forward kinematics at any position and to compute
inverse kinematics with any warm start. A robot `Articulation` does not need to be present on the USD stage. See [Kinematics Solvers](Robot_Simulation.md) for more details.

Additionally, sending an inverse kinematic solution directly to the robot is not likely to be useful beyond demonstrations.
In a realistic scenario, you need to determine not only the end position of the robot, but also the path to get there. An IK solver on its own can make
for only a rudimentary trajectory through space that is not likely to be optimal.

## Summary

This tutorial reviews how to load the `LulaKinematicsSolver` class and use it alongside the `ArticulationKinematicsSolver`
helper class to compute forward and inverse kinematics at any frame specified in the robot URDF file.

### Further Learning

To understand the motivation behind the structure and usage of `LulaKinematicsSolver` in NVIDIA Isaac Sim, reference the [Motion Generation](concepts/index.html#isaac-sim-motion-generation)
page.

---

# Lula Trajectory Generator

This tutorial explores how the [Lula Trajectory Generator](Robot_Simulation.md) in the [Motion Generation](concepts/index.html#isaac-sim-motion-generation) extension can be used to create both task-space and c-space trajectories that can be easily applied to a simulated robot `Articulation`.

## Getting Started

**Prerequisites**

* Complete the [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md) tutorial prior to beginning this tutorial.
* You can reference the [Lula Robot Description and XRDF Editor](Robot_Simulation.md) to understand how to generate your own `robot_description.yaml` file to be able to use the [Lula Trajectory Generator](Robot_Simulation.md) on unsupported robots.
* Review the [Loaded Scenario Extension Template](Templates.md) to understand how this tutorial is structured and run.

To follow along with the tutorial, you can search and enable the **Motion Generation Tutorials** extension within your running Isaac Sim 6.0 instance.
Within the isaacsim.robot\_motion.motion\_generation.tutorials extension, there an example of the `LulaTaskSpaceTrajectorygenerator` and `LulaCSpaceTrajectoryGenerator` being used to generate trajectories
connecting specified c-space and task-space points.
The sections of this tutorial build up the file `scenario.py` from basic functionality to the completed code.

## Generating a C-Space Trajectory

The `LulaCSpaceTrajectoryGenerator` class is able to generate a trajectory that connects a provided set of c-space waypoints.
The code snippet below demonstrates how, given appropriate config files,
the `LulaCSpaceTrajectoryGenerator` class can be initialized and used to create a sequence
of `ArticulationAction` that can be set on each frame to produce the desired trajectory.

The code snippet below shows the relevant contents of `/Trajectory_Generator_python/scenario.py` from the provided example.

```python
  1import os
  2
  3import carb
  4import lula
  5import numpy as np
  6from isaacsim.core.api.objects.cuboid import FixedCuboid
  7from isaacsim.core.prims import SingleArticulation as Articulation
  8from isaacsim.core.prims import SingleXFormPrim as XFormPrim
  9from isaacsim.core.utils.extensions import get_extension_path_from_name
 10from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
 11from isaacsim.core.utils.prims import delete_prim, get_prim_at_path
 12from isaacsim.core.utils.stage import add_reference_to_stage
 13from isaacsim.robot_motion.motion_generation import (
 14    ArticulationTrajectory,
 15    LulaCSpaceTrajectoryGenerator,
 16    LulaKinematicsSolver,
 17    LulaTaskSpaceTrajectoryGenerator,
 18)
 19from isaacsim.storage.native import get_assets_root_path
 20
 21
 22class UR10TrajectoryGenerationExample:
 23    def __init__(self):
 24        self._c_space_trajectory_generator = None
 25        self._taskspace_trajectory_generator = None
 26        self._kinematics_solver = None
 27
 28        self._action_sequence = []
 29        self._action_sequence_index = 0
 30
 31        self._articulation = None
 32
 33    def load_example_assets(self):
 34        # Add the Franka and target to the stage
 35        # The position in which things are loaded is also the position in which they
 36
 37        robot_prim_path = "/ur10"
 38        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"
 39
 40        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
 41        self._articulation = Articulation(robot_prim_path)
 42
 43        # Return assets that were added to the stage so that they can be registered with the core.World
 44        return [self._articulation]
 45
 46    def setup(self):
 47        # Config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
 48        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
 49        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")
 50
 51        # Initialize a LulaCSpaceTrajectoryGenerator object
 52        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
 53            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
 54            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
 55        )
 56
 57        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
 58            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
 59            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
 60        )
 61
 62        self._kinematics_solver = LulaKinematicsSolver(
 63            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
 64            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
 65        )
 66
 67        self._end_effector_name = "ee_link"
 68
 69    def setup_cspace_trajectory(self):
 70        c_space_points = np.array(
 71            [
 72                [
 73                    -0.41,
 74                    0.5,
 75                    -2.36,
 76                    -1.28,
 77                    5.13,
 78                    -4.71,
 79                ],
 80                [
 81                    -1.43,
 82                    1.0,
 83                    -2.58,
 84                    -1.53,
 85                    6.0,
 86                    -4.74,
 87                ],
 88                [
 89                    -2.83,
 90                    0.34,
 91                    -2.11,
 92                    -1.38,
 93                    1.26,
 94                    -4.71,
 95                ],
 96                [
 97                    -0.41,
 98                    0.5,
 99                    -2.36,
100                    -1.28,
101                    5.13,
102                    -4.71,
103                ],
104            ]
105        )
106
107        timestamps = np.array([0, 5, 10, 13])
108
109        trajectory_time_optimal = self._c_space_trajectory_generator.compute_c_space_trajectory(c_space_points)
110        trajectory_timestamped = self._c_space_trajectory_generator.compute_timestamped_c_space_trajectory(
111            c_space_points, timestamps
112        )
113
114        # Visualize c-space targets in task space
115        for i, point in enumerate(c_space_points):
116            position, rotation = self._kinematics_solver.compute_forward_kinematics(self._end_effector_name, point)
117            add_reference_to_stage(
118                get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}"
119            )
120            frame = XFormPrim(f"/visualized_frames/target_{i}", scale=[0.04, 0.04, 0.04])
121            frame.set_world_pose(position, rot_matrices_to_quats(rotation))
122
123        if trajectory_time_optimal is None or trajectory_timestamped is None:
124            carb.log_warn("No trajectory could be computed")
125            self._action_sequence = []
126        else:
127            physics_dt = 1 / 60
128            self._action_sequence = []
129
130            # Follow both trajectories in a row
131
132            articulation_trajectory_time_optimal = ArticulationTrajectory(
133                self._articulation, trajectory_time_optimal, physics_dt
134            )
135            self._action_sequence.extend(articulation_trajectory_time_optimal.get_action_sequence())
136
137            articulation_trajectory_timestamped = ArticulationTrajectory(
138                self._articulation, trajectory_timestamped, physics_dt
139            )
140            self._action_sequence.extend(articulation_trajectory_timestamped.get_action_sequence())
141
142    def update(self, step: float):
143        if len(self._action_sequence) == 0:
144            return
145
146        if self._action_sequence_index >= len(self._action_sequence):
147            self._action_sequence_index += 1
148            self._action_sequence_index %= (
149                len(self._action_sequence) + 10
150            )  # Wait 10 frames before repeating trajectories
151            return
152
153        if self._action_sequence_index == 0:
154            self._teleport_robot_to_position(self._action_sequence[0])
155
156        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])
157
158        self._action_sequence_index += 1
159        self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories
160
161    def reset(self):
162        # Delete any visualized frames
163        if get_prim_at_path("/visualized_frames"):
164            delete_prim("/visualized_frames")
165
166        self._action_sequence = []
167        self._action_sequence_index = 0
168
169    def _teleport_robot_to_position(self, articulation_action):
170        initial_positions = np.zeros(self._articulation.num_dof)
171        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions
172
173        self._articulation.set_joint_positions(initial_positions)
174        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))
```

On lines 53-56, the `LulaCSpaceTrajectoryGenerator` class is initialized using a URDF and
[Lula Robot Description File](Robot_Simulation.md).
The `LulaCSpaceTrajectoryGenerator` takes in a series of waypoints, and it connects them in configuration space using spline-based interpolation.
There are two main objectives that can be fulfilled by the trajectory generator:

* time-optimal
* time-stamped

The provided example shows a trajectory that runs quickly, and then runs slowly. This is seen in the code on lines (80-81 and 99-103).
On line 80, a time-optimal trajectory is created in the form of a `LulaTrajectory` object, which fulfills the [Trajectory Interface](Robot_Simulation.md).
On line 81, a time-stamped trajectory is created that will hit the same waypoints at the times `[0,5,10,13]` seconds (line 78). Time optimality is
defined as saturating at least one of velocity, acceleration, or jerk limits of the robot throughout a trajectory.

On lines 99-103, These `LulaTrajectory` objects are passed to `ArticulationTrajectory` to generate a sequence of `ArticulationAction` that can be passed directly to the
robot `Articulation`. The function `ArticulationTrajectory.get_action_sequence()` returns a list of `ArticulationAction` that is meant to be consumed at the specified
rate. In this case, the framerate of physics is assumed to be fixed at `1/60` seconds.

If no trajectory can be computed that connects the c-space waypoints, the trajectory returned by `LulaCSpaceTrajectoryGenerator.compute_c_space_trajectory`
will be `None`. This can occur when one of the specified c-space waypoints is not reachable or is very close to a joint limit.
This case is handled on lines 90-92.

On lines 84-88, a visualization of the original `c_space_points` is created by converting them to task-space points.
This code is not functional, but it helps to verify that the robot is hitting every target.

The `update()` function is programmed to play the sequence of `ArticulationActions` in a loop, taking a pause of `10 frames` for dramatic effect between trajectories.

## Generating a Task-Space Trajectory

### Simple Case: Linearly Connecting Waypoints

Generating a task-space trajectory is similar to generating a c-space trajectory.
In the simplest use-case, you can pass in a set of task-space position and quaternion orientation targets,
which will be linearly interpolated in task-space to produce the resulting trajectory.
An example is provided in the code snippet below:

```python
class UR10TrajectoryGenerationExample:
    def __init__(self):
        self._c_space_trajectory_generator = None
        self._taskspace_trajectory_generator = None
        self._kinematics_solver = None

        self._action_sequence = []
        self._action_sequence_index = 0

        self._articulation = None

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/ur10"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        # Return assets that were added to the stage so that they can be registered with the core.World
        return [self._articulation]

    def setup(self):
        # Config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        # Initialize a LulaCSpaceTrajectoryGenerator object
        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
        )

        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
        )

        self._end_effector_name = "ee_link"

    def setup_taskspace_trajectory(self):
        task_space_position_targets = np.array(
            [[0.3, -0.3, 0.1], [0.3, 0.3, 0.1], [0.3, 0.3, 0.5], [0.3, -0.3, 0.5], [0.3, -0.3, 0.1]]
        )

        task_space_orientation_targets = np.tile(np.array([0, 1, 0, 0]), (5, 1))

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_points(
            task_space_position_targets, task_space_orientation_targets, self._end_effector_name
        )

        # Visualize task-space targets in task space
        for i, (position, orientation) in enumerate(zip(task_space_position_targets, task_space_orientation_targets)):
            add_reference_to_stage(
                get_assets_root_path() + "/Isaac/Props/UIElements/frame_prim.usd", f"/visualized_frames/target_{i}"
            )
            frame = XFormPrim(f"/visualized_frames/target_{i}", scale=[0.04, 0.04, 0.04])
            frame.set_world_pose(position, orientation)

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)

            # Get a sequence of ArticulationActions that are intended to be passed to the robot at 1/60 second intervals
            self._action_sequence = articulation_trajectory.get_action_sequence()

    def update(self, step: float):
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= (
                len(self._action_sequence) + 10
            )  # Wait 10 frames before repeating trajectories
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])

        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories

    def reset(self):
        # Delete any visualized frames
        if get_prim_at_path("/visualized_frames"):
            delete_prim("/visualized_frames")

        self._action_sequence = []
        self._action_sequence_index = 0

    def _teleport_robot_to_position(self, articulation_action):
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))
```

In moving to the task-space trajectory generator, there are few code changes required. The initialization is nearly the same on line 36 as for the
c-space trajectory generator. The main changes are on lines 59-61 where a task-space trajectory is specified. When using the function
`LulaTaskSpaceTrajectoryGenerator.compute_task_space_trajectory_from_points`, a position and orientation target must be specified for each task-space waypoint.
Additionally, a frame from the robot URDF must be specified as the end effector frame.
If the waypoints cannot be connected to form a trajectory, the `compute_task_space_trajectory_from_points` function will return `None`.
This case is checked on line 69.

### Defining Complicated Trajectories

The `LulaTaskSpaceTrajectoryGenerator` can be used to create paths with more complicated specifications than to connect a set of task-space targets linearly.
Using the class `lula.TaskSpacePathSpec`, you can define paths with arcs and circles with multiple options for orientation targets.
The code snippet below demonstrates creating a `lula.TaskSpacePathSpec` and gives an example of each available function for adding to a task-space path.
Additionally, it shows how a `lula.TaskSpacePathSpec` can be combined with a `lula.CSpacePathSpec` in a `lula.CompositePathSpec` to specify trajectories
that contain both c-space and task-space waypoints.

```python
class UR10TrajectoryGenerationExample:
    def __init__(self):
        self._c_space_trajectory_generator = None
        self._taskspace_trajectory_generator = None
        self._kinematics_solver = None

        self._action_sequence = []
        self._action_sequence_index = 0

        self._articulation = None

    def load_example_assets(self):
        # Add the Franka and target to the stage
        # The position in which things are loaded is also the position in which they

        robot_prim_path = "/ur10"
        path_to_robot_usd = get_assets_root_path() + "/Isaac/Robots/UniversalRobots/ur10/ur10.usd"

        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        # Return assets that were added to the stage so that they can be registered with the core.World
        return [self._articulation]

    def setup(self):
        # Config files for supported robots are stored in the motion_generation extension under "/motion_policy_configs"
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        rmp_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        # Initialize a LulaCSpaceTrajectoryGenerator object
        self._c_space_trajectory_generator = LulaCSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
        )

        self._taskspace_trajectory_generator = LulaTaskSpaceTrajectoryGenerator(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
        )

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=rmp_config_dir + "/universal_robots/ur10/rmpflow/ur10_robot_description.yaml",
            urdf_path=rmp_config_dir + "/universal_robots/ur10/ur10_robot.urdf",
        )

        self._end_effector_name = "ee_link"

    def setup_advanced_trajectory(self):
        # The following code demonstrates how to specify a complicated cspace and taskspace path
        # using the lula.CompositePathSpec object

        initial_c_space_robot_pose = np.array([0, 0, 0, 0, 0, 0])

        # Combine a cspace and taskspace trajectory
        composite_path_spec = lula.create_composite_path_spec(initial_c_space_robot_pose)

        #############################################################################
        # Demonstrate all the available movements in a taskspace path spec:

        # Lula has its own classes for Rotations and 6 DOF poses: Rotation3 and Pose3
        r0 = lula.Rotation3(np.pi / 2, np.array([1.0, 0.0, 0.0]))
        t0 = np.array([0.3, -0.1, 0.3])
        task_space_spec = lula.create_task_space_path_spec(lula.Pose3(r0, t0))

        # Add path linearly interpolating between r0,r1 and t0,t1
        t1 = np.array([0.3, -0.1, 0.5])
        r1 = lula.Rotation3(np.pi / 3, np.array([1, 0, 0]))
        task_space_spec.add_linear_path(lula.Pose3(r1, t1))

        # Add pure translation.  Constant rotation is assumed
        task_space_spec.add_translation(t0)

        # Add pure rotation.
        task_space_spec.add_rotation(r0)

        # Add three-point arc with constant orientation.
        t2 = np.array(
            [
                0.3,
                0.3,
                0.3,
            ]
        )
        midpoint = np.array([0.3, 0, 0.5])
        task_space_spec.add_three_point_arc(t2, midpoint, constant_orientation=True)

        # Add three-point arc with tangent orientation.
        task_space_spec.add_three_point_arc(t0, midpoint, constant_orientation=False)

        # Add three-point arc with orientation target.
        task_space_spec.add_three_point_arc_with_orientation_target(lula.Pose3(r1, t2), midpoint)

        # Add tangent arc with constant orientation. Tangent arcs are circles that connect two points
        task_space_spec.add_tangent_arc(t0, constant_orientation=True)

        # Add tangent arc with tangent orientation.
        task_space_spec.add_tangent_arc(t2, constant_orientation=False)

        # Add tangent arc with orientation target.
        task_space_spec.add_tangent_arc_with_orientation_target(lula.Pose3(r0, t0))

        ###################################################
        # Demonstrate the usage of a c_space path spec:
        c_space_spec = lula.create_c_space_path_spec(np.array([0, 0, 0, 0, 0, 0]))

        c_space_spec.add_c_space_waypoint(np.array([0, 0.5, -2.0, -1.28, 5.13, -4.71]))

        ##############################################################
        # Combine the two path specs together into a composite spec:

        # specify how to connect initial_c_space and task_space points with transition_mode option
        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_task_space_path_spec(task_space_spec, transition_mode)

        transition_mode = lula.CompositePathSpec.TransitionMode.FREE
        composite_path_spec.add_c_space_path_spec(c_space_spec, transition_mode)

        # Transition Modes:
        # lula.CompositePathSpec.TransitionMode.LINEAR_TASK_SPACE:
        #      Connect cspace to taskspace points linearly through task space.  This mode is only available when adding a task_space path spec.
        # lula.CompositePathSpec.TransitionMode.FREE:
        #      Put no constraints on how cspace and taskspace points are connected
        # lula.CompositePathSpec.TransitionMode.SKIP:
        #      Skip the first point of the path spec being added, using the last pose instead

        trajectory = self._taskspace_trajectory_generator.compute_task_space_trajectory_from_path_spec(
            composite_path_spec, self._end_effector_name
        )

        if trajectory is None:
            carb.log_warn("No trajectory could be computed")
            self._action_sequence = []
        else:
            physics_dt = 1 / 60
            articulation_trajectory = ArticulationTrajectory(self._articulation, trajectory, physics_dt)

            # Get a sequence of ArticulationActions that are intended to be passed to the robot at 1/60 second intervals
            self._action_sequence = articulation_trajectory.get_action_sequence()

    def update(self, step: float):
        if len(self._action_sequence) == 0:
            return

        if self._action_sequence_index >= len(self._action_sequence):
            self._action_sequence_index += 1
            self._action_sequence_index %= (
                len(self._action_sequence) + 10
            )  # Wait 10 frames before repeating trajectories
            return

        if self._action_sequence_index == 0:
            self._teleport_robot_to_position(self._action_sequence[0])

        self._articulation.apply_action(self._action_sequence[self._action_sequence_index])

        self._action_sequence_index += 1
        self._action_sequence_index %= len(self._action_sequence) + 10  # Wait 10 frames before repeating trajectories

    def reset(self):
        # Delete any visualized frames
        if get_prim_at_path("/visualized_frames"):
            delete_prim("/visualized_frames")

        self._action_sequence = []
        self._action_sequence_index = 0

    def _teleport_robot_to_position(self, articulation_action):
        initial_positions = np.zeros(self._articulation.num_dof)
        initial_positions[articulation_action.joint_indices] = articulation_action.joint_positions

        self._articulation.set_joint_positions(initial_positions)
        self._articulation.set_joint_velocities(np.zeros_like(initial_positions))
```

The code snippet above creates a `lula.CompositePathSpec` on line 55 with an initial c-space position. It is combined with a
`lula.TaskSpacePathSpec` on lines 108-109 and it is combined with a `lula.CSpacePathSpec` on lines 111-112. The resulting path
is one that starts at the specified `initial_c_space_robot_pose`, then follows a series of taskspace targets, then hits two c-space
targets. When combining path specs, a transition mode must be specified to determine how c-space and task-space points should be connected
to each other. Reference lines 114-120 to see the possible options. In this case, no constraint is made on how the `LulaTrajectoryGenerator`
connects these points.

Each available option for specifying a `lula.TaskSpacePathSpec` is demonstrated between lines 63-94.
The code snippet above moves mainly between three translations: `t0, t1, t2` with possible rotations `r0, r1`.
The `lula.TaskSpacePathSpec` object is created with an initial position on line 63.
Each following `add` function that is called adds a path between the last position in the `path_spec` so far and a new position.
The basic possibilities are:

> 1. Linearly interpolate translation to a new point while keeping rotation fixed (line 71)
> 2. Linearly interpolate rotation to a new point while keeping translation fixed (line 74)
> 3. Linearly interpolate both rotation and translation to a new 6 DOF point (line 68)

The `lula.TaskSpacePathSpec` also makes it easy to define various arcs and circular paths that connect points in space.
A three-point arc can be defined that moves through a midpoint to a translation target.
There are three options for the orientation of the robot while moving along the path:

> 1. Keep rotation constant (line 79)
> 2. Always stay oriented tangent to the arc (line 82)
> 3. Linearly interpolate rotation to a rotation target (line 85)

Finally, a circular path can be specified without defining a midpoint as on lines 88, 91, and 94.
The same three options for specifying orientation are available.

## Summary

This tutorial shows how to use the [Lula Trajectory Generator](Robot_Simulation.md) to generate c-space and task-space trajectories for a robot. Task-space trajectories can be specified using a series of task-space waypoints that will be connected linearly, or they can be defined piecewise with many different options for connecting each pair of points in space.

### Further Learning

Reference the [Motion Generation](concepts/index.html#isaac-sim-motion-generation) page for a complete description of trajectories in NVIDIA Isaac Sim.

---

# Configuring RMPflow for a New Manipulator

In this tutorial, you learn how the [RMPflow](Robot_Simulation.md) algorithm can be fully configured following the creation of a Robot Description File.

## Getting Started

**Prerequisites**

* **Complete** the [Lula Robot Description and XRDF Editor](Robot_Simulation.md) tutorial to create one of the two required configuration files for using RmpFlow.
* Review the [Tutorial 7: Configure a Manipulator](Robot_Setup.md) prior to beginning this tutorial to obtain a robot Articulation USD asset.

This tutorial provides a URDF file and USD file describing the **Cobotta Pro 900** robot. The USD file was generated from the URDF using the process discussed in [Tutorial 6: Setup a Manipulator](Robot_Setup.md).

You can download the following zip file to follow along with the tutorial:

[`Cobotta_Pro_900_Tutorial_Assets`](../_downloads/43f1f07841f3ef71cc54f320218ced44/Cobotta_Pro_900_Assets.zip)

## Using the Lula Test Widget

This tutorial demonstrates how RMPflow can be configured and tested on a new robot using the Lula Test Widget. The Lula Test Widget is an extension that can
be enabled in the Extensions menu as shown below, and then accessed under **Tools > Robotics > Lula Test Widget**.

This extension allows you to select their RMPflow config files along with a selected robot Articulation on the USD stage and run scenarios to verify that
RMPflow is working as intended. After each type of required config file is created, they can be loaded and used in the Lula Test Widget.

## Template RmpFlow Config File

There are three files to describe the robot and parameterize the
[RMPflow](Robot_Simulation.md) algorithm:

> * A **URDF** (universal robot description file), used for specifying robot kinematics
>   :   as well as joint and link names. Position limits for each joint are also required.
>       Other properties in the URDF are ignored and can be omitted; these include masses,
>       moments of inertia, visual and collision meshes.
> * A **supplementary robot description file** in YAML format. This file can be generated using the Lula Robot Description Editor UI tool.
> * A **RMPflow configuration file** in YAML format, containing parameters for all enabled RMPs.

This tutorial assumes that you is starting with a URDF file describing their robot and has created a Robot Description File using the
[Lula Robot Description and XRDF Editor](Robot_Simulation.md).
In this tutorial, a template files is provided for the remaining RMPflow configuration, which is
modified to match the **Cobotta Pro 900** robot. The provided tutorial zip file contains a completed robot\_description.yaml for the **Cobotta Pro 900**.

### Template RmpFlow Config YAML File

The RMPflow algorithm has over 50 settable parameters, but these parameters tend to generalize between robots with similar kinematic structures
and length scales. The values in the template have been tuned specifically for the Franka Emika Panda,
but serve as a good starting point for many 6- and 7-dof robot arms. The template file can be found in
the provided tutorial zip at ./rmpflow\_configs/template\_rmpflow\_config.yaml.

```python
 1# Artificially limit the robot joints.  For example:
 2# A joint with range +-pi would be limited to +-(pi-.01)
 3joint_limit_buffers: [.01, .01, .01, .01, .01, .01, .01]
 4
 5# RMPflow has many modifiable parameters, but these serve as a great start.
 6# Most parameters will not need to be modified
 7rmp_params:
 8    cspace_target_rmp:
 9        metric_scalar: 50.
10        position_gain: 100.
11        damping_gain: 50.
12        robust_position_term_thresh: .5
13        inertia: 1.
14    cspace_trajectory_rmp:
15        p_gain: 100.
16        d_gain: 10.
17        ff_gain: .25
18        weight: 50.
19    cspace_affine_rmp:
20        final_handover_time_std_dev: .25
21        weight: 2000.
22    joint_limit_rmp:
23        metric_scalar: 1000.
24        metric_length_scale: .01
25        metric_exploder_eps: 1e-3
26        metric_velocity_gate_length_scale: .01
27        accel_damper_gain: 200.
28        accel_potential_gain: 1.
29        accel_potential_exploder_length_scale: .1
30        accel_potential_exploder_eps: 1e-2
31    joint_velocity_cap_rmp:
32        max_velocity: 4.
33        velocity_damping_region: 1.5
34        damping_gain: 1000.0
35        metric_weight: 100.
36    target_rmp:
37        accel_p_gain: 30.
38        accel_d_gain: 85.
39        accel_norm_eps: .075
40        metric_alpha_length_scale: .05
41        min_metric_alpha: .01
42        max_metric_scalar: 10000
43        min_metric_scalar: 2500
44        proximity_metric_boost_scalar: 20.
45        proximity_metric_boost_length_scale: .02
46        xi_estimator_gate_std_dev: 20000.
47        accept_user_weights: false
48    axis_target_rmp:
49        accel_p_gain: 210.
50        accel_d_gain: 60.
51        metric_scalar: 10
52        proximity_metric_boost_scalar: 3000.
53        proximity_metric_boost_length_scale: .08
54        xi_estimator_gate_std_dev: 20000.
55        accept_user_weights: false
56    collision_rmp:
57        damping_gain: 50.
58        damping_std_dev: .04
59        damping_robustness_eps: 1e-2
60        damping_velocity_gate_length_scale: .01
61        repulsion_gain: 800.
62        repulsion_std_dev: .01
63        metric_modulation_radius: .5
64        metric_scalar: 10000.
65        metric_exploder_std_dev: .02
66        metric_exploder_eps: .001
67    damping_rmp:
68        accel_d_gain: 30.
69        metric_scalar: 50.
70        inertia: 100.
71
72canonical_resolve:
73    max_acceleration_norm: 50.
74    projection_tolerance: .01
75    verbose: false
76
77
78# body_cylinders are used to promote self-collision avoidance between the robot and its base
79# The example below defines the robot base to be a capsule defined by the absolute coordinates pt1 and pt2.
80# The semantic name provided for each body_cylinder does not need to be present in the robot URDF.
81body_cylinders:
82     - name: base
83       pt1: [0,0,.333]
84       pt2: [0,0,0.]
85       radius: .05
86
87
88# body_collision_controllers defines spheres located at specified frames in the robot URDF
89# These spheres will not be allowed to collide with the capsules enumerated under body_cylinders
90# By design, most frames in industrial robots are kinematically unable to collide with the robot base.
91# It is often only necessary to define body_collision_controllers near the end effector
92body_collision_controllers:
93     - name: end_effector
94       radius: .05
```

This tutorial focuses on three fields in this file:

* `joint_limit_buffers` introduces artificial joint limits around the joint limits stated in the robot URDF. The shape of the provided `joint_limit_buffers` must match the c-space given in the `robot_description.yaml` file. Imagining that the template robot has seven revolute joints, the given buffers of .01 on the seven c-space joints mean that RMPflow will drive the robot up to .01 radians from the joint limits given in the robot URDF. If the robot has prismatic joints, a value of .01 would be expressed implicitly in meters.
* `body_cylinders` and `body_collision_controllers` help RMPflow to avoid self-collision between the end effector and the robot base. `body_cylinders` define an imagined robot base using a set of capsules.
* `body_collision_controllers` define collision spheres placed on different frames of the robot URDF. The template code above defines an unmoving capsule in absolute coordinates and a sphere centered around the “end\_effector” frame in the robot URDF. RMPflow will not allow a collision between the sphere and capsule.

Apart from preventing the end effector from colliding with the base, RMPflow does not directly avoid self-collisions based on collision geometry.

For most applications, however, joint limits are sufficient to prevent links in the middle of the kinematic chain from colliding with each other.

## Modifying the Template for the Cobotta Pro 900

### Doing the Bare Minimum

The minimum changes required to get the **Cobotta** to be able to use RMPflow to follow a target.

The `rmpflow_config` file requires little work to get started (./rmpflow\_configs/rmpflow\_config\_basic.yaml in the provided tutorial zip):

```python
 1# Artificially limit the robot joints.  For example:
 2# A joint with range +-pi would be limited to +-(pi-.01)
 3joint_limit_buffers: [.01, .01, .01, .01, .01, .01]
 4
 5#Omitting `rmp_params` argument
 6
 7# body_cylinders are used to promote self-collision avoidance between the robot and its base
 8# The example below defines the robot base to be a capsule defined by the absolute coordinates pt1 and pt2.
 9# The semantic name provided for each body_cylinder does not need to be present in the robot URDF.
10body_cylinders:
11     - name: base
12       pt1: [0,0,.333]
13       pt2: [0,0,0.]
14       radius: .05
15
16
17# body_collision_controllers defines spheres located at specified frames in the robot URDF
18# These spheres will not be allowed to collide with the capsules enumerated under body_cylinders
19# By design, most frames in industrial robots are kinematically unable to collide with the robot base.
20# It is often only necessary to define body_collision_controllers near the end effector
21body_collision_controllers:
22     - name: right_inner_finger
23       radius: .05
```

To get the robot moving around, you can ignore the `rmp_params` argument for now. Modify the `joint_limit_buffers`
argument to represent that the robot only has six DOFs rather than the seven listed in the template. You have to provide
`body_cylinders`, you will represent the robot base later. One change was
required to the default `body_collision_controllers` argument, that was to change the frame at which you place a collision
sphere. There is no `end_effector` frame in the **Cobotta** URDF, so for now pick a frame that is near the end effector:
`right_inner_finger`.

In the Lula Test Widget, observe that the robot is able to follow the target and avoid obstacles.
Notice that the frame that RMPflow is moving to the target position is not in the center of the gripper. In the Lula Test Widget, the
`right_inner_finger` is selected as the end effector frame. The available end effector frames come from the robot URDF file, and there is not a frame resting
in the center of the gripper.

### Avoiding Self-Collision: Configuring Body Cylinders and Body Collision Controllers

With a completed Robot Description File, the robot will avoid collisions with external obstacles, but it will not avoid self-collision.
There is limited tooling available for avoiding self-collision because industrial robot arms typically remove most potential for self-collision
with joint limits. However, some exploration is required with a particular robot to learn what types of self-collision are possible.
With the preliminary configuration of body cylinders and body collision controllers. You set in the `Cobotta_Pro_900_Assets/rmpflow_configs/cobotta_rmpflow_config_basic.yaml` file,
it is easy to cause collisions between the robot end effector and the robot base.

`body_cylinders` define an imagined robot base using a set of capsules. `body_collision_controllers` define collision spheres placed
on different frames of the robot URDF.

RMPflow will not allow these spheres and capsules to come into contact with each other. In the basic `rmpflow` config, you defined the base as the capsule
connecting two spheres of radius `.05 m` at the absolute coordinates ([0,0,0], [0,0,.333]) (refer to [Doing the Bare Minimum](#isaac-sim-tutorial-configure-rmpflow-bare-minimum)),
and you define a single `body_collision_controller` at the `right_inner_finger` frame.

In the video above, you observe that the gripper will not pass directly
through the robot base, but it is easy to facilitate a self-collision with the edge of the robot base, or the base of the second link.

The self-collision tooling available in RMPflow does not allow you to avoid all self-collisions without sacrificing some acceptable robot configurations as well.

To make self collisions completely impossible for the **Cobotta**, you need a very conservative estimate of the robot base.
You would not allow the gripper to move close to the base at all. Choosing the best possible configuration is use-case dependent.
There is no reason to take away maneuverability around the robot base unless you observe that the robot is self-colliding.

One potential configuration in this tutorial covers the other frames in the gripper and exaggerates the size of the robot base to make it
harder for the gripper to intersect with the robot’s second link. The sizes and locations for the capsule and spheres are based on the collision spheres that
you’ve already added.

```python
 1# body_cylinders are used to promote self-collision avoidance between the robot and its base
 2# The example below defines the robot base to be a capsule defined by the absolute coordinates pt1 and pt2.
 3# The semantic name provided for each body_cylinder does not need to be present in the robot URDF.
 4body_cylinders:
 5     - name: base
 6       pt1: [0,0,.12]
 7       pt2: [0,0,0.]
 8       radius: .08
 9     - name: second_link
10       pt1: [0,0,.12]
11       pt2: [0,0,.12]
12       radius: .16
13
14
15# body_collision_controllers defines spheres located at specified frames in the robot URDF
16# These spheres will not be allowed to collide with the capsules enumerated under body_cylinders
17# By design, most frames in industrial robots are kinematically unable to collide with the robot base.
18# It is often only necessary to define body_collision_controllers near the end effector
19body_collision_controllers:
20     - name: J5
21       radius: .05
22     - name: J6
23       radius: .05
24     - name: right_inner_finger
25       radius: .02
26     - name: left_inner_finger
27       radius: .02
28     - name: right_inner_knuckle
29       radius: .02
30     - name: left_inner_knuckle
31       radius: .02
```

You represent the robot base link “J1” with a capsule of radius .08 m, which matches the size of the collision spheres in near the base of the robot.
You represent the robot’s second link with a large sphere of radius .12.
In the Lula Test Widget, you observe the robot does a much better job avoiding collisions with the first and second link.
As expected, it is still possible to cause a self-collision, but the cases are much more limited.

### Creating an End Effector Frame

Observe that the chosen end effector frame `right_inner_finger` does not directly
represent the position of the robot’s gripper. The frame that RMPflow considers to be the end effector must be present in the robot URDF.
In this tutorial, you selected a frame near the end of the robot as the best option. To directly control where the center of the gripper is, you have two options:

* Manually compute transforms between the desired target and the target you send to RMPflow at runtime.
* Add a frame to the robot’s URDF.

This tutorial covers the second option by adding a frame to the **Cobotta Pro 900** URDF. Typically, the end effector position is in
the center of the gripper, with two principal axes aligned with the gripper fingers.

Investigating the **Cobotta Pro 900** URDF, you observe how the “right\_inner\_finger” frame is connected to the robot arm.
In the URDF, you observe that the “right\_inner\_finger” joint is a grandchild of the “onrobot\_rg6\_base\_link” frame, which is at the gripper base.

```python
 1<joint name="right_inner_finger_joint" type="revolute">
 2    <origin rpy="0 0 0" xyz="0 -0.047334999999999995 0.064495"/>
 3    <parent link="right_outer_knuckle"/>
 4    <child link="right_inner_finger"/>
 5    <axis xyz="1 0 0"/>
 6    <limit effort="1000" lower="-0.872665" upper="0.872665" velocity="2.0"/>
 7    <mimic joint="finger_joint" multiplier="1" offset="0"/>
 8  </joint>
 9
10<joint name="right_outer_knuckle_joint" type="revolute">
11    <origin rpy="0 0 3.141592653589793" xyz="0 0.024112 0.136813"/>
12    <parent link="onrobot_rg6_base_link"/>
13    <child link="right_outer_knuckle"/>
14    <axis xyz="1 0 0"/>
15    <limit effort="1000" lower="-0.628319" upper="0.628319" velocity="2.0"/>
16    <mimic joint="finger_joint" multiplier="-1" offset="0"/>
17  </joint>
```

This tells us that you can create a frame that is offset from the “`onrobot_rg6_base_link`” frame by a pure Z offset of `.064495+.136813=.2013` to represent a point in the center of the gripper, aligned with the “`right_inner_finger_joint`” and “`left_inner_finger_joint`”. To get closer with the tips of the fingers, increase the Z offset to .24.

Add a link to the URDF called “gripper\_center”, whose offset from the parent link “`onrobot_rg6_base_link`” is defined by the connection
“`gripper_center_joint`”. In the tutorial file, the modified URDF is saved as `./cobotta_pro_900_gripper_frame.urdf`.

```python
1<link name="gripper_center"/>
2  <joint name="gripper_center_joint" type="fixed">
3    <origin rpy="0 0 0" xyz="0.0 0.0 .24"/>
4    <parent link="onrobot_rg6_base_link"/>
5    <child link="gripper_center"/>
6  </joint>
```

You observe in the video that the Z axis of the target lies along the center of the gripper and that the Y axis of the target is aligned with the gripper plane.

This video uses three of the provided config files:

```python
./robot_description.yaml
./cobotta_pro_900_gripper_frame.urdf
./rmpflow_configs/cobotta_rmpflow_config_basic.yaml
```

### Modifying RMPflow Parameters

There is one remaining piece of the RMPflow config files left to modify, the RMPflow parameters in `rmpflow_config.yaml`.
Typically not much modification of the parameters from the template is needed. RMPflow terms work well for robots with similar scales.
The template RMPflow config was tuned based on the **Franka Emika Panda** robot.

There is one RMPflow parameter that is robot-specific, `joint_velocity_cap_rmp`. This term sets a limit on the maximum velocity that is allowed by RMPflow for any joint in the specified configuration space.
Investigating the URDF, you observe that each joint in the **Cobotta Pro 900** has a velocity limit of `1 rad/s`.

```python
1<joint name="joint_6" type="revolute">
2    <parent link="J5"/>
3    <child link="J6"/>
4    <origin rpy="0.000000 0.000000 0.000000" xyz="0.000000 0.120000 0.160000"/>
5    <axis xyz="-0.000000 -0.000000 1.000000"/>
6    <limit effort="1" lower="-6.28318530717959" upper="6.28318530717959" velocity="1"/>
7    <dynamics damping="0" friction="0"/>
8  </joint>
```

To make sure that RMPflow respects these joint velocity limits, you can modify template parameters so that RMPflow will start damping the velocity of a joint when it comes within `.3 rad/s` of the `1 rad/sec` limit:

```python
1joint_velocity_cap_rmp:
2    max_velocity: 1.
3    velocity_damping_region: .3
4    damping_gain: 1000.0
5    metric_weight: 100.
```

Note

The PD gains from the provided Cobotta Pro 900 USD file are based off the PD gains that you chose for the Franka Emika Panda of P=10000 N\*m and D=1000 N\*m\*s. These values produced oscillations in the Cobotta Pro 900 when you reduced the `max_velocity joint_velocity_cap_rmp` term to `1 rad/sec`. The USD provided for the Cobotta robot in this tutorial has a proportional gain of 10000 N\*m damping gain of 10000 N\*m\*s.

Refer to [RMPflow Tuning Guide](Robot_Simulation.md) for more details about the meaning of each RMPflow parameter with and a description of how to improve RMPflow parameters for new robots.

## Summary

This tutorial builds on the Lula Robot Description Editor tutorial to complete the process of configuring RMPflow on a new robot. In it, you:

> 1. Modify a template rmpflow\_config.yaml file to fit a specific robot.
> 2. Tune self-collision avoidance behavior.
> 3. Create a new end effector frame that can be used by RMPflow.

### Further Learning

To understand the motivation behind the structure and usage of RmpFlow in NVIDIA Isaac Sim, reference the [Motion Generation](concepts/index.html#isaac-sim-motion-generation) page.

---

# cuRobo and cuMotion

Note

There are known issues with using NvBlox examples within cuRobo. This tutorial will be updated when cuRobo is updated to resolve these issues.

Note

This cuRobo tutorial is not supported on aarch64 platforms.

## Learning Objectives

[cuRobo](https://curobo.org) (also on [GitHub](https://github.com/NVlabs/curobo)) is a high-performance,
GPU-accelerated robotics motion generation library for robot manipulators, developed by NVIDIA Research.
It is a standalone Python library that interfaces directly with NVIDIA Isaac Sim, simplifying both testing in simulation
and deploying on physical robots.

[NVIDIA cuMotion](https://nvidia-isaac-ros.github.io/concepts/manipulation/index.html#nvidia-cumotion),
available as a Developer Preview in Isaac 3.0, is a production motion generation package for
manipulators. The current version leverages cuRobo as its backend, providing collision-free motion planning using a
plugin for [MoveIt 2](https://moveit.picknik.ai) and a set of supporting ROS 2 packages. For an example of using
cuMotion with NVIDIA Isaac Sim using the ROS 2 bridge, see the relevant
[section](https://nvidia-isaac-ros.github.io/concepts/manipulation/cumotion_moveit/tutorial_isaac_sim.html)
of the Isaac ROS documentation. This example is somewhat limited in Isaac 3.0 but will be expanded in a future
release.

In the remainder of this tutorial, we focus on direct integration of cuRobo into NVIDIA Isaac Sim, covering cuRobo
installation and use, with examples for collision-free inverse kinematics, motion planning, and reactive
control (MPPI).

## Getting Started

**Prerequisites**

* Complete the [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md) tutorial prior to beginning this tutorial.

## Installation

Follow the [cuRobo installation instructions](https://curobo.org/get_started/1_install_instructions.html) for
installing cuRobo and required libraries. cuRobo supports NVIDIA Isaac Sim 2022.2.1 and later. Follow the
[workstation installation instructions](Installation.md) to install NVIDIA Isaac Sim.

## Examples

### Using Isaac Sim with cuRobo

In the cuRobo documentation, refer to the
[“Using Isaac Sim” section](https://curobo.org/get_started/2b_isaacsim_examples.html) for an overview of how cuRobo
is interfaced to Isaac Sim, along with a series of standalone examples demonstrating collision checking, motion
generation, inverse kinematics, model-predictive control, and multi-arm reaching.

### Using Isaac Sim with cuRobo and nvblox

In the cuRobo documentation, refer to the
[“Using with Depth Camera” section](https://curobo.org/get_started/2d_nvblox_demo.html) for examples of
obstacle-aware motion generation in NVIDIA Isaac Sim, both with pre-generated signed distance fields (SDFs)
from [nvblox](https://github.com/nvidia-isaac/nvblox) and with online mapping leveraging nvblox with a
physical RealSense depth camera.

---

# Surface Gripper Extension

## About

The [Surface Gripper Extension](#isaac-surface-grippers) Extension is used to create a suction-cup gripper behavior for an end-effector. It works by parsing the Surface Gripper properties on the USD Surface Gripper Schema, and managing a set of D6 Joints between the parent and child rigid bodies at the gripper points of contact.

The physical properties of the gripper are defined on the D6 Joints properties, such as joint limits across the different degrees of freedom, and the stiffness and damping of the joint. The Surface gripper Object then handles the activation of the constraints, and define which objects to be grasped based on the grip threshold.

This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.robot.surface_gripper`.

To create a surface gripper through the GUI, Go to the menu `Create` > `Robots` > `Surface Gripper`. This will create a surface gripper prim in the stage.

## API Documentation

See the [API Documentation](../py/source/extensions/isaacsim.robot.surface_gripper/docs/index.html) for usage information.

## Setting up a Surface Gripper

The Surface Gripper Has the following properties:

| Property | Description |
| --- | --- |
| Attachment Points | The list of Joints that will be used to attach the gripper to the object |
| Status | (Read-Only) The current state of the gripper |
| Gripped Objects | (Read-Only) The list of objects that are currently grasped by the gripper |
| Max Grip Distance | Distance from the gripper point that will accept closing contact |
| Retry Interval | How long the gripper will remain attempting to close on an object |
| Shear Force Limit | The maximum lateral force that the gripper can apply to an object before it will break the constraint |
| Coaxial Force Limit | The maximum axial force that the gripper can apply to an object before it will break the constraint |

### Attachment Joints

The joints that will be used to attach the gripper to the object are defined on the `Attachment Points` property. This is a list of paths to the D6 Joints that will be used to attach the gripper to the object. These joints must be defined on the USD file at the gripper points of contact, and must be of type `D6`. Any physical properties for the joint are defined on the D6 Joint Schema, but there are a few properties that are required to be set for the Joint:

* Joint must be enabled
* All joints Body 0 must be the same.
* Joint must have “Exclude from Articulation” set to True. If this is not set, the surface gripper manager will set it to True at runtime.

### Attachment Point API

The joints that are defined on the `Attachment Points` property are automatically assigned the `Attachment API`. This API is responsible for providing additional attributes to the joint, which are necessary for the Surface Gripper Manager to handle the gripper. In the attachment point API, the following attributes are available:

* ClearanceOffset: This registers the distance from the joint to the parent object’s surface. Since the surface gripper works by sending a raycast from the joint world position, this offset will be added to the raycast origin to avoid false positive hits with the parent object. If this offset is not defined, it will start at the joint’s world position, and whenever it clears the parent object collider, it will author the offset at runtime for future use.
* Forward Axis: This registers which joint axis will be used to attempt to close the gripper. The default value is `X`.

#### Adding Attachment Joint API

To add an attachment joint API, select the joint in the stage, and in the right panel under the Properties tab, check the + Add button, and select `Edit API Schema`. Search for AttachmentPointAPI and apply it to the joint.

## Tutorials & Examples

Activate the `Robotics Examples` content browser from **Windows** > **Examples** > **Robotics Examples**. Navigate to **Manipulation**, select the Surface Gripper Example, and click the load button in the information window on the right side of the Robotics Examples content browser. You may need to adjust the GUI to see the load button.

### Surface Gripper Example

This example shows a Surface gripper mounted to a gantry, and contains cubes that can be grasped by the gripper. This Surface gripper is Added by code, and also connected throug the surface gripper Omniverse OmniGraph node.

To run the Example:

1. Press the **Load** button. The scene should begin playing.
2. You can move the Gantry with the gamepad axis, or by manually editing the gantry joint target positions.
3. Move the gantry near some cube or set of cubes, and click on the “Open/Close” Button - the button label reflects the current gripper state. The gripper can also be closed by the down face button on the gamepad (e.g X on playstation controllers, or A on Xbox controllers).
4. The gripper will attempt to close on the cubes, and if successful, the cubes will be grasped by the gripper.
5. Lift the gantry, and the cubes will remain grasped by the gripper, or forces may be excessive and break the gripper constraint.

## Omniverse OmniGraph Node

The Surface Gripper extension provides a implementation through Omniverse OmniGraph. To use it, Add a surface gripper node to the desired graph, and select the Surface gripper prim it will control.

## Creating a Surface Gripper fully on code

This section describes how to implement a surface gripper completely from code. These are snippets from the Surface Gripper Example code, and is not complete.

### Defining the Surface Gripper Properties

```python
# Relevant Imports
import isaacsim.robot.surface_gripper._surface_gripper as surface_gripper
import usd.schema.isaac.robot_schema as robot_schema

# [...]

self.gripper_prim_path = "/World/SurfaceGripper"
self.gripper_interface = surface_gripper.acquire_surface_gripper_interface()

# Create the Surface Gripper Prim
# Once it is created it can be saved and this doesn't need to be redone
robot_schema.CreateSurfaceGripper(self._stage, self.gripper_prim_path)
gripper_prim = self._stage.GetPrimAtPath(self.gripper_prim_path)
attachment_points_rel = gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name)

# Select the joints to the gripper
# The joints should be D6 joints defined in the usd file.
# All joint attributes can be defined as desired, except for:
# Joint Should be enabled
# Joint Type should be D6
# All Joint Parents should be the same Rigid body
# Exclude from Articulation must be checked
# No Break force/Torque should be set
# Joint drives can be used to derive the desired joint bounce/stretch behavior
# Enable/Disable the joint DoFs and limits as desired.

gripper_joints = [p.GetPath() for p in self._stage.GetPrimAtPath("/World/Surface_Gripper_Joints").GetChildren()]
attachment_points_rel.SetTargets(gripper_joints)

# Define the distance the joint can grasp, and at what distance from the origin of the joints it will settle
gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.011)
# Define the Override Break limits
gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(0.005)
gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(5)

# How long the gripper will try to close if it is open
gripper_prim.GetAttribute(robot_schema.Attributes.RETRY_INTERVAL.name).Set(1.0)

# [...]
```

### Get Gripper State

The Surface Gripper is updated on every simulation step, and the state can be retrieved at any time through the interface:

```python
self.gripper_interface = surface_gripper.acquire_surface_gripper_interface()
status = self.gripper_interface.get_gripper_status(self.gripper_prim_path)
print(status)  # Open, Closed, or Closing
```

### Controlling the Gripper

The Gripper State is controlled through the `open` and `close` methods of the interface. Alternativel, there’s also the `set_gripper_action`, which receives a numeric value between -1 and 1, where `< -0.3` will open the gripper, `> 0.3` will close it, and anything in between will be ignored.

```python
1self.gripper_interface.close_gripper(gripper_prim_path)
2
3self.gripper_interface.open_gripper(gripper_prim_path)
4
5self.gripper_interface.set_gripper_action(gripper_prim_path, 0.5)  # Closes the gripper
6self.gripper_interface.set_gripper_action(gripper_prim_path, -0.5)  # Opens the gripper
```

### Keeping USD Scene in Sync

In order to optimize the Surface Gripper Update performance, the USD Scene update is disabled by default. When the USD writeback is disabled, the Properties panel for the Surface Gripper prim will not be updated automatically. The surface gripper status can still be retrieved through get\_gripper\_status method of the surface gripper interface, and objects currently grasped by the gripper can be retrieved through get\_gripped\_objects method of the surface gripper interface.

> The USD writeback can be enabled by setting the `set_write_to_usd` property to `True` on the Surface Gripper interface. This is a global setting for all surface gripper instances.

---

# Grasp Editor

## Learning Objectives

This tutorial explains how to use the Grasp Editor extension in NVIDIA Isaac Sim to hand-author and
simulate grasps for a specific gripper/object pair. These grasps are stored in an isaac\_grasp
YAML file that can be imported and used with a motion generation algorithm to move the gripper into
place and grasp the desired object.

## Getting Started

To get started using the Grasp Editor extension, you need to prepare your assets in NVIDIA Isaac Sim.

* You must have an Articulation capable of grasping. This can be a floating gripper, or it can be a gripper attached to an arm.
* You must have a USD version of the object you want to grasp.

For both the gripper and the object, you must be ready to identify the USD frame that should be used to
represent location. This is often the frame in the center of the object mesh and at the base of the gripper.

You can download the stage used in this tutorial
[`here`](../_downloads/4d1aeb9e29208ad4bf35f0a38d105e49/Grasp_Editor_Tutorial_Stage.zip)
and follow along.

## What is an Isaac Grasp File?

The output of the Grasp Editor extension is a YAML in the isaac\_grasp file format. A single isaac\_grasp
file stores a list of grasps for a specific gripper/object pair. The file follows a simple format:

```python
 1format: isaac_grasp
 2format_version: 1.0
 3
 4object_frame_link: /World/mug
 5gripper_frame_link: /World/panda_hand
 6
 7grasps:
 8  grasp_0:
 9      confidence: 1.0
10      position: [-0.04346, 0.06759, 0.19895]
11      orientation: {w: 0.00332, xyz: [0.98453, 0.16837, 0.04837]}
12      cspace_position:
13        panda_finger_joint1: 0.00943
14      pregrasp_cspace_position:
15        panda_finger_joint1: 0.04
```

isaac\_grasp files do not need to originate with the Grasp Editor extension. The Grasp Editor
extension is useful for both authoring isaac\_grasp files and importing grasps that were authored
elsewhere for visualization and validation.

A grasp is defined by the relative position of the gripper and object. In order for this relative
position to have meaning, a representative frame must be chosen for the gripper and object positions.
The Grasp Editor writes the USD paths of these representative frames to an isaac\_grasp file
under the object\_frame\_link and gripper\_frame\_link fields. Because isaac\_grasp files may be
authored externally (possibly without going through USD at all), the Grasp Editor ignores the
object\_frame\_link and gripper\_frame\_link fields when importing grasps. This makes it the user’s
responsibility to identify the correct USD frames when using the Grasp Editor for importing.

Each grasp in an isaac\_grasp file has a unique name (e.g. grasp\_0). The fields for a named
grasp are:

* confidence: A parameter describing the quality of a grasp.
* position: The translation of the gripper frame relative to the object frame.
* orientation: The orientation of the gripper frame relative to the object frame.
* cspace\_position: A dictionary of joint positions for every joint that is used to control the gripper.
  These joint positions are the state of the gripper as it is actively grasping the object.
* pregrasp\_cspace\_position: A dictionary of joint positions for every joint that is used to control the gripper.
  These joint positions represent the open position of the gripper.

All together, a grasp may be applied in practice by moving the gripper to the correct relative position and orientation
while in the pregrasp\_cspace\_position, then closing the gripper until the joints are in cspace\_position.
If the object’s position and orientation in the world frame of reference is given by \(T\_o, R\_o\), with
the position and orientation fields specifying relative transformation \(^oT\_g, ^o\!\!R\_g\)
(i.e. the translation and rotation of the gripper according to the object frame of reference),
the desired position of the gripper in the world frame \(T\_g , R\_g\) is given by:

\[\begin{split}T\_g = R\_o \cdot {^oT\_g + T\_o} \\
R\_g = R\_o \cdot {^o\!R\_g}\end{split}\]

## Using the Grasp Editor

### Selection Frame

The Grasp Editor is a UI-based extension that can be used to author and import isaac\_grasp files.
In NVIDIA Isaac Sim, the Grasp Editor can be found in the toolbar under **Tools** > **Robotics** > **Grasp Editor**.
The first step is to add an Articulation and an object to the stage. The Articulation may be an
isolated gripper, or it may be a gripper attached to a robot arm. The object can be any
non-Articulation that has an associated mesh.

In the Selection Frame, select the Articulation and object of interest. The prim path for the object
can be copied by right clicking on the desired prim and selecting “Copy Prim Path”. An export path
must be chosen for the isaac\_grasp file (this should end in ‘.yaml’). The Grasp Editor may be used
to author a sequence of grasps to the selected export file, but it does not support modifying an existing
file. If an export path is supplied that already exists, the existing file will be overwritten with
a new isaac\_grasp file.

This tutorial will author grasps between the Panda hand gripper (isolated from the Franka Emika Panda robot)
and a mug. When “Ready” is clicked, the Grasp Editor will validate each field and perform all necessary
conversions of the selected object prim (the mug) to make it graspable. Specifically, it applies the
Rigid Body and Collision APIs from Usd Physics so that the object has a collision geometry and can be moved
by external forces.

Note

The Grasp Editor does not revert these changes to the object asset, and so it is best not to save the USD stage unless these changes are specifically desired.

Warning

There is a known issue that the mug may “dissappear”, this is a visual bug. You can press “STOP”, then “PLAY” again to make it reappear.

### Select Frames of Reference

In this panel, you may select the frames of reference that should be used to describe the position
in space of the gripper and object. It is critical to understand this panel and to make the proper
selections before moving on.

Most motion generation algorithms do not natively consume USD files. It is common for motion generation
algorithms to reference a URDF file. If the Grasp Editor
uses a frame that is not defined in a corresponding URDF file, an authored grasp becomes meaningless from the
perspective of any such motion generation algorithms.

Similarly, the selected frame of reference for the object
must correspond to the existing pipeline in which the object is being manipulated. For example, if a
camera is being used to identify object pose, there is an implicit frame of reference for the object
associated with that vision system. In this case, the selected frame for the object must correspond to this
implicit frame of reference. If there is not already a frame in the USD that represents the correct frame of
reference, a new one should be authored on the stage under the selected object path (e.g. nested under “/World/mug”).

In this tutorial, the base frames for the gripper and object are used. If the entire Franka Panda robot
were being used, the correct frame of reference for the gripper would still be the panda\_hand frame.
Once “Finalize” is clicked, these frames of reference become global to the output isaac\_grasp file and
cannot be changed.

**The Grasp Editor will write the USD paths for the frames of reference to the output isaac\_grasp file,
but this information will not be interpretable by a motion generation algorithm that does not consume USD.**

### Joint Settings

In this menu, you must select which joints in the Articulation are active degrees of freedom (DOFs) in
the gripper. The Panda hand is a two finger gripper, but one of the joints is a mimic joint. Observe
in the figure below that changing the value of panda\_finger\_joint1 causes panda\_finger\_joint\_2 to
move at the same time. This means that the Panda hand gripper is effectively controlled by a single DOF.

Each active DOF in the gripper should be checked as “Part of Gripper”. This will open a new menu of
joint settings that define how the grasp will be simulated and what gets written to the output isaac\_grasp file.

* Position When Open: The position of DOF that is considered to be open. Each grasp will be simulated
  by moving from the open position towards the closed position.
* Position When Closed: The position of the DOF that is considered to be fully closed.
* Grasp Speed: The speed at which the DOF will move from the open position towards the closed position when simulating.
* Max Effort Magnitude: The maximum force/torque (N or N\*m) that this DOF will be able to apply on the object when simulating.

At least one DOF must be marked as part of the Gripper in order to author a grasp. Only active gripper DOFs will
be written to the output isaac\_grasp file.

### Utils

The Utils menu has two useful utility functions that assist in using the Grasp Editor.

The Mask Collision button will mask collisions between the gripper and object. This may be helpful
when moving the object into place in order to test a grasp. Masked collisions are unmasked when a grasp
is simulated. When importing a grasp, collisions are masked automatically.

If the simulated grasp does not appear to have complete contact between the object and gripper,
you can use the “Show Physics Colliders” button to visualize the collision geometry associated
with your assets. It is outside of the scope of this extension to fix incorrect collider geometry,
but the Grasp Editor does allow you to author grasps without simulating them. In this situation
you can mask collisions and move things into place visually.

### Author a Grasp

A grasp may be authored with the aid of simulation. Moving assets by hand into what appears to be the right
position is imprecise. In the figure below, the mug is moved into roughly the right position to
be grasped, and the “Simulate” button is clicked to close the gripper according to its joint settings.
This causes the lip of the mug to be pushed into the exact center of the gripper fingers, and it
leaves the gripper fingers in the exact position of contact with the object. This gives a high
degree of confidence to the grasp that is written to the output file. Once the simulation is complete,
the export panel will populate, and the grasp may be written to file.

There may be reasons that the grasp simulation does not support your use-case such as:

* The physics colliders for your assets are not accurate.
* The mechanics for opening and closing the gripper are more complicated than is represented in the Grasp Editor.

In either case, the best way to make use of the Grasp Editor is to move things into place through
external means and export the grasp without simulating by clicking the “Skip Sim” button. For example,
some real robot grippers have heavily coupled degrees of freedom with somewhat complicated mechanics.
For such a gripper, you would want to replicate the exact movement programmatically and send joint
commands to the USD asset accordingly. In this case, you could turn on collisions and use an external
script or OmniGraph node to drive your gripper into a grasping position, then use the export function of the
Grasp Editor to export the current state of grasp on the USD stage to your isaac\_grasp file.

#### Adding External Forces and Torques

An extra feature of the Grasp Editor is that you can apply external forces and torques as part of the
grasp simulation. This may help to discern which grasps have the best force closure over the object.
The amount of force and torque applied may be selected in the “Add External Rigid Body Forces” panel.
A single scalar value may be chosen for force and for torque. A non-zero value \(v\) for force will cause
a force of \(\pm v\) N along each axis, centered at the base frame of the rigid body.
Likewise for torque, a value \(v\) will cause a torque of \(\pm v\) N\*m to be applied about each axis, centered
at the base frame of the rigid body.

The figure below demonstrates closing the grasp and then applying forces of 3 N. This test fails
when the mug flies away under a force of \([3, 0, 0]\). A smaller force value of 0.5 N is then
chosen, and the mug moves under the force, but the grasp is maintained.

### Exporting Grasps

The export frame becomes available once a grasp has been fully simulated, or the option to simulate has been declined.
On clicking “Export”, the current state of the stage is used to fill in the relevant fields of the
isaac\_grasp file.

* The confidence field takes on the value of the “Confidence” field in the Export panel.
* The position and orientation fields for the grasp are determined by finding the relative position
  of the gripper in the object’s frame of reference. This uses the frames defined in
  [Select Frames of Reference](#isaac-sim-app-tutorial-grasp-editor-reference-frames).
* The cspace\_position field is determined based on the current positions of the DOFs that have been marked as
  part of the gripper.
* The pregrasp\_cspace\_position field is taken from the “Position When Open” field of Joint Settings for each
  DOF that has been marked as part of the gripper.

At this stage, multiple grasps may be authored in a row and sequentially exported to the same isaac\_grasp file.

### Importing Grasps

Apart from authoring grasps, the Grasp Editor may be used to validate grasps that were authored
elsewhere. This can be done in the Import panel by selecting an isaac\_grasp file and clicking Import.
This tutorial uses the same file that is used for export, but this does not need to be the case.

In the figure below, multiple grasps have been authored and written to file using the Grasp Editor.
These grasps are imported, and can now be quickly visualized and simulated in sequence.

## Using Authored Grasps in Isaac Sim

The Grasp Editor is primarily a UI-based extension, but it offers some utility for importing and
using authored grasps within NVIDIA Isaac Sim through a Python API.

This section presents the following stage with the goal of determining where the robot should go
to execute one of the authored grasps.

The following code snippet imports the grasp file demonstrated in [Importing Grasps](#isaac-sim-app-tutorial-grasp-editor-import) and
determines where the panda\_hand frame should be in order to duplicate grasp\_1.

```python
from isaacsim.core.utils.xforms import get_world_pose
from isaacsim.robot_setup.grasp_editor import GraspSpec, import_grasps_from_file

import_file_path = "/path/to/franka_mug_grasp.yaml"
grasp_spec = import_grasps_from_file(import_file_path)

mug_reference_frame = "/World/mug"

grasp_names = grasp_spec.get_grasp_names()

mug_trans, mug_quat = get_world_pose(mug_reference_frame)
gripper_trans_target, gripper_orientation_target = grasp_spec.compute_gripper_pose_from_rigid_body_pose(
    "grasp_1", mug_trans, mug_quat
)

print("Grasp Names:", grasp_names)
print("Gripper Translation Target:", gripper_trans_target)
print("Gripper Orientation Target:", gripper_orientation_target)
```

```python
Grasp Names: ['grasp_0', 'grasp_1', 'grasp_2']
Gripper Translation Target: [ 0.41496072 -0.03612298  0.27738899]
Gripper Orientation Target: [-0.1690746   0.63886658  0.12752551  0.73959483]
```

The result of the code snippet shows the name of each grasp in the isaac\_grasp file, and
the translation and orientation targets that should be set for the
panda\_hand frame in the full Franka robot. Note that the code snippet uses the frame of reference
for the mug that was selected in the Grasp Editor. It is outside of the scope of this tutorial to
use a motion generation algorithm to achieve this grasp.

Check out the GraspSpec class in our [API Documentation](../py/source/extensions/isaacsim.robot_setup.grasp_editor/docs/index.html) to see the complete set of functionality.

---

# Reinforcement Learning Policies Examples in Isaac Sim

## About

The isaac\_sim\_policy\_example Extension is a framework and has a set of helper functions to deploy Isaac Lab Reinforcement Learning Policies in Isaac Sim.
For details for training and building the policy in Isaac Sim, visit [deploying policy in Isaac Sim](Isaac_Lab.md).

This Extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.robot.policy.example`.
To run examples below activate **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` tab.

### Unitree H1 Humanoid Example

1. The Unitree H1 humanoid example can be accessed by creating a empty stage.
2. Open the example menu using **Robotics Examples** > **POLICY** > **Humanoid**.
3. Press **LOAD** to open the scene.

This example uses the H1 Flat Terrain Policy trained in Isaac Lab to control the humanoid’s locomotion.

Controls:

* Forward: UP ARROW / NUM 8
* Turn Left: LEFT ARROW / NUM 4
* Turn Right: RIGHT ARROW / NUM 6

### Boston Dynamics Spot Quadruped Example

1. The Boston Dynamics Spot quadruped example can be accessed by creating a empty stage.
2. Open the example menu using **Robotics Examples** > **POLICY** > **Quadruped**.
3. Press **LOAD** to open the scene.

This example uses the Spot Flat Terrain Policy trained in Isaac Lab to control the quadruped’s locomotion.

Controls:

* Forward: UP ARROW / NUM 8
* Backward: BACK ARROW / NUM 2
* Move Left: LEFT ARROW / NUM 4
* Move Right: RIGHT ARROW / NUM 6
* Turn Left: N / NUM 7
* Turn Right: M / NUM 9

### Franka Panda Open Drawer Example

1. The Franka Panda Open Drawer example can be accessed by creating a empty stage.
2. Open the example menu using **Robotics Examples** > **POLICY** > **Franka**.
3. Press **LOAD** to open the scene.

This example uses the Franka Open Drawer Policy trained in Isaac Lab to control the robot’s arm.
The robot will open the drawer, hold it open until the would reset.

## Policies Files

The policies used in the examples are trained in Isaac Lab and are available here:

|  |  |  |
| --- | --- | --- |
| Name | Policy | Parameters |
| H1 Flat Terrain Policy | [H1 Flat Terrain Policy](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/H1_Policies/h1_policy.pt) | [H1 Flat Terrain Policy Environment Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/H1_Policies/h1_env.yaml)  [H1 Flat Terrain Policy Network Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/H1_Policies/agent.yaml) |
| Spot Flat Terrain Policy | [Spot Flat Terrain Policy](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Spot_Policies/spot_policy.pt) | [Spot Flat Terrain Policy Environment Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Spot_Policies/spot_env.yaml)  [Spot Flat Terrain Policy Network Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Spot_Policies/agent.yaml) |
| ANYmal C Flat Terrain Policy | [ANYmal C Flat Terrain Policy](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Anymal_Policies/anymal_policy.pt)  [Anymal C Motor Policy](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Anymal_Policies/sea_net_jit2.pt) | [ANYmal C Flat Terrain Policy Environment Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Anymal_Policies/anymal_env.yaml)  [ANYmal C Flat Terrain Policy Network Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Anymal_Policies/agent.yaml) |
| Franka Panda Open Drawer Policy | [Franka Panda Open Drawer Policy](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Franka_Policies/Open_Drawer_Policy/policy.pt) | [Franka Panda Open Drawer Policy Environment Parameters](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/Franka_Policies/Open_Drawer_Policy/env.yaml) |

Note

The policies can also be downloaded directly from the Content Browser by right clicking the policy and selecting `Download`.

## API Documentation

See the [API documentation](../py/source/extensions/isaacsim.robot.policy.examples/docs/index.html) for complete usage information.

## Standalone Examples

**h1\_standalone.py**

* This standalone example demonstrates a Unitree H1 controlled by a flat terrain policy, following a set of predetermined command sequences. It may be run via the following command:

  > ```python
  > ./python.sh standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py --num-robots <number of robot> --env-url </path/to/environment>
  > ```
  >
  > For example, this will spawn 5 robots on the flat grid scene below:
  >
  > ```python
  > ./python.sh standalone_examples/api/isaacsim.robot.policy.examples/h1_standalone.py --num-robots 5 --env-url /Isaac/Environments/Grid/default_environment.usd
  > ```
  >
  > 

**spot\_standalone.py**

* This standalone example demonstrates a Boston Dynamics Spot controlled by a flat terrain policy, following a set of predetermined command sequences. It may be run via the following command:

  > ```python
  > ./python.sh standalone_examples/api/isaacsim.robot.policy.examples/spot_standalone.py
  > ```
  >
  > 

**anymal\_standalone.py**

* This standalone example demonstrates an ANYmal C robot that is controlled by a neural network policy. The rough terrain policy was trained in Isaac Lab and takes as input the state of the robot, the commanded base velocity, and the surrounding terrain and outputs joint position targets. The example may be run via the following command:

  > ```python
  > ./python.sh standalone_examples/api/isaacsim.robot.policy.examples/anymal_standalone.py
  > ```
  >
  > 

Controls:

* Forward: UP ARROW / NUM 8
* Backward: BACK ARROW / NUM 2
* Move Left: LEFT ARROW / NUM 4
* Move Right: RIGHT ARROW / NUM 6
* Turn Left: N / NUM 7
* Turn Right: M / NUM 9

---

# Robot Simulation Tips

## Improve Simulation Performance

* You can speed up the simulation by reducing the number of objects in the scene, reducing the complexity of the objects in the scene, or reducing the number of simulation steps. For more information, see [Isaac Sim Performance Optimization Handbook](Isaac_Sim_Performance_Optimization_Handbook.md) for more details.
* Alternatively, you can reduce the number of sensors, or reduce the resolution of sensors in the scene. See [Isaac Sim Benchmarks](Isaac_Sim_Benchmarks.md) for more performance benchmarks.

## Simulation Time Stepping and Rendering Rate

* adjust the rendering and physics stepping in simulation. If your system is GPU limited, decrease the rending rate could allow more physics stepping to happen per frame, so although it may appear less smooth with a lower frame rate, the physics simulation may be more accurate and more simulation time would have elapsed per frame.
* These parameters can be set using the Simulation Manager and the Rendering Manager, assessbile via Python

## Adjusting friction for wheeled robots

* add and adjust friction parameters to both the wheels and the ground. See [Adding Contact and Friction Parameters](Robot_Setup.md) for instructions.
* Coefficient of friction is the ratio between between the normal force and the friction force. Increasing the Coefficient of static or dynamic friciton to a higher value will increase the friction force for the same normal force. Coefficient of friction should not exceed 1.0 in most cases.
* modify the frictiion combine mode to adjust how friction is computed between two objects.

## Gripper not picking up objects

* You can increase the friction parameters on both the fingers and object. See [Adding Contact and Friction Parameters](Robot_Setup.md) for instructions.
* Use the Physics Authoring Toolbar (Tools > Physics Toolbar), especially the [Mass Distribution Tool](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.supportui/docs/dev_guide/authoring_tools.html#mass-distribution-manipulator) to make sure the weight of the object and weight of the arms are reasonable.
* The gripper is not following your commands accurately, consider increase the stiffness and damping gains in the controller.

## Colliders

* Colliders should only be applied to the parts of the robot that need to interact with the environment.
* Use simple shape colliders (box, sphere, capsule, convex hull) or convex hull whenever possible for better performance.
* Only use convex decomposition colliders when necessary, such as tips of end effectors, as they are more computationally expensive. Adjust the Error Percentage, Shrink Wrap, and ofset parameters in the advanced tab for better accuracy.
* Apply collision filters to avoid unnecessary collision checks between parts of the robot that should not collide with each other, such as the rubber pads on the finger and the finger itself. Overlapping colliders can cause instability in the simulation and cause the robot to “explode”. Collision filters can be set via *Physics Collision Group*
* For dynamic collisions, use convex hull, convex decomposition, box, sphere, or SDF approximations only. Triangle mesh, and Mesh simplification only works for static objects.

## Masses

* For accurate simulation, the mass, center of mass, diagonal inertia, principal axes of the rigid body should be set using the MassAPI, and match the real world masses as closely as possible.
* If it’s not specified, the mass will be estimated based on the volume of the mesh, with dentisty set to 1000 kg/m^3 by default.

---

# Useful Links

Some important concepts that are useful to understand when working with robot simulation in Isaac Sim are located in the following sections:

* [Physics Simulation Fundamentals](Physics.md): Provides basic summary about rigid bodies, colliders, joints, articulation, as well as simulation timelines and steps.
* [Core API Overview](Python_Scripting_and_Tutorials.md): Provides an overview of how the core API in Isaac Sim interfaces with the physics backend.