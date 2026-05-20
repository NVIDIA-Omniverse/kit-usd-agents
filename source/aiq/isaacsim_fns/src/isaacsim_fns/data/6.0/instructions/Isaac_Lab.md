# Isaac Lab

## Overview

Isaac Lab is the official robot learning framework for Isaac Sim, providing APIs and examples for reinforcement learning,
imitation learning, and more. The framework provides the ability to design tasks in different workflows, including
a modular design to easily and efficiently create robot learning environments, while leveraging the latest simulation capabilities.

Some of its core features include:

* Modular configuration-driven system to easily create and modify environments
* Flexible user-designed workflow for optimized performance
* Suite of robot learning environments for training and evaluation
* Support for different reinforcement learning and imitation learning libraries
* Connection to peripheral devices, such as game-pads and keyboards, for collecting demonstrations
* Ability to augment simulation with custom actuator models for sim-to-real transfer

## Isaac Lab Resources

For more information and documentation for Isaac Lab, see the following external references:

* [Isaac Lab Repository](https://github.com/isaac-sim/IsaacLab)
* [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab)

## Suggested Isaac Sim Tutorials

The following set of tutorials details usage of reinforcement learning related components in Isaac Sim.

**Robot Setup**

* [Importing URDF](Importers_and_Exporters.md)
* [Importing MJCF](Importers_and_Exporters.md)
* [Simulation Fundamentals](Physics.md)

**Deploying Policies**

* [Rigging a Legged Robot for Policy Inference](Robot_Setup.md)
* [Policy Deployment](Isaac_Lab.md)
* [Policy Deployment in ROS 2](ROS_2.md)

**Data Generation**

* [Getting Started with Cloner](Isaac_Lab.md)
* [Instanceable Assets](Isaac_Lab.md)

**Python Scripting**

* [Python Scripting](Python_Scripting_and_Tutorials.md)

## Troubleshooting

- Isaac Lab Troubleshooting

Common Isaac Lab issues and their solutions are documented in the [Isaac Lab Troubleshooting](troubleshooting.html#isaac-sim-isaac-lab-troubleshooting) page. For general simulation troubleshooting, see [Troubleshooting](Help_FAQ.md).

## Deprecated Frameworks

Isaac Lab will be replacing previously released frameworks for robot learning and reinforcement learning,
including [IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) for the
[Isaac Gym Preview Release](https://developer.nvidia.com/isaac-gym), [OmniIsaacGymEnvs](https://github.com/isaac-sim/OmniIsaacGymEnvs) for
Isaac Sim, and [Orbit](https://isaac-orbit.github.io) for Isaac Sim.

These frameworks are now deprecated in favor of continuing development in Isaac Lab.
We encourage users of these frameworks to migrate your work over to Isaac Lab.
Migration guides are available to support the migration process:

* Migrating from IsaacGymEnvs and Isaac Gym Preview Release: [link](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
* Migrating from OmniIsaacGymEnvs: [link](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_omniisaacgymenvs.html)
* Migrating from Orbit: [link](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_orbit.html)

---

# Deploying Policies in Isaac Sim

The objective of this tutorial is to explain the process of deploying a policy trained in Isaac Lab by going through an example and exploring robot definition files.

There are many use cases in which you might want to deploy your policy in Isaac Sim; such as enabling robots to accomplish more complex locomotion, testing and integrating the policy with other stacks such as navigation and localization in simulated environments, and interfacing it using with existing bridges such as ROS 2.

## Learning Objectives

In this tutorial, you will walk through the policy based robot examples:

1. H1 and Spot flat terrain policy controller demo
2. Training and exporting policies in Isaac Lab
3. Reading the environment parameter file from Isaac Lab
4. Robot definition class
5. Position to torque conversion
6. Debugging tips
7. Sim to Real deployment

## Demos

First activate **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` tab.

### Unitree H1 Humanoid Example

1. The Unitree H1 humanoid example can be accessed by creating a empty stage.
2. Open the example using **Robotics Examples** > **POLICY** > **Humanoid**.
3. Press **LOAD** to open the scene.

This example uses the H1 Flat Terrain Policy trained in Isaac Lab to control the humanoid’s locomotion.

Controls:

* Forward: UP ARROW / NUM 8
* Turn Left: LEFT ARROW / NUM 4
* Turn Right: RIGHT ARROW / NUM 6

### Boston Dynamics Spot Quadruped Example

1. The Boston Dynamics Spot quadruped example can be accessed by creating a empty stage.
2. Open the example using **Robotics Examples** > **POLICY** > **Quadruped**.
3. Press **LOAD** to open the scene.

This example uses the Spot Flat Terrain Policy trained in Isaac Lab to control the quadruped’s locomotion.

Controls:

* Forward: UP ARROW / NUM 8
* Backward: BACK ARROW / NUM 2
* Move Left: LEFT ARROW / NUM 4
* Move Right: RIGHT ARROW / NUM 6
* Turn Left: N / NUM 7
* Turn Right: M / NUM 9

Note

See [isaac sim policy example extension document](Robot_Simulation.md) for standalone example workflow and the policy files used in the examples.

## Training and Exporting Policies in Isaac Lab

### Training

Training the policy from Isaac Lab is the first step to deploying the policy.
Consult the [Isaac Lab tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html) for training an existing or custom policy.

The policies trained used in the examples above are Isaac-Velocity-Flat-H1-v0 for the Unitree H1 humanoid and Isaac-Velocity-Flat-Spot-v0 for the Boston Dynamics Spot robot.

Note

For example, in Isaac Lab 2.0, use the following command to train the H1 flat terrain policy:

```python
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-H1-v0 --headless
```

### Exporting

Policies trained using `RSL_rl`, the policies can be exported using the `scripts/reinforcement_learning/rsl_rl/play.py` inside the Isaac Lab workspace. The exported files are generated in the `exported` folder.

It is also possible to inference using a policy trained in a different framework or with an iteration snapshot, however additional data such as neural network structure may be required.
Follow the documentation of your desired framework for more information.

Note

For example, in Isaac Lab 2.0, use the following command to export the H1 flat terrain policy:

```python
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-H1-v0 --num_envs 32
```

Note

The trained policy files used in the examples are available to download [here](Robot_Simulation.md).

## Understanding the Environment Parameter File

The `agent.yaml` and `env.yaml` are generated with trained policies to describe the policy configurations and they are located in the `logs/rsl_rl/<task_name>/<time>/params/` folder.

* `agent.yaml` describes the neural network parameters.
* `env.yaml` describes the environment and robot configurations.

The below snippets are taken from Isaac-Velocity-Flat-H1-v0.

### Simulation Setup

```python
sim:
physics_prim_path: /physicsScene
dt: 0.005
render_interval: 4
gravity: !!python/tuple
- 0.0
- 0.0
- -9.81
enable_scene_query_support: false
use_fabric: true
disable_contact_processing: true
use_gpu_pipeline: true
device: cuda:0
```

The first snippet describes the simulation environment, the simulation physics is required to run at 0.005s (200hz), with gravity pointing downwards at 9.81m/s^2

### Robot Setup

The `scene:robot:init_state` section describes the robot’s initial position, orientation, velocity, as well as default joint position and velocity.

```python
init_state:
  pos: !!python/tuple
  - 0.0
  - 0.0
  - 1.05
  rot: &id003 !!python/tuple
  - 1.0
  - 0.0
  - 0.0
  - 0.0
  lin_vel: &id001 !!python/tuple
  - 0.0
  - 0.0
  - 0.0
  ang_vel: *id001
  joint_pos:
    .*_hip_yaw: 0.0
    .*_hip_roll: 0.0
    .*_hip_pitch: -0.28
    .*_knee: 0.79
    .*_ankle: -0.52
    torso: 0.0
    .*_shoulder_pitch: 0.28
    .*_shoulder_roll: 0.0
    .*_shoulder_yaw: 0.0
    .*_elbow: 0.52
  joint_vel:
    .*: 0.0
```

The `scene:robot:init_state:actuators` section below describes the robot joint properties such as effort and velocity limit, stiffness and dampening.

```python
actuators:
  legs:
    class_type: omni.isaac.lab.actuators.actuator_pd:ImplicitActuator
    joint_names_expr:
    - .*_hip_yaw
    - .*_hip_roll
    - .*_hip_pitch
    - .*_knee
    - torso
    effort_limit: 300
    velocity_limit: 100.0
    stiffness:
      .*_hip_yaw: 150.0
      .*_hip_roll: 150.0
      .*_hip_pitch: 200.0
      .*_knee: 200.0
      torso: 200.0
    damping:
      .*_hip_yaw: 5.0
      .*_hip_roll: 5.0
      .*_hip_pitch: 5.0
      .*_knee: 5.0
      torso: 5.0
    armature: null
    friction: null
```

### Observations Parameters

The observation parameters describes the observations required by the policy, as well as scale or clipping factors that need to be applied to the observation.

```python
observations:
    policy:
        concatenate_terms: true
        enable_corruption: true
        base_lin_vel:
        func: omni.isaac.lab.envs.mdp.observations:base_lin_vel
        params: {}
        noise:
            func: omni.isaac.lab.utils.noise.noise_model:uniform_noise
            operation: add
            n_min: -0.1
            n_max: 0.1
        clip: null
        scale: null
```

### Actions Parameters

The actions parameters describes the action outputted by the policy, as well as scaling factors and offsets that need to be applied to the actions.

```python
actions:
    joint_pos:
        class_type: omni.isaac.lab.envs.mdp.actions.joint_actions:JointPositionAction
        asset_name: robot
        debug_vis: false
        joint_names:
        - .*
        scale: 0.5
        offset: 0.0
        use_default_offset: true
```

### Commands Parameters

Finally, the command section describers the type of command for the policy, as well as acceptable command ranges for the policy.

```python
commands:
    base_velocity:
        class_type: omni.isaac.lab.envs.mdp.commands.velocity_command:UniformVelocityCommand
        resampling_time_range: !!python/tuple
        - 10.0
        - 10.0
        debug_vis: true
        asset_name: robot
        heading_command: true
        heading_control_stiffness: 0.5
        rel_standing_envs: 0.02
        rel_heading_envs: 1.0
        ranges:
            lin_vel_x: !!python/tuple
            - 0.0
            - 1.0
            lin_vel_y: *id006
            ang_vel_z: !!python/tuple
            - -1.0
            - 1.0
            heading: !!python/tuple
            - -3.141592653589793
            - 3.141592653589793
```

## Policy Controller Class

The robot definition class defines the robot prim, imports the robot policy, sets up the robot configurations, builds the observation tensor, and finally applies the policy control action to the robot.

### Constructor

The Constructor will spawn the robot USD, and create a single articulation object for controlling the robot.

### Load Policy

This class will load in the policy file and the corresponding environment file which the policy controller will use to set up the Isaac Sim environment.

### Initialize

The initialize function must be called once after simulation started. The purpose of this function is to match the robot configurations to the policy, by setting the robot effort mode, control mode, joint gains, joint max effort, joint max velocity, and articulation root.

### `_set_articulation_prop`

This function parses the articulation root property and set these properties to the robot.

### `_compute_action`

This function will compute the action from the observation.

### `_compute_observation`

This function must be overload by the inherited class and it is called by `advance()` during every physics step. The purpose of this function is to create an observation tensor in the format expected by the policy.
For example, the code snippet below creates the observation tensor for the H1 flat terrain policy.

```python
obs = torch.zeros(69, device=torch.device(str(self.robot._device)))
# Base lin vel
obs[:3] = lin_vel_b.squeeze()
# Base ang vel
obs[3:6] = ang_vel_b.squeeze()
# Gravity
obs[6:9] = gravity_b.squeeze()
# Command
obs[9:12] = command
# Joint states
current_joint_pos = wp.to_torch(self.robot.get_dof_positions())
current_joint_vel = wp.to_torch(self.robot.get_dof_velocities())
obs[12:31] = current_joint_pos - self.default_pos
obs[31:50] = current_joint_vel - self.default_vel
# Previous Action
obs[50:69] = self._previous_action
```

Note

Remember to multiply the observation terms by the observation scale specified in the `env.yaml`.

### Forward

This function must be overload by the inherited class and is called every physics step to generate control action for the robot.
For example, the code snippet below creates the controls for the H1 flat terrain policy.

```python
if self._policy_counter % self._decimation == 0:
    obs = self._compute_observation(command)
    self.action = self._compute_action(obs)
    self._previous_action = self.action.clone()

self.robot.set_dof_position_targets(positions=wp.from_torch(self.default_pos + (self.action * self._action_scale)))
self._policy_counter += 1
```

Note

* The policy does not need to be called every step, refer to the decimation parameter in `env.yaml`.
* Remember to multiply the action output by the action scale specified in `env.yaml`.

Warning

For position based controls, do not use `set_joint_position()` as that will teleport the joint to the desired position.

## Position to Torque Controls

Some robots may require torque control as output. If the policy generates position as an output, then you must convert position to torque. There are many ways to do this, here an actuator network is used to convert position to torque.

The actuator network class is defined in `source/extensions/isaacsim.robot.policy.examples/isaacsim/robot/policy/examples/utils/actuator_network.py`.
The actuator network policy for the Anymal robot is stored on the Content Browser at *SAMPLES* > *POLICY* > *ANYMAL\_POLICIES*

### Import Policy

For our LSTMSeaNetwork implementation, the policy file is loaded into the helper actuator network using the snippet below from the Anymal Flat Terrain Policy class:

```python
def initialize(self, physics_sim_view=None) -> None:
    """
    Initialize the articulation interface and set up drive mode.

    Args:
        physics_sim_view: The physics simulation view
    """
    super().initialize(physics_sim_view=physics_sim_view, control_mode="effort")

    # Actuator network
    assets_root_path = get_assets_root_path()
    file_content = omni.client.read_file(
        assets_root_path + "/Isaac/IsaacLab/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt"
    )[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    self._actuator_network = LstmSeaNetwork()
    self._actuator_network.setup(file, self.default_pos)
    self._actuator_network.reset()
```

### Run the Actuator Network

In the advance function, insert the position outputs from the locomotion policy into the actuator network and apply the torque to the robot using the snippet below:

```python
current_joint_pos = self.get_joint_positions()
current_joint_vel = self.get_joint_velocities()

joint_torques, _ = self._actuator_network.compute_torques(
    current_joint_pos, current_joint_vel, self._action_scale * self.action
)

self.set_joint_efforts(joint_torques)
```

## Debugging Tips

If your robot doesn’t work right away, you can use the following tips to start debugging:

### Verify Your Policy

You can start by verifying that your policy is working properly by [playing it in Isaac Lab.](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/run_rl_training.html#playing-the-trained-agent)

Remember to use the correct `play.py` for your workflow and select the correct task.

### Verify the Robot Joint Properties

#### Robot Joint Order

If the policy is working on Isaac Lab, then you should verify is the joint order of the robot, joint properties, and default joint positions.

To see the joint order, open your asset USD, create an articulation with the robot prim, start the simulation, initialize articulation, and call the `dof_names` function.

```python
from isaacsim.core.experimental.prims import Articulation

# Open your USD and PLAY the simulation before running this snippet
prim = Articulation(prim_paths_expr="<your_robot_prim_path>")
prim.initialize()
print(str(prim.dof_names))
```

Print out the `dof_names` for both the Isaac Sim asset and the asset you used to train in Isaac Lab, make sure that the names and orders match exactly.

The ANYmal robot below has control commands in the wrong order, as a result the robot is falling over.

#### Default Joint Position

After you have the joint positions, verify that your default joint positions are inserted correctly. If the joint positions are incorrect, the robot joints will not go to the correct position.

For example, in the video below, the ankle joint was set incorrectly and the H1 humanoid was tip toeing, doing a “moonwalk”.

#### Robot Joint Properties

If you observe the joints are moving too much or not enough, then the joint properties may not be set up correctly.

```python
from isaacsim.core.experimental.prims import Articulation

# Open your USD and PLAY the simulation before running this snippet
prim = Articulation(prim_paths_expr="<your_robot_prim_path>")
prim.initialize()
print(str(prim.dof_properties))
```

Then, you can compare the joint properties with the env YAML file generated by Isaac Lab. Check the articulation API documentation for the properties for the DOFs.

For example, in the video below, the spot robot’s stiffness and dampening are set too high, resulting in underactuated movement.

For example, in the video below, the H1 robot’s arm stiffness and dampening are set too low, resulting in over movement.

### Verify the Simulation Environment

If the robot matches exactly and the inference examples are still not working, then it’s time to check the simulation parameters.

#### Physics Scene

Physics scene describes the time stepping with `Time Steps Per Second (Hz)`, so take the inverse of the `dt` parameter in the `env.yaml` and set this correctly.
Also match the physics scene properties with the physx section of the `env.yaml` file.

For example, in the video below, time step was set to 60Hz, instead of the 500Hz expected by the controller.

### Verify the Observation and Action Tensor

Finally, verify the observation and action tensors, and make sure your tensor structures are correct, the data passed in to the tensors are correct, and the correct scale factors are applied to the input and outputs.

Also, make sure the actions output from the policy matches the expected type of inputs of articulation and are in the correct order to correctly power the robot.

## Sim To Real Deployment

Congratulations, your robot and policy are working correctly in Isaac Sim now and you have tested it with the rest of your stack. Now it’s time to deploy it on a real robot.

Please read this [article](https://developer.nvidia.com/blog/closing-the-sim-to-real-gap-training-spot-quadruped-locomotion-with-nvidia-isaac-lab/) on deploying an reinforcement learning policy to a spot robot.

---

# Running a Reinforcement Learning Policy through ROS 2 and Isaac Sim

## Learning Objectives

In this example, you learn to run a reinforcement learning policy through ROS 2 and Isaac Sim. You will learn to:

* Setup a ROS 2 node to publish observations and receive actions from Isaac Sim for the H1 flat terrain locomotion policy
* Setup Isaac Sim environment to run a reinforcement learning policy

## Getting Started

**Prerequisite**

* The `torch` package is required to run this sample. Follow the [PyTorch](https://pytorch.org/get-started/locally) installation instructions to install it (if not already installed).
  Since PyTorch will run on a separate process, no specific version is required (it doesn’t have to match Isaac Sim’s PyTorch version).
* Enable the `isaacsim.ros2.bridge` Extension in the Extension Manager window by navigating to **Window** > **Extensions**.
* This tutorial requires the `h1_fullbody_controller` ROS 2 package, which is provided in [IsaacSim-ros\_workspaces](https://github.com/isaac-sim/IsaacSim-ros_workspaces) repo. Complete [ROS 2 Installation (Default)](ROS_2.md) to make sure the ROS 2 workspace environment is setup correctly.
* This tutorial requires the completion of [Tutorial 13: Rigging a Legged Robot for Locomotion Policy](Robot_Setup.md) to setup the robot joint configurations based on the locomotion policy parameter, see the section below.

Hint

* If you encounter `error: externally-managed-environment` when installing PyTorch, try installing it in a virtual Python environment.
* If you encounter `ModuleNotFoundError: No module named 'yaml'`, install PyYaml using `pip`.

## About the H1 Flat Terrain Locomotion Policy

The policy is trained on based on the **Isaac-Velocity-Flat-H1-v0** environment from Isaac Lab. This policy tracks a velocity command on a flat terrain for the H1 humanoid robot. The policy is capable of walking forward and turning left/right. The policy does not support moving backwards nor sideways.

## Set Up Robot Joint Configurations

Follow the steps in [Tutorial 13: Rigging a Legged Robot for Locomotion Policy](Robot_Setup.md) to setup the robot joint configurations based on the locomotion policy parameter. This step is very important, because mismatching the joint configurations can result in unexpected robot behavior.

> * The H1 flat terrain policy environment definition file is in [YAML file](https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0/Isaac/Samples/Policies/H1_Policies/h1_env.yaml).
> * The angle units specified in the policy environment definition file are in radians. The Isaac Sim USD GUI interface expects the angles to be specified in degrees.
> * The rigged H1 robot is available in the content browser at `Isaac/Samples/Rigging/H1/h1_rigged.usd`.

## Add IMU Sensor

Use the IMU sensor to obtain the body frame linear acceleration, angular velocity, and orientation.
The flat terrain policy requires the linear velocity, angular velocity, and gravity vector from the pelvis link. You need to add an IMU sensor to the pelvis link to compute these values.

* You can create an IMU sensor by right clicking on the `/h1/pelvis` and select **Create** > **Isaac** > **Sensors** > **Imu Sensor**.

Warning

If you add the IMU to a different link, for example, the torso link, you must first transform the IMU data to the pelvis link frame before using it in the policy.

## Set up ROS 2 Node for the H1 Humanoid Robot

The ROS 2 node publishes the observations and receives the actions from Isaac Sim. As specified in the environment definition file, the observations requires the following information:

* Body frame linear velocity
* Body frame angular velocity
* Body frame gravity vector
* Command (linear and angular velocity)
* Relative joint position
* Relative joint velocity
* Previous Action

You can obtain the body frame linear velocity, angular velocity, and gravity vector from processing the IMU data.
The command is the desired linear and angular velocity of the robot, which can be retrieved from a ROS 2 twist message.
The relative joint position and velocity can be computed from the Isaac Sim joint state topic.
The previous action is the action applied last iteration and can be tracked by the policy node.

The action is a joint state message, which is a dictionary of joint names and their desired positions.

In this section, we will setup OmniGraph nodes that publishes the observations and receives the actions from Isaac Sim on physics step.

### Create an On Demand OmniGraph

1. Open the H1 Unitree robot model that you rigged in the [Tutorial 13: Rigging a Legged Robot for Locomotion Policy](Robot_Setup.md) tutorial.
2. Create a scope to hold the ActionGraphs by right clicking on the stage and selecting **Create** > **Scope**, rename it “Graph”.
3. Right click on the stage and select **Create** > **Visual Scripting** > **ActionGraph**.
4. Rename the ActionGraph to “ROS\_Imu”, drag and drop this ActionGraph into the “Graph” scope.
5. Left click on the ActionGraph node, scroll down in the property editor and set the `pipelineStage` to `pipelineStageOnDemand`.

This will ensure the ActionGraph node runs when the Isaac Sim physics steps.

### Create Imu Publisher Node

This node publishes the IMU data to ROS 2, which contains the body frame linear acceleration, angular velocity, and orientation.

1. Right click on the actionGraph node and select **Open Graph**.
2. Copy the following nodes into the Action graph:

   * `On Physics Step`: This node is triggered when the Isaac Sim physics steps, and runs the entire graph.
   * `ROS2 Context`: This node creates a context for the ROS 2 node.
   * `ROS2 QoS Profile`: This node sets the QoS profile for the ROS 2 node.
   * `Isaac Read IMU Node`: This node reads the IMU data from Isaac Sim.
   * `Isaac Read Simulation Time`: This node reads the simulation time from Isaac Sim.
   * `ROS2 Publish IMU`: This node publishes the IMU data to ROS 2 using the `Isaac Read IMU Node` and `Isaac Read Simulation Time` nodes as source.
3. Connect the nodes as shown in the image below.

   * Set the `Isaac Read IMU Node` input `IMU Prim` to `/h1/pelvis/Imu_Sensor` to read the IMU sensor data.
   * Uncheck input `Read Gravity` of the `Isaac Read IMU Node` to avoid reading the gravity vector from the pelvis link. This is because we only want the linear and angular velocity from the pelvis link.
   * Check the `Reset on Stop` input of `Read Simulation Time` node to reset the simulation time when the simulation stops.

### Create Joint State Publisher and Subscriber Nodes

This node publishes the joint states to ROS 2, which contains the joint names, positions, and velocities, and subscribes to the joint state commands from Isaac Sim.

1. Create a new ActionGraph node and rename it to “ROS\_Joint\_States”.
2. Set the `pipelineStage` to `pipelineStageOnDemand`.
3. Copy the following nodes into the Action graph:

   * `On Physics Step`: This node is triggered when the Isaac Sim physics steps, and runs the entire graph.
   * `ROS2 Context`: This node creates a context for the ROS 2 node.
   * `ROS2 QoS Profile`: This node sets the QoS profile for the ROS 2 node.
   * `ROS2 Subscribe Joint State`: This node subscribes to the joint states commands from the external policy node.
   * `ROS2 Publish Joint State`: This node publishes the current joint states to ROS 2 from Isaac Sim.
   * `Isaac Read Simulation Time`: This node reads the simulation time from Isaac Sim.
   * `Articulation Controller`: This node will execute the joint state commands from the Subscribe joint States node.
4. Connect the nodes as shown in the image below.

   * Set the `ROS2 Publish Joint State` input `Target Prim` to `/h1`, and `Topic Name` to `/joint_states`.
   * Set the `ROS2 Subscribe Joint State` input `Topic Name` to `/joint_command`.
   * Set the `Articulation Controller` input `Target Prim` to `/h1`.
   * Check the `Reset on Stop` input of `Read Simulation Time` node to reset the simulation time when the simulation stops.

Note

The completed asset is available in the content browser at `Isaac Sim/Samples/ROS2/Robots/h1_ROS.usd`.

## Publish ROS Clock and Set Up Environment

Now that the asset is set up, create a simulation scenario to place the robot in, configure the physics settings, and ROS time publish.

### Setup Simulation Scenario

1. Create a new file, in the Content Browser, go to `Isaac Sim/Environments/Simple_Warehouse` and drag the `warehouse.usd` asset into the stage.
2. Drag and drop the `h1_ROS.usd` asset that you made earlier into the stage. Set the Z transform to `1.0` so it is above the ground.
3. Create a `Physics Scene` by right clicking on the stage and selecting **Create** > **Physics** > **Physcis Scene**.
4. Select the `Physics Scene` and set `Time Steps Per Second` to `200`.
5. Because you only have one robot, use CPU physics for better performance.

   * Uncheck `Enable GPU Dynamics`
   * Set the `Broadphase Type` to `MBP`

### Setup ROS 2 Clock Publisher

1. Create a new ActionGraph node and rename it to “ROS\_Clock”.
2. Set the `pipelineStage` to `pipelineStageOnDemand`.
3. Copy the following nodes into the Action graph:

   * `On Physics Step`: This node is triggered when the Isaac Sim physics steps, and runs the entire graph.
   * `ROS2 Context`: This node creates a context for the ROS 2 node.
   * `ROS2 QoS Profile`: This node sets the QoS profile for the ROS 2 node.
   * `ROS2 Publish Clock`: This node publishes the ROS 2 clock to ROS 2.
   * `Read Simulation Time`: This node reads the simulation time from Isaac Sim.
4. Connect the nodes as shown in the image below.

   * Check the `Reset on Stop` input of `Read Simulation Time` node to reset the simulation time when the simulation stops.

Note

The completed environment is available in the content browser at `Isaac Sim/Samples/ROS2/Scenario/h1_ros_locomotion_policy_tutorial.usd`.

## Run ROS 2 Policy

The asset is set up, you can run the ROS 2 policy. Build the ROS 2 workspace and source the `setup.bash` file.

1. Launch the `h1_fullbody_controller` ROS 2 package by running the following command in the environment with PyTorch installed:

   > ```python
   > ros2 launch h1_fullbody_controller h1_fullbody_controller.launch.py
   > ```

Note

This ROS 2 package computes observations and actions using the ROS messages that you published above and the flat terrain locomotion policy.
When no command velocities are received, the robot will stand still and maintain balance. Make sure to start the ROS 2 policy before starting the simulation, otherwise the robot will fall over.

2. Open the H1 scenario you created earlier and press **PLAY** to start the simulation.
3. In a separate terminal, source ROS and launch `teleop_twist_keyboard` or another desired package to publish Twist messages:

   > ```python
   > ros2 run teleop_twist_keyboard teleop_twist_keyboard
   > ```

You can now control the H1 humanoid robot using your keyboard. Try the controls and observe if the robot moves as expected.

* Forward: `i`
* Forward + Turn Left: `u`
* Forward + Turn Right: `o`
* Turn Left: `j`
* Turn Right: `l`
* Stand Still: `k`

Important

* Moving backwards is not supported in this version of the policy. Pressing `m`, `,`, `.` key will cause the robot to fall over.
* Setting linear and angular velocity above 0.75 exceeds the velocity limits of the policy and will cause the robot to fall over.
* The robot might drift overtime when there’s no command velocities. This is expected behavior.

## Summary

This tutorial covered:

1. Creating and setting up an ROS 2 node to publish observations and receive actions from Isaac Sim for the H1 flat terrain locomotion policy.
2. Setting up Isaac Sim environment to run a reinforcement learning policy.

### Next Steps

* Learn more about [Isaac Lab](Isaac_Lab.md) here and the Isaac Sim native method for [policy deployment](Isaac_Lab.md).

---

# Getting Started with Cloner

Training reinforcement learning policies can often benefit from collecting trajectories from vectorized copies of environments performing the same task. The Cloner interface is designed to simplify the environment design process for such a scene by providing APIs that allow users to clone a given environment as many times as desired.

In addition to providing cloning functionality, the Cloner interface also provides utilities to generate target paths, automatically compute target transforms, as well as filtering out collisions between clones.

## Learning Objectives

In this tutorial, we will walk through the Cloner interface:

1. Set up an example using the Cloner class
2. Set up an example using the GridCloner class
3. Use APIs from isaacsim.core.api to access cloned objects
4. Understand advanced cloning with physics replication and additional parameters

*10-15 Minute Tutorial*

## Getting Started

We will first launch Isaac Sim and enable the Cloner extension. Open the Extensions window from the UI by navigating to Window > Extensions from the top menu bar. Find the Isaac Sim Cloner extension, or isaacsim.core.cloner and enable the extension via the toggle switch on the right side of the extension name.

Next, open the Script Editor window from the UI by navigating to Window > Script Editor from the top menu bar. All example code in this tutorial can be pasted into the Script Editor window and executed by clicking on Run.

## Introduction to Cloner

Please make sure isaacsim.core.cloner is enabled from the Extensions window before running the snippets.

Let’s first start with a simple use case of the Cloner interface. In this example, we will create a scene with 4 cubes.

```python
from isaacsim.core.cloner import Cloner  # import Cloner interface
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom

# create our base environment with one cube
base_env_path = "/World/Cube_0"
UsdGeom.Cube.Define(get_current_stage(), base_env_path)

# create a Cloner instance
cloner = Cloner()

# generate 4 paths that begin with "/World/Cube" - path will be appended with _{index}
target_paths = cloner.generate_paths("/World/Cube", 4)

# clone the cube at target paths
cloner.clone(source_prim_path="/World/Cube_0", prim_paths=target_paths)
```

We should now have 4 cubes in our stage: “/World/Cube\_0”, “/World/Cube\_1”, “/World/Cube\_2”, “/World/Cube\_3”. But you may have noticed that the cubes have all been created at the same position.

We can add a transform to each cube, simply replace the last line of the previous code with the following:

```python
import numpy as np

cube_positions = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]])

# clone the cube at target paths at specified positions
cloner.clone(source_prim_path="/World/Cube_0", prim_paths=target_paths, positions=cube_positions)
```

It is also possible to specify the orientations of each clone by passing in an orientations argument, which should also be a np.ndarray.

## Grid Cloner

Grid Cloner is a specialized Cloner class that automatically places clones in a grid, without requiring pre-computed translations and orientations from the user.

To use the Grid Cloner, we will need to specify the spacing we would like between each clone at initialization.

```python
from isaacsim.core.cloner import GridCloner  # import GridCloner interface
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom

# create our base environment with one cube
base_env_path = "/World/Cube_0"
UsdGeom.Cube.Define(get_current_stage(), base_env_path)

# create a GridCloner instance
cloner = GridCloner(spacing=3)

# generate 4 paths that begin with "/World/Cube" - path will be appended with _{index}
target_paths = cloner.generate_paths("/World/Cube", 4)

# clone the cube at target paths
cloner.clone(source_prim_path="/World/Cube_0", prim_paths=target_paths)
```

Now we have a scene with 4 cubes placed in a grid!

## Accessing Cloned Objects

Now that we have created our scene with the Cloner interface, we can access states for the cloned objects using APIs from isaacsim.core.api. These APIs allow us to collect and apply data as vectorized tensors to all or a subset of objects at once, avoiding iterating through objects in loops.

We will show a simple example of retrieving the global transforms of all of the boxes in the scene, as well as setting a new translation on the boxes.

```python
# import the XFormPrimView interface from isaacsim.core.api for APIs for XForm prims
from isaacsim.core.prims import XFormPrimView

# retrieve a View containing all 4 boxes by using a wildcard expression that matches the prim paths for all boxes
boxes = XFormPrimView("/World/Cube_*")

# retrieve the global transforms of all boxes
#   - positions will be a vector of shape (4, 3) for X, Y, Z axes of translation
#   - orientations will be a vector of shape (4, 4) for W, X, Y, Z axes of quaternion
positions, orientations = boxes.get_world_poses()

# increase positions on the Z axis to move boxes up by 1.5 units
positions[:, 2] += 1.5
# apply the new positions
boxes.set_world_poses(positions, orientations)
```

## Physics Replication

The cloning process can take advantage of faster physics parsing by replicating physics directly in PhysX, avoiding copying of USD physics properties. This feature can be enabled by passing in a new parameter replicate\_physics=True when cloning objects in the scene. Note that to use this feature, the user must also specify some additional parameters: base\_env\_paths and root\_path. base\_env\_paths points to the ancestry prim of all clones and root\_path specifies the prefix of each target clone path before the index. This also imposes the limitation that all target clone paths must be appended by an incremental index. If both define\_base\_env() and generate\_paths() APIs have already been called before cloning, the user can avoid specifying base\_env\_paths and root\_path parameters as the information has already been provided to the Cloner class.

```python
cloner.clone(
    source_prim_path="/World/Ants/Ant_0",
    prim_paths=target_paths,
    position_offsets=position_offsets,
    replicate_physics=True,
    base_env_path="/World/Ants",
    root_path="/World/Ants/Ant_",
)
```

A full example using physics replication can be found at standalone\_examples/api/isaacsim.core.cloner/cloner\_ants.py.

There are currently some features that are not supported by physics replication. For example, runtime modification of shape properties are not allowed on prims that have been created using physics replication. For scenes that require randomization or modification of shape properties (such as materials, friction, restitution, etc.) at run time, please do not enable physics replication when cloning objects.

## Additional Parameters

In addition to physics replication, the Cloner also provides an option to copy from the source prim. This flag can be set with the copy\_from\_source argument.

```python
cloner.clone(
    source_prim_path="/World/Ants/Ant_0",
    prim_paths=target_paths,
    position_offsets=position_offsets,
    replicate_physics=True,
    base_env_path="/World/Ants",
    root_path="/World/Ants/Ant_",
    copy_from_source=True,
)
```

By default, copy\_from\_source is set to False, in which case the cloned prims will be defined as [USD Inherits](https://openusd.org/release/api/class_usd_inherits.html) of the source prim. The cloning process will be faster when USD Inherits are used for cloning. However, any changes that are made to the source prim *after* cloning will also reflect in the cloned prims.

If this behavior is undesired, please set copy\_from\_source to True. When copy\_from\_source is set to True, the cloned prims will be defined as **copies** of the source prim. After cloning, each cloned prim will be an individual entity and any changes in the source prim **will not** be reflected on the cloned prims. This setting can be useful in cases where cloned environments are not designed to be identical.

## Summary

This tutorial covered the following topics:

1. How to use the Cloner interface
2. How to use the GridCloner interface
3. How to access states of cloned objects with isaacsim.core.api APIs
4. Advanced cloning with physics replication and additional parameters

### Next Steps

Continue on to the next tutorial in our Reinforcement Learning Tutorials series, [Instanceable Assets](Isaac_Lab.md), to learn about instanceable assets for improving memory efficiency.

---

# Instanceable Assets

Reinforcement learning often requires training in large simulation scenes with multiple clones of the same robots. As we add more and more robots into the simulation environment, the memory consumption also increases for each additional set of robot and mesh assets added. To reduce memory consumption, we can take advantage of USD’s [Scenegraph Instancing](https://graphics.pixar.com/usd/dev/api/_usd__page__scenegraph_instancing.html) functionality to mark common meshes shared by different copies of the same robots as instanceable.

By doing so, each copy of the robot will reference a single copy of meshes, avoiding the need to create multiple copies of the same meshes in the scene, thus reducing memory usage in the overall simulation environment.

## Learning Objectives

In this tutorial, we will show how to create instanceable assets in Isaac Sim. We will

1. Explain requirements for making assets instanceable
2. Use the URDF and MJCF importers to create instanceable assets
3. Show utility methods to convert existing assets to instanceable assets

*10-15 Minute Tutorial*

## Getting Started

* Please refer to USD Documentation on [Scenegraph Instancing](https://graphics.pixar.com/usd/dev/api/_usd__page__scenegraph_instancing.html) for more details on instancing.
* Please refer to [Tutorial: Import URDF](Importers_and_Exporters.md) and [Tutorial: Import MJCF](Importers_and_Exporters.md) for more details on importer functionalities.

## Hierarchy Requirement for Instanceable Assets

USD prohibits modifying properties of prims on descendants of instanced prims. Therefore, we generally only perform instancing on mesh prims for robot assets, since properties on meshes will not differ across different environments during simulation. However, the transforms of the meshes may be different during simulation when robots in each environment are being moved in varying ways. Thus, we have to define the topology of our robot hierarchy in a specific structure in the asset tree definition in order for the instanceable flag to take action.

To mark any mesh or primitive geometry prim in the asset as instanceable, the mesh prim requires a parent Xform prim to be present, which will be used to add a reference to a master USD file containing definition of the mesh prim.

For example, the following definition cannot be marked instanceable:

```python
World
  |_ Robot
       |_ Collisions
               |_ Sphere
               |_ Box
```

Instead, it will have to be modified to:

```python
World
  |_ Robot
       |_ Collisions
               |_ Sphere_Xform
               |      |_ Sphere
               |_ Box_Xform
                      |_ Box
```

Any references that exist on the original Sphere and Box prims would have to be moved to Sphere\_Xform and Box\_Xform prims.

## Using URDF and MJCF Importers

Isaac Sim provides two importers - URDF and MJCF - for converting robot assets to USD format to be used in Isaac Sim. Both importers support the option to import robot assets directly as instanceable assets. By selecting this option, imported assets will be split into two separate USD files that follow the above hierarchy definition. Any mesh data will be written to an USD stage to be referenced by the main USD stage, which contains the main robot definition.

To use the Instanceable option in the importers, first check the Create Instanceable Asset option. Then, specify a file path to indicate the location for saving the mesh data in the Instanceable USD Path textbox. This will default to ./instanceable\_meshes.usd, which will generate a file instanceable\_meshes.usd that is saved to the current directory.

Once the asset is imported with these options enabled, you will see the robot definition in the stage - we will refer to this stage as the master stage. If we expand the robot hierarchy in the Stage, we will notice that the parent prims that have mesh descendants have been marked as Instanceable and they reference a prim in our Instanceable USD Path USD file. We are also no longer able to modify attributes of descendant meshes.

To add our instanced asset into a new stage, we will simply need to add our master USD file.

## Modifying Existing Assets

Due to limitations of the topology requirement for making assets instanceable, it is not as straightforward to convert existing non-instanceable assets to become instanceable. Here, we will try to provide a few small utility methods to help make the process simpler.

All utilities should be copied into and run from the script editor, which can be opened from Window > Script Editor.

First, we need to make sure our existing asset follows the hierarchy constraint defined above, where all mesh prims have a parent XForm prim present that can be used to mark the prim as instanceable. To help with the process of creating new parent prims, we provide a utility method create\_parent\_xforms() below to automatically insert a new Xform prim as a parent of every mesh prim in the stage.

```python
import omni.client
import omni.usd
from pxr import Sdf, UsdGeom

def create_parent_xforms(asset_usd_path, source_prim_path, save_as_path=None):
    """Adds a new UsdGeom.Xform prim for each Mesh/Geometry prim under source_prim_path.
    Moves material assignment to new parent prim if any exists on the Mesh/Geometry prim.

    Args:
        asset_usd_path (str): USD file path for asset
        source_prim_path (str): USD path of root prim
        save_as_path (str): USD file path for modified USD stage. Defaults to None, will save in same file.
    """
    omni.usd.get_context().open_stage(asset_usd_path)
    stage = omni.usd.get_context().get_stage()

    prims = [stage.GetPrimAtPath(source_prim_path)]
    edits = Sdf.BatchNamespaceEdit()
    while len(prims) > 0:
        prim = prims.pop(0)
        print(prim)
        if prim.GetTypeName() in ["Mesh", "Capsule", "Sphere", "Box"]:
            new_xform = UsdGeom.Xform.Define(stage, str(prim.GetPath()) + "_xform")
            print(prim, new_xform)
            edits.Add(Sdf.NamespaceEdit.Reparent(prim.GetPath(), new_xform.GetPath(), 0))
            continue

        children_prims = prim.GetChildren()
        prims = prims + children_prims

    stage.GetRootLayer().Apply(edits)

    if save_as_path is None:
        omni.usd.get_context().save_stage()
    else:
        omni.usd.get_context().save_as_stage(save_as_path)
```

This method can be run on an existing non-instanced USD file for an asset from the script editor, where:

* asset\_usd\_path is the file path to the current existing USD asset
* source\_prim\_path is the USD prim path to the root prim of the asset
* save\_as\_path is a different file path to same the modified asset to. This can be left unspecified to overwrite the existing file.

```python
create_parent_xforms(asset_usd_path=ASSET_USD_PATH, source_prim_path=SOURCE_PRIM_PATH, save_as_path=SAVE_AS_PATH)
```

It is worth noting that any [USD Relationships](https://graphics.pixar.com/usd/dev/api/class_usd_relationship.html) on the referenced meshes will be removed. This is because those USD Relationships originally have targets set to prims in the original prim that may no longer be valid and hence cannot be accessed from the new stage. Common examples of USD Relationships that could exist on the meshes are visual materials, physics materials, and filtered collision pairs. Therefore, it is recommended to set these USD Relationships on the meshes’ parent Xforms instead of the meshes themselves.

The above method can also be run as part of an overall conversion process, which is defined in the utility below. This utility will first insert new parent prims if create\_xforms=True is specified, and generate a new USD file that is used for referencing. It will then traverse through the asset tree and mark the parent prim of any mesh or primitive type prims as instanceable, along with inserting a reference to the mesh USD stage.

```python
create_parent_xforms(asset_usd_path=ASSET_USD_PATH, source_prim_path=SOURCE_PRIM_PATH, save_as_path=SAVE_AS_PATH)
```

## Summary

This tutorial covered the following topics:

1. Requirements for creating instanceable assets
2. Using the URDF and MJCF Importers to create instanceable assets
3. Making existing assets instanceable

---

# Isaac Lab Troubleshooting

This page consolidates troubleshooting information for Isaac Lab components in Isaac Sim.

## Common Issues

### Installation Issues

* Make sure you have the correct Python version (3.9+) when setting up Isaac Lab
* If encountering ModuleNotFoundError, ensure all dependencies are installed via pip install -e . in the Isaac Lab repository
* For GPU compatibility issues, verify that your CUDA version matches the requirements for Isaac Lab

### Performance Issues

* For slow training performance, try reducing the number of environments or the complexity of the scene
* Memory issues can be resolved by setting smaller batch sizes or reducing environment complexity
* To improve frame rates during training, consider using fewer sensors or reducing their resolution

### Environment Setup Issues

* If robots fail to initialize, check that the URDF/USD files are correctly specified
* For task initialization failures, ensure your task configuration files are properly formatted
* Make sure reward terms and observations are correctly defined in your environment configurations

### Policy Deployment Issues

* When deploying trained policies, verify the observation space matches the training environment
* For poor policy performance after deployment, check if the simulation parameters match the training settings
* If policy files fail to load, verify they are in the correct format supported by Isaac Lab