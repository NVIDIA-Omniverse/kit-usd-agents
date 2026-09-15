# Isaac Sim Assets

Isaac Sim provides a variety of assets and robots to help you build your virtual world. Some are made specifically for Isaac Sim and robotics applications,
others are made for other NVIDIA Omniverse-based applications. The ones that are available to you by default are all located in the **Window > Browsers** tab.

The [Content Browser](Browsers.md) is where you can find all of Isaac Sim assets and files. This includes all of the assets listed below, as well as URDF file, config files, policy binaries, and more.

Sample assets are available for download with the [Latest Release](Installation.md) of Isaac Sim.
To use this content, you must download the files to the local disk or a Nucleus server.
All asset paths below are assumed to be relative to the default asset root path in the persistent.isaac.asset\_root.default setting. See [Local Assets Packs](Installation.md)

Note

Assets will take longer to load when they are accessed for the first time; robots may take multiple minutes to load and larger environment scenes may take as long as ten minutes or more.

## Categories

- Robot Assets
- Camera and Depth Sensors
- Non-Visual Sensors
- Prop Assets
- Environment Assets
- Featured Assets
- Third-Party SimReady USD Assets
- Neural Volume Rendering

## Omniverse Activity UI

The [Omniverse Activity UI](https://docs.omniverse.nvidia.com/kit/docs/omni.activity.ui) allows you to monitor the progress and activities when assets are being loaded.

Enable the `omni.activity.ui` extension in the Extension Manager (**Window > Extensions** menu),
or launch Isaac Sim from a terminal with the argument `--enable omni.activity.ui`.
Then, open the **Activity Progress** window (**Window > Utilities > Activity Progress** menu) before opening or loading the USD asset to monitor its loading progress.

---

# Robot Assets

NVIDIA Isaac Sim supports a wide range of robots with differential bases, form factors, and functions.

These robots can be categorized as wheeled robots, holonomic robots, quadruped robots, robotic manipulator and aerial robots (drones). They can be found in the Content Browser in the `Isaac Sim/Robots` folder.

Wheeled

**iRobot**

Create3

**USD Path:** iRobot/Create3/create\_3.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 4 |
| Number of Links | 5 |
| Number of DOFs | 4 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Turtlebot**

Turtlebot3

**USD Path:** Turtlebot/Turtlebot3/turtlebot3\_burger.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

**NVIDIA**

Robomaker

**USD Path:** NVIDIA/Robomaker/aws\_robomaker\_jetbot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 5 |
| Number of Links | 6 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

NovaCarter

**USD Path:** NVIDIA/NovaCarter/nova\_carter.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 7 |

| Sensor/Accessory | Count |
| --- | --- |
| Camera | 12 |
| IMU | 5 |
| OmniSensor Lidar | 3 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

Leatherback

**USD Path:** NVIDIA/Leatherback/leatherback.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 26 |
| Number of Links | 27 |
| Number of DOFs | 26 |

| Sensors | Count |
| --- | --- |
| Camera | 4 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX CollisionAPI

Jetbot

**USD Path:** NVIDIA/Jetbot/jetbot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

| Sensors | Count |
| --- | --- |
| Camera | 2 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

Carter

Variant 1

**USD Path:** NVIDIA/Carter/carter\_v1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 4 |

| Sensors | Count |
| --- | --- |
| Camera | 5 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

Variant 2

**USD Path:** NVIDIA/Carter/carter\_v1\_physx\_lidar.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 4 |

| Sensors | Count |
| --- | --- |
| Camera | 4 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

**IsaacSim**

ForkliftC

**USD Path:** IsaacSim/ForkliftC/forklift\_c.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 7 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX CollisionAPI

ForkliftB

Variant 1

**USD Path:** IsaacSim/ForkliftB/forklift\_b.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 7 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX CollisionAPI

Variant 2

**USD Path:** IsaacSim/ForkliftB/forklift\_b\_sensor.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

| Sensor/Accessory | Count |
| --- | --- |
| Camera | 6 |
| IMU | 3 |
| OmniSensor Lidar | 1 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX CollisionAPI

**Idealworks**

iwhub

Variant 1

**USD Path:** Idealworks/iwhub/iw\_hub.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 7 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

Variant 2

**USD Path:** Idealworks/iwhub/iw\_hub\_sensors.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

| Sensors | Count |
| --- | --- |
| Camera | 2 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX CollisionAPI
* PhysX JointAPI
* PhysX ArticulationAPI
* PhysX SceneAPI

Variant 3

**USD Path:** Idealworks/iwhub/iw\_hub\_static.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Fraunhofer**

Evobot

**USD Path:** Fraunhofer/Evobot/evobot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX ArticulationAPI
* PhysX SceneAPI

**Clearpath**

Jackal

Variant 1

**USD Path:** Clearpath/Jackal/jackal.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 4 |
| Number of Links | 5 |
| Number of DOFs | 4 |

| Sensor/Accessory | Count |
| --- | --- |
| Camera | 2 |
| IMU | 1 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** Clearpath/Jackal/jackal\_basic.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 4 |
| Number of Links | 5 |
| Number of DOFs | 4 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Dingo

Variant 1

**USD Path:** Clearpath/Dingo/dingo.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

| Sensors | Count |
| --- | --- |
| Camera | 2 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** Clearpath/Dingo/dingo\_basic.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**AgilexRobotics**

limo

**USD Path:** AgilexRobotics/limo/limo.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 4 |
| Number of Links | 5 |
| Number of DOFs | 4 |

| Sensors | Count |
| --- | --- |
| Camera | 1 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

Manipulator

**Yaskawa**

Motoman Next

NHC12

**USD Path:** Yaskawa/Motoman Next/NHC12/NHC12\_A00.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

NEX7

**USD Path:** Yaskawa/Motoman Next/NEX7/NEX7\_C00\_c00.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

NEX4

**USD Path:** Yaskawa/Motoman Next/NEX4/NEX4\_C00.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

NEX35

**USD Path:** Yaskawa/Motoman Next/NEX35/NEX35\_C00.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

NEX20

**USD Path:** Yaskawa/Motoman Next/NEX20/NEX20\_C00.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

NEX10

**USD Path:** Yaskawa/Motoman Next/NEX10/NEX10\_C00.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

**Yahboom**

Dofbot

**USD Path:** Yahboom/Dofbot/dofbot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 11 |
| Number of Links | 12 |
| Number of DOFs | 11 |

| Sensors | Count |
| --- | --- |
| Camera | 1 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

**WonikRobotics**

AllegroHand

Variant 1

**USD Path:** WonikRobotics/AllegroHand/allegro.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 20 |
| Number of Links | 21 |
| Number of DOFs | 16 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** WonikRobotics/AllegroHand/allegro\_hand.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 20 |
| Number of Links | 21 |
| Number of DOFs | 16 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

Variant 3

**USD Path:** WonikRobotics/AllegroHand/allegro\_hand\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 20 |
| Number of Links | 21 |
| Number of DOFs | 16 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

**UniversalRobots**

ur5e

**USD Path:** UniversalRobots/ur5e/ur5e.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur5

**USD Path:** UniversalRobots/ur5/ur5.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur3e

**USD Path:** UniversalRobots/ur3e/ur3e.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur30

**USD Path:** UniversalRobots/ur30/ur30.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX ArticulationAPI

ur3

**USD Path:** UniversalRobots/ur3/ur3.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur20

**USD Path:** UniversalRobots/ur20/ur20.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur16e

**USD Path:** UniversalRobots/ur16e/ur16e.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur10e

**USD Path:** UniversalRobots/ur10e/ur10e.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Accessories:**

* Robotiq\_2f\_140
* Robotiq\_2f\_85

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

ur10

**USD Path:** UniversalRobots/ur10/ur10.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 6 |

**Accessories:**

* Long\_Suction
* Short\_Suction

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX ArticulationAPI

**Unitree**

Z1

**USD Path:** Unitree/Z1/z1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

Dex5

**USD Path:** Unitree/Dex5/Dex5-URDF-R.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 20 |
| Number of Links | 21 |
| Number of DOFs | 20 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

Dex3

**USD Path:** Unitree/Dex3/dex3\_1\_r.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 7 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

**Ufactory**

xarm\_gripper

**USD Path:** Ufactory/xarm\_gripper/xarm\_gripper.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX ArticulationAPI

xarm7

**USD Path:** Ufactory/xarm7/xarm7.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 13 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX ArticulationAPI

xarm6

**USD Path:** Ufactory/xarm6/xarm6.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 13 |
| Number of Links | 14 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX ArticulationAPI

uf850

**USD Path:** Ufactory/uf850/uf850.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

lite6\_gripper

**USD Path:** Ufactory/lite6\_gripper/uf\_lite\_gripper.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX ArticulationAPI
* PhysX RigidBodyAPI

lite6

**USD Path:** Ufactory/lite6/lite6.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

**Techman**

TM12

**USD Path:** Techman/TM12/tm12.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 9 |
| Number of Links | 10 |
| Number of DOFs | 6 |

| Sensors | Count |
| --- | --- |
| Camera | 1 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

**ShadowRobot**

ShadowHand

Variant 1

**USD Path:** ShadowRobot/ShadowHand/shadow\_hand.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 25 |
| Number of Links | 26 |
| Number of DOFs | 24 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

Variant 2

**USD Path:** ShadowRobot/ShadowHand/shadow\_hand\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 25 |
| Number of Links | 26 |
| Number of DOFs | 24 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

**Robotiq**

Hand-E

Variant 1

**USD Path:** Robotiq/Hand-E/Robotiq\_Hand\_E\_base.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX ArticulationAPI

Variant 2

**USD Path:** Robotiq/Hand-E/Robotiq\_Hand\_E\_edit.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX ArticulationAPI

2F-85

**USD Path:** Robotiq/2F-85/Robotiq\_2F\_85\_edit.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX ArticulationAPI
* PhysX JointAPI

2F-140

Variant 1

**USD Path:** Robotiq/2F-140/2f140\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 10 |
| Number of Links | 11 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** Robotiq/2F-140/Robotiq\_2F\_140\_base.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 8 |

**Physics APIs:**

* PhysX ArticulationAPI

Variant 3

**USD Path:** Robotiq/2F-140/Robotiq\_2F\_140\_config.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 10 |
| Number of Links | 11 |
| Number of DOFs | 10 |

| Sensor/Accessory | Count |
| --- | --- |
| Contact Sensor | 1 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX SceneAPI
* PhysX ResidualReportingAPI
* PhysX MimicJointAPI

Variant 4

**USD Path:** Robotiq/2F-140/Robotiq\_2F\_140\_physics\_edit.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 8 |

| Sensor/Accessory | Count |
| --- | --- |
| Contact Sensor | 1 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX MimicJointAPI
* PhysX JointAPI

Variant 5

**USD Path:** Robotiq/2F-140/Collected\_2f140\_instanceable/2f140\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 10 |
| Number of Links | 11 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

**RobotStudio**

so101\_new\_calib

**USD Path:** RobotStudio/so101\_new\_calib/so101\_new\_calib.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

so100

**USD Path:** RobotStudio/so100/so100.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

**RethinkRobotics**

Sawyer

**USD Path:** RethinkRobotics/Sawyer/sawyer\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 12 |
| Number of Links | 13 |
| Number of DOFs | 8 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

**Kuka**

KR210\_L150

**USD Path:** Kuka/KR210\_L150/kr210\_l150.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Kinova**

Jaco2

Variant 1

**USD Path:** Kinova/Jaco2/J2N7S300/j2n7s300\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 13 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** Kinova/Jaco2/J2N6S300/j2n6s300\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 13 |
| Number of Links | 14 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Gen3

**USD Path:** Kinova/Gen3/gen3n7\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 7 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Kawasaki**

RS080N

**USD Path:** Kawasaki/RS080N/rs080n\_onrobot\_rg2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

RS025N

**USD Path:** Kawasaki/RS025N/rs025n\_onrobot\_rg2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

RS013N

**USD Path:** Kawasaki/RS013N/rs013n\_onrobot\_rg2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

RS007N

**USD Path:** Kawasaki/RS007N/rs007n\_onrobot\_rg2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

RS007L

**USD Path:** Kawasaki/RS007L/rs007l\_onrobot\_rg2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

**FrankaRobotics**

FrankaPanda

**USD Path:** FrankaRobotics/FrankaPanda/franka.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 10 |
| Number of Links | 11 |
| Number of DOFs | 9 |

**Accessories:**

* AlternateFinger
* Default
* Robotiq\_2F\_85

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX RigidBodyAPI

FrankaFR3

**USD Path:** FrankaRobotics/FrankaFR3/fr3.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 12 |
| Number of Links | 13 |
| Number of DOFs | 9 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

FrankaEmika

**USD Path:** FrankaRobotics/FrankaEmika/panda\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 10 |
| Number of Links | 11 |
| Number of DOFs | 9 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

FactoryFranka

Variant 1

**USD Path:** FrankaRobotics/FactoryFranka/factory\_franka.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 11 |
| Number of Links | 12 |
| Number of DOFs | 9 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

Variant 2

**USD Path:** FrankaRobotics/FactoryFranka/factory\_franka\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 11 |
| Number of Links | 12 |
| Number of DOFs | 9 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Flexiv**

Rizon4

**USD Path:** Flexiv/Rizon4/flexiv\_rizon4.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 7 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

**Festo**

FestoCobot

**USD Path:** Festo/FestoCobot/festo\_cobot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 7 |
| Number of Links | 8 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

**Fanuc**

CRX10IAL

**USD Path:** Fanuc/CRX10IAL/crx10ial.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 9 |
| Number of Links | 10 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Denso**

CobottaPro900

**USD Path:** Denso/CobottaPro900/cobotta\_pro\_900.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

CobottaPro1300

**USD Path:** Denso/CobottaPro1300/cobotta\_pro\_1300.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

**OpenArm**

openarm\_unimanual

**USD Path:** OpenArm/openarm\_unimanual/openarm\_unimanual.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 11 |
| Number of Links | 12 |
| Number of DOFs | 11 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI
* PhysX MimicJointAPI

openarm\_bimanual

**USD Path:** OpenArm/openarm\_bimanual/openarm\_bimanual.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 22 |
| Number of Links | 23 |
| Number of DOFs | 22 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI
* PhysX MimicJointAPI

Humanoid

**XiaoPeng**

PX5

Variant 1

**USD Path:** XiaoPeng/PX5/px5.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 16 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

Variant 2

**USD Path:** XiaoPeng/PX5/px5\_without\_housing.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 16 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

**X-Humanoid**

Tien Kung

**USD Path:** XHumanoid/Tien Kung/tienkung.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 59 |
| Number of Links | 60 |
| Number of DOFs | 54 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

**Unitree**

H1

**USD Path:** Unitree/H1/h1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 24 |
| Number of Links | 25 |
| Number of DOFs | 19 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

G1\_23dof

Variant 1

**USD Path:** Unitree/G1\_23dof/g1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Physics APIs:**

* PhysX SceneAPI
* PhysX CollisionAPI
* PhysX JointAPI
* PhysX ArticulationAPI

Variant 2

**USD Path:** Unitree/G1\_23dof/g1\_minimal.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Physics APIs:**

* PhysX SceneAPI
* PhysX CollisionAPI
* PhysX JointAPI
* PhysX ArticulationAPI

G1

**USD Path:** Unitree/G1/g1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 45 |
| Number of Links | 46 |
| Number of DOFs | 43 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX ArticulationAPI

**SanctuaryAI**

Phoenix

**USD Path:** SanctuaryAI/Phoenix/phoenix.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 77 |
| Number of Links | 78 |
| Number of DOFs | 77 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**RobotEra**

STAR1

**USD Path:** RobotEra/STAR1/star1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 55 |
| Number of Links | 56 |
| Number of DOFs | 55 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

**Ihmcrobotics**

Valkyrie

**USD Path:** Ihmcrobotics/Valkyrie/valkyrie.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 25 |
| Number of Links | 26 |
| Number of DOFs | 25 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

**FourierIntelligence**

GR-1

Variant 1

**USD Path:** FourierIntelligence/GR-1/GR1T2\_fourier\_hand\_6dof/GR1T2\_fourier\_hand\_6dof.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 54 |
| Number of Links | 55 |
| Number of DOFs | 54 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX SceneAPI
* PhysX ArticulationAPI

* This robot is in Isaac Lab

Variant 2

**USD Path:** FourierIntelligence/GR-1/GR1T1/GR1\_T1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 40 |
| Number of Links | 41 |
| Number of DOFs | 32 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Agility**

Digit

**USD Path:** Agility/Digit/digit\_v4.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 42 |
| Number of Links | 43 |
| Number of DOFs | 38 |

| Sensors | Count |
| --- | --- |
| Camera | 4 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX ArticulationAPI

* This robot is in Isaac Lab

Cassie

**USD Path:** Agility/Cassie/cassie.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 14 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

**Agibot**

A2D

**USD Path:** Agibot/A2D/A2D.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 34 |
| Number of Links | 35 |
| Number of DOFs | 34 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX ArticulationAPI

**1X**

Neo

**USD Path:** 1X/Neo/Neo.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 33 |
| Number of Links | 34 |
| Number of DOFs | 33 |

**Physics APIs:**

* PhysX MimicJointAPI
* PhysX JointAPI
* PhysX ArticulationAPI

**BoosterRobotics**

BoosterT1

**USD Path:** BoosterRobotics/BoosterT1/T1\_locomotion.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 23 |
| Number of Links | 24 |
| Number of DOFs | 23 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

Quadruped

**Unitree**

laikago

**USD Path:** Unitree/laikago/laikago.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 12 |
| Number of Links | 13 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

aliengo

**USD Path:** Unitree/aliengo/aliengo.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 12 |
| Number of Links | 13 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

Go2

**USD Path:** Unitree/Go2/go2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 38 |
| Number of Links | 39 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Go1

Variant 1

**USD Path:** Unitree/Go1/go1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** Unitree/Go1/go1\_sensor.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

| Sensor/Accessory | Count |
| --- | --- |
| Contact Sensor | 4 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

B2

**USD Path:** Unitree/B2/b2.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 30 |
| Number of Links | 31 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

A1

**USD Path:** Unitree/A1/a1.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

**IsaacSim**

Ant

Variant 1

**USD Path:** IsaacSim/Ant/ant.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 8 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** IsaacSim/Ant/ant\_colored.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX CollisionAPI
* PhysX SceneAPI

Variant 3

**USD Path:** IsaacSim/Ant/ant\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 8 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

**BostonDynamics**

spot

**USD Path:** BostonDynamics/spot/spot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

**ANYbotics**

anymal\_d

**USD Path:** ANYbotics/anymal\_d/anymal\_d.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX ArticulationAPI

anymal\_c

**USD Path:** ANYbotics/anymal\_c/anymal\_c.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 17 |
| Number of Links | 18 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

anymal\_b

**USD Path:** ANYbotics/anymal\_b/anymal\_b.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX JointAPI
* PhysX ArticulationAPI

**DeepRobotics**

X30

**USD Path:** DeepRobotics/X30/X30.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

M20

**USD Path:** DeepRobotics/M20/M20.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 16 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

Lite3

**USD Path:** DeepRobotics/Lite3/Lite3.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 16 |
| Number of Links | 17 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

Holonomic

**NVIDIA**

Kaya

Variant 1

**USD Path:** NVIDIA/Kaya/kaya.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 33 |
| Number of Links | 34 |
| Number of DOFs | 33 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI

Variant 2

**USD Path:** NVIDIA/Kaya/kaya\_ogn\_gamepad.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Physics APIs:**

* PhysX SceneAPI
* PhysX CollisionAPI
* PhysX JointAPI
* PhysX ArticulationAPI

**Fraunhofer**

O3dyn

Variant 1

**USD Path:** Fraunhofer/O3dyn/o3dyn.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 76 |
| Number of Links | 77 |
| Number of DOFs | 64 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

Variant 2

**USD Path:** Fraunhofer/O3dyn/o3dyn\_controller.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX CollisionAPI
* PhysX JointAPI
* PhysX ArticulationAPI
* PhysX SceneAPI

Variant 3

**USD Path:** Fraunhofer/O3dyn/o3dyn\_trimmed.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 52 |
| Number of Links | 53 |
| Number of DOFs | 40 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI
* PhysX CollisionAPI
* PhysX SceneAPI

Aerial

**NASA**

Ingenuity

**USD Path:** NASA/Ingenuity/ingenuity.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 4 |
| Number of Links | 5 |
| Number of DOFs | 4 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX CollisionAPI

**IsaacSim**

Quadcopter

**USD Path:** IsaacSim/Quadcopter/quadcopter.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 8 |
| Number of Links | 9 |
| Number of DOFs | 8 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

**Bitcraze**

Crazyflie

**USD Path:** Bitcraze/Crazyflie/cf2x.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 4 |
| Number of Links | 5 |
| Number of DOFs | 4 |

**Physics APIs:**

* PhysX ArticulationAPI

* This robot is in Isaac Lab

Isaac Sim Simple

**IsaacSim**

Vehicle

**USD Path:** IsaacSim/Vehicle/basic\_vehicle\_m.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | N/A |
| Number of Links | N/A |
| Number of DOFs | N/A |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX CollisionAPI
* PhysX SceneAPI

SimpleArticulation

Variant 1

**USD Path:** IsaacSim/SimpleArticulation/articulation\_3\_joints.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 3 |
| Number of Links | 4 |
| Number of DOFs | 3 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX CollisionAPI

Variant 2

**USD Path:** IsaacSim/SimpleArticulation/revolute\_articulation.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 1 |
| Number of Links | 2 |
| Number of DOFs | 1 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX CollisionAPI

Variant 3

**USD Path:** IsaacSim/SimpleArticulation/simple\_articulation.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX CollisionAPI

Humanoid28

**USD Path:** IsaacSim/Humanoid28/humanoid\_28.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 14 |
| Number of Links | 15 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Humanoid

Variant 1

**USD Path:** IsaacSim/Humanoid/humanoid.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Variant 2

**USD Path:** IsaacSim/Humanoid/humanoid\_instanceable.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 15 |
| Number of Links | 16 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

DifferentialBase

**USD Path:** IsaacSim/DifferentialBase/differential\_base.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX SceneAPI
* PhysX JointAPI

Cartpole

**USD Path:** IsaacSim/Cartpole/cartpole.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 2 |
| Number of Links | 3 |
| Number of DOFs | 2 |

**Physics APIs:**

* PhysX ArticulationAPI
* PhysX JointAPI

CartDoublePendulum

**USD Path:** IsaacSim/CartDoublePendulum/cart\_double\_pendulum.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 3 |
| Number of Links | 4 |
| Number of DOFs | 3 |

**Physics APIs:**

* PhysX JointAPI
* PhysX ArticulationAPI

BalanceBot

**USD Path:** IsaacSim/BalanceBot/balance\_bot.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 6 |
| Number of Links | 7 |
| Number of DOFs | 6 |

**Physics APIs:**

* PhysX RigidBodyAPI
* PhysX ArticulationAPI
* PhysX JointAPI

Mobile Manipulator

**Clearpath**

RidgebackUr

**USD Path:** Clearpath/RidgebackUr/ridgeback\_ur5.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 9 |
| Number of Links | 10 |
| Number of DOFs | 9 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

RidgebackFranka

**USD Path:** Clearpath/RidgebackFranka/ridgeback\_franka.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 18 |
| Number of Links | 19 |
| Number of DOFs | 12 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

* This robot is in Isaac Lab

**BostonDynamics**

spot

**USD Path:** BostonDynamics/spot/spot\_with\_arm.usd

Properties

|  |  |
| --- | --- |
| Number of Joints | 19 |
| Number of Links | 20 |
| Number of DOFs | 19 |

**Physics APIs:**

* PhysX SceneAPI
* PhysX ArticulationAPI
* PhysX JointAPI

---

# Camera and Depth Sensors

Isaac Sim supports camera and depth sensors, with digital twins found in the Content Browser
:   under `Isaac Sim/Sensors`, organized into subfolders by manufacturer.

## Cameras

For more information about camera modeling in Isaac Sim, see [here](Sensors.md).

### Leopard Imaging

#### Hawk Stereo Camera

The Hawk Stereo Camera ([LI-AR0234CS-STEREO-GMSL2-30](https://leopardimaging.com/product/platform-partners/qualcomm/iot-robotics-qualcomm/li-ar0234cs-stereo-gmsl2-qualcomm/li-ar0234cs-stereo-gmsl2-30/)) from Leopard Imaging consists of two OnSemi AR0234CS RGB image sensors and a 6-axis IMU, both are simulated in the NVIDIA Isaac Sim.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>LeopardImaging>Hawk*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>LeopardImaging>Hawk>hawk\_v1.1\_nominal.usd*

Features and Specification

Camera Features

| name | camera\_left | camera\_right |
| --- | --- | --- |
| focalLength | 2.8734347820281982 | 2.8779797554016113 |
| focusDistance | 0.6000000238418579 | 0.6000000238418579 |
| fStop | 240.0 | 240.0 |
| projection | perspective | perspective |
| stereoRole | left | right |
| horizontalAperture | 5.760000228881836 | 5.760000228881836 |
| verticalAperture | 3.5999999046325684 | 3.5999999046325684 |
| clippingRange | (0.076, 100000) | (0.076, 100000) |
| cameraProjectionType | fisheyePolynomial | fisheyePolynomial |
| nominalWidth | 1920.0 | 1920.0 |
| nominalHeight | 1200.0 | 1200.0 |
| opticalCenterX | 957.85107421875 | 954.709228515625 |
| opticalCenterY | 589.5376586914062 | 588.3735961914062 |
| maxFOV | 150.0 | 150.0 |
| polyK0 | 5.0055230531143025e-05 | 8.962746505858377e-05 |
| polyK1 | 0.0010426010703667998 | 0.001039923052303493 |
| polyK2 | 9.85131620723223e-09 | 1.502240820627776e-08 |
| polyK3 | 1.6426542417957712e-11 | 5.982795422271314e-12 |
| polyK4 | 2.9886398802796144e-14 | 3.6818906078281075e-14 |
| polyK5 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | [0.147811, -0.032313, -0.000194, -0.000035, 0.008823, 0.517913, -0.06708, 0.01695] | [6.815791, 5.172144, -0.000246, -0.000128, 0.353267, 7.180808, 7.640372, 1.596375] |
| physicalDistortionModel | rational\_polynomial | rational\_polynomial |

**Other Features**

* Waterproof: IP65
* Dimensions: 180 mm (length) by 44.33 mm (depth) by 25.0 mm (height)
* Operating Temperature: -20C to 50C

IMU to Hawk sensor (left camera) transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 90 | 0.0 |
| Translation (meters) | 0.0 | -0.0947 | 0.0061 |

Note

For the datasheet and full list of specifications, visit the [Hawk stereo camera product page](https://leopardimaging.com/leopard-imaging-hawk-stereo-camera/) and [purchase here](https://leopardimaging.com/product/platform-partners/qualcomm/iot-robotics-qualcomm/li-ar0234cs-stereo-gmsl2-qualcomm/li-ar0234cs-stereo-gmsl2-30/).

#### Owl Fisheye camera

The Owl camera ([LI-AR0234CS-GMSL2-OWL](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/maxim-gmsl-2-cameras/li-ar0234cs-gmsl2-owl/li-ar0234cs-gmsl2-owl/)) from Leopard Imaging consists of a 2.3MP OnSemi AR0234CS RGB image sensor, capable of producing crisp images in low-light and bright scenes.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>LeopardImaging>Owl*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>LeopardImaging>Owl>owl.usd*

Features and Specification

Camera Features

| name | camera |
| --- | --- |
| focalLength | 1.3646053075790405 |
| focusDistance | 0.6000000238418579 |
| fStop | 180.0 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 3.5999999046325684 |
| clippingRange | (0.076, 100000) |
| cameraProjectionType | fisheyePolynomial |
| nominalWidth | 1920.0 |
| nominalHeight | 1200.0 |
| opticalCenterX | 943.99462890625 |
| opticalCenterY | 602.3110961914062 |
| maxFOV | 235.0 |
| polyK0 | 0.0002725422091316432 |
| polyK1 | 0.0021866457536816597 |
| polyK2 | 1.2340817079348199e-07 |
| polyK3 | -1.079574096785052e-09 |
| polyK4 | 5.997452426180494e-13 |
| polyK5 | 0.0 |
| p0 | -0.00037 |
| p1 | -0.00074 |
| s0 | -0.00058 |
| s1 | -0.00022 |
| s2 | 0.00019 |
| s3 | -0.0002 |
| physicalDistortionCoefficients | [0.057225, 0.012671, -0.002978, -0.000472] |
| physicalDistortionModel | kannalaBrandt |

**Other Features**

* Dimensions: 50 mm (length) by 37.63 mm (depth) by 25.0 mm (height)
* Operating Temperature: -20C to 50C

Note

For full list of specifications, visit the [product page](https://leopardimaging.com/leopard-imaging-hawk-stereo-camera/) ,and the owl cameras can be [purchased here](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/maxim-gmsl-2-cameras/li-ar0234cs-gmsl2-owl/li-ar0234cs-gmsl2-owl/).

### Sensing

#### SG2-AR0233C-5200-G2A-H100F1A Camera (Certified by Sensing)

SG2-AR0233C-5200-G2A-H100F1A from the [SG2-AR0233C-5200-G2A-Hxxx Series](https://www.sensing-world.com/en/pd.jsp?id=18) is a megapixel high performance automotive camera module, primarily used for ADAS, HDR imaging functionalities.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG2-AR0233C-5200-G2A-H100F1A*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG2>H100F1A>SG2-AR0233C-5200-G2A-H100F1A.usd*

Features and Specification

Camera Features

| name | SG2\_AR0233C\_5200\_G2A\_H100F1A\_01 |
| --- | --- |
| focalLength | 3.549999952316284 |
| focusDistance | 270.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 3.240000009536743 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeOrthographic |
| nominalWidth | 1920.0 |
| nominalHeight | 1080.0 |
| opticalCenterX | 998.0842895507812 |
| opticalCenterY | 520.5062866210938 |
| maxFOV | 100.0 |
| polyK0 | 0.9293811321258545 |
| polyK1 | 0.15743136405944824 |
| polyK2 | 0.008131147362291813 |
| polyK3 | 1.358112096786499 |
| polyK4 | 0.4388065040111542 |
| polyK5 | 0.035474397242069244 |
| p0 | -1.8616799934534356e-05 |
| p1 | -0.000114203299744986 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG2-AR0233C-5200-G2A-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=18)

#### SG2-OX03CC-5200-GMSL2-H60YA Series Camera (Certified by Sensing)

SG2-OX03CC-5200-GMSL2-H60YA from the [SG2-OX03CC-5200-GMSL2-Hxxx Series](https://www.sensing-world.com/en/pd.jsp?id=106&id=106) is a megapixel high performance automotive camera module, primarily used for ADAS, HDR imaging functionalities.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG2-OX03CC-5200-GMSL2-H60YA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG2>H60YA>Camera\_SG2\_OX03CC\_5200\_GMSL2\_H60YA.usd*

Features and Specification

Camera Features

| name | Camera\_SG2\_OX03CC\_5200\_GMSL2\_H60YA |
| --- | --- |
| focalLength | 5.75 |
| focusDistance | 700.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 3.240000009536743 |
| clippingRange | (0.1, 1000000) |
| cameraProjectionType | fisheyeOrthographic |
| nominalWidth | 1920.0 |
| nominalHeight | 1080.0 |
| opticalCenterX | 959.595947265625 |
| opticalCenterY | 647.6140747070312 |
| maxFOV | 60.0 |
| polyK0 | 0.7182272672653198 |
| polyK1 | 60.113136291503906 |
| polyK2 | 2.598527431488037 |
| polyK3 | 1.1977670192718506 |
| polyK4 | 60.394771575927734 |
| polyK5 | 31.610383987426758 |
| p0 | 0.0004008802934549749 |
| p1 | -0.0013344850158318877 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG2-OX03CC-5200-GMSL2F-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=106&id=106)

#### SG3-ISX031C-GMSL2F-H190XA (Certified by Sensing)

[SG3-ISX031C-GMSL2F-H190XA](https://www.sensing-world.com/en/pd.jsp?id=23#_jcp=2) is a 3 megapixels automotive camera for automotive surround view.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG3-ISX031C-GMSL2F-H190XA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG3>H190XA>SG3S-ISX031C-GMSL2F-H190XA.usd*

Features and Specification

Camera Features

| name | SG3S\_ISX031C\_GMSL2F\_H190XA\_01 |
| --- | --- |
| focalLength | 1.5099999904632568 |
| focusDistance | 39.0 |
| fStop | 2.0 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 4.607999801635742 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeEquidistant |
| nominalWidth | 1920.0 |
| nominalHeight | 1536.0 |
| opticalCenterX | 960.6082153320312 |
| opticalCenterY | 768.0 |
| maxFOV | 190.0 |
| polyK0 | 0.13215887546539307 |
| polyK1 | -0.031036589294672012 |
| polyK2 | -0.004391151946038008 |
| polyK3 | 0.0018116832943633199 |
| polyK4 | 0.0 |
| polyK5 | 0.0 |
| p0 | 0.0 |
| p1 | 0.0 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG3-ISX031C-GMSL2F-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=23#_jcp=2)

#### SG5-IMX490C-5300-GMSL2-H110SA (Certified by Sensing)

[SG5-IMX490C-5300-GMSL2-H110SA](https://www.sensing-world.com/en/pd.jsp?id=24#_jcp=2) is a 5 megapixels automotive camera for automotive surround view, ADAS and viewing fusion.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG5-IMX490C-5300-GMSL2-H110SA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG5>H100SA>SG5-IMX490C-5300-GMSL2-H110SA.usd*

Features and Specification

Camera Features

| name | Camera\_SG5\_IMX490C\_5300\_GMSL2\_H110SA |
| --- | --- |
| focalLength | 4.260000228881836 |
| focusDistance | 220.0 |
| fStop | 2.799999952316284 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.640000343322754 |
| verticalAperture | 5.579999923706055 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeOrthographic |
| nominalWidth | 2880.0 |
| nominalHeight | 1860.0 |
| opticalCenterX | 1442.3316650390625 |
| opticalCenterY | 926.6644287109375 |
| maxFOV | 110.0 |
| polyK0 | 0.6106576919555664 |
| polyK1 | -0.11334560066461563 |
| polyK2 | -0.014692608267068863 |
| polyK3 | 0.9237731099128723 |
| polyK4 | -0.011052233166992664 |
| polyK5 | -0.051484767347574234 |
| p0 | 2.2259799152379856e-05 |
| p1 | -7.929380080895498e-05 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C
* Multi camera synchronization support

Note

For the datasheet and full list of specifications, visit the [SG5-IMX490C-5300-GMSL2-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=24#_jcp=2)

#### SG8S-AR0820C-5300-G2A-H30YA Camera (Certified by Sensing)

[SG8S-AR0820C-5300-G2A-H30YA](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2) is a 4k high performance automotive grade camera that supports advanced on sensor HDR and multi-camera synchronization.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG8S-AR0820C-5300-G2A-H30YA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG8>H30YA>SG8S-AR0820C-5300-G2A-H30YA.usd*

Features and Specification

Camera Features

| name | SG8S\_AR0820C\_5300\_G2A\_H30YA\_01 |
| --- | --- |
| focalLength | 15.300000190734863 |
| focusDistance | 7070.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.064000129699707 |
| verticalAperture | 4.535999774932861 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | pinhole |
| nominalWidth | 3840.0 |
| nominalHeight | 2160.0 |
| opticalCenterX | 1864.240478515625 |
| opticalCenterY | 986.3945922851562 |
| maxFOV | 30.0 |
| polyK0 | -0.6564998626708984 |
| polyK1 | -4.156541347503662 |
| polyK2 | 245.6761932373047 |
| polyK3 | -0.43839189410209656 |
| polyK4 | -4.5701212882995605 |
| polyK5 | 251.74969482421875 |
| p0 | -0.000658363220281899 |
| p1 | 8.901000114747148e-07 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Dimensions: 40 mm (length) by 23 mm (depth) by 40 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG8-AR820C-5300-G2A-Hxxx series cameras product page.](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2)

#### SG8S-AR0820C-5300-G2A-H60SA Camera (Certified by Sensing)

[SG8S-AR0820C-5300-G2A-H60SA](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2) is a 4k high performance automotive grade camera that supports advanced on sensor HDR and multi-camera synchronization.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG8S-AR0820C-5300-G2A-H60SA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG8>H60SA>SG8S-AR0820C-5300-G2A-H60SA.usd*

Features and Specification

Camera Features

| name | SG8S\_AR0820C\_5300\_G2A\_H60SA\_01 |
| --- | --- |
| focalLength | 7.869999885559082 |
| focusDistance | 1670.0 |
| fStop | 1.7999999523162842 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.064000129699707 |
| verticalAperture | 4.535999774932861 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeEquidistant |
| nominalWidth | 3840.0 |
| nominalHeight | 2160.0 |
| opticalCenterX | 1919.1090087890625 |
| opticalCenterY | 1087.7274169921875 |
| maxFOV | 60.0 |
| polyK0 | 0.8600332140922546 |
| polyK1 | -0.30780455470085144 |
| polyK2 | -0.05103735625743866 |
| polyK3 | 1.5231009721755981 |
| polyK4 | 0.0005489090108312666 |
| polyK5 | -0.25151902437210083 |
| p0 | 6.143400241853669e-05 |
| p1 | -4.332419848651625e-05 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Dimensions: 40 mm (length) by 23 mm (depth) by 40 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG8-AR820C-5300-G2A-Hxxx series cameras product page.](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2)

#### SG8S-AR0820C-5300-G2A-H120YA Camera (Certified by Sensing)

[SG8S-AR0820C-5300-G2A-H120YA](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2) is a 4k high performance automotive grade camera that supports advanced on sensor HDR and multi-camera synchronization.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG8S-AR0820C-5300-G2A-H120YA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG8>H120YA>SG8S-AR0820C-5300-G2A-H120YA.usd*

Features and Specification

Camera Features

| name | SG8S\_AR0820C\_5300\_G2A\_H120YA\_01 |
| --- | --- |
| focalLength | 4.010000228881836 |
| focusDistance | 480.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.064000129699707 |
| verticalAperture | 4.535999774932861 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeEquidistant |
| nominalWidth | 3840.0 |
| nominalHeight | 2160.0 |
| opticalCenterX | 1919.1090087890625 |
| opticalCenterY | 1087.7274169921875 |
| maxFOV | 120.0 |
| polyK0 | 0.8600332140922546 |
| polyK1 | -0.30780455470085144 |
| polyK2 | -0.05103735625743866 |
| polyK3 | 1.5231009721755981 |
| polyK4 | 0.0005489090108312666 |
| polyK5 | -0.25151902437210083 |
| p0 | 6.143400241853669e-05 |
| p1 | -4.332419848651625e-05 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Dimensions: 40 mm (length) by 23 mm (depth) by 40 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG8-AR820C-5300-G2A-Hxxx series cameras product page.](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2)

### SICK

#### Inspector83x (Certified by SICK)

##### Inspector83x (Certified)

The [SICK Inspector83x](https://www.sick.com/inspector83x) is a 2D camera, which helps to rapidly solve vision applications such as quality assurance, defect detection, and sorting.

###### Features and Specification

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [Inspector83x product page](https://www.sick.com/inspector83x).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>SICK>Inspector83x*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>Inspector83x>SICK\_Inspector83x.usd*

## Depth Sensors

For more information about depth sensor modeling in Isaac Sim, see [here](Sensors.md).

### Realsense

#### Realsense D455 (Certified by Realsense)

##### Realsense D455 (Certified)

The [Realsense D455](https://realsenseai.com/products/real-sense-depth-camera-d455f) consists of multiple RGB and depth image sensors and a 6-axis IMU.

###### Features and Specification

**Other Features**

* Dimensions: 124 mm (length) by 25 mm (depth) by 29 mm (height)
* IMU: Bosch BM1055
* Ideal Range: 0.6m to 6m
* Minimum Depth Distance at Max resolution: 52cm
* Depth accuracy: under 2% at 4m

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [D455 product page](https://realsenseai.com/products/real-sense-depth-camera-d455f).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Realsense>Realsense D455*

Camera Attributes

Camera Features

| name | Camera\_Pseudo\_Depth | Camera\_OmniVision\_OV9782\_Color | Camera\_OmniVision\_OV9782\_Left | Camera\_OmniVision\_OV9782\_Right |
| --- | --- | --- | --- | --- |
| focalLength | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 |
| focusDistance | 0.6000000238418579 | 0.5 | 0.5 | 0.5 |
| fStop | 2.0 | 2.0 | 2.0 | 2.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 |
| verticalAperture | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 |
| clippingRange | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) |
| cameraProjectionType | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 100.5999984741211 | 98.0 | 98.0 | 98.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

IMU to RealSense transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 0.0 | 0.0 |
| Translation (meters) | 0.016 | -0.01728 | 0.0074 |

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Realsense>D455>rsd455.usd*

#### Realsense D457 (Certified by Realsense)

##### Realsense D457

The [Realsense D457](https://www.realsenseai.com/products/d457-gmsl-fakra/) is a ruggedized, IP65-rated stereo depth camera featuring a GMSL/FAKRA interface for secure, long-distance high-bandwidth connectivity. It utilizes the same optical module as the D455 and is designed for autonomous mobile robots (AMRs) and automotive infotainment.

###### Features and Specification

**Other Features**

* Dimensions: 124 mm (length) by 36 mm (depth) by 29 mm (height)
* Environment: Indoor/Outdoor (IP65 Rated)
* Vision Processor: Vision Processor D4 Board V5
* Sensors: Global Shutter (Depth and RGB)
* Depth FOV: 87° × 58° (Resolution up to 1280 × 720 at 90 fps)
* RGB FOV: 90° × 65° (Resolution up to 1280 × 800 at 60 fps)
* Minimum Depth Distance: 26-52 cm
* Connectors: GMSL/FAKRA (Maxim Integrated), USB-C
* IMU: Built-in 6-axis IMU

> ℹ️ **Note**
> For the full datasheet, visit the [D457 Datasheet](https://www.realsenseai.com/wp-content/uploads/2023/01/Intel-RealSense-D457-Datasheet-January2023.pdf).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Realsense>Realsense D457*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Realsense>D457>rsd457.usd*

Camera Attributes

Camera Features

| name | Camera\_Pseudo\_Depth | Camera\_OmniVision\_OV9782\_Color | Camera\_OmniVision\_OV9782\_Left | Camera\_OmniVision\_OV9782\_Right |
| --- | --- | --- | --- | --- |
| focalLength | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 |
| focusDistance | 0.6000000238418579 | 0.5 | 0.5 | 0.5 |
| fStop | 2.0 | 2.0 | 2.0 | 2.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 |
| verticalAperture | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 |
| clippingRange | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) |
| cameraProjectionType | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 100.5999984741211 | 98.0 | 98.0 | 98.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

IMU to RealSense transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 0.0 | 0.0 |
| Translation (meters) | 0.016 | -0.01728 | 0.0074 |

#### Realsense D555 (Certified by Realsense)

##### Realsense D555

The [Realsense D555](https://www.realsenseai.com/products/d555-poe/) is a ruggedized, IP65-rated stereo depth camera designed for industrial and outdoor environments. It features the new RealSense Vision SoC V5, on-chip Power over Ethernet (PoE), and global shutter sensors for both RGB and Depth.

###### Features and Specification

**Other Features**

* Dimensions: 167 mm (length) by 42 mm (depth) by 48 mm (height)
* Environment: Indoor/Outdoor (IP65 Rated)
* Vision Processor: RealSense Vision SoC V5
* Sensors: Global Shutter (Depth and RGB)
* Depth FOV: 87° × 58° (Resolution up to 1280 × 720 at 90 fps)
* RGB FOV: 90° × 65° (Resolution up to 1280 × 800 at 60 fps)
* Minimum Depth Distance: 26-52 cm
* Connectors: PoE (RJ45), USB-C (Power/Data), GMSL/FAKRA, External HW Sync via USB
* Native ROS Support: Powered by SafeDDS (ISO 26262-certified) and interoperable with Fast DDS, enabling plug-and-play ROS 2 streaming over Ethernet without additional installation.

> ℹ️ **Note**
> For the full datasheet, visit the [D555 Datasheet](https://www.realsenseai.com/wp-content/uploads/2025/08/D555-Datasheet-v1.1.pdf).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Realsense>Realsense D555*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Realsense>D555>rsd555.usd*

Camera Attributes

Camera Features

| name | Camera\_Pseudo\_Depth | Camera\_OmniVision\_OV9782\_Color | Camera\_OmniVision\_OV9782\_Left | Camera\_OmniVision\_OV9782\_Right |
| --- | --- | --- | --- | --- |
| focalLength | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 |
| focusDistance | 0.6000000238418579 | 0.5 | 0.5 | 0.5 |
| fStop | 2.0 | 2.0 | 2.0 | 2.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 |
| verticalAperture | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 |
| clippingRange | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) |
| cameraProjectionType | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 100.5999984741211 | 98.0 | 98.0 | 98.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

IMU to RealSense transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 0.0 | 0.0 |
| Translation (meters) | 0.016 | -0.01728 | 0.0074 |

### Orbbec

#### Orbbec Gemini 2 (Certified by Orbbec)

The [Orbbec Gemini 2](https://www.orbbec.com/products/stereo-vision-camera/gemini-2/) is a depth camera based on Active Stereo IR technology.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec Gemini 2*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>Gemini 2>orbbec\_gemini2\_V1.0.usd*

Features and Specification

Camera Features

| name | Stream\_rgb | Stream\_depth | Stream\_ir\_left | Stream\_ir\_right |
| --- | --- | --- | --- | --- |
| focalLength | 2.9700000286102295 | 1.809999942779541 | 1.809999942779541 | 1.809999942779541 |
| focusDistance | 80.0 | 45.0 | 45.0 | 45.0 |
| fStop | 0.0 | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 5.539000034332275 | 3.619999885559082 | 3.880000114440918 | 3.880000114440918 |
| verticalAperture | 3.0920000076293945 | 2.440000057220459 | 2.440000057220459 | 2.440000057220459 |
| clippingRange | (0.005, 1000) | (0.005, 1000) | (0.005, 1000) | (0.005, 1000) |
| cameraProjectionType | pinhole | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 90 mm (length) by 25 mm (depth) by 30 mm (height)
* IMU supported with multi camera synchronization
* Ideal Range: 0.15m to 10m
* Depth accuracy: under 2% at 2m

Note

For the datasheet and full list of specifications, visit the [Gemini 2 product page.](https://www.orbbec.com/products/stereo-vision-camera/gemini-2)

#### Orbbec Femto Mega (Certified by Orbbec)

The [Orbbec Femto Mega](https://www.orbbec.com/products/tof-camera/femto-mega/) is a programmable multi-mode Depth and RGB camera.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec FemtoMega*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>FemtoMega>orbbec\_femtomega\_v1.0.usd*

Features and Specification

Camera Features

| name | camera\_rgb | camera\_tof\_nfov | camera\_tof\_wfov |
| --- | --- | --- | --- |
| focalLength | 3.25 | 1.690000057220459 | 1.690000057220459 |
| focusDistance | 150.0 | 44.0 | 44.0 |
| fStop | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective |
| stereoRole | mono | mono | mono |
| horizontalAperture | 5.449999809265137 | 2.5899999141693115 | 5.849999904632568 |
| verticalAperture | 3.0999999046325684 | 2.1500000953674316 | 5.849999904632568 |
| clippingRange | (0.01, 1000000) | (0.01, 1000) | (0.01, 1000) |
| cameraProjectionType | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 115mm (length) by 145 mm (depth) by 40mm (height)
* IMU supported
* Ideal Range: 0.25m to 5.46m
* Depth accuracy: under 11mm + 0.1% distance

Note

For the datasheet and full list of specifications, visit the [Femto Mega product page.](https://www.orbbec.com/products/tof-camera/femto-mega/)

#### Orbbec Gemini 335 (Certified by Orbbec)

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec Gemini 335*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>Gemini335>orbbec\_gemini\_335.usd*

Features and Specification

Camera Features

| name | Stream\_rgb | Stream\_ir\_left | Stream\_ir\_right |
| --- | --- | --- | --- |
| focalLength | 2.9700000286102295 | 1.809999942779541 | 1.809999942779541 |
| focusDistance | 80.0 | 45.0 | 45.0 |
| fStop | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective |
| stereoRole | mono | right | left |
| horizontalAperture | 5.539000034332275 | 3.880000114440918 | 3.880000114440918 |
| verticalAperture | 3.0920000076293945 | 2.440000057220459 | 2.440000057220459 |
| clippingRange | (0.005, 1000) | (0.005, 1000) | (0.005, 1000) |
| cameraProjectionType | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable |

#### Orbbec Gemini 335L (Certified by Orbbec)

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec Gemini 335L*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>Gemini335L>orbbec\_gemini\_335L.usd*

Features and Specification

Camera Features

| name | Camera\_ir\_left | Camera\_ir\_right | Camera\_rgb |
| --- | --- | --- | --- |
| focalLength | 1.809999942779541 | 1.809999942779541 | 1.809999942779541 |
| focusDistance | 45.0 | 45.0 | 0.44999998807907104 |
| fStop | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective |
| stereoRole | mono | mono | mono |
| horizontalAperture | 3.8399999141693115 | 3.8399999141693115 | 3.8399999141693115 |
| verticalAperture | 2.4000000953674316 | 2.4000000953674316 | 2.4000000953674316 |
| clippingRange | (0.005, 100000) | (0.005, 100000) | (0.005, 100000) |
| cameraProjectionType | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable |

### Stereolabs

#### ZED X (Certified by Stereolabs)

The [ZED X Stereo Camera](https://www.stereolabs.com/zed-x/) from Stereolabs consists of two 1200p 60fps RGB image sensors and a 6-axis IMU, all simulated in the NVIDIA Isaac Sim.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Stereolabs>ZED\_X*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Stereolabs>ZED\_X>ZED\_X.usd*

Features and Specification

Camera Features

| name | CameraLeft | CameraRight |
| --- | --- | --- |
| focalLength | 2.2079999446868896 | 2.2079999446868896 |
| focusDistance | 28.0 | 28.0 |
| fStop | 0.0 | 0.0 |
| projection | perspective | perspective |
| stereoRole | left | right |
| horizontalAperture | 5.760000228881836 | 5.760000228881836 |
| verticalAperture | 3.240000009536743 | 3.240000009536743 |
| clippingRange | (0.01, 100000) | (0.01, 100000) |
| cameraProjectionType | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 163.4 mm (length) by 31.8 mm (depth) by 36.7 mm (height)
* Operating Temperature: -20C to 55C

IMU to ZED X transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | -90.0 | 0.0 | 0.0 |
| Translation (meters) | 0.06 | -0.0 | 0.00185 |

Note

For the datasheet and full list of specifications, visit the [ZED X datasheet](https://www.stereolabs.com/datasheets), for usage in Isaac Sim, see [Stereolabs Documentation](https://www.stereolabs.com/docs/isaac-sim).

#### ZED X Mini (Certified by Stereolabs)

The [ZED X Mini Stereo Camera](https://www.stereolabs.com/zed-x/) from Stereolabs consists of two 1200p 60fps RGB image sensors and a 6-axis IMU, all simulated in the NVIDIA Isaac Sim.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Stereolabs>ZED\_X\_mini>ZED\_X\_Mini.usd*

Features and Specification

Camera Features

| name | CameraRight | CameraLeft |
| --- | --- | --- |
| focalLength | 2.2079999446868896 | 2.2079999446868896 |
| focusDistance | 28.0 | 28.0 |
| fStop | 0.0 | 0.0 |
| projection | perspective | perspective |
| stereoRole | right | left |
| horizontalAperture | 5.760000228881836 | 5.760000228881836 |
| verticalAperture | 3.240000009536743 | 3.240000009536743 |
| clippingRange | (0.01 100000) | (0.01 100000) |
| cameraProjectionType | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 93.6 mm (length) by 31.8 mm (depth) by 36.7 mm (height)
* Operating Temperature: -20C to 55C

IMU to ZED X Mini transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | -90.0 | 0.0 | 0.0 |
| Translation (meters) | 0.06 | -0.0 | 0.00185 |

Note

For the datasheet and full list of specifications, visit the [ZED X Mini datasheet](https://www.stereolabs.com/datasheets), for usage in Isaac Sim, see [Stereolabs Documentation](https://www.stereolabs.com/docs/isaac-sim).

---

# Non-Visual Sensors

Isaac Sim models many types of non-visual sensors models, with digital twins found in the Content Browser under `Isaac Sim/Sensors`, organized into subfolders by manufacturer.

Some non-visual sensor types do not have digital twins. For more information about these sensors,
including how to create them from the GUI, follow the links below:

* [Contact sensors](Sensors.md)
* [IMU sensors](Sensors.md)
* [Lightbeam sensors](Sensors.md)
* [PhysX Lidars](Sensors.md)
* [RTX Radars](Sensors.md)

## RTX Lidars

RTX Lidars marked as “certified” have Lidar configurations verified by the sensor manufacturer and tested before release.

Some Lidar models feature multiple configurations or profiles, which are implemented as [USD Variants](https://docs.omniverse.nvidia.com/workflows/latest/variant-workflows.html).
In those cases, the available variants and their characteristics will also be provided as tables in the appropriate section below.

### NVIDIA

There are several example Lidar configuration files that ship with Isaac Sim. Note none of these Lidars have a mesh,
so only a prim will appear in the Stage window when they are created. To create them via the UI, select the appropriate
option below from the menu: *Create>Sensors>RTX Lidar>NVIDIA*.

* **Example Rotary 2D** - a 10Hz rotary Lidar configuration with emitters in a single plane.
* **Example Rotary** - a 10Hz rotary Lidar configuration with emitters in a single plane.
* **Example Rotary Beams** - a 10Hz rotary Lidar configuration using a Gaussian beam ray type.
* **Example Solid State** - a solid state Lidar configuration.
* **Example Solid State Beams** - a solid state Lidar configuration using a Gaussian beam ray type.
* **Simple Example Solid State** - a simple 12-emitter solid state Lidar configuration, used to debug solid state Lidar issues.

### HESAI

#### XT32 SD10

[HESAI XT32 SD10](https://www.hesaitech.com/product/xt32/) is a high precision, 32 Channels 360 degrees spinning mid range Lidar.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>HESAI>XT32 SD10*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>HESAI>XT32\_SD10>HESAI\_XT32\_SD10.usd*

Features and Specification

XT32 SD10 Features

| name | XT-32 10hz |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20000 |
| numberOfEmitters | 32 |
| nearRangeM | 0.05 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.004 |
| rangeAccuracyM | 0.02 |
| minDistBetweenEchos | 0.05 |
| minReflectance | 0.1 |
| minReflectanceRange | 80.0 |
| wavelengthNm | 905.0 |
| pulseTimeNs | 10 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.015 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.0 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 100 mm (Top Diameter) by 103 mm (Bottom Diameter) by 76.0 mm (Height)

Note

For the datasheet and full list of specifications, visit the [XT32 SD10 product page.](https://www.hesaitech.com/product/xt32/)

### Ouster

#### OS0

[Ouster OS0](https://ouster.com/products/hardware/os0-lidar-sensor) is a high precision Lidar for autonomous vehicles, heavy machinery, robot, and mapping solutions. Isaac Sim has several pre-configured frequencies and resolutions that can be added to the stage easily.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>OS0*, then select the desired sensor configuration.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>OS0>OS0.usd*

Features and Specification

OS0 Rev6 Features

10 Hz

512 Resolution

Variant: OS0\_REV6\_128ch10hz512res

| name | OS0 REV6 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV6\_128ch10hz1024res

| name | OS0 REV6 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS0\_REV6\_128ch10hz2048res

| name | OS0 REV6 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS0\_REV6\_128ch20hz512res

| name | OS0 REV6 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV6\_128ch20hz1024res

| name | OS0 REV6 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

OS0 Rev7 Features

10 Hz

512 Resolution

Variant: OS0\_REV7\_128ch10hz512res

| name | OS0 REV7 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV7\_128ch10hz1024res

| name | OS0 REV7 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS0\_REV7\_128ch10hz2048res

| name | OS0 REV7 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS0\_REV7\_128ch20hz512res

| name | OS0 REV7 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV7\_128ch20hz1024res

| name | OS0 REV7 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

**Other Features**

* Rotation Rate: 10 or 20 hz (configurable)
* Dimensions: 87 mm (Diameter) by 58.35 mm (Height). With thermal cap, height is 74.2 mm.
* IMU supported: [InvenSense IAM-20680HT](https://invensense.tdk.com/download-pdf/iam-20680ht-datasheet/)

Note

For the datasheet and full list of specifications, visit the [OS0 product page.](https://ouster.com/products/hardware/os0-lidar-sensor)

#### OS1

[Ouster OS1](https://ouster.com/products/hardware/os1-lidar-sensor) is a high precision Lidar for autonomous vehicles, heavy machinery, robot, and mapping solutions.
Isaac Sim has several pre-configured frequencies and resolutions that can be easily added to the stage.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>OS1*.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>OS1>OS1.usd*

Features and Specification

OS1 Rev6 Features

32 Channels 10 Hz

512 Resolution

Variant: OS1\_REV6\_32ch10hz512res

| name | OS1 REV6 32 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_32ch10hz1024res

| name | OS1 REV6 32 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS1\_REV6\_32ch10hz2048res

| name | OS1 REV6 32 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

32 Channels 20 Hz

512 Resolution

Variant: OS1\_REV6\_32ch20hz512res

| name | OS1 REV6 32 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_32ch20hz1024res

| name | OS1 REV6 32 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

128 Channels 10 Hz

512 Resolution

Variant: OS1\_REV6\_128ch10hz512res

| name | OS1 REV6 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_128ch10hz1024res

| name | OS1 REV6 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS1\_REV6\_128ch10hz2048res

| name | OS1 REV6 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

128 Channels 20 Hz

512 Resolution

Variant: OS1\_REV6\_128ch20hz512res

| name | OS1 REV6 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_128ch20hz1024res

| name | OS1 REV6 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

OS1 Rev7 Features

10 Hz

512 Resolution

Variant: OS1\_REV7\_128ch10hz512res

| name | OS1 REV7 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV7\_128ch10hz1024res

| name | OS1 REV7 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS1\_REV7\_128ch10hz2048res

| name | OS1 REV7 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS1\_REV7\_128ch20hz512res

| name | OS1 REV7 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV7\_128ch20hz1024res

| name | OS1 REV7 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 87 mm (Diameter) by 58.35 mm (Height). With thermal cap, height is 74.2 mm.
* IMU supported: [InvenSense IAM-20680HT](https://invensense.tdk.com/download-pdf/iam-20680ht-datasheet/)

Note

For the datasheet and full list of specifications, visit the [OS1 product page.](https://ouster.com/products/hardware/os1-lidar-sensor)

#### OS2

[Ouster OS2](https://ouster.com/products/hardware/os2-lidar-sensor) is a high precision Lidar for autonomous vehicles, heavy machinery, robot, and mapping solutions.
Isaac Sim has several pre-configured frequencies and resolutions that can be easily added to the stage.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>OS2*.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>OS2>OS2.usd*

Features and Specification

OS2 Rev6 Features

10 Hz

512 Resolution

Variant: OS2\_REV6\_128ch10hz512res

| name | OS2 REV6 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV6\_128ch10hz1024res

| name | OS2 REV6 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS2\_REV6\_128ch10hz2048res

| name | OS2 REV6 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS2\_REV6\_128ch20hz512res

| name | OS2 REV6 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV6\_128ch20hz1024res

| name | OS2 REV6 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

OS2 Rev7 Features

10 Hz

512 Resolution

Variant: OS2\_REV7\_128ch10hz512res

| name | OS2 REV7 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV7\_128ch10hz1024res

| name | OS2 REV7 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS2\_REV7\_128ch10hz2048res

| name | OS2 REV7 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS2\_REV7\_128ch20hz512res

| name | OS2 REV7 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV7\_128ch20hz1024res

| name | OS2 REV7 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 87 mm (Diameter) by 58.35 mm (Height). With thermal cap, height is 74.2 mm.
* IMU supported: [InvenSense IAM-20680HT](https://invensense.tdk.com/download-pdf/iam-20680ht-datasheet/)

Note

For the datasheet and full list of specifications, visit the [OS2 product page.](https://ouster.com/products/hardware/os2-lidar-sensor)

#### VLS 128

[Ouster VLS 128](https://ouster.com/products/hardware/vls-128) is a long range, ultra high resolution 3D Lidar for autonomous vehicles.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>VLS 128*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>VLS\_128>Ouster\_VLS\_128.usd*

Features and Specification

VLS 128 Features

| name | Velodyne VLS-128 |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 18761 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 200.0 |
| rangeResolutionM | 0.004 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 120.0 |
| wavelengthNm | 903.0 |
| pulseTimeNs | 6 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 165.5 mm (Diameter) by 141.3 mm (Height)
* Operating Temperature: -20C to 60C

Note

[VLS 128 product page.](https://ouster.com/products/hardware/vls-128)

### SICK

#### LRS4581R (Certified)

The [SICK LRS4581R](https://www.sick.com/LRS4000) of the LRS4000 family is a 2D LiDAR sensor for large scanning ranges in outdoor applications or for localization tasks.

##### Features and Specification

| Profile | Scan frequency | Angular resolution |
| --- | --- | --- |
| Profile\_1 | 12.5 Hz | 0.02° |
| Profile\_2 | 12.5 Hz | 0.04° |
| Profile\_3 | 12.5 Hz | 0.06° |
| Profile\_4 | 12.5 Hz | 0.1° |
| Profile\_5 | 12.5 Hz | 0.12° |
| Profile\_6 | 25 Hz | 0.04° |
| Profile\_7 | 25 Hz | 0.08° |
| Profile\_8 | 25 Hz | 0.12° |
| Profile\_9 | 25 Hz | 0.2° |
| Profile\_10 | 25 Hz | 0.24° |
| Profile\_Extended\_1 | 12.5 Hz | 0.04° |
| Profile\_Extended\_2 | 12.5 Hz | 0.08° |
| Profile\_Extended\_3 | 12.5 Hz | 0.12° |
| Profile\_Extended\_4 | 12.5 Hz | 0.24° |
| Profile\_Extended\_5 | 25 Hz | 0.08° |
| Profile\_Extended\_6 | 25 Hz | 0.16° |
| Profile\_Extended\_7 | 25 Hz | 0.24° |
| Profile\_Extended\_8 | 25 Hz | 0.48° |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [LRS4581R product page](https://www.sick.com/LRS4000).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>LRS4581R*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>LRS4581R>SICK\_LRS4581R.usd*

#### microScan3 (Certified)

The [SICK microScan3](https://www.sick.com/microScan3) safety laser scanner stands for the protection of very different applications: from stationary to mobile, from simple to complex and delivers high-precision measurement data.

##### Features and Specification

| Profile | Protective field range | Scan frequency |
| --- | --- | --- |
| Profile\_1 | 4.0 m | 33.3 Hz |
| Profile\_2 | 4.0 m | 25.0 Hz |
| Profile\_3 | 5.5 m | 33.3 Hz |
| Profile\_4 | 5.5 m | 25.0 Hz |
| Profile\_5 | 9.0 m | 25.0 Hz |
| Profile\_6 | 9.0 m | 20.0 Hz |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [microScan3 product page](https://www.sick.com/microScan3).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>microScan3*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>microScan3>SICK\_microScan3.usd*

#### MRS1104C (Certified)

The [SICK MRS1104C](https://www.sick.com/MRS1000) of the MRS1000 family is a 3D LiDAR sensor for collision protection and assistance for all traveling objects in production facilities or reliable monitoring in traffic management and  building security.

##### Features and Specification

| Profile | Scan frequency | Angular resolution |
| --- | --- | --- |
| Profile\_1 | 12.5 Hz | 0.25° |
| Profile\_2\_Interlaced | 6.25 Hz | 0.125° |
| Profile\_3\_Interlaced | 3.125 Hz | 0.0625° |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [MRS1104C product page](https://www.sick.com/MRS1000).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>MRS1104C*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>MRS1104C>SICK\_MRS1104C.usd*

#### multiScan136 (Certified)

The SICK [multiScan136](http://www.sick.com/multiScan100) of the multiScan100 family is a 3D LiDAR sensor for mobile and stationary applications and reliably detects drop-off edges and obstacles ahead.

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [multiScan136 product page](http://www.sick.com/multiScan100).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>multiScan136*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>multiScan136>SICK\_multiScan136.usd*

#### multiScan165 (Certified)

The SICK [multiScan165](http://www.sick.com/multiScan100) of the multiScan100 family is a 3D LiDAR sensor for mobile and stationary applications and reliably detects drop-off edges and obstacles ahead.

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [multiScan165 product page](http://www.sick.com/multiScan100).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>multiScan165*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>multiScan165>SICK\_multiScan165.usd*

#### nanoScan3 (Certified)

The [SICK nanoScan3](https://www.sick.com/nanoScan3) is the smallest safety laser scanner, which is well suited for the protection and localization of mobile platforms.

##### Features and Specification

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [nanoScan3 product page](https://www.sick.com/nanoScan3).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>nanoScan3*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>nanoScan3>SICK\_nanoScan3.usd*

#### picoScan150 (Certified)

The [SICK picoScan150](http://www.sick.com/picoScan100) of the picoScan100 family is a 2D LiDAR sensor for solving demanding industrial applications such as collision avoidance or measurement and monitoring in indoor and outdoor areas.

##### Features and Specification

| Profile | Scan frequency | Angular resolution |
| --- | --- | --- |
| Profile\_1 | 15 Hz | 0.5° |
| Profile\_2 | 15 Hz | 0.33° |
| Profile\_3 | 20 Hz | 0.1° |
| Profile\_4 | 20 Hz | 0.25° |
| Profile\_5 | 25 Hz | 0.25° |
| Profile\_6 | 30 Hz | 0.1° |
| Profile\_7 | 40 Hz | 0.25° |
| Profile\_8 | 50 Hz | 0.25° |
| Profile\_9 | 15 Hz | 0.05° |
| Profile\_10 | 40 Hz | 0.125° |
| Profile\_11 | 15 Hz | 1.0° |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [picoScan150 product page](http://www.sick.com/picoScan100).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>picoScan150*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>picoScan150>SICK\_picoScan150.usd*

#### TiM781 (Certified)

The [SICK TiM781](http://www.sick.com/TiM) of the TiM family is a 2D LiDAR sensor for collision protection for mobile applications, object measurement or monitoring of objects.

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [TiM781 product page](http://www.sick.com/TiM).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>TiM781*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>tim781.usd*

### SLAMTEC

#### RPLIDAR S2E

[SLAMTEC RPLIDAR S2E](https://download-en.slamtec.com/api/download/rplidar-s2m1-RxE-datasheet/1.8?lang=en) is a low cost 360 degrees 2D laser scanner Lidar from SLAMTEC.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Slamtec>RPLIDAR S2E*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Slamtec>RPLIDAR\_S2E.usd*

Features and Specification

RPLIDAR S2E Features

|  |  |
| --- | --- |
| name | RPLIDAR S2E |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 32000 |
| numberOfEmitters | 1 |
| nearRangeM | 0.05 |
| farRangeM | 30.0 |
| rangeResolutionM | 0.013 |
| rangeAccuracyM | 0.03 |
| minDistBetweenEchos | 0.05 |
| minReflectance | 0.1 |
| minReflectanceRange | 10.0 |
| wavelengthNm | 905.0 |
| pulseTimeNs | 5 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.0 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.0 |
| maxReturns | 1 |

Note

For the datasheet and full list of specifications, vist the [RPLIDAR S2 product page.](https://www.slamtec.com/en/support#rplidar-s2)

### ZVISION

#### ML-30s+ (Certified)

[ZVISION ML-30s+](http://zvision.xyz/en/h-col-262.html) is a short range automotive grade solid state Lidar. Note there is no mesh for this lidar, so
when it is created via the UI, only a prim will appear in the Stage window.

Features and Specification

ML-30s+ Features

| name | ML-30s+ |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10 |
| numberOfEmitters | 51200 |
| numberOfChannels | 51200 |
| nearRangeM | 0.2 |
| farRangeM | 45.0 |
| effectiveApertureSize | 0.01 |
| focusDistM | 0.12 |
| rangeResolutionM | 0.03 |
| rangeAccuracyM | 0.03 |
| minDistBetweenEchos | 0.2 |
| minReflectance | 0.1 |
| minReflectanceRange | 270.0 |
| wavelengthNm | 905.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.025 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.025 |
| maxReturns | 2 |

To create the sensor from the menu: *Create>Sensors>RTX Lidar>ZVISION>ML30S+*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>ZVISION>ZVISION\_ML30S.usda*

Note

For the datasheet and full list of specifications, visit the [ML-30s+ product page.](http://zvision.xyz/en/h-col-262.html)

#### ML-Xs (Certified)

[ZVISION ML-Xs](http://zvision.xyz/en/h-col-279.html) is a long range automotive high performance grade solid state Lidar. Note there is no mesh for this lidar, so
when it is created via the UI, only a prim will appear in the Stage window.

Features and Specification

ML-30s+ Features

| name | ML-Xs |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10 |
| numberOfEmitters | 108000 |
| numberOfChannels | 108000 |
| nearRangeM | 0.5 |
| farRangeM | 250 |
| effectiveApertureSize | 0.01 |
| focusDistM | 0.12 |
| rangeResolutionM | 0.03 |
| rangeAccuracyM | 0.05 |
| minReflectance | 0.1 |
| minReflectanceRange | 270.0 |
| wavelengthNm | 1550.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.025 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.025 |
| maxReturns | 2 |

To create the Lidar prim: *Create>Sensors>RTX Lidar>ZVISION>MLXS*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>ZVISION>ZVISION\_MLXS.usda*

Note

For the datasheet and full list of specifications, visit the [ML-Xs product page.](http://zvision.xyz/en/h-col-279.html)

## Tactile Sensors

### Tashan Technology

#### Universal Tactile Sensor TS-F-A (Certified)

[Tashan Technology Universal Tactile Sensor TS-F-A](https://github.com/TashanTec/Tashan-Isaac-Sim) is a tactile simulation model based on real products to advance research and innovation in robotic tactile perception technology and promote the development of embodied intelligent robots.

Features and Specification

Outputs 11 dimensional feature channels:
:   * Proximity sensing [1]
    * Tactile sensing [2-4]: Normal force, tangential force, tangential force direction
    * Raw capacitance values [5-11]: 7-channel raw capacitance data

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Tashan>TS-F-A>TS-F-A.usd*

Note

For usage in Isaac Sim, visit the [Tashan Technology Tactile Simulation Platform User Manual.](https://github.com/TashanTec/Tashan-Isaac-Sim)

## Sensor Gizmo in Viewport

In Isaac Sim, the sensor functions are decoupled from physical meshes, and you can have sensors on stage without any mesh associated with the sensor. We use sensor gizmo to track the location of the actual sensing functions regardless of mesh. The gizmos are not visible by default, but you can toggle them on or off in the viewport.

To toggle the sensor gizmos, go to **Viewport Menu** >  > **Show By Type** > **Sensors**.

---

# Prop Assets

## Characters

Listed below are a few characters available in Isaac-Sim, located in the Content Browser inside the `Isaac Sim` folder.

### Police Man

Male character in police uniform with retargeted skeleton.

`People/Characters/original_male_adult_police_04/male_adult_police_04.usd` in the Content Browser.

### Male Doctor

Male character in doctor uniform with retargeted skeleton.

`People/Characters/origial_male_adult_medical_01/male_adult_medical_01.usd` in the Content Browser.

### Police Woman

Female character in police uniform with retargeted skeleton.

`People/Characters/female_adult_police_02/female_adult_police_02.usd` in the Content Browser.

### Construction Worker

Male character in construction uniform with retargeted skeleton.

`People/Characters/origial_male_adult_construction_03/male_adult_construction_03.usd` in the Content Browser.

Note

User can change a character’s clothing color by modifying material’s `Property -> Material and Shader` value

Here is an example of how to change male\_adult\_construction\_03’s safety hat’s color

* First, expand the character on the stage menu and navigate to their `Looks` folder. Example - `/World/male_adult_construction_03/Looks`.
* Next, select your target material (Example - `opaque__plastic__hardhat`) and change material’s `Property -> Material and Shader -> Albedo -> Color Tint` value to adjust character’s color.

### April Tags

We provide a simple mdl material that can index into a April Tag mosaic image.

To use, add the material to your stage using `Create->April Tag->`

Then create a mesh cube using `Create->Mesh->Cube` and assign the AprilTag material to that prim

The material has the following parameters which need to be configured:

* `Mosaic texture` The path to the texture that contains the grid of April tag images
* `Tag Size` The width/height of the tag in pixels
* `Tags Per Row` The number of tag images per row in the mosaic
* `Spacing` The number of padding pixels between each tag image
* `Tag ID` The index of the tag to use.

The figure below shows example usage using `tag36h11.png`,
after manually creating the mesh cube and assigning the material as described above.

---

# Environment Assets

## Simple Grid

This simple environment contains a flat ground and sides with a grid texture. Three configurations are provided; the first two have square corners, the third has curved corners.

|  |  |  |
| --- | --- | --- |
| [Square Grid Room](../_images/isaac_assets_default_grid_room.png)   search `default_environment.usd` in the Content Browser or using the create menu: *Create>Environments>Flat Grid*   Flat Grid. | [Square Grid Room](../_images/isaac_assets_square_grid_room.png)   search `gridroom_black.usd` in the Content Browser or using the create menu: *Create>Environments>Black Grid*   Black Grid. | [Curved Grid Room](../_images/isaac_assets_curved_grid_room.png)   search `gridroom_curved.usd` in the Content Browser   Curved Grid. |

## Simple Room

A simple room containing a table.

search `simple_room.usd` in the Content Browser or using the create menu: *Create>Environments>Simple Room*

## Warehouse

A warehouse environment with shelving and objects that can be placed on them. Four configurations are provided:

|  |  |
| --- | --- |
| [Simple Warehouse](../_images/isaac_assets_simple_warehouse.png)   search `warehouse.usd` in the Content Browser.   A small warehouse with a single shelf. | [Simple Warehouse](../_images/isaac_assets_simple_warehouse.png)   search `warehouse_with_forklifts.usd` in the Content Browser.   A small warehouse with a single shelf and forklifts. |
| [Small warehouse multiple shelves](../_images/isaac_assets_simple_warehouse_multiple_shelves.png)   search `warehouse_multiple_shelves.usd` in the Content Browser.   A small warehouse with multiple shelves. | [Full Warehouse](../_images/isaac_assets_full_warehouse.png)   search `full_warehouse.usd` in the Content Browser.   A full-sized warehouse with shelves, obstacles on the floors, and forklifts. |

## Hospital

A hospital environment, with multiple rooms and spaces.

Search `hospital.usd` in the Content Browser.

## Office

An Office Environment, with multiple rooms and an open plan floor.

Search `office.usd` in the Content Browser.

## JetRacer Track

A jetracer track outlined on the ground plane.

Search `jetracer_track_solid.usd` in the Content Browser.

## Small Warehouse Digital Twin

A digital twin of a small warehouse, it can be created using

Search `small_warehouse_digital_twin.usd` in the Content Browser.

---

# Featured Assets

**Nova Carter**

Powered by the [Nova Orin™](https://developer.nvidia.com/isaac/nova-orin) sensor and compute architecture, Nova Carter is a complete robotics development platform that accelerates the development and deployment of next-generation Autonomous Mobile Robots (AMRs).

Nova Carter is being used as a reference platform for both Isaac AMR and Isaac ROS software, enabling real-world and simulation-based development. Nova Carter robots may be purchased from [Segway Robotics](https://robotics.segway.com/nova-carter).

For more information on the fully-featured Nova Carter Isaac Sim Asset, please refer to the [Nova Carter](Isaac_Sim_Assets.md) documentation page.

Warning

Nova Carter robot may take multiple minutes to load for the first time.

---

# Nova Carter

Powered by the [Nova Orin™](https://developer.nvidia.com/isaac/nova-orin) sensor and compute architecture, Nova Carter is a complete robotics development platform that accelerates the development and deployment of next-generation Autonomous Mobile Robots (AMRs).

Nova Carter is being used as a reference platform for both Isaac AMR and Isaac ROS software, enabling real-world and simulation-based development. Nova Carter robots may be purchased from [Segway Robotics](https://robotics.segway.com/nova-carter).

The robot features the full Nova Orin sensor set, including four Leopard Imaging Hawk stereo cameras, four Leopard Imaging Owl fisheye cameras, IMUs, two 2D RPLidars, and one XT-32 3D Lidar. The robot digital twins with the cameras, Lidars, and IMU sensors are simulated in NVIDIA Isaac Sim and connected to the ROS 2 bridge for different robotics applications.

## Assets

### Nova Carter

The Nova Carter assets can be found on nucleus after NVIDIA Isaac Sim is installed, the Nova Carter assets are in the `/Isaac/Robots/NVIDIA/NovaCarter` folder, the ROS 2 assets are in the `/Isaac/Samples` folder, and the sample environments are in the `/Isaac/Sampls/ROS2/Scenarios/` folder.

Nova Carter

* `/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd`, the Nova Carter robot with no sensors attached.
* `/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd`, the Nova Carter robot with sensors attached and ROS 2 action graph enabled. This asset has been tested and verified before release.

Furthermore the nova\_carter.usd asset in NVIDIA Isaac Sim also contains prebuilt variants for different simulation and animation applications.

* Configuration:
  :   + Base: The Nova Carter robot all individual parts assembled
      + Fully Merged: The Nova Carter robot with all fixed parts merged into a single mesh for faster simulation
      + No\_Internals: The Nova Carter robot with no internal components like circuit board, battery.
      + Skirt\_only: The Nova Carter robot with only the skirt and wheels, useful for simulating the robot base without the upper structure or to mount custom parts on top.
* Physics:
  :   + No\_Physics: The Nova Carter robot with no physics, useful for visual only applications like animation
      + Physics\_Base: The Nova Carter robot with physics enabled
* Sensors
  :   + None: The Nova Carter robot with no sensors attached.
      + All\_Sensors: The Nova Carter robot with all sensors attached.

Frames and Topic names

Nova Carter Frame Names and Topic Names

| Device Name | Frame ID | Topic name |
| --- | --- | --- |
| Front Hawk | front\_stereo\_camera\_left\_optical | front\_stereo\_camera/left/image\_raw front\_stereo\_camera/left/camera\_info |
| Front Hawk | front\_stereo\_camera\_right\_optical | front\_stereo\_camera/right/image\_raw front\_stereo\_camera/right/camera\_info |
| Front Hawk | front\_stereo\_camera\_imu | /front\_stereo\_imu/imu |
| Front Hawk | front\_stereo\_camera\_left |  |
| Front Hawk | front\_stereo\_camera\_right |  |
| Front Hawk | front\_stereo\_camera |  |
| Back Hawk | back\_stereo\_camera\_left\_optical | back\_stereo\_camera/left/image\_raw back\_stereo\_camera/left/camera\_info |
| Back Hawk | back\_stereo\_camera\_right\_optical | back\_stereo\_camera/right/image\_raw back\_stereo\_camera/right/camera\_info |
| Back Hawk | back\_stereo\_camera\_imu | back\_stereo\_imu/imu |
| Back Hawk | back\_stereo\_camera\_left |  |
| Back Hawk | back\_stereo\_camera\_right |  |
| Back Hawk | back\_stereo\_camera |  |
| Left Hawk | left\_stereo\_camera\_left\_optical | left\_stereo\_camera/left/image\_raw left\_stereo\_camera/left/camera\_info |
| Left Hawk | left\_stereo\_camera\_right\_optical | left\_stereo\_camera/right/image\_raw left\_stereo\_camera/right/camera\_info |
| Left Hawk | left\_stereo\_camera\_imu | left\_stereo\_imu/imu |
| Left Hawk | left\_stereo\_camera\_left |  |
| Left Hawk | left\_stereo\_camera\_right |  |
| Left Hawk | left\_stereo\_camera |  |
| Right Hawk | right\_stereo\_camera\_left\_optical | right\_stereo\_camera/left/image\_raw right\_stereo\_camera/left/camera\_info |
| Right Hawk | right\_stereo\_camera\_right\_optical | right\_stereo\_camera/right/image\_raw right\_stereo\_camera/right/camera\_info |
| Right Hawk | right\_stereo\_camera\_imu | right\_stereo\_imu/imu |
| Right Hawk | right\_stereo\_camera\_left |  |
| Right Hawk | right\_stereo\_camera\_right |  |
| Right Hawk | right\_stereo\_camera |  |
| Front RP lidar | front\_2d\_lidar | front\_2d\_lidar/scan |
| Back RP lidar | back\_2d\_lidar | back\_2d\_lidar/scan |
| XT 32 | front\_3d\_lidar | front\_3d\_lidar/lidar\_points |
| Front owl | front\_fisheye\_camera\_optical | front\_fisheye\_camera/left/image\_raw front\_fisheye\_camera/left/camera\_info |
| Front owl | front\_fisheye\_camera |  |
| Back owl | back\_fisheye\_camera\_optical | back\_fisheye\_camera/left/image\_raw back\_fisheye\_camera/left/camera\_info |
| Back owl | back\_fisheye\_camera |  |
| Left owl | left\_fisheye\_camera\_optical | left\_fisheye\_camera/left/image\_raw left\_fisheye\_camera/left/camera\_info |
| Left owl | left\_fisheye\_camera |  |
| Right owl | right\_fisheye\_camera\_optical | right\_fisheye\_camera/left/image\_raw right\_fisheye\_camera/left/camera\_info |
| Right owl | right\_fisheye\_camera |  |
| Chassis IMU | chassis\_imu | /chassis/imu |
| Odometry |  | /chassis/odom |
| TF |  | /tf |

### Nova Dev Kit

The Nova Dev Kit is a development platform consist of 3 hawk stereo cameras and 3 owl fisheye cameras.

Nova Dev Kit

* `/Isaac/Robots/NVIDIA/NovaCarterDevKit/nova_dev_kit_sensors.usd`, the Nova Dev Kit with sensors attached.
* `/Isaac/Samples/ROS2/Robots/Nova_Dev_Kit_ROS.usd`, the Nova Dev Kit with sensors attached and ROS 2 action graph enabled.
* `/Isaac/Samples/ROS2/Robots/Nova_Dev_Kit_On_Robot_ROS.usd`, the Nova Dev Kit ROS model attached to a Nova Carter base. This asset has been tested and verified before release.

Frames and Topic names

Nova Dev Kit Frame Names and Topic Names

| Device Name | Frame ID | Topic name |
| --- | --- | --- |
| Front Hawk | front\_stereo\_camera\_left\_optical | front\_stereo\_camera/left/image\_raw front\_stereo\_camera/left/camera\_info |
| Front Hawk | front\_stereo\_camera\_right\_optical | front\_stereo\_camera/right/image\_raw front\_stereo\_camera/right/camera\_info |
| Front Hawk | front\_stereo\_camera\_imu | /front\_stereo\_imu/imu |
| Front Hawk | front\_stereo\_camera\_left |  |
| Front Hawk | front\_stereo\_camera\_right |  |
| Front Hawk | front\_stereo\_camera |  |
| Left Hawk | left\_stereo\_camera\_left\_optical | left\_stereo\_camera/left/image\_raw left\_stereo\_camera/left/camera\_info |
| Left Hawk | left\_stereo\_camera\_right\_optical | left\_stereo\_camera/right/image\_raw left\_stereo\_camera/right/camera\_info |
| Left Hawk | left\_stereo\_camera\_imu | left\_stereo\_imu/imu |
| Left Hawk | left\_stereo\_camera\_left |  |
| Left Hawk | left\_stereo\_camera\_right |  |
| Left Hawk | left\_stereo\_camera |  |
| Right Hawk | right\_stereo\_camera\_left\_optical | right\_stereo\_camera/left/image\_raw right\_stereo\_camera/left/camera\_info |
| Right Hawk | right\_stereo\_camera\_right\_optical | right\_stereo\_camera/right/image\_raw right\_stereo\_camera/right/camera\_info |
| Right Hawk | right\_stereo\_camera\_imu | right\_stereo\_imu/imu |
| Right Hawk | right\_stereo\_camera\_left |  |
| Right Hawk | right\_stereo\_camera\_right |  |
| Right Hawk | right\_stereo\_camera |  |
| Front owl | front\_fisheye\_camera\_optical | front\_fisheye\_camera/left/image\_raw front\_fisheye\_camera/left/camera\_info |
| Front owl | front\_fisheye\_camera |  |
| Left owl | left\_fisheye\_camera\_optical | left\_fisheye\_camera/left/image\_raw left\_fisheye\_camera/left/camera\_info |
| Left owl | left\_fisheye\_camera |  |
| Right owl | right\_fisheye\_camera\_optical | right\_fisheye\_camera/left/image\_raw right\_fisheye\_camera/left/camera\_info |
| Right owl | right\_fisheye\_camera |  |
| Odometry |  | /chassis/odom |
| TF |  | /tf |

## Sensors

### Hawk Stereo Camera

The Hawk stereo camera features two RGB camera sensors and a 6 axis IMU, and it is located at `Isaac/Sensors/LeopardImaging/Hawk/hawk_v1.1_nominal.usd`
The detailed specs can be found [here](Sensors.md).

Note: The front hawk is enabled by default for Nova\_Carter\_ROS.usd, additional hawk cameras can be enabled as needed, with increase load on computation.

* To enable to other hawk sensors, go to *Window > Graph Editors > ActionGraph*
* Click *Edit Action Graph* and select the action graph for the sensor to enable. (For example, */nova\_carter\_ros2\_sensors/back\_hawk*)
* Select the *Isaac Create Render Product Node* for the camera (there is one for the left camera, and one for the right camera), click *enabled*

### RPLidar

RP Lidar is a RTX based 2D lidar, that can be enabled by default and can be created by clicking *Create > Isaac > Sensors > RTX Lidar > SLAMTEC > RPLIDAR S2E*

Note

The RP Lidars are disabled by default, to enable them, follow the dropdown above and check `enabled`

### XT-32

XT-32 is a RTX based 3D lidar, that can be enabled by default and can be created by clicking *Create > Isaac > Sensors > RTX Lidar > HESAI > PandarXT-32 10hz*

Note

The XT-32 are enabled by default, to disable it, follow the dropdown above and uncheck `enabled`

## Getting Started

NVIDIA Isaac Sim has provided several ROS 2 samples with the Nova Carter robot for control and navigation.

### ROS 2 Sample Scene

First activate **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` tab.
The sample scene can be loaded after [enabling the ROS 2 Bridge Extension](ROS_2.md) by clicking *Robotics Examples > ROS2 > Isaac ROS > Sample Scene*.

This scene showcases a Nova Carter inside a small warehouse, with all Lidars and front hawk camera running from the robot frame. Please follow [Multiple Sensors in RViz2 section](ROS_2.md)
for visualizing the sensors and install `teleop-twist-keyboard` by following the [Driving Turtlebot Tutorial](ROS_2.md).

### ROS 2 Navigation Scene

The navigation scene can be loaded after [enabling the ROS 2 Bridge Extension](ROS_2.md) by clicking *Robotics Examples > ROS2 > Navigation > Nova Carter*.

Please follow [ROS 2 Navigation](ROS_2.md) tutorial for usage.

## Other Resources

* [Nova Orin](https://developer.nvidia.com/isaac/nova-orin)
* [Isaac AMR](https://docs.nvidia.com/isaac/doc/index.html)
* [Segway Nova Carter](https://robotics.segway.com/nova-carter)
* [Isaac Sim Overview](What_Is_Isaac_Sim.md)
* [ROS 2 Tutorials](ROS_2.md)
* [Hawk Stereo Camera](Sensors.md)

---

# Third-Party SimReady USD Assets

Isaac Sim welcomes open source assets from the community. This page outlines links to third-party assets that are compatible with Isaac Sim.

Third-Party SimReady USD Assets

| Link | Description |
| --- | --- |
| [Lightwheel SimReady store](https://simready.com) | Catalog of open sourced and closed source environments and prop assets. |
| [X-Humanoid ArtVIP Dataset](https://huggingface.co/datasets/x-humanoid-robomind/ArtVIP) | A large-scale dataset of articulated 3D objects and scenes. |
| [Synthesis Asset pack](https://synthesis.extwin.com) | A collection of assets for synthetic data generation. |
| [SpatialVerse dataset](https://huggingface.co/spatialverse) | InteriorAgent, InteriorGS, InteriorAgent\_Nav datasets for synthetic data generation. |
| [XGrid Scan to Simulation Tutorial](https://developer.xgrids.com/#/document?titleId=en-1761533581983) | Tutorial for converting a 3D scan to a simulation environment in NVIDIA Isaac Sim. |

---

# Neural Volume Rendering

NuRec (Neural Reconstruction) enables scene rendering in Omniverse using neural volumes derived from real-world images. These scenes, based on 3D Gaussian models, can be loaded into Isaac Sim as standard USD assets for visualization and simulation.

For more details on how NuRec works in Omniverse, including data preparation, rendering settings, and known limitations, see the [NuRec documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/neural-rendering.html). To generate compatible scenes, you can use the open-source project [3DGruT](https://github.com/nv-tlabs/3dgrut) which provides tools for training 3D Gaussian models from image collections and exporting them in a USDZ-based format suitable for use in Omniverse applications.

## Example

The following example demonstrates how to load a NuRec scene into Isaac Sim and run a simulation. The snippet iterates over the provided examples and starts by loading the provided stage, it then loads the carter navigation asset and sets the start location. It then checks if a collision ground plane needs to be created at the spawn location, and if so, creates a plane prim with a collision API applied. It then sets the carter navigation target prim location and runs the simulation for the given number of steps. During the simulation the wheeled robot will navigate towards the target location.

The example script can be run directly from the [Script Editor](Development_Tools.md) or as a [Standalone Application](Workflows.md).

### Prerequisites

* Download the NVIDIA NuRec Dataset from [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec).
* Update the `USER_PATH` variable in the script: `USER_PATH = "/home/user/PhysicalAI-Robotics-NuRec"`

Script Editor

Script Editor

```python
import asyncio
import os

import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path_async
from pxr import PhysxSchema, UsdGeom, UsdPhysics

# User path of the HF NuRec dataset
USER_PATH = "/home/user/PhysicalAI-Robotics-NuRec"

# Paths for loading and placing the Nova Carter navigation asset and its target.
NOVA_CARTER_NAV_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
NOVA_CARTER_NAV_USD_PATH = "/World/NovaCarterNav"
NOVA_CARTER_NAV_TARGET_PATH = f"{NOVA_CARTER_NAV_USD_PATH}/targetXform"
# Scenarios for testing navigation in the environments
EXAMPLE_CONFIGS = [
    {
        "name": "Voyager Cafe",
        "stage_url": f"{USER_PATH}/nova_carter-cafe/stage.usdz",
        "nav_start_loc": (0, 0, 0),
        "nav_relative_target_loc": (-3, -1.5, 0),
        "create_collision_ground_plane": False,
        "num_simulation_steps": 500,
    },
    {
        "name": "Galileo Lab",
        "stage_url": f"{USER_PATH}/nova_carter-galileo/stage.usdz",
        "nav_start_loc": (-2.5, 2.5, 0),
        "nav_relative_target_loc": (4, 0, 0),
        "create_collision_ground_plane": False,
        "num_simulation_steps": 500,
    },
    {
        "name": "Wormhole",
        "stage_url": f"{USER_PATH}/nova_carter-wormhole/stage.usdz",
        "nav_start_loc": (0, 0, 0),
        "nav_relative_target_loc": (5, 0, 0),
        "create_collision_ground_plane": False,
        "num_simulation_steps": 500,
    },
    {
        "name": "ZH Lounge",
        "stage_url": f"{USER_PATH}/zh_lounge/usd/zh_lounge.usda",
        "nav_start_loc": (-1.5, -3, -1.6),
        "nav_relative_target_loc": (-0.5, 5, -1.6),
        "create_collision_ground_plane": True,
        "num_simulation_steps": 500,
    },
]

async def run_example_async(example_config):
    example_name = example_config.get("name")
    print(f"Running example: '{example_name}'")

    # Open the stage
    stage_url = example_config.get("stage_url")
    if not stage_url:
        print(f"Stage URL not provided, exiting")
        return
    if not os.path.exists(stage_url):
        print(f"Stage URL does not exist: '{stage_url}', exiting")
        return

    print(f"Opening stage: '{stage_url}'")
    await omni.usd.get_context().open_stage_async(stage_url)
    stage = omni.usd.get_context().get_stage()

    # Make sure the physics scene is set to synchronous for the navigation to work
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            physx_scene.GetUpdateTypeAttr().Set("Synchronous")
            break

    # Load the carter navigation asset
    assets_root_path = await get_assets_root_path_async()
    carter_nav_path = assets_root_path + NOVA_CARTER_NAV_URL
    print(f"Loading carter nova asset: '{carter_nav_path}'")
    carter_nav_prim = add_reference_to_stage(usd_path=carter_nav_path, prim_path=NOVA_CARTER_NAV_USD_PATH)

    # Set the carter navigation start location
    nav_start_loc = example_config.get("nav_start_loc")
    if not nav_start_loc:
        print(f"Navigation start location not provided, exiting")
        return
    print(f"Setting carter navigation start location to: {nav_start_loc}")
    if not carter_nav_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_nav_prim).AddTranslateOp()
    carter_nav_prim.GetAttribute("xformOp:translate").Set(nav_start_loc)

    # Check if a collision ground plane needs to be created at the spawn location
    if example_config.get("create_collision_ground_plane"):
        plane_path = "/World/CollisionPlane"
        print(f"Creating collision ground plane {plane_path} at {nav_start_loc}")
        omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_path=plane_path, prim_type="Plane")
        plane_prim = stage.GetPrimAtPath(plane_path)
        plane_prim.GetAttribute("xformOp:scale").Set((10, 10, 1))
        plane_prim.GetAttribute("xformOp:translate").Set(nav_start_loc)
        if not plane_prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI.Apply(plane_prim)
        else:
            collision_api = UsdPhysics.CollisionAPI(plane_prim)
        collision_api.CreateCollisionEnabledAttr(True)
        plane_prim.GetAttribute("visibility").Set("invisible")

    # Set the carter navigation target prim location
    nav_relative_target_loc = example_config.get("nav_relative_target_loc")
    if not nav_relative_target_loc:
        print(f"Navigation relative target location not provided, exiting")
        return
    print(f"Setting carter navigation target location to: {nav_relative_target_loc}")
    carter_navigation_target_prim = stage.GetPrimAtPath(NOVA_CARTER_NAV_TARGET_PATH)
    if not carter_navigation_target_prim.IsValid():
        print(f"Carter navigation target prim not found at path: '{NOVA_CARTER_NAV_TARGET_PATH}', exiting")
        return
    if not carter_navigation_target_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_navigation_target_prim).AddTranslateOp()
    carter_navigation_target_prim.GetAttribute("xformOp:translate").Set(nav_relative_target_loc)

    # Run the simulation for the given number of steps
    num_simulation_steps = example_config.get("num_simulation_steps")
    if not num_simulation_steps:
        print(f"Number of simulation steps not provided, exiting")
        return
    print(f"Running {num_simulation_steps} simulation steps")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for i in range(num_simulation_steps):
        if i % 10 == 0:
            print(f"Step {i}, time: {timeline.get_current_time():.4f}")
        await omni.kit.app.get_app().next_update_async()

    print(f"Simulation complete, pausing timeline")
    timeline.pause()

async def run_examples_async():
    for example_config in EXAMPLE_CONFIGS:
        await run_example_async(example_config)

asyncio.ensure_future(run_examples_async())
```

Standalone Application

Standalone Application

```python
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import omni.kit.app
import omni.kit.commands
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, UsdGeom, UsdPhysics

# User path of the HF NuRec dataset
USER_PATH = "/home/user/PhysicalAI-Robotics-NuRec"

# Paths for loading and placing the Nova Carter navigation asset and its target.
NOVA_CARTER_NAV_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
NOVA_CARTER_NAV_USD_PATH = "/World/NovaCarterNav"
NOVA_CARTER_NAV_TARGET_PATH = f"{NOVA_CARTER_NAV_USD_PATH}/targetXform"
# Scenarios for testing navigation in the environments
EXAMPLE_CONFIGS = [
    {
        "name": "Voyager Cafe",
        "stage_url": f"{USER_PATH}/nova_carter-cafe/stage.usdz",
        "nav_start_loc": (0, 0, 0),
        "nav_relative_target_loc": (-3, -1.5, 0),
        "create_collision_ground_plane": False,
        "num_simulation_steps": 500,
    },
    {
        "name": "Galileo Lab - 1",
        "stage_url": f"{USER_PATH}/nova_carter-galileo/stage.usdz",
        "nav_start_loc": (-2.5, 2.5, 0),
        "nav_relative_target_loc": (4, 0, 0),
        "create_collision_ground_plane": False,
        "num_simulation_steps": 500,
    },
    {
        "name": "Wormhole",
        "stage_url": f"{USER_PATH}/nova_carter-wormhole/stage.usdz",
        "nav_start_loc": (0, 0, 0),
        "nav_relative_target_loc": (5, 0, 0),
        "create_collision_ground_plane": False,
        "num_simulation_steps": 500,
    },
    {
        "name": "ZH Lounge",
        "stage_url": f"{USER_PATH}/zh_lounge/usd/zh_lounge.usda",
        "nav_start_loc": (-1.5, -3, -1.6),
        "nav_relative_target_loc": (-0.5, 5, -1.6),
        "create_collision_ground_plane": True,
        "num_simulation_steps": 500,
    },
]

def run_example(example_config):
    example_name = f"{example_config.get('name')} - {example_config.get('num_simulation_steps')}"
    print(f"Running example: '{example_name}'")

    # Open the stage
    stage_url = example_config.get("stage_url")
    if not stage_url:
        print("Stage URL not provided, exiting")
        return
    if not os.path.exists(stage_url):
        print(f"Stage URL does not exist: '{stage_url}', exiting")
        return

    print(f"Opening stage: '{stage_url}'")
    omni.usd.get_context().open_stage(stage_url)
    stage = omni.usd.get_context().get_stage()

    # Make sure the physics scene is set to synchronous for the navigation to work
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            physx_scene.GetUpdateTypeAttr().Set("Synchronous")
            break

    # Load the carter navigation asset
    assets_root_path = get_assets_root_path()
    carter_nav_path = assets_root_path + NOVA_CARTER_NAV_URL
    print(f"Loading carter nova asset: '{carter_nav_path}'")
    carter_nav_prim = add_reference_to_stage(usd_path=carter_nav_path, prim_path=NOVA_CARTER_NAV_USD_PATH)

    # Set the carter navigation start location
    nav_start_loc = example_config.get("nav_start_loc")
    if not nav_start_loc:
        print(f"Navigation start location not provided, exiting")
        return
    print(f"Setting carter navigation start location to: {nav_start_loc}")
    if not carter_nav_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_nav_prim).AddTranslateOp()
    carter_nav_prim.GetAttribute("xformOp:translate").Set(nav_start_loc)

    # Check if a collision ground plane needs to be created at the spawn location
    if example_config.get("create_collision_ground_plane"):
        plane_path = "/World/CollisionPlane"
        print(f"Creating collision ground plane {plane_path} at {nav_start_loc}")
        omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_path=plane_path, prim_type="Plane")
        plane_prim = stage.GetPrimAtPath(plane_path)
        plane_prim.GetAttribute("xformOp:scale").Set((10, 10, 1))
        plane_prim.GetAttribute("xformOp:translate").Set(nav_start_loc)
        if not plane_prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = UsdPhysics.CollisionAPI.Apply(plane_prim)
        else:
            collision_api = UsdPhysics.CollisionAPI(plane_prim)
        collision_api.CreateCollisionEnabledAttr(True)
        plane_prim.GetAttribute("visibility").Set("invisible")

    # Set the carter navigation target prim location
    nav_relative_target_loc = example_config.get("nav_relative_target_loc")
    if not nav_relative_target_loc:
        print(f"Navigation relative target location not provided, exiting")
        return
    print(f"Setting carter navigation target location to: {nav_relative_target_loc}")
    carter_navigation_target_prim = stage.GetPrimAtPath(NOVA_CARTER_NAV_TARGET_PATH)
    if not carter_navigation_target_prim.IsValid():
        print(f"Carter navigation target prim not found at path: '{NOVA_CARTER_NAV_TARGET_PATH}', exiting")
        return
    if not carter_navigation_target_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_navigation_target_prim).AddTranslateOp()
    carter_navigation_target_prim.GetAttribute("xformOp:translate").Set(nav_relative_target_loc)

    # Run the simulation for the given number of steps
    num_simulation_steps = example_config.get("num_simulation_steps")
    if not num_simulation_steps:
        print(f"Number of simulation steps not provided, exiting")
        return
    print(f"Running {num_simulation_steps} simulation steps")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for i in range(num_simulation_steps):
        if i % 10 == 0:
            print(f"Step {i}, time: {timeline.get_current_time():.4f}")
        simulation_app.update()

    print(f"Simulation complete, pausing timeline")
    timeline.pause()

def run_examples():
    for example_config in EXAMPLE_CONFIGS:
        run_example(example_config)

run_examples()

simulation_app.close()
```