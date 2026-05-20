# Quick Tutorials

If you are new to NVIDIA Isaac Sim, we recommend that you review the GUI references and complete the two tutorials listed below.

- Isaac Sim Basic Usage Tutorial
- Basic Robot Tutorial
- Tutorial Reference Table

In the *Quick Tutorials*, all the actions that can be performed in the GUI, can be performed using Python. You can switch between performing actions in the GUI and in Python. Anything you make inside the GUI can be saved as part the USD file.

For example, you can create the world, include the actions needed for your robots using the GUI. Then pull the entire USD file into a standalone Python script and systematically modify properties there as needed.

## For Beginners: Robot Setup Tutorials

If you are new to NVIDIA Isaac Sim and want to learn how to set up and rig robots, we **strongly recommend** exploring our comprehensive [Robot Setup Tutorials](Robot_Setup.md). These tutorials will teach you:

* How to set up environments and stages
* How to assemble and rig robots from basic shapes
* How to add joints, articulations, and sensors
* How to import and configure manipulators
* Advanced techniques for complex robot structures

The Robot Setup Tutorials are designed as a complete learning path that takes you from basic concepts to advanced robot rigging techniques. They are perfect for beginners who want to understand how to work with robots in Isaac Sim.

**Start here:** [Tutorial 1: Stage Setup](Robot_Setup.md)

## What’s Next

After completing either the Robot Setup Tutorials or Quick Tutorials, explore these additional resources:

* **Examples and Demos:** Access a library of examples and demos in [Examples](Examples.md) to explore use-cases and capabilities of NVIDIA Isaac Sim
* **Available Assets:** Browse the [Assets](Isaac_Sim_Assets.md) section to see what assets are available to you
* **Advanced Python API:** See [Build a Robot using Core Python API](Python_Scripting_and_Tutorials.md) for more complex tutorials

---

# Isaac Sim Basic Usage Tutorial

This tutorial covers the basics of Isaac Sim, including navigating the GUI, adding objects to the stage, looking up basic properties of objects, and running simulations.

In this tutorial, you will go from a blank stage to a moving robot using your choice of three different workflows. The purpose of including the three different workflows is to illustrate that Isaac Sim can be used in different ways depending on your needs.

You can review the scripts in both workflows to see how they differ. Comparing and contrasting can help you understand how to perform the exact same tasks:

* The **extension script** can be found in **Window > Examples > Robotics Examples**, then click on **Open Script** on the right upper corner of the browser.
* The **standalone script** can be found in the `<isaac-sim-root-dir>/standalone_examples/tutorials/` folder.

You can try the “hot-reloading” feature out by editing any of the scripts in the Extension examples. Save the file and see the changes reflected immediately without shutting down the simulator.

For a description of workflow concepts, see [Workflows](Workflows.md).

## Tutorial

There are three tabs for this tutorial, all three perform the same actions and reach the same outcome. Go through the full page under the same tab to learn about each workflow. Toggle between tabs to compare the different workflows or to perform the tutorial steps for your environment.

* GUI
* Extensions
* Standalone Python

GUI

Launch

1. Launch Isaac Sim from installation root folder.

   Linux

   ```python
   cd ~/isaacsim
   ./isaac-sim.sh
   ```

   Windows

   ```python
   cd C:\isaacsim
   isaac-sim.bat
   ```

   After the simulator is fully loaded, create a new scene:
2. From the top Menu Bar, click **File > New**. The first time you launch Isaac Sim, it may take a five - ten minutes to complete.

Add a Ground Plane

Add a ground plane to the scene:

1. From the top Menu Bar, click **Create > Physics > Ground Plane**.

Add a Light Source

You can add a light source to the scene to illuminate the objects in the scene. If you have a light source in the scene, but no object to reflect the light, the scene will still be dark.

Add a Distant Light source to the scene:

1. From the top Menu Bar, click **Create > Lights > Distant Light**.

Add a Visual Cube

A “visual” cube is a cube with no physics properties attached, for example, no mass, no collision. This cube will not fall under gravity or collide with other objects.

Add a cube to the scene:

1. From the top Menu Bar, click **Create > Shape > Cube**.
2. From the far left side of the UI locate the arrow icon and press **Play**. The cube does not do anything when simulation is running.

Move, Rotate, and Scale the Cube

Use the various gizmos on the left hand side toolbar to manipulate the cube.

1. Press “W” or click on the Move Gizmo to drag and move the cube. You can move it in only one axis by clicking on the arrows and drag, in two axes by clicking on the colored squares and drag, or in all three axes by clicking on the dot in the center of the gizmo and drag.
2. Press “E” or click on the Rotate Gizmo to rotate the cube.
3. Press “R” or click on the Scale Gizmo to scale the cube. You can scale it in one dimension by clicking on the the arrows and drag, two dimensions by clicking on the colored squares and drag, or in all three dimensions by clicking on the circle in the center of the gizmo and drag.
4. Press “esc” to deselect the cube.

For “Move” and “Rotate”, you can indicate if you are maneuvering in local or world coordinates. Click and hold on the gizmos to see the options.

You can make more precise modifications to the cube through its **Property** panel by typing in the exact numbers in the corresponding boxes. Click on the blue square next to the boxes to reset the values to default.

Add Physics and Collision Properties

Common physics properties are mass and inertia matrix, which are the properties that allow the object to fall under gravity. Collision Properties are the properties that allow the object to collide with other objects.

Physics and collision properties can be added separately, so you can have an object that collides with other objects but does not fall under gravity, or falls under gravity but does not collide with other objects. But in many cases, they are added together.

To add physics and collision properties to the cube:

1. Find the object (“/World/Cube”) on the stage tree and highlight it.
2. From the **Property** panel on the bottom right of the Workspace, click on the **Add** button and select **Physics** on the dropdown menu. This will show a list of properties that can be added to the object.
3. Select **Rigid Body with Colliders Preset** to add both physics and collision meshes to the object.
4. Press the **Play** button to see the cube fall under gravity and collide with the ground plane.

Extension

Launch

We will demonstrate the property of an Extension workflow using an existing Extension module called the “Script Editor”. The Script Editor allows the users to interact with the stage using Python. You will see that we will be mostly using the same Python APIs as in the Standalone Python workflow. The difference between the two workflows will become clear when we start to interact with the simulation timeline, especially in the [next tutorial](Quick_Tutorials.md).

Launch a fresh instance of Isaac Sim, go the top Menu Bar and click **Window > Script Editor**.

Add a Ground Plane

To add a ground plane using the interactive Python, copy paste the following snippet in the Script Editor and run it by clicking the **Run** button on the bottom.

```python
import omni.usd
from pxr import Sdf, UsdLux

stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)
```

Add a Light Source

You can add a light source to the scene to illuminate the objects in the scene. If you have a light source in the scene, but no object to reflect the light, the scene will still be dark.

1. Open a new tab in the Script Editor (**Tab > Add Tab**).
2. Add a light source by copy-pasting the following snippet in the Script Editor and running it.

```python
import omni.usd
from pxr import Sdf, UsdLux

stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)
```

Add a Visual Cube

A “visual” cube is a cube with no physics properties attached. No mass, no collision. This cube will not fall under gravity or collide with other objects. You can press **Play** to see that the cube does not do anything when the simulation is running.

1. Open a new tab in the Script Editor (**Tab > Add Tab**).
2. Add two cubes by copy-pasting the following snippet in the Script Editor and run it. We’ll keep one as visual-only, and add physics and collision properties to the other for comparison.

```python
import omni.usd
from pxr import Sdf, UsdLux

stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)
```

Isaac Sim Core API are wrappers for raw USD and physics engine APIs. You can add a visual cube (without physics and color properties) using raw USD API. Notice that the raw USD API is more verbose, but gives you more control over each property.

```python
import omni.usd
from pxr import Sdf, UsdLux

stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(300)
```

Add Physics and Collision Properties

Common physics properties are mass and inertia matrix, which are the properties that allow the object to fall under gravity. Collision Properties are the properties that allow the object to collide with other objects.

Physics and collision properties can be added separately, so that you can have an object that collides with other objects but does not fall under gravity, or falls under gravity but does not collide with other objects. But in many cases, they are added together.

In Isaac Sim core API, we’ve written wrappers for frequently used objects with all its physics and collision properties attached. You can add a cube with physics and collision properties using the following snippet.

```python
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

DynamicCuboid(
    prim_path="/dynamic_cube",
    name="dynamic_cube",
    position=np.array([0, -1.0, 1.0]),
    scale=np.array([0.6, 0.5, 0.2]),
    size=1.0,
    color=np.array([255, 0, 0]),
)
```

Alternatively, if you want to modify an existing object to have physics and collision properties, you can use the following snippet.

```python
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

DynamicCuboid(
    prim_path="/dynamic_cube",
    name="dynamic_cube",
    position=np.array([0, -1.0, 1.0]),
    scale=np.array([0.6, 0.5, 0.2]),
    size=1.0,
    color=np.array([255, 0, 0]),
)
```

Click the **Play** button to see the cubes fall under gravity and collide with the ground plane.

Move, Rotate, and Scale the Cube

Moving an object using core API:

```python
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

DynamicCuboid(
    prim_path="/dynamic_cube",
    name="dynamic_cube",
    position=np.array([0, -1.0, 1.0]),
    scale=np.array([0.6, 0.5, 0.2]),
    size=1.0,
    color=np.array([255, 0, 0]),
)
```

Moving an object using raw USD API:

```python
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

DynamicCuboid(
    prim_path="/dynamic_cube",
    name="dynamic_cube",
    position=np.array([0, -1.0, 1.0]),
    scale=np.array([0.6, 0.5, 0.2]),
    size=1.0,
    color=np.array([255, 0, 0]),
)
```

Standalone Python

Launch

The script that runs Part I, [Isaac Sim Basic Usage Tutorial](#isaac-sim-app-intro-quickstart), is located in `standalone_examples/tutorials/getting_started.py`.

To run the script, open a terminal, navigate to the root of the Isaac Sim installation, and run the following command:

Linux

```python
./python.sh standalone_examples/tutorials/getting_started.py
```

Windows

```python
python.bat standalone_examples\tutorials\getting_started.py
```

Code Explained

**Add a Ground Plane**

The lines inside `getting_started.py` that are relevant to adding a ground plane to the scene are below.

```python
from isaacsim.core.api.objects.ground_plane import GroundPlane

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
```

**Add a Light Source**

You can add a light source to the scene to illuminate the objects in the scene. If you have a light source in the scene, but no object to reflect the light, the scene will still be dark.

The lines inside `getting_started.py` that add a Distant Light are:

```python
from isaacsim.core.api.objects.ground_plane import GroundPlane

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
```

**Add a Visual Cube**

A “visual” cube is a cube with no physics properties attached. No mass, no collision. This cube will not fall under gravity or collide with other objects. You can press **Play** to see that the cube does not do anything when the simulation is running.

The lines inside `getting_started.py` that add a visual cube to the scene are:

```python
from isaacsim.core.api.objects.ground_plane import GroundPlane

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
```

**Add Physics and Collision Properties**

Common physics properties are mass and inertia matrix, which are the properties that allow the object to fall under gravity. Collision properties are the properties that allow the object to collide with other objects.

Physics and collision properties can be added separately, so you can have an object that collides with other objects but does not fall under gravity, or falls under gravity but does not collide with other objects. But in many cases, they are added together.

In the standalone Python script, physics properties can be added to the cube by turning it into a `RigidPrim` object as shown below. This gives the object the ability to fall under gravity.

```python
from isaacsim.core.api.objects.ground_plane import GroundPlane

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
```

The snippet below shows the lines that add collision properties, which gives the object the ability to collide with other objects.

```python
from isaacsim.core.api.objects.ground_plane import GroundPlane

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
```

Move, Rotate, and Scale the Cube

The snippet below shows the lines that moved the objects in the scene using the core API.

```python
from isaacsim.core.api.objects.ground_plane import GroundPlane

GroundPlane(prim_path="/World/GroundPlane", z_position=0)
```

Save your work.

You can now proceed to [the next tutorial](Quick_Tutorials.md).

---

# Basic Robot Tutorial

[Basic Robot Tutorial](#isaac-sim-app-intro-quickstart-robot), describes how to add a robot to the stage, move the robot, and examine the robot.

You must complete the previous [Isaac Sim Basic Usage Tutorial](Quick_Tutorials.md) before starting this one.

## Tutorial

GUI

Add a robot to Stage

1. Start with a new stage, **File > New Stage**.
2. Add robot to the scene, from the top Menu Bar, click **Create > Robots > Franka Emika Panda Arm**.

Examine the robot

Use the Physics Inspector to examine the robot’s joint properties.

1. Go to **Tools > Physics > Physics Inspector**. A window opens on the right.
2. Select Franka to inspect. The window will populate the joint information, such as the upper and lower limits as well as its default position by default.
3. Click on the hamburger icon on the top right to see more options, such as the joint stiffness and damping.
4. Optionally, make any changes to these values to see the robot move on the Stage corresponding to the change. A green check mark will appear.
5. To commit the changes to be the new default values for the robot, click the green check mark.

Control the Robot

The GUI-based robot controllers are inside the Omniverse visual programming tool, OmniGraphs. There are more involved tutorials about OmniGraph in the [Omnigraph](Omnigraph.md) section. For the purpose of this tutorial, we will generate the graph using a shortcut tool, and then examine the graph in the OmniGraph editor.

1. Open the graph generator by going to **Tools > Robotics > Omnigraph Controllers > Joint Position**.
2. In the newly appeared **Articulation Position Controller Inputs** popup window, click **Add** for the **Robot Prim** field.
3. Select **Franka** as the Target.
4. Click **OK** to generate the graph.

To move the robot:

1. In the Stage tab to the upper right, select **Graph > Position\_Controller**.
2. Select the **JointCommandArray** node. You can do this by either selecting the node on the Stage tree, or selecting the node in the graph editor.
3. In the **Property** tab to the lower right, you can see the joint command values. The **Inputs** under the **Construct Array Node** correspond to joints on the robot, starting with the base joint.
4. Press **Play** to start the simulation.
5. Click+hold+drag various value fields or type different values to see the robot arm change position.

To visualize the generated graph:

1. Open an graph editor window, **Window > Graph Editors > Action Graph**. The editor window opens in the tab below the Viewport tab that contains the robot.
2. Pull up the newly opened browser tab.
3. Click **Edit Action Graph** that is in the middle of the graph editor window.
4. Select the only existing graph on the list.
5. Select an array and review the **Stage** and **Property** tabs to see the values associated with each array node.
6. Select the **Articulation Controller** object in the graph to review its properties.

Extension

Add a robot to Stage

Start with a new Stage (File > New). To add a robot to the scene, copy-paste the following code snippet into the Script Editor and run it.

```python
import carb
import numpy as np
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
prim_path = "/World/Arm"

add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
arm_handle = Articulation(prim_paths_expr=prim_path, name="Arm")
arm_handle.set_world_poses(positions=np.array([[0, -1, 0]]))
```

Examine the robot

Isaac Sim Core API has many function calls to retrieve information about the robot. Here are some examples for finding the number of joints and the joint names, various joint properties, and joint states.

Open a new tab in the Script Editor, copy-paste the following code snippet. This can only be run after the previous adding robot step, where `arm_handle` has already been established. Press **Play** before running the snippet. Physics must be running for these commands to work.

```python
import carb
import numpy as np
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
usd_path = assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
prim_path = "/World/Arm"

add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
arm_handle = Articulation(prim_paths_expr=prim_path, name="Arm")
arm_handle.set_world_poses(positions=np.array([[0, -1, 0]]))
```

Notice when you pressed “Run”, it only prints the state once, even if the simulation is running. You’d have to keep pressing “Run” if you want to see more recent states. If you want to see the information printed at every physics step, you would need to insert these commands into a physics callback that runs at each physics step. We will go more in depth on how time stepping works in the next section [Workflows](Workflows.md).

To insert the commands into a physics callback, run the following snippet in a separate tab in the Script Editor.

```python
import asyncio

from isaacsim.core.api.simulation_context import SimulationContext

async def test():
    def print_state(dt):
        joint_positions = arm_handle.get_joint_positions()
        print("Joint positions: ", joint_positions)

    simulation_context = SimulationContext()
    await simulation_context.initialize_simulation_context_async()
    await simulation_context.reset_async()
    simulation_context.add_physics_callback("printing_state", print_state)

asyncio.ensure_future(test())
```

Start the simulation by pressing play, then run the snippet. You should see the information printed at every physics step into the terminal.

If printing at every physics step is no longer necessary, you can remove the physics callback by running the following snippet.

```python
import asyncio

from isaacsim.core.api.simulation_context import SimulationContext

async def test():
    def print_state(dt):
        joint_positions = arm_handle.get_joint_positions()
        print("Joint positions: ", joint_positions)

    simulation_context = SimulationContext()
    await simulation_context.initialize_simulation_context_async()
    await simulation_context.reset_async()
    simulation_context.add_physics_callback("printing_state", print_state)

asyncio.ensure_future(test())
```

Control the Robot

There are many ways to control the robot in Isaac Sim. The lowest level is sending direct joint commands to set position, velocity, and efforts. Here is an example of how to control the robot using the Articulation API at the joint level.

Open a new tab in the Script Editor, copy-paste the following code snippet. This can only be run after the previous adding robot step, where `arm_handle` has already been established. Press **Play** before running the snippet. Physics must be running for these commands to work. We will provide two positions for you to toggle between. If you’ve added the print state snippet above to each physics step, you should be able to see the printed joints numbers change as the robot moves.

```python
import asyncio

from isaacsim.core.api.simulation_context import SimulationContext

async def test():
    def print_state(dt):
        joint_positions = arm_handle.get_joint_positions()
        print("Joint positions: ", joint_positions)

    simulation_context = SimulationContext()
    await simulation_context.initialize_simulation_context_async()
    await simulation_context.reset_async()
    simulation_context.add_physics_callback("printing_state", print_state)

asyncio.ensure_future(test())
```

```python
# Set all joints to 0
arm_handle.set_joint_positions([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
```

Similar to the `get_joint_positions` function above, the `set_joint_positions` here is only get executed once when you pressed “Run”. If you wish to send commands at every physics step, you would need to insert these commands into a physics callback that runs at each physics step.

Standalone Python

Recall that the script that runs this tutorial is located in `standalone_examples/tutorials/getting_started_robot.py`. To run the script, open a terminal, navigate to the root of the Isaac Sim installation, and run the following command:

Linux

```python
./python.sh standalone_examples/tutorials/getting_started_robot.py
```

Windows

```python
python.bat standalone_examples\tutorials\getting_started_robot.py
```

Code Explained

Line 14 to 50 in getting\_started\_robot.py script sets up the scene and adds robots to the stage. The script starts by import necessary modules, add the ground plane, set the camera angle, and add two robots — one arm, one mobile —- to the scene, using the same APIs that’s used in the Extension workflow.

The notable differences between the Extension workflow Python and Standalone Python are:

**Starting the Simulator at the top**

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open
```

**Using a “World” object**

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open
```

We will explain more about the World object in the [Core API Overview](Python_Scripting_and_Tutorials.md). For now, think of it as the object that controls everything about this virtual world, such as physics and rendering stepping, and holding object handles.

**Stepping the Simulation explicitly**

At the bottom of the script, there is a loop, and a stepping function `my_world.step()` is called every iteration. Inside this stepping function, it move forward a fixed number of rendering and physics calculation.

The script will run for 4 cycles, and at each cycle, the arm and the car will move or stop moving. The car’s joint positions will be printed at every physics step in the last cycle.

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open
```

Notice the get\_joint\_positions and set\_joint\_positions functions are the same as the ones that are used in the Extension workflow, but since the stepping is explicit, the commands simply exist as part of the loop and gets executed every physics step by default. This is the main difference between the Extension and Standalone Python workflows. Go to the next section [Workflows](Workflows.md) for more details.

Save your work.

The next set of recommend tutorials are the GUI reference [Robot Setup Tutorials Series](Robot_Setup.md).

Or, you can continue to the next section to explore use-cases and capabilities of NVIDIA Isaac Sim by accessing a library of examples and demos in [Examples](Examples.md).

---

# Tutorial Reference Table

## Tutorial Series

The following tutorial series are available in Isaac Sim:

| Tutorial Series | Description |
| --- | --- |
| [Core API Tutorial Series](Python_Scripting_and_Tutorials.md) | Introductory tutorials for using the Isaac Sim Core Python API to manipulate the simulation environment and control robots |
| [Robot Setup Tutorials Series](Robot_Setup.md) | Introductory tutorials for assembling robots in Isaac Sim and asset optimization |
| [Importer and Exporter Tutorial Series](Importers_and_Exporters.md) | Tutorials for using the URDF, USD, and CAD importers and URDF exporter to import and export assets in Isaac Sim |
| [Isaac Lab Tutorials](Isaac_Lab.md) | Reinforcement Learning tutorials in Isaac Sim |
| [ROS Tutorial Series](ROS_2.md) | ROS 2 bridge and integration tutorials with Isaac Sim |
| [Synthetic Data Generation Tutorials](Synthetic_Data_Generation.md) | Synthetic data generation tutorials with Isaac Sim |
| [Sensor Tutorials](Sensors.md) | Tutorials for using RTX and PhysX sensors in Isaac Sim |
| [Motion Generation Tutorial Series](Robot_Simulation.md) | Tutorials for using motion generation in Isaac Sim |
| [Omnigraph Tutorials](Omnigraph.md) | Tutorials for using the Omnigraph to create and edit graphs in Isaac Sim |