# Omnigraph

OmniGraph is Omniverse’s visual programming framework. It provides a graph framework that connects functions from multiple systems inside Omniverse. It is also a compute framework that allows for highly customized nodes so that you can integrate your own functionality into Omniverse and automatically harness the efficient computation backend.

Inside NVIDIA Isaac Sim, OmniGraph is the main engine for the Replicators, ROS 2 bridge, sensor access, controllers, external input/output devices, UI, and much more.

To access OmniGraph’s editor, go to **Window > Graph Editors > Action Graph**.

## Tutorials

---

# Commonly Used Omnigraph Shortcuts

Isaac Sim has shortcuts for populating some of the most commonly used Omnigraphs. They can be found under **Tools > Robotics > Omnigraph Controllers**. After selecting the graph you want to create, you are prompted to provide a minimal set of parameters to populate the graph.

The shortcuts are:

[Controller Graphs](#omnigraph-shortcuts-controller-graphs)

* Joint Position Controller
* Joint Velocity Controller
* Differential Controller
* Open Loop Gripper Controller

For information on how to use ROS Graphs, go to each of the relevant [ROS 2 Tutorials (Linux and Windows)](ROS_2.md).

Note

* *No* validation is done to detect a graph with the same tasks or that controls the same robot. You must ensure that your graphs are unique in the scene.
* These are just shortcuts to create the graph. You can always modify the graph after it’s created to suit your needs.

To use Python scripting to create these graphs:

> 1. Click on the icon next to **Python Script for Graph Generation** on the bottom of the popup window.
>    It takes you to the Python script used to generate the graphs for the given shortcut.
> 2. `make_graph()` is where the creation occurs. The relevant commands may or may not all be in one continuous block depending on how the shortcut is setup.

## Controller Graphs

The controller shortcuts for moving the robots are:

* Articulation (Joint Position and Velocity) Controllers
* Differential Drive Controller
* Gripper Controllers

### Articulation Controllers

Both Position and Velocity Controllers issue commands directly to each joint in the articulation.

* **Robot Prim**: The parent prim of the robot.
* **Graph Path**: The path to the graph generated. It is default to be under an independent tree called “/Graph/{type}\_controller”. If a graph already exist in the path given, it’ll find the next available path by appending a number to the end of that path.
* **Add to Existing Graph** (optional): Default to False. If checked, it’ll add the nodes to an existing graph and use an existing tick node if there exist one, but will add new controller nodes regardless of existing ones.

#### Use the Articulation Controller

To use the controller to move the robot:

1. Highlight the **JointCommandArray** node under the newly created graph.
2. Press *play* to start the simulation.
3. Move the robot by changing the values in the **JointCommandArray** node in the Property Tab.

If you had initial targets for position or velocity saved as part of the USD, it immediately moves towards those targets when you press **play**.

### Differential Controller

The Differential Controller takes in linear and angular velocities and converts them to individual wheel velocities.

* **Robot Prim**: The Robot Prim.
* **Graph Path**: The path to the graph generated. By default, it is under an independent tree called “/Graph/{type}\_controller”. If a graph already exist in the path given, it finds the next available path by appending a number to the end of that path.
* **Wheel Radius**: The radius of the wheel in meters.
* **Distance between wheels**: The distance between the two wheels in meters.
* **Right/Left Joint Names** (optional): Names of the joints that control the right and left wheels.
* **Right/Left Joint Index** (optional): The index of the joints that control the right and left wheels in the articulation chain.
* **Use Keyboard Control** (optional): Default to none. If checked, it also populates the graph that receives WASD as keyboard inputs to move the robot forward, backward, spin left, and spin right.
* **Add to Existing Graph** (optional): Defaults to False. If checked, it adds the nodes to an existing graph and uses an existing tick node if there is one, but will add new controller nodes regardless of existing ones.

#### Use the Differential Controller

* In some robots, there are only two controllable joints, so you do not have to specify joint names or indices. For robots with multiple actuated joints in an articulation chain, you must specify either the names or the indices of the joints that control the right and left wheels.
* If you did not include the WASD keyboard control in the graph, you can always test the controller by manually changing the “Desired Angular Velocity” and “Desired Linear Velocity” in the **DifferentialController** node under the newly created graph.

* If you are using the WASD Keyboard control, there are two scaling values used to scale the binary input from the keyboard to a linear velocity and an angular velocity that make sense for the vehicle’s size. The values are inside the nodes “ScaleLinear” and “ScaleAngular” respectively. You can print the output of the “DifferentialController” node to see relative affects of the scaling values. You want to tune them so that the rotating commands results in similar magnitude changes in the wheels’ velocities as the forward and backward commands.

* If you are using Isaac Sim Assets, the default values of the wheel radius and distance between wheels can be found on the bottom of the page for Wheeled Robots in [Robot Assets](Isaac_Sim_Assets.md)

### Gripper Controller

The Gripper Controller works for any end-effector that has only one-degree of actuation per finger. This includes all parallel jaw grippers, as well as any multi-finger, multi-DOF-per-finger hands where each finger has only one degree of actuation.

* **Parent Robot**: The robot that contains the gripper. This could be the gripper itself, or if the gripper is part of an arm, this could be the prim for the entire manipulator.
* **Gripper Root**: The prim that contains all the gripper joints.
* **Graph Path**: The path to the graph generated. It is default to be under an independent tree called “/Graph/{type}\_controller”. If a graph already exists in the path given, it finds the next available path by appending a number to the end of that path.
* **Gripper Speed**: The speed at which the gripper closes or opens in meters (or radian) per second.
* **Gripper Joint Names**: The names of the joints that control the gripper fingers. List them all out separated by commas.
* **Open/Close Position Limit** (optional): The joint position that’s considered fully open. Unit: meter (prismatic) or radian (revolute). If left blank, it defaults to the joint limits inside the asset’s USD file.
* **Use Keyboard Control** (optional): Default to none. If checked, it populates the graph that receives “O”,”C”, and “N” as keyboard inputs to open, close, and stop the gripper.
* **Add to Existing Graph** (optional): Defaults to False. If checked, it adds the nodes to an existing graph and uses an existing tick node if one exists, but will add new controller nodes regardless of existing ones.

#### Use the Gripper Controller

If no joint limits are given, the gripper defaults to the joint limits inside the asset’s USD file. If the Open Position Limit and Close Position Limit are flipped, the gripper controller automatically corrects for it. The controller makes the assumption that the joint limits for opened position is greater than closed position. So if it is the opposite for your gripper, you would have to either adjust your definition of open and close or modify the Python script accordingly.

* Only uniform speed and same joint limits are supported using the shortcut. If you want variable speed or different joint limits for each of the fingers, you can modify the graph by adding arrays for the speed and joint limit inputs.
* If the articulation chain you are working with contains both an arm and a gripper and you wish to control the arm using the Articulcation Position Controller and the Gripper Controller for the gripper separately:

  1. Remove the joints that control the gripper from the arm controller graph.
  2. Validate that there is no conflict between the two graphs.

---

# Custom Python Nodes

There already exist a large number of default nodes that comes with Isaac Sim. You can find the definitions and descriptions for them in either the [Omnigraph Node Library](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph/node-library/node-library.html "(in Omniverse Extensions)") or [API Documentation](API_Documentation.md). If those prove to be insufficient, you can write your own and integrate them into Isaac Sim.

A node is defined by two files, an .ogn file, which is a JSON file that defines the structure of the node, including its inputs, outputs, and parameters. Either a Python file or a C++ file can be used to define its function. Here we will focus on Python nodes.

## Node Files

All OmniGraph Node files starts with “Ogn” as a prefix. This is expected by the parser.

### Node Definition (.ogn)

The .ogn file is a JSON file that defines the structure of the node, including its inputs, outputs, and parameters. Here is an example of a simple node definition:

```python
 1{
 2 "NodeName": {
 3     "version": 1,
 4     "categories": "examples",
 5     "description": ["Minimum Example"],
 6     "language": "python",
 7     "metadata": {
 8         "uiName": "minimum example"
 9     },
10     "inputs": {
11                        "execIn": {
12             "description": "the trigger input that starts the node",
13             "type": "execution",
14         },
15                        "value_input": {
16             "type": "double",
17             "description": "a number",
18             "default": 0.0,
19          },
20     },
21     "outputs": {
22         "output_bool": {
23             "type": "bool",
24             "description": "let output be a boolean",
25          }
26      }
27   }
28}
```

A note about the input “execIn”. This is a special input that is used to trigger the node. This trigger is only relevant in an Action Graph, where you must explicitly trigger the node to run, such as on a physics tick, or a stage event, like opening and closing a stage. In a Push Graph, the node will run automatically at every frame and the ‘execIn’ input is not necessary.

### Function Definition

Here’s a minimum example of a Python node that takes an input number and outputs a boolean value based on whether the input is greater than 0:

```python
class OgnNodeName:
    @staticmethod
    def compute(db):
        db.outputs.out = bool(db.inputs.value_input > 0.0)
        return True
```

Notes:

* the class name must match the name of the node in the .ogn file, and the file name must match the class name.
* the “compute” function is what the ‘execIn’ input triggers. It takes a single argument, the database, which contains the inputs and outputs of the node. The function should return True if the node ran successfully, and False if it failed.
* this node has no internal state, which means all data that passes through it is gone the next tick. If you need to store data between ticks, you can use the “internal state” to store it.

## Using the Custom Node

You can simply insert your custom node’s `.py` and `.ogn` files into any of extensions that already have a directory that contains the `.py` and `.ogn` files for existing nodes and thereby avoid creating your own extension that way.

You can also create your own extension and insert the files there. (link to the new template generator)

## Isaac Sim Nodes as Examples

You are welcome to dig into the code behind some of our existing OmniGraph nodes to find examples of how to structure a node, or even modify them to suite your own need. To find the backend `.py` and `.ogn` files for a particular node. Hover your mouse over the node in the editor window, a tooltip window will appear and the name of the extension will be written in the parentheses. You can then navigate to the extensions’s folder that contains the backend scripts for the nodes by going to `exts/isaacsim.<ext_name>/isaacsim/<ext_name>/ogn/python/nodes/`.

Not all of the nodes are written in Python, some have C++ backends, so if you won’t necessarily see a corresponding `.py` and `.ogn` files for all the nodes on the list. Note that if you found a folder with a list of `Ogn<node_name>Database.py`, this is NOT the directory that contains the Python description of the node.

---

# Custom C++ Nodes

For C++ nodes, the [Node Definition (.ogn)](Omnigraph.md) is the same as the one used for Custom Python Nodes.

Examples of how to include Omnigraph nodes can be found in the extension template’s [GitHub repo](https://github.com/NVIDIA-Omniverse/kit-extension-template-cpp/tree/main/source/extensions/omni.example.cpp.omnigraph_node).

To use the custom C++ nodes, you will need also build your custom C++ extension. Follow [Kit C++ Extension Template](https://docs.omniverse.nvidia.com/kit/docs/kit-extension-template-cpp/latest/index.html) for the detailed instructions.

---

# Isaac Sim Omnigraph Tutorial

This tutorial introduces you to the world of visual programming via OmniGraph.
We highly recommend that you also read [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)"), because it is a key component in Omniverse Kit.

## Learning Objectives

This tutorial aims to

* walk you through building an action graph to control a robot in Isaac Sim, specifically, the Jetbot.
* show you how to use the Omnigraph shortcuts to generate a differential controller graph for the Jetbot.

## Build the Graph

Let’s build an action graph to control a robot in Isaac Sim the Jetbot.

### Setting Up the Stage

1. On a new stage, start by right clicking and selecting **create > Physics > Ground Plane**.
2. In the Content Browser, navigate to `Isaac Sim/Robots/NVIDIA/Jetbot/jetbot.usd`.
3. Click and drag `jetbot.usd` onto the stage.
4. Position the JetBot just above the ground plane.
5. When completed, verify that the JetBot is under `/World/jetbot` in the context tree and that the stage looks similar to:

Jetbot on the stage

Note

Click play! Validate that the JetBot falls and lands on the stage. Click stop before continuing.

Depending on your default render settings, the camera of the JetBot may have a placeholder mesh (it looks like a gray television camera).
To hide these meshes, click on the  icon in the viewport and select **Show By Type –> Cameras**.

### Building the Graph

1. Select **Window > Graph Editors > Action Graph** from the dropdown menu at the top of the editor.
   The Graph Editor appears in the same pane as the Content browser.
2. Click **New Action Graph** to open an empty graph.
3. Type `controller` in the search bar of the graph editor.
4. Drag an `Articulation Controller` and a `Differential Controller` onto the graph.

The `Articulation Controller` applies driver commands (in the form of force, position, or velocity) to the specified joints
of any prim with an articulation root.

To tell the controller which robot it’s going to control:

1. Select the `Articulation Controller` node in the graph and open up the property pane.
2. You can either:

   * Click **usePath** and Type in the path to the robot */World/jetbot* in **robotPath**

     **OR**
   * Click **Add Targets** near the top of the pane for `input:targetPrim` and select **JetBot** in the pop up window.

The `Differential Controller` computes drive commands for a two wheeled robot given some target linear and angular velocity. Like the
`Articulation Controller`, it also needs to be configured.

1. Select the `Differential Controller` node in the graph.
2. In the properties pane, set the `wheelDistance` to 0.1125, the `wheelRadius` to 0.03, and `maxAngularSpeed` to 0.2.

The `Articulation Controller` also needs to know which joints to articulate. It expects this information in the form of a list of tokens or index values. Each joint in a robot has a name and the JetBot has exactly two. Verify this by examining the JetBot in the stage context tree. Within `/World/jetbot/chassis`
are two revolute physics joints named `left_wheel_joint` and `right_wheel_joint`.

Stage Tree

1. Type `token` into the search bar of the graph editor.
2. Add two `Constant Token` nodes to the graph.
3. Select one and set it’s value to `left_wheel_joint` in the properties pane.
4. Repeat this for the other constant token node, but set the value to `right_wheel_joint`.
5. Type `make array` into the search bar of the graph editor.
6. Add a `Make Array` node to the graph.
7. Select the `Make Array` node and click on the `+` icon in the `inputs` section of the property pane menu to add a second input.
8. Set the `arraySize` to 2 and set the input type to `token[]` from the dropdown menu in the same pane.
9. Connect the constant token nodes to `input0` and `input1` of the `Make Array` node, and then the output of that node to the `Joint Names` input of the `Articulation Controller` node.

The last node is the event node.

1. Search for `playback` in the search bar of the graph editor.
2. Add an `On Playback Tick` node to the graph. This node emits an execution event for every frame, but only while the simulation is playing.
3. Connect the `Tick` output of the `On Playback Tick` node to the `Exec In` input of both controller nodes.
4. Connect the `Velocity Command` output of the differential controller to the `Velocity Command` input of the articulation controller.
5. Validate that the graph looks similar to:

Simple differential control for the JetBot

1. Press the play button.
2. Select the `Differential Controller` node in the graph.
3. Click and drag on either the angular or linear velocity values in the properties pane to change it’s value (or just click and type in the desired value).

Note

Explore the available OmniGraph nodes and try to setup a graph to control the JetBot with the keyboard. The graph
below is an example graph for controlling the JetBot with a keyboard.

Keyboard control Action graph for the JetBot

## Omnigraph Shortcuts

Putting the graph from scratch can be tedious, especially when you have to iterate. We made some shortcuts for frequently used graphs, so that within a couple clicks, you can generate a complex graph with multiple nodes and connections. They can be found under `Tools -> Robotics -> Omnigraph Controllers`, and the instructions for them are in [Commonly Used Omnigraph Shortcuts](Omnigraph.md).

To use the Differential Controller graph from the menu shortcut:

1. Delete (or Disable if that is an option) any previous OmniGraphs that controls the Jetbot.
2. Go to the Menu bar and click on **Tools -> Robotics -> Omnigraph Controllers -> Differential Controller**.
3. You are prompted for the necessary parameters.
4. Add “/World/jetbot” to `Articulation Root`, set the **distance between wheels** to 0.1125, and the **wheel radius** to 0.03.
5. Given JetBot only has two controllable joints, you can leave the rest of the fields empty.
6. Turn **Use Keyboard Control (WASD)** on.
7. Click **OK** to generate the graph. You can open the generated graph under `/Graph/differential_controller`.
8. Press **Play** to start simulation.
9. Verify that you can move the JetBot using the WASD keys on the keyboard.

## Summary

This tutorial covered:

* Basic concepts of OmniGraph
* Setting up a stage with a robot
* Using OmniGraph to construct interfaces to a robot
* Using the Omnigraph shortcuts to generate differential controller graph

### Further Learning

* More in-depth concepts in [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)")
* More details about all the OmniGraph shortcuts [Commonly Used Omnigraph Shortcuts](Omnigraph.md)
* Examples for composing OmniGraph via Python scripting: [OmniGraph via Python Scripting Tutorial](Omnigraph.md)
* Examples for writing custom Python nodes: [Custom Python Nodes](Omnigraph.md)

---

# OmniGraph via Python Scripting Tutorial

While OmniGraph is intended to be a visual scripting tool, it does have Python scripting interfaces. This tutorial will give some examples of how to script an action graph using Python.

## Learning Objectives

This tutorial will

* walk you through examples of scripting an Omnigraph using purely Python APIs
* introduce the basic concepts and frequently used parameters in OmniGraphs and showcase them using scripted examples

## Getting Started

**Prerequisites**

* Review the GUI Tutorial series, especially [Isaac Sim Omnigraph Tutorial](Omnigraph.md) and [Omniverse Script Editor](Development_Tools.md) prior to beginning this tutorial.
* Review the Core API Tutorial series, especially [Hello World](Python_Scripting_and_Tutorials.md) to become familiar with the extension workflow via Python, as well as the Python Standalone workflow.

## Code Snippets

### Creating a Graph

First let’s build a simple action graph that prints “Hello World” to the console on every simulation frame.

1. Open ‘Window > Script Editor’ and paste the following code:

   > ```python
   > import omni.graph.core as og
   >
   > keys = og.Controller.Keys
   > (graph_handle, list_of_nodes, _, _) = og.Controller.edit(
   >     {"graph_path": "/action_graph", "evaluator_name": "execution"},
   >     {
   >         keys.CREATE_NODES: [("tick", "omni.graph.action.OnTick"), ("print", "omni.graph.ui_nodes.PrintText")],
   >         keys.SET_VALUES: [
   >             ("print.inputs:text", "Hello World"),
   >             (
   >                 "print.inputs:logLevel",
   >                 "Warning",
   >             ),  # setting the log level to warning so we can see the printout in terminal
   >         ],
   >         keys.CONNECT: [("tick.outputs:tick", "print.inputs:execIn")],
   >     },
   > )
   > ```
2. Press ‘Run’ to execute the script. You should see a new prim `/action_graph` created on the Stage tree.
3. Expand the prim on stage, the nodes “tick” and “print” should be listed under the graph. These nodes can be accessed just like any other prim on the stage.
4. Press “play” to start the simulation. You should see “Hello World” printed to the console on every frame.
5. Open graph editor by going to Window > Graph Editors > Action Graph.
6. With the newly created graph highlighted on the Stage tree on the right, open the graph by clicking on the icon for ‘Edit Action Graph’ in the graph editor window. You should see two nodes connected with each other by a line.

### Editing a Graph

Once a graph has been created, there are specific APIs to manipulate the graph’s terms.

**Getting and Setting Attribute Values**

Open another tab in the Script Editor, paste the snippet below, and run.

```python
# get existing value from an attribute
existing_text = og.Controller.attribute("/action_graph/print.inputs:text").get()
print("Existing Text: ", existing_text)

# set new value
og.Controller.attribute("/action_graph/print.inputs:text").set("New Texts to print")
```

This will change the value in the “Print Text” node from “Hello World” to “New Texts to print”. But this affect won’t take place until the first tick through the graph. So when you press ‘Run’ in the script editor, the graph has yet to be ticked, so it should fetch the current value from the node, and print out a single string of “Existing Text: Hello World” in the Script Editor’s console (as well as the terminal if you are using that, or the main Omniverse’s console if you include “Info” to be printed).

Now press ‘Play’ and start the simulation. It should now print, at the rate of one string per tick, the updated text “New Texts to print”, in the terminal or the main Omniverse console (though not the Script Editor’s console).

**Adding Nodes and Connections**

Open a third tab in the Script Editor to add nodes and make more connections to an existing graph.

```python
og.Controller.create_node("/action_graph/new_node_name", "omni.graph.nodes.ConstantString")
og.Controller.attribute("/action_graph/new_node_name.inputs:value").set("This is a new node")
og.Controller.connect("/action_graph/new_node_name.inputs:value", "/action_graph/print.inputs:text")
```

A new node named “new\_node\_name” will be created and connected to the “Print Text” node. If you have the graph editor (Window > Graph Editors > Action Graph) open, you can see that there are now three nodes connected to each other instead of two.

### Graph Execution

By default, the graph is evaluated on every frame. You can change this behavior by setting the graph to evaluate only when you call it.

You can also trigger each graph explicitly by making execute only when you call it. To do this, there is a special parameter called “pipeline\_stage” where you can set the graph to execute “On Demand”. Most of the times we want to set this variable during the creation of the graph:

1. Delete the previous graph by selecting it on the stage tree and pressing ‘Delete’ key.
2. Open a new tab in the Script Editor and paste the following code

   > ```python
   > (demand_graph_handle, _, _, _) = og.Controller.edit(
   >     {
   >         "graph_path": "/ondemand_graph",
   >         "evaluator_name": "execution",
   >         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
   >     },
   >     {
   >         keys.CREATE_NODES: [("tick", "omni.graph.action.OnTick"), ("print", "omni.graph.ui_nodes.PrintText")],
   >         keys.SET_VALUES: [("print.inputs:text", "On Demand Graph"), ("print.inputs:logLevel", "Warning")],
   >         keys.CONNECT: [("tick.outputs:tick", "print.inputs:execIn")],
   >     },
   > )
   > ```
3. Press ‘Run’ in the Script Editor. A new graph `/ondemand_graph` will be created.
4. Start simulation by press “play”, nothing should be printed from this graph because we did not explicitly call to evaluate it.
5. To manually trigger a graph, open another tab, and paste in `demand_graph_handle.evaluate()`
6. Make sure simulation is still running. Click ‘Run’ in the Script Editor. You should see “On Demand Graph” printed to the console once.

Alternatively, you can also set it for an existing graph by `demand_graph_handle.change_pipeline_stage(og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND)`

A more in-depth example of attaching graphs to physics callbacks and/or rendering callbacks can be found in standalone\_examples/api/isaacsim.core.api/omnigraph\_triggers.py

## Summary

In this tutorial, we introduced scripting OmniGraph via Python.

### Further Reading

For more Python Scripting API in [OmniGraph APIs](https://docs.omniverse.nvidia.com/kit/docs/omni.graph/latest/omni.graph.core.html)
