# Digital Twin

## Warehouse Logistics

The warehouse logistics section contains extensions for building warehouses, generating conveyor belts, animating people, and using NVIDIA cuOpt for routing optimization.

- Warehouse Creator Extension
- Conveyor Belt Utility
- Static Warehouse Assets
- NVIDIA cuOpt

## Cortex

Cortex ties the robotics tooling of Isaac Sim together into a cohesive collaborative robotic system. The Cortex tutorials start with an overview of the core concepts and then steps through a series of examples of increasing sophistication.

- Isaac Cortex: Overview

## Mapping

- Mapping

### Troubleshooting

- Digital Twin Troubleshooting

Common Digital Twin issues and their solutions are documented in the [Digital Twin Troubleshooting](troubleshooting.html#isaac-sim-digital-twin-troubleshooting) page. For general simulation troubleshooting, see [Troubleshooting](Help_FAQ.md).

---

# Warehouse Creator Extension

The warehouse Creator Extension uses the new Modular Warehouse Assets to build a custom shape warehouse.

## Installing and Enabling the Extension

The extension can be installed/enabled by:

1. Navigating to **Window > Extensions** from the top-level menu
2. Searching for Warehouse Creator in the text field of the **Extension Manager** window, to reveal the `omni.warehouse_creator` result.
3. Clicking on the **Install** or **Update** button if not previously installed.
4. Toggling to enable the extension and checking the autoload checkbox if you like.

## How to Use

Start by navigating to **Tools > Modular Warehouse Creator**. Verify that you see:

> 

In the **Instructions** tab, there is a brief description on how to use it.

For faster warehouse creation, Download Isaac sim assets locally, and update your assets path. Refer to [Latest Release](Installation.md) for the download pack. On the Warehosue Creator tab, in the dataset source, click on the Folder icon and select the `[Isaac Sim Assets Path]/Isaac/Environments/Modular_Warehouse/Props` folder in the downloaded location.

### Drawing

To begin the Warehouse Generation, click on **Build Warehouse**. It places the viewport on build mode. A curve draw dialog will display while in this mode. **Do not interact with that dialog as it may disrupt the warehouse creation**.

Every click on the viewport gets translated into a segment for your warehouse wall in the draw mode. The warehouse is built in a counter-clockwise order. Start by placing a starting point, and moving your way along the shape you want to make. All points are automatically aligned by the warehouse tile size.

There are two methods to finish drawing:

1. Place a final point at the start. It closes the loop and places the center tiles to complete the warehouse.
2. If last point added is aligned with the first, **Finish** automatically closes the drawing and the warehouse interior is built.

Note

* The points must form a perimeter of the warehouse shape. The builder does not handle crossing lines during warehouse generation.
* The UI prevents you from placing two points too close to each other. To place the closing point, you can zoom in to the start point and click nearby it. The most effective way of achieving it is by placing it along the start edge opposite to the edge direction, as shown in the demonstration video.

### Styling

Most warehouse blocks have a selection of styles to choose from. By selecting the blocks, on the **Property** panel you can pick the style you want to use for that **Tile**. Each style may serve a different purpose in a warehouse. For example, like a loading dock, or an access panel.

Note

To facilitate selecting the blocks by clicking at them on the viewport, change the **Select Mode** by right-clicking the toolbar and set it to **Component**.

To select the style scroll down in the **Property** panel, and choose the desired option for each block type. It will affect all selected blocks of that type.

> 

#### Block Styles

The block styles are split by the type of blocks, which can vary from straight walls, corners in and out, or a center piece.
:   

### Editing Column Placement

Every block’s internal corner contains a quarter of a column to it. By combining all adjacent blocks, a column is formed through all the sub-components. Figuring out which components to enable or disable when adding or removing a column can become easily cumbersome. To facilitate this effort, there is the column Placement editor. To use it, select the floor plan prim, and then click on **Edit Column Placement** on the warehouse creator window.
This puts the editor in a column placement mode. The ceiling and details of the warehouse are hidden, and the floor plan with the columns shows.

By clicking on a column, it switches from “Enabled” to “Disabled” and vice-versa. “Enabled” columns are displayed in their default materials and appearance, while disabled columns are displayed in a translucent green. You can select multiple at once by click-dragging on the viewport. To enable or disable all columns, you can push the corresponding button in the UI. The **Flip All** button will reverse the enabled status in all columns.
When you are done with column edit, Click **Confirm** to save your current selection. To revert to how it was before entering editing mode, click **Cancel**.

> 

---

# Conveyor Belt Utility

## About

The Conveyor Belt Utility Extension provides an utility to turn Rigid bodies into conveyors in NVIDIA Isaac Sim.

## Usage

To use the extension:

1. Select **Window > Extensions**.
2. Search for “conveyor”.
3. Select `isaacsim.asset.gen.conveyor.ui`, and click on **Enable**. This will enable both the extension and it’s UI interface.

To auto-load this extension in the future, click on **Autoload** near the top of the `isaacsim.asset.gen.conveyor.ui` information pane of the extension manager.

To create a conveyor:

1. Select a rigid body or a mesh in the stage.
2. Go to the **Create > Isaac Sim > Warehouse Items > Conveyor**, to create a Omniverse OmniGraph node that will manage the conveyor speed and animation, with the following properties:

   > * conveyorPrim: The target that will have the conveyor velocity applied. If it’s not a Rigid body it will be automatically configured as one, with default collision models. Only a single prim is allowed per ConveyorNode.
   > * Animate Direction: Texture animation direction in the UV map.
   > * Animate Scale: ratio between conveyor velocity and texture animation.
   > * Animate Texture: flag to enable texture animation.
   > * Curved: Flag to indicate a curved conveyor belt. When true, applies angular velocity instead of linear velocity. The velocity is applied along the specified Direction as a rotation axis. The Direction axis can be scaled to adjust the velocity, with values greater than 1 increasing the curvature radius and values less than 1 decreasing it. For example, setting Direction to (0, 0, 1) will rotate about the Z axis, a common use case.
   > * Direction: Conveyor velocity direction in local coordinates.
   > * Enabled: Flag to enable/disable the conveyor system.
   > * Velocity: Conveyor velocity.

This Omniverse OmniGraph comes preconfigured with a variable for the velocity, so it can be changed by selecting the Omniverse OmniGraph prim directly. If you have multiple conveyors on a scene, you can also synchronize all velocities by selecting a single Omniverse OmniGraph’s variable in the read variable node (read\_speed).

To emulate a conveyor animation, you can use a tiled texture, and set the **Animate** properties to have the texture translate in the same direction and velocity of the conveyor movement.

Alternatively, you can define your own Omniverse OmniGraph and manually add the Conveyor nodes to it, letting you have multiple conveyor nodes on the same Omniverse OmniGraph.

For convenience, multiple conveyor pieces are provided with the Isaac Sim assets package are available standalone on the Isaac Sim default assets package at Isaac/Props/Conveyors.

When authoring the Conveyor functionality for these assets, be sure to have the Belt or Rollers prim selected, as these are the prims that contains the meshes for the conveyor elements.

## Digital Twin Library Conveyor System Generator

To facilitate the creation of Digital Twins, a utility to generate conveyor systems is provided at **Tools > Conveyor Track Builder**. This utility ships with our Digital Twin assets pack for conveyors, but you can use your own dataset, provided that you change the configuration file.

If an item selected on the screen is a component of the conveyor dataset, it will try to connect to one of the conveyor endpoints, as defined by the configuration, otherwise it will use the selection as a parent for the insertion of the new piece.

The configurator is made with loose integration with the assets, allowing flexibility when creating the conveyor system, with a minimal set of rules to facilitate the creation. This may cause the need for some minimal post-processing after creating the system, being a compromise so it won’t block you from fully customizing their track after it’s modeled.

### User Interface

| Ref # | Option | Result |
| --- | --- | --- |
| 1 | Conveyor Style | Styles of Conveyor Available, Can be Roller, Belt, or Dual |
| 2 | Track Type | Track Types, Can be Start, T-style split, straight, Y-style split, end. |
| 3 | Curvature | Track Curvature, Can be None, Half (usually for 90 degrees), or Full (usually for 180 degrees turn), to the left or right. |
| 4 | Elevation | Track Elevation. Can be one-level or two-levels up from the entry point, either Up or Down. |
| 5 | Selected Track | Shows the current selected track on screen, its endpoints, and the Delete button to remove the current track from the system. |
| 6 | New Track | Shows the piece marked for addition on the system. Lets you choose the input point, the track variants available on the dataset, and in some cases, gives the option to use a mirror of the piece |
| 7 | Track Variants | Shows the additional variants for the filter selection |
| 8 | Selected Endpoint | Each option relates to one of the track endpoints. Endpoints already used will not show on the UI, unless all endpoints are already connected. |
| 9 | Mirror | Mirrors the selected piece on the primary belt direction |

### Dataset

The dataset is a collection of USD files used for the system creation. Each USD file must:

* Have a Default Prim defined. That prim and all its children is what will be loaded when your asset is loaded as a reference.
* Have the default prim with an empty transform (Translate and rotation components set to zero).
* Have each conveyor track defined as an Xform Prim, with all visual/collision meshes parented by this Prim.
* Have the entry point of the tracks at the Origin, with the track aligned with the X-axis, with the origin at the middle of the track on the Y axis.
* Have the anchor points defined at Height zero (Z = 0), at the end of the track, aligned to the middle of the track in the Y axis. The X axis must be aligned with the base direction of the Track.
* Have individual materials defined for each track. Meshes that are part of the same conveyor base Prim can share materials.
* Be contained on the same base folder. They can have references to assets outside this base folder.

Accompanying the assets dataset, there is a JSON file that contains the metadata needed to the UI workflow, and to configure the conveyor physics, if the original assets don’t have the conveyor physics already embedded.

> ```python
> {
>     "assets": {
>         "ConveyorBelt_A01": {  # File name of the asset, without the extension.
>             "style": "DUAL",  # Conveyor style, can be ROLLER, BELT, or DUAL
>             "start_level": 0,  # Conveyor level for the track, can be any positive number
>             "angle": "HALF",  # Conveyor turn type, can be NONE, HALF, FULL"
>             "curvature": "SMALL",  # Conveyor radius of curvature, can be NONE, SMALL, MEDIUM, LARGE. Currently not used by the filter.
>             "ramp": "FLAT",  # Ramp level. How many levels it increases or decreases start level, can be FLAT, ONE, TWO, THREE, FOUR.
>             "type": "STRAIGHT",  # Track Type, can be START, STRAIGHT (used for all single track types, including curves and ramps), Y_MERGE, T_MERGE, FORK_MERGE, END.
>             "anchors": [  # All Prim children paths that correspond to  endpoints on the asset.
>                 "",  # This is the root of the conveyor, which is also an endpoint.
>                 "/Anchorpoint",  # For all the other anchors, keep the trailing / on the child prim name
>             ],
>             "conveyor_nodes": {  # All Child Prims to be configured as conveyors using the OmniGraph node. Each track should have its own configuration (in the case of merge and splits), even if it's of the same style
>                 "Rollers": {
>                     "animate_scale": 0.01,
>                     "animate_direction": [0.0, 1.0],
>                     "direction": [1.0, 0.0, -37.0],
>                     "curved": true,
>                 },
>                 "Belt": {
>                     "animate_scale": 0.5,
>                     "animate_direction": [1.0, 0.0],
>                     "direction": [0, 0.0, -37.0],
>                     "curved": true,
>                 },
>             },
>         }
>     }
> }
> ```

Note

Strict JSON types do not have comments, the snippet above have them included to explain the data. If you copy it, remember to remove the comments otherwise it will fail the extension.
For a full version of the JSON file, check the data folder in the extension.

### Changing the Configuration and Dataset Source

To change the dataset to be used, and the configuration file with your own:

1. Go to **Edit** > **Preferences** > **Conveyor Builder**.
2. Choose the source path to be used in either. The assets must be in the direct folder listed in the Conveyor Assets Location.

If you want to restore the original settings, click **Reset To Default**.

### Improving Load Time

By default, the tool uses the cloud-based assets folder. Because the tool only downloads the assets from the cloud the first time they are used, it can result in long wait times while the asset is loaded. To reduce this time, you can download the assets locally and update your assets location to the local path.

#### Available Tracks

|  |  |  |
| --- | --- | --- |
| [Conveyor A01](../../_images/isaac_conveyor_ConveyorBelt_A01.usd.png) | [Conveyor A02](../../_images/isaac_conveyor_ConveyorBelt_A02.usd.png) | [Conveyor A03](../../_images/isaac_conveyor_ConveyorBelt_A03.usd.png) |
| [Conveyor A04](../../_images/isaac_conveyor_ConveyorBelt_A04.usd.png) | [Conveyor A05](../../_images/isaac_conveyor_ConveyorBelt_A05.usd.png) | [Conveyor A06](../../_images/isaac_conveyor_ConveyorBelt_A06.usd.png) |
| [Conveyor A07](../../_images/isaac_conveyor_ConveyorBelt_A07.usd.png) | [Conveyor A08](../../_images/isaac_conveyor_ConveyorBelt_A08.usd.png) | [Conveyor A09](../../_images/isaac_conveyor_ConveyorBelt_A09.usd.png) |
| [Conveyor A10](../../_images/isaac_conveyor_ConveyorBelt_A10.usd.png) | [Conveyor A11](../../_images/isaac_conveyor_ConveyorBelt_A11.usd.png) | [Conveyor A12](../../_images/isaac_conveyor_ConveyorBelt_A12.usd.png) |
| [Conveyor A13](../../_images/isaac_conveyor_ConveyorBelt_A13.usd.png) | [Conveyor A14](../../_images/isaac_conveyor_ConveyorBelt_A14.usd.png) | [Conveyor A15](../../_images/isaac_conveyor_ConveyorBelt_A15.usd.png) |
| [Conveyor A16](../../_images/isaac_conveyor_ConveyorBelt_A16.usd.png) | [Conveyor A17](../../_images/isaac_conveyor_ConveyorBelt_A17.usd.png) | [Conveyor A18](../../_images/isaac_conveyor_ConveyorBelt_A18.usd.png) |
| [Conveyor A19](../../_images/isaac_conveyor_ConveyorBelt_A19.usd.png) | [Conveyor A20](../../_images/isaac_conveyor_ConveyorBelt_A20.usd.png) | [Conveyor A21](../../_images/isaac_conveyor_ConveyorBelt_A21.usd.png) |
| [Conveyor A22](../../_images/isaac_conveyor_ConveyorBelt_A22.usd.png) | [Conveyor A23](../../_images/isaac_conveyor_ConveyorBelt_A23.usd.png) | [Conveyor A24](../../_images/isaac_conveyor_ConveyorBelt_A24.usd.png) |
| [Conveyor A25](../../_images/isaac_conveyor_ConveyorBelt_A25.usd.png) | [Conveyor A26](../../_images/isaac_conveyor_ConveyorBelt_A26.usd.png) | [Conveyor A27](../../_images/isaac_conveyor_ConveyorBelt_A27.usd.png) |
| [Conveyor A28](../../_images/isaac_conveyor_ConveyorBelt_A28.usd.png) | [Conveyor A29](../../_images/isaac_conveyor_ConveyorBelt_A29.usd.png) | [Conveyor A30](../../_images/isaac_conveyor_ConveyorBelt_A30.usd.png) |
| [Conveyor A31](../../_images/isaac_conveyor_ConveyorBelt_A31.usd.png) | [Conveyor A32](../../_images/isaac_conveyor_ConveyorBelt_A32.usd.png) | [Conveyor A33](../../_images/isaac_conveyor_ConveyorBelt_A33.usd.png) |
| [Conveyor A34](../../_images/isaac_conveyor_ConveyorBelt_A34.usd.png) | [Conveyor A37](../../_images/isaac_conveyor_ConveyorBelt_A37.usd.png) | [Conveyor A38](../../_images/isaac_conveyor_ConveyorBelt_A38.usd.png) |
| [Conveyor A39](../../_images/isaac_conveyor_ConveyorBelt_A39.usd.png) | [Conveyor A40](../../_images/isaac_conveyor_ConveyorBelt_A40.usd.png) | [Conveyor A41](../../_images/isaac_conveyor_ConveyorBelt_A41.usd.png) |
| [Conveyor A42](../../_images/isaac_conveyor_ConveyorBelt_A42.usd.png) | [Conveyor A43](../../_images/isaac_conveyor_ConveyorBelt_A43.usd.png) | [Conveyor A44](../../_images/isaac_conveyor_ConveyorBelt_A44.usd.png) |
| [Conveyor A45](../../_images/isaac_conveyor_ConveyorBelt_A45.usd.png) | [Conveyor A46](../../_images/isaac_conveyor_ConveyorBelt_A46.usd.png) | [Conveyor A47](../../_images/isaac_conveyor_ConveyorBelt_A47.usd.png) |
| [Conveyor A48](../../_images/isaac_conveyor_ConveyorBelt_A48.usd.png) | [Conveyor A49](../../_images/isaac_conveyor_ConveyorBelt_A49.usd.png) |  |

---

# Static Warehouse Assets

Isaac Sim comes with a multitude of assets for you to build your own application. Additionally, there are extra asset libraries provided by NVIDIA that you can use. Open **Window** > **Browsers** > **NVIDIA Assets**, and the window **NVIDIA Assets** will show, where you can browse for all content to build your environment.

Let’s start by setting up the warehouse building. click on the **+** Icon next to **Industrial**, then on **Buildings**, and select **Warehouse**. By dragging **Warehouse01** to the scene, you’ll load a reference to the asset on your stage. Alternatively, you can also [build a custom warehouse](Digital_Twin.md).

Note

If you drag on the viewport window, it will let you place it at an arbitrary position, If instead you want it placed at the origin or on a given xform, drag it into the Stage window on top of the desired Prim.

Depending on which assets you goals, you may find **NVIDIA Assets** that are currently on a centimeter scale. This is because **NVIDIA Assets** are created by our art team, while **Isaac Sim Assets** have been curated with intent. Be mindful of the units! When importing certain assets, you may need to manually scale them down to units of 0.01. To do this, select the asset prim, click on “Add Transform” on the Properties pane, and set the scale to 0.01 on all directions.

Note

You can add a 0.01 scale on the parent prim you are adding the assets to instead (for example create a prim at /World/Warehouse\_Import and always drag the assets into it), and then all assets will be imported already scaled.

Now you can add some shelves for empty shelves, or racks for shelves filled with boxes.

Any asset in **NVIDIA Assets** can be used to compose your scene, browse around the categories to find the asset you need, or search by name.

## Simulation Needs

These assets are purely visual, so any simulation needs you may have need to be authored on top of it. In that case, the recommendation is to create a new stage, and drag one single asset to it and perform the desired authoring as a variation of the original asset, and save it on your nucleus. Then, on your environment, drag the asset you saved that contains the modifications.

## SimReady Assets

Omniverse also contains a suite of [SimReady Assets](https://docs.omniverse.nvidia.com/simready/latest/index.html), which are assets curated for machine learning and digital twins. These assets come fully annotated for Semantic Labeling, and also contains a preset physics setup so you can get started with your digital twin operation. For more details, visit the [NVIDIA On Demand session: SimReady Specification](https://www.nvidia.com/en-us/on-demand/session/omniverse2020-om1742/?playlistId=playList-63b157fe-95fe-4b93-8b9b-d731be32ec29)

### Example

Let’s make a variation of WarehousePile\_A04 that contains physics properties, with boxes being individual rigid bodies.

We start with a brand new Stage, and create an Xform under World with the name “Import”, and set its scale to 0.01

Then we drag the WarehousePile\_A04 into it.

To simplify the tree, we can bring the imported prim at the root. Click on the Option button on the stage, and select Show Root, then drag WareHousePile\_A04 on the Root, then right-click it, and select Set as Default Prim. Delete /World and /Environment. Next, select all children of /WareHousePile\_A04, and on the Properties pane, press **Add** > **Physics** > **Rigid Body with Colliders preset**. To see the effect of the rigid body API, add a ground plane by going to **Create** > **Physics** > **Ground Plane**, and start simulation. Try shift-click to drag one of the lower boxes, they should all fall over each other.

You can now go back to the previously saved asset to customize it to contain physics material properties, different mass properties, and so on. All changes will be stored locally and be applied on top of the original asset. To see the local changes, you can go to the Layer tab, right click the Root layer, and click on Edit.

You will see that the USD file opens in edit mode on your text editor, containing the reference to the original asset, and all “deltas” that are being applied to it.

---

# NVIDIA cuOpt

## Learning Objectives

Demonstrate and provide a reference for the use of [NVIDIA cuOpt](https://developer.nvidia.com/cuopt-logistics-optimization) to solve routing
optimization problems in simulation.

**Topics include:**

* Creation of waypoint network
* Basic interaction with the cuOpt service
* Visualization and processing of optimization specific data
* Intra-warehouse transport use case demonstration

*15-20 Minute Tutorial*

## Getting Started

**Prerequisites**

* Access to the NVIDIA [cuOpt server](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/index.html) and follow the [cuOpt Quickstart Guide](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/quick-start.html) to setup the cuOpt server.
* Review the Core API [Hello World](Python_Scripting_and_Tutorials.md) and introductory Tutorial series [Robot Setup Tutorials Series](Robot_Setup.md).
* NVIDIA cuOpt sample extensions are disabled by default. Enable the extensions required for this tutorial from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)")
  by searching for `omni.cuopt.examples`. Enabling `omni.cuopt.examples` will automatically enable `omni.cuopt.service` and `omni.cuopt.visualization`

## NVIDIA cuOpt Examples

> 

This tutorial is based around a set of three examples from the `omni.cuopt.examples` extension.
These examples are arranged in increasing fidelity from simple randomized routing problems
with only basic visualization, to an intra-warehouse transport scenario complete with high fidelity
warehouse assets. Each example leverages supporting extensions in order to interact with the cuOpt service
and visualize optimization data. The code for all examples and supporting extensions is available and
can be extended for use in other applications.

### Example Overview

* **Create Network** : An example demonstrating use of visualization tools to create a network. This network can be saved
  and re-used to represent the waypoint graph for optimization problems. Utilities from the `omni.cuopt.visualization` extension
  are used to create and display the waypoint graph.
* **Simple Cost Matrix** : A minimal example using simple primitives to represent a depot based fleet routing problem
  where vehicles start from a depot (Cone) and must fulfill the demand across all locations (Spheres). A cost Matrix
  representing the cost of travel between locations is created by measuring Euclidean distance between points. `omni.cuopt.service`
  is used for communication between scene data and a running cuOpt service instance.
* **Simple Waypoint Graph** : cuOpt also supports a weighted waypoint graph representation of the optimization environment,
  which is the focus of this example. Waypoint graphs are a common representation for interior environments where the cost
  between locations might not be predetermined and straight line distance is not sufficient. Here the waypoint graph represents
  the travel network, and target locations (a subset of graph nodes) represent the locations to be visited and the location
  of the fleet. In addition to using `omni.cuopt.service`, utilities from the `omni.cuopt.visualization` extension are used
  to process and display the waypoint graph as well as the resulting optimized routes. The waypoint graph, order data, and vehicle
  data are all loaded from JSON files that exist alongside the source code for this example and can be modified as needed.
* **Intra-warehouse Transport Demo** : In this example a more complex waypoint graph is generated to represent the transportation network
  for a warehouse environment. You are able to create and place semantic zones to denote high cost areas of travel to be avoided.
  In addition to the utilities used in the Simple Waypoint Graph example, additional functionality from the `omni.cuopt.visualization`
  extension is used to generate the warehouse environment from a JSON configuration files alongside the source code for the example.

### Supporting Extension Overview

* `omni.cuopt.service`: This extension contains a thin wrapper around the cuOpt service that is used for preprocessing
  scene data as well as formatting and sending requests to the cuOpt service. This extension also contains utilities for representing
  the optimization data and formatting text results to be displayed in the UI for the examples.
* `omni.cuopt.visualization`: This extension contains utilities for generating scene data including the waypoint graph, semantics zones
  and the warehouse environment. This extension also contains helper functions for adjusting the weight of graph edges based
  on proximity to a given semantic zone.

## Running cuOpt Examples

### Create Network

1. Starting from a New Isaac Sim Session (`CTRL + N`) navigate to the cuOpt menu item now present in the Isaac Sim interface and select Create Network.

   > 
2. In the Create Node section, click CREATE NODE:

   * **Create Node**: Creates a network node at default location. Move the node around to desired position. Multiple network nodes can be created
     one by one.
   > 
3. In the Create Edge section, click CREATE EDGE:

   * **Create Edge**: Creates an edge between two nodes. Select two nodes and click on create edge. Multiple edges between nodes can be created in
     the network one by one.
   > 
4. The created network will have Nodes and Edges that looks like:

   > 
5. Save the network file as USD for future use in optimization problems.
6. Click **Open Source Code** to view the reference implementation.

   > 

### Simple Cost Matrix

1. Starting from a New Isaac Sim Session (`CTRL + N`) navigate to the cuOpt menu item now present in the Isaac Sim interface and select Simple Cost Matrix.

   > 
2. Enter the credentials assigned to you for the NVIDIA cuOpt managed service.

   > See [Running cuOpt Examples](#credentials-cuopt).
   >
   > 
3. In the Optimization Problem Setup section, select values (or use defaults) for the following, then click SETUP PROBLEM:

   * **Fleet Size**: The maximum number of vehicles available. **Note** If a solution can be found using fewer vehicles that solution will be returned.
   * **Vehicle Capacity**: The number of stops (of demand=1) each vehicle can visit.
   * **Number of Locations**: The number of non-depot locations that must be visited.
   * **Solver Time Limit**: The amount of time the cuOpt solver is given to find an optimized solution. **Note** To maintain solution quality additional time should be given for larger problems.
   > 
4. In the Run cuOpt section, click SOLVE to return optimized routes. A text representation of the routes is displayed in the UI and results are also shown in the viewport.

   > 
5. Click **Open Source Code** to view the reference implementation.

   > 

### Simple Waypoint Graph

1. Starting from a New Isaac Sim Session (`CTRL + N`) navigate to the cuOpt menu item now present in the Isaac Sim interface and select Simple Waypoint Graph.

   > 
2. Enter the credentials assigned to you for the NVIDIA cuOpt managed service.

   > See [Running cuOpt Examples](#credentials-cuopt).
   >
   > 
3. In the Optimization Problem Setup section, click the LOAD buttons from top to bottom (Waypoint Graph, Orders, Vehicles) to setup the problem:

   * **Load Waypoint Graph** Clicking LOAD JSON loads a sample waypoint graph from `/extension_data/waypoint_graph.json`, which exists alongside
     the source code for this example. To load a network from a USD file created using the Create Network tools, drop the file into Stage window
     and click LOAD SCENE. A sample Network.usda is provided in `/extension_data/Network.usda`, which exists alongside
     the source code for this example.
   * **Load Orders** loads sample order data from `/extension_data/order_data.json`, which exists alongside the source code for this example.
     Order locations now appear in green.
   * **Load Vehicles** loads sample vehicle data from `/extension_data/vehicle_data.json`, which exists alongside the source code for this example.
     **Note** Vehicles are assigned to start from Node\_0 position, but are not shown in the viewport.
   > 
4. In the Run cuOpt section, click SOLVE to return optimized routes. A text representation of the routes is displayed in the UI and results are also shown in the viewport.

   > 
5. Click **Open Source Code** to view the reference implementation.

   > 

### Intra-warehouse Transport Demo

1. Starting from a New Isaac Sim Session (`CTRL + N`), navigate to the cuOpt menu item now present in the Isaac Sim interface and select Intra-warehouse Transport Demo.

   > 
2. Enter the credentials assigned to you for the NVIDIA cuOpt managed service.

   > See [Running cuOpt Examples](#credentials-cuopt).
   >
   > 
3. In the Optimization Problem Setup section, click the LOAD buttons from top to bottom (Sample Warehouse, Waypoint Graph, Orders, Vehicles, Semantic Zone) to setup the problem:

   * **Load Sample Warehouse** loads a sample warehouse defined by `/extension_data/warehouse_building_data.json`, conveyors defined by
     `/extension_data/warehouse_conveyors_data.json` and shelves defined by `/extension_data/warehouse_shelves_data.json`.
     All JSON files can be found alongside the source code for this example.
   * **Load Waypoint Graph** loads a sample waypoint graph from `/extension_data/waypoint_graph.json`, which exists alongside
     the source code for this example.
   * **Load Orders** loads sample order data from `/extension_data/order_data.json`, which exists alongside the source code for this example.
     Order locations now appear in green.
   * **Load Vehicles** loads sample vehicle data from `/extension_data/vehicle_data.json`, which exists alongside the source code for this example.
     **Note** Vehicles are assigned to start from Node\_0 position but are not shown in the viewport.
   * **(OPTIONAL) Create Semantic Zone** creates a semantic zone of user defined size starting at location `(0,0,0)`. If the generated semantic zone
     is placed over one or more edges in the waypoint graph, the edge within that semantic zone is assigned a very high travel cost. cuOpt attempts
     to avoid these edges if possible in the optimized solution. **Note** Each time the Generate button is clicked a new semantic zone is created.
   > 
4. In the Run cuOpt section, if a semantic zone has been created or moved, click UPDATE to capture the current weights. Then
   click SOLVE to return optimized routes. A text representation of the routes is displayed in the UI and results is also shown in the viewport.

   > 
5. Click **Open Source Code** to view the reference implementation.

   > 

## Additional Information

The examples shown here demonstrate only a small subset of cuOpt functionality. For additional features and advanced usage
see the [cuOpt Documentation](https://docs.nvidia.com/cuopt/) and the [cuOpt-Resources Repository](https://github.com/NVIDIA/cuOpt-Resources).

---

# Isaac Cortex: Overview

Cortex ties the robotics tooling of Isaac Sim together into a cohesive collaborative robotic
system. Collaborative robotic systems are complex and it will take some iteration for us to get this right. We provide
these tools to demonstrate where we’re headed and to give a sneak peak at the behavior programming
model we’re developing.

## Tutorial Sequence

The Cortex tutorials start with an overview of the core concepts (this tutorial), and then step
through a series of examples of increasing sophistication.

- Decider networks
- Behavior Examples: Peck Games
- Walkthrough: Franka Block Stacking
- Walkthrough: UR10 Bin Stacking
- Building Cortex Based Extensions

It’s best to step through the tutorials in order.

For more information, see
Nathan Ratliff’s [Isaac Cortex GTC22 talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42693/)

## Overview of Cortex

Industrial robots do productive work by leveraging the robot’s speed and repeatability. Integrators
structure the world around the robots so highly scripted behavior becomes a useful part of the
workcell and assembly line. But these robots are often programmed only to move through their joint
motions, and are generally unaware of their surroundings. That makes them easy to script, but also
extremely unsafe to be around. These fast moving, dangerous, machines usually live in cages for
safety to separate them from human workers.

Collaborative robotic systems are fundamentally different. These robots are designed to work in
close proximity with people, uncaged, on the long tail of problems that require more intelligence
and dexterity than can be addressed currently by scripted industrial robots. They need to be
inherently reactive and adapt quickly to their surroundings (especially when human co-workers are
nearby). But at the same time, we need them to remain easily programmable since their value is in the
diversity of applications they can support. Cortex is a framework built atop Isaac Sim designed to
address these issues. Cortex aims to make developing intelligent collaborative robotic systems as
easy as game development.

Collaborative robotic systems typically have perception modules streaming information into a world
model, and the robot must decide which skills to execute at any given moment to progress toward
accomplishing its task. Often there are many skills to choose from. The robot must decide when to
pick up an object, when to open a door, when to press a button, and ultimately which skills or
sequence of skills are most suited to accomplish these steps. We’re, therefore, faced with the
problem of organizing the available policies and controllers into an easily accessible API, and
enabling straightforward programming of these decisions. Additionally, we want to do this in a way
that enables users to develop first in simulation, then straightforwardly connect to perception and
real-robot controllers for controlling physical robots.

The Cortex pipeline (detailed in the next section) revolves around leveraging the simulator as the
world model. Isaac Sim has an expressive world representation based on USD with PhysX built on it, so
this world model is both a detailed database of state information and physically realistic.

To easily switch between simulation and reality, we need to be able to modularly remove the perception
component and operate directly on simulated ground truth during development. We also need to easily
switch between controlling just the simulated robot and streaming commands to an external robot.
Moreover, the decision framework should support reactivity as a first class citizen. All of these
requirements are innately supported by Cortex as described below.

Advanced note

Modern deep learning techniques often learn abstract latent spaces encoding world information,
but that doesn’t remove the need for a central database of information. We’re still a ways away
from end-to-end trained systems that are robust and transparent enough to be deployed in
production as holistic solutions. Until then, real-world systems will have many parts including
perception modules and many policies representing skills that we need to orchestrate into a
complete system. Even if each of these individual parts benefit from abstract latent spaces and
its own perceptual streams (for example, specialized skills with feedback), they still must be organized
into a programmable coordinated system. Cortex is where the parts come together into a complete
system.

### The Cortex Pipeline

Cortex centers around a 6 stage processing pipeline which is stepped every cycle at 60hz (see also
the diagram in the figure above):

1. **Perception:** Sensory streams enter the perception module and are processed into information
   about both what is in the world and where those objects are.
2. **World modeling:** This information is written into our USD database. USD represents our world
   belief capturing all available information. Importantly, this world model is visualizable, giving
   a window into the robot’s mind.
3. **Logical state monitoring:** A collection of logical state monitors monitor the world and record
   the current logical state of the environment. Logical state includes discrete information such as
   whether a door is open or closed or whether a particular object is currently in the gripper.
4. **Decision making:** Based on the world model and logical state, the system needs to decide
   what to do. What to do is defined by what commands are available through the exposed command API
   (see next item and below). The most basic form of decision model is a state machine. We build
   state machines into a new form of hierarchical decision data structure called a Decider Network
   which is based on years of research into collaborative robotics system programming at NVIDIA.
5. **Command API (policies):** Behavior is driven by policies, and each policy is governed by a set
   of parameters. For example, motion with collision avoidance, is governed by sophisticated motion
   generation algorithms, but parameterized by motion commands that specify the target end-effector
   pose and the direction along which the end-effector should approach. Developers can expose
   custom command API for available policies to be accessed by the decision layer.
6. **Control:** And finally, low-level control synchronizes the internal robotic state with the
   physical robot for real time execution.

Layers 2 through 5 (world, logical, decisions, commands) operate on the belief model (the simulation
running in the mind of the robot) and can be used entirely in simulation atop the Isaac Sim core API
without any notion of physical (or simulated) reality. They enable complex systems to be designed in
simulation first, focusing first on shaping the system’s behavior, before connecting to a physical
(or simulated) world. Then perception can be added connecting into the world model via ROS and
control can be added again connecting to the physical robot via ROS. Both of these stages can (and
arguably should) be tested in simulation using simulated perception and control. For that purpose,
we can use entirely separate simulated models with synthetic sensor data feeding into a real
perception and control modules which will be running in practice. That allows us to adjust the noise
characteristics, delays, and other real-world artifacts to profile and debug the end-to-end system
thoroughly entirely before trying it in the physical world.

This concept of separate belief and sim (or real) worlds is fundamental to Cortex. Cortex operates
on a simulation known as the belief (the mind of the robot). There may be a separate “external”
simulation running as well which simulates the real world. Or (equivalently, as far as the belief
simulation is concerned), the belief could be operating alongside the real physical world. Often we
depict these two worlds with one robot in front of the other (see the figure above): the robot in
front is the belief and the robot in back is the reality simulation.

There are two extensions associated with Cortex:

1. `Isaacsim.cortex.framework`: This extension handles the Cortex framework and base classes.
2. `isaacsim.cortex.behavior`: This extension handles the sample behaviors constructed with the framework.

Currently, the Cortex programming model works only with the standalone Python app workflow.

### A Basic Example

As an example, imagine you would like to have a robot arm follow a magical floating ball. In the
world there is the robot, the ball, and a camera. Here is how the 6 stages of processing map to this
robotics problem and world for each cycle:

1. **Perception:** An image is captured from the camera streamed and processed into the ball’s world
   transform, streamed as a `tf` via ROS to Cortex.
2. **World modeling:** The world is modeled in USD. The measured ball’s transform is recorded and made
   available to logical state monitors and behaviors which choose when and how frequently to
   synchronize the internal world model.
3. **Logical state monitoring:** A monitor is used to determine if the robot is gripping the ball. If
   the robot is gripping the ball the `has_ball` state is set to `True`, otherwise it is set to
   `False`.
4. **Decision making:** The ball’s current location, the robot’s current state, and the `has_ball` state
   are used in state machine to determine what the robot should do, either move towards the ball or
   do nothing.
5. **Command API:** If the robot should move towards the ball, the move towards the ball command is
   sent.
6. **Control:** Based on the commands received from the command API, low-level control commands are sent
   to the real robot via ROS.

## Command API

PhysX represents robots in generalized coordinates as what are called articulations. In Isaac Sim,
we wrap those into an `Articulation` class which provides a nice API for commanding the joints of the
articulation. Those joint commands correspond to low-level control commands. But often *policies* will
govern subsets of those joints to provide skills. For instance, a Franka arm’s articulation has 9
degrees of freedom, 7 for the arm and one for each of the two fingers. However, arm control on the
physical robot is separate from the gripper control, with the gripper commands being discrete (move
to position at a given velocity, close until the fingers feel a force).

Cortex provides a `Commander` abstraction which operates on a subset of the articulation’s joints
and exposes an interface to sending higher-level commands to a policy governing those joints. A
robot in the Cortex model is an articulation which has an associated collection of commanders
governing the joints. For instance, a `CortexFranka` robot has a `MotionCommander`
encapsulating the RMPflow algorithm governing the arm joints and a `FrankaGripper` commander
governing the hand joints. Commands can be sent to commanders either discretely or at every cycle,
and they’re processed by the commander into low-level joint commands every cycle. For example, for the
Franka, we can send a motion command (target pose with approach direction) and the commander will
incrementally move the joints until it reaches that target. That target can either be changed every
cycle (adapting to moving objects) or set just once; in either case, the motion commander will
incrementally move the joints to the latest command.

The commanders along with their commands constitute the *command API* exposed for a given robot. For
instance, if `robot` is a `CortexFranka` object, the two above mentioned commanders are exposed
as `robot.arm` (the `MotionCommander`) and `robot.gripper` (the `FrankaGripper`), so anyone
holding the `robot` object has access to the command APIs exposed by those commander objects. For
instance, we can call `robot.arm.send(MotionCommand(target_pose))` or its convenience method
`robot.arm.send_end_effector(target_pose)` to command the arm, and we can call methods such as
`robot.gripper.close()` to command the gripper.

Information about the latest command and the latest articulation action (low-level joint command) is
cached off in the commander and accessible by modules in the **control** layer for translating those
commands to the physical robot.

See `CortexFranka` in `isaacsim.cortex.framework/isaacsim/cortex/framework/robot.py` as an example of how these
tools come together.

## Note on Rotation Matrix Calculations

In many of the examples, especially the complete examples stepped through in
[Walkthrough: Franka Block Stacking](Digital_Twin.md)
and [Walkthrough: UR10 Bin Stacking](Digital_Twin.md)
we perform calculations on the end-effector or block transforms to calculate targets. To understand
those blocks of code, note that a rotation matrix can be interpreted as a frame in a particulate
coordinate system. Specifically, each column of the rotation matrix is an axis of the frame.

This is convenient when the axes have semantic meaning, such as for the end-effector. The Franka’s
end-effector frame is as shown in the following figure with the x-, y-, and z-axes depicted in red,
green, and blue, respectively.

These x-, y-, and z-axes, as vectors in world coordinates, form the column vectors of the
end-effector’s rotation matrix in world coordinates.

For instance, we might retrieve the end-effector’s rotation matrix and extract the corresponding
axes using using:

```python
import isaacsim.cortex.framework.math_util as math_util

R = robot.arm.get_fk_R()
ax, ay, az = math_util.unpack_R(R)
```

And we can compute the rotation matrix (target) that has the z-axis pointing down and maintains the
most similar y-axis using:

```python
import isaacsim.cortex.framework.math_util as math_util

R = robot.arm.get_fk_R()
ax, ay, az = math_util.unpack_R(R)
```

This type of math is common and can be understood as basic geometric manipulations to an
orthogonal frame.

---

# Decider networks

This tutorial steps through the basics of decider networks and demonstrates the concepts with some
simple examples. Decider networks, as a class of decision tools, include state machines, and we
include some examples of how to construct state machines using our built in tooling.

In all command line examples, we use the abbreviation `isaac_python` for the Isaac Sim python
script (`<isaac_sim_root>/python.sh` on Linux and `<isaac_sim_root>\python.bat` on Windows).
The command lines are written relative to the working directory
`standalone_examples/api/isaacsim.cortex.framework`.

Each example will launch Isaac Sim without playing the simulation. Press play to run the simulation
and behavior.

Related tutorials: [Behavior Examples: Peck Games](Digital_Twin.md) steps through scripting
a series of simple games for the Franka robot, building off the concepts presented here. The
tutorial emphasizes some of the limitations of state machines and illustrates how decider networks
simplify the development of reactive behaviors. Likewise,
[Walkthrough: Franka Block Stacking](Digital_Twin.md) walks through a complete demo of how
decider networks and state machines are used to develop an interactive block stacking behavior for
the Franka.

## Basics of decider networks

All behaviors in Cortex are decider networks. This section describes decider networks and covers the
basics of the framework tooling for implementing them. We also show how to implement state machines
using the framework.

### Decision framework tooling

A decider network is similar to a decision tree (although not strictly a tree, as described below),
but with a notion of statefulness. The full *decision framework* is implemented in
`isaacsim.cortex.framework/isaacsim/cortex/framework/df.py`. The decider network is represented by the
`DfNetwork` class.

Decider networks are formally directed acyclic graphs of `DfDecider` nodes. One node is designated
the *root*, and *leaf* nodes are nodes with no children. Each decider node’s job is to choose among
its children. This choice is made by the decider node’s `decide()` method. Every step of the
behavior (generally at 60hz, one step per physics step) the decider network algorithm traces from
the root down to a leaf following the sequence of decisions made by the decider nodes encountered
along the way.

The decider network is represented by the `DfNetwork` class. The root is passed into the
`DfNetwork` object on construction along with a custom *context* object which handles monitoring
logical state and gives each decider node access to the command API. Each decider node can access
the context object as `self.context` during execution.

Context objects derive from `DfLogicalState` which provides an API for adding logical state
monitors. Logical state monitors are simply functions that take the context object as input (e.g.
member functions of the context object are common); they’re called once every cycle in the order
they’re added. The context object also generally provides access to the robot’s command API. See for
instance `DfContext` the behavior tools in the module
`isaacsim.cortex.framework/isaacsim/cortex/framework/dfb.py`. The `DfContext` is a common base class which
additionally provides access to the robot along with its command API.

Here’s a simple example:

```python
import time

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.cortex.framework.df import DfDecider, DfDecision, DfNetwork
from isaacsim.cortex.framework.dfb import DfBasicContext
from isaacsim.cortex.framework.robot import add_franka_to_stage

class CustomContext(DfBasicContext):
    def __init__(self, robot):
        super().__init__(robot)

    def reset(self):
        # Called before the behavior is run. This is where logical state can be initialized.
        self.has_work = False

    def monitor_work(self):
        # Set the self.has_work logical state member if there's currently work to do.
        pass

class GoHome(DfDecider):
    def __init__(self):
        pass

class DoWork(DfDecider):
    def __init__(self):
        pass

class Dispatch(DfDecider):
    def __init__(self):
        super().__init__()
        self.add_child("go_home", GoHome())
        self.add_child("do_work", DoWork())

    def decide(self):
        # The decide method has access to the context object
        if self.context.has_work:
            return DfDecision("do_work")
        else:
            return DfDecision("go_home")

world = CortexWorld()
robot = world.add_robot(add_franka_to_stage(name="franka", prim_path="/World/franka"))
world.scene.add_default_ground_plane()

decider_network = DfNetwork(Dispatch(), context=CustomContext(robot))
world.add_decider_network(decider_network)

start_time = time.time()
world.run(simulation_app, is_done_cb=lambda: time.time() - start_time > 10)
simulation_app.close()
```

The section at the bottom illustrates how to create a world, add a robot and behavior, and run it.
See `isaacsim.cortex.framework/isaacsim/cortex/framework/cortex_world.py` for more information on the cortex world
and its API. Its stepped automatically from `world.run(simulation_app)` in a standard loop runner.
Stepping the world processes the logical state, decision (behavior), and command API (policy) layers
of the Cortex pipeline:

```python
def step(self, render: bool = True, step_sim: bool = True) -> None:
    if self._task_scene_built:
        ...
        if self.is_playing():
            # Cortex pipeline: Process logical state monitors, then make decisions based on that
            # logical state (sends commands to the robot's commanders), and finally step the
            # robot's commanders to handle those commands.
            for ls_monitor in self._logical_state_monitors.values():
                ls_monitor.pre_step()
            for behavior in self._behaviors.values():
                behavior.pre_step()
            for robot in self._robots.values():
                robot.pre_step()
        ...
```

The `world.add_decider_network(decider_network)` automatically adds a logical state monitor which
calls the context object’s monitors and adds a behavior which steps the decider network. More
generally, logical state monitors can be added using `world.add_logical_state_monitor(...)` and
behaviors can be added using `world.add_behavior(...)`.

### Statefulness of decider nodes and state machines

Every cycle, the decider network algorithm traces from the root to a leaf creating an execution path
(the sequence of decider nodes visited). (See the above figure.) From cycle to cycle it keeps track
of the execution path and uses the previous path to determine whether the decider nodes are making
the same decisions as before, or making different decisions. Each decide node has an `enter()` and
`exit()` method, and every time a new branch to a leaf is chosen, `exit()` is called on the
branch no longer taken in reverse order from the leaf to the branching point, and `enter()` is
called along the new branch in the order the new decider nodes are visited. The full decider node
API is

```python
class DfDecider(DfBindable):
    ...

    def enter(self):
        pass

    def decide(self):
        pass

    def exit(self):
        pass
```

`enter()` is called (along with `decide()`) only when the decider node is entered in the sense
defined above. As long as the execution path to this node remains consistent from step to step, only
`decide()` is called. Once it’s no longer reached, `exit()` is called. (`DfBindable`
indicates that objects of this type will be able to access the custom context object as
`self.context` during execution.)

These concepts are analogous to the entry and exit concepts of state machines. The decision
framework provides a `DfState` base class for defining state machines with an analogous API:

```python
class DfState(DfBindable):
    ...

    def enter(self):
        pass

    def step(self):
        pass

    def exit(self):
        pass
```

`enter()` is called on entry to the state, `step()` is called while in the state, and `exit()`
is called when the state is exited. The `step()` method indicates the state machine transition
through its return value. E.g. returning `self` will transition back to itself, and returning
`None` will transition to the terminal “do nothing” state. More generally, state transitions to
new states are implemented by returning a reference to that state object.

Since the concepts of entry, step/decide, and exit align between state machines and decider nodes
they are compatible within the decision framework. A `DfStateMachineDecider` is a decider node
which takes a start state of a state machine on construction and runs the state machine. The decider
node’s `enter()` method resets the state machine to the start state and the `decide()` method
steps the state machine.

One common use case is the sequential state machine. If `State1`, `State2`, and `State3` are
each `DfState` objects which transition to themselves while doing work and terminate (transition
to `None`) when finished, we can string them together into a sequential state machine using
`DfStateSequence([State1(), State2(), State3()])`. A `DfStateSequence` is itself a `DfState`
object which transitions back to itself, making it a hierarchical state machine. Internally, it runs
the states in sequence, transitioning to the next state whenever a state terminates. We can loop the
sequence using `DfStateSequence([State1(), State2(), State3()], loop=True)`

We can create a decider network that runs this state machine using:

```python
state = DfStateSequence([State1(), State2(), State3()], loop=True)
decider_network = DfNetwork(DfStateMachineDecider(state), context=DfContext(robot))
```

To see a complete example of using a looping sequential state machine run:

```python
isaac_python example_command_api_main.py
```

The robot will move the end-effector to a fixed target and maintain that target while changing the
nullspace arm configuration and opening and closing the gripper.

## Simple follow example

Run the follow example:

```python
isaac_python follow_example_main.py
```

It’ll launch the robot with a sphere at the end-effector. Select the sphere and drag it around with
the Move gizmo.

We’ll modify this simple example below. The final modified code is shown in
`follow_example_modified_main.py` for reference.

### Add an end-effector monitor

Currently, the decider network is created with just the default context object `DfContext`. We’ll
modify it to include a logical state monitor that monitors whether the end-effector has converged.

Add the following code

```python
class FollowContext(DfContext):
    def __init__(self, robot):
        super().__init__(robot)
        self.reset()

        self.add_monitors([FollowContext.monitor_end_effector, FollowContext.monitor_diagnostics])

    def reset(self):
        self.is_target_reached = False

    def monitor_end_effector(self):
        eff_p = self.robot.arm.get_fk_p()
        target_p, _ = self.robot.follow_sphere.get_world_pose()
        self.is_target_reached = np.linalg.norm(target_p - eff_p) < 0.01

    def monitor_diagnostics(self):
        print("is_target_reached: {}".format(self.is_target_reached))
```

Then modify the creation of the decider network to use this context object.

```python
class FollowContext(DfContext):
    def __init__(self, robot):
        super().__init__(robot)
        self.reset()

        self.add_monitors([FollowContext.monitor_end_effector, FollowContext.monitor_diagnostics])

    def reset(self):
        self.is_target_reached = False

    def monitor_end_effector(self):
        eff_p = self.robot.arm.get_fk_p()
        target_p, _ = self.robot.follow_sphere.get_world_pose()
        self.is_target_reached = np.linalg.norm(target_p - eff_p) < 0.01

    def monitor_diagnostics(self):
        print("is_target_reached: {}".format(self.is_target_reached))
```

Run the example again, and you’ll see `is_target_reached: <val>` printed out where `<val>` is
`False` when the end-effector is away from the target and `True` when it reaches the target.

### Setup automatic action on the monitored logical state

Adding the end-effector monitor toggles the `is_target_reached` logical state, but doesn’t do
anything with it. Now we’ll add a second monitor to the `FollowContext` class to automatically
open and close the gripper based on whether the end-effector is at the target.

```python
class FollowContext(DfContext):
    def __init__(self, robot):
        super().__init__(robot)
        self.reset()

        # New: add FollowContext.monitor_gripper to the monitor list
        self.add_monitors(
            [FollowContext.monitor_end_effector, FollowContext.monitor_gripper, FollowContext.monitor_diagnostics]
        )

    def reset(self):
        self.is_target_reached = False

    def monitor_end_effector(self):
        eff_p = self.robot.arm.get_fk_p()
        target_p, _ = self.robot.follow_sphere.get_world_pose()
        self.is_target_reached = np.linalg.norm(target_p - eff_p) < 0.01

    # New: Implement monitor_gripper()
    def monitor_gripper(self):
        if self.context.is_target_reached:
            self.robot.gripper.close()
        else:
            self.robot.gripper.open()

    def monitor_diagnostics(self):
        print("is_target_reached: {}".format(self.is_target_reached))
```

This will close the gripper once the target’s been reached and open it when it’s not.

Run the example again and play with the sphere target. If you move the target away from the
end-effector, you’ll see the gripper open and the end-effector each toward the target. Once the
target is reached, the gripper will close.

## Simple state machine

Run the following to launch an example of a simple state machine.

```python
isaac_python franka_examples_main.py --behavior=simple_state_machine
```

You’ll see the robot move its end-effector up and down moving between two pre-specified points.

## Simple decider network

Run the following to launch an example of a simple state machine.

```python
isaac_python franka_examples_main.py --behavior=simple_decider_network
```

You’ll see “<middle>” printed in the console. Select the `/World/motion_commander_target` prim in
the stage listing and select the Move gizmo. Move the end-effector to the left and right. When it
enters the left region (from the user’s perspective) it’ll print out “<left>”; when it moves back
into the middle region it’ll print out “<middle>”; and when it moves into the right region it’ll
print out “<right>”.

Note that this example additionally demonstrates passing parameters to a decider node.

```python
class Dispatch(DfDecider):
    def __init__(self):
        super().__init__()
        self.add_child("print_left", PrintAction("<left>"))
        self.add_child("print_right", PrintAction("<right>"))
        self.add_child("print", PrintAction())

    def decide(self):
        if self.context.is_middle:
            return DfDecision("print", "<middle>")  # Send parameters down to generic print.

        if self.context.is_left:
            return DfDecision("print_left")
        else:
            return DfDecision("print_right")
```

## Running other behaviors

Any of the behaviors listed in `isaacsim.cortex.behaviors/isaacsim/cortex/behaviors/franka` can be
loaded with this `franka_examples_main.py` example.

The full command line is

```python
isaac_python franka_examples_main.py --behavior=<behavior_name>
```

with `<behavior_name>` set to any of the following

```python
block_stacking_behavior
peck_state_machine
peck_decider_network
peck_game
simple_state_machine
simple_decider_network
```

Alternatively, you can load behaviors directly from their Python module:

```python
isaac_python franka_examples_main.py --behavior=<path_to_behavior>
```

This tutorial stepped through the last two “simple” behaviors. The `peck_state_machine`,
`peck_decider_network` and `peck_game` behaviors will be covered in
[Behavior Examples: Peck Games](Digital_Twin.md), and the `block_stacking_behavior` is
walked through in detail in [Walkthrough: Franka Block Stacking](Digital_Twin.md).

---

# Behavior Examples: Peck Games

This tutorial shows how to design simple behaviors and explores the tradeoffs between state machines
and decider networks. It steps through two implementations of a simple ground-pecking behavior with
the Franka robot where the robot must peck around the blocks. The first implementation uses a state
machine and is unable to react to blocks moved in front of its path. We fix that issue in the second
implementation with a simple decider network that internally leverages parts of the original state
machine. The state machine is effectively the same, but the higher-level decider can preempt the
state machine as needed for reactivity. Finally, we implement a reactive pick game using a pure
decider network. In that final example, we demonstrate the utility of the custom context object and
its monitors.

In all command line examples, we use the abbreviation `isaac_python` for the Isaac Sim python
script (`<isaac_sim_root>/python.sh` on Linux and `<isaac_sim_root>\python.bat` on Windows).
The command lines are written relative to the working directory
`standalone_examples/api/isaacsim.cortex.framework`.

Each example will launch Isaac Sim without playing the simulation. Press play to run the simulation
and behavior.

## Designing reactivity using decider networks

We start with a simple behavior that has the robot peck at the ground avoiding regions occupied by
blocks.

### State machine implementation

The `peck_state_machine` module implements this simple peck behavior as a
state machine. Run the behavior using:

```python
isaac_python franka_examples_main.py --behavior=peck_state_machine
```

The Franka robot will peck at the ground avoiding the blocks. You can move the blocks around to see
how that affects where the robot chooses to peck.

The implementation is straightforward:

```python
class PeckState(DfState):
    ...

    def enter(self):
        # On entry, sample a target.
        target_p = self.sample_target_p_away_from_obs()
        target_q = make_target_rotation(target_p)
        self.target = PosePq(target_p, target_q)
        approach_params = ApproachParams(direction=np.array([0.0, 0.0, -0.1]), std_dev=0.04)
        self.context.robot.arm.send_end_effector(self.target, approach_params=approach_params)

    def step(self):
        target_dist = np.linalg.norm(self.context.robot.arm.get_fk_p() - self.target.p)
        if target_dist < 0.01:
            return None  # Exit
        return self  # Keep going

def make_decider_network(robot):
    root = DfStateMachineDecider(
        DfStateSequence(
            [DfCloseGripper(width=0.0), PeckState(), DfTimedDeciderState(DfLift(height=0.05), activity_duration=0.25)],
            loop=True,
        )
    )
    return DfNetwork(root, context=PeckContext(robot))
```

With the simple state machine implementation, however, there’s an error case in reactivity. Try
moving a block directly into the path of a current peck. The behavior will hang trying
unsuccessfully to get the end-effector to the target.

The state machine chooses the target on entry and keeps it fixed throughout the behavior. It,
therefore, doesn’t react to the changing environment. State machines, by themselves, aren’t great at
modeling reactive behavior. We’ll use decider networks to fix this problem.

### Decider network implementation

The `peck_decider_network` module augments this simple peck behavior by adding
a reactive `Dispatch` decider node. Run the behavior using:

```python
isaac_python franka_examples_main.py --behavior=peck_decider_network
```

The decider network uses a logical state monitor to monitor whether there’s a block that would
prevent the end-effector from reaching the current peck target. If there is, it triggers the system
to re-choose the target.

```python
class PeckContext(DfContext):
    def __init__(self, robot):
        super().__init__(robot)
        self.robot = robot
        self.reset()
        self.add_monitors([PeckContext.monitor_active_target_p])

        def reset(self):
            self.is_done = True
            self.active_target_p = None

        # Monitor whether a block is too close to the active target.
        def monitor_active_target_p(self):
            if self.active_target_p is not None and self.is_near_obs(self.active_target_p):
                self.is_done = True

        # Called by a special state at the end of the peck behavior.
        def set_is_done(self):
            self.is_done = True

    ...

class PeckState(DfState):
    def enter(self):
        target_p = self.context.active_target_p
        target_q = make_target_rotation(target_p)
        self.target = PosePq(target_p, target_q)
        approach_params = ApproachParams(direction=np.array([0.0, 0.0, -0.1]), std_dev=0.04)
        self.context.robot.arm.send_end_effector(self.target, approach_params=approach_params)

    def step(self):
        # Send the command each cycle so exponential smoothing will converge.
        target_dist = np.linalg.norm(self.context.robot.arm.get_fk_p() - self.target.p)
        if target_dist < 0.01:
            return None  # Exit
        return self  # Keep going

class Dispatch(DfDecider):
    def __init__(self):
        super().__init__()

        self.add_child("choose_target", ChooseTarget())
        self.add_child(
            "peck",
            DfStateMachineDecider(
                DfStateSequence(
                    [
                        CloseGripper(),
                        PeckState(),
                        DfTimedDeciderState(DfLift(height=0.05), activity_duration=0.25),
                        DfWriteContextState(lambda context: context.set_is_done()),
                    ]
                )
            ),
        )

    def decide(self):
        if self.context.is_done:
            return DfDecision("choose_target")
        else:
            return DfDecision("peck")

def make_decider_network(robot):
    return DfNetwork(Dispatch(), context=PeckContext(robot))
```

Note that the top-level `Dispatch()` decider can immediately preempt the sequential “peck” state
machine if the monitor `monitor_active_target_p()` detects a block to close to the target.

Try moving the block under the end-effector. This time, every time the block gets too close to the
end-effector’s target, it immediately chooses a different target.

## Designing logical state contexts

Now let’s implement a simple game where the robot pecks the block that’s most recently been moved.
We use decider networks to make it simple to program reactivity to the block movements. This example
demonstrates how simple behaviors can be when the logical state is sufficiently modeled by the
context object.

The behavior is implemented in `peck_game`. Run the behavior using:

```python
isaac_python franka_examples_main.py --behavior=peck_game
```

The `PeckContext` class handles monitoring block movement and setting the latest active target
accordingly. It also monitors whether the end-effector is close to the block which is useful in
deciding whether the robot needs to lift away from the block before moving to it’s next target.

```python
class PeckContext(DfLogicalState):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot

        self.monitors = [
            PeckContext.monitor_block_movement,
            PeckContext.monitor_active_target_p,
            PeckContext.monitor_active_block,
            PeckContext.monitor_eff_block_proximity,
            PeckContext.monitor_diagnostics,
        ]
```

Given the logical state monitored by the context object, the main logic can be concisely written as
the following `Dispatch` decider node:

```python
class PeckContext(DfLogicalState):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot

        self.monitors = [
            PeckContext.monitor_block_movement,
            PeckContext.monitor_active_target_p,
            PeckContext.monitor_active_block,
            PeckContext.monitor_eff_block_proximity,
            PeckContext.monitor_diagnostics,
        ]
```

In words, it reasons according to the following rules: If the end-effector is close to an inactive
block, we need to just lift away from it (it’s too close). Otherwise, if there’s an active block,
move to peck it. If no block is active, go home. This `decide()` method is ticked every cycle so
immediately once the active block monitor notices a block has moved, it acts on it.

---

# Walkthrough: Franka Block Stacking

This tutorial walks through a complete reactive block stacking application. This example builds
the scene entirely using the Isaac Sim core Python API and runs the behavior. See
[Walkthrough: UR10 Bin Stacking](Digital_Twin.md) for an example of designing a behavior for
an existing USD environment.

In all command line examples, we use the abbreviation `isaac_python` for the Isaac Sim python
script (`<isaac_sim_root>/python.sh` on Linux and `<isaac_sim_root>\python.bat` on Windows).
The command lines are written relative to the working directory
`standalone_examplesomni/api/isaacsim.cortex.framework`.

Run the following demo

```python
isaac_python franka_examples_main.py --behavior=block_stacking_behavior
```

Press play once Isaac Sim has started up. You’ll see the Franka block stacking demo running.

This tutorial will step through the demo.

The environment has a Franka robot with a set of 4 blocks of different colors. Its goal is to stack
the blocks into a tower in a pre-defined order. This behavior is reactive and robust to user
interaction. Users can move the blocks as shown in the video and the robot will adapt as needed to
continue progressing toward its goal.

## Block stacking decider network

The decider network is shown in the figure below.

In words, the dispatch revolves around the state of the tower and the gripper. If the tower’s done,
then go home. If there’s more work to do, if there’s a block in the gripper, place it somewhere.
Otherwise, if there’s no block in the gripper, acquire a block. At the next level down, it decides
where to place the block and which block to acquire. The pick block and place at target decider
nodes (action nodes) are the same in all cases and key off parameters passed in from the decider
nodes one level up.

## Top-level dispatch

The behavior is constructed as a decider network:

```python
def make_decider_network(robot):
    return DfNetwork(
        BlockPickAndPlaceDispatch(), context=BuildTowerContext(robot, tower_position=np.array([0.25, 0.3, 0.0]))
    )
```

with top-level dispatch decider node `BlockPickAndPlaceDispatch`. The dispatch decider node’s
implementation is simple, directly modeling the logic shown in the above diagram:

```python
def make_decider_network(robot):
    return DfNetwork(
        BlockPickAndPlaceDispatch(), context=BuildTowerContext(robot, tower_position=np.array([0.25, 0.3, 0.0]))
    )
```

If the tower’s complete, then go home. Otherwise, there’s more to do with the tower. If there’s
nothing in the gripper (gripper clear) then pick up a block. Otherwise, there’s a block in the
gripper, and place it somewhere. The rest is about both choosing what to pick or where to place the
current block, and how to perform this specific action.

Each of these decisions map to the children setup on construction. Both the pick and place behaviors
are modeled as `DfRldsDecider` nodes as described next.

## Robust Logical Dynamical Systems (RLDS)

Both the pick and place behaviors are modeled as Robust Logical Dynamical Systems (RLDS). See the
[RLDS paper on ArXiv](https://arxiv.org/abs/1908.01896). An RLDS is a sequence of behaviors, each
of which has a entry condition on the logical state defining whether it’s runnable. The RLDS
algorithm steps backward from the last behavior in the sequence checking each node’s runnability
condition. It executes the first (most distal) behavior that says it’s runnable. In that sense, the
more distal the behavior, the higher the priority. The sequence is known as its *priority sequence*.

See `class DfRldsDecider` for the RLDS implementation as a decider node. The `decide()` method
implements the reverse sweep through the RLDS sequence. See also Nathan Ratliff’s [Isaac Cortex
GTC22 talk](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s42693/) for additional
details on RLDSs.

### The pick RLDS decider

The pick RLDS decider has three behaviors in its priority sequence.

```python
def make_pick_rlds():
    rlds = DfRldsDecider()
    ...
    rlds.append_rlds_node("reach_to_block", reach_to_block_rd)
    rlds.append_rlds_node("pick_block", PickBlockRd())
    rlds.append_rlds_node("open_gripper", open_gripper_rd)  # Always open the gripper if it's not.

    return rlds
```

It’s most intuitive to read these nodes in reverse since that’s the order they’re processed by the
RLDS algorithm. In order of highest priority to lowest priority (reverse order) we have:

1. **Open gripper:** If the gripper isn’t open, the highest priority action is to open the gripper.
2. **Pick block:** To get here, the gripper must be open. If the block is between the fingers, pick it.
3. **Reach to block:** To get here, the gripper is open but not at the block yet. Reach toward the
   block.

Each node in the sequence executes an action designed to drive the system toward satisfying the
runnable condition of the next behavior in the sequence.

The decision of which block to pick is handled by the `reach_to_block_rd` node. It’s constructed
in the section of code hidden by the `...` above:

```python
open_gripper_rd = OpenGripperRd(dist_thresh_for_open=0.15)
reach_to_block_rd = ReachToBlockRd()
choose_block = ChooseNextBlock()
approach_grasp = DfApproachGrasp()

reach_to_block_rd.link_to("choose_block", choose_block)
choose_block.link_to("approach_grasp", approach_grasp)
```

It first chooses the block in a `choose_block` node, then passes the chosen block as parameters down to
`approach_grasp`. `ChooseNextBlock` itself has two possible children
`ChooseNextBlockForTowerBuildUp` and `ChooseNextBlockForTowerTeardown`, and the decision is made
based on whether the stack is currently in the right order.

### The place RLDS decider

The place RLDS decider is similar in nature to the pick decider:

```python
def make_place_rlds():
    rlds = DfRldsDecider()
    rlds.append_rlds_node("reach_to_placement", ReachToPlacementRd())
    rlds.append_rlds_node("place_block", PlaceBlockRd())
    return rlds
```

Again, `ReachToPlacementRd` itself decides where to place base on the current logical state of the
tower.

### Pick and place atomic actions

The `pick_block` node is itself an state machine implemented by the `PickBlockRd` node. Its
state sequence is built on construction:

```python
class PickBlockRd(DfStateMachineDecider, DfRldsNode):
    def __init__(self):
        # This behavior uses the locking feature of the decision framework to run a state machine
        # sequence as an atomic unit.
        super().__init__(
            DfStateSequence(
                [
                    DfSetLockState(set_locked_to=True, decider=self),
                    DfTimedDeciderState(DfCloseGripper(), activity_duration=0.5),
                    DfTimedDeciderState(Lift(0.1), activity_duration=0.25),
                    DfWriteContextState(lambda ctx: ctx.mark_block_in_gripper()),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )

    ...
```

The state machine shouldn’t be interrupted to ensure successful grasps, so we make it atomic,
by locking the decider network in the beginning and unlocking it in the end.
Locking the decider network ensures that the decision path from the root to this node remain the
same until it’s unlocked. If we didn’t do that, the higher-level dispatch could preempt this node
half way through and prevent a successful pick. This model is the reverse of most other frameworks
to promote reactivity. Everything is preemptable unless it’s specifically locked.

The full sequence for a pick is

1. Lock the decider network so the pick behavior is atomic.
2. Close the gripper for .5 seconds.
3. Lift for .25 seconds.
4. Mark the block as being in the gripper.
5. Unlock the decider network.

Much of this behavior is simply a timed sequence. If for any reason it’s unsuccessful, the decider
network will be reactive to that at a higher level and recover. So we can model this behavior as
simply a blind behavior that executes at the right time. It records its belief (that the block is in
the gripper), but the context will continue to monitor whether that’s true.

The `PlaceBlockRd` is another atomic sequential state machine similar to the pick state machine:

```python
class PlaceBlockRd(DfStateMachineDecider, DfRldsNode):
    def __init__(self):
        # This behavior uses the locking feature of the decision framework to run a state machine
        # sequence as an atomic unit.
        super().__init__(
            DfStateSequence(
                [
                    DfSetLockState(set_locked_to=True, decider=self),
                    DfTimedDeciderState(DfOpenGripper(), activity_duration=0.5),
                    DfTimedDeciderState(Lift(0.15), activity_duration=0.35),
                    DfWriteContextState(lambda ctx: ctx.clear_gripper()),
                    DfWriteContextState(set_top_block_aligned),
                    DfTimedDeciderState(DfCloseGripper(), activity_duration=0.25),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )
```

This state sequence includes opening, lifting, writing logical state, and closing the gripper. Note
that `set_top_block_aligned` marks an `is_aligned` flag in the block. In this demo we don’t do
anything with that flag, but in a real-world system implementation, we would likely have two
behaviors, a place behavior, then an align behavior. Placement might be inaccurate, so on initial
placement the `is_aligned` flag is `False`. Then the decider would see that and run a pinch
alignment behavior which results in `is_aligned` being set to `True`. This a good example of
logical state that’s unobservable. Perception modules might not be accurate enough to fully detect
whether the block is misaligned, but pinching the block we know will align it to higher precision.
So we can run that behavior, and simply mark that it’s been run.

In this example, keep the `is_aligned` logical state in there, but for brevity we don’t implement
the pinch alignment behavior.

## Logical state context

All of these behaviors are supported by the logical state information extracted by the
`BuildTowerContext`. The easiest way to understand what information the context extracts as
logical state is to look at the collection of logical state monitors it uses:

```python
class BuildTowerContext(DfContext):
    ...

    def __init__(self, robot, tower_position):
        ...
        self.monitors = [
            BuildTowerContext.monitor_perception,
            BuildTowerContext.monitor_block_tower,
            BuildTowerContext.monitor_gripper_has_block,
            BuildTowerContext.monitor_suppression_requirements,
            BuildTowerContext.monitor_diagnostics,
        ]
```

In order, the set of monitors are:

1. **Monitor perception:** Periodically sync measured block transforms to the belief blocks. Syncing
   is suppressed when the block is in the end-effector and the measured transform is within a radius
   of the belief. Since the block is moving, we expect the belief to be more accurate during this
   period than perception, but it’s still reactive to the block falling from the gripper.
2. **Monitor block tower:** Based on the positions of the blocks, it infers what the current state of
   the block tower is and populates the `block_tower` data structure appropriately. Decider nodes
   can query the state of the block tower using that data structure and it will always reflect the
   current state.
3. **Monitor gripper has block:** Monitors whether there’s a block in the gripper. There’s an
   interaction between this monitor and the pick behavior. The pick behavior will seed the belief
   that there’s a block in the gripper since that was the intent, and this monitor simply verifies
   that it’s true.
4. **Monitor suppression requirements:** Generally, collision avoidance is active between the robot
   and the blocks (especially important to avoid knocking the tower down). But the robot needs to
   interact with the blocks when picking and placing. This monitor automatically suppresses
   collisions when interaction with blocks is expected.
5. **Monitor diagnostics:** Prints some information about the logical state at a readable
   (throttled) rate.

The information collected by the logical state monitors is everything needed for the decider network
to make robust reactive decisions.

---

# Walkthrough: UR10 Bin Stacking

This tutorial walks through a complete bin stacking application using the UR10 robot. In this
example, we use a pre-designed USD environment containing a conveyor belt, a pallet where the bins
should be stacked, and a UR10 robot with a suction gripper. Our application doesn’t add to the scene
(aside from invisible collision obstacles), and instead controls the existing USD elements in the
pre-designed USD environment.

In all command line examples, we use the abbreviation `isaac_python` for the Isaac Sim python
script (`<isaac_sim_root>/python.sh` on Linux and `<isaac_sim_root>\python.bat` on Windows).
The command lines are written relative to the working directory
`standalone_examples/api/isaacsim.cortex.framework`.

Run the following demo

```python
isaac_python demo_ur10_conveyor_main.py
```

Press play once Isaac Sim has started up. You’ll see the bin stacking demo running.

This tutorial will step through the demo.

The setting is a UR10 robot with a suction gripper moving bins from a conveyor to a pallet. Bins
need to be stacked upside down, so any bin that that comes right-side up is flipped at a flip
station before stacking.

This demo uses a shallow decider network with a top-level dispatch node choosing among multiple
sequential state machines.

The individual behaviors demonstrate a common RMPflow programming technique where obstacle regions
are automatically toggled on and off strategically to shape the motion behavior. These obstacle
regions are modeled in the scene USD but are invisible by default. Try toggling their visibility. Go
to `World/Ur10Table/Obstacles` and toggle the visibility of the `FlipStationSphere`,
`NavigationDome`, `NavigationBarrier`, and `NavigationFlipStation` to see them.

The bin placement behavior also leverages the reactivity of RMPflow in conjunction with its approach
direction parameters to create an automatic adjustment behavior to correct on the fly for
misalignment between bins.

## Top-level dispatch

The entry point to the decider network is the `Dispatch` node as we can see in the construction.
`DfNetwork` is the decider network structure; it’s passed the root (the `Dispatch` node) and the
context object that will be available as a member within every decider/state node:

```python
def make_decider_network(robot):
    return DfNetwork(Dispatch(), context=BinStackingContext(robot))
```

The context object gives each node access to the robot’s command API as well as any logical state
extracted by its monitors. The `Dispatch` node’s `decide()` logic is pretty simple given the
logical state.

```python
def make_decider_network(robot):
    return DfNetwork(Dispatch(), context=BinStackingContext(robot))
```

The logical state includes:

* **stack\_complete:** Notes whether all bins are on the pallet.
* **active\_bin:** The bin that’s currently in play. The bin remains active until it’s been placed on
  the stack. Then a new bin is selected from the bins at the end of the conveyor.
* **active\_bin.is\_attached:** Indicates whether the active bin is attached to the end-effector via
  the suction gripper.
* **active\_bin.needs\_flip:** Indicates whether the bin attached to the end-effector is right-side-up
  (needs flip) or up-side-down (doesn’t need flip).

The decision logic becomes simply: If the stack is complete or there’s no active bin, then go home.
Otherwise, if there’s an active bin, pick the bin if it’s not already in the gripper, and flip it if
it needs to be flipped. Then place the bin on the stack. Decider networks make it easy to write this
decision logic in a readable form.

## Sequential state machines

The sequential state machines that implement the pick, flip and place behaviors are each similar in
structure.

```python
class PickBin(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPick(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    CloseSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.3), activity_duration=0.4),
                    DfSetLockState(set_locked_to=False, decider=self),
                ],
            )
        )

class FlipBin(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    LiftAndTurn(),
                    MoveToFlipStation(),
                    DfSetLockState(set_locked_to=True, decider=self),
                    OpenSuctionGripper(),
                    ReleaseFlipStationBin(duration=0.65),
                    DfSetLockState(set_locked_to=False, decider=self),
                ]
            )
        )

class PlaceBin(DfStateMachineDecider):
    def __init__(self):
        super().__init__(
            DfStateSequence(
                [
                    ReachToPlace(),
                    DfWaitState(wait_time=0.5),
                    DfSetLockState(set_locked_to=True, decider=self),
                    OpenSuctionGripper(),
                    DfTimedDeciderState(DfLift(0.1), activity_duration=0.25),
                    DfWriteContextState(lambda ctx: ctx.mark_active_bin_as_complete()),
                    DfSetLockState(set_locked_to=False, decider=self),
                ],
            )
        )
```

One feature to note is the use of locking and unlocking the decider network. Decider networks are
reactive by nature, so atomic state machine behaviors that shouldn’t be preempted need to be
explicitly locked. The sequential state machines make use of `DfSetLockState` to lock and unlock
the decider network. Additionally, `PlaceBin` uses `DfWriteContextState` to call a context
function which marks the active bin as complete once it’s performed the placement procedure.

## Navigation obstacle monitors

The underlying motion generator for the `MotionCommander` is `RMPflow` which will automatically
avoid registered obstacles. However, there are times where those obstacles need to be turned off to
enable interaction. E.g. en route we want to avoid a manipulable object as obstacle, but once we’re
there we should grab it. The `ObstacleMonitor` and `ObstacleMonitorContext` classes of
`isaacsim.cortex.framework/isaacsim/cortex/framework/obstacle_monitor_context.py` facilitate developing obstacle
monitors which automatically toggle obstacles on and off based on programmed conditions.

Take a look at the `ObstacleMonitor` implementation in the file listed above. On construction it
takes a set of obstacles which it will monitor as well as the context object, and the API requires
deriving classes to implement `is_obstacle_required()` to define when the obstacles should be
enabled and disabled based on information accessible from the context object. The method has access
to the context as `self.context` similar to the decider / state objects. The API also supplies
`activate_autotoggle()` and `deactivate_autotoggle()` to activate and deactivate the monitor.
When active, it’ll automatically enable and disable the obstacles based on the truth value of
`is_obstacle_required()`. When deactivated, the obstacles will be disabled and remain disabled
until the monitor is reactivated.

`ObstacleMonitorContext` is a convenient base class which adds a `monitor_obstacles` logical
state monitor automatically, so deriving classes only need to add the obstacle monitor objects using
`add_obstacle_monitors()`.

The `BinStackingContext` object derives from `ObstacleMonitorContext` and adds two obstacle
monitors which it uses to shape both the navigation behavior moving between the pallet and the
conveyor and the navigation around the bin flip station while flipping the bin. They’re constructed
and added in the `BinStackingContext` constructor:

```python
class BinStackingContext(ObstacleMonitorContext):
    def __init__(self, robot):
        super().__init__()

        ...

        self.flip_station_obs_monitor = FlipStationObstacleMonitor(self)
        self.navigation_obs_monitor = NavigationObstacleMonitor(self)
        self.add_obstacle_monitors([self.flip_station_obs_monitor, self.navigation_obs_monitor])
```

Both monitors have simple logic:

```python
class BinStackingContext(ObstacleMonitorContext):
    def __init__(self, robot):
        super().__init__()

        ...

        self.flip_station_obs_monitor = FlipStationObstacleMonitor(self)
        self.navigation_obs_monitor = NavigationObstacleMonitor(self)
        self.add_obstacle_monitors([self.flip_station_obs_monitor, self.navigation_obs_monitor])
```

The `FlipStationObstacleMonitor` monitors the `flip_station_sphere` which is a spherical
obstacle around the flip station. When active, it’ll enable the obstacle until the end-effector is
descending along the approach direction of its motion command toward the pose target. The monitor is
used to avoid the flip station and bin (resting on the station) after releasing it from the bottom
and moving to pick it from the top. It’s activated on entry to the `ReachToPick` class (used any
time the bin needs to be picked, independent of whether the bin is on the flip station) and
deactivated on exit.

```python
class BinStackingContext(ObstacleMonitorContext):
    def __init__(self, robot):
        super().__init__()

        ...

        self.flip_station_obs_monitor = FlipStationObstacleMonitor(self)
        self.navigation_obs_monitor = NavigationObstacleMonitor(self)
        self.add_obstacle_monitors([self.flip_station_obs_monitor, self.navigation_obs_monitor])
```

Similarly, the `NavigationObstacleMonitor` monitors a collection of obstacles which shape the
navigation behavior between the pallet and conveyor (both directions) to avoid the robot base and
current bin stack in transit. They’re needed while moving from one to the other, but not needed once
the arm reaches the region of its destination (either the pallet or conveyor).

The `MoveWithNavObs` state object extends the `Move` state with entry and exit conditions that
automatically toggle the navigation obstacle:

```python
class MoveWithNavObs(Move):
    def enter(self):
        super().enter()
        self.context.navigation_obs_monitor.activate_autotoggle()

    def exit(self):
        super().exit()
        self.context.navigation_obs_monitor.deactivate_autotoggle()
```

This class is the base class for both the `ReachToPick` state and the `ReachToPlace` state used
by `PickBin` and `PlaceBin` listed above.

## Robustness reactivity on placement

The bin attachment to the end-effector and the stacking alignment of the bins are both physically
simulated. Just blindly grasping and moving the bin to a target without adjusting for errors will
result in slightly misaligned bins which don’t rest against each other correctly. Since we’re using
a reactive motion generator (RMPflow), implementing reactive adjustments is straightforward. In
`ReachToPlace` the adjustments are made every cycle in the `step()` method:

```python
class ReachToPlace(MoveWithNavObs):
    ...

    def step(self):
        if self.bin_under is not None:
            bin_under_p, _ = self.bin_under.bin_obj.get_world_pose()
            bin_grasped_p, _ = self.context.active_bin.bin_obj.get_world_pose()
            xy_err = bin_under_p[:2] - bin_grasped_p[:2]
            if np.linalg.norm(xy_err) < 0.02:
                self.target_p[:2] += 0.1 * (bin_under_p[:2] - bin_grasped_p[:2])

        target_pose = PosePq(self.target_p, math_util.matrix_to_quat(self.target_R))

        approach_params = ApproachParams(direction=0.15 * np.array([0.0, 0.0, -1.0]), std_dev=0.005)
        posture_config = self.context.robot.default_config
        self.update_command(
            MotionCommand(target_pose=target_pose, approach_params=approach_params, posture_config=posture_config)
        )
```

If we’re placing a bin on top of another bin (`bin_under`) this code adjusts the end-effector
target based on the xy position alignment error between the bin in the gripper and the bin under
it. (The orientational alignment generally is already sufficient for successful placement.)

Additionally, RMPflow is configured to take reactive state feedback from the simulator, and we use
tight approach parameters for reaching the target (`std_dev=0.005`) so it needs to follow a narrow
funnel on approach. If the bin is misaligned on first approach, the bin physics will shove the
end-effector out of that funnel and it’ll attempt the approach again.

In combination, this gets the robot to reactively adjust the positioning of the bin and retry the
approach (repeatedly if needed) until it gets it right. Often the adjustment process is sufficient,
but periodically it needs to retry the approach. Usually a single retry suffices. This is a subtle
behavior and the code is concise, but it’s the difference between approximately 85% successful bin
placement and 100% success.

## Logical state context

The `BinStackingContext` object additionally monitors all logical state needed to support the
above behaviors. These monitors are set up in the constructor and the logical state is
reset/initialized in the `reset()` method:

```python
class BinStackingContext(ObstacleMonitorContext):
    def __init__(self, robot):
        super().__init__()
        ...

        self.add_monitors(
            [
                BinStackingContext.monitor_bins,
                BinStackingContext.monitor_active_bin,
                BinStackingContext.monitor_active_bin_grasp_T,
                BinStackingContext.monitor_active_bin_grasp_reached,
                self.diagnostics_monitor.monitor,
            ]
        )

        def reset(self):
            super().reset()

            # Find the collection of bins in the world scene.
            self.bins = []
            i = 0
            while True:
                name = "bin_{}".format(i)
                bin_obj = self.world.scene.get_object(name)
                if bin_obj is None:
                    break
                self.bins.append(BinState(bin_obj))
                i += 1

            self.active_bin = None
            self.stacked_bins.clear()
```

These monitors perform the following:

1. **Monitor bins:** If there’s no active bin, it checks whether there’s a bin at the end of the
   conveyor and activates it if so.
2. **Monitor the active bin:** Deactivates a bin if it’s dropped on the floor.
3. **Monitor the grasp transform of the active bin:** Monitors the best grasp for the current active bin.
4. **Monitor whether the active bin is reached:** Sets the `active_bin.{is_grasp_reached,is_attached}`
   flags based on the proximity between the end-effector and desired grasp transform, and whether
   the suction gripper is “closed”.
5. **Monitor diagnostics:** Prints some information about the logical state at a readable
   (throttled) rate.

---

# Building Cortex Based Extensions

This tutorial covers the use of Cortex in a custom extension running directly on Isaac Sim App instead of the Python SimulationApp. For this we use the same behaviors from [Walkthrough: Franka Block Stacking](Digital_Twin.md) and [Walkthrough: UR10 Bin Stacking](Digital_Twin.md). To use Cortex, similar to [Hello Robot](Python_Scripting_and_Tutorials.md), but we create a modified version of the Base Sample that replaces the Core World with a Cortex World:

```python
import gc
from abc import abstractmethod

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.tasks.base_task import BaseTask
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async
from isaacsim.cortex.framework.cortex_world import CortexWorld
from isaacsim.examples.interactive import base_sample

class CortexBase(base_sample.BaseSample):
    async def load_world_async(self):
        """
        Function called when clicking load buttton.
        The difference between this class and Base Sample is that we initialize a CortexWorld specialization.
        """
        if CortexWorld.instance() is None:
            await create_new_stage_async()
            self._world = CortexWorld(**self._world_settings)
            await self._world.initialize_simulation_context_async()
            self.setup_scene()
        else:
            self._world = CortexWorld.instance()
        self._current_tasks = self._world.get_current_tasks()
        await self._world.reset_async()
        await self._world.pause_async()
        await self.setup_post_load()
        if len(self._current_tasks) > 0:
            self._world.add_physics_callback("tasks_step", self._world.step_async)
        return
```

Now, we need to define the world Task, to define how the world behaves, and the Robot Cortex task. That code is equivalent to the standalone examples, except that the functions to step, start and reset the simulation are moved on the callbacks for the task step, and reset callbacks.

## Franka Cortex Examples

The UI is defined in the `exts/isaacsim.examples.interactive/isaacsim/examples/interactive/franka_cortex/franka_cortex_extension.py`, This sample shows how to load many different decider networks for Franka.

First activate **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` tab.
To load the sample navigate to Robotics Examples > Cortex > Franka Cortex Examples.

First, select the behavior you want from the drop-down, then click on LOAD. To begin the decider network, click on START.

On the Diagnostic monitor, you can check the decision stack. Due to the different nature of the tasks, the task diagnostics is showed as a diagnostic message, containing the important information for each task.

Note

Pressing **STOP**, then **PLAY** in this workflow might not reset the world properly. Use
the **RESET** button instead.

### Hot-Swapping Behaviors

Cortex allows you to select different behavior policies to run on your robot. In this example you can select which policy is running on the robot, even while it’s executing the previous policy. It will change the behavior to conform to the new policy. To do so, choose a new behavior in the drop-down.

## UR10 Palletizing Example

The UI is defined in the `exts/isaacsim.examples.interactive/isaacsim/examples/interactive/ur10_palletizing/ur10_palletizing_extension.py`.

To load the sample navigate to Robotics Examples > Cortex > UR10 Palletizing.

Click on LOAD to load all the assets and setup the scene. Then, Click on START PALLETIZING to begin the task.

On the Diagnostics section, you can inspect the Cortex Decision stack for the robot, and the flags used by the decision network to move forward to the next steps.

---

# Mapping

NVIDIA Isaac Sim mapping extension supports 2D occupancy map generation for a specified height.

## Occupancy Map Generator

The [Mapping](#ext-isaacsim-asset-generator-occupancy-map) Extension is used to generate a binary map of whether or not an area in the scene is occupied at a given height. It uses physics collision geometry in the [Stage](Glossary.md) to determine if a location is occupied or not.

This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.asset.gen.omap`.

To access this Extension, go to the top menu bar and click **Tools** > **Robotics** >> **Occupancy Map**.

### Conventions

* All geometry must have Collisions Enabled to be detected by the Occupancy Map Generator. Otherwise the geometry will not appear in the final map.
* The Start location of the map cannot be occupied.

Note

If mapping does not work correctly make sure the start location is not occupied. You can view the physics geometry by clicking the Show/Hide (eye icon) in the viewport window and selecting **Show By Type** > **Physics Mesh** > **All**.

### API Documentation

See the [API Documentation](../py/source/extensions/isaacsim.asset.gen.omap/docs/index.html) for usage information.

### User Interface

The user interface is composed of two parts, the configuration window (named *Occupancy Map*) and the *Visualization* window.

#### Occupancy Map window

* **Origin**: An open location inside of the area you wish to map.
* **Lower/Upper Bound**: Areas outside of these bounds will not be mapped. These are maximal bounds, the mapped area may be smaller than these limits.
* **Positioning**:

  > + **CENTER TO SELECTION**: The origin will be moved to the center of a selected prim or prims.
  > + **BOUND SELECTION**: The bounds will updated to incorporate the selected prim or prims.

* **Cell Size**: The number of meters each pixel in the final image represents.
* **Occupancy Map**:

  > + **CALCULATE**: Compute the occupancy map.
  > + **VISUALIZE IMAGE**: Open a new window to preview and save the resulting map as an image.
* **Use PhysX Collision Geometry**: When set to True (default), the current collision approximations are used by the PhysX based Lidar to generate the occupancy map. If set to False, the collision approximations are temporarily removed and the RTX Lidar uses the original triangle meshes to generate the occupancy map.

**Example:**

The following steps show how to create and visualize an occupancy map of a certain scene:

> 1. Create a new Cone shape (**Create > Shape > Cone** menu) and add the physics Collision property to it (right click and **Add > Physics > Collider Preset**, or in the *Property* panel).
> 2. Translate the shape 0.3 meters in the X-axis and orient it 90º in the X-axis Euler angles by modifying its *Transform* in the *Property* panel.
> 3. Click on the **Tools > Robotics > Occupancy Map** menu to open the *Occupancy Map* window docked to the button panel.
> 4. Set the Occupancy Map’s Origin Z-axis value to 0.1 meters to map the area at that height
> 5. Click on **CALCULATE** followed by **VISUALIZE IMAGE**. A Visualization popup will appear as shown in the image in the next subsection.
> 6. Finally, click **Save Image** to save the map to an easily accessible location. You will need it for later steps in this guide!

#### Visualization window

* **Occupied Color**: The color chosen to represent space that is “occupied”.
* **Freespace Color**: The color chosen to represent space that is “free”.
* **Unknown Color**: The color chosen to represent space that is interstitial or “unknown”.
* **Rotate Image**: Rotates the coordinates of the image space. A rotation of \(\text{180}^{\circ}\) will result in a Heightmap orientation that matches that of the original source stage of the occupancy map.
* **Coordinate Type**: Determines the format of the output in the information window. Stage Space coordinates reports values in the space of the stage, while the “ROS Occupancy Map Parameters File” returns the needed parameters for the ROS Occupancy Map.
* **RE-GENERATE IMAGE**: This will regenerate the image and information window if you changed the stage.
* **Save Image**: Opens the file picker interface to save the image.

## Heightmap Importer

### Heightmap Importer

The Heightmap Importer Extension converts a 2D occupancy map into a 3D heightmap terrain.
In this extension black pixels in the occupancy map are considered occupied and white pixels are considered free space.
The generated 3D terrain automatically has a collision mesh applied for all the occupied pixels.

To access this Extension, go to the top menu bar and click **Tools** > **Robotics** > **Heightmap Importer**.

* **Cell Size**: Real-world units represented by a single pixel in the 2D occupancy image. The default unit in Isaac Sim is in meters.
* **Load** : Load the desired occupancy image.
* **Generate**: Button to generate the 3D heightmap terrain.

### Heightmap Usage Example

To run the Example:

1. Save the following image to disk:

2. Go to the top menu bar and click **Tools** > **Robotics** > **Heightmap Importer**.
3. Press the **Load Image** button and open the saved image. A window titled **Visualization** will appear.
4. Press the **Generate Heightmap** button to create geometry corresponding to the input occupancy map in the [Stage](Glossary.md).

---

# Digital Twin Troubleshooting

This page consolidates troubleshooting information for Digital Twin components in Isaac Sim.

## Warehouse Logistics Issues

### Warehouse Creator Issues

* If warehouse components don’t appear after generation, check for errors in the console logs
* For layout issues, ensure the grid dimensions and spacing are properly configured
* If textures appear incorrect, verify your material settings and check GPU compatibility

### Conveyor Belt Issues

* For non-functioning conveyors, ensure the physics settings are correctly applied
* If objects fall through conveyors, adjust collision settings and physics parameters
* Animation speed issues can be resolved by checking the conveyor speed settings

## Cortex Issues

### Decider Network Issues

* If decision networks fail to initialize, check that all required extensions are enabled
* For unexpected behavior, review your network configurations and connections
* Debug flows by enabling verbose logging and tracing through decisions step by step

### Asset Loading Issues

* Missing assets can be resolved by checking file paths and ensuring assets are available
* For slow loading of complex assets, consider using simpler versions for testing
* USD file compatibility issues may require updating to the latest USD schema

## Mapping Issues

### Occupancy Map Issues

* If occupancy maps fail to generate, ensure the scene has proper collision geometry
* For inaccurate maps, adjust the resolution and sensor parameters
* Missing areas in the map may indicate occlusion issues or raycast failures