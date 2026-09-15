# Python Scripting and Tutorials

## Concepts

- Python Scripting Concepts
- Core API Overview
- Python Environment

## Snippets

- Scene Setup Snippets
- Util Snippets
- Robot Simulation Snippets

## API Reference

- API Documentation

## Tutorials

- Core API Tutorial Series

---

# Python Scripting Concepts

## Standalone vs Interactive Python

Python scripting in NVIDIA Isaac Sim can be done in two ways: standalone and interactive. Standalone Python scripts are executed from the command line and are used to automate tasks or run simulations. Interactive Python scripts are executed in the Python console and are used to explore the NVIDIA Isaac Sim API and test code snippets. Both types of scripts can be used to create custom extensions, such as new robot controllers or sensors, and to interact with the Omniverse application.

---

# Core API Overview

Important

Isaac Sim 5.0.0 has introduced the [Core Experimental API](../py/docs/overview/experimental.html): a rewritten implementation of the current Core API
designed to be more robust, flexible, and powerful, yet still maintain the core utilities and wrapper concepts.

Going forward, it will become the base API used in all Isaac Sim source code.
The current Core API will be deprecated and removed in future releases.

Therefore, **we strongly encourage early adoption and use of the Core Experimental API**.

## Core API is a Wrapper

Isaac Sim Core API are wrappers for raw USD and physics engine APIs, tailored to suit robotics applications. Here is adding a cube and apply physics properties to it using the raw USD

```python
import omni
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()

# Setting up Physics Scene
gravity = 9.8
scene = UsdPhysics.Scene.Define(stage, "/World/physics")
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(gravity)
PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physics"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physics")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")

# Setting up Ground Plane
PhysicsSchemaTools.addGroundPlane(stage, "/World/groundPlane", "Z", 15, Gf.Vec3f(0, 0, 0), Gf.Vec3f(0.7))

# Adding a Cube
path = "/World/Cube"
cubeGeom = UsdGeom.Cube.Define(stage, path)
cubePrim = stage.GetPrimAtPath(path)
size = 0.5
offset = Gf.Vec3f(0.5, 0.2, 1.0)
cubeGeom.CreateSizeAttr(size)
cubeGeom.AddTranslateOp().Set(offset)

# Attach Rigid Body and Collision Preset
rigid_api = UsdPhysics.RigidBodyAPI.Apply(cubePrim)
rigid_api.CreateRigidBodyEnabledAttr(True)
UsdPhysics.CollisionAPI.Apply(cubePrim)
```

Here is adding a cube with physics and material properties to stage using Core API.

```python
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid

DynamicCuboid(
    prim_path="/new_cube_2",
    name="cube_1",
    position=np.array([0, 0, 1.0]),
    scale=np.array([0.6, 0.5, 0.2]),
    size=1.0,
    color=np.array([255, 0, 0]),
)
```

## Application vs Simulation vs World vs Scene vs Stage

Everything in USD is a primitive (prim) with attributes.

A **Simulation** (the sim) moves these prims forward through time by literally changing these attributes programmatically.

The **Application** is the thing that manages the gross aspects of the simulation (how things are rendered, for example) and how the user interacts with it. If there is a GUI for the sim, it is a part of the application.

A **Stage** is a USD concept, and defines the logical and relational context for prims in the simulation. If a mug prim is on a table prim then that relationship is expressed by the relative locations of those prims on the stage, and the specific attributes each has. In this way, the stage provides context for the application: prims cannot exist without a stage and so an application concerned with prims requires a stage to function.

Similarly, the **World** is what provides context to the simulation, defining which prims are relevant to the ongoing flow of time, the **scene**, and managing the aspects of the simulation that are most important to the user.

For example, imagine you are going to see a play at a theater. The theater is like the **application**, your gateway to the play, while the **simulation** is the play itself, defined by a program. You take your seat and you can see the **stage**, where the play will take place. When the play starts, the curtain rises and reveals a **scene** composed props and actors that then act out that part of the play. When it’s time to move to the next scene, the curtain falls, the scene is reset, and then the curtain rises again, revealing the next part of the play. The stage crew and all the mechanical devices behind the scene that manages the curtain and the props is the **world** of the play.

---

# Python Environment

This document will cover:

* Details about how running standalone Python scripts works.
* A short list of interesting/useful standalone Python scripts to try.
* Resources to develop Python scripts for NVIDIA Isaac Sim, such as VSCode and Jupyter Notebook support.

## Details: How `python.sh` works

Note

* On Windows use python.bat instead of python.sh
* The details of how python.sh works below are similar to how python.bat works

This script first defines the location of the apps folder so the contained .kit files can be located at runtime.

```python
# Get path to the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# The apps directory is relative to where the script lives
export EXP_PATH=$SCRIPT_DIR/apps
```

Then we source the NVIDIA Isaac Sim Python environment so all extension interfaces can be loaded correctly.

```python
source ${SCRIPT_DIR}/setup_python_env.sh
```

The setup\_python\_env.sh script update/defined the following environment variables:

* ISAAC\_PATH: Path to the main isaac folder
* PYTHONPATH: Paths to each extensions Python interfaces
* LD\_LIBRARY\_PATH: Paths to binary interfaces required to find symbols at runtime
* CARB\_APP\_PATH: path to the core Omniverse kit executable

Finally, we execute the Python interpreter that is packaged with Omniverse:

```python
python_exe=${PYTHONEXE:-"${SCRIPT_DIR}/kit/python/bin/python3"}
...
$python_exe $@
```

## SimulationApp

The [SimulationApp Class](../py/source/extensions/isaacsim.simulation_app/docs/index.html) provides convenience functions to manage the lifetime of a NVIDIA Isaac Sim application.

### Usage Example:

The following code provides a usage example for how SimulationApp can be used to create an app, step forward in time and then exit.

Note

Any Omniverse level imports **must** occur after the class is instantiated.
Because APIs are provided by the extension/runtime plugin system, it must be loaded before they will be available to import.

Important

When running headless:

* Set `"headless": True` in the config when initializing `SimulationApp`
* Any calls that create/open a matplotlib window need to be commented out

```python
from isaacsim import SimulationApp

# Simple example showing how to start and stop the helper
simulation_app = SimulationApp({"headless": True})

### Perform any omniverse imports here after the helper loads ###

simulation_app.update()  # Render a single frame
simulation_app.close()  # Cleanup application
```

### Details: How `SimulationApp` works

Although `SimulationApp` further configures the application and exposes APIs, there are some fundamental steps in any Omniverse Kit-based implementation that must be executed.

The first is to get the carbonite framework.
Here the environment variables (e.g.: `CARB_APP_PATH`, `ISAAC_PATH` and `EXP_PATH`) were defined when running the python.sh script.

```python
import carb
import omni.kit.app

framework = carb.get_framework()
framework.load_plugins(
    loaded_file_wildcards=["omni.kit.app.plugin"],
    search_paths=[os.path.abspath(f'{os.environ["CARB_APP_PATH"]}/kernel/plugins')],
)
```

After loading the framework, it is possible to configure the start arguments before loading the application. For example:

```python
import carb
import omni.kit.app

framework = carb.get_framework()
framework.load_plugins(
    loaded_file_wildcards=["omni.kit.app.plugin"],
    search_paths=[os.path.abspath(f'{os.environ["CARB_APP_PATH"]}/kernel/plugins')],
)
```

And then start the application.

```python
app = omni.kit.app.get_app()
app.startup("Isaac-Sim", os.environ["CARB_APP_PATH"], sys.argv)
```

Shutting down a running application is done by calling `shutdown` and then unloading the framework:

```python
app = omni.kit.app.get_app()
app.startup("Isaac-Sim", os.environ["CARB_APP_PATH"], sys.argv)
```

### Enabling additional extensions

There are two methods for adding additional extensions:

1. Under `[dependencies]` section in an experience file (e.g.: `apps/isaacsim.exp.base.python.kit`):

   > ```python
   > # [dependencies]
   > # # Enable the layers and stage windows in the UI
   > # "omni.kit.window.stage" = {}
   > # "omni.kit.widget.layers" = {}
   > ```
2. From Python code:

   ```python
   from isaacsim import SimulationApp

   # Start the application
   simulation_app = SimulationApp({"headless": False})

   # Get the utility to enable extensions
   from isaacsim.core.utils.extensions import enable_extension

   # Enable the layers and stage windows in the UI
   enable_extension("omni.kit.widget.stage")
   enable_extension("omni.kit.widget.layers")

   simulation_app.update()
   ```

## Standalone Example Scripts

### Time Stepping

This sample shows how to start an Omniverse Kit Python app and then create callbacks which get called each rendering frame and each physics timestep. It also shows the different ways to step physics and rendering.

The sample can be executed by running the following:

```python
./python.sh standalone_examples/api/isaacsim.core.api/time_stepping.py
```

### Load USD Stage

This sample demonstrates how to load a USD stage and start simulating it.

The sample can be executed by running the following, specify `usd_path` to a location on your nucleus server:

```python
./python.sh standalone_examples/api/isaacsim.simulation_app/load_stage.py --usd_path /Isaac/Environments/Simple_Room/simple_room.usd
```

### URDF Import

This sample demonstrates how to use the URDF Python API, configure its physics and then simulate it for a fixed number of frames.

The sample can be executed by running the following:

```python
./python.sh standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py
```

### Change Resolution

This sample demonstrates how to change the resolution of the viewport at runtime.

The sample can be executed by running the following:

```python
./python.sh standalone_examples/api/isaacsim.simulation_app/change_resolution.py
```

### Convert Assets to USD

This sample demonstrates how to batch convert OBJ/STL/FBX assets to USD.

To execute it with sample data, run the following:

```python
./python.sh standalone_examples/api/omni.kit.asset_converter/asset_usd_converter.py --folders standalone_examples/data/cube standalone_examples/data/torus
```

The input folders containing OBJ/STL/FBX assets are specified as argument
and it will output in terminal the path to converted USD files.

```python
Converting folder standalone_examples/data/cube...
---Added standalone_examples/data/cube_converted/cube_fbx.usd

Converting folder standalone_examples/data/torus...
---Added standalone_examples/data/torus_converted/torus_stl.usd
```

This sample leverages Python APIs from the [Asset Importer](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html "(in Omniverse Extensions)") extension.

The details about the import options can be found [here](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-importer.html "(in Omniverse Extensions)").

### Livestream

This sample demonstrates how to enable livestreaming when running in native Python.

See [Isaac Sim WebRTC Streaming Client](Installation.md) for more information on running the client.

```python
./python.sh standalone_examples/api/isaacsim.simulation_app/livestream.py
```

Note

* Running livestream.py will not have all of the default Isaac Sim extensions enabled. See [enabling additional extensions](#isaac-sim-python-additional-extensions) for more information.

---

# Scene Setup Snippets

## Objects Creation and Manipulation

Note

The following scripts should only be run on the default new stage and only once. You can try these by creating a new stage via File > New and running from Window > Script Editor

### Rigid Object Creation

The following snippet adds a dynamic cube with given properties and a ground plane to the scene.

```python
import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.physics_context import PhysicsContext

PhysicsContext()
GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
DynamicCuboid(
    prim_path="/World/cube",
    position=np.array([-0.5, -0.2, 1.0]),
    scale=np.array([0.5, 0.5, 0.5]),
    color=np.array([0.2, 0.3, 0.0]),
)
```

### View Objects

View classes in this extension are collections of similar prims. View classes manipulate the underlying objects in a vectorized way.
Most View APIs require the world and the physics simulation to be initialized before they can be used.
This can be achieved by adding the view class to the World’s scene and resetting the world as follows

```python
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim

# View classes are initialized when they are added to the scene and the world is reset
world = World()
cube = DynamicCuboid(prim_path="/World/cube_0")
rigid_prim = RigidPrim(prim_paths_expr="/World/cube_[0-100]")
world.scene.add(rigid_prim)
world.reset()
# rigid_prim is now initialized and can be used
```

which works when running the script via the Isaac Sim Python script. When using Window > Script Editor, to run the snippets you need to use the asynchronous version of `reset` as follows

```python
import asyncio

from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim

async def init():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane(z_position=-1.0)
    cube = DynamicCuboid(prim_path="/World/cube_0")
    rigid_prim = RigidPrim(prim_paths_expr="/World/cube_[0-100]")
    # View classes are internally initialized when they are added to the scene and the world is reset
    world.scene.add(rigid_prim)
    await world.reset_async()
    # rigid_prim is now initialized and can be used

asyncio.ensure_future(init())
```

See [Workflows](Workflows.md) tutorial for more details about various workflows for developing in Isaac Sim.

### Create RigidPrim

The following snippet adds three cubes to the scene and creates a RigidPrim (formerly RigidPrimView) to manipulate the batch.

```python
import asyncio

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim

async def example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane(z_position=-1.0)

    # create rigid cubes
    for i in range(3):
        DynamicCuboid(prim_path=f"/World/cube_{i}")

    # create the view object to batch manipulate the cubes
    rigid_prim = RigidPrim(prim_paths_expr="/World/cube_[0-2]")
    world.scene.add(rigid_prim)
    await world.reset_async()
    # set world poses
    rigid_prim.set_world_poses(positions=np.array([[0, 0, 2], [0, -2, 2], [0, 2, 2]]))

asyncio.ensure_future(example())
```

See the [API Documentation](../py/source/extensions/isaacsim.core.prims/docs/index.html#isaacsim.core.prims.RigidPrim) for all the possible operations supported by `RigidPrim`.

### Create RigidContactView

There are scenarios where you are interested in net contact forces on each body and contact forces between specific bodies. This can be achieved via the RigidContactView object managed by the RigidPrim

```python
import asyncio

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim

async def example():
    if World.instance():
        World.instance().clear_instance()
    world = World()
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()

    # create three rigid cubes sitting on top of three others
    for i in range(3):
        DynamicCuboid(prim_path=f"/World/bottom_box_{i+1}", size=2, color=np.array([0.5, 0, 0]), mass=1.0)
        DynamicCuboid(prim_path=f"/World/top_box_{i+1}", size=2, color=np.array([0, 0, 0.5]), mass=1.0)

    # as before, create RigidContactView to manipulate bottom boxes but this time specify top boxes as filters to the view object
    # this allows receiving contact forces between the bottom boxes and top boxes
    bottom_box = RigidPrim(
        prim_paths_expr="/World/bottom_box_*",
        name="bottom_box",
        positions=np.array([[0, 0, 1.0], [-5.0, 0, 1.0], [5.0, 0, 1.0]]),
        contact_filter_prim_paths_expr=["/World/top_box_*"],
    )
    # create a RigidContactView to manipulate top boxes
    top_box = RigidPrim(
        prim_paths_expr="/World/top_box_*",
        name="top_box",
        positions=np.array([[0.0, 0, 3.0], [-5.0, 0, 3.0], [5.0, 0, 3.0]]),
        track_contact_forces=True,
    )

    world.scene.add(top_box)
    world.scene.add(bottom_box)
    await world.reset_async()

    # net contact forces acting on the bottom boxes
    print(bottom_box.get_net_contact_forces())
    # contact forces between the top and the bottom boxes
    print(bottom_box.get_contact_force_matrix())

asyncio.ensure_future(example())
```

More detailed information about the friction and contact forces can be obtained from the `get_friction_data` and `get_contact_force_data` respectively.
These APIs provide all the contact forces and contact points between pairs of the sensor prims and filter prims. `get_contact_force_data` API provides the contact distances and contact normal vectors as well.

In the example below, we add three boxes to the scene and apply a tangential force of magnitude 10 to each. Then we use the aforementioned APIs to receive all the contact information and sum across all the contact points to find the friction/normal forces between the boxes and the ground plane.

```python
import asyncio

import numpy as np
from isaacsim.core.api.materials.physics_material import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.world import World
from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.stage import create_new_stage_async, update_stage_async

async def contact_force_example():
    g = 10
    await create_new_stage_async()
    if World.instance():
        World.instance().clear_instance()
    world = World()
    world.scene.add_default_ground_plane()
    await world.initialize_simulation_context_async()
    material = PhysicsMaterial(
        prim_path="/World/PhysicsMaterials",
        static_friction=0.5,
        dynamic_friction=0.5,
    )
    # create three rigid cubes sitting on top of three others
    for i in range(3):
        DynamicCuboid(
            prim_path=f"/World/Box_{i+1}", size=2, color=np.array([0, 0, 0.5]), mass=1.0
        ).apply_physics_material(material)

    # Creating RigidPrim with contact relevant keywords allows receiving contact information
    # In the following we indicate that we are interested in receiving up to 30 contact points data between the boxes and the ground plane
    box_view = RigidPrim(
        prim_paths_expr="/World/Box_*",
        positions=np.array([[0, 0, 1.0], [-5.0, 0, 1.0], [5.0, 0, 1.0]]),
        contact_filter_prim_paths_expr=["/World/defaultGroundPlane/GroundPlane/CollisionPlane"],
        max_contact_count=3 * 10,  # we don't expect more than 10 contact points for each box
    )

    world.scene.add(box_view)
    await world.reset_async()

    forces = np.array([[g, 0, 0], [g, 0, 0], [g, 0, 0]])
    box_view.apply_forces(forces)
    await update_stage_async()

    # tangential forces
    friction_forces, friction_points, friction_pair_contacts_count, friction_pair_contacts_start_indices = (
        box_view.get_friction_data(dt=1 / 60)
    )
    # normal forces
    forces, points, normals, distances, pair_contacts_count, pair_contacts_start_indices = (
        box_view.get_contact_force_data(dt=1 / 60)
    )
    # pair_contacts_count, pair_contacts_start_indices are tensors of size num_sensors x num_filters
    # friction_pair_contacts_count, friction_pair_contacts_start_indices are tensors of size num_sensors x num_filters
    # use the following tensors to sum across all the contact points
    force_aggregate = np.zeros((box_view._contact_view.num_shapes, box_view._contact_view.num_filters, 3))
    friction_force_aggregate = np.zeros((box_view._contact_view.num_shapes, box_view._contact_view.num_filters, 3))

    # process contacts for each pair i, j
    for i in range(pair_contacts_count.shape[0]):
        for j in range(pair_contacts_count.shape[1]):
            start_idx = pair_contacts_start_indices[i, j]
            friction_start_idx = friction_pair_contacts_start_indices[i, j]
            count = pair_contacts_count[i, j]
            friction_count = friction_pair_contacts_count[i, j]
            # sum/average across all the contact points for each pair
            pair_forces = forces[start_idx : start_idx + count]
            pair_normals = normals[start_idx : start_idx + count]
            force_aggregate[i, j] = np.sum(pair_forces * pair_normals, axis=0)

            # sum/average across all the friction pairs
            pair_forces = friction_forces[friction_start_idx : friction_start_idx + friction_count]
            friction_force_aggregate[i, j] = np.sum(pair_forces, axis=0)

    print("friction forces: \n", friction_force_aggregate)
    print("contact forces: \n", force_aggregate)
    # get_contact_force_matrix API is equivalent to the summation of the individual contact forces computed above
    print("contact force matrix: \n", box_view.get_contact_force_matrix(dt=1 / 60))
    # get_net_contact_forces API is the summation of the all forces
    # in the current example because all the potential contacts are captured by the choice of our filter prims (/World/defaultGroundPlane/GroundPlane/CollisionPlane)
    # the following is similar to the reduction of the contact force matrix above across the filters
    print("net contact force: \n", box_view.get_net_contact_forces(dt=1 / 60))

asyncio.ensure_future(contact_force_example())
```

See the [API Documentation](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.RigidContactView) for more information about `RigidContactView`.

### Set Mass Properties for a Mesh

The snippet below shows how to set the mass of a physics object. Density can also be specified as an alternative

```python
import omni
from omni.physx.scripts import utils
from pxr import UsdPhysics

stage = omni.usd.get_context().get_stage()
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the prim
cube_prim = stage.GetPrimAtPath(path)
# Make it a rigid body
utils.setRigidBody(cube_prim, "convexHull", False)

mass_api = UsdPhysics.MassAPI.Apply(cube_prim)
mass_api.CreateMassAttr(10)
### Alternatively set the density
mass_api.CreateDensityAttr(1000)
```

### Get Size of a Mesh

The snippet below shows how to get the size of a mesh.

```python
import omni
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cone")
# Get the prim
prim = stage.GetPrimAtPath(path)
# Get the size
bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
bbox_cache.Clear()
prim_bbox = bbox_cache.ComputeWorldBound(prim)
prim_range = prim_bbox.ComputeAlignedRange()
prim_size = prim_range.GetSize()
print(prim_size)
```

### Apply Semantic Data on Entire Stage

The snippet below shows how to programmatically apply semantic data on objects by iterating the entire stage.

```python
import omni.usd
from isaacsim.core.utils.semantics import add_labels

def remove_prefix(name, prefix):
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name

def remove_numerical_suffix(name):
    suffix = name.split("_")[-1]
    if suffix.isnumeric():
        return name[: -len(suffix) - 1]
    return name

def remove_underscores(name):
    return name.replace("_", "")

stage = omni.usd.get_context().get_stage()
for prim in stage.Traverse():
    if prim.GetTypeName() == "Mesh":
        label = str(prim.GetPrimPath()).split("/")[-1]
        label = remove_prefix(label, "SM_")
        label = remove_numerical_suffix(label)
        label = remove_underscores(label)
        add_labels(prim, labels=[label], instance_name="class")
```

### Convert Asset to USD

The below script will convert a non-USD asset like OBJ/STL/FBX to USD. This is meant to be used inside the [Script Editor](Development_Tools.md). For running it as a [Standalone Application](Workflows.md), Check [Python Environment](Python_Scripting_and_Tutorials.md).

```python
import asyncio

import carb
import omni

async def convert_asset_to_usd(input_obj: str, output_usd: str):
    import omni.kit.asset_converter

    def progress_callback(progress, total_steps):
        pass

    converter_context = omni.kit.asset_converter.AssetConverterContext()
    # setup converter and flags
    # converter_context.ignore_material = False
    # converter_context.ignore_animation = False
    # converter_context.ignore_cameras = True
    # converter_context.single_mesh = True
    # converter_context.smooth_normals = True
    # converter_context.preview_surface = False
    # converter_context.support_point_instancer = False
    # converter_context.embed_mdl_in_usd = False
    # converter_context.use_meter_as_world_unit = True
    # converter_context.create_world_as_default_root_prim = False
    instance = omni.kit.asset_converter.get_instance()
    task = instance.create_converter_task(input_obj, output_usd, progress_callback, converter_context)
    success = await task.wait_until_finished()
    if not success:
        carb.log_error(task.get_status(), task.get_detailed_error())
    print("converting done")

asyncio.ensure_future(
    convert_asset_to_usd(
        "</path/to/mesh.obj>",
        "</path/to/mesh.usd>",
    )
)
```

The details about the optional import options in lines 13-23 can be found [here](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html "(in Omniverse Extensions)").

## Physics How-Tos

### Create A Physics Scene

```python
import omni
from pxr import Gf, Sdf, UsdPhysics

stage = omni.usd.get_context().get_stage()
# Add a physics scene prim to stage
scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/World/physicsScene"))
# Set gravity vector
scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
scene.CreateGravityMagnitudeAttr().Set(981.0)
```

The following can be added to set specific settings, in this case use CPU physics and the TGS solver

```python
from pxr import PhysxSchema

PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")
```

Adding a ground plane to a stage can be done via the following code:
It creates a Z up plane with a size of 100 cm at a Z coordinate of -100

```python
from pxr import PhysxSchema

PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/physicsScene"))
physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage, "/World/physicsScene")
physxSceneAPI.CreateEnableCCDAttr(True)
physxSceneAPI.CreateEnableStabilizationAttr(True)
physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
physxSceneAPI.CreateSolverTypeAttr("TGS")
```

### Enable Physics And Collision For a Mesh

The script below assumes there is a physics scene in the stage.

```python
import omni
from omni.physx.scripts import utils

# Create a cube mesh in the stage
stage = omni.usd.get_context().get_stage()
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the prim
cube_prim = stage.GetPrimAtPath(path)
# Enable physics on prim
# If a tighter collision approximation is desired use convexDecomposition instead of convexHull
utils.setRigidBody(cube_prim, "convexHull", False)
```

If a tighter collision approximation is desired use convexDecomposition

```python
import omni
from omni.physx.scripts import utils

# Create a cube mesh in the stage
stage = omni.usd.get_context().get_stage()
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the prim
cube_prim = stage.GetPrimAtPath(path)
# Enable physics on prim
# If a tighter collision approximation is desired use convexDecomposition instead of convexHull
utils.setRigidBody(cube_prim, "convexDecomposition", False)
```

To verify that collision meshes have been successfully enabled, click the “eye” icon > “Show By Type” >
“Physics Mesh” > “All”. This will show the collision meshes as pink outlines on the objects.

### Traverse a stage and assign collision meshes to children

```python
import omni
from omni.physx.scripts import utils
from pxr import Gf, Usd, UsdGeom

stage = omni.usd.get_context().get_stage()

def add_cube(stage, path, size: float = 10, offset: Gf.Vec3d = Gf.Vec3d(0, 0, 0)):
    cubeGeom = UsdGeom.Cube.Define(stage, path)
    cubeGeom.CreateSizeAttr(size)
    cubeGeom.AddTranslateOp().Set(offset)

### The following prims are added for illustrative purposes
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Torus")
# all prims under AddCollision will get collisions assigned
add_cube(stage, "/World/Cube_0", offset=Gf.Vec3d(100, 100, 0))
# create a prim nested under without a parent
add_cube(stage, "/World/Nested/Cube", offset=Gf.Vec3d(100, 0, 100))
###

# Traverse all prims in the stage starting at this path
curr_prim = stage.GetPrimAtPath("/")

for prim in Usd.PrimRange(curr_prim):
    # only process shapes and meshes
    if (
        prim.IsA(UsdGeom.Cylinder)
        or prim.IsA(UsdGeom.Capsule)
        or prim.IsA(UsdGeom.Cone)
        or prim.IsA(UsdGeom.Sphere)
        or prim.IsA(UsdGeom.Cube)
    ):
        # use a ConvexHull for regular prims
        utils.setCollider(prim, approximationShape="convexHull")
    elif prim.IsA(UsdGeom.Mesh):
        # "None" will use the base triangle mesh if available
        # Can also use "convexDecomposition", "convexHull", "boundingSphere", "boundingCube"
        utils.setCollider(prim, approximationShape="None")
    pass
pass
```

### Do Overlap Test

These snippets detect and report when objects overlap with a specified cubic/spherical region.
The following is assumed: the stage contains a physics scene, all objects have collision meshes enabled,
and the play button has been clicked.

The parameters: extent, origin and rotation (or origin and radius) define the cubic/spherical region to check overlap against.
The output of the physX query is the number of objects that overlaps with this cubic/spherical region.

```python
import carb
import omni
import omni.physx
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, UsdGeom, Vt

def report_hit(hit):
    # When a collision is detected, the object color changes to red.
    hitColor = Vt.Vec3fArray([Gf.Vec3f(180.0 / 255.0, 16.0 / 255.0, 0.0)])
    usdGeom = UsdGeom.Mesh.Get(omni.usd.get_context().get_stage(), hit.rigid_body)
    usdGeom.GetDisplayColorAttr().Set(hitColor)
    return True

def check_overlap():
    # Defines a cubic region to check overlap with
    extent = carb.Float3(20.0, 20.0, 20.0)
    origin = carb.Float3(0.0, 0.0, 0.0)
    rotation = carb.Float4(0.0, 0.0, 1.0, 0.0)
    # physX query to detect number of hits for a cubic region
    numHits = get_physx_scene_query_interface().overlap_box(extent, origin, rotation, report_hit, False)
    # physX query to detect number of hits for a spherical region
    # numHits = get_physx_scene_query_interface().overlap_sphere(radius, origin, report_hit, False)
    return numHits > 0
```

### Do Raycast Test

This snippet detects the closest object that intersects with a specified ray.
The following is assumed: the stage contains a physics scene, all objects have collision meshes enabled,
and the play button has been clicked.

The parameters: origin, rayDir and distance define a ray along which a ray hit might be detected.
The output of the query can be used to access the object’s reference, and its distance from the raycast origin.

```python
import carb
import omni
import omni.physx
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, UsdGeom, Vt

def check_raycast():
    # Projects a raycast from 'origin', in the direction of 'rayDir', for a length of 'distance' cm
    # Parameters can be replaced with real-time position and orientation data  (e.g. of a camera)
    origin = carb.Float3(0.0, 0.0, 0.0)
    rayDir = carb.Float3(1.0, 0.0, 0.0)
    distance = 100.0
    # physX query to detect closest hit
    hit = get_physx_scene_query_interface().raycast_closest(origin, rayDir, distance)
    if hit["hit"]:
        # Change object color to yellow and record distance from origin
        usdGeom = UsdGeom.Mesh.Get(omni.usd.get_context().get_stage(), hit["rigidBody"])
        hitColor = Vt.Vec3fArray([Gf.Vec3f(255.0 / 255.0, 255.0 / 255.0, 0.0)])
        usdGeom.GetDisplayColorAttr().Set(hitColor)
        distance = hit["distance"]
        return usdGeom.GetPath().pathString, distance
    return None, 10000.0

print(check_raycast())
```

## USD How-Tos

### Creating, Modifying, Assigning Materials

```python
import omni
from pxr import Gf, Sdf, UsdShade

mtl_created_list = []
# Create a new material using OmniGlass.mdl
omni.kit.commands.execute(
    "CreateAndBindMdlMaterialFromLibrary",
    mdl_name="OmniGlass.mdl",
    mtl_name="OmniGlass",
    mtl_created_list=mtl_created_list,
)
# Get reference to created material
stage = omni.usd.get_context().get_stage()
mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
# Set material inputs, these can be determined by looking at the .mdl file
# or by selecting the Shader attached to the Material in the stage window and looking at the details panel
omni.usd.create_material_input(mtl_prim, "glass_color", Gf.Vec3f(0, 1, 0), Sdf.ValueTypeNames.Color3f)
omni.usd.create_material_input(mtl_prim, "glass_ior", 1.0, Sdf.ValueTypeNames.Float)
# Create a prim to apply the material to
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the path to the prim
cube_prim = stage.GetPrimAtPath(path)
# Bind the material to the prim
cube_mat_shade = UsdShade.Material(mtl_prim)
UsdShade.MaterialBindingAPI(cube_prim).Bind(cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)
```

Assigning a texture to a material that supports it can be done as follows:

```python
import carb
import omni
from pxr import Sdf, UsdShade

# Change the server to your Nucleus install, default is set to localhost in omni.isaac.sim.base.kit
default_server = carb.settings.get_settings().get("/persistent/isaac/asset_root/default")
mtl_created_list = []
# Create a new material using OmniPBR.mdl
omni.kit.commands.execute(
    "CreateAndBindMdlMaterialFromLibrary",
    mdl_name="OmniPBR.mdl",
    mtl_name="OmniPBR",
    mtl_created_list=mtl_created_list,
)
stage = omni.usd.get_context().get_stage()
mtl_prim = stage.GetPrimAtPath(mtl_created_list[0])
# Set material inputs, these can be determined by looking at the .mdl file
# or by selecting the Shader attached to the Material in the stage window and looking at the details panel
omni.usd.create_material_input(
    mtl_prim,
    "diffuse_texture",
    default_server + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png",
    Sdf.ValueTypeNames.Asset,
)
# Create a prim to apply the material to
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the path to the prim
cube_prim = stage.GetPrimAtPath(path)
# Bind the material to the prim
cube_mat_shade = UsdShade.Material(mtl_prim)
UsdShade.MaterialBindingAPI(cube_prim).Bind(cube_mat_shade, UsdShade.Tokens.strongerThanDescendants)
```

### Adding a transform matrix to a prim

```python
import omni
from pxr import Gf, UsdGeom

# Create a cube mesh in the stage
stage = omni.usd.get_context().get_stage()
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Get the prim and set its transform matrix
cube_prim = stage.GetPrimAtPath("/World/Cube")
xform = UsdGeom.Xformable(cube_prim)
transform = xform.AddTransformOp()
mat = Gf.Matrix4d()
mat.SetTranslateOnly(Gf.Vec3d(0.10, 1, 1.5))
mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 1, 0), 290))
transform.Set(mat)
```

### Align two USD prims

```python
import omni
from pxr import Gf, UsdGeom

stage = omni.usd.get_context().get_stage()
# Create a cube
result, path_a = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
prim_a = stage.GetPrimAtPath(path_a)
# change the cube pose
xform = UsdGeom.Xformable(prim_a)
transform = xform.AddTransformOp()
mat = Gf.Matrix4d()
mat.SetTranslateOnly(Gf.Vec3d(0.10, 1, 1.5))
mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 1, 0), 290))
transform.Set(mat)
# Create a second cube
result, path_b = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
prim_b = stage.GetPrimAtPath(path_b)
# Get the transform of the first cube
pose = omni.usd.utils.get_world_transform_matrix(prim_a)
# Clear the transform on the second cube
xform = UsdGeom.Xformable(prim_b)
xform.ClearXformOpOrder()
# Set the pose of prim_b to that of prim_b
xform_op = xform.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionDouble, "")
xform_op.Set(pose)
```

### Get World Transform At Current Timestamp For Selected Prims

```python
import omni
from pxr import Gf, UsdGeom

usd_context = omni.usd.get_context()
stage = usd_context.get_stage()

#### For testing purposes we create and select a prim
#### This section can be removed if you already have a prim selected
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
cube_prim = stage.GetPrimAtPath(path)
# change the cube pose
xform = UsdGeom.Xformable(cube_prim)
transform = xform.AddTransformOp()
mat = Gf.Matrix4d()
mat.SetTranslateOnly(Gf.Vec3d(0.10, 1, 1.5))
mat.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0, 1, 0), 290))
transform.Set(mat)
omni.usd.get_context().get_selection().set_prim_path_selected(path, True, True, True, False)
####

# Get list of selected primitives
selected_prims = usd_context.get_selection().get_selected_prim_paths()
# Get the current timecode
timeline = omni.timeline.get_timeline_interface()
timecode = timeline.get_current_time() * timeline.get_time_codes_per_seconds()
# Loop through all prims and print their transforms
for s in selected_prims:
    curr_prim = stage.GetPrimAtPath(s)
    print("Selected", s)
    pose = omni.usd.utils.get_world_transform_matrix(curr_prim, timecode)
    print("Matrix Form:", pose)
    print("Translation: ", pose.ExtractTranslation())
    q = pose.ExtractRotation().GetQuaternion()
    print("Rotation: ", q.GetReal(), ",", q.GetImaginary()[0], ",", q.GetImaginary()[1], ",", q.GetImaginary()[2])
```

### Save current stage to USD

This can be useful if generating a stage in Python and you want to store it to reload later to debugging

```python
import carb
import omni

# Create a prim
result, path = omni.kit.commands.execute("CreateMeshPrimCommand", prim_type="Cube")
# Change the path as needed
omni.usd.get_context().save_as_stage("/path/to/asset/saved.usd", None)
```

---

# Util Snippets

## Simple Async Task

```python
import asyncio

import omni

# Async task that pauses simulation once the incoming task is complete
async def pause_sim(task):
    done, pending = await asyncio.wait({task})
    if task in done:
        print("Waited until next frame, pausing")
        omni.timeline.get_timeline_interface().pause()

# Start simulation, then wait a frame and run the pause_sim task
omni.timeline.get_timeline_interface().play()
task = asyncio.ensure_future(omni.kit.app.get_app().next_update_async())
asyncio.ensure_future(pause_sim(task))
```

## Get Camera Parameters

The below script show how to get the camera parameters associated with a viewport.

```python
import math

import omni
from omni.syntheticdata import helpers

stage = omni.usd.get_context().get_stage()
viewport_api = omni.kit.viewport.utility.get_active_viewport()
# Set viewport resolution, changes will occur on next frame
viewport_api.set_texture_resolution((512, 512))
# get resolution
(width, height) = viewport_api.get_texture_resolution()
aspect_ratio = width / height
# get camera prim attached to viewport
camera = stage.GetPrimAtPath(viewport_api.get_active_camera())
focal_length = camera.GetAttribute("focalLength").Get()
horiz_aperture = camera.GetAttribute("horizontalAperture").Get()
vert_aperture = camera.GetAttribute("verticalAperture").Get()
# Pixels are square so we can also do:
# vert_aperture = height / width * horiz_aperture
near, far = camera.GetAttribute("clippingRange").Get()
fov = 2 * math.atan(horiz_aperture / (2 * focal_length))
# helper to compute projection matrix
proj_mat = helpers.get_projection_matrix(fov, aspect_ratio, near, far)

# compute focal point and center
focal_x = height * focal_length / vert_aperture
focal_y = width * focal_length / horiz_aperture
center_x = height * 0.5
center_y = width * 0.5
```

## Rendering

There are three primary APIs you should use when making frequent updates to large amounts of geometry: `UsdGeom.Points`,
`UsdGeom.PointInstancer`, and `DebugDraw`. The different advantages and limitations of each of these methods are explained
below, and can help guide you on which method to use.

### UsdGeom.Points

Use the `UsdGeom.Points` API when the geometry needs to interact with the renderer.
The `UsdGeom.Points` API is the most efficient method to render large amounts of point geometry.

> ```python
> import random
>
> import omni.usd
> from pxr import UsdGeom
>
>
> class Example:
>     def create(self):
>         # Create Point List
>         N = 500
>         self.point_list = [
>             (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0)) for _ in range(N)
>         ]
>         self.sizes = [0.05 for _ in range(N)]
>
>         points_path = "/World/Points"
>         stage = omni.usd.get_context().get_stage()
>         self.points = UsdGeom.Points.Define(stage, points_path)
>         self.points.CreatePointsAttr().Set(self.point_list)
>         self.points.CreateWidthsAttr().Set(self.sizes)
>         self.points.CreateDisplayColorPrimvar("constant").Set([(1, 0, 1)])
>
>     def update(self):
>         # modify the point list
>         for i in range(len(self.point_list)):
>             self.point_list[i] = (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0))
>         # update the points
>         self.points.GetPointsAttr().Set(self.point_list)
>
>
> import asyncio
>
> import omni
>
> example = Example()
> example.create()
>
>
> async def update_points():
>     # Update 10 times, waiting 10 frames between each update
>     for _ in range(10):
>         for _ in range(10):
>             await omni.kit.app.get_app().next_update_async()
>         example.update()
>
>
> asyncio.ensure_future(update_points())
> ```

### UsdGeom.PointInstancer

Use the `UsdGeom.PointInstancer` API when the geometry needs to interact with the physics scene.
The `UsdGeom.PointInstancer` API lets you efficiently replicate an instance of a prim — with all of its USD properties —
and update all instances with a list of positions, colors, and sizes.

See the [PointInstancer Reference](https://openusd.org/release/api/class_usd_geom_point_instancer.html) for more information regarding the PointInstancer API.

Below are code snippets for how to create and update geometry with `UsdGeom.PointInstancer`:

> ```python
> import random
>
> import omni.usd
> from pxr import Gf, UsdGeom
>
>
> class Example:
>     def create(self):
>         # Create Point List
>         N = 500
>         scale = 0.05
>         self.point_list = [
>             (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0)) for _ in range(N)
>         ]
>         self.colors = [(1, 1, 1, 1) for _ in range(N)]
>         self.sizes = [(1.0, 1.0, 1.0) for _ in range(N)]
>
>         # Set up Geometry to be Instanced
>         cube_path = "/World/Cube"
>         stage = omni.usd.get_context().get_stage()
>         cube = UsdGeom.Cube(stage.DefinePrim(cube_path, "Cube"))
>         cube.AddScaleOp().Set(Gf.Vec3d(1, 1, 1) * scale)
>         cube.CreateDisplayColorPrimvar().Set([(0, 1, 1)])
>         # Set up Point Instancer
>
>         instance_path = "/World/PointInstancer"
>         self.point_instancer = UsdGeom.PointInstancer(stage.DefinePrim(instance_path, "PointInstancer"))
>         # Create & Set the Positions Attribute
>         self.positions_attr = self.point_instancer.CreatePositionsAttr()
>         self.positions_attr.Set(self.point_list)
>         self.scale_attr = self.point_instancer.CreateScalesAttr()
>         self.scale_attr.Set(self.sizes)
>         # Set the Instanced Geometry
>         self.point_instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])
>
>         self.proto_indices_attr = self.point_instancer.CreateProtoIndicesAttr()
>         self.proto_indices_attr.Set([0] * len(self.point_list))
>
>     def update(self):
>         # modify the point list
>         for i in range(len(self.point_list)):
>             self.point_list[i] = (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0))
>         # update the points
>         self.positions_attr.Set(self.point_list)
>
>
> import asyncio
>
> import omni
>
> example = Example()
> example.create()
>
>
> async def update_points():
>     # Update 10 times, waiting 10 frames between each update
>     for _ in range(10):
>         for _ in range(10):
>             await omni.kit.app.get_app().next_update_async()
>         example.update()
>
>
> asyncio.ensure_future(update_points())
> ```

### DebugDraw

The [Debug Drawing Extension API](Debugging_Profiling.md) API is useful for purely visualizing geometry in the Viewport. Geometry drawn with the `debug_draw_interface`
cannot be rendered and does not interact with the physics scene. However, it is the most performance-efficient method of visualizing geometry.

> See the [API documentation](../py/docs/extsbuild/isaacsim.util.debug_draw/docs/index.html) for complete usage information.

Below are code snippets for how to create and update geometry visualed with `DebugDraw`:

> ```python
> import random
>
> from isaacsim.util.debug_draw import _debug_draw
>
>
> class Example:
>     def create(self):
>         self.draw = _debug_draw.acquire_debug_draw_interface()
>         N = 500
>         self.point_list = [
>             (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0)) for _ in range(N)
>         ]
>         self.color_list = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(N)]
>         self.size_list = [10.0 for _ in range(N)]
>
>     def update(self):
>         # modify the point list
>         for i in range(len(self.point_list)):
>             self.point_list[i] = (random.uniform(-2.0, 2.0), random.uniform(-0.1, 0.1), random.uniform(-1.0, 1.0))
>
>         # draw the points
>         self.draw.clear_points()
>         self.draw.draw_points(self.point_list, self.color_list, self.size_list)
>
>
> import asyncio
>
> import omni
>
> example = Example()
> example.create()
>
>
> async def update_points():
>     # Update 10 times, waiting 10 frames between each update
>     for _ in range(10):
>         for _ in range(10):
>             await omni.kit.app.get_app().next_update_async()
>         example.update()
>
>
> asyncio.ensure_future(update_points())
> ```

### Rendering Frame Delay

The default rendering pipeline in the app experiences have upto 3 frames in flight to be rendered, which results in higher FPS since the simulation is not blocked until the latest state is rendered completely.

For applications that need the rendered data to correspond to the latest simulation state with no delay, the following experience file should be used `apps/omni.isaac.sim.zero_delay.python.kit`. Below is an example of how to use the experience file in a standlone workflow.

```python
import os

from isaacsim import SimulationApp

SimulationApp({"headless": True}, experience=f"{os.environ['EXP_PATH']}/isaacsim.exp.base.zero_delay.kit")
```

Alternatively, if you would like to use the specific settings instead, you can set them with extra\_args as well:

```python
import os

from isaacsim import SimulationApp

SimulationApp({"headless": True}, experience=f"{os.environ['EXP_PATH']}/isaacsim.exp.base.zero_delay.kit")
```

---

# Robot Simulation Snippets

Hint

Refer to the [Articulation](../py/source/extensions/isaacsim.core.experimental.prims/docs/index.html#isaacsim.core.experimental.prims.Articulation) class documentation for more details on the API.

## Wrapping Articulations

Note

The following snippets should only be run once on a new stage.
Create a new stage (File > New menu) and run the snippets in the Script Editor (Window > Script Editor menu).

Adds two Franka robots to the stage and wraps them via an [Articulation](../py/source/extensions/isaacsim.core.experimental.prims/docs/index.html#isaacsim.core.experimental.prims.Articulation) object to control them simultaneously.

```python
 1import isaacsim.core.experimental.utils.app as app_utils
 2import isaacsim.core.experimental.utils.stage as stage_utils
 3from isaacsim.core.experimental.prims import Articulation
 4from isaacsim.storage.native import get_assets_root_path
 5
 6# Add Franka robots to the stage
 7usd_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
 8variants = [("Gripper", "AlternateFinger"), ("Mesh", "Quality")]
 9stage_utils.add_reference_to_stage(usd_path, path="/World/Franka_1", variants=variants)
10stage_utils.add_reference_to_stage(usd_path, path="/World/Franka_2", variants=variants)
11
12# Wrap Franka robots via an Articulation object
13articulations = Articulation(
14    "/World/Franka_.*",
15    positions=[[-1, -1, 0], [1, 1, 0]],
16    reset_xform_op_properties=True,
17)
```

Play the simulation.
Then, open a new tab in the Script Editor window (Tab > Add Tab menu) and execute the following code to set the DOF positions for each articulation.

```python
1# Set the joint positions for each articulation
2articulations.set_dof_position_targets(
3    [
4        [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 0.0],
5        [-1.5, -1.5, -1.5, -1.5, -1.5, -1.5, -1.5, 0.04, 0.04],
6    ]
7)
```

## DOF Control

Note

The following snippets should only be run once on a new stage that has the Franka robot at the `/Franka` prim path,
and while the simulation is playing.

Prepare the scene:

1. Add a Franka robot to the stage via the Create > Robots > Franka Emika Panda Arm menu.
2. Play the simulation.

Warning

The snippets are disparate examples, running them out of order may have unintended consequences.
The resulting movements may not respect the robot’s kinematic limitations.

Make sure there is a Franka robot at the `/Franka` prim path and that the simulation is playing.
Then, open the Script Editor window (Window > Script Editor menu) and run the following snippets.

### Query Articulation

```python
 1from isaacsim.core.experimental.prims import Articulation
 2
 3articulation = Articulation("/Franka")
 4# Get articulation information
 5print("DOF count:", articulation.num_dofs)
 6print("DOF names:", articulation.dof_names)
 7print("DOF paths:", articulation.dof_paths)
 8print("DOF types:", articulation.dof_types)
 9print("Link count:", articulation.num_links)
10print("Link names:", articulation.link_names)
11print("Link paths:", articulation.link_paths)
```

### Read DOF States

```python
1from isaacsim.core.experimental.prims import Articulation
2
3articulation = Articulation("/Franka")
4# Get all DOF states
5print("DOF positions:", articulation.get_dof_positions())
6print("DOF velocities:", articulation.get_dof_velocities())
7print("DOF efforts:", articulation.get_dof_efforts())
```

### DOF Position Control

```python
1import numpy as np
2from isaacsim.core.experimental.prims import Articulation
3
4articulation = Articulation("/Franka")
5# Set all DOF positions to random values between -1 and 1
6articulation.set_dof_position_targets(np.random.rand(9) * 2 - 1)
```

### Single DOF Position Control

```python
1import numpy as np
2from isaacsim.core.experimental.prims import Articulation
3
4articulation = Articulation("/Franka")
5# Set the 'panda_finger_joint1' DOF position to 0.04.
6# The 'panda_finger_joint2' will mimic the value, as they are linked
7articulation.set_dof_position_targets(0.04, dof_indices=articulation.get_dof_indices("panda_finger_joint1"))
```

### DOF Velocity Control

```python
1import numpy as np
2from isaacsim.core.experimental.prims import Articulation
3
4articulation = Articulation("/Franka")
5# Switch to velocity control mode
6articulation.switch_dof_control_mode("velocity")
7# Set all DOF velocities to random values between -10 and 10
8articulation.set_dof_velocity_targets(10 * (np.random.rand(9) * 2 - 1))
```

### Single DOF Velocity Control

```python
1import numpy as np
2from isaacsim.core.experimental.prims import Articulation
3
4articulation = Articulation("/Franka")
5# Switch to velocity control mode
6articulation.switch_dof_control_mode("velocity")
7# Set the 'panda_joint4' DOF velocity to 0.25
8articulation.set_dof_velocity_targets(0.25, dof_indices=articulation.get_dof_indices("panda_joint4"))
```

### DOF Effort Control

```python
1import numpy as np
2from isaacsim.core.experimental.prims import Articulation
3
4articulation = Articulation("/Franka")
5# Switch to effort control mode
6articulation.switch_dof_control_mode("effort")
7# Set all DOF efforts to random values between -100 and 100
8articulation.set_dof_efforts(100 * (np.random.rand(9) * 2 - 1))
```

---

# API Documentation

Each of the following links navigate away from the doc set you are currently in.

* [Isaac Sim API](py/index.html)
* [Omniverse API Documentation](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/kit_overview.html)

---

# Core API Tutorial Series

The Core API tutorials is for beginner NVIDIA Isaac Sim users. This tutorial series details how to control wheeled robots and manipulators with controllers while logging robot and environment data.

- Hello World
- Hello Robot
- Adding a Manipulator Robot
- Adding Multiple Robots
- Multiple Robot Scenarios
- Adding Props
- Data Logging

---

# Hello World

[NVIDIA Omniverse™ Kit](https://docs.omniverse.nvidia.com/dev-guide/latest/kit-architecture.html "(in Omniverse Developer Guide)"), the toolkit that NVIDIA Isaac Sim uses to build its applications, provides a Python interpreter for scripting. This means every single GUI command, as well as many additional functions are available as Python APIs. However, the learning curve for interfacing with Omniverse Kit using Pixar’s USD Python API is steep and steps are frequently tedious. Therefore we’ve provided a set of APIs that are designed to be used in robotics applications, APIs that abstract away the complexity of USD APIs and merge multiple steps into one for frequently performed tasks.

In this tutorial, we will present the concepts of Core APIs and how to use them. We will start with adding a cube to an empty stage, and we’ll build upon it to create a scene with multiple robots executing multiple tasks simultaneously, as seen below.

## Learning Objectives

This tutorial series introduces the Core API. After this tutorial, you learn:

* How to use the Core APIs to manipulate the USD stage.
* How to add a rigid body to the [Stage](Glossary.md) and simulate it using Python in NVIDIA Isaac Sim.
* The difference between running Python in an **Extension Workflow** vs a **Standalone Workflow**.

*10-15 Minute Tutorial*

## Getting Started

**Prerequisites**

* Intermediate knowledge in Python and asynchronous programming is required for this tutorial.
* Please download and install [Visual Studio Code](https://code.visualstudio.com/download) prior to beginning this tutorial.
* Please review [Quick Tutorials](Quick_Tutorials.md) and [Workflows](Workflows.md) prior to beginning this tutorial.

Begin by opening the *Hello World* example. First activate **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` tab.

1. Click **Robotics Examples > General > Hello World**.
2. Verify that the window for the *Hello World* example extension is visible in the workspace.
3. Click the **Open Source Code** button to launch the source code for editing in [Visual Studio Code](https://code.visualstudio.com/download).
4. Click the **Open Containing Folder** button to open the directory containing the example files.

This folder contains three files: `hello_world.py`, `hello_world_extension.py`, and `__init__.py`.

The `hello_world.py` script is where the logic of the application will be added, while the UI
elements of the application will be added in `hello_world_extension.py` script and thus
linked to the logic.

1. Click the **LOAD** button to load the World.
2. click **File > New From Stage Template > Empty** to create a new stage, click **Don’t Save** when prompted to save the current stage.
3. Click the **LOAD** button to load the World again.
4. Open `hello_world.py` and press “Ctrl+S” to use the hot-reload feature. You will
   notice that the menu disappears from the workspace (because it was restarted).
5. Open the example menu again and click the **LOAD** button.

Now you can begin adding to this example.

## Code Overview

This example inherits from BaseSample, which is a boilerplate extension application that
sets up the basics for every robotics extension application. The following are a few examples of the
actions BaseSample performs:

1. Loading assets into the stage using a button.
2. Clearing the stage when a new stage is created.
3. Resetting objects to their default states.
4. Handling hot reloading.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2from isaacsim.examples.base.base_sample_experimental import BaseSample
 3from isaacsim.storage.native import get_assets_root_path
 4
 5
 6class HelloWorld(BaseSample):
 7    def __init__(self) -> None:
 8        super().__init__()
 9
10    # This function is called to setup the assets in the scene for the first time
11    def setup_scene(self):
12        # Add ground plane directly to the stage
13        ground_plane = stage_utils.add_reference_to_stage(
14            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
15            path="/World/ground",
16        )
```

### Key Concepts

**Stage Utilities**: The `stage_utils` module provides functions for directly manipulating the USD stage,
such as adding references, creating prims, and managing stage hierarchy.

**Prim Classes**: The API provides prim wrapper classes like `RigidPrim`, `GeomPrim`,
and `Articulation` that give you direct control over USD prims with physics capabilities.

**SimulationManager**: For callbacks and simulation events, the `SimulationManager` class provides
methods to register and deregister callbacks for various simulation events.

## Adding to the Scene

Use the Python API to add a cube as a rigid body to the scene. With the Core APIs,
create the geometry first, then apply collision and rigid body properties.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
 6from isaacsim.examples.base.base_sample_experimental import BaseSample
 7from isaacsim.storage.native import get_assets_root_path
 8
 9
10class HelloWorld(BaseSample):
11    def __init__(self) -> None:
12        super().__init__()
13
14    def setup_scene(self):
15        # Add ground plane
16        ground_plane = stage_utils.add_reference_to_stage(
17            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
18            path="/World/ground",
19        )
20
21        # Create a blue visual material for the cube
22        visual_material = PreviewSurfaceMaterial("/World/Materials/blue")
23        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])
24
25        # Create the cube geometry
26        self._cube_shape = Cube(
27            paths="/World/fancy_cube",
28            positions=np.array([[0.0, 0.0, 1.0]]),  # Starting position 1m above ground
29            sizes=[1.0],
30            scales=np.array([[0.5015, 0.5015, 0.5015]]),  # Scale the cube
31            reset_xform_op_properties=True,
32        )
33
34        # Apply collision APIs to enable physics collision
35        GeomPrim(paths=self._cube_shape.paths, apply_collision_apis=True)
36
37        # Make it a rigid body (dynamic object that responds to physics)
38        self._cube = RigidPrim(paths=self._cube_shape.paths)
39
40        # Apply the blue material
41        self._cube_shape.apply_visual_materials(visual_material)
```

1. Press **Ctrl+S** to save the code and hot-reload NVIDIA Isaac Sim.
2. Open the menu again.
3. click **File > New From Stage Template > Empty**, then the **LOAD** button. You need to perform this action
   if you change anything in the **setup\_scene**. Otherwise, you only need to press the
   **LOAD** button.
4. See the dynamic cube falling as the simulation starts automatically.

Note

Every time the code is edited or changed, press **Ctrl+S** to save the code and hot-reload
NVIDIA Isaac Sim.

### Understanding the Prim Classes

The experimental API uses a layered approach to create physics-enabled objects:

1. **Cube** (or other shape classes): Creates the visual geometry on the USD stage.
2. **GeomPrim**: Wraps the geometry and can apply collision APIs for physics interactions.
3. **RigidPrim**: Adds rigid body dynamics, making the object respond to gravity and forces.

This modular approach gives you fine-grained control - you can create static colliders
(GeomPrim without RigidPrim) or fully dynamic objects (with both).

## Inspecting Object Properties

Print the world pose and velocity of the cube. The highlighted lines show how you can query object properties.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
 6from isaacsim.examples.base.base_sample_experimental import BaseSample
 7from isaacsim.storage.native import get_assets_root_path
 8
 9
10class HelloWorld(BaseSample):
11    def __init__(self) -> None:
12        super().__init__()
13
14    def setup_scene(self):
15        # Add ground plane
16        ground_plane = stage_utils.add_reference_to_stage(
17            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
18            path="/World/ground",
19        )
20
21        # Create a blue visual material for the cube
22        visual_material = PreviewSurfaceMaterial("/World/Materials/blue")
23        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])
24
25        # Create the cube geometry
26        self._cube_shape = Cube(
27            paths="/World/fancy_cube",
28            positions=np.array([[0.0, 0.0, 1.0]]),
29            sizes=[1.0],
30            scales=np.array([[0.5015, 0.5015, 0.5015]]),
31            reset_xform_op_properties=True,
32        )
33
34        # Apply collision and rigid body
35        GeomPrim(paths=self._cube_shape.paths, apply_collision_apis=True)
36        self._cube = RigidPrim(paths=self._cube_shape.paths)
37        self._cube_shape.apply_visual_materials(visual_material)
38
39    # This function is called after load button is pressed
40    # It's called after setup_scene and after one physics time step
41    # to propagate physics handles needed to retrieve physical properties
42    async def setup_post_load(self):
43        # Query cube properties using RigidPrim methods
44        positions, orientations = self._cube.get_world_poses()
45        # get_velocities() returns a tuple: (linear_velocities, angular_velocities)
46        linear_velocities, angular_velocities = self._cube.get_velocities()
47
48        # Convert from warp arrays to numpy for printing
49        # Note: experimental APIs return batched results (even for single objects)
50        print("Cube position is : " + str(positions.numpy()[0]))
51        print("Cube's orientation is : " + str(orientations.numpy()[0]))
52        print("Cube's linear velocity is : " + str(linear_velocities.numpy()[0]))
```

Note

The experimental APIs return batched results as warp arrays. Use `.numpy()` to convert
them to numpy arrays, and index with `[0]` to get the first (and only) element when
working with a single object.

### Continuously Inspecting the Object Properties during Simulation

Print the world pose and velocity of the cube during simulation at every physics step
executed. As mentioned in [Workflows](Workflows.md), in this workflow the
application is running asynchronously and can’t control when to step physics. However, you can add
callbacks to ensure certain things happen before certain events.

Add a physics callback using the SimulationManager:

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
 6from isaacsim.core.simulation_manager import SimulationManager
 7from isaacsim.examples.base.base_sample_experimental import BaseSample
 8from isaacsim.storage.native import get_assets_root_path
 9
10
11class HelloWorld(BaseSample):
12    def __init__(self) -> None:
13        super().__init__()
14        self._physics_callback_id = None
15
16    def setup_scene(self):
17        # Add ground plane
18        ground_plane = stage_utils.add_reference_to_stage(
19            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
20            path="/World/ground",
21        )
22
23        # Create a blue visual material for the cube
24        visual_material = PreviewSurfaceMaterial("/World/Materials/blue")
25        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])
26
27        # Create the cube geometry
28        self._cube_shape = Cube(
29            paths="/World/fancy_cube",
30            positions=np.array([[0.0, 0.0, 1.0]]),
31            sizes=[1.0],
32            scales=np.array([[0.5015, 0.5015, 0.5015]]),
33            reset_xform_op_properties=True,
34        )
35
36        # Apply collision and rigid body
37        GeomPrim(paths=self._cube_shape.paths, apply_collision_apis=True)
38        self._cube = RigidPrim(paths=self._cube_shape.paths)
39        self._cube_shape.apply_visual_materials(visual_material)
40
41    async def setup_post_load(self):
42        # Register a physics callback using SimulationManager
43        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
44
45        self._physics_callback_id = SimulationManager.register_callback(
46            self.print_cube_info, IsaacEvents.POST_PHYSICS_STEP
47        )
48
49    # Physics callback function - called after each physics step
50    # Takes dt (delta time) and context as arguments
51    def print_cube_info(self, dt, context):
52        positions, orientations = self._cube.get_world_poses()
53        linear_velocities, angular_velocities = self._cube.get_velocities()
54
55        print("Cube position is : " + str(positions.numpy()[0]))
56        print("Cube's orientation is : " + str(orientations.numpy()[0]))
57        print("Cube's linear velocity is : " + str(linear_velocities.numpy()[0]))
58
59    def physics_cleanup(self):
60        # Clean up callback when the extension is unloaded
61        if self._physics_callback_id is not None:
62            SimulationManager.deregister_callback(self._physics_callback_id)
63            self._physics_callback_id = None
```

## Converting the Example to a Standalone Application

Note

* On windows use python.bat instead of python.sh
* The details of how python.sh works below are similar to how python.bat works

As mentioned in [Workflows](Workflows.md), in this workflow, the robotics
application is started when launched from Python right away.

1. Open a new `my_application.py` file and add the following:

```python
 1# Launch Isaac Sim before any other imports
 2# Default first two lines in any standalone application
 3from isaacsim import SimulationApp
 4
 5simulation_app = SimulationApp({"headless": False})  # we can also run as headless
 6
 7# Now import Isaac Sim modules
 8import isaacsim.core.experimental.utils.stage as stage_utils
 9import numpy as np
10import omni.timeline
11from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
12from isaacsim.core.experimental.objects import Cube
13from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
14from isaacsim.core.simulation_manager import SimulationManager
15from isaacsim.storage.native import get_assets_root_path
16
17# Add ground plane
18ground_plane = stage_utils.add_reference_to_stage(
19    usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
20    path="/World/ground",
21)
22
23# Create a blue visual material for the cube
24visual_material = PreviewSurfaceMaterial("/World/Materials/blue")
25visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])
26
27# Create the cube geometry
28cube_shape = Cube(
29    paths="/World/fancy_cube",
30    positions=np.array([[0.0, 0.0, 1.0]]),
31    sizes=[1.0],
32    scales=np.array([[0.5, 0.5, 0.5]]),
33    reset_xform_op_properties=True,
34)
35
36# Apply collision and rigid body
37GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
38cube = RigidPrim(paths=cube_shape.paths)
39cube_shape.apply_visual_materials(visual_material)
40
41# Start the timeline (physics simulation)
42omni.timeline.get_timeline_interface().play()
43simulation_app.update()
44
45# Run the simulation loop
46for i in range(50):
47    # Only query when physics is actively simulating
48    if SimulationManager.is_simulating():
49        positions, orientations = cube.get_world_poses()
50        linear_velocities, angular_velocities = cube.get_velocities()
51
52        # Will be shown on terminal
53        print("Cube position is : " + str(positions.numpy()[0]))
54        print("Cube's orientation is : " + str(orientations.numpy()[0]))
55        print("Cube's linear velocity is : " + str(linear_velocities.numpy()[0]))
56
57    # Step the app (physics + rendering)
58    simulation_app.update()
59
60simulation_app.close()  # close Isaac Sim
```

1. Run it using `./python.sh ./exts/isaacsim.examples.interactive/isaacsim/examples/interactive/user_examples/my_application.py`.

## Summary

This tutorial covered the following topics:

1. Overview of the Core APIs for direct stage manipulation.
2. Using `stage_utils` to add assets to the stage.
3. Creating dynamic objects with `Cube`, `GeomPrim`, and `RigidPrim`.
4. Registering physics callbacks with `SimulationManager`.
5. Accessing dynamic properties for objects using prim wrapper methods.
6. The main differences in a standalone application.

### Next Steps

Continue to [Hello Robot](Python_Scripting_and_Tutorials.md) to learn how to add a robot to the simulation.

Note

The next tutorials will be developed mainly using the extensions application workflow.
However, conversion to other workflows is similar given what was covered
in this tutorial.

---

# Hello Robot

## Learning Objectives

This tutorial details how to add and move a mobile robot in NVIDIA Isaac Sim in an extension application.
After this tutorial, you will understand how to add a robot to the simulation and apply actions to
its wheels using Python.

*10-15 Minute Tutorial*

## Getting Started

**Prerequisites**

* Review [Hello World](Python_Scripting_and_Tutorials.md) prior to beginning this tutorial.

Begin with the source code of the **Hello World** example developed in the previous tutorial:
[Hello World](Python_Scripting_and_Tutorials.md).

## Adding a Robot

Begin by adding a NVIDIA Jetbot to the scene, which allows you to access the library of NVIDIA Isaac Sim
robots, sensors, and environments located on a [Omniverse Nucleus](Glossary.md) Server using Python,
as well as navigate through it using the **Content** window.

Note

The server shown in these steps has been connected to in [Workstation Setup](Installation.md). Follow these steps first before proceeding.

1. Add the assets by simply dragging them to the stage window or the viewport.
2. Try to do the same thing through Python in the **Hello World** example.
3. Create a new stage: **File > new > Don’t Save**
4. Open the `hello_world.py` file by clicking the **Open Source Code**
   button in the **Hello World** window.

```python
 1import carb
 2import isaacsim.core.experimental.utils.stage as stage_utils
 3from isaacsim.core.experimental.prims import Articulation
 4from isaacsim.examples.base.base_sample_experimental import BaseSample
 5from isaacsim.storage.native import get_assets_root_path
 6
 7
 8class HelloWorld(BaseSample):
 9    def __init__(self) -> None:
10        super().__init__()
11
12    def setup_scene(self):
13        # Add ground plane
14        ground_plane = stage_utils.add_reference_to_stage(
15            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
16            path="/World/ground",
17        )
18
19        # Get the assets root path from the Nucleus server
20        assets_root_path = get_assets_root_path()
21        if assets_root_path is None:
22            carb.log_error("Could not find nucleus server with /Isaac folder")
23            return
24
25        # Add the Jetbot robot to the stage
26        asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
27        stage_utils.add_reference_to_stage(usd_path=asset_path, path="/World/Fancy_Robot")
28
29    async def setup_post_load(self):
30        # Wrap the Jetbot with the Articulation class for control
31        self._jetbot = Articulation("/World/Fancy_Robot")
32
33        # Print info about the Jetbot
34        print("Number of DOFs: " + str(self._jetbot.num_dofs))
35        print("DOF names: " + str(self._jetbot.dof_names))
36        print("Joint Positions: " + str(self._jetbot.get_dof_positions().numpy()))
```

Click the **LOAD** button to load the scene and see the Jetbot appear. Although it is being simulated,
it is not moving. The next section walks through how to make the robot move.

## Move the Robot

In NVIDIA Isaac Sim, Robots are constructed of physically accurate articulated joints. Applying actions
to these articulations make them move.

Next, apply random velocities to the Jetbot’s wheel joints to get it moving.

```python
 1import carb
 2import isaacsim.core.experimental.utils.stage as stage_utils
 3import numpy as np
 4from isaacsim.core.experimental.prims import Articulation
 5from isaacsim.core.simulation_manager import SimulationManager
 6from isaacsim.examples.base.base_sample_experimental import BaseSample
 7from isaacsim.storage.native import get_assets_root_path
 8
 9
10class HelloWorld(BaseSample):
11    def __init__(self) -> None:
12        super().__init__()
13        self._physics_callback_id = None
14
15    def setup_scene(self):
16        # Add ground plane
17        ground_plane = stage_utils.add_reference_to_stage(
18            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
19            path="/World/ground",
20        )
21
22        # Get the assets root path from the Nucleus server
23        assets_root_path = get_assets_root_path()
24        if assets_root_path is None:
25            carb.log_error("Could not find nucleus server with /Isaac folder")
26            return
27
28        # Add the Jetbot robot to the stage
29        asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
30        stage_utils.add_reference_to_stage(usd_path=asset_path, path="/World/Fancy_Robot")
31
32    async def setup_post_load(self):
33        # Wrap the Jetbot with the Articulation class for control
34        self._jetbot = Articulation("/World/Fancy_Robot")
35
36        # Register a physics callback to send actions every physics step
37        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
38
39        self._physics_callback_id = SimulationManager.register_callback(
40            self.send_robot_actions, IsaacEvents.POST_PHYSICS_STEP
41        )
42
43    def send_robot_actions(self, dt, context):
44        # Apply random velocity targets to the wheel joints
45        # Jetbot has 2 DOFs: left_wheel_joint and right_wheel_joint
46        random_velocities = 5 * np.random.rand(1, 2)  # Shape: (1, num_dofs)
47        self._jetbot.set_dof_velocity_targets(random_velocities)
48
49    def physics_cleanup(self):
50        # Clean up callback when the extension is unloaded
51        if self._physics_callback_id is not None:
52            SimulationManager.deregister_callback(self._physics_callback_id)
53            self._physics_callback_id = None
```

Click the **LOAD** button to load the scene and watch the Jetbot move with random velocities.

Note

Pressing **STOP**, then **PLAY** in this workflow might not reset the world properly. Use
the **RESET** button instead.

### Extra Practice

This example applies random velocities to the Jetbot articulation controller. Try the following
exercises:

1. Make the Jetbot move backwards (hint: use negative velocities).
2. Make the Jetbot turn right (hint: apply different velocities to each wheel).
3. Make the Jetbot stop after 5 seconds (hint: track elapsed time in the callback).

## Controlling Specific Joints

You can also control specific joints by their names or indices. Here’s how to get the wheel
joint indices and apply velocities only to specific joints:

```python
 1import carb
 2import isaacsim.core.experimental.utils.stage as stage_utils
 3import numpy as np
 4from isaacsim.core.experimental.prims import Articulation
 5from isaacsim.core.simulation_manager import SimulationManager
 6from isaacsim.examples.base.base_sample_experimental import BaseSample
 7from isaacsim.storage.native import get_assets_root_path
 8
 9
10class HelloWorld(BaseSample):
11    def __init__(self) -> None:
12        super().__init__()
13        self._physics_callback_id = None
14
15    def setup_scene(self):
16        # Add ground plane
17        ground_plane = stage_utils.add_reference_to_stage(
18            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
19            path="/World/ground",
20        )
21
22        # Add the Jetbot robot to the stage
23        assets_root_path = get_assets_root_path()
24        asset_path = assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd"
25        stage_utils.add_reference_to_stage(usd_path=asset_path, path="/World/Fancy_Robot")
26
27    async def setup_post_load(self):
28        # Wrap the Jetbot with the Articulation class
29        self._jetbot = Articulation("/World/Fancy_Robot")
30
31        # Print available DOF names
32        print("Available DOFs:", self._jetbot.dof_names)
33
34        # Get indices for specific wheel joints
35        self._wheel_indices = self._jetbot.get_dof_indices(["left_wheel_joint", "right_wheel_joint"]).numpy()
36        print("Wheel indices:", self._wheel_indices)
37
38        # Register physics callback
39        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
40
41        self._physics_callback_id = SimulationManager.register_callback(
42            self.send_robot_actions, IsaacEvents.POST_PHYSICS_STEP
43        )
44
45    def send_robot_actions(self, dt, context):
46        # Apply velocity targets to specific DOF indices
47        wheel_velocities = np.array([[10.0, 10.0]])  # Both wheels same speed = forward
48        self._jetbot.set_dof_velocity_targets(wheel_velocities, dof_indices=self._wheel_indices)
49
50    def physics_cleanup(self):
51        if self._physics_callback_id is not None:
52            SimulationManager.deregister_callback(self._physics_callback_id)
53            self._physics_callback_id = None
```

## Summary

This tutorial covered the following topics:

1. Adding NVIDIA Isaac Sim library components from a Nucleus Server
2. Adding a robot to the stage using `stage_utils.add_reference_to_stage()`
3. Wrapping a robot with the `Articulation` class for control
4. Using `set_dof_velocity_targets()` to apply velocity control
5. Registering physics callbacks with `SimulationManager`
6. Controlling specific joints by name or index

### Next Steps

Continue on to the next tutorial in the Essential Tutorials series, [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md),
to learn how to add a manipulator robot to the simulation.

### Further Learning

**Nucleus Server**

* For an overview of how to best leverage a Nucleus Server, see the [Nucleus Overview in NVIDIA Omniverse](https://youtu.be/JaoIQ4YBnBE)
  tutorial.

**Robot Specific Extensions**

* NVIDIA Isaac Sim provides several robot extensions such as `isaacsim.robot.manipulators.examples.franka`, `isaacsim.robot.manipulators.examples.universal_robots`,
  and many more. To learn more, check out the standalone examples located at `standalone_examples/api/isaacsim.robot.manipulators/franka`
  and `standalone_examples/api/isaacsim.robot.manipulators/universal_robots/`.

---

# Adding a Manipulator Robot

## Learning Objectives

This tutorial introduces a manipulator robot to the simulation, a Franka Panda.
It describes how to add the robot to the scene and execute a pick-and-place operation.
After this tutorial, you will have more experience using manipulator robots and
controlling them with inverse kinematics in NVIDIA Isaac Sim.

*15-20 Minute Tutorial*

## Getting Started

**Prerequisites**

* Review [Hello Robot](Python_Scripting_and_Tutorials.md) prior to beginning this tutorial.

Begin with the source code open from the [Hello Robot](Python_Scripting_and_Tutorials.md) tutorial,
by clicking the **Open Source Code** button in the Hello World Example window.

Note

Pressing **STOP**, then **PLAY** in this workflow might not reset the world properly. Use
the **RESET** button instead.

## Creating the Scene with a Franka Robot

Add a Franka robot and a cube for the robot to pick up using the `FrankaExperimental` class.
This class inherits from `Articulation` and provides high-level control methods including
inverse kinematics and gripper control.

When you set `create_robot=True` in the constructor, `FrankaExperimental` automatically
spawns the Franka robot USD asset at the specified path.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
 6from isaacsim.examples.base.base_sample_experimental import BaseSample
 7from isaacsim.robot.manipulators.examples.franka import FrankaExperimental
 8from isaacsim.storage.native import get_assets_root_path
 9
10
11class HelloWorld(BaseSample):
12    def __init__(self) -> None:
13        super().__init__()
14
15    def setup_scene(self):
16        # Add ground plane
17        ground_plane = stage_utils.add_reference_to_stage(
18            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
19            path="/World/ground",
20        )
21
22        # Create the Franka robot - constructor spawns the robot when create_robot=True
23        self._robot = FrankaExperimental(robot_path="/World/robot", create_robot=True)
24
25        # Create a blue cube for the robot to pick up
26        visual_material = PreviewSurfaceMaterial("/World/Materials/blue")
27        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])
28
29        cube_shape = Cube(
30            paths="/World/Cube",
31            positions=np.array([[0.5, 0.0, 0.0258]]),
32            sizes=[1.0],
33            scales=np.array([[0.0515, 0.0515, 0.0515]]),
34            reset_xform_op_properties=True,
35        )
36
37        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
38        RigidPrim(paths=cube_shape.paths)
39        cube_shape.apply_visual_materials(visual_material)
```

Click the **LOAD** button to see the Franka robot and cube in the scene.

The `FrankaExperimental` class provides these key methods for robot control:

* `set_end_effector_pose(position, orientation)` - Move end-effector using inverse kinematics
* `open_gripper()` / `close_gripper()` - Control the gripper
* `get_current_state()` - Get DOF positions and end-effector pose
* `get_downward_orientation()` - Get quaternion for downward-facing orientation
* `reset_to_default_pose()` - Reset robot to home position

## Using FrankaPickPlace for Complete Pick-and-Place

For a complete pick-and-place operation, use the `FrankaPickPlace` class. This class has a
`setup_scene()` method that spawns everything needed for pick-and-place: the Franka robot,
ground plane, and a cube to manipulate.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.simulation_manager import SimulationManager
 4from isaacsim.examples.base.base_sample_experimental import BaseSample
 5from isaacsim.robot.manipulators.examples.franka import FrankaPickPlace
 6
 7
 8class HelloWorld(BaseSample):
 9    def __init__(self) -> None:
10        super().__init__()
11        self._physics_callback_id = None
12
13    def setup_scene(self):
14        # FrankaPickPlace.setup_scene() spawns the complete scene:
15        # - Ground plane
16        # - Franka robot (using FrankaExperimental)
17        # - Blue cube for manipulation
18        self._controller = FrankaPickPlace()
19        self._controller.setup_scene()
20
21    async def setup_post_load(self):
22        # Reset the controller to initialize the robot position
23        self._controller.reset()
24
25        # Register physics callback to execute pick-place steps
26        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
27
28        self._physics_callback_id = SimulationManager.register_callback(
29            self.physics_step, IsaacEvents.POST_PHYSICS_STEP
30        )
31
32    def physics_step(self, dt, context):
33        # Execute one step of the pick-and-place operation
34        if not self._controller.is_done():
35            self._controller.forward()
36        else:
37            print("Pick-and-place completed!")
38            self._timeline.pause()
39
40    # This function is called after Reset button is pressed
41    # Resetting anything in the world should happen here
42    async def setup_post_reset(self):
43        self._controller.reset()
44        self._franka.gripper.set_joint_positions(self._franka.gripper.joint_opened_positions)
45        await self._world.play_async()
46        return
47
48    def physics_step(self, step_size):
49        cube_position, _ = self._fancy_cube.get_world_pose()
50        goal_position = np.array([-0.3, -0.3, 0.0515 / 2.0])
51        current_joint_positions = self._franka.get_joint_positions()
52        actions = self._controller.forward(
53            picking_position=cube_position,
54            placing_position=goal_position,
55            current_joint_positions=current_joint_positions,
56        )
57        self._franka.apply_action(actions)
58        # Only for the pick and place controller, indicating if the state
59        # machine reached the final state.
60        if self._controller.is_done():
61            self._world.pause()
62        return
```

Click the **LOAD** button to start the pick-and-place operation. The robot will automatically
execute all phases of picking up and placing the cube.

## Customizing the FrankaPickPlace Scene

The `setup_scene()` method accepts parameters to customize the cube position, size,
and target position:

```python
1self._controller = FrankaPickPlace()
2self._controller.setup_scene(
3    cube_initial_position=np.array([0.4, 0.2, 0.0258]),
4    cube_size=np.array([0.05, 0.05, 0.05]),
5    target_position=np.array([-0.4, 0.2, 0.12]),
6)
```

## Understanding the Pick-and-Place State Machine

The `FrankaPickPlace` class uses a state machine with the following phases:

Pick-and-Place Phases

| Phase | Description | Default Steps |
| --- | --- | --- |
| 0 | Move to x,y position above cube | 60 |
| 1 | Approach down to cube | 40 |
| 2 | Close gripper to grasp | 20 |
| 3 | Lift cube upward | 40 |
| 4 | Move cube to target location | 80 |
| 5 | Open gripper to release | 20 |
| 6 | Move up and away | 20 |

You can customize the phase durations by passing `events_dt` to the constructor:

```python
# Custom phase durations (steps for each phase)
controller = FrankaPickPlace(events_dt=[80, 60, 30, 60, 100, 30, 30])
```

## Summary

This tutorial covered the following topics:

1. Adding a Franka manipulator robot using `FrankaExperimental` with `create_robot=True`
2. Using the `FrankaPickPlace.setup_scene()` method to spawn a complete pick-and-place scene
3. Executing pick-and-place operations with the `forward()` method
4. Understanding and customizing the pick-and-place state machine phases

### Next Steps

Continue to the next tutorial in our Essential Tutorials series, [Adding Multiple Robots](Python_Scripting_and_Tutorials.md),
to learn how to add multiple robots to the simulation.

---

# Adding Multiple Robots

## Learning Objectives

This tutorial integrates two different types of robots into the same simulation:
a mobile robot (Jetbot) and a manipulator (Franka), working together to accomplish a task.
The Jetbot pushes a cube towards the Franka, which then picks it up.
After this tutorial, you will have experience building multi-robot simulations with object interaction.

*15-20 Minute Tutorial*

## Getting Started

**Prerequisites**

* Review [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md) prior to beginning this tutorial.

Begin with the source code open from the previous tutorial, [Adding a Manipulator Robot](Python_Scripting_and_Tutorials.md).

Note

Pressing **STOP**, then **PLAY** in this workflow might not reset the world properly. Use
the **RESET** button instead.

## Creating the Scene

Begin by adding the Jetbot, Franka Panda, and the Cube from the previous tutorials to the scene.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
 6from isaacsim.examples.base.base_sample_experimental import BaseSample
 7from isaacsim.storage.native import get_assets_root_path
 8
 9
10class HelloWorld(BaseSample):
11    def __init__(self) -> None:
12        super().__init__()
13
14    def setup_scene(self):
15        assets_root_path = get_assets_root_path()
16
17        # Add ground plane
18        stage_utils.add_reference_to_stage(
19            usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
20            path="/World/ground",
21        )
22
23        # Add Jetbot mobile robot
24        stage_utils.add_reference_to_stage(
25            usd_path=assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
26            path="/World/Jetbot",
27        )
28
29        # Add a cube in front of Jetbot for it to push
30        visual_material = PreviewSurfaceMaterial("/World/Materials/red")
31        visual_material.set_input_values("diffuseColor", [1.0, 0.0, 0.0])
32        cube_shape = Cube(
33            paths="/World/Cube",
34            positions=np.array([[0.15, 0.0, 0.025]]),  # In front of Jetbot
35            sizes=[1.0],
36            scales=np.array([[0.05, 0.05, 0.05]]),
37            reset_xform_op_properties=True,
38        )
39        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
40        RigidPrim(paths=cube_shape.paths)
41        cube_shape.apply_visual_materials(visual_material)
42
43        # Add Franka manipulator at a position the Jetbot will push the cube to
44        stage_utils.add_reference_to_stage(
45            usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
46            path="/World/Franka",
47        )
48
49        # Position Franka so the cube will be pushed into its workspace
50        franka_xform = XformPrim("/World/Franka")
51        franka_xform.set_world_poses(positions=np.array([[0.8, -0.5, 0.0]]))
52
53    async def setup_post_load(self):
54        # Create Articulation handles for both robots
55        self._jetbot = Articulation("/World/Jetbot")
56        self._franka = Articulation("/World/Franka")
57
58        # Print robot info
59        print(f"Jetbot DOFs: {self._jetbot.num_dofs}, names: {self._jetbot.dof_names}")
60        print(f"Franka DOFs: {self._franka.num_dofs}, names: {self._franka.dof_names}")
```

Click the **LOAD** button to see both robots and the cube in the scene.

## Controlling Multiple Robots

Now add physics callbacks to control both robots simultaneously. The Jetbot will push the cube
forward while the Franka prepares to receive it.

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
 6from isaacsim.core.simulation_manager import SimulationManager
 7from isaacsim.examples.base.base_sample_experimental import BaseSample
 8from isaacsim.storage.native import get_assets_root_path
 9
10
11class HelloWorld(BaseSample):
12    def __init__(self) -> None:
13        super().__init__()
14        self._physics_callback_id = None
15        self._step_counter = 0
16
17    def setup_scene(self):
18        assets_root_path = get_assets_root_path()
19
20        # Add ground plane
21        stage_utils.add_reference_to_stage(
22            usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
23            path="/World/ground",
24        )
25
26        # Add Jetbot mobile robot
27        stage_utils.add_reference_to_stage(
28            usd_path=assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
29            path="/World/Jetbot",
30        )
31
32        # Add a cube in front of Jetbot for it to push
33        visual_material = PreviewSurfaceMaterial("/World/Materials/red")
34        visual_material.set_input_values("diffuseColor", [1.0, 0.0, 0.0])
35        cube_shape = Cube(
36            paths="/World/Cube",
37            positions=np.array([[0.15, 0.0, 0.025]]),
38            sizes=[1.0],
39            scales=np.array([[0.05, 0.05, 0.05]]),
40            reset_xform_op_properties=True,
41        )
42        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
43        RigidPrim(paths=cube_shape.paths)
44        cube_shape.apply_visual_materials(visual_material)
45
46        # Add Franka manipulator
47        stage_utils.add_reference_to_stage(
48            usd_path=assets_root_path + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
49            path="/World/Franka",
50        )
51
52        # Position Franka forward and to the right of Jetbot's path
53        franka_xform = XformPrim("/World/Franka")
54        franka_xform.set_world_poses(positions=np.array([[0.8, -0.5, 0.0]]))
55
56    async def setup_post_load(self):
57        # Create Articulation handles
58        self._jetbot = Articulation("/World/Jetbot")
59        self._franka = Articulation("/World/Franka")
60        self._cube = RigidPrim("/World/Cube")
61        self._step_counter = 0
62
63        # Register physics callback
64        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
65
66        self._physics_callback_id = SimulationManager.register_callback(
67            self.physics_step, IsaacEvents.POST_PHYSICS_STEP
68        )
69
70    def physics_step(self, dt, context):
71        self._step_counter += 1
72        if self._step_counter < 300:
73            # Drive Jetbot forward to push the cube
74            self._jetbot.set_dof_velocity_targets([[10.0, 10.0]])
75        else:
76            # Stop the Jetbot after pushing
77            self._jetbot.set_dof_velocity_targets([[0.0, 0.0]])
78
79    def physics_cleanup(self):
80        if self._physics_callback_id is not None:
81            SimulationManager.deregister_callback(self._physics_callback_id)
82            self._physics_callback_id = None
```

Watch as the Jetbot pushes the cube towards the Franka!

## Adding State Machine Logic

Create a state machine to coordinate the robots: first the Jetbot pushes the cube towards Franka,
then backs up to give space, and finally Franka executes a full pick-and-place sequence using
the `FrankaExperimental` class for IK-based end-effector control:

```python
  1import isaacsim.core.experimental.utils.stage as stage_utils
  2import numpy as np
  3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
  4from isaacsim.core.experimental.objects import Cube
  5from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
  6from isaacsim.core.simulation_manager import SimulationManager
  7from isaacsim.examples.base.base_sample_experimental import BaseSample
  8from isaacsim.robot.manipulators.examples.franka import FrankaExperimental
  9from isaacsim.storage.native import get_assets_root_path
 10
 11
 12class HelloWorld(BaseSample):
 13    def __init__(self) -> None:
 14        super().__init__()
 15        self._physics_callback_id = None
 16        self._state = 0
 17
 18    def setup_scene(self):
 19        assets_root_path = get_assets_root_path()
 20
 21        # Add ground plane
 22        stage_utils.add_reference_to_stage(
 23            usd_path=assets_root_path + "/Isaac/Environments/Grid/default_environment.usd",
 24            path="/World/ground",
 25        )
 26
 27        # Add Jetbot at origin
 28        stage_utils.add_reference_to_stage(
 29            usd_path=assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
 30            path="/World/Jetbot",
 31        )
 32
 33        # Add cube in front of Jetbot
 34        visual_material = PreviewSurfaceMaterial("/World/Materials/blue")
 35        visual_material.set_input_values("diffuseColor", [0.0, 0.0, 1.0])
 36        cube_shape = Cube(
 37            paths="/World/Cube",
 38            positions=np.array([[0.15, 0.0, 0.0258]]),
 39            sizes=[1.0],
 40            scales=np.array([[0.05, 0.05, 0.05]]),
 41            reset_xform_op_properties=True,
 42        )
 43        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
 44        RigidPrim(paths=cube_shape.paths)
 45        cube_shape.apply_visual_materials(visual_material)
 46
 47        # Add Franka using FrankaExperimental for IK and gripper control
 48        self._franka = FrankaExperimental(robot_path="/World/Franka", create_robot=True)
 49        franka_xform = XformPrim("/World/Franka")
 50        franka_xform.set_world_poses(positions=[[0.8, -0.3, 0.0]])
 51
 52    async def setup_post_load(self):
 53        self._jetbot = Articulation("/World/Jetbot")
 54        self._cube = RigidPrim("/World/Cube")
 55        self._cube_goal = np.array([1.2, 0.0, 0.0])  # Target: Franka reaches from the side
 56        self._step_counter = 0
 57        self._pick_phase = 0
 58
 59        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
 60
 61        self._physics_callback_id = SimulationManager.register_callback(
 62            self.physics_step, IsaacEvents.POST_PHYSICS_STEP
 63        )
 64        self._state = 0
 65
 66    def physics_step(self, dt, context):
 67        if self._state == 0:
 68            # Jetbot pushes cube to Franka
 69            cube_pos = self._cube.get_world_poses()[0].numpy()[0]
 70            if np.linalg.norm(cube_pos[:2] - self._cube_goal[:2]) > 0.05:
 71                self._jetbot.set_dof_velocity_targets([[10.0, 10.0]])
 72            else:
 73                self._jetbot.set_dof_velocity_targets([[0.0, 0.0]])
 74                print("Cube delivered! Backing up...")
 75                self._state = 1
 76                self._step_counter = 0
 77
 78        elif self._state == 1:
 79            # Jetbot backs up
 80            self._jetbot.set_dof_velocity_targets([[-8.0, -8.0]])
 81            self._step_counter += 1
 82            if self._step_counter > 100:
 83                self._jetbot.set_dof_velocity_targets(np.array([[0.0, 0.0]]))
 84                print("Franka starting pick-and-place...")
 85                self._state = 2
 86                self._step_counter = 0
 87                self._franka.open_gripper()
 88
 89        elif self._state == 2:
 90            # Franka pick-and-place sequence using step counter
 91            cube_pos = self._cube.get_world_poses()[0].numpy()[0]
 92            down_orient = self._franka.get_downward_orientation()
 93            self._step_counter += 1
 94
 95            if self._pick_phase == 0:
 96                # Move above cube (wait 120 steps)
 97                self._franka.set_end_effector_pose(
 98                    np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.2]]), down_orient
 99                )
100                if self._step_counter > 120:
101                    self._pick_phase = 1
102                    self._step_counter = 0
103            elif self._pick_phase == 1:
104                # Lower to cube (wait 100 steps)
105                self._franka.set_end_effector_pose(
106                    np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.1]]), down_orient
107                )
108                if self._step_counter > 100:
109                    self._franka.close_gripper()
110                    self._pick_phase = 2
111                    self._step_counter = 0
112            elif self._pick_phase == 2:
113                # Close the gripper (wait 50 steps)
114                self._franka.close_gripper()
115                if self._step_counter > 50:
116                    self._pick_phase = 3
117                    self._step_counter = 0
118            elif self._pick_phase == 3:
119                # Lift cube (wait 100 steps)
120                self._franka.set_end_effector_pose(
121                    np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.25]]), down_orient
122                )
123                if self._step_counter > 100:
124                    self._pick_phase = 4
125                    self._step_counter = 0
126            elif self._pick_phase == 4:
127                # Move to target (wait 150 steps)
128                self._franka.set_end_effector_pose(np.array([[0.3, 0.3, 0.15]]), down_orient)
129                if self._step_counter > 150:
130                    self._franka.open_gripper()
131                    self._pick_phase = 5
132                    self._step_counter = 0
133            elif self._pick_phase == 5:
134                # Lift the arm (wait 150 steps)
135                self._franka.set_end_effector_pose(
136                    np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.5]]), down_orient
137                )
138                if self._step_counter > 150:
139                    self._step_counter = 0
140
141    async def setup_post_reset(self):
142        self._state = 0
143        self._step_counter = 0
144        self._pick_phase = 0
145        self._franka.reset_to_default_pose()
146
147    def physics_cleanup(self):
148        if self._physics_callback_id is not None:
149            SimulationManager.deregister_callback(self._physics_callback_id)
150            self._physics_callback_id = None
```

## Summary

This tutorial covered the following topics:

1. Adding multiple robots and objects (cube) to the scene
2. Using `Cube`, `GeomPrim`, and `RigidPrim` to create pushable objects
3. Using the `Articulation` class to control different robot types
4. Having a mobile robot (Jetbot) push objects towards a manipulator (Franka)
5. Building state machine logic to coordinate pushing, backing up, and picking
6. Using `FrankaExperimental` for IK-based end-effector control and gripper operations

### Next Steps

Continue on to the next tutorial in our Essential Tutorials series, [Multiple Robot Scenarios](Python_Scripting_and_Tutorials.md),
to learn how to add multiple tasks and manage them.

---

# Multiple Robot Scenarios

## Learning Objectives

This tutorial describes how to create and manage multiple robot scenarios in NVIDIA Isaac Sim.
It explains how to use parameterization and Python classes to scale your simulations with
multiple instances of robots performing similar tasks. After this tutorial, you will have
more experience building scalable multi-robot simulations in NVIDIA Isaac Sim.

*15-20 Minute Tutorial*

## Getting Started

**Prerequisites**

* Review [Adding Multiple Robots](Python_Scripting_and_Tutorials.md) prior to beginning this tutorial.

Begin with the source code open from the previous tutorial, [Adding Multiple Robots](Python_Scripting_and_Tutorials.md).

Note

Pressing **STOP**, then **PLAY** in this workflow might not reset the world properly. Use
the **RESET** button instead.

## Organizing Robot Scenarios with Classes

When working with multiple robots performing similar tasks, it’s helpful to encapsulate the
robot setup and control logic into reusable classes. This approach allows you to easily
create multiple instances with different parameters (like position offsets).

Create a `RobotScenario` class that manages a Jetbot pushing a cube to a Franka:

```python
  1import isaacsim.core.experimental.utils.stage as stage_utils
  2import numpy as np
  3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
  4from isaacsim.core.experimental.objects import Cube
  5from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
  6from isaacsim.core.simulation_manager import SimulationManager
  7from isaacsim.examples.base.base_sample_experimental import BaseSample
  8from isaacsim.robot.manipulators.examples.franka import FrankaExperimental
  9from isaacsim.storage.native import get_assets_root_path
 10
 11
 12class RobotScenario:
 13    """Encapsulates a Jetbot + Franka + Cube scenario with an offset."""
 14
 15    def __init__(self, name: str, offset: np.ndarray = np.array([0.0, 0.0, 0.0])):
 16        self.name = name
 17        self.offset = offset
 18        self.state = 0
 19        self.step_counter = 0
 20        self.pick_phase = 0
 21        self.jetbot = None
 22        self.franka = None
 23        self.cube = None
 24        self.cube_goal = np.array([1.2, 0.0, 0.0]) + offset
 25
 26    def setup_scene(self):
 27        """Create the robots and cube for this scenario."""
 28        assets_root_path = get_assets_root_path()
 29        base_path = f"/World/{self.name}"
 30
 31        # Add Jetbot
 32        stage_utils.add_reference_to_stage(
 33            usd_path=assets_root_path + "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
 34            path=f"{base_path}/Jetbot",
 35        )
 36        jetbot_xform = XformPrim(f"{base_path}/Jetbot")
 37        jetbot_xform.reset_xform_op_properties()
 38        jetbot_xform.set_world_poses(positions=[self.offset.tolist()])
 39
 40        # Add cube in front of Jetbot
 41        cube_pos = self.offset + np.array([0.15, 0.0, 0.025])
 42        visual_material = PreviewSurfaceMaterial(f"{base_path}/Materials/red")
 43        visual_material.set_input_values("diffuseColor", [1.0, 0.0, 0.0])
 44        cube_shape = Cube(
 45            paths=f"{base_path}/Cube",
 46            positions=np.array([cube_pos]),
 47            sizes=[1.0],
 48            scales=np.array([[0.05, 0.05, 0.05]]),
 49            reset_xform_op_properties=True,
 50        )
 51        GeomPrim(paths=cube_shape.paths, apply_collision_apis=True)
 52        RigidPrim(paths=cube_shape.paths)
 53        cube_shape.apply_visual_materials(visual_material)
 54
 55        # Add Franka
 56        franka_pos = self.offset + np.array([0.8, -0.3, 0.0])
 57        self.franka = FrankaExperimental(robot_path=f"{base_path}/Franka", create_robot=True)
 58        franka_xform = XformPrim(f"{base_path}/Franka")
 59        franka_xform.reset_xform_op_properties()
 60        franka_xform.set_world_poses(positions=[franka_pos.tolist()])
 61
 62    def initialize(self):
 63        """Initialize articulation handles after scene load."""
 64        base_path = f"/World/{self.name}"
 65        self.jetbot = Articulation(f"{base_path}/Jetbot")
 66        self.cube = RigidPrim(f"{base_path}/Cube")
 67
 68    def reset(self):
 69        """Reset the scenario state."""
 70        self.state = 0
 71        self.step_counter = 0
 72        self.pick_phase = 0
 73        self.franka.reset_to_default_pose()
 74
 75    def step(self):
 76        """Execute one step of the scenario logic."""
 77        if self.state == 0:
 78            # Jetbot pushes cube
 79            cube_pos = self.cube.get_world_poses()[0].numpy()[0]
 80            if np.linalg.norm(cube_pos[:2] - self.cube_goal[:2]) > 0.05:
 81                self.jetbot.set_dof_velocity_targets([[10.0, 10.0]])
 82            else:
 83                self.jetbot.set_dof_velocity_targets([[0.0, 0.0]])
 84                self.state = 1
 85                self.step_counter = 0
 86
 87        elif self.state == 1:
 88            # Jetbot backs up
 89            self.jetbot.set_dof_velocity_targets([[-8.0, -8.0]])
 90            self.step_counter += 1
 91            if self.step_counter > 100:
 92                self.jetbot.set_dof_velocity_targets([[0.0, 0.0]])
 93                self.state = 2
 94                self.step_counter = 0
 95                self.franka.open_gripper()
 96
 97        elif self.state == 2:
 98            # Franka pick-and-place
 99            self._franka_pick_place()
100
101    def _franka_pick_place(self):
102        """Execute Franka pick-and-place state machine."""
103        cube_pos = self.cube.get_world_poses()[0].numpy()[0]
104        down_orient = self.franka.get_downward_orientation()
105        self.step_counter += 1
106
107        if self.pick_phase == 0:
108            self.franka.set_end_effector_pose(np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.2]]), down_orient)
109            if self.step_counter > 120:
110                self.pick_phase = 1
111                self.step_counter = 0
112        elif self.pick_phase == 1:
113            self.franka.set_end_effector_pose(np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.1]]), down_orient)
114            if self.step_counter > 100:
115                self.franka.close_gripper()
116                self.pick_phase = 2
117                self.step_counter = 0
118        elif self.pick_phase == 2:
119            self.franka.close_gripper()
120            if self.step_counter > 50:
121                self.pick_phase = 3
122                self.step_counter = 0
123        elif self.pick_phase == 3:
124            self.franka.set_end_effector_pose(np.array([[cube_pos[0], cube_pos[1], cube_pos[2] + 0.25]]), down_orient)
125            if self.step_counter > 100:
126                self.pick_phase = 4
127                self.step_counter = 0
128        elif self.pick_phase == 4:
129            target = self.offset + np.array([0.3, 0.3, 0.15])
130            self.franka.set_end_effector_pose(np.array([target]), down_orient)
131            if self.step_counter > 150:
132                self.franka.open_gripper()
133                self.step_counter = 0
134                self.pick_phase = 5
135        elif self.pick_phase == 5:
136            # Lift the arm from target position (don't use cube_pos - cube was dropped)
137            target = self.offset + np.array([0.3, 0.3, 0.4])  # Lift above drop location
138            self.franka.set_end_effector_pose(np.array([target]), down_orient)
139            if self.step_counter > 150:
140                self.step_counter = 0
141                self.state = 5  # Done
142
143
144class HelloWorld(BaseSample):
145    def __init__(self) -> None:
146        super().__init__()
147        self._physics_callback_id = None
148        self._scenario = None
149
150    def setup_scene(self):
151        # Add ground plane
152        stage_utils.add_reference_to_stage(
153            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
154            path="/World/ground",
155        )
156        # Create a single scenario
157        self._scenario = RobotScenario(name="scenario_0", offset=np.array([0.0, 0.0, 0.0]))
158        self._scenario.setup_scene()
159
160    async def setup_post_load(self):
161        self._scenario.initialize()
162
163        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
164
165        self._physics_callback_id = SimulationManager.register_callback(
166            self.physics_step, IsaacEvents.POST_PHYSICS_STEP
167        )
168
169    def physics_step(self, dt, context):
170        self._scenario.step()
171
172    async def setup_post_reset(self):
173        self._scenario.reset()
174
175    def physics_cleanup(self):
176        if self._physics_callback_id is not None:
177            SimulationManager.deregister_callback(self._physics_callback_id)
178            self._physics_callback_id = None
```

## Scaling to Multiple Scenarios

```python
 1import isaacsim.core.experimental.utils.stage as stage_utils
 2import numpy as np
 3from isaacsim.core.experimental.materials import PreviewSurfaceMaterial
 4from isaacsim.core.experimental.objects import Cube
 5from isaacsim.core.experimental.prims import Articulation, GeomPrim, RigidPrim, XformPrim
 6from isaacsim.core.simulation_manager import SimulationManager
 7from isaacsim.examples.base.base_sample_experimental import BaseSample
 8from isaacsim.robot.manipulators.examples.franka import FrankaExperimental
 9from isaacsim.storage.native import get_assets_root_path
10
11# RobotScenario class definition (same as above)
12# ... (include the full RobotScenario class from the previous example)
13
14
15class HelloWorld(BaseSample):
16    def __init__(self) -> None:
17        super().__init__()
18        self._physics_callback_id = None
19        self._scenarios = []
20        self._num_scenarios = 2  # Number of parallel scenarios
21
22    def setup_scene(self):
23        # Add ground plane
24        stage_utils.add_reference_to_stage(
25            usd_path=get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd",
26            path="/World/ground",
27        )
28
29        # Create multiple scenarios with Y-axis offsets
30        for i in range(self._num_scenarios):
31            offset = np.array([0.0, (i - 1) * 2.0, 0.0])  # Spread along Y-axis
32            scenario = RobotScenario(name=f"scenario_{i}", offset=offset)
33            scenario.setup_scene()
34            self._scenarios.append(scenario)
35
36    async def setup_post_load(self):
37        # Initialize all scenarios
38        for scenario in self._scenarios:
39            scenario.initialize()
40
41        from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
42
43        self._physics_callback_id = SimulationManager.register_callback(
44            self.physics_step, IsaacEvents.POST_PHYSICS_STEP
45        )
46
47    def physics_step(self, dt, context):
48        # Step all scenarios
49        for scenario in self._scenarios:
50            scenario.step()
51
52    async def setup_post_reset(self):
53        # Reset all scenarios
54        for scenario in self._scenarios:
55            scenario.reset()
56
57    def physics_cleanup(self):
58        if self._physics_callback_id is not None:
59            SimulationManager.deregister_callback(self._physics_callback_id)
60            self._physics_callback_id = None
61        self._scenarios = []
```

## Adding Randomization

To make simulations more interesting, you can add randomization to the scenario parameters.
Modify the `RobotScenario` class to accept randomization options:

```python
 1class RobotScenario:
 2    """Encapsulates a Jetbot + Franka + Cube scenario with randomization."""
 3
 4    def __init__(self, name: str, offset: np.ndarray = np.array([0.0, 0.0, 0.0]), randomize: bool = False):
 5        self.name = name
 6        self.offset = offset
 7        self.randomize = randomize
 8        self.state = 0
 9        self.step_counter = 0
10        self.pick_phase = 0
11
12        # Randomize cube goal position if enabled
13        if randomize:
14            random_x = np.random.uniform(1.1, 1.4)
15            self.cube_goal = np.array([random_x, 0.0, 0.0]) + offset
16        else:
17            self.cube_goal = np.array([1.2, 0.0, 0.0]) + offset
18
19        # ... rest of the class remains the same
```

Then create scenarios with randomization enabled:

```python
1# Create multiple scenarios with randomization
2for i in range(self._num_scenarios):
3    offset = np.array([0.0, (i - 1) * 2.0, 0.0])
4    scenario = RobotScenario(name=f"scenario_{i}", offset=offset, randomize=True)  # Enable randomization
5    scenario.setup_scene()
6    self._scenarios.append(scenario)
```

## Best Practices for Scaling

When creating large-scale multi-robot simulations:

1. **Use unique paths**: Each scenario should use unique USD prim paths to avoid conflicts.
   The `RobotScenario` class uses the scenario name to create unique paths like
   `/World/scenario_0/Jetbot`.
2. **Manage state independently**: Each scenario instance maintains its own state variables,
   allowing scenarios to progress independently.
3. **Clean up properly**: The `physics_cleanup` method ensures callbacks are deregistered
   and scenario lists are cleared when the simulation is stopped.
4. **Consider performance**: With many scenarios, consider reducing physics step frequency
   or using GPU-accelerated simulation for better performance.

## Summary

This tutorial covered the following topics:

1. Organizing robot scenarios into reusable Python classes
2. Using the `offset` parameter to position multiple scenarios in the world
3. Scaling to multiple parallel scenarios with a simple loop
4. Adding randomization to scenario parameters
5. Best practices for managing multiple robot instances

---

# Adding Props

## Learning Objectives

This tutorial shows how to add objects to the scene and configure them for simulation.

*10-15 Minute Tutorial*

## Adding Rubik’s Cube

Start by adding a Rubik’s Cube to the scene.

1. Create a new stage on Isaac Sim by clicking on the **File** tab and then clicking on **New Stage**.
2. In the Content Browser, go to `Isaac Sim` > `Props` > `Rubiks_Cube` > `rubiks_cube.usd` and drag and drop the `rubiks_cube.usd` file into the stage. This will add a Rubik’s Cube to the scene as a payload.
3. Left click on the Rubik’s Cube and in the properties panel, set the `Position` to `(0, 0, 0.1)`.
4. On the stage, right click `Create` > `Isaac` > `Environment` > `Flat Grid` to create a flat ground.
5. Click `PLAY` to start the simulation, you will see the Rubik’s Cube is not falling to the ground. This is because the Rubik’s Cube is not a rigid body.
6. Click `STOP` to stop the simulation.

## Configure Physics Properties

### Add Rigid Body Properties

1. Right click on the Rubik’s Cube and select `Add` > `Physics` > `Rigid Body`. This will add a rigid body attribute to the Rubik’s Cube and it will be affected by physics.
2. Now, click `PLAY` to start the simulation, you will see the Rubik’s Cube fall through the ground, this is because the Rubik’s Cube does not have a collision shape. Click `STOP` to stop the simulation.

### Add Collision Properties

1. Right click on the Rubik’s Cube and select `Add` > `Physics` > `Collider Presets`. This will add a collision attribute to the Rubik’s Cube and it will collide with other objects.
2. Now, click `PLAY` to start the simulation, you will see the Rubik’s Cube fall on the ground. Click `STOP` to stop the simulation.

### Add Mass

In addition to collision, you can also add mass, inertia, and center of mass to the Rubik’s Cube to configure its physical properties.

1. Right click on the Rubik’s Cube and select `Add` > `Physics` > `Mass`. This will add a mass attribute to the Rubik’s Cube.
2. In the properties panel, scroll down to the `Mass` section and set the `Mass` to `0.1` to make it weigh 100 grams.

Note

In addition to mass, you can also set the `Density`, `Center of Mass`, `Diagonal Inertia`, and `Principal Axes` of the object.

Setting the mass to 0 will make the simulation to compute it at runtime based on its volume (assuming 1000 kg/m^3 if density is not specified).

### Visualize Collision Shapes

Right click on the `Eye` on the top left of the viewport and select `Show By Type` > `Physics` > `Coliders` > `All`. This will show the collision shapes everything in the scene.

The ground plane’s collider is pink to denote it is a static object. The Rubik’s Cube is a dynamic object, so it falls to the ground and its collider is green.

Note

You can adjust the collider type by left clicking on the `RubikCube` mesh at `World/rubiks_cube/RubikCube` and scroll down to the `Physics/Collider` section, and select a different approximate type in the `Approximation` tab.

### Customize Collider

Let’s customize the collider for the Rubik’s Cube, by making it a sphere and easier to roll

1. Left click on the `RubikCube` mesh at `World/rubiks_cube/RubikCube` and scroll down to the `Physics/Collider` section, press the `x` on the right to delete the current collider.
2. Left click on the `RubikCube` mesh and select `Create` > `Shape` > `Sphere`. This will add a sphere shape around the Rubik’s Cube.
3. Scroll down to the `Geometry` section and set the `Radius` to `0.07` to make the sphere smaller to match the Rubik’s Cube.
4. Add a Collider to the sphere by selecting `Add` > `Physics` > `Collider Presets`.
5. Hide the Sphere by unckecking the eye icon to the right of the sphere on the stage.
6. Slant the groundplane by going to `FlatGrid` and Click on `Toggle Offset Mode` icon on the right of `Transform` in the Properties panel, then setting the `Rotation` to `(10, 0, 0)` to give it a 10 degree slope.
7. Click `PLAY` to start the simulation, you will see the Rubik’s Cube rolls on the ground. Click `STOP` to stop the simulation.

### Add Physics Materials

You can also apply surface properties to the Rubik’s Cube by adding a physics material.

1. Left click on the Rubik’s Cube and in the properties panel, set the `Position` to `(0, 0, 1)` to move it up.
2. Right click on the Rubik’s Cube and select `Create` > `Physics` > `Physics Material`. This will add a physics material attribute to the Rubik’s Cube. Drag it to the `World/rubiks_cube/Looks` folder.
3. In the properties panel, scroll down to the `Physics Material` section and set the `Restitution` to `1` to make it bounce.
4. Select the `Sphere` collider we created earlier and in the properties panel, scroll down to the `Physics/Physics material on selected Material` section and select the `Physics Material` we just created at `/World/rubiks_cube/Looks/PhysicsMaterial`.
5. Click `PLAY` to start the simulation, you will see the Rubik’s Cube rolls on the ground and bounces. Click `STOP` to stop the simulation.

Note

You can also set the `Static Friction` and `Dynamic Friction` as well.

Note

The completed asset is available at `Isaac Sim` > `Samples` > `Rigging` > `RubiksCube` > `rubiks_cube.usd` in the Content Browser.

### Tips

* Object rigid body api should be applied to the default prim of the object.
* collision API should be applied to the mesh prim of the object, and it should be applied as a **physXSchema**

### What’s Next?

Extending from the concepts above, you assemble more complex collision shapes using basic shapes. For example, in the image below, we approximated a bearing collider using cylinders and rectangles.

## Summary

This tutorial covered the following topics:

1. Adding objects to the scene.
2. Configuring object physics properties.
3. Customize object collision shapes.
4. Apply physics materials to objects.

---

# Data Logging

## Learning Objectives

This tutorial shows how to record data using the DataLogger and play it back in NVIDIA Isaac Sim.
After this tutorial, you can record and play back states and actions in your pipeline while using NVIDIA Isaac Sim.

*10-15 Minute Tutorial*

## Recording Data

Let’s begin by using the following target extension example to record data

* Open the Follow Target example by first activating **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` window. Next enable the example from **Robotics Examples** > **Manipulation** > **Follow Target Task**.
* Click on **LOAD** under the World Controls to load Franka with a visual target cube.
* Choose the output directory of the JSON file to save the data under Data Logging in the **Follow Target** menu.
* Click on **Follow Target** under Task Controls.
* Click on **START LOGGING**
* Move the visual cube around so that Franka follows it using the RMPFlowController
* After a few seconds, click on **SAVE DATA**
* Click on **File > New From Stage Template > Empty** to create new stage.
* The data should be recorded under the chosen file shown in the **Output Directory** text field.

### Code Overview

Open the extension example code located at ISAAC\_SIM\_DIR/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/follow\_target/follow\_target.py using the open-source code button located at the top of the menu.

First, let’s look at the logging function.

> ```python
> def _on_logging_event(self, val):
>     world = self.get_world()
>     data_logger = world.get_data_logger()  # a DataLogger object is defined in the World by default
>     if not world.get_data_logger().is_started():
>         robot_name = self._task_params["robot_name"]["value"]
>         target_name = self._task_params["target_name"]["value"]
>
>         # A data logging function is called at every time step index if the data logger is started already.
>         # We define the function here. The tasks and scene are passed to this function when called.
>
>         def frame_logging_func(tasks, scene):
>             return {
>                 "joint_positions": scene.get_object(robot_name)
>                 .get_joint_positions()
>                 .tolist(),  # save data as lists since its a JSON file.
>                 "applied_joint_positions": scene.get_object(robot_name).get_applied_action().joint_positions.tolist(),
>                 "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
>             }
>
>         data_logger.add_data_frame_logging_func(
>             frame_logging_func
>         )  # adds the function to be called at each physics time step.
>     if val:
>         data_logger.start()  # starts the data logging
>     else:
>         data_logger.pause()
>     return
> ```

Now let’s look at how to save the data collected so far.

> ```python
> def _on_save_data_event(self, log_path):
>     world = self.get_world()
>     data_logger = world.get_data_logger()  # a DataLogger object is defined in the World by default
>     data_logger.save(log_path=log_path)  # Saves the collected data to the json file specified.
>     data_logger.reset()  # Resets the DataLogger internal state so that another set of data can be collected and saved separately.
>     return
> ```

### Inspect the Data

As shown below, the DataLogger takes care of logging the time in seconds and time step corresponding to each data frame.

> ```python
> {
>     "Isaac Sim Data": [
>         {
>             "current_time": 1.4833334106951952,
>             "current_time_step": 89,
>             "data": {
>                 "joint_positions": [
>                     0.07561380416154861,
>                     -1.2318825721740723,
>                     0.11344202607870102,
>                     -2.4259397983551025,
>                     0.0970514565706253,
>                     1.6226640939712524,
>                     0.8470714688301086,
>                     4.0,
>                     3.997776985168457,
>                 ],
>                 "applied_joint_positions": [
>                     0.07291083037853241,
>                     -1.2202218770980835,
>                     0.1190749853849411,
>                     -2.39223575592041,
>                     0.11230156570672989,
>                     1.3975754976272583,
>                     0.9029524326324463,
>                     4.0,
>                     4.0,
>                 ],
>                 "target_position": [0.0, 10.0, 70.0],
>             },
>         },
>         {
>             "current_time": 1.5000000782310963,
>             "current_time_step": 90,
>             "data": {
>                 "joint_positions": [
>                     0.07484950870275497,
>                     -1.2287049293518066,
>                     0.11509127914905548,
>                     -2.416816234588623,
>                     0.09880664199590683,
>                     1.603981614112854,
>                     0.8490884304046631,
>                     4.0,
>                     3.997793197631836,
>                 ],
>                 "applied_joint_positions": [
>                     0.07221028953790665,
>                     -1.2172484397888184,
>                     0.1205318346619606,
>                     -2.3833296298980713,
>                     0.11395926028490067,
>                     1.3804354667663574,
>                     0.9063650369644165,
>                     4.0,
>                     4.0,
>                 ],
>                 "target_position": [0.0, 10.0, 70.0],
>             },
>         },
>         {
>             "current_time": 1.5166667457669973,
>             "current_time_step": 91,
>             "data": {
>                 "joint_positions": [
>                     0.07410304993391037,
>                     -1.2255841493606567,
>                     0.11668683588504791,
>                     -2.4077510833740234,
>                     0.10055229812860489,
>                     1.58543062210083,
>                     0.8511277437210083,
>                     4.0,
>                     3.997816562652588,
>                 ],
>                 "applied_joint_positions": [
>                     0.07153353840112686,
>                     -1.2144113779067993,
>                     0.1219213530421257,
>                     -2.3745198249816895,
>                     0.11559101194143295,
>                     1.363703727722168,
>                     0.9096996784210205,
>                     4.0,
>                     4.0,
>                 ],
>                 "target_position": [0.0, 10.0, 70.0],
>             },
>         },
>     ]
> }
> ```

## Replaying Back Data

Similarly, we will use another extension example provided under **Robotics Examples** to play back the recorded data.

* Open the Follow Target example by first activating **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` window. Next enable the example from **Robotics Examples** > **Manipulation** > **Replay Follow Target Task**.
* Click on **LOAD** under the World Controls to load Franka with a visual target cube.
* Point to the JSON file saved in the last step in **Recording Data** section under **Data File** at the Data Replay section in the menu.
* Click on **Replay Trajectory** under Data Replay to replay the actions only and wait for the trajectory to finish replaying.
* Click on **Reset**
* Similarly, click on **Replay Scene** under Data Replay to replay the actions and the cube position.
* Click on **File > New From Stage Template > Empty** to create new stage.

### Code Overview

Open the extension example code using the open-source code button located at the top of the menu located at:

ISAAC\_SIM\_DIR/exts/isaacsim.examples.interactive/isaacsim/examples/interactive/replay\_follow\_target/replay\_follow\_target.py

First, let’s look at how to load the data and replay the trajectory.

> ```python
> def _on_logging_event(self, val):
>     world = self.get_world()
>     data_logger = world.get_data_logger()  # a DataLogger object is defined in the World by default
>     if not world.get_data_logger().is_started():
>         robot_name = self._task_params["robot_name"]["value"]
>         target_name = self._task_params["target_name"]["value"]
>
>         # A data logging function is called at every time step index if the data logger is started already.
>         # We define the function here. The tasks and scene are passed to this function when called.
>
>         def frame_logging_func(tasks, scene):
>             return {
>                 "joint_positions": scene.get_object(robot_name)
>                 .get_joint_positions()
>                 .tolist(),  # save data as lists since its a JSON file.
>                 "applied_joint_positions": scene.get_object(robot_name).get_applied_action().joint_positions.tolist(),
>                 "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
>             }
>
>         data_logger.add_data_frame_logging_func(
>             frame_logging_func
>         )  # adds the function to be called at each physics time step.
>     if val:
>         data_logger.start()  # starts the data logging
>     else:
>         data_logger.pause()
>     return
> ```

Now let’s look at how to replay the scene, including the goal cube position.

> ```python
> def _on_replay_scene_step(self, step_size):
>     if self._world.current_time_step_index < self._data_logger.get_num_of_data_frames():
>         target_name = self._task_params["target_name"]["value"]
>         data_frame = self._data_logger.get_data_frame(data_frame_index=self._world.current_time_step_index)
>         self._articulation_controller.apply_action(
>             ArticulationAction(joint_positions=data_frame.data["applied_joint_positions"])
>         )
>         # Sets the world position of the goal cube to the same recorded position
>         self._world.scene.get_object(target_name).set_world_pose(position=np.array(data_frame.data["target_position"]))
>     return
> ```

## Summary

This tutorial covered the following topics:

1. Using the DataLogger to save data.
2. Using the DataLogger to replay data.