# Physics

On a high-level, simulations with Omniverse™ Physics work as follows:

* The USD Physics schema of robot and environment assets are parsed and corresponding simulation objects are created in the selected physics backend.
* Then, for each discrete-time step of the simulation, Physics advances the simulation objects given their current state and additional inputs such as, for example, control-policy torques.
* The updated state is written back to USD by default, where the state can be further processed by the user, a reinforcement-learning policy, or other extensions such as the Omniverse RTX Renderer.
* Omniverse™ Physics propagates runtime changes to physics parameters in USD to the physics objects.

Isaac Sim supports multiple physics backends: the default PhysX SDK backend and the experimental Newton backend.

## Tools

## Additional Resources

* Omniverse™ Physics [core documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html "(in Omni Physics)") and [programming guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html)
* [USD Physics Schemas](https://openusd.org/release/api/usd_physics_page_front.html) and PhysX SDK-engine-specific [Physx Schemas](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/annotated.html)
* Explore further Omniverse [simulation extensions](https://docs.omniverse.nvidia.com/extensions/latest/ext_simulation.html#simoverview "(in Omniverse Extensions)").
* [PhysX SDK](https://nvidia-omniverse.github.io/PhysX/physx/5.4.2/index.html)
* [Omniverse Visual Debugger](https://nvidia-omniverse.github.io/PhysX/physx/5.4.2/docs/OmniVisualDebugger.html)
* [Flow: Fluid Dynamics](https://docs.omniverse.nvidia.com/extensions/latest/ext_fluid-dynamics.html "(in Omniverse Extensions)")
* [NVIDIA Warp](https://nvidia.github.io/warp/index.html)

---

# Physics Simulation Fundamentals

## Physics in USD Schemas

The physics properties of assets are all well-defined using [USD Physics Schemas](https://openusd.org/release/api/usd_physics_page_front.html) and [Physx Schemas](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/annotated.html) . The documentation of Physics properties and how to access them in code is defined in C++, but you can follow [these guidelines](https://developer.nvidia.com/usd/apinotes) to find the equivalent calls in Python. For example, where generic names are used to represent an arbitrary API, the general usage is:

```python
import omni.usd
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
prim = stage.GetPrimAtPath("/Path/To/Prim")
physics_api_prim = UsdPhysics.SomePhysicsAPI(prim)
physx_api_prim = PhysxSchema.AnotherPhysxAPI(prim)

# Check if the API is Applied, if not, Apply it.
if not physics_api_prim:
    physics_api_prim = UsdPhysics.SomePhysicsAPI.Apply(prim)

physics_attr = physics_api_prim.GetSomePhysicsAttr()
physx_attr = physx_api_prim.GetPhysxAttr()

# Check if Attribute is authored, otherwise create it
if not physics_attr:
    physics_attr = physics_api_prim.CreateSomePhysicsAttr(1.0)
print(physics_attr.Get())
physics_attr.Set(10.0)
```

In some cases, you may need to have additional parameters when casting the Prim to a given API, for example, [Joint State](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_joint_state_a_p_i.html#afff2009176797852a1389d7244caa875) does require the joint type (“Prismatic”, or “Angular”, for instance). In these cases the C++ signature will contain a “TfToken” type. Replace it with a basic string and it should work in Python.

If you need to know the attribute name of some physics attribute you see on the UI, Hover over the attribute in the properties panel, and it will show its name in the tooltip. The attribute name standard is `schema_name:attribute_name`, so for example something like `physics:velocity` on a rigid body means it’s using the Physics Rigid Body API and the attribute name is `velocity`, so the corresponding attribute getter would be `UsdPhysics.RigidBodyAPI(prim).GetVelocityAttr()`.

## Simulation Timeline

Simulation time **differs** from real-time. Depending on system configuration and the size of the simulated environment, each time step may be computed faster or slower than the time it’s simulating, resulting in a warped speed if results are presented sequentially (often, physics simulation in Isaac Sim is faster than real-time). To mitigate this, Isaac Sim is configured by default with a limiter to match real-time speed.

Moreover, the simulation may run at a faster pace than rendering, meaning there may be more than one simulation time-step occurring in the background for every rendered frame. In the simplified example below, the simulation is set to run at 120 time steps per second, while rendering is set to 60 frames per second, resulting in two physics steps per rendered frame:

Note

The physics step time doesn’t necessarily coincide with system time (from the simulation start). In cases where the simulation can run faster than real-time, it’s possible to run an accelerated version of the simulation in a timeline without rendering or frame-rate blocking.

Ideally, simulation and rendering would match or be multiples of each other, but when this isn’t the case, each rendered frame may contain an uneven number of simulation timesteps. For example, simulation set to 100 steps per second, rendering set to 30 frames per second, resulting in most render updates having 3 simulation steps but occasionally 4 in a frame.

There are three event streams on the timeline (among a few others, but these are notably the most relevant for Isaac Sim). You can subscribe directly to Simulation Events or to Frame update events, either pre or post-rendering. OmniGraph nodes are typically updated on a pre-render event, but there are ways to set them to update on different events, such as every physics step.

### Configuring Frame Rate

The simulation rendering frame rate can be configured by adjusting the current stage metadata. In the **Layer** tab, select the **Root Layer**, and in the properties panel modify the **Timecodes per second** property.

### Configuring Simulation Timesteps

Simulation steps per second are determined in the Physics Scene. If there’s no Physics scene in your stage, it uses the default value, which is 60 steps per second.

To add a Simulation Scene element:

1. Click on **Create** > **Physics** > **Simulation Scene**.
2. Select the Simulation scene.
3. In the Properties panel check the element **Simulation Steps per Second**.

For more details on other parameters in the Physics Scene, refer to [Omniverse Physics Developer Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html "(in Omni Physics)").

## Simulation Components

Prims in a USD stage do not have physics enabled by default, but you may add simulation properties through the UI or using Python scripts. The following creates an example scene to which elements are added as we progress through the basic physics object types.

To begin, create a new scene and add a ground plane to it: **File** > **New**, then **Create** > **Physics** > **Ground Plane**.

### Rigid Body

This is the most basic element. Adding rigid body dynamics enables an element to be subject to gravitational acceleration and other external forces.

1, Add a container to use as our Rigid Body: **Create** > **Xform**.
2. Move it up to `Z=10` in the properties panel.
3. To make it a rigid body, right-click on it in the stage, then **Add** > **Physics** > **Rigid Body**.

Verify that the Xform is now be a rigid body, although you may not see much because it has no visual meshes.

You can fix that by nesting a Cube in it:

1. **Create** > **Mesh** > **Cube**, and drag it into the Xform.
2. Ensure the cube’s Translate is set to [0,0,0.5].

After you’ve completed the same setup as the screenshot above, hit play and see what happens:

Review the following:

* Notice how the Z position gets updated as the object falls - this is because we are highlighting the rigid body directly. Try again selecting the cube, and you’ll notice that it doesn’t change.
* The cube falls straight through the ground. We need to let the simulation know it needs to collide with other objects.

### Colliders

To make our rigid body collide, you must indicate to the simulation that you want it to. For that, there’s the Collider API.

1. Select the Cube prim, and click on the **Add Button** > **Physics** > **Collider**.
2. Run the simulation again and verify that the rigid body stops at the ground.

Colliders can also be added to non-movable objects. Let’s experiment:

1. Create a new cube and place it at Z=3.0.
2. Then change its scale to [2,2,0.01] to create a 2x2 meter platform.
3. Add the collider to it just like before, without adding the Rigid body.

Play the simulation again, and verify that this is the result:

Raise the Xform position to `Z=80`.
Play the simulation again.

With this example, you are solving some of the common issues of physics simulation. Because time is discretized, if objects move too fast, during one time-step the object is above the platform, and in the next it has completely passed through it, with no collision captured.
This doesn’t occur with the ground plane because it implements a “force field” that pushes penetrated objects towards the ground surface.

To remedy this, enable an option in the physics scene called **Enable CCD** (Continuous Collision Detection). CCD sweeps the object from one pose to the next. This option must also be enabled in the rigid body itself:

1. Select the Xform.
2. In the properties panel, enable CCD under the rigid body properties.

There are other ways to solve this issue, but for this scenario, this is the most effective.

Remember that collision has nothing to do with what you see on screen. For instance, you could hide the cube and the collider would behave the same, or you could add another cube or a sphere under Xform and it would have no effect unless you apply the Collision API to it.

Many object colliders are made using a composition of multiple mesh elements, giving it its shape and behavior. They work as a single rigid body even if they are physically separated on the stage, as long as they are all children of a rigid body.

Try adding and removing colliders to this rigid body or adding more rigid bodies to this scene and see how they behave.

#### Convex Hull

This next experiment with colliders removes the platform you added before and returns our Xform to `Z=10`.

1. Add a Torus mesh in the place of the platform at `Z=3.0` and scale it to [5.0, 5.0, 5.0].
2. **Add** > **Physics** > **Rigid Body With Colliders Preset**.
3. Run the simulation.

The Cube sits on top of the torus hole because the default approximation for mesh geometry is a convex hull. This is an approximation that the simulation engine can process efficiently, i.e. they are a good choice for performant simulations. We will review more complex, and therefore more computationally expensive approximations below.

To see the collision shape in use:

1. Click on the eye icon on the top-left side of the Viewport.
2. **Show by type** > **Physics** > **Colliders** > **Selected**.
3. Verify that green lines appear on the Torus.

This is a debug view of the collision shape.

You can also view a solid display of the colliders by opening the Physics debug menu:
1. **Window** > **Simulation** > **Debug**.
2. In the debug window, scroll to “Collision Mesh Debug Visualization”.
3. Check “Solid Mesh Collision Visualization”.
4 Verify that when you select the torus, its shape displays solidly.

|  |  |
| --- | --- |
| ../_images/isaac_sim_torus.png | ../_images/isaac_sim_torus_collision.png |

#### Convex Decomposition

At a small expense, the torus collider can have the hole by a composition of convex shapes. This composition can be:

* manually created by adding multiple shapes
* computed with Physics Convex Decomposition

1. Select the Torus.
2. In the properties panel, scroll down to the Collision section, and select **Convex Decomposition** from the drop-down.
3. By opening the Advanced tab, you can adjust the parameters until you find a decomposition to your satisfaction.

Note

Fewer convex hulls typically results in higher performance.

In the Simulation Debug tab, you can also increase the Explode View distance to split the collider shapes and better understand how the composition is made.

The Collider drop-down contains more options to explore, like Bounding Cube and Sphere - the cheapest collisions possible, and a mode “Sphere Approximation”, which is similar to Convex decomposition but directly uses a group of spheres instead of conforming meshes.

Note

While triangle mesh and mesh simplification are not supported by rigid bodies and fall back to convex hull, it is possible to use a triangle mesh geometry directly on a rigid body by adding a signed-distance field to it; select **SDF Mesh** in the approximation drop-down to do so.

For more details on Rigid Bodies and Colliders, check [Rigid Body Simulation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html "(in Omni Physics)") and [Colliders](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html "(in Omni Physics)").

#### Contact and Rest Offset

In the Collider Advanced tab, there are two more parameters that can be important tuning parameters when there are collision issues, in particular with small and thin objects.

The Rest Offset can be tuned to inflate or shrink the collision geometry set; it can be useful to adjust in cases where the visual mesh is larger or smaller than the collision geometry so that the collision locations are consistent with the visual representation.

The Contact Offset dictates how far from the collision geometry, irrespective of Rest Offset, the simulation engine starts generating contact constraints. The tradeoff for tuning the contact offset is performance vs. collision fidelity: A larger Contact Offset results in many contact constraints being generated which is more computationally expensive; a smaller offset can result in issues with contacts being detected too late, and symptoms include jittering or missed contacts or even tunneling (see notes on CCD above).

### Contacts and Friction

Besides making sure that object do not interpenetrate, collisions can transfer or dissipate energy as modeled by restitution and friction.

The parameters for the contact model are available in Physics materials. To create a Physics material:

1. Go to **Create** > **Physics** > **Physics Material**.
2. Select Rigid Body Material.

Physics materials are typically assigned to **Collider Geometry** but behave analogous to USD render materials otherwise; see [Physics Materials](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html "(in Omni Physics)") for a full explanation of USD material resolution logic. For example, you may assign different materials to different collision geometry of a rigid body, or you may assign a material to the rigid body prim and configure it to override any materials set on the collider children.

To assign a physics material:

1. Select the collider prim.
2. Scroll to the Collider settings.
3. In Physics Materials on Selected Models, select the desired material. The list only allows picking materials that have physics properties.

Note that you may also add a physics material to a render material with **“Add”** > **Physics** > **Rigid Body Material** and assign the material in the render material section; the physics properties will be picked up.

#### Compliant Contacts

You may configure the rigid material to produce compliant (i.e. spring-damper) contact dynamics in the Advanced tab. This may be useful for approximating deformable bodies with rigid bodies.

#### Combine Modes

Because contacts are an interaction between two bodies, each contact parameter is not enough to describe how this interaction plays out. Just like in the real world, one surface material property may dominate the interaction or they may seamlessly combine into an average value. To replicate that, friction, restitution, and compliant-contact damping have a configurable combine mode field. Because both sides of the contact have this combine mode, the precedence of the combine mode matters:

The lower in the drop-down, the lower the priority of a mode in a combine mismatch resolution; so `average < min < multiply < max`.

For example, if Collider A has a friction combine mode average while Collider B has min, their interaction resolves as the minimum friction between the two. If a body C with combine mode max contacts A and B, the friction between A and C are resolved with max, as well as B and C.

### Joints

Robots are typically composed of multiple jointed rigid bodies. Joints create constraints between two bodies. In the following, you use a **Revolute Joint**, but the steps are similar for other joint types, see a list in [Joints](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html#joints "(in Omni Physics)").

You must configure the relative pose of the joint frames for each body to be jointed. Find more details, in particular the local scaling aspect of joint frames in the [Joint Frames Section](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html#jointframes "(in Omni Physics)").

Note that when creating a joint through the UI, the joint’s frames are set to match the pose of the second rigid body selected for the creation.

Now create a joint as follows:

1. Select first the Xform rigid body, and then the Torus rigid body.
2. Go to **Create** > **Physics** > **Joints** > **(Joint Type)**.

For this tutorial, use the **Revolute Joint** type. Because the Torus was selected second, the joint is at its center.

You will notice a circle on-screen, representing the origin and range of motion for the joint. If you start the simulation now, the Torus and Cube fall together. When the torus hits the ground, the cube stops moving. It’s in a stable position, but if you nudge it, it moves down in a circular pattern. Interact with the cube by pressing shift and left-clicking the cube.

Check the properties panel and review the following attributes:

1. Body 0: /World/XForm
2. Body 1: /World/Torus

These are the Poses relative to the bodies. You will notice that Position 0 is `Z=-7.0`.

1. Position 0: `[0, 0, -7.0]`
2. Rotation 0: `[0, 0, 0.0]`
3. Position 1: `[0, 0, 0.0]`
4. Rotation 1: `[0, 0, 0.0]`

Note

When setting up joints that are part of an articulation, make sure that Body 0 will be the parent of Body 1 in the articulation-tree hierarchy. This way, joint-related quantities like link incoming joint forces or joint drive targets have a one-to-one correspondence in the PhysX SDK and USD.

#### Joint Axis

A revolute joint provides one degree of freedom and you may choose what axis of the joint frames is free. By default, the X axis is selected. You can change that in Properties, under the Revolute Joint section.

#### Joint Limits

The joint limits determine how far the joint can move from its original position. By default, when a joint is created, it comes without limits. With the joint selected, scroll down in the Properties panel and modify the Lower Limit and Upper Limit under the Revolute Joint section. Remember that USD uses degrees, not radians to represent angles.

#### Adding a Joint Drive

You may control the position and velocity of the degree of freedom that the joint added using a Joint Drive. You can do that by clicking the **Add Button** > **Physics** > **Angular Drive**. For details on configuring a joint drive, refer to [Tutorial 11: Tuning Joint Drive Gains](Robot_Setup.md).

## Articulation

An articulation is an optimized simulation structure for jointed bodies that provides superior performance, fidelity, and features for robotics. There are some limitations regarding topology (loop-closing) and joint support, which you can learn about in [Articulations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html#articulations "(in Omni Physics)"). For a complete guide in tuning articulations, refer to [Articulation Stability Guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html).

For overall Simulation Hints and FAQ, refer to [Physx Simulation Hints and FAQ](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides.html).

## Stepping an OmniGraph with Physics

To guarantee one graph step per physics step at the moment it happens, you must use a modified version of an OmniGraph.

1. Create a new Action Graph through **Create** > **Visual Scripting** > **Action Graph**.
2. Select the created graph on the stage and in the **Raw USD Properties** section, in the pipeline stage, select *PipelineStageOnDemand*.
3. On the Action Graph window, search for **On Physics Step**. Drag and Drop it on your OmniGraph.
4. Continue your OmniGraph as usual.

## Simulation Residuals

The physics simulation provides a metric to check how well it converged to a solution, i.e. how well it resolved constraints. To check for this result there is another API that can be applied to a few physics elements.

To check the Residuals:

1. Click on the selected physics element.
2. **“Add”** > **Physics** > **Residual Reporting**.
3. Verify that you can see the Residual plot over time on the Simulation Data Visualizer: **(eye icon on viewport)** > **Show by Type** > **Physics** > **Simulation Data Visualizer**.

The types of Physics Objects that report residuals are Simulation Scenes, the Articulation Roots, and Joints.

---

# Newton Physics Backend

Isaac Sim supports multiple physics backends. In addition to the default PhysX SDK backend, you can now use Newton as the simulation backend.

[Newton](https://newton-physics.github.io/newton/) is a GPU-accelerated, extensible, and differentiable physics simulation engine designed for robotics and research. Built on [NVIDIA Warp](https://nvidia.github.io/warp/) and integrating [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), Newton provides high-performance simulation with multiple solver implementations including XPBD, MuJoCo, Featherstone, and SemiImplicit. Newton is an open-source project maintained by Disney Research, Google DeepMind, and NVIDIA.

Note

Newton integration in Isaac Sim is experimental. The API and features may change in future releases.

## Overview

Isaac Sim integrates Newton through three key extensions:

* **isaacsim.physics.newton**: The Newton physics backend implementation that:

  + Parses USD stage from your scene and builds Newton simulation objects
  + Synchronizes simulation state with Fabric for rendering and data access
  + Provides a tensor-based API (`isaacsim.physics.newton.tensors`) compatible with NumPy, PyTorch, and Warp
  + Registers Newton with the unified physics interface
* **isaacsim.core.simulation\_manager**: `SimulationManager` provides functionality for switching between physics engines at runtime, along with scene configuration classes (`PhysicsScene`, `NewtonMjcScene`) for Newton-specific settings.
* **isaacsim.core.experimental.prims**: Uses `isaacsim.physics.newton.tensors` as its tensor backend when Newton is active. This extension provides engine-agnostic prim wrappers that work consistently across all physics backends.

When Newton is active, it replaces PhysX as the simulation backend while maintaining compatibility with standard USD Physics schemas used by your robot and environment assets.

### Using the Experimental Core API

The `isaacsim.core.experimental` extension provides engine-agnostic building blocks that ensure compatibility across different physics backends. User extensions and applications are highly recommended to use `isaacsim.core.experimental` to write simulation code that works with all physics backends (PhysX, Newton). See the [Core Experimental API documentation](../py/docs/overview/experimental.html) for more details.

## Launching Isaac Sim with Newton

You can launch Isaac Sim with Newton as the default physics backend using the dedicated application file.

Linux

```python
./isaac-sim.newton.sh
```

Windows

```python
isaac-sim.newton.bat
```

When launched with this application, Newton is automatically enabled and PhysX is disabled.

## Switching Physics Engines at Runtime

You can switch between physics engines programmatically using the `SimulationManager` class. Use `get_available_physics_engines()` to list registered engines and `switch_physics_engine()` to activate Newton:

```python
from isaacsim.core.simulation_manager import SimulationManager

engines = SimulationManager.get_available_physics_engines(verbose=True)
success = SimulationManager.switch_physics_engine("newton")
if success:
    print("Switched to Newton physics engine")
```

Note

Switching physics engines should be done before starting the simulation. The switch deactivates the previous engine and activates the new one.
Currently, only one physics engine can be active at a time.

## Basic Usage Example

The following example demonstrates setting up a simple physics scene with Newton:

```python
import omni.kit.actions.core
import omni.timeline
import omni.usd
from isaacsim.core.experimental.objects import Cube, Plane
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene
from pxr import Sdf, UsdGeom, UsdLux, UsdPhysics

omni.usd.get_context().new_stage()
SimulationManager.switch_physics_engine("newton")
stage = omni.usd.get_context().get_stage()

# Enable camera light and add distant light
action_registry = omni.kit.actions.core.get_action_registry()
action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera").execute()
UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(500)

# Create physics scene
mjc_scene = NewtonMjcScene("/World/PhysicsScene")
mjc_scene.set_gravity((0.0, 0.0, -9.81))

# Create ground plane (collision + visual)
UsdGeom.Xform.Define(stage, "/World/GroundPlane")
Plane("/World/GroundPlane/CollisionPlane", axes="Z")
UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/World/GroundPlane/CollisionPlane"))
visual_mesh = UsdGeom.Mesh.Define(stage, "/World/GroundPlane/VisualMesh")
size = 50.0
visual_mesh.CreatePointsAttr([(-size, -size, 0), (size, -size, 0), (size, size, 0), (-size, size, 0)])
visual_mesh.CreateFaceVertexCountsAttr([4])
visual_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
visual_mesh.CreateDisplayColorAttr([(0.5, 0.5, 0.5)])

# Create dynamic cube
Cube("/World/Cube", sizes=0.5, positions=[[0.0, 0.0, 2.0]])
cube_prim = stage.GetPrimAtPath("/World/Cube")
UsdPhysics.CollisionAPI.Apply(cube_prim)
UsdPhysics.RigidBodyAPI.Apply(cube_prim)

# Start simulation
timeline = omni.timeline.get_timeline_interface()
timeline.play()
```

## Scene Configuration

### Newton USD Schemas

Newton uses custom USD schemas to configure physics scenes. The [Newton USD Schemas](https://github.com/newton-physics/newton-usd-schemas) project provides extensions to OpenUSD’s UsdPhysics specification, allowing USD layers to fully specify Newton runtime parameters. These schemas follow a minimalist approach, capturing parameters that generalize across simulators and have clear physical meaning.

The key schemas include:

* **NewtonSceneAPI**: Base Newton schema applied to all physics scenes, providing common attributes like timestep (`newton:timeStepsPerSecond`), gravity settings, and solver iterations.
* **MjcSceneAPI**: MuJoCo solver-specific schema with integrator type, constraint solver algorithm, tolerance, and contact settings.

### PhysicsScene Base Class

The `PhysicsScene` class provides a Python interface to the `NewtonSceneAPI` schema attributes. When you create a `PhysicsScene`, it automatically applies the `NewtonSceneAPI` to the underlying USD prim, allowing you to configure common Newton settings:

```python
from isaacsim.core.simulation_manager import PhysicsScene

physics_scene = PhysicsScene("/World/PhysicsScene")
physics_scene.set_gravity((0.0, 0.0, -9.81))
physics_scene.set_dt(0.001)
physics_scene.set_enabled_gravity(True)
physics_scene.set_max_solver_iterations(100)
```

### MuJoCo Solver Configuration

MuJoCo-specific parameters can be stored in USD through the MJC USD schemas, which capture settings for scenes, bodies, joints, and other elements. The `MjcSceneAPI` is one of these schemas, providing scene-level simulation parameters. The `NewtonMjcScene` class provides a Python interface to the `MjcSceneAPI` attributes, allowing you to configure MuJoCo solver settings directly on USD Physics Scene prims.

When you create a `NewtonMjcScene`, it applies both `NewtonSceneAPI` and `MjcSceneAPI` to the prim:

```python
from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene

mjc_scene = NewtonMjcScene("/World/PhysicsScene")
mjc_scene.set_dt(0.002)
mjc_scene.set_integrator("implicit")  # euler, rk4, implicit, implicitfast
mjc_scene.set_solver("newton")  # pgs, cg, newton
mjc_scene.set_iterations(100)
mjc_scene.set_tolerance(1e-8)
mjc_scene.set_cone("elliptic")  # pyramidal, elliptic
```

Note

Additional engine-specific scene classes to incorporate other solver-specific schemas (XPBD, Featherstone) are under development and will be available in future releases.

### Robot Simulation Example

The following example loads a Franka robot and simulates it with Newton:

```python
import omni.kit.actions.core
import omni.timeline
import omni.usd
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.mjc_scene import NewtonMjcScene
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Sdf, UsdLux

omni.usd.get_context().new_stage()
SimulationManager.switch_physics_engine("newton")
stage = omni.usd.get_context().get_stage()

action_registry = omni.kit.actions.core.get_action_registry()
action_registry.get_action("omni.kit.viewport.menubar.lighting", "set_lighting_mode_camera").execute()
UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight")).CreateIntensityAttr(500)

mjc_scene = NewtonMjcScene("/World/PhysicsScene")
mjc_scene.set_dt(0.002)
mjc_scene.set_integrator("implicit")
mjc_scene.set_gravity((0.0, 0.0, -9.81))

asset_path = get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
add_reference_to_stage(usd_path=asset_path, prim_path="/World/Franka")

timeline = omni.timeline.get_timeline_interface()
timeline.play()
```

The physics engine selector in the viewport menu.

To compare simulation results between Newton and PhysX: stop the simulation, switch the physics engine from “newton” to “physx” using the menu shown above, and play the simulation again.

## Asset Compatibility

Existing PhysX-based assets in Isaac Sim are compatible with Newton. However, these assets are tuned for PhysX and may not produce optimal results with Newton/MuJoCo out of the box. Users may need to adjust physics parameters (including: contact settings, solver iterations, timestep) to achieve desired simulation behavior with Newton/MuJoCo.

With the new asset structure and MJCF/URDF importers, we are working toward converting each asset to both PhysX schemas and MJC USD schemas. This will enable consistent simulation behavior between the original MJCF asset (using MuJoCo) and the converted MJC USD asset (using Newton).

## Additional Resources

* [Newton Physics Documentation](https://newton-physics.github.io/newton/)
* [Newton USD Schemas](https://github.com/newton-physics/newton-usd-schemas)
* [NVIDIA Warp Documentation](https://nvidia.github.io/warp/)
* [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp)

---

# Omniverse™ Physics and PhysX SDK Limitations

This section provides a table of known limitations with the PhysX engine and workarounds.

PhysX Engine Limitations

| Feature | Description of Limitation | Recommended Workaround | IsaacSim GPU RL Perf Impact | Perf Impact Reason |
| --- | --- | --- | --- | --- |
| Custom Geometry GPU pipeline contact reports | Custom geometry contact data is not made available via the tensor API GPU contact information getters. | Use GPU native collision approximation like convex hull, SDF tri-mesh, base geoms like sphere/box/capsule. Disable custom geometry cylinders and cones in physics stage settings. | Yes | Contacts are generated by CPU custom geometry callbacks |
| Custom Geometry collision against GPU Features | Custom geometry does not collide with GPU features (particles and deformable bodies), and may have poor collision quality against SDF tri-mesh colliders. | Use GPU native collision approximation like convex hull, SDF tri-mesh, base geoms like sphere/box/capsule. Disable custom geometry cylinders and cones in physics stage settings. | Yes | Contacts are generated by CPU custom geometry callbacks |
| Particles and deformable body contact reports | Particles and deformable body do not support contact reports | NA | No |  |
| Simulation-resume determinacy | Replaying a simulation from an in-contact simulation state saved out in the middle of a simulation run can be nondeterministic. The PhysX SDK is using internal contact state that can persist over multiple simulation steps and that cannot currently be serialized and recovered from USD data. | Restart simulations from the beginning to achieve determinism. | No | NA |
| Conveyor Belts / Kinematics with nonzero velocity | * Deformable bodies and particles do not support conveyor belts and will not contact/fall through. * Collision behavior between conveyor belts and SDF tri-mesh dynamics is inadequate | Do not use the conveyor belt feature with particles, deformable bodies, or SDF tri-mesh collision geometry. Use rigid bodies with non-SDF collision geometry instead. | Yes | The conveyor belt feature will trigger a CPU code path, so it is best to avoid for maximum GPU pipeline performance. |
| Isosurface | Isosurface may be leaking memory | Do not use isosurface feature if memory leaks are an issue. It is a render-only feature and does not affect underlying fluid simulation. | Yes | Memory leak will lead to out of memory (OOM) |
| Deformable Bodies and Particles static friction | Static friction is not supported. | NA | No |  |
| Deformable Bodies and Particle Friction Combine Mode | Friction combine mode is not supported. Any interaction is using the dynamic friction set on the particle/deformable actor. | NA | No |  |
| Particles simulation | Particles can roll off flat collision surfaces that are perpendicular to gravity due to solver ghost forces. | NA | No |  |
| Articulation Tendons | The simulation fidelity and behavior can be inadequate. Articulation joint incoming force reported will be excessively high when using nonzero TGS velocity iterations. | Fixed tendons: Use the [Mimic Joint feature](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/class_physx_schema_physx_mimic_joint_a_p_i.html). Spatial tendons: Apply external forces to the links to mimic the spatial tendons. Force sensing: Use TGS with zero velocity iterations. | No |  |
| GPU Convex Hull Vertex/Face Limit | For performance/memory footprint, the SDK limits GPU-compatible convex hulls to 64 vertices and faces. This can lead to poor approximation quality to the asset collider tri-mesh. | Use convex decomposition or SDF tri-meshes to capture details better. Be aware that a convex decomposition with the same set of vertices as a single convex hull may not produce the exact same behavior because contact detection runs on each convex independently. | Yes | Switching to convex decomposition or SDF will have a simulation perf impact due to higher computational cost. |
| Spherical Articulation Joints on links with nonidentity center-of-mass transform (cmasslocalpose) | Joint limits and drives for spherical articulation joints may not respond correctly in certain joint state ranges if the articulation joint is setup on a link with nonidentity mass frame setup (cmasslocalpose) | If possible, transform asset such that prim and mass frame coincide such that an identity transform can be used for the mass frame. | No |  |
| TGS Velocity Iterations | * With the current release (2023.1), the PhysX SDK no longer silently converts velocity iterations exceeding four to position iterations. Omniverse Physics will issue a corresponding warning to the log. This changing of iteration counts can affect physics behavior. | * manually converting the velocity iterations >4 to position iterations in assets to recover the behavior before the fix * try setting zero or very few velocity iterations * try the PGS solver | No |  |
| D6 Joint Drive | D6 Joint Drive does not behave exactly as expected when the TGS solver is employed. | If there are drive behavior issues then use PGS instead. | No |  |
| D6 Joint Drive | D6 Joint Drive does not work well with a combination of TGS and velocity iterations. | When TGS is employed it is recommended to focus computational effort on position iterations and to have zero velocity iterations. | No |  |
| Articulation Link Force Sensors | The force sensors are deprecated and will be removed in a future version. They reported incorrect/implausible values for many use cases. | Use contact force reports or link incoming joint force instead. | No |  |
| Articulation Joint Solver Forces | The joint solver force reporting is deprecated and will be removed in a future version. The reported forces were incorrect/implausible. | Use link incoming joint force reporting instead. | No |  |
| Particle Cloth | Particle cloth will be deprecated when replaced with surface deformable bodies when they are ready. We don’t have an ETA for this yet. | Do not use particle cloth. | No |  |
| Articulation Loop-closing using D6 Joints | We have seen issues with loop-closures using D6 Joints where the simulation goes unstable, mainly using GPU simulation. | Try increasing simulation time steps per second on the scene (i.e. decrease the simulation time step), try increasing articulation solver iterations, and try PGS and TGS solvers. | No |  |
| Articulation joint friction | There are reports of differing effective joint friction between PGS and TGS solver.  The friction model may not be suitable for all applications,see details in the [API documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/_api_build/class_px_articulation_joint_reduced_coordinate.html#_CPPv4N36PxArticulationJointReducedCoordinate22setFrictionCoefficientEK6PxReal). | A more common velocity-proportional dynamic friction model can be implemented using a joint drive with zero target velocity and a suitable damping parameter. | No |  |

---

# Physics Inspector

Detailed documentation regarding the Physics Inspector can be found [here](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.supportui/docs/dev_guide/authoring_tools.html#physics-inspector).

Note that the path to enable the Physics Inspector and the Physics Authoring Tool bar described on the page linked above is slightly different than in Isaac Sim. In Isaac Sim, the paths are below:

1. Physics Authoring Toolbar: Tools > Physics Toolbar
2. Physics Inspector: Tools > Physics > Physics Inspector

Warning

Since the Physics Inspector partially initializes `omni.physx`, it is expected for general simulations to not behave properly.
Such behaviour can be reversed by simply closing the Physics Inspector window/panel.

---

# Physics Static Collision Extension

The [Physics Static Collision Extension](#isaac-static-collision-utils) Extension is used to visualize collision meshes. Use this Utility extension to add static collision APIs to an entire [Stage](Glossary.md). The extension can also be used to remove all physics related APIs for testing purposes.

This extension is enabled by default. If it is ever disabled, it can be re-enabled from the [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)") by searching for `isaacsim.utils.physics`.

To access this Extension, go to the top menu bar and click **Tools** > **Physics API Editor**.

Note

Dynamic objects are currently not supported.

## User Interface

The User Interface provides options to add or clear static collision on selected static objects.

### Configuration Options

* **Apply to children**: Recursively create collision on all selected children; otherwise, create collision for just the selected object.
* **Visible only**: Ensure the prim is visible before creating collision. (Ignores hidden prims)
* **Collision Type**: Type of collision approximation to use
* **Apply Static**: Applies collision to the current selection.
* **Remove Collision API**: Clears the collision from the current selection.
* **Remove All Physics APIs**: Remove all Physics-related APIs (including collision) from the current selection.

### Enable Visualization

To visualize collision in any viewport:

1. **Select**: the  eye icon.
2. **Select**: Show by type.
3. **Select**: Physics Mesh.
4. **Check**: All.

Note

Enable visualization **after** collision APIs have been applied or removed. Otherwise there will be a loss in performance while the extension traverses the desired subtree.

---

# Simulation Data Visualizer

The [Simulation Data Visualizer](#isaac-inspect-physics) is used to visualize information for the selected prim. You can use this tool to better understand the behaviors of physics-enabled geometry during simulation.

If a non-physics prim is selected, position changes over the course of simulation are tracked. However, when a physics element is selected, it shows more physics properties, including position and velocities (linear, angular).

## Conventions

The simulation data visualizer provides the following information:

* **Position**: in [Stage](Glossary.md) units [X, Y, Z]
* **Rotation**: in degrees [X, Y, Z]
* **Linear Velocity**: in [Stage](Glossary.md) units/s
* **Angular Velocity**: in degrees/s
* **Linear Acceleration**: in [Stage](Glossary.md) units/s^2
* **Mass**: in [Stage](Glossary.md) mass unit
* **Moment of Inertia**: in [Stage](Glossary.md) mass unit\*[Stage](Glossary.md) units^2

For velocities, there’s a fourth plot M, which is the magnitude of the vector.

Articulations and Physics Scenes have residual error reporting. The available residual information is Position and Velocity RMS and Max in [Stage](Glossary.md) units.

## Inspect Physics Example

To run this utility:

1. Open the Simulation Data Visualizer by going to the **Visibility Menu (eye icon on viewport) > Show by Type > Physics > Simulation Data Visualizer**.
2. Activate **Windows** > **Examples** > **Robotics Examples** which will open the `Robotics Examples` tab.
3. Load some simulation-ready example, such as the Cortex Franka example, by clicking **Robotics Examples > Cortex > Franka Cortex Examples**.
4. Press the **Load Robot** button.
5. Select the **/World/Franka/panda\_hand** prim from the [Stage](Glossary.md).
6. Press the **START** button to begin simulating.

After simulation starts, the physics state of the selected rigid body updates in the **Inspect Physics** window.

**Inspect Residual Error**

1. With the previous example open, select the physics scene in the **Stage** window.
2. On the **Properties** panel, click on the **Add > Physics > Residual Reporting**.
3. Scroll down on the **Properties** panel, and under the **Advanced** tab, check **Enable Residual Reporting**.
4. Click on **Reset**.
5. Click on **Start**.

The types of Physics Objects that report residuals are:

* Simulation Scene
* Articulations (Place at the Root)
* Joints

