# Renaming Extensions in Isaac Sim 4.5

## Renamed Extensions

The following table specifies the new extension name(s) for deprecated Isaac Sim extensions.

Note

Several deprecated extensions (eg. `omni.isaac.sensor`) have been split into multiple new extensions,
and some new extensions (eg. `isaacsim.sensors.physx`) contain APIs from multiple deprecated extensions.
Deprecation warnings, as described [above](#extension-renaming-apis), will reflect these changes.

Note

Support for the deprecated extensions listed below will be removed in Isaac Sim 5.0 release.

Note

For more information about each extension’s functionality, please review the [API Documentation](API_Documentation.md).

Deprecated and New Extensions

| Isaac Sim 4.2 Extension(s) | Isaac Sim 4.5 Extension(s) |
| --- | --- |
| omni.exporter.urdf | isaacsim.asset.exporter.urdf |
| omni.importer.mjcf | isaacsim.asset.importer.mjcf |
| omni.importer.urdf | isaacsim.asset.importer.urdf |
| omni.isaac.app.selector | isaacsim.app.selector |
| omni.isaac.app.setup | isaacsim.app.setup |
| omni.isaac.articulation\_inspector | omni.physx.inspector |
| omni.isaac.asset\_browser   omni.isaac.assets\_check | isaacsim.asset.browser |
| omni.isaac.benchmark.services | isaacsim.benchmark.services |
| omni.isaac.benchmarks | isaacsim.benchmark.examples |
| omni.isaac.block\_world | isaacsim.asset.importer.heightmap |
| omni.isaac.camera\_inspector | isaacsim.util.camera\_inspector |
| omni.isaac.cloner | isaacsim.core.cloner |
| omni.isaac.conveyor | isaacsim.asset.gen.conveyor |
| omni.isaac.conveyor.ui | isaacsim.asset.gen.conveyor.ui |
| omni.isaac.core\_nodes | isaacsim.core.nodes |
| omni.isaac.core | isaacsim.core.api   isaacsim.core.prims   isaacsim.core.utils |
| omni.isaac.cortex | isaacsim.cortex.framework |
| omni.isaac.cortex.sample\_behaviors | isaacsim.cortex.behaviors |
| omni.isaac.debug\_draw | isaacsim.util.debug\_draw |
| omni.isaac.diff\_usd | isaacsim.util |
| omni.isaac.doctest | isaacsim.test.docstring |
| omni.isaac.examples | isaacsim.examples.interactive |
| omni.isaac.extension\_templates | isaacsim.examples.extension |
| omni.isaac.franka | isaacsim.robot.manipulators.examples |
| omni.isaac.gain\_tuner | isaacsim.robot\_setup.gain\_tuner |
| omni.isaac.grasp\_editor | isaacsim.robot\_setup.grasp\_editor |
| omni.isaac.import\_wizard | isaacsim.robot\_setup.import\_wizard |
| omni.isaac.jupyter\_notebook | isaacsim.code\_editor.jupyter |
| omni.isaac.kit | isaacsim.simulation\_app |
| omni.isaac.lula\_test\_widget | isaacsim.robot\_motion.lula\_test\_widget |
| omni.isaac.lula | isaacsim.robot\_motion.lula |
| omni.isaac.manipulators | isaacsim.robot.manipulators |
| omni.isaac.manipulators.ui | isaacsim.robot.manipulators.ui |
| omni.isaac.menu | isaacsim.gui.menu |
| omni.isaac.merge\_mesh | isaacsim.util.merge\_mesh |
| omni.isaac.motion\_generation | isaacsim.robot\_motion.motion\_generation |
| omni.isaac.nucleus | isaacsim.storage.native |
| omni.isaac.occupancy\_map | isaacsim.asset.gen.omap |
| omni.isaac.occupancy\_map.ui | isaacsim.asset.gen.omap.ui |
| omni.isaac.physics\_utilities | isaacsim.util.physics |
| omni.isaac.proximity\_sensor | isaacsim.sensors.physx |
| omni.isaac.quadruped | isaacsim.robot.policy.examples |
| omni.isaac.range\_sensor | isaacsim.sensors.physx |
| omni.isaac.range\_sensor.examples | isaacsim.sensors.physx.examples |
| omni.isaac.range\_sensor.ui | isaacsim.sensors.physx.ui |
| omni.isaac.robot\_assembler | isaacsim.robot\_setup.assembler |
| omni.isaac.robot\_description\_editor | isaacsim.robot\_setup.xrdf\_editor   isaacsim.robot\_setup.lula\_editor |
| omni.isaac.ros\_bridge | isaacsim.ros1.bridge |
| omni.isaac.ros2\_bridge | isaacsim.ros2.bridge |
| omni.isaac.ros2\_bridge.robot\_description | isaacsim.ros2.urdf |
| omni.isaac.scene\_blox | isaacsim.replicator.scene\_blox |
| omni.isaac.sensor | isaacsim.sensors.camera   isaacsim.sensors.camera.ui   isaacsim.sensors.physics   isaacsim.sensors.physics.examples   isaacsim.sensors.physics.ui   isaacsim.sensors.physx   isaacsim.sensors.rtx   isaacsim.sensors.rtx.ui |
| omni.isaac.surface\_gripper | isaacsim.robot.surface\_gripper |
| omni.isaac.surface\_gripper.ui | isaacsim.robot.surface\_gripper.ui |
| omni.isaac.synthetic\_recorder | isaacsim.replicator.synthetic\_recorder |
| omni.isaac.tests | isaacsim.test.collection |
| omni.isaac.tf\_viewer | isaacsim.ros2.tf\_viewer |
| omni.isaac.throttling | isaacsim.core.throttling |
| omni.isaac.ui\_template | isaacsim.examples.ui |
| omni.isaac.ui | isaacsim.gui.components |
| omni.isaac.unit\_converter | omni.usd.metrics\_assembler |
| omni.isaac.universal\_robots | isaacsim.robot.manipulators.examples |
| omni.isaac.utils | isaacsim.core.utils |
| omni.isaac.version | isaacsim.core.version |
| omni.isaac.vscode | isaacsim.code\_editor.vscode |
| omni.isaac.wheeled\_robots | isaacsim.robot.wheeled\_robots   isaacsim.robot.wheeled\_robots.examples |
| omni.isaac.wheeled\_robots.ui | isaacsim.robot.wheeled\_robots.ui |
| omni.isaac.window.about | isaacsim.app.about |
| omni.kit.property.isaac | isaacsim.gui.property |
| omni.replicator.agent.camera\_calibration | isaacsim.replicator.agent.camera\_calibration |
| omni.replicator.agent.core | isaacsim.replicator.agent.core |
| omni.replicator.agent.ui | isaacsim.replicator.agent.ui |
| omni.replicator.isaac | isaacsim.replicator.domain\_randomization   isaacsim.replicator.examples   isaacsim.replicator.writers |

## Deprecated Extensions

The following table specifies extensions deprecated, but not renamed, in Isaac Sim 4.5.

Note

Support for the deprecated extensions listed below will be removed in Isaac Sim 5.0 release.

| Deprecated Extensions |
| --- |
| omni.isaac.dynamic\_control |
| omni.isaac.examples\_nodes |
| omni.isaac.repl |

## Removed Extensions

The following table specifies extensions removed from Isaac Sim 4.5.

| Removed Extensions | Notes |
| --- | --- |
| omni.isaac.benchmark\_environments | Deprecated since Isaac Sim 4.0. |
| omni.isaac.cortex\_sync | Deprecated since Isaac Sim 4.0. |
| omni.isaac.dofbot | `dofbot` example no longer supported as of Isaac Sim 4.5 |
| omni.isaac.partition | Partition tool no longer supported as of Isaac Sim 4.5 |
| omni.isaac.physics\_inspector | Deprecated since Isaac Sim 4.0. Replaced by [Omniverse Physics Inspector](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/ux/source/omni.physx.supportui/docs/dev_guide/authoring_tools.html#physics-inspector). |
| omni.isaac.robot\_benchmark | Deprecated since Isaac Sim 4.0. |
| omni.isaac.ocs2 | No longer supported as of Isaac Sim 4.5 |

# FAQ

## Why were extensions renamed in Isaac Sim 4.5?

We chose to rename and split extensions for Isaac Sim 4.5 to support ongoing brand standardization,
and improve modularity for end-users who may choose to build custom apps using Isaac Sim extensions,
including Isaac Lab.

## How will renamed extensions affect my Isaac Sim workflows?

Extension renaming will affect your workflows in three primary ways:

1. Setting names referencing specific extensions have changed (eg. `/exts/omni.isaac.ros2_bridge/ros_distro` has become `/exts/isaacsim.ros2.bridge/ros_distro`).
2. OmniGraph node names referencing specific extensions have changed (eg. `omni.isaac.core_nodes.IsaacReadSimulationTime` has become `isaacsim.core.nodes.IsaacReadSimulationTime`).
3. Extension APIs have been renamed and/or moved (eg. `omni.isaac.sensor.LidarRtx` has become `isaacsim.sensors.rtx.LidarRtx`).

## How should I update my existing Isaac Sim workflows for Isaac Sim 4.5?

Isaac Sim 4.5 has introduced the `isaacsim.core.deprecation_manager` extension to support backwards compatibility with existing end-user workflows for Isaac Sim 4.5. While we cannot
guarantee perfect compatibility, this section will explain how the new extension can simplify updating your workflows, and what steps you should take to fully transition to Isaac Sim 4.5.

Note

In addition to renaming extensions, to improve startup time compared to Isaac Sim 4.2, several extensions were removed from the Isaac Sim 4.5 base Kit app.
If after trying the solutions below you are still encountering errors related to missing or renamed extensions, consider manually enabling the extension via the
`isaacsim.core.utils.extensions.enable_extension` API.

### Renaming settings

The `isaacsim.core.deprecation_manager` extension will automatically copy values of settings associated with a deprecated Isaac Sim extension
to its corresponding new extension’s settings. For example, if you execute the following:

```python
./isaac-sim.sh --/exts/omni.isaac.ros2_bridge/ros_distro=humble
```

the `isaacsim.core.deprecation_manager` extension will automatically set the value of `/exts/isaacsim.ros2.bridge/ros_distro` to `humble`.

If your workflow relies on custom values for Isaac Sim extension settings, rename the settings by referencing [the table below](#extension-renaming-deprecated-to-new).

### Renaming OmniGraph nodes

The `isaacsim.core.deprecation_manager` extension will automatically update the prim type of any OmniGraph node belonging to a deprecated Isaac Sim extension
when a scene opens, and print the changes to the console along with a message in the UI. The scene must be saved to retain the changes.

Note

If the OmniGraph node is part of a reference or payload asset in the scene, the name change will be registered as a delta to the reference in the opened scene.
`isaacsim.core.deprecation_manager` **will not** recursively updated USD files. To ensure the referenced USD is properly updated, you will need to
manually open the referenced USD on its own and then save it with the updated OmniGraph nodes.

Any USD asset shipped with Isaac Sim 4.5 that includes OmniGraph nodes will already be updated to use the new extensions’s OmniGraph nodes.

### Renaming extension APIs

All deprecated extensions have been included in Isaac Sim 4.5. When loaded from a kit app, they will print deprecation warning(s) indicating to which new
extension(s) APIs have been moved.

To remove those deprecation warnings when running your own workflows, migrate your workflows to the new extensions by updating Python import statements to use the
new extensions, eg.

```python
from omni.isaac.sensor import LidarRtx
```

would become

```python
from omni.isaac.sensor import LidarRtx
```
