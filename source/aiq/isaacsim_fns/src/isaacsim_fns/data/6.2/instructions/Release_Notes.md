# Release Notes

## 6.0.0 Early Developer Release

* Updated to Kit SDK 109.0.2
* Updated to Python 3.12
* Updated to use RT 2.0 rendering mode by default
* In the early developer release, Neural Reconstruction (NuRec) scenes which include a matte object will not render correctly. To avoid this, you may disable the matte object by making it invisible, or you can force the older RT 1.0 rendering mode using these command-line settings:

  `--/persistent/rtx/modes/rt/enabled=true`
  `--/rtx/rendermode=RaytracedLighting`

### Kit SDK Version

Changed: 107.3.3+isaac.229672.69cbf6ad.gl -> 109.0.2+production.256123.dc36eb6f.gl

### Dependencies

#### Added

* isaacsim.replicator.agent.schema: 0.0.1
* isaacsim.replicator.incident.core: 0.11.2
* isaacsim.replicator.incident.ui: 0.6.0
* isaacsim.replicator.object.core: 0.11.3
* isaacsim.replicator.object.ui: 0.11.1
* isaacsim.sensors.rtx.calibration: 0.3.2
* omni.ai.langchain.agent.chat\_iro: 2.2.5
* omni.ai.langchain.core: 2.2.0
* omni.ai.langchain.widget.core: 3.0.0
* omni.anim.behavior.asset: 109.0.6
* omni.anim.behavior.bundle: 109.0.6
* omni.anim.behavior.core: 109.0.7
* omni.anim.behavior.ui: 109.0.6
* omni.behavior.composer: 0.5.3
* omni.behavior.composer.schema: 0.4.0
* omni.behavior.composer.ui: 0.5.2
* omni.kit.livestream.app: 9.0.0
* omni.mesh\_tools.libs: 109.0.8
* omni.replicator.nv: 1.0.0
* omni.scene.optimizer.analysis: 109.0.2
* omni.scene.optimizer.validators: 109.0.2

#### Removed

* isaacsim.replicator.incident: 0.1.28
* isaacsim.replicator.object: 0.4.13
* isaacsim.xr.input\_devices: 1.0.2
* isaacsim.xr.openxr: 1.0.0
* omni.anim.people: 0.7.9
* omni.cuopt.examples: 1.3.0
* omni.kit.streamsdk.plugins: 7.6.3
* omni.kit.window.modifier.titlebar: 105.2.16
* omni.kit.xr.advertise: 107.3.109
* omni.kit.xr.profile.ar: 107.3.109
* omni.kit.xr.profile.common: 107.3.109
* omni.kit.xr.profile.tabletar: 107.3.109
* omni.kit.xr.profile.vr: 107.3.109
* omni.kit.xr.scene\_view.core: 107.3.109
* omni.kit.xr.scene\_view.utils: 107.3.109
* omni.services.livestream.nvcf: 7.2.0

#### Changed

* isaacsim.action\_and\_event\_data\_generation.setup: 0.0.9 -> 0.6.0
* isaacsim.anim.robot: 0.0.15 -> 1.0.3
* isaacsim.exp.full: 5.1.0 -> 6.0.0
* isaacsim.replicator.agent.core: 0.7.28 -> 1.0.11
* isaacsim.replicator.agent.ui: 0.7.11 -> 1.0.7
* isaacsim.replicator.caption.core: 0.0.32 -> 0.6.6
* isaacsim.sensors.rtx.placement: 0.6.14 -> 0.16.4
* isaacsim.util.debug\_draw: 3.1.0 -> 3.2.0
* omni.anim.asset: 107.3.0 -> 109.0.6
* omni.anim.behavior.schema: 107.3.0 -> 109.0.6
* omni.anim.curve.bundle: 1.2.3 -> 1.3.0
* omni.anim.curve.core: 1.3.1 -> 1.5.3
* omni.anim.curve.ui: 1.4.1 -> 1.6.1
* omni.anim.curve\_editor: 106.4.1 -> 109.0.0
* omni.anim.graph.bundle: 107.3.3 -> 109.0.6
* omni.anim.graph.core: 107.3.4 -> 109.0.6
* omni.anim.graph.schema: 107.3.3 -> 109.0.6
* omni.anim.graph.ui: 107.3.3 -> 109.0.6
* omni.anim.navigation.bundle: 107.3.3 -> 109.0.6
* omni.anim.navigation.core: 107.3.8 -> 109.0.6
* omni.anim.navigation.schema: 107.3.3 -> 109.0.6
* omni.anim.navigation.ui: 107.3.6 -> 109.0.6
* omni.anim.retarget.bundle: 107.3.3 -> 109.0.6
* omni.anim.retarget.core: 107.3.3 -> 109.0.6
* omni.anim.retarget.preview: 107.3.3 -> 109.0.6
* omni.anim.retarget.ui: 107.3.3 -> 109.0.6
* omni.anim.shared.core: 107.0.1 -> 109.0.2
* omni.anim.skelJoint: 107.3.3 -> 109.0.6
* omni.anim.timeline: 107.0.0 -> 109.0.1
* omni.anim.widget.timeline: 0.1.14 -> 0.3.0
* omni.anim.window.timeline: 106.5.0 -> 109.0.2
* omni.asset\_validator.core: 1.1.6 -> 1.8.0
* omni.asset\_validator.ui: 1.1.6 -> 1.8.0
* omni.convexdecomposition: 107.3.26 -> 109.0.10
* omni.cuopt.service: 1.3.0 -> 1.3.1
* omni.cuopt.visualization: 1.3.0 -> 1.3.2
* omni.curve.manipulator: 107.0.4 -> 108.2.0
* omni.flowusd: 107.1.8 -> 109.0.2
* omni.genproc.core: 107.0.3 -> 109.0.0
* omni.graph.action: 1.130.0 -> 2.0.0
* omni.graph.action\_nodes: 1.50.4 -> 2.0.2
* omni.graph.action\_nodes\_core: 1.2.0 -> 2.0.1
* omni.graph.bundle.action: 2.30.0 -> 3.0.1
* omni.graph.examples.cpp: 1.50.2 -> 2.0.1
* omni.graph.nodes: 1.170.10 -> 2.1.5
* omni.graph.nodes\_core: 1.1.0 -> 2.0.3
* omni.graph.scriptnode: 1.50.0 -> 2.1.2
* omni.graph.telemetry: 2.40.2 -> 3.1.3
* omni.graph.ui: 1.101.6 -> 2.1.5
* omni.graph.ui\_nodes: 1.50.5 -> 2.0.3
* omni.graph.visualization.nodes: 2.1.3 -> 2.1.4
* omni.graph.window.action: 1.50.2 -> 2.2.1
* omni.graph.window.core: 2.0.0 -> 3.1.0
* omni.graph.window.generic: 1.50.2 -> 2.2.1
* omni.importer.onshape: 1.0.1 -> 2.0.1
* omni.kit.asset\_converter: 5.0.17 -> 5.0.22
* omni.kit.browser.asset: 1.3.12 -> 1.3.15
* omni.kit.browser.core: 2.3.13 -> 2.3.17
* omni.kit.browser.folder.core: 1.10.9 -> 1.12.1
* omni.kit.browser.material: 1.6.2 -> 1.6.5
* omni.kit.converter.cad: 205.0.0 -> ~207.0
* omni.kit.converter.common: 507.1.2 -> 508.0.1
* omni.kit.converter.dgn: 509.1.0 -> 509.3.0
* omni.kit.converter.dgn\_core: 510.1.0 -> 511.2.0
* omni.kit.converter.hoops: 509.1.0 -> 509.2.4
* omni.kit.converter.hoops\_core: 509.1.0 -> 510.1.3
* omni.kit.converter.jt: 508.1.0 -> 508.2.5
* omni.kit.converter.jt\_core: 508.1.0 -> 508.2.7
* omni.kit.core.collection: 0.2.3 -> 0.2.4
* omni.kit.data2ui.core: 1.1.2 -> 1.1.4
* omni.kit.data2ui.usd: 1.1.2 -> 1.1.4
* omni.kit.environment.core: 1.3.24 -> 1.4.1
* omni.kit.gfn: 107.0.4 -> 108.0.0
* omni.kit.graph.delegate.default: 1.2.3 -> 1.2.5
* omni.kit.graph.delegate.modern: 1.10.9 -> 1.10.11
* omni.kit.graph.editor.core: 1.5.3 -> 1.5.4
* omni.kit.graph.usd.commands: 1.3.1 -> 1.3.2
* omni.kit.graph.widget.variables: 2.1.0 -> 2.1.1
* omni.kit.livestream.core: 7.5.0 -> 9.0.0
* omni.kit.livestream.webrtc: 7.0.0 -> 9.0.2
* omni.kit.mesh.raycast: 107.0.1 -> 108.0.0
* omni.kit.playlist.core: 1.3.5 -> 1.3.7
* omni.kit.pointclouds: 1.5.14 -> 1.6.5
* omni.kit.preferences.animation: 1.2.0 -> 1.4.0
* omni.kit.prim.icon: 1.0.15 -> 1.1.0
* omni.kit.profiler.window: 2.3.5 -> 2.3.7
* omni.kit.property.collection: 0.2.3 -> 0.2.4
* omni.kit.property.environment: 1.2.2 -> 1.2.3
* omni.kit.property.physx: 107.3.26 -> 109.0.10
* omni.kit.scripting: 107.3.2 -> 109.0.4
* omni.kit.sequencer.core: 108.0.2 -> 108.1.1
* omni.kit.sequencer.usd: 108.0.2 -> 108.1.1
* omni.kit.stage\_column.payload: 2.0.3 -> 2.0.5
* omni.kit.stage\_column.variant: 1.0.17 -> 1.0.20
* omni.kit.stagerecorder.bundle: 105.0.2 -> 109.0.0
* omni.kit.stagerecorder.core: 107.0.3 -> 109.0.0
* omni.kit.stagerecorder.ui: 107.0.1 -> 109.0.0
* omni.kit.thumbnails.mdl: 1.0.27 -> 1.0.28
* omni.kit.timeline.minibar: 1.2.11 -> 1.2.13
* omni.kit.tool.asset\_importer: 4.3.2 -> 5.1.3
* omni.kit.tool.measure: 107.0.2 -> 200.0.4
* omni.kit.tool.remove\_unused.controller: 0.1.4 -> 0.2.0
* omni.kit.tool.remove\_unused.core: 0.1.3 -> 0.1.4
* omni.kit.variant.editor: 107.5.3 -> 107.5.6
* omni.kit.variant.presenter: 107.0.0 -> 107.1.1
* omni.kit.viewport.menubar.lighting: 107.3.1 -> 107.3.2
* omni.kit.waypoint.core: 1.4.62 -> 1.6.3
* omni.kit.waypoint.playlist: 1.0.9 -> 1.1.1
* omni.kit.widget.collection: 0.3.1 -> 0.3.3
* omni.kit.widget.material\_preview: 1.0.16 -> 1.1.2
* omni.kit.widget.schema\_api: 1.0.3 -> 1.0.4
* omni.kit.widget.sliderbar: 1.0.13 -> 1.1.0
* omni.kit.widget.timeline: 107.0.1 -> 107.0.2
* omni.kit.widget.zoombar: 1.0.6 -> 1.0.7
* omni.kit.widgets.custom: 1.0.13 -> 1.1.2
* omni.kit.window.collection: 0.3.1 -> 0.3.4
* omni.kit.window.material: 1.7.2 -> 1.8.0
* omni.kit.window.material\_graph: 1.9.1 -> 1.9.5
* omni.kit.window.movie\_capture: 2.5.6 -> 2.7.3
* omni.kit.window.section: 107.0.3 -> 107.1.2
* omni.kit.window.usddebug: 1.1.2 -> 1.1.4
* omni.kit.xr.core: 107.3.109 -> 109.0.0
* omni.kit.xr.system.openxr: 107.3.109 -> 109.0.0
* omni.kit.xr.system.simulatedxr: 107.3.109 -> 109.0.0
* omni.kit.xr.ui.stage: 107.3.109 -> 109.0.0
* omni.kit.xr.ui.window.profile: 107.3.109 -> 109.0.0
* omni.kit.xr.ui.window.viewport: 107.3.109 -> 109.0.0
* omni.kvdb: 107.3.26 -> 109.0.10
* omni.localcache: 107.3.26 -> 109.0.10
* omni.metropolis.utils: 0.1.20 -> 0.14.7
* omni.no\_code\_ui.bundle: 1.1.2 -> 1.1.4
* omni.physics: 107.3.26 -> 109.0.10
* omni.physics.physx: 107.3.26 -> 109.0.10
* omni.physics.stageupdate: 107.3.26 -> 109.0.10
* omni.physics.tensors: 107.3.26 -> 109.0.10
* omni.physx: 107.3.26 -> 109.0.10
* omni.physx.asset\_validator: 107.3.26 -> 109.0.10
* omni.physx.bundle: 107.3.26 -> 109.0.10
* omni.physx.camera: 107.3.26 -> 109.0.10
* omni.physx.cct: 107.3.26 -> 109.0.10
* omni.physx.commands: 107.3.26 -> 109.0.10
* omni.physx.cooking: 107.3.26 -> 109.0.10
* omni.physx.demos: 107.3.26 -> 109.0.10
* omni.physx.fabric: 107.3.26 -> 109.0.10
* omni.physx.foundation: 107.3.26 -> 109.0.10
* omni.physx.graph: 107.3.26 -> 109.0.10
* omni.physx.pvd: 107.3.26 -> 109.0.10
* omni.physx.supportui: 107.3.26 -> 109.0.10
* omni.physx.telemetry: 107.3.26 -> 109.0.10
* omni.physx.tensors: 107.3.26 -> 109.0.10
* omni.physx.tests: 107.3.26 -> 109.0.10
* omni.physx.tests.visual: 107.3.26 -> 109.0.10
* omni.physx.ui: 107.3.26 -> 109.0.10
* omni.physx.vehicle: 107.3.26 -> 109.0.10
* omni.ramp: 107.0.1 -> 107.0.2
* omni.replicator.core: 1.12.27 -> 1.12.34
* omni.scene.optimizer.bundle: 107.3.12 -> 109.0.2
* omni.scene.optimizer.core: 107.3.12 -> 109.0.2
* omni.scene.optimizer.ui: 107.3.12 -> 109.0.2
* omni.scene.visualization.core: 107.0.2 -> 109.0.1
* omni.services.client: 0.5.3 -> 0.5.4
* omni.services.convert.asset: 508.0.2 -> 509.0.0
* omni.services.convert.cad: 507.0.2 -> 507.1.5
* omni.services.core: 1.9.0 -> 1.9.3
* omni.services.facilities.base: 1.0.4 -> 1.0.5
* omni.services.facilities.monitoring.metrics: 0.3.0 -> 0.3.1
* omni.services.pip\_archive: 0.16.0 -> 0.18.3
* omni.services.starfleet.auth: 0.1.5 -> 0.1.6
* omni.services.transport.client.base: 1.2.4 -> 1.2.5
* omni.services.transport.client.http\_async: 1.4.0 -> 1.4.2
* omni.services.transport.server.base: 1.1.1 -> 1.1.2
* omni.services.transport.server.http: 1.3.1 -> 1.3.2
* omni.services.transport.server.zeroconf: 1.0.9 -> 1.0.10
* omni.services.usd: 1.1.0 -> 1.1.1
* omni.simready.explorer: 1.1.3 -> 1.1.4
* omni.tools.array: 107.0.0 -> 108.0.0
* omni.usd.fileformat.e57: 1.4.3 -> 1.7.0
* omni.usd.fileformat.pts: 107.1.1 -> 108.0.0
* omni.usd.metrics.assembler: 107.3.1 -> 109.0.0
* omni.usd.metrics.assembler.physics: 107.3.26 -> 109.0.10
* omni.usd.metrics.assembler.ui: 107.3.1 -> 109.0.0
* omni.usd.schema.flow: 107.1.1 -> 109.0.1
* omni.usd.schema.metrics.assembler: 107.3.1 -> 109.0.0
* omni.usd.schema.physx: 107.3.26 -> 109.0.10
* omni.usd.schema.sequence: 3.0.1 -> 3.1.2
* omni.usdex.libs: 1.2.2 -> 2.1.2
* omni.usdphysics: 107.3.26 -> 109.0.10
* omni.usdphysics.ui: 107.3.26 -> 109.0.10
* omni.vdb\_timesample\_editor: 0.2.0 -> 0.2.3
* omni.warp: 1.8.2 -> 1.10.0
* omni.warp.core: 1.8.2 -> 1.10.0

## Extensions

* **isaacsim.app.about**

  > + Changed
  >
  >   - Update description
* **isaacsim.app.setup**

  > + Changed
  >
  >   - Add wait for viewport to be ready before printing app ready status
  >   - Change startup behavior so that app ready status is delayed until after the app has started
  >   - Remove unused imports
  >   - Remove unused omni.pip.cloud from test dependencies
  >   - Remove extra omni.rtx.settings.core from test dependencies
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.asset.browser**

  > + Changed
  >
  >   - Update assets path
  >   - Migrate extension implementation to core experimental API
  >   - Remove omni.pip.cloud from extension.toml
  >   - Remove requests dependency, use urllib instead
  > + Fixed
  >
  >   - Update unit tests for kit 109.0
  >   - Add missing documentation for the asset browser extension
  >   - Replaced deprecated onclick\_fn with onclick\_action in “Isaac Sim Assets” menu item to eliminate deprecation warnings
  >   - Registered proper toggle action for the asset browser
  >   - Fix issue where cache json file could not be created if the cache directory did not exist
* **isaacsim.asset.exporter.urdf**

  > + Changed
  >
  >   - Update lxml==6.0.2
  >   - Migrate extension implementation to core experimental API
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.asset.gen.conveyor**

  > + Changed
  >
  >   - Update to Kit 109 and Python 3.12
* **isaacsim.asset.gen.conveyor.ui**

  > + Changed
  >
  >   - Migrate to Events 2.0.
* **isaacsim.asset.gen.omap**

  > + Changed
  >
  >   - Update to use new debug draw plugin interface
  >   - Improve docstrings and cleanup codebase
  >   - Update to Kit 109 and Python 3.12
  >   - Migrate extension implementation to core experimental API
  >   - Add omni.physics.stageupdate to extension.toml
* **isaacsim.asset.gen.omap.ui**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Refactor codebase and improve docstrings
  >   - Migrate extension implementation to core experimental API
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.asset.importer.heightmap**

  > + Changed
  >
  >   - Renamed Block World Generator to Heightmap Importer
  >   - Refactored to separate importer logic from extension UI
  >   - Standardized terminology from “block world” to “heightmap” throughout codebase
  >   - Updated all documentation, comments, and test names to use heightmap terminology
  >   - Migrate extension implementation to core experimental API
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.asset.importer.mjcf**

  > + Changed
  >
  >   - Update description
  >   - Update to Kit 109 and Python 3.12
* **isaacsim.asset.importer.urdf**

  > + Changed
  >
  >   - Update description
  >   - Migrate to Events 2.0.
  >   - Add missing docstrings
  >   - Restore deprecated behavior of merging bodies with inertia; issue warning
  >   - Update to Kit 109 and Python 3.12
  > + Fixed
  >
  >   - Fixed issue that caused crash of Isaac sim on failed mesh conversion
* **isaacsim.asset.validation**

  > + Changed
  >
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.benchmark.examples**

  > + Changed
  >
  >   - Update description
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.benchmark.services**

  > + Added
  >
  >   - Add error handling if set\_phase() is called without a matching store\_measurements()
  > + Removed
  >
  >   - Removed stop\_recording\_runtime arg to benchmark.store\_measurements()
  > + Changed
  >
  >   - Converted log statements to use logger for independent visibility control
  >   - Update description
  >   - Migrate to Events 2.0.
  >   - Get the CUDA device names using Warp API
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
  > + Fixed
  >
  >   - Write privacy.toml file to temporary directory
* **isaacsim.code\_editor.jupyter**

  > + Changed
  >
  >   - Migrate to Events 2.0.
* **isaacsim.core.api**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Add missing docstrings
  >   - Update to Kit 109 and Python 3.12
  >   - Remove unused omni.pip.cloud from dependencies
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove extra carb settings from tests
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.core.cloner**

  > + Changed
  >
  >   - Fix clang tidy issues in cpp code
  >   - Add missing docstrings
  >   - Update to Kit 109 and Python 3.12
  >   - Convert input arguments to NumPy without explicitly import PyTorch
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.core.deprecation\_manager**

  > + Added
  >
  >   - Expose function to import a deprecated/removed module safely
  > + Changed
  >
  >   - Removed code to enable the omni.isaac.ml\_archive extension when importing PyTorch via import\_module
  >   - Enable the omni.isaac.ml\_archive extension when importing PyTorch via import\_module
* **isaacsim.core.experimental.materials**

  > + Removed
  >
  >   - Remove checking for the deformable beta feature, as it is now active by default
  > + Changed
  >
  >   - Define ranges for visual material inputs and clip them accordingly
  >   - Standardize test args in extension.toml
* **isaacsim.core.experimental.objects**

  > + Changed
  >
  >   - Define the reset\_xform\_op\_properties parameter to True by default for all objects
* **isaacsim.core.experimental.prims**

  > + Removed
  >
  >   - Remove checking for the deformable beta feature, as it is now active by default
  > + Changed
  >
  >   - Migrate to Events 2.0
  >   - Update check condition on DOF to ensure it checks if it’s a valid DOF before checking limits
  >   - Update implementation to Warp 1.10.0
  >   - Update array output in docstrings example due to changes in the NumPy representation
  >   - Make isaacsim.storage.native an explicit test dependency
  >   - Replace the use of deprecated core utils functions within implementations
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
  > + Fixed
  >
  >   - Fix physics setup when a prim instance is created while the simulation is running
* **isaacsim.core.experimental.utils**

  > + Added
  >
  >   - Add timeline-related functions to app utils
  >   - Add xform utils
  >   - Support USD schemas when getting a prim or prim path
  >   - Add app utils
  >   - Add semantics utils
  >   - Add stage utils functions to:
  >   - Check whether the stage is loading
  >   - Generate a string representation of the stage
  >   - Move a prim to a different location on the stage hierarchy
  >   - Delete a prim from the stage
  >   - Add prim utils functions to:
  >   - Find all the prim paths in the stage that match the given (regex) path
  >   - Check whether a prim corresponds to a non-root link in an articulation
  > + Changed
  >
  >   - Update app module docstrings example and add module summaries for docs purposes
  >   - Return the input as it is when getting a prim or prim path, provided that the input is of the expected return type
* **isaacsim.core.includes**

  > + Changed
  >
  >   - Run clang tidy
  >   - Migrate BaseResetNode to Events 2.0.
  >   - Update to Kit 109 and Python 3.12
* **isaacsim.core.nodes**

  > + Changed
  >
  >   - Migrate OgnOnPhysicsStep to Events 2.0.
  >   - Update description
  >   - Migrate to Events 2.0.
  >   - Set ResetOnStop to True for all Simulation Time OG nodes
  >   - Moved handle interface to isaacsim.ros2.nodes extension where it was used.
  >   - Update to Kit 109 and Python 3.12
  >   - Update deprecated python unittest methods
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated dependencies
  >   - Remove deprecated time related APIs from CoreNodes interface
  >   - Remove extra carb settings from tests
* **isaacsim.core.prims**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
  > + Fixed
  >
  >   - Fix the PhysxCollisionAPI schema when checking for collision properties
* **isaacsim.core.simulation\_manager**

  > + Added
  >
  >   - Add the SimulationEvent enum
  >   - Allow to perform a fabric update when stepping physics
  > + Changed
  >
  >   - Run clang tidy
  >   - Raise a RuntimeError if the physics dt is being set while simulation is running/playing
  >   - Mark as deprecated the IsaacEvents enum and the backend-related methods
  >   - Make set\_physics\_dt a classmethod
  >   - Add unit tests for SimulationManager
  >   - Update to Kit 109 and Python 3.12
  >   - Replace the use of deprecated core utils functions by the core experimental implementations
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.core.throttling**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Remove extra carb settings from tests
* **isaacsim.core.utils**

  > + Removed
  >
  >   - Removed deprecated nucleus module (use isaacsim.storage.native instead):
  >   - get\_url\_root, create\_folder, delete\_folder, \_list\_files, download\_assets\_async
  >   - check\_server, check\_server\_async, build\_server\_list, find\_nucleus\_server
  >   - get\_server\_path, get\_server\_path\_async, verify\_asset\_root\_path
  >   - get\_full\_asset\_path, get\_full\_asset\_path\_async
  >   - get\_nvidia\_asset\_root\_path, get\_isaac\_asset\_root\_path
  >   - get\_assets\_root\_path, get\_assets\_root\_path\_async, get\_assets\_server
  >   - \_collect\_files, is\_dir\_async, is\_file\_async, is\_file
  >   - recursive\_list\_folder, list\_folder
  >   - Removed deprecated create\_hydra\_texture from render\_product (use omni.replicator.core.create.render\_product instead)
  >   - Removed deprecated semantics functions using old SemanticsAPI (use new LabelsAPI equivalents):
  >   - add\_update\_semantics -> use add\_labels
  >   - remove\_all\_semantics -> use remove\_labels
  >   - get\_semantics -> use get\_labels
  >   - check\_missing\_semantics -> use check\_missing\_labels
  >   - check\_incorrect\_semantics -> use check\_incorrect\_labels
  >   - count\_semantics\_in\_scene -> use count\_labels\_in\_scene
  > + Changed
  >
  >   - set\_camera\_prim\_path now also applies the OmniRtxCameraExposureAPI\_1 schema to the camera prim
  >   - set\_camera\_prim\_path now also sets the exposure:time attribute to 0.02
  >   - Add missing docstrings
  >   - Update to Kit 109 and Python 3.12
  >   - Fix invalid escape sequences
  >   - Update deprecated python unittest methods
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated dependencies
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.cortex.behaviors**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.examples.browser**

  > + Fixed
  >
  >   - Replaced deprecated onclick\_fn with onclick\_action in “Robotics Examples” menu item to eliminate deprecation warnings
  >   - Registered proper toggle action for the examples browser
* **isaacsim.examples.extension**

  > + Changed
  >
  >   - Fix event name usage.
  >   - Migrate to Events 2.0.
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.examples.interactive**

  > + Removed
  >
  >   - base\_sample\_experimental.py
  >   - base\_sample\_extension\_experimental.py
  >   - Build window function use
  >   - The Simple Stack example
  >   - The Franka Pick-and-Place and UR10 Follow Target examples have been removed from this extension and moved to a new location
  > + Changed
  >
  >   - Update description
  >   - Migrate to Events 2.0.
  >   - Updated inference examples to use GPU physics and the new experimental APIs
  >   - Moved policy based examples to isaacsim.robot.policy.examples
  >   - Add missing docstrings
  >   - Update imports from isaacsim.base\_samples to isaacsim.examples.base
  >   - The Start with Robot, Kaya Gamepad, Omnigraph Keyboard, and Hello World examples now depend on the new Warp-based APIs
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated dependencies
  >   - Remove extra carb settings from tests
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
  > + Fixed
  >
  >   - Fix Kaya Gamepad example test
* **isaacsim.examples.ui**

  > + Changed
  >
  >   - Rename startup.py to test\_startup.py
* **isaacsim.gui.components**

  > + Changed
  >
  >   - Add missing docstrings
  >   - Migrate extension implementation to core experimental API
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.gui.content\_browser**

  > + Changed
  >
  >   - Fix missing icon error in Isaac content browser
  >   - Update description
  >   - Revert the protocol to match the current Isaac Sim version
  >   - Fix navigation issue caused by incorrect protocol
  >   - Update assets path
  >   - Update to Kit 109 and Python 3.12
  > + Fixed
  >
  >   - Fix getting Carb settings API for protocol designation
* **isaacsim.gui.menu**

  > + Changed
  >
  >   - Update golden image for environment test
  >   - Update description
  >   - Renamed Block World Generator to Heightmap Importer
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.gui.property**

  > + Added
  >
  >   - Introduced widgets for the Robot Schema
  > + Changed
  >
  >   - Add missing license headers
* **isaacsim.gui.sensors.icon**

  > + Removed
  >
  >   - Remove the deprecated and unused isaacsim.core.utils dependency
  > + Changed
  >
  >   - Update description
  >   - Migrate to Events 2.0.
  >   - Update test module import
* **isaacsim.replicator.behavior**

  > + Changed
  >
  >   - Added explicit seed to randomizers to make them deterministic
  >   - Updated sdg pipeline golden images
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.replicator.behavior.ui**

  > + Changed
  >
  >   - Update test module import
* **isaacsim.replicator.domain\_randomization**

  > + Changed
  >
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.replicator.examples**

  > + Changed
  >
  >   - Improved simready assets SDG example output results
  >   - Fixed sequential sphere scan randomizer example to work in script editor in sync with docs
  >   - Added explicit .reset() to events 2.0 subscribers in sync with docs examples
  >   - Migrate to Events 2.0.
  >   - Added an app update after switching to pathtracing in the palletizing example test
  >   - Fixed scatter plane parent path in scene based SDG example test
  >   - Fixed SDG box stacking randomizer example test by waiting for the data to be written to disk
  >   - Make consistent use of SimulationManager
  >   - Added scene based SDG example test
  >   - Added object based SDG example test
  >   - Added AMR navigation example test
  >   - Switched to RealtimePathTracing in the motion blur example
  >   - Updated replicator examples to use replicator functional api where applicable
  >   - Writers use explicit backends to write data to disk
  >   - Changed data augmentation tests to use a fixed seed in the kernel functions as well, updated golden images
  >   - UR10 palletizing example uses realtime pathtracing and backend for its writer
  >   - Switched to core.experimental rigid prims where applicable
  >   - Switched to SimulationManager instead of World
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.replicator.mobility\_gen**

  > + Changed
  >
  >   - Fix clang tidy issues in cpp code
  >   - Update to Kit 109 and Python 3.12
  >   - Update deprecated python unittest methods
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.replicator.mobility\_gen.examples**

  > + Changed
  >
  >   - Fix USD path for placement of front camera on Carter with latest USD asset
  >   - Fix issue where H1 and Spot policies command must be provided as torch tensor
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.replicator.mobility\_gen.ui**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.replicator.synthetic\_recorder**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Separated recorder and UI configuration
  >   - Refactored to use explicit backend\_type and backend\_params for custom backend support
  >   - Added file validation to tests and reorganized them by specific test cases for clarity and speed
  >   - Renamed test files (test\_recorder\_outputs.py, test\_recorder\_timeline.py)
  >   - Added colorize\_depth parameter for depth visualization
* **isaacsim.replicator.writers**

  > + Changed
  >
  >   - Deprecate DOPEWriter and YCBVideoWriter writers
  >   - Deprecate OgnPose and OgnDope nodes
  >   - Updated pose writer to support explicit backends
  >   - Updated pose writer tests to use golden images and functional API
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.robot.manipulators**

  > + Changed
  >
  >   - Update description
  >   - Add missing docstrings
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.robot.manipulators.examples**

  > + Added
  >
  >   - Franka Pick-and-Place and UR10 Follow Target interactive examples
  > + Removed
  >
  >   - Build window function use
  > + Changed
  >
  >   - Updated description
  >   - Add missing docstrings
  >   - Update imports from isaacsim.base\_samples to isaacsim.examples.base
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.robot.manipulators.ui**

  > + Changed
  >
  >   - Update description
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.robot.policy.examples**

  > + Changed
  >
  >   - Removed rendering manager test time dependency (moved to base sample)
  >   - Removed unnecessary dependencies
  >   - Removed remaining experimental api references
  >   - Changed the backend to experimental API using warp and torch
  >   - Enabled GPU physics to inference policies
  >   - Moved policy based interactive examples to the isaacsim.robot.policy.examples folder
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.robot.schema**

  > + Changed
  >
  >   - Add missing docstrings
  >   - Fixed parsing of robot tree to ignore bodies in joints that are not rigid bodies
  >   - Updated Robot Schema definitions:
  >   - Removed Attributes for DofOrder
  >   - Created DofOrderOP list to be used with DofType tokens
  >   - Updated Add RobotAPI util such that it automatically scans the robot prim for Links and joints and populates it in the traversal order
  >   - Update to Kit 109 and Python 3.12
  >   - Fixed issue in \_\_init\_\_.py with running with coverage.py
* **isaacsim.robot.surface\_gripper**

  > + Changed
  >
  >   - Fix clang tidy issues in cpp code
  >   - Update to Kit 109 and Python 3.12
  >   - Performance Updates
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Update test cases to use Python’s compliant regex when instantiating the view class
* **isaacsim.robot.surface\_gripper.ui**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.robot.wheeled\_robots**

  > + Changed
  >
  >   - Add missing docstrings
  >   - Update to Kit 109 and Python 3.12
  >   - Delete deprecated AckermannControllerDeprecated node
  >   - Delete deprecated ackermann\_controller\_deprecated.py file
  >   - Fix invalid escape sequences
  >   - Update deprecated python unittest methods
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated dependencies
* **isaacsim.robot.wheeled\_robots.ui**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.robot\_motion.lula**

  > + Changed
  >
  >   - Update fix build issues with py 3.12 lula package
  > + Fixed
  >
  >   - Issue with python bindings for lula working with numpy 2.x
* **isaacsim.robot\_motion.lula\_test\_widget**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Add missing docstrings
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.robot\_motion.motion\_generation**

  > + Changed
  >
  >   - Increased tolerances on flaky tests in tests/test\_trajectory\_generator.py.
  >   - Update to Kit 109 and Python 3.12
  >   - Fix invalid escape sequences
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove extra carb settings from tests
* **isaacsim.robot\_setup.assembler**

  > + Changed
  >
  >   - Fix event name usage.
  >   - Migrate to Events 2.0.
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove extra carb settings from tests
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.robot\_setup.gain\_tuner**

  > + Removed
  >
  >   - Remove unused import statement and commented code
  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
  > + Fixed
  >
  >   - Fixed consumption of events downstream on UI builder
* **isaacsim.robot\_setup.grasp\_editor**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
  > + Fixed
  >
  >   - Fixed Events 2.0 assets loaded and timeline play / stop events
* **isaacsim.robot\_setup.xrdf\_editor**

  > + Changed
  >
  >   - Considers visual mesh scaling when generating collision spheres.
  >   - No longer deletes portions of the robot prim when generating collision spheres.
  >   - Migrate to Events 2.0.
  >   - Add missing docstrings
  >   - Update deprecated numpy in1d to np.isin
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.ros2.bridge**

  > + Changed
  >
  >   - Split extension into multiple extensions.
  >   - isaacsim.ros2.core: Core ROS 2 libraries and backend functionality
  >   - isaacsim.ros2.examples: ROS 2 examples
  >   - isaacsim.ros2.nodes: ROS 2 OmniGraph nodes and components
  >   - isaacsim.ros2.ui: ROS 2 UI components
  >   - Replace import statements with the deprecation function when importing PyTorch in tests
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated dependencies
* **isaacsim.ros2.sim\_control**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.ros2.tf\_viewer**

  > + Changed
  >
  >   - Added CUDA build dependencies
  >   - Update to Kit 109 and Python 3.12
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.ros2.urdf**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.sensors.camera**

  > + Added
  >
  >   - Unit test for get\_view\_matrix\_ros
  > + Changed
  >
  >   - Added validation checks and warmup warnings to camera sensor data methods to handle unavailable data
  >   - Added warmup tests for camera sensor checking for warnings and data availability
  >   - Migrate to Events 2.0.
  >   - Replace import statements with the deprecation function when importing PyTorch
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated time related APIs from CoreNodes interface
  > + Fixed
  >
  >   - Removed do\_array\_copy=True workaround in tiled sensor (fixed upstream in replicator.core 1.12.32 by changing strides type from int32 to int64 to avoid warp array arithmetic when getting annotator data)
  >   - Fixed issue with tiled sensor data slicing by copying the data from the annotator (do\_array\_copy=True)
* **isaacsim.sensors.camera.ui**

  > + Added
  >
  >   - New Realsense category, with D455, D457, and D55 models.
  > + Removed
  >
  >   - Intel as category
  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.sensors.physics**

  > + Added
  >
  >   - Add dedicated GPU codepath for IMU to use separate stream and pinned memory buffer
  > + Changed
  >
  >   - Fix clang tidy issues in cpp code
  >   - Migrate to Events 2.0.
  >   - Update to Kit 109 and Python 3.12
  >   - Update deprecated python unittest methods
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove deprecated time related APIs from CoreNodes interface
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.sensors.physics.examples**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.sensors.physics.ui**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.sensors.physx**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Update to Kit 109 and Python 3.12
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove extra carb settings from tests
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.sensors.physx.examples**

  > + Changed
  >
  >   - Migrate to Events 2.0.
  >   - Fix invalid escape sequences
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Migrate PhysX subscription and simulation control interfaces to Omni Physics
* **isaacsim.sensors.physx.ui**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.sensors.rtx**

  > + Added
  >
  >   - IsaacCreateRTXRadarPointCloud annotator to explicitly support RTX Radar
  >   - Link sensor\_checker utility as Python module
  >   - Add sensor\_checker to unit tests to verify supported Lidar configs
  > + Removed
  >
  >   - No longer dependent on RtxSensorMetadata AOV
  > + Changed
  >
  >   - RtxLidar.get\_object\_ids correctly handles GenericModelOutput.objId
  >   - Migrate to Events 2.0.
  >   - OgnIsaacCreateRTXLidarScanBuffer uses lambda function to initialize and allocate buffers
  >   - OgnIsaacCreateRTXLidarScanBuffer includes support for RTX Radar metadata
  >   - Fix tests after updating to kit 109.0.1.
  >   - Compute maximum points per Lidar scan from Lidar configuration
  >   - Update to Kit 109 and Python 3.12
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Exclude Simple Example Solid State config from tests
* **isaacsim.sensors.rtx.ui**

  > + Changed
  >
  >   - Change test to use RealTimePathTracing render mode
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.simulation\_app**

  > + Added
  >
  >   - Add carb settings for RealTimePathTracing mode
  >   - Fix create\_new\_stage not working correctly
  > + Changed
  >
  >   - Increased MAX\_FRAMES in \_wait\_for\_viewport for Windows so NEW\_FRAME event fires when expected (again)
  >   - Change startup behavior so that app ready status is delayed until after the app has started
  >   - Increased MAX\_FRAMES in \_wait\_for\_viewport for Windows so NEW\_FRAME event fires when expected
  >   - Add missing docstrings
  >   - Change default renderer to RealTimePathTracing
* **isaacsim.storage.native**

  > + Added
  >
  >   - Added resolve\_asset\_path function to synchronously resolve asset paths with the same logic as the async variant.
  > + Changed
  >
  >   - Update description
  >   - Add missing docstrings
  >   - Add docstring tests
  >   - Update assets path
  > + Fixed
  >
  >   - Update assets path
* **isaacsim.test.collection**

  > + Removed
  >
  >   - Remove commented code
  > + Changed
  >
  >   - Updated deprecated imports to isaacsim.storage.native
  >   - Update deprecated python unittest methods
  >   - Make omni.isaac.ml\_archive an explicit test dependency
  >   - Remove extra carb settings from tests
* **isaacsim.test.docstring**

  > + Changed
  >
  >   - Update description
  >   - Add API documentation
  >   - Add missing docstrings
  >   - Add more example usage to documentation
* **isaacsim.test.utils**

  > + Added
  >
  >   - Specify –/app/settings/fabricDefaultStageFrameHistoryCount=3 for startup test
  >   - Added compare\_images\_in\_directories() function to compare images in two directories
  > + Removed
  >
  >   - Remove omni.replicator.core as an explicit test dependency
  > + Changed
  >
  >   - Update description
  >   - Add omni.replicator.core as an explicit dependency for image capture utils
  >   - Fix invalid escape sequence
* **isaacsim.ucx.core**

  > + Added
  >
  >   - Add UCXListenerRegistry::tryRemoveListener for reference-counted listener cleanup
  >   - Added UCXListener::tagSendWithRequest for better monitoring of send requests.
  >   - Added UcxUtils.h.
  >   - Added UCX Python dependencies.
  > + Changed
  >
  >   - Fix issues found by clang tidy
  >   - Regenerate pip prebundle
  >   - Refactored UCXListener’s tag messaging functions.
* **isaacsim.util.camera\_inspector**

  > + Changed
  >
  >   - Make omni.isaac.ml\_archive an explicit test dependency
* **isaacsim.util.physics**

  > + Changed
  >
  >   - Update description
  >   - Add missing docstrings
* **omni.isaac.core\_archive**

  > + Changed
  >
  >   - Update to kiwisolver-1.4.5
  >   - Removed numba and gunicorn from dependencies
  >   - Remove omni.pip.cloud from dependencies, users should explicitly enable if needed
  >   - Remove unused tornado and pint packages
  >   - Remove markupsafe from dependencies, its in omni.kit.pip\_archive
* **omni.kit.loop-isaac**

  > + Fixed
  >
  >   - Issue where setting manual mode to false in the carb settings did not work if set before app startup completed
* **omni.pip.cloud**

  > + Changed
  >
  >   - Update to aioboto3==15.2.0
  >   - Update to aiobotocore==2.24.2
  >   - Update to boto3==1.40.18
  >   - Update to botocore==1.40.18
  >   - Update to msal==1.29.0
* **omni.pip.compute**

  > + Changed
  >
  >   - Update opencv-python-headless==4.12.0.88

- Known Issues
- General
- Warnings
- Errors
- Hang
- Crash

---

# Known Issues

## General

1. On some Windows systems, you may encounter an error like the following:

   > ```python
   > OSError: [WinError 126] The specified module could not be found. Error loading "C:\path\to\omni.isaac.ml_archive\pip_prebundle\torch\lib\fbgemm.dll" or one of its dependencies.
   > ```
   >
   > This issue is caused by missing build tools. To resolve it, install Visual Studio 2022 and then install `MSVC v143 - VS 2022 c++ x64/86 build tools` through the Visual Studio interface.
2. The replicator Scatter3D OmniGraph node breaks physics when called on a stage using world.
3. If running NVIDIA Isaac Sim headless connected via the remote client and you exit on shutdown, the following error can occur, it can be ignored:

   > ```python
   > [ext: omni.physx] shutdown
   > Fatal Python error: Segmentation fault
   >
   > Thread 0x00007f46f8faa740 (most recent call first):
   > File "..._build/target-deps/kit_sdk_release/_build/linux-x86_64/release/extsPhysics/omni.physx/omni/physx/scripts/extension.py", line 30 in on_shutdown
   > File "..._build/target-deps/kit_sdk_release/_build/linux-x86_64/release/plugins/bindings-python/omni/ext/impl/_internal.py", line 225 in shutdown_all
   > File "..._build/target-deps/kit_sdk_release/_build/linux-x86_64/release/plugins/bindings-python/omni/ext/impl/_internal.py", line 261 in shutdown_all_extensions
   > File "..._build/target-deps/kit_sdk_release/_build/linux-x86_64/release", line 3 in <module>
   > ```
4. When running in a windowed container, the following errors may be ignored and the app continues to run after waiting for awhile:

   > ```python
   > ERROR: Could not find a version that satisfies the requirement psutil (from versions: none)
   > ERROR: No matching distribution found for psutil
   > WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.VerifiedHTTPSConnection object at 0x7fcdcc284dd8>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution',)': /simple/psutil/``
   > ```
5. If exiting standalone Python scripts with `Ctrl-C`, it may need to be done twice to exit.
6. If more than one asset in URDF contains the same material name, only one material is created. Regardless if the parameters in the material are different. For example, if two meshes have materials with the name “material”, one is blue and the other is red, both meshes will be either red or blue. This also applies for textured materials.
7. MJCF importer does not show the built-in bookmark in the file picker dialog. The bookmark is still available in the content pane and can be copy-pasted into the file picker dialog.
8. If you see a black screen when running on Windows, use the `--vulkan` command-line argument during startup.
9. Debug Visualizations are not present in fisheye lens cameras that are not pinhole, because that feature is not implemented.
10. Assigning a viewport resolution that exceeds the available VRAM results in the application throwing `ERROR_OUT_OF_DEVICE_MEMORY` errors. Subsequently, reducing the resolution to a smaller value may lead to a crash.
11. When using `World` or `SimulationContext` from `isaacsim.core.api` with OmniGraph, make sure the graphs are created before the World or SimulationContext are initialized.
12. Using [replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/basic_functionalities.html "(in Omniverse Extensions)")’s `rep.new_layer()` functionality, which creates a new layer in which to place and randomize assets, may lead to issues in simulation scenarios where these assets are used. In such cases the use of `rep.new_layer()` can be omitted.
13. When running through Python, selecting physics objects and then another object on screen may result in one or more `omni.kit.manupulator` errors. This error is non-detrimental to the code execution and may be safely ignored.
14. Dragging and dropping an asset with default values in OmniGraph nodes should be saved with scene and reloaded before hitting `Play` to make sure all values are correctly set.
15. Error on exiting ./isaac-sim.streaming.sh from UI.

    > ```python
    > [Error] [carb.livestream-rtc.plugin] nvstPushStreamData timeout for eye 0, stream 0000000000000000.
    > ```
16. Cortex samples with ROS synchronization may perform abnormally and not be able to execute the task if the running FPS drops below 25 FPS.
17. If randomized materials are not loaded on time for synthetic data generation the [rt\_subframes](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)") must be set to be at least 2.
18. Some grippers with parallel mechanism (that is, Robotiq 2F-85 and 2F-C2) have links that do not move with rest of the gripper.
19. [Multiple Robot ROS2 Navigation](ROS_2.md) tutorial has high CPU usage. If you observe instances of robots colliding or experiencing localization issues, it’s likely because the Nav2 stack is unable to properly synchronize with sensor data, resulting in missed controller commands.
20. There can be many warnings and other messages when running Isaac Sim. The amount of log output can be reduced by using the following command line arguments:

    > ```python
    > --/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error
    > ```
21. Using Replicator to write to S3 buckets with the built-in backend in Windows may require setting the credentials in the environment variables instead of the AWS config files. This is because of a possible path parsing error in boto3 on Windows.
22. XR extensions do not work properly on Windows.
23. When running standalone examples in Windows, in some scenarios, threads may not be properly cleaned up when the application is closed. This can usually be ignored because the application will still successfully close. As a workaround, you can add multiple `standalone_app.update()` calls before calling `standalone_app.close()`.

    > ```python
    > Windows fatal exception: access violation
    >
    > Thread 0x00000634 (most recent call first):
    > ```
24. The ROS 2 QoS Profile OmniGraph node is unable to save custom profiles unless you manually change the createProfile input to “Custom” first before updating the other fields.
25. On some multi GPU systems when creating a render product the main viewport will go black, the render product will continue to work correctly
26. USD to URDF Exporter:
    :   * The Collider meshes may be improperly included in the visuals. They can be manually removed from the URDF file.
        * The Body and Joints are authored in the URDF file in alphabetical order. They can be manually reordered in the URDF file.
        * Depending on the robot structure, some body names may be overriden due to the merging of different frames. Review the output and verify that it’s accurate.
        * The URDF exporter adds joint effort and velocity limits as inf when unbounded. This may make the URDF not import correctly if the URDF parser does not support inf values in Float.
27. In certain instances, prolonged execution of the ROS 2 `carter_warehouse_navigation.usd` sample scene or the ROS 2 Joint State publisher with the `franka.usd` asset may lead to a memory leak.
28. The Isaac Sim asset path does not work directly with the Omniverse Kit file picker dialog. As a workaround, when using an S3 asset path with the Omniverse Kit file picker, copy and paste the path and hit `enter` instead of clicking **Select**.

    > 
29. Gains produced by the gain turner may not perfectly track the robot’s commanded movements. (E.g. as seen in the Cobotta Pro robot)
30. URDF files links, joints, and meshes must comply with USD naming conventions to import with the URDF importer. Link names, joint names, mesh names cannot contain special characters, and cannot start with an underscore, or numbers.
31. When navigating assets to import, if the folder name contains a supported extension type at the end (e.g. `*.stp, *.obj, *.urdf`), the asset browser will show the import options for the supported format, and a pre-import procedure may happen, which could cause an error message to appear in the log. This message can be safely ignored.
32. Replicator synthetic data generation may require more subframes to be rendered for scenes with significant changes (e.g. moving objects or changing lighting conditions). See [RT Subframes Parameter](Synthetic_Data_Generation.md) and [subframes examples](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)") for more information.
33. Franka Open Drawer example is not able to open the drawer by default on Blackwell GPU, increase the `self._physics_rate` to 600 will work correctly. The issue is currently under investigation.
34. When running in standalone mode, the Replicator `CosmosWriter` might skip generating videos from the recorded frames. In this case, run a few app updates before detaching the writer to make sure the videos will be generated.
35. When using Replicator for synthetic data generation (SDG) workflows, it is recommended to set the DLSS model to Quality mode to avoid rendering artifacts. At lower resolutions (especially below 600x600), the default Performance mode may cause issues such as transparent or incorrectly rendered edges in the generated images.

    > ```python
    > import carb.settings
    >
    > # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    > carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
    > ```
36. When using Replicator, frames may be skipped due to the `isaacsim.core.throttling` extension toggling `/app/asyncRendering=True` by default when the timeline is stopped. Since Replicator remains in STARTED mode, it does not re-initialize and toggle the setting back to False, leading to frames being skipped. To resolve this issue, launch with the following flag to disable async rendering toggling:

    > ```python
    > --/exts/isaacsim.core.throttling/enable_async=false
    > ```
37. When running SDG pipelines with Replicator in standalone mode on Windows, the first frame may be skipped by writers or the data might be missing in annotators. As a workaround, add an extra capture call (`rep.orchestrator.step()`) before the SDG pipeline starts to ensure all frames are recorded correctly. See [First Frame Missing in Windows Standalone Mode](Synthetic_Data_Generation.md) for details.

## Warnings

1. Warnings similar to the following can be ignored:

   > * `[Warning] [omni.usd] Warning (secondary thread)`
   > * `[Warning] [carb.tasking.plugin] Counter 0x7f25e002f8d0`
   > * `[Warning] [rtx.neuraylib.plugin] [MDLC:COMPILER]   1.0   MDLC   comp warn`
   > * `[Warning] [rtx.mdltranslator.plugin] Unable to resolve`
   > * `[Warning] [omni.tagging.plugin] Failed to discover tagging service`
   > * `[Warning] [omni.isaac.dynamic_control.plugin] DcFindArticulationDof: Function called while not simulating`
   > * `[Warning] [omni.isaac.dynamic_control.plugin] DcSetDofProperties: Function called while not simulating`
   > * `[Warning] [omni.client.plugin]  Tick: authentication: Could not connect to discovery service at "wss://...`
2. If Physics is not needed this message can be ignored:

   > * `[Warning] [omni.physx.plugin] Physics USD: Physics scene not found. A temporary default PhysicsScene prim was added automatically!`
3. If there is unwanted noise in simulated depth images, disable anti-aliasing under the **Render Settings > Ray Tracing > Anti-Aliasing** tab by setting the `Algorithm` to `None`.
4. Pyperclip is used to copy text in some extensions, if you see the following message refer to the link to install a supported copy/paste mechanism:

   > ```python
   > Pyperclip could not find a copy/paste mechanism for your system.
   > For more information, see https://pyperclip.readthedocs.io/en/latest/index.html#not-implemented-error
   > ```
5. If the parent prim of a joint does not correspond to body 0 then the value returned from physX will be the negation of the USD value.
6. If you see the following UI warning, followed by a call stack, it can be ignored:

   > * `[Warning] [omni.ui] Container::addChild attempting to add a child during a draw callback`
7. If you see warnings similar to the following, they can be ignored:

   > ```python
   > [Warning] [omni.client.python] Detected a blocking function. This will cause hitches or hangs in the UI. Please switch to the async version
   > ```
   >
   > Make sure you have access to the Internet or a Nucleus server.

## Errors

1. Errors similar to the following can be ignored when running headless:

   > ```python
   > [Error] [carb.windowing-glfw.plugin] GLFW initialization failed.
   > ```
   >
   > ```python
   > [Error] [carb] Failed to startup plugin carb.windowing-glfw.plugin (interfaces: [carb::windowing::IGLContext v1.0],[carb::windowing::IWindowing v1.2]) (impl: carb.windowing-glfw.plugin)
   > ```
   >
   > ```python
   > [Error] [carb.scripting-python.plugin] RuntimeError: Failed to acquire interface: carb::windowing::IWindowing (pluginName: nullptr)
   >
   > At:
   > /isaac-sim/kit/exts/omni.kit.window.cursor/omni/kit/window/cursor/cursor.py(27): on_startup
   > /isaac-sim/kit/plugins/bindings-python/omni/ext/impl/_internal.py(141): _startup_ext
   > /isaac-sim/kit/plugins/bindings-python/omni/ext/impl/_internal.py(174): startup_all_extensions_in_module
   > /isaac-sim/kit/plugins/bindings-python/omni/ext/impl/_internal.py(225): startup_all_extensions_in_module
   > PythonExtension.cpp::startup()(2): <module>
   >
   > [Error] [omni.ext.plugin] [ext: omni.kit.window.cursor-1.0.1] Failed to process python module extension in '/isaac-sim/kit/exts/omni.kit.window.cursor/.'.
   > ```
2. These errors can be ignored while running the Omniverse Streaming Remote Client, if the client works as normal:

   > ```python
   > ERROR [BifrostClient: Streamer] {2B436200} -  updateVideoSettingsForNVbProfile: profile 8 is not handled
   > ERROR [NVST:ClientSession] {1E8D2700} -  Number of channels(2) is not valid for surround configuration
   > ERROR [NVST:ClientSession] {1E8D2700} -  Either in stereo or error in receiving opus information from server
   >
   > ERROR [LAVCDecoder] {FC4A0700} - GERONIMO_ERROR 0xC0040006 GERONIMO_LAVCDECODER_DECODE: Packet decode failure.
   > ERROR [GIOInterface] {DAFFD700} - GERONIMO_ERROR 0xC0020003 GERONIMO_IOINTERFACE_INVALID_AUDIO_FUNC: Audio Renderer is NULL.
   > ERROR [NVST:ClientLibraryWrapper] {1E8D2700} -  Cannot find streamEventRaised for stream.media type 1
   > ERROR [NVST:ClientLibraryWrapper] {1E8D2700} -  Cannot find streamEventRaised for stream.media type 2
   >
   > ERROR [NVST:RtspSessionPocoBase] {FB280700} -  perform() failed: 0
   > ERROR [NVST:RtspPocoEvent] {FB280700} -  RTSP-XNvEvent Polling failed: 0, rc: 408
   >
   > ERROR [NVST:UdpRtpSource] {F9A7D700} -  UDP RTP Source: failed to receive data (Error: 0x80000013)
   > ERROR [NVST:RtpSourceQueue] {F9A7D700} -  RtpSourceQueue: failed to read RTP packet (Result: 0X80000013)
   > ERROR [BifrostClient: NvscWrapper] {1D0CF700} - Received old frame - current 7536 received 0
   > ERROR [BifrostClient: Interface] {2B436200} - nvbSendInputEvent(). SessionIdentifier is ''
   > ```
   >
   > ```python
   > ERROR [Geronimo::Analytics] {875D4140} - Failed to load dll with error: /../..//NvTelemetryAPI.so: cannot open shared object file: No such file or directory
   > ERROR [BifrostClient: Streamer] {875D4140} - updateVideoSettingsForNVbProfile: profile 8 is not handled
   > ERROR [NVST:ClientSession] {7EC38640} - Number of channels(2) is not valid for surround configuration
   > ERROR [NVST:ServerControl] {4BFFF640} - Unknown server notification
   > ERROR [GIOInterface] {49FFB640} - GERONIMO_ERROR 0xC0020003 GERONIMO_IOINTERFACE_INVALID_AUDIO_FUNC: Audio Renderer is NULL.
   > ERROR [NVST:ClientLibraryWrapper] {7EC38640} - Cannot find streamEventRaised for stream.media type 1
   > ERROR [NVST:ClientLibraryWrapper] {7EC38640} - Cannot find streamEventRaised for stream.media type 2
   > [h264 @ 0x55580065b160] Cannot parallelize slice decoding with deblocking filter type 1, decoding such frames in sequential order
   > To parallelize slice decoding you need video encoded with disable_deblocking_filter_idc set to 2 (deblock only edges that do not cross slices).
   > Setting the flags2 libavcodec option to +fast (-flags2 +fast) will disable deblocking across slices and enable parallel slice decoding but will generate non-standard-compliant output.
   > ERROR [BifrostClient: Streamer] {875D4140} - Sending Input event failed: Nvsc Error: NVST_R_GENERIC_ERROR (0x800b0000)
   > ```
3. Error similar to the following happens when STOPPING and STARTING simulation again when using Isaac core world class. To stop the error trail, Reset the scene with one of the `world.reset` methods (in the core samples, this happen when pressing the RESET button on the UI).

   > ```python
   > 2021-06-01 19:16:51 [65,842ms] [Error] [omni.kit.app._impl] [py stderr]: AttributeError: 'NoneType' object has no attribute '<...>'
   >
   > At:
   > <...>
   > ```
4. If an asset contains an Action Graph and the Action Graph window is closed before re-opening the same asset, the following errors may appear and can be ignored:

   > ```python
   > [Error] [omni.usd] TF_PYTHON_EXCEPTION (secondary thread): in TfPyConvertPythonExceptionToTfErrors at line 114 of /buildAgent/work/ca6c508eae419cf8/USD/pxr/base/tf/pyError.cpp -- Tf Python Exception
   >
   > [Error] [omni.kit.app.impl] [py stderr]: sys:1: RuntimeWarning: coroutine 'OmniGraphModel.__delayed_prim_changed' was never awaited
   > RuntimeWarning: Enable tracemalloc to get the object allocation traceback
   >
   > [Error] [omni.graph] Invalid GraphObj object in Py_Graph in getNode
   > ```
5. Errors when converting ShapeNet models may appear from the `omni.kit.asset_converter` extension when textures for the target model are missing from the input dataset. These errors can be ignored.

   > ```python
   > [Error] [omni.kit.asset_converter.impl.omni_client_wrapper] Cannot copy from */images/texture2.jpg to */textures/texture2.jpg, error code: Result.ERROR_NOT_FOUND.
   > ```
6. Errors when livestreaming. These errors can be ignored.

   > ```python
   > [Error] [carb.livestream.plugin] nvstPushStreamData timeout for eye 0, stream (nil).
   > ```
7. Errors when using a Jupyter notebook. These errors can be ignored.

   > ```python
   > [Error] [omni.kit.app.plugin] Can`t delay app ready event, it was already sent. Requester name: omni.usd
   > ```
8. Errors while generating samples with `flying_things_4d.yaml` for larger value of `--num-scenes` (specially when running for stress test):

   > ```python
   > [Error] [omni.physicsschema.plugin] Rigid Body of (/Replicator/SampledAssets/Population_9658d6d1/Ref_Xform_05/Ref) missing xformstack reset when child of rigid body (/Replicator/SampledAssets/Population_9658d6d1/Ref_Xform_05) in hierarchy. Simulation of multiple RigidBodyAPIs in a hierarchy will cause unpredicted results. Please fix the hierarchy or use XformStack reset.
   > ```
9. Error like this when running Composer can be ignored.

   > ```python
   > [Error] [omni.graph.core.plugin] /Replicator/SDGPipeline/OgnSampleCombine_03: [/Replicator/SDGPipeline] Assertion raised in compute - AttributeData 'OgnSampleCombine_03.outputs:samples' of type 'float3[]' required array of elements of length 3, got array with elements of size 1
   > ```
10. When using the surface gripper between two objects that contain articulation root, the following error may appear and the surface Gripper won’t work. To avoid it, disable the Articulation API from the picked object.

    > ```python
    > [Error] [omni.physx.plugin] PhysX error: PxD6JointCreate: actors must be different
    > ```
11. Fatal error regarding `omni.sensors` plugin when running RTX Radar. Unless you have manually disabled Vulkan and MotionBVH, this error appears if you are using a below-minimum-spec GPU. Your GPU must be Ampere architecture or newer.

    > ```python
    > [Warning] [omni.sensors.nv.radar.wpm_dmatapprox.plugin] MotionBVH activation state 0 doesn\'t match requested state 1
    > [Fatal] [omni.sensors.nv.radar.wpm_dmatapprox.plugin] Running radar without MotionBVH is disallowed, to force it use --/app/sensors/nv/radar/runWithoutMBVH=true
    > If you are running on Windows and have motionBVH enabled, be sure to enable Vulkan as well by passing --vulkan
    > ```
12. Error regarding failure to process writer attach request when playing scene containing an OmniGraph, after changing timeCodesPerSecond setting.
    To resolve, save the scene, reopen it, then play it again.

    > ```python
    > [Error] [omni.graph] Invalid Node object passed to Graph.get_graph_from_node
    > [Error] [isaacsim.core.nodes.impl.base_writer_node] Could not process writer attach request (<omni.replicator.core.scripts.writers.NodeWriter object at 0x7355b3d175b0>, None), Invalid NodeObj object in Py_Node in getAttributes
    > ```
13. In `omni.replicator.object`, the description file `demo_shader_attributes_diffuse.yaml` can have a corrupted JPEG error that the picture is not written before it’s used. We are looking into fixing it.

    > ```python
    > [Error] [gpu.foundation.plugin] Couldn\'t process /tmp/carb.F3srY8/randomized_output.jpg, it might not have written completely. Reason: Failed to load image: Corrupt JPEG
    > ```
14. omni.usd LoadModule errors can be ignored.

    > ```python
    > [Error] [omni.usd] USD_MDL: in LoadModule ...
    > ```
15. WinError 123 errors similar to below may appear when clicking an asset in the Isaac Sim Asset Browser. These errors can be ignored.

    > ```python
    > [WinError 123] The filename, directory name, or volume label syntax is incorrect: `https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/Environments/Simple_Warehouse/full_warehouse.usd`
    > ```
16. Following errors can be ignored when running ROS 2 Navigation and ROS\_Nav2\_Waypoint\_Follower ActionGraph. These errors occur when starting and stopping simulation without ticking the impulse node inside the ROS\_Nav2\_Waypoint\_Follower ActionGraph. The prim attribute for OrientReadPrimAttribute and TranslateReadPrimAttribute nodes will be set after ticking this graph.

    > ```python
    > [Error] [omni.graph.core.plugin] /Graph/ROS_Nav2_Waypoint_Follower/OrientReadPrimAttribute: [/Graph/ROS_Nav2_Waypoint_Follower] Prim has not been set
    > [Error] [omni.graph.core.plugin] /Graph/ROS_Nav2_Waypoint_Follower/TranslateReadPrimAttribute: [/Graph/ROS_Nav2_Waypoint_Follower] Prim has not been set
    > ```
17. If you are encountering any issues regarding the dependencies on `omni.replicator.character` or `omni.replicator.agent`, the extension is now renamed to `isaacsim.replicator.agent`. Revise your code accordingly.
18. CUDA driver failures from the `omni.sensors.nv.lidar.lidar_core.plugin` (example below) on Ubuntu may be due to a system-level CUDA installation mismatch with the `omni.sensors` runtime-compiled libraries.

    > ```python
    > [Error] [omni.sensors.nv.lidar.lidar_core.plugin] CUDA Driver CALL FAILED at line 522: the provided PTX was compiled with an unsupported toolchain.
    > [Error] [omni.sensors.nv.lidar.lidar_core.plugin] CUDA Driver CALL FAILED at line 548: named symbol not found
    > ```
    >
    > One workaround may be to set the `LD_LIBRARY_PATH` enviroment variable as follows, where `</path/to/isaac_sim_installation>` should be replaced with the path to your local Isaac Sim installation.
    >
    > ```python
    > export LD_LIBRARY_PATH=</path/to/isaac_sim_installation>/extscache/omni.sensors.nv.common-2.5.0-coreapi+lx64.r.cp310/bin:$LD_LIBRARY_PATH
    > ```
19. Physics Inspector “failed to find internal joint” errors for robots with mimic joints does not affect the functionality of the mimic joints and can be ignored.

    > ```python
    > [Error] [omni.physx.plugin] Usd Physics: failed to find internal joint object for PhysxMimicJointAPI at /Franka/panda_hand/panda_finger_joint2. Please ensure that the prim is a supported joint type and is part of an articulation.
    > ```
20. The `omni.kit.telemetry` extension startup error with code `(error = 206)` on Windows is caused by a file path exceeding the length limit. Verify that the file path of `omni.telemetry.transmitter.exe` does not exceed 260 characters.
21. If you encounter the error message `Windows fatal exception: int divide by zero` once the app is started, it could be due to GPU overclocking software such as MSI Afterburner. Try disabling the software to resolve the issue.”
22. Python errors related to `tkinter` like the following indicate the user is attempting to use `tkinter` with the Python distribution shipped with Isaac Sim. This is not supported.

    > ```python
    > File "/path/to/isaac_sim/installation/kit/python/lib/python3.11/tkinter/__init__.py", line 38, in <module>
    >     import _tkinter # If this fails your Python may not be configured for Tk
    >     ^^^^^^^^^^^^^^^
    > ModuleNotFoundError: No module named '_tkinter'
    > ```
23. Error when using [depth sensor AOVs](Sensors.md). The AOV number (eg. 38 below) may change, depending on the selected AOV.

    > ```python
    > [Error] [rtx.postprocessing.plugin] DepthSensor: Texture sizes do not match: inColorTexDesc 1920x1080x1:11@0 inDepthTexDesc 1500x843x1:33@0
    > [Error] [rtx.postprocessing.plugin] DepthSensor: Failed to allocate view resources for view 1 device 0
    > [Error] [carb.scenerenderer-rtx.plugin] Failed to export AOV 38 to render product. The renderer did not generate the AOV texture
    > ```

## Hang

1. On windows, when using the extension manager, clicking on the dependencies window will cause the viewport to go black and hang.
2. The WebRTC Client on Firefox may appear to hang after a few seconds when clicking the Play button. Using the Google Chrome or Chromium browser is recommended.
3. Isaac Sim may hang if a browser pop-up for logging into Nucleus is closed before completing the login. Force restart of Isaac Sim is required.

## Crash

1. Using compound nodes in Omnigraph may lead to a crash, we do not recommend using compound nodes in Omnigraph.
2. Shutting down the physics.tensors extension before the Python garbage collector cleans up the related objects can lead to a crash. To prevent this, manually set the related tensor API objects in Python to None before unloading the extension.