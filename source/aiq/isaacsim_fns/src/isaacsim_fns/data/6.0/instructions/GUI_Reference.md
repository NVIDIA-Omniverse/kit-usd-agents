# GUI Reference

The GUI interface provides access to most of the tools available in Isaac Sim. Here are the frequently used general tools in Isaac Sim. Many additional GUI-based tools can also be found [Omniverse Extension Documentations](https://docs.omniverse.nvidia.com/extensions/latest/ext_core.html "(in Omniverse Extensions)").

- User Interface Reference
- Keyboard Shortcuts Reference
- Create Menu
- Replicator Menu
- Preferences
- Selection Modes
- Examples Browser

## GUI Extensions Used in Isaac Sim

These items link to GUI elements that are extensions that are maintained by other products:

* [Opening Page](GUI_Reference.md)
* [Viewport](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_viewport.html "(in Omniverse Extensions)")
* [Extension Manager](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html "(in Omniverse Extensions)")
* [Stage](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_stage.html "(in Omniverse Extensions)")
* [Property Window](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_property-panel.html "(in Omniverse Extensions)")
* [Script Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_script-editor.html "(in Omniverse Extensions)")
* [Layers](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_layers.html "(in Omniverse Extensions)")
* [Console](https://docs.omniverse.nvidia.com/extensions/latest/ext_console.html "(in Omniverse Extensions)")
* [Layout Templates](layouts.html#isaac-sim-app-gui-layouts)

---

# User Interface Reference

[NVIDIA Omniverse™ Isaac Sim](../overview/overview.html) is built on [NVIDIA Omniverse](https://docs.omniverse.nvidia.com/) platform, so it shares the same UI elements as many Omniverse apps.

## Opening Page

Here’s a summary of the Isaac Sim frequently mentioned elements on the opening page. For more detailed view of all the elements on the page, go to [Omniverse User Interface](https://docs.omniverse.nvidia.com/composer/latest/interface.html "(in Omniverse USD Composer)").

| Ref # | Option | Result |
| --- | --- | --- |
| 1 | Menu Bar | Isaac Sim [Menu Bar](#isaac-sim-menu-bar) |
| 2 | Viewport | The primary way of viewing assets. See [Viewport](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_viewport.html "(in Omniverse Extensions)") for more details. |
| 3 | Main Toolbar | [Tool Bar](#toolbar) for manipulating the assets and start/stop simulation buttons are located. |
| 4 | Browsers | The default location for asset and example browsers. |
| 5 | Stage | The Stage window allows you to see all the assets in your current USD Scene. See the [Stage](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_stage.html "(in Omniverse Extensions)") docs for more details. |
| 6 | Property Panel | The window that displays the details of selected prim. See [Property Window](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_property-panel.html "(in Omniverse Extensions)") for more details. |

## Menu Bar

The Isaac Sim menu layout may be different from the layout of other Omniverse applications. Here are the ones unique to Isaac Sim.

| Ref # | Option | Result |
| --- | --- | --- |
| 1 | Create | The menu for creating various primitives and other simulation objects |
| 2 | Window | Opens various windows of loaded extensions, in this case, the ones composing the GUI and other extensions |
| 3 | Tools | The menu of available simulation tools for animation, physics, replicator, robotics, and USD |
| 4 | Utilities | Access various diagnostic and developer utilities such as debugging and extension templates |
| 5 | Layout | Opens menu for selecting preferred gui layouts |

## Tool Bar

| Icon | Menu Item | Action |
| --- | --- | --- |
| [tb_sel_mod](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_select_models.svg) / [tb_sel_prim](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_select_prims.svg) | [Selection Modes](GUI_Reference.md) | Allows user to pick select and object in the viewport.  This is also the default viewport mouse behavior. |
| [tb_mv_glob](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_move_global.svg) / [tb_mv_local](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_move_local.svg) | Move (Global / Local) | Instantiates a user widget that allows user to move a  selected object or group of objects |
| [tb_rot_glob](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_rotate_global.svg) | Rotate (Global / Local) | Instantiates a user widget that allows user to rotate  a selected object or group of objects |
| [tb_scl](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_scale.svg) | Scale | Instantiates a user widget that allows user to scale a  selected object or group of objects |
| [tb_snap](../_images/isim_4.5_base_ref_gui_kit_reference-guide_toolbar_snap.svg) | Snap (enable/disable) | Sets snapping to specified increments or surface snap. |
| [tb_anim_trans](../_images/isim_4.5_base_ref_gui_kit_reference-guide_animation-bar-1.png) | Select Mode | Toggles transform widgets between local and global  translation modes |
| [tb_anim_trans](../_images/isim_4.5_base_ref_gui_kit_reference-guide_animation-bar-1.png) | * Play | Start an animation |
| [tb_anim_trans](../_images/isim_4.5_base_ref_gui_kit_reference-guide_animation-bar-1.png) | * Stop | Stop an animation |

Note

Tools with a small triangle below their icon denotes additional options are available by right clicking the icon.

## Tabs

The Layout of the windows can be rearranged by moving the tabbed windows around, and docking them to different locations.

1. Panel Being Dragged (See Note below).
2. Panels Original location.
3. Acceptable Docking Locations.

Note

A tab can be “torn-off” and moved to another panel or window by click-hold-drag on the tabs title-bar and dragging it to another location or UI pane.

### OS Tabs

Certain tabs in the interface can be detached from the main window, which can be useful on multiple monitors and wide aspect ratio monitors.

To Detach a Tabbed panel use the following procedure.

1. `Right Click` on a `Tab` to invoke the `Move to New OS Window` option.

   > 
2. `Left Click` Select `Move to OS Window` action.

   > 
3. Position the window wherever you wish by using `Left-Click` + Dragging.

   > 

## Grab Handles

Grab handles are found in all Omniverse Apps and allow you to resize panels.

1. Grab Handle.

They are “invisible” UI element dividers that, when rolled over, will illuminate and can be click-dragged. This allows for UI customization, which is especially helpful in managing window content.

Note

Sliding is restricted to horizontal or vertical only.

### See Also

* [Omniverse User Interface](https://docs.omniverse.nvidia.com/composer/latest/interface.html "(in Omniverse USD Composer)") - Detailed overview of Omniverse UI elements
* [Viewport](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_viewport.html "(in Omniverse Extensions)") - In-depth guide to the viewport functionality
* [Stage](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_stage.html "(in Omniverse Extensions)") - Comprehensive documentation of the Stage window
* [Property Window](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_property-panel.html "(in Omniverse Extensions)") - Detailed guide to the Property Panel

---

# Keyboard Shortcuts Reference

Keyboard shortcuts can reduce the amount of clicking one must do by providing “hot keys” that allow for “one touch” operation.

## Most Commonly Used Shortcuts

The gizmos for manipulating an object are on the left hand side toolbar.

> * Press “W” or click on the Move Gizmo to drag and move, for example, a Cube. You can move it in only one axis by clicking on the arrows and drag, in two axes by clicking on the colored squares and drag, or in all three axes by clicking on the dot in the center of the gizmo and drag.
> * Press “E” or click on the Rotate Gizmo to rotate.
> * Press “R” or click on the Scale Gizmo to scale. You can scale in one dimension by clicking on the the arrows and drag, two dimensions by clicking on the colored squares and drag, or in all three dimensions by clicking on the circle in the center of the gizmo and drag.
> * Press “ESCAPE” to deselect an object.

## Viewport Controls

| Input | Alternate Input | Result |
| --- | --- | --- |
| RMB + W | RMB + Up Arrow | Move Forward |
| RMB + S | RMB + Down Arrow | Move Backward |
| RMB + A | RMB + Left Arrow | Move Left |
| RMB + D | RMB + Right Arrow | Move Right |
| RMB + Q | RMB + Page Up | Move Up |
| RMB + E | RMB + Page Down | Move Down |
| Scroll Wheel | Opt + RMB | Zoom |
| LMB |  | Select |
| ESCAPE |  | Deselect |
| Select + ‘F’ |  | Zoom Camera to Selected Objects |
| Deselect + ‘F’ |  | Zoom Camera to All |
| Opt + LMB |  | Orbit about the Viewport Center |
| MMB (Hold) |  | Pan |
| RMB (Hold) |  | Pivot Camera |
| RMB (Click) |  | Invoke Contextual Menus |
| Shift + H |  | Show / Hide Grid and HUD information |
| F7 |  | Enables and disables the visibility of the UI |
| F11 |  | Toggles full screen mode |
| F10 |  | Capture Screen Shot |

Note

While using any move command, Shift can be held to double the movement speed. Control can be used to halve the movement speed.

## Selection

| Input | Alternate Input | Result |
| --- | --- | --- |
| Ctrl + A |  | Selects all assets in the current scene |
| Ctrl + I |  | Selects all assets not selected and deselects all selected assets |
| Esc |  | Deselects all assets in the current scene |

## File Operations

| Input | Alternate Input | Result |
| --- | --- | --- |
| Ctrl + S |  | Save File |
| Ctrl + O |  | Open File |

## Asset Control

| Input | Alternate Input | Result |
| --- | --- | --- |
| Del |  | Deletes selected asset |
| Ctrl + Shift + I |  | Create an instance of the current asset |
| Ctrl + D |  | Duplicates current asset |
| Ctrl + G |  | Groups selected assets into a container |
| H |  | Toggles selected asset visibility |

## Animation Controls

| Input | Alternate Input | Result |
| --- | --- | --- |
| Space |  | Plays/Pauses animations |

## Custom Hotkeys

You can create your own custom hotkey combinations to work faster and more effectively by using the [Hotkeys Extension](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_hotkeys.html "(in Omniverse Extensions)").

---

# Create Menu

This table outlines the common things needed for scene creation and layout in Isaac Sim under the Create tab.

| Menu Items | Action |
| --- | --- |
| Light | Create custom Lights. See the Lighting docs for details. |
| Camera | Add Cameras to the stage. |
| Material | Create Omniverse Materials based on templates. |
| Physics | See the [Physics](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/index.html "(in Omni Physics)") docs for details. |
| Sensors | Adds [cameras](Sensors.md) and [RTX-based](Sensors.md), [PhysX-based](Sensors.md), or [physics-based](Sensors.md) sensors to the Stage. |
| Robots | Adds different robots to the Stage. |
| Environments | Adds different environments to the Stage. |
| April Tag | Adds [April Tag Asset](Isaac_Sim_Assets.md) to the Stage. |

---

# Replicator Menu

The Replicator (Tools > Replicator) menu has a suite of useful tools and extensions for [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") and generating, visualizing, and recording synthetic data.

| UI Element | Reference |
| --- | --- |
| Semantics Schema Editor | Add semantic information. Refer to the [Semantics Schema Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html "(in Omniverse Extensions)") docs for more details. |
| Synthetic Data Recorder | Synthetic Data Recorder. Refer to the [Synthetic Data Recorder](Synthetic_Data_Generation.md) docs for more details. |
| ReplicatorYAML | Generating dataset using [Replicator YAML](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/yaml_manual.html#replicator-yaml-manual "(in Omniverse Extensions)"). |
| Start | Starts the randomizations and writing to disk [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)"). |
| Step | Performs a single randomization operations with writing to disk [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)"). |
| Preview | Performs a singe randomization iteration without writing to disk [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)"). |

---

# Preferences

The Preferences panel is found under `Edit > Preferences` and hosts a list of settings that are
applicable to the current app. Here we list the most commonly modified ones. These settings apply to the entire application and are not stored with individual USD stages.

## Audio

There are some application preferences that can help to control the behavior of audio output globally in Omniverse USD Composer. These preferences affect all USD stages loaded in Omniverse USD Composer. These settings are not stored as part of the USD stage. The app preferences window can be opened by going to the **Edit** menu and choosing **Preferences**. The audio preferences can be found by selecting **Audio** in the sections list on the left.

> 

### Audio Output

Input and Output device preferences.

| Option | Description |
| --- | --- |
| Output Device | Displays a drop-down box containing the names of all audio output devices connected to the system. This may be used to select the desired device for output in Omniverse USD Presenter. This affects output for the main USD stage and all UI audio. Once a device is selected from the list, the **Apply** button must be pushed to accept the change. Changing both this setting and the speaker configuration below will cause the output of all open audio contexts to be changed. If the state of devices attached to the system has changed recently (that is: a new device was connected or a device was disconnected from the system), the **Refresh** button can be used to collect the new device list. By default, the system’s default output device will be chosen.    If the selected device is disconnected from the system between launches of Omniverse USD Presenter or the device list changes between launches, the previously selected device will attempt to be found first on the next launch. If it is still attached to the system, it will be used. If it could not be found in the device list, the system’s default output device will be used instead. |
| Input Device | Displays a drop-down box containing the names of all audio input devices connected to the system. This may be used to select the desired device to use for recording in Omniverse View. This affects input for all USD stages. Once a device is selected from the list, the **Apply** button must be pushed to accept the change. The input device is unaffected by the “Speaker Configuration” setting. If the state of devices attached to the system has changed recently (that is: a new device was connected or a device was disconnected from the system), the **Refresh** button can be used to collect the new device list. By default, the system’s default input device will be chosen.    If the selected device is disconnected from the system between launches of Omniverse USD Presenter or the device list changes between launches, the previously selected device will attempt to be found first on the next launch. If it is still attached to the system, it will be used. If it could not be found in the device list, the system’s default output device will be used instead. |
| Speaker Configuration | Sets the speaker configuration to use for output. All configurations are supported regardless of the device’s capabilities (that is: a 5.1 configuration is still supported on a stereo device). In the case the output mode is not directly supported by the selected device, the final output of the audio system will be down-mixed to the device’s preferred configuration. As much of the original stream as is possible will be preserved in the down-mixed output.    If the “auto-detect” configuration is selected, the output will try to match the device’s preferred format. Note that this could result in extra processing requirements on some devices due to the larger number of speaker channels.    The **Apply** button must be pushed (or Omniverse USD Composer relaunched) after changing this setting for this to take effect. The **Refresh** button refreshes the device lists for audio input and output. If a new device is connected to the system or an existing device is removed, pushing this button will refresh the device lists to reflect the new device sets for both input and output.    Note: this button may disappear in the future in favor of auto-detecting system device changes. |
| Apply | Applies all changes to the audio input and output device selections. This will have no effect if none of the options have changed. However, if a device is in use at the time (that is: actively recording audio or actively playing audio in a stage), this could result in a brief interruption in audio. It is best to ensure that all audio recording and playback has been stopped before pushing this button. |

### Audio Parameters

| Option | Description |
| --- | --- |
| Auto Stream Threshold | Defines the asset size at which the audio system will decide to stream a compressed audio asset instead of decompress it into memory. This threshold is expressed in kilobytes. If this is set to zero the auto-streaming feature will be disabled. If this is set to any larger value, any compressed audio asset with a decompressed size larger than this threshold will be streamed from the original compressed object instead of being decompressed. The benefit of this is lower memory usage and faster asset loading. However, streaming sounds do require slightly more processing time. The default value is 0KB. |

### Audio Player Parameters

| Option | Description |
| --- | --- |
| Auto Stream Threshold | Defines the asset size at which the audio player will decompress a compressed asset on-the-fly during playback instead of decompressing it into memory on load. This has the benefit of using less memory and allowing playback to start sooner (for large assets at least). This threshold is expressed in kilobytes. If set to zero, all assets will be decompressed before playing. If set to a non-zero value, any compressed asset with a decompressed size larger than this value will be decompressed as it plays. This defaults to 256KB. |
| Close Audio Player on Stop | Determines whether the audio player will close on its own once the first playback of a sound completes. This is useful when previewing large numbers of audio assets from the content browser. If this option is left unchecked, the audio player window will remain open after playback completes. This defaults to unchecked. |

### Volume Levels

Adjust volume properties of sounds.

| Option | Description |
| --- | --- |
| Master Volume | Defines the master volume level for all audio output. All other volume levels are effectively multiplied by this volume level to get the final overall volume. Setting this to 0.0 will result in silence (though audio data will still be fully processed). Setting this to 1.0 will be full volume. The volume level changes linearly across this range. This defaults to 1.0.    If the selected device is disconnected from the system between launches or the device list changes between launches, the previously selected device will attempt to be found first on the next launch. If it is still attached to the system, it will be used. If it could not be found in the device list, the system’s default output device will be used instead. |
| USD Volume | Defines the volume level to be used by all audio for the USD stage audio output. This affects all spatial and non-spatial sounds. Setting this to 0.0 will result in silence (though audio data will still be fully processed). Setting this to 1.0 will be full volume. The volume level changes linearly across this range. This defaults to 1.0. |
| Spatial Voice Volume | Defines the volume level to be used for all spatial sounds in the USD stage. This volume level is effectively multiplied by the “USD Volume” level setting as well before output to get the final volume level for spatial sounds. Setting this to 0.0 will result in silence (though audio data will still be fully processed). Setting this to 1.0 will be full volume. The volume level changes linearly across this range. This defaults to 1.0. |
| Non-spatial Voice Volume | Defines the volume level to be used for all non-spatial sounds in the USD stage. This volume level is effectively multiplied by the “USD Volume” level setting as well before output to get the final volume level for non-spatial sounds. Setting this to 0.0 will result in silence (though audio data will still be fully processed). Setting this to 1.0 will be full volume. The volume level changes linearly across this range. This defaults to 1.0. |
| UI Audio Volume | Defines the volume level to be used for all UI audio sounds in Omniverse USD Presenter. Setting this to 0.0 will result in silence (though audio data will still be fully processed). Setting this to 1.0 will be full volume. The volume level changes linearly across this range. This defaults to 1.0. |

### Debug

Sound debugging options.

| Option | Description |
| --- | --- |
| Stream Dump Filename | Defines the filename to be used when dumping the USD stage audio output to file. This will be written out in WAVE file format regardless of the extension on the filename. The channel count and data format will match the current output device’s selected channel count and format. This file will be written to disk as audio is played and will always try to remain within a few milliseconds of audio away from what is playing on the device (as close as possible).    The output file must be on a local file volume. Sending output to an Omniverse location is not supported. Once stream dumping is enabled, the output file will be created and it will be written to as new audio data is produced. The output will continue until stream dumping is disabled or The Omniverse App is exited. The default value for this setting is an empty string.    Note that as long as this feature is left enabled, data will continue to be written to the output file. Since this is written as uncompressed data, this file will tend to grow rather quickly. For example, a 48KHz stereo floating point signal will write approximately 22MB per minute. For this reason, the “Enable Stream Dump” setting is not persistent in this Omniverse app’s user configuration. It will always be off when the Omniverse App launches. |
| Enable Stream Dump | Defines whether stream dumping is currently enabled. As soon as this is enabled and a valid filename is selected in “Stream Dump Filename”, writing to the output file will begin. Stream dumping will continue until this setting is disabled or this Omniverse app is exited. This setting does not persist in this Omniverse app’s user configuration. It will always be disabled on a fresh launch. |

## Capture Screenshot

### Capture Screenshot

| Option | Description |
| --- | --- |
| Path to Save Screenshots | The path where captured screenshots are saved. |
| Capture only 3D viewport | Checked (Default): Will only Capture the Viewport.  Unchecked: Captures Interface and Viewport. |

## Datetime Format

### Datetime Format

| Option | Description |
| --- | --- |
| Display Date As | Sets the format of the datetime string in the screenshot filename.  MM/DD/YYYY (Default): Month/Day/Year  DD.MM.YYYY: Day.Month.Year  DD-MM-YYYY: Day-Month-Year  YYYY-MM-DD: Year-Month-Day  YYYY/MM/DD: Year/Month/Day  YYYY.MM.DD: Year.Month.Day |

## Developer

### Throttle Rendering

| Option | Description |
| --- | --- |
| Async Rendering | Toggles asynchronous rendering. This defaults to unchecked. |
| Skip Rendering While Minimized | Toggles skipping rendering while the viewport is minimized. This defaults to unchecked. |
| Yield ‘ms’ while in focus | Sets the amount of time [ms] to yield while the viewport is in focus. This defaults to 0ms. |
| Yield ‘ms’ while not in focus | Sets the amount of time [ms] to yield while the viewport is not in focus. This defaults to 0ms. |
| Enable UI FPS Limit | Limits the Viewport rendering framerate to the specified FPS Limit. This defaults to checked. |
| UI FPS Limit uses Busy Loop | Limits the Viewport rendering framerate with a busy loop. This defaults to unchecked. |
| UI FPS Limit | Sets the framerate in frames per second (FPS) if Set FPS Limit is checked. This defaults to 120 FPS. |

### Mip Mapping in ui.image

| Option | Description |
| --- | --- |
| Generate Mips | Toggles mip mapping in ui.image. This defaults to unchecked. |

## Live

### Join Live

| Option | Description |
| --- | --- |
| Quick Join Enabled | Toggles quick join for live sessions. This defaults to checked. |
| Session List Selection | Selects the live session to join. This defaults to the last session. |

## Material

### Material

| Option | Description |
| --- | --- |
| Binding Strength | Sets the binding strength for the material to be weaker or stronger than descendants. This defaults to weaker than descendants. |

### Render Context Material Network

| Option | Description |
| --- | --- |
| Render Context Material Network | If a UsdShade.Material prim contains definitions for multiple contexts, this list defines the order in which those contexts are selected. |

## Stage

### New Stage

Parameters used to establish new stages when created.

| Option | Description |
| --- | --- |
| Default Up Axis | Sets the default up axis for new stages. This defaults to Z. |
| Default Animation Rate | Sets the default animation rate for new stages. This defaults to 60.0. |
| Default Meters per Unit | Sets the default meters per unit for new stages. This defaults to 1.0. |
| Default Time Code Range | Sets the default time code range for new stages. This defaults to 0.0 to 1000000.0. |
| Default DefaultPrim Name | Sets the default default prim name for new stages. This defaults to World. |
| Interpolation Type | Sets the default interpolation type for new stages. This defaults to Linear. |
| Start with Transform Op on Prim Creation | Toggles enabling a transform op on prim creation for new stages. This defaults to checked. |
| Default Transform Op Type | Sets the default transform op type for new stages. This defaults to `Scale, Orient, Translate`. |
| Default Rotation Order | Sets the default rotation order for new stages. This defaults to ZYX. |
| Default XForm Op Order | Sets the default xform op order for new stages. This defaults to `xformOp:translate, xformOp:orient, xformOp:scale`. |
| Default XForm Precision | Sets the default xform precision for new stages. This defaults to Double. |

### Authoring

| Option | Description |
| --- | --- |
| Keep Prim World Transform when ReParenting | When reparenting a prim, this setting determines if the prim’s world transform is kept, inherited from the parent, or determined manually by the user. This defaults to Inherit Parent Transform. |
| Set Instanceable when Creating Reference | Toggles setting the prim to instanceable when creating a reference. This defaults to unchecked. |
| Transform Gizmo Manipulates Separately | Toggles the transform gizmo manipulating the prim separately or as a group. This defaults to unchecked. |

### Logging

| Option | Description |
| --- | --- |
| Mute USD Coding Error from USD Diagnostic Manager | Toggles muting USD coding errors from the USD Diagnostic Manager. This defaults to unchecked. |

## Template Startup

### New Stage Template

| Option | Description |
| --- | --- |
| Path to User Templates | The path to the user templates. This defaults to `${app_documents}/scripts/new_stage`. |
| Default Template | The default template to use when creating a new stage. This defaults to sunlight. |

## Rendering

### White Mode

| Option | Description |
| --- | --- |
| Material | Sets the material used in White Mode. This defaults to DebugWhite. |
| Exception | Excludes this list of prims from White Mode. This defaults to `GizmoTex, Gizmo, OmniGlass, SunsetSkyMat`. |

---

# Selection Modes

There are two selection modes in Omniverse: Selection by *type* and selection by *model kind*.
Selection by *type* selects as deep in the tree possible and Selection by *model kind* starts at the clicked mesh and searches up the stage tree until it finds a prim of the currently specified *model kind*.
A prim’s *type* is set when the prim is created and a prim’s *kind* is a property that can be set by the user in the properties window.

## Changing Selection Mode

To toggle between *type* and *model kind* selection, click on the top-most icon in the toolbar immediately to the right of the viewport as shown below:

Click on the selection mode button to toggle between *type* and *model kind* selection.

Note

Press T Hot-Key to quickly toggle between selection modes. Press Q Hot-Key to switch to select from a transform mode.

In addition, right clicking the selection mode button displays the selection rollout, which has specific *type* and *kind* options.
These options will be explained in the respective *type* and *model kind* subsections.

Right click on the selection mode button to display *type* and *model kind* selection mode options.

### Type Selection

The selection toggle button will appear as two grey boxes and one orange box as shown below to indicate *type* selection mode.

Two grey boxes and one orange box will display in the selection mode toggle button if *type* selection is active.

While in this mode, clicking on an item in the viewport will select the lowest corresponding item in the stage tree.
This will typically be a mesh, but could also be a light or a camera.

After right clicking the selection mode toggle button there are four filtering options for this selection mode: *All Prim Types*, *Meshes*, *Lights* and *Cameras*.
*All Prim Types* is selected by default and does not filter the selection. *Meshes*, *Lights* and *Cameras* will filter the selection by the specified *type*.

The prim type is set when a prim is created and cannot be edited; it is inherent to the prim.

Note

When in prim mode, you can select a parent group by selecting the containment outline (bounding box).

### Model Kind Selection

The selection toggle button will appear as a single grey box as shown below to indicate *model kind* selection mode.

A single grey box will display in the selection mode toggle button if *model kind* selection is active.

While in this mode, clicking on an item in the viewport will start with the deepest item in the stage tree (what *type* mode would have selected) and then search up the tree for the first prim of the corresponding *kind*.
The *Kind* is an attribute which can be set in the properties pane for any prim.

After right clicking the selection mode toggle button there are five filtering options for this selection mode: *All Model Kinds*, *Assembly*, *Group*, *Component* and *Subcomponent*.
*All Model Kinds* is selected by default and will simply select the first prim with any *Kind* set.
*Assembly*, *Group*, *Component* and *Subcomponent* will each navigate up the tree until encountering a prim of the specified *kind*.

So, a user can allow for sophisticated hierarchical group selection by purposefully choosing *Model Kind* selection filters and setting the *kind* attribute throughout their stage structure.