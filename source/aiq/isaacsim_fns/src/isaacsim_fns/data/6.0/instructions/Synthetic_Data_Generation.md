# Synthetic Data Generation

Synthetic Data Generation (SDG) is a collection of tools and workflows for generating synthetic data in Isaac Sim.

- Perception Data Generation (Replicator)
- Action and Event Data Generation
- Grasping Synthetic Data Generation
- Data Generation with MobilityGen

---

# Perception Data Generation (Replicator)

Isaac Sim Replicator offers various tools and workflows for synthetic data generation (SDG), primarily using the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension. The examples and tutorials in this section showcase practical applications for robotics, including domain randomization, sensor simulation, and data collection with [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") and [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)").

## Basics and Getting Started

- Overview
- Synthetic Data Recorder
- Getting Started Scripts

## Tutorials

- Scene Based Synthetic Dataset Generation
- Object Based Synthetic Dataset Generation
- Environment Based Synthetic Dataset Generation with Infinigen
- Randomization in Simulation – AMR Navigation
- Randomization in Simulation – UR10 Palletizing
- Cosmos Synthetic Data Generation

## Customization Tools and Techniques

- Data Augmentation
- Custom Replicator Randomization Nodes
- Modular Behavior Scripting
- Randomization Snippets
- Useful Snippets

## Troubleshooting

- Replicator Troubleshooting

---

# Overview

Isaac Sim Replicator offers various tools and workflows for synthetic data generation (SDG), with its core functionalities mostly provided by, but not limited to, the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension. This page provides an overview of these tools and extensions, including semantic labeling, sensor visualization, GUI-based data recording, config file-based SDG workflows, and getting started scripts (examples). To enable SDG relevant UI panels you can use the [Synthetic Data Generation Layout](../gui/layouts.html#isaac-sim-app-gui-layouts).

## The Semantics Schema Editor

The [Semantics Schema Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html "(in Omniverse Extensions)") is a GUI-based extension that enables you to view, add, edit, or remove semantic labels on prims in a stage. Semantically labeling prims is necessary for annotators like semantic segmentation or bounding boxes to include semantic information in the synthetic data. You can access the editor through **Tools > Replicator > Semantics Schema Editor**. To programmatically label prims in a stage, see the following [example snippet](Python_Scripting_and_Tutorials.md).

## The Synthetic Data Visualizer

The [Synthetic Data Visualizer](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/visualization.html "(in Omniverse Extensions)") tool enables sensor output visualization directly in the Viewport window, it can be accessed using the  icon and selecting the desired output formats.

Note

* Cross Correspondence visualization requires a specific two-camera setup explained in the Cross Correspondence section of the [annotator details](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") page.

## The Synthetic Data Recorder

The [Synthetic Data Recorder](Synthetic_Data_Generation.md) is a GUI-based tool that allows you to record synthetic data directly from the editor. It is built on top of `omni.replicator` using `BasicWriter` as its default writer, it is useful for rapid iterations of synthetic data recordings for testing purposes. You can access the recorder via **Tools > Replicator > Synthetic Data Recorder**.

## Replicator YAML

[Replicator YAML](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/yaml_workflow.html "(in Omniverse Extensions)") is a configuration file-based workflow built on top of the Replicator API. It allows you to define randomizations and data capture pipelines as configuration files. These configurations are transformed through the Replicator API into an [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") workflow for synthetic data generation. You can access the YAML workflow using **Tools > Replicator > Replicator YAML**.

## Getting Started Scripts

The [Getting Started Scripts](Synthetic_Data_Generation.md) provides a starting point for typical Isaac Sim Replicator workflows. These tutorials cover basic topics such as accessing data from [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") or [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)"), and using Replicator randomizers together with custom USD/Isaac Sim API randomizers triggered independently from the data capture.

---

# Synthetic Data Recorder

This tutorial introduces the Synthetic Data Recorder for Isaac Sim, which is a GUI extension for recording synthetic data with the possibility of using [custom writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/custom_writer.html "(in Omniverse Extensions)") to record the data in various formats.

The Synthetic Data Recorder requires assets to be [semantically labelled](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html "(in Omniverse Extensions)") for all of the annotators to work correctly. The recorder uses the `BasicWriter` by default with access to most common [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)").

## Getting Started

The UI window can be opened from the main menu using **Tools** > **Replicator** > **Synthetic Data Recorder**.

This tutorial uses the following stage as an example:

```python
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd
```

The stage asset can be found in the **Content Browser** under **Isaac Sim** > **Samples** > **Replicator** > **Stage** > **full\_warehouse\_worker\_and\_anim\_cameras.usd**, or can be loaded using by inserting the whole URL in the path field.

The example stage comes preloaded with semantic annotations and multiple cameras. Some of the included cameras are animated to move around the scene when running the simulation. To create custom camera movement animations, review the [Camera Animation Tutorial](https://docs.omniverse.nvidia.com/extensions/latest/ext_animation-timeline.html "(in Omniverse Extensions)").

## Basic Usage

The recorder is split into two main parts:

* the **Writer** frame - containing sensor, data, and output parameters
* the **Control** frame - containing the recording functionalities such as start, stop, pause, and parameters such as the number of frames to execute

### Writer Frame

The **Writer** frame provides access to **Render Products**, **Parameters**, **Output**, and **Config** options.

The **Render Products** frame allows the creation of a list of render product entries using the **Add New Render Product** button. By default, a new entry is added to the list using the active viewport camera as its camera path (see left figure). If cameras are selected in the stage viewer, these are added to the render products list (see right figure). The render products list can include the same camera path multiple times, with each instance having a different resolution. All entry values, such as camera path or resolution, can be manually edited in the input fields.

The **Parameters** frame offers a choice between the default built-in Replicator writer (`BasicWriter`) and a custom writer. Default writer parameters, primarily annotators, can be selected from the checkbox list. Parameters for custom writers, which are unknown beforehand, must be provided in the form of a JSON file containing all required parameters. The path to the JSON file is entered in the **Parameters Path** input field.

The **Output** frame (left figure) specifies the working directory path where the data is saved, along with the folder name for the current recording. The output folder name is incremented in case of conflicts. The recorder also supports writing to S3 buckets by enabling **Use S3**, entering the required fields, and ensuring AWS credentials are properly configured.

Note

When writing to S3, the **Increment** folder naming feature is not supported and defaults to **Timestamp**.

The **Config** frame (right figure) allows loading and saving the GUI writer state as a JSON configuration file. By default, the extension loads the most recently used configuration state.

### Control Frame

The **Control** frame contains the recording functionalities such as Start/Stop and Pause/Resume, and parameters such as the number of frames to record or the number of subframes to render for each recorded frame.

* The **Start** button creates a writer, given the selected parameters, and starts the recording.
* The **Stop** button stops the recording and clears the writer.
* The **Pause** button pauses the recording without clearing the writer.
* The **Resume** button resumes the recording.
* The **Number of Frames** input field sets the number of frames to record, after which the recorder is stopped and the writer cleared. If the value is set to `0`, the recording runs indefinitely or until the **Stop** button is pressed.
* The **RTSubframes** field sets the number of additional subframes to render for each per frame. This can be used if randomized materials are not loaded in time or if temporal rendering artifacts (such as ghosting) are present due to objects being teleported.
* The **Control Timeline** checkbox starts, stops, pauses, and resumes the timeline together with the recorder.
* The **Verbose** checkbox enables verbose logging for the recorder (events such as start, stop, pause, resume, and the number of frames recorded).

Note

To improve the rendering quality, or to avoid any rendering artifacts caused by low lighting conditions or fast-moving objects, increase the **RTSubframes** parameter. This renders multiple subframes for each frame, thereby improving the quality of recorded data at the expense of longer rendering times per frame. For more details, see the [subframes](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)") documentation.

## Custom Writer Example

To support custom data formats, the custom writer can be registered and loaded from the GUI. In this example, a custom writer called `MyCustomWriter` is registered using the [Script Editor](Development_Tools.md) for use with the recorder.

MyCustomWriter

```python
import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry

class MyCustomWriter(Writer):
    def __init__(
        self,
        output_dir,
        rgb=True,
        normals=False,
    ):
        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.output_dir = output_dir
        self.annotators = []
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        if normals:
            self.annotators.append(AnnotatorRegistry.get_annotator("normals"))
        self._frame_id = 0

    def write(self, data: dict):
        for annotator in data.keys():
            # If there are multiple render products the data will be stored in subfolders
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            # rgb
            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                filename = f"{render_product_path}rgb_{self._frame_id}.png"
                print(f"[{self._frame_id}] Writing {self.output_dir}/{filename} ..")
                self.backend.write_image(filename, data[annotator])

            # normals
            if annotator.startswith("normals"):
                if multi_render_prod:
                    render_product_path += "normals/"
                filename = f"{render_product_path}normals_{self._frame_id}.png"
                print(f"[{self._frame_id}] Writing {self.output_dir}/{filename} ..")
                colored_data = ((data[annotator] * 0.5 + 0.5) * 255).astype(np.uint8)
                self.backend.write_image(filename, colored_data)

        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0

WriterRegistry.register(MyCustomWriter)
```

my\_params.json

```python
1{
2    "rgb": true,
3    "normals": true
4}
```

### Data Visualization Writer

The **Data Visualization** writer is a custom writer that can be used to visualize the annotation data on top of rendered images. The writer and its implementation details can be found in `/isaacsim.replicator.writers/python/scripts/writers/data_visualization_writer.py`, and can be imported using `from isaacsim.replicator.writers import DataVisualizationWriter`. The custom writer can be selected from the **Parameters** frame and its parameters can be loaded from a JSON file using the **Parameters Path** input field. Here is an example JSON file that can be used to parameterize the writer:

my\_data\_visualization\_params.json

```python
 1{
 2    "bounding_box_2d_tight": true,
 3    "bounding_box_2d_tight_params": {
 4        "background": "rgb",
 5        "outline": "green",
 6        "fill": null
 7    },
 8    "bounding_box_2d_loose": true,
 9    "bounding_box_2d_loose_params": {
10        "background": "normals",
11        "outline": "red",
12        "fill": null
13    },
14    "bounding_box_3d": true,
15    "bounding_box_3d_params": {
16        "background": "rgb",
17        "fill": "blue",
18        "width": 2
19    }
20}
```

And the resulting data:

For more information on the supported parameters, see the class docstring:

DataVisualizationWriter class docstring

```python
"""Data Visualization Writer

This writer can be used to visualize various annotator data.

Supported annotators:
- bounding_box_2d_tight
- bounding_box_2d_loose
- bounding_box_3d

Supported backgrounds:
- rgb
- normals

Args:
    output_dir (str):
        Output directory for the data visualization files forwarded to the backend writer.
    bounding_box_2d_tight (bool, optional):
        If True, 2D tight bounding boxes will be drawn on the selected background (transparent by default).
        Defaults to False.
    bounding_box_2d_tight_params (dict, optional):
        Parameters for the 2D tight bounding box annotator. Defaults to None.
    bounding_box_2d_loose (bool, optional):
        If True, 2D loose bounding boxes will be drawn on the selected background (transparent by default).
        Defaults to False.
    bounding_box_2d_loose_params (dict, optional):
        Parameters for the 2D loose bounding box annotator. Defaults to None.
    bounding_box_3d (bool, optional):
        If True, 3D bounding boxes will be drawn on the selected background (transparent by default). Defaults to False.
    bounding_box_3d_params (dict, optional):
        Parameters for the 3D bounding box annotator. Defaults to None.
    frame_padding (int, optional):
        Number of digits used for the frame number in the file name. Defaults to 4.

"""
```

## Replicator Randomized Cameras

To take advantage of Replicator randomization techniques, randomized cameras can be loaded using the [Script Editor](Development_Tools.md) before starting the recorder to run scene randomizations during recording. In this example a randomized camera is created using the Replicator API. This can be attached as a render product to the recorder and for each frame the camera is randomized with the given parameters.

```python
import omni.replicator.core as rep

camera = rep.create.camera()
with rep.trigger.on_frame():
    with camera:
        rep.modify.pose(
            position=rep.distribution.uniform((-5, 5, 1), (-1, 15, 5)),
            look_at="/Root/Warehouse/SM_CardBoxA_3",
        )
```

## Recording Loop Overview

The **Synthetic Data Recorder** is a GUI extension for Isaac Sim that uses the `BasicWriter` or custom Replicator writers for capturing data. Its implementation is located in `/isaacsim.replicator.synthetic_recorder/isaacsim/replicator/synthetic_recorder/synthetic_recorder.py` and utilizes the `orchestrator.step(rt_subframes, pause_timeline, delta_time)` function to manage the recording process. This function ensures that recorded frames remain synchronized with the stage by waiting for any “frames in flight” from the renderer. For integration with the UI, the recorder uses the asynchronous version of this function: `step_async`.

```python
while self._current_frame < num_frames:
    timeline = omni.timeline.get_timeline_interface()

    if self.control_timeline and not timeline.is_playing():
        timeline.play()
        timeline.commit()

    await rep.orchestrator.step_async(rt_subframes=self.rt_subframes, delta_time=None, pause_timeline=False)

    self._current_frame += 1
```

The recording loop offers flexibility for different use cases. It can advance the timeline for dynamic scenes, such as simulations or animations, or operate without advancing the timeline for static captures. This approach enables recording scenarios like randomizing views, adjusting lighting conditions, or repositioning objects.

---

# Getting Started Scripts

This guide outlines a series of example scripts designed to facilitate typical Isaac Sim Replicator workflows. The examples include both “asynchronous” usage through the [Script Editor](Development_Tools.md) and “synchronous” usage through the [Standalone Application](Workflows.md). These scripts cover simulation-based scenarios and configurations for synthetic data generation (SDG).

## Prerequisites

Before starting with these examples, ensure you have:

* Basic understanding of Python programming
* Familiarity with USD (Universal Scene Description) concepts
* Access to NVIDIA Omniverse™ Isaac Sim
* Sufficient disk space for data capture (varies based on resolution and number of frames)
* GPU with sufficient memory for rendering (recommended: 8GB+)

## Setup and Configuration

This section introduces configurations typically used in such workflows.

## Orchestrator Step Function

In Replicator, the `orchestrator.step()` function is used to trigger the entire synthetic data generation (SDG) process, including executing randomizations and capturing data. For Isaac Sim workflows, this function is used solely to trigger data capture only, with randomization triggers assigned to custom events and manually activated.

The `step()` function has the following signature:

```python
rep.orchestrator.step(rt_subframes: int = -1, pause_timeline: bool = True, delta_time: float = None)
```

Where:

* `rt_subframes`: Specifies the number of subframes to render. A value greater than 0 enables subframe generation, reducing rendering artifacts or allowing materials to load fully.
* `pause_timeline`: Pauses the timeline (if currently playing) after the step if set to `True`.
* `delta_time`: Specifies the time to advance the timeline during a step. Defaults to the timeline’s rate if `None`.

More details on graph-based replicator randomizers can be found in the [Randomizer Details](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)"), and for custom Isaac Sim or USD API-based randomizations, refer to the [Isaac Sim Randomizers Guide](Synthetic_Data_Generation.md).

## Capture on Play Flag

By default, Replicator captures data every frame during playback. For Isaac Sim workflows, data capture is configured to occur at user-defined frames using the `step()` function. To achieve this, the capture-on-play flag is disabled:

```python
import omni.replicator.core as rep

rep.orchestrator.set_capture_on_play(False)
# OR
import carb.settings

carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
```

## RT Subframes Parameter

In scenarios where reducing temporal rendering artifacts is needed, such as ghosting caused by quickly moving or teleporting assets, or under weak lighting conditions, RTSubframes can be used to render the same frame multiple times. This pauses the simulation and renders additional subframes, improving rendering quality.

The `rt_subframes` parameter is typically set during the capture request in the `step()` function but can also be configured globally:

```python
# Set the rt_subframes parameter for a specific capture step
rep.orchestrator.step(rt_subframes=4)

# Set the rt_subframes parameter globally
import carb.settings

carb.settings.get_settings().set("/omni/replicator/RTSubframes", 4)
```

Refer to the [documentation examples](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)") for additional details.

## DLSS Quality Mode for SDG

When using Replicator for synthetic data generation (SDG) workflows, it is recommended to set the DLSS model to Quality mode to avoid rendering artifacts. At lower resolutions (especially below 600x600), the default Performance mode may cause issues such as transparent or incorrectly rendered edges in the generated images.

```python
import carb.settings

# Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto))
carb.settings.get_settings().set("/rtx/post/dlss/execMode", 2)
```

## Custom Event Randomizations

To provide flexibility, replicator randomizers can be triggered independently using custom events. This is achieved by registering the randomizer trigger through `trigger.on_custom_event` and activating it with `utils.send_og_event`. For instance, the following example creates a randomization graph for a dome light and randomizes its color. The randomization graph is then triggered manually through its custom event name. The `step()` function does not trigger this randomization graph.

```python
# Create a randomization graph for creating a dome light and randomizing its color
with rep.trigger.on_custom_event(event_name="randomize_dome_light_color"):
    rep.create.light(light_type="Dome", color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

# Trigger the randomization graph using its custom event name
rep.utils.send_og_event(event_name="randomize_dome_light_color")
```

An example snippet for custom events is also available [here](Synthetic_Data_Generation.md).

## Wait Until Complete

Ensuring that all data is fully written to disk before closing the application is essential to prevent data loss. High data throughput, such as from multiple cameras or large resolutions, may introduce I/O bottlenecks; refer to the [I/O Optimization Guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/io_guidelines.html "(in Omniverse Extensions)") for strategies to mitigate such issues.

The `wait_until_complete` function ensures that all writing tasks are finalized by waiting for the writer backend to complete its operations. This process allows the application to continue updating until all writing tasks are complete, safeguarding against potential data loss.

```python
from omni.replicator.core import BackendDispatch
import omni.kit.app

async def wait_until_complete():
    while not BackendDispatch.is_done_writing():
        await omni.kit.app.get_app().next_update_async()
```

Alternatively, use the documented helper functions: `rep.orchestrator.wait_until_complete()` for synchronous contexts or `await rep.orchestrator.wait_until_complete_async()` for asynchronous contexts.

## Examples

### Data Capture: BasicWriter

This example demonstrates how to use the `BasicWriter` for data capture with RGB and bounding box annotators. It sets up a scene with a cube and a dome light, attaches semantic labels to the cube, and saves captured data to disk. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_01.py
```

Script Editor

```python
import asyncio
import os

import carb.settings
import omni.replicator.core as rep
import omni.usd

async def run_example_async():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup the stage with a dome light and a cube
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Create a render product using the viewport perspective camera
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, (512, 512), name="MyRenderProduct")

    # Write data using the basic writer with the rgb and bounding box annotators
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer")
    backend.initialize(output_dir=out_dir)
    print(f"Output directory: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, bounding_box_2d_tight=True)
    writer.attach(rp)

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        await rep.orchestrator.step_async()

    # Wait for the data to be written to disk and clean up resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

# Run the example
asyncio.ensure_future(run_example_async())
```

Standalone Application

```python
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import carb.settings
import omni.replicator.core as rep
import omni.usd

def run_example():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup the stage with a dome light and a cube
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Create a render product using the viewport perspective camera
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, (512, 512), name="MyRenderProduct")

    # Write data using the basic writer with the rgb and bounding box annotators
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer")
    backend.initialize(output_dir=out_dir)
    print(f"Output directory: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, bounding_box_2d_tight=True)
    writer.attach(rp)

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        rep.orchestrator.step()

    # Wait for the data to be written to disk and clean up resources
    rep.orchestrator.wait_until_complete()
    writer.detach()
    rp.destroy()

# Run the example
run_example()

simulation_app.close()
```

The output directory will contain the captured data, including RGB images and bounding box annotations in `.npy` and `.json` formats:

### Custom Writer and Annotators with Multiple Cameras

This example demonstrates data capture by creating a custom writer to access annotator data such as camera parameters and 3D bounding boxes. It configures two cameras (custom and viewport perspective), uses annotators to access data directly, writes data to disk using `PoseWriter`. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_02.py
```

Script Editor

```python
import asyncio
import os

import carb.settings
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core import Writer

# Create a custom writer to access annotator data
class MyWriter(Writer):
    def __init__(self, camera_params: bool = True, bounding_box_3d: bool = True):
        # Organize data from render product perspective (legacy, annotator, renderProduct)
        self.data_structure = "renderProduct"
        self.annotators = []
        if camera_params:
            self.annotators.append(rep.annotators.get("camera_params"))
        if bounding_box_3d:
            self.annotators.append(rep.annotators.get("bounding_box_3d"))
        self._frame_id = 0

    def write(self, data: dict):
        print(f"[MyWriter][{self._frame_id}] data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        self._frame_id += 1

# Register the writer
rep.writers.register_writer(MyWriter)

async def run_example_async():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup stage
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Capture from two perspectives, a custom camera and a perspective camera
    top_cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0), parent="/World", name="TopCamera")
    persp_cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="PerspCamera")

    # Create the render products
    rp_top = rep.create.render_product(top_cam.GetPath(), (400, 400), name="top_view")
    rp_persp = rep.create.render_product(persp_cam.GetPath(), (512, 512), name="persp_view")

    # Use the annotators to access the data directly, each annotator is attached to a render product
    rgb_annotator_top = rep.annotators.get("rgb")
    rgb_annotator_top.attach(rp_top)
    rgb_annotator_persp = rep.annotators.get("rgb")
    rgb_annotator_persp.attach(rp_persp)

    # Use the custom writer to access the annotator data
    custom_writer = rep.writers.get("MyWriter")
    custom_writer.initialize(camera_params=True, bounding_box_3d=True)
    custom_writer.attach([rp_top, rp_persp])

    # Use the pose writer to write the data to disk
    pose_writer = rep.WriterRegistry.get("PoseWriter")
    out_dir = os.path.join(os.getcwd(), "_out_pose_writer")
    print(f"Output directory: {out_dir}")
    pose_writer.initialize(output_dir=out_dir, write_debug_images=True)
    pose_writer.attach([rp_top, rp_persp])

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        await rep.orchestrator.step_async()

        # Get the data from the annotators
        rgb_data_cam = rgb_annotator_top.get_data()
        rgb_data_persp = rgb_annotator_persp.get_data()
        print(f"[Annotator][Cam][{i}] rgb_data_cam shape: {rgb_data_cam.shape}")
        print(f"[Annotator][Persp][{i}] rgb_data_persp shape: {rgb_data_persp.shape}")

    # Wait for the data to be written to disk and clean up resources
    await rep.orchestrator.wait_until_complete_async()
    pose_writer.detach()
    custom_writer.detach()
    rgb_annotator_top.detach()
    rgb_annotator_persp.detach()
    rp_top.destroy()
    rp_persp.destroy()

# Run the example
asyncio.ensure_future(run_example_async())
```

Standalone Application

```python
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import carb.settings
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core import Writer

# Create a custom writer to access annotator data
class MyWriter(Writer):
    def __init__(self, camera_params: bool = True, bounding_box_3d: bool = True):
        # Organize data from render product perspective (legacy, annotator, renderProduct)
        self.data_structure = "renderProduct"
        self.annotators = []
        if camera_params:
            self.annotators.append(rep.annotators.get("camera_params"))
        if bounding_box_3d:
            self.annotators.append(rep.annotators.get("bounding_box_3d"))
        self._frame_id = 0

    def write(self, data: dict):
        print(f"[MyWriter][{self._frame_id}] data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        self._frame_id += 1

# Register the writer
rep.writers.register_writer(MyWriter)

def run_example():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup stage
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Capture from two perspectives, a custom camera and a perspective camera
    top_cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0), parent="/World", name="TopCamera")
    persp_cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="PerspCamera")

    # Create the render products
    rp_top = rep.create.render_product(top_cam.GetPath(), (400, 400), name="top_view")
    rp_persp = rep.create.render_product(persp_cam.GetPath(), (512, 512), name="persp_view")

    # Use the annotators to access the data directly, each annotator is attached to a render product
    rgb_annotator_top = rep.annotators.get("rgb")
    rgb_annotator_top.attach(rp_top)
    rgb_annotator_persp = rep.annotators.get("rgb")
    rgb_annotator_persp.attach(rp_persp)

    # Use the custom writer to access the annotator data
    custom_writer = rep.writers.get("MyWriter")
    custom_writer.initialize(camera_params=True, bounding_box_3d=True)
    custom_writer.attach([rp_top, rp_persp])

    # Use the pose writer to write the data to disk
    pose_writer = rep.WriterRegistry.get("PoseWriter")
    out_dir = os.path.join(os.getcwd(), "_out_pose_writer")
    print(f"Output directory: {out_dir}")
    pose_writer.initialize(output_dir=out_dir, write_debug_images=True)
    pose_writer.attach([rp_top, rp_persp])

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        rep.orchestrator.step()

        # Get the data from the annotators
        rgb_data_top = rgb_annotator_top.get_data()
        rgb_data_persp = rgb_annotator_persp.get_data()
        print(f"[Annotator][Top][{i}] rgb_data_top shape: {rgb_data_top.shape}")
        print(f"[Annotator][Persp][{i}] rgb_data_persp shape: {rgb_data_persp.shape}")

    # Wait for the data to be written to disk and clean up resources
    rep.orchestrator.wait_until_complete()
    pose_writer.detach()
    custom_writer.detach()
    rgb_annotator_top.detach()
    rgb_annotator_persp.detach()
    rp_top.destroy()
    rp_persp.destroy()

run_example()

simulation_app.close()
```

The output directory will contain the captured data, including RGB with the 3D bounding box annotations as overlays together with `.json` files with the frame data. The annotator and custom writer data is printed to the terminal.

### Custom Randomizations: Replicator Graph and USD API

This example demonstrates creating a custom randomization using Replicator’s graph-based randomizers triggered by custom events and a custom USD API-based randomization. A dome light’s color is randomized through custom events, while a cube’s location is randomized through USD API. Data is captured using the `BasicWriter` with semantic segmentation. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_03.py
```

Script Editor

```python
import asyncio
import os
import random

import carb.settings
import omni.replicator.core as rep
import omni.usd

# Randomize the location of a prim without the graph-based randomizer
def randomize_location(prim):
    random_pos = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
    rep.functional.modify.position(prim, random_pos)

async def run_example_async():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)
    random.seed(42)
    rep.set_global_seed(42)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup stage
    rep.functional.create.xform(name="World")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Create a replicator randomizer with custom event trigger
    with rep.trigger.on_custom_event(event_name="randomize_dome_light_color"):
        rep.create.light(light_type="Dome", color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    # Create a render product using the viewport perspective camera
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, (512, 512))

    # Write data using the basic writer with the rgb and bounding box annotators
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer_rand")
    backend.initialize(output_dir=out_dir)
    print(f"Output directory: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
    writer.attach(rp)

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        # Trigger the custom graph-based event randomizer every second step
        if i % 2 == 1:
            rep.utils.send_og_event(event_name="randomize_dome_light_color")

        # Run the custom USD API location randomizer on the prims
        randomize_location(cube)

        # Since the replicator randomizer is set to trigger on custom events, step will only trigger the writer
        await rep.orchestrator.step_async(rt_subframes=32)

    # Wait for the data to be written to disk and clean up resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

# Run the example
asyncio.ensure_future(run_example_async())
```

Standalone Application

```python
import os
import random

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import carb.settings
import omni.replicator.core as rep
import omni.usd

# Randomize the location of a prim without the graph-based randomizer
def randomize_location(prim):
    random_pos = (random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1))
    rep.functional.modify.position(prim, random_pos)

def run_example():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)
    random.seed(42)
    rep.set_global_seed(42)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Setup stage
    rep.functional.create.xform(name="World")
    cube = rep.functional.create.cube(parent="/World", name="Cube")
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")

    # Create a replicator randomizer with custom event trigger
    with rep.trigger.on_custom_event(event_name="randomize_dome_light_color"):
        rep.create.light(light_type="Dome", color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    # Create a render product using the viewport perspective camera
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, (512, 512))

    # Write data using the basic writer with the rgb and bounding box annotators
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer_rand")
    backend.initialize(output_dir=out_dir)
    print(f"Output directory: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
    writer.attach(rp)

    # Trigger a data capture request (data will be written to disk by the writer)
    for i in range(3):
        print(f"Step {i}")
        # Trigger the custom graph-based event randomizer every second step
        if i % 2 == 1:
            rep.utils.send_og_event(event_name="randomize_dome_light_color")

        # Run the custom USD API location randomizer on the prims
        randomize_location(cube)

        # Since the replicator randomizer is set to trigger on custom events, step will only trigger the writer
        rep.orchestrator.step(rt_subframes=32)

    # Destroy the render product to release resources by detaching it from the writer first
    writer.detach()
    rp.destroy()

    # Wait for the data to be written to disk
    rep.orchestrator.wait_until_complete()

# Run the example
run_example()

simulation_app.close()
```

The output directory will contain the RGB and semantic segmentation images with the captured data. The cube is randomized each capture, while the dome light color is randomized every second capture.

### Event-Triggered Data Capture: Timeline and Simulation

This example shows how to capture simulation data when specific conditions are met. A cube and sphere are dropped in a physics simulation, and data is captured at specific intervals based on the cube’s height. The timeline is paused during capture to ensure data consistency. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_04.py
```

Script Editor

```python
import asyncio
import os

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from pxr import UsdGeom

async def run_example_async():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Add a light
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")

    # Create a cube with colliders and rigid body dynamics at a specific location
    cube = rep.functional.create.cube(name="Cube", parent="/World")
    rep.functional.modify.position(cube, (0, 0, 2))
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")
    rep.functional.physics.apply_rigid_body(cube, with_collider=True)

    # Createa a sphere with colliders and rigid body dynamics next to the cube
    sphere = rep.functional.create.sphere(name="Sphere", parent="/World")
    rep.functional.modify.position(sphere, (-1, -1, 2))
    rep.functional.modify.semantics(sphere, {"class": "my_sphere"}, mode="add")
    rep.functional.physics.apply_rigid_body(sphere, with_collider=True)

    # Create a render product using the viewport perspective camera
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, (512, 512))

    # Write data using the basic writer with the rgb and bounding box annotators
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer_sim")
    backend.initialize(output_dir=out_dir)
    print(f"Output directory: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
    writer.attach(rp)

    # Start the timeline (will only advance with app update)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Wrap the cube with as a RigidPrim for easy access to its world poses and velocities
    cube_rigid = RigidPrim(str(cube.GetPrimPath()))

    # Wrap the cube as an Imageable object to toggle visibility during capture
    cube_imageable = UsdGeom.Imageable(cube)

    # Define the capture interval in meters
    capture_interval_meters = 0.5
    cube_pos = cube_rigid.get_world_poses(indices=[0])[0].numpy()
    previous_capture_height = cube_pos[0, 2]

    # Update the app which will advance the timeline (and implicitly the simulation)
    for i in range(100):
        await omni.kit.app.get_app().next_update_async()
        cube_pos = cube_rigid.get_world_poses(indices=[0])[0].numpy()
        current_height = cube_pos[0, 2]
        distance_dropped = previous_capture_height - current_height
        print(f"Step {i}; cube height: {current_height:.3f}; drop since last capture: {distance_dropped:.3f}")

        # Stop the simulation if the cube falls below the ground
        if current_height < 0:
            print(f"\t Cube fell below the ground at height {current_height:.3f}, stopping simulation..")
            break

        # Capture every time the cube drops by the threshold distance
        if distance_dropped >= capture_interval_meters:
            print(f"\t Capturing at height {current_height:.3f}")
            previous_capture_height = current_height

            # Setting delta_time to 0.0 will make sure the timeline is not advanced during capture
            await rep.orchestrator.step_async(delta_time=0.0)

            # Capture again with the cube hidden
            print("\t Capturing with cube hidden")
            cube_imageable.MakeInvisible()
            await rep.orchestrator.step_async(delta_time=0.0)
            cube_imageable.MakeVisible()

            # Resume the timeline to continue the simulation
            timeline.play()

    # Pause the simulation
    timeline.pause()

    # Wait for the data to be written to disk and clean up resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

# Run the example
asyncio.ensure_future(run_example_async())
```

Standalone Application

```python
import os

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import RigidPrim
from pxr import UsdGeom

def run_example():
    # Create a new stage and disable capture on play
    omni.usd.get_context().new_stage()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Add a light
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")

    # Create a cube with colliders and rigid body dynamics at a specific location
    cube = rep.functional.create.cube(name="Cube", parent="/World")
    rep.functional.modify.position(cube, (0, 0, 2))
    rep.functional.modify.semantics(cube, {"class": "my_cube"}, mode="add")
    rep.functional.physics.apply_rigid_body(cube, with_collider=True)

    # Createa a sphere with colliders and rigid body dynamics next to the cube
    sphere = rep.functional.create.sphere(name="Sphere", parent="/World")
    rep.functional.modify.position(sphere, (-1, -1, 2))
    rep.functional.modify.semantics(sphere, {"class": "my_sphere"}, mode="add")
    rep.functional.physics.apply_rigid_body(sphere, with_collider=True)

    # Create a render product using the viewport perspective camera
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, (512, 512))

    # Write data using the basic writer with the rgb and bounding box annotators
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), "_out_basic_writer_sim")
    backend.initialize(output_dir=out_dir)
    print(f"Output directory: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
    writer.attach(rp)

    # Start the timeline (will only advance with app update)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Wrap the cube with as a RigidPrim for easy access to its world poses and velocities
    cube_rigid = RigidPrim(str(cube.GetPrimPath()))

    # Wrap the cube as an Imageable object to toggle visibility during capture
    cube_imageable = UsdGeom.Imageable(cube)

    # Define the capture interval in meters
    capture_interval_meters = 0.5
    cube_pos = cube_rigid.get_world_poses(indices=[0])[0].numpy()
    previous_capture_height = cube_pos[0, 2]

    # Update the app which will advance the timeline (and implicitly the simulation)
    for i in range(100):
        simulation_app.update()
        cube_pos = cube_rigid.get_world_poses(indices=[0])[0].numpy()
        current_height = cube_pos[0, 2]
        distance_dropped = previous_capture_height - current_height
        print(f"Step {i}; cube height: {current_height:.3f}; drop since last capture: {distance_dropped:.3f}")

        # Stop the simulation if the cube falls below the ground
        if current_height < 0:
            print(f"\t Cube fell below the ground at height {current_height:.3f}, stopping simulation..")
            break

        # Capture every time the cube drops by the threshold distance
        if distance_dropped >= capture_interval_meters:
            print(f"\t Capturing at height {current_height:.3f}")
            previous_capture_height = current_height

            # Setting delta_time to 0.0 will make sure the timeline is not advanced during capture
            rep.orchestrator.step(delta_time=0.0)

            # Capture again with the cube hidden
            print("\t Capturing with cube hidden")
            cube_imageable.MakeInvisible()
            rep.orchestrator.step(delta_time=0.0)
            cube_imageable.MakeVisible()

            # Resume the timeline to continue the simulation
            timeline.play()

    # Pause the simulation
    timeline.pause()

    # Wait for the data to be written to disk and clean up resources
    rep.orchestrator.wait_until_complete()
    writer.detach()
    rp.destroy()

# Run the example
run_example()

simulation_app.close()
```

The output directory will contain the RGB and semantic segmentation images with the captured data at specific simulation times (cube drop height intervals) and the cube hidden during capture. During every second capture with the cube hidden, the timeline will not advance (`delta_time=0.0`) ensuring the same simulation state can be captured multiple times.

## Troubleshooting

For troubleshooting information related to the Getting Started Scripts, refer to the [Getting Started Scripts Issues](troubleshooting.html#isaac-sim-replicator-troubleshooting-getting-started) section in the Replicator Troubleshooting page.

## Next Steps

After completing these examples, consider exploring:

1. Advanced randomizations using the [Randomizer Details](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)")
2. Custom annotators for specialized data capture
3. Distributed data generation using multiple GPUs
4. Integration with machine learning pipelines
5. Advanced physics-based simulations

For more information, refer to:
- [Replicator Documentation](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/basic_functionalities.html "(in Omniverse Extensions)")
- [Isaac Sim Randomizers Guide](Synthetic_Data_Generation.md)
- [I/O Optimization Guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/io_guidelines.html "(in Omniverse Extensions)")

---

# Scene Based Synthetic Dataset Generation

This tutorial illustrates the process of generating synthetic datasets using the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension. The resulting data is stored offline (on disk), making it readily available for training deep neural networks. The examples can be executed within the Isaac Sim Python [standalone](Workflows.md) environment. The example uses Isaac Sim and Replicator to create synthetic datasets offline (on disk) for the training of machine learning models.

In this tutorial you:

* Utilize and set up external customizable config files (YAML/JSON) to adjust simulation and scenario parameters
* Load custom environments
* Spawn assets using the Isaac Sim API
* Run randomized physics simulations
* Register various Replicator randomization [graphs](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)")
* Create cameras and render products with the Replicator API
* Use Replicator writers to save data to disk

## Prerequisites

* Familiarity with the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension, including its [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") and [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)").
* Basic understanding of Isaac Sim’s [Stage](Glossary.md) and [World](Glossary.md) concepts, further explained in the [Hello World](Python_Scripting_and_Tutorials.md) tutorial.
* Running simulations as [Standalone Applications](Workflows.md) or via the [Script Editor](Development_Tools.md).
* Familiarity with Replicator [randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)") and [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") for a better understanding of the randomization pipeline.

## Scenario

By default, the scenario is executed in a warehouse environment. Within this setting, a forklift is randomly placed in a designated area. Based on the forklift’s position, a pallet is placed in front of it at a randomized distance. Using Replicator’s `scatter_2d` randomization function with the collision check argument `check_for_collisions` set to `True`, the pallet is scattered with boxes, ensuring the boxes do not self-collide. The scatter graph node randomly scatters the boxes in each capture frame. Additionally, a traffic cone is randomly positioned at one of the bottom corners of the forklift’s oriented bounding box (OBB). Before the synthetic data generation (SDG) pipeline starts, a short physics simulation is executed, during which several boxes are dropped onto a pallet situated behind the forklift.

Three camera views are used for the synthetic data generation (SDG). The first (`top_view_cam`) offers a top-down view of the scenario (left), the second (`pallet_cam`) captures a randomized view of the boxes scattered on the pallet (center), and the third is overlooking the pallet from the driver’s place in the forklift using various heights (right).
The data is collected using Replicator writers with configurable backends. The default setup uses `BasicWriter` with a `DiskBackend`. The writer’s config parameters are loaded from the `writer_config` entry and used to initialize the writer with annotators including rgb, semantic\_segmentation, and bounding\_box\_3d. The output directory is specified in `backend_params`, which by default is `<working_dir>/_out_scene_based_sdg`.

## Getting Started

The main script of the tutorial is located at `<install_path>/standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py` and it is set to run as a standalone application. The default configurations are stored in the script itself in the form of a Python dictionary, there is no need to provide a config file.

To overwrite the default configuration parameters, you can provided custom config files as a command-line argument for the script by using `--config <path/to/file.json/yaml>`. Example config files are stored in `scene_based_sdg/config/*`. In the provided examples, the configuration files serve as templates to illustrate and showcase the configurability of the script.

Helper functions are located in the `scene_based_sdg_utils.py` file.

To generate a synthetic dataset, run the following command for the Standalone Application (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py
```

## Implementation

The following section provides an implementation overview of the script. It includes details regarding the configuration parameters, scene generation helper functions, randomizations (Isaac Sim and Replicator), and data capture loop. As standalone example the script is split into two files: the main script and a utilities module.

Script Editor

Utils module and main script

```python
import asyncio
import math
import os

import carb.settings
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.experimental.utils.semantics import remove_all_labels
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils import prims
from isaacsim.core.utils.bounds import (
    compute_combined_aabb,
    compute_obb,
    create_bbox_cache,
    get_obb_corners,
)
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, Usd, UsdGeom

def setup_writer(config: dict) -> rep.Writer | None:
    """Setup and initialize writer with optional backend support."""

    def normalize_output_dir(params):
        if "output_dir" in params and not os.path.isabs(params["output_dir"]):
            params["output_dir"] = os.path.join(os.getcwd(), params["output_dir"])

    writer_type = config.get("writer", "BasicWriter")
    if writer_type not in rep.WriterRegistry.get_writers():
        print(f"[SDG] Writer type '{writer_type}' not found in registry.")
        return None

    writer = rep.WriterRegistry.get(writer_type)
    writer_kwargs = dict(config.get("writer_config", {}))
    normalize_output_dir(writer_kwargs)

    backend_type = config.get("backend_type")
    backend = None
    if backend_type:
        try:
            backend = rep.backends.get(backend_type)
        except Exception as e:
            print(f"[SDG] Backend '{backend_type}' not found: {e}")
            return None

        backend_params = dict(config.get("backend_params", {}))
        normalize_output_dir(backend_params)

        try:
            print(f"[SDG] Backend: {backend_type} | Params: {backend_params}")
            backend.initialize(**backend_params)
        except TypeError as e:
            print(f"[SDG] Invalid backend params: {e}")
            return None

    if "output_dir" in writer_kwargs:
        print(f"[SDG] Output: {writer_kwargs['output_dir']}")

    backend_info = f" + {backend_type}" if backend else ""
    print(f"[SDG] Writer: {writer_type}{backend_info} | Config: {writer_kwargs}")

    try:
        if backend:
            writer.initialize(backend=backend, **writer_kwargs)
        else:
            writer.initialize(**writer_kwargs)
    except TypeError as e:
        print(f"[SDG] Invalid writer params: {e}")
        return None

    return writer

async def simulate_falling_objects_async(
    forklift_prim: Usd.Prim,
    assets_root_path: str,
    config: dict,
    max_sim_steps: int = 250,
    num_boxes: int = 8,
    rng: np.random.Generator | None = None,
) -> None:
    """Run physics simulation to drop boxes on pallet near forklift."""
    if rng is None:
        rng = np.random.default_rng()

    forklift_transform = omni.usd.get_world_transform_matrix(forklift_prim)
    sim_pallet_offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(rng.uniform(-1, 1), rng.uniform(-4, -3.6), 0))
    sim_pallet_position = (sim_pallet_offset * forklift_transform).ExtractTranslation()
    sim_pallet_rotation = euler_angles_to_quat([0, 0, rng.uniform(0, math.pi)])

    sim_pallet = prims.create_prim(
        prim_path="/World/SimulatedPallet",
        position=sim_pallet_position,
        orientation=sim_pallet_rotation,
        usd_path=assets_root_path + config["pallet"]["url"],
        semantic_label=config["pallet"]["class"],
    )
    sim_pallet_geom = GeomPrim(f"{str(sim_pallet.GetPrimPath())}/.*", apply_collision_apis=True)
    sim_pallet_geom.set_collision_approximations("boundingCube")

    bbox_cache = create_bbox_cache()
    current_height = bbox_cache.ComputeLocalBound(sim_pallet).GetRange().GetSize()[2] * 1.1

    sim_box_rigid_prims = []
    for box_index in range(num_boxes):
        box_xy_offset = Gf.Vec3d(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), current_height)
        sim_box = prims.create_prim(
            prim_path=f"/World/SimulatedCardbox_{box_index}",
            position=sim_pallet_position + box_xy_offset,
            orientation=sim_pallet_rotation,
            usd_path=assets_root_path + config["cardbox"]["url"],
            semantic_label=config["cardbox"]["class"],
        )
        current_height += bbox_cache.ComputeLocalBound(sim_box).GetRange().GetSize()[2] * 1.1

        sim_box_geom = GeomPrim(f"{str(sim_box.GetPrimPath())}/.*", apply_collision_apis=True)
        sim_box_geom.set_collision_approximations("convexHull")
        sim_box_rigid_prims.append(RigidPrim(str(sim_box.GetPrimPath())))

    SimulationManager.set_physics_dt(1.0 / 90.0)
    SimulationManager.initialize_physics()

    velocity_threshold = 0.01
    for step in range(max_sim_steps):
        SimulationManager.step()
        if sim_box_rigid_prims:
            top_box_velocity = sim_box_rigid_prims[-1].get_velocities(indices=[0])[0].numpy()
            if np.linalg.norm(top_box_velocity) < velocity_threshold:
                print(f"[SDG] Simulation settled at step {step}")
                break
        await omni.kit.app.get_app().next_update_async()

def setup_camera_bounds(
    pallet_prim: Usd.Prim, forklift_prim: Usd.Prim, pallet_tf: Gf.Matrix4d, forklift_tf: Gf.Matrix4d
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Calculate camera randomization bounds for pallet, top view, and driver cameras."""
    pallet_pos = pallet_tf.ExtractTranslation()
    pallet_cam_bounds = {
        "min": (pallet_pos[0] - 2, pallet_pos[1] - 2, 2),
        "max": (pallet_pos[0] + 2, pallet_pos[1] + 2, 4),
    }

    forklift_pos = forklift_tf.ExtractTranslation()
    top_cam_bounds = {
        "min": (forklift_pos[0], forklift_pos[1], 9),
        "max": (forklift_pos[0], forklift_pos[1], 11),
    }

    driver_cam_pos = forklift_pos + Gf.Vec3d(0.0, 0.0, 1.9)
    driver_cam_bounds = {
        "min": (driver_cam_pos[0], driver_cam_pos[1], driver_cam_pos[2] - 0.25),
        "max": (driver_cam_pos[0], driver_cam_pos[1], driver_cam_pos[2] + 0.25),
    }

    return {
        "pallet_cam": pallet_cam_bounds,
        "top_cam": top_cam_bounds,
        "driver_cam": driver_cam_bounds,
    }

def create_scatter_plane_for_prim(
    prim: Usd.Prim, prim_tf: Gf.Matrix4d, parent_path: str, scale_factor: float = 0.8, visible: bool = False
) -> Usd.Prim:
    """Create scatter plane sized and aligned to prim surface."""
    bb_cache = create_bbox_cache()
    prim_bbox = bb_cache.ComputeLocalBound(prim)
    prim_bbox.Transform(prim_tf)
    prim_size = prim_bbox.GetRange().GetSize()

    prim_quat = prim_tf.ExtractRotation().GetQuaternion()
    prim_quat_xyzw = (prim_quat.GetReal(), *prim_quat.GetImaginary())
    prim_rotation_deg = quat_to_euler_angles(np.array(prim_quat_xyzw), degrees=True)

    prim_pos = prim_tf.ExtractTranslation()
    scatter_plane_scale = (prim_size[0] * scale_factor, prim_size[1] * scale_factor, 1)
    scatter_plane_pos = prim_pos + Gf.Vec3d(0, 0, prim_size[2])

    scatter_plane = rep.functional.create.plane(
        scale=scatter_plane_scale,
        position=tuple(scatter_plane_pos),
        rotation=tuple(prim_rotation_deg),
        visible=visible,
        parent=parent_path,
    )

    return scatter_plane

def setup_cone_placement_corners(
    forklift_prim: Usd.Prim, bb_cache=None, scale_factor: float = 1.3
) -> tuple[list[list[float]], tuple[float, float, float]]:
    """Calculate forklift OBB corners for cone placement."""
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    forklift_obb_center, forklift_obb_axes, forklift_obb_extent = compute_obb(bb_cache, forklift_prim.GetPrimPath())
    enlarged_extent = (
        forklift_obb_extent[0] * scale_factor,
        forklift_obb_extent[1] * scale_factor,
        forklift_obb_extent[2],
    )
    forklift_obb_corners = get_obb_corners(forklift_obb_center, forklift_obb_axes, enlarged_extent)

    cone_placement_corners = [
        forklift_obb_corners[0].tolist(),
        forklift_obb_corners[2].tolist(),
        forklift_obb_corners[4].tolist(),
        forklift_obb_corners[6].tolist(),
    ]

    forklift_obb_quat = Gf.Matrix3d(forklift_obb_axes).ExtractRotation().GetQuaternion()
    forklift_obb_quat_xyzw = (forklift_obb_quat.GetReal(), *forklift_obb_quat.GetImaginary())
    forklift_rotation_deg = quat_to_euler_angles(np.array(forklift_obb_quat_xyzw), degrees=True)

    return cone_placement_corners, forklift_rotation_deg

def register_lights_graph_randomizer(forklift_prim: Usd.Prim, pallet_prim: Usd.Prim, event_name: str) -> None:
    """Register graph randomizer for sphere lights."""
    bb_cache = create_bbox_cache()
    combined_bounds = compute_combined_aabb(bb_cache, [forklift_prim.GetPrimPath(), pallet_prim.GetPrimPath()])
    light_pos_min = (combined_bounds[0], combined_bounds[1], 6)
    light_pos_max = (combined_bounds[3], combined_bounds[4], 7)

    with rep.trigger.on_custom_event(event_name):
        rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0.2, 0.1, 0.1), (0.9, 0.8, 0.8)),
            intensity=rep.distribution.uniform(2000, 4000),
            position=rep.distribution.uniform(light_pos_min, light_pos_max),
            scale=rep.distribution.uniform(1, 4),
            count=3,
        )

def register_cardboxes_materials_graph_randomizer(
    cardboxes: list[Usd.Prim], cardbox_material_urls: list[str], event_name: str
) -> None:
    """Register graph randomizer for cardbox materials."""
    cardbox_mesh_paths = []
    for cardbox in cardboxes:
        meshes = [child for child in cardbox.GetChildren() if child.IsA(UsdGeom.Mesh)]
        cardbox_mesh_paths.extend([mesh.GetPrimPath() for mesh in meshes])

    with rep.trigger.on_custom_event(event_name):
        cardbox_mesh_group_node = rep.create.group(cardbox_mesh_paths)
        with cardbox_mesh_group_node:
            rep.randomizer.materials(cardbox_material_urls)

async def run_example_async(config):
    assets_root_path = await get_assets_root_path_async()
    if assets_root_path is None:
        print("[SDG] Could not get nucleus server path")
        return

    # Load environment stage
    env_url = config.get("env_url", "/Isaac/Environments/Grid/default_environment.usd")
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    print(f"[SDG] Loading Stage {env_url}")
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()

    await omni.kit.app.get_app().next_update_async()

    # Initialize randomization
    rep.set_global_seed(42)
    rng = np.random.default_rng(42)

    # Configure replicator for manual triggering
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode for best SDG results
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Clear previous semantic labels
    if config.get("clear_previous_semantics", True):
        for prim in stage.Traverse():
            remove_all_labels(prim, include_descendants=True)

    # Create SDG scope for organizing all generated objects
    sdg_scope = stage.DefinePrim("/SDG", "Scope")

    # Spawn forklift at random pose
    forklift_prim = prims.create_prim(
        prim_path="/SDG/Forklift",
        position=(rng.uniform(-20, -2), rng.uniform(-1, 3), 0),
        orientation=euler_angles_to_quat([0, 0, rng.uniform(0, math.pi)]),
        usd_path=assets_root_path + config["forklift"]["url"],
        semantic_label=config["forklift"]["class"],
    )

    # Spawn pallet in front of forklift with random offset
    forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
    pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, rng.uniform(-1.8, -1.2), 0))
    pallet_pos = (pallet_offset_tf * forklift_tf).ExtractTranslation()
    forklift_quat = forklift_tf.ExtractRotationQuat()
    forklift_quat_xyzw = (forklift_quat.GetReal(), *forklift_quat.GetImaginary())

    pallet_prim = prims.create_prim(
        prim_path="/SDG/Pallet",
        position=pallet_pos,
        orientation=forklift_quat_xyzw,
        usd_path=assets_root_path + config["pallet"]["url"],
        semantic_label=config["pallet"]["class"],
    )

    # Create cardboxes for pallet scattering
    cardboxes = []
    for i in range(5):
        cardbox = prims.create_prim(
            prim_path=f"/SDG/CardBox_{i}",
            usd_path=assets_root_path + config["cardbox"]["url"],
            semantic_label=config["cardbox"]["class"],
        )
        cardboxes.append(cardbox)

    # Create traffic cone for corner placement
    cone = prims.create_prim(
        prim_path="/SDG/Cone",
        usd_path=assets_root_path + config["cone"]["url"],
        semantic_label=config["cone"]["class"],
    )

    # Create cameras
    rep.functional.create.scope(name="Cameras", parent="/SDG")
    driver_cam = rep.functional.create.camera(
        focus_distance=400.0,
        focal_length=24.0,
        clipping_range=(0.1, 10000000.0),
        name="DriverCam",
        parent="/SDG/Cameras",
    )
    pallet_cam = rep.functional.create.camera(name="PalletCam", parent="/SDG/Cameras")
    top_view_cam = rep.functional.create.camera(clipping_range=(6.0, 1000000.0), name="TopCam", parent="/SDG/Cameras")

    await omni.kit.app.get_app().next_update_async()

    # Setup render products
    resolution = config.get("resolution", (512, 512))
    forklift_rp = rep.create.render_product(top_view_cam, resolution, name="TopView")
    driver_rp = rep.create.render_product(driver_cam, resolution, name="DriverView")
    pallet_rp = rep.create.render_product(pallet_cam, resolution, name="PalletView")

    render_products = [forklift_rp, driver_rp, pallet_rp]
    for render_product in render_products:
        render_product.hydra_texture.set_updates_enabled(False)

    # Initialize writer and attach to render products
    writer = setup_writer(config)
    if not writer:
        print("[SDG] Failed to setup writer")
        return

    writer.attach(render_products)

    for render_product in render_products:
        render_product.hydra_texture.set_updates_enabled(True)

    rt_subframes = config.get("rt_subframes", -1)

    # Calculate camera randomization bounds
    pallet_tf = omni.usd.get_world_transform_matrix(pallet_prim)
    camera_bounds = setup_camera_bounds(pallet_prim, forklift_prim, pallet_tf, forklift_tf)
    pallet_cam_bounds_min = camera_bounds["pallet_cam"]["min"]
    pallet_cam_bounds_max = camera_bounds["pallet_cam"]["max"]
    top_cam_bounds_min = camera_bounds["top_cam"]["min"]
    top_cam_bounds_max = camera_bounds["top_cam"]["max"]
    driver_cam_bounds_min = camera_bounds["driver_cam"]["min"]
    driver_cam_bounds_max = camera_bounds["driver_cam"]["max"]

    # Setup scatter plane and cone placement
    scatter_plane = create_scatter_plane_for_prim(pallet_prim, pallet_tf, parent_path="/SDG", scale_factor=0.8)
    cone_placement_corners, forklift_rotation_deg = setup_cone_placement_corners(forklift_prim)

    # Register graph-based randomizers for lights and materials
    register_lights_graph_randomizer(forklift_prim, pallet_prim, event_name="randomize_lights")

    cardbox_material_urls = [
        f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/Materials/MI_PaperNotes_01.mdl",
        f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/Materials/MI_CardBoxB_05.mdl",
    ]
    register_cardboxes_materials_graph_randomizer(
        cardboxes, cardbox_material_urls, event_name="randomize_cardboxes_materials"
    )

    # Run physics simulation to settle boxes on pallet
    await simulate_falling_objects_async(forklift_prim, assets_root_path, config, rng=rng)

    # SDG loop - generate frames with randomizations
    num_frames = config.get("num_frames", 10)
    print(f"[SDG] Running SDG for {num_frames} frames")
    for i in range(num_frames):
        print(f"[SDG] Frame {i}/{num_frames}")

        print(f"[SDG]  Randomizing boxes on pallet.")
        rep.functional.randomizer.scatter_2d(
            prims=cardboxes, surface_prims=scatter_plane, check_for_collisions=True, rng=rng
        )

        print(f"[SDG]  Randomizing boxes materials.")
        rep.utils.send_og_event(event_name="randomize_cardboxes_materials")
        print(f"[SDG]  Randomizing lights.")
        rep.utils.send_og_event(event_name="randomize_lights")

        print(f"[SDG]  Randomizing pallet camera.")
        rep.functional.modify.pose(
            pallet_cam,
            position_value=rng.uniform(pallet_cam_bounds_min, pallet_cam_bounds_max),
            look_at_value=pallet_prim,
            look_at_up_axis=(0, 0, 1),
        )

        print(f"[SDG]  Randomizing driver camera.")
        rep.functional.modify.pose(
            driver_cam,
            position_value=rng.uniform(driver_cam_bounds_min, driver_cam_bounds_max),
            look_at_value=pallet_prim,
            look_at_up_axis=(0, 0, 1),
        )

        if i % 2 == 0:
            print(f"[SDG]  Randomizing cone position.")
            selected_corner = cone_placement_corners[rng.integers(0, len(cone_placement_corners))]
            rep.functional.modify.pose(
                cone,
                position_value=selected_corner,
            )

        if i % 4 == 0:
            print(f"[SDG]  Randomizing top view camera.")
            roll_angle = rng.uniform(0, 2 * np.pi)
            rep.functional.modify.pose(
                top_view_cam,
                position_value=rng.uniform(top_cam_bounds_min, top_cam_bounds_max),
                look_at_value=forklift_prim,
                look_at_up_axis=(np.cos(roll_angle), np.sin(roll_angle), 0.0),
            )

        print(f"[SDG]  Capturing frame with rt_subframes={rt_subframes}")
        await rep.orchestrator.step_async(delta_time=0.0, rt_subframes=rt_subframes)

    # Cleanup
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    for render_product in render_products:
        render_product.destroy()

    print("[SDG] Complete")

config = {
    "resolution": [512, 512],
    "rt_subframes": 32,
    "num_frames": 10,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "BasicWriter",
    "backend_type": "DiskBackend",
    "backend_params": {
        "output_dir": "_out_scene_based_sdg",
    },
    "writer_config": {
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "traffic_cone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "cardbox",
    },
}

asyncio.ensure_future(run_example_async(config))
```

Standalone Application

Main script

```python
"""Generate offline synthetic dataset"""

import argparse
import json
import math
import os

import numpy as np
import yaml
from isaacsim import SimulationApp

# Default configuration
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 32,
    "num_frames": 10,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "BasicWriter",
    "backend_type": "DiskBackend",
    "backend_params": {
        "output_dir": "_out_scene_based_sdg",
    },
    "writer_config": {
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "traffic_cone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "cardbox",
    },
    "close_app_after_run": True,
}

import carb

# Parse command line arguments for optional config file
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()

# Load config file if provided
args_config = {}
if args.config and os.path.isfile(args.config):
    print("File exist")
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Clear default writer_config if overridden in args
if "writer_config" in args_config:
    config["writer_config"].clear()

# Merge args config into default config
config.update(args_config)

# Initialize simulation app
simulation_app = SimulationApp(launch_config=config["launch_config"])

import carb.settings

# Runtime modules (must import after SimulationApp creation)
import omni.replicator.core as rep
import omni.usd
import scene_based_sdg_utils
from isaacsim.core.experimental.utils.semantics import remove_all_labels
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf

# Get assets root path from nucleus server
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

# Load environment stage
print(f"[SDG] Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage{config['env_url']}, closing application..")
    simulation_app.close()

# Initialize randomization
rep.set_global_seed(42)
rng = np.random.default_rng(42)

# Configure replicator for manual triggering
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode for best SDG results
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Clear previous semantic labels
if config["clear_previous_semantics"]:
    for prim in get_current_stage().Traverse():
        remove_all_labels(prim, include_descendants=True)

# Create SDG scope for organizing all generated objects
stage = get_current_stage()
sdg_scope = stage.DefinePrim("/SDG", "Scope")

# Spawn forklift at random pose
forklift_prim = prims.create_prim(
    prim_path="/SDG/Forklift",
    position=(rng.uniform(-20, -2), rng.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, rng.uniform(0, math.pi)]),
    usd_path=assets_root_path + config["forklift"]["url"],
    semantic_label=config["forklift"]["class"],
)

# Spawn pallet in front of forklift with random offset
forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, rng.uniform(-1.8, -1.2), 0))
pallet_pos = (pallet_offset_tf * forklift_tf).ExtractTranslation()
forklift_quat = forklift_tf.ExtractRotationQuat()
forklift_quat_xyzw = (forklift_quat.GetReal(), *forklift_quat.GetImaginary())

pallet_prim = prims.create_prim(
    prim_path="/SDG/Pallet",
    position=pallet_pos,
    orientation=forklift_quat_xyzw,
    usd_path=assets_root_path + config["pallet"]["url"],
    semantic_label=config["pallet"]["class"],
)

# Create cardboxes for pallet scattering
cardboxes = []
for i in range(5):
    cardbox = prims.create_prim(
        prim_path=f"/SDG/CardBox_{i}",
        usd_path=assets_root_path + config["cardbox"]["url"],
        semantic_label=config["cardbox"]["class"],
    )
    cardboxes.append(cardbox)

# Create traffic cone for corner placement
cone = prims.create_prim(
    prim_path="/SDG/Cone",
    usd_path=assets_root_path + config["cone"]["url"],
    semantic_label=config["cone"]["class"],
)

# Create cameras
rep.functional.create.scope(name="Cameras", parent="/SDG")
driver_cam = rep.functional.create.camera(
    focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name="DriverCam", parent="/SDG/Cameras"
)
pallet_cam = rep.functional.create.camera(name="PalletCam", parent="/SDG/Cameras")
top_view_cam = rep.functional.create.camera(clipping_range=(6.0, 1000000.0), name="TopCam", parent="/SDG/Cameras")

# Setup render products
resolution = config.get("resolution", (512, 512))
forklift_rp = rep.create.render_product(top_view_cam, resolution, name="TopView")
driver_rp = rep.create.render_product(driver_cam, resolution, name="DriverView")
pallet_rp = rep.create.render_product(pallet_cam, resolution, name="PalletView")

render_products = [forklift_rp, driver_rp, pallet_rp]
for render_product in render_products:
    render_product.hydra_texture.set_updates_enabled(False)

# Initialize writer and attach to render products
writer = scene_based_sdg_utils.setup_writer(config)
if not writer:
    carb.log_error("[SDG] Failed to setup writer, closing application.")
    simulation_app.close()

writer.attach(render_products)

for render_product in render_products:
    render_product.hydra_texture.set_updates_enabled(True)

# Configure raytracing subframes for material loading and motion artifacts
rt_subframes = config.get("rt_subframes", -1)

# Calculate camera randomization bounds
pallet_tf = omni.usd.get_world_transform_matrix(pallet_prim)
camera_bounds = scene_based_sdg_utils.setup_camera_bounds(pallet_prim, forklift_prim, pallet_tf, forklift_tf)
pallet_cam_bounds_min = camera_bounds["pallet_cam"]["min"]
pallet_cam_bounds_max = camera_bounds["pallet_cam"]["max"]
top_cam_bounds_min = camera_bounds["top_cam"]["min"]
top_cam_bounds_max = camera_bounds["top_cam"]["max"]
driver_cam_bounds_min = camera_bounds["driver_cam"]["min"]
driver_cam_bounds_max = camera_bounds["driver_cam"]["max"]

# Setup scatter plane and cone placement
scatter_plane = scene_based_sdg_utils.create_scatter_plane_for_prim(
    pallet_prim, pallet_tf, parent_path="/SDG", scale_factor=0.8
)
cone_placement_corners, forklift_rotation_deg = scene_based_sdg_utils.setup_cone_placement_corners(forklift_prim)

# Register graph-based randomizers for lights and materials
scene_based_sdg_utils.register_lights_graph_randomizer(forklift_prim, pallet_prim, event_name="randomize_lights")

cardbox_material_urls = [
    f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/Materials/MI_PaperNotes_01.mdl",
    f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/Materials/MI_CardBoxB_05.mdl",
]
scene_based_sdg_utils.register_cardboxes_materials_graph_randomizer(
    cardboxes, cardbox_material_urls, event_name="randomize_cardboxes_materials"
)

# Run physics simulation to settle boxes on pallet
scene_based_sdg_utils.simulate_falling_objects(forklift_prim, assets_root_path, config, rng=rng)

# SDG loop - generate frames with randomizations
num_frames = config.get("num_frames", 0)
print(f"[SDG] Running SDG for {num_frames} frames")
for i in range(num_frames):
    print(f"[SDG] Frame {i}/{num_frames}")

    print(f"[SDG]  Randomizing boxes on pallet.")
    rep.functional.randomizer.scatter_2d(
        prims=cardboxes, surface_prims=scatter_plane, check_for_collisions=True, rng=rng
    )

    print(f"[SDG]  Randomizing boxes materials.")
    rep.utils.send_og_event(event_name="randomize_cardboxes_materials")
    print(f"[SDG]  Randomizing lights.")
    rep.utils.send_og_event(event_name="randomize_lights")

    print(f"[SDG]  Randomizing pallet camera.")
    rep.functional.modify.pose(
        pallet_cam,
        position_value=rng.uniform(pallet_cam_bounds_min, pallet_cam_bounds_max),
        look_at_value=pallet_prim,
        look_at_up_axis=(0, 0, 1),
    )

    print(f"[SDG]  Randomizing driver camera.")
    rep.functional.modify.pose(
        driver_cam,
        position_value=rng.uniform(driver_cam_bounds_min, driver_cam_bounds_max),
        look_at_value=pallet_prim,
        look_at_up_axis=(0, 0, 1),
    )

    if i % 2 == 0:
        print(f"[SDG]  Randomizing cone position.")
        selected_corner = cone_placement_corners[rng.integers(0, len(cone_placement_corners))]
        rep.functional.modify.pose(
            cone,
            position_value=selected_corner,
        )

    if i % 4 == 0:
        print(f"[SDG]  Randomizing top view camera.")
        roll_angle = rng.uniform(0, 2 * np.pi)
        rep.functional.modify.pose(
            top_view_cam,
            position_value=rng.uniform(top_cam_bounds_min, top_cam_bounds_max),
            look_at_value=forklift_prim,
            look_at_up_axis=(np.cos(roll_angle), np.sin(roll_angle), 0.0),
        )

    print(f"[SDG]  Capturing frame with rt_subframes={rt_subframes}")
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)

# Cleanup
rep.orchestrator.wait_until_complete()
writer.detach()
for render_product in render_products:
    render_product.destroy()

# Check if the application should keep running after data generation
close_app_after_run = config.get("close_app_after_run", True)
if config["launch_config"]["headless"]:
    if not close_app_after_run:
        print("[SDG] 'close_app_after_run' is ignored when running headless. The application will be closed.")
elif not close_app_after_run:
    print("[SDG] The application will not be closed after the run. Make sure to close it manually.")
    while simulation_app.is_running():
        simulation_app.update()
simulation_app.close()
```

Utils module

```python
import math
import os

import carb
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils import prims
from isaacsim.core.utils.bounds import compute_combined_aabb, compute_obb, create_bbox_cache, get_obb_corners
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from pxr import Gf, Usd, UsdGeom

def setup_writer(config: dict) -> rep.Writer | None:
    """Setup and initialize writer with optional backend support and error handling."""

    def normalize_output_dir(params):
        """Convert relative output_dir to absolute path."""
        if "output_dir" in params and not os.path.isabs(params["output_dir"]):
            params["output_dir"] = os.path.join(os.getcwd(), params["output_dir"])

    # Get writer from registry
    writer_type = config.get("writer", "BasicWriter")
    if writer_type not in rep.WriterRegistry.get_writers():
        carb.log_error(f"[SDG] Writer type '{writer_type}' not found in registry.")
        return None

    writer = rep.WriterRegistry.get(writer_type)
    writer_kwargs = dict(config.get("writer_config", {}))
    normalize_output_dir(writer_kwargs)

    # Initialize backend if specified
    backend_type = config.get("backend_type")
    backend = None
    if backend_type:
        try:
            backend = rep.backends.get(backend_type)
        except Exception as e:
            carb.log_error(f"[SDG] Backend '{backend_type}' not found: {e}")
            return None

        backend_params = dict(config.get("backend_params", {}))
        normalize_output_dir(backend_params)

        try:
            print(f"[SDG] Backend: {backend_type} | Params: {backend_params}")
            backend.initialize(**backend_params)
        except TypeError as e:
            carb.log_error(f"[SDG] Invalid backend params: {e}")
            return None

    # Initialize writer
    if "output_dir" in writer_kwargs:
        print(f"[SDG] Output: {writer_kwargs['output_dir']}")

    backend_info = f" + {backend_type}" if backend else ""
    print(f"[SDG] Writer: {writer_type}{backend_info} | Config: {writer_kwargs}")

    try:
        if backend:
            writer.initialize(backend=backend, **writer_kwargs)
        else:
            writer.initialize(**writer_kwargs)
    except TypeError as e:
        carb.log_error(f"[SDG] Invalid writer params: {e}")
        return None

    return writer

def simulate_falling_objects(
    forklift_prim: Usd.Prim,
    assets_root_path: str,
    config: dict,
    max_sim_steps: int = 250,
    num_boxes: int = 8,
    rng: np.random.Generator | None = None,
) -> None:
    """Run physics simulation to drop boxes on pallet near forklift."""
    if rng is None:
        rng = np.random.default_rng()

    # Spawn pallet at random position relative to forklift
    forklift_transform = omni.usd.get_world_transform_matrix(forklift_prim)
    sim_pallet_offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(rng.uniform(-1, 1), rng.uniform(-4, -3.6), 0))
    sim_pallet_position = (sim_pallet_offset * forklift_transform).ExtractTranslation()
    sim_pallet_rotation = euler_angles_to_quat([0, 0, rng.uniform(0, math.pi)])

    sim_pallet = prims.create_prim(
        prim_path="/World/SimulatedPallet",
        position=sim_pallet_position,
        orientation=sim_pallet_rotation,
        usd_path=assets_root_path + config["pallet"]["url"],
        semantic_label=config["pallet"]["class"],
    )
    sim_pallet_geom = GeomPrim(f"{str(sim_pallet.GetPrimPath())}/.*", apply_collision_apis=True)
    sim_pallet_geom.set_collision_approximations("boundingCube")

    # Spawn boxes stacked above pallet
    bbox_cache = create_bbox_cache()
    current_height = bbox_cache.ComputeLocalBound(sim_pallet).GetRange().GetSize()[2] * 1.1

    sim_box_rigid_prims = []
    for box_index in range(num_boxes):
        box_xy_offset = Gf.Vec3d(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), current_height)
        sim_box = prims.create_prim(
            prim_path=f"/World/SimulatedCardbox_{box_index}",
            position=sim_pallet_position + box_xy_offset,
            orientation=sim_pallet_rotation,
            usd_path=assets_root_path + config["cardbox"]["url"],
            semantic_label=config["cardbox"]["class"],
        )
        current_height += bbox_cache.ComputeLocalBound(sim_box).GetRange().GetSize()[2] * 1.1

        sim_box_geom = GeomPrim(f"{str(sim_box.GetPrimPath())}/.*", apply_collision_apis=True)
        sim_box_geom.set_collision_approximations("convexHull")
        sim_box_rigid_prims.append(RigidPrim(str(sim_box.GetPrimPath())))

    # Run physics simulation
    SimulationManager.set_physics_dt(1.0 / 90.0)
    SimulationManager.initialize_physics()

    # Simulate until boxes settle or max steps reached
    velocity_threshold = 0.01
    for step in range(max_sim_steps):
        SimulationManager.step()
        if sim_box_rigid_prims:
            top_box_velocity = sim_box_rigid_prims[-1].get_velocities(indices=[0])[0].numpy()
            if np.linalg.norm(top_box_velocity) < velocity_threshold:
                print(f"[SDG] Simulation settled at step {step}")
                break

def setup_camera_bounds(
    pallet_prim: Usd.Prim, forklift_prim: Usd.Prim, pallet_tf: Gf.Matrix4d, forklift_tf: Gf.Matrix4d
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Calculate camera randomization bounds for pallet, top view, and driver cameras."""
    pallet_pos = pallet_tf.ExtractTranslation()
    pallet_cam_bounds = {
        "min": (pallet_pos[0] - 2, pallet_pos[1] - 2, 2),
        "max": (pallet_pos[0] + 2, pallet_pos[1] + 2, 4),
    }

    forklift_pos = forklift_tf.ExtractTranslation()
    top_cam_bounds = {
        "min": (forklift_pos[0], forklift_pos[1], 9),
        "max": (forklift_pos[0], forklift_pos[1], 11),
    }

    driver_cam_pos = forklift_pos + Gf.Vec3d(0.0, 0.0, 1.9)
    driver_cam_bounds = {
        "min": (driver_cam_pos[0], driver_cam_pos[1], driver_cam_pos[2] - 0.25),
        "max": (driver_cam_pos[0], driver_cam_pos[1], driver_cam_pos[2] + 0.25),
    }

    return {
        "pallet_cam": pallet_cam_bounds,
        "top_cam": top_cam_bounds,
        "driver_cam": driver_cam_bounds,
    }

def create_scatter_plane_for_prim(
    prim: Usd.Prim, prim_tf: Gf.Matrix4d, parent_path: str, scale_factor: float = 0.8, visible: bool = False
) -> Usd.Prim:
    """Create scatter plane sized and aligned to prim surface."""
    bb_cache = create_bbox_cache()
    prim_bbox = bb_cache.ComputeLocalBound(prim)
    prim_bbox.Transform(prim_tf)
    prim_size = prim_bbox.GetRange().GetSize()

    prim_quat = prim_tf.ExtractRotation().GetQuaternion()
    prim_quat_xyzw = (prim_quat.GetReal(), *prim_quat.GetImaginary())
    prim_rotation_deg = quat_to_euler_angles(np.array(prim_quat_xyzw), degrees=True)

    prim_pos = prim_tf.ExtractTranslation()
    scatter_plane_scale = (prim_size[0] * scale_factor, prim_size[1] * scale_factor, 1)
    scatter_plane_pos = prim_pos + Gf.Vec3d(0, 0, prim_size[2])

    scatter_plane = rep.functional.create.plane(
        scale=scatter_plane_scale,
        position=tuple(scatter_plane_pos),
        rotation=tuple(prim_rotation_deg),
        visible=visible,
        parent=parent_path,
    )

    return scatter_plane

def setup_cone_placement_corners(
    forklift_prim: Usd.Prim, bb_cache=None, scale_factor: float = 1.3
) -> tuple[list[list[float]], tuple[float, float, float]]:
    """Calculate forklift OBB corners for cone placement, returns (corner_positions, rotation_degrees)."""
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    forklift_obb_center, forklift_obb_axes, forklift_obb_extent = compute_obb(bb_cache, forklift_prim.GetPrimPath())
    enlarged_extent = (
        forklift_obb_extent[0] * scale_factor,
        forklift_obb_extent[1] * scale_factor,
        forklift_obb_extent[2],
    )
    forklift_obb_corners = get_obb_corners(forklift_obb_center, forklift_obb_axes, enlarged_extent)

    cone_placement_corners = [
        forklift_obb_corners[0].tolist(),
        forklift_obb_corners[2].tolist(),
        forklift_obb_corners[4].tolist(),
        forklift_obb_corners[6].tolist(),
    ]

    forklift_obb_quat = Gf.Matrix3d(forklift_obb_axes).ExtractRotation().GetQuaternion()
    forklift_obb_quat_xyzw = (forklift_obb_quat.GetReal(), *forklift_obb_quat.GetImaginary())
    forklift_rotation_deg = quat_to_euler_angles(np.array(forklift_obb_quat_xyzw), degrees=True)

    return cone_placement_corners, forklift_rotation_deg

def register_lights_graph_randomizer(forklift_prim: Usd.Prim, pallet_prim: Usd.Prim, event_name: str) -> None:
    """Register graph randomizer to create sphere lights with varying color, intensity, and position."""
    bb_cache = create_bbox_cache()
    combined_bounds = compute_combined_aabb(bb_cache, [forklift_prim.GetPrimPath(), pallet_prim.GetPrimPath()])
    light_pos_min = (combined_bounds[0], combined_bounds[1], 6)
    light_pos_max = (combined_bounds[3], combined_bounds[4], 7)

    with rep.trigger.on_custom_event(event_name):
        rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0.2, 0.1, 0.1), (0.9, 0.8, 0.8)),
            intensity=rep.distribution.uniform(2000, 4000),
            position=rep.distribution.uniform(light_pos_min, light_pos_max),
            scale=rep.distribution.uniform(1, 4),
            count=3,
        )

def register_cardboxes_materials_graph_randomizer(
    cardboxes: list[Usd.Prim], cardbox_material_urls: list[str], event_name: str
) -> None:
    """Register graph randomizer to apply random materials to cardbox meshes."""
    cardbox_mesh_paths = []
    for cardbox in cardboxes:
        meshes = [child for child in cardbox.GetChildren() if child.IsA(UsdGeom.Mesh)]
        cardbox_mesh_paths.extend([mesh.GetPrimPath() for mesh in meshes])

    with rep.trigger.on_custom_event(event_name):
        cardbox_mesh_group_node = rep.create.group(cardbox_mesh_paths)
        with cardbox_mesh_group_node:
            rep.randomizer.materials(cardbox_material_urls)
```

## Config Scenarios

The following provides details about the various config scenarios:

Built-in

Without an explicit config file, the script uses the default parameters stored in the script itself. The default parameters are the following:

Built-in (default) Config

```python
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "resolution": [512, 512],
    "rt_subframes": 32,
    "num_frames": 10,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "BasicWriter",
    "backend_type": "DiskBackend",
    "backend_params": {
        "output_dir": "_out_scene_based_sdg",
    },
    "writer_config": {
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "distance_to_image_plane": True,
        "bounding_box_3d": True,
        "occlusion": True,
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "forklift",
    },
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "traffic_cone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "cardbox",
    },
    "close_app_after_run": True,
}
```

The following command runs the script with the default parameters:

```python
./python.sh standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py
```

Basic Writer

Using the `config_basic_writer.yaml` config file explictly chooses `BasicWriter` with the given `writer_config` configurations. It also changes the environment to `/Isaac/Environments/Grid/default_environment.usd`.

Custom YAML Config

```python
launch_config:
  renderer: RealTimePathTracing
  headless: false
resolution: [512, 512]
env_url: "/Isaac/Environments/Grid/default_environment.usd"
rt_subframes: 32
writer: BasicWriter
backend_type: DiskBackend
backend_params:
  output_dir: _out_basicwriter
writer_config:
  rgb: true
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py \
    --config standalone_examples/replicator/scene_based_sdg/config/config_basic_writer.yaml
```

Default Writer

The `config_default_writer.json` uses the default writer (which is still the `BasicWriter`) and changes the `writer_config` values to **rgb** and **instance\_segmentation** annotators.

Custom JSON Config

```python
{
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": false
    },
    "resolution": [512, 512],
    "backend_type": "DiskBackend",
    "backend_params": {
        "output_dir": "_out_defaultwriter"
    },
    "writer_config": {
        "rgb": true,
        "instance_segmentation": true
    }
}
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py \
    --config standalone_examples/replicator/scene_based_sdg/config/config_default_writer.json
```

Kitti Writer

The `config_kitti_writer.yaml` config file uses `KittiWriter` with the given `writer_config` configurations.

Custom YAML Config using KittiWriter

```python
launch_config:
  renderer: RealTimePathTracing
  headless: true
resolution: [512, 512]
num_frames: 5
clear_previous_semantics: false
writer: KittiWriter
backend_type: null
writer_config:
  output_dir: _out_kitti
  colorize_instance_segmentation: true
  mapping_dict:
    forklift: [11, 110, 223, 255]
    pallet: [211, 210, 223, 255]
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py \
    --config standalone_examples/replicator/scene_based_sdg/config/config_kitti_writer.yaml
```

Coco Writer

The `config_coco_writer.yaml` config file uses `CocoWriter` with the given `writer_config` configurations.

Custom YAML Config using CocoWriter

```python
launch_config:
  renderer: RealTimePathTracing
  headless: true
resolution: [512, 512]
num_frames: 5
clear_previous_semantics: true
backend_type: null
writer: CocoWriter
writer_config:
  output_dir: _out_coco
  coco_categories:
    forklift:
      name: forklift
      id: 333
      supercategory: warehouse
      color: [211, 111, 211]
      isthing: 1
    pallet:
      name: pallet
      id: 313
      supercategory: warehouse
      color: [141, 111, 131]
      isthing: 1
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/scene_based_sdg/scene_based_sdg.py \
    --config standalone_examples/replicator/scene_based_sdg/config/config_coco_writer.yaml
```

## Loading the Environment

The environment is a USD stage. Use `get_assets_root_path_async` to get the path to the nucleus server and then load the environment using `omni.usd.get_context().open_stage()`.

Load the Environment

```python
# Get assets root path from nucleus server
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

# Load environment stage
print(f"[SDG] Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage{config['env_url']}, closing application..")
    simulation_app.close()

# Initialize randomization
rep.set_global_seed(42)
rng = np.random.default_rng(42)

# Configure replicator for manual triggering
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode for best SDG results
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Clear previous semantic labels
if config["clear_previous_semantics"]:
    for prim in get_current_stage().Traverse():
        remove_all_labels(prim, include_descendants=True)
```

## Creating the Cameras and the Writer

The example provides two ways (Replicator and Isaac Sim API) of creating cameras `rep.create.camera` and `prims.create_prim`. ``` prims.create_prim``is used as render products to generate the data. The created render products are attached to the built-in ``BasicWriter ``` to collect the data from the selected annotators (rgb, semantic\_segmentation, bounding\_box\_3d) and to write it to the given output path. Use ``` rep.get.prim_at_path``to access ``driver_cam_prim ``` wrapped in an OmniGraph node so that it can be randomized by each step of the randomization graph generated by Replicator.

The cameras used in the examples are created using `rep.functional.create.camera`, which create camera prims used by render products.

Cameras

```python
# Create cameras
rep.functional.create.scope(name="Cameras", parent="/SDG")
driver_cam = rep.functional.create.camera(
    focus_distance=400.0, focal_length=24.0, clipping_range=(0.1, 10000000.0), name="DriverCam", parent="/SDG/Cameras"
)
pallet_cam = rep.functional.create.camera(name="PalletCam", parent="/SDG/Cameras")
top_view_cam = rep.functional.create.camera(clipping_range=(6.0, 1000000.0), name="TopCam", parent="/SDG/Cameras")
```

From the cameras, render products are created and disabled until the SDG pipeline starts to improve performance by avoiding unnecessary rendering. The writer setup is handled by a helper function that supports optional backend configuration for flexible output handling.

Writer and Render Products

```python
# Setup render products
resolution = config.get("resolution", (512, 512))
forklift_rp = rep.create.render_product(top_view_cam, resolution, name="TopView")
driver_rp = rep.create.render_product(driver_cam, resolution, name="DriverView")
pallet_rp = rep.create.render_product(pallet_cam, resolution, name="PalletView")

render_products = [forklift_rp, driver_rp, pallet_rp]
for render_product in render_products:
    render_product.hydra_texture.set_updates_enabled(False)

# Initialize writer and attach to render products
writer = scene_based_sdg_utils.setup_writer(config)
if not writer:
    carb.log_error("[SDG] Failed to setup writer, closing application.")
    simulation_app.close()

writer.attach(render_products)

for render_product in render_products:
    render_product.hydra_texture.set_updates_enabled(True)
```

The `setup_writer` helper function handles writer initialization with optional backend support:

```python
def setup_writer(config: dict) -> rep.Writer | None:
    """Setup and initialize writer with optional backend support and error handling."""

    def normalize_output_dir(params):
        """Convert relative output_dir to absolute path."""
        if "output_dir" in params and not os.path.isabs(params["output_dir"]):
            params["output_dir"] = os.path.join(os.getcwd(), params["output_dir"])

    # Get writer from registry
    writer_type = config.get("writer", "BasicWriter")
    if writer_type not in rep.WriterRegistry.get_writers():
        carb.log_error(f"[SDG] Writer type '{writer_type}' not found in registry.")
        return None

    writer = rep.WriterRegistry.get(writer_type)
    writer_kwargs = dict(config.get("writer_config", {}))
    normalize_output_dir(writer_kwargs)

    # Initialize backend if specified
    backend_type = config.get("backend_type")
    backend = None
    if backend_type:
        try:
            backend = rep.backends.get(backend_type)
        except Exception as e:
            carb.log_error(f"[SDG] Backend '{backend_type}' not found: {e}")
            return None

        backend_params = dict(config.get("backend_params", {}))
        normalize_output_dir(backend_params)

        try:
            print(f"[SDG] Backend: {backend_type} | Params: {backend_params}")
            backend.initialize(**backend_params)
        except TypeError as e:
            carb.log_error(f"[SDG] Invalid backend params: {e}")
            return None

    # Initialize writer
    if "output_dir" in writer_kwargs:
        print(f"[SDG] Output: {writer_kwargs['output_dir']}")

    backend_info = f" + {backend_type}" if backend else ""
    print(f"[SDG] Writer: {writer_type}{backend_info} | Config: {writer_kwargs}")

    try:
        if backend:
            writer.initialize(backend=backend, **writer_kwargs)
        else:
            writer.initialize(**writer_kwargs)
    except TypeError as e:
        carb.log_error(f"[SDG] Invalid writer params: {e}")
        return None

    return writer
```

## Domain Randomization

The following snippet provides examples of various randomization possibilities using Isaac Sim and Replicator API. The example uses a seeded random number generator (`numpy.random.Generator`) for reproducible randomization. It starts by spawning a forklift using the Isaac Sim API to a randomly generated pose. It then uses the forklift pose to place a pallet in front of it within the bounds of a random distance. Cardboxes and a traffic cone are also created upfront for later randomization.

Isaac Sim API Asset Spawning

```python
# Spawn forklift at random pose
forklift_prim = prims.create_prim(
    prim_path="/SDG/Forklift",
    position=(rng.uniform(-20, -2), rng.uniform(-1, 3), 0),
    orientation=euler_angles_to_quat([0, 0, rng.uniform(0, math.pi)]),
    usd_path=assets_root_path + config["forklift"]["url"],
    semantic_label=config["forklift"]["class"],
)

# Spawn pallet in front of forklift with random offset
forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, rng.uniform(-1.8, -1.2), 0))
pallet_pos = (pallet_offset_tf * forklift_tf).ExtractTranslation()
forklift_quat = forklift_tf.ExtractRotationQuat()
forklift_quat_xyzw = (forklift_quat.GetReal(), *forklift_quat.GetImaginary())

pallet_prim = prims.create_prim(
    prim_path="/SDG/Pallet",
    position=pallet_pos,
    orientation=forklift_quat_xyzw,
    usd_path=assets_root_path + config["pallet"]["url"],
    semantic_label=config["pallet"]["class"],
)

# Create cardboxes for pallet scattering
cardboxes = []
for i in range(5):
    cardbox = prims.create_prim(
        prim_path=f"/SDG/CardBox_{i}",
        usd_path=assets_root_path + config["cardbox"]["url"],
        semantic_label=config["cardbox"]["class"],
    )
    cardboxes.append(cardbox)

# Create traffic cone for corner placement
cone = prims.create_prim(
    prim_path="/SDG/Cone",
    usd_path=assets_root_path + config["cone"]["url"],
    semantic_label=config["cone"]["class"],
)
```

The new Replicator API uses `rep.functional` for direct randomization without graph registration. A scatter plane is created using a helper function, and boxes are scattered directly using `rep.functional.randomizer.scatter_2d` in the SDG loop. Material randomization is handled through a separate graph-based randomizer.

Scatter Plane Setup

```python
def create_scatter_plane_for_prim(
    prim: Usd.Prim, prim_tf: Gf.Matrix4d, parent_path: str, scale_factor: float = 0.8, visible: bool = False
) -> Usd.Prim:
    """Create scatter plane sized and aligned to prim surface."""
    bb_cache = create_bbox_cache()
    prim_bbox = bb_cache.ComputeLocalBound(prim)
    prim_bbox.Transform(prim_tf)
    prim_size = prim_bbox.GetRange().GetSize()

    prim_quat = prim_tf.ExtractRotation().GetQuaternion()
    prim_quat_xyzw = (prim_quat.GetReal(), *prim_quat.GetImaginary())
    prim_rotation_deg = quat_to_euler_angles(np.array(prim_quat_xyzw), degrees=True)

    prim_pos = prim_tf.ExtractTranslation()
    scatter_plane_scale = (prim_size[0] * scale_factor, prim_size[1] * scale_factor, 1)
    scatter_plane_pos = prim_pos + Gf.Vec3d(0, 0, prim_size[2])

    scatter_plane = rep.functional.create.plane(
        scale=scatter_plane_scale,
        position=tuple(scatter_plane_pos),
        rotation=tuple(prim_rotation_deg),
        visible=visible,
        parent=parent_path,
    )

    return scatter_plane
```

Material randomization for cardboxes is registered as a graph-based randomizer triggered by custom events:

Material Randomization Graph

```python
def register_cardboxes_materials_graph_randomizer(
    cardboxes: list[Usd.Prim], cardbox_material_urls: list[str], event_name: str
) -> None:
    """Register graph randomizer to apply random materials to cardbox meshes."""
    cardbox_mesh_paths = []
    for cardbox in cardboxes:
        meshes = [child for child in cardbox.GetChildren() if child.IsA(UsdGeom.Mesh)]
        cardbox_mesh_paths.extend([mesh.GetPrimPath() for mesh in meshes])

    with rep.trigger.on_custom_event(event_name):
        cardbox_mesh_group_node = rep.create.group(cardbox_mesh_paths)
        with cardbox_mesh_group_node:
            rep.randomizer.materials(cardbox_material_urls)
```

The traffic cone is positioned at one of the forklift’s bounding box corners. A helper function calculates the corner positions:

Cone Placement Setup

```python
def setup_cone_placement_corners(
    forklift_prim: Usd.Prim, bb_cache=None, scale_factor: float = 1.3
) -> tuple[list[list[float]], tuple[float, float, float]]:
    """Calculate forklift OBB corners for cone placement, returns (corner_positions, rotation_degrees)."""
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    forklift_obb_center, forklift_obb_axes, forklift_obb_extent = compute_obb(bb_cache, forklift_prim.GetPrimPath())
    enlarged_extent = (
        forklift_obb_extent[0] * scale_factor,
        forklift_obb_extent[1] * scale_factor,
        forklift_obb_extent[2],
    )
    forklift_obb_corners = get_obb_corners(forklift_obb_center, forklift_obb_axes, enlarged_extent)

    cone_placement_corners = [
        forklift_obb_corners[0].tolist(),
        forklift_obb_corners[2].tolist(),
        forklift_obb_corners[4].tolist(),
        forklift_obb_corners[6].tolist(),
    ]

    forklift_obb_quat = Gf.Matrix3d(forklift_obb_axes).ExtractRotation().GetQuaternion()
    forklift_obb_quat_xyzw = (forklift_obb_quat.GetReal(), *forklift_obb_quat.GetImaginary())
    forklift_rotation_deg = quat_to_euler_angles(np.array(forklift_obb_quat_xyzw), degrees=True)

    return cone_placement_corners, forklift_rotation_deg
```

Light randomization is registered as a graph-based randomizer triggered by custom events:

Light Randomization Graph

```python
def register_lights_graph_randomizer(forklift_prim: Usd.Prim, pallet_prim: Usd.Prim, event_name: str) -> None:
    """Register graph randomizer to create sphere lights with varying color, intensity, and position."""
    bb_cache = create_bbox_cache()
    combined_bounds = compute_combined_aabb(bb_cache, [forklift_prim.GetPrimPath(), pallet_prim.GetPrimPath()])
    light_pos_min = (combined_bounds[0], combined_bounds[1], 6)
    light_pos_max = (combined_bounds[3], combined_bounds[4], 7)

    with rep.trigger.on_custom_event(event_name):
        rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0.2, 0.1, 0.1), (0.9, 0.8, 0.8)),
            intensity=rep.distribution.uniform(2000, 4000),
            position=rep.distribution.uniform(light_pos_min, light_pos_max),
            scale=rep.distribution.uniform(1, 4),
            count=3,
        )
```

Similar to the above examples, Replicator has support for many other randomizations. For more information, see Replicator’s [randomizer examples tutorials](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)").

Camera bounds are calculated using a helper function to determine the randomization ranges:

Camera Bounds Setup

```python
def setup_camera_bounds(
    pallet_prim: Usd.Prim, forklift_prim: Usd.Prim, pallet_tf: Gf.Matrix4d, forklift_tf: Gf.Matrix4d
) -> dict[str, dict[str, tuple[float, float, float]]]:
    """Calculate camera randomization bounds for pallet, top view, and driver cameras."""
    pallet_pos = pallet_tf.ExtractTranslation()
    pallet_cam_bounds = {
        "min": (pallet_pos[0] - 2, pallet_pos[1] - 2, 2),
        "max": (pallet_pos[0] + 2, pallet_pos[1] + 2, 4),
    }

    forklift_pos = forklift_tf.ExtractTranslation()
    top_cam_bounds = {
        "min": (forklift_pos[0], forklift_pos[1], 9),
        "max": (forklift_pos[0], forklift_pos[1], 11),
    }

    driver_cam_pos = forklift_pos + Gf.Vec3d(0.0, 0.0, 1.9)
    driver_cam_bounds = {
        "min": (driver_cam_pos[0], driver_cam_pos[1], driver_cam_pos[2] - 0.25),
        "max": (driver_cam_pos[0], driver_cam_pos[1], driver_cam_pos[2] + 0.25),
    }

    return {
        "pallet_cam": pallet_cam_bounds,
        "top_cam": top_cam_bounds,
        "driver_cam": driver_cam_bounds,
    }
```

After setting up the randomizers and before running the data collection, a short physics simulation is run. The example drops several stacked boxes on a pallet behind the forklift using `SimulationManager` and the experimental `GeomPrim` and `RigidPrim` classes.

Isaac Sim Simulation

```python
def simulate_falling_objects(
    forklift_prim: Usd.Prim,
    assets_root_path: str,
    config: dict,
    max_sim_steps: int = 250,
    num_boxes: int = 8,
    rng: np.random.Generator | None = None,
) -> None:
    """Run physics simulation to drop boxes on pallet near forklift."""
    if rng is None:
        rng = np.random.default_rng()

    # Spawn pallet at random position relative to forklift
    forklift_transform = omni.usd.get_world_transform_matrix(forklift_prim)
    sim_pallet_offset = Gf.Matrix4d().SetTranslate(Gf.Vec3d(rng.uniform(-1, 1), rng.uniform(-4, -3.6), 0))
    sim_pallet_position = (sim_pallet_offset * forklift_transform).ExtractTranslation()
    sim_pallet_rotation = euler_angles_to_quat([0, 0, rng.uniform(0, math.pi)])

    sim_pallet = prims.create_prim(
        prim_path="/World/SimulatedPallet",
        position=sim_pallet_position,
        orientation=sim_pallet_rotation,
        usd_path=assets_root_path + config["pallet"]["url"],
        semantic_label=config["pallet"]["class"],
    )
    sim_pallet_geom = GeomPrim(f"{str(sim_pallet.GetPrimPath())}/.*", apply_collision_apis=True)
    sim_pallet_geom.set_collision_approximations("boundingCube")

    # Spawn boxes stacked above pallet
    bbox_cache = create_bbox_cache()
    current_height = bbox_cache.ComputeLocalBound(sim_pallet).GetRange().GetSize()[2] * 1.1

    sim_box_rigid_prims = []
    for box_index in range(num_boxes):
        box_xy_offset = Gf.Vec3d(rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), current_height)
        sim_box = prims.create_prim(
            prim_path=f"/World/SimulatedCardbox_{box_index}",
            position=sim_pallet_position + box_xy_offset,
            orientation=sim_pallet_rotation,
            usd_path=assets_root_path + config["cardbox"]["url"],
            semantic_label=config["cardbox"]["class"],
        )
        current_height += bbox_cache.ComputeLocalBound(sim_box).GetRange().GetSize()[2] * 1.1

        sim_box_geom = GeomPrim(f"{str(sim_box.GetPrimPath())}/.*", apply_collision_apis=True)
        sim_box_geom.set_collision_approximations("convexHull")
        sim_box_rigid_prims.append(RigidPrim(str(sim_box.GetPrimPath())))

    # Run physics simulation
    SimulationManager.set_physics_dt(1.0 / 90.0)
    SimulationManager.initialize_physics()

    # Simulate until boxes settle or max steps reached
    velocity_threshold = 0.01
    for step in range(max_sim_steps):
        SimulationManager.step()
        if sim_box_rigid_prims:
            top_box_velocity = sim_box_rigid_prims[-1].get_velocities(indices=[0])[0].numpy()
            if np.linalg.norm(top_box_velocity) < velocity_threshold:
                print(f"[SDG] Simulation settled at step {step}")
                break
```

## Running the Script

The SDG loop runs randomizations directly using `rep.functional` APIs, triggers graph-based randomizers via custom events, and captures frames. The loop uses the seeded random number generator for reproducible results.

SDG Loop Execution

```python
# SDG loop - generate frames with randomizations
num_frames = config.get("num_frames", 0)
print(f"[SDG] Running SDG for {num_frames} frames")
for i in range(num_frames):
    print(f"[SDG] Frame {i}/{num_frames}")

    print(f"[SDG]  Randomizing boxes on pallet.")
    rep.functional.randomizer.scatter_2d(
        prims=cardboxes, surface_prims=scatter_plane, check_for_collisions=True, rng=rng
    )

    print(f"[SDG]  Randomizing boxes materials.")
    rep.utils.send_og_event(event_name="randomize_cardboxes_materials")
    print(f"[SDG]  Randomizing lights.")
    rep.utils.send_og_event(event_name="randomize_lights")

    print(f"[SDG]  Randomizing pallet camera.")
    rep.functional.modify.pose(
        pallet_cam,
        position_value=rng.uniform(pallet_cam_bounds_min, pallet_cam_bounds_max),
        look_at_value=pallet_prim,
        look_at_up_axis=(0, 0, 1),
    )

    print(f"[SDG]  Randomizing driver camera.")
    rep.functional.modify.pose(
        driver_cam,
        position_value=rng.uniform(driver_cam_bounds_min, driver_cam_bounds_max),
        look_at_value=pallet_prim,
        look_at_up_axis=(0, 0, 1),
    )

    if i % 2 == 0:
        print(f"[SDG]  Randomizing cone position.")
        selected_corner = cone_placement_corners[rng.integers(0, len(cone_placement_corners))]
        rep.functional.modify.pose(
            cone,
            position_value=selected_corner,
        )

    if i % 4 == 0:
        print(f"[SDG]  Randomizing top view camera.")
        roll_angle = rng.uniform(0, 2 * np.pi)
        rep.functional.modify.pose(
            top_view_cam,
            position_value=rng.uniform(top_cam_bounds_min, top_cam_bounds_max),
            look_at_value=forklift_prim,
            look_at_up_axis=(np.cos(roll_angle), np.sin(roll_angle), 0.0),
        )

    print(f"[SDG]  Capturing frame with rt_subframes={rt_subframes}")
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)
```

After the SDG loop completes, proper cleanup ensures all data is written and resources are released:

Cleanup

```python
# Cleanup
rep.orchestrator.wait_until_complete()
writer.detach()
for render_product in render_products:
    render_product.destroy()

# Check if the application should keep running after data generation
close_app_after_run = config.get("close_app_after_run", True)
if config["launch_config"]["headless"]:
    if not close_app_after_run:
        print("[SDG] 'close_app_after_run' is ignored when running headless. The application will be closed.")
elif not close_app_after_run:
    print("[SDG] The application will not be closed after the run. Make sure to close it manually.")
    while simulation_app.is_running():
        simulation_app.update()
simulation_app.close()
```

## Summary

This tutorial covered the following topics:

1. Starting a `SimulationApp` instance of Isaac Sim to work with Replicator
2. Loading a stage and custom assets at random locations using Isaac Sim API with seeded randomization
3. Setting up cameras using `rep.functional.create.camera` with organized stage structure
4. Configuring writers with optional backend support for flexible output handling
5. Using `rep.functional` APIs for direct randomization (scatter, pose modification)
6. Creating graph-based randomizers for lights and materials triggered by custom events
7. Running physics simulations with `SimulationManager` and experimental `GeomPrim`/`RigidPrim` classes
8. Proper cleanup of writers and render products

## Next Steps

One possible use for the created data is with the TAO Toolkit. After the generated synthetic data is in Kitti format, you can use the TAO Toolkit to
train a model. TAO provides [segmentation, classification and object detection models](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html#pre-trained-models).
This example uses object detection with the [Detectnet V2 model](https://docs.nvidia.com/tao/tao-toolkit-archive/5.2.0/text/object_detection/detectnet_v2.html)
as a use case.

To get started with TAO, follow the [set-up instruction video](https://docs.nvidia.com/tao/tao-toolkit/text/quick_start_guide/index.html).

TAO uses Jupyter notebooks to guide you through the training process.
In the folder cv\_samples\_v1.3.0, you can find notebooks for multiple models.
You can use any of the object detection networks for this use case, but this example uses Detectnet\_V2.

In the detectnet\_v2 folder, you can find the Jupyter notebook and the specs folder.
The [TAO Detectnet V2 documentation](https://docs.nvidia.com/tao/tao-toolkit-archive/5.2.0/text/object_detection/detectnet_v2.html)
goes into more detail about this sample. TAO works with configuration files that can be found in the
specs folder. Here, you must modify the specs to refer to the generated synthetic data as the
input.

To prepare the data, you must run the following command.

```python
tao detectnet_v2 dataset-convert [-h] -d DATASET_EXPORT_SPEC -o OUTPUT_FILENAME [-f VALIDATION_FOLD]
```

This is in the Jupyter notebook with a sample configuration. Modify the spec file to match the folder
structure of your synthetic data. The data is in TFrecord format and is ready for training.
Again, you need to change the spec file for training to represent the path to the synthetic data and
the classes being detected.

```python
tao detectnet_v2 train [-h] -k <key>
                        -r <result directory>
                        -e <spec_file>
                        [-n <name_string_for_the_model>]
                        [--gpus <num GPUs>]
                        [--gpu_index <comma separate gpu indices>]
                        [--use_amp]
                        [--log_file <log_file>]
```

For any questions regarding the TAO Toolkit, refer to the [TAO documentation](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html).

## Further Learning

To learn how to use NVIDIA Isaac Sim to create data sets in an interactive manner, see the
[Synthetic Data Recorder](Synthetic_Data_Generation.md) and then visualize them with the [Synthetic Data Visualizer](Synthetic_Data_Generation.md).

---

# Object Based Synthetic Dataset Generation

This document is an example of using Isaac Sim and [Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") to generate object-centric synthetic datasets. The script spawns labeled and distractor assets in a predefined area (closed off with invisible collision walls) and captures scenes from multiple camera viewpoints. The script also demonstrates how to randomize the camera poses, apply random velocities to the objects, and trigger custom events to randomize the scene. The randomizers can be Replicator-based or custom Isaac Sim/USD API based and can be triggered at specific times.

## Learning Objectives

The goal of this tutorial is to demonstrate how to use Isaac Sim and [replicator randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)") in a hybrid way in simulated environments. The tutorial covers the following topics:

* How to create a custom USD stage and add [rigid-body](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html "(in Omni Physics)") enabled assets with colliders.

  > + How to spawn and add colliders and rigid body dynamics to assets.
  > + How to create a collision box area around the assets to prevent them from drifting away.
  > + How to add a physics scene and set custom physics settings.
* How to create custom randomizers and trigger them at specific times.

  > + How to randomize the camera poses to look at a random target asset.
  > + How to randomize the shape distractor colors and apply random velocities to the floating shape distractors.
  > + How to randomize the lights in the working area and the dome background.
* How to capture motion blur by combining the number of pathtraced subframes samples simulated for the given duration.

  > + How to enable motion blur and set the number of sub samples to render for motion blur in PathTracing mode.
  > + How to set the render mode to PathTracing.
* How to create a custom synthetic dataset generation pipeline.
* Performance optimization by enabling rendering and data processing only for the frames to be captured.
* Use custom writers to export the data.

## Prerequisites

* Familiarity with USD / Isaac Sim APIs for creating and manipulating USD stages.
* Familiarity with [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)"), its [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)"), and [randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)").
* Basic understanding of [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") for the Replicator randomization and trigger pipeline.
* Familiarity with [rigid-body](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html "(in Omni Physics)") dynamics and physics simulation in Isaac Sim.
* Running simulations as [Standalone Applications](Workflows.md) or via the [Script Editor](Development_Tools.md).

## Getting Started

The main script of the tutorial is located at `<install_path>/standalone_examples/replicator/object_based_sdg/object_based_sdg.py` with its util functions at `<install_path>/standalone_examples/replicator/object_based_sdg/object_based_sdg_utils.py`.

* The script can be run as a standalone application (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/replicator/object_based_sdg/object_based_sdg.py
```

To overwrite the default configuration parameters, you can provide custom config files as a command-line argument for the script by using `--config <path/to/file.json/yaml>`. Example config files are stored in `<install_path>/standalone_examples/replicator/object_based_sdg/config/*`.

* Example of running the script with a custom config file:

```python
./python.sh standalone_examples/replicator/object_based_sdg/object_based_sdg.py \
    --config standalone_examples/replicator/object_based_sdg/config/<example_config>.yaml
```

## Implementation

The following section provides an implementation overview of the script. It includes details regarding the configuration parameters, scene generation helper functions, randomizations (Isaac Sim and Replicator), and data capture loop.

The complete implementation consists of two files: the main script and a utilities module.

Script Editor

Utils module and main script

```python
import asyncio
import os
import random
import time
from itertools import chain

import carb.settings
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import get_active_viewport
from omni.physx import get_physx_interface, get_physx_scene_query_interface
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

def add_colliders(root_prim: Usd.Prim) -> None:
    """Enable collisions on the asset (without rigid body dynamics the asset will be static)."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

def create_collision_box_walls(
    stage: Usd.Stage,
    path: str,
    width: float,
    depth: float,
    height: float,
    thickness: float = 0.5,
    visible: bool = False,
) -> None:
    """Create a collision box area wrapping the given working area with origin at (0, 0, 0)."""
    walls = [
        ("floor", (0, 0, (height + thickness) / -2.0), (width, depth, thickness)),
        ("ceiling", (0, 0, (height + thickness) / 2.0), (width, depth, thickness)),
        ("left_wall", ((width + thickness) / -2.0, 0, 0), (thickness, depth, height)),
        ("right_wall", ((width + thickness) / 2.0, 0, 0), (thickness, depth, height)),
        ("front_wall", (0, (depth + thickness) / 2.0, 0), (width, thickness, height)),
        ("back_wall", (0, (depth + thickness) / -2.0, 0), (width, thickness, height)),
    ]
    for name, location, size in walls:
        prim = stage.DefinePrim(f"{path}/{name}", "Cube")
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        rep.functional.modify.pose(prim, position_value=location, scale_value=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()

def get_random_transform_values(
    loc_min=(0, 0, 0), loc_max=(1, 1, 1), rot_min=(0, 0, 0), rot_max=(360, 360, 360), scale_min_max=(0.1, 1.0)
):
    """Create random transformation values for location, rotation, and scale."""
    location = (
        random.uniform(loc_min[0], loc_max[0]),
        random.uniform(loc_min[1], loc_max[1]),
        random.uniform(loc_min[2], loc_max[2]),
    )
    rotation = (
        random.uniform(rot_min[0], rot_max[0]),
        random.uniform(rot_min[1], rot_max[1]),
        random.uniform(rot_min[2], rot_max[2]),
    )
    scale = tuple([random.uniform(scale_min_max[0], scale_min_max[1])] * 3)
    return location, rotation, scale

def get_random_pose_on_sphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    """Generate a random pose on a sphere looking at the origin."""
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = radius * np.sin(theta) * np.cos(phi)
    location = origin + Gf.Vec3f(x, y, z)
    direction = origin - location
    direction_normalized = direction.GetNormalized()
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    orientation = Gf.Quatf(rotation.GetQuat())
    return location, orientation

def set_render_products_updates(render_products, enabled, include_viewport=False):
    """Enable or disable the render products and viewport rendering."""
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled

def apply_velocities_towards_target(prims, target=(0, 0, 0), strength_range=(0.1, 1.0)):
    """Apply velocities to prims directing them towards a target point."""
    for prim in prims:
        loc = prim.GetAttribute("xformOp:translate").Get()
        strength = random.uniform(strength_range[0], strength_range[1])
        velocity = (
            (target[0] - loc[0]) * strength,
            (target[1] - loc[1]) * strength,
            (target[2] - loc[2]) * strength,
        )
        prim.GetAttribute("physics:velocity").Set(velocity)

def apply_random_velocities(prims, linear_range=(-2.5, 2.5), angular_range=(-45, 45)):
    """Apply random linear and angular velocities to prims."""
    for prim in prims:
        lin_vel = (
            random.uniform(linear_range[0], linear_range[1]),
            random.uniform(linear_range[0], linear_range[1]),
            random.uniform(linear_range[0], linear_range[1]),
        )
        ang_vel = (
            random.uniform(angular_range[0], angular_range[1]),
            random.uniform(angular_range[0], angular_range[1]),
            random.uniform(angular_range[0], angular_range[1]),
        )
        prim.GetAttribute("physics:velocity").Set(lin_vel)
        prim.GetAttribute("physics:angularVelocity").Set(ang_vel)

async def run_example_async(config: dict) -> None:
    """Run the object-based SDG example asynchronously."""
    assets_root_path = get_assets_root_path()
    stage = None

    # ENVIRONMENT
    env_url = config.get("env_url", "")
    if env_url:
        env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
        omni.usd.get_context().open_stage(env_path)
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            upgrade_prim_semantics_to_labels(prim, include_descendants=True)
            remove_labels(prim, include_descendants=True)
    else:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        rep.functional.create.xform(name="World")
        rep.functional.create.distant_light(intensity=400.0, rotation=(0, 60, 0), name="DistantLight")

    working_area_size = config.get("working_area_size", (3, 3, 3))
    working_area_min = (working_area_size[0] / -2, working_area_size[1] / -2, working_area_size[2] / -2)
    working_area_max = (working_area_size[0] / 2, working_area_size[1] / 2, working_area_size[2] / 2)

    create_collision_box_walls(
        stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
    )

    rep.functional.physics.create_physics_scene("/PhysicsScene", timeStepsPerSecond=60)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

    # TRAINING ASSETS
    labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
    floating_labeled_prims = []
    falling_labeled_prims = []
    labeled_prims = []
    rep.functional.create.scope(name="Labeled", parent="/World")
    for obj in labeled_assets_and_properties:
        obj_url = obj.get("url", "")
        label = obj.get("label", "unknown")
        count = obj.get("count", 1)
        floating = obj.get("floating", False)
        scale_min_max = obj.get("randomize_scale", (1, 1))
        for i in range(count):
            rand_loc, rand_rot, rand_scale = get_random_transform_values(
                loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
            )
            asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
            prim = rep.functional.create.reference(
                usd_path=asset_path,
                parent="/World/Labeled",
                name=label,
                position=rand_loc,
                rotation=rand_rot,
                scale=rand_scale,
            )
            add_colliders(prim)
            rep.functional.physics.apply_rigid_body(prim, disableGravity=floating)
            add_labels(prim, labels=[label], instance_name="class")
            if floating:
                floating_labeled_prims.append(prim)
            else:
                falling_labeled_prims.append(prim)
    labeled_prims = floating_labeled_prims + falling_labeled_prims

    # DISTRACTORS
    shape_distractors_types = config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
    shape_distractors_scale_min_max = config.get("shape_distractors_scale_min_max", (0.02, 0.2))
    shape_distractors_num = config.get("shape_distractors_num", 350)
    shape_distractors = []
    floating_shape_distractors = []
    falling_shape_distractors = []
    for i in range(shape_distractors_num):
        rand_loc, rand_rot, rand_scale = get_random_transform_values(
            loc_min=working_area_min, loc_max=working_area_max, scale_min_max=shape_distractors_scale_min_max
        )
        rand_shape = random.choice(shape_distractors_types)
        prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{rand_shape}", False)
        prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
        rep.functional.modify.pose(prim, position_value=rand_loc, rotation_value=rand_rot, scale_value=rand_scale)
        disable_gravity = random.choice([True, False])
        add_colliders(prim)
        rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
        if disable_gravity:
            floating_shape_distractors.append(prim)
        else:
            falling_shape_distractors.append(prim)
        shape_distractors.append(prim)

    mesh_distactors_urls = config.get("mesh_distractors_urls", [])
    mesh_distactors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
    mesh_distactors_num = config.get("mesh_distractors_num", 10)
    mesh_distractors = []
    floating_mesh_distractors = []
    falling_mesh_distractors = []
    for i in range(mesh_distactors_num):
        rand_loc, rand_rot, rand_scale = get_random_transform_values(
            loc_min=working_area_min, loc_max=working_area_max, scale_min_max=mesh_distactors_scale_min_max
        )
        mesh_url = random.choice(mesh_distactors_urls)
        prim_name = os.path.basename(mesh_url).split(".")[0]
        asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
        prim = rep.functional.create.reference(
            usd_path=asset_path,
            parent="/World/Distractors",
            name=prim_name,
            position=rand_loc,
            rotation=rand_rot,
            scale=rand_scale,
        )
        disable_gravity = random.choice([True, False])
        add_colliders(prim)
        rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
        if disable_gravity:
            floating_mesh_distractors.append(prim)
        else:
            falling_mesh_distractors.append(prim)
        mesh_distractors.append(prim)
        upgrade_prim_semantics_to_labels(prim, include_descendants=True)
        remove_labels(prim, include_descendants=True)

    # REPLICATOR
    rep.set_global_seed(42)
    random.seed(42)
    rep.orchestrator.set_capture_on_play(False)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    cameras = []
    num_cameras = config.get("num_cameras", 1)
    camera_properties_kwargs = config.get("camera_properties_kwargs", {})
    rep.functional.create.scope(name="Cameras", parent="/World")
    for i in range(num_cameras):
        cam_prim = rep.functional.create.camera(parent="/World/Cameras", name="cam", **camera_properties_kwargs)
        cameras.append(cam_prim)

    camera_colliders = []
    camera_collider_radius = config.get("camera_collider_radius", 0)
    if camera_collider_radius > 0:
        for cam in cameras:
            cam_path = cam.GetPath()
            cam_collider = stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
            cam_collider.GetAttribute("radius").Set(camera_collider_radius)
            rep.functional.physics.apply_collider(cam_collider)
            collision_api = UsdPhysics.CollisionAPI(cam_collider)
            collision_api.GetCollisionEnabledAttr().Set(False)
            UsdGeom.Imageable(cam_collider).MakeInvisible()
            camera_colliders.append(cam_collider)

    await omni.kit.app.get_app().next_update_async()

    render_products = []
    resolution = config.get("resolution", (640, 480))
    for cam in cameras:
        rp = rep.create.render_product(cam.GetPath(), resolution)
        render_products.append(rp)

    disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
    if disable_render_products_between_captures:
        set_render_products_updates(render_products, False, include_viewport=False)

    writer_type = config.get("writer_type", None)
    writer_kwargs = config.get("writer_kwargs", {})
    if out_dir := writer_kwargs.get("output_dir"):
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.getcwd(), out_dir)
            writer_kwargs["output_dir"] = out_dir
        print(f"[SDG] Writing data to: {out_dir}")
    if writer_type is not None and len(render_products) > 0:
        writer = rep.writers.get(writer_type)
        writer.initialize(**writer_kwargs)
        writer.attach(render_products)

    # RANDOMIZERS
    def on_overlap_hit(hit) -> bool:
        prim = stage.GetPrimAtPath(hit.rigid_body)
        if prim not in camera_colliders:
            rand_vel = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 8))
            prim.GetAttribute("physics:velocity").Set(rand_vel)
        return True

    overlap_area_thickness = 0.1
    overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
    overlap_area_extent = (
        working_area_size[0] / 2 * 0.99,
        working_area_size[1] / 2 * 0.99,
        overlap_area_thickness / 2 * 0.99,
    )

    def on_physics_step(dt: float) -> None:
        get_physx_scene_query_interface().overlap_box(
            carb.Float3(overlap_area_extent),
            carb.Float3(overlap_area_origin),
            carb.Float4(0, 0, 0, 1),
            on_overlap_hit,
            False,
        )

    physx_sub = get_physx_interface().subscribe_physics_step_events(on_physics_step)

    camera_distance_to_target_min_max = config.get("camera_distance_to_target_min_max", (0.1, 0.5))
    camera_look_at_target_offset = config.get("camera_look_at_target_offset", 0.2)

    def randomize_camera_poses() -> None:
        for cam in cameras:
            target_asset = random.choice(labeled_prims)
            loc_offset = (
                random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
                random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
                random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            )
            target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
            distance = random.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
            cam_loc, quat = get_random_pose_on_sphere(origin=target_loc, radius=distance)
            rep.functional.modify.pose(cam, position_value=cam_loc, rotation_value=quat)

    async def simulate_camera_collision_async(num_frames: int = 1) -> None:
        for cam_collider in camera_colliders:
            collision_api = UsdPhysics.CollisionAPI(cam_collider)
            collision_api.GetCollisionEnabledAttr().Set(True)
        if not timeline.is_playing():
            timeline.play()
        for _ in range(num_frames):
            await omni.kit.app.get_app().next_update_async()
        for cam_collider in camera_colliders:
            collision_api = UsdPhysics.CollisionAPI(cam_collider)
            collision_api.GetCollisionEnabledAttr().Set(False)

    with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
        shape_distractors_paths = [
            prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)
        ]
        shape_distractors_group = rep.create.group(shape_distractors_paths)
        with shape_distractors_group:
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

    with rep.trigger.on_custom_event(event_name="randomize_lights"):
        lights = rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
            temperature=rep.distribution.normal(6500, 500),
            intensity=rep.distribution.normal(35000, 5000),
            position=rep.distribution.uniform(working_area_min, working_area_max),
            scale=rep.distribution.uniform(0.1, 1),
            count=3,
        )

    with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
        dome_textures = [
            assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
            assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
        ]
        dome_light = rep.create.light(light_type="Dome")
        with dome_light:
            rep.modify.attribute("inputs:texture:file", rep.distribution.choice(dome_textures))
            rep.randomizer.rotation()

    async def capture_with_motion_blur_and_pathtracing_async(
        duration: float = 0.05, num_samples: int = 8, spp: int = 64
    ) -> None:
        orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
        target_physics_fps = 1 / duration * num_samples
        if target_physics_fps > orig_physics_fps:
            physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)
        is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
        if not is_motion_blur_enabled:
            carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
        carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)
        prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
        carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
        carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
        carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
        carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)
        if not timeline.is_playing():
            timeline.play()
        await rep.orchestrator.step_async(delta_time=duration, pause_timeline=False)
        if target_physics_fps > orig_physics_fps:
            physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
        carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)

    async def run_simulation_loop_async(duration: float) -> None:
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()
        elapsed_time = 0.0
        previous_time = timeline.get_current_time()
        while elapsed_time <= duration:
            await omni.kit.app.get_app().next_update_async()
            elapsed_time += timeline.get_current_time() - previous_time
            previous_time = timeline.get_current_time()

    # SDG
    num_frames = config.get("num_frames", 10)
    rt_subframes = config.get("rt_subframes", -1)
    sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.025)

    rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
    rep.utils.send_og_event(event_name="randomize_dome_background")
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    timeline = omni.timeline.get_timeline_interface()
    timeline.set_start_time(0)
    timeline.set_end_time(1000000)
    timeline.set_looping(False)
    timeline.play()
    timeline.commit()
    await omni.kit.app.get_app().next_update_async()

    wall_time_start = time.perf_counter()

    for i in range(num_frames):
        if i % 3 == 0:
            randomize_camera_poses()
            if camera_colliders:
                await simulate_camera_collision_async(num_frames=4)
        if i % 10 == 0:
            apply_velocities_towards_target(list(chain(labeled_prims, shape_distractors, mesh_distractors)))
        if i % 5 == 0:
            rep.utils.send_og_event(event_name="randomize_lights")
        if i % 15 == 0:
            rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
        if i % 25 == 0:
            rep.utils.send_og_event(event_name="randomize_dome_background")
        if i % 17 == 0:
            apply_random_velocities(list(chain(floating_shape_distractors, floating_mesh_distractors)))

        if disable_render_products_between_captures:
            set_render_products_updates(render_products, True, include_viewport=False)

        print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
        if i % 5 == 0:
            await capture_with_motion_blur_and_pathtracing_async(duration=0.025, num_samples=8, spp=128)
        else:
            await rep.orchestrator.step_async(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

        if disable_render_products_between_captures:
            set_render_products_updates(render_products, False, include_viewport=False)

        if sim_duration_between_captures > 0:
            await run_simulation_loop_async(sim_duration_between_captures)
        else:
            await omni.kit.app.get_app().next_update_async()

    await rep.orchestrator.wait_until_complete_async()

    wall_duration = time.perf_counter() - wall_time_start
    sim_duration = timeline.get_current_time()
    num_captures = num_frames * num_cameras
    print(
        f"[SDG] Captured {num_frames} frames, {num_captures} entries in {wall_duration:.2f} seconds.\n"
        f"\t Simulation duration: {sim_duration:.2f}\n"
    )

    physx_sub.unsubscribe()
    physx_sub = None
    await omni.kit.app.get_app().next_update_async()
    timeline.stop()

config = {
    "env_url": "",
    "working_area_size": (5, 5, 3),
    "rt_subframes": 4,
    "num_frames": 10,
    "num_cameras": 2,
    "camera_collider_radius": 1.25,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 0.05,
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focal_length": 24.0,
        "focus_distance": 400,
        "f_stop": 0.0,
        "clipping_range": (0.01, 10000),
    },
    "camera_look_at_target_offset": 0.15,
    "camera_distance_to_target_min_max": (0.25, 0.75),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": "_out_obj_based_sdg_pose_writer",
        "format": None,
        "use_subfolders": False,
        "write_debug_images": True,
        "skip_empty_frames": False,
    },
    "labeled_assets_and_properties": [
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "label": "pudding_box",
            "count": 5,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            "label": "mustard_bottle",
            "count": 7,
            "floating": False,
            "scale_min_max": (0.85, 3.25),
        },
    ],
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 150,
    "mesh_distractors_urls": [
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.35, 1.35),
    "mesh_distractors_num": 75,
}

asyncio.ensure_future(run_example_async(config))
```

Standalone Application

Main script

```python
import argparse
import json
import os

import yaml
from isaacsim import SimulationApp

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "env_url": "",
    "working_area_size": (4, 4, 3),
    "rt_subframes": 4,
    "num_frames": 4,
    "num_cameras": 2,
    "camera_collider_radius": 0.5,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 0.05,
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focal_length": 24.0,
        "focus_distance": 400,
        "f_stop": 0.0,
        "clipping_range": (0.01, 10000),
    },
    "camera_look_at_target_offset": 0.15,
    "camera_distance_to_target_min_max": (0.25, 0.75),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": "_out_obj_based_sdg_pose_writer",
        "format": None,
        "use_subfolders": False,
        "write_debug_images": True,
        "skip_empty_frames": False,
    },
    "labeled_assets_and_properties": [
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "label": "pudding_box",
            "count": 5,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            "label": "mustard_bottle",
            "count": 7,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
    ],
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 350,
    "mesh_distractors_urls": [
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.35, 1.35),
    "mesh_distractors_num": 75,
}

import carb

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)

import random
import time
from itertools import chain

import carb.settings

# Custom util functions for the example
import object_based_sdg_utils
import omni.physics.core
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels, remove_labels, upgrade_prim_semantics_to_labels
from isaacsim.storage.native import get_assets_root_path
from omni.physics.core import get_physics_scene_query_interface
from pxr import PhysicsSchemaTools, PhysxSchema, UsdGeom, UsdPhysics

# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()
stage = None

# ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()
    # Remove any previous semantics in the loaded stage
    for prim in stage.Traverse():
        # Make sure old semantics api are upgraded to the new labels api
        upgrade_prim_semantics_to_labels(prim, include_descendants=True)
        remove_labels(prim, include_descendants=True)
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    rep.functional.create.xform(name="World")
    rep.functional.create.distant_light(intensity=400.0, rotation=(0, 60, 0), name="DistantLight")

# Get the working area size and bounds (width=x, depth=y, height=z)
working_area_size = config.get("working_area_size", (3, 3, 3))
working_area_min = (working_area_size[0] / -2, working_area_size[1] / -2, working_area_size[2] / -2)
working_area_max = (working_area_size[0] / 2, working_area_size[1] / 2, working_area_size[2] / 2)

# Create a collision box area around the assets to prevent them from drifting away
object_based_sdg_utils.create_collision_box_walls(
    stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

rep.functional.physics.create_physics_scene("/PhysicsScene", timeStepsPerSecond=60)
physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

# TRAINING ASSETS
# Add the objects to be trained in the environment with their labels and properties
labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
floating_labeled_prims = []
falling_labeled_prims = []
labeled_prims = []
rep.functional.create.scope(name="Labeled", parent="/World")
for obj in labeled_assets_and_properties:
    obj_url = obj.get("url", "")
    label = obj.get("label", "unknown")
    count = obj.get("count", 1)
    floating = obj.get("floating", False)
    scale_min_max = obj.get("randomize_scale", (1, 1))
    for i in range(count):
        # Create a prim and add the asset reference
        rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
            loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
        )
        asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
        prim = rep.functional.create.reference(
            usd_path=asset_path,
            parent="/World/Labeled",
            name=label,
            position=rand_loc,
            rotation=rand_rot,
            scale=rand_scale,
        )
        # Apply colliders and rigid body dynamics
        object_based_sdg_utils.add_colliders(prim)
        rep.functional.physics.apply_rigid_body(prim, disableGravity=False)
        #  Label the asset (any previous 'class' label will be overwritten)
        add_labels(prim, labels=[label], instance_name="class")
        if floating:
            floating_labeled_prims.append(prim)
        else:
            falling_labeled_prims.append(prim)
labeled_prims = floating_labeled_prims + falling_labeled_prims

# DISTRACTORS
# Add shape distractors to the environment as floating or falling objects
shape_distractors_types = config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
shape_distractors_scale_min_max = config.get("shape_distractors_scale_min_max", (0.02, 0.2))
shape_distractors_num = config.get("shape_distractors_num", 350)
shape_distractors = []
floating_shape_distractors = []
falling_shape_distractors = []
for i in range(shape_distractors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        loc_min=working_area_min, loc_max=working_area_max, scale_min_max=shape_distractors_scale_min_max
    )
    rand_shape = random.choice(shape_distractors_types)
    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{rand_shape}", False)
    prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
    rep.functional.modify.pose(prim, position_value=rand_loc, rotation_value=rand_rot, scale_value=rand_scale)
    disable_gravity = random.choice([True, False])
    object_based_sdg_utils.add_colliders(prim)
    rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
    if disable_gravity:
        floating_shape_distractors.append(prim)
    else:
        falling_shape_distractors.append(prim)
    shape_distractors.append(prim)

# Add mesh distractors to the environment as floating of falling objects
mesh_distactors_urls = config.get("mesh_distractors_urls", [])
mesh_distactors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
mesh_distactors_num = config.get("mesh_distractors_num", 10)
mesh_distractors = []
floating_mesh_distractors = []
falling_mesh_distractors = []
for i in range(mesh_distactors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        loc_min=working_area_min, loc_max=working_area_max, scale_min_max=mesh_distactors_scale_min_max
    )
    mesh_url = random.choice(mesh_distactors_urls)
    prim_name = os.path.basename(mesh_url).split(".")[0]
    asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
    prim = rep.functional.create.reference(
        usd_path=asset_path,
        parent="/World/Distractors",
        name=prim_name,
        position=rand_loc,
        rotation=rand_rot,
        scale=rand_scale,
    )
    disable_gravity = random.choice([True, False])
    object_based_sdg_utils.add_colliders(prim)
    rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
    if disable_gravity:
        floating_mesh_distractors.append(prim)
    else:
        falling_mesh_distractors.append(prim)
    mesh_distractors.append(prim)
    # Remove any previous semantics on the mesh distractor
    upgrade_prim_semantics_to_labels(prim, include_descendants=True)
    remove_labels(prim, include_descendants=True)

# REPLICATOR
# Initialize randomization
rep.set_global_seed(42)
random.seed(42)

# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Create the camera prims and their properties
cameras = []
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
rep.functional.create.scope(name="Cameras", parent="/World")
for i in range(num_cameras):
    cam_prim = rep.functional.create.camera(parent="/World/Cameras", name="cam", **camera_properties_kwargs)
    cameras.append(cam_prim)

# Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
camera_colliders = []
camera_collider_radius = config.get("camera_collider_radius", 0)
if camera_collider_radius > 0:
    for cam in cameras:
        cam_path = cam.GetPath()
        cam_collider = stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
        cam_collider.GetAttribute("radius").Set(camera_collider_radius)
        rep.functional.physics.apply_collider(cam_collider)
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)
        UsdGeom.Imageable(cam_collider).MakeInvisible()
        camera_colliders.append(cam_collider)

# Wait an app update to ensure the prim changes are applied
simulation_app.update()

# Create render products using the cameras
render_products = []
resolution = config.get("resolution", (640, 480))
for cam in cameras:
    rp = rep.create.render_product(cam.GetPath(), resolution)
    render_products.append(rp)

# Enable rendering only at capture time
disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
if disable_render_products_between_captures:
    object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

# Create the writer and attach the render products
writer_type = config.get("writer_type", "PoseWriter")
writer_kwargs = config.get("writer_kwargs", {})
# If not an absolute path, set it relative to the current working directory
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")
if writer_type is not None and len(render_products) > 0:
    writer = rep.writers.get(writer_type)
    writer.initialize(**writer_kwargs)
    writer.attach(render_products)

# RANDOMIZERS
def on_overlap_hit(hit) -> bool:
    """Apply a random upwards velocity to objects overlapping the bounce area."""
    prim_path = str(PhysicsSchemaTools.intToSdfPath(hit.rigid_body))
    prim = stage.GetPrimAtPath(prim_path)
    # Skip the camera collision spheres
    if prim not in camera_colliders:
        rand_vel = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 8))
        prim.GetAttribute("physics:velocity").Set(rand_vel)
    return True  # return True to continue the query

# Area to check for overlapping objects (above the bottom collision box)
overlap_area_thickness = 0.1
overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
overlap_area_extent = (
    working_area_size[0] / 2 * 0.99,
    working_area_size[1] / 2 * 0.99,
    overlap_area_thickness / 2 * 0.99,
)

def on_physics_step(dt: float, context) -> None:
    """Check for overlapping objects on every physics update step."""
    get_physics_scene_query_interface().overlap_box(
        carb.Float3(overlap_area_extent),
        carb.Float3(overlap_area_origin),
        carb.Float4(0, 0, 0, 1),
        on_overlap_hit,
    )

# Subscribe to the physics step events to check for objects overlapping the 'bounce' area
physics_sub = omni.physics.core.get_physics_simulation_interface().subscribe_physics_on_step_events(
    pre_step=False, order=0, on_update=on_physics_step
)

camera_distance_to_target_min_max = config.get("camera_distance_to_target_min_max", (0.1, 0.5))
camera_look_at_target_offset = config.get("camera_look_at_target_offset", 0.2)

def randomize_camera_poses() -> None:
    """Randomize camera poses to look at a random target asset with random distance and offset."""
    for cam in cameras:
        target_asset = random.choice(labeled_prims)
        # Add a look_at offset so the target is not always in the center of the camera view
        loc_offset = (
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
        )
        target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
        distance = random.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
        cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(origin=target_loc, radius=distance)
        rep.functional.modify.pose(cam, position_value=cam_loc, rotation_value=quat)

def simulate_camera_collision(num_frames: int = 1) -> None:
    """Enable camera colliders temporarily and simulate to push out overlapping objects."""
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(True)
    if not timeline.is_playing():
        timeline.play()
    for _ in range(num_frames):
        simulation_app.update()
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)

# Create a randomizer for the shape distractors colors, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

# Create a randomizer for lights in the working area, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_lights"):
    lights = rep.create.light(
        light_type="Sphere",
        color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform(working_area_min, working_area_max),
        scale=rep.distribution.uniform(0.1, 1),
        count=3,
    )

# Create a randomizer for the dome background, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
    dome_textures = [
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
    ]
    dome_light = rep.create.light(light_type="Dome")
    with dome_light:
        rep.modify.attribute("inputs:texture:file", rep.distribution.choice(dome_textures))
        rep.randomizer.rotation()

def capture_with_motion_blur_and_pathtracing(
    physx_scene: PhysxSchema.PhysxSceneAPI, duration: float = 0.05, num_samples: int = 8, spp: int = 64
) -> None:
    """Capture motion blur by combining pathtraced subframe samples simulated for the given duration."""
    # For small step sizes the physics FPS needs to be temporarily increased to provide movements every sub sample
    orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    target_physics_fps = 1 / duration * num_samples
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Enable motion blur (if not enabled)
    is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
    if not is_motion_blur_enabled:
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    # Number of sub samples to render for motion blur in PathTracing mode
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the original physics FPS
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Restoring physics FPS from {target_physics_fps} to {orig_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

    # Restore the previous render and motion blur  settings
    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)

def run_simulation_loop(duration: float) -> None:
    """Update the app until a given simulation duration has passed."""
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not timeline.is_playing():
        timeline.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )

# SDG
# Number of frames to capture
num_frames = config.get("num_frames", 10)

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)

# Amount of simulation time to wait between captures
sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.025)

# Initial trigger for randomizers before the SDG loop with several app updates (ensures materials/textures are loaded)
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
rep.utils.send_og_event(event_name="randomize_dome_background")
for _ in range(5):
    simulation_app.update()

# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
timeline.play()
timeline.commit()
simulation_app.update()

# Store the wall start time for stats
wall_time_start = time.perf_counter()

# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
for i in range(num_frames):
    # Cameras will be moved to a random position and look at a randomly selected labeled asset
    if i % 3 == 0:
        print(f"\t Randomizing camera poses")
        randomize_camera_poses()
        # Temporarily enable camera colliders and simulate for a few frames to push out any overlapping objects
        if camera_colliders:
            simulate_camera_collision(num_frames=4)

    # Apply a random velocity towards the origin to the working area to pull the assets closer to the center
    if i % 10 == 0:
        print(f"\t Applying velocity towards the origin")
        object_based_sdg_utils.apply_velocities_towards_target(
            list(chain(labeled_prims, shape_distractors, mesh_distractors))
        )

    # Randomize lights locations and colors
    if i % 5 == 0:
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")

    # Randomize the colors of the primitive shape distractors
    if i % 15 == 0:
        print(f"\t Randomizing shape distractors colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    # Randomize the texture of the dome background
    if i % 25 == 0:
        print(f"\t Randomizing dome background")
        rep.utils.send_og_event(event_name="randomize_dome_background")

    # Apply a random velocity on the floating distractors (shapes and meshes)
    if i % 17 == 0:
        print(f"\t Randomizing shape distractors velocities")
        object_based_sdg_utils.apply_random_velocities(
            list(chain(floating_shape_distractors, floating_mesh_distractors))
        )

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if i % 5 == 0:
        capture_with_motion_blur_and_pathtracing(physx_scene, duration=0.025, num_samples=8, spp=128)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()

# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration
num_captures = num_frames * num_cameras
avg_capture_fps = num_captures / wall_duration
print(
    f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
    f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
)

# Unsubscribe the physics overlap checks and stop the timeline
physics_sub = None
simulation_app.update()
timeline.stop()

simulation_app.close()
```

Utils module

```python
import random

import numpy as np
import omni.replicator.core as rep
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

def add_colliders(root_prim: Usd.Prim) -> None:
    """Enable collisions on the asset (without rigid body dynamics the asset will be static)."""
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            # Set PhysX specific properties
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

def create_collision_box_walls(
    stage: Usd.Stage,
    path: str,
    width: float,
    depth: float,
    height: float,
    thickness: float = 0.5,
    visible: bool = False,
) -> None:
    """Create a collision box area wrapping the given working area with origin at (0, 0, 0)."""
    # Define the walls (name, location, size) with thickness towards outside of the working area
    walls = [
        ("floor", (0, 0, (height + thickness) / -2.0), (width, depth, thickness)),
        ("ceiling", (0, 0, (height + thickness) / 2.0), (width, depth, thickness)),
        ("left_wall", ((width + thickness) / -2.0, 0, 0), (thickness, depth, height)),
        ("right_wall", ((width + thickness) / 2.0, 0, 0), (thickness, depth, height)),
        ("front_wall", (0, (depth + thickness) / 2.0, 0), (width, thickness, height)),
        ("back_wall", (0, (depth + thickness) / -2.0, 0), (width, thickness, height)),
    ]
    for name, location, size in walls:
        prim = stage.DefinePrim(f"{path}/{name}", "Cube")
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        rep.functional.modify.pose(prim, position_value=location, scale_value=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()

def get_random_transform_values(
    loc_min: tuple[float, float, float] = (0, 0, 0),
    loc_max: tuple[float, float, float] = (1, 1, 1),
    rot_min: tuple[float, float, float] = (0, 0, 0),
    rot_max: tuple[float, float, float] = (360, 360, 360),
    scale_min_max: tuple[float, float] = (0.1, 1.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Create random transformation values for location, rotation, and scale."""
    location = (
        random.uniform(loc_min[0], loc_max[0]),
        random.uniform(loc_min[1], loc_max[1]),
        random.uniform(loc_min[2], loc_max[2]),
    )
    rotation = (
        random.uniform(rot_min[0], rot_max[0]),
        random.uniform(rot_min[1], rot_max[1]),
        random.uniform(rot_min[2], rot_max[2]),
    )
    scale = tuple([random.uniform(scale_min_max[0], scale_min_max[1])] * 3)
    return location, rotation, scale

def get_random_pose_on_sphere(
    origin: tuple[float, float, float],
    radius: float,
    camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
) -> tuple[Gf.Vec3f, Gf.Quatf]:
    """Generate a random pose on a sphere looking at the origin."""
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = radius * np.sin(theta) * np.cos(phi)

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

def set_render_products_updates(render_products: list, enabled: bool, include_viewport: bool = False) -> None:
    """Enable or disable the render products and viewport rendering."""
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled

def apply_velocities_towards_target(
    prims: list[Usd.Prim],
    target: tuple[float, float, float] = (0, 0, 0),
    strength_range: tuple[float, float] = (0.1, 1.0),
) -> None:
    """Apply velocities to prims directing them towards a target point."""
    for prim in prims:
        loc = prim.GetAttribute("xformOp:translate").Get()
        strength = random.uniform(strength_range[0], strength_range[1])
        velocity = ((target[0] - loc[0]) * strength, (target[1] - loc[1]) * strength, (target[2] - loc[2]) * strength)
        prim.GetAttribute("physics:velocity").Set(velocity)

def apply_random_velocities(
    prims: list[Usd.Prim],
    linear_range: tuple[float, float] = (-2.5, 2.5),
    angular_range: tuple[float, float] = (-45, 45),
) -> None:
    """Apply random linear and angular velocities to prims."""
    for prim in prims:
        lin_vel = (
            random.uniform(linear_range[0], linear_range[1]),
            random.uniform(linear_range[0], linear_range[1]),
            random.uniform(linear_range[0], linear_range[1]),
        )
        ang_vel = (
            random.uniform(angular_range[0], angular_range[1]),
            random.uniform(angular_range[0], angular_range[1]),
            random.uniform(angular_range[0], angular_range[1]),
        )
        prim.GetAttribute("physics:velocity").Set(lin_vel)
        prim.GetAttribute("physics:angularVelocity").Set(ang_vel)
```

## Config Scenarios

The script has the following main configuration parameters:

* **launch\_config** (dict): Configuration for the launch settings, such as the renderer and headless mode.
* **env\_url** (str): The URL of the environment to load, if empty a new empty stage is created.
* **working\_area\_size** (tuple): The size of the area (width, depth, height) in which the objects will be placed, this area will be surrounded by invisible collision walls to prevent objects from drifting away.
* **num\_frames** (int): The number of frames to capture (the total number of entries will be num\_frames \* num\_cameras).
* **num\_cameras** (int): The number of cameras to use for capturing the frames, these will be randomized and moved to look at different targets.
* **disable\_render\_products\_between\_captures** (bool): If True, the render products will be disabled between captures to save resources.
* **simulation\_duration\_between\_captures** (float): The amount of simulation time to run between data captures.
* **camera\_properties\_kwargs** (dict): The camera properties to set for the cameras (focal length, focus distance, f-stop, clipping range).
* **writer\_type** (str): The writer type to use to write the data to disk. For example, PoseWriter or BasicWriter.
* **writer\_kwargs** (dict): The writer parameters to use when initializing the writer. For example, output\_dir, format, use\_subfolders.
* **labeled\_assets\_and\_properties** (list): A list of dictionaries with the labeled assets to add to the environment with their properties.
* **shape\_distractors\_types** (list): A list of shape types to use for the distractors (capsule, cone, cylinder, sphere, cube).
* **shape\_distractors\_num** (int): The number of shape distractors to add to the environment.
* **mesh\_distractors\_urls** (list): A list of mesh URLs to use for the distractors. For example, `/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd` or `omniverse://...`.
* **mesh\_distractors\_num** (int): The number of mesh distractors to add to the environment.

The following provides details about the various config scenarios:

Built-in

Without an explicit config file, the script uses the default parameters stored in the script itself. The default parameters are the following:

Built-in (default) Config

```python
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "env_url": "",
    "working_area_size": (4, 4, 3),
    "rt_subframes": 4,
    "num_frames": 4,
    "num_cameras": 2,
    "camera_collider_radius": 0.5,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 0.05,
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focal_length": 24.0,
        "focus_distance": 400,
        "f_stop": 0.0,
        "clipping_range": (0.01, 10000),
    },
    "camera_look_at_target_offset": 0.15,
    "camera_distance_to_target_min_max": (0.25, 0.75),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": "_out_obj_based_sdg_pose_writer",
        "format": None,
        "use_subfolders": False,
        "write_debug_images": True,
        "skip_empty_frames": False,
    },
    "labeled_assets_and_properties": [
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "label": "pudding_box",
            "count": 5,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            "label": "mustard_bottle",
            "count": 7,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
    ],
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 350,
    "mesh_distractors_urls": [
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.35, 1.35),
    "mesh_distractors_num": 75,
}
```

The following command runs the script with the default parameters:

```python
./python.sh standalone_examples/replicator/object_based_sdg/object_based_sdg.py
```

Basic Writer

The `object_based_sdg_config.yaml` config file uses `BasicWriter` with extended labeled assets and mesh distractors configurations.

Custom YAML Config using BasicWriter

```python
launch_config:
  renderer: RealTimePathTracing
  headless: false
env_url: ''
working_area_size:
- 4
- 4
- 3
rt_subframes: 4
num_frames: 10
num_cameras: 2
disable_render_products_between_captures: false
simulation_duration_between_captures: 0.0
resolution:
- 640
- 480
camera_look_at_target_offset: 0.15
camera_distance_to_target_min_max:
  - 0.25
  - 0.75
writer_type: BasicWriter
writer_kwargs:
  output_dir: _out_obj_based_sdg_basic_writer
  rgb: true
  semantic_segmentation: true
  use_common_output_dir: true
labeled_assets_and_properties:
- url: /Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd
  label: pudding_box
  count: 5
  floating: true
  scale_min_max:
    - 0.85
    - 1.25
- url: /Isaac/Props/YCB/Axis_Aligned/011_banana.usd
  label: banana
  count: 10
  floating: false
  scale_min_max:
    - 0.85
    - 1.25
- url: /Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd
  label: mustard_bottle
  count: 7
  floating: true
  scale_min_max:
    - 0.85
    - 1.25
shape_distractors_types:
- capsule
- cone
- cylinder
- sphere
- cube
shape_distractors_scale_min_max:
  - 0.015
  - 0.15
shape_distractors_num: 350
mesh_distractors_urls:
- /Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd
- /Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd
- /Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd
- /Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd
- /Isaac/Environments/Simple_Warehouse/Props/SM_BarelPlastic_B_03.usd
- /Isaac/Environments/Office/Props/SM_Board.usd
- /Isaac/Environments/Office/Props/SM_Book_03.usd
- /Isaac/Environments/Office/Props/SM_Book_34.usd
- /Isaac/Environments/Office/Props/SM_BookOpen_01.usd
- /Isaac/Environments/Office/Props/SM_Briefcase.usd
- /Isaac/Environments/Office/Props/SM_Extinguisher.usd
- /Isaac/Environments/Hospital/Props/SM_GasCart_01b.usd
- /Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd
- /Isaac/Environments/Hospital/Props/SM_MedicalBox_01g.usd
- /Isaac/Environments/Hospital/Props/SM_Toweldispenser_01a.usd
mesh_distractors_scale_min_max:
  - 0.35
  - 1.35
mesh_distractors_num: 75
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/object_based_sdg/object_based_sdg.py \
    --config standalone_examples/replicator/object_based_sdg/config/object_based_sdg_config.yaml
```

PoseWriter (DOPE)

The `object_based_sdg_dope_config.yaml` config file uses `PoseWriter` with DOPE format output for training DOPE networks.

Custom YAML Config using PoseWriter with DOPE format

```python
writer_type: PoseWriter
writer_kwargs:
  output_dir: _out_obj_based_sdg_pose_writer_dope
  format: dope
  write_debug_images: true
  skip_empty_frames: false
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/object_based_sdg/object_based_sdg.py \
    --config standalone_examples/replicator/object_based_sdg/config/object_based_sdg_dope_config.yaml
```

PoseWriter (CenterPose)

The `object_based_sdg_centerpose_config.yaml` config file uses `PoseWriter` with CenterPose format output for training CenterPose networks.

Custom YAML Config using PoseWriter with CenterPose format

```python
writer_type: PoseWriter
writer_kwargs:
  output_dir: _out_obj_based_sdg_pose_writer_centerpose
  format: centerpose
  write_debug_images: true
  skip_empty_frames: false
```

The following command runs the script with the custom parameters:

```python
./python.sh standalone_examples/replicator/object_based_sdg/object_based_sdg.py \
    --config standalone_examples/replicator/object_based_sdg/config/object_based_sdg_centerpose_config.yaml
```

### Util Functions

The script uses the `rep.functional` API directly for common operations such as setting transforms (`rep.functional.modify.pose`), creating assets (`rep.functional.create.reference`, `rep.functional.create.camera`), and applying physics properties (`rep.functional.physics.apply_rigid_body`, `rep.functional.physics.apply_collider`). Additional helper functions are provided in a separate utils module for custom operations.

Replicator Functional API for Transforms

The `rep.functional.modify.pose` function is used to set position, rotation, and scale on prims. This replaces the need for custom transform helper functions.

```python
def get_random_transform_values(
    loc_min: tuple[float, float, float] = (0, 0, 0),
    loc_max: tuple[float, float, float] = (1, 1, 1),
    rot_min: tuple[float, float, float] = (0, 0, 0),
    rot_max: tuple[float, float, float] = (360, 360, 360),
    scale_min_max: tuple[float, float] = (0.1, 1.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Create random transformation values for location, rotation, and scale."""
    location = (
        random.uniform(loc_min[0], loc_max[0]),
        random.uniform(loc_min[1], loc_max[1]),
        random.uniform(loc_min[2], loc_max[2]),
    )
    rotation = (
        random.uniform(rot_min[0], rot_max[0]),
        random.uniform(rot_min[1], rot_max[1]),
        random.uniform(rot_min[2], rot_max[2]),
    )
    scale = tuple([random.uniform(scale_min_max[0], scale_min_max[1])] * 3)
    return location, rotation, scale
```

Example usage for creating and positioning shape distractors:

```python
falling_shape_distractors = []
for i in range(shape_distractors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        loc_min=working_area_min, loc_max=working_area_max, scale_min_max=shape_distractors_scale_min_max
    )
    rand_shape = random.choice(shape_distractors_types)
    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Distractors/{rand_shape}", False)
    prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
    rep.functional.modify.pose(prim, position_value=rand_loc, rotation_value=rand_rot, scale_value=rand_scale)
    disable_gravity = random.choice([True, False])
    object_based_sdg_utils.add_colliders(prim)
    rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
    if disable_gravity:
        floating_shape_distractors.append(prim)
    else:
        falling_shape_distractors.append(prim)
    shape_distractors.append(prim)
```

Generate 3D Transform Values

The following functions are used to generate random 3D transform values for various scenarios.

```python
def get_random_transform_values(
    loc_min: tuple[float, float, float] = (0, 0, 0),
    loc_max: tuple[float, float, float] = (1, 1, 1),
    rot_min: tuple[float, float, float] = (0, 0, 0),
    rot_max: tuple[float, float, float] = (360, 360, 360),
    scale_min_max: tuple[float, float] = (0.1, 1.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Create random transformation values for location, rotation, and scale."""
    location = (
        random.uniform(loc_min[0], loc_max[0]),
        random.uniform(loc_min[1], loc_max[1]),
        random.uniform(loc_min[2], loc_max[2]),
    )
    rotation = (
        random.uniform(rot_min[0], rot_max[0]),
        random.uniform(rot_min[1], rot_max[1]),
        random.uniform(rot_min[2], rot_max[2]),
    )
    scale = tuple([random.uniform(scale_min_max[0], scale_min_max[1])] * 3)
    return location, rotation, scale
```

Example of generating a random pose on a sphere looking at the origin:

```python
def get_random_pose_on_sphere(
    origin: tuple[float, float, float],
    radius: float,
    camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
) -> tuple[Gf.Vec3f, Gf.Quatf]:
    """Generate a random pose on a sphere looking at the origin."""
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = radius * np.sin(theta) * np.cos(phi)

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation
```

Rigid-body Dynamics

Physics properties are applied using the `rep.functional.physics` API. The `apply_rigid_body` function adds rigid body dynamics, while `apply_collider` adds collision properties to prims. For custom collision settings (such as mesh approximation types), a helper function is still used.

```python
def add_colliders(root_prim: Usd.Prim) -> None:
    """Enable collisions on the asset (without rigid body dynamics the asset will be static)."""
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            # Set PhysX specific properties
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")
```

Example usage for creating labeled assets with colliders and rigid body:

```python
scale_min_max = obj.get("randomize_scale", (1, 1))
for i in range(count):
    # Create a prim and add the asset reference
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
    )
    asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
    prim = rep.functional.create.reference(
        usd_path=asset_path,
        parent="/World/Labeled",
        name=label,
        position=rand_loc,
        rotation=rand_rot,
        scale=rand_scale,
    )
    # Apply colliders and rigid body dynamics
    object_based_sdg_utils.add_colliders(prim)
    rep.functional.physics.apply_rigid_body(prim, disableGravity=False)
    #  Label the asset (any previous 'class' label will be overwritten)
    add_labels(prim, labels=[label], instance_name="class")
    if floating:
        floating_labeled_prims.append(prim)
    else:
        falling_labeled_prims.append(prim)
```

### Randomizers

The following snippets show the various randomizations used throughout the script.

* **|isaac-sim\_short|/USD based:** bounce randomizer, randomizing camera poses, applying custom velocities to assets
* **Replicator based:** randomizing lights, shape distractors colors, dome background, and floating distractors velocities

Overlap Triggered Velocity Randomizer

The following snippet simulates a bouncing area above the bottom collision box. The function checks for overlapping objects in the area and applies a random velocity to the objects. The function is triggered every physics update step to check for objects overlapping the ‘bounce’ area.

```python
# RANDOMIZERS
def on_overlap_hit(hit) -> bool:
    """Apply a random upwards velocity to objects overlapping the bounce area."""
    prim_path = str(PhysicsSchemaTools.intToSdfPath(hit.rigid_body))
    prim = stage.GetPrimAtPath(prim_path)
    # Skip the camera collision spheres
    if prim not in camera_colliders:
        rand_vel = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 8))
        prim.GetAttribute("physics:velocity").Set(rand_vel)
    return True  # return True to continue the query

# Area to check for overlapping objects (above the bottom collision box)
overlap_area_thickness = 0.1
overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
overlap_area_extent = (
    working_area_size[0] / 2 * 0.99,
    working_area_size[1] / 2 * 0.99,
    overlap_area_thickness / 2 * 0.99,
)

def on_physics_step(dt: float, context) -> None:
    """Check for overlapping objects on every physics update step."""
    get_physics_scene_query_interface().overlap_box(
        carb.Float3(overlap_area_extent),
        carb.Float3(overlap_area_origin),
        carb.Float4(0, 0, 0, 1),
        on_overlap_hit,
    )

# Subscribe to the physics step events to check for objects overlapping the 'bounce' area
physics_sub = omni.physics.core.get_physics_simulation_interface().subscribe_physics_on_step_events(
    pre_step=False, order=0, on_update=on_physics_step
)
```

Camera Randomization

The camera randomization function uses the `rep.functional` API along with Isaac Sim/USD API to look at a randomly chosen labeled asset from a randomized distance together with an offset to avoid always looking at the center of the asset. Cameras are created using `rep.functional.create.camera` and positioned using `rep.functional.modify.pose`. If camera colliders are enabled, the function will temporarily enable them and simulate for a few frames to push out any overlapping objects.

```python
def randomize_camera_poses() -> None:
    """Randomize camera poses to look at a random target asset with random distance and offset."""
    for cam in cameras:
        target_asset = random.choice(labeled_prims)
        # Add a look_at offset so the target is not always in the center of the camera view
        loc_offset = (
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
            random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
        )
        target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
        distance = random.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
        cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(origin=target_loc, radius=distance)
        rep.functional.modify.pose(cam, position_value=cam_loc, rotation_value=quat)

def simulate_camera_collision(num_frames: int = 1) -> None:
    """Enable camera colliders temporarily and simulate to push out overlapping objects."""
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(True)
    if not timeline.is_playing():
        timeline.play()
    for _ in range(num_frames):
        simulation_app.update()
    for cam_collider in camera_colliders:
        collision_api = UsdPhysics.CollisionAPI(cam_collider)
        collision_api.GetCollisionEnabledAttr().Set(False)
```

Apply Velocities Towards a Target

The following function applies velocities to the prims with a random magnitude towards the given target (center of the working area). This is making sure in the example scenario that the objects don’t drift away and are occasionally pulled towards the center to clutter the scene.

```python
def apply_velocities_towards_target(
    prims: list[Usd.Prim],
    target: tuple[float, float, float] = (0, 0, 0),
    strength_range: tuple[float, float] = (0.1, 1.0),
) -> None:
    """Apply velocities to prims directing them towards a target point."""
    for prim in prims:
        loc = prim.GetAttribute("xformOp:translate").Get()
        strength = random.uniform(strength_range[0], strength_range[1])
        velocity = ((target[0] - loc[0]) * strength, (target[1] - loc[1]) * strength, (target[2] - loc[2]) * strength)
        prim.GetAttribute("physics:velocity").Set(velocity)
```

Randomize Sphere Lights

The following snippet creates the given number of lights that will be added to a replicator randomization graph that will randomize the lights attributes (color, temperature, intensity, position, scale) when manually triggered (`rep.utils.send_og_event(event_name="randomize_lights")`).

```python
# Create a randomizer for lights in the working area, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_lights"):
    lights = rep.create.light(
        light_type="Sphere",
        color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform(working_area_min, working_area_max),
        scale=rep.distribution.uniform(0.1, 1),
        count=3,
    )
```

Randomize Shape Distractors Colors

The following snippet creates a randomizer graph for the shape distractors colors, manually triggered at custom events (`rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")`. The paths of the shape distractors prims are used to create a graph node representing the distractor prims, which are then used in the built-in Replicator color randomizer (`rep.randomizer.color`).

```python
# Create a randomizer for the shape distractors colors, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
```

### SDG Loop

The following snippet shows the main data capture loop that runs the simulation for a given number of frames and captures the data at custom intervals. The loop triggers the randomizations and actions at custom frame intervals. For example, randomizing camera poses, applying velocities towards the origin, randomizing lights, shape distractors colors, dome background, and floating distractors velocities.

SDG Loop

```python
# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
for i in range(num_frames):
    # Cameras will be moved to a random position and look at a randomly selected labeled asset
    if i % 3 == 0:
        print(f"\t Randomizing camera poses")
        randomize_camera_poses()
        # Temporarily enable camera colliders and simulate for a few frames to push out any overlapping objects
        if camera_colliders:
            simulate_camera_collision(num_frames=4)

    # Apply a random velocity towards the origin to the working area to pull the assets closer to the center
    if i % 10 == 0:
        print(f"\t Applying velocity towards the origin")
        object_based_sdg_utils.apply_velocities_towards_target(
            list(chain(labeled_prims, shape_distractors, mesh_distractors))
        )

    # Randomize lights locations and colors
    if i % 5 == 0:
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")

    # Randomize the colors of the primitive shape distractors
    if i % 15 == 0:
        print(f"\t Randomizing shape distractors colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    # Randomize the texture of the dome background
    if i % 25 == 0:
        print(f"\t Randomizing dome background")
        rep.utils.send_og_event(event_name="randomize_dome_background")

    # Apply a random velocity on the floating distractors (shapes and meshes)
    if i % 17 == 0:
        print(f"\t Randomizing shape distractors velocities")
        object_based_sdg_utils.apply_random_velocities(
            list(chain(floating_shape_distractors, floating_mesh_distractors))
        )

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if i % 5 == 0:
        capture_with_motion_blur_and_pathtracing(physx_scene, duration=0.025, num_samples=8, spp=128)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()
```

### Motion Blur

The following snippet captures the frames using path tracing and motion blur, it selects the duration of the movement to capture and the number of frames to combine.

Example of a captured frame using motion blur and path tracing:

Motion Blur

```python
def capture_with_motion_blur_and_pathtracing(
    physx_scene: PhysxSchema.PhysxSceneAPI, duration: float = 0.05, num_samples: int = 8, spp: int = 64
) -> None:
    """Capture motion blur by combining pathtraced subframe samples simulated for the given duration."""
    # For small step sizes the physics FPS needs to be temporarily increased to provide movements every sub sample
    orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    target_physics_fps = 1 / duration * num_samples
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Enable motion blur (if not enabled)
    is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
    if not is_motion_blur_enabled:
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    # Number of sub samples to render for motion blur in PathTracing mode
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not timeline.is_playing():
        timeline.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the original physics FPS
    if target_physics_fps > orig_physics_fps:
        print(f"[SDG] Restoring physics FPS from {target_physics_fps} to {orig_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

    # Restore the previous render and motion blur  settings
    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)
```

### Performance Optimization

To optimize the performance of the SDG pipeline, especially if there are many frames computed between captures, the render products (rendering and processing) can be disabled by default and only enabled during the capture time. This can be achieved by setting the `disable_render_products_between_captures` parameter to **True** in the configuration. Setting the `include_viewport` argument to **True** in the `set_render_products_updates` function will also disable the viewport (UI) rendering, this will disable any live feedback in the viewport during the simulation, this can be especially useful if the pipeline is running on a headless server.

Toggle Render Products

```python
def set_render_products_updates(render_products: list, enabled: bool, include_viewport: bool = False) -> None:
    """Enable or disable the render products and viewport rendering."""
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled
```

### Writer

By default the script uses the `PoseWriter` writer to write the data to disk. The writer parameters are as follows:

* **output\_dir** (str): The output directory to write the data to
* **format** (str): The format to use for the output files (for example, CenterPose, DOPE), if None a default format will be used writing all the available data.
* **use\_subfolders** (bool): If True, the data will be written to subfolders based on the camera name.
* **write\_debug\_images** (bool): If True, debug images will also be written (for example, bounding box overlays).
* **skip\_empty\_frames** (bool): If True, empty frames will be skipped when writing the data.

The `PoseWriter` implementation can be found in the `pose_writer.py` file in the `isaacsim.replicator.writers` extension. Examples of using various output formats can be found in the `/config/object_based_sdg_dope_config.yaml` and `/config/object_based_sdg_centerpose_config.yaml` configuration files. Where the `format` parameter is set to **dope** and **centerpose** respectively.

To use a custom writer, the `writer_type` and `writer_kwargs` parameters can be set in the config files or in the script to load a custom writer implementation.

```python
"writer_type": "MyCustomWriter",
"writer_kwargs": {
    "arg1": "val1",
    "arg2": "val2",
    "argn": "valn",
}
```

## SyntheticaDETR

SyntheticaDETR is a 2D object detection network aimed to detect indoor objects in RGB images. It is built on top of RT-DETR, a state of the art 2D object detection network on COCO dataset, with training done on data collected entirely in simulation using the Isaac Sim Replicator. As of today SyntheticaDETR is the top performing object detection network on the BOP leaderboard for YCBV dataset.

Leaderboard link: <https://bop.felk.cvut.cz/leaderboards/detection-bop22/ycb-v/>

### Data Generation

Generate data using Isaac Sim and Replicator with procedurally generated scenes. Objects are dropped from ceilings and simulation is run with physics enabled to avoid interpenetrations to allow for objects to settle into stable configurations on the floor. The RGB renderings are captured during the process along with the ground truth segmentation, depth and bounding boxes of visible objects in the view frustum. The image and ground truth pair are used to train networks using supervised learning.

### Data Generation with Real World Asset Capture

While the above data generation process is suited for objects with known 3D assets already available in digital form, including USD and OBJ format, there are scenarios where such assets are not available apriori.

Therefore, use the AR Code app for iPad/iPhone to capture the assets. The app uses LiDAR and multiple images captured from various diverse viewpoints to obtain the 3D asset model directly in USD format suited for rendering with the Isaac Sim and Replicator.

Below are the asset models captured using the app and visualized from different viewpoints.

These assets were used in the Synthetica rendering framework to obtain rendered images:

The results of the detector trained on this synthetic data and tested directly on the real world images are shown below:

The numbers next to the labels on the bounding boxes represent the confidence values with which the detector is certain about the identity of the object.

### SyntheticaDETR Model and Isaac ROS RT-DETR

The SyntheticaDETR model is available in the NGC Catalog at the following link:
[SyntheticaDETR in NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/synthetica_detr)

Furthermore, to run the model in ROS, refer to this thorough tutorial:
[Isaac ROS RT-DETR Tutorial](https://nvidia-isaac-ros.github.io/repositories_and_packages/isaac_ros_object_detection/isaac_ros_rtdetr/index.html)

---

# Environment Based Synthetic Dataset Generation with Infinigen

This tutorial explains how to set up a synthetic data generation (SDG) pipeline in Isaac Sim using the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension and procedurally generated environments from [Infinigen](https://infinigen.org/). The example uses the [standalone](Workflows.md) workflow.

Example of Infinigen generated rooms.

Example data collected from the synthetic dataset generation pipeline.

## Learning Objectives

In this tutorial, you will learn how to:

* Load procedurally generated environments from Infinigen as background scenes.
* Prepare the environments for SDG and physics simulations.
* Load physics-enabled target assets (labeled) for data collection and distractor assets (unlabeled) for scene diversity.
* Use built-in Replicator randomizer graphs manually triggered at custom intervals, detached from the writing process.
* Use custom USD / Isaac Sim API functions for custom randomizers.
* Use multiple Replicator Writers and cameras (render products) to save different types of data from different viewpoints.
* Use config files to easily customize the simulation and data collection process.
* Understand and customize configuration parameters for flexibility.

## Prerequisites

Before starting this tutorial, you should be familiar with:

* USD / Isaac Sim APIs for creating and manipulating USD stages.
* [Rigid-body dynamics](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/rigid_bodies.html "(in Omni Physics)") and physics simulation in Isaac Sim.
* Replicator [randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)") and [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") for a better understanding of the Replicator randomization graphs pipeline.
* Running simulations as [Standalone Applications](Workflows.md).
* Procedurally generating environments using [Infinigen](https://infinigen.org/).

## Generating Infinigen Environments

1. **Install Infinigen**: Follow the installation instructions on the [Infinigen GitHub Repository](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md).
2. **Generate Environments**: Use the [Hello Room](https://github.com/princeton-vl/infinigen/blob/main/docs/HelloRoom.md) instructions to generate indoor scenes using various settings and parameters.
3. **Example Script**: Use the following example script (Linux) to generate multiple dining room environments with different seeds. The script can be run directly from the terminal.

   ```python
   # Loop from 1 to 10 to generate 10 scenes
   for i in {1..10}
   do
     # Create the output folders for both the Infinigen generation and the Omniverse export
     mkdir -p outputs/indoors/dining_room_$i
     mkdir -p outputs/omniverse/dining_room_$i

     # Step 1: Run Infinigen scene generation for a DiningRoom scene with a specific seed
     python -m infinigen_examples.generate_indoors \
       --seed $i \
       --task coarse \
       --output_folder outputs/indoors/dining_room_$i \
       -g fast_solve.gin singleroom.gin \
       -p compose_indoors.terrain_enabled=False restrict_solving.restrict_parent_rooms=\[\"DiningRoom\"\] &&

     # Step 2: Export the generated scene to Omniverse-compatible format
     python -m infinigen.tools.export \
       --input_folder outputs/indoors/dining_room_$i \
       --output_folder outputs/omniverse/dining_room_$i \
       -f usdc \
       -r 1024 \
       --omniverse
   done
   ```

   * This script generates 10 unique dining room environments by varying the seed value.
   * The `infinigen_examples.generate_indoors` command generates the environments and stores them in `outputs/indoors/dining_room_$i`.
   * The `infinigen.tools.export` command exports the generated environments to the selected format, saving them to `outputs/omniverse/dining_room_$i`.
   * The `-f usdc` flag specifies the format of the exported file to USD.
   * The `--omniverse` flag ensures compatibility with Omniverse applications.

## Scenario Overview

In this tutorial, we will use procedurally generated environments as backdrops for synthetic data generation. These environments are then configured with colliders and physics properties, enabling physics-based simulations. Within each indoor environment, we define a “working area”—in this case, the dining table—where we will place both labeled target assets and unlabeled distractor assets.

The assets are divided into two categories:

* **Falling assets**: Physics-enabled objects that interact with the environment and settle onto surfaces, such as the ground or table.
* **Floating assets**: Objects equipped only with colliders that remain floating in the air.

For each background environment, we will capture frames in two scenarios:

1. Assets floating around the working area.
2. Physics-enabled assets that have settled on surfaces like the ground or table.

To capture these frames, we use multiple cameras (render products) configured with one or multiple writers. The cameras will be randomized for each frame, changing their positions around the working area and orienting toward randomly selected target assets.

Once the captures for one environment are complete, a new environment will be loaded, configured with colliders and physics properties, and the process will repeat until the desired number of captures is achieved.

During the capture process, we will apply randomizers at various frames to introduce variability into the scene. These randomizations include:

* Object poses.
* Lighting configurations, including dome light settings.
* Colors of shape distractors.

By incorporating these randomizations, we increase the diversity of the dataset, making it more robust for training machine learning models.

## Getting Started

The main script for this tutorial is located at:

`<install_path>/standalone_examples/replicator/infinigen/infinigen_sdg.py`

This script is designed to run as a **Standalone Application**. The default configurations are stored within the script itself in the form of a Python dictionary. You can override these defaults by providing custom configuration files in JSON or YAML format.

Helper functions are located in the `infinigen_sdg_utils.py` file. These functions help with loading environments, spawning assets, randomizing object poses, and running physics simulations.

To generate a synthetic dataset using the default configuration, run the following command (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/replicator/infinigen/infinigen_sdg.py
```

To use a custom configuration file that supports multiple writers and other custom settings, use the –config argument:

```python
./python.sh standalone_examples/replicator/infinigen/infinigen_sdg.py \
    --config standalone_examples/replicator/infinigen/config/infinigen_multi_writers_pt.yaml
```

## Implementation

The following sections provide an overview of the key steps involved in setting up and running the synthetic data generation pipeline.
The complete implementation consists of two files: the main script and a utilities module.

Main script

```python
"""Generate synthetic datasets using infinigen (https://infinigen.org/) generated environments."""

import argparse
import json
import os

import yaml
from isaacsim import SimulationApp

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "environments": {
        # List of background environments (list of folders or files)
        "folders": ["/Isaac/Samples/Replicator/Infinigen/dining_rooms/"],
        "files": [],
    },
    "capture": {
        # Number of captures (frames = total_captures * num_cameras)
        "total_captures": 9,
        # Number of captures per environment before running the simulation (objects in the air)
        "num_floating_captures_per_env": 2,
        # Number of captures per environment after running the simulation (objects fallen)
        "num_dropped_captures_per_env": 2,
        # Number of cameras to capture from (each camera will have a render product attached)
        "num_cameras": 2,
        # Resolution of the captured frames
        "resolution": (720, 480),
        # Disable render products throughout the piepline, enable them only when capturing the frames
        "disable_render_products": False,
        # Number of subframes to render (RealTimePathTracing) to avoid temporal rendering artifacts (e.g. ghosting)
        "rt_subframes": 8,
        # Use PathTracing renderer or RealTimePathTracing when capturing the frames
        "path_tracing": False,
        # Offset to avoid the images always being in the image center
        "camera_look_at_target_offset": 0.1,
        # Distance between the camera and the target object
        "camera_distance_to_target_range": (1.15, 1.45),
        # Number of scene lights to create in the working area
        "num_scene_lights": 3,
    },
    "writers": [
        {
            # Type of the writer to use (e.g. PoseWriter, BasicWriter, etc.) and the kwargs to pass to the writer init
            "type": "PoseWriter",
            "kwargs": {
                "output_dir": "_out_infinigen_posewriter",
                "format": None,
                "use_subfolders": True,
                "write_debug_images": True,
                "skip_empty_frames": False,
            },
        }
    ],
    "labeled_assets": {
        # Labeled assets with auto-labeling (e.g. 002_banana -> banana) using regex pattern replacement on the asset name
        "auto_label": {
            # Number of labeled assets to create from the given files/folders list
            "num": 6,
            # Chance to disable gravity for the labeled assets (0.0 - all the assets will fall, 1.0 - all the assets will float)
            "gravity_disabled_chance": 0.25,
            # List of folders and files to search for the labeled assets
            "folders": ["/Isaac/Props/YCB/Axis_Aligned/"],
            "files": ["/Isaac/Props/YCB/Axis_Aligned/036_wood_block.usd"],
            # Regex pattern to replace in the asset name (e.g. "002_banana" -> "banana")
            "regex_replace_pattern": r"^\d+_",
            "regex_replace_repl": "",
        },
        # Manually labeled assets with specific labels and properties
        "manual_label": [
            {
                "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
                "label": "pudding_box",
                "num": 2,
                "gravity_disabled_chance": 0.25,
            },
            {
                "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
                "label": "mustard_bottle",
                "num": 2,
                "gravity_disabled_chance": 0.25,
            },
        ],
    },
    "distractors": {
        # Shape distractors (unlabeled background assets) to drop in the scene (e.g. capsules, cones, cylinders)
        "shape_distractors": {
            # Amount of shape distractors to create
            "num": 20,
            # Chance to disable gravity for the shape distractors
            "gravity_disabled_chance": 0.25,
            # List of shape types to randomly choose from
            "types": ["capsule", "cone", "cylinder", "sphere", "cube"],
        },
        # Mesh distractors (unlabeled background assets) to drop in the scene
        "mesh_distractors": {
            # Amount of mesh distractors to create
            "num": 8,
            # Chance to disable gravity for the mesh distractors
            "gravity_disabled_chance": 0.25,
            # List of folders and files to search to randomly choose from
            "folders": [
                "/Isaac/Environments/Simple_Warehouse/Props/",
                "/Isaac/Environments/Office/Props/",
            ],
            "files": [
                "/Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd",
                "/Isaac/Environments/Hospital/Props/SM_MedicalBox_01g.usd",
            ],
        },
    },
    # Hide ceilling to get a top-down view of the scene, move viewport camera to the top-down view
    "debug_mode": True,
}

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
parser.add_argument(
    "--close-on-completion", action="store_true", help="Ensure the app closes on completion even in debug mode"
)
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            print(f"[SDG] Warning: Config file {args.config} is not json or yaml, using default config")
else:
    print(f"[SDG] Warning: Config file {args.config} does not exist, using default config")

# Update the default config dict with the external one
config.update(args_config)

simulation_app = SimulationApp(launch_config={"headless": False})

from itertools import cycle

import carb.settings
import infinigen_sdg_utils as infinigen_utils
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.viewports import set_camera_view

# Run the SDG pipeline on the scenarios
def run_sdg(config):
    # Load the config parameters
    env_config = config.get("environments", {})
    env_urls = infinigen_utils.get_usd_paths(
        files=env_config.get("files", []), folders=env_config.get("folders", []), skip_folder_keywords=[".thumbs"]
    )
    if not env_urls:
        print("[SDG] Error: No environment USD files found. Please check the 'environments' config.")
        return
    print(f"[SDG] Found {len(env_urls)} environment(s)")
    capture_config = config.get("capture", {})
    writers_config = config.get("writers", {})
    labeled_assets_config = config.get("labeled_assets", {})
    distractors_config = config.get("distractors", {})

    # Create a new stage
    print("[SDG] Creating a new stage")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Disable capture on play
    rep.orchestrator.set_capture_on_play(False)

    # Disable UJITSO cooking ([Warning] [omni.ujitso] UJITSO : Build storage validation failed)
    carb.settings.get_settings().set("/physics/cooking/ujitsoCollisionCooking", False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Initialize randomization
    rep.set_global_seed(12)
    rng = np.random.default_rng(12)

    # Debug mode (hide ceiling, move viewport camera to the top-down view)
    debug_mode = config.get("debug_mode", False)

    # Create the cameras
    cameras = []
    num_cameras = capture_config.get("num_cameras", 0)
    rep.functional.create.scope(name="Cameras")
    for i in range(num_cameras):
        cam_prim = rep.functional.create.camera(parent="/Cameras", name=f"cam_{i}", clipping_range=(0.25, 1000))
        cameras.append(cam_prim)
    print(f"[SDG] Created {len(cameras)} cameras")

    # Create the render products for the cameras
    render_products = []
    resolution = capture_config.get("resolution", (1280, 720))
    disable_render_products = capture_config.get("disable_render_products", False)
    for cam in cameras:
        rp = rep.create.render_product(cam.GetPath(), resolution, name=f"rp_{cam.GetName()}")
        if disable_render_products:
            rp.hydra_texture.set_updates_enabled(False)
        render_products.append(rp)
    print(f"[SDG] Created {len(render_products)} render products")

    # Only create the writers if there are render products to attach to
    writers = []
    if render_products:
        for writer_config in writers_config:
            writer = infinigen_utils.setup_writer(writer_config)
            if writer:
                writer.attach(render_products)
                writers.append(writer)
                print(
                    f"[SDG] {writer_config['type']}'s out dir: {writer_config.get('kwargs', {}).get('output_dir', '')}"
                )
    print(f"[SDG] Created {len(writers)} writers")

    # Load target assets with auto-labeling (e.g. 002_banana -> banana)
    auto_label_config = labeled_assets_config.get("auto_label", {})
    auto_floating_assets, auto_falling_assets = infinigen_utils.load_auto_labeled_assets(auto_label_config, rng)
    print(f"[SDG] Loaded {len(auto_floating_assets)} floating auto-labeled assets")
    print(f"[SDG] Loaded {len(auto_falling_assets)} falling auto-labeled assets")

    # Load target assets with manual labels
    manual_label_config = labeled_assets_config.get("manual_label", [])
    manual_floating_assets, manual_falling_assets = infinigen_utils.load_manual_labeled_assets(manual_label_config, rng)
    print(f"[SDG] Loaded {len(manual_floating_assets)} floating manual-labeled assets")
    print(f"[SDG] Loaded {len(manual_falling_assets)} falling manual-labeled assets")
    target_assets = auto_floating_assets + auto_falling_assets + manual_floating_assets + manual_falling_assets

    # Load the shape distractors
    shape_distractors_config = distractors_config.get("shape_distractors", {})
    floating_shapes, falling_shapes = infinigen_utils.load_shape_distractors(shape_distractors_config, rng)
    print(f"[SDG] Loaded {len(floating_shapes)} floating shape distractors")
    print(f"[SDG] Loaded {len(falling_shapes)} falling shape distractors")
    shape_distractors = floating_shapes + falling_shapes

    # Load the mesh distractors
    mesh_distractors_config = distractors_config.get("mesh_distractors", {})
    floating_meshes, falling_meshes = infinigen_utils.load_mesh_distractors(mesh_distractors_config, rng)
    print(f"[SDG] Loaded {len(floating_meshes)} floating mesh distractors")
    print(f"[SDG] Loaded {len(falling_meshes)} falling mesh distractors")
    mesh_distractors = floating_meshes + falling_meshes

    # Resolve any centimeter-meter scale issues of the assets
    infinigen_utils.resolve_scale_issues_with_metrics_assembler()

    # Create lights to randomize in the working area
    scene_lights = []
    num_scene_lights = capture_config.get("num_scene_lights", 0)
    for i in range(num_scene_lights):
        light_prim = stage.DefinePrim(f"/Lights/SphereLight_scene_{i}", "SphereLight")
        scene_lights.append(light_prim)
    print(f"[SDG] Created {len(scene_lights)} scene lights")

    # Register replicator randomizers and trigger them once
    print("[SDG] Registering replicator graph randomizers")
    infinigen_utils.register_dome_light_randomizer()
    infinigen_utils.register_shape_distractors_color_randomizer(shape_distractors)

    # Check if the render mode needs to be switched to path tracing for the capture (by default: RealTimePathTracing)
    use_path_tracing = capture_config.get("path_tracing", False)

    # Capture detail using subframes (https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html)
    rt_subframes = capture_config.get("rt_subframes", 3)

    # Min and max distance between the camera and the target object
    camera_distance_to_target_range = capture_config.get("camera_distance_to_target_range", (0.5, 1.5))

    # Number of captures (frames = total_captures * num_cameras)
    # NOTE: if captured frames have no labeled data, they can be skipped (e.g. PoseWriter with skip_empty_frames=True)
    total_captures = capture_config.get("total_captures", 0)

    # Number of captures per environment with the objects in the air or dropped
    num_floating_captures_per_env = capture_config.get("num_floating_captures_per_env", 0)
    num_dropped_captures_per_env = capture_config.get("num_dropped_captures_per_env", 0)

    # Start the SDG loop
    env_cycle = cycle(env_urls)
    capture_counter = 0
    while capture_counter < total_captures:
        # Load the next environment
        env_url = next(env_cycle)

        # Load the new environment
        print(f"[SDG] Loading environment: {env_url}")
        infinigen_utils.load_env(env_url, prim_path="/Environment")

        # Setup the environment (add collision, fix lights, etc.) and update the app once to apply the changes
        print(f"[SDG] Setting up the environment")
        infinigen_utils.setup_env(root_path="/Environment", hide_top_walls=debug_mode)
        simulation_app.update()

        # Get the location of the prim above which the assets will be randomized
        working_area_loc = infinigen_utils.get_matching_prim_location(
            match_string="TableDining", root_path="/Environment"
        )

        # Move viewport above the working area to get a top-down view of the scene
        if debug_mode:
            camera_loc = (working_area_loc[0], working_area_loc[1], working_area_loc[2] + 10)
            set_camera_view(eye=np.array(camera_loc), target=np.array(working_area_loc))

        # Get the spawn areas as offseted location ranges from the working area (min_x, min_y, min_z, max_x, max_y, max_z)
        print(f"[SDG] Randomizing {len(target_assets)} target assets around the working area")
        target_loc_range = infinigen_utils.offset_range((-0.5, -0.5, 1, 0.5, 0.5, 1.5), working_area_loc)
        infinigen_utils.randomize_poses(
            target_assets,
            location_range=target_loc_range,
            rotation_range=(0, 360),
            scale_range=(0.95, 1.15),
            rng=rng,
        )

        # Mesh distractors
        print(f"[SDG] Randomizing {len(mesh_distractors)} mesh distractors around the working area")
        mesh_loc_range = infinigen_utils.offset_range((-1, -1, 1, 1, 1, 2), working_area_loc)
        infinigen_utils.randomize_poses(
            mesh_distractors,
            location_range=mesh_loc_range,
            rotation_range=(0, 360),
            scale_range=(0.3, 1.0),
            rng=rng,
        )

        # Shape distractors
        print(f"[SDG] Randomizing {len(shape_distractors)} shape distractors around the working area")
        shape_loc_range = infinigen_utils.offset_range((-1.5, -1.5, 1, 1.5, 1.5, 2), working_area_loc)
        infinigen_utils.randomize_poses(
            shape_distractors,
            location_range=shape_loc_range,
            rotation_range=(0, 360),
            scale_range=(0.01, 0.1),
            rng=rng,
        )

        print(f"[SDG] Randomizing {len(scene_lights)} scene lights properties and locations around the working area")
        lights_loc_range = infinigen_utils.offset_range((-2, -2, 1, 2, 2, 3), working_area_loc)
        infinigen_utils.randomize_lights(
            scene_lights,
            location_range=lights_loc_range,
            intensity_range=(500, 2500),
            color_range=(0.1, 0.1, 0.1, 0.9, 0.9, 0.9),
            rng=rng,
        )

        print("[SDG] Randomizing dome lights")
        rep.utils.send_og_event(event_name="randomize_dome_lights")

        print("[SDG] Randomizing shape distractor colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

        # Run the physics simulation for a few frames to solve any collisions
        print("[SDG] Fixing collisions through physics simulation")
        simulation_app.update()
        infinigen_utils.run_simulation(num_frames=4, render=True)

        # Check if the render products need to be enabled for the capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(True)

        # Check if the render mode needs to be switched to path tracing for the capture
        if use_path_tracing:
            print("[SDG] Switching to PathTracing render mode")
            carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")

        # Capture frames with the objects in the air
        for i in range(num_floating_captures_per_env):
            # Check if the total captures have been reached
            if capture_counter >= total_captures:
                break
            # Randomize the camera poses
            print(f"[SDG] Randomizing camera poses ({len(cameras)} cameras)")
            infinigen_utils.randomize_camera_poses(
                cameras, target_assets, camera_distance_to_target_range, polar_angle_range=(0, 75), rng=rng
            )
            print(
                f"[SDG] Capturing floating assets {i+1}/{num_floating_captures_per_env} (total: {capture_counter+1}/{total_captures})"
            )
            rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
            capture_counter += 1

        # Check if the render products need to be disabled until the next capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(False)

        # Check if the render mode needs to be switched back to raytracing until the next capture
        if use_path_tracing:
            carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")

        print("[SDG] Running the simulation")
        infinigen_utils.run_simulation(num_frames=200, render=False)

        # Check if the render products need to be enabled for the capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(True)

        # Check if the render mode needs to be switched to path tracing for the capture
        if use_path_tracing:
            carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")

        for i in range(num_dropped_captures_per_env):
            # Check if the total captures have been reached
            if capture_counter >= total_captures:
                break
            # Spawn the cameras with a smaller polar angle to have mostly a top-down view of the objects
            print("[SDG] Randomizing camera poses")
            infinigen_utils.randomize_camera_poses(
                cameras,
                target_assets,
                distance_range=camera_distance_to_target_range,
                polar_angle_range=(0, 45),
                rng=rng,
            )
            print(
                f"[SDG] Capturing dropped assets {i+1}/{num_dropped_captures_per_env} (total: {capture_counter+1}/{total_captures})"
            )
            rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
            capture_counter += 1

        # Check if the render products need to be disabled until the next capture
        if disable_render_products:
            for rp in render_products:
                rp.hydra_texture.set_updates_enabled(False)

        # Check if the render mode needs to be switched back to raytracing until the next capture
        if use_path_tracing:
            carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")

    # Wait until the data is written to the disk
    rep.orchestrator.wait_until_complete()

    # Detach the writers
    print("[SDG] Detaching writers")
    for writer in writers:
        writer.detach()

    # Destroy render products
    print("[SDG] Destroying render products")
    for rp in render_products:
        rp.destroy()

    print(f"[SDG] Finished, captured {capture_counter * num_cameras} frames")

# Check if debug mode is enabled
debug_mode = config.get("debug_mode", False)

# Start the SDG pipeline
print("[SDG] Starting the SDG pipeline")
run_sdg(config)
print("[SDG] Pipeline finished")

# Make sure the app closes on completion even if in debug mode
if args.close_on_completion:
    simulation_app.close()

# In debug mode, keep the app running until manually closed
if debug_mode:
    while simulation_app.is_running():
        simulation_app.update()

simulation_app.close()
```

Utils module

```python
import math
import os
import re
from itertools import chain

import numpy as np
import omni.client
import omni.kit.app
import omni.kit.commands
import omni.physics.core
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import add_labels, remove_labels
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

def add_colliders(root_prim: Usd.Prim, approximation_type: str = "convexHull") -> None:
    """Add collision attributes to mesh and geometry primitives under the root prim."""
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Gprim):
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approximation_type)

def add_rigid_body(prim: Usd.Prim, disable_gravity: bool = False, ensure_mass: bool = False) -> None:
    """Apply rigid body physics, optionally ensuring a valid mass property exists (defaults to 1.0 kg)."""
    rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
    if not ensure_mass:
        return
    if not prim.HasAPI(UsdPhysics.MassAPI):
        UsdPhysics.MassAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI(prim)
    mass_attr = mass_api.GetMassAttr()
    if not mass_attr or mass_attr.Get() is None or mass_attr.Get() <= 0:
        mass_api.CreateMassAttr(1.0)

def get_random_pose_on_sphere(
    origin: tuple[float, float, float],
    radius_range: tuple[float, float],
    polar_angle_range: tuple[float, float],
    camera_forward_axis: tuple[float, float, float] = (0, 0, -1),
    rng: np.random.Generator = None,
) -> tuple[Gf.Vec3d, Gf.Quatf]:
    """Generate a random pose on a sphere looking at the origin, with specified radius and polar angle ranges."""
    if rng is None:
        rng = np.random.default_rng()

    # https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/reference_conventions.html
    # Convert degrees to radians for polar angles (theta)
    polar_angle_min_rad = math.radians(polar_angle_range[0])
    polar_angle_max_rad = math.radians(polar_angle_range[1])

    # Generate random spherical coordinates
    radius = rng.uniform(radius_range[0], radius_range[1])
    polar_angle = rng.uniform(polar_angle_min_rad, polar_angle_max_rad)
    azimuthal_angle = rng.uniform(0, 2 * math.pi)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * math.sin(polar_angle) * math.cos(azimuthal_angle)
    y = radius * math.sin(polar_angle) * math.sin(azimuthal_angle)
    z = radius * math.cos(polar_angle)

    # Calculate the location in 3D space
    location = Gf.Vec3d(origin[0] + x, origin[1] + y, origin[2] + z)

    # Calculate direction vector from camera to look_at point
    direction = Gf.Vec3d(origin) - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), direction_normalized)
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

def randomize_camera_poses(
    cameras: list[Usd.Prim],
    targets: list[Usd.Prim],
    distance_range: tuple[float, float],
    polar_angle_range: tuple[float, float] = (0, 180),
    look_at_offset: tuple[float, float] = (-0.1, 0.1),
    rng: np.random.Generator = None,
) -> None:
    """Randomize the poses of cameras to look at random targets with adjustable distance and offset."""
    for cam in cameras:
        # Get a random target asset to look at
        target_asset = targets[rng.integers(len(targets))]

        # Add a look_at offset so the target is not always in the center of the camera view
        target_loc = target_asset.GetAttribute("xformOp:translate").Get()
        target_loc = (
            target_loc[0] + rng.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[1] + rng.uniform(look_at_offset[0], look_at_offset[1]),
            target_loc[2] + rng.uniform(look_at_offset[0], look_at_offset[1]),
        )

        # Generate random camera pose
        loc, quat = get_random_pose_on_sphere(target_loc, distance_range, polar_angle_range, rng=rng)

        # Set the camera's transform attributes to the generated location and orientation
        rep.functional.modify.pose(cam, position_value=loc, rotation_value=quat)

def get_usd_paths_from_folder(
    folder_path: str, recursive: bool = True, usd_paths: list[str] = None, skip_keywords: list[str] = None
) -> list[str]:
    """Retrieve USD file paths from a folder, optionally searching recursively and filtering by keywords."""
    if usd_paths is None:
        usd_paths = []
    skip_keywords = skip_keywords or []

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.client"):
        ext_manager.set_extension_enabled_immediate("omni.client", True)

    result, entries = omni.client.list(folder_path)
    if result != omni.client.Result.OK:
        print(f"[SDG] Error: Could not list assets in path: {folder_path}")
        return usd_paths

    for entry in entries:
        if any(keyword.lower() in entry.relative_path.lower() for keyword in skip_keywords):
            continue
        _, ext = os.path.splitext(entry.relative_path)
        if ext in [".usd", ".usda", ".usdc"]:
            path_posix = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            usd_paths.append(path_posix)
        elif recursive and entry.flags & omni.client.ItemFlags.CAN_HAVE_CHILDREN:
            sub_folder = os.path.join(folder_path, entry.relative_path).replace("\\", "/")
            get_usd_paths_from_folder(sub_folder, recursive=recursive, usd_paths=usd_paths, skip_keywords=skip_keywords)

    return usd_paths

def get_usd_paths(
    files: list[str] = None, folders: list[str] = None, skip_folder_keywords: list[str] = None
) -> list[str]:
    """Retrieve USD paths from specified files and folders, optionally filtering out specific folder keywords."""

    def resolve_path(path: str, assets_root: str, is_folder: bool = False) -> str:
        """Resolve path to full URL: remote URLs and existing local paths used as-is, otherwise prefixed with assets_root."""
        # Remote URLs - use as-is
        if path.startswith(("omniverse://", "http://", "https://", "file://")):
            return path
        # Windows absolute path (e.g., C:\path or C:/path) - use as-is
        if len(path) > 2 and path[1] == ":" and path[2] in ("/", "\\"):
            return path
        # Local absolute path that exists - use as-is
        if path.startswith("/") and (os.path.isfile(path) or (is_folder and os.path.isdir(path))):
            return path
        # Nucleus relative path - prepend assets root
        return assets_root + path

    files = files or []
    folders = folders or []
    skip_folder_keywords = skip_folder_keywords or []

    assets_root_path = get_assets_root_path()
    env_paths = []

    for file_path in files:
        env_paths.append(resolve_path(file_path, assets_root_path, is_folder=False))

    for folder_path in folders:
        resolved_folder = resolve_path(folder_path, assets_root_path, is_folder=True)
        env_paths.extend(get_usd_paths_from_folder(resolved_folder, recursive=True, skip_keywords=skip_folder_keywords))

    return env_paths

def load_env(usd_path: str, prim_path: str, remove_existing: bool = True) -> Usd.Prim:
    """Load an environment from a USD file into the stage at the specified prim path, optionally removing any existing prim."""
    stage = omni.usd.get_context().get_stage()

    # Remove existing prim if specified
    if remove_existing and stage.GetPrimAtPath(prim_path):
        omni.kit.commands.execute("DeletePrimsCommand", paths=[prim_path])

    root_prim = add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
    return root_prim

def add_colliders_to_env(root_path: str | None = None, approximation_type: str = "none") -> None:
    """Add colliders to all mesh prims within the specified root path in the stage."""
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    for prim in Usd.PrimRange(prim):
        if prim.IsA(UsdGeom.Mesh):
            add_colliders(prim, approximation_type)

def find_matching_prims(
    match_strings: list[str], root_path: str | None = None, prim_type: str | None = None, first_match_only: bool = False
) -> Usd.Prim | list[Usd.Prim] | None:
    """Find prims matching specified strings, with optional type filtering and single match return."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    matching_prims = []
    for prim in Usd.PrimRange(root_prim):
        if any(match in str(prim.GetPath()) for match in match_strings):
            if prim_type is None or prim.GetTypeName() == prim_type:
                if first_match_only:
                    return prim
                matching_prims.append(prim)

    return matching_prims if not first_match_only else None

def hide_matching_prims(match_strings: list[str], root_path: str | None = None, prim_type: str | None = None) -> None:
    """Set visibility of prims matching specified strings to 'invisible' within the root path."""
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPseudoRoot() if root_path is None else stage.GetPrimAtPath(root_path)

    for prim in Usd.PrimRange(root_prim):
        if prim_type is None or prim.GetTypeName() == prim_type:
            if any(match in str(prim.GetPath()) for match in match_strings):
                prim.GetAttribute("visibility").Set("invisible")

def setup_env(root_path: str | None = None, approximation_type: str = "none", hide_top_walls: bool = False) -> None:
    """Set up the environment with colliders, ceiling light adjustments, and optional top wall hiding."""
    # Fix ceiling lights: meshes are blocking the light and need to be set to invisible
    ceiling_light_meshes = find_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")
    for light_mesh in ceiling_light_meshes:
        light_mesh.GetAttribute("visibility").Set("invisible")

    # Hide ceiling light meshes for lighting fix
    hide_matching_prims(["001_SPLIT_GLA"], root_path, "Xform")

    # Hide top walls for better debug view, if specified
    if hide_top_walls:
        hide_matching_prims(["_exterior", "_ceiling"], root_path)

    # Add colliders to the environment
    add_colliders_to_env(root_path, approximation_type)

    # Fix dining table collision by setting it to a bounding cube approximation
    table_prim = find_matching_prims(
        match_strings=["TableDining"], root_path=root_path, prim_type="Xform", first_match_only=True
    )
    if table_prim is not None:
        add_colliders(table_prim, approximation_type="boundingCube")
    else:
        print("[SDG] Warning: Could not find dining table prim in the environment")

def create_shape_distractors(
    num_distractors: int,
    shape_types: list[str],
    root_path: str,
    gravity_disabled_chance: float,
    rng: np.random.Generator = None,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create shape distractors with optional gravity settings, returning lists of floating and falling shapes."""
    if rng is None:
        rng = np.random.default_rng()
    stage = omni.usd.get_context().get_stage()
    floating_shapes = []
    falling_shapes = []
    for _ in range(num_distractors):
        rand_shape = shape_types[rng.integers(len(shape_types))]
        disable_gravity = rng.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{root_path}/{name_prefix}{rand_shape}", False)
        prim = stage.DefinePrim(prim_path, rand_shape.capitalize())
        add_colliders(prim)
        add_rigid_body(prim, disable_gravity=disable_gravity, ensure_mass=True)
        (floating_shapes if disable_gravity else falling_shapes).append(prim)
    return floating_shapes, falling_shapes

def load_shape_distractors(
    shape_distractors_config: dict, rng: np.random.Generator = None
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load shape distractors based on configuration, returning lists of floating and falling shapes."""
    num_shapes = shape_distractors_config.get("num", 0)
    shape_types = shape_distractors_config.get("shape_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
    shape_gravity_disabled_chance = shape_distractors_config.get("gravity_disabled_chance", 0.0)
    return create_shape_distractors(num_shapes, shape_types, "/Distractors", shape_gravity_disabled_chance, rng)

def create_mesh_distractors(
    num_distractors: int,
    mesh_urls: list[str],
    root_path: str,
    gravity_disabled_chance: float,
    rng: np.random.Generator = None,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create mesh distractors from specified URLs with optional gravity settings."""
    if rng is None:
        rng = np.random.default_rng()
    stage = omni.usd.get_context().get_stage()
    floating_meshes = []
    falling_meshes = []
    for _ in range(num_distractors):
        rand_mesh_url = mesh_urls[rng.integers(len(mesh_urls))]
        disable_gravity = rng.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_name = os.path.basename(rand_mesh_url).split(".")[0]
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{root_path}/{name_prefix}{prim_name}", False)
        try:
            prim = add_reference_to_stage(usd_path=rand_mesh_url, prim_path=prim_path)
        except Exception as e:
            print(f"[SDG] Error: Failed to load mesh distractor '{rand_mesh_url}': {e}")
            continue
        add_colliders(prim)
        add_rigid_body(prim, disable_gravity=disable_gravity, ensure_mass=True)
        (floating_meshes if disable_gravity else falling_meshes).append(prim)
    return floating_meshes, falling_meshes

def load_mesh_distractors(
    mesh_distractors_config: dict, rng: np.random.Generator = None
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load mesh distractors based on configuration, returning lists of floating and falling meshes."""
    num_meshes = mesh_distractors_config.get("num", 0)
    mesh_gravity_disabled_chance = mesh_distractors_config.get("gravity_disabled_chance", 0.0)
    mesh_folders = mesh_distractors_config.get("folders", [])
    mesh_files = mesh_distractors_config.get("files", [])
    mesh_urls = get_usd_paths(
        files=mesh_files, folders=mesh_folders, skip_folder_keywords=["material", "texture", ".thumbs"]
    )
    floating_meshes, falling_meshes = create_mesh_distractors(
        num_meshes, mesh_urls, "/Distractors", mesh_gravity_disabled_chance, rng
    )
    for prim in chain(floating_meshes, falling_meshes):
        remove_labels(prim, include_descendants=True)
    return floating_meshes, falling_meshes

def create_auto_labeled_assets(
    num_assets: int,
    asset_urls: list[str],
    root_path: str,
    regex_replace_pattern: str,
    regex_replace_repl: str,
    gravity_disabled_chance: float,
    rng: np.random.Generator = None,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create assets with automatic labels, applying optional gravity settings."""
    if rng is None:
        rng = np.random.default_rng()
    stage = omni.usd.get_context().get_stage()
    floating_assets = []
    falling_assets = []
    for _ in range(num_assets):
        asset_url = asset_urls[rng.integers(len(asset_urls))]
        disable_gravity = rng.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        basename = os.path.basename(asset_url)
        name_without_ext = os.path.splitext(basename)[0]
        label = re.sub(regex_replace_pattern, regex_replace_repl, name_without_ext)
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{root_path}/{name_prefix}{label}", False)
        try:
            prim = add_reference_to_stage(usd_path=asset_url, prim_path=prim_path)
        except Exception as e:
            print(f"[SDG] Error: Failed to load asset '{asset_url}': {e}")
            continue
        add_colliders(prim)
        add_rigid_body(prim, disable_gravity=disable_gravity, ensure_mass=True)
        remove_labels(prim, include_descendants=True)
        add_labels(prim, labels=[label], instance_name="class")
        (floating_assets if disable_gravity else falling_assets).append(prim)
    return floating_assets, falling_assets

def load_auto_labeled_assets(
    auto_label_config: dict, rng: np.random.Generator = None
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load auto-labeled assets based on configuration, returning lists of floating and falling assets."""
    num_assets = auto_label_config.get("num", 0)
    gravity_disabled_chance = auto_label_config.get("gravity_disabled_chance", 0.0)
    assets_files = auto_label_config.get("files", [])
    assets_folders = auto_label_config.get("folders", [])
    assets_urls = get_usd_paths(
        files=assets_files, folders=assets_folders, skip_folder_keywords=["material", "texture", ".thumbs"]
    )
    regex_replace_pattern = auto_label_config.get("regex_replace_pattern", "")
    regex_replace_repl = auto_label_config.get("regex_replace_repl", "")
    return create_auto_labeled_assets(
        num_assets,
        assets_urls,
        "/Assets",
        regex_replace_pattern,
        regex_replace_repl,
        gravity_disabled_chance,
        rng,
    )

def create_labeled_assets(
    num_assets: int,
    asset_url: str,
    label: str,
    root_path: str,
    gravity_disabled_chance: float,
    rng: np.random.Generator = None,
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Create labeled assets with optional gravity settings, returning lists of floating and falling assets."""
    if rng is None:
        rng = np.random.default_rng()
    stage = omni.usd.get_context().get_stage()
    assets_root_path = get_assets_root_path()
    asset_url = (
        asset_url
        if asset_url.startswith(("omniverse://", "http://", "https://", "file://"))
        else assets_root_path + asset_url
    )
    floating_assets = []
    falling_assets = []
    for _ in range(num_assets):
        disable_gravity = rng.random() < gravity_disabled_chance
        name_prefix = "floating_" if disable_gravity else "falling_"
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{root_path}/{name_prefix}{label}", False)

        prim = add_reference_to_stage(usd_path=asset_url, prim_path=prim_path)
        add_colliders(prim)
        add_rigid_body(prim, disable_gravity=disable_gravity, ensure_mass=True)
        remove_labels(prim, include_descendants=True)
        add_labels(prim, labels=[label], instance_name="class")
        (floating_assets if disable_gravity else falling_assets).append(prim)
    return floating_assets, falling_assets

def load_manual_labeled_assets(
    manual_labeled_assets_config: list[dict], rng: np.random.Generator = None
) -> tuple[list[Usd.Prim], list[Usd.Prim]]:
    """Load manually labeled assets based on configuration, returning lists of floating and falling assets."""
    labeled_floating_assets = []
    labeled_falling_assets = []
    for labeled_asset_config in manual_labeled_assets_config:
        asset_url = labeled_asset_config.get("url", "")
        asset_label = labeled_asset_config.get("label", "")
        num_assets = labeled_asset_config.get("num", 0)
        gravity_disabled_chance = labeled_asset_config.get("gravity_disabled_chance", 0.0)
        floating_assets, falling_assets = create_labeled_assets(
            num_assets,
            asset_url,
            asset_label,
            "/Assets",
            gravity_disabled_chance,
            rng,
        )
        labeled_floating_assets.extend(floating_assets)
        labeled_falling_assets.extend(falling_assets)
    return labeled_floating_assets, labeled_falling_assets

def resolve_scale_issues_with_metrics_assembler() -> None:
    """Enable and execute metrics assembler to resolve scale issues in the stage."""
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    if not ext_manager.is_extension_enabled("omni.usd.metrics.assembler"):
        ext_manager.set_extension_enabled_immediate("omni.usd.metrics.assembler", True)
    from omni.metrics.assembler.core import get_metrics_assembler_interface

    stage_id = omni.usd.get_context().get_stage_id()
    get_metrics_assembler_interface().resolve_stage(stage_id)

def get_matching_prim_location(match_string, root_path=None):
    prim = find_matching_prims(
        match_strings=[match_string], root_path=root_path, prim_type="Xform", first_match_only=True
    )
    if prim is None:
        print("[SDG] Warning: Could not find matching prim, returning (0, 0, 0)")
        return (0, 0, 0)
    if prim.HasAttribute("xformOp:translate"):
        return prim.GetAttribute("xformOp:translate").Get()
    elif prim.HasAttribute("xformOp:transform"):
        return prim.GetAttribute("xformOp:transform").Get().ExtractTranslation()
    else:
        print(f"[SDG] Warning: Could not find location attribute for '{prim.GetPath()}', returning (0, 0, 0)")
        return (0, 0, 0)

def offset_range(
    range_coords: tuple[float, float, float, float, float, float], offset: tuple[float, float, float]
) -> tuple[float, float, float, float, float, float]:
    """Offset the min and max coordinates of a range by the specified offset."""
    return (
        range_coords[0] + offset[0],  # min_x
        range_coords[1] + offset[1],  # min_y
        range_coords[2] + offset[2],  # min_z
        range_coords[3] + offset[0],  # max_x
        range_coords[4] + offset[1],  # max_y
        range_coords[5] + offset[2],  # max_z
    )

def randomize_poses(
    prims: list[Usd.Prim],
    location_range: tuple[float, float, float, float, float, float],
    rotation_range: tuple[float, float],
    scale_range: tuple[float, float],
    rng: np.random.Generator = None,
) -> None:
    """Randomize the location, rotation, and scale of a list of prims within specified ranges."""
    if rng is None:
        rng = np.random.default_rng()
    for prim in prims:
        rand_loc = (
            rng.uniform(location_range[0], location_range[3]),
            rng.uniform(location_range[1], location_range[4]),
            rng.uniform(location_range[2], location_range[5]),
        )
        rand_rot = (
            rng.uniform(rotation_range[0], rotation_range[1]),
            rng.uniform(rotation_range[0], rotation_range[1]),
            rng.uniform(rotation_range[0], rotation_range[1]),
        )
        rand_scale = rng.uniform(scale_range[0], scale_range[1])
        rep.functional.modify.pose(prim, position_value=rand_loc, rotation_value=rand_rot, scale_value=rand_scale)

def run_simulation(num_frames: int, render: bool = True) -> None:
    """Run a simulation for a specified number of frames, optionally without rendering."""
    if render:
        # Start the timeline and advance the app, this will render the physics simulation results every frame
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1000000)
        timeline.set_looping(False)
        timeline.play()
        for _ in range(num_frames):
            omni.kit.app.get_app().update()
        timeline.pause()
    else:
        # Run the physics simulation steps without advancing the app
        stage = omni.usd.get_context().get_stage()
        physx_scene = None

        # Search for or create a physics scene
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                break

        if physx_scene is None:
            physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

        # Get simulation parameters
        physx_dt = 1 / physx_scene.GetTimeStepsPerSecondAttr().Get()
        physics_sim_interface = omni.physics.core.get_physics_simulation_interface()

        # Run physics simulation for each frame
        for _ in range(num_frames):
            physics_sim_interface.simulate(physx_dt, 0)
            physics_sim_interface.fetch_results()

def register_dome_light_randomizer() -> None:
    """Register a replicator graph randomizer for dome lights using various sky textures."""
    assets_root_path = get_assets_root_path()
    dome_textures = [
        assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Evening/evening_road_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Night/kloppenheim_02_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Night/moonlit_golf_4k.hdr",
    ]
    with rep.trigger.on_custom_event(event_name="randomize_dome_lights"):
        rep.create.light(light_type="Dome", texture=rep.distribution.choice(dome_textures))

def register_shape_distractors_color_randomizer(shape_distractors: list[Usd.Prim]) -> None:
    """Register a replicator graph randomizer to change colors of shape distractors."""
    with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
        shape_distractors_paths = [prim.GetPath() for prim in shape_distractors]
        shape_distractors_group = rep.create.group(shape_distractors_paths)
        with shape_distractors_group:
            rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))

def randomize_lights(
    lights: list[Usd.Prim],
    location_range: tuple[float, float, float, float, float, float] | None = None,
    color_range: tuple[float, float, float, float, float, float] | None = None,
    intensity_range: tuple[float, float] | None = None,
    rng: np.random.Generator = None,
) -> None:
    """Randomize location, color, and intensity of specified lights within given ranges."""
    if rng is None:
        rng = np.random.default_rng()
    for light in lights:
        # Randomize the location of the light
        if location_range is not None:
            rand_loc = (
                rng.uniform(location_range[0], location_range[3]),
                rng.uniform(location_range[1], location_range[4]),
                rng.uniform(location_range[2], location_range[5]),
            )
            rep.functional.modify.pose(light, position_value=rand_loc)

        # Randomize the color of the light
        if color_range is not None:
            rand_color = (
                rng.uniform(color_range[0], color_range[3]),
                rng.uniform(color_range[1], color_range[4]),
                rng.uniform(color_range[2], color_range[5]),
            )
            light.GetAttribute("inputs:color").Set(rand_color)

        # Randomize the intensity of the light
        if intensity_range is not None:
            rand_intensity = rng.uniform(intensity_range[0], intensity_range[1])
            light.GetAttribute("inputs:intensity").Set(rand_intensity)

def setup_writer(config: dict) -> None:
    """Setup a writer based on configuration settings, initializing with specified arguments."""
    writer_type = config.get("type", None)
    if writer_type is None:
        print("[SDG] Warning: No writer type specified, skipping writer setup")
        return None

    try:
        writer = rep.writers.get(writer_type)
    except Exception as e:
        print(f"[SDG] Error: Writer type '{writer_type}' not found: {e}")
        return None

    writer_kwargs = config.get("kwargs", {})
    if out_dir := writer_kwargs.get("output_dir"):
        # If not an absolute path, make path relative to the current working directory
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(os.getcwd(), out_dir)
            writer_kwargs["output_dir"] = out_dir

    writer.initialize(**writer_kwargs)
    return writer
```

### Configuration Files

Example configuration files are provided in the `infinigen/config` directory. These files allow you to customize various aspects of the simulation, such as the number of captures, assets to include, randomization parameters, and writers to use.

Here’s an example of a custom YAML configuration file that demonstrates the use of multiple writers:

Custom YAML Configuration File

```python
environments:
  # List of background environments (list of folders or files)
  folders:
    - /Isaac/Samples/Replicator/Infinigen/dining_rooms/
  files: []

capture:
  # Number of captures (frames = total_captures * num_cameras)
  total_captures: 12
  # Number of captures per environment before running the simulation (objects in the air)
  num_floating_captures_per_env: 2
  # Number of captures per environment after running the simulation (objects dropped)
  num_dropped_captures_per_env: 3
  # Number of cameras to capture from (each camera will have a render product attached)
  num_cameras: 2
  # Resolution of the captured frames
  resolution: [640, 480]
  # Disable render products throughout the pipeline, enable them only when capturing the frames
  disable_render_products: true
  # Number of subframes to render (RealTimePathTracing) to avoid temporal rendering artifacts (e.g. ghosting)
  rt_subframes: 8
  # Use PathTracing renderer instead of RealTimePathTracing when capturing the frames
  path_tracing: true
  # Offset to avoid the images always being in the image center
  camera_look_at_target_offset: 0.1
  # Distance between the camera and the target object
  camera_distance_to_target_range: [1.05, 1.25]
  # Number of scene lights to create in the working area
  num_scene_lights: 4

writers:
  # Type of the writer to use (e.g. PoseWriter, BasicWriter, etc.) and the kwargs to pass to the writer init
  - type: BasicWriter
    kwargs:
      output_dir: "_out_infinigen_basicwriter_pt"
      rgb: true
      semantic_segmentation: true
      colorize_semantic_segmentation: true
      use_common_output_dir: false
  - type: DataVisualizationWriter
    kwargs:
      output_dir: "_out_infinigen_dataviswriter_pt"
      bounding_box_2d_tight: true
      bounding_box_2d_tight_params:
        background: rgb
      bounding_box_3d: true
      bounding_box_3d_params:
        background: normals

labeled_assets:
  # Labeled assets with auto-labeling (e.g. 002_banana -> banana) using regex pattern replacement on the asset name
  auto_label:
    # Number of labeled assets to create from the given files/folders list
    num: 5
    # Chance to disable gravity for the labeled assets (0.0 - all the assets will fall, 1.0 - all the assets will float)
    gravity_disabled_chance: 0.25
    # List of folders and files to search for the labeled assets
    folders:
      - /Isaac/Props/YCB/Axis_Aligned/
    files:
      - /Isaac/Props/YCB/Axis_Aligned/036_wood_block.usd
    # Regex pattern to replace in the asset name (e.g. "002_banana" -> "banana")
    regex_replace_pattern: "^\\d+_"
    regex_replace_repl: ""

  # Manually labeled assets with specific labels and properties
  manual_label:
    - url: /Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd
      label: pudding_box
      num: 2
      gravity_disabled_chance: 0.25
    - url: /Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd
      label: mustard_bottle
      num: 2
      gravity_disabled_chance: 0.25

distractors:
  # Shape distractors (unlabeled background assets) to drop in the scene (e.g. capsules, cones, cylinders)
  shape_distractors:
    # Amount of shape distractors to create
    num: 30
    # Chance to disable gravity for the shape distractors
    gravity_disabled_chance: 0.25
    # List of shape types to randomly choose from
    types: ["capsule", "cone", "cylinder", "sphere", "cube"]

  # Mesh distractors (unlabeled background assets) to drop in the scene
  mesh_distractors:
    # Amount of mesh distractors to create
    num: 10
    # Chance to disable gravity for the mesh distractors
    gravity_disabled_chance: 0.25
    # List of folders and files to search to randomly choose from
    folders:
      - /Isaac/Environments/Office/Props/
    files:
      - /Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd
      - /Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd
      - /Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd
      - /Isaac/Environments/Simple_Warehouse/Props/S_WetFloorSign.usd
      - /Isaac/Environments/Hospital/Props/SM_MedicalBag_01a.usd
      - /Isaac/Environments/Hospital/Props/SM_MedicalBox_01g.usd

# Hide ceiling, move viewport camera to top-down view above the working area
debug_mode: true
```

### Configuration Parameters

Here is an explanation of the configuration parameters:

* **environments**:

  + **folders**: List of directories containing the Infinigen environments to be used.
  + **files**: Specific USD files of environments to be loaded.
* **capture**:

  + **total\_captures**: Total number of captures to generate.
  + **num\_floating\_captures\_per\_env**: Number of captures to take before running the physics simulation (assets are floating).
  + **num\_dropped\_captures\_per\_env**: Number of captures to take after the physics simulation (assets have settled).
  + **num\_cameras**: Number of cameras to use for capturing images.
  + **resolution**: Resolution of the rendered images (width, height).
  + **disable\_render\_products**: If true, render products are disabled between captures to improve performance.
  + **rt\_subframes**: Number of subframes to render for each capture.
  + **path\_tracing**: If true, uses path tracing for rendering (higher quality, slower).
  + **camera\_look\_at\_target\_offset**: Random offset applied when cameras look at target assets.
  + **camera\_distance\_to\_target\_range**: Range of distances for cameras from the target assets.
  + **num\_scene\_lights**: Number of additional lights to add to the scene.
* **writers**: List of writers to use for data output.

  + **type**: Type of writer (e.g., BasicWriter, DataVisualizationWriter).
  + **kwargs**: Arguments specific to each writer type.
* **labeled\_assets**:

  + **auto\_label**: Configuration for automatically labeled assets.

    - **num**: Number of assets to spawn.
    - **gravity\_disabled\_chance**: Probability that an asset will have gravity disabled (will float).
    - **folders** and **files**: Sources for the asset USD files.
    - **regex\_replace\_pattern** and **regex\_replace\_repl**: Used to generate labels from file names.
  + **manual\_label**: List of assets with manually specified labels.

    - **url**: USD file path of the asset.
    - **label**: Semantic label to assign.
    - **num**: Number of instances to spawn.
    - **gravity\_disabled\_chance**: Probability of gravity being disabled.
* **distractors**:

  + **shape\_distractors**: Configuration for primitive shape distractors.

    - **num**: Number of distractors to spawn.
    - **gravity\_disabled\_chance**: Probability of gravity being disabled.
    - **types**: List of primitive shapes to use.
  + **mesh\_distractors**: Configuration for mesh distractors.

    - **num**: Number of distractors to spawn.
    - **gravity\_disabled\_chance**: Probability of gravity being disabled.
    - **folders** and **files**: Sources for the distractor USD files.
* **debug\_mode**: When set to true, certain elements like ceilings are hidden to provide a better view of the scene during development and debugging.

### Loading Infinigen Environments

We will load environments generated by Infinigen into the Isaac Sim stage. The environments are specified in the configuration file, either through folders or individual files.

Loading Infinigen Environments

```python
def run_sdg(config):
    # Load the config parameters
    env_config = config.get("environments", {})
    env_urls = infinigen_utils.get_usd_paths(
        files=env_config.get("files", []), folders=env_config.get("folders", []), skip_folder_keywords=[".thumbs"]
    )
    if not env_urls:
        print("[SDG] Error: No environment USD files found. Please check the 'environments' config.")
        return
    print(f"[SDG] Found {len(env_urls)} environment(s)")
```

```python
# Start the SDG loop
env_cycle = cycle(env_urls)
capture_counter = 0
while capture_counter < total_captures:
    # Load the next environment
    env_url = next(env_cycle)

    # Load the new environment
    print(f"[SDG] Loading environment: {env_url}")
    infinigen_utils.load_env(env_url, prim_path="/Environment")
```

In the above code, we use the `get_usd_paths` utility function to collect all USD files from the specified folders and files in the configuration. The `skip_folder_keywords` parameter filters out directories containing specified keywords (e.g., `.thumbs` thumbnail folders). We then cycle through these environments to load them one by one.

### Setting Up the Scene

After loading the environment, we set up the scene by:

* Hiding unnecessary elements (e.g., ceiling) for better visibility if the debugging mode is selected.
* Adding colliders to the environment for physics simulation.
* Loading labeled assets and distractors with physics properties.
* Randomizing asset poses within the working area.

Loading Assets

```python
# Load target assets with auto-labeling (e.g. 002_banana -> banana)
auto_label_config = labeled_assets_config.get("auto_label", {})
auto_floating_assets, auto_falling_assets = infinigen_utils.load_auto_labeled_assets(auto_label_config, rng)
print(f"[SDG] Loaded {len(auto_floating_assets)} floating auto-labeled assets")
print(f"[SDG] Loaded {len(auto_falling_assets)} falling auto-labeled assets")

# Load target assets with manual labels
manual_label_config = labeled_assets_config.get("manual_label", [])
manual_floating_assets, manual_falling_assets = infinigen_utils.load_manual_labeled_assets(manual_label_config, rng)
print(f"[SDG] Loaded {len(manual_floating_assets)} floating manual-labeled assets")
print(f"[SDG] Loaded {len(manual_falling_assets)} falling manual-labeled assets")
target_assets = auto_floating_assets + auto_falling_assets + manual_floating_assets + manual_falling_assets

# Load the shape distractors
shape_distractors_config = distractors_config.get("shape_distractors", {})
floating_shapes, falling_shapes = infinigen_utils.load_shape_distractors(shape_distractors_config, rng)
print(f"[SDG] Loaded {len(floating_shapes)} floating shape distractors")
print(f"[SDG] Loaded {len(falling_shapes)} falling shape distractors")
shape_distractors = floating_shapes + falling_shapes

# Load the mesh distractors
mesh_distractors_config = distractors_config.get("mesh_distractors", {})
floating_meshes, falling_meshes = infinigen_utils.load_mesh_distractors(mesh_distractors_config, rng)
print(f"[SDG] Loaded {len(floating_meshes)} floating mesh distractors")
print(f"[SDG] Loaded {len(falling_meshes)} falling mesh distractors")
mesh_distractors = floating_meshes + falling_meshes
```

Setting Up the Environment and Randomizing Poses

```python
# Setup the environment (add collision, fix lights, etc.) and update the app once to apply the changes
print(f"[SDG] Setting up the environment")
infinigen_utils.setup_env(root_path="/Environment", hide_top_walls=debug_mode)
simulation_app.update()

# Get the location of the prim above which the assets will be randomized
working_area_loc = infinigen_utils.get_matching_prim_location(
    match_string="TableDining", root_path="/Environment"
)

# Move viewport above the working area to get a top-down view of the scene
if debug_mode:
    camera_loc = (working_area_loc[0], working_area_loc[1], working_area_loc[2] + 10)
    set_camera_view(eye=np.array(camera_loc), target=np.array(working_area_loc))

# Get the spawn areas as offseted location ranges from the working area (min_x, min_y, min_z, max_x, max_y, max_z)
print(f"[SDG] Randomizing {len(target_assets)} target assets around the working area")
target_loc_range = infinigen_utils.offset_range((-0.5, -0.5, 1, 0.5, 0.5, 1.5), working_area_loc)
infinigen_utils.randomize_poses(
    target_assets,
    location_range=target_loc_range,
    rotation_range=(0, 360),
    scale_range=(0.95, 1.15),
    rng=rng,
)

# Mesh distractors
print(f"[SDG] Randomizing {len(mesh_distractors)} mesh distractors around the working area")
mesh_loc_range = infinigen_utils.offset_range((-1, -1, 1, 1, 1, 2), working_area_loc)
infinigen_utils.randomize_poses(
    mesh_distractors,
    location_range=mesh_loc_range,
    rotation_range=(0, 360),
    scale_range=(0.3, 1.0),
    rng=rng,
)

# Shape distractors
print(f"[SDG] Randomizing {len(shape_distractors)} shape distractors around the working area")
shape_loc_range = infinigen_utils.offset_range((-1.5, -1.5, 1, 1.5, 1.5, 2), working_area_loc)
infinigen_utils.randomize_poses(
    shape_distractors,
    location_range=shape_loc_range,
    rotation_range=(0, 360),
    scale_range=(0.01, 0.1),
    rng=rng,
)
```

**Explanation:**

* **Loading Assets**: Assets are loaded once at the beginning of the pipeline. The `load_auto_labeled_assets` function automatically generates labels from file names using regex patterns (e.g., `002_banana` becomes `banana`). The `load_manual_labeled_assets` function uses explicitly defined labels. Both functions return separate lists of floating (gravity disabled) and falling (gravity enabled) assets.
* **Environment Setup**: The `setup_env` utility function adds colliders to the environment and hides top walls if `debug_mode` is `true`. Hiding the top walls provides a clear view of the scene during debugging.
* **Working Area Location**: We use `get_matching_prim_location` to find the location of the dining table, which serves as our working area.
* **Randomizing Poses**: The `randomize_poses` function takes explicit `location_range`, `rotation_range`, and `scale_range` parameters. The `offset_range` helper function creates location ranges relative to the working area location.

### Creating Cameras and Render Products

We create multiple cameras to capture images from different viewpoints. Each camera is assigned a render product, which is used by Replicator writers to save data.

Creating Cameras and Render Products

```python
# Create the cameras
cameras = []
num_cameras = capture_config.get("num_cameras", 0)
rep.functional.create.scope(name="Cameras")
for i in range(num_cameras):
    cam_prim = rep.functional.create.camera(parent="/Cameras", name=f"cam_{i}", clipping_range=(0.25, 1000))
    cameras.append(cam_prim)
print(f"[SDG] Created {len(cameras)} cameras")

# Create the render products for the cameras
render_products = []
resolution = capture_config.get("resolution", (1280, 720))
disable_render_products = capture_config.get("disable_render_products", False)
for cam in cameras:
    rp = rep.create.render_product(cam.GetPath(), resolution, name=f"rp_{cam.GetName()}")
    if disable_render_products:
        rp.hydra_texture.set_updates_enabled(False)
    render_products.append(rp)
print(f"[SDG] Created {len(render_products)} render products")
```

**Explanation:**

* We use Replicator’s `rep.functional.create.scope` to create an organizational scope for cameras.
* Cameras are created using `rep.functional.create.camera` which provides a cleaner API for camera creation with configurable clipping range.
* Render products are created using Replicator’s `create.render_product` function.
* If `disable_render_products` is set to `true` in the configuration, we disable the render products during creation. They will be enabled only during capture to save computational resources.

### Setting Up Replicator Writers

We use multiple Replicator writers to collect and store different types of data generated during the simulation. Writers are specified in the configuration file and can include various types such as `BasicWriter`, `DataVisualizationWriter`, `PoseWriter`, and custom writers.

Setting Up Replicator Writers

```python
# Only create the writers if there are render products to attach to
writers = []
if render_products:
    for writer_config in writers_config:
        writer = infinigen_utils.setup_writer(writer_config)
        if writer:
            writer.attach(render_products)
            writers.append(writer)
            print(
                f"[SDG] {writer_config['type']}'s out dir: {writer_config.get('kwargs', {}).get('output_dir', '')}"
            )
print(f"[SDG] Created {len(writers)} writers")
```

**Explanation:**

* Writers are only created if there are render products available to attach to.
* The `setup_writer` utility function initializes writers based on the configuration, handling output directory paths and writer-specific arguments.
* Writers are attached to the render products (cameras) to capture data from the specified viewpoints.
* Multiple writers can be used simultaneously to generate different dataset types.

### Domain Randomization

To enhance the diversity of the dataset, we apply domain randomization to various elements in the scene:

* **Randomizing Object Poses**: Positions, orientations, and scales of assets are randomized within specified ranges.
* **Randomizing Lights**: Scene lights are randomized in terms of position, intensity, and color.
* **Randomizing Dome Light**: The environment dome light is randomized to simulate different lighting conditions.
* **Randomizing Shape Distractor Colors**: Colors of shape distractors are randomized to increase visual diversity.

Creating and Registering Randomizers

```python
# Create lights to randomize in the working area
scene_lights = []
num_scene_lights = capture_config.get("num_scene_lights", 0)
for i in range(num_scene_lights):
    light_prim = stage.DefinePrim(f"/Lights/SphereLight_scene_{i}", "SphereLight")
    scene_lights.append(light_prim)
print(f"[SDG] Created {len(scene_lights)} scene lights")

# Register replicator randomizers and trigger them once
print("[SDG] Registering replicator graph randomizers")
infinigen_utils.register_dome_light_randomizer()
infinigen_utils.register_shape_distractors_color_randomizer(shape_distractors)
```

Triggering Randomizations

```python
print(f"[SDG] Randomizing {len(scene_lights)} scene lights properties and locations around the working area")
lights_loc_range = infinigen_utils.offset_range((-2, -2, 1, 2, 2, 3), working_area_loc)
infinigen_utils.randomize_lights(
    scene_lights,
    location_range=lights_loc_range,
    intensity_range=(500, 2500),
    color_range=(0.1, 0.1, 0.1, 0.9, 0.9, 0.9),
    rng=rng,
)

print("[SDG] Randomizing dome lights")
rep.utils.send_og_event(event_name="randomize_dome_lights")

print("[SDG] Randomizing shape distractor colors")
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
```

**Explanation:**

* **Scene Lights**: Additional sphere lights are created using the USD API (`stage.DefinePrim`) and stored for later randomization.
* **Randomizers Registration**: Custom Replicator graph randomizers for dome lights and shape distractor colors are registered once during setup.
* **Light Randomization**: The `randomize_lights` utility function randomizes light properties (location, intensity, color) within specified ranges.
* **Event-Based Triggering**: Randomizations are triggered using `rep.utils.send_og_event` which sends OmniGraph events to the registered randomizer graphs.

### Running Physics Simulation

We run physics simulations to allow objects to interact naturally within the environment. This involves:

* Running a short simulation to resolve any initial overlaps.
* Capturing images before objects have settled (floating captures).
* Running a longer simulation to let objects fall and settle.
* Capturing images after objects have settled (dropped captures).

Running Physics Simulation

```python
# Run the physics simulation for a few frames to solve any collisions
print("[SDG] Fixing collisions through physics simulation")
simulation_app.update()
infinigen_utils.run_simulation(num_frames=4, render=True)

# Check if the render products need to be enabled for the capture
if disable_render_products:
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(True)

# Check if the render mode needs to be switched to path tracing for the capture
if use_path_tracing:
    print("[SDG] Switching to PathTracing render mode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")

# Capture frames with the objects in the air
for i in range(num_floating_captures_per_env):
    # Check if the total captures have been reached
    if capture_counter >= total_captures:
        break
    # Randomize the camera poses
    print(f"[SDG] Randomizing camera poses ({len(cameras)} cameras)")
    infinigen_utils.randomize_camera_poses(
        cameras, target_assets, camera_distance_to_target_range, polar_angle_range=(0, 75), rng=rng
    )
    print(
        f"[SDG] Capturing floating assets {i+1}/{num_floating_captures_per_env} (total: {capture_counter+1}/{total_captures})"
    )
    rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
    capture_counter += 1

# Check if the render products need to be disabled until the next capture
if disable_render_products:
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(False)

# Check if the render mode needs to be switched back to raytracing until the next capture
if use_path_tracing:
    carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")

print("[SDG] Running the simulation")
infinigen_utils.run_simulation(num_frames=200, render=False)

# Check if the render products need to be enabled for the capture
if disable_render_products:
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(True)

# Check if the render mode needs to be switched to path tracing for the capture
if use_path_tracing:
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")

for i in range(num_dropped_captures_per_env):
    # Check if the total captures have been reached
    if capture_counter >= total_captures:
        break
    # Spawn the cameras with a smaller polar angle to have mostly a top-down view of the objects
    print("[SDG] Randomizing camera poses")
    infinigen_utils.randomize_camera_poses(
        cameras,
        target_assets,
        distance_range=camera_distance_to_target_range,
        polar_angle_range=(0, 45),
        rng=rng,
    )
    print(
        f"[SDG] Capturing dropped assets {i+1}/{num_dropped_captures_per_env} (total: {capture_counter+1}/{total_captures})"
    )
    rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
    capture_counter += 1

# Check if the render products need to be disabled until the next capture
if disable_render_products:
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(False)

# Check if the render mode needs to be switched back to raytracing until the next capture
if use_path_tracing:
    carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")
```

**Explanation:**

* **Initial Simulation**: A short simulation resolves any initial overlaps among assets.
* **Render Product Management**: Render products are enabled only during capture and disabled during simulation to save computational resources.
* **Path Tracing**: When enabled, the render mode switches to PathTracing for higher quality captures and back to RealTimePathTracing during simulation.
* **Floating Captures**: We capture images while assets are still floating, with cameras positioned using larger polar angles (0-75°) for varied viewpoints.
* **Physics Simulation**: A longer simulation (200 frames) allows assets to fall and settle according to physics, without rendering for efficiency.
* **Dropped Captures**: We capture images after assets have settled, using smaller polar angles (0-45°) for mostly top-down views.
* **Capture Counter**: Each capture increments the counter, with early exit checks to respect the total capture limit.

### Capturing Data

We capture data at specified intervals, ensuring that we have a diverse set of images covering various object states and viewpoints.

* **Randomizing Camera Poses**: Cameras are positioned randomly around target assets to capture images from different angles.
* **Triggering Randomizations**: Randomizations are applied at each environment to ensure diversity.

Capturing Data Loop

```python
# Start the SDG loop
env_cycle = cycle(env_urls)
capture_counter = 0
while capture_counter < total_captures:
    # Load the next environment
    env_url = next(env_cycle)

    # Load the new environment
    print(f"[SDG] Loading environment: {env_url}")
    infinigen_utils.load_env(env_url, prim_path="/Environment")

    # Setup the environment (add collision, fix lights, etc.)
    infinigen_utils.setup_env(root_path="/Environment", hide_top_walls=debug_mode)
    simulation_app.update()

    # Get the location of the working area (e.g., dining table)
    working_area_loc = infinigen_utils.get_matching_prim_location(
        match_string="TableDining", root_path="/Environment"
    )

    # Randomize poses for target assets, mesh distractors, shape distractors
    # ... (randomization code as shown in previous snippets)

    # Trigger graph-based randomizers
    rep.utils.send_og_event(event_name="randomize_dome_lights")
    rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")

    # Run physics simulation and capture floating assets
    infinigen_utils.run_simulation(num_frames=4, render=True)
    for i in range(num_floating_captures_per_env):
        if capture_counter >= total_captures:
            break
        infinigen_utils.randomize_camera_poses(cameras, target_assets, ...)
        rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
        capture_counter += 1

    # Run longer simulation for dropped assets
    infinigen_utils.run_simulation(num_frames=200, render=False)
    for i in range(num_dropped_captures_per_env):
        if capture_counter >= total_captures:
            break
        infinigen_utils.randomize_camera_poses(cameras, target_assets, ...)
        rep.orchestrator.step(rt_subframes=rt_subframes, delta_time=0.0)
        capture_counter += 1

# Cleanup: wait for data, detach writers, destroy render products
rep.orchestrator.wait_until_complete()
for writer in writers:
    writer.detach()
for rp in render_products:
    rp.destroy()
print(f"[SDG] Finished, captured {capture_counter * num_cameras} frames")
```

**Explanation:**

* We loop through the environments using `cycle` to repeat environments if needed.
* The `capture_counter` is incremented inside each capture loop (floating and dropped), not at the end of the environment iteration.
* After loading each environment, we call `simulation_app.update()` to apply changes before proceeding.
* Randomizations are triggered using OmniGraph events for each environment.
* After all captures are complete, we wait for the data to be written, then properly cleanup by detaching writers and destroying render products.

## Summary

In this tutorial, you learned how to generate synthetic datasets using Infinigen environments in NVIDIA Omniverse Isaac Sim. The key steps included:

1. **Generating Infinigen Environments**: Using Infinigen to create photorealistic indoor environments.
2. **Understanding Configuration Parameters**: Customizing the simulation and data generation process through configuration files.
3. **Setting Up the Simulation**: Running Isaac Sim as a standalone application and loading Infinigen environments.
4. **Spawning Assets**: Using the Isaac Sim API to place labeled assets and distractors in the environment.
5. **Configuring the SDG Pipeline**: Creating cameras, render products, and using multiple Replicator writers to generate different datasets.
6. **Applying Domain Randomization**: Enhancing dataset diversity through randomizations.
7. **Running Physics Simulations**: Simulating object interactions for realistic scenes.
8. **Capturing and Saving Data**: Collecting images and annotations using multiple Replicator writers.

By following this tutorial, you now have the foundation to create rich, diverse synthetic datasets using procedurally generated environments and advanced randomization techniques.

## Next Steps

With the generated datasets, you can proceed to train machine learning models for tasks like object detection, segmentation, and pose estimation. Consider exploring the [TAO Toolkit](https://docs.nvidia.com/tao/) for training workflows and pre-trained models.

---

# Randomization in Simulation – AMR Navigation

Example of using Isaac Sim and Replicator to capture synthetic data from simulated environments (AMR Navigation).

## Learning Objectives

The goal of this tutorial is to demonstrate how to setup an Isaac Sim simulation scenario together with the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension to capture synthetic data using diverse randomization techniques.

In this tutorial you:

* Implement scene randomizations using USD / Isaac Sim APIs:

  > + Randomize poses of assets in the scene
  > + Switch between different background environments
* Collect synthetic data at specific simulation events with Replicator
* Create and destroy render products on the fly to improve runtime performance
* Create and destroy Replicator capture graphs within the same simulation instance

### Prerequisites

* Familiarity with USD / Isaac Sim APIs for scene creation and manipulation.
* Familiarity with [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") and its [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)").
* Basic understanding of [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") for the navigation implementation.
* Running simulations as [Standalone Applications](Workflows.md) or via the [Script Editor](Development_Tools.md).

## Scenario

This tutorial uses the Nova Carter robot equipped with an [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") navigation stack, notably without collision avoidance features. The navigation stack constantly drives the robot towards a designated Xform target (`<..>/targetXform`), positioned at the location of the randomized objects of interest. As the robot comes in the proximity of the object of interest, a synthetic data generation (SDG) pipeline is triggered to capture data from its two main camera sensors. After the data is captured the objects of interest are re-randomized and the simulation continues. After a certain number of frames (`env_interval`) the background environment is changed as well. After `num_frames` the application terminates.

The `use_temp_rp` flag is used to provide an option to use temporary render products to improve the runtime performance. This speeds up the simulation by only using the render products when capturing the data, thus avoiding the overhead of rendering the sensor views when not capturing data.

The scenario uses the left and right camera sensors of Nova Carter (`<..>/stereo_cam_<left/right>_sensor_frame/camera_sensor_<left/right>`) to collect **LdrColor** (rgb) [annotator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") data using Replicator. By default, the data is written to `<working_dir>/_out_nav_sdg_demo` and runs for `num_frames=9` iterations.

Furthermore, it changes the background environment every `env_interval=3` captured frames. The `use_temp_rp` flag can be used to optimize performance by disabling the sensor render products during simulation and temporarily enabling them during data capture.

The following image provides an illustration of the resulting data from the various environments.

## Implementation

The following section provides an overview and explanation of the implementation and examples on how to run the demo.

Standalone Application

To run the example as a standalone application, use the following command to execute the provided script. The script also accepts several optional arguments to customize its behavior (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/replicator/amr_navigation.py
```

Arguments include:

* `--use_temp_rp` flag to use temporary render products (default: False)
* `--num_frames` the number of frames to be captured (default: 9)
* `--env_interval` the capture interval at which the background environment is changed (default: 3)

For example, to run the application with all the arguments:

```python
./python.sh standalone_examples/replicator/amr_navigation.py --use_temp_rp --num_frames 9 --env_interval 3
```

Standalone Script

```python
"""Generate synthetic data from an AMR navigating to random locations."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import builtins
import os
import random
from itertools import cycle

import carb.eventdispatcher
import carb.settings
import omni.client
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import omni.usd.commands
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

ENV_URLS = [
    "/Isaac/Environments/Grid/default_environment.usd",
    "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "/Isaac/Environments/Grid/gridroom_black.usd",
]

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=9, help="The number of frames to capture")
parser.add_argument("--env_interval", type=int, default=3, help="Interval at which to change the environments")
parser.add_argument("--use_temp_rp", action="store_true", help="Create and destroy render products for each SDG frame")
args, unknown = parser.parse_known_args()

class NavSDGDemo:
    """Demonstration of synthetic data generation using an AMR navigating towards a target."""

    CARTER_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
    DOLLY_URL = "/Isaac/Props/Dolly/dolly.usd"
    PROPS_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics"
    LEFT_CAMERA_REL_PATH = "sensors/front_hawk/left/camera_left"
    RIGHT_CAMERA_REL_PATH = "sensors/front_hawk/right/camera_right"

    def __init__(self) -> None:
        """Initialize the navigation SDG demo with default values."""
        self._carter_chassis = None
        self._carter_nav_target = None
        self._dolly = None
        self._dolly_light = None
        self._props = []
        self._cycled_env_urls = None
        self._env_interval = 1
        self._timeline = None
        self._timeline_sub = None
        self._stage_event_sub = None
        self._stage = None
        self._trigger_distance = 2.0
        self._num_frames = 0
        self._frame_counter = 0
        self._writer = None
        self._out_dir = None
        self._render_products = []
        self._use_temp_rp = False
        self._in_running_state = False

    def start(
        self,
        num_frames: int = 10,
        out_dir: str | None = None,
        env_urls: list[str] = [],
        env_interval: int = 3,
        use_temp_rp: bool = False,
        seed: int | None = None,
    ) -> None:
        """Start the SDG demo with the given configuration."""
        print(f"[SDG] Starting")
        if seed is not None:
            rep.set_global_seed(seed)
            random.seed(seed)
        self._num_frames = num_frames
        self._out_dir = out_dir if out_dir is not None else os.path.join(os.getcwd(), "_out_nav_sdg_demo")
        self._cycled_env_urls = cycle(env_urls)
        self._env_interval = env_interval
        self._use_temp_rp = use_temp_rp
        self._frame_counter = 0
        self._trigger_distance = 2.0
        self._load_env()
        self._randomize_dolly_pose()
        self._randomize_dolly_light()
        self._randomize_prop_poses()
        self._setup_sdg()
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.play()
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="amr_navigation.NavSDGDemo._on_timeline_event",
        )
        self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSING),
            on_event=self._on_stage_closing_event,
            observer_name="amr_navigation.NavSDGDemo._on_stage_closing_event",
        )
        self._in_running_state = True

    def clear(self) -> None:
        """Reset all state variables and unsubscribe from events."""
        self._cycled_env_urls = None
        self._carter_chassis = None
        self._carter_nav_target = None
        self._dolly = None
        self._dolly_light = None
        self._timeline = None
        self._frame_counter = 0
        self._stage_event_sub = None
        self._timeline_sub = None
        self._clear_sdg_render_products()
        self._stage = None
        self._in_running_state = False

    def is_running(self) -> bool:
        """Return whether the SDG demo is currently running."""
        return self._in_running_state

    def _is_running_in_script_editor(self) -> bool:
        """Return whether the script is running in the Isaac Sim script editor."""
        return builtins.ISAAC_LAUNCHED_FROM_TERMINAL is True

    def _on_stage_closing_event(self, e: carb.eventdispatcher.Event):
        """Handle stage closing event by clearing state."""
        self.clear()

    def _load_env(self) -> None:
        """Create a new stage and load environment, robot, dolly, light, and props."""
        create_new_stage()
        self._stage = omni.usd.get_context().get_stage()
        rep.functional.physics.create_physics_scene(
            "/PhysicsScene", enableCCD=True, broadphaseType="MBP", enableGPUDynamics=False
        )

        # Environment
        assets_root_path = get_assets_root_path()
        rep.functional.create.reference(usd_path=assets_root_path + next(self._cycled_env_urls), name="Environment")

        # Nova Carter
        rep.functional.create.scope(name="NavWorld")
        carter = rep.functional.create.reference(
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            usd_path=assets_root_path + self.CARTER_URL,
            parent="/NavWorld",
            name="CarterNav",
        )

        # Iterate children until targetXform (for navigation target) and chassis_link (for current location) are found
        for child in carter.GetChildren():
            if child.GetName() == "targetXform":
                self._carter_nav_target = child
                break
        for child in carter.GetChildren():
            if child.GetName() == "chassis_link":
                self._carter_chassis = child
                break

        # Dolly
        self._dolly = rep.functional.create.reference(
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            usd_path=assets_root_path + self.DOLLY_URL,
            parent="/NavWorld",
            name="Dolly",
        )

        # # Add colliders to the dolly and its geometry primitives
        for desc_prim in self._dolly.GetChildren():
            if desc_prim.IsA(UsdGeom.Gprim):
                rep.functional.physics.apply_rigid_body(desc_prim)

        # Light
        self._dolly_light = rep.functional.create.sphere_light(
            position=(0, 0, 0),
            intensity=250000,
            radius=0.3,
            color=(1.0, 1.0, 1.0),
            parent="/NavWorld",
            name="DollyLight",
        )

        # Props
        props_urls = []
        props_folder_path = assets_root_path + self.PROPS_URL
        result, entries = omni.client.list(props_folder_path)
        if result != omni.client.Result.OK:
            carb.log_error(f"Could not list assets in path: {props_folder_path}")
            return
        for entry in entries:
            _, ext = os.path.splitext(entry.relative_path)
            if ext == ".usd":
                props_urls.append(f"{props_folder_path}/{entry.relative_path}")

        cycled_props_url = cycle(props_urls)
        for i in range(15):
            prop_url = next(cycled_props_url)
            prop_name = os.path.splitext(os.path.basename(prop_url))[0]
            path = f"/NavWorld/Props/Prop_{prop_name}_{i}"
            prim = self._stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(prop_url)
            self._props.append(prim)

    def _randomize_dolly_pose(self) -> None:
        """Set random dolly position ensuring minimum distance from Carter."""
        min_dist_from_carter = 4
        carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
        for _ in range(100):
            x, y = random.uniform(-6, 6), random.uniform(-6, 6)
            dist = (Gf.Vec2f(x, y) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
            if dist > min_dist_from_carter:
                self._dolly.GetAttribute("xformOp:translate").Set((x, y, 0))
                self._carter_nav_target.GetAttribute("xformOp:translate").Set((x, y, 0))
                break
        self._dolly.GetAttribute("xformOp:rotateXYZ").Set((0, 0, random.uniform(-180, 180)))

    def _randomize_dolly_light(self) -> None:
        """Position light above dolly with random color."""
        dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
        self._dolly_light.GetAttribute("xformOp:translate").Set(dolly_loc + (0, 0, 3))
        self._dolly_light.GetAttribute("inputs:color").Set(
            (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        )

    def _randomize_prop_poses(self) -> None:
        """Stack props above the dolly with random horizontal offsets."""
        spawn_loc = self._dolly.GetAttribute("xformOp:translate").Get()
        spawn_loc[2] = spawn_loc[2] + 0.5
        for prop in self._props:
            prop.GetAttribute("xformOp:translate").Set(spawn_loc + (random.uniform(-1, 1), random.uniform(-1, 1), 0))
            spawn_loc[2] = spawn_loc[2] + 0.2

    def _setup_sdg(self) -> None:
        """Configure SDG settings, camera parameters, writer, and render products."""
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        # Set camera sensors fStop to 0.0 to get well lit sharp images
        left_camera_path = self._carter_chassis.GetPath().AppendPath(self.LEFT_CAMERA_REL_PATH)
        left_camera_prim = self._stage.GetPrimAtPath(left_camera_path)
        left_camera_prim.GetAttribute("fStop").Set(0.0)
        right_camera_path = self._carter_chassis.GetPath().AppendPath(self.RIGHT_CAMERA_REL_PATH)
        right_camera_prim = self._stage.GetPrimAtPath(right_camera_path)
        right_camera_prim.GetAttribute("fStop").Set(0.0)

        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=self._out_dir)
        print(f"[SDG] Writing data to: {self._out_dir}")
        self._writer = rep.writers.get("BasicWriter")
        self._writer.initialize(backend=backend, rgb=True)
        self._setup_sdg_render_products()

    def _setup_sdg_render_products(self) -> None:
        """Create and attach render products for left and right cameras."""
        print(f"[SDG] Creating SDG render products")
        left_camera_path = self._carter_chassis.GetPath().AppendPath(self.LEFT_CAMERA_REL_PATH)
        rp_left = rep.create.render_product(
            str(left_camera_path),
            (1024, 1024),
            name="left_sensor",
            force_new=True,
        )
        right_camera_path = self._carter_chassis.GetPath().AppendPath(self.RIGHT_CAMERA_REL_PATH)
        rp_right = rep.create.render_product(
            str(right_camera_path),
            (1024, 1024),
            name="right_sensor",
            force_new=True,
        )
        self._render_products = [rp_left, rp_right]
        # For better performance the render products can be disabled when not in use, and re-enabled only during SDG
        if self._use_temp_rp:
            self._disable_render_products()
        self._writer.attach(self._render_products)

    def _clear_sdg_render_products(self) -> None:
        """Detach writer and destroy all render products."""
        print(f"[SDG] Clearing SDG render products")
        if self._writer:
            self._writer.detach()
        for rp in self._render_products:
            rp.destroy()
        self._render_products.clear()
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    def _enable_render_products(self) -> None:
        """Enable texture updates on all render products."""
        print(f"[SDG] Enabling render products for SDG..")
        for rp in self._render_products:
            rp.hydra_texture.set_updates_enabled(True)

    def _disable_render_products(self) -> None:
        """Disable texture updates on all render products."""
        print(f"[SDG] Disabling render products (enabled only during SDG)..")
        for rp in self._render_products:
            rp.hydra_texture.set_updates_enabled(False)

    def _run_sdg(self) -> None:
        """Execute one SDG capture step synchronously."""
        if self._use_temp_rp:
            self._enable_render_products()
        rep.orchestrator.step(rt_subframes=16)
        if self._use_temp_rp:
            self._disable_render_products()

    async def _run_sdg_async(self) -> None:
        """Execute one SDG capture step asynchronously."""
        if self._use_temp_rp:
            self._enable_render_products()
        await rep.orchestrator.step_async(rt_subframes=16)
        if self._use_temp_rp:
            self._disable_render_products()

    def _load_next_env(self) -> None:
        """Replace current environment with the next one from the cycle."""
        if self._stage.GetPrimAtPath("/Environment"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Environment"])
        assets_root_path = get_assets_root_path()
        rep.functional.create.scope(name="Environment")
        rep.functional.create.reference(usd_path=assets_root_path + next(self._cycled_env_urls), name="Environment")

    def _on_sdg_done(self, task) -> None:
        """Callback invoked when async SDG step completes."""
        self._setup_next_frame()

    def _setup_next_frame(self) -> None:
        """Prepare scene for next frame or finish if all frames captured."""
        self._frame_counter += 1
        if self._frame_counter >= self._num_frames:
            print(f"[SDG] Finished")
            # Make sure the data has been written to disk before clearing the state
            if self._is_running_in_script_editor():
                import asyncio

                task = asyncio.ensure_future(rep.orchestrator.wait_until_complete_async())
                task.add_done_callback(lambda t: self.clear())
            else:
                rep.orchestrator.wait_until_complete()
                self.clear()
            return

        self._randomize_dolly_pose()
        self._randomize_dolly_light()
        self._randomize_prop_poses()
        if self._frame_counter % self._env_interval == 0:
            self._load_next_env()
        # Set a new random distance from which to take capture the next frame
        self._trigger_distance = random.uniform(1.75, 2.5)
        self._timeline.play()
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="amr_navigation.NavSDGDemo._on_timeline_event",
        )

    def _on_timeline_event(self, e: carb.eventdispatcher.Event):
        """Check distance to dolly and trigger SDG capture when close enough."""
        carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
        dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
        dist = (Gf.Vec2f(dolly_loc[0], dolly_loc[1]) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
        if dist < self._trigger_distance:
            print(f"[SDG] Starting SDG for frame no. {self._frame_counter}")
            self._timeline.pause()
            if self._is_running_in_script_editor():
                import asyncio

                task = asyncio.ensure_future(self._run_sdg_async())
                task.add_done_callback(self._on_sdg_done)
            else:
                self._run_sdg()
                self._setup_next_frame()

out_dir = os.path.join(os.getcwd(), "_out_nav_sdg_demo", "")
nav_demo = NavSDGDemo()
nav_demo.start(
    num_frames=args.num_frames,
    out_dir=out_dir,
    env_urls=ENV_URLS,
    env_interval=args.env_interval,
    use_temp_rp=args.use_temp_rp,
    seed=22,
)

while simulation_app.is_running() and nav_demo.is_running():
    simulation_app.update()

simulation_app.close()
```

Script Editor

To run the example from the script editor, the following code must be executed:

Script Editor Script

```python
import asyncio
import builtins
import os
import random
from itertools import cycle

import carb.settings
import omni.client
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
import omni.usd.commands
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

ENV_URLS = [
    "/Isaac/Environments/Grid/default_environment.usd",
    "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "/Isaac/Environments/Grid/gridroom_black.usd",
]
NUM_FRAMES = 9
ENV_INTERVAL = 3
USE_TEMP_RP = True

class NavSDGDemo:
    """Demonstration of synthetic data generation using an AMR navigating towards a target."""

    CARTER_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
    DOLLY_URL = "/Isaac/Props/Dolly/dolly.usd"
    PROPS_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics"
    LEFT_CAMERA_REL_PATH = "sensors/front_hawk/left/camera_left"
    RIGHT_CAMERA_REL_PATH = "sensors/front_hawk/right/camera_right"

    def __init__(self) -> None:
        """Initialize the navigation SDG demo with default values."""
        self._carter_chassis = None
        self._carter_nav_target = None
        self._dolly = None
        self._dolly_light = None
        self._props = []
        self._cycled_env_urls = None
        self._env_interval = 1
        self._timeline = None
        self._timeline_sub = None
        self._stage_event_sub = None
        self._stage = None
        self._trigger_distance = 2.0
        self._num_frames = 0
        self._frame_counter = 0
        self._writer = None
        self._out_dir = None
        self._render_products = []
        self._use_temp_rp = False
        self._in_running_state = False
        self._completion_event = None

    async def run_async(
        self,
        num_frames: int = 10,
        out_dir: str | None = None,
        env_urls: list[str] = [],
        env_interval: int = 3,
        use_temp_rp: bool = False,
        seed: int | None = None,
    ) -> None:
        """Run the SDG demo asynchronously and wait for completion."""
        self._completion_event = asyncio.Event()
        self.start(
            num_frames=num_frames,
            out_dir=out_dir,
            env_urls=env_urls,
            env_interval=env_interval,
            use_temp_rp=use_temp_rp,
            seed=seed,
        )
        await self._completion_event.wait()

    def start(
        self,
        num_frames: int = 10,
        out_dir: str | None = None,
        env_urls: list[str] = [],
        env_interval: int = 3,
        use_temp_rp: bool = False,
        seed: int | None = None,
    ) -> None:
        """Start the SDG demo with the given configuration."""
        print(f"[SDG] Starting")
        if seed is not None:
            rep.set_global_seed(seed)
            random.seed(seed)
        self._num_frames = num_frames
        self._out_dir = out_dir if out_dir is not None else os.path.join(os.getcwd(), "_out_nav_sdg_demo")
        self._cycled_env_urls = cycle(env_urls)
        self._env_interval = env_interval
        self._use_temp_rp = use_temp_rp
        self._frame_counter = 0
        self._trigger_distance = 2.0
        self._load_env()
        self._randomize_dolly_pose()
        self._randomize_dolly_light()
        self._randomize_prop_poses()
        self._setup_sdg()
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline.play()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.CURRENT_TIME_TICKED), self._on_timeline_event
        )
        self._stage_event_sub = (
            omni.usd.get_context()
            .get_stage_event_stream()
            .create_subscription_to_pop_by_type(int(omni.usd.StageEventType.CLOSING), self._on_stage_closing_event)
        )
        self._in_running_state = True

    def clear(self) -> None:
        """Reset all state variables and unsubscribe from events."""
        self._cycled_env_urls = None
        self._carter_chassis = None
        self._carter_nav_target = None
        self._dolly = None
        self._dolly_light = None
        self._timeline = None
        self._frame_counter = 0
        if self._stage_event_sub:
            self._stage_event_sub.unsubscribe()
        self._stage_event_sub = None
        if self._timeline_sub:
            self._timeline_sub.unsubscribe()
        self._timeline_sub = None
        self._clear_sdg_render_products()
        self._stage = None
        self._in_running_state = False
        # Signal completion for async waiters
        if self._completion_event:
            self._completion_event.set()
            self._completion_event = None

    def is_running(self) -> bool:
        """Return whether the SDG demo is currently running."""
        return self._in_running_state

    def _is_running_in_script_editor(self) -> bool:
        """Return whether the script is running in the Isaac Sim script editor."""
        return builtins.ISAAC_LAUNCHED_FROM_TERMINAL is True

    def _on_stage_closing_event(self, e: carb.events.IEvent) -> None:
        """Handle stage closing event by clearing state."""
        self.clear()

    def _load_env(self) -> None:
        """Create a new stage and load environment, robot, dolly, light, and props."""
        create_new_stage()
        self._stage = omni.usd.get_context().get_stage()
        rep.functional.physics.create_physics_scene(
            "/PhysicsScene", enableCCD=True, broadphaseType="MBP", enableGPUDynamics=False
        )

        # Environment
        assets_root_path = get_assets_root_path()
        rep.functional.create.reference(usd_path=assets_root_path + next(self._cycled_env_urls), name="Environment")

        # Nova Carter
        rep.functional.create.scope(name="NavWorld")
        carter = rep.functional.create.reference(
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            usd_path=assets_root_path + self.CARTER_URL,
            parent="/NavWorld",
            name="CarterNav",
        )

        # Iterate children until targetXform (for navigation target) and chassis_link (for current location) are found
        for child in carter.GetChildren():
            if child.GetName() == "targetXform":
                self._carter_nav_target = child
                break
        for child in carter.GetChildren():
            if child.GetName() == "chassis_link":
                self._carter_chassis = child
                break

        # Dolly
        self._dolly = rep.functional.create.reference(
            position=(0, 0, 0),
            rotation=(0, 0, 0),
            usd_path=assets_root_path + self.DOLLY_URL,
            parent="/NavWorld",
            name="Dolly",
        )

        # Add colliders to the dolly and its geometry primitives
        for desc_prim in self._dolly.GetChildren():
            if desc_prim.IsA(UsdGeom.Gprim):
                rep.functional.physics.apply_rigid_body(desc_prim)

        # Light
        self._dolly_light = rep.functional.create.sphere_light(
            position=(0, 0, 0),
            intensity=250000,
            radius=0.3,
            color=(1.0, 1.0, 1.0),
            parent="/NavWorld",
            name="DollyLight",
        )

        # Props
        props_urls = []
        props_folder_path = assets_root_path + self.PROPS_URL
        result, entries = omni.client.list(props_folder_path)
        if result != omni.client.Result.OK:
            carb.log_error(f"Could not list assets in path: {props_folder_path}")
            return
        for entry in entries:
            _, ext = os.path.splitext(entry.relative_path)
            if ext == ".usd":
                props_urls.append(f"{props_folder_path}/{entry.relative_path}")

        cycled_props_url = cycle(props_urls)
        for i in range(15):
            prop_url = next(cycled_props_url)
            prop_name = os.path.splitext(os.path.basename(prop_url))[0]
            path = f"/NavWorld/Props/Prop_{prop_name}_{i}"
            prim = self._stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(prop_url)
            self._props.append(prim)

    def _randomize_dolly_pose(self) -> None:
        """Set random dolly position ensuring minimum distance from Carter."""
        min_dist_from_carter = 4
        carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
        for _ in range(100):
            x, y = random.uniform(-6, 6), random.uniform(-6, 6)
            dist = (Gf.Vec2f(x, y) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
            if dist > min_dist_from_carter:
                self._dolly.GetAttribute("xformOp:translate").Set((x, y, 0))
                self._carter_nav_target.GetAttribute("xformOp:translate").Set((x, y, 0))
                break
        self._dolly.GetAttribute("xformOp:rotateXYZ").Set((0, 0, random.uniform(-180, 180)))

    def _randomize_dolly_light(self) -> None:
        """Position light above dolly with random color."""
        dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
        self._dolly_light.GetAttribute("xformOp:translate").Set(dolly_loc + (0, 0, 3))
        self._dolly_light.GetAttribute("inputs:color").Set(
            (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        )

    def _randomize_prop_poses(self) -> None:
        """Stack props above the dolly with random horizontal offsets."""
        spawn_loc = self._dolly.GetAttribute("xformOp:translate").Get()
        spawn_loc[2] = spawn_loc[2] + 0.5
        for prop in self._props:
            prop.GetAttribute("xformOp:translate").Set(spawn_loc + (random.uniform(-1, 1), random.uniform(-1, 1), 0))
            spawn_loc[2] = spawn_loc[2] + 0.2

    def _setup_sdg(self) -> None:
        """Configure SDG settings, camera parameters, writer, and render products."""
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        # Set camera sensors fStop to 0.0 to get well lit sharp images
        left_camera_path = self._carter_chassis.GetPath().AppendPath(self.LEFT_CAMERA_REL_PATH)
        left_camera_prim = self._stage.GetPrimAtPath(left_camera_path)
        left_camera_prim.GetAttribute("fStop").Set(0.0)
        right_camera_path = self._carter_chassis.GetPath().AppendPath(self.RIGHT_CAMERA_REL_PATH)
        right_camera_prim = self._stage.GetPrimAtPath(right_camera_path)
        right_camera_prim.GetAttribute("fStop").Set(0.0)

        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=self._out_dir)
        print(f"[SDG] Writing data to: {self._out_dir}")
        self._writer = rep.writers.get("BasicWriter")
        self._writer.initialize(backend=backend, rgb=True)
        self._setup_sdg_render_products()

    def _setup_sdg_render_products(self) -> None:
        """Create and attach render products for left and right cameras."""
        print(f"[SDG] Creating SDG render products")
        left_camera_path = self._carter_chassis.GetPath().AppendPath(self.LEFT_CAMERA_REL_PATH)
        rp_left = rep.create.render_product(
            str(left_camera_path),
            (1024, 1024),
            name="left_sensor",
            force_new=True,
        )
        right_camera_path = self._carter_chassis.GetPath().AppendPath(self.RIGHT_CAMERA_REL_PATH)
        rp_right = rep.create.render_product(
            str(right_camera_path),
            (1024, 1024),
            name="right_sensor",
            force_new=True,
        )
        self._render_products = [rp_left, rp_right]
        # For better performance the render products can be disabled when not in use, and re-enabled only during SDG
        if self._use_temp_rp:
            self._disable_render_products()
        self._writer.attach(self._render_products)

    def _clear_sdg_render_products(self) -> None:
        """Detach writer and destroy all render products."""
        print(f"[SDG] Clearing SDG render products")
        if self._writer:
            self._writer.detach()
        for rp in self._render_products:
            rp.destroy()
        self._render_products.clear()
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    def _enable_render_products(self) -> None:
        """Enable texture updates on all render products."""
        print(f"[SDG] Enabling render products for SDG..")
        for rp in self._render_products:
            rp.hydra_texture.set_updates_enabled(True)

    def _disable_render_products(self) -> None:
        """Disable texture updates on all render products."""
        print(f"[SDG] Disabling render products (enabled only during SDG)..")
        for rp in self._render_products:
            rp.hydra_texture.set_updates_enabled(False)

    def _run_sdg(self) -> None:
        """Execute one SDG capture step synchronously."""
        if self._use_temp_rp:
            self._enable_render_products()
        rep.orchestrator.step(rt_subframes=16)
        if self._use_temp_rp:
            self._disable_render_products()

    async def _run_sdg_async(self) -> None:
        """Execute one SDG capture step asynchronously."""
        if self._use_temp_rp:
            self._enable_render_products()
        await rep.orchestrator.step_async(rt_subframes=16)
        if self._use_temp_rp:
            self._disable_render_products()

    def _load_next_env(self) -> None:
        """Replace current environment with the next one from the cycle."""
        if self._stage.GetPrimAtPath("/Environment"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Environment"])
        assets_root_path = get_assets_root_path()
        rep.functional.create.scope(name="Environment")
        rep.functional.create.reference(usd_path=assets_root_path + next(self._cycled_env_urls), name="Environment")

    def _on_sdg_done(self, task) -> None:
        """Callback invoked when async SDG step completes."""
        self._setup_next_frame()

    def _setup_next_frame(self) -> None:
        """Prepare scene for next frame or finish if all frames captured."""
        self._frame_counter += 1
        if self._frame_counter >= self._num_frames:
            print(f"[SDG] Finished")
            if self._is_running_in_script_editor():
                task = asyncio.ensure_future(rep.orchestrator.wait_until_complete_async())
                task.add_done_callback(lambda t: self.clear())
            else:
                rep.orchestrator.wait_until_complete()
                self.clear()
            return

        self._randomize_dolly_pose()
        self._randomize_dolly_light()
        self._randomize_prop_poses()
        if self._frame_counter % self._env_interval == 0:
            self._load_next_env()
        # Set a new random distance from which to capture the next frame
        self._trigger_distance = random.uniform(1.75, 2.5)
        self._timeline.play()
        self._timeline_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.CURRENT_TIME_TICKED), self._on_timeline_event
        )

    def _on_timeline_event(self, e: carb.events.IEvent) -> None:
        """Check distance to dolly and trigger SDG capture when close enough."""
        carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
        dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
        dist = (Gf.Vec2f(dolly_loc[0], dolly_loc[1]) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
        if dist < self._trigger_distance:
            print(f"[SDG] Starting SDG for frame no. {self._frame_counter}")
            self._timeline.pause()
            self._timeline_sub.unsubscribe()
            if self._is_running_in_script_editor():
                task = asyncio.ensure_future(self._run_sdg_async())
                task.add_done_callback(self._on_sdg_done)
            else:
                self._run_sdg()
                self._setup_next_frame()

out_dir = os.path.join(os.getcwd(), "_out_nav_sdg_demo", "")
nav_demo = NavSDGDemo()
asyncio.ensure_future(
    nav_demo.run_async(
        num_frames=NUM_FRAMES,
        out_dir=out_dir,
        env_urls=ENV_URLS,
        env_interval=ENV_INTERVAL,
        use_temp_rp=USE_TEMP_RP,
        seed=22,
    )
)
```

Code Explanation

This tab describes each section of the larger sample script that is used for this tutorial. By reviewing the descriptions and code snippets you can understand how the script is working and how you might customize it for your use.

The following snippets can be used to load and start the demo scene. Each of the snippets has an explanation that can be expanded. The snippets and explanations are collapsed so that you can control opening them as you read and work through the tutorial for yourself.

**Running the AMR Navigation SDG Demo**

The following snippet is from the end of the code sample, it runs for the given `num_frames` and changes the background environment every `env_interval`. The output is written to the given `out_dir path`. The `use_temp_rp` parameter can be used to optimize performance by creating render products only for the frames when the data is captured.

The start method loads and runs the demo with the specified parameters, while clear halts the demo and clears any active subscribers and render products. You can use `is_running` to verify whether the demo is still running.

Running the NavSDGDemo Python Script Example

```python
out_dir = os.path.join(os.getcwd(), "_out_nav_sdg_demo", "")
nav_demo = NavSDGDemo()
nav_demo.start(
    num_frames=args.num_frames,
    out_dir=out_dir,
    env_urls=ENV_URLS,
    env_interval=args.env_interval,
    use_temp_rp=args.use_temp_rp,
    seed=22,
)

while simulation_app.is_running() and nav_demo.is_running():
    simulation_app.update()

simulation_app.close()
```

**NavSDGDemo Class and Attributes**

The demo script is wrapped in its own class called `NavSDGDemo`.

NavSDGDemo Class Snippet

```python
class NavSDGDemo:
    """Demonstration of synthetic data generation using an AMR navigating towards a target."""

    CARTER_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
    DOLLY_URL = "/Isaac/Props/Dolly/dolly.usd"
    PROPS_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics"
    LEFT_CAMERA_REL_PATH = "sensors/front_hawk/left/camera_left"
    RIGHT_CAMERA_REL_PATH = "sensors/front_hawk/right/camera_right"

    def __init__(self) -> None:
        """Initialize the navigation SDG demo with default values."""
        self._carter_chassis = None
        self._carter_nav_target = None
        self._dolly = None
        self._dolly_light = None
        self._props = []
        self._cycled_env_urls = None
        self._env_interval = 1
        self._timeline = None
        self._timeline_sub = None
        self._stage_event_sub = None
        self._stage = None
        self._trigger_distance = 2.0
        self._num_frames = 0
        self._frame_counter = 0
        self._writer = None
        self._out_dir = None
        self._render_products = []
        self._use_temp_rp = False
        self._in_running_state = False
```

The attributes of this class include:

* `self._carter_chassis` and `self._carter_nav_target` prims are used to track Nova Carter and its target Xform in the navigation graph
* `self._dolly` is used as the target for the navigation target of Nova Carter and to track the distance to Nova Carter
* `self._dolly_light` randomized light placed above the dolly each captured frame
* `self._props` list of prop prims to place and simulate above the dolly each captured frame
* `self._cycled_env_urls` the paths for the background environments to cycle through
* `self._env_interval` is used to determine after how many frames to change the background environment
* `self._timeline` is used to control (play/pause) the simulation timeline between frame captures
* `self._timeline_sub` is the subscriber to the timeline ticks. It is used as the feedback loop to trigger the synthetic data generation
* `self._stage_event_sub` is a subscriber to stage closing events used to clear the demo in case a new stage is opened
* `self._stage` is used to access the active stage in order to create, access, and delete prims of interest
* `self._trigger_distance` is used to determine the distance between Nova Carter and the dolly at which the synthetic data generation should trigger, the value is randomized after each capture
* `self._num_frames` and `self._frame_counter` are used to track and stop the demo after the given number of frames
* `self._writer` is the writer used to write the synthetic data to disk
* `self._render_products` are the two render products attached to the left and right camera sensors of Nova Carter, the writer is attached to these to access data from the annotators
* `self._use_temp_rp` is a flag, which when set to `True`, causes the demo to disable render products when not capturing. Otherwise the render products are always enabled
* `self._in_running_state` indicates the running state of the demo used to track whether the demo has finished or not
* `self._completion_event` is an asyncio event used for the `run_async` method to signal when the demo has completed

**Workflow and Start Function**

The workflow’s main functions are `start` and the `_on_timeline_event` callback functions. `start` creates a new environment with:

* navigation specific physics scene
* Nova Carter
* navigation graph with the target Xform
* dolly
* randomization light
* props to drop around the dolly

It also creates the timeline subscriber with `_on_timeline_event` as the callback function triggered with each timeline tick. The `_on_timeline_event` function checks if Nova Carter is close enough to the dolly, if so it pauses the simulation, unsubscribes the timeline callback, and triggers the synthetic data generation (SDG). Depending on whether the demo is running in the script editor or as a standalone application it runs the SDG synchronously or asynchronously.

Workflow Snippet

```python
def start(
    self,
    num_frames: int = 10,
    out_dir: str | None = None,
    env_urls: list[str] = [],
    env_interval: int = 3,
    use_temp_rp: bool = False,
    seed: int | None = None,
) -> None:
    """Start the SDG demo with the given configuration."""
    print(f"[SDG] Starting")
    if seed is not None:
        rep.set_global_seed(seed)
        random.seed(seed)
    self._num_frames = num_frames
    self._out_dir = out_dir if out_dir is not None else os.path.join(os.getcwd(), "_out_nav_sdg_demo")
    self._cycled_env_urls = cycle(env_urls)
    self._env_interval = env_interval
    self._use_temp_rp = use_temp_rp
    self._frame_counter = 0
    self._trigger_distance = 2.0
    self._load_env()
    self._randomize_dolly_pose()
    self._randomize_dolly_light()
    self._randomize_prop_poses()
    self._setup_sdg()
    self._timeline = omni.timeline.get_timeline_interface()
    self._timeline.play()
    self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
        on_event=self._on_timeline_event,
        observer_name="amr_navigation.NavSDGDemo._on_timeline_event",
    )
    self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSING),
        on_event=self._on_stage_closing_event,
        observer_name="amr_navigation.NavSDGDemo._on_stage_closing_event",
    )
    self._in_running_state = True
```

```python
def _on_timeline_event(self, e: carb.eventdispatcher.Event):
    """Check distance to dolly and trigger SDG capture when close enough."""
    carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
    dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
    dist = (Gf.Vec2f(dolly_loc[0], dolly_loc[1]) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
    if dist < self._trigger_distance:
        print(f"[SDG] Starting SDG for frame no. {self._frame_counter}")
        self._timeline.pause()
        if self._is_running_in_script_editor():
            import asyncio

            task = asyncio.ensure_future(self._run_sdg_async())
            task.add_done_callback(self._on_sdg_done)
        else:
            self._run_sdg()
            self._setup_next_frame()
```

**Randomizations Explanation**

To randomize the environment before the synthetic data capture, the following functions are used:

* `_randomize_dolly_pose`: places the dolly at a random pose with a given minimum distance from Nova Carter. After such a pose is found, the navigation target is placed at the dolly’s position.
* `_randomize_dolly_light`: places the dolly light above the dolly with a new random color.
* `_randomize_prop_poses`: places the props above the dolly at random locations, which eventually starts to fall after the simulation starts.

Randomizations Snippet

```python
def _randomize_dolly_pose(self) -> None:
    """Set random dolly position ensuring minimum distance from Carter."""
    min_dist_from_carter = 4
    carter_loc = self._carter_chassis.GetAttribute("xformOp:translate").Get()
    for _ in range(100):
        x, y = random.uniform(-6, 6), random.uniform(-6, 6)
        dist = (Gf.Vec2f(x, y) - Gf.Vec2f(carter_loc[0], carter_loc[1])).GetLength()
        if dist > min_dist_from_carter:
            self._dolly.GetAttribute("xformOp:translate").Set((x, y, 0))
            self._carter_nav_target.GetAttribute("xformOp:translate").Set((x, y, 0))
            break
    self._dolly.GetAttribute("xformOp:rotateXYZ").Set((0, 0, random.uniform(-180, 180)))

def _randomize_dolly_light(self) -> None:
    """Position light above dolly with random color."""
    dolly_loc = self._dolly.GetAttribute("xformOp:translate").Get()
    self._dolly_light.GetAttribute("xformOp:translate").Set(dolly_loc + (0, 0, 3))
    self._dolly_light.GetAttribute("inputs:color").Set(
        (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    )

def _randomize_prop_poses(self) -> None:
    """Stack props above the dolly with random horizontal offsets."""
    spawn_loc = self._dolly.GetAttribute("xformOp:translate").Get()
    spawn_loc[2] = spawn_loc[2] + 0.5
    for prop in self._props:
        prop.GetAttribute("xformOp:translate").Set(spawn_loc + (random.uniform(-1, 1), random.uniform(-1, 1), 0))
        spawn_loc[2] = spawn_loc[2] + 0.2
```

**Synthetic Data Generation (SDG) Explanation**

When executing the synthetic data generation (SDG) pipeline the `rep.orchestrator.step` function is called to initiate the data capture and the execution of the writer’s write function.

Depending on the value of the `use_temp_rp` flag, the sensor’s render products are handled differently:

* If set to `True`, the render products are only enabled during data capture.
* `False` is the default. It renders the render products and processes every frame.

Synthetic Data Generation (SDG) Snippet

```python
def _run_sdg(self) -> None:
    """Execute one SDG capture step synchronously."""
    if self._use_temp_rp:
        self._enable_render_products()
    rep.orchestrator.step(rt_subframes=16)
    if self._use_temp_rp:
        self._disable_render_products()
```

**Next Frame Explanation**

After the synthetic data generation (SDG) completes, the `_setup_next_frame` function prepares the simulation for the next frame. This involves incrementing the frame counter (`self._frame_counter`), randomizing the dolly, dolly light, and props. Then changing the background environment, if the `env_interval` is reached. Additionally the timeline and its subscriber are re-started.

If the `_num_frames` is reached the demo makes sure the the writer backend is finished with writing the data to disk (`rep.orchestrator.wait_until_complete`) and clears the demo.

Next Frame Snippet

```python
def _setup_next_frame(self) -> None:
    """Prepare scene for next frame or finish if all frames captured."""
    self._frame_counter += 1
    if self._frame_counter >= self._num_frames:
        print(f"[SDG] Finished")
        # Make sure the data has been written to disk before clearing the state
        if self._is_running_in_script_editor():
            import asyncio

            task = asyncio.ensure_future(rep.orchestrator.wait_until_complete_async())
            task.add_done_callback(lambda t: self.clear())
        else:
            rep.orchestrator.wait_until_complete()
            self.clear()
        return

    self._randomize_dolly_pose()
    self._randomize_dolly_light()
    self._randomize_prop_poses()
    if self._frame_counter % self._env_interval == 0:
        self._load_next_env()
    # Set a new random distance from which to take capture the next frame
    self._trigger_distance = random.uniform(1.75, 2.5)
    self._timeline.play()
    self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
        on_event=self._on_timeline_event,
        observer_name="amr_navigation.NavSDGDemo._on_timeline_event",
    )
```

---

# Randomization in Simulation – UR10 Palletizing

Example of using Isaac Sim and Replicator to capture synthetic data from simulated environments (UR10 palletizing).

## Learning Objectives

The goal of this tutorial is to provide an example on how to extend an existing Isaac Sim simulation to trigger a synthetic data generation (SDG) pipeline to randomize the environment and collect synthetic data at specific simulation events using the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension.

Note

The tutorial makes sure that the SDG pipeline does not change the outcome of the running simulation and cleans up its changes after each capture.

This tutorial teaches you to:

* Collect synthetic data at specific simulation events with Replicator:

  > + Using annotators to collect the data and manually write it to disk
  > + Using writers to implicitly write the data to disk
* Setup various Replicator randomization graphs to:

  > + Randomize lights around the object of interest
  > + Randomize materials and textures of objects of interest running at different rates
* Create and destroy Replicator randomization and capture graphs within the same simulation instance
* Switch between different rendering modes on the fly
* Create and destroy render products on the fly to improve runtime performance

## Prerequisites

* Familiarity with the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension and its [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") and [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)").
* Familiarity with Replicator [randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)") and [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") for a better understanding of the randomization pipeline.
* Executing code from the [Script Editor](Development_Tools.md).

## Scenario

For this tutorial, you build on top of the UR10 palletizing demo scene, which is programmatically loaded and started by the provided script.

The demo scene depicts a simple palletizing scenario where the UR10 robot picks up bins from a conveyor belt and places them on a pallet.

For bins that are flipped, the robot flips them right side up with a helper object before placing them on the pallet.

In the above images, data collected from the actions in the left side image belong to the **bin flip scenario**.

In the above images, data collected from the right side image belongs to the **bin on pallet scenario**.

For each frame in this scenario, the camera pose is iterated through in a predefined sequence, while the custom lights’ parameters are randomized. Data is generated for each manipulated bin in the palletizing demo scene.

The events for which synthetic data are collected are:

* When the bin is placed on the flipping helper object
* When the bin is placed on the pallet (or on another bin that is already on the pallet)

Below, in each captured frame the bin colors are randomized. At a lower randomization rate, the camera poses and pallet textures are also randomized.

The [annotator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") data collected by the scenario includes the **LdrColor** (rgb) and **instance segmentation**.

The data is directly accessed from the annotators and saved to disk using custom helper functions.

The data is written to disk using a built-in Replicator [writer](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)") (`BasicWriter`).

## Implementation

Script Editor

The example can be run from UI using the [Script Editor](Development_Tools.md):

Full Script Editor Script

```python
import asyncio
import json
import os

import carb.settings
import omni
import omni.kit.app
import omni.kit.commands
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.bounds import create_bbox_cache
from isaacsim.storage.native import get_assets_root_path
from omni.physx import get_physx_scene_query_interface
from omni.replicator.core.functional import write_image
from pxr import UsdShade

DEFAULT_NUM_CAPTURES = 4  # Number bins to capture
DEFAULT_BIN_FLIP_FRAMES = 2  # Number of frames to capture for the bin flip scenario
DEFAULT_PALLET_FRAMES = 2  # Number of frames to capture for the pallet scenario
MAX_BINS = 36  # Maximum number of bins available in the scene

class PalletizingSDGDemo:
    BINS_FOLDER_PATH = "/World/Ur10Table/bins"
    FLIP_HELPER_PATH = "/World/Ur10Table/pallet_holder"
    PALLET_PRIM_MESH_PATH = "/World/Ur10Table/pallet/Xform/Mesh_015"

    def __init__(self):
        # There are 36 bins in total
        self._bin_counter = 0
        self._num_captures = MAX_BINS
        self._bin_flip_frames = DEFAULT_BIN_FLIP_FRAMES
        self._pallet_frames = DEFAULT_PALLET_FRAMES
        self._stage = None
        self._active_bin = None

        # Cleanup in case the user closes the stage
        self._stage_event_sub = None

        # Simulation state flags
        self._in_running_state = False
        self._bin_flip_scenario_done = False

        # Used to pause/resume the simulation
        self._timeline = None

        # Used to actively track the active bins surroundings (e.g., in contact with pallet)
        self._timeline_sub = None
        self._overlap_extent = None

        # SDG
        self._rep_camera = None
        self._output_dir = os.path.join(os.getcwd(), "_out_palletizing_sdg_demo")
        print(f"[PalletizingSDGDemo] Output directory: {self._output_dir}")

    def start(self, num_captures, bin_flip_frames, pallet_frames):
        self._num_captures = num_captures if 1 <= num_captures <= 36 else 36
        self._bin_flip_frames = bin_flip_frames
        self._pallet_frames = pallet_frames
        if self._init():
            self._start()

    def is_running(self):
        return self._in_running_state

    def _init(self):
        self._stage = omni.usd.get_context().get_stage()
        self._active_bin = self._stage.GetPrimAtPath(f"{self.BINS_FOLDER_PATH}/bin_{self._bin_counter}")

        if not self._active_bin:
            print("[PalletizingSDGDemo] Could not find bin, make sure the palletizing demo is loaded..")
            return False

        bb_cache = create_bbox_cache()
        half_ext = bb_cache.ComputeLocalBound(self._active_bin).GetRange().GetSize() * 0.5
        self._overlap_extent = carb.Float3(half_ext[0], half_ext[1], half_ext[2] * 1.1)

        self._timeline = omni.timeline.get_timeline_interface()
        if not self._timeline.is_playing():
            print("[PalletizingSDGDemo] Please start the palletizing demo first..")
            return False

        # Disable capture on play for replicator, data capture will be triggered manually
        rep.orchestrator.set_capture_on_play(False)

        # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
        carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

        # Clear any previously generated SDG graphs
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

        return True

    def _start(self):
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_timeline_event",
        )
        self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSING),
            on_event=self._on_stage_closing_event,
            observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_stage_closing_event",
        )
        self._in_running_state = True
        print("[PalletizingSDGDemo] Starting the palletizing SDG demo..")

    def clear(self):
        if self._timeline_sub:
            self._timeline_sub.reset()
            self._timeline_sub = None
        if self._stage_event_sub:
            self._stage_event_sub.reset()
            self._stage_event_sub = None
        self._in_running_state = False
        self._bin_counter = 0
        self._active_bin = None
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    def _on_stage_closing_event(self, e: carb.eventdispatcher.Event):
        # Make sure the subscribers are unsubscribed for new stages
        self.clear()

    def _on_timeline_event(self, e: carb.eventdispatcher.Event):
        self._check_bin_overlaps()

    def _check_bin_overlaps(self):
        bin_pose = omni.usd.get_world_transform_matrix(self._active_bin)
        origin = bin_pose.ExtractTranslation()
        quat_gf = bin_pose.ExtractRotation().GetQuaternion()

        any_hit_flag = False
        hit_info = get_physx_scene_query_interface().overlap_box(
            carb.Float3(self._overlap_extent),
            carb.Float3(origin[0], origin[1], origin[2]),
            carb.Float4(
                quat_gf.GetImaginary()[0],
                quat_gf.GetImaginary()[1],
                quat_gf.GetImaginary()[2],
                quat_gf.GetReal(),
            ),
            self._on_overlap_hit,
            any_hit_flag,
        )

    def _on_overlap_hit(self, hit):
        # Skip self-hits
        if hit.rigid_body == self._active_bin.GetPrimPath():
            return True

        # Handle flip scenario (only once per bin)
        if not self._bin_flip_scenario_done and hit.rigid_body.startswith(self.FLIP_HELPER_PATH):
            self._timeline.pause()
            if self._timeline_sub:
                self._timeline_sub.reset()
                self._timeline_sub = None
            asyncio.ensure_future(self._run_bin_flip_scenario())
            return False

        # Handle pallet landing scenario
        is_pallet_hit = hit.rigid_body.startswith(self.PALLET_PRIM_MESH_PATH)
        is_other_bin_hit = hit.rigid_body.startswith(f"{self.BINS_FOLDER_PATH}/bin_")
        if is_pallet_hit or is_other_bin_hit:
            self._timeline.pause()
            if self._timeline_sub:
                self._timeline_sub.reset()
                self._timeline_sub = None
            asyncio.ensure_future(self._run_pallet_scenario())

        return True  # No relevant hit, return True to continue the query

    def _switch_to_pathtracing(self, spp=32, total_spp=32):
        carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
        carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
        carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", total_spp)

    def _switch_to_realtime_pathtracing(self):
        carb.settings.get_settings().set("/rtx/rendermode", "RealTimePathTracing")

    async def _run_bin_flip_scenario(self):
        await omni.kit.app.get_app().next_update_async()
        print(f"[PalletizingSDGDemo] Running bin flip scenario for bin {self._bin_counter}..")

        self._switch_to_pathtracing(spp=16, total_spp=32)
        await omni.kit.app.get_app().next_update_async()
        self._create_bin_flip_graph()

        rgb_annot = rep.annotators.get("rgb")
        instance_segmentation_annot = rep.annotators.get("instance_segmentation", init_params={"colorize": True})
        rp = rep.create.render_product(self._rep_camera, (512, 512))
        rgb_annot.attach(rp)
        instance_segmentation_annot.attach(rp)
        out_dir = os.path.join(self._output_dir, f"annot_bin_{self._bin_counter}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"[PalletizingSDGDemo] Starting capturing data for bin flip scenario for bin {self._bin_counter}..")
        for i in range(self._bin_flip_frames):
            print(f"  [PalletizingSDGDemo] Capturing frame {i + 1}/{self._bin_flip_frames}")
            await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)

            rgb_data = rgb_annot.get_data()
            rgb_file_path = os.path.join(out_dir, f"rgb_{i}.png")
            write_image(path=rgb_file_path, data=rgb_data)

            instance_segmentation_data = instance_segmentation_annot.get_data()
            instance_segmentation_file_path = os.path.join(out_dir, f"instance_segmentation_{i}.png")
            write_image(path=instance_segmentation_file_path, data=instance_segmentation_data["data"])
            with open(os.path.join(out_dir, f"instance_segmentation_info_{i}.json"), "w") as f:
                json.dump(instance_segmentation_data["info"], f, indent=4)

        # Wait for the data to be written to disk and free up resources after the capture
        await rep.orchestrator.wait_until_complete_async()
        rgb_annot.detach()
        instance_segmentation_annot.detach()
        rp.destroy()

        # Cleanup the generated SDG graph
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

        self._switch_to_realtime_pathtracing()

        # Set the flag to indicate that the bin flip scenario is done and the simulation can continue to the next bin
        self._bin_flip_scenario_done = True
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_timeline_event",
        )
        self._timeline.play()

    def _create_bin_flip_graph(self):
        # Create new random lights using the color palette for the color attribute
        color_palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

        def randomize_bin_flip_lights():
            lights = rep.create.light(
                light_type="Sphere",
                temperature=rep.distribution.normal(6500, 2000),
                intensity=rep.distribution.normal(45000, 15000),
                position=rep.distribution.uniform((0.25, 0.25, 0.5), (1, 1, 0.75)),
                scale=rep.distribution.uniform(0.5, 0.8),
                color=rep.distribution.choice(color_palette),
                count=3,
            )
            return lights.node

        rep.randomizer.register(randomize_bin_flip_lights)

        # Move the camera to the given location sequences and look at the predefined location
        camera_positions = [
            (1.96, 0.72, -0.34),
            (1.48, 0.70, 0.90),
            (0.79, -0.86, 0.12),
            (-0.49, 1.47, 0.58),
        ]
        self._rep_camera = rep.create.camera()
        with rep.trigger.on_frame():
            rep.randomizer.randomize_bin_flip_lights()
            with self._rep_camera:
                rep.modify.pose(
                    position=rep.distribution.sequence(camera_positions),
                    look_at=(0.78, 0.72, -0.1),
                )

    async def _run_pallet_scenario(self):
        await omni.kit.app.get_app().next_update_async()
        print(f"[PalletizingSDGDemo] Running pallet scenario for bin {self._bin_counter}..")
        mesh_to_orig_mats = {}
        pallet_mesh = self._stage.GetPrimAtPath(self.PALLET_PRIM_MESH_PATH)
        pallet_orig_mat, _ = UsdShade.MaterialBindingAPI(pallet_mesh).ComputeBoundMaterial()
        mesh_to_orig_mats[pallet_mesh] = pallet_orig_mat
        for i in range(self._bin_counter + 1):
            bin_mesh = self._stage.GetPrimAtPath(f"{self.BINS_FOLDER_PATH}/bin_{i}/Visuals/FOF_Mesh_Magenta_Box")
            bin_orig_mat, _ = UsdShade.MaterialBindingAPI(bin_mesh).ComputeBoundMaterial()
            mesh_to_orig_mats[bin_mesh] = bin_orig_mat

        self._create_bin_and_pallet_graph()

        out_dir = os.path.join(self._output_dir, f"writer_bin_{self._bin_counter}", "")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            backend=backend,
            rgb=True,
            instance_segmentation=True,
            colorize_instance_segmentation=True,
        )
        rp = rep.create.render_product(self._rep_camera, (512, 512))
        writer.attach(rp)

        print(f"[PalletizingSDGDemo] Starting capturing data for pallet scenario for bin {self._bin_counter}..")
        for i in range(self._pallet_frames):
            print(f"  [PalletizingSDGDemo] Capturing frame {i + 1}/{self._pallet_frames}")
            await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)

        # Make sure the backend finishes writing the data before clearing the generated SDG graph
        await rep.orchestrator.wait_until_complete_async()

        # Free up resources after the capture
        writer.detach()
        rp.destroy()

        # Cleanup the generated SDG graph
        print(f"[PalletizingSDGDemo] Restoring {len(mesh_to_orig_mats)} original materials")
        for mesh, mat in mesh_to_orig_mats.items():
            UsdShade.MaterialBindingAPI(mesh).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

        # Cleanup the generated SDG graph
        if self._stage.GetPrimAtPath("/Replicator"):
            omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

        # Return in paused state if there are no more bins to capture
        if not self._next_bin():
            return

        # Resume the simulation and continue with the next bin
        self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
            event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
            on_event=self._on_timeline_event,
            observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_timeline_event",
        )
        self._timeline.play()

    def _create_bin_and_pallet_graph(self):
        # Bin material randomization
        bin_paths = [
            f"{self.BINS_FOLDER_PATH}/bin_{i}/Visuals/FOF_Mesh_Magenta_Box" for i in range(self._bin_counter + 1)
        ]
        bins_node = rep.get.prim_at_path(bin_paths)

        with rep.trigger.on_frame():
            mats = rep.create.material_omnipbr(
                diffuse=rep.distribution.uniform((0.2, 0.1, 0.3), (0.6, 0.6, 0.7)),
                roughness=rep.distribution.choice([0.1, 0.9]),
                count=10,
            )
            with bins_node:
                rep.randomizer.materials(mats)

        # Camera and pallet texture randomization at a slower rate
        assets_root_path = get_assets_root_path()
        texture_paths = [
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Oak/Oak_BaseColor.png",
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Ash/Ash_BaseColor.png",
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
            assets_root_path + "/NVIDIA/Materials/Base/Wood/Timber/Timber_BaseColor.png",
        ]
        pallet_node = rep.get.prim_at_path(self.PALLET_PRIM_MESH_PATH)
        pallet_prim = pallet_node.get_output_prims()["prims"][0]
        pallet_loc = omni.usd.get_world_transform_matrix(pallet_prim).ExtractTranslation()
        self._rep_camera = rep.create.camera()
        with rep.trigger.on_frame(interval=4):
            with pallet_node:
                rep.randomizer.texture(texture_paths, texture_rotate=rep.distribution.uniform(80, 95))
            with self._rep_camera:
                rep.modify.pose(
                    position=rep.distribution.uniform((0, -2, 1), (2, 1, 2)),
                    look_at=(pallet_loc[0], pallet_loc[1], pallet_loc[2]),
                )

    def _next_bin(self):
        self._bin_counter += 1
        if self._bin_counter >= self._num_captures:
            self.clear()
            print("[PalletizingSDGDemo] Palletizing SDG demo finished..")
            return False
        self._active_bin = self._stage.GetPrimAtPath(f"{self.BINS_FOLDER_PATH}/bin_{self._bin_counter}")
        print(f"[PalletizingSDGDemo] Moving to bin {self._bin_counter}..")
        self._bin_flip_scenario_done = False
        return True

async def run_example_async(num_captures, bin_flip_frames, pallet_frames):
    import random

    from isaacsim.examples.interactive.ur10_palletizing.ur10_palletizing import (
        BinStacking,
    )

    # Createa new stage
    await omni.usd.get_context().new_stage_async()

    # Seed for the bin drop stage(if it needs to be flipped or not)
    random.seed(42)

    # Seed for the replicator randomization
    rep.set_global_seed(42)

    # Load the bin stacking stage and start the demo
    bin_staking_sample = BinStacking()
    print(f"[PalletizingSDGDemo] Loading the bin stacking stage..")
    await bin_staking_sample.load_world_async()
    print(f"[PalletizingSDGDemo] Starting bin stacking..")
    await bin_staking_sample.on_event_async()

    # Wait a few frames for the stage to fully load then start the SDG pipeline
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    print(f"[PalletizingSDGDemo] Starting SDG pipeline with {num_captures} bins to capture")
    sdg_demo = PalletizingSDGDemo()
    sdg_demo.start(num_captures, bin_flip_frames, pallet_frames)

    # Wait until the SDG pipeline demo is finished
    while sdg_demo.is_running():
        await omni.kit.app.get_app().next_update_async()
    print("[PalletizingSDGDemo] SDG pipeline finished, pausing the simulation..")
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()

asyncio.ensure_future(
    run_example_async(
        num_captures=DEFAULT_NUM_CAPTURES, bin_flip_frames=DEFAULT_BIN_FLIP_FRAMES, pallet_frames=DEFAULT_PALLET_FRAMES
    )
)
```

Code Explanation

This tab describes each section of the larger sample script that is used for this tutorial. By reviewing the descriptions and code snippets you can understand how the script is working and how you might customize it for your use.

**Running the UR10 Palletizing Demo Scene**

The following snippet is from the end of the code sample, it loads and starts the default UR10 Palletizing demo scene, followed by the synthetic data generation (SDG) that runs and captures the requested number of iterations (`num_captures`). You can modify NUM\_CAPTURES to run for a different number of frame captures.

Running the Example Snippet

```python
async def run_example_async(num_captures, bin_flip_frames, pallet_frames):
    import random

    from isaacsim.examples.interactive.ur10_palletizing.ur10_palletizing import (
        BinStacking,
    )

    # Create a new stage
    await omni.usd.get_context().new_stage_async()

    # Seed for the bin drop stage (if it needs to be flipped or not)
    random.seed(42)

    # Seed for the replicator randomization
    rep.set_global_seed(42)

    # Load the bin stacking stage and start the demo
    bin_staking_sample = BinStacking()
    print(f"[PalletizingSDGDemo] Loading the bin stacking stage..")
    await bin_staking_sample.load_world_async()
    print(f"[PalletizingSDGDemo] Starting bin stacking..")
    await bin_staking_sample.on_event_async()

    # Wait a few frames for the stage to fully load then start the SDG pipeline
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    print(f"[PalletizingSDGDemo] Starting SDG pipeline with {num_captures} bins to capture")
    sdg_demo = PalletizingSDGDemo()
    sdg_demo.start(num_captures, bin_flip_frames, pallet_frames)

    # Wait until the SDG pipeline demo is finished
    while sdg_demo.is_running():
        await omni.kit.app.get_app().next_update_async()
    print("[PalletizingSDGDemo] SDG pipeline finished, pausing the simulation..")
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()

asyncio.ensure_future(
    run_example_async(
        num_captures=DEFAULT_NUM_CAPTURES,
        bin_flip_frames=DEFAULT_BIN_FLIP_FRAMES,
        pallet_frames=DEFAULT_PALLET_FRAMES
    )
)
```

**PalletizingSDGDemo Class**

The demo script is wrapped in the `PalletizingSDGDemo` class. It oversees the simulation environment and manages the synthetic data generation.

PalletizingSDGDemo Class Snippet

```python
class PalletizingSDGDemo:
    BINS_FOLDER_PATH = "/World/Ur10Table/bins"
    FLIP_HELPER_PATH = "/World/Ur10Table/pallet_holder"
    PALLET_PRIM_MESH_PATH = "/World/Ur10Table/pallet/Xform/Mesh_015"

    def __init__(self):
        # There are 36 bins in total
        self._bin_counter = 0
        self._num_captures = MAX_BINS
        self._bin_flip_frames = DEFAULT_BIN_FLIP_FRAMES
        self._pallet_frames = DEFAULT_PALLET_FRAMES
        self._stage = None
        self._active_bin = None

        # Cleanup in case the user closes the stage
        self._stage_event_sub = None

        # Simulation state flags
        self._in_running_state = False
        self._bin_flip_scenario_done = False

        # Used to pause/resume the simulation
        self._timeline = None

        # Used to actively track the active bins surroundings (e.g., in contact with pallet)
        self._timeline_sub = None
        self._overlap_extent = None

        # SDG
        self._rep_camera = None
        self._output_dir = os.path.join(os.getcwd(), "_out_palletizing_sdg_demo")
        print(f"[PalletizingSDGDemo] Output directory: {self._output_dir}")
```

The attributes of this class include:

* `self._bin_counter` and `self._num_captures` are used to track the current bin index and the requested number of frames to capture
* `self._stage` is used to access objects of interest in the environment during the simulation
* `self._active_bin` is tracking the current active bin
* `self._stage_event_sub` is a subscriber to stage closing events, it is used to cleanup the demo if the stage is closed
* `self._in_running_state` indicates whether the demo is currently running
* `self._bin_flip_scenario_done` is a flag to mark if the bin flip scenario has been completed, to avoid triggering it again
* `self._timeline` is used to pause and resume the simulation in response to Synthetic Data Generation (SDG) events
* `self._timeline_sub` is a subscriber to timeline events, allowing the monitoring of the simulation state (tracking the active bin’s surroundings)
* `self._overlap_extent` represents an extent cache of the bin size, which is used to query for overlaps around the active bin
* `self._rep_camera` points the temporary replicator camera to capture SDG data
* `self._output_dir` is the output directory where the SDG data gets stored

**Start Function**

The `start` function initializes and starts the SDG demo. During initialization (using `self._init()`), it checks whether the UR10 palletizing demo is loaded and running. Additionally, it sets up the `self._stage` and `self._active_bin` attributes. The demo is then started with the `self._start()` function. This function subscribes to timeline events through `self._timeline_sub`, which uses the `self._on_timeline_event` callback function to monitor the simulation state.

Start Function Workflow Snippet

```python
def start(self, num_captures, bin_flip_frames, pallet_frames):
    self._num_captures = num_captures if 1 <= num_captures <= 36 else 36
    self._bin_flip_frames = bin_flip_frames
    self._pallet_frames = pallet_frames
    if self._init():
        self._start()

def is_running(self):
    return self._in_running_state

def _init(self):
    self._stage = omni.usd.get_context().get_stage()
    self._active_bin = self._stage.GetPrimAtPath(f"{self.BINS_FOLDER_PATH}/bin_{self._bin_counter}")

    if not self._active_bin:
        print("[PalletizingSDGDemo] Could not find bin, make sure the palletizing demo is loaded..")
        return False

    bb_cache = create_bbox_cache()
    half_ext = bb_cache.ComputeLocalBound(self._active_bin).GetRange().GetSize() * 0.5
    self._overlap_extent = carb.Float3(half_ext[0], half_ext[1], half_ext[2] * 1.1)

    self._timeline = omni.timeline.get_timeline_interface()
    if not self._timeline.is_playing():
        print("[PalletizingSDGDemo] Please start the palletizing demo first..")
        return False

    # Disable capture on play for replicator, data capture will be triggered manually
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Clear any previously generated SDG graphs
    if self._stage.GetPrimAtPath("/Replicator"):
        omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    return True

def _start(self):
    self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
        on_event=self._on_timeline_event,
        observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_timeline_event",
    )
    self._stage_event_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_event_name(omni.usd.StageEventType.CLOSING),
        on_event=self._on_stage_closing_event,
        observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_stage_closing_event",
    )
    self._in_running_state = True
    print("[PalletizingSDGDemo] Starting the palletizing SDG demo..")
```

**Timeline Advance and Bin Overlaps**

On every timeline advance update, the `self._check_bin_overlaps` function is called to monitor the surroundings of the active bin. If an overlap is detected, the `self._on_overlap_hit` callback function is invoked. This function determines if the overlap is relevant to one of two scenarios:

> * bin flip
> * bin on pallet

If relevant, the simulation is paused, the timeline event subscription is removed, and the Synthetic Data Generation (SDG) starts for the current active bin. Depending on the current simulation state, the SDG is initiated by the `self._run_bin_flip_scenario` or the `self._run_pallet_scenario` function.

Bin Tracking Snippet

```python
def _on_timeline_event(self, e: carb.eventdispatcher.Event):
    self._check_bin_overlaps()

def _check_bin_overlaps(self):
    bin_pose = omni.usd.get_world_transform_matrix(self._active_bin)
    origin = bin_pose.ExtractTranslation()
    quat_gf = bin_pose.ExtractRotation().GetQuaternion()

    any_hit_flag = False
    hit_info = get_physx_scene_query_interface().overlap_box(
        carb.Float3(self._overlap_extent),
        carb.Float3(origin[0], origin[1], origin[2]),
        carb.Float4(
            quat_gf.GetImaginary()[0],
            quat_gf.GetImaginary()[1],
            quat_gf.GetImaginary()[2],
            quat_gf.GetReal(),
        ),
        self._on_overlap_hit,
        any_hit_flag,
    )

def _on_overlap_hit(self, hit):
    # Skip self-hits
    if hit.rigid_body == self._active_bin.GetPrimPath():
        return True

    # Handle flip scenario (only once per bin)
    if not self._bin_flip_scenario_done and hit.rigid_body.startswith(self.FLIP_HELPER_PATH):
        self._timeline.pause()
        if self._timeline_sub:
            self._timeline_sub.reset()
            self._timeline_sub = None
        asyncio.ensure_future(self._run_bin_flip_scenario())
        return False

    # Handle pallet landing scenario
    is_pallet_hit = hit.rigid_body.startswith(self.PALLET_PRIM_MESH_PATH)
    is_other_bin_hit = hit.rigid_body.startswith(f"{self.BINS_FOLDER_PATH}/bin_")
    if is_pallet_hit or is_other_bin_hit:
        self._timeline.pause()
        if self._timeline_sub:
            self._timeline_sub.reset()
            self._timeline_sub = None
        asyncio.ensure_future(self._run_pallet_scenario())

    return True  # No relevant hit, return True to continue the query
```

When the active bin is positioned on the flip helper object, it triggers the **bin flip scenario**. In this scenario, path tracing is chosen as the rendering mode. To collect the data, Replicator annotators are used directly to access the data and the `write_image` function from `omni.replicator.core.functional` writes the data to disk.

The `_create_bin_flip_graph` function is used to create the Replicator randomization graphs for the **bin flip scenario**. This includes the creation of a camera and randomized lights. After setting up the graph, a delayed preview command is dispatched, ensuring the graph is fully created prior to launching the Synthetic Data Generation (SDG).

The `rep.orchestrator.step_async` function is called for the requested number of frames (`self._bin_flip_frames`) to advance the randomization graph by one frame and provide the annotators with the new data. The data is then retrieved using the `get_data()` function and saved to disk using `write_image`. To optimize simulation performance, render products are discarded after each SDG pipeline and the constructed Replicator graphs are removed.

After the SDG scenario is completed, the render mode is set back to realtime path tracing. The timeline then resumes the simulation and the timeline subscriber is reactivated to continue monitoring the simulation environment. To ensure that the **bin flip scenario** doesn’t re-trigger, given that the bin remains in contact with the flip helper object, the `self._bin_flip_scenario_done` flag is set to `True`.

Bin Flip Scenario Snippet

```python
async def _run_bin_flip_scenario(self):
    await omni.kit.app.get_app().next_update_async()
    print(f"[PalletizingSDGDemo] Running bin flip scenario for bin {self._bin_counter}..")

    self._switch_to_pathtracing(spp=16, total_spp=32)
    await omni.kit.app.get_app().next_update_async()
    self._create_bin_flip_graph()

    rgb_annot = rep.annotators.get("rgb")
    instance_segmentation_annot = rep.annotators.get("instance_segmentation", init_params={"colorize": True})
    rp = rep.create.render_product(self._rep_camera, (512, 512))
    rgb_annot.attach(rp)
    instance_segmentation_annot.attach(rp)
    out_dir = os.path.join(self._output_dir, f"annot_bin_{self._bin_counter}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PalletizingSDGDemo] Starting capturing data for bin flip scenario for bin {self._bin_counter}..")
    for i in range(self._bin_flip_frames):
        print(f"  [PalletizingSDGDemo] Capturing frame {i + 1}/{self._bin_flip_frames}")
        await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)

        rgb_data = rgb_annot.get_data()
        rgb_file_path = os.path.join(out_dir, f"rgb_{i}.png")
        write_image(path=rgb_file_path, data=rgb_data)

        instance_segmentation_data = instance_segmentation_annot.get_data()
        instance_segmentation_file_path = os.path.join(out_dir, f"instance_segmentation_{i}.png")
        write_image(path=instance_segmentation_file_path, data=instance_segmentation_data["data"])
        with open(os.path.join(out_dir, f"instance_segmentation_info_{i}.json"), "w") as f:
            json.dump(instance_segmentation_data["info"], f, indent=4)

    # Wait for the data to be written to disk and free up resources after the capture
    await rep.orchestrator.wait_until_complete_async()
    rgb_annot.detach()
    instance_segmentation_annot.detach()
    rp.destroy()

    # Cleanup the generated SDG graph
    if self._stage.GetPrimAtPath("/Replicator"):
        omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    self._switch_to_realtime_pathtracing()

    # Set the flag to indicate that the bin flip scenario is done
    self._bin_flip_scenario_done = True
    self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
        on_event=self._on_timeline_event,
        observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_timeline_event",
    )
    self._timeline.play()
```

For the **bin flip scenario**, the Replicator randomization graph uses a predefined color palette list. This list provides options for the system to randomly select colors when varying the lights using `rep.distribution.choice(color_palette)`. Meanwhile, the camera operates from a set of predefined locations. Instead of random selections, the camera sequentially transitions between these locations using `rep.distribution.sequence(camera_positions)`. Both the randomization of lights and the systematic camera movement are programmed to execute with every frame capture, as indicated by `rep.trigger.on_frame()`.

Bin Flip Randomization Graph Snippet

```python
def _create_bin_flip_graph(self):
    # Create new random lights using the color palette for the color attribute
    color_palette = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    def randomize_bin_flip_lights():
        lights = rep.create.light(
            light_type="Sphere",
            temperature=rep.distribution.normal(6500, 2000),
            intensity=rep.distribution.normal(45000, 15000),
            position=rep.distribution.uniform((0.25, 0.25, 0.5), (1, 1, 0.75)),
            scale=rep.distribution.uniform(0.5, 0.8),
            color=rep.distribution.choice(color_palette),
            count=3,
        )
        return lights.node

    rep.randomizer.register(randomize_bin_flip_lights)

    # Move the camera to the given location sequences and look at the predefined location
    camera_positions = [
        (1.96, 0.72, -0.34),
        (1.48, 0.70, 0.90),
        (0.79, -0.86, 0.12),
        (-0.49, 1.47, 0.58),
    ]
    self._rep_camera = rep.create.camera()
    with rep.trigger.on_frame():
        rep.randomizer.randomize_bin_flip_lights()
        with self._rep_camera:
            rep.modify.pose(
                position=rep.distribution.sequence(camera_positions),
                look_at=(0.78, 0.72, -0.1),
            )
```

When the active bin is placed on the pallet, or on top of another bin on the pallet, it triggers the **bin on pallet scenario**. Because the randomization graph is modifying the materials and textures of the bins and the pallet, these original materials are cached. This ensures that they can be reapplied after the simulation resumes.

The `_create_bin_and_pallet_graph` function sets up the Replicator randomization graphs for this scenario. These graphs include the camera, which randomizes its position around the pallet, the varying materials for the bins placed on the pallet, and the alternating textures for the pallet itself. After the graph is created, a delayed preview command is dispatched to ensure that it is fully generated before the Synthetic Data Generation (SDG) begins.

For data writing, the **bin on pallet scenario** uses a `DiskBackend` with the built-in Replicator `BasicWriter`. For each frame defined by `self._pallet_frames`, the `rep.orchestrator.step_async` function advances the randomization graph by a single frame. This action also triggers the writer to save the data to disk. To improve performance during the simulation, the created render products are discarded after each scenario and the generated graphs are removed.

After the scenario completes, the cached materials are re-applied. The system then checks to see if it has processed the last bin. If not, the simulation is resumed, designating the next bin as active and reactivating the timeline subscriber to continue monitoring the simulation environment.

Bin on Pallet Scenario Snippet

```python
async def _run_pallet_scenario(self):
    await omni.kit.app.get_app().next_update_async()
    print(f"[PalletizingSDGDemo] Running pallet scenario for bin {self._bin_counter}..")
    mesh_to_orig_mats = {}
    pallet_mesh = self._stage.GetPrimAtPath(self.PALLET_PRIM_MESH_PATH)
    pallet_orig_mat, _ = UsdShade.MaterialBindingAPI(pallet_mesh).ComputeBoundMaterial()
    mesh_to_orig_mats[pallet_mesh] = pallet_orig_mat
    for i in range(self._bin_counter + 1):
        bin_mesh = self._stage.GetPrimAtPath(f"{self.BINS_FOLDER_PATH}/bin_{i}/Visuals/FOF_Mesh_Magenta_Box")
        bin_orig_mat, _ = UsdShade.MaterialBindingAPI(bin_mesh).ComputeBoundMaterial()
        mesh_to_orig_mats[bin_mesh] = bin_orig_mat

    self._create_bin_and_pallet_graph()

    out_dir = os.path.join(self._output_dir, f"writer_bin_{self._bin_counter}", "")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out_dir)
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        backend=backend,
        rgb=True,
        instance_segmentation=True,
        colorize_instance_segmentation=True,
    )
    rp = rep.create.render_product(self._rep_camera, (512, 512))
    writer.attach(rp)

    print(f"[PalletizingSDGDemo] Starting capturing data for pallet scenario for bin {self._bin_counter}..")
    for i in range(self._pallet_frames):
        print(f"  [PalletizingSDGDemo] Capturing frame {i + 1}/{self._pallet_frames}")
        await rep.orchestrator.step_async(rt_subframes=16, delta_time=0.0)

    # Make sure the backend finishes writing the data before clearing the generated SDG graph
    await rep.orchestrator.wait_until_complete_async()

    # Free up resources after the capture
    writer.detach()
    rp.destroy()

    # Cleanup the generated SDG graph
    print(f"[PalletizingSDGDemo] Restoring {len(mesh_to_orig_mats)} original materials")
    for mesh, mat in mesh_to_orig_mats.items():
        UsdShade.MaterialBindingAPI(mesh).Bind(mat, UsdShade.Tokens.strongerThanDescendants)

    # Cleanup the generated SDG graph
    if self._stage.GetPrimAtPath("/Replicator"):
        omni.kit.commands.execute("DeletePrimsCommand", paths=["/Replicator"])

    # Return in paused state if there are no more bins to capture
    if not self._next_bin():
        return

    # Resume the simulation and continue with the next bin
    self._timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
        on_event=self._on_timeline_event,
        observer_name="test_sdg_ur10_palletizing.PalletizingSDGDemo._on_timeline_event",
    )
    self._timeline.play()
```

For the **bin on pallet scenario**, the Replicator randomization graph randomizes the colors of the bin materials. A predefined list of textures is used, from which the graph randomly selects and applies th pallet textures, this is done by `rep.randomizer.texture(texture_paths,..)`. The camera’s position varies around the pallet using `rep.distribution.uniform(..)` and is oriented towards the pallet’s location. The trigger is split into two parts:

* the bin materials are changed **every frame** as shown by `rep.trigger.on_frame()`
* while the pallet textures and the camera positions are executed every **four frames**, represented by `rep.trigger.on_frame(interval=4)`

Bin on Pallet Randomization Graph Snippet

```python
def _create_bin_and_pallet_graph(self):
    # Bin material randomization
    bin_paths = [
        f"{self.BINS_FOLDER_PATH}/bin_{i}/Visuals/FOF_Mesh_Magenta_Box" for i in range(self._bin_counter + 1)
    ]
    bins_node = rep.get.prim_at_path(bin_paths)

    with rep.trigger.on_frame():
        mats = rep.create.material_omnipbr(
            diffuse=rep.distribution.uniform((0.2, 0.1, 0.3), (0.6, 0.6, 0.7)),
            roughness=rep.distribution.choice([0.1, 0.9]),
            count=10,
        )
        with bins_node:
            rep.randomizer.materials(mats)

    # Camera and pallet texture randomization at a slower rate
    assets_root_path = get_assets_root_path()
    texture_paths = [
        assets_root_path + "/NVIDIA/Materials/Base/Wood/Oak/Oak_BaseColor.png",
        assets_root_path + "/NVIDIA/Materials/Base/Wood/Ash/Ash_BaseColor.png",
        assets_root_path + "/NVIDIA/Materials/Base/Wood/Plywood/Plywood_BaseColor.png",
        assets_root_path + "/NVIDIA/Materials/Base/Wood/Timber/Timber_BaseColor.png",
    ]
    pallet_node = rep.get.prim_at_path(self.PALLET_PRIM_MESH_PATH)
    pallet_prim = pallet_node.get_output_prims()["prims"][0]
    pallet_loc = omni.usd.get_world_transform_matrix(pallet_prim).ExtractTranslation()
    self._rep_camera = rep.create.camera()
    with rep.trigger.on_frame(interval=4):
        with pallet_node:
            rep.randomizer.texture(texture_paths, texture_rotate=rep.distribution.uniform(80, 95))
        with self._rep_camera:
            rep.modify.pose(
                position=rep.distribution.uniform((0, -2, 1), (2, 1, 2)),
                look_at=(pallet_loc[0], pallet_loc[1], pallet_loc[2]),
            )
```

---

# Cosmos Synthetic Data Generation

This tutorial demonstrates generating multi-modal synthetic data for [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) using the `CosmosWriter` in Isaac Sim. The writer captures synchronized RGB, depth, segmentation, and edge data from a robot navigating a warehouse environment.

The generated data serves as ground truth input for [Cosmos Transfer](https://docs.nvidia.com/cosmos/latest/), which transforms low-resolution control signals into high-quality visual simulations through its Multi-ControlNet architecture.

## Why Use the CosmosWriter?

The CosmosWriter bridges the gap between simulation and real-world robotics applications by generating rich, multi-modal datasets from synthetic environments. Key use cases include:

* **Sim-to-Real Transfer**: Transform synthetic simulation videos into photorealistic scenes with varied materials, lighting, and environmental conditions using Cosmos Transfer
* **Domain Adaptation**: Generate diverse training data from a single simulation, creating variations in scene styles, materials, and lighting without re-running expensive simulations or capturing real-world data
* **Data Augmentation**: Expand limited datasets by generating multiple visual variations while preserving robot motions, object positions, and scene structure

For examples of sim-to-real transformations in robotics, see the [Cosmos Cookbook Robotics Gallery](https://nvidia-cosmos.github.io/cosmos-cookbook/gallery/robotics_inference.html), which showcases how synthetic kitchen scenes can be transformed into photorealistic environments with different cabinet styles, robot materials, and lighting conditions.

## Prerequisites

* Familiarity with the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension and its [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)")
* Basic understanding of Isaac Sim’s SDG [Getting Started Scripts](Synthetic_Data_Generation.md)
* Running simulations as [Standalone Applications](Workflows.md) or via the [Script Editor](Development_Tools.md).

## What the CosmosWriter Generates

The writer outputs five synchronized modalities from the robot’s camera:

* **RGB** - Color imagery (vis control)
* **Depth** - Distance-to-camera for spatial understanding
* **Segmentation** - Instance masks for object tracking
* **Shaded Segmentation** - Instance masks with realistic shading
* **Edges** - Canny edge detection for boundaries

These modalities correspond to [Cosmos Transfer’s](https://docs.nvidia.com/cosmos/latest/#controlnet-specification) control branches:

* **vis**: Uses RGB imagery with bilateral blurring
* **edge**: Applies Canny edge detection (tunable thresholds)
* **depth**: Depth maps for 3D structure understanding
* **seg**: Segmentation masks for object identification

Each control branch can be weighted (0.0-1.0) to balance adherence vs. creative freedom in the generated output.

## Implementation

This example demonstrates a Carter Nova robot autonomously navigating through a warehouse environment. As the robot moves from its starting position to a target location, the `CosmosWriter` captures synchronized multi-modal data (RGB, depth, segmentation, shaded segmentation, and edges) from the robot’s front camera. The captured data is organized into clips, with each clip containing a sequence of frames that can be used as input for Cosmos Transfer.

Standalone Application

The example can be run as a standalone application using the following commands in the terminal (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/replicator/cosmos_writer_warehouse.py
```

Full Standalone Script

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import os

import carb
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdGeom

# Capture parameters
START_DELAY = 0.1  # Timeline duration delay before capturing the first clip
NUM_CLIPS = 2  # Number of video clips to capture with the CosmosWriter
NUM_FRAMES_PER_CLIP = 10  # Number of frames for each clip
CAPTURE_INTERVAL = 2  # Capture interval between frames (capture every N simulation steps)

# Stage and asset paths
STAGE_URL = "/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd"
CARTER_NAV_ASSET_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
CARTER_NAV_PATH = "/NavWorld/CarterNav"
CARTER_NAV_TARGET_PATH = f"{CARTER_NAV_PATH}/targetXform"
CARTER_CAMERA_PATH = f"{CARTER_NAV_PATH}/chassis_link/sensors/front_hawk/left/camera_left"
CARTER_NAV_POSITION = (-6, 4, 0)
CARTER_NAV_TARGET_POSITION = (3, 3, 0)

def advance_timeline_by_duration(duration: float, max_updates: int = 1000):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    target_time = current_time + duration

    if timeline.get_end_time() < target_time:
        timeline.set_end_time(1000000)

    if not timeline.is_playing():
        timeline.play()

    print(f"Advancing timeline from {current_time:.4f}s to {target_time:.4f}s")
    step_count = 0
    while current_time < target_time:
        if step_count >= max_updates:
            print(f"Max updates reached: {step_count}, finishing timeline advance.")
            break

        prev_time = current_time
        simulation_app.update()
        current_time = timeline.get_current_time()
        step_count += 1

        if step_count % 10 == 0:
            print(f"\tStep {step_count}, {current_time:.4f}s/{target_time:.4f}s")

        if current_time <= prev_time:
            print(f"Warning: Timeline did not advance at update {step_count} (time: {current_time:.4f}s).")
    print(f"Finished advancing timeline to {current_time:.4f}s (target {target_time:.4f}s) in {step_count} steps")

def run_sdg_pipeline(
    camera_path, num_clips, num_frames_per_clip, capture_interval, use_instance_id=True, segmentation_mapping=None
):
    rp = rep.create.render_product(camera_path, (1280, 720))
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), f"_out_cosmos_warehouse")
    print(f"output_directory: {out_dir}")
    backend.initialize(output_dir=out_dir)
    cosmos_writer.initialize(
        backend=backend, use_instance_id=use_instance_id, segmentation_mapping=segmentation_mapping
    )
    cosmos_writer.attach(rp)

    # Make sure the timeline is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    print(
        f"Starting SDG pipeline. Capturing {num_clips} clips with {num_frames_per_clip} frames each, every {capture_interval} simulation step(s)."
    )

    for clip_index in range(num_clips):
        print(f"Starting clip {clip_index + 1}/{num_clips}")

        frames_captured_count = 0
        simulation_step_index = 0
        while frames_captured_count < num_frames_per_clip:
            print(f"Simulation step {simulation_step_index}")
            if simulation_step_index % capture_interval == 0:
                print(f"\t Capturing frame {frames_captured_count + 1}/{num_frames_per_clip} for clip {clip_index + 1}")
                rep.orchestrator.step(pause_timeline=False)
                frames_captured_count += 1
            else:
                simulation_app.update()
            simulation_step_index += 1

        print(f"Finished clip {clip_index + 1}/{num_clips}. Captured {frames_captured_count} frames")

        # Move to next clip if not the last clip
        if clip_index < num_clips - 1:
            print(f"Moving to next clip...")
            cosmos_writer.next_clip()

    print("Waiting to finish processing and writing the data")
    rep.orchestrator.wait_until_complete()
    print(f"Finished SDG pipeline. Captured {num_clips} clips with {num_frames_per_clip} frames each")
    cosmos_writer.detach()
    rp.destroy()
    timeline.pause()

def run_example(
    num_clips,
    num_frames_per_clip,
    capture_interval,
    start_delay=0.0,
    use_instance_id=True,
    segmentation_mapping=None,
):
    assets_root_path = get_assets_root_path()
    stage_path = assets_root_path + STAGE_URL
    print(f"Opening stage: '{stage_path}'")
    omni.usd.get_context().open_stage(stage_path)
    stage = omni.usd.get_context().get_stage()

    # Enable script nodes
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Disable capture on play on the new stage, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Load carter nova asset with its navigation graph
    carter_url_path = assets_root_path + CARTER_NAV_ASSET_URL
    print(f"Loading carter nova asset: '{carter_url_path}' at prim path: '{CARTER_NAV_PATH}'")
    carter_nav_prim = add_reference_to_stage(usd_path=carter_url_path, prim_path=CARTER_NAV_PATH)

    if not carter_nav_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_nav_prim).AddTranslateOp()
    carter_nav_prim.GetAttribute("xformOp:translate").Set(CARTER_NAV_POSITION)

    # Set the navigation target position
    carter_navigation_target_prim = stage.GetPrimAtPath(CARTER_NAV_TARGET_PATH)
    if not carter_navigation_target_prim.IsValid():
        print(f"Carter navigation target prim not found at path: {CARTER_NAV_TARGET_PATH}, exiting")
        return
    if not carter_navigation_target_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_navigation_target_prim).AddTranslateOp()
    carter_navigation_target_prim.GetAttribute("xformOp:translate").Set(CARTER_NAV_TARGET_POSITION)

    # Use the carter nova front hawk camera for capturing data
    camera_prim = stage.GetPrimAtPath(CARTER_CAMERA_PATH)
    if not camera_prim.IsValid():
        print(f"Camera prim not found at path: {CARTER_CAMERA_PATH}, exiting")
        return

    # Advance the timeline with the start delay if provided
    if start_delay is not None and start_delay > 0:
        advance_timeline_by_duration(start_delay)

    # Run the SDG pipeline
    run_sdg_pipeline(
        camera_prim.GetPath(), num_clips, num_frames_per_clip, capture_interval, use_instance_id, segmentation_mapping
    )

# Setup the environment and run the example
run_example(
    num_clips=NUM_CLIPS,
    num_frames_per_clip=NUM_FRAMES_PER_CLIP,
    capture_interval=CAPTURE_INTERVAL,
    start_delay=START_DELAY,
    use_instance_id=True,
)

simulation_app.close()
```

Script Editor

Full Script Editor Script

```python
import asyncio
import os

import carb
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path_async
from pxr import UsdGeom

# Capture parameters
START_DELAY = 0.1  # Timeline duration delay before capturing the first clip
NUM_CLIPS = 3  # Number of video clips to capture with the CosmosWriter
NUM_FRAMES_PER_CLIP = 120  # Number of frames for each clip
CAPTURE_INTERVAL = 2  # Capture interval between frames (capture every N simulation steps)

# Stage and asset paths
STAGE_URL = "/Isaac/Samples/Replicator/Stage/full_warehouse_worker_and_anim_cameras.usd"
CARTER_NAV_ASSET_URL = "/Isaac/Samples/Replicator/OmniGraph/nova_carter_nav_only.usd"
CARTER_NAV_PATH = "/NavWorld/CarterNav"
CARTER_NAV_TARGET_PATH = f"{CARTER_NAV_PATH}/targetXform"
CARTER_CAMERA_PATH = f"{CARTER_NAV_PATH}/chassis_link/sensors/front_hawk/left/camera_left"
CARTER_NAV_POSITION = (-6, 4, 0)
CARTER_NAV_TARGET_POSITION = (3, 3, 0)

async def advance_timeline_by_duration_async(duration: float, max_updates: int = 1000):
    timeline = omni.timeline.get_timeline_interface()
    current_time = timeline.get_current_time()
    target_time = current_time + duration

    if timeline.get_end_time() < target_time:
        timeline.set_end_time(1000000)

    if not timeline.is_playing():
        timeline.play()

    print(f"Advancing timeline from {current_time:.4f}s to {target_time:.4f}s")
    step_count = 0
    while current_time < target_time:
        if step_count >= max_updates:
            print(f"Max updates reached: {step_count}, finishing timeline advance.")
            break

        prev_time = current_time
        await omni.kit.app.get_app().next_update_async()
        current_time = timeline.get_current_time()
        step_count += 1

        if step_count % 10 == 0:
            print(f"\tStep {step_count}, {current_time:.4f}s/{target_time:.4f}s")

        if current_time <= prev_time:
            print(f"Warning: Timeline did not advance at update {step_count} (time: {current_time:.4f}s).")
    print(f"Finished advancing timeline to {current_time:.4f}s (target {target_time:.4f}s) in {step_count} steps")

async def run_sdg_pipeline_async(
    camera_path,
    num_clips,
    num_frames_per_clip,
    capture_interval,
    use_instance_id=True,
    segmentation_mapping=None,
):
    rp = rep.create.render_product(camera_path, (1280, 720))
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), f"_out_cosmos_warehouse")
    print(f"output_directory: {out_dir}")
    backend.initialize(output_dir=out_dir)
    cosmos_writer.initialize(
        backend=backend, use_instance_id=use_instance_id, segmentation_mapping=segmentation_mapping
    )
    cosmos_writer.attach(rp)

    # Make sure the timeline is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    print(
        f"Starting SDG pipeline. Capturing {num_clips} clips with {num_frames_per_clip} frames each, every {capture_interval} simulation step(s)."
    )

    for clip_index in range(num_clips):
        print(f"Starting clip {clip_index + 1}/{num_clips}")

        frames_captured_count = 0
        simulation_step_index = 0
        while frames_captured_count < num_frames_per_clip:
            print(f"Simulation step {simulation_step_index}")
            if simulation_step_index % capture_interval == 0:
                print(f"\t Capturing frame {frames_captured_count + 1}/{num_frames_per_clip} for clip {clip_index + 1}")
                await rep.orchestrator.step_async(pause_timeline=False)
                frames_captured_count += 1
            else:
                await omni.kit.app.get_app().next_update_async()
            simulation_step_index += 1

        print(f"Finished clip {clip_index + 1}/{num_clips}. Captured {frames_captured_count} frames")

        # Move to next clip if not the last clip
        if clip_index < num_clips - 1:
            print(f"Moving to next clip...")
            cosmos_writer.next_clip()

    print("Waiting to finish processing and writing the data")
    await rep.orchestrator.wait_until_complete_async()
    print(f"Finished SDG pipeline. Captured {num_clips} clips with {num_frames_per_clip} frames each")
    cosmos_writer.detach()
    rp.destroy()
    timeline.pause()

async def run_example_async(
    num_clips,
    num_frames_per_clip,
    capture_interval,
    start_delay=0.0,
    use_instance_id=True,
    segmentation_mapping=None,
):
    assets_root_path = await get_assets_root_path_async()
    stage_path = assets_root_path + STAGE_URL
    print(f"Opening stage: '{stage_path}'")
    omni.usd.get_context().open_stage(stage_path)
    stage = omni.usd.get_context().get_stage()

    # Enable script nodes
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Disable capture on play on the new stage, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Load carter nova asset with its navigation graph
    carter_url_path = assets_root_path + CARTER_NAV_ASSET_URL
    print(f"Loading carter nova asset: '{carter_url_path}' at prim path: '{CARTER_NAV_PATH}'")
    carter_nav_prim = add_reference_to_stage(usd_path=carter_url_path, prim_path=CARTER_NAV_PATH)

    if not carter_nav_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_nav_prim).AddTranslateOp()
    carter_nav_prim.GetAttribute("xformOp:translate").Set(CARTER_NAV_POSITION)

    # Set the navigation target position
    carter_navigation_target_prim = stage.GetPrimAtPath(CARTER_NAV_TARGET_PATH)
    if not carter_navigation_target_prim.IsValid():
        print(f"Carter navigation target prim not found at path: {CARTER_NAV_TARGET_PATH}, exiting")
        return
    if not carter_navigation_target_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(carter_navigation_target_prim).AddTranslateOp()
    carter_navigation_target_prim.GetAttribute("xformOp:translate").Set(CARTER_NAV_TARGET_POSITION)

    # Use the carter nova front hawk camera for capturing data
    camera_prim = stage.GetPrimAtPath(CARTER_CAMERA_PATH)
    if not camera_prim.IsValid():
        print(f"Camera prim not found at path: {CARTER_CAMERA_PATH}, exiting")
        return

    # Advance the timeline with the start delay if provided
    if start_delay is not None and start_delay > 0:
        await advance_timeline_by_duration_async(start_delay)

    # Run the SDG pipeline
    await run_sdg_pipeline_async(
        camera_prim.GetPath(),
        num_clips,
        num_frames_per_clip,
        capture_interval,
        use_instance_id,
        segmentation_mapping,
    )

# Setup the environment and run the example
asyncio.ensure_future(
    run_example_async(
        num_clips=NUM_CLIPS,
        num_frames_per_clip=NUM_FRAMES_PER_CLIP,
        capture_interval=CAPTURE_INTERVAL,
        start_delay=START_DELAY,
        use_instance_id=True,
    )
)
```

Code Explanation

This tab explains how the warehouse navigation example works and how the CosmosWriter captures multi-modal data during robot movement.

**Script Overview**

The script simulates a Carter Nova robot navigating through a warehouse while capturing synchronized multi-modal data from its front camera. The robot moves from a starting position to a target location, and the CosmosWriter generates ground truth data for Cosmos Transfer.

Main Execution Flow

```python
# Setup the environment and run the example
run_example(
    num_clips=NUM_CLIPS,
    num_frames_per_clip=NUM_FRAMES_PER_CLIP,
    capture_interval=CAPTURE_INTERVAL,
    start_delay=START_DELAY,
    use_instance_id=True,
)

simulation_app.close()
```

**Key Configuration Parameters**

Capture Parameters

* `NUM_CLIPS = 2`: Generate 2 separate video clips
* `NUM_FRAMES_PER_CLIP = 10`: Each clip contains 10 frames
* `CAPTURE_INTERVAL = 2`: Capture every 2nd simulation step
* `START_DELAY = 0.1`: Custom delay to start capturing at a specific time

**Data Capture Pipeline**

The `run_sdg_pipeline` function orchestrates the entire capture process:

SDG Pipeline Implementation

```python
def run_sdg_pipeline(
    camera_path, num_clips, num_frames_per_clip, capture_interval, use_instance_id=True, segmentation_mapping=None
):
    rp = rep.create.render_product(camera_path, (1280, 720))
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    backend = rep.backends.get("DiskBackend")
    out_dir = os.path.join(os.getcwd(), f"_out_cosmos_warehouse")
    print(f"output_directory: {out_dir}")
    backend.initialize(output_dir=out_dir)
    cosmos_writer.initialize(
        backend=backend, use_instance_id=use_instance_id, segmentation_mapping=segmentation_mapping
    )
    cosmos_writer.attach(rp)

    # Make sure the timeline is playing
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    print(
        f"Starting SDG pipeline. Capturing {num_clips} clips with {num_frames_per_clip} frames each, every {capture_interval} simulation step(s)."
    )

    for clip_index in range(num_clips):
        print(f"Starting clip {clip_index + 1}/{num_clips}")

        frames_captured_count = 0
        simulation_step_index = 0
        while frames_captured_count < num_frames_per_clip:
            print(f"Simulation step {simulation_step_index}")
            if simulation_step_index % capture_interval == 0:
                print(f"\t Capturing frame {frames_captured_count + 1}/{num_frames_per_clip} for clip {clip_index + 1}")
                rep.orchestrator.step(pause_timeline=False)
                frames_captured_count += 1
            else:
                simulation_app.update()
            simulation_step_index += 1

        print(f"Finished clip {clip_index + 1}/{num_clips}. Captured {frames_captured_count} frames")

        # Move to next clip if not the last clip
        if clip_index < num_clips - 1:
            print(f"Moving to next clip...")
            cosmos_writer.next_clip()

    print("Waiting to finish processing and writing the data")
    rep.orchestrator.wait_until_complete()
    print(f"Finished SDG pipeline. Captured {num_clips} clips with {num_frames_per_clip} frames each")
    cosmos_writer.detach()
    rp.destroy()
    timeline.pause()
```

**Key aspects:**
- The render product is created from the robot’s front camera at 1280x720 resolution
- `pause_timeline=False` allows the robot to continue moving during capture
- The simulation advances between captures to show navigation progress

**CosmosWriter Configuration**

Writer Modes and Parameters

The CosmosWriter supports two segmentation modes:

1. **Instance ID Mode** (default):

   ```python
   cosmos_writer.initialize(
       backend=backend, use_instance_id=use_instance_id, segmentation_mapping=segmentation_mapping
   )
   ```
2. **Semantic Segmentation Mode**:

   ```python
   segmentation_mapping = {
       "floor": [255, 0, 0, 255],  # Red
       "wall": [0, 255, 0, 255],  # Green
       "rack": [0, 0, 255, 255],  # Blue
   }

   # Note: This overrides instance ID mode and requires semantic annotations
   cosmos_writer.initialize(backend=backend, segmentation_mapping=segmentation_mapping)
   ```

**Timeline Management**

The script uses a helper function to advance the timeline before starting capture:

> Timeline Advancement
>
> ```python
> def advance_timeline_by_duration(duration: float, max_updates: int = 1000):
>     timeline = omni.timeline.get_timeline_interface()
>     current_time = timeline.get_current_time()
>     target_time = current_time + duration
>
>     if timeline.get_end_time() < target_time:
>         timeline.set_end_time(1000000)
>
>     if not timeline.is_playing():
>         timeline.play()
>
>     print(f"Advancing timeline from {current_time:.4f}s to {target_time:.4f}s")
>     step_count = 0
>     while current_time < target_time:
>         if step_count >= max_updates:
>             print(f"Max updates reached: {step_count}, finishing timeline advance.")
>             break
>
>         prev_time = current_time
>         simulation_app.update()
>         current_time = timeline.get_current_time()
>         step_count += 1
>
>         if step_count % 10 == 0:
>             print(f"\tStep {step_count}, {current_time:.4f}s/{target_time:.4f}s")
>
>         if current_time <= prev_time:
>             print(f"Warning: Timeline did not advance at update {step_count} (time: {current_time:.4f}s).")
>     print(f"Finished advancing timeline to {current_time:.4f}s (target {target_time:.4f}s) in {step_count} steps")
> ```

## Output Structure

After running the script, an output folder (e.g., `_out_cosmos_warehouse`) is created containing organized multi-modal data optimized for Cosmos Transfer and other foundation model training pipelines. Each clip represents a continuous sequence of frames captured during robot navigation:

```python
_out_cosmos_warehouse/
  clip_0000/                    # First clip sequence
    rgb/                        # Standard color images
      rgb_0000.png, rgb_0001.png, ...
    depth/                      # Colorized depth visualization
      depth_0000.png, depth_0001.png, ...
    segmentation/              # Instance/semantic masks
      segmentation_0000.png, segmentation_0001.png, ...
    shaded_seg/                # Segmentation with realistic shading
      shaded_seg_0000.png, shaded_seg_0001.png, ...
    edges/                      # Canny edge detection results
      edges_0000.png, edges_0001.png, ...
    rgb.mp4                     # Combined RGB video
    depth.mp4                   # Combined depth video
    segmentation.mp4            # Combined segmentation video
    shaded_seg.mp4              # Combined shaded segmentation video
    edges.mp4                   # Combined edges video
  clip_0001/                    # Next clip sequence
```

**What Each Modality Provides:**

* **RGB (rgb.mp4)**: The visual input video used with Cosmos Transfer’s `vis` control branch for preserving lighting and camera properties
* **Depth (depth.mp4)**: 3D spatial information used with the `depth` control branch to maintain perspective and spatial relationships
* **Segmentation (segmentation.mp4)**: Instance or semantic masks used with the `seg` control branch for object-level transformations
* **Shaded Segmentation (shaded\_seg.mp4)**: Combines segmentation with realistic shading for enhanced visual coherence
* **Edges (edges.mp4)**: Structural boundaries used with the `edge` control branch to preserve object shapes while allowing material and lighting changes

These MP4 files can be directly passed to Cosmos Transfer as control inputs. The PNG sequences are provided for frame-level inspection or custom processing pipelines.

## Advanced Usage

**Custom Segmentation Colors:**

Map specific semantic labels to custom colors when you need consistent class identification across datasets. Use this when training models that require specific object classes to maintain the same color/ID across all training data, ensuring Cosmos Transfer preserves class relationships.

```python
segmentation_mapping = {
    "floor": [255, 0, 0, 255],  # Red
    "wall": [0, 255, 0, 255],  # Green
    "rack": [0, 0, 255, 255],  # Blue
}

# Note: This overrides instance ID mode and requires semantic annotations
cosmos_writer.initialize(backend=backend, segmentation_mapping=segmentation_mapping)
```

**Edge Detection Tuning:**

Adjust Canny edge detection parameters for the hysteresis procedure when generating edge maps. The Canny algorithm uses two thresholds:

* **Low threshold**: Edges with gradient magnitude above this value are considered as potential edges
* **High threshold**: Edges with gradient magnitude above this value are definitely edges

Lower threshold values detect more edges (including noise), while higher values produce cleaner output with only strong edges. Values typically range from 10-200.

```python
cosmos_writer.initialize(
    backend=backend,
    use_instance_id=True,
    canny_threshold_low=10,  # Low threshold for hysteresis
    canny_threshold_high=100,  # High threshold for hysteresis
)
```

## Using Data with Cosmos Transfer

The generated data can be used with [Cosmos Transfer](https://docs.nvidia.com/cosmos/latest/) to create high-quality visual simulations. This enables sim-to-real transfer where synthetic scenes are transformed into photorealistic environments while preserving robot motions and scene structure.

For real-world examples of this workflow, see the [Cosmos Cookbook Robotics Gallery](https://nvidia-cosmos.github.io/cosmos-cookbook/gallery/robotics_inference.html), which demonstrates:

* **Edge-only control**: Transform simulation videos into diverse kitchen styles (white cabinets, red cabinets, wood tones) and robot materials (plastic, metal, gold) while preserving exact robot motions
* **Multi-control**: Combine depth, edge, and segmentation controls for precise scene manipulation

Here’s how the modalities map to Transfer’s control branches:

**Basic Single Control Example:**

```python
{
    "prompt": "A modern warehouse with autonomous robots...",
    "input_video_path": "_out_cosmos_warehouse/clip_0000/rgb.mp4",
    "edge": {
        "control_weight": 1.0
    }
}
```

**Multi-Modal Control Example:**

```python
{
    "prompt": "High-quality warehouse simulation...",
    "input_video_path": "_out_cosmos_warehouse/clip_0000/rgb.mp4",
    "vis": {"control_weight": 0.25},
    "edge": {"control_weight": 0.25},
    "depth": {
        "input_control": "_out_cosmos_warehouse/clip_0000/depth.mp4",
        "control_weight": 0.25
    },
    "seg": {
        "input_control": "_out_cosmos_warehouse/clip_0000/segmentation.mp4",
        "control_weight": 0.25
    }
}
```

**Key Considerations:**

* **Control Weights**: Values 0.0-1.0 control adherence (higher = stricter following, lower = more creative freedom)
* **Automatic Normalization**: If total weights > 1.0, they’re normalized automatically
* **Prompting**: Focus on single scenes with rich descriptions; avoid camera control instructions
* **Safety**: Human faces are automatically blurred by Cosmos Guardrail

For advanced features like spatiotemporal control maps and prompt upsampling, refer to the [Cosmos Transfer documentation](https://docs.nvidia.com/cosmos/latest/).

## Summary

This tutorial demonstrated using the CosmosWriter to generate synchronized multi-modal data from a robot navigating a warehouse. The output provides ground truth for Cosmos Transfer to create high-quality visual simulations for physical AI applications.

**Next Steps:**

1. **Explore your output**: Navigate to the generated output folder (e.g., `_out_cosmos_warehouse`) to inspect the RGB, depth, segmentation, and edge data
2. **Use with Cosmos Transfer**: Pass the generated MP4 files to Cosmos [Transfer1](https://docs.nvidia.com/cosmos/latest/transfer1/index.html) or [Transfer2.5](https://docs.nvidia.com/cosmos/latest/transfer2.5/index.html) using the JSON configuration examples above
3. **See real examples**: Visit the [Cosmos Cookbook Robotics Gallery](https://nvidia-cosmos.github.io/cosmos-cookbook/gallery/robotics_inference.html) for examples of sim-to-real transformations using similar data
4. **Customize for your use case**: Adjust capture parameters, segmentation mappings, and edge detection thresholds to optimize for your specific training pipeline

---

# Data Augmentation

Example of using Isaac Sim and Replicator to capture augmented synthetic data.

## Learning Objectives

This tutorial provides examples on how to use omni.replicator [augmentations](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/augmentation_examples.html "(in Omniverse Extensions)") on annotators or writers. The provided examples will showcase how to augment **rgb** and **depth** annotator data using warp (GPU) or NumPy (CPU) kernel/filters. The use of warp is particularly advantageous for executing parallelizable tasks, especially if the data already resides in the GPUs memory, thus avoiding memory copies from GPU to CPU.

* For a better understanding of the tutorial, familiarity with [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)"), [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)"), [writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)") and [warp](https://docs.omniverse.nvidia.com/extensions/latest/ext_warp.html "(in Omniverse Extensions)") is recommended.

## Scenario

The depicted figure showcases the example augmentations used throughout the examples. The first image is an illustrative example switching the red and blue channels of the image. The second image is a composed augmentation of converting the rgb data to hsv, adding gaussian noise, and converting back to rgb. The third and forth image are results of applying gaussian noise filters with various sigma values to the depth data.

For the example scenario a red cube is spawned with a camera looking at it from a top view. For the cube a replicator randomization graph is created which will trigger a random rotation for every frame capture.

## Implementation

The tutorial is split into two parts, the first example will showcase how to augment annotators directly, and secondly how to augment writers. Both examples can be run as [Standalone Applications](Workflows.md) or from the UI using the [Script Editor](Development_Tools.md).

### Annotator Augmentation

The annotator example will output rgb images with the red and blue channels switched, and two depth images with different gaussian noise levels (saved as grayscale PNGs). The example can switch between using warp or NumPy augmentations.

Standalone Application

The example can be run as a standalone application using the following commands in the terminal (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/replicator/augmentation/annotator_augmentation.py
```

Optionally the following arguments can be used to change the default behavior:

* `--use_warp` – flag to use warp (GPU) instead of numpy (CPU) for the augmentation functions (default: False)
* `--num_frames` – the number of frames to be captured (default: 25)

```python
./python.sh standalone_examples/replicator/augmentation/annotator_augmentation.py --use_warp --num_frames 25
```

Full Standalone Script

```python
"""Generate augmented synthetic data from annotators."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
import time

import carb.settings
import numpy as np
import omni.replicator.core as rep
import omni.usd
import warp as wp
from isaacsim.core.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path
from omni.replicator.core.functional import write_image

parser = argparse.ArgumentParser()
parser.add_argument("--num_frames", type=int, default=5, help="The number of frames to capture")
parser.add_argument(
    "--use_warp",
    action="store_true",
    help="Use warp augmentations instead of numpy",
)
parser.add_argument("--resolution", nargs=2, type=int, default=[512, 512], help="Camera resolution")
parser.add_argument("--env_url", type=str, default="", help="USD environment URL (empty for basic scene)")
args, unknown = parser.parse_known_args()

num_frames = args.num_frames
use_warp = args.use_warp
resolution = args.resolution
env_url = args.env_url or None
SEED = 42

# Enable warp scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

def rgb_to_bgr_np(data_in):
    """Swap RGBA red and blue channels using NumPy (CPU)."""
    data_in[:, :, [0, 2]] = data_in[:, :, [2, 0]]
    return data_in

@wp.kernel
def rgb_to_bgr_wp(data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8)):
    """Swap RGBA red and blue channels using Warp (GPU)."""
    i, j = wp.tid()
    data_out[i, j, 0] = data_in[i, j, 2]
    data_out[i, j, 1] = data_in[i, j, 1]
    data_out[i, j, 2] = data_in[i, j, 0]
    data_out[i, j, 3] = data_in[i, j, 3]

def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to depth values using NumPy (CPU)."""
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

rep.annotators.register_augmentation(
    "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=SEED)
)

@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
):
    """Add Gaussian noise to depth values using Warp (GPU)."""
    i, j = wp.tid()
    # Unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)

rep.annotators.register_augmentation(
    "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=SEED)
)

def convert_depth_to_uint8(data):
    """Normalize depth data and convert it to uint8 grayscale."""
    if isinstance(data, wp.array):
        data = data.numpy()
    depth = data.astype(np.float32, copy=False)
    depth[np.isinf(depth)] = np.nan
    mean_val = np.nanmean(depth)
    if np.isnan(mean_val):
        mean_val = 0.0
    depth = np.nan_to_num(depth, nan=mean_val, copy=False)
    min_val = depth.min()
    max_val = depth.max()
    if max_val <= min_val:
        return np.zeros(depth.shape, dtype=np.uint8)
    normalized = (depth - min_val) / (max_val - min_val)
    return (normalized * 255.0).astype(np.uint8)

def run_example(num_frames: int, resolution: tuple[int, int], use_warp: bool, env_url: str | None = None) -> float:
    """Run the capture pipeline using step() to trigger a randomization and data capture."""
    print(f"Running example with num_frames: {num_frames}, resolution: {resolution}, use_warp: {use_warp}")

    if env_url is not None and env_url != "":
        assets_root_path = get_assets_root_path()
        stage_path = assets_root_path + env_url
        print(f"Opening stage: {stage_path}")
        open_stage(stage_path)
    else:
        omni.usd.get_context().new_stage()
        rep.functional.create.dome_light(intensity=1000, rotation=(270, 0, 0))
        ground_plane = rep.functional.create.plane(scale=(10, 10, 1), position=(0, 0, 0))
        rep.functional.physics.apply_collider(ground_plane)

    # Use a fixed global seed for reproducibility
    rep.set_global_seed(SEED)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Augment the RGB and depth annotators
    rgb_to_bgr_augm = rep.annotators.Augmentation.from_function(rgb_to_bgr_wp if use_warp else rgb_to_bgr_np)
    depth_aug = rep.annotators.get_augmentation("gn_depth_wp" if use_warp else "gn_depth_np")
    rgb_to_bgr_annot = rep.annotators.augment(
        source_annotator=rep.annotators.get("rgb"),
        augmentation=rgb_to_bgr_augm,
    )
    depth_annot_1 = rep.annotators.get("distance_to_camera")
    depth_annot_1.augment(depth_aug)
    depth_annot_2 = rep.annotators.get("distance_to_camera")
    depth_annot_2.augment(depth_aug, sigma=0.5)

    # Create the render product and attach the annotators to it
    cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
    rp = rep.create.render_product(cam, resolution)
    rgb_to_bgr_annot.attach(rp)
    depth_annot_1.attach(rp)
    depth_annot_2.attach(rp)

    # Create a red cube and randomize its rotation every capture frame using a replicator randomizer graph
    red_cube = rep.functional.create.cube(position=(0, 0, 0.71))
    rep.functional.create.material(mdl="OmniPBR.mdl", bind_prims=[red_cube], diffuse_color_constant=(1, 0, 0))

    with rep.trigger.on_frame():
        red_cube_node = rep.get.prim_at_path(red_cube.GetPath())
        with red_cube_node:
            rep.randomizer.rotation()

    # Output directory
    out_dir = os.path.join(os.getcwd(), f"_out_augm_annot_{'warp' if use_warp else 'numpy'}")
    print(f"Writing data to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    capture_start = time.time()
    for frame_idx in range(num_frames):
        print(f"  Capturing frame {frame_idx + 1}/{num_frames}")
        rep.orchestrator.step(rt_subframes=32)

        # Get the data from the annotators
        rgb_data = rgb_to_bgr_annot.get_data()
        depth_data_1 = depth_annot_1.get_data()
        depth_data_2 = depth_annot_2.get_data()

        # Schedule the write of the data to disk
        write_image(path=os.path.join(out_dir, f"annot_rgb_{frame_idx}.png"), data=rgb_data)
        write_image(
            path=os.path.join(out_dir, f"annot_depth_1_{frame_idx}.png"),
            data=convert_depth_to_uint8(depth_data_1),
        )
        write_image(
            path=os.path.join(out_dir, f"annot_depth_2_{frame_idx}.png"),
            data=convert_depth_to_uint8(depth_data_2),
        )

    # Wait for the data to be written to disk and release resources
    rep.orchestrator.wait_until_complete()
    rgb_to_bgr_annot.detach()
    depth_annot_1.detach()
    depth_annot_2.detach()
    rp.destroy()

    return time.time() - capture_start

duration = run_example(num_frames, resolution, use_warp, env_url)
average = duration / num_frames if num_frames else 0.0
mode_label = "warp" if use_warp else "numpy"
print(
    f"The duration for capturing {num_frames} frames using '{mode_label}' was: {duration:.4f} seconds, "
    f"with an average of {average:.4f} seconds per frame."
)

simulation_app.close()
```

Script Editor

Full Script Editor Script

```python
import asyncio
import os
import time

import carb.settings
import numpy as np
import omni.replicator.core as rep
import warp as wp
from isaacsim.core.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path_async
from omni.replicator.core.functional import write_image

NUM_FRAMES = 5
RESOLUTION = (512, 512)
USE_WARP = False
ENV_URL = "/Isaac/Environments/Grid/default_environment.usd"
SEED = 42

# Enable warp scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

def rgb_to_bgr_np(data_in):
    """Swap RGBA red and blue channels using NumPy (CPU)."""
    data_in[:, :, [0, 2]] = data_in[:, :, [2, 0]]
    return data_in

@wp.kernel
def rgb_to_bgr_wp(data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8)):
    """Swap RGBA red and blue channels using Warp (GPU)."""
    i, j = wp.tid()
    data_out[i, j, 0] = data_in[i, j, 2]
    data_out[i, j, 1] = data_in[i, j, 1]
    data_out[i, j, 2] = data_in[i, j, 0]
    data_out[i, j, 3] = data_in[i, j, 3]

def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to depth values using NumPy (CPU)."""
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

rep.annotators.register_augmentation(
    "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=SEED)
)

@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
):
    """Add Gaussian noise to depth values using Warp (GPU)."""
    i, j = wp.tid()
    # Unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)

rep.annotators.register_augmentation(
    "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=SEED)
)

def convert_depth_to_uint8(data):
    """Normalize depth data and convert it to uint8 grayscale."""
    if isinstance(data, wp.array):
        data = data.numpy()
    depth = data.astype(np.float32, copy=False)
    depth[np.isinf(depth)] = np.nan
    mean_val = np.nanmean(depth)
    if np.isnan(mean_val):
        mean_val = 0.0
    depth = np.nan_to_num(depth, nan=mean_val, copy=False)
    min_val = depth.min()
    max_val = depth.max()
    if max_val <= min_val:
        return np.zeros(depth.shape, dtype=np.uint8)
    normalized = (depth - min_val) / (max_val - min_val)
    return (normalized * 255.0).astype(np.uint8)

# Run the capture pipeline using step() to trigger a randomization and data capture
async def run_example_async(num_frames: int, resolution: tuple[int, int], use_warp: bool) -> float:
    print(f"Running example with num_frames: {num_frames}, resolution: {resolution}, use_warp: {use_warp}")

    # Open a new stage
    assets_root_path = await get_assets_root_path_async()
    stage_path = assets_root_path + ENV_URL
    print(f"Opening stage: {stage_path}")
    open_stage(stage_path)

    # Use a fixed global seed for reproducibility
    rep.set_global_seed(SEED)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Augment the RGB and depth annotators
    rgb_to_bgr_augm = rep.annotators.Augmentation.from_function(rgb_to_bgr_wp if use_warp else rgb_to_bgr_np)
    depth_aug = rep.annotators.get_augmentation("gn_depth_wp" if use_warp else "gn_depth_np")
    rgb_to_bgr_annot = rep.annotators.augment(
        source_annotator=rep.annotators.get("rgb"),
        augmentation=rgb_to_bgr_augm,
    )
    depth_annot_1 = rep.annotators.get("distance_to_camera")
    depth_annot_1.augment(depth_aug)
    depth_annot_2 = rep.annotators.get("distance_to_camera")
    depth_annot_2.augment(depth_aug, sigma=0.5)

    # Create the render product and attach the annotators to it
    cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
    rp = rep.create.render_product(cam, resolution)
    rgb_to_bgr_annot.attach(rp)
    depth_annot_1.attach(rp)
    depth_annot_2.attach(rp)

    # Create a red cube and randomize its rotation every capture frame using a replicator randomizer graph
    red_cube = rep.functional.create.cube(position=(0, 0, 0.71))
    rep.functional.create.material(mdl="OmniPBR.mdl", bind_prims=[red_cube], diffuse_color_constant=(1, 0, 0))

    with rep.trigger.on_frame():
        red_cube_node = rep.get.prim_at_path(red_cube.GetPath())
        with red_cube_node:
            rep.randomizer.rotation()

    # Output directory
    out_dir = os.path.join(os.getcwd(), "_out_augm_annot")
    print(f"Writing data to: {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    capture_start = time.time()
    for frame_idx in range(num_frames):
        print(f"  Capturing frame {frame_idx + 1}/{num_frames}")
        await rep.orchestrator.step_async(rt_subframes=32)

        # Get the data from the annotators
        rgb_data = rgb_to_bgr_annot.get_data()
        depth_data_1 = depth_annot_1.get_data()
        depth_data_2 = depth_annot_2.get_data()

        # Schedule the write of the data to disk
        write_image(path=os.path.join(out_dir, f"annot_rgb_{frame_idx}.png"), data=rgb_data)
        write_image(
            path=os.path.join(out_dir, f"annot_depth_1_{frame_idx}.png"),
            data=convert_depth_to_uint8(depth_data_1),
        )
        write_image(
            path=os.path.join(out_dir, f"annot_depth_2_{frame_idx}.png"),
            data=convert_depth_to_uint8(depth_data_2),
        )

    # Wait for the data to be written to disk and release resources
    await rep.orchestrator.wait_until_complete_async()
    rgb_to_bgr_annot.detach()
    depth_annot_1.detach()
    depth_annot_2.detach()
    rp.destroy()

    return time.time() - capture_start

def on_task_done(task: asyncio.Task):
    """Report timing information when capture completes."""
    duration = task.result()
    average = duration / NUM_FRAMES if NUM_FRAMES else 0.0
    mode_label = "warp" if USE_WARP else "numpy"
    print(
        f"The duration for capturing {NUM_FRAMES} frames using '{mode_label}' was: {duration:.4f} seconds, "
        f"with an average of {average:.4f} seconds per frame."
    )

task = asyncio.ensure_future(run_example_async(NUM_FRAMES, RESOLUTION, USE_WARP))
task.add_done_callback(on_task_done)
```

Code Explanation

To be able to run the augmentation functions, enable scripting in the settings:

Enable Scripting

```python
# Enable warp scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
```

To augment the **rgb** data we provide for illustrative purposes a function that switches the red and blue channels in the rgb data using NumPy (CPU) and warp (GPU) kernels:

RGB to BGR using Warp and Numpy

```python
def rgb_to_bgr_np(data_in):
    """Swap RGBA red and blue channels using NumPy (CPU)."""
    data_in[:, :, [0, 2]] = data_in[:, :, [2, 0]]
    return data_in

@wp.kernel
def rgb_to_bgr_wp(data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8)):
    """Swap RGBA red and blue channels using Warp (GPU)."""
    i, j = wp.tid()
    data_out[i, j, 0] = data_in[i, j, 2]
    data_out[i, j, 1] = data_in[i, j, 1]
    data_out[i, j, 2] = data_in[i, j, 0]
    data_out[i, j, 3] = data_in[i, j, 3]
```

For the **depth** data we use gaussian noise filters. Note that the functions are registered in the annotator registry for later access:

Depth Gaussian Noise using Warp and Numpy

```python
def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to depth values using NumPy (CPU)."""
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

rep.annotators.register_augmentation(
    "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=SEED)
)

@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
):
    """Add Gaussian noise to depth values using Warp (GPU)."""
    i, j = wp.tid()
    # Unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)

rep.annotators.register_augmentation(
    "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=SEED)
)
```

Create the augmentations (warp or NumPy) once using the function directly and once from the registry:

Augmentations using Warp or Numpy

```python
# Augment the RGB and depth annotators
rgb_to_bgr_augm = rep.annotators.Augmentation.from_function(rgb_to_bgr_wp if use_warp else rgb_to_bgr_np)
depth_aug = rep.annotators.get_augmentation("gn_depth_wp" if use_warp else "gn_depth_np")
rgb_to_bgr_annot = rep.annotators.augment(
    source_annotator=rep.annotators.get("rgb"),
    augmentation=rgb_to_bgr_augm,
)
depth_annot_1 = rep.annotators.get("distance_to_camera")
depth_annot_1.augment(depth_aug)
depth_annot_2 = rep.annotators.get("distance_to_camera")
depth_annot_2.augment(depth_aug, sigma=0.5)
```

You can also register a new annotator together with its augmentation:

Register Augmentated Annotator

```python
rgb_to_bgr_annot = rep.annotators.augment(
    source_annotator=rep.annotators.get("rgb"),
    augmentation=rgb_to_bgr_augm,
)
depth_annot_1 = rep.annotators.get("distance_to_camera")
depth_annot_1.augment(depth_aug)
depth_annot_2 = rep.annotators.get("distance_to_camera")
depth_annot_2.augment(depth_aug, sigma=0.5)
```

Finally create the augmented annotators (1x **rgb**, 2x **depth**) and attach them to a render product to generate data:

Annotator Augmentation

```python
# Create the render product and attach the annotators to it
cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, resolution)
rgb_to_bgr_annot.attach(rp)
depth_annot_1.attach(rp)
depth_annot_2.attach(rp)
```

### Writer Augmentation

The **writer** example will output gaussian noise augmented RGB and depth annotator data from a writer.

Standalone Application

The example can be run as a standalone application using the following commands in the terminal (on Windows use `python.bat` instead of `python.sh`):

> ```python
> ./python.sh standalone_examples/replicator/augmentation/writer_augmentation.py
> ```
>
> Optionally the following arguments can be used to change the default behavior:
>
> * `--use_warp` – flag to use warp (GPU) instead of NumPy (CPU) for the augmentation functions (default: False)
> * `--num_frames` – the number of frames to be captured (default: 25)
>
> ```python
> ./python.sh standalone_examples/replicator/augmentation/writer_augmentation.py --use_warp --num_frames 25
> ```
>
> Full Standalone Script
>
> ```python
> """Generate augmented synthetic from a writer"""
>
> from isaacsim import SimulationApp
>
> simulation_app = SimulationApp(launch_config={"headless": False})
>
> import argparse
> import os
> import time
>
> import carb.settings
> import numpy as np
> import omni.replicator.core as rep
> import omni.usd
> import warp as wp
> from isaacsim.core.utils.stage import open_stage
> from isaacsim.storage.native import get_assets_root_path
>
> parser = argparse.ArgumentParser()
> parser.add_argument("--num_frames", type=int, default=5, help="The number of frames to capture")
> parser.add_argument(
>     "--use_warp",
>     action="store_true",
>     help="Use warp augmentations instead of numpy",
> )
> parser.add_argument("--resolution", nargs=2, type=int, default=[512, 512], help="Camera resolution")
> parser.add_argument("--env_url", type=str, default="", help="USD environment URL (empty for basic scene)")
> args, unknown = parser.parse_known_args()
>
> num_frames = args.num_frames
> use_warp = args.use_warp
> resolution = args.resolution
> env_url = args.env_url or None
> SEED = 42
>
> # Enable warp scripts
> carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
>
>
> def gaussian_noise_rgb_np(data_in, sigma: float, seed: int):
>     """Add Gaussian noise to RGB data using NumPy (CPU)."""
>     np.random.seed(seed)
>     # Convert to float32 space
>     data_in = data_in.astype(np.float32)
>     # Add Gaussian noise to each channel
>     data_in[:, :, 0] = data_in[:, :, 0] + np.random.randn(*data_in.shape[:-1]) * sigma
>     data_in[:, :, 1] = data_in[:, :, 1] + np.random.randn(*data_in.shape[:-1]) * sigma
>     data_in[:, :, 2] = data_in[:, :, 2] + np.random.randn(*data_in.shape[:-1]) * sigma
>     # Clip to [0, 255] and convert to uint8
>     data_in = np.clip(data_in, 0, 255).astype(np.uint8)
>     return data_in
>
>
> @wp.kernel
> def gaussian_noise_rgb_wp(
>     data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8), sigma: float, seed: int
> ):
>     """Add Gaussian noise to RGB data using Warp (GPU)."""
>     # Get thread coordinates and image dimensions to calculate unique pixel ID for random generation
>     i, j = wp.tid()
>     dim_i = data_in.shape[0]
>     dim_j = data_in.shape[1]
>     pixel_id = i * dim_i + j
>
>     # Use pixel_id as offset to create unique seeds for each pixel and channel (ensure independent noise patterns across R,G,B channels)
>     state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
>     state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
>     state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))
>
>     # Apply noise to each channel independently using unique seeds; work in float32 space, then clip and convert to uint8
>     val_r = wp.float32(data_in[i, j, 0]) + sigma * wp.randn(state_r)
>     val_g = wp.float32(data_in[i, j, 1]) + sigma * wp.randn(state_g)
>     val_b = wp.float32(data_in[i, j, 2]) + sigma * wp.randn(state_b)
>
>     # Clip to [0, 255] and convert to uint8
>     data_out[i, j, 0] = wp.uint8(wp.clamp(val_r, 0.0, 255.0))
>     data_out[i, j, 1] = wp.uint8(wp.clamp(val_g, 0.0, 255.0))
>     data_out[i, j, 2] = wp.uint8(wp.clamp(val_b, 0.0, 255.0))
>     data_out[i, j, 3] = data_in[i, j, 3]
>
>
> def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
>     """Add Gaussian noise to depth values using NumPy (CPU)."""
>     np.random.seed(seed)
>     result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
>     return np.clip(result, 0, None).astype(data_in.dtype)
>
>
> rep.AnnotatorRegistry.register_augmentation(
>     "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=None)
> )
>
>
> @wp.kernel
> def gaussian_noise_depth_wp(
>     data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
> ):
>     """Add Gaussian noise to depth values using Warp (GPU)."""
>     i, j = wp.tid()
>     # Unique ID for random seed per pixel
>     scalar_pixel_id = i * data_in.shape[1] + j
>     state = wp.rand_init(seed, scalar_pixel_id)
>     data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)
>
>
> rep.AnnotatorRegistry.register_augmentation(
>     "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=None)
> )
>
>
> def run_example(num_frames: int, resolution: tuple[int, int], use_warp: bool, env_url: str | None = None) -> float:
>     """Run the capture pipeline using step() to trigger a randomization and data capture."""
>     print(f"Running example with num_frames: {num_frames}, resolution: {resolution}, use_warp: {use_warp}")
>
>     if env_url is not None and env_url != "":
>         assets_root_path = get_assets_root_path()
>         stage_path = assets_root_path + env_url
>         print(f"Opening stage: {stage_path}")
>         open_stage(stage_path)
>     else:
>         omni.usd.get_context().new_stage()
>         rep.functional.create.dome_light(intensity=1000, rotation=(270, 0, 0))
>         ground_plane = rep.functional.create.plane(scale=(10, 10, 1), position=(0, 0, 0))
>         rep.functional.physics.apply_collider(ground_plane)
>
>     # Use a fixed global seed for reproducibility
>     rep.set_global_seed(SEED)
>
>     # Disable capture on play, data is captured manually using the step function
>     rep.orchestrator.set_capture_on_play(False)
>
>     # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
>     carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
>
>     # Augment the annotators
>     rgb_to_hsv_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_rgb_to_hsv)
>     hsv_to_rgb_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_hsv_to_rgb)
>
>     # Augment the RGB and depth annotators
>     gn_rgb_augm = rep.annotators.Augmentation.from_function(
>         gaussian_noise_rgb_wp if use_warp else gaussian_noise_rgb_np, sigma=15.0, seed=SEED
>     )
>     gn_depth_augm = rep.annotators.get_augmentation("gn_depth_wp" if use_warp else "gn_depth_np")
>
>     # Create a writer and apply the augmentations to its corresponding annotators
>     out_dir = os.path.join(os.getcwd(), f"_out_augm_writer_{'warp' if use_warp else 'numpy'}")
>     backend = rep.backends.get("DiskBackend")
>     backend.initialize(output_dir=out_dir)
>     print(f"Writing data to: {out_dir}")
>     writer = rep.writers.get("BasicWriter")
>     writer.initialize(backend=backend, rgb=True, distance_to_camera=True, colorize_depth=True)
>
>     # Apply the augmentations to the RGB and depth annotators
>     augmented_rgb_annot = rep.annotators.get("rgb").augment_compose(
>         [rgb_to_hsv_augm, gn_rgb_augm, hsv_to_rgb_augm], name="rgb"
>     )
>     writer.add_annotator(augmented_rgb_annot)
>     writer.augment_annotator("distance_to_camera", gn_depth_augm)
>
>     # Create a camera and a render product and attach them to the writer
>     cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
>     rp = rep.create.render_product(cam, resolution)
>     writer.attach(rp)
>
>     # Create a red cube and randomize its rotation every capture frame using a replicator randomizer graph
>     red_cube = rep.functional.create.cube(position=(0, 0, 0.71))
>     rep.functional.create.material(mdl="OmniPBR.mdl", bind_prims=[red_cube], diffuse_color_constant=(1, 0, 0))
>     with rep.trigger.on_frame():
>         red_cube_node = rep.get.prim_at_path(red_cube.GetPath())
>         with red_cube_node:
>             rep.randomizer.rotation()
>
>     capture_start = time.time()
>     for frame_idx in range(num_frames):
>         print(f"  Capturing frame {frame_idx + 1}/{num_frames}")
>         rep.orchestrator.step(rt_subframes=32)
>
>     # Wait for the data to be written to disk and release resources
>     rep.orchestrator.wait_until_complete()
>     writer.detach()
>     rp.destroy()
>
>     return time.time() - capture_start
>
>
> duration = run_example(num_frames, resolution, use_warp, env_url)
> average = duration / num_frames if num_frames else 0.0
> mode_label = "warp" if use_warp else "numpy"
> print(
>     f"The duration for capturing {num_frames} frames using '{mode_label}' was: {duration:.4f} seconds, "
>     f"with an average of {average:.4f} seconds per frame."
> )
>
> simulation_app.close()
> ```

Script Editor

Full Script Editor Script

```python
import asyncio
import os
import time

import carb.settings
import numpy as np
import omni.replicator.core as rep
import warp as wp
from isaacsim.core.utils.stage import open_stage
from isaacsim.storage.native import get_assets_root_path_async

NUM_FRAMES = 5
RESOLUTION = (512, 512)
USE_WARP = False
ENV_URL = "/Isaac/Environments/Grid/default_environment.usd"
SEED = 42

# Enable warp scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

def gaussian_noise_rgb_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to RGB data using NumPy (CPU)."""
    np.random.seed(seed)
    # Convert to float32 space
    data_in = data_in.astype(np.float32)
    # Add Gaussian noise to each channel
    data_in[:, :, 0] = data_in[:, :, 0] + np.random.randn(*data_in.shape[:-1]) * sigma
    data_in[:, :, 1] = data_in[:, :, 1] + np.random.randn(*data_in.shape[:-1]) * sigma
    data_in[:, :, 2] = data_in[:, :, 2] + np.random.randn(*data_in.shape[:-1]) * sigma
    # Clip to [0, 255] and convert to uint8
    data_in = np.clip(data_in, 0, 255).astype(np.uint8)
    return data_in

@wp.kernel
def gaussian_noise_rgb_wp(
    data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8), sigma: float, seed: int
):
    """Add Gaussian noise to RGB data using Warp (GPU)."""
    # Get thread coordinates and image dimensions to calculate unique pixel ID for random generation
    i, j = wp.tid()
    dim_i = data_in.shape[0]
    dim_j = data_in.shape[1]
    pixel_id = i * dim_i + j

    # Use pixel_id as offset to create unique seeds for each pixel and channel (ensure independent noise patterns across R,G,B channels)
    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Apply noise to each channel independently using unique seeds; work in float32 space, then clip and convert to uint8
    val_r = wp.float32(data_in[i, j, 0]) + sigma * wp.randn(state_r)
    val_g = wp.float32(data_in[i, j, 1]) + sigma * wp.randn(state_g)
    val_b = wp.float32(data_in[i, j, 2]) + sigma * wp.randn(state_b)

    # Clip to [0, 255] and convert to uint8
    data_out[i, j, 0] = wp.uint8(wp.clamp(val_r, 0.0, 255.0))
    data_out[i, j, 1] = wp.uint8(wp.clamp(val_g, 0.0, 255.0))
    data_out[i, j, 2] = wp.uint8(wp.clamp(val_b, 0.0, 255.0))
    data_out[i, j, 3] = data_in[i, j, 3]

def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to depth values using NumPy (CPU)."""
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

rep.annotators.register_augmentation(
    "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=SEED)
)

@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
):
    """Add Gaussian noise to depth values using Warp (GPU)."""
    i, j = wp.tid()
    # Unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)

rep.annotators.register_augmentation(
    "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=SEED)
)

# Run the capture pipeline using step() to trigger a randomization and data capture
async def run_example_async(num_frames: int, resolution: tuple[int, int], use_warp: bool) -> float:
    print(f"Running example with num_frames: {num_frames}, resolution: {resolution}, use_warp: {use_warp}")

    # Open a new stage
    assets_root_path = await get_assets_root_path_async()
    stage_path = assets_root_path + ENV_URL
    print(f"Opening stage: {stage_path}")
    open_stage(stage_path)

    # Use a fixed global seed for reproducibility
    rep.set_global_seed(SEED)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Augment the annotators
    rgb_to_hsv_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_rgb_to_hsv)
    hsv_to_rgb_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_hsv_to_rgb)

    # Augment the RGB and depth annotators
    gn_rgb_augm = rep.annotators.Augmentation.from_function(
        gaussian_noise_rgb_wp if use_warp else gaussian_noise_rgb_np, sigma=15.0, seed=SEED
    )
    gn_depth_augm = rep.annotators.get_augmentation("gn_depth_wp" if use_warp else "gn_depth_np")

    # Create a writer and apply the augmentations to its corresponding annotators
    out_dir = os.path.join(os.getcwd(), "_out_augm_writer")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out_dir)
    print(f"Writing data to: {out_dir}")
    writer = rep.writers.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True, distance_to_camera=True, colorize_depth=True)

    # Apply the augmentations to the RGB and depth annotators
    augmented_rgb_annot = rep.annotators.get("rgb").augment_compose(
        [rgb_to_hsv_augm, gn_rgb_augm, hsv_to_rgb_augm], name="rgb"
    )
    writer.add_annotator(augmented_rgb_annot)
    writer.augment_annotator("distance_to_camera", gn_depth_augm)

    # Create a camera and a render product and attach them to the writer
    cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
    rp = rep.create.render_product(cam, resolution)
    writer.attach(rp)

    # Create a red cube and randomize its rotation every capture frame using a replicator randomizer graph
    red_cube = rep.functional.create.cube(position=(0, 0, 0.71))
    rep.functional.create.material(mdl="OmniPBR.mdl", bind_prims=[red_cube], diffuse_color_constant=(1, 0, 0))
    with rep.trigger.on_frame():
        red_cube_node = rep.get.prim_at_path(red_cube.GetPath())
        with red_cube_node:
            rep.randomizer.rotation()

    capture_start = time.time()
    for frame_idx in range(num_frames):
        print(f"  Capturing frame {frame_idx + 1}/{num_frames}")
        await rep.orchestrator.step_async(rt_subframes=32)

    # Wait for the data to be written to disk and release resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

    return time.time() - capture_start

def on_task_done(task: asyncio.Task):
    """Report timing information when capture completes."""
    duration = task.result()
    average = duration / NUM_FRAMES if NUM_FRAMES else 0.0
    mode_label = "warp" if USE_WARP else "numpy"
    print(
        f"The duration for capturing {NUM_FRAMES} frames using '{mode_label}' was: {duration:.4f} seconds, "
        f"with an average of {average:.4f} seconds per frame."
    )

task = asyncio.ensure_future(run_example_async(NUM_FRAMES, RESOLUTION, USE_WARP))
task.add_done_callback(on_task_done)
```

Code Explanation

To be able to run the augmentation functions one needs to enable scripting in the settings:

Enable Scripting

```python
# Enable warp scripts
carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)
```

For the **rgb** (**LdrColor**) annotator of the writer, we provide gaussian noise functions using NumPy (CPU) and warp (GPU) kernels, applied on the RGB channels of the RGBA provided data format.

RGB Gaussian Noise using Warp and Numpy

```python
def gaussian_noise_rgb_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to RGB data using NumPy (CPU)."""
    np.random.seed(seed)
    # Convert to float32 space
    data_in = data_in.astype(np.float32)
    # Add Gaussian noise to each channel
    data_in[:, :, 0] = data_in[:, :, 0] + np.random.randn(*data_in.shape[:-1]) * sigma
    data_in[:, :, 1] = data_in[:, :, 1] + np.random.randn(*data_in.shape[:-1]) * sigma
    data_in[:, :, 2] = data_in[:, :, 2] + np.random.randn(*data_in.shape[:-1]) * sigma
    # Clip to [0, 255] and convert to uint8
    data_in = np.clip(data_in, 0, 255).astype(np.uint8)
    return data_in

@wp.kernel
def gaussian_noise_rgb_wp(
    data_in: wp.array3d(dtype=wp.uint8), data_out: wp.array3d(dtype=wp.uint8), sigma: float, seed: int
):
    """Add Gaussian noise to RGB data using Warp (GPU)."""
    # Get thread coordinates and image dimensions to calculate unique pixel ID for random generation
    i, j = wp.tid()
    dim_i = data_in.shape[0]
    dim_j = data_in.shape[1]
    pixel_id = i * dim_i + j

    # Use pixel_id as offset to create unique seeds for each pixel and channel
    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Apply noise to each channel independently using unique seeds
    val_r = wp.float32(data_in[i, j, 0]) + sigma * wp.randn(state_r)
    val_g = wp.float32(data_in[i, j, 1]) + sigma * wp.randn(state_g)
    val_b = wp.float32(data_in[i, j, 2]) + sigma * wp.randn(state_b)

    # Clip to [0, 255] and convert to uint8
    data_out[i, j, 0] = wp.uint8(wp.clamp(val_r, 0.0, 255.0))
    data_out[i, j, 1] = wp.uint8(wp.clamp(val_g, 0.0, 255.0))
    data_out[i, j, 2] = wp.uint8(wp.clamp(val_b, 0.0, 255.0))
    data_out[i, j, 3] = data_in[i, j, 3]
```

For the **depth** annotator of the writer, there are gaussian noise functions using NumPy (CPU) and warp (GPU) kernels, applied on the 2D array of float32 values. The functions are registered in the annotator registry for later access:

Depth Gaussian Noise using Warp and Numpy

```python
def gaussian_noise_depth_np(data_in, sigma: float, seed: int):
    """Add Gaussian noise to depth values using NumPy (CPU)."""
    np.random.seed(seed)
    result = data_in.astype(np.float32) + np.random.randn(*data_in.shape) * sigma
    return np.clip(result, 0, None).astype(data_in.dtype)

rep.AnnotatorRegistry.register_augmentation(
    "gn_depth_np", rep.annotators.Augmentation.from_function(gaussian_noise_depth_np, sigma=0.1, seed=None)
)

@wp.kernel
def gaussian_noise_depth_wp(
    data_in: wp.array2d(dtype=wp.float32), data_out: wp.array2d(dtype=wp.float32), sigma: float, seed: int
):
    """Add Gaussian noise to depth values using Warp (GPU)."""
    i, j = wp.tid()
    # Unique ID for random seed per pixel
    scalar_pixel_id = i * data_in.shape[1] + j
    state = wp.rand_init(seed, scalar_pixel_id)
    data_out[i, j] = data_in[i, j] + sigma * wp.randn(state)

rep.AnnotatorRegistry.register_augmentation(
    "gn_depth_wp", rep.annotators.Augmentation.from_function(gaussian_noise_depth_wp, sigma=0.1, seed=None)
)
```

Access the default (**rgb**) augmentations from replicator:

Built-in Replicator Augmentations

```python
# Augment the annotators
rgb_to_hsv_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_rgb_to_hsv)
hsv_to_rgb_augm = rep.annotators.Augmentation.from_function(rep.augmentations_default.aug_hsv_to_rgb)
```

Furthermore the custom augmentations are created (warp or NumPy), after using the function directly and once from the registry:

Augmentations using Warp or Numpy

```python
# Augment the RGB and depth annotators
gn_rgb_augm = rep.annotators.Augmentation.from_function(
    gaussian_noise_rgb_wp if use_warp else gaussian_noise_rgb_np, sigma=15.0, seed=SEED
)
gn_depth_augm = rep.annotators.get_augmentation("gn_depth_wp" if use_warp else "gn_depth_np")
```

Finally the writer is created and initialized to use the **rgb** and **depth** (**distance\_to\_camera**) annotators. The built-in `rgb` annotator is replaced by a new augmented one by using the same `name="rgb"` name and adding it to the writer (`add_annotator`). The augmented RGB annotator uses a composition by switching the data to hsv, adding gaussian noise, and switching back to RGB. The `distance_to_camera` annotator is augmented by using the built-in `augment_annotator` function:

Writer Augmentation

```python
# Create a writer and apply the augmentations to its corresponding annotators
out_dir = os.path.join(os.getcwd(), f"_out_augm_writer_{'warp' if use_warp else 'numpy'}")
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=out_dir)
print(f"Writing data to: {out_dir}")
writer = rep.writers.get("BasicWriter")
writer.initialize(backend=backend, rgb=True, distance_to_camera=True, colorize_depth=True)

# Apply the augmentations to the RGB and depth annotators
augmented_rgb_annot = rep.annotators.get("rgb").augment_compose(
    [rgb_to_hsv_augm, gn_rgb_augm, hsv_to_rgb_augm], name="rgb"
)
writer.add_annotator(augmented_rgb_annot)
writer.augment_annotator("distance_to_camera", gn_depth_augm)

# Create a camera and a render product and attach them to the writer
cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))
rp = rep.create.render_product(cam, resolution)
writer.attach(rp)
```

---

# Custom Replicator Randomization Nodes

This tutorial provides an example of how to create custom randomization nodes for the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension.

## Learning Objectives

The goal of this tutorial is to demonstrate how to create custom [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") randomization nodes. These nodes can then be further integrated into the Synthetic Data Generation (SDG) pipeline graph of [Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)").

This tutorial will showcase how to:

* Create custom scene randomization Python scripts.
* Wrap the scripts as OmniGraph nodes and manually add them to an existing SDG pipeline graph.
* Encapsulate the OmniGraph nodes as **ReplicatorItems** to be automatically added to the SDG pipeline graph using Replicator’s API.

## Prerequisites

* Familiarity with USD / Isaac Sim APIs for creating custom scene randomizers. See [Randomization Snippets](Synthetic_Data_Generation.md) for more details.
* Familiarity with [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") and its randomization API [replicator randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)").
* Basic knowledge of [OmniGraph](https://docs.omniverse.nvidia.com/extensions/latest/ext_omnigraph.html "(in Omniverse Extensions)") and how to create [OmniGraph Nodes](https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/WritingNodes.html#omnigraph-nodes).
* Experience running simulations via the [Script Editor](Development_Tools.md).

## Implementation

This tutorial will showcase how to create custom scene randomization Python scripts. These scripts will create prims in a new stage and randomize their rotation and locations: **in a sphere**, **on a sphere**, and **between two spheres**.

The following image shows the result after running the randomization in the Script Editor:

Code Explanation

The following functions take as input the radius (or radii) of the spheres and generate a random 3D point on the surface of a sphere, within a sphere, and between two spheres. These points will determine the prim locations.

Randomization Functions

```python
# Generate a random 3D point on the surface of a sphere of a given radius.
def random_point_on_sphere(radius):
    # Generate a random direction by spherical coordinates (phi, theta)
    phi = random.uniform(0, 2 * math.pi)
    # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
    costheta = random.uniform(-1, 1)
    theta = math.acos(costheta)

    # Convert from spherical to Cartesian coordinates
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    return x, y, z

# Generate a random 3D point within a sphere of a given radius, ensuring a uniform distribution throughout the volume.
def random_point_in_sphere(radius):
    # Generate a random direction by spherical coordinates (phi, theta)
    phi = random.uniform(0, 2 * math.pi)
    # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
    costheta = random.uniform(-1, 1)
    theta = math.acos(costheta)

    # Scale the radius uniformly within the sphere, applying the cube root to a random value
    # to account for volume's cubic growth with radius (r^3), ensuring spatial uniformity.
    r = radius * (random.random() ** (1 / 3))

    # Convert from spherical to Cartesian coordinates
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z

# Generate a random 3D point between two spheres, ensuring a uniform distribution throughout the volume.
def random_point_between_spheres(radius1, radius2):
    # Ensure radius1 < radius2
    if radius1 > radius2:
        radius1, radius2 = radius2, radius1

    # Generate a random direction by spherical coordinates (phi, theta)
    phi = random.uniform(0, 2 * math.pi)
    # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
    costheta = random.uniform(-1, 1)
    theta = math.acos(costheta)

    # Uniformly distribute points between two spheres by weighting the radius to match volume growth (r^3),
    # ensuring spatial uniformity by taking the cube root of a value between the radii cubed.
    r = (random.uniform(radius1**3, radius2**3)) ** (1 / 3.0)

    # Convert from spherical to Cartesian coordinates
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z
```

The following snippet creates prims in a new stage and randomizes their rotation and locations using the previously defined functions.

Spawning and Randomizing Prims

```python
stage = omni.usd.get_context().get_stage()
prim_count = 500
prim_scale = 0.1
rad_in = 0.5
rad_on = 1.5
rad_bet1 = 2.5
rad_bet2 = 3.5

# Create the default prims
on_sphere_prims = [stage.DefinePrim(f"/World/sphere_{i}", "Sphere") for i in range(prim_count)]
in_sphere_prims = [stage.DefinePrim(f"/World/cube_{i}", "Cube") for i in range(prim_count)]
between_spheres_prims = [stage.DefinePrim(f"/World/cylinder_{i}", "Cylinder") for i in range(prim_count)]

# Add xformOps and scale to the prims
for prim in chain(on_sphere_prims, in_sphere_prims, between_spheres_prims):
    if not prim.HasAttribute("xformOp:translate"):
        UsdGeom.Xformable(prim).AddTranslateOp()
    if not prim.HasAttribute("xformOp:scale"):
        UsdGeom.Xformable(prim).AddScaleOp()
    if not prim.HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(prim).AddRotateXYZOp()
    prim.GetAttribute("xformOp:scale").Set((prim_scale, prim_scale, prim_scale))

# Randomize the prims
for _ in range(10):
    for in_sphere_prim in in_sphere_prims:
        rand_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        in_sphere_prim.GetAttribute("xformOp:rotateXYZ").Set(rand_rot)
        rand_loc = random_point_in_sphere(rad_in)
        in_sphere_prim.GetAttribute("xformOp:translate").Set(rand_loc)

    for on_sphere_prim in on_sphere_prims:
        rand_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        on_sphere_prim.GetAttribute("xformOp:rotateXYZ").Set(rand_rot)
        rand_loc = random_point_on_sphere(rad_on)
        on_sphere_prim.GetAttribute("xformOp:translate").Set(rand_loc)

    for between_spheres_prim in between_spheres_prims:
        rand_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        between_spheres_prim.GetAttribute("xformOp:rotateXYZ").Set(rand_rot)
        rand_loc = random_point_between_spheres(rad_bet1, rad_bet2)
        between_spheres_prim.GetAttribute("xformOp:translate").Set(rand_loc)
```

Script Editor

Snippet to run in the Script Editor:

Full Script Editor Script

```python
import math
import random
from itertools import chain

import omni.replicator.core as rep
import omni.usd
from pxr import UsdGeom

# Generate a random 3D point on the surface of a sphere of a given radius.
def random_point_on_sphere(radius):
    # Generate a random direction by spherical coordinates (phi, theta)
    phi = random.uniform(0, 2 * math.pi)
    # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
    costheta = random.uniform(-1, 1)
    theta = math.acos(costheta)

    # Convert from spherical to Cartesian coordinates
    x = radius * math.sin(theta) * math.cos(phi)
    y = radius * math.sin(theta) * math.sin(phi)
    z = radius * math.cos(theta)

    return x, y, z

# Generate a random 3D point within a sphere of a given radius, ensuring a uniform distribution throughout the volume.
def random_point_in_sphere(radius):
    # Generate a random direction by spherical coordinates (phi, theta)
    phi = random.uniform(0, 2 * math.pi)
    # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
    costheta = random.uniform(-1, 1)
    theta = math.acos(costheta)

    # Scale the radius uniformly within the sphere, applying the cube root to a random value
    # to account for volume's cubic growth with radius (r^3), ensuring spatial uniformity.
    r = radius * (random.random() ** (1 / 3))

    # Convert from spherical to Cartesian coordinates
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z

# Generate a random 3D point between two spheres, ensuring a uniform distribution throughout the volume.
def random_point_between_spheres(radius1, radius2):
    # Ensure radius1 < radius2
    if radius1 > radius2:
        radius1, radius2 = radius2, radius1

    # Generate a random direction by spherical coordinates (phi, theta)
    phi = random.uniform(0, 2 * math.pi)
    # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
    costheta = random.uniform(-1, 1)
    theta = math.acos(costheta)

    # Uniformly distribute points between two spheres by weighting the radius to match volume growth (r^3),
    # ensuring spatial uniformity by taking the cube root of a value between the radii cubed.
    r = (random.uniform(radius1**3, radius2**3)) ** (1 / 3.0)

    # Convert from spherical to Cartesian coordinates
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)

    return x, y, z

stage = omni.usd.get_context().get_stage()
prim_count = 500
prim_scale = 0.1
rad_in = 0.5
rad_on = 1.5
rad_bet1 = 2.5
rad_bet2 = 3.5

# Create the default prims
on_sphere_prims = [stage.DefinePrim(f"/World/sphere_{i}", "Sphere") for i in range(prim_count)]
in_sphere_prims = [stage.DefinePrim(f"/World/cube_{i}", "Cube") for i in range(prim_count)]
between_spheres_prims = [stage.DefinePrim(f"/World/cylinder_{i}", "Cylinder") for i in range(prim_count)]

# Add xformOps and scale to the prims
for prim in chain(on_sphere_prims, in_sphere_prims, between_spheres_prims):
    if not prim.HasAttribute("xformOp:translate"):
        UsdGeom.Xformable(prim).AddTranslateOp()
    if not prim.HasAttribute("xformOp:scale"):
        UsdGeom.Xformable(prim).AddScaleOp()
    if not prim.HasAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(prim).AddRotateXYZOp()
    prim.GetAttribute("xformOp:scale").Set((prim_scale, prim_scale, prim_scale))

# Randomize the prims
for _ in range(10):
    for in_sphere_prim in in_sphere_prims:
        rand_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        in_sphere_prim.GetAttribute("xformOp:rotateXYZ").Set(rand_rot)
        rand_loc = random_point_in_sphere(rad_in)
        in_sphere_prim.GetAttribute("xformOp:translate").Set(rand_loc)

    for on_sphere_prim in on_sphere_prims:
        rand_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        on_sphere_prim.GetAttribute("xformOp:rotateXYZ").Set(rand_rot)
        rand_loc = random_point_on_sphere(rad_on)
        on_sphere_prim.GetAttribute("xformOp:translate").Set(rand_loc)

    for between_spheres_prim in between_spheres_prims:
        rand_rot = (random.uniform(0, 360), random.uniform(0, 360), random.uniform(0, 360))
        between_spheres_prim.GetAttribute("xformOp:rotateXYZ").Set(rand_rot)
        rand_loc = random_point_between_spheres(rad_bet1, rad_bet2)
        between_spheres_prim.GetAttribute("xformOp:translate").Set(rand_loc)
```

As a next step, custom [OmniGraph Nodes](https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/WritingNodes.html#omnigraph-nodes) are created for the randomization functions. The node descriptions and implementations can be found in the following code snippets:

Node Descriptions

OgnSampleInSphere.ogn

```python
{
    "OgnSampleInSphere": {
        "version": 1,
        "description": "Assigns a uniformly sampled location in a sphere.",
        "language": "Python",
        "metadata": {
            "uiName": "Sample In Sphere"
        },
        "inputs": {
            "prims": {
                "type": "target",
                "description": "prims to randomize",
                "default": []
            },
            "execIn": {
                "type": "execution",
                "description": "exec",
                "default": 0
            },
            "radius": {
                "type": "float",
                "description": "sphere radius",
                "default": 1.0
            }
        },
        "outputs": {
            "execOut": {
                "type": "execution",
                "description": "exec"
            }
        }
    }
}
```

OgnSampleOnSphere.ogn

```python
{
    "OgnSampleOnSphere": {
        "version": 1,
        "description": "Assignes uniformly sampled location on a sphere.",
        "language": "Python",
        "metadata": {
            "uiName": "Sample On Sphere"
        },
        "inputs": {
            "prims": {
                "type": "target",
                "description": "prims to randomize",
                "default": []
            },
            "execIn": {
                "type": "execution",
                "description": "exec",
                "default": 0
            },
            "radius": {
                "type": "float",
                "description": "sphere radius",
                "default": 1.0
            }
        },
        "outputs": {
            "execOut": {
                "type": "execution",
                "description": "exec"
            }
        }
    }
}
```

OgnSampleBetweenSpheres.ogn

```python
{
    "OgnSampleBetweenSpheres": {
        "version": 1,
        "description": "Assigns a uniformly sampled location between two spheres.",
        "language": "Python",
        "metadata": {
            "uiName": "Sample Between Spheres"
        },
        "inputs": {
            "prims": {
                "type": "target",
                "description": "prims to randomize",
                "default": []
            },
            "execIn": {
                "type": "execution",
                "description": "exec",
                "default": 0
            },
            "radius1": {
                "type": "float",
                "description": "inner sphere radius",
                "default": 0.5
            },
            "radius2": {
                "type": "float",
                "description": "outer sphere radius",
                "default": 1.0
            }
        },
        "outputs": {
            "execOut": {
                "type": "execution",
                "description": "exec"
            }
        }
    }
}
```

Node Implementations

OgnSampleInSphere.py

```python
import numpy as np
import omni.graph.core as og
import omni.usd
from pxr import Sdf, UsdGeom

class OgnSampleInSphere:
    @staticmethod
    def compute(db) -> bool:
        prim_paths = db.inputs.prims
        if len(prim_paths) == 0:
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        stage = omni.usd.get_context().get_stage()
        prims = [stage.GetPrimAtPath(str(path)) for path in prim_paths]

        radius = db.inputs.radius

        try:
            for prim in prims:
                if not UsdGeom.Xformable(prim):
                    prim_type = prim.GetTypeName()
                    raise ValueError(
                        f"Expected prim at {prim.GetPath()} to be an Xformable prim but got type {prim_type}"
                    )
                if not prim.HasAttribute("xformOp:translate"):
                    UsdGeom.Xformable(prim).AddTranslateOp()
            if radius <= 0:
                raise ValueError(f"Radius must be positive, got {radius}")

        except Exception as error:
            db.log_error(str(error))
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        samples = []
        for _ in range(len(prims)):
            # Generate a random direction by spherical coordinates (phi, theta)
            phi = np.random.uniform(0, 2 * np.pi)
            # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)

            # Scale the radius uniformly within the sphere, applying the cube root to a random value
            # to account for volume's cubic growth with radius (r^3), ensuring spatial uniformity.
            r = radius * (np.random.random() ** (1 / 3))

            # Convert from spherical to Cartesian coordinates
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            samples.append((x, y, z))

        with Sdf.ChangeBlock():
            for prim, sample in zip(prims, samples):
                prim.GetAttribute("xformOp:translate").Set(sample)

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
```

OgnSampleOnSphere.py

```python
import numpy as np
import omni.graph.core as og
import omni.usd
from pxr import Sdf, UsdGeom

class OgnSampleOnSphere:
    @staticmethod
    def compute(db) -> bool:
        prim_paths = db.inputs.prims
        if len(prim_paths) == 0:
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        stage = omni.usd.get_context().get_stage()
        prims = [stage.GetPrimAtPath(str(path)) for path in prim_paths]

        radius = db.inputs.radius

        try:
            for prim in prims:
                if not UsdGeom.Xformable(prim):
                    prim_type = prim.GetTypeName()
                    raise ValueError(
                        f"Expected prim at {prim.GetPath()} to be an Xformable prim but got type {prim_type}"
                    )
                if not prim.HasAttribute("xformOp:translate"):
                    UsdGeom.Xformable(prim).AddTranslateOp()
            if radius <= 0:
                raise ValueError(f"Radius must be positive, got {radius}")

        except Exception as error:
            db.log_error(str(error))
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        samples = []
        for _ in range(len(prims)):
            # Generate a random direction by spherical coordinates (phi, theta)
            phi = np.random.uniform(0, 2 * np.pi)
            # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)

            # Convert from spherical to Cartesian coordinates
            x = radius * np.sin(theta) * np.cos(phi)
            y = radius * np.sin(theta) * np.sin(phi)
            z = radius * np.cos(theta)

            samples.append((x, y, z))

        with Sdf.ChangeBlock():
            for prim, sample in zip(prims, samples):
                prim.GetAttribute("xformOp:translate").Set(sample)

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
```

OgnSampleBetweenSpheres.py

```python
import numpy as np
import omni.graph.core as og
import omni.usd
from pxr import Sdf, UsdGeom

class OgnSampleBetweenSpheres:
    @staticmethod
    def compute(db) -> bool:
        prim_paths = db.inputs.prims
        if len(prim_paths) == 0:
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        stage = omni.usd.get_context().get_stage()
        prims = [stage.GetPrimAtPath(str(path)) for path in prim_paths]

        radius1 = db.inputs.radius1
        radius2 = db.inputs.radius2

        # Ensure radius1 < radius2
        if radius1 > radius2:
            radius1, radius2 = radius2, radius1

        try:
            for prim in prims:
                if not UsdGeom.Xformable(prim):
                    prim_type = prim.GetTypeName()
                    raise ValueError(
                        f"Expected prim at {prim.GetPath()} to be an Xformable prim but got type {prim_type}"
                    )
                if not prim.HasAttribute("xformOp:translate"):
                    UsdGeom.Xformable(prim).AddTranslateOp()
            if radius1 < 0 or radius2 <= 0:
                raise ValueError(
                    f"Radius must be positive and larger radius larger than 0, got {radius1} and {radius2}"
                )

        except Exception as error:
            db.log_error(str(error))
            db.outputs.execOut = og.ExecutionAttributeState.DISABLED
            return False

        samples = []
        for _ in range(len(prims)):
            # Generate a random direction by spherical coordinates (phi, theta)
            phi = np.random.uniform(0, 2 * np.pi)
            # Sample costheta to ensure uniform distribution of points on the sphere (surface is proportional to sin(theta))
            costheta = np.random.uniform(-1, 1)
            theta = np.arccos(costheta)

            # Uniformly distribute points between two spheres by weighting the radius to match volume growth (r^3),
            # ensuring spatial uniformity by taking the cube root of a value between the radii cubed.
            r = (np.random.uniform(radius1**3, radius2**3)) ** (1 / 3.0)

            # Convert from spherical to Cartesian coordinates
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            samples.append((x, y, z))

        with Sdf.ChangeBlock():
            for prim, sample in zip(prims, samples):
                prim.GetAttribute("xformOp:translate").Set(sample)

        db.outputs.execOut = og.ExecutionAttributeState.ENABLED
        return True
```

After this step, the randomizers will be available as nodes in the graph editor. For this tutorial the nodes are already added to the built-in `isaacsim.replicator.examples` extension and are available by default. Other custom nodes created through the OmniGraph tutorial will be accessible through the `omni.new.extension` extension (if the default tutorial-provided extension name was used). An example of accessing the nodes in an action graph is depicted below:

Note

If the custom nodes are not available, the newly created extension needs to be enabled. This can be done by navigating to **Window > Extensions > THIRD PARTY > ``omni.new.extension`` > ENABLED**:

After the OmniGraph randomization nodes are created, they can be manually added to a pre-existing SDG pipeline graph. To create a basic SDG graph, the following snippet can be used in the Script Editor to randomize the rotations of the created cubes every frame.

Basic SDG Pipeline

```python
import omni.replicator.core as rep

cube = rep.create.cube(count=50, scale=0.1)
with rep.trigger.on_frame():
    with cube:
        rep.randomizer.rotation()
```

After the snippet is executed in the Script Editor, the generated graph can be opened at `/Replicator/SDGPipeline` and the custom nodes can be added to the graph. The following image shows the result after the custom nodes are added to the SDG pipeline graph together with the resulting randomization (from the UI using `Tools` > `Replicator` > `Preview` or `Step`):

To avoid manually adding the custom nodes to the SDG pipeline graph, the Replicator API can be used to automatically insert the nodes into the graph. For this purpose, the nodes need to be encapsulated as **ReplicatorItems** using the `@ReplicatorWrapper` decorator. The following code snippet demonstrates how **ReplicatorItems** can be created for the custom nodes:

ReplicatorWrapper

```python
import omni.replicator.core as rep
from omni.replicator.core.scripts.utils import (
    ReplicatorItem,
    ReplicatorWrapper,
    create_node,
    set_target_prims,
)

@ReplicatorWrapper
def on_sphere(
    radius: float = 1.0,
    input_prims: ReplicatorItem | list[str] | None = None,
) -> ReplicatorItem:

    node = create_node("isaacsim.replicator.examples.OgnSampleOnSphere", radius=radius)
    if input_prims:
        set_target_prims(node, "inputs:prims", input_prims)
    return node

@ReplicatorWrapper
def in_sphere(
    radius: float = 1.0,
    input_prims: ReplicatorItem | list[str] | None = None,
) -> ReplicatorItem:

    node = create_node("isaacsim.replicator.examples.OgnSampleInSphere", radius=radius)
    if input_prims:
        set_target_prims(node, "inputs:prims", input_prims)
    return node

@ReplicatorWrapper
def between_spheres(
    radius1: float = 0.5,
    radius2: float = 1.0,
    input_prims: ReplicatorItem | list[str] | None = None,
) -> ReplicatorItem:

    node = create_node("isaacsim.replicator.examples.OgnSampleBetweenSpheres", radius1=radius1, radius2=radius2)
    if input_prims:
        set_target_prims(node, "inputs:prims", input_prims)
    return node

prim_count = 50
prim_scale = 0.1
rad_in = 0.5
rad_on = 1.5
rad_bet1 = 2.5
rad_bet2 = 3.5

# Create the default prims
sphere = rep.create.sphere(count=prim_count, scale=prim_scale)
cube = rep.create.cube(count=prim_count, scale=prim_scale)
cylinder = rep.create.cylinder(count=prim_count, scale=prim_scale)

# Create the randomization graph
with rep.trigger.on_frame():
    with sphere:
        rep.randomizer.rotation()
        in_sphere(rad_in)

    with cube:
        rep.randomizer.rotation()
        on_sphere(rad_on)

    with cylinder:
        rep.randomizer.rotation()
        between_spheres(rad_bet1, rad_bet2)
```

Note

For this tutorial the `create_node` function uses `"isaacsim.replicator.examples.OgnSampleInSphere"` as the node path, this path needs to be replaced in case the custom nodes are not part of the built-in `isaacsim.replicator.examples` extension.

After the snippet is executed in the Script Editor, the custom nodes will be automatically added to the SDG pipeline graph. To trigger the randomization, `Tools` > `Replicator` > `Preview` (or `Step`) can be called from the UI. The following image shows the generated graph and the resulting randomization:

---

# Modular Behavior Scripting

## Overview

This tutorial introduces the `isaacsim.replicator.behavior` extension, providing multiple examples of modular behavior scripts in Isaac Sim Replicator for synthetic data generation (SDG). By utilizing [Behavior Scripts (Python Scripting Component)](https://docs.omniverse.nvidia.com/extensions/latest/ext_python-scripting-component/user_manual.html), reusable, shareable, and easily modifiable behaviors can be developed and attached to prims in a USD stage, acting as randomizers or custom smart-asset behaviors.

The behavior script examples can be found under:

`/exts/isaacsim.replicator.behavior/isaacsim/replicator/behavior/behaviors/*`

### Learning Objectives

After completing this tutorial, you will understand how to:

* **Use pre-built behavior scripts** for common synthetic data generation tasks, including:

  + **Location Randomizer** - randomizes prim positions within specified bounds for object placement variety
  + **Rotation Randomizer** - applies random rotations to enhance orientation diversity in datasets
  + **Look At Behavior** - makes prims continuously face target locations or other prims for camera tracking
  + **Light Randomizer** - randomizes light properties like color and intensity to simulate different lighting conditions
  + **Texture Randomizer** - applies random textures to materials for increased visual variety
  + **Volume Stack Randomizer** - uses physics simulation to randomly stack objects for realistic arrangements
* **Understand behavior script architecture** - how modular Python scripts attach to prims and can be customized through exposed USD attributes, with configurable parameters like update intervals and randomization ranges
* **Control behavior execution** - configure behaviors to run on timeline events (start, update, stop) or trigger them independently using custom events for advanced workflows
* **Create custom behavior scripts** - develop your own behaviors using the provided templates and base classes for specific synthetic data generation needs
* **Build complex SDG pipelines** - combine multiple behaviors, simulations, and events to create sophisticated data generation workflows, such as physics-based object stacking followed by automated data capture

### Prerequisites

It is recommended that you have a basic understanding of the following concepts before proceeding with the tutorial:

* USD and Isaac Sim APIs for creating and manipulating USD stages
* [Python Scripting Component](https://docs.omniverse.nvidia.com/extensions/latest/ext_python-scripting-component/user_manual.html) in Isaac Sim
* The [timeline](https://docs.omniverse.nvidia.com/extensions/latest/ext_animation-timeline.html "(in Omniverse Extensions)") and [custom events](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/events.html) system
* [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") and its Isaac Sim [tutorials](Synthetic_Data_Generation.md) for synthetic data generation
* [Writers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/writer_examples.html "(in Omniverse Extensions)") and [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html "(in Omniverse Extensions)") for data capture
* Running scripts using the [Script Editor](https://docs.omniverse.nvidia.com/extensions/latest/ext_script-editor.html "(in Omniverse Extensions)") to setup and run pipelines

### Demonstration

The [example section](#isaac-sim-app-tutorial-replicator-modular-scripting-example) provides a demonstration of how to use the behavior scripts to create a custom synthetic data generation pipeline:

### Behavior Scripts

**Behavior Scripts** are modular Python scripts attached to prims in a USD stage. By default, they include template code that responds to timeline events such as start, pause, stop, and update. These scripts define specific behaviors or randomizations applied to prims during simulation or data generation.

Attaching scripts directly to prims integrates the behaviors into the USD, making them modular because scripts can be easily attached, detached, or swapped on prims without altering core logic. They are sharable because behaviors can be embedded within assets and shared across different projects or stages.

They are configurable because variables can be exposed through USD attributes for customization without modifying the script code. Additionally, they are persistent; because scripts reside on the prims, they persist with the USD stage and can be versioned and managed accordingly.

The advantages of behavior scripts include reusability, allowing them to be written once and reused across multiple prims or projects. They offer encapsulation by containing behavior logic within the prims, reducing external dependencies. They provide interactivity because parameters can be adjusted through the UI, enabling modifications without programming. Finally, they ensure integration by becoming an integral part of the asset, which maintains consistency across different environments.

### Exposing Variables Through USD Attributes

To enhance flexibility and accessibility, the input parameters in the provided behavior scripts examples can be exposed as USD attributes on prims. This approach allows you to modify behavior parameters directly from the UI without altering the script code.

The benefits of exposing variables include customization, interactivity, and consistency. Parameters such as target locations, ranges, or other settings can be adjusted per prim instance, using the UI to tweak behaviors and observe immediate effects, while maintaining a uniform interface for modifying behaviors across different scripts.

The exposed variables are implemented using the USD API to create custom attributes with appropriate namespaces on the prim. These attributes are then read by the behavior scripts during execution to adjust their logic accordingly.

The UI implementation for exposing the variables is done in `isaacsim.replicator.behavior.ui`. It extends the **Property** panel of the selected prims in the stage with a custom section for the exposed variables. The UI is automatically generated based on the exposed variables defined in the behavior script, displaying them as editable fields in the generated widget.

**Example of Exposed Variables Definition:**

```python
VARIABLES_TO_EXPOSE = [
    {
        "attr_name": "targetLocation",
        "attr_type": Sdf.ValueTypeNames.Vector3d,
        "default_value": Gf.Vec3d(0.0, 0.0, 0.0),
        "doc": "The 3D vector specifying the location to look at.",
    },
    {
        "attr_name": "targetPrimPath",
        "attr_type": Sdf.ValueTypeNames.String,
        "default_value": "",
        "doc": "The path of the target prim to look at. If specified, it has priority over the target location.",
    },
    # Additional variables...
]
```

### Custom Event-Based Behavior Scripts

While behavior scripts are timeline-based by default, some behaviors need to operate independently of the simulation timeline. **Event-based scripting** allows behaviors to be triggered by [custom events](https://docs.omniverse.nvidia.com/kit/docs/kit-manual/latest/guide/events.html), providing greater control over when and how they execute. This is achieved by skipping the default behavior functions and instead listening to and publishing custom events.

Custom events are defined and managed within Omniverse using an event bus system, enabling scripts to publish or subscribe to these events and facilitating communication between different components or behaviors.

Event-based scripting offers flexibility by allowing customization of when behaviors are executed, independent of the simulation timeline. It enhances modularity by decoupling behaviors from the core simulation loop, making them more modular. Additionally, it improves scalability by managing complex workflows through orchestrating multiple behaviors via events.

For example, the volume\_stack\_randomizer.py script randomizes the stacking of objects by simulating physics before the simulation starts. By using custom events, behaviors can be triggered before the simulation, execution flow can be controlled by starting, stopping, or resetting behaviors based on specific events rather than timeline updates, and performance can be enhanced by avoiding unnecessary computations during each simulation frame through decoupling certain behaviors.

## Script Examples

In this section, various behavior scripts available in the `isaacsim.replicator.behavior` extension are explored. Each script provides specific functionality that can enhance synthetic data generation workflows. The scripts are designed to be modular, reusable, and customizable through exposed variables.

The folder path for the behavior scripts is:

`/exts/isaacsim.replicator.behavior/isaacsim/replicator/behavior/behaviors/*`

### Location Randomizer

The `location_randomizer.py` script randomizes the location of prims within specified bounds during runtime, providing position variability for enhanced synthetic datasets.

Overview

**Purpose:** Randomizes prim positions within defined bounds to create variety in object placement.

**Key Features:**

* Position range randomization within minimum and maximum bounds
* Relative positioning support using target prims as reference points
* Child prim inclusion for hierarchical randomization
* Configurable update intervals for performance control

**Exposed Variables:**

Configuration Parameters

* **range:minPosition** (Vector3d): Minimum position bounds for randomization
* **range:maxPosition** (Vector3d): Maximum position bounds for randomization
* **frame:useRelativeFrame** (Bool): Enable relative positioning mode
* **frame:targetPrimPath** (String): Reference prim path for relative positioning
* **includeChildren** (Bool): Include child prims in randomization
* **interval** (UInt): Update frequency (0 = every frame)

Implementation Details

**Child Prim Inclusion:**

Child Prim Selection Logic

```python
def _setup(self):
    include_children = self._get_exposed_variable("includeChildren")
    if include_children:
        self._valid_prims = [prim for prim in Usd.PrimRange(self.prim) if prim.IsA(UsdGeom.Xformable)]
    elif self.prim.IsA(UsdGeom.Xformable):
        self._valid_prims = [self.prim]
    else:
        self._valid_prims = []
        carb.log_warn(f"[{self.prim_path}] No valid prims found.")
```

* When **includeChildren** is True: Uses Usd.PrimRange to select all transformable descendant prims
* When **includeChildren** is False: Only includes the assigned prim if it’s transformable
* Logs warning if no valid prims are found

**Randomization Logic:**

Core Randomization Implementation

```python
def _randomize_location(self, prim):
    # Generate random offset within bounds
    random_offset = Gf.Vec3d(
        random.uniform(self._min_position[0], self._max_position[0]),
        random.uniform(self._min_position[1], self._max_position[1]),
        random.uniform(self._min_position[2], self._max_position[2]),
    )

    # Calculate final location based on target prim and relative frame settings
    if self._target_prim:
        target_loc = get_world_location(self._target_prim)
        loc = (
            target_loc + self._target_offsets[prim] + random_offset
            if self._use_relative_frame
            else target_loc + random_offset
        )
    else:
        loc = self._initial_locations[prim] + random_offset if self._use_relative_frame else random_offset

    self._set_location(prim, loc)
```

* Generates random offset within specified bounds
* Handles target prim relative positioning
* Applies relative frame calculations when enabled
* Updates prim location using internal API

Usage Example

**Basic Setup:**

Step-by-Step Configuration

1. **Attach Script**: Add location\_randomizer.py to your target prim
2. **Set Bounds**: Configure range:minPosition and range:maxPosition
3. **Enable Children**: Set includeChildren to True for hierarchical randomization
4. **Set Interval**: Use interval to control update frequency

**Example Configuration:**

* **range:minPosition**: (-5.0, -5.0, 0.0)
* **range:maxPosition**: (5.0, 5.0, 2.0)
* **includeChildren**: True
* **interval**: 5 (updates every 5 frames)

**Use Cases:**

* **Background Objects**: Randomize prop positions for scene variety
* **Relative Positioning**: Move objects relative to a moving target
* **Hierarchical Randomization**: Apply randomization to object groups

### Rotation Randomizer

The `rotation_randomizer_1.py` script applies random rotations to prims during runtime, enhancing orientation diversity in synthetic datasets.

Overview

**Purpose:** Applies random rotations to prims within specified Euler angle bounds.

**Key Features:**

* Rotation range randomization within minimum and maximum angle bounds
* Child prim inclusion for hierarchical rotation randomization
* Configurable update intervals for performance optimization

**Exposed Variables:**

Configuration Parameters

* **range:minRotation** (Vector3d): Minimum rotation angles in degrees (X, Y, Z)
* **range:maxRotation** (Vector3d): Maximum rotation angles in degrees (X, Y, Z)
* **includeChildren** (Bool): Include child prims in rotation randomization
* **interval** (UInt): Update frequency (0 = every frame)

Implementation Details

**Child Prim Selection:**

Child Prim Selection Logic

```python
def _setup(self):
    include_children = self._get_exposed_variable("includeChildren")
    if include_children:
        self._valid_prims = [prim for prim in Usd.PrimRange(self.prim) if prim.IsA(UsdGeom.Xformable)]
    elif self.prim.IsA(UsdGeom.Xformable):
        self._valid_prims = [self.prim]
    else:
        self._valid_prims = []
        carb.log_warn(f"[{self.prim_path}] No valid prims found.")
```

* When **includeChildren** is True: All transformable descendant prims are included
* When **includeChildren** is False: Only the assigned prim is considered if transformable
* Warning logged if no valid prims found

**Rotation Randomization:**

Core Rotation Implementation

```python
def _randomize_rotation(self, prim):
    rotation = (
        Gf.Rotation(Gf.Vec3d.XAxis(), random.uniform(self._min_rotation[0], self._max_rotation[0]))
        * Gf.Rotation(Gf.Vec3d.YAxis(), random.uniform(self._min_rotation[1], self._max_rotation[1]))
        * Gf.Rotation(Gf.Vec3d.ZAxis(), random.uniform(self._min_rotation[2], self._max_rotation[2]))
    )
    set_rotation_with_ops(prim, rotation)
```

* Generates random Euler angles within specified bounds for each axis
* Creates composite rotation by multiplying X, Y, and Z axis rotations
* Applies rotation using set\_rotation\_with\_ops for proper transformation handling

Usage Example

**Basic Setup:**

Step-by-Step Configuration

1. **Attach Script**: Add rotation\_randomizer\_1.py to your target prim
2. **Set Rotation Bounds**: Configure range:minRotation and range:maxRotation
3. **Enable Children**: Set includeChildren to True for hierarchical rotation
4. **Set Interval**: Use interval to control update frequency

**Example Configuration:**

* **range:minRotation**: (-180.0, -90.0, 0.0) degrees
* **range:maxRotation**: (180.0, 90.0, 360.0) degrees
* **includeChildren**: True
* **interval**: 10 (updates every 10 frames)

**Use Cases:**

* **Object Variety**: Randomize prop orientations for diverse scenes
* **Tumbling Effects**: Simulate falling or floating objects
* **Presentation Angles**: Vary object viewing angles for training data

### Look At Behavior

The `look_at_behavior.py` script orients prims to continuously face a specified target, ideal for camera tracking and sensor alignment.

Overview

**Purpose:** Orients prims to continuously face a target location or another prim.

**Key Features:**

* Target specification using fixed coordinates or dynamic prim tracking
* Up axis control for maintaining consistent orientation
* Child prim inclusion for hierarchical look-at behavior
* Configurable update intervals for performance control

**Exposed Variables:**

Configuration Parameters

* **targetLocation** (Vector3d): Fixed 3D coordinates to look at
* **targetPrimPath** (String): Path to target prim (overrides targetLocation)
* **upAxis** (Vector3d): Up axis for orientation (e.g., (0, 0, 1) for +Z)
* **includeChildren** (Bool): Include child prims in look-at behavior
* **interval** (UInt): Update frequency (0 = every frame)

Implementation Details

**Target Prim Handling:**

Target Prim Resolution

```python
def _setup(self):
    target_prim_path = self._get_exposed_variable("targetPrimPath")
    if target_prim_path:
        self._target_prim = self.stage.GetPrimAtPath(target_prim_path)
        if not self._target_prim or not self._target_prim.IsValid() or not self._target_prim.IsA(UsdGeom.Xformable):
            self._target_prim = None
            carb.log_warn(f"[{self.prim_path}] Invalid target prim path: {target_prim_path}")
```

* **targetPrimPath** takes precedence over **targetLocation** when specified
* Validates target prim exists and is transformable
* Logs warning if target prim is invalid

**Orientation Calculation:**

Look-At Rotation Implementation

```python
def _apply_behavior(self):
    target_location = self._get_target_location()
    for prim in self._valid_prims:
        eye = get_world_location(prim)
        if (target_location - eye).GetLength() < 1e-9:
            continue  # Already at target; skip rotation to avoid undefined look-at
        look_at_rotation = calculate_look_at_rotation(eye, target_location, self._up_axis)
        set_rotation_with_ops(prim, look_at_rotation)
```

* Retrieves current prim position using get\_world\_location
* Calculates required rotation using calculate\_look\_at\_rotation
* Applies rotation while preserving existing transformation operations

Usage Example

**Basic Setup:**

Step-by-Step Configuration

1. **Attach Script**: Add look\_at\_behavior.py to your camera or sensor prim
2. **Set Target**: Configure either targetLocation or targetPrimPath
3. **Adjust Up Axis**: Set upAxis to maintain desired orientation
4. **Set Interval**: Use interval to control update frequency

**Example Configuration:**

* **targetPrimPath**: /World/MovingObject/Prim
* **upAxis**: (0, 0, 1) (Z-up orientation)
* **includeChildren**: False (camera only)
* **interval**: 1 (update every frame)

**Use Cases:**

* **Camera Tracking**: Make cameras follow moving subjects
* **Sensor Alignment**: Point sensors at targets of interest
* **Lighting Direction**: Orient lights to follow objects

### Light Randomizer

The `light_randomizer.py` script randomizes light properties to simulate different lighting conditions for enhanced scene variability.

Overview

**Purpose:** Randomizes light color and intensity properties to create diverse lighting scenarios.

**Key Features:**

* Color randomization varying RGB values within specified ranges
* Intensity randomization adjusting brightness between minimum and maximum values
* Child light inclusion for hierarchical lighting randomization
* Configurable update intervals for performance optimization

**Exposed Variables:**

Configuration Parameters

* **includeChildren** (Bool): Include child light prims in randomization
* **interval** (UInt): Update frequency (0 = every frame)
* **range:minColor** (Color3f): Minimum RGB values for color randomization
* **range:maxColor** (Color3f): Maximum RGB values for color randomization
* **range:intensity** (Float2): Intensity range as (min, max) values

Implementation Details

**Light Property Randomization:**

Color and Intensity Randomization

```python
def _apply_behavior(self):
    for prim in self._valid_prims:
        rand_color = (
            random.uniform(self._min_color[0], self._max_color[0]),
            random.uniform(self._min_color[1], self._max_color[1]),
            random.uniform(self._min_color[2], self._max_color[2]),
        )
        prim.GetAttribute("inputs:color").Set(rand_color)

        rand_intensity = random.uniform(self._intensity_range[0], self._intensity_range[1])
        prim.GetAttribute("inputs:intensity").Set(rand_intensity)
```

* Generates random RGB values within specified color ranges
* Applies random intensity values within defined bounds
* Updates light attributes directly using USD API

**Child Light Selection:**

Light Prim Discovery

```python
def _setup(self):
    include_children = self._get_exposed_variable("includeChildren")
    if include_children:
        self._valid_prims = [prim for prim in Usd.PrimRange(self.prim) if prim.HasAPI(UsdLux.LightAPI)]
    elif self.prim.HasAPI(UsdLux.LightAPI):
        self._valid_prims = [self.prim]
    else:
        self._valid_prims = []
        carb.log_warn(f"[{self.prim_path}] No valid light prims found.")
```

* Uses UsdLux.LightAPI to identify valid light prims
* Includes child lights when **includeChildren** is enabled
* Validates that target prim or children have light API

Usage Example

**Basic Setup:**

Step-by-Step Configuration

1. **Attach Script**: Add light\_randomizer.py to a light prim or parent containing lights
2. **Set Color Range**: Configure range:minColor and range:maxColor
3. **Set Intensity Range**: Define range:intensity min/max values
4. **Enable Children**: Set includeChildren to True for multiple lights

**Example Configuration:**

* **range:minColor**: (0.8, 0.8, 0.8) (warm white minimum)
* **range:maxColor**: (1.0, 1.0, 1.0) (bright white maximum)
* **range:intensity**: (1000.0, 5000.0) (intensity range)
* **includeChildren**: True
* **interval**: 0 (update every frame)

**Use Cases:**

* **Day/Night Cycles**: Simulate changing lighting conditions
* **Dynamic Environments**: Create flickering or varying light sources
* **Color Temperature**: Randomize between warm and cool lighting

### Texture Randomizer

The `texture_randomizer.py` script randomly applies textures to materials for increased visual variety of objects.

Overview

**Purpose:** Randomly applies textures to visual prims to create diverse material appearances.

**Key Features:**

* Texture selection from provided asset arrays or CSV lists
* Material creation with randomized parameters (scale, rotation, UV projection)
* Child prim inclusion for hierarchical texture randomization
* Configurable update intervals for performance control

**Exposed Variables:**

Configuration Parameters

* **includeChildren** (Bool): Include child prims in texture randomization
* **interval** (UInt): Update frequency (0 = every frame)
* **textures:assets** (AssetArray): List of texture assets to use
* **textures:csv** (String): CSV string of texture URLs
* **projectUvwProbability** (Float): Probability of enabling project\_uvw
* **textureScaleRange** (Float2): Texture scale range as (min, max)
* **textureRotateRange** (Float2): Texture rotation range in degrees (min, max)

Implementation Details

**Texture Application:**

Material and Shader Randomization

```python
def _apply_behavior(self):
    if not self._texture_urls:
        carb.log_warn(f"[{self.prim_path}] No texture URLs provided; skipping.")
        return
    for mat in self._texture_materials:
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(mat.GetPrim(), get_prim=True))
        if not shader:
            continue
        diffuse_texture = random.choice(self._texture_urls)
        if shader.GetInput("diffuse_texture"):
            shader.GetInput("diffuse_texture").Set(diffuse_texture)

        project_uvw = random.choices(
            [True, False], weights=[self._project_uvw_probability, 1 - self._project_uvw_probability]
        )[0]
        shader.GetInput("project_uvw").Set(bool(project_uvw))

        texture_scale = random.uniform(self._texture_scale_range[0], self._texture_scale_range[1])
        shader.GetInput("texture_scale").Set((texture_scale, texture_scale))

        texture_rotate = random.uniform(self._texture_rotate_range[0], self._texture_rotate_range[1])
        shader.GetInput("texture_rotate").Set(texture_rotate)
```

* Randomly selects textures from provided asset list
* Applies probabilistic UV projection settings
* Randomizes texture scale and rotation parameters
* Updates shader inputs directly via USD API

**Child Prim Selection:**

Geometric Prim Discovery

```python
def _setup(self):
    include_children = self._get_exposed_variable("includeChildren")
    if include_children:
        self._valid_prims = [prim for prim in Usd.PrimRange(self.prim) if prim.IsA(UsdGeom.Gprim)]
    elif self.prim.IsA(UsdGeom.Gprim):
        self._valid_prims = [self.prim]
    else:
        self._valid_prims = []
        carb.log_warn(f"[{self.prim_path}] No valid prims found.")
```

* Uses UsdGeom.Gprim to identify geometric prims suitable for materials
* Includes child prims when **includeChildren** is enabled
* Validates that target prims can receive material bindings

Usage Example

**Basic Setup:**

Step-by-Step Configuration

1. **Attach Script**: Add texture\_randomizer.py to a geometric prim
2. **Provide Textures**: Set textures:assets or textures:csv with texture paths
3. **Configure Parameters**: Adjust scale, rotation, and UV projection settings
4. **Enable Children**: Set includeChildren to True for multiple objects

**Example Configuration:**

* **textures:csv**: “texture1.jpg,texture2.png,texture3.exr”
* **textureScaleRange**: (0.5, 2.0) (scale variation)
* **textureRotateRange**: (0.0, 360.0) (full rotation)
* **projectUvwProbability**: 0.3 (30% chance of UV projection)
* **includeChildren**: True

**Use Cases:**

* **Material Variety**: Create diverse surface appearances for objects
* **Background Variation**: Randomize textures on environmental elements
* **Asset Augmentation**: Enhance object datasets with texture variation

### Volume Stack Randomizer

The `volume_stack_randomizer.py` script uses physics simulation to randomly stack objects for realistic object arrangements.

Overview

**Purpose:** Randomly drops and stacks assets within specified areas using physics simulation.

**Key Features:**

* Asset randomization from provided lists or CSV paths
* Physics simulation for natural stacking behavior
* Event-based execution independent of simulation timeline
* Customizable parameters for drop height, asset count, and rendering

**Exposed Variables:**

Configuration Parameters

* **includeChildren** (Bool): Include child prims in the behavior
* **event:input** (String): Event name to subscribe to for behavior control
* **event:output** (String): Event name to publish after behavior execution
* **assets:assets** (AssetArray): List of asset references to spawn
* **assets:csv** (String): CSV string of asset URLs to spawn
* **assets:numRange** (Int2): Range for number of assets to spawn (min, max)
* **dropHeight** (Float): Height from which to drop the assets
* **renderSimulation** (Bool): Whether to render simulation steps
* **removeRigidBodyDynamics** (Bool): Remove rigid body dynamics after simulation
* **preserveSimulationState** (Bool): Keep final simulation state

Implementation Details

**Core Structure:**

Class Architecture

```python
class VolumeStackRandomizer(BehaviorScript):
    BEHAVIOR_NS = "volumeStackRandomizer"
    EVENT_NAME_IN = f"{EXTENSION_NAME}.{BEHAVIOR_NS}.in"
    EVENT_NAME_OUT = f"{EXTENSION_NAME}.{BEHAVIOR_NS}.out"
    ACTION_FUNCTION_MAP = {
        "setup": "_setup_async",
        "run": "_run_behavior_async",
        "reset": "_reset_async",
    }

    async def _setup_async(self):
        # Asynchronous setup logic...
        pass

    async def _run_behavior_async(self):
        # Asynchronous behavior execution...
        pass

    async def _reset_async(self):
        # Asynchronous reset logic...
        pass
```

* Event-based behavior using custom events for lifecycle management
* Asynchronous methods for non-blocking physics simulation
* Action function mapping for external event control

**Child Prim Selection:**

Surface Area Discovery

```python
async def _setup_async(self):
    include_children = self._get_exposed_variable("includeChildren")
    if include_children:
        self._valid_prims = [prim for prim in Usd.PrimRange(self.prim) if prim.IsA(UsdGeom.Gprim)]
    elif self.prim.IsA(UsdGeom.Gprim):
        self._valid_prims = [self.prim]
    else:
        self._valid_prims = []
        carb.log_warn(f"[{self.prim_path}] No valid prims found.")
```

* Identifies geometric prims suitable for object stacking surfaces
* Includes child prims when **includeChildren** is enabled
* Validates surface prims can receive physics objects

Event-Based Control

**Custom Event System:**

Event-Based Execution Control

The Volume Stack Randomizer operates using custom events rather than timeline-based updates, allowing for precise control over when stacking operations occur.

**Event Flow:**

1. **Reset Phase**: Cleans up previous simulation state
2. **Setup Phase**: Spawns assets and prepares physics simulation
3. **Run Phase**: Executes physics simulation for object stacking
4. **Completion**: Publishes completion event with final state

**Event Control Example:**

```python
async def run_stacking_simulation_async(prim_path=None):
    actions = [("reset", "RESET", 10), ("setup", "SETUP", 500), ("run", "FINISHED", 1500)]
    for action, state, wait in actions:
        await publish_event_and_wait_for_completion_async(
            publish_payload={"prim_path": prim_path, "action": action},
            expected_payload={"prim_path": prim_path, "state_name": state},
            publish_event_name=VolumeStackRandomizer.EVENT_NAME_IN,
            subscribe_event_name=VolumeStackRandomizer.EVENT_NAME_OUT,
            max_wait_updates=wait,
        )
```

**Integration Benefits:**

* **Precise Control**: Execute stacking at specific workflow points
* **Sequential Operations**: Chain multiple stacking operations
* **State Management**: Track completion of each simulation phase
* **External Orchestration**: Control from external scripts or systems

Usage Example

**Basic Setup:**

Step-by-Step Configuration

1. **Attach Script**: Add volume\_stack\_randomizer.py to surface prims
2. **Configure Assets**: Set assets:csv or assets:assets with object paths
3. **Set Parameters**: Define assets:numRange, dropHeight, and other settings
4. **Control Events**: Use custom events to trigger stacking operations

**Example Configuration:**

* **assets:csv**: “box1.usd,box2.usd,cylinder.usd”
* **assets:numRange**: (5, 20) (spawn 5-20 objects)
* **dropHeight**: 2.0 (drop from 2 units above surface)
* **renderSimulation**: True (show simulation steps)
* **preserveSimulationState**: True (keep final arrangement)

**Use Cases:**

* **Object Arrangement**: Create realistic piles of objects
* **Physics Validation**: Test object interactions and stability
* **Scene Preparation**: Set up complex scenes before data capture
* **Simulation Workflows**: Integrate physics-based randomization into pipelines

### Templates

This section provides template scripts that serve as starting points for creating custom behaviors.

Available Templates

**Template Scripts:**

* **example\_behavior.py**: Basic template with boilerplate code for new behaviors
* **base\_behavior.py** and **example\_base\_behavior.py**: Demonstrate base behavior class inheritance for structured development
* **example\_custom\_event\_behavior.py**: Shows implementation of event-based behaviors

**Key Template Features:**

* **Variable Exposure**: Demonstrates exposing variables as USD attributes for UI customization
* **Behavior Structure**: Provides necessary methods (on\_init, on\_play, on\_update, on\_stop, on\_destroy) for timeline integration
* **Extensibility**: Base behavior classes enable easy extension and reuse in new behaviors
* **Event Integration**: Shows both timeline-based and custom event-based approaches

## Example

Below is an example demonstrating the use of behavior scripts to set up and run synthetic data generation in Isaac Sim. It showcases how to utilize behavior scripts for stacking simulations, texture randomization, light behavior, and camera tracking, ultimately capturing synthetic data with randomized scene configurations.

**Key Highlights of the Example:**

* **Volume Stacking Simulation**: Randomly stack assets using physics simulation to create realistic arrangements.
* **Texture Randomization**: Apply randomized textures to assets for scene diversity.
* **Light and Camera Behaviors**: Add randomization to light properties and make the camera track a specific target.
* **Synthetic Data Capture**: Generate and save synthetic images with the configured behaviors.

**Example Script:**

The demo script can be run directly from the [Script Editor](Development_Tools.md):

Behavior script-based SDG script:

```python
import asyncio
import inspect
import os

import isaacsim.core.experimental.utils.semantics as semantics_utils
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.replicator.behavior.behaviors import (
    LightRandomizer,
    LocationRandomizer,
    LookAtBehavior,
    RotationRandomizer,
    TextureRandomizer,
    VolumeStackRandomizer,
)
from isaacsim.replicator.behavior.global_variables import EXPOSED_ATTR_NS
from isaacsim.replicator.behavior.utils.behavior_utils import (
    add_behavior_script_with_parameters_async,
    publish_event_and_wait_for_completion_async,
)
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, UsdGeom

async def setup_and_run_stacking_simulation_async(prim, seed: int | None = None):
    STACK_ASSETS_CSV = (
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd,"
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd,"
        "/Isaac/Props/KLT_Bin/small_KLT_visual.usd,"
    )

    # Add the behavior script with custom parameters
    script_path = inspect.getfile(VolumeStackRandomizer)
    parameters = {
        f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:assets:csv": STACK_ASSETS_CSV,
        f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:assets:numRange": Gf.Vec2i(2, 15),
    }
    if seed is not None:
        parameters[f"{EXPOSED_ATTR_NS}:{VolumeStackRandomizer.BEHAVIOR_NS}:seed"] = seed
    await add_behavior_script_with_parameters_async(prim, script_path, parameters)

    # Helper function to handle publishing and waiting for events
    async def handle_event(action, expected_state, max_wait):
        return await publish_event_and_wait_for_completion_async(
            publish_payload={"prim_path": prim.GetPath(), "action": action},
            expected_payload={"prim_path": prim.GetPath(), "state_name": expected_state},
            publish_event_name=VolumeStackRandomizer.EVENT_NAME_IN,
            subscribe_event_name=VolumeStackRandomizer.EVENT_NAME_OUT,
            max_wait_updates=max_wait,
        )

    # Define and execute the stacking simulation steps
    actions = [("reset", "RESET", 10), ("setup", "SETUP", 500), ("run", "FINISHED", 1500)]
    for action, state, wait in actions:
        print(f"Executing '{action}' and waiting for state '{state}'...")
        if not await handle_event(action, state, wait):
            print(f"Failed to complete '{action}' with state '{state}'.")
            return

    print("Stacking simulation finished.")

async def setup_texture_randomizer_async(prim, seed: int | None = None):
    TEXTURE_ASSETS_CSV = (
        "/Isaac/Materials/Textures/Patterns/nv_bamboo_desktop.jpg,"
        "/Isaac/Materials/Textures/Patterns/nv_wood_boards_brown.jpg,"
        "/Isaac/Materials/Textures/Patterns/nv_wooden_wall.jpg,"
    )

    script_path = inspect.getfile(TextureRandomizer)
    parameters = {
        f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:interval": 5,
        f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:textures:csv": TEXTURE_ASSETS_CSV,
    }
    if seed is not None:
        parameters[f"{EXPOSED_ATTR_NS}:{TextureRandomizer.BEHAVIOR_NS}:seed"] = seed
    await add_behavior_script_with_parameters_async(prim, script_path, parameters)

async def setup_light_behaviors_async(prim, light_seed: int | None = None, location_seed: int | None = None):
    # Light randomization
    light_script_path = inspect.getfile(LightRandomizer)
    light_parameters = {
        f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:interval": 4,
        f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:range:intensity": Gf.Vec2f(20000, 120000),
    }
    if light_seed is not None:
        light_parameters[f"{EXPOSED_ATTR_NS}:{LightRandomizer.BEHAVIOR_NS}:seed"] = light_seed
    await add_behavior_script_with_parameters_async(prim, light_script_path, light_parameters)

    # Location randomization
    location_script_path = inspect.getfile(LocationRandomizer)
    location_parameters = {
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:interval": 2,
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:minPosition": Gf.Vec3d(-1.25, -1.25, 0.0),
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:maxPosition": Gf.Vec3d(1.25, 1.25, 0.0),
    }
    if location_seed is not None:
        location_parameters[f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:seed"] = location_seed
    await add_behavior_script_with_parameters_async(prim, location_script_path, location_parameters)

async def setup_target_asset_behaviors_async(prim, rotation_seed: int | None = None, location_seed: int | None = None):
    # Rotation randomization
    rotation_script_path = inspect.getfile(RotationRandomizer)
    rotation_parameters = {}
    if rotation_seed is not None:
        rotation_parameters[f"{EXPOSED_ATTR_NS}:{RotationRandomizer.BEHAVIOR_NS}:seed"] = rotation_seed
    await add_behavior_script_with_parameters_async(prim, rotation_script_path, rotation_parameters)

    # Location randomization
    location_script_path = inspect.getfile(LocationRandomizer)
    location_parameters = {
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:interval": 3,
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:minPosition": Gf.Vec3d(-0.2, -0.2, -0.2),
        f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:range:maxPosition": Gf.Vec3d(0.2, 0.2, 0.2),
    }
    if location_seed is not None:
        location_parameters[f"{EXPOSED_ATTR_NS}:{LocationRandomizer.BEHAVIOR_NS}:seed"] = location_seed
    await add_behavior_script_with_parameters_async(prim, location_script_path, location_parameters)

async def setup_camera_behaviors_async(prim, target_prim_path):
    # Look at behavior following the target asset
    script_path = inspect.getfile(LookAtBehavior)
    parameters = {
        f"{EXPOSED_ATTR_NS}:{LookAtBehavior.BEHAVIOR_NS}:targetPrimPath": target_prim_path,
    }
    await add_behavior_script_with_parameters_async(prim, script_path, parameters)

async def setup_writer_and_capture_data_async(camera_path, num_captures):
    # Create the writer and the render product
    rp = rep.create.render_product(camera_path, (512, 512))
    writer = rep.writers.get("BasicWriter")
    output_directory = os.path.join(os.getcwd(), "_out_behaviors_sdg")
    print(f"output_directory: {output_directory}")
    writer.initialize(output_dir=output_directory, rgb=True, distance_to_image_plane=True, colorize_depth=True)
    writer.attach(rp)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Use the timeline to control the behavior scripts execution and frame captures
    timeline = omni.timeline.get_timeline_interface()

    # Start the SDG pipeline
    for i in range(num_captures):
        # Advance the timeline with one update and then pause it to avoid triggering the behavior scripts by the step_async internal updates
        timeline.play()
        await omni.kit.app.get_app().next_update_async()
        timeline.pause()
        timeline.commit()

        # Capture the frame
        print(f"Capturing frame {i} at time {timeline.get_current_time():.4f}")
        await rep.orchestrator.step_async(rt_subframes=32, delta_time=0.0)

    # Stop the timeline (and trigger the behavior scripts to stop)
    timeline.stop()
    await omni.kit.app.get_app().next_update_async()

    # Make sure all the frames are written from the backend queue and free the rendering resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

async def run_example_async(num_captures, seed: int | None = None):
    STAGE_URL = "/Isaac/Samples/Replicator/Stage/warehouse_pallets_behavior_scripts.usd"
    PALLETS_ROOT_PATH = "/Root/Pallets"
    LIGHTS_ROOT_PATH = "/Root/Lights"
    CAMERA_PATH = "/Root/Camera_01"
    TARGET_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd"
    TARGET_ASSET_PATH = "/Root/Target"
    TARGET_ASSET_LABEL = "power_drill"
    TARGET_ASSET_LOCATION = (-1.5, 5.5, 1.5)

    # Generate unique seeds per behavior instance to ensure determinism regardless of execution order
    if seed is not None:
        seed_rng = np.random.default_rng(seed)
        stacking_seed = int(seed_rng.integers(0, 2**31))
        texture_seed = int(seed_rng.integers(0, 2**31))
        light_intensity_seed = int(seed_rng.integers(0, 2**31))
        light_location_seed = int(seed_rng.integers(0, 2**31))
        target_rotation_seed = int(seed_rng.integers(0, 2**31))
        target_location_seed = int(seed_rng.integers(0, 2**31))
    else:
        stacking_seed = texture_seed = None
        light_intensity_seed = light_location_seed = None
        target_rotation_seed = target_location_seed = None

    # Open stage
    assets_root_path = await get_assets_root_path_async()
    print(f"Opening stage from {assets_root_path + STAGE_URL}")
    await omni.usd.get_context().open_stage_async(assets_root_path + STAGE_URL)
    stage = omni.usd.get_context().get_stage()

    # Check if all required prims exist in the stage
    pallets_root_prim = stage.GetPrimAtPath(PALLETS_ROOT_PATH)
    lights_root_prim = stage.GetPrimAtPath(LIGHTS_ROOT_PATH)
    camera_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not all([pallets_root_prim.IsValid(), lights_root_prim.IsValid(), camera_prim.IsValid()]):
        print(f"Not all required prims exist in the stage.")
        return

    # Spawn the target asset at the requested location, label it with the target asset label
    target_prim = stage.DefinePrim(TARGET_ASSET_PATH, "Xform")
    target_prim.GetReferences().AddReference(assets_root_path + TARGET_ASSET_URL)
    if not target_prim.HasAttribute("xformOp:translate"):
        UsdGeom.Xformable(target_prim).AddTranslateOp()
    target_prim.GetAttribute("xformOp:translate").Set(TARGET_ASSET_LOCATION)
    semantics_utils.remove_all_labels(target_prim, include_descendants=True)
    semantics_utils.add_labels(target_prim, labels=[TARGET_ASSET_LABEL], taxonomy="class")

    # Setup and run the stacking simulation before capturing the data
    # Note: Physics simulation is non-deterministic, final positions may vary
    await setup_and_run_stacking_simulation_async(pallets_root_prim, seed=stacking_seed)

    # Setup texture randomizer
    await setup_texture_randomizer_async(pallets_root_prim, seed=texture_seed)

    # Setup the light behaviors
    await setup_light_behaviors_async(
        lights_root_prim, light_seed=light_intensity_seed, location_seed=light_location_seed
    )

    # Setup the target asset behaviors
    await setup_target_asset_behaviors_async(
        target_prim, rotation_seed=target_rotation_seed, location_seed=target_location_seed
    )

    # Setup the camera behaviors
    await setup_camera_behaviors_async(camera_prim, str(target_prim.GetPath()))

    # Setup the writer and capture the data, behavior scripts are triggered by running the timeline
    await setup_writer_and_capture_data_async(camera_path=camera_prim.GetPath(), num_captures=num_captures)

asyncio.ensure_future(run_example_async(num_captures=6))
```

---

# Randomization Snippets

Examples of randomization using USD and Isaac Sim APIs. These examples demonstrate how to randomize scenes for synthetic data generation (SDG) in scenarios where default [replicator randomizers](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/randomizer_details.html "(in Omniverse Extensions)") are not sufficient or applicable.

The snippets are designed to align with the structure and function names used in the replicator example snippets. In comparison they also have the option to write the data to disk by stetting `write_data=True`.

Prerequisites:

* Familiarity with [USD](https://developer.nvidia.com/usd/tutorials).
* Ability to execute code from the [Script Editor](Development_Tools.md).
* Understanding basic replicator concepts, such as [subframes](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)").

## Randomizing Light Sources

This snippet sets up a new environment containing a cube and a sphere.
It then spawns a given number of lights and randomizes selected attributes for these lights over a specified number of frames.

Randomizing Light Sources

```python
import asyncio
import os

import numpy as np
import omni.kit.commands
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels
from pxr import Gf, Sdf, UsdGeom

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()

sphere = stage.DefinePrim("/World/Sphere", "Sphere")
UsdGeom.Xformable(sphere).AddTranslateOp().Set((0.0, 1.0, 1.0))
add_labels(sphere, labels=["sphere"], instance_name="class")

cube = stage.DefinePrim("/World/Cube", "Cube")
UsdGeom.Xformable(cube).AddTranslateOp().Set((0.0, -2.0, 2.0))
add_labels(cube, labels=["cube"], instance_name="class")

plane_path = "/World/Plane"
omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_path=plane_path, prim_type="Plane")
plane_prim = stage.GetPrimAtPath(plane_path)
plane_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False).Set(Gf.Vec3d(10, 10, 1))

def sphere_lights(num):
    lights = []
    for i in range(num):
        # "CylinderLight", "DiskLight", "DistantLight", "DomeLight", "RectLight", "SphereLight"
        prim_type = "SphereLight"
        next_free_path = omni.usd.get_stage_next_free_path(stage, f"/World/{prim_type}", False)
        light_prim = stage.DefinePrim(next_free_path, prim_type)
        UsdGeom.Xformable(light_prim).AddTranslateOp().Set((0.0, 0.0, 0.0))
        UsdGeom.Xformable(light_prim).AddRotateXYZOp().Set((0.0, 0.0, 0.0))
        UsdGeom.Xformable(light_prim).AddScaleOp().Set((1.0, 1.0, 1.0))
        light_prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(True)
        light_prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(6500.0)
        light_prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(0.5)
        light_prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(30000.0)
        light_prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set((1.0, 1.0, 1.0))
        light_prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(0.0)
        light_prim.CreateAttribute("inputs:diffuse", Sdf.ValueTypeNames.Float).Set(1.0)
        light_prim.CreateAttribute("inputs:specular", Sdf.ValueTypeNames.Float).Set(1.0)
        lights.append(light_prim)
    return lights

async def run_randomizations_async(num_frames, lights, write_data, delay=None):
    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_rand_lights")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="Camera")
        rp = rep.create.render_product(cam, resolution=(512, 512))
        writer.attach(rp)

    for _ in range(num_frames):
        for light in lights:
            light.GetAttribute("xformOp:translate").Set(
                (np.random.uniform(-5, 5), np.random.uniform(-5, 5), np.random.uniform(4, 6))
            )
            scale_rand = np.random.uniform(0.5, 1.5)
            light.GetAttribute("xformOp:scale").Set((scale_rand, scale_rand, scale_rand))
            light.GetAttribute("inputs:colorTemperature").Set(np.random.normal(4500, 1500))
            light.GetAttribute("inputs:intensity").Set(np.random.normal(25000, 5000))
            light.GetAttribute("inputs:color").Set(
                (np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9))
            )

        if write_data:
            await rep.orchestrator.step_async(rt_subframes=16)
        else:
            await omni.kit.app.get_app().next_update_async()
        # Optional delay between frames to better visualize the randomization in the viewport
        if delay is not None and delay > 0:
            await asyncio.sleep(delay)

    # Wait for the data to be written to disk and cleanup writer and render product
    if write_data:
        await rep.orchestrator.wait_until_complete_async()
        writer.detach()
        rp.destroy()

num_frames = 10
lights = sphere_lights(10)
asyncio.ensure_future(run_randomizations_async(num_frames=num_frames, lights=lights, write_data=True, delay=0.2))
```

## Randomizing Textures

The snippet sets up an environment, spawns a given number of cubes and spheres, and randomizes their textures for the given number of frames. After the randomizations their original materials are reassigned. The snippet also showcases how to create a new material and assign it to a prim.

Randomizing Textures

```python
import asyncio
import os

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.utils.semantics import add_labels, get_labels
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, Sdf, UsdGeom, UsdShade

omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
dome_light = stage.DefinePrim("/World/DomeLight", "DomeLight")
dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(1000.0)

sphere = stage.DefinePrim("/World/Sphere", "Sphere")
UsdGeom.Xformable(sphere).AddTranslateOp().Set((0.0, 0.0, 1.0))
add_labels(sphere, labels=["sphere"], instance_name="class")

num_cubes = 10
for _ in range(num_cubes):
    prim_type = "Cube"
    next_free_path = omni.usd.get_stage_next_free_path(stage, f"/World/{prim_type}", False)
    cube = stage.DefinePrim(next_free_path, prim_type)
    UsdGeom.Xformable(cube).AddTranslateOp().Set((np.random.uniform(-3.5, 3.5), np.random.uniform(-3.5, 3.5), 1))
    scale_rand = np.random.uniform(0.25, 0.5)
    UsdGeom.Xformable(cube).AddScaleOp().Set((scale_rand, scale_rand, scale_rand))
    add_labels(cube, labels=["cube"], instance_name="class")

plane_path = "/World/Plane"
omni.kit.commands.execute("CreateMeshPrimWithDefaultXform", prim_path=plane_path, prim_type="Plane")
plane_prim = stage.GetPrimAtPath(plane_path)
plane_prim.CreateAttribute("xformOp:scale", Sdf.ValueTypeNames.Double3, False).Set(Gf.Vec3d(10, 10, 1))

def get_shapes():
    stage = omni.usd.get_context().get_stage()
    shapes = []
    for prim in stage.Traverse():
        labels = get_labels(prim)
        if class_labels := labels.get("class"):
            if "cube" in class_labels or "sphere" in class_labels:
                shapes.append(prim)
    return shapes

shapes = get_shapes()

def create_omnipbr_material(mtl_url, mtl_name, mtl_path):
    stage = omni.usd.get_context().get_stage()
    omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=mtl_url, mtl_name=mtl_name, mtl_path=mtl_path)
    material_prim = stage.GetPrimAtPath(mtl_path)
    shader = UsdShade.Shader(omni.usd.get_shader_from_material(material_prim, get_prim=True))

    # Add value inputs
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f)
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float)

    # Add texture inputs
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("metallic_texture", Sdf.ValueTypeNames.Asset)

    # Add other attributes
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool)

    # Add texture scale and rotate
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2)
    shader.CreateInput("texture_rotate", Sdf.ValueTypeNames.Float)

    material = UsdShade.Material(material_prim)
    return material

def create_materials(num):
    MDL = "OmniPBR.mdl"
    mtl_name, _ = os.path.splitext(MDL)
    MAT_PATH = "/World/Looks"
    materials = []
    for _ in range(num):
        prim_path = omni.usd.get_stage_next_free_path(stage, f"{MAT_PATH}/{mtl_name}", False)
        mat = create_omnipbr_material(mtl_url=MDL, mtl_name=mtl_name, mtl_path=prim_path)
        materials.append(mat)
    return materials

materials = create_materials(len(shapes))

async def run_randomizations_async(num_frames, materials, write_data, delay=None):
    assets_root_path = await get_assets_root_path_async()
    textures = [
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/aggregate_exposed_diff.jpg",
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_diff.jpg",
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/gravel_track_ballast_multi_R_rough_G_ao.jpg",
        assets_root_path + "/NVIDIA/Materials/vMaterials_2/Ground/textures/rough_gravel_rough.jpg",
    ]

    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_rand_textures")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="Camera")
        rp = rep.create.render_product(cam, resolution=(512, 512))
        writer.attach(rp)

    # Apply the new materials and store the initial ones to reassign later
    initial_materials = {}
    for i, shape in enumerate(shapes):
        cur_mat, _ = UsdShade.MaterialBindingAPI(shape).ComputeBoundMaterial()
        initial_materials[shape] = cur_mat
        UsdShade.MaterialBindingAPI(shape).Bind(materials[i], UsdShade.Tokens.strongerThanDescendants)

    for _ in range(num_frames):
        for mat in materials:
            shader = UsdShade.Shader(omni.usd.get_shader_from_material(mat, get_prim=True))
            diffuse_texture = np.random.choice(textures)
            shader.GetInput("diffuse_texture").Set(diffuse_texture)
            project_uvw = np.random.choice([True, False], p=[0.9, 0.1])
            shader.GetInput("project_uvw").Set(bool(project_uvw))
            texture_scale = np.random.uniform(0.1, 1)
            shader.GetInput("texture_scale").Set((texture_scale, texture_scale))
            texture_rotate = np.random.uniform(0, 45)
            shader.GetInput("texture_rotate").Set(texture_rotate)

        if write_data:
            await rep.orchestrator.step_async(rt_subframes=4)
        else:
            await omni.kit.app.get_app().next_update_async()

        # Optional delay between frames to better visualize the randomization in the viewport
        if delay is not None and delay > 0:
            await asyncio.sleep(delay)

    # Wait for the data to be written to disk and cleanup writer and render product
    if write_data:
        await rep.orchestrator.wait_until_complete_async()
        writer.detach()
        rp.destroy()

    # Reassign the initial materials
    for shape, mat in initial_materials.items():
        if mat:
            UsdShade.MaterialBindingAPI(shape).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
        else:
            UsdShade.MaterialBindingAPI(shape).UnbindAllBindings()

num_frames = 10
asyncio.ensure_future(run_randomizations_async(num_frames, materials, write_data=True, delay=0.2))
```

## Sequential Randomizations

The snippet provides an example of more complex randomizations, where the results of the first randomization are used to determine the next randomization. It uses a custom sampler function to set the location of the camera by iterating over (almost) equidistant points on a sphere. The snippet starts by setting up the environment, a forklift, a pallet, a bin, and a dome light. For every randomization frame, it cycles through the dome light textures, moves the pallet to a random location, and then moves the bin so that it is fully on top of the pallet. Finally, it moves the camera to a new location on the sphere, ensuring it faces the bin.

Sequential Randomizations

```python
import asyncio
import itertools
import os

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, Usd, UsdGeom, UsdLux

# Fibonacci sphere algorithm: https://arxiv.org/pdf/0912.4540
def next_point_on_sphere(idx, num_points, radius=1, origin=(0, 0, 0)):
    offset = 2.0 / num_points
    inc = np.pi * (3.0 - np.sqrt(5.0))
    z = ((idx * offset) - 1) + (offset / 2)
    phi = ((idx + 1) % num_points) * inc
    r = np.sqrt(1 - pow(z, 2))
    y = np.cos(phi) * r
    x = np.sin(phi) * r
    return [(x * radius) + origin[0], (y * radius) + origin[1], (z * radius) + origin[2]]

async def run_randomizations_async(
    num_frames, forklift_path, pallet_path, bin_path, dome_textures, write_data, delay=None
):
    assets_root_path = await get_assets_root_path_async()

    await omni.usd.get_context().new_stage_async()
    stage = omni.usd.get_context().get_stage()

    dome_light = UsdLux.DomeLight.Define(stage, "/World/Lights/DomeLight")
    dome_light.GetIntensityAttr().Set(1000)

    forklift_prim = stage.DefinePrim("/World/Forklift", "Xform")
    forklift_prim.GetReferences().AddReference(assets_root_path + forklift_path)
    if not forklift_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(forklift_prim).AddTranslateOp()
    forklift_prim.GetAttribute("xformOp:translate").Set((-4.5, -4.5, 0))

    pallet_prim = stage.DefinePrim("/World/Pallet", "Xform")
    pallet_prim.GetReferences().AddReference(assets_root_path + pallet_path)
    if not pallet_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(pallet_prim).AddTranslateOp()
    if not pallet_prim.GetAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(pallet_prim).AddRotateXYZOp()

    bin_prim = stage.DefinePrim("/World/Bin", "Xform")
    bin_prim.GetReferences().AddReference(assets_root_path + bin_path)
    if not bin_prim.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(bin_prim).AddTranslateOp()
    if not bin_prim.GetAttribute("xformOp:rotateXYZ"):
        UsdGeom.Xformable(bin_prim).AddRotateXYZOp()

    view_cam = stage.DefinePrim("/World/Camera", "Camera")
    if not view_cam.GetAttribute("xformOp:translate"):
        UsdGeom.Xformable(view_cam).AddTranslateOp()
    if not view_cam.GetAttribute("xformOp:orient"):
        UsdGeom.Xformable(view_cam).AddOrientOp()

    dome_textures_full = [assets_root_path + tex for tex in dome_textures]
    textures_cycle = itertools.cycle(dome_textures_full)

    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_rand_sphere_scan")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        persp_cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), name="PerspCamera")
        rp_persp = rep.create.render_product(persp_cam, (512, 512), name="PerspView")
        rp_view = rep.create.render_product(view_cam, (512, 512), name="SphereView")
        writer.attach([rp_view, rp_persp])

    bb_cache = UsdGeom.BBoxCache(time=Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    pallet_size = bb_cache.ComputeWorldBound(pallet_prim).GetRange().GetSize()
    pallet_length = pallet_size.GetLength()
    bin_size = bb_cache.ComputeWorldBound(bin_prim).GetRange().GetSize()

    for i in range(num_frames):
        # Set next background texture every nth frame and run an app update
        if i % 5 == 0:
            dome_light.GetTextureFileAttr().Set(next(textures_cycle))
            await omni.kit.app.get_app().next_update_async()

        # Randomize pallet pose
        pallet_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(np.random.uniform(-1.5, 1.5), np.random.uniform(-1.5, 1.5), 0)
        )
        rand_z_rot = np.random.uniform(-90, 90)
        pallet_prim.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3d(0, 0, rand_z_rot))
        pallet_tf_mat = omni.usd.get_world_transform_matrix(pallet_prim)
        pallet_rot = pallet_tf_mat.ExtractRotation()
        pallet_pos = pallet_tf_mat.ExtractTranslation()

        # Randomize bin position on top of the rotated pallet area making sure the bin is fully on the pallet
        rand_transl_x = np.random.uniform(-pallet_size[0] / 2 + bin_size[0] / 2, pallet_size[0] / 2 - bin_size[0] / 2)
        rand_transl_y = np.random.uniform(-pallet_size[1] / 2 + bin_size[1] / 2, pallet_size[1] / 2 - bin_size[1] / 2)

        # Adjust bin position to account for the random rotation of the pallet
        rand_z_rot_rad = np.deg2rad(rand_z_rot)
        rot_adjusted_transl_x = rand_transl_x * np.cos(rand_z_rot_rad) - rand_transl_y * np.sin(rand_z_rot_rad)
        rot_adjusted_transl_y = rand_transl_x * np.sin(rand_z_rot_rad) + rand_transl_y * np.cos(rand_z_rot_rad)
        bin_prim.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(
                pallet_pos[0] + rot_adjusted_transl_x,
                pallet_pos[1] + rot_adjusted_transl_y,
                pallet_pos[2] + pallet_size[2] + bin_size[2] / 2,
            )
        )
        # Keep bin rotation aligned with pallet
        bin_prim.GetAttribute("xformOp:rotateXYZ").Set(pallet_rot.GetAxis() * pallet_rot.GetAngle())

        # Get next camera position on a sphere looking at the bin with a randomized distance
        rand_radius = np.random.normal(3, 0.5) * pallet_length
        bin_pos = omni.usd.get_world_transform_matrix(bin_prim).ExtractTranslation()
        cam_pos = next_point_on_sphere(i, num_points=num_frames, radius=rand_radius, origin=bin_pos)
        view_cam.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*cam_pos))

        eye = Gf.Vec3d(*cam_pos)
        target = Gf.Vec3d(*bin_pos)
        up_axis = Gf.Vec3d(0, 0, 1)
        look_at_quatd = Gf.Matrix4d().SetLookAt(eye, target, up_axis).GetInverse().ExtractRotation().GetQuat()
        view_cam.GetAttribute("xformOp:orient").Set(Gf.Quatf(look_at_quatd))

        if write_data:
            await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0)
        else:
            await omni.kit.app.get_app().next_update_async()
        # Optional delay between frames to better visualize the randomization in the viewport
        if delay is not None and delay > 0:
            await asyncio.sleep(delay)

    # Wait for the data to be written to disk and cleanup writer and render products
    if write_data:
        await rep.orchestrator.wait_until_complete_async()
        writer.detach()
        rp_persp.destroy()
        rp_view.destroy()

NUM_FRAMES = 90
FORKLIFT_PATH = "/Isaac/Props/Forklift/forklift.usd"
PALLET_PATH = "/Isaac/Props/Pallet/pallet.usd"
BIN_PATH = "/Isaac/Props/KLT_Bin/small_KLT_visual.usd"
DOME_TEXTURES = [
    "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
    "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
    "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
    "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
]
asyncio.ensure_future(
    run_randomizations_async(
        NUM_FRAMES, FORKLIFT_PATH, PALLET_PATH, BIN_PATH, DOME_TEXTURES, write_data=True, delay=0.2
    )
)
```

## Physics-based Randomized Volume Filling

The snippet randomizes the stacking of objects on multiple surfaces. It randomly spawns a given number of pallets in the selected areas and then spawns physically simulated boxes on top of them. A temporary collision box area is created around the pallets to prevent the boxes from falling off. After all the boxes have been dropped, they are moved in various directions and finally pulled towards the center of the pallet for more stable stacking. Finally, the collision area is removed, after which the boxes can also fall to the ground. To allow easier sliding of the boxes into more stable positions, their friction is temporarily reduced during the simulation.

Physics-based Randomized Volume Filling

```python
import asyncio
import os
import random
from itertools import chain

import carb
import omni.kit.app
import omni.physics.core
import omni.replicator.core as rep
import omni.usd
from isaacsim.storage.native import get_assets_root_path_async
from pxr import Gf, PhysicsSchemaTools, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdUtils

# Add transformation properties to the prim (if not already present)
def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)

# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(prim):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")

# Enables rigid body dynamics (physics simulation) on the prim (having valid colliders is recommended)
def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    # Physics
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    else:
        rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    # PhysX
    if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    else:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
    physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    if angular_damping is not None:
        physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)

# Create a new prim with the provided asset URL and transform properties
def create_asset(stage, asset_url, path, location=None, rotation=None, orientation=None, scale=None):
    prim_path = omni.usd.get_stage_next_free_path(stage, path, False)
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(asset_url)
    set_transform_attributes(prim, location=location, rotation=rotation, orientation=orientation, scale=scale)
    return prim

# Create a new prim with the provided asset URL and transform properties including colliders
def create_asset_with_colliders(stage, asset_url, path, location=None, rotation=None, orientation=None, scale=None):
    prim = create_asset(stage, asset_url, path, location, rotation, orientation, scale)
    add_colliders(prim)
    return prim

# Create collision walls around the top surface of the prim with the given height and thickness
def create_collision_walls(stage, prim, bbox_cache=None, height=2, thickness=0.3, material=None, visible=False):
    # Use the untransformed axis-aligned bounding box to calculate the prim surface size and center
    if bbox_cache is None:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    local_range = bbox_cache.ComputeWorldBound(prim).GetRange()
    width, depth, local_height = local_range.GetSize()
    # Raise the midpoint height to the prim's surface
    mid = local_range.GetMidpoint() + Gf.Vec3d(0, 0, local_height / 2)

    # Define the walls (name, location, size) with the specified thickness added externally to the surface and height
    walls = [
        ("floor", (mid[0], mid[1], mid[2] - thickness / 2), (width, depth, thickness)),
        ("ceiling", (mid[0], mid[1], mid[2] + height + thickness / 2), (width, depth, thickness)),
        (
            "left_wall",
            (mid[0] - (width + thickness) / 2, mid[1], mid[2] + height / 2),
            (thickness, depth, height),
        ),
        (
            "right_wall",
            (mid[0] + (width + thickness) / 2, mid[1], mid[2] + height / 2),
            (thickness, depth, height),
        ),
        (
            "front_wall",
            (mid[0], mid[1] + (depth + thickness) / 2, mid[2] + height / 2),
            (width, thickness, height),
        ),
        (
            "back_wall",
            (mid[0], mid[1] - (depth + thickness) / 2, mid[2] + height / 2),
            (width, thickness, height),
        ),
    ]

    # Use the parent prim path to create the walls as children (use local coordinates)
    prim_path = prim.GetPath()
    collision_walls = []
    for name, location, size in walls:
        prim = stage.DefinePrim(f"{prim_path}/{name}", "Cube")
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        set_transform_attributes(prim, location=location, scale=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()
        if material is not None:
            mat_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            mat_binding_api.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
        collision_walls.append(prim)
    return collision_walls

# Slide the assets independently in perpendicular directions and then pull them all together towards the given center
async def apply_forces_async(stage, boxes, pallet, strength=550, strength_center_multiplier=2):
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    # Get the pallet center and forward vector to apply forces in the perpendicular directions and towards the center
    pallet_tf: Gf.Matrix4d = UsdGeom.Xformable(pallet).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pallet_center = pallet_tf.ExtractTranslation()
    pallet_rot: Gf.Rotation = pallet_tf.ExtractRotation()
    force_forward = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(1, 0, 0))) * strength
    force_right = Gf.Vec3d(pallet_rot.TransformDir(Gf.Vec3d(0, 1, 0))) * strength

    physics_simulation_interface = omni.physics.core.get_physics_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    for box_prim in boxes:
        body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
        forces = [force_forward, force_right, -force_forward, -force_right]
        for force in chain(forces, forces):
            box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            box_position = carb.Float3(*box_tf.ExtractTranslation())
            physics_simulation_interface.add_force_at_pos(
                stage_id, body_path, carb.Float3(force), box_position, omni.physics.core.ForceMode.FORCE
            )
            for _ in range(10):
                await omni.kit.app.get_app().next_update_async()

    # Pull all box at once to the pallet center
    for box_prim in boxes:
        body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
        box_tf: Gf.Matrix4d = UsdGeom.Xformable(box_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        box_location = box_tf.ExtractTranslation()
        force_to_center = (pallet_center - box_location) * strength * strength_center_multiplier
        physics_simulation_interface.add_force_at_pos(
            stage_id,
            body_path,
            carb.Float3(*force_to_center),
            carb.Float3(*box_location),
            omni.physics.core.ForceMode.FORCE,
        )
    for _ in range(20):
        await omni.kit.app.get_app().next_update_async()
    timeline.pause()

# Create a new stage and and run the example scenario
async def stack_boxes_on_pallet_async(pallet_prim, boxes_urls_and_weights, num_boxes, drop_height=1.5, drop_margin=0.2):
    pallet_path = pallet_prim.GetPath()
    print(f"[BoxStacking] Running scenario for pallet {pallet_path} with {num_boxes} boxes..")
    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

    # Create a custom physics material to allow the boxes to easily slide into stacking positions
    material_path = f"{pallet_path}/Looks/PhysicsMaterial"
    default_material = UsdShade.Material.Define(stage, material_path)
    physics_material = UsdPhysics.MaterialAPI.Apply(default_material.GetPrim())
    physics_material.CreateRestitutionAttr().Set(0.0)  # Inelastic collision (no bouncing)
    physics_material.CreateStaticFrictionAttr().Set(0.01)  # Small friction to allow sliding of stationary boxes
    physics_material.CreateDynamicFrictionAttr().Set(0.01)  # Small friction to allow sliding of moving boxes

    # Apply the physics material to the pallet
    mat_binding_api = UsdShade.MaterialBindingAPI.Apply(pallet_prim)
    mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")

    # Create collision walls around the top of the pallet and apply the physics material to them
    collision_walls = create_collision_walls(
        stage, pallet_prim, bbox_cache, height=drop_height + drop_margin, material=default_material
    )

    # Create the random boxes (without physics) with the specified weights and sort them by size (volume)
    box_urls, box_weights = zip(*boxes_urls_and_weights)
    rand_boxes_urls = random.choices(box_urls, weights=box_weights, k=num_boxes)
    boxes = [create_asset(stage, box_url, f"{pallet_path}_Boxes/Box_{i}") for i, box_url in enumerate(rand_boxes_urls)]
    boxes.sort(key=lambda box: bbox_cache.ComputeLocalBound(box).GetVolume(), reverse=True)

    # Calculate the drop area above the pallet taking into account the pallet surface, drop height and the margin
    # Note: The boxes can be spawned colliding with the surrounding collision walls as they will be pushed inwards
    pallet_range = bbox_cache.ComputeWorldBound(pallet_prim).GetRange()
    pallet_width, pallet_depth, pallet_heigth = pallet_range.GetSize()
    # Move the spawn center at the given height above the pallet surface
    spawn_center = pallet_range.GetMidpoint() + Gf.Vec3d(0, 0, pallet_heigth / 2 + drop_height)
    spawn_width, spawn_depth = pallet_width / 2 - drop_margin, pallet_depth / 2 - drop_margin

    # Use the pallet local-to-world transform to apply the local random offsets relative to the pallet
    pallet_tf: Gf.Matrix4d = UsdGeom.Xformable(pallet_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pallet_rot: Gf.Rotation = pallet_tf.ExtractRotation()

    # Simulate dropping the boxes from random poses on the pallet
    timeline = omni.timeline.get_timeline_interface()
    for box_prim in boxes:
        # Create a random location and orientation for the box within the drop area in local frame
        local_loc = spawn_center + Gf.Vec3d(
            random.uniform(-spawn_width, spawn_width), random.uniform(-spawn_depth, spawn_depth), 0
        )
        axes = [Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)]
        angles = [random.choice([180, 90, 0, -90, -180]) + random.uniform(-3, 3) for _ in axes]
        local_rot = Gf.Rotation()
        for axis, angle in zip(axes, angles):
            local_rot *= Gf.Rotation(axis, angle)

        # Transform the local pose to the pallet's world coordinate system
        world_loc = pallet_tf.Transform(local_loc)
        world_quat = Gf.Quatf((pallet_rot * local_rot).GetQuat())

        # Set the spawn pose and enable collisions and rigid body dynamics with dampened angular movements
        set_transform_attributes(box_prim, location=world_loc, orientation=world_quat)
        add_colliders(box_prim)
        add_rigid_body_dynamics(box_prim, angular_damping=0.9)

        # Bind the physics material to the box (allow frictionless sliding)
        mat_binding_api = UsdShade.MaterialBindingAPI.Apply(box_prim)
        mat_binding_api.Bind(default_material, UsdShade.Tokens.weakerThanDescendants, "physics")
        # Wait for an app update to load the new attributes
        await omni.kit.app.get_app().next_update_async()

        # Play simulation for a few frames for each box
        timeline.play()
        for _ in range(20):
            await omni.kit.app.get_app().next_update_async()
        timeline.pause()

    # Iteratively apply forces to the boxes to move them around then pull them all together towards the pallet center
    await apply_forces_async(stage, boxes, pallet_prim)

    # Remove rigid body dynamics of the boxes until all other scenarios are completed
    for box in boxes:
        UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Set(False)

    # Increase the friction to prevent sliding of the boxes on the pallet before removing the collision walls
    physics_material.CreateStaticFrictionAttr().Set(0.9)
    physics_material.CreateDynamicFrictionAttr().Set(0.9)

    # Remove collision walls
    for wall in collision_walls:
        stage.RemovePrim(wall.GetPath())
    return boxes

# Run the example scenario
async def run_box_stacking_scenarios_async(num_pallets, env_url=None, write_data=False):
    # Get assets root path once for all asset loading operations
    assets_root_path = await get_assets_root_path_async()

    # List of pallets and boxes to randomly choose from with their respective weights
    pallets_urls_and_weights = [
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd", 0.25),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_02.usd", 0.75),
    ]
    boxes_urls_and_weights = [
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.02),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd", 0.06),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd", 0.12),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.80),
    ]

    # Load a predefined or create a new stage
    if env_url is not None:
        env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
        omni.usd.get_context().open_stage(env_path)
        stage = omni.usd.get_context().get_stage()
    else:
        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
        distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(400.0)
        if not distant_light.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(distant_light).AddRotateXYZOp()
        distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))
        dome_light = stage.DefinePrim("/World/Lights/DomeLight", "DomeLight")
        dome_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(500.0)

    # Spawn the pallets
    pallets = []
    pallets_urls, pallets_weights = zip(*pallets_urls_and_weights)
    rand_pallet_urls = random.choices(pallets_urls, weights=pallets_weights, k=num_pallets)
    # Custom pallet poses for the evnironment
    custom_pallet_locations = [
        (-9.3, 5.3, 1.3),
        (-9.3, 7.3, 1.3),
        (-9.3, -0.6, 1.3),
    ]
    random.shuffle(custom_pallet_locations)
    for i, pallet_url in enumerate(rand_pallet_urls):
        # Use a custom location for every other pallet
        if env_url is not None:
            if i % 2 == 0 and custom_pallet_locations:
                rand_loc = Gf.Vec3d(*custom_pallet_locations.pop())
            else:
                rand_loc = Gf.Vec3d(-6.5, i * 1.75, 0) + Gf.Vec3d(random.uniform(-0.2, 0.2), random.uniform(0, 0.2), 0)
        else:
            rand_loc = Gf.Vec3d(i * 1.5, 0, 0) + Gf.Vec3d(random.uniform(0, 0.2), random.uniform(-0.2, 0.2), 0)
        rand_rot = (0, 0, random.choice([180, 90, 0, -90, -180]) + random.uniform(-15, 15))
        pallet_prim = create_asset_with_colliders(
            stage, pallet_url, f"/World/Pallet_{i}", location=rand_loc, rotation=rand_rot
        )
        pallets.append(pallet_prim)

    # Stack the boxes on the pallets
    total_boxes = []
    for pallet in pallets:
        if env_url is not None:
            rand_num_boxes = random.randint(8, 15)
            stacked_boxes = await stack_boxes_on_pallet_async(
                pallet, boxes_urls_and_weights, num_boxes=rand_num_boxes, drop_height=1.0
            )
        else:
            rand_num_boxes = random.randint(12, 20)
            stacked_boxes = await stack_boxes_on_pallet_async(pallet, boxes_urls_and_weights, num_boxes=rand_num_boxes)
        total_boxes.extend(stacked_boxes)

    # Re-enable rigid body dynamics of the boxes and run the simulation for a while
    for box in total_boxes:
        UsdPhysics.RigidBodyAPI(box).GetRigidBodyEnabledAttr().Set(True)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(200):
        await omni.kit.app.get_app().next_update_async()
    timeline.pause()

    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_box_stacking")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        cam = rep.functional.create.camera(position=(5, -5, 2), look_at=(0, 0, 0), name="PalletCamera")
        rp = rep.create.render_product(cam, resolution=(512, 512))
        writer.attach(rp)

        # Capture the data and wait for the data to be written to disk
        await rep.orchestrator.step_async(rt_subframes=8)

        # Wait for the data to be written to disk and cleanup
        await rep.orchestrator.wait_until_complete_async()
        writer.detach()
        rp.destroy()

# asyncio.ensure_future(run_box_stacking_scenarios_async(num_pallets=1, write_data=True))
asyncio.ensure_future(
    run_box_stacking_scenarios_async(
        num_pallets=6, env_url="/Isaac/Environments/Simple_Warehouse/warehouse.usd", write_data=True
    )
)
```

## Simready Assets SDG Example

Script editor example for using [SimReady Assets](https://developer.nvidia.com/omniverse/simready-assets) to randomize the scene. SimReady Assets are physically accurate 3D objects with realistic properties, behavior, and data connections that are optimized for simulation.

Note

The example can only run in async mode and requires the [SimReady Explorer](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_browser-extensions/simready-explorer.html "(in Omniverse Extensions)") window to be enabled to process the search requests.

The example script will create an SDG randomization and capture pipeline scenario with a table, a plate, and a number of items on top of the plate. The scene will be simulated for a while and then the captured images will be saved to disk.

The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/simready_assets_sdg.py
```

Script Editor

Simready Assets SDG Example

```python
import asyncio
import os
import time

import carb.settings
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import upgrade_prim_semantics_to_labels
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

# Make sure the simready explorer extension is enabled
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("omni.simready.explorer"):
    ext_manager.set_extension_enabled_immediate("omni.simready.explorer", True)
import omni.simready.explorer as sre

def enable_simready_explorer() -> None:
    """Enable the SimReady Explorer window if not already open."""
    if sre.get_instance().browser_model is None:
        import omni.kit.actions.core as actions

        actions.execute_action("omni.simready.explorer", "toggle_window")

def set_prim_variants(prim: Usd.Prim, variants: dict[str, str]) -> None:
    """Set variant selections on a prim from a dictionary of variant set names to values."""
    vsets = prim.GetVariantSets()
    for name, value in variants.items():
        vset = vsets.GetVariantSet(name)
        if vset:
            vset.SetVariantSelection(value)

async def search_assets_async() -> tuple[list, list, list]:
    """Search for SimReady assets (tables, dishes, items) asynchronously."""
    print(f"[SDG] Searching for SimReady assets...")
    start_time = time.time()
    tables = await sre.find_assets(["table", "furniture"])
    print(f"[SDG]   - Found {len(tables)} tables ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    plates = await sre.find_assets(["plate"])
    print(f"[SDG]   - Found {len(plates)} plates ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    bowls = await sre.find_assets(["bowl"])
    print(f"[SDG]   - Found {len(bowls)} bowls ({time.time() - start_time:.2f}s)")
    dishes = plates + bowls
    start_time = time.time()
    fruits = await sre.find_assets(["fruit"])
    print(f"[SDG]   - Found {len(fruits)} fruits ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    vegetables = await sre.find_assets(["vegetable"])
    print(f"[SDG]   - Found {len(vegetables)} vegetables ({time.time() - start_time:.2f}s)")
    items = fruits + vegetables
    return tables, dishes, items

async def run_simready_randomization_async(
    stage: Usd.Stage,
    camera_prim: Usd.Prim,
    render_product,
    tables: list,
    dishes: list,
    items: list,
    rng: np.random.Generator = None,
) -> None:
    """Randomize a scene with SimReady assets, run physics, and capture the result."""
    if rng is None:
        rng = np.random.default_rng()

    print(f"[SDG]   Creating anonymous variation layer for the randomizations...")
    root_layer = stage.GetRootLayer()
    variation_layer = Sdf.Layer.CreateAnonymous("variation")
    root_layer.subLayerPaths.insert(0, variation_layer.identifier)
    stage.SetEditTarget(variation_layer)

    # Load the simready assets with rigid body properties
    variants = {"PhysicsVariant": "RigidBody"}
    rep.functional.create.scope(name="Assets")

    # Choose a random table and add it to the stage
    print(f"[SDG]   Loading assets...")
    table_asset = tables[rng.integers(len(tables))]
    start_time = time.time()
    table_prim = rep.functional.create.reference(usd_path=table_asset.main_url, parent="/Assets", name=table_asset.name)
    set_prim_variants(table_prim, variants)
    upgrade_prim_semantics_to_labels(table_prim)
    print(f"[SDG]     - Table: '{table_asset.name}' ({time.time() - start_time:.2f}s)")
    await omni.kit.app.get_app().next_update_async()

    # Keep only colliders on the table (disable rigid body dynamics)
    UsdPhysics.RigidBodyAPI(table_prim).GetRigidBodyEnabledAttr().Set(False)

    # Compute table dimensions from its bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    table_bbox = bbox_cache.ComputeWorldBound(table_prim)
    table_extent = table_bbox.GetRange().GetSize()

    # Choose a random dish and add it to the stage
    dish_asset = dishes[rng.integers(len(dishes))]
    start_time = time.time()
    dish_prim = rep.functional.create.reference(usd_path=dish_asset.main_url, parent="/Assets", name=dish_asset.name)
    set_prim_variants(dish_prim, variants)
    upgrade_prim_semantics_to_labels(dish_prim)
    print(f"[SDG]     - Dish: '{dish_asset.name}' ({time.time() - start_time:.2f}s)")
    await omni.kit.app.get_app().next_update_async()

    # Compute dish dimensions from its bounding box
    dish_bbox = bbox_cache.ComputeWorldBound(dish_prim)
    dish_extent = dish_bbox.GetRange().GetSize()

    # Calculate random position for the dish near the center of the table
    center_region_scale = 0.75
    dish_range_x = max(0, (table_extent[0] - dish_extent[0]) / 2 * center_region_scale)
    dish_range_y = max(0, (table_extent[1] - dish_extent[1]) / 2 * center_region_scale)
    dish_position = (
        rng.uniform(-dish_range_x, dish_range_x) if dish_range_x > 0 else 0,
        rng.uniform(-dish_range_y, dish_range_y) if dish_range_y > 0 else 0,
        table_extent[2] + dish_extent[2] / 2,
    )
    dish_prim.GetAttribute("xformOp:translate").Set(dish_position)

    # Add random items above the dish
    num_items = rng.integers(2, 5)
    item_prims = []
    for _ in range(num_items):
        item_asset = items[rng.integers(len(items))]
        start_time = time.time()
        item_prim = rep.functional.create.reference(
            usd_path=item_asset.main_url, parent="/Assets", name=item_asset.name
        )
        set_prim_variants(item_prim, variants)
        upgrade_prim_semantics_to_labels(item_prim)
        print(f"[SDG]     - Item: '{item_asset.name}' ({time.time() - start_time:.2f}s)")
        item_prims.append(item_prim)
        await omni.kit.app.get_app().next_update_async()

    # Position items stacked above the dish
    print(f"[SDG]   Positioning assets on table...")
    stack_height = dish_position[2]
    item_scatter_radius = max(0, dish_extent[0] / 4)
    for item_prim in item_prims:
        item_bbox = bbox_cache.ComputeWorldBound(item_prim)
        item_extent = item_bbox.GetRange().GetSize()
        scatter_x = rng.uniform(-item_scatter_radius, item_scatter_radius) if item_scatter_radius > 0 else 0
        scatter_y = rng.uniform(-item_scatter_radius, item_scatter_radius) if item_scatter_radius > 0 else 0
        item_position = (
            dish_position[0] + scatter_x,
            dish_position[1] + scatter_y,
            stack_height + item_extent[2] / 2,
        )
        item_prim.GetAttribute("xformOp:translate").Set(item_position)
        stack_height += item_extent[2]

    # Run physics simulation for items to settle
    num_sim_steps = 25
    print(f"[SDG]   Running physics simulation ({num_sim_steps} steps)...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(num_sim_steps):
        await omni.kit.app.get_app().next_update_async()
    timeline.pause()

    print(f"[SDG]   Setting edit target to root layer...")
    stage.SetEditTarget(root_layer)

    print(f"[SDG]   Positioning camera and capturing frame...")
    camera_position = (
        dish_position[0] + rng.uniform(-0.5, 0.5),
        dish_position[1] + rng.uniform(-0.5, 0.5),
        dish_position[2] + 1.5 + rng.uniform(-0.5, 0.5),
    )
    rep.functional.modify.pose(
        camera_prim, position_value=camera_position, look_at_value=dish_prim, look_at_up_axis=(0, 0, 1)
    )
    render_product.hydra_texture.set_updates_enabled(True)
    await rep.orchestrator.step_async(delta_time=0.0, rt_subframes=16)
    render_product.hydra_texture.set_updates_enabled(False)

    print(f"[SDG]   Removing temp variation layer...")
    variation_layer.Clear()
    root_layer.subLayerPaths.remove(variation_layer.identifier)

async def run_simready_randomizations_async(num_scenarios: int) -> None:
    """Run multiple SimReady randomization scenarios and capture the results."""
    print(f"[SDG] Initializing scene...")
    await omni.usd.get_context().new_stage_async()
    stage = omni.usd.get_context().get_stage()

    # Initialize randomization
    rng = np.random.default_rng(34)
    rep.set_global_seed(34)

    # Data capture will happen manually using step()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Add lights to the scene
    print(f"[SDG] Setting up lighting...")
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    rep.functional.create.distant_light(intensity=2500, parent="/World", name="DistantLight", rotation=(-75, 0, 0))

    # Simready explorer window needs to be created for the search to work
    enable_simready_explorer()

    # Search for the simready assets
    tables, dishes, items = await search_assets_async()

    # Create the writer and the render product for capturing the scene
    output_dir = os.path.join(os.getcwd(), "_out_simready_assets")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_dir)
    writer = rep.writers.get("BasicWriter")
    print(f"[SDG] Initializing writer, output directory: {output_dir}...")
    writer.initialize(backend=backend, rgb=True)

    # Create camera and render product (disabled by default, enabled only when capturing)
    print(f"[SDG] Creating camera and render product...")
    camera_prim = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(camera_prim, (512, 512))
    rp.hydra_texture.set_updates_enabled(False)
    writer.attach(rp)

    # Generate randomized scenarios
    for i in range(num_scenarios):
        print(f"[SDG] Scenario {i + 1}/{num_scenarios}")
        await run_simready_randomization_async(
            stage=stage, camera_prim=camera_prim, render_product=rp, tables=tables, dishes=dishes, items=items, rng=rng
        )

    # Finalize and cleanup
    print("[SDG] Wait for the data to be written and cleanup render products...")
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

num_scenarios = 5
print(f"[SDG] Starting SDG pipeline with {num_scenarios} scenarios...")
asyncio.ensure_future(run_simready_randomizations_async(num_scenarios))
```

Standalone Application

Simready Assets SDG Example

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import asyncio
import os
import time

import carb.settings
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.utils.semantics import upgrade_prim_semantics_to_labels
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

parser = argparse.ArgumentParser()
parser.add_argument("--num_scenarios", type=int, default=5, help="Number of randomization scenarios to create")
args, _ = parser.parse_known_args()
num_scenarios = args.num_scenarios

# Make sure the simready explorer extension is enabled
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("omni.simready.explorer"):
    ext_manager.set_extension_enabled_immediate("omni.simready.explorer", True)
import omni.simready.explorer as sre

def enable_simready_explorer() -> None:
    """Enable the SimReady Explorer window if not already open."""
    if sre.get_instance().browser_model is None:
        import omni.kit.actions.core as actions

        actions.execute_action("omni.simready.explorer", "toggle_window")

def set_prim_variants(prim: Usd.Prim, variants: dict[str, str]) -> None:
    """Set variant selections on a prim from a dictionary of variant set names to values."""
    vsets = prim.GetVariantSets()
    for name, value in variants.items():
        vset = vsets.GetVariantSet(name)
        if vset:
            vset.SetVariantSelection(value)

async def search_assets_async() -> tuple[list, list, list]:
    """Search for SimReady assets (tables, dishes, items) asynchronously."""
    print(f"[SDG] Searching for SimReady assets...")
    start_time = time.time()
    tables = await sre.find_assets(["table", "furniture"])
    print(f"[SDG]   - Found {len(tables)} tables ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    plates = await sre.find_assets(["plate"])
    print(f"[SDG]   - Found {len(plates)} plates ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    bowls = await sre.find_assets(["bowl"])
    print(f"[SDG]   - Found {len(bowls)} bowls ({time.time() - start_time:.2f}s)")
    dishes = plates + bowls
    start_time = time.time()
    fruits = await sre.find_assets(["fruit"])
    print(f"[SDG]   - Found {len(fruits)} fruits ({time.time() - start_time:.2f}s)")
    start_time = time.time()
    vegetables = await sre.find_assets(["vegetable"])
    print(f"[SDG]   - Found {len(vegetables)} vegetables ({time.time() - start_time:.2f}s)")
    items = fruits + vegetables
    return tables, dishes, items

def run_simready_randomization(
    stage: Usd.Stage,
    camera_prim: Usd.Prim,
    render_product,
    tables: list,
    dishes: list,
    items: list,
    rng: np.random.Generator = None,
) -> None:
    """Randomize a scene with SimReady assets, run physics, and capture the result."""
    if rng is None:
        rng = np.random.default_rng()

    print(f"[SDG]   Creating anonymous variation layer for the randomizations...")
    root_layer = stage.GetRootLayer()
    variation_layer = Sdf.Layer.CreateAnonymous("variation")
    root_layer.subLayerPaths.insert(0, variation_layer.identifier)
    stage.SetEditTarget(variation_layer)

    # Load the simready assets with rigid body properties
    variants = {"PhysicsVariant": "RigidBody"}
    rep.functional.create.scope(name="Assets")

    # Choose a random table and add it to the stage
    print(f"[SDG]   Loading assets...")
    table_asset = tables[rng.integers(len(tables))]
    start_time = time.time()
    table_prim = rep.functional.create.reference(usd_path=table_asset.main_url, parent="/Assets", name=table_asset.name)
    set_prim_variants(table_prim, variants)
    upgrade_prim_semantics_to_labels(table_prim)
    print(f"[SDG]     - Table: '{table_asset.name}' ({time.time() - start_time:.2f}s)")
    simulation_app.update()

    # Keep only colliders on the table (disable rigid body dynamics)
    UsdPhysics.RigidBodyAPI(table_prim).GetRigidBodyEnabledAttr().Set(False)

    # Compute table dimensions from its bounding box
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    table_bbox = bbox_cache.ComputeWorldBound(table_prim)
    table_extent = table_bbox.GetRange().GetSize()

    # Choose a random dish and add it to the stage
    dish_asset = dishes[rng.integers(len(dishes))]
    start_time = time.time()
    dish_prim = rep.functional.create.reference(usd_path=dish_asset.main_url, parent="/Assets", name=dish_asset.name)
    set_prim_variants(dish_prim, variants)
    upgrade_prim_semantics_to_labels(dish_prim)
    print(f"[SDG]     - Dish: '{dish_asset.name}' ({time.time() - start_time:.2f}s)")
    simulation_app.update()

    # Compute dish dimensions from its bounding box
    dish_bbox = bbox_cache.ComputeWorldBound(dish_prim)
    dish_extent = dish_bbox.GetRange().GetSize()

    # Calculate random position for the dish near the center of the table
    center_region_scale = 0.75
    dish_range_x = max(0, (table_extent[0] - dish_extent[0]) / 2 * center_region_scale)
    dish_range_y = max(0, (table_extent[1] - dish_extent[1]) / 2 * center_region_scale)
    dish_position = (
        rng.uniform(-dish_range_x, dish_range_x) if dish_range_x > 0 else 0,
        rng.uniform(-dish_range_y, dish_range_y) if dish_range_y > 0 else 0,
        table_extent[2] + dish_extent[2] / 2,
    )
    dish_prim.GetAttribute("xformOp:translate").Set(dish_position)

    # Add random items above the dish
    num_items = rng.integers(2, 5)
    item_prims = []
    for _ in range(num_items):
        item_asset = items[rng.integers(len(items))]
        start_time = time.time()
        item_prim = rep.functional.create.reference(
            usd_path=item_asset.main_url, parent="/Assets", name=item_asset.name
        )
        set_prim_variants(item_prim, variants)
        upgrade_prim_semantics_to_labels(item_prim)
        print(f"[SDG]     - Item: '{item_asset.name}' ({time.time() - start_time:.2f}s)")
        item_prims.append(item_prim)
        simulation_app.update()

    # Position items stacked above the dish
    print(f"[SDG]   Positioning assets on table...")
    stack_height = dish_position[2]
    item_scatter_radius = max(0, dish_extent[0] / 4)
    for item_prim in item_prims:
        item_bbox = bbox_cache.ComputeWorldBound(item_prim)
        item_extent = item_bbox.GetRange().GetSize()
        scatter_x = rng.uniform(-item_scatter_radius, item_scatter_radius) if item_scatter_radius > 0 else 0
        scatter_y = rng.uniform(-item_scatter_radius, item_scatter_radius) if item_scatter_radius > 0 else 0
        item_position = (
            dish_position[0] + scatter_x,
            dish_position[1] + scatter_y,
            stack_height + item_extent[2] / 2,
        )
        item_prim.GetAttribute("xformOp:translate").Set(item_position)
        stack_height += item_extent[2]

    # Run physics simulation for items to settle
    num_sim_steps = 25
    print(f"[SDG]   Running physics simulation ({num_sim_steps} steps)...")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(num_sim_steps):
        simulation_app.update()
    timeline.pause()

    print(f"[SDG]   Setting edit target to root layer...")
    stage.SetEditTarget(root_layer)

    print(f"[SDG]   Positioning camera and capturing frame...")
    camera_position = (
        dish_position[0] + rng.uniform(-0.5, 0.5),
        dish_position[1] + rng.uniform(-0.5, 0.5),
        dish_position[2] + 1.5 + rng.uniform(-0.5, 0.5),
    )
    rep.functional.modify.pose(
        camera_prim, position_value=camera_position, look_at_value=dish_prim, look_at_up_axis=(0, 0, 1)
    )
    render_product.hydra_texture.set_updates_enabled(True)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=16)
    render_product.hydra_texture.set_updates_enabled(False)

    print(f"[SDG]   Removing temp variation layer...")
    variation_layer.Clear()
    root_layer.subLayerPaths.remove(variation_layer.identifier)

def run_simready_randomizations(num_scenarios: int) -> None:
    """Run multiple SimReady randomization scenarios and capture the results."""
    print(f"[SDG] Initializing scene...")
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()

    # Initialize randomization
    rng = np.random.default_rng(34)
    rep.set_global_seed(34)

    # Data capture will happen manually using step()
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Add lights to the scene
    print(f"[SDG] Setting up lighting...")
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
    rep.functional.create.distant_light(intensity=2500, parent="/World", name="DistantLight", rotation=(-75, 0, 0))

    # Simready explorer window needs to be created for the search to work
    enable_simready_explorer()

    # Search for the simready assets and wait until the task is complete
    search_task = asyncio.ensure_future(search_assets_async())
    while not search_task.done():
        simulation_app.update()
    tables, dishes, items = search_task.result()

    # Create the writer and the render product for capturing the scene
    output_dir = os.path.join(os.getcwd(), "_out_simready_assets")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_dir)
    writer = rep.writers.get("BasicWriter")
    print(f"[SDG] Initializing writer, output directory: {output_dir}...")
    writer.initialize(backend=backend, rgb=True)

    # Create camera and render product (disabled by default, enabled only when capturing)
    print(f"[SDG] Creating camera and render product...")
    camera_prim = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(camera_prim, (512, 512))
    rp.hydra_texture.set_updates_enabled(False)
    writer.attach(rp)

    # Generate randomized scenarios
    for i in range(num_scenarios):
        print(f"[SDG] Scenario {i + 1}/{num_scenarios}")
        run_simready_randomization(
            stage=stage, camera_prim=camera_prim, render_product=rp, tables=tables, dishes=dishes, items=items, rng=rng
        )

    # Finalize and cleanup
    print("[SDG] Wait for the data to be written and cleanup render products...")
    rep.orchestrator.wait_until_complete()
    writer.detach()
    rp.destroy()

print(f"[SDG] Starting SDG pipeline with {num_scenarios} scenarios...")
run_simready_randomizations(num_scenarios)

simulation_app.close()
```

---

# Useful Snippets

Various examples of Isaac Sim Replicator snippets that can be run as [Standalone Applications](Workflows.md) or from the UI using the [Script Editor](Development_Tools.md).

## Annotator and Custom Writer Data from Multiple Cameras

Example on how to access data from multiple cameras in a scene using annotators or custom writers. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/multi_camera.py
```

Script Editor

Annotator and Custom Writer Data from Multiple Cameras

```python
import asyncio
import os

import carb.settings
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core import Writer
from omni.replicator.core.backends import DiskBackend
from omni.replicator.core.functional import write_image

NUM_FRAMES = 5

# Randomize cube color every frame using a graph-based replicator randomizer
def cube_color_randomizer():
    cube_prims = rep.get.prims(path_pattern="Cube")
    with cube_prims:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
    return cube_prims.node

# Example of custom writer class to access the annotator data
class MyWriter(Writer):
    def __init__(self, rgb: bool = True):
        # Organize data from render product perspective (legacy, annotator, renderProduct)
        self.data_structure = "renderProduct"
        self.annotators = []
        self._frame_id = 0
        if rgb:
            # Create a new rgb annotator and add it to the writer's list of annotators
            self.annotators.append(rep.annotators.get("rgb"))
        # Create writer output directory and initialize DiskBackend
        output_dir = os.path.join(os.getcwd(), "_out_mc_writer")
        print(f"Writing writer data to {output_dir}")
        self.backend = DiskBackend(output_dir=output_dir, overwrite=True)

    def write(self, data):
        if "renderProducts" in data:
            for rp_name, rp_data in data["renderProducts"].items():
                if "rgb" in rp_data:
                    file_path = f"{rp_name}_frame_{self._frame_id}.png"
                    self.backend.schedule(write_image, data=rp_data["rgb"]["data"], path=file_path)
        self._frame_id += 1

rep.WriterRegistry.register(MyWriter)

# Create a new stage
omni.usd.get_context().new_stage()

# Set global random seed for the replicator randomizer
rep.set_global_seed(11)

# Disable capture on play to capture data manually using step
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Setup stage
rep.functional.create.xform(name="World")
rep.functional.create.dome_light(intensity=900, parent="/World", name="DomeLight")
cube = rep.functional.create.cube(parent="/World", name="Cube", semantics={"class": "my_cube"})

# Register the graph-based cube color randomizer to trigger on every frame
rep.randomizer.register(cube_color_randomizer)
with rep.trigger.on_frame():
    rep.randomizer.cube_color_randomizer()

# Create cameras
cam_top = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0), parent="/World", name="CamTop")
cam_side = rep.functional.create.camera(position=(2, 2, 0), look_at=(0, 0, 0), parent="/World", name="CamSide")
cam_persp = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="CamPersp")

# Create the render products
rp_top = rep.create.render_product(cam_top, resolution=(320, 320), name="RpTop")
rp_side = rep.create.render_product(cam_side, resolution=(640, 640), name="RpSide")
rp_persp = rep.create.render_product(cam_persp, resolution=(1024, 1024), name="RpPersp")

# Example of accessing the data through a custom writer
writer = rep.WriterRegistry.get("MyWriter")
writer.initialize(rgb=True)
writer.attach([rp_top, rp_side, rp_persp])

# Example of accessing the data directly through annotators
rgb_annotators = []
for rp in [rp_top, rp_side, rp_persp]:
    # Create a new rgb annotator for each render product
    rgb = rep.annotators.get("rgb")
    # Attach the annotator to the render product
    rgb.attach(rp)
    rgb_annotators.append(rgb)

# Create annotator output directory
output_dir_annot = os.path.join(os.getcwd(), "_out_mc_annot")
print(f"Writing annotator data to {output_dir_annot}")
os.makedirs(output_dir_annot, exist_ok=True)

async def run_example_async():
    for i in range(NUM_FRAMES):
        print(f"Step {i}")
        # The step function triggers registered graph-based randomizers, collects data from annotators,
        # and invokes the write function of attached writers with the annotator data
        await rep.orchestrator.step_async(rt_subframes=32)
        for j, rgb_annot in enumerate(rgb_annotators):
            file_path = os.path.join(output_dir_annot, f"rp{j}_step_{i}.png")
            write_image(path=file_path, data=rgb_annot.get_data())

    # Wait for the data to be written and release resources
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    for annot in rgb_annotators:
        annot.detach()
    for rp in [rp_top, rp_side, rp_persp]:
        rp.destroy()

asyncio.ensure_future(run_example_async())
```

Standalone Application

Annotator and Custom Writer Data from Multiple Cameras

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import os

import carb.settings
import omni.replicator.core as rep
import omni.usd
from omni.replicator.core import Writer
from omni.replicator.core.backends import DiskBackend
from omni.replicator.core.functional import write_image

NUM_FRAMES = 5

# Randomize cube color every frame using a graph-based replicator randomizer
def cube_color_randomizer():
    cube_prims = rep.get.prims(path_pattern="Cube")
    with cube_prims:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
    return cube_prims.node

# Example of custom writer class to access the annotator data
class MyWriter(Writer):
    def __init__(self, rgb: bool = True):
        # Organize data from render product perspective (legacy, annotator, renderProduct)
        self.data_structure = "renderProduct"
        self.annotators = []
        self._frame_id = 0
        if rgb:
            # Create a new rgb annotator and add it to the writer's list of annotators
            self.annotators.append(rep.annotators.get("rgb"))
        # Create writer output directory and initialize DiskBackend
        output_dir = os.path.join(os.getcwd(), "_out_mc_writer")
        print(f"Writing writer data to {output_dir}")
        self.backend = DiskBackend(output_dir=output_dir, overwrite=True)

    def write(self, data):
        if "renderProducts" in data:
            for rp_name, rp_data in data["renderProducts"].items():
                if "rgb" in rp_data:
                    file_path = f"{rp_name}_frame_{self._frame_id}.png"
                    self.backend.schedule(write_image, data=rp_data["rgb"]["data"], path=file_path)
        self._frame_id += 1

rep.WriterRegistry.register(MyWriter)

# Create a new stage
omni.usd.get_context().new_stage()

# Set global random seed for the replicator randomizer
rep.set_global_seed(11)

# Disable capture on play to capture data manually using step
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Setup stage
rep.functional.create.xform(name="World")
rep.functional.create.dome_light(intensity=900, parent="/World", name="DomeLight")
cube = rep.functional.create.cube(parent="/World", name="Cube", semantics={"class": "my_cube"})

# Register the graph-based cube color randomizer to trigger on every frame
rep.randomizer.register(cube_color_randomizer)
with rep.trigger.on_frame():
    rep.randomizer.cube_color_randomizer()

# Create cameras
cam_top = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0), parent="/World", name="CamTop")
cam_side = rep.functional.create.camera(position=(2, 2, 0), look_at=(0, 0, 0), parent="/World", name="CamSide")
cam_persp = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="CamPersp")

# Create the render products
rp_top = rep.create.render_product(cam_top, resolution=(320, 320), name="RpTop")
rp_side = rep.create.render_product(cam_side, resolution=(640, 640), name="RpSide")
rp_persp = rep.create.render_product(cam_persp, resolution=(1024, 1024), name="RpPersp")

# Example of accessing the data through a custom writer
writer = rep.WriterRegistry.get("MyWriter")
writer.initialize(rgb=True)
writer.attach([rp_top, rp_side, rp_persp])

# Example of accessing the data directly through annotators
rgb_annotators = []
for rp in [rp_top, rp_side, rp_persp]:
    # Create a new rgb annotator for each render product
    rgb = rep.annotators.get("rgb")
    # Attach the annotator to the render product
    rgb.attach(rp)
    rgb_annotators.append(rgb)

# Create annotator output directory
output_dir_annot = os.path.join(os.getcwd(), "_out_mc_annot")
print(f"Writing annotator data to {output_dir_annot}")
os.makedirs(output_dir_annot, exist_ok=True)

for i in range(NUM_FRAMES):
    print(f"Step {i}")
    # The step function triggers registered graph-based randomizers, collects data from annotators,
    # and invokes the write function of attached writers with the annotator data
    rep.orchestrator.step(rt_subframes=32)
    for j, rgb_annot in enumerate(rgb_annotators):
        file_path = os.path.join(output_dir_annot, f"rp{j}_step_{i}.png")
        write_image(path=file_path, data=rgb_annot.get_data())

# Wait for the data to be written and release resources
rep.orchestrator.wait_until_complete()
writer.detach()
for annot in rgb_annotators:
    annot.detach()
for rp in [rp_top, rp_side, rp_persp]:
    rp.destroy()

simulation_app.close()
```

## Synthetic Data Access at Specific Simulation Timepoints

Example on how to access synthetic data (RGB, semantic segmentation) from multiple cameras in a simulation scene at specific events using annotators or writers. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/simulation_get_data.py
```

Script Editor

Synthetic Data Access at Specific Simulation Timepoints

```python
import asyncio
import json
import os

import carb.settings
import numpy as np
import omni
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.simulation_manager import SimulationManager
from omni.replicator.core.functional import write_image, write_json
from pxr import UsdPhysics

# Util function to save semantic segmentation annotator data
def write_sem_data(sem_data, file_path):
    id_to_labels = sem_data["info"]["idToLabels"]
    write_json(path=file_path + ".json", data=id_to_labels)
    sem_image_data = sem_data["data"]
    write_image(path=file_path + ".png", data=sem_image_data)

# Create a new stage
omni.usd.get_context().new_stage()

# Setting capture on play to False will prevent the replicator from capturing data each frame
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Add a dome light and a ground plane
rep.functional.create.xform(name="World")
rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
ground_plane = GroundPlane("/World/GroundPlane")
rep.functional.modify.semantics(ground_plane.prims, {"class": "ground_plane"}, mode="add")

# Create a camera and render product to collect the data from
rep.functional.create.xform(name="World")
cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
rp = rep.create.render_product(cam, resolution=(512, 512), name="MyRenderProduct")

# Set the output directory for the data
out_dir = os.path.join(os.getcwd(), "_out_sim_event")
writer_dir = os.path.join(out_dir, "writer")
annotator_dir = os.path.join(out_dir, "annotator")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(writer_dir, exist_ok=True)
os.makedirs(annotator_dir, exist_ok=True)

print(f"Outputting data to {out_dir}..")
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=writer_dir)

# Example of using a writer to save the data
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
writer.attach(rp)

# Example of accesing the data directly from annotators
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach(rp)
sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": True})
sem_annot.attach(rp)

# Initialize the simulation manager
SimulationManager.initialize_physics()

async def run_example_async():
    # Spawn and drop a few cubes, capture data when they stop moving
    for i in range(5):
        cube = rep.functional.create.cube(name=f"Cuboid_{i}", parent="/World")
        rep.functional.modify.position(cube, (0, 0, 10 + i))
        rep.functional.modify.semantics(cube, {"class": "cuboid"}, mode="add")
        rep.functional.physics.apply_rigid_body(cube, with_collider=True)
        physics_rigid_body_api = UsdPhysics.RigidBodyAPI(cube)

        for s in range(500):
            SimulationManager.step()
            linear_velocity = physics_rigid_body_api.GetVelocityAttr().Get()
            speed = np.linalg.norm(linear_velocity)

            if speed < 0.1:
                print(f"Cube_{i} stopped moving after {s} simulation steps, writing data..")
                # Tigger the writer and update the annotators with new data
                await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0, pause_timeline=False)
                rgb_path = os.path.join(annotator_dir, f"Cube_{i}_step_{s}_rgb.png")
                sem_path = os.path.join(annotator_dir, f"Cube_{i}_step_{s}_sem")
                write_image(path=rgb_path, data=rgb_annot.get_data())
                write_sem_data(sem_annot.get_data(), sem_path)
                break

    # Wait for the data to be written to disk and clean up resources
    await rep.orchestrator.wait_until_complete_async()
    rgb_annot.detach()
    sem_annot.detach()
    writer.detach()
    rp.destroy()

asyncio.ensure_future(run_example_async())
```

Standalone Application

Synthetic Data Access at Specific Simulation Timepoints

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import json
import os

import carb.settings
import numpy as np
import omni
import omni.replicator.core as rep
from isaacsim.core.experimental.objects import GroundPlane
from isaacsim.core.simulation_manager import SimulationManager
from omni.replicator.core.functional import write_image, write_json
from pxr import UsdPhysics

# Util function to save semantic segmentation annotator data
def write_sem_data(sem_data, file_path):
    id_to_labels = sem_data["info"]["idToLabels"]
    write_json(path=file_path + ".json", data=id_to_labels)
    sem_image_data = sem_data["data"]
    write_image(path=file_path + ".png", data=sem_image_data)

# Create a new stage
omni.usd.get_context().new_stage()

# Setting capture on play to False will prevent the replicator from capturing data each frame
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Add a dome light and a ground plane
rep.functional.create.xform(name="World")
rep.functional.create.dome_light(intensity=500, parent="/World", name="DomeLight")
ground_plane = GroundPlane("/World/GroundPlane")
rep.functional.modify.semantics(ground_plane.prims, {"class": "ground_plane"}, mode="add")

# Create a camera and render product to collect the data from
cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
rp = rep.create.render_product(cam, resolution=(512, 512), name="MyRenderProduct")

# Set the output directory for the data
out_dir = os.path.join(os.getcwd(), "_out_sim_event")
writer_dir = os.path.join(out_dir, "writer")
annotator_dir = os.path.join(out_dir, "annotator")

os.makedirs(out_dir, exist_ok=True)
os.makedirs(writer_dir, exist_ok=True)
os.makedirs(annotator_dir, exist_ok=True)

print(f"Outputting data to {out_dir}..")
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=writer_dir)

# Example of using a writer to save the data
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(backend=backend, rgb=True, semantic_segmentation=True, colorize_semantic_segmentation=True)
writer.attach(rp)

# Example of accesing the data directly from annotators
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach(rp)
sem_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": True})
sem_annot.attach(rp)

# Initialize the simulation manager
simulation_manager = SimulationManager()
simulation_manager.initialize_physics()

# Spawn and drop a few cubes, capture data when they stop moving
for i in range(5):
    cube = rep.functional.create.cube(name=f"Cuboid_{i}", parent="/World")
    rep.functional.modify.position(cube, (0, 0, 10 + i))
    rep.functional.modify.semantics(cube, {"class": "cuboid"}, mode="add")
    rep.functional.physics.apply_rigid_body(cube, with_collider=True)
    physics_rigid_body_api = UsdPhysics.RigidBodyAPI(cube)

    for s in range(500):
        simulation_manager.step()
        linear_velocity = physics_rigid_body_api.GetVelocityAttr().Get()
        speed = np.linalg.norm(linear_velocity)

        if speed < 0.1:
            print(f"Cube_{i} stopped moving after {s} simulation steps, writing data..")
            # Tigger the writer and update the annotators with new data
            rep.orchestrator.step(rt_subframes=4, delta_time=0.0, pause_timeline=False)
            rgb_path = os.path.join(annotator_dir, f"Cube_{i}_step_{s}_rgb.png")
            write_image(path=rgb_path, data=rgb_annot.get_data())
            sem_path = os.path.join(annotator_dir, f"Cube_{i}_step_{s}_sem")
            write_sem_data(sem_annot.get_data(), sem_path)
            break

# Wait for the data to be written to disk and clean up resources
rep.orchestrator.wait_until_complete()
rgb_annot.detach()
sem_annot.detach()
writer.detach()
rp.destroy()

simulation_app.close()
```

## Custom Event Randomization and Writing

The following example showcases the use of custom events to trigger randomizations and data writing at various times throughout the simulation. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/custom_event_and_write.py
```

Script Editor

Custom Event Randomization and Writing

```python
import asyncio
import os

import carb.settings
import omni.replicator.core as rep
import omni.usd

omni.usd.get_context().new_stage()

# Set global random seed for the replicator randomizer to ensure reproducibility
rep.set_global_seed(11)

# Setting capture on play to False will prevent the replicator from capturing data each frame
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

rep.functional.create.xform(name="World")
rep.functional.create.distant_light(intensity=4000, rotation=(315, 0, 0), parent="/World", name="DistantLight")
small_cube = rep.functional.create.cube(scale=0.75, position=(-1.5, 1.5, 0), parent="/World", name="SmallCube")
large_cube = rep.functional.create.cube(scale=1.25, position=(1.5, -1.5, 0), parent="/World", name="LargeCube")

# Graph-based randomizations triggered on custom events
with rep.trigger.on_custom_event(event_name="randomize_small_cube"):
    small_cube_node = rep.get.prim_at_path(small_cube.GetPath())
    with small_cube_node:
        rep.randomizer.rotation()

with rep.trigger.on_custom_event(event_name="randomize_large_cube"):
    large_cube_node = rep.get.prim_at_path(large_cube.GetPath())
    with large_cube_node:
        rep.randomizer.rotation()

# Use the disk backend to write the data to disk
out_dir = os.path.join(os.getcwd(), "_out_custom_event")
print(f"Writing data to {out_dir}")
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=out_dir)

cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
rp = rep.create.render_product(cam, (512, 512))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(backend=backend, rgb=True)
writer.attach(rp)

async def run_example_async():
    print(f"Capturing at original positions")
    await rep.orchestrator.step_async(rt_subframes=8)

    print("Randomizing small cube rotation (graph-based) and capturing...")
    rep.utils.send_og_event(event_name="randomize_small_cube")
    await rep.orchestrator.step_async(rt_subframes=8)

    print("Moving small cube position (USD API) and capturing...")
    small_cube.GetAttribute("xformOp:translate").Set((-1.5, 1.5, -2))
    await rep.orchestrator.step_async(rt_subframes=8)

    print("Randomizing large cube rotation (graph-based) and capturing...")
    rep.utils.send_og_event(event_name="randomize_large_cube")
    await rep.orchestrator.step_async(rt_subframes=8)

    print("Moving large cube position (USD API) and capturing...")
    large_cube.GetAttribute("xformOp:translate").Set((1.5, -1.5, 2))
    await rep.orchestrator.step_async(rt_subframes=8)

    # Wait until all the data is saved to disk and cleanup writer and render product
    await rep.orchestrator.wait_until_complete_async()
    writer.detach()
    rp.destroy()

asyncio.ensure_future(run_example_async())
```

Standalone Application

Custom Event Randomization and Writing

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import os

import carb.settings
import omni.replicator.core as rep
import omni.usd

omni.usd.get_context().new_stage()

# Set global random seed for the replicator randomizer to ensure reproducibility
rep.set_global_seed(11)

# Setting capture on play to False will prevent the replicator from capturing data each frame
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

rep.functional.create.xform(name="World")
rep.functional.create.distant_light(intensity=4000, rotation=(315, 0, 0), parent="/World", name="DistantLight")
small_cube = rep.functional.create.cube(scale=0.75, position=(-1.5, 1.5, 0), parent="/World", name="SmallCube")
large_cube = rep.functional.create.cube(scale=1.25, position=(1.5, -1.5, 0), parent="/World", name="LargeCube")

# Graph-based randomizations triggered on custom events
with rep.trigger.on_custom_event(event_name="randomize_small_cube"):
    small_cube_node = rep.get.prim_at_path(small_cube.GetPath())
    with small_cube_node:
        rep.randomizer.rotation()

with rep.trigger.on_custom_event(event_name="randomize_large_cube"):
    large_cube_node = rep.get.prim_at_path(large_cube.GetPath())
    with large_cube_node:
        rep.randomizer.rotation()

# Use the disk backend to write the data to disk
out_dir = os.path.join(os.getcwd(), "_out_custom_event")
print(f"Writing data to {out_dir}")
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir=out_dir)

cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
rp = rep.create.render_product(cam, (512, 512))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(backend=backend, rgb=True)
writer.attach(rp)

def run_example():
    print(f"Capturing at original positions")
    rep.orchestrator.step(rt_subframes=8)

    print("Randomizing small cube rotation (graph-based) and capturing...")
    rep.utils.send_og_event(event_name="randomize_small_cube")
    rep.orchestrator.step(rt_subframes=8)

    print("Moving small cube position (USD API) and capturing...")
    small_cube.GetAttribute("xformOp:translate").Set((-1.5, 1.5, -2))
    rep.orchestrator.step(rt_subframes=8)

    print("Randomizing large cube rotation (graph-based) and capturing...")
    rep.utils.send_og_event(event_name="randomize_large_cube")
    rep.orchestrator.step(rt_subframes=8)

    print("Moving large cube position (USD API) and capturing...")
    large_cube.GetAttribute("xformOp:translate").Set((1.5, -1.5, 2))
    rep.orchestrator.step(rt_subframes=8)

    # Wait until all the data is saved to disk and cleanup writer and render product
    rep.orchestrator.wait_until_complete()
    writer.detach()
    rp.destroy()

run_example()

simulation_app.close()
```

## Motion Blur

This example demonstrates how to capture motion blur data using [RTX Real-Time](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_rt.html) and [RTX Interactive (Path Tracing)](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html) rendering modes. For the RTX - Real-Time mode, refer to [motion blur parameters](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx_post-processing.html#motion-blur). For the RTX – Interactive (Path Tracing) mode, motion blur is achieved by rendering multiple subframes (`/omni/replicator/pathTracedMotionBlurSubSamples`) and combining them to create the effect.

The example uses animated and physics-enabled assets with synchronized motion. Keyframe animated assets can be advanced at any custom delta time due to their interpolated motion, whereas physics-enabled assets require a custom physics FPS to ensure motion samples at any custom delta time. The example showcases how to compute the target physics FPS, change it if needed, and restore the original physics FPS after capturing the motion blur.

The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/motion_blur.py
```

Script Editor

Motion Blur

```python
import asyncio
import os

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, UsdPhysics

# Paths to the animated and physics-ready assets
PHYSICS_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
ANIM_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"

# -z velocities and start locations of the animated (left side) and physics (right side) assets (stage units/s)
ASSET_VELOCITIES = [0, 5, 10]
ASSET_X_MIRRORED_LOCATIONS = [(0.5, 0, 0.3), (0.3, 0, 0.3), (0.1, 0, 0.3)]

# Used to calculate how many frames to animate the assets to maintain the same velocity as the physics assets
ANIMATION_DURATION = 10

# Number of frames to capture for each scenario
NUM_FRAMES = 3

# Configuration for motion blur examples
DELTA_TIMES = [None, 1 / 30, 1 / 60, 1 / 240]
SAMPLES_PER_PIXEL = [32, 128]
MOTION_BLUR_SUBSAMPLES = [4, 16]

def setup_stage():
    """Create a new USD stage with animated and physics-enabled assets with synchronized motion."""
    omni.usd.get_context().new_stage()
    settings = carb.settings.get_settings()
    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    settings.set("rtx/post/dlss/execMode", 2)

    # Capture data only on request
    rep.orchestrator.set_capture_on_play(False)

    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(ANIMATION_DURATION)

    # Create lights
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=100, parent="/World", name="DomeLight")
    rep.functional.create.distant_light(intensity=2500, rotation=(315, 0, 0), parent="/World", name="DistantLight")

    # Setup the physics assets with gravity disabled and the requested velocity
    assets_root_path = get_assets_root_path()
    physics_asset_url = assets_root_path + PHYSICS_ASSET_URL
    for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
        prim = rep.functional.create.reference(
            usd_path=physics_asset_url,
            parent="/World",
            name=f"physics_asset_{int(abs(velocity))}",
            position=location,
        )
        physics_rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        physics_rigid_body_api.GetVelocityAttr().Set((0, 0, -velocity))
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(True)
        physx_rigid_body_api.GetAngularDampingAttr().Set(0.0)
        physx_rigid_body_api.GetLinearDampingAttr().Set(0.0)

    # Setup animated assets maintaining the same velocity as the physics assets
    anim_asset_url = assets_root_path + ANIM_ASSET_URL
    for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
        start_location = (-location[0], location[1], location[2])
        prim = rep.functional.create.reference(
            usd_path=anim_asset_url,
            parent="/World",
            name=f"anim_asset_{int(abs(velocity))}",
            position=start_location,
        )
        animation_distance = velocity * ANIMATION_DURATION
        end_location = (start_location[0], start_location[1], start_location[2] - animation_distance)
        end_keyframe_time = timeline.get_time_codes_per_seconds() * ANIMATION_DURATION
        # Timesampled keyframe (animated) translation
        prim.GetAttribute("xformOp:translate").Set(start_location, time=0)
        prim.GetAttribute("xformOp:translate").Set(end_location, time=end_keyframe_time)

async def run_motion_blur_example_async(
    num_frames=NUM_FRAMES, delta_time=None, use_path_tracing=True, motion_blur_subsamples=8, samples_per_pixel=64
):
    """Capture motion blur frames with the given delta time step and render mode."""
    setup_stage()
    stage = omni.usd.get_context().get_stage()
    settings = carb.settings.get_settings()

    # Enable motion blur capture
    settings.set("/omni/replicator/captureMotionBlur", True)

    # Set motion blur settings based on the render mode
    if use_path_tracing:
        print("[MotionBlur] Setting PathTracing render mode motion blur settings")
        settings.set("/rtx/rendermode", "PathTracing")
        # (int): Total number of samples for each rendered pixel, per frame.
        settings.set("/rtx/pathtracing/spp", samples_per_pixel)
        # (int): Maximum number of samples to accumulate per pixel. When this count is reached the rendering stops until a scene or setting change is detected, restarting the rendering process. Set to 0 to remove this limit.
        settings.set("/rtx/pathtracing/totalSpp", samples_per_pixel)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)
        # Number of sub samples to render if in PathTracing render mode and motion blur is enabled.
        settings.set("/omni/replicator/pathTracedMotionBlurSubSamples", motion_blur_subsamples)
    else:
        print("[MotionBlur] Setting RealTimePathTracing render mode motion blur settings")
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        # 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
        settings.set("/rtx/post/aa/op", 2)
        # (float): The fraction of the largest screen dimension to use as the maximum motion blur diameter.
        settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.02)
        # (float): Exposure time fraction in frames (1.0 = one frame duration) to sample.
        settings.set("/rtx/post/motionblur/exposureFraction", 1.0)
        # (int): Number of samples to use in the filter. A higher number improves quality at the cost of performance.
        settings.set("/rtx/post/motionblur/numSamples", 8)

    # Setup backend
    mode_str = f"pt_subsamples_{motion_blur_subsamples}_spp_{samples_per_pixel}" if use_path_tracing else "rt"
    delta_time_str = "None" if delta_time is None else f"{delta_time:.4f}"
    output_directory = os.path.join(os.getcwd(), f"_out_motion_blur_func_dt_{delta_time_str}_{mode_str}")
    print(f"[MotionBlur] Output directory: {output_directory}")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_directory)

    # Setup writer and render product
    camera = rep.functional.create.camera(
        position=(0, 1.5, 0), look_at=(0, 0, 0), parent="/World", name="MotionBlurCam"
    )
    render_product = rep.create.render_product(camera, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True)
    writer.attach(render_product)

    # Run a few updates to make sure all materials are fully loaded for capture
    for _ in range(5):
        await omni.kit.app.get_app().next_update_async()

    # Create or get the physics scene
    rep.functional.physics.create_physics_scene(path="/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

    # Check the target physics depending on the delta time and the render mode
    target_physics_fps = stage.GetTimeCodesPerSecond() if delta_time is None else 1 / delta_time
    if use_path_tracing:
        target_physics_fps *= motion_blur_subsamples

    # Check if the physics FPS needs to be increased to match the delta time
    original_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    if target_physics_fps > original_physics_fps:
        print(f"[MotionBlur] Changing physics FPS from {original_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Start the timeline for physics updates in the step function
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Capture frames
    for i in range(num_frames):
        print(f"[MotionBlur] \tCapturing frame {i}")
        await rep.orchestrator.step_async(delta_time=delta_time)

    # Restore the original physics FPS
    if target_physics_fps > original_physics_fps:
        print(f"[MotionBlur] Restoring physics FPS from {target_physics_fps} to {original_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(original_physics_fps)

    # Switch back to the raytracing render mode
    if use_path_tracing:
        print("[MotionBlur] Restoring render mode to RealTimePathTracing")
        settings.set("/rtx/rendermode", "RealTimePathTracing")

    # Wait until the data is fully written
    await rep.orchestrator.wait_until_complete_async()

    # Cleanup
    writer.detach()
    render_product.destroy()

async def run_motion_blur_examples_async(num_frames, delta_times, samples_per_pixel, motion_blur_subsamples):
    print(
        f"[MotionBlur] Running with delta_times={delta_times}, samples_per_pixel={samples_per_pixel}, motion_blur_subsamples={motion_blur_subsamples}"
    )

    for delta_time in delta_times:
        # RayTracing examples
        await run_motion_blur_example_async(num_frames=num_frames, delta_time=delta_time, use_path_tracing=False)
        # PathTracing examples
        for motion_blur_subsample in motion_blur_subsamples:
            for samples_per_pixel_value in samples_per_pixel:
                await run_motion_blur_example_async(
                    num_frames=num_frames,
                    delta_time=delta_time,
                    use_path_tracing=True,
                    motion_blur_subsamples=motion_blur_subsample,
                    samples_per_pixel=samples_per_pixel_value,
                )

asyncio.ensure_future(
    run_motion_blur_examples_async(
        num_frames=NUM_FRAMES,
        delta_times=DELTA_TIMES,
        samples_per_pixel=SAMPLES_PER_PIXEL,
        motion_blur_subsamples=MOTION_BLUR_SUBSAMPLES,
    )
)

async def run_motion_blur_examples_async():
    motion_blur_step_duration = [None, 1 / 30, 1 / 60, 1 / 240]
    for custom_delta_time in motion_blur_step_duration:
        # RayTracing examples
        await run_motion_blur_example_async(delta_time=custom_delta_time, use_path_tracing=False)
        # PathTracing examples
        spps = [32, 128]
        motion_blur_sub_samples = [4, 16]
        for motion_blur_sub_sample in motion_blur_sub_samples:
            for spp in spps:
                await run_motion_blur_example_async(
                    delta_time=custom_delta_time,
                    use_path_tracing=True,
                    motion_blur_subsamples=motion_blur_sub_sample,
                    samples_per_pixel=spp,
                )

asyncio.ensure_future(run_motion_blur_examples_async())
```

Standalone Application

Motion Blur

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysxSchema, UsdPhysics

# Paths to the animated and physics-ready assets
PHYSICS_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd"
ANIM_ASSET_URL = "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd"

# -z velocities and start locations of the animated (left side) and physics (right side) assets (stage units/s)
ASSET_VELOCITIES = [0, 5, 10]
ASSET_X_MIRRORED_LOCATIONS = [(0.5, 0, 0.3), (0.3, 0, 0.3), (0.1, 0, 0.3)]

# Used to calculate how many frames to animate the assets to maintain the same velocity as the physics assets
ANIMATION_DURATION = 10

# Number of frames to capture for each scenario
NUM_FRAMES = 3

def parse_delta_time(value):
    """Convert string to float or None. Accepts 'None', -1, 0, or numeric values."""
    if value.lower() == "none":
        return None
    float_value = float(value)
    return None if float_value in (-1, 0) else float_value

parser = argparse.ArgumentParser()
parser.add_argument(
    "--delta_times",
    nargs="*",
    type=parse_delta_time,
    default=[None, 1 / 30, 1 / 240],
    help="List of delta times (seconds per frame) to use for motion blur captures. Use 'None' for default stage time.",
)
parser.add_argument(
    "--samples_per_pixel",
    nargs="*",
    type=int,
    default=[32, 128],
    help="List of samples per pixel (spp) values for path tracing",
)
parser.add_argument(
    "--motion_blur_subsamples",
    nargs="*",
    type=int,
    default=[4, 16],
    help="List of motion blur subsample values for path tracing",
)
args, _ = parser.parse_known_args()
delta_times = args.delta_times
samples_per_pixel = args.samples_per_pixel
motion_blur_subsamples = args.motion_blur_subsamples

def setup_stage():
    """Create a new USD stage with animated and physics-enabled assets with synchronized motion."""
    omni.usd.get_context().new_stage()
    settings = carb.settings.get_settings()
    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    settings.set("rtx/post/dlss/execMode", 2)

    # Capture data only on request
    rep.orchestrator.set_capture_on_play(False)

    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_end_time(ANIMATION_DURATION)

    # Create lights
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=100, parent="/World", name="DomeLight")
    rep.functional.create.distant_light(intensity=2500, rotation=(315, 0, 0), parent="/World", name="DistantLight")

    # Setup the physics assets with gravity disabled and the requested velocity
    assets_root_path = get_assets_root_path()
    physics_asset_url = assets_root_path + PHYSICS_ASSET_URL
    for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
        prim = rep.functional.create.reference(
            usd_path=physics_asset_url, parent="/World", name=f"physics_asset_{int(abs(velocity))}", position=location
        )
        physics_rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        physics_rigid_body_api.GetVelocityAttr().Set((0, 0, -velocity))
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(True)
        physx_rigid_body_api.GetAngularDampingAttr().Set(0.0)
        physx_rigid_body_api.GetLinearDampingAttr().Set(0.0)

    # Setup animated assets maintaining the same velocity as the physics assets
    anim_asset_url = assets_root_path + ANIM_ASSET_URL
    for location, velocity in zip(ASSET_X_MIRRORED_LOCATIONS, ASSET_VELOCITIES):
        start_location = (-location[0], location[1], location[2])
        prim = rep.functional.create.reference(
            usd_path=anim_asset_url, parent="/World", name=f"anim_asset_{int(abs(velocity))}", position=start_location
        )
        animation_distance = velocity * ANIMATION_DURATION
        end_location = (start_location[0], start_location[1], start_location[2] - animation_distance)
        end_keyframe_time = timeline.get_time_codes_per_seconds() * ANIMATION_DURATION
        # Timesampled keyframe (animated) translation
        prim.GetAttribute("xformOp:translate").Set(start_location, time=0)
        prim.GetAttribute("xformOp:translate").Set(end_location, time=end_keyframe_time)

def run_motion_blur_example(
    num_frames, delta_time=None, use_path_tracing=True, motion_blur_subsamples=8, samples_per_pixel=64
):
    """Capture motion blur frames with the given delta time step and render mode."""
    setup_stage()
    stage = omni.usd.get_context().get_stage()
    settings = carb.settings.get_settings()

    # Enable motion blur capture
    settings.set("/omni/replicator/captureMotionBlur", True)

    # Set motion blur settings based on the render mode
    if use_path_tracing:
        print("[MotionBlur] Setting PathTracing render mode motion blur settings")
        settings.set("/rtx/rendermode", "PathTracing")
        # (int): Total number of samples for each rendered pixel, per frame.
        settings.set("/rtx/pathtracing/spp", samples_per_pixel)
        # (int): Maximum number of samples to accumulate per pixel. When this count is reached the rendering stops until a scene or setting change is detected, restarting the rendering process. Set to 0 to remove this limit.
        settings.set("/rtx/pathtracing/totalSpp", samples_per_pixel)
        settings.set("/rtx/pathtracing/optixDenoiser/enabled", 0)
        # Number of sub samples to render if in PathTracing render mode and motion blur is enabled.
        settings.set("/omni/replicator/pathTracedMotionBlurSubSamples", motion_blur_subsamples)
    else:
        print("[MotionBlur] Setting RealTimePathTracing render mode motion blur settings")
        settings.set("/rtx/rendermode", "RealTimePathTracing")
        # 0: Disabled, 1: TAA, 2: FXAA, 3: DLSS, 4:RTXAA
        settings.set("/rtx/post/aa/op", 2)
        # (float): The fraction of the largest screen dimension to use as the maximum motion blur diameter.
        settings.set("/rtx/post/motionblur/maxBlurDiameterFraction", 0.02)
        # (float): Exposure time fraction in frames (1.0 = one frame duration) to sample.
        settings.set("/rtx/post/motionblur/exposureFraction", 1.0)
        # (int): Number of samples to use in the filter. A higher number improves quality at the cost of performance.
        settings.set("/rtx/post/motionblur/numSamples", 8)

    # Setup backend
    mode_str = f"pt_subsamples_{motion_blur_subsamples}_spp_{samples_per_pixel}" if use_path_tracing else "rt"
    delta_time_str = "None" if delta_time is None else f"{delta_time:.4f}"
    output_directory = os.path.join(os.getcwd(), f"_out_motion_blur_func_dt_{delta_time_str}_{mode_str}")
    print(f"[MotionBlur] Output directory: {output_directory}")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=output_directory)

    # Setup writer and render product
    camera = rep.functional.create.camera(
        position=(0, 1.5, 0), look_at=(0, 0, 0), parent="/World", name="MotionBlurCam"
    )
    render_product = rep.create.render_product(camera, (1280, 720))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True)
    writer.attach(render_product)

    # Run a few updates to make sure all materials are fully loaded for capture
    for _ in range(5):
        simulation_app.update()

    # Create or get the physics scene
    rep.functional.physics.create_physics_scene(path="/PhysicsScene")
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

    # Check the target physics depending on the delta time and the render mode
    target_physics_fps = stage.GetTimeCodesPerSecond() if delta_time is None else 1 / delta_time
    if use_path_tracing:
        target_physics_fps *= motion_blur_subsamples

    # Check if the physics FPS needs to be increased to match the delta time
    original_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
    if target_physics_fps > original_physics_fps:
        print(f"[MotionBlur] Changing physics FPS from {original_physics_fps} to {target_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

    # Start the timeline for physics updates in the step function
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Capture frames
    for i in range(num_frames):
        print(f"[MotionBlur] \tCapturing frame {i}")
        rep.orchestrator.step(delta_time=delta_time)

    # Restore the original physics FPS
    if target_physics_fps > original_physics_fps:
        print(f"[MotionBlur] Restoring physics FPS from {target_physics_fps} to {original_physics_fps}")
        physx_scene.GetTimeStepsPerSecondAttr().Set(original_physics_fps)

    # Switch back to the raytracing render mode
    if use_path_tracing:
        print("[MotionBlur] Restoring render mode to RealTimePathTracing")
        settings.set("/rtx/rendermode", "RealTimePathTracing")

    # Wait until the data is fully written
    rep.orchestrator.wait_until_complete()

    # Cleanup
    writer.detach()
    render_product.destroy()

def run_motion_blur_examples(num_frames, delta_times, samples_per_pixel, motion_blur_subsamples):
    print(
        f"[MotionBlur] Running with delta_times={delta_times}, samples_per_pixel={samples_per_pixel}, motion_blur_subsamples={motion_blur_subsamples}"
    )
    for delta_time in delta_times:
        # RayTracing examples
        run_motion_blur_example(num_frames=num_frames, delta_time=delta_time, use_path_tracing=False)
        # PathTracing examples
        for motion_blur_subsample in motion_blur_subsamples:
            for samples_per_pixel_value in samples_per_pixel:
                run_motion_blur_example(
                    num_frames=num_frames,
                    delta_time=delta_time,
                    use_path_tracing=True,
                    motion_blur_subsamples=motion_blur_subsample,
                    samples_per_pixel=samples_per_pixel_value,
                )

run_motion_blur_examples(
    num_frames=NUM_FRAMES,
    delta_times=delta_times,
    samples_per_pixel=samples_per_pixel,
    motion_blur_subsamples=motion_blur_subsamples,
)

simulation_app.close()
```

## Subscribers and Events at Custom FPS

Examples of subscribing to various events (such as stage, physics, and render/app), setting custom update rates, and adjusting various related settings. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/subscribers_and_events.py
```

Script Editor

Subscribers and Events at Custom FPS

```python
import asyncio
import time

import carb.eventdispatcher
import carb.settings
import omni.kit.app
import omni.physics.core
import omni.timeline
import omni.usd
from pxr import PhysxSchema, UsdPhysics

# TIMELINE / STAGE
USE_CUSTOM_TIMELINE_SETTINGS = True
USE_FIXED_TIME_STEPPING = True
PLAY_EVERY_FRAME = True
PLAY_DELAY_COMPENSATION = 0.0
SUBSAMPLE_RATE = 1
STAGE_FPS = 30.0

# PHYSX
USE_CUSTOM_PHYSX_FPS = False
PHYSX_FPS = 60.0
MIN_SIM_FPS = 30

# Simulations can also be enabled/disabled at runtime
DISABLE_SIMULATIONS = False

# APP / RENDER
LIMIT_APP_FPS = False
APP_FPS = 120

# Number of app updates to run while collecting events
NUM_APP_UPDATES = 100

# Print the captured events
VERBOSE = False

async def run_subscribers_and_events_async():
    def on_timeline_event(event: carb.eventdispatcher.Event):
        nonlocal timeline_events
        timeline_events.append(event)
        if VERBOSE:
            print(f"  [timeline][{len(timeline_events)}] {event}")

    def on_physics_step(dt: float, context):
        nonlocal physics_events
        physics_events.append(dt)
        if VERBOSE:
            print(f"  [physics][{len(physics_events)}] dt={dt}")

    def on_stage_render_event(event: carb.eventdispatcher.Event):
        nonlocal stage_render_events
        stage_render_events.append(event.event_name)
        if VERBOSE:
            print(f"  [stage render][{len(stage_render_events)}] {event.event_name}")

    def on_app_update(event: carb.eventdispatcher.Event):
        nonlocal app_update_events
        app_update_events.append(event.event_name)
        if VERBOSE:
            print(f"  [app update][{len(app_update_events)}] {event.event_name}")

    stage = omni.usd.get_context().get_stage()
    timeline = omni.timeline.get_timeline_interface()

    if USE_CUSTOM_TIMELINE_SETTINGS:
        # Ideal to make simulation and animation synchronized.
        # Default: True in editor, False in standalone.
        # NOTE:
        # - It may limit the frame rate (see 'timeline.set_play_every_frame') such that the elapsed wall clock time matches the frame's delta time.
        # - If the app runs slower than this, animation playback may slow down (see 'CompensatePlayDelayInSecs').
        # - For performance benchmarks, turn this off or set a very high target in `timeline.set_target_framerate`
        carb.settings.get_settings().set("/app/player/useFixedTimeStepping", USE_FIXED_TIME_STEPPING)

        # This compensates for frames that require more computation time than the frame's fixed delta time, by temporarily speeding up playback.
        # The parameter represents the length of these "faster" playback periods, which means that it must be larger than the fixed frame time to take effect.
        # Default: 0.0
        # NOTE:
        # - only effective if `useFixedTimeStepping` is set to True
        # - setting a large value results in long fast playback after a huge lag spike
        carb.settings.get_settings().set("/app/player/CompensatePlayDelayInSecs", PLAY_DELAY_COMPENSATION)

        # If set to True, no frames are skipped and in every frame time advances by `1 / TimeCodesPerSecond`.
        # Default: False
        # NOTE:
        # - only effective if `useFixedTimeStepping` is set to True
        # - simulation is usually faster than real-time and processing is only limited by the frame rate of the runloop
        # - useful for recording
        # - same as `carb.settings.get_settings().set("/app/player/useFastMode", PLAY_EVERY_FRAME)`
        timeline.set_play_every_frame(PLAY_EVERY_FRAME)

        # Timeline sub-stepping, i.e. how many times updates are called (update events are dispatched) each frame.
        # Default: 1
        # NOTE: same as `carb.settings.get_settings().set("/app/player/timelineSubsampleRate", SUBSAMPLE_RATE)`
        timeline.set_ticks_per_frame(SUBSAMPLE_RATE)

        # Time codes per second for the stage
        # NOTE: same as `stage.SetTimeCodesPerSecond(STAGE_FPS)` and `carb.settings.get_settings().set("/app/stage/timeCodesPerSecond", STAGE_FPS)`
        timeline.set_time_codes_per_second(STAGE_FPS)

    # Create a PhysX scene to set the physics time step
    if USE_CUSTOM_PHYSX_FPS:
        physx_scene = None
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
                break
        if physx_scene is None:
            UsdPhysics.Scene.Define(stage, "/PhysicsScene")
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

        # Time step for the physics simulation
        # Default: 60.0
        physx_scene.GetTimeStepsPerSecondAttr().Set(PHYSX_FPS)

        # Minimum simulation frequency to prevent clamping; if the frame rate drops below this,
        # physics steps are discarded to avoid app slowdown if the overall frame rate is too low.
        # Default: 30.0
        # NOTE: Matching `minFrameRate` with `TimeStepsPerSecond` ensures a single physics step per update.
        carb.settings.get_settings().set("/persistent/simulation/minFrameRate", MIN_SIM_FPS)

    # Throttle Render/UI/Main thread update rate
    if LIMIT_APP_FPS:
        # Enable rate limiting of the main run loop (UI, rendering, etc.)
        # Default: False
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", LIMIT_APP_FPS)

        # FPS limit of the main run loop (UI, rendering, etc.)
        # Default: 120
        # NOTE: disabled if `/app/player/useFixedTimeStepping` is False
        carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", int(APP_FPS))

    # Simulations can be selectively disabled (or toggled at specific times)
    if DISABLE_SIMULATIONS:
        carb.settings.get_settings().set("/app/player/playSimulations", False)

    print("Configuration:")
    print(f"  Timeline:")
    print(f"    - Stage FPS: {STAGE_FPS}  (/app/stage/timeCodesPerSecond)")
    print(f"    - Fixed time stepping: {USE_FIXED_TIME_STEPPING}  (/app/player/useFixedTimeStepping)")
    print(f"    - Play every frame: {PLAY_EVERY_FRAME}  (/app/player/useFastMode)")
    print(f"    - Subsample rate: {SUBSAMPLE_RATE}  (/app/player/timelineSubsampleRate)")
    print(f"    - Play delay compensation: {PLAY_DELAY_COMPENSATION}s  (/app/player/CompensatePlayDelayInSecs)")
    print(f"  Physics:")
    print(f"    - PhysX FPS: {PHYSX_FPS}  (physxScene.timeStepsPerSecond)")
    print(f"    - Min simulation FPS: {MIN_SIM_FPS}  (/persistent/simulation/minFrameRate)")
    print(f"    - Simulations enabled: {not DISABLE_SIMULATIONS}  (/app/player/playSimulations)")
    print(f"  Rendering:")
    print(f"    - App FPS limit: {APP_FPS if LIMIT_APP_FPS else 'unlimited'}  (/app/runLoops/main/rateLimitFrequency)")

    # Start the timeline
    print(f"Starting the timeline...")
    timeline.set_current_time(0)
    timeline.set_end_time(10000)
    timeline.set_looping(False)
    timeline.play()
    timeline.commit()
    wall_start_time = time.time()

    # Subscribe to events
    print(f"Subscribing to events...")
    timeline_events = []
    timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
        on_event=on_timeline_event,
        observer_name="test_sdg_useful_snippets_timeline_based.on_timeline_event",
    )
    physics_events = []
    physics_sub = omni.physics.core.get_physics_simulation_interface().subscribe_physics_on_step_events(
        pre_step=False, order=0, on_update=on_physics_step
    )
    stage_render_events = []
    stage_render_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.usd.get_context().stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME, True),
        on_event=on_stage_render_event,
        observer_name="subscribers_and_events.on_stage_render_event",
    )
    app_update_events = []
    app_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
        event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
        on_event=on_app_update,
        observer_name="subscribers_and_events.on_app_update",
    )

    # Run app updates and cache events
    print(f"Starting running the application for {NUM_APP_UPDATES} updates...")
    for i in range(NUM_APP_UPDATES):
        if VERBOSE:
            print(f"[app update loop][{i+1}/{NUM_APP_UPDATES}]")
        await omni.kit.app.get_app().next_update_async()
    elapsed_wall_time = time.time() - wall_start_time
    print(f"Finished running the application for {NUM_APP_UPDATES} updates...")

    # Stop timeline and unsubscribe from all events
    print(f"Stopping timeline and unsubscribing from all events...")
    timeline.stop()
    if app_sub:
        app_sub.reset()
        app_sub = None
    if stage_render_sub:
        stage_render_sub.reset()
        stage_render_sub = None
    if physics_sub:
        physics_sub.unsubscribe()
        physics_sub = None
    if timeline_sub:
        timeline_sub.reset()
        timeline_sub = None

    # Print summary statistics
    print("\nStats:")
    print(f"- App updates: {NUM_APP_UPDATES}")
    print(f"- Wall time: {elapsed_wall_time:.4f} seconds")
    print(f"- Timeline events: {len(timeline_events)}")
    print(f"- Physics events: {len(physics_events)}")
    print(f"- Stage render events: {len(stage_render_events)}")
    print(f"- App update events: {len(app_update_events)}")

    # Calculate and display real-time performance factor
    if len(physics_events) > 0:
        sim_time = sum(physics_events)
        realtime_factor = sim_time / elapsed_wall_time if elapsed_wall_time > 0 else 0
        print(f"- Simulation time: {sim_time:.4f}s")
        print(f"- Real-time factor: {realtime_factor:.2f}x")

asyncio.ensure_future(run_subscribers_and_events_async())
```

Standalone Application

Subscribers and Events at Custom FPS

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import time

import carb.eventdispatcher
import carb.settings
import omni.kit.app
import omni.physics.core
import omni.timeline
import omni.usd
from pxr import PhysxSchema, UsdPhysics

# TIMELINE / STAGE
USE_CUSTOM_TIMELINE_SETTINGS = True
USE_FIXED_TIME_STEPPING = True
PLAY_EVERY_FRAME = True
PLAY_DELAY_COMPENSATION = 0.0
SUBSAMPLE_RATE = 1
STAGE_FPS = 30.0

# PHYSX
USE_CUSTOM_PHYSX_FPS = False
PHYSX_FPS = 60.0
MIN_SIM_FPS = 30

# Simulations can also be enabled/disabled at runtime
DISABLE_SIMULATIONS = False

# APP / RENDER
LIMIT_APP_FPS = False
APP_FPS = 120

# Number of app updates to run while collecting events
NUM_APP_UPDATES = 100

# Print the captured events
VERBOSE = False

def on_timeline_event(event: carb.eventdispatcher.Event):
    global timeline_events
    timeline_events.append(event)
    if VERBOSE:
        print(f"  [timeline][{len(timeline_events)}] {event}")

def on_physics_step(dt, context):
    global physics_events
    physics_events.append(dt)
    if VERBOSE:
        print(f"  [physics][{len(physics_events)}] dt={dt}")

def on_stage_render_event(event: carb.eventdispatcher.Event):
    global stage_render_events
    stage_render_events.append(event.event_name)
    if VERBOSE:
        print(f"  [stage render][{len(stage_render_events)}] {event.event_name}")

def on_app_update(event: carb.eventdispatcher.Event):
    global app_update_events
    app_update_events.append(event.event_name)
    if VERBOSE:
        print(f"  [app update][{len(app_update_events)}] {event.event_name}")

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

if USE_CUSTOM_TIMELINE_SETTINGS:
    # Ideal to make simulation and animation synchronized.
    # Default: True in editor, False in standalone.
    # NOTE:
    # - It may limit the frame rate (see 'timeline.set_play_every_frame') such that the elapsed wall clock time matches the frame's delta time.
    # - If the app runs slower than this, animation playback may slow down (see 'CompensatePlayDelayInSecs').
    # - For performance benchmarks, turn this off or set a very high target in `timeline.set_target_framerate`
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", USE_FIXED_TIME_STEPPING)

    # This compensates for frames that require more computation time than the frame's fixed delta time, by temporarily speeding up playback.
    # The parameter represents the length of these "faster" playback periods, which means that it must be larger than the fixed frame time to take effect.
    # Default: 0.0
    # NOTE:
    # - only effective if `useFixedTimeStepping` is set to True
    # - setting a large value results in long fast playback after a huge lag spike
    carb.settings.get_settings().set("/app/player/CompensatePlayDelayInSecs", PLAY_DELAY_COMPENSATION)

    # If set to True, no frames are skipped and in every frame time advances by `1 / TimeCodesPerSecond`.
    # Default: False
    # NOTE:
    # - only effective if `useFixedTimeStepping` is set to True
    # - simulation is usually faster than real-time and processing is only limited by the frame rate of the runloop
    # - useful for recording
    # - same as `carb.settings.get_settings().set("/app/player/useFastMode", PLAY_EVERY_FRAME)`
    timeline.set_play_every_frame(PLAY_EVERY_FRAME)

    # Timeline sub-stepping, i.e. how many times updates are called (update events are dispatched) each frame.
    # Default: 1
    # NOTE: same as `carb.settings.get_settings().set("/app/player/timelineSubsampleRate", SUBSAMPLE_RATE)`
    timeline.set_ticks_per_frame(SUBSAMPLE_RATE)

    # Time codes per second for the stage
    # NOTE: same as `stage.SetTimeCodesPerSecond(STAGE_FPS)` and `carb.settings.get_settings().set("/app/stage/timeCodesPerSecond", STAGE_FPS)`
    timeline.set_time_codes_per_second(STAGE_FPS)

# Create a PhysX scene to set the physics time step
if USE_CUSTOM_PHYSX_FPS:
    physx_scene = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(prim)
            break
    if physx_scene is None:
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))

    # Time step for the physics simulation
    # Default: 60.0
    physx_scene.GetTimeStepsPerSecondAttr().Set(PHYSX_FPS)

    # Minimum simulation frequency to prevent clamping; if the frame rate drops below this,
    # physics steps are discarded to avoid app slowdown if the overall frame rate is too low.
    # Default: 30.0
    # NOTE: Matching `minFrameRate` with `TimeStepsPerSecond` ensures a single physics step per update.
    carb.settings.get_settings().set("/persistent/simulation/minFrameRate", MIN_SIM_FPS)

# Throttle Render/UI/Main thread update rate
if LIMIT_APP_FPS:
    # Enable rate limiting of the main run loop (UI, rendering, etc.)
    # Default: False
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitEnabled", LIMIT_APP_FPS)

    # FPS limit of the main run loop (UI, rendering, etc.)
    # Default: 120
    # NOTE: disabled if `/app/player/useFixedTimeStepping` is False
    carb.settings.get_settings().set("/app/runLoops/main/rateLimitFrequency", int(APP_FPS))

# Simulations can be selectively disabled (or toggled at specific times)
if DISABLE_SIMULATIONS:
    carb.settings.get_settings().set("/app/player/playSimulations", False)

print("Configuration:")
print(f"  Timeline:")
print(f"    - Stage FPS: {STAGE_FPS}  (/app/stage/timeCodesPerSecond)")
print(f"    - Fixed time stepping: {USE_FIXED_TIME_STEPPING}  (/app/player/useFixedTimeStepping)")
print(f"    - Play every frame: {PLAY_EVERY_FRAME}  (/app/player/useFastMode)")
print(f"    - Subsample rate: {SUBSAMPLE_RATE}  (/app/player/timelineSubsampleRate)")
print(f"    - Play delay compensation: {PLAY_DELAY_COMPENSATION}s  (/app/player/CompensatePlayDelayInSecs)")
print(f"  Physics:")
print(f"    - PhysX FPS: {PHYSX_FPS}  (physxScene.timeStepsPerSecond)")
print(f"    - Min simulation FPS: {MIN_SIM_FPS}  (/persistent/simulation/minFrameRate)")
print(f"    - Simulations enabled: {not DISABLE_SIMULATIONS}  (/app/player/playSimulations)")
print(f"  Rendering:")
print(f"    - App FPS limit: {APP_FPS if LIMIT_APP_FPS else 'unlimited'}  (/app/runLoops/main/rateLimitFrequency)")

# Start the timeline
print(f"Starting the timeline...")
timeline.set_current_time(0)
timeline.set_end_time(10000)
timeline.set_looping(False)
timeline.play()
timeline.commit()
wall_start_time = time.time()

# Subscribe to events
print(f"Subscribing to events...")
timeline_events = []
timeline_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
    event_name=omni.timeline.GLOBAL_EVENT_CURRENT_TIME_TICKED,
    on_event=on_timeline_event,
    observer_name="subscribers_and_events.on_timeline_event",
)
physics_events = []
physics_sub = omni.physics.core.get_physics_simulation_interface().subscribe_physics_on_step_events(
    pre_step=False, order=0, on_update=on_physics_step
)
stage_render_events = []
stage_render_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
    event_name=omni.usd.get_context().stage_rendering_event_name(omni.usd.StageRenderingEventType.NEW_FRAME, True),
    on_event=on_stage_render_event,
    observer_name="subscribers_and_events.on_stage_render_event",
)
app_update_events = []
app_sub = carb.eventdispatcher.get_eventdispatcher().observe_event(
    event_name=omni.kit.app.GLOBAL_EVENT_UPDATE,
    on_event=on_app_update,
    observer_name="subscribers_and_events.on_app_update",
)

# Run app updates and cache events
print(f"Starting running the application for {NUM_APP_UPDATES} updates.")
for i in range(NUM_APP_UPDATES):
    if VERBOSE:
        print(f"[app update loop][{i+1}/{NUM_APP_UPDATES}]")
    simulation_app.update()
elapsed_wall_time = time.time() - wall_start_time
print(f"Finished running the application for {NUM_APP_UPDATES} updates...")

# Stop timeline and unsubscribe from all events
timeline.stop()
if app_sub:
    app_sub.reset()
    app_sub = None
if stage_render_sub:
    stage_render_sub.reset()
    stage_render_sub = None
if physics_sub:
    physics_sub.unsubscribe()
    physics_sub = None
if timeline_sub:
    timeline_sub.reset()
    timeline_sub = None

# Print summary statistics
print("\nStats:")
print(f"- App updates: {NUM_APP_UPDATES}")
print(f"- Wall time: {elapsed_wall_time:.4f} seconds")
print(f"- Timeline events: {len(timeline_events)}")
print(f"- Physics events: {len(physics_events)}")
print(f"- Stage render events: {len(stage_render_events)}")
print(f"- App update events: {len(app_update_events)}")

# Calculate and display real-time performance factor
if len(physics_events) > 0:
    sim_time = sum(physics_events)
    realtime_factor = sim_time / elapsed_wall_time if elapsed_wall_time > 0 else 0
    print(f"- Simulation time: {sim_time:.4f}s")
    print(f"- Real-time factor: {realtime_factor:.2f}x")

simulation_app.close()
```

## Accessing Writer and Annotator Data at Custom FPS

Example of how to trigger a writer and access annotator data at a custom FPS, with product rendering disabled when the data is not needed. The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/custom_fps_writer_annotator.py
```

Note

It is currently not possible to change timeline (stage) FPS after the replicator graph creation as it causes a graph reset. This issue is being addressed. As a workaround make sure you are setting the timeline (stage) parameters before creating the replicator graph.

Script Editor

Accessing Writer and Annotator Data at Custom FPS

```python
import asyncio
import os

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd

# Configuration
NUM_CAPTURES = 6
VERBOSE = True

# NOTE: To avoid FPS delta misses make sure the sensor framerate is divisible by the timeline framerate
STAGE_FPS = 100.0
SENSOR_FPS = 10.0
SENSOR_DT = 1.0 / SENSOR_FPS

async def run_custom_fps_example_async(duration_seconds):
    # Create a new stage
    await omni.usd.get_context().new_stage_async()

    # Disable capture on play to capture data manually using step
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Make sure fixed time stepping is set (the timeline will be advanced with the same delta time)
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)

    # Create scene with a semantically annotated cube with physics
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=250, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(position=(0, 0, 2), parent="/World", name="Cube", semantics={"class": "cube"})
    rep.functional.physics.apply_collider(cube)
    rep.functional.physics.apply_rigid_body(cube)

    # Create render product (disabled until data capture is needed)
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, resolution=(512, 512), name="rp")
    rp.hydra_texture.set_updates_enabled(False)

    # Create the backend for the writer
    out_dir_rgb = os.path.join(os.getcwd(), "_out_writer_fps_rgb")
    print(f"Writer data will be written to: {out_dir_rgb}")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out_dir_rgb)

    # Create a writer and an annotator as examples of different ways of accessing data
    writer_rgb = rep.WriterRegistry.get("BasicWriter")
    writer_rgb.initialize(backend=backend, rgb=True)
    writer_rgb.attach(rp)

    # Create an annotator to access the data directly
    annot_depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    annot_depth.attach(rp)

    # Run the simulation for the given number of frames and access the data at the desired framerates
    print(
        f"Starting simulation: {duration_seconds:.2f}s duration, {SENSOR_FPS:.0f} FPS sensor, {STAGE_FPS:.0f} FPS timeline"
    )

    # Set the timeline parameters
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    timeline.set_current_time(0.0)
    timeline.set_end_time(10)
    timeline.set_time_codes_per_second(STAGE_FPS)
    timeline.play()
    timeline.commit()

    # Run the simulation for the given number of frames and access the data at the desired framerates
    frame_count = 0
    previous_time = timeline.get_current_time()
    elapsed_time = 0.0
    iteration = 0

    while timeline.get_current_time() < duration_seconds:
        current_time = timeline.get_current_time()
        delta_time = current_time - previous_time
        elapsed_time += delta_time

        # Simulation progress
        if VERBOSE:
            print(f"Step {iteration}: timeline time={current_time:.3f}s, elapsed time={elapsed_time:.3f}s")

        # Trigger sensor at desired framerate (use small epsilon for floating point comparison)
        if elapsed_time >= SENSOR_DT - 1e-9:
            elapsed_time -= SENSOR_DT  # Reset with remainder to maintain accuracy

            rp.hydra_texture.set_updates_enabled(True)
            await rep.orchestrator.step_async(delta_time=0.0, pause_timeline=False, rt_subframes=16)
            annot_data = annot_depth.get_data()

            print(f"\n  >> Capturing frame {frame_count} at time={current_time:.3f}s | shape={annot_data.shape}\n")
            frame_count += 1

            rp.hydra_texture.set_updates_enabled(False)

        previous_time = current_time
        # Advance the app (timeline) by one frame
        await omni.kit.app.get_app().next_update_async()
        iteration += 1

    # Wait for writer to finish
    await rep.orchestrator.wait_until_complete_async()

    # Cleanup
    timeline.pause()
    writer_rgb.detach()
    annot_depth.detach()
    rp.destroy()

# Run example with duration for all captures plus a buffer of 5 frames
duration = (NUM_CAPTURES * SENSOR_DT) + (5.0 / STAGE_FPS)
asyncio.ensure_future(run_custom_fps_example_async(duration_seconds=duration))
```

Standalone Application

Accessing Writer and Annotator Data at Custom FPS

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os

import carb.settings
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd

# Configuration
NUM_CAPTURES = 6
VERBOSE = True

# NOTE: To avoid FPS delta misses make sure the sensor framerate is divisible by the timeline framerate
STAGE_FPS = 100.0
SENSOR_FPS = 10.0
SENSOR_DT = 1.0 / SENSOR_FPS

def run_custom_fps_example(duration_seconds):
    # Create a new stage
    omni.usd.get_context().new_stage()

    # Disable capture on play to capture data manually using step
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Make sure fixed time stepping is set (the timeline will be advanced with the same delta time)
    carb.settings.get_settings().set("/app/player/useFixedTimeStepping", True)

    # Create scene with a semantically annotated cube with physics
    rep.functional.create.xform(name="World")
    rep.functional.create.dome_light(intensity=250, parent="/World", name="DomeLight")
    cube = rep.functional.create.cube(position=(0, 0, 2), parent="/World", name="Cube", semantics={"class": "cube"})
    rep.functional.physics.apply_collider(cube)
    rep.functional.physics.apply_rigid_body(cube)

    # Create render product (disabled until data capture is needed)
    cam = rep.functional.create.camera(position=(5, 5, 5), look_at=(0, 0, 0), parent="/World", name="Camera")
    rp = rep.create.render_product(cam, resolution=(512, 512), name="rp")
    rp.hydra_texture.set_updates_enabled(False)

    # Create the backend for the writer
    out_dir_rgb = os.path.join(os.getcwd(), "_out_writer_fps_rgb")
    print(f"Writer data will be written to: {out_dir_rgb}")
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=out_dir_rgb)

    # Create a writer and an annotator as examples of different ways of accessing data
    writer_rgb = rep.WriterRegistry.get("BasicWriter")
    writer_rgb.initialize(backend=backend, rgb=True)
    writer_rgb.attach(rp)

    # Create an annotator to access the data directly
    annot_depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    annot_depth.attach(rp)

    # Run the simulation for the given number of frames and access the data at the desired framerates
    print(
        f"Starting simulation: {duration_seconds:.2f}s duration, {SENSOR_FPS:.0f} FPS sensor, {STAGE_FPS:.0f} FPS timeline"
    )

    # Set the timeline parameters
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_looping(False)
    timeline.set_current_time(0.0)
    timeline.set_end_time(10)
    timeline.set_time_codes_per_second(STAGE_FPS)
    timeline.play()
    timeline.commit()

    # Run the simulation for the given number of frames and access the data at the desired framerates
    frame_count = 0
    previous_time = timeline.get_current_time()
    elapsed_time = 0.0
    iteration = 0

    while timeline.get_current_time() < duration_seconds:
        current_time = timeline.get_current_time()
        delta_time = current_time - previous_time
        elapsed_time += delta_time

        # Simulation progress
        if VERBOSE:
            print(f"Step {iteration}: timeline time={current_time:.3f}s, elapsed time={elapsed_time:.3f}s")

        # Trigger sensor at desired framerate (use small epsilon for floating point comparison)
        if elapsed_time >= SENSOR_DT - 1e-9:
            elapsed_time -= SENSOR_DT  # Reset with remainder to maintain accuracy

            rp.hydra_texture.set_updates_enabled(True)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=16)
            annot_data = annot_depth.get_data()

            print(f"\n  >> Capturing frame {frame_count} at time={current_time:.3f}s | shape={annot_data.shape}\n")
            frame_count += 1

            rp.hydra_texture.set_updates_enabled(False)

        previous_time = current_time
        # Advance the app (timeline) by one frame
        simulation_app.update()
        iteration += 1

    # Wait for writer to finish
    rep.orchestrator.wait_until_complete()

    # Cleanup
    timeline.pause()
    writer_rgb.detach()
    annot_depth.detach()
    rp.destroy()

# Run example with duration for all captures plus a buffer of 5 frames
duration = (NUM_CAPTURES * SENSOR_DT) + (5.0 / STAGE_FPS)
run_custom_fps_example(duration_seconds=duration)

simulation_app.close()
```

## Cosmos Writer Example

This example demonstrates the `CosmosWriter` for capturing multi-modal synthetic data compatible with [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/) world foundation models. It creates a simple falling box scene and captures synchronized RGB, segmentation, depth, and edge data (images and videos) that can be used with Cosmos Transfer to generate photorealistic variations.

For a more detailed tutorial please see [Cosmos Synthetic Data Generation](Synthetic_Data_Generation.md).

The standalone example can also be run directly (on Windows use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.examples/cosmos_writer_simple.py
```

Script Editor

Cosmos Writer Example

```python
import asyncio
import os

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd

SEGMENTATION_MAPPING = {
    "plane": [0, 0, 255, 255],
    "cube": [255, 0, 0, 255],
    "sphere": [0, 255, 0, 255],
}
NUM_FRAMES = 60

async def run_cosmos_example_async(num_frames, segmentation_mapping=None):
    # Create a new stage
    omni.usd.get_context().new_stage()

    # CosmosWriter requires script nodes to be enabled
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Set the stage properties
    rep.settings.set_stage_up_axis("Z")
    rep.settings.set_stage_meters_per_unit(1.0)
    rep.functional.create.dome_light(intensity=500)

    # Create the scenario with a ground plane and a falling sphere and cube.
    plane = rep.functional.create.plane(position=(0, 0, 0), scale=(10, 10, 1), semantics={"class": "plane"})
    rep.functional.physics.apply_collider(plane)

    sphere = rep.functional.create.sphere(position=(0, 0, 3), semantics={"class": "sphere"})
    rep.functional.physics.apply_collider(sphere)
    rep.functional.physics.apply_rigid_body(sphere)

    cube = rep.functional.create.cube(position=(1, 1, 2), scale=0.5, semantics={"class": "cube"})
    rep.functional.physics.apply_collider(cube)
    rep.functional.physics.apply_rigid_body(cube)

    # Set up the writer
    camera = rep.functional.create.camera(position=(5, 5, 3), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, (1280, 720))
    out_dir = os.path.join(os.getcwd(), "_out_cosmos_simple")
    print(f"Output directory: {out_dir}")
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    cosmos_writer.initialize(output_dir=out_dir, segmentation_mapping=segmentation_mapping)
    cosmos_writer.attach(rp)

    # Start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Capture a frame every app update
    for i in range(num_frames):
        print(f"Frame {i+1}/{num_frames}")
        await omni.kit.app.get_app().next_update_async()
        await rep.orchestrator.step_async(delta_time=0.0, pause_timeline=False)
    timeline.pause()

    # Wait for all data to be written
    await rep.orchestrator.wait_until_complete_async()
    print("Data generation complete!")
    cosmos_writer.detach()
    rp.destroy()

asyncio.ensure_future(run_cosmos_example_async(num_frames=NUM_FRAMES, segmentation_mapping=SEGMENTATION_MAPPING))
```

Standalone Application

Cosmos Writer Example

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import os

import carb.settings
import omni.replicator.core as rep
import omni.timeline
import omni.usd

SEGMENTATION_MAPPING = {
    "plane": [0, 0, 255, 255],
    "cube": [255, 0, 0, 255],
    "sphere": [0, 255, 0, 255],
}
NUM_FRAMES = 60

def run_cosmos_example(num_frames, segmentation_mapping=None):
    # Create a new stage
    omni.usd.get_context().new_stage()

    # CosmosWriter requires script nodes to be enabled
    carb.settings.get_settings().set_bool("/app/omni.graph.scriptnode/opt_in", True)

    # Disable capture on play, data is captured manually using the step function
    rep.orchestrator.set_capture_on_play(False)

    # Set DLSS to Quality mode (2) for best SDG results (Options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

    # Set the stage properties
    rep.settings.set_stage_up_axis("Z")
    rep.settings.set_stage_meters_per_unit(1.0)
    rep.functional.create.dome_light(intensity=500)

    # Create the scenario with a ground plane and a falling sphere and cube.
    plane = rep.functional.create.plane(position=(0, 0, 0), scale=(10, 10, 1), semantics={"class": "plane"})
    rep.functional.physics.apply_collider(plane)

    sphere = rep.functional.create.sphere(position=(0, 0, 3), semantics={"class": "sphere"})
    rep.functional.physics.apply_collider(sphere)
    rep.functional.physics.apply_rigid_body(sphere)

    cube = rep.functional.create.cube(position=(1, 1, 2), scale=0.5, semantics={"class": "cube"})
    rep.functional.physics.apply_collider(cube)
    rep.functional.physics.apply_rigid_body(cube)

    # Set up the writer
    camera = rep.functional.create.camera(position=(5, 5, 3), look_at=(0, 0, 0))
    rp = rep.create.render_product(camera, (1280, 720))
    out_dir = os.path.join(os.getcwd(), "_out_cosmos_simple")
    print(f"Output directory: {out_dir}")
    cosmos_writer = rep.WriterRegistry.get("CosmosWriter")
    cosmos_writer.initialize(output_dir=out_dir, segmentation_mapping=segmentation_mapping)
    cosmos_writer.attach(rp)

    # Start the simulation
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    # Capture a frame every app update
    for i in range(num_frames):
        print(f"Frame {i+1}/{num_frames}")
        simulation_app.update()
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False)
    timeline.pause()

    # Wait for all data to be written
    rep.orchestrator.wait_until_complete()
    print("Data generation complete!")
    cosmos_writer.detach()
    rp.destroy()

run_cosmos_example(num_frames=NUM_FRAMES, segmentation_mapping=SEGMENTATION_MAPPING)

simulation_app.close()
```

---

# Replicator Troubleshooting

This page consolidates troubleshooting information for Replicator components in Isaac Sim.

## Replicator Rendering Issues

If there is unwanted noise in simulated depth images, disable anti-aliasing under the **Render Settings > Ray Tracing > Anti-Aliasing** tab by setting the `Algorithm` to `None`.

If randomized materials are not loaded on time for synthetic data generation, the [rt\_subframes](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)") must be set to be at least `2`.

The replicator Scatter3D OmniGraph node breaks physics when called on a stage using world. Avoid using these together or use alternative methods for object placement.

If ghosting artifacts are observed in the captured data, especially for scenes with moving objects or significant changes in lighting conditions, increase the `rt_subframes` value when capturing the data to a value until the renderer is able to remove the artifacts. For more information see [RT Subframes Parameter](Synthetic_Data_Generation.md) and [subframes examples](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html#subframes-examples "(in Omniverse Extensions)").

If the captured images are written as black, try starting Isaac Sim once with the `--reset-user` to clear any previous user settings.

## Async Rendering and Frame Skipping

When using Replicator, frames may be skipped due to the `isaacsim.core.throttling` extension toggling `/app/asyncRendering=True` by default when the timeline is stopped. Since Replicator remains in STARTED mode, it does not re-initialize and toggle the setting back to False, leading to frames being skipped during capture.

**Solution:** Launch Isaac Sim with the following flag to disable async rendering toggling from the throttling extension:

```python
--/exts/isaacsim.core.throttling/enable_async=false
```

This occurs because when the timeline is stopped, the throttling extension enables async rendering for performance. However, when Replicator schedules frames for capture before the timeline starts playing again, those frames may be skipped due to async rendering being enabled. The flag above prevents the throttling extension from toggling async rendering, ensuring all scheduled frames are captured properly.

## Replicator Data Storage Issues

Using Replicator to write to S3 buckets with the built-in backend in Windows may require setting the credentials in the environment variables instead of the AWS config files. This is because of a possible path parsing error in Boto3 on Windows.

When working with large datasets or high-resolution images, you may experience storage bottlenecks. Consider:
1. Using a faster storage device
2. Reducing the image resolution or compression level
3. Using batch processing with smaller batches

## Replicator Layers and Randomization

Using [replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/basic_functionalities.html "(in Omniverse Extensions)")’s `rep.new_layer()` functionality, which creates a new layer in which to place and randomize assets, may lead to issues in simulation scenarios where these assets are used. In such cases the use of `rep.new_layer()` can be omitted.

When using multiple randomizers, be aware that they may conflict with each other. Test your randomization settings carefully to ensure they produce the expected results.

## Replicator Performance Issues

For complex scenes with many objects and randomizers, you may experience performance issues. Consider:
1. Reducing the number of objects in the scene
2. Simplifying the randomization parameters
3. Using fewer sensors or lower resolution sensors
4. Running with headless mode for improved performance during data generation

## Replicator API Changes

If you are encountering any issues regarding the dependencies on `omni.replicator.character` or `omni.replicator.agent`, the extension is now renamed to `isaacsim.replicator.agent`. Revise your code accordingly.

## Getting Started Scripts Issues

Common issues and solutions for the Getting Started Scripts:

1. **Data not being captured**
   - Ensure the capture-on-play flag is properly set
   - Check if the render products are correctly attached to writers
   - Verify the output directory has write permissions
2. **Rendering artifacts**
   - Try increasing RTSubframes value
   - Check if materials are fully loaded before capture
   - Ensure proper lighting setup
3. **Performance issues**
   - Reduce resolution or number of cameras
   - Use headless mode for faster processing
   - Optimize scene complexity
4. **Memory issues**
   - Reduce batch size
   - Clear unused resources with `destroy()`
   - Monitor GPU memory usage

## First Frame Missing in Windows Standalone Mode

On Windows, when running SDG pipelines with Replicator in standalone mode, the first frame may be skipped by writers or data may be missing from annotators.

### Workaround

Call a few “warmup” steps to advance the simulation before the first capture to avoid missing the initial frame. For example:

```python
# Warmup the simulation
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(2):
    standalone_app.update()
```

Alternative (depending on your Replicator control flow):

```python
import omni.replicator.core as rep
# [..] initialize writer [..]
rep.orchestrator.step()
# [..] start SDG pipeline [..]
```

---

# Action and Event Data Generation

**Action and Event Data Generation** is a reference application for Isaac Sim that provides a suite of extensions for realistic indoor simulation and large-scale synthetic data generation. It is designed to address the challenges of collecting high-quality, diverse, and richly labeled datasets for training Vision AI models.

Real-world data collection often faces limitations in scalability, cost, and the ability to capture rare or dangerous scenarios (such as accidents or near-misses). This application enables the programmatic generation of synthetic data that is accurate and diverse, effectively bridging the gap between simulation and real-world deployment.

## Key Features

* **Ground Truth Generation**: Provides accurate ground truth across multiple modalities by leveraging [Replicator](Synthetic_Data_Generation.md) for precise data capture and rich annotation.
* **Rare Event Generation**: Enables the programmatic creation of rare and long-tail events to improve model robustness.
* **Scalable Workflow**: Supports both an interactive interface for rapid prototyping and a headless batch generation mode for producing massive, reproducible datasets.
* **Configurable Control**: Utilizes YAML configuration files to define scenes, agents, and events, ensuring the data generation process is versionable and reproducible.

## Architecture and Workflow

Built on the Omniverse platform, this toolset integrates technologies such as [Omniverse Animation](https://docs.omniverse.nvidia.com/extensions/latest/ext_anim.html) for character behaviors, [Omniverse Flow](https://docs.omniverse.nvidia.com/extensions/latest/ext_fluid-dynamics/using.html) for dynamic events, and [Replicator](Synthetic_Data_Generation.md) for data capture.

The architecture employs a layered approach to scene construction and data capture. **Object Simulation** defines the static environment, which serves as the foundation for dynamic elements introduced by **Event Generation** and **Actor Simulation**. The pipeline culminates in the data acquisition phase, where **Sensor Placement** optimizes sensor coverage and **VLM Scene Captioning** synthesizes semantic descriptions.

## Launching the Application

To launch the app, use:

* **Linux**: `./isaac-sim.action_and_event_data_generation.sh`
* **Windows**: `.\isaac-sim.action_and_event_data_generation.bat`

The application launches with the Action and Event Data Generation extensions pre-enabled and a custom workspace layout.

## Action and Event Data Generation Stack

## Extensions

The core functionality is provided by a set of five application-level extensions and supporting tools:

| Extension | API Name | Description |
| --- | --- | --- |
| Actor Simulation and SDG | `isaacsim.replicator.agent.core` | The **Isaac Sim Replicator Agent (IRA)** extension simulates intelligent actors in 3D environments. It handles complex human and robot behaviors, from large-scale routines like warehouse operations (e.g., workers patrolling, forklifts roaming) to specific reactions to dynamic events. It captures diverse data and action metadata. |
| Object Simulation and SDG | `isaacsim.replicator.object.core` | The **Isaac Sim Replicator Object (IRO)** extension allows you to programmatically create and place objects at scale. It can procedurally generate unique shapes, automatically stack racks, and pack boxes before applying physics to settle the scene realistically. |
| Physical Space Event Generation | `isaacsim.replicator.incident.core` | The **Isaac Sim Replicator Incident (IRI)** extension generates realistic, configurable physical events. It orchestrates simulations using Omniverse Flow and PhysX to create scenarios ranging from spills and toppling boxes to complex fires with smoke, all with rich annotation and event metadata. |
| VLM Scene Captioning | `isaacsim.replicator.caption.core` | The **Isaac Sim Replicator Caption (IRC)** extension bridges the gap between vision and language. It analyzes the scene to build a scene graph (objects and spatial relationships) and uses an LLM to generate rich, human-readable descriptions (global and brief captions) and visualized scene graphs. |
| RTX Sensor Placement | `isaacsim.sensors.rtx.placement` | The **RTX Sensor Placement (ISP)** extension automates camera positioning. It algorithmically places sensors to maximize visual coverage, focus on points of interest, control occlusion, or create Bird’s-Eye-View groups, while extracting intrinsic and extrinsic calibration data. |
| RTX Sensor Calibration | `isaacsim.sensors.rtx.calibration` | The **RTX Sensor Calibration (ISC)** extension generates camera calibration data for deployed cameras in the scene. |
| Behavior Composer | `omni.behavior.composer` | The **Behavior Composer (OBC)** extension implements the classic behavior tree system for Omniverse Kit applications. It provides tools and APIs to author entity behaviors using OpenUSD and a standalone C++ core runtime API to power simulation engines. |
| Animated Robot Controller | `isaacsim.anim.robot` | The **Animated Robot Controller (IAR)** extension enables realistic robot animation by playing back captured simulation motion data. It bridges physics-based simulation and animation, allowing for precise robot movements without the overhead of real-time physics. |
| Action and Event Generation Utilities | `omni.metropolis.utils` | The **Action and Event Generation Utilities (OMU)** extension provides shared utilities across the Action and Event Generation extension stack. |
| Chat IRO | `omni.ai.langchain.agent.chat_iro` | **Chat IRO** is an AI assistant that enables natural language scene authoring for the **Object Simulation (IRO)** extension. It allows users to describe scenes in plain English to automatically generate YAML configurations, providing immediate viewport previews and iterative editing capabilities. |
| Isaac Agent Planner | `isaacsim.agent.planner.core`, `isaacsim.agent.planner.bridge` | The **Isaac Agent Planner (IAP)** extension automatically generates behavior trees for actors (characters and cameras) from natural language scenario descriptions. It uses LLMs and RAG to transform plain English like *“Alice picks up the mug”* into executable behavior trees compatible with Omni Behavior Composer. |

## Extension Tutorials

- Telemetry and Performance Tracking
- Actor Simulation and Synthetic Data Generation
- Object Simulation and Synthetic Data Generation
- VLM Scene Captioning
- Physical Space Event Generation
- RTX Sensors Placement and Calibration
- Isaac Agent Planner (IAP)

---

# Telemetry and Performance Tracking

The Action and Event Data Generation extensions include built-in telemetry capabilities to track performance metrics and usage patterns. The telemetry system captures various metrics across extensions, providing valuable insights into system behavior, performance characteristics, and usage patterns.

## Overview

Telemetry in the Action and Event Data Generation ecosystem helps developers and users:

* **Monitor Performance**: Track execution times, resource usage, and system performance
* **Understand Usage Patterns**: Gain insights into how features are being used
* **Identify Issues**: Detect bottlenecks and performance problems early
* **Improve User Experience**: Use data-driven insights to optimize workflows

The telemetry system is implemented across multiple extensions and provides a standardized approach to metric collection and reporting.

Local telemetry logs can be found in the `~/.nvidia-omniverse/logs/` directory.

## Telemetry Architecture

The telemetry system is built on NVIDIA Omniverse’s structured logging framework and consists of:

* **Schema Definition**: Structured schemas defining telemetry events and their attributes
* **Event Generation**: Automated Python bindings generated from schema definitions
* **Data Collection**: Instrumented code that emits telemetry events
* **Storage and Analysis**: Events logged locally and transmitted to analysis platforms

For more details, see the [Omniverse Telemetry Walkthrough](https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/structuredlog/Walkthrough.html).

## Telemetry Modes

The telemetry system supports different operational modes:

* **Production Mode**: `--/telemetry/mode=prod` - Default mode for production deployments.
* **Test Mode**: `--/telemetry/mode=test` - Internal mode for QA, validation, and testing.
* **Dev Mode**: `--/telemetry/mode=dev` - Internal mode for development.

Note that different modes have different data collection and transmission policies.

To disable telemetry, set the `--/telemetry/enableAnonymousData=false` command line argument.

Regardless of the mode, data is saved locally to the user’s home directory in the `~/.nvidia-omniverse/logs/` directory.

## Configuring Telemetry

Telemetry can be enabled or disabled through extension settings in `extension.toml`.

The following extensions contain specific telemetry settings:

* `isaacsim.replicator.agent`
* `omni.metropolis.utils`

```python
[settings]
exts."isaacsim.replicator.agent".telemetry_enabled = true
```

To modify telemetry settings at runtime:

```python
import carb.settings

settings = carb.settings.get_settings()

# Enable telemetry for specific extensions
settings.set("/exts/isaacsim.replicator.agent/telemetry_enabled", True)
settings.set("/exts/omni.metropolis.utils/telemetry_enabled", True)

# Disable telemetry for specific extensions
settings.set("/exts/isaacsim.replicator.agent/telemetry_enabled", False)
settings.set("/exts/omni.metropolis.utils/telemetry_enabled", False)
```

### Available Telemetry Events

The following telemetry events are available across the Action and Event Data Generation extensions:

**omni.metropolis.utils**

* `file_read` - Tracks file read operations
* `file_write` - Tracks file write operations

**isaacsim.replicator.agent.core**

* `data_generation` - Tracks data generation operations
* `load_asset_to_scene` - Tracks asset loading events
* `stage_setup_event` - Tracks stage setup operations
* `writer_initialized_event` - Tracks writer initialization events

## Related Documentation

* [Action and Event Data Generation Overview](What_Is_Isaac_Sim.md)
* [Actor Simulation and Synthetic Data Generation](Synthetic_Data_Generation.md)
* [Object Simulation and Synthetic Data Generation](Synthetic_Data_Generation.md)

## Additional Resources

* [Omniverse Telemetry Walkthrough](https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/structuredlog/Walkthrough.html)

---

# Actor Simulation and Synthetic Data Generation

Detecting and tracking animated actors or agents like human characters and robots in diverse environments offers significant value across industries like retail, manufacturing, and logistics. It helps optimize layouts, improve safety, and enhance efficiency. However, collecting real-world data to train detection models is often costly and unscalable.

Synthetic data generation offers a flexible, scalable solution. The `Omni.Metropolis.Core` (OMC), `Isaacsim.Replicator.Agent` (IRA), `Isaacsim.Anim.Robot.Core` (IAR) extensions together provide a way to set up human characters and robots in 3D environments and generate synthetic data.
This framework also provides control over actor behaviors, environments, sensors, via configuration file. It aims to provide a GPU-accelerated solution for training computer vision models and testing software-in-the-loop systems.

This framework simplifies simulation customization with features like:

* **Codeless Interaction**: Configurations are expressed in yaml file. No code is needed to get synthetic data.
* **Simplified Setup**: Included in Isaac Sim, it offers both GUI and scripting interfaces for interactive and headless workflows.
* **High-Fidelity Data**: Leverages Omniverse’s SimReady assets, physics, and rendering to produce realistic imagery and accurate annotations essential for AI training.
* **Seamless Integration**: As part of Kit extensions, it works natively with `omni.anim.behavior`, `omni.anim.navigation`, and `omni.replicator.core`.

Before enabling this extension, read [What Is Isaac Sim?](../overview/overview.html) to learn about Isaac Sim and follow [Installation](Installation.md) to install Isaac Sim.

## Enable Extensions

1. Follow the [Omniverse Extension Manager guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html) to enable the `Omni.Metropolis.Core`, `Isaacsim.Replicator.Agent.Core & UI` and `Isaacsim.Anim.Robot.Core`.

   > * The extensions fetch sample assets from Isaac Sim Assets during start. Refer to [Isaac Sim Assets](Isaac_Sim_Assets.md) if you encounter issues for loading assets.
   > * If loading the UI appears to be hanging, try starting Isaac Sim with the flag `--/persistent/isaac/asset_root/timeout=1.0`.
2. The UI panel is accessible by **Tools > Action and Event Data Generation > Actor SDG** and it opens on the right side of the screen.

Note

* To have the extension auto-loaded on startup, check the **autoload** checkbox in the extension manager.
* Because of extension dependencies, a restart of the Isaac Sim app might be required.

Tip

If you encounter unexpected errors, try launching Isaac Sim with the `--reset-user` flag to clear previous user settings.

```python
./isaac-sim.sh --reset-user
```

## Getting Started in UI

It is recommended to use UI for first-time users. Please refer to [Running from script](#actor-sim-running-from-script) section for running with python script in IsaacSim headless mode.

1. Follow the [Enable Extensions](#actor-sim-getting-started) and open the UI panel.
2. The default minimal config is loaded by default. You can also load a seperate config file using the folder browser icon.

   > * All the sample config files are in `[Isaac Sim App Path]/extscache/isaacsim.replicator.agent.core-[current-version]/data/sample_configs/`.
   > * THe minimal config file does not have actors and cameras. For a more comprehensive example, please use `warehouse.yaml` in the above folder. Note that this example can take up more loading time.

3. [Optional] Modify the configuration file to your needs.

   > * Use Save or Save As icon to save the changes in UI to config file.
   > * Use Reload icon to reset changes in UI and load the original config file again.
4. Click the **Set Up Simulation** button from the top of the UI and it will start loading simulation assets (scene, cameras, actors) according the UI.

   > * The scene requires a NavMesh to spawn assets and control them correctly. The scenes in the example config has NavMesh set up in advance. If you are using a external scene, please refer to [Navigation Mesh](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh.html "(in Omniverse Extensions)") for NavMesh set up.
   > * You can also go to **Window > Navigation > NavMesh** and turn off **Auto-Bake** in the NavMesh settings. Turning it off can increase the performance.
5. Click the **Start Data Generation** button from the top of the UI and the simulation and data generation will start. It will run for the duration (in seconds) specified in the **Simulation Duration** in **Actor SDG Setup** panel.
6. When data generation finsihes, the output data can be found from the **Output Directory** according to the output direcotry in **Replicator** panel.

   > * By default, it is in the User folder for Windows and the home folder for Linux.

## Running from Script

For large-scale data generation, it can be more efficient to launch it from script. IRA provides an automatic script (`actor_sdg.py`) to run offline data generation.

To run from script, open a terminal from where Isaac Sim is installed and run the following commands.

* For Linux:
  :   `./python.sh tools/actor_sdg/actor_sdg.py -c [config file path]`
* For Windows:
  :   `.\python.bat tools\actor_sdg\actor_sdg.py -c [config file path]`

Note

* `[config file path]` is the path to the IRA configuration file.
* You must use the `python.sh` or `python.bat` bundled with Isaac Sim to run the script.
* An example config file is also provided in the `/tools/actor_sdg` folder. For a sample Linux run, execute: `./python.sh tools/actor_sdg/actor_sdg.py -c tools/actor_sdg/sample_config.yaml`

## Configuration File

The configuration file is the central place to define your simulation. It controls everything from the environment and characters to the sensors and data output. The file uses the YAML format.

The configuration file is organized into these top-level sections:

* `environment`: Defines the simulation environment and assets.
* `character`: Configures human characters.
* `robot`: Configures robots.
* `sensor`: Configures RTX sensors.
* `replicator`: Configures data generation and output.

For detailed configuration instructions, parameter lists, and examples, refer to the following document:

- Configuration File Guide

## Actor Behaviors

Actor behaviors are achieved by OMC, IRA and IAR together.

Actors perform a “routine-trigger” behavior loop at play. This pattern is configurable by the behaviors and triggers assigned to the actor.

### The Routine Trigger Loop

When no actor triggers are activated, actors perform routine loop by repeatedly pick behaviors under routines to perform by their probability weights, using the `actor global seed`.

When any trigger is activated, actor will pause routine and start performing the behaviors under each active trigger. Running triggers will be paused and pushed to queue if a trigger with higher priority happens (triggers with lower priority will be skipped).
The trigger will be marked complete when its behaviors are all finished. Then the first trigger in queue will resume running.
Once all active triggers complete, actors will fallback to routine.

### Configuerate Behaviors

After actors are loaded into scene by config file, the configurations are embedded in the USD API schemas and USD Prims. Each actor is represented by MetroAgentAPI schema and its derived type.
For human character, it is the `IRACharacterAPI` attached on the SkelRoot prim. For animated robot, it is the `AnimRobotAPI` attached on the root prim of the robot payload.
Each behavior and trigger becomes individual USD Prims that actor USD API can have reference to, each actor trigger prim can also have reference to a list of behaviors.

The actor USD API schema defines basic information of the actor: name, group, seed, a routine reference slot and a triger reference slot.
At play, the name, group and seed will be combined and hashed into a single seed as `actor global seed`. This seed will be used for all the “randomness” of the actor, including random routine picking for the actor itself and the picking within each behavior such as picking a speed from speed range.
This also means the same `actor global seed` will display same result if other settings and the environment don’t change.

Each type of actor behavior is represented by a USD Prim type. It defines the configuration of the behavior: weight, repeat and behavior specific parameters.
For human characters, the behavior prim types follows `CharacterXXXBehavior` naming pattern. For animated robots, they are `RobotXXXBehavior`.

Each actor trigger is also a USD Prim. It defines the trigger prioirty and has a refrence of behavior list to be executed sequentially when this trigger activates.
Human characters and anim robots share the same trigger types that’s defined in OMC with naming `MetroXXXTrigger`.

In addition, actors leverage `omni.behavior.behavior` (Human characters) and `isaacsim.anim.robot.core` (Animated robots) as their animation implementation.
For more information about them, please refer to the following documents:

- Animated Robot Controller

## Terminology

Isaacsim.Replicator.Agent.Core

The core extension that manages the simulation state. It contains the essential API and modules for setting up the simulation and capturing the synthetic data. Its modules can be called independently.

Isaacsim.Replicator.Agent.UI

The UI extension for IRA. When this extension loads, the core extension is loaded automatically. This extension contains the UI components for easy interaction with the extension.

Configuration File

A `.yaml` file that contains configuration data that defines the key components of a simulation, including the randomization seed, duration of the simulation, number of the actors, and output format. To use the extension, you must load a configuration file or use the UI to generate a YAML file first.

Actor

Actors are controlled by the respective controllers (omni.behavior.composer and isaacsim.anim.robot) and perform actions in the simulation. The extension supports human characters and robots (Nova Carter, iw.hub) as actors. The terms “actor” and “agent” are used interchangeably in this documentation.

Seed

Randomization seed. Given the same seed, the extension can generate the same randomized result for camera and agent location and agent behaviors. With the same seed and the same sequence of operations, the same data is guaranteed to be generated.

Replicator (Omni.Replicator.Core)

The data capturing extension that our extension is based on. More information about the Replicator extension can be found in [Replicator Official Documentation](Synthetic_Data_Generation.md).

---

# Configuration File Guide

This guide describes how to configure the Isaac Sim Replicator Agent (IRA) for simulation and synthetic data generation. The configuration controls the environment, sensor generation and placement, character/robot agents, behaviors, and data generation.

## Concepts & Workflow

Before diving into detailed configuration, it is helpful to understand the general workflow and key concepts of an IRA simulation.

### Workflow Overview

1. **Environment Setup**: Define the static 3D environment where the simulation takes place.
2. **Agent & Sensor Definition**: Configure characters, robots, and cameras (sensors) to populate the environment.
3. **Behavior Configuration**: Assign routines (weighted random actions like walking, idling) and triggers (reactive behaviors like when a collision occurs) to actors.
4. **Data Generation**: Configure the Replicator writers to generate ground-truth data (RGB, segmentation, etc.).

#### Key Concepts

* **Environment**: The static 3D world (USD stage) loaded for the simulation. It also defines the NavMesh for navigation.
* **Agents (Actors)**: Dynamic entities in the scene, which can be **Characters** (humans) or **Robots**.
* **Behaviors**: Atomic actions an actor can perform, such as `wander`, `patrol`, or `idle`.
* **Routines**: A collection of behaviors assigned to an actor group. Actors randomly select behaviors from this pool based on assigned weights.
* **Triggers**: Conditional logic that interrupts normal routines. When a condition is met (e.g., a specific time or event), the trigger executes its defined list of behaviors in sequence. Once the trigger sequence is complete, the agent resumes its standard routine until another trigger activates.
* **Sensors**: Cameras placed in the scene to observe the simulation.
* **Replicator**: The system responsible for rendering frames and writing annotated data (ground truth) to disk or cloud storage.

## Top-level Structure

Configs are YAML files with a single root key `isaacsim.replicator.agent`:

```python
isaacsim.replicator.agent:
  version: 1.0.0
  environment: { ... }            # required
  seed: 123456789                 # optional; 32-bit (0..4294967295); autogenerated if omitted
  simulation_duration: 60.0       # optional; defaults to 60.0
  character: { ... }              # optional
  robot: { ... }                  # optional
  sensor: { ... }                 # optional
  replicator: { ... }             # optional
```

### Root Parameters

* **version** (required): Semantic version of the configuration schema (e.g., “1.0.0”).
* **environment** (required): Defines the simulation world.
* **seed** (optional): A 32-bit unsigned integer (0..4,294,967,295).
  - Used to initialize random number generators for deterministic simulations (e.g., character spawn locations, routine variations).
  - If omitted, a seed is generated based on the current system time.
* **simulation\_duration** (optional): The total run time of the simulation in seconds.
  - The simulation runs at a fixed time step corresponding to **30 FPS**.
  - Defaults to `60.0` seconds.
* **character** (optional): Configures human agents (appearance, behaviors like wander/patrol, and triggers).
* **robot** (optional): Configures robot agents (config path, behaviors/commands, data collection).
* **sensor** (optional): Configures static cameras using placement strategies (e.g., aim at targets, coverage).
* **replicator** (optional): Configures data writers (e.g., output directory, annotators like RGB/segmentation).

### Quick Start (Minimal)

```python
isaacsim.replicator.agent:
  version: 1.0.0
  environment:
    base_stage_asset_path: "Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
```

## Sections

### Environment

Defines the static 3D environment and additional assets to load.

* `base_stage_asset_path` (required): Path or URL to the main USD stage.
  - Supports `http(s)://` (including S3 presigned URLs), Windows/UNC paths, and local filesystem paths.
  - Also supports paths relative to the [Isaac Sim Assets](Isaac_Sim_Assets.md) root.
* `prop_asset_paths` (optional): A list of additional USD assets to load as **sublayers** into the stage. This is useful for adding props or lighting to a base environment without modifying it. Supports paths relative to the [Isaac Sim Assets](Isaac_Sim_Assets.md) root.

**Example:**

```python
environment:
  base_stage_asset_path: "Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
  prop_asset_paths:
    - "Isaac/Props/Conveyors/ConveyorBelt_A08.usd"
```

### Character

Defines groups of human characters, their appearance, and their behavior.

* `root_prim_path` (optional): Root path for spawning characters (default: `/World/Characters`).
* `groups`: Dictionary of character groups.

#### Character Group Parameters

* `num` (required): Number of characters to spawn.
* `asset_path` (optional): USD path to character assets. Supports paths relative to the [Isaac Sim Assets](Isaac_Sim_Assets.md) root. Default: `Isaac/People/Characters/`.
* `spawn_areas` (optional): List of **NavMesh area names** where characters can spawn. If empty, spawns anywhere on the NavMesh.
* `semantic_labels` (optional): List of `[type, data]` pairs for semantic segmentation. Default: `[["class", "character"]]`.
* `motion_library_path` (optional): Path to a custom motion library file. Supports paths relative to the [Isaac Sim Assets](Isaac_Sim_Assets.md) root. Default: `Isaac/People/MotionLibrary/HumanMotionLibrary.usd`.
* `routines` (optional): List of behaviors the characters will execute. Default: `[{ wander: {} }]`.
* `triggers` (optional): List of event-based triggers that interrupt routines.

#### Behaviors

Behaviors are defined in the `routines` list. Common fields:

* `weight` (default 1): Probability weight for selecting this behavior.
* `repeat` (default 1): How many times to repeat this behavior before choosing a new one.

**Supported Behaviors:**

1. **wander**: Randomly walk and idle.

   * `walk`:
     - `speed_range`: [min, max] m/s (default [1.0, 1.0]).
     - `distance_range`: [min, max] distance to travel per walk leg (default [5.0, 15.0]).
     - `navigation_areas`: List of allowed NavMesh area tags.
   * `idle`: Array of idle options.
     - `animation`: Name of the animation (must exist in motion lib).
     - `time_range`: [min, max] duration in seconds (default [2.0, 5.0]).
     - `weight`: Selection probability.
2. **patrol**: Follow a specific path.

   * `speed_range`: [min, max] m/s (default [1.0, 1.0]).
   * One of:
     - `path_points`: List of 3D points `[[x,y,z], [x,y,z], ...]`.
     - `target_prims`: List of prim paths to visit.

     Note

     Both `path_points` and `target_prims` must be on the [NavMesh](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh.html "(in Omniverse Extensions)") and reachable by the actors.
3. **custom\_behavior**: Execute a Behavior Tree from a file.

   * `behavior_name`: Identifier.
   * `behavior_subtree`: Path to the behavior JSON file.
   * `params`: Dictionary of arbitrary parameters passed to the tree.

#### Triggers

Triggers define events that interrupt normal routines to execute a specific reaction behavior.

* `priority` (default 1): Higher priority triggers override lower ones.
* `behavior`: List of behaviors to execute when triggered.

**Trigger Types:**

* `event_trigger`: Fires on a named event (e.g., custom simulation events).
* `time_trigger`: Fires after a specific duration (in seconds). *Not available in Isaac Sim 6.0 EA yet. (Coming in 6.0 GA)*

  Tip

  **Dispatching Events from Python**

  The method for dispatching events differs between **Characters** and **Robots**.

  **For Characters:**
  Use the `AgentManager` to update the behavior tree blackboard directly.

  ```python
  from isaacsim.replicator.agent.core.agent_manager import AgentManager

  # Arguments:
  # - prim: The prim path of the specific character instance
  # - key: The internal blackboard key for events (always "event_trigger_key")
  # - value: The name of the event to trigger (matches the YAML 'event' field)
  AgentManager._set_tree_blackboard(
      prim="/World/Characters/warehouse_workers/Character",
      key="event_trigger_key",
      value="test_event"
  )
  ```

  **For Robots:**
  Use the `carb.eventdispatcher` system.

  ```python
  import carb

  # Arguments:
  # - event_name: The name of the event to trigger (matches the YAML 'event' field, e.g., "my_test_event")
  # - payload: A dictionary containing:
  #     - "prim_path": The prim path of the specific robot instance
  carb.eventdispatcher.get_eventdispatcher().dispatch_event(
      "my_test_event",
      payload={"prim_path": "/World/Robots/iw_hub/iw_hub"}
  )
  ```

**Example:**

```python
character:
  root_prim_path: "/World/Characters"
  groups:
    warehouse_workers:
      asset_path: "Isaac/People/Characters/"
      num: 10
      spawn_areas: ["warehouse_floor"]
      routines:
        - wander:
            walk:
              speed_range: [0.8, 1.5]
              distance_range: [5.0, 10.0]
            idle:
              - animation: look_around
                time_range: [2.0, 5.0]
      triggers:
        - time_trigger:
            time: 30.0
            priority: 10
            behavior:
              - patrol: # Move to break room
                  speed_range: [1.0, 1.2]
                  target_prims: ["/World/BreakRoom"]
        - event_trigger:
            event: test_event
            priority: 10
            behavior:
              - patrol:
                  speed_range: [5.0, 5.5]
                  path_points:
                    - [0, 0, 0]
                    - [0, -5, 0]
```

### Robot

Defines robot agents. Robots can be controlled via behaviors OR an external command file.

* `root_prim_path` (optional): Root path for robots (default: `/World/Robots`).
* `groups`: Dictionary of robot groups.

#### Robot Group Parameters

* `num` (required): Number of robots.
* `config_file_path` (required): Path to the **IAR (Isaac Agent) configuration file** for this robot type. Supports absolute paths or paths relative to the IAR built-in robot config sample folder (located in `isaacsim/anim/robot/agent/configs` within the `isaacsim.anim.robot` extension).
* `command_file_path` (optional): Path to a global command file. **Mutually exclusive with** `routines`.
* `spawn_areas` (optional): NavMesh areas for spawning.
* `agent_radius` (optional): Radius in meters (default 0.5) used for NavMesh queries.
* `write_data` (optional): If `true`, enables data collection from the robot’s onboard cameras. *Not available in Isaac Sim 6.0 EA yet. (Coming in 6.0 GA)*
* `camera_prim_paths` (optional): List of specific camera prims on the robot to use. If empty and `write_data` is true, *all* cameras on the robot are used. *Not available in Isaac Sim 6.0 EA yet. (Coming in 6.0 GA)*
* `semantic_labels` (optional): Default `[["class", "robot"]]`.
* `semantic_label_path` (optional): Relative path under the robot prim to apply semantics.
* `routines` / `triggers`: Similar to characters but with robot-specific behaviors.

#### Robot Behaviors

* **wander**:
  - `move`: { `distance_range`: [min, max] (default [10.0, 15.0]), `navigation_areas`: [tags] }
  - `idle`: { `time_range`: [min, max] (default [2.0, 5.0]) }
* **patrol**:
  - `path_points`: List of 3D points.
* **stop**:
  - `time_range`: [min, max] seconds to remain stopped (default [5.0, 5.0]).

### Sensor

Defines static cameras in the scene. Cameras are organized into named **groups**.

* `root_prim_path` (optional): Absolute prim path where all camera groups will be created (default: `/World/Cameras`).
* `groups`: A dictionary where keys are group names and values define the camera configuration.

#### Group Configuration

Each group must specify `num` (number of cameras) and **one** placement strategy (`aim_at_targets` OR `maximum_coverage`).
- For `aim_at_targets`, `num` must be >= 0.
- For `maximum_coverage`, `num` can be >= -1. If `-1`, the number of cameras is automatically calculated based on the grid resolution and coverage ratio.

**1. Placement Strategy: aim\_at\_targets**

Places cameras to look at specific targets.

* `targets` (optional): List of target prim paths (e.g., `/World/Characters`) or identifiers.
* `raycast_density` (optional): Density of rays used to find valid camera positions. Higher values are more precise but slower.
* `yaw_range` (optional): [min, max] degrees (0..360) for the camera’s rotation around the target.
* `occlusion_threshold` (optional): Threshold for filtering occluded views (-1 to disable).

**2. Placement Strategy: maximum\_coverage**

Places cameras to maximize visual coverage of the environment.

* `target_coverage_ratio` (optional): Desired coverage ratio (0.0 to 1.0). Default `0.9`.
* `grid_resolution` (optional): Size of the grid cells (in meters) used for coverage calculation. Default `1.0`.

**Shared Parameters**

These apply to all placement strategies:

* `height_range`: [min, max] height in meters (Z-axis).
* `look_down_angle_range`: [min, max] pitch angle in degrees (0 = horizontal, 90 = straight down).
* `focal_length_range`: [min, max] focal length in millimeters.
* `distance_range`: [min, max] distance from the camera to its target/interest point in meters.

**Example:**

```python
sensor:
  root_prim_path: "/World/Cameras"
  groups:
    ceiling_cameras:
      num: 20
      aim_at_targets:
        targets: ["/World/Characters"]
        distance_range: [5, 10]
        height_range: [7, 10]
        focal_length_range: [10, 15]
        look_down_angle_range: [30, 45]
    coverage_cameras:
      num: 5
      maximum_coverage:
        target_coverage_ratio: 0.8
        height_range: [2, 5]
```

### Replicator

Controls the generation of synthetic data (images, annotations) using Omniverse Replicator.

* `writers` (required): Dictionary of writer configurations.
* `hide_debug_visualization` (optional, default `true`): Hides debug visualizations (NavMesh, skeletons, lights) during data capture.

#### Writer Configuration

Supported writers: `BasicWriter`, `IRABasicWriter`, `CosmosIRAWriter`, `RTSPWriter`, `CustomWriter`.

**Common Settings per Writer:**

* **Timing**:
  - `start_frame` / `end_frame`: Frame-based control (inclusive/exclusive). Defaults to `start_frame: 30` if not specified.
  - `start_time` / `end_time`: Time-based control (seconds).
* **Sensors**:
  - `sensor_prim_list`: Optional list of cameras to use. If omitted, uses all cameras defined in `sensor.root_prim_path`.

**Output Settings:**

* `output_dir`: Local directory for output. Defaults to `~/IRA_output` if not set.
* `s3_bucket`, `s3_region`, `s3_endpoint`: For direct S3 upload.

**Common Annotators (Parameters):**

* `rgb`: RGB Image.
* `bounding_box_2d_tight` / `loose`: 2D Bounding boxes.
* `bounding_box_3d`: 3D Bounding boxes.
* `semantic_segmentation`: Pixel-wise semantic class IDs.
* `instance_segmentation`: Pixel-wise instance IDs.
* `distance_to_camera`: Depth map.
* `normals`: Surface normals.
* `motion_vectors`: Pixel motion.
* `colorize_*`: e.g., `colorize_semantic_segmentation` (save as visible color map vs raw ID).

#### Specialized Writers

1. **IRABasicWriter**:

   The foundational writer for Agent simulations, derived from Replicator’s `BasicWriter`. It organizes output into separate folders per annotator and consolidates object/agent metadata into `object_detection.json`.

   * **Key Features**:

     + **Folder Structure**: Outputs each annotator’s data into separate folders for better readability.
     + **Object Detection**: Consolidates bounding box and skeleton data into a single file named `object_detection.json`.
     + **Default Semantic Filter**: `class:character|robot;id:*` (captures characters and robots).
   * **Overwritten Annotators**:
     The following standard annotators are replaced by specialized `object_info_*` versions and written to `object_detection.json`:

     + `bounding_box_2d_tight`
     + `bounding_box_2d_loose`
     + `bounding_box_3d`
     + `skeleton_data` (replaced by `agent_info_skeleton_data`)
   * **Defaults**:

     + `rgb`: Enabled.
     + `camera_params`: Enabled.
     + S3-related parameters: Disabled.
   * **Special Parameters**:

     + `video_rendering_annotator_list`: Generates .mp4 videos for specified annotators (e.g., `["rgb", "semantic_segmentation"]`).
     + `agent_info_skeleton_data`: Exports 2D/3D skeleton joints for characters.
2. **CosmosIRAWriter**:

   * Adds “Cosmos” specific post-processing.
   * `shaded_seg`: Shaded segmentation visualization.
   * `canny_edge`: Canny edge detection filter (with `canny_threshold_low/high`).
3. **RTSPWriter**:

   Streams selected annotators live over RTSP instead of saving frames to disk. It spins up an `ffmpeg` process per camera and annotator and pushes raw frame buffers into that stream.

   * **Key Features**:

     + **Live RTSP Streaming**: Publishes each camera and annotator to `rtsp://<host>:<port>/<topic>_<camera>_<annotator>`.
     + **Per-annotator Streams**: Each enabled annotator gets its own RTSP endpoint.
     + **Hardware/Software Encoding**: NVENC for supported 8-bit RGBA annotators with a software fallback for others.
     + **Automatic GPU Distribution**: Multiple GPUs are automatically distributed across available render products.

   RTSP setup (FFmpeg)

   Before streaming from Isaac Sim using `RTSPWriter`, install FFmpeg.

   Run the following command on Linux:

   Install FFmpeg on Linux

   ```python
   sudo apt update && sudo apt install -y ffmpeg
   ```

   Run the following command on Windows 10/11:

   Install FFmpeg on Windows 10/11

   ```python
   winget install ffmpeg
   ```

   RTSP parameters and annotators

   Each of these parameters is represented as a boolean toggle in the UI. Enabling any of them will include that data type in the RTSP output stream.

   * `rtsp_stream_url`: Base RTSP server URL.
   * `rtsp_rgb`: Toggle RGB stream (LdrColor).
   * `rtsp_semantic_segmentation`: Toggle semantic segmentation stream.
   * `rtsp_instance_id_segmentation`: Toggle instance ID segmentation stream.
   * `rtsp_instance_segmentation`: Toggle instance segmentation stream.
   * `rtsp_normals`: Toggle normals stream.
   * `rtsp_distance_to_image_plane`: Toggle distance to image plane stream.
   * `rtsp_distance_to_camera`: Toggle distance to camera stream.
   * `device`: NVENC GPU index.

   Supported RTSP annotators:

   * `rgb` (LdrColor)
   * `semantic_segmentation`
   * `instance_id_segmentation`
   * `instance_segmentation`
   * `normals`
   * `distance_to_camera`
   * `distance_to_image_plane`

   RTSP stream URL format

   Each stream URL follows this format:

   `[rtsp_stream_url]/RTSPWriter[camera_prim_path_with_underscores]_[annotator_name]`

   Where:

   * `rtsp_stream_url`: Base RTSP server URL (for example, `rtsp://localhost:8554/RTSPWriter`).
   * `camera_prim_path_with_underscores`: The camera prim path with forward slashes replaced by underscores.
   * `annotator_name`: The original annotator name (for example, `rgb` or `distance_to_camera`).

   Examples (`rtsp_stream_url` = `rtsp://localhost:8554/RTSPWriter`):

   * RGB stream for `/World/Cameras/Camera_01`:
     `rtsp://localhost:8554/RTSPWriter_World_Cameras_Camera_01_rgb`
   * Distance-to-camera stream for `/World/Cameras/Camera`:
     `rtsp://localhost:8554/RTSPWriter_World_Cameras_Camera_distance_to_camera`

   RTSP defaults

   * `rtsp_rgb`: Enabled.
   * All other `rtsp_*` annotators: Disabled.
   * `rtsp_stream_url`: `rtsp://localhost:8554/RTSPWriter`.
   * `device`: `0` (NVENC GPU index).

   RTSP runtime notes

   Warning messages are posted to the console to show the match between annotator names and RTSP stream URLs.
   Initializing RTSP streaming may take a while.
   During RTSP stream initialization, the first few frames may exhibit visual artifacts or corruption. This is expected behavior and resolves after the stream is fully established.

Note

In Isaac Sim 6.0 EA release, we support only the `IRABasicWriter`. Other writers will be supported in the Isaac Sim 6.0 GA release.

**Example:**

```python
replicator:
  writers:
    IRABasicWriter:
      # Output Config
      output_dir: "/home/IRA_Output"
      s3_post_upload: false

      # Timing
      start_frame: 0
      end_frame: 300

      # Annotators
      rgb: true
      camera_params: true
      bounding_box_2d_tight: true
      semantic_segmentation: true
      agent_info_skeleton_data: true

      # Video
      video_rendering_annotator_list: ["rgb", "semantic_segmentation"]
```

---

# Animated Robot Controller

The `isaacsim.anim.robot.core` extension provides functionality for generating animated robots in Isaac Sim Replicator Agent (IRA). This extension is powered by [Behavior Script](https://docs.omniverse.nvidia.com/extensions/latest/ext_python-scripting-component/user_manual.html) , which enables reactive behavior based on USD stage events.

`isaacsim.anim.robot.core` generates animated robots by first simulating the robot, capturing its motion data, and then playing it back in the form of commands. Robots are simulated by `isaacsim.robot.wheeled_robots` for motions such as moving forward and turning round, which is recorded by `omni.kit.stagerecorder.core`. Then, these motion capture data are converted into commands such as `GoTo` and `Idle`.

`isaacsim.anim.robot.core` supports Nova Carter and iw.hub robots, However, custom actors can be added by setting up its `dataclass`.

## Customization

Robot behavior and animation can be customized by modifying the agent configuration YAML file located at:
`{isaacsim.anim.robot.core extension path}/isaacsim/anim/robot/agent/configs/{robot type}.yaml`

## Robot Attributes

The following attributes can be configured for each robot actor (based on the BaseAgentConfig schema):

* **agent\_name** (str): Display name of the agent (default: “BaseAgent”).
* **linear\_velocity** (float): Forward movement speed in meters per second (m/s).
* **angular\_velocity** (float): Turning speed in degrees per second (deg/s).
* **forward\_vec** (list[float]): Initial forward direction vector (default: [1.0, 0.0, 0.0]).
* **joints** (list[str]): List of joint prim relative paths that can be animated.
* **drive\_base** (str): Robot’s drive system type. Supported values: `differential`, `omni_directional`.
* **path\_planner** (str): Path planner type. Supported values: `navmesh`, `base` (default: `navmesh`).
* **states** (list[str]): List of Finite State Machine (FSM) state names (e.g., `["idle", "turn_left", "turn_right", "forward"]`).
* **transitions** (dict[str, list[str]]): State transition graph defining valid transitions between states.
* **animation\_paths** (dict[str, str]): Mapping of state names to folder paths containing animation USDs.
* **asset\_path** (str | null): Relative or absolute path/URL to the agent USD. If relative, it tries local file first, then Isaac Sim asset root.
* **radius** (float | null): Override the radius of the agent for path planning. If not provided, the radius is calculated from the agent’s bounding box.

**Example Configuration (iw\_hub.yaml):**

```python
agent_name: "iw_hub"
linear_velocity: 0.5
angular_velocity: 30.0
forward_vec: [1.0, 0.0, 0.0]
joints:
  - "/chassis/lift"
  - "/chassis/left_wheel"
  - "/chassis/right_wheel"
  - "/chassis/left_swivel/left_caster"
  - "/chassis/right_swivel/right_caster"
  - "/chassis/left_swivel"
  - "/chassis/right_swivel"
drive_base: "differential"
path_planner: "navmesh"
animation_paths:
  turn_left: "${ext_path}/data/iw_hub/turn_left"
  turn_right: "${ext_path}/data/iw_hub/turn_right"
  forward: "${ext_path}/data/iw_hub/forward"
  lift_up: "${ext_path}/data/iw_hub/lift_up"
  lift_down: "${ext_path}/data/iw_hub/lift_down"
states: ["idle", "turn_left", "turn_right", "forward", "lift_up", "lift_down"]
transitions:
  idle: ["turn_left", "turn_right", "idle", "forward", "lift_up", "lift_down"]
  turn_left: ["forward"]
  turn_right: ["forward"]
  forward: ["turn_left", "turn_right", "idle"]
  lift_up: ["idle"]
  lift_down: ["idle"]
asset_path: "Isaac/Samples/AnimRobot/iw_hub.usd"
```

## Customizing Animations

To create custom animations:

1. Simulate the robot in Isaac Sim.
2. Use `omni.kit.stagerecorder.core` to capture motion data.
3. Update the `animation_paths` attribute with new animation file paths.

Animation files should be organized in state-specific folders. For example, `iw.hub`’s turn-left animation is located at:
`{Isaac Sim App Path}/extcache/isaacsim.anim.robot.core/data/iw_hub/turn_left/`

Store each joint’s animation data in a file named after the joint.

---

# Object Simulation and Synthetic Data Generation

`isaacsim.replicator.object` (IRO) is a no-code-change-required tool that generates synthetic data for model training that can be used on a range of tasks from retail object detection to robotics. The extension can be run from the UI or the `isaac-sim` container.

It takes a YAML description file that describes a mutable scene, or a hierarchy of such stacked description files as input, and outputs a description file along with graphics content including RGB, 2D/3D bounding boxes, and segmentation masks.

## Motivation

Training deep learning models with synthetic data is in high demand, while 3D software that is used to generate synthetic data often take a long time to learn, including stages such as getting familiar with UI panels. IRO aims at providing you an easy way to compose scenes that are uniquely domain randomized. For example, a typical user for this product is a data scientist without experience in using 3D modeling software, such as Maya and 3ds Max.

In a domain randomization scenario, rather than the actual detailed content in the 3D scene, a data scientist often focuses more on the rules that governs how the scene is randomized, and the relationship among these randomized rules. IRO provides a set of tools, using macros, to abstractly, intuitively, and compactly describe a randomized 3D scene.

## Chat IRO: Natural Language Interface for IRO

Chat IRO is a new extension that lets you describe scenes in plain English and automatically generates IRO description files (YAML). It applies the configuration to the stage, shows an immediate viewport preview, can run simulations, and supports saving and loading YAML files enabling fast, iterative scene authoring without manual YAML editing.

- Chat IRO: Natural Language Interface for Isaac Sim Replicator Object

## End-to-end Pipeline

An end-to-end pipeline is made up of groupings of the larger steps that go into using IRO.

**Acquire Graphics Resources**

To compose a randomized scene, IRO requires imported 3D models to be in USD format. Common 3D formats such as Wavefront OBJ can be converted to USD using [asset converter](https://docs.omniverse.nvidia.com/extensions/latest/ext_asset-converter.html).

**Compose a Description File**

The specifications of a description file is described in this multi-page documentation. It’s recommended that you start with the video guides in [best practices](#best-practices).

**Generate Synthetic Data**

Follow the guidelines below to run IRO.

**Train a CV Model; Deployment and Real-World Application**

An example notebook showing steps to train an object detection model on the synthetic images created using IRO is in TAO 6.0.

## Run from the UI

1. Follow the [Omniverse Extension Manager guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html) to enable the `isaacsim.replicator.object.core` and `isaacsim.replicator.object.ui` extensions.
2. If the extension is successfully enabled, Object SDG panel will be available at the top right, and **Tools** > **Action and Event Data Generation** will have options **Object SDG** and **Distribution Visualizer**.

   If not, disable and enable the extension again. The Object SDG panel is turned on by default, and you can turn it off or on again by **Tools** > **Action and Event Data Generation** > **Object SDG**.

   
3. Click on the folder icon or the Visual Studio Code icon on the right side of the opened extension panel as shown above. The root folder of the extension opens.
4. Under `PATH_TO_CORE_EXTENSION/isaacsim/replicator/object/core/configs` there are many description files in YAML format.
   It’s recommended that you start with `demo_kaleidoscope.yaml`.
5. Go to `global.yaml` and update `output_path` to any local folder where you can store the simulation output.

Note

Select description files from the dropdown below the **Simulate** button. When the extension is loaded, all `.yaml` files in the configs folder will have their names included in this list.

**Placeholders in description files**

Some example description files have placeholders. The paths need to be replaced with valid paths.

For example:

* In `global.yaml` and `minimum.yaml`, replace `PATH_TO_OUTPUT` with a valid path.
* In `demo_bottle.yaml`, replace `PATH_TO_LABEL_IMAGES` with a folder that contains JPEG images.
* For `tutorial_harmonizer_permutate.yaml`, `demo_macro.yaml`, `tutorial_macro.yaml`, `tutorial_scene_graph.yaml` and `tutorial_scene_graph_randomized.yaml` to run:

  > + replace `PATH_TO_ORO` in `global.yaml` with the absolute path of `data/oro_tutorial_models/oro.usd` in the extension’s root folder.
* In `doc_observatory.yaml`:

  > + replace `PATH_TO_OBSERVATORY_SCOPE` with the absolute path of `data/oro_tutorial_models/observatory_scope.usd`
  > + replace `PATH_TO_OBSERVATORY_BASE` with the absolute path of `data/oro_tutorial_models/observatory_base.usd`
  > + replace `PATH_TO_OBSERVATORY_SHAFT` with the absolute path of `data/oro_tutorial_models/observatory_shaft.usd`.
* To make `demo_bin_pack.yaml`, `demo_bins_of_bins_rack_2_layers.yaml`, `demo_bins_of_bins_rack.yaml`, `demo_bins_of_bins.yaml`, `demo_table.yaml` and `demo_transform_operator.yaml` work:

  > + replace `PATH_TO_BOXES` with a folder containing USD files of boxes (or other USDs) in `global.yaml`.
* In `demo_shader_attributes.yaml`,
  :   + replace `PATH_TO_USD` with a path to a USD file.
* In `demo_frustum.yaml`:

  > + replace `PATH_TO_MAIN_OBJECTS` with a folder containing USD files to be used as main objects.
  > + replace `PATH_TO_DISTRACTORS` with a folder containing USD files to be used as distractors.
  > + replace `PATH_TO_BACKGROUND_IMAGES` with a folder containing JPEG images to be used as background images.

You can adjust the scale, if things are not showing up correctly, because different USD files have different sizes.

1. Select `demo_kaleidoscope` from the dropdown box; `demo_kaleidoscope` will appear in the **Description File** text box. You can also use the full absolute path `PATH_TO_CORE_EXTENSION/isaacsim/replicator/object/core/configs/demo_kaleidoscope.yaml` to load a description file.
2. Click **Simulate** to start the simulation. The progress bar will show the simulation progress.

In the above and following content, `PATH_TO_CORE_EXTENSION` varies, for **Isaac on Windows** it is something like `C:\isaacsim\extscache\isaacsim.replicator.object.core-0.x.y\isaacsim\replicator\object\core\configs\demo_kaleidoscope.yaml`, while for **Isaac on Linux** it is something like `~/isaacsim/extscache/isaacsim.replicator.object.core-0.x.y/isaacsim/replicator/object/core/configs/demo_kaleidoscope.yaml`

A guide on how to use the extension is available [here](#best-practices).

## Run from Docker

To install the Isaac Sim Docker container, visit [Container Deployment](Installation.md).

To run the Isaac Sim Docker container:

```python
docker run --gpus device=0 --entrypoint /bin/bash -v LOCAL_PATH:/tmp --network host -it ISAAC_SIM_DOCKER_CONTAINER_URL
```

Accordingly, update `global.yaml` to have `output_path` to be any folder under `/tmp`.

For example, to launch the simulation with `demo_kaleidoscope`:

```python
bash isaac-sim.sh --no-window --enable isaacsim.replicator.object.core --allow-root --/log/file=/tmp/isaacsim.replicator.object.log --/log/level=warn --/windowless=True --/config/file=PATH_TO_CORE_EXTENSION/isaacsim/replicator/object/core/configs/demo_kaleidoscope.yaml
```

`/tmp/isaacsim.replicator.object.log` contains the messages from execution as well as from the extension. You can search the messages from the extension by filtering the file with METROPERF.

Note

If it is not generating anything on the first run inside Docker container, run it again.

## Embedded Interface

When writing graphics content to disk is not needed, the embedded interface is a quick way to prototype a description file.

To use the embedded interface, select a description file, and then click on the **Initialize Scene Randomization** button in the **Object Detection SDG** panel to load the description file. Randomization symbols will be created and connected accordingly. From then on, the scene is randomized per click on the **Randomize Scene** button.

Note

After clicking on the **Initialize Scene Randomization** button and before clicking on the **Randomize Scene** button, it is normal that the viewport is black. To see anything of interest at this stage, press “F” to focus on the selected prim.

To preview physically, click on the triangular **Play** button on the left column of widgets.

## Expected Output

After the simulation, the output is stored in `output_path`. The output content is determined by the [output switches](Synthetic_Data_Generation.md) setting.

For example, the image output of `demo_bottle` is:

While the segmentation output is:

The 2D bounding box is:

```python
bottle_0 0 -1.0 0 1028 333 1362 2159 0 0 0 0 0 0 0
bottle_1 0 -1.0 0 1895 112 2277 1694 0 0 0 0 0 0 0
bottle_2 0 -1.0 0 1281 462 1854 2159 0 0 0 0 0 0 0
```

in which the four positive numbers indicate `x_min`, `x_max`, `y_min`, `y_max`. The number `-1` is where the occlusion rate should be, but because a bottle is transparent, it is `-1` here.

As another example, the image output of `demo_kaleidoscope` is:

While the segmentation output is:

## Concepts

Description File

The description file is a YAML file that has a main key named `isaacsim.replicator.object`.

The description file consists of key-value pairs. Each key-value pair is a [Mutable](Synthetic_Data_Generation.md), a [Harmonizer](Synthetic_Data_Generation.md), or a [Setting](Synthetic_Data_Generation.md).

The description file generates frames as specified. Each frame the scene is randomized, [graphics content](Synthetic_Data_Generation.md) is captured, and output to disk. [Settings](Synthetic_Data_Generation.md) describe how the scene is configured and how data is output. For example, you can set the number of frames to output, whether or not to output 2D bounding boxes, or set the gravity and friction of physics simulation.

The description file populates the scene with objects that are called [mutables](Synthetic_Data_Generation.md).

Mutables randomize every frame. Sometimes you might want to constrain how they randomize. For example, to know how other mutables are randomizing and randomize correspondingly. To do so, define [harmonizers](Synthetic_Data_Generation.md).

Example Minimal Description File Definition

```python
isaacsim.replicator.object:
   version: 0.x.y
   num_frames: 3
   output_path: OUTPUT_PATH
   screen_height: 1080
   screen_width: 1920
   seed: 0
```

Simulation Workflow

Every time a simulation is launched, an initialization stage happens in the beginning, and a per-frame simulation stage happens every frame.

In the initialization stage, the description file is parsed by a description parser. Symbols are created for every [mutable attribute](Synthetic_Data_Generation.md) that requires a resolution to get its actual value. These symbols will resolve to actual values when they are used to interact with the USD scene once, after they are initialized; and also in every per-frame simulation.

Each time a symbol is resolved, the dependent symbols of it are also recursively resolved. If an unresolved harmonized mutable attribute is met, the parser enters `AWAITING_HARMONIZATION` status, and then the [harmonizers](Synthetic_Data_Generation.md) harmonizes (collect information from the `pitch` attribute and randomize), and propagate output back to harmonized mutable attributes. After all harmonized mutable attributes are resolved, the parser will be out of `AWAITING_HARMONIZATION` status.

After this, the resolved values are used to update the USD scene. If gravity is turned on, physics is resolved so that objects move away from each other when they overlap or drop onto a surface (for more details, see [physics simulation explained](Synthetic_Data_Generation.md)). And [graphics content](Synthetic_Data_Generation.md) is captured. Eventually, the state of the scene in this frame is recorded and saved, such that later on, it can be restored or inspected.

More details can be found in [harmonization example](Synthetic_Data_Generation.md).

Scene Restoration

To support multiple-sampling for pretrained models:

In the output content, you can use the output saved from logging of a specific frame to generate the exact same graphics content as when this frame was generated. Or you can slightly modify it to have something different but everything else is the same.

## Main Simulation Workflow Walkthrough

Here is a walkthrough on how to run the main simulation workflow.

The first step is to set the description files. Turn on the extension manager, search for `isaacsim.replicator.object.core`, and click on the **Open Extension Folder** button, as shown below.

Note

If `isaacsim.replicator.object.core` and/or `isaacsim.replicator.object.ui` are not enabled, click on the capsule icons to enable them.

In the folder, go to `PATH_TO_CORE_EXTENSION/isaacsim/replicator/object/core/configs`. On Windows, the folder is opened after the **Open Extension Folder** button is clicked. On Linux, it can bring up the browser with the URL as `file://EXTENSION_PATH`, in this case, navigate to `EXTENSION_PATH` using the command line or `xdg-open`.

Edit the `global.yaml` file. Set `OUTPUT_PATH` to a folder where you want to store the output. Also, update `PATH_TO_BOXES` to a folder that contains USD files of boxes.

Select `demo_table` from the dropdown box, and click on the **Simulate** button. The simulation will run, and the output will be stored in the folder specified by `OUTPUT_PATH`.

## Compose a Description File

To compose a description file that generates a scene that has a table with randomized objects dropping onto it:

Suppose we have the following assets:

* an HDRI texture for the dome light at `PATH_TO_HDRI`
* a USD model as a table at `PATH_TO_TABLE`
* a folder that contains USD models of objects to be scattered onto the table at `PATH_TO_OBJECTS`

Plan the distribution of graphics assets before composing a description file. The assets are dragged into the viewport, to get an idea of them, refer to the image. Here a dome light is created, and its texture is set to `PATH_TO_HDRI`; then a table from `PATH_TO_TABLE`; then one of the objects from `PATH_TO_OBJECTS`.

Adjust the position of the object about to be scattered onto the table, for a reasonable range of its position.

From observation, from `(-13, 100, -70)` to `(13, 100, 70)` is a reasonable range for the position of the box. Compose a description file as follows:

```python
isaacsim.replicator.object:
  # the minimum
  version: 0.x.y
  num_frames: 3
  seed: 0
  output_path: PATH_TO_OUTPUT
  screen_height: 2160
  screen_width: 3840

  # physics parameters
  gravity: 10000
  friction: 0.3
  simulation_time: 10
  linear_damping: 4

  # light
  bright_light:
    type: light
    subtype: dome
    intensity: 1000
    transform_operators:
    - rotateX: 270
    texture_path: PATH_TO_HDRI

  # camera; transforms page has more details on how to construct a list of transform operators
  focal_length: 14.228393962367306
  horizontal_aperture: 20.955
  camera_parameters:
    screen_width: $[/screen_width]
    screen_height: $[/screen_height]
    focal_length: $[/focal_length]
    horizontal_aperture: $[/horizontal_aperture]
    near_clip: 0.001
    far_clip: 100000
  default_camera:
    type: camera
    camera_parameters: $[/camera_parameters]
    transform_operators:
    - translate:
      - 0
      - 50
      - 0
    - rotateY:
        distribution_type: range
        start: -180
        end: 180
    - rotateX: -30
    - translate_local:
      - 0
      - 0
      - 400

  # boxes
  box:
    count: 10
    type: geometry
    subtype: mesh
    physics: rigidbody # reacts to gravity, collisions, etc.
    tracked: true # if true, the bounding boxes, segmentation, etc. will be recorded in the output
    usd_path:
      distribution_type: folder
      value: PATH_TO_OBJECTS
      suffix: usd
    transform_operators:
    - translate: # as planned
        distribution_type: range
        start:
        - -13
        - 100
        - -70
        end:
        - 13
        - 100
        - 70
    - rotateXYZ:
        distribution_type: range
        start:
        - -180
        - -180
        - -180
        end:
        - 180
        - 180
        - 180
    - scale:
      - 0.2
      - 0.2
      - 0.2

  table:
    type: geometry
    subtype: mesh
    physics: collision # rigidbodies will collide with it, but it doesn't move
    usd_path: PATH_TO_TABLE
    transform_operators:
    - rotateX: -90
```

Run the simulation by clicking on the **Simulate** button to generate RGB images like:

And segmentation masks like:

Note

Check whether the YAML text is formatted correctly in the description (for example, indentation). If you meet an error `mapping values are not allowed here` it can be due to a formatting problem.

## Scene Editing

For the convenience of scene planning, a basic scene editing widget is provided to toggle the visibility of prims.

In a scene created by the [embedded interface](#embedded-interface) using [this description file](Synthetic_Data_Generation.md), you can create a cube, change its translate and size (but not its rotation), and move it around to toggle visibility of prims that has its position included within the spatial range of the cube, as shown below:

Note

Before clicking on the **Toggle Visibility of selected region** button, make sure the cube is selected.

## Catalog

Conventions in the linked catalog files:

`Type` in the tables indicates the expected data types. Where a type is expected, a macro string can be used for later evaluation of that specific type. For example, if you expect int in a value, you can either give an int or something like `$[index]`. See [Macro](Synthetic_Data_Generation.md) for details.

Within a mutable, aside from these options, you can also specify a [Mutable Attribute](Synthetic_Data_Generation.md) to evaluate to this type.

`numeric` means literal or evaluated `float` or `int`.

- Setting
- Mutable
- Camera
- Geometry
- Light
- Mutable Attribute
- Transformation
- Harmonizer
- Macro
- Distribution Visualizer
- Randomization Dependency: Incremental Examples

## 3rd-party Libraries Used

py3dbp (modified), MIT License
PyYaml, MIT License
trimesh, MIT License
regex, Apache License

---

# Chat IRO: Natural Language Interface for Isaac Sim Replicator Object

Vision-language and scene-generation workflows often require users to hand‑write
YAML configuration files for [Isaacsim.replicator.object](Synthetic_Data_Generation.md) (IRO).
This can be error‑prone and slow, especially for complex layouts, harmonizers,
physics setups, and camera rigs.

`Chat IRO` is a natural‑language interface that converts plain English
descriptions into executable IRO YAML configurations and runs them directly
inside Isaac Sim. It sits on top of the IRO extension and automates
configuration authoring, validation, and execution.

Chat IRO has the following features:

* Convert English descriptions into IRO YAML scenes.
* Use a Retrieval‑Augmented Generation (RAG) system with thousands of
  production YAML examples to improve correctness and reuse best practices.
* Validate generated YAML for syntax and common structural issues before
  execution.
* Preview the generated scene immediately in the Isaac Sim viewport.
* Save and reload configuration files for iterative workflows.

Note

This extension is not available on Linux aarch64 in Isaac Sim 6.0 EA. Support will be added in Isaac Sim 6.0 GA.

## Workflow

Chat IRO uses the following workflow to generate scenes:

1. You type a natural‑language request such as
   `Create a scene with 10 random size and color cubes` into the
   Chat IRO window.
2. The extension optionally queries its RAG index of existing IRO YAML files
   and injects relevant examples into the LLM context.
3. The LLM generates a candidate YAML configuration for
   `isaacsim.replicator.object`.
4. Chat IRO validates the YAML, fixes common issues, and executes it through
   IRO to create or update the scene.
5. The resulting synthetic scene is rendered in the viewport. You can
   iteratively refine the configuration by sending follow‑up prompts.

### Prerequisites

Before using Chat IRO, ensure the following requirements are met:

* `isaacsim.replicator.object.ui` extension enabled
* A supported operating system (Linux is the primary platform; Windows is
  experimental).
* An NVIDIA GPU with CUDA support (recommended).
* At least 8 GB of RAM (16 GB or more is recommended for large scenes).
* The `omni.ai.langchain.agent.chat_iro` extension enabled.
* A valid NVIDIA API key for LLM access.

Note

The LLM features require a valid NVIDIA API key and sufficient
credits. Visit the [NVIDIA API portal](https://build.nvidia.com) to
obtain a key and manage credits. See the [NVIDIA API reference page](https://docs.api.nvidia.com/nim/reference/llm-apis) for more details.

## Enable `Chat IRO` Extension

1. Follow the [Omniverse Extension Manager guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html)
   to enable the `omni.ai.langchain.agent.chat_iro` extension.
2. Launch Isaac Sim and open the Extension Manager if it is not already open:

   * In the main menu, select **Window > Extensions**.
   * Search for `Chat IRO`.
   * Enable the extension and optionally enable **AUTOLOAD** so it is loaded
     automatically on future launches.
3. Configure the NVIDIA API key by setting it as an environment variable.

   **Linux/macOS**

   ```python
   # Set API key for the current shell session
   export NVIDIA_API_KEY="nvapi-YOUR-KEY-HERE"

   # Make the setting persistent (for bash)
   echo 'export NVIDIA_API_KEY="nvapi-YOUR-KEY-HERE"' >> ~/.bashrc
   source ~/.bashrc
   ```

   **Windows (Command Prompt)**

   ```python
   REM Set API key for the current Command Prompt session
   set NVIDIA_API_KEY=nvapi-YOUR-KEY-HERE

   REM To make the setting persistent, add the variable in
   REM System Properties > Environment Variables.
   ```

Note

If LLM authentication fails, verify that `NVIDIA_API_KEY` is set
and has remaining credits.

### Accessing the Chat IRO Panel

Once the extension is enabled:

1. Open the main Chat IRO window:

   * From the menu bar, select **Window > Chat IRO**.
   * A dockable Chat IRO panel opens, typically on the right side of the
     viewport.
2. Select a model from the **Model** drop‑down menu. Verified working models include:

   * `meta/llama-4-maverick-17b-128e-instruct` (recommended, 256K context, default)
   * `qwen/qwen3-next-80b-a3b-instruct` (128K+ context)
3. Confirm that the status line in the Chat IRO panel indicates that the
   model is ready and the extension is authenticated.

### Using Chat IRO

Chat IRO can be used in the following ways:

* [Using the UI panel](#chat-iro-using-ui-panel)
* [Generating new IRO scenes](#chat-iro-generate-scenes)
* [Editing existing IRO YAML files](#chat-iro-edit-yaml)

### Using the UI Panel

To create and preview scenes using the Chat IRO panel:

1. In the Chat IRO input box, type a prompt such as:

   ```python
   Create a scene with 7 cubes and 6 spheres. All objects are randomly positioned, random color, and sized.
   ```
2. Press `Enter` to send the prompt.
3. Chat IRO retrieves relevant YAML patterns from its RAG index, generates
   an IRO configuration, validates it, and executes it in Isaac Sim.
4. Inspect the viewport to verify that the generated scene matches the
   requested behavior (object counts, colors, positioning, lighting, and
   camera placement).

   
5. Refine the scene with follow‑up prompts that modify the existing
   configuration. For example:

   ```python
   Make all cubes blue and add rigidbody physics
   ```

   The extension updates the YAML configuration in place, reapplies it, and
   refreshes the viewport.
6. By default, configuration files are automatically stored in a directory similar to:

   `~/Documents/ChatIRO_Results/config_files/my_scene.yaml`

   You can also specify a custom path by asking Chat IRO to save the file to a different location.

### Generating New IRO Scenes

Chat IRO is optimized for generating complete IRO scenes from concise,
well‑specified prompts. Good prompts include:

* `Create 20 purple cubes arranged in a circular formation with radius 900 at Y = 50.`
* `Pack 8 cubes and 6 spheres scaled 1.2x into a bin sized (300, 400, 500) at (5, 0, 0).`

For example, the following prompt:

```python
Create a scene with 7 cubes and 6 spheres. All objects are randomly positioned,
random color, and sized.
```

will typically produce an IRO configuration similar to:

```python
isaacsim.replicator.object:
  version: 0.10.0
  parent_config: standard
  seed: 42
  num_frames: 10
  output_path: /Documents/ChatIRO_Results
  screen_height: 2160
  screen_width: 3840
  focal_length: 14.228393962367306
  horizontal_aperture: 20.955

  camera_parameters:
    screen_width: $[/screen_width]
    screen_height: $[/screen_height]
    focal_length: $[/focal_length]
    horizontal_aperture: $[/horizontal_aperture]
    near_clip: 0.001
    far_clip: 100000

  cube:
    count: 7
    type: geometry
    subtype: cube
    tracked: true
    color:
      distribution_type: range
      start:
      - 0
      - 0
      - 0
      end:
      - 1
      - 1
      - 1
    transform_operators:
    - rotateX: 0
    - rotateY: 0
    - rotateZ: 0
    - translate:
        distribution_type: range
        start:
        - -500
        - 50
        - -500
        end:
        - 500
        - 50
        - 500
    - scale:
        distribution_type: range
        start:
        - 0.5
        - 0.5
        - 0.5
        end:
        - 1.5
        - 1.5
        - 1.5

  sphere:
    count: 6
    type: geometry
    subtype: sphere
    tracked: true
    color:
      distribution_type: range
      start:
      - 0
      - 0
      - 0
      end:
      - 1
      - 1
      - 1
    transform_operators:
    - rotateX: 0
    - rotateY: 0
    - rotateZ: 0
    - translate:
        distribution_type: range
        start:
        - -500
        - 50
        - -500
        end:
        - 500
        - 50
        - 500
    - scale:
        distribution_type: range
        start:
        - 0.5
        - 0.5
        - 0.5
        end:
        - 1.5
        - 1.5
        - 1.5

  default_camera:
    camera_parameters: $[/camera_parameters]
    transform_operators:
    - rotateX: -30
    - rotateY: 45
    - rotateZ: 0
    - translate:
      - 0
      - 0
      - 1000
    - scale:
      - 1
      - 1
      - 1
    type: camera

  dome_light:
    intensity: 1500
    subtype: dome
    transform_operators:
    - rotateX: 270
    type: light
```

#### More Prompt Examples

Use these prompts to explore richer scenes:

**Bin packing**

```python
Create a scene that packs 8 spheres and 10 cubes scaled 1.2 times
into a bin sized (300, 400, 500) at position (5, 0, 0)
```

**Grid layout**

```python
Create 25 cubes arranged in a 5x5 grid with spacing of 100 units
```

**Physics**

```python
Create 20 spheres with rigidbody physics falling from height 500
onto a ground plane
```

Note

Complex mathematical layouts (for example, circular or grid‑based
arrangements) may require a few iterations. If object placement does not
match expectations, use a follow‑up prompt that focuses only on correcting
the formulas or spacing.

#### Using Existing USD Scenes

You can also reference existing USD stages or assets in your prompts:

**Populate a warehouse stage**

::

```python
Create a warehouse scene from the USD stage at /home/user/Assets/warehouse.usd and populate the shelves
with 50 random sized boxes and 10 pallets
```

**Enhance a robot lab**

::

```python
Create a robot lab scene from the USD stage at /home/user/Assets/robot_lab.usd and add 5 cubes and 5 spheres
on the main table with random colors and sizes
```

### Editing Existing IRO YAML Files

Chat IRO can also load and modify YAML configuration files that you have
created manually or with other tools.

Typical workflow:

1. Ask Chat IRO to load a file:

   ```python
   load /home/user/Documents/ChatIRO_Results/config_files/my_scene.yaml
   ```
2. Inspect the generated scene in the viewport.
3. Apply edits using natural language, such as:

   ```python
   Add 5 more cubes with random colors.

   Increase dome light intensity to 3000.

   Add a rotating camera that orbits the scene 360 degrees.
   ```
4. Save the updated configuration:

```python
save /absolute/path/to/my_scene_v2.yaml
```

Behind the scenes, Chat IRO reuses the same validation and execution pipeline
used for newly generated configurations.

### Managing Output Files and Directories

By default, Chat IRO saves generated configuration files and simulation
outputs to a structured directory under your home folder.

#### Default Output Location

All Chat IRO outputs are organized in:

```python
~/Documents/ChatIRO_Results/
├── config_files/              # YAML configuration files
├── simulation_results/        # IRO simulation outputs
└── .cache/                    # Temporary files (hidden)
```

* The `config_files/` directory contains YAML files that define scenes.
* The `simulation_results/` directory contains rendered images, sensor
  data, and other outputs generated when executing the YAML configurations.
* The `.cache/` directory stores temporary processing files.

Note

If `~/Documents/ChatIRO_Results/` does not exist, Chat IRO creates it
automatically on first use.

#### Changing the Output Directory

You can change the default output directory with an environment variable:

```python
# Linux/macOS
export CHAT_IRO_OUTPUT_DIR="~/MyProjects/IRO_Results"

# To make it persistent, add to your shell startup file, for example:
echo 'export CHAT_IRO_OUTPUT_DIR="~/MyProjects/IRO_Results"' >> ~/.bashrc
source ~/.bashrc
```

```python
REM Windows (Command Prompt)
set CHAT_IRO_OUTPUT_DIR=C:\Users\YourName\IRO_Results

REM Add to System Environment Variables for persistence
```

Note

Advanced users can also configure the default output directory in the
Chat IRO extension settings or via the Python APIs that ship with the
extension.

#### Natural‑Language File Commands

Chat IRO understands simple text commands for loading, saving, and running
configurations:

**Loading files**

```python
load /path/to/my_scene.yaml
```

**Saving files**

```python
save /absolute/path/to/my_warehouse_scene.yaml

save this as /absolute/path/to/production_config.yaml
```

**Simulating with specific parameters**

```python
simulate with seed 123
```

Note

For reliable behavior, always specify an absolute path when saving, for example:
`save /absolute/path/to/my_scene.yaml`. Using only a file name (for example,
`save my_scene.yaml`) is not recommended because the save location can vary
depending on your environment and configuration.

### Chat IRO RAG Configuration

Chat IRO includes a Retrieval‑Augmented Generation system that provides deep
knowledge of existing IRO scenes and best‑practice configurations.

The behavior of the RAG system can be customized in `extension.toml`:

```python
[settings.exts."omni.ai.langchain.agent.chat_iro"]
enable_rag = true                  # Enable/disable RAG (default: true)
rag_top_k = 15                     # Number of documents to retrieve
rag_max_tokens = 8000              # Maximum tokens for RAG context
enable_multi_query_rag = true      # Enable multi‑query decomposition
max_sub_queries = 3                # Maximum number of sub‑queries

# Optional cross‑encoder reranking
enable_rag_reranking = false
reranker_model = "BAAI/bge-reranker-large"
```

When enabled, RAG allows Chat IRO to:

* Break complex prompts into multiple focused sub‑queries.
* Retrieve relevant YAML snippets for geometry, harmonizers, and cameras.
* Merge and rerank results to provide higher‑quality configurations.

Note

Enabling cross‑encoder reranking typically improves retrieval accuracy by
10–30% at the cost of additional latency (around 100–200 ms per request).
For simple prompts or low‑latency environments, keep
`enable_rag_reranking = false`.

### Best Practices

Chat IRO relies on LLMs that interpret natural language. Clear, specific
prompts lead to more reliable IRO configurations.

Recommended prompting guidelines:

* Specify concrete numbers rather than vague terms.

  *Good:* `Create 20 cubes in a circular formation with radius 900 at Y = 50.`

  *Avoid:* `Create some objects in a circle.`
* Explicitly describe sizes, positions, and physics requirements.
* Build scenes iteratively and validate each step in the viewport.
* Save working configurations frequently and version them as you refine.

If the generated YAML does not execute or the scene appears empty:

* Ask Chat IRO to regenerate with corrected structure, for example:

  ```python
  Regenerate the configuration using valid YAML syntax and complete
  all missing parameters.
  ```
* Focus corrective prompts on specific errors (spacing, rotations, counts,
  physics flags) instead of rewriting the entire scene.

### Troubleshooting

Common issues and remedies:

**LLM authentication failed**

* Symptom: Error message about missing or invalid API key; no YAML generated.
* Action: Verify `NVIDIA_API_KEY` in your environment and `extension.toml`,
  confirm that your account has remaining credits, and restart Isaac Sim.

**No scene is rendered**

* Symptom: Chat IRO responds, but the viewport remains empty.
* Action:

  + Inspect the generated YAML in the Chat IRO window.
  + Look for error messages in the Isaac Sim console or logs.
  + Try a simple prompt such as `Create 5 cubes` to verify basic behavior.

**YAML syntax errors**

* Symptom: Messages such as `Failed to parse YAML`.
* Action:

  + Ask Chat IRO to fix the YAML syntax.
  + Simplify the prompt and ensure that you request a single, self‑contained
    configuration.

**Slow responses**

* Symptom: Noticeable delay between sending a prompt and receiving an answer.
* Action:

  + Reduce `rag_top_k` and disable reranking in `extension.toml`.
  + Split very complex scenes into multiple, smaller prompts.

### Session Management

Over very long sessions, the LLM may drift from the original constraints or
produce inconsistent configurations.

To reset the conversation:

* Click the \(+\) button in the upper‑left corner of the Chat IRO window to
  start a new session.
* Optionally restart Isaac Sim if behavior remains inconsistent.
* Begin the new session with a clear instruction such as:

  ```python
  You are a YAML configuration generator for Isaac Sim Replicator Object.
  Generate only valid YAML with proper structure. Create a scene with
  10 cubes in a grid layout.
  ```

---

# Setting

If the key-value pair in the description file is neither a [Mutable](Synthetic_Data_Generation.md) nor a [Harmonizer](Synthetic_Data_Generation.md), it’s a setting. You can define a description with only settings.

## Required Keys

There are several required keys for settings:

| Required Key | Type | Description |
| --- | --- | --- |
| output\_path | string | The output folders in which folders corresponding to  each [output switch](#output-switches) are created |
| num\_frames | int | Number of frames to output |
| screen\_width | int | Screen width of output images |
| screen\_height | int | Screen height of output images |
| seed | int | Global randomization seed |
| version | string | Version number of isaacsim.replicator.object.core |

output\_switches

The setting output\_path controls what is output to disk per frame. It has these switches:

| Switch | Data |
| --- | --- |
| images | The RGB image of the frame |
| labels | 2D tight bounding box and the occlusion rate information for each visible tracked object. Each line corresponds to an object, and it has Kitti format `usd_base_name 0 occlusion 0 x_min y_min x_max y_max 0 0 0 0 0 0 0` |
| 3d\_labels | 3D bounding box information stored as Objectron format |
| descriptions | A description file logging the current state of the scene - Using this file as input description, the same graphics content is output |
| segmentation | The segmentation mask of tracked mutables |
| depth | The depth map of the scene, showing the distance to image plane |
| normal | The normal map of the frame |

Setting a switch to True or not setting the switch creates the corresponding folder under `output_path` and writes corresponding data into it.

`usd_base_name` is the mutable name or the USD file base name of USD file when a geometry `mesh` is loaded, which means it’s not allowed to load different USD files with the same base name. Using `${resource_root_1}/apple.usd` and `${resource_root_1}/inner/apple.usd` in the same simulation causes unexpected behavior.

For example, an output switch could be:

```python
output_switches:
  images: True
  labels: True
  descriptions: True
  3d_labels: True
  segmentation: True
  depth: False
```

## Optional Keys

There are also optional keys, where if not set, have default values:

| Optional Keys with Default Value | Type | Default value | Description |
| --- | --- | --- | --- |
| parent\_config | string | None | Specifies the description file that this description file inherits from, in the same parent folder. Values re-defined in the current description file override values defined in parent configs. |
| path\_tracing | bool | False | Render mode selection |
| inter\_frame\_time/simulation\_time | numeric | 0 | The simulation time between 2 frames |
| extra\_rendering\_time | numeric | 0 | Extra rendering time per frame |
| output\_name | string | `$[seed]_$[camera]` | The output name of a frame that can be customized. Seed, camera, and frame macros are available. |
| skip\_frames\_with\_no\_visible\_tracked\_mutables | bool | False | If set to true, and if there are no visible tracked mutables in the scene, the frame is skipped |
| gravity | numeric | 0 | Resolves gravity during [physics resolution stage](Synthetic_Data_Generation.md) |
| friction | numeric | 1 | Friction among objects during physics resolution stage. Lower values indicate that the object is more slippery. |
| linear\_damping | numeric | 0 | Linear damping of objects during physics resolution stage. |
| angular\_damping | numeric | 0 | Angular damping of objects during physics resolution stage. |
| occlusion\_threshold | numeric | 1 | If the occlusion of an object is bigger than this threshold, the object will be skipped in the labels. |
| max\_area\_threshold | numeric | None | If the bounding box area of an object as a percentage of the screen area is bigger than this threshold, the object will be skipped in the labels. |
| min\_area\_threshold | numeric | None | If the bounding box area of an object as a percentage of the screen area is smaller than this threshold, the object will be skipped in the labels. |

**Suggestions and More Information**

**path\_tracing**

Turning it on uses the path tracer, which makes simulation slower but image quality higher. Turning it off uses real-time RTX.

**inter\_frame\_time/simulation\_time and extra\_rendering\_time**

For complex scenes, leave more time for physics resolution and rendering.

**output\_name**

Three macros are available:

* $[seed] evaluates to the seed of the current frame
* $[camera] evaluates to the camera name
* $[frame] evaluates to the frame index. Refer to seed for details.

**$[seed]**

Each frame is randomized with its own seed, which equals the global seed plus the frame index. For example, if global seed is `2`, and three images are output, the frame indices for these three images are `0, 1, 2`; and the seeds are `2, 3, 4`, respectively.

**Physics simulation explained**

When objects are randomized in the scene for each frame, they can start at an overlapping position. Resolution of physics de-penetrates these objects. The de-penetration accelerates the objects, such that they can start off with a high speed. Increase linear/angular damping to keep object movement contained.

However, if linear or angular damping is set too high, objects get lazy and they don’t move much. This can be bad in a gravity enabled setting, where we want objects to be in close contact with a surface. Because different objects have different sizes and shapes, it’s good to tune these physics properties to reach a good appearance.

Similarly, too high of a value for friction makes objects cluster if they are in close contact; while too low of a value for friction makes them slippery and glide off surfaces.

Note

If there is no object in the scene when you are expecting some objects, one reason might be that they flew away from the view frustum. Check your physics settings.

---

# Mutable

If the pair is a dictionary with a key `type`, it’s a mutable. There are three types of mutables:

* [Camera](Synthetic_Data_Generation.md) where we are
* [Geometry](Synthetic_Data_Generation.md) the things we want to observe
* [Light](Synthetic_Data_Generation.md) how we observe things

Each mutable consists of attributes. Each key-value pair of a mutable is an attribute. An attribute can be a [Mutable Attribute](Synthetic_Data_Generation.md), which mutates per frame.

Available attributes of mutables are:

| Name | Type | Description |
| --- | --- | --- |
| type | string | The type of the mutable, `camera`, `geometry`, or `light`. |
| count | int | The number of identically defined mutables |
| tracked | bool | If the mutable is tracked, its 2d/3D bounding boxes will be output, and it will have a corresponding highlighted color on the segmentation mask. |
| transform\_operators | list | The transformation of the object. |

**Transform operators**

Specially, to define its pose in space, the mutable can define a sequenced list of [transform operators](Synthetic_Data_Generation.md). A transform operator is also a key-value pair, in which the value can be a mutable attribute.

**Shader attributes**

In Omniverse, a shader has many attributes describing how a mesh is shaded. For example, `diffuse_texture` that points to the RGB image, and `texture_rotate` that specifies how its texture should be rotated. In ORO, you can control these attributes just like any other mutable attributes. For example, the following description randomizes the tint, and rotates and scales the texture:

```python
mesh:
  type: geometry
  subtype: mesh
  usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Residential/Furniture/Desks/Desk_01.usd
  transform_operators:
  - rotateXYZ:
    - -90
    - 0
    - 0
  shader_attributes:
    texture_rotate:
      distribution_type: range
      start: -180
      end: 180
    diffuse_tint:
      distribution_type: range
      start:
      - 0
      - 0
      - 0
      end:
      - 2
      - 2
      - 2
    texture_scale:
      distribution_type: range
      start:
      - 0.2
      - 0.2
      end:
      - 0.7
      - 0.7
```

Specifically, you can do common computer vision operations, such as color mapping to a mesh that has an RGB image as diffuse texture, by doing:

```python
shader_attributes:
  diffuse_texture:
    distribution_type: texture
    operation: color_map
```

Available options are: `color_map`, `transform`, `add_noise`, `apply_blur`, `color_shift`, `invert_color`, `sobel_edges`, and `random_mutation`.

**Name, count, and index**

A mutable has a `name`, which is the key in the key-value pair and potentially an index, if defined in group with a `count` attribute. For example:

```python
mesh:
  count: 2
  ...
```

During initialization stage, two mutables are initialized after the description is parsed. For example:

```python
mesh_0:
  count: 2
  index: 0
  name: mesh_0
  ...

mesh_1:
  count: 2
  index: 1
  name: mesh_1
  ...
```

The `count` is still there so that you can access how many mutables are in the group. You can use these values to define macros.

---

# Camera

If a mutable has an attribute `type` of `camera`, it’s a camera. Typically, a basic pinhole camera model is used.

Required attributes of a camera:

| Name | Type |
| --- | --- |
| camera\_parameters | dict |

`camera_parameters` is a dictionary with the following six required keys:

| Name | Type |
| --- | --- |
| screen\_width | int |
| screen\_height | int |
| focal\_length | numeric |
| horizontal\_aperture | numeric |
| near\_clip | numeric |
| far\_clip | numeric |

## Pinhole Model

3D objects are projected onto a 2D plane, like this:

Looking from a top view, towards the negative Y-axis:

In the picture, f, which is the distance from the camera to the projection plane, is `focal_length`. hA is the distance from the left edge (upper end) to the right edge (lower end) and stands for `horizontal_aperture`.

`near_clip` and `far_clip` define two planes perpendicular to the line of vision between which you can observe things.

## Assumptions

The following assumptions are made for the frame of reference and conversion from/to other common representations.

If no transform operator is applied on the camera, the Y-axis points upwards and X-axis points to the right. The camera looks towards negative direction of the Z-axis.

---

# Geometry

If a mutable has attribute `type` of `geometry`, it’s a geometry. A geometry is a substance in space.

Available attributes of `geometry`:

| Name | Type | Description |
| --- | --- | --- |
| subtype | string | refer to Basic shapes, Deformed shape, and Mesh loaded from USD |
| physics | string | `collision` or `rigidbody` |
| is\_instance | bool | whether the geometry is instanced - default to true; required to be false for shader attribute randomization |

If `physics` is set to `rigidbody`, the object is a dynamic object that responds to physics. If it’s set to `collision`, the object is a static object that dynamic objects interact with. For example, a wall can have `collision` and a ping-pong ball bouncing off of it has `rigidbody`.

**Basic shapes**

If `subtype` is one of `cone`, `cube`, `cylinder`, `disk`, `torus`, `plane`, or `sphere` it defines the corresponding basic geometry.

**Deformed shape**

Physics simulation for bottles is not supported.

If `subtype` is `bottle`, it defines a bottle shape, which is a parameterized deformed geometry controlled by the following parameters:

| Name | Type | Description |
| --- | --- | --- |
| base\_effector | string | vertical position of the base effector |
| neck\_effector | string | vertical position of the neck effector |
| horizontal\_effector | string | horizontal position of the body effector |
| vertical\_effector | string | vertical position of the body effector |

This image illustrates how the shape of the bottle is controlled by these four effectors.

Note

We currently don’t yet have collision detection for deformed shapes; they don’t have physics.

**Mesh loaded from USD**

If `subtype` is `mesh`, it defines a mesh loaded from USD.

Additional attribute of mesh:

| Name | Type | Description |
| --- | --- | --- |
| usd\_path | string | The path to the USD |

---

# Light

If a mutable has attribute `type` of `light`, it’s a light. There are directional lights and dome lights. A light has an `intensity` attribute. Specifically, a dome light has a `texture_path` attribute.

Aside from ordinary attributes of mutables, additional available attributes of lights are:

| Name | Type |
| --- | --- |
| intensity | numeric |
| color | numeric, dimension three list |

**Direct light**

In the default pose, a direct light shines towards negative Z-direction.

**Dome light**

A dome light has light beams coming from all directions.

Additionally, available attributes of dome light are:

| Name | Type |
| --- | --- |
| texture\_path | string |

`texture_path` is a path to a spherical image that is a skybox. It has a color value in all directions, so that when the camera is rotated, you can observe different parts of the image.

---

# Mutable Attribute

A value in the description file is a mutable attribute if it is a dictionary that has a key `distribution_type`, or a string that contains [macros](Synthetic_Data_Generation.md) (`$[...]`). A mutable attribute does not have to be part of a mutable; you can have standalone mutable attributes.

You can define mutable attributes with `distribution_type` as `folder`, `set`, `range`, `frustum`, and `harmonized`.

**Folder**

A `folder` mutable attribute uniformly samples a file from the specified folder with the specified suffix. To define a `folder` type, there are two additional required keys, `suffix` and `value`.

```python
distractor:
  type: geometry
  subtype: mesh
  usd_path:
    distribution_type: folder
    suffix: usd
    value: $[/resources_root]/distractors
```

In this example, a geometry named `distractor`, which is a mesh loaded from a USD file, is defined. And the USD file is randomly selected from all files in `$[/resources_root]/distractors` that has a `.usd` extension.

Note

Some example description files have [placeholders](Synthetic_Data_Generation.md).

**Set**

A `set` attribute randomly selects a value from a set.

```python
dome_light:
  type: light
  subtype: dome
  texture_path:
    distribution_type: set
    values:
    - $[/skies]/adams_place_bridge_4k.hdr
    - $[/skies]/autoshop_01_4k.hdr
```

In this example, a dome light is defined with a texture of either `$[/skies]/adams_place_bridge_4k.hdr` or `$[/skies]/autoshop_01_4k.hdr`, selected randomly.

**Range**

A `range` attribute specifies the range of randomization for a numeric value.

```python
dome_light:
  type: light
  subtype: dome
  intensity:
    distribution_type: range
    start: 1000
    end: 3000
```

Here the dome light defined has an intensity as a random number within `[1000, 3000]`.

**Camera frustum**

A `camera_frustum` attribute is specially used for sampling a value for the translate operator (Refer to [Transformation](Synthetic_Data_Generation.md)). It samples a position in a view frustum defined by `camera_parameters`, which is the same as in [Camera](Synthetic_Data_Generation.md).

```python
main_object:
  ...
  transform_operators:
  - translate:
      distribution_type: camera_frustum
      camera_parameters: $[/camera_parameters]
      distance_min: 200
      distance_max: 600
      screen_space_range: 0.5
```

`distance_min` and `distance_max` are the minimum and maximum distance from the view point. `screen_space_range` is the range in screen space on which to scatter objects. For example, if you set it to `0.5`, the objects are only scattered in the space projected to the area specified within the dotted lines:

Camera frustum doesn’t scatter objects uniformly along the line of vision. It’s scattered more often in the near field and the far field, such that the probability density of projected area is constant. For example, below is a uniformly sampled in (a) while sampling more in the near field in (b). In (b), the projected areas are more evenly spaced compared to (a).

For the same object, it’s more likely to be sampled near `distance_min` than `distance_max` such that a position that gives a projection ten pixels wide has the same possibility to be sampled with a position that gives a projection twenty pixels wide.

Such a distance is given by:

\[distance = \frac{distanceMin \cdot distanceMax}{distanceMin + (distanceMax - distanceMin) \cdot randomUnit}\]

in which \(randomUnit\) is uniformly sampled within `[0,1]`.

**Harmonized**

A `harmonized` attribute defines an attribute that retrieves its value from a [Harmonizer](Synthetic_Data_Generation.md) after [harmonization](Synthetic_Data_Generation.md). More details can be found in the [harmonization example](Synthetic_Data_Generation.md).

---

# Transformation

This page discusses how to move things around.

The position, orientation, and size of an object in the scene must be defined by a sequence of transform operators (also known as a scene graph). This sequence is ordered such that global transforms are towards the top, while local transforms are towards the bottom. If you are not familiar with what “global” and “local” means, here is an example:

## Scene Graph Example

Imagine that there is an observatory that has a movable base, a dome that can rotate around and a retractable scope that can rotate up and down. Inside the observatory sits a bird, Oro. You are sitting at the scope head, looking at Oro. And assume that Oro is frozen in space, so that if the observatory moves, it moves relative to Oro and you want to see Oro from different perspectives. The whole setting is:

The scope head points towards the positive direction of the Z-axis, so we are looking towards the negative direction of Z-axis. Because the scope is retractable, start at zero length, so that we are inside Oro’s body. This is the starting pose, if no transform operators are defined at all.

Define these entities in your descriptions. A camera, with camera parameters defined as described in [Camera](Synthetic_Data_Generation.md); A [Dome light](Synthetic_Data_Generation.md) so that you can see things; and Oro, which is a [Geometry](Synthetic_Data_Generation.md). The observatory is only conceptual, you don’t need to see it.

```python
dome_light:
  type: light
  subtype: dome
  intensity: 1000

default_camera:
  type: camera
  camera_parameters: $[/camera_parameters]

penguin:
  type: geometry
  subtype: mesh
  usd_path: [PATH_TO_PENGUIN]
```

Note

If no camera is defined, no images are output, because nothing is there to see.

Extend the scope along the Z-axis using the `translate` operator, so that you are 1000 units way from Oro, and take a picture.

```python
default_camera:
  # ...
  transform_operators:
  - translate:
    - 0
    - 0
    - 1000
```

Then rotate the scope around the X-axis by 30 degrees. This applies a `rotateX` operator, before the original translate.

```python
transform_operators:
- rotateX: -30
- translate:
  - 0
  - 0
  - 1000
```

To go the other way around, rotate the muzzle itself, and translate it along the Z-axis. In this case the camera looks away from Oro, which is not the intention.

Rotate the turret, giving another operator `rotateY`:

```python
transform_operators:
- rotateY: 60
- rotateX: -30
- translate:
  - 0
  - 0
  - 1000
```

And eventually, drive the observatory forward, which is yet another translate, so that you don’t always have Oro at the center of the screen. Because you are defining two translates, add a suffix `translate_global`:

```python
transform_operators:
- translate_global:
  - 0
  - 0
  - 1000
- rotateY: 60
- rotateX: -30
- translate:
  - 0
  - 0
  - 1000
```

Note

Duplicated names of transform operators are not allowed. Add `_suffix` to differentiate.

To randomize all transform operators with mutable attributes and generate five images:

```python
transform_operators:
- translate_global:
    distribution_type: range
    start:
    - -500
    - 0
    - -500
    end:
    - 500
    - 0
    - 500
- rotateY:
    distribution_type: range
    start: -180
    end: 180
- rotateX:
    distribution_type: range
    start: -60
    end: 60
- translate:
    distribution_type: range
    start:
    - 0
    - 0
    - 800
    end:
    - 0
    - 0
    - 1200
```

Now you have different views of Oro. The AI model you are about to train will get a better understanding of Oro.

## Transform Operators

All available transform operators are:

**Translate operator**

| Operator Name | Required Format |
| --- | --- |
| translate, rotateXYZ, rotateXZY, rotateYXZ, rotateYZX, rotateZXY, rotateZYX, scale | numeric, list of three elements |
| orient | numeric, list of four elements |
| rotateX, rotateY, rotateZ | numeric |
| transform | numeric, list of lists of four by four elements |

Required format indicates the dimension and type of expected input. `numeric` means float or int, or a value evaluated to float or int by macro or mutable attribute. For example:

```python
rotateXYZ:
- $[../index]
- 5
- 10
```

is valid, while:

```python
rotateXYZ:
- True
- abc
```

is not valid.

Note

* `orient` is represented by a quaternion in wxyz order, in which w is the scalar part; all other rotate operators describe rotation in degrees.
* The Euler angle sequence is represented from local to global from left to right. For example, rotateXYZ means Y is global rotation relative to X, and Z is global rotation relative to Y.
* Scale operators appear at the bottom. It’s not recommended to define a scale above a translate or rotate, unless this is intended.

## Practical Example of Flexible xformOps

A translation applied globally to a rotation, is different than the other way around. In an ordinary setting, from global to local, you translate, rotate, and scale an object. In IRO, you can swap the order of linear transformations, because of the flexibility in USD xformOps. To scatter cubes on a section of a sphere using only combination of randomizations in translation and rotation in a different order:

```python
isaacsim.replicator.object:
  version: 0.x.y
  num_frames: 3
  seed: 0
  output_path: PATH_TO_OUTPUT
  simulation_time: 1
  gravity: 981

  dome_light:
    intensity: 3000
    subtype: dome
    type: light

  size_coef:
    count: 400
    distribution_type: range
    start: 0.0
    end: 1.0
  size_min: 0.5
  size_max: 0.8
  basic_shape:
    count: 400
    type: geometry
    subtype: cube
    tracked: true
    physics: rigidbody
    color:
    - 0.0 + $[/size_coef_$[index]] * 1.0
    - 0.0 + $[/size_coef_$[index]] * 0.0
    - 1.0 + $[/size_coef_$[index]] * -1.0
    size: $[/size_min] + $[/size_coef_$[index]] * ($[/size_max] - $[/size_min])
    transform_operators:
    - rotateY:
        distribution_type: range
        start: -160
        end: 160
    - rotateX:
        distribution_type: range
        start: -60
        end: 0
    - translate:
        distribution_type: range
        start:
        - 0
        - 0
        - 0
        end:
        - 0
        - 0
        - 500
    - rotateXYZ:
        distribution_type: range
        start:
        - -180
        - -180
        - -180
        end:
        - 180
        - 180
        - 180
    - scale:
      - $[../size]
      - $[../size]
      - $[../size]

  plane:
    type: geometry
    subtype: plane
    tracked: true
    physics: collision
    color:
    - 0.5
    - 0.7
    - 0.7
    transform_operators:
    - scale:
      - 10
      - 10
      - 10

  screen_height: 2160
  screen_width: 3840
  focal_length: 14.228393962367306
  horizontal_aperture: 20.955
  camera_parameters:
    screen_width: $[/screen_width]
    screen_height: $[/screen_height]
    focal_length: $[/focal_length]
    horizontal_aperture: $[/horizontal_aperture]
    near_clip: 0.001
    far_clip: 100000
  default_camera:
    camera_parameters: $[/camera_parameters]
    transform_operators:
    - rotateY: 30
    - rotateX: -30
    - translate:
      - 0
      - 0
      - 5000
    type: camera
```

The created scene with [embedded interface](Synthetic_Data_Generation.md):

The visualization using the [distribution visualizer](Synthetic_Data_Generation.md):

---

# Harmonizer

A harmonizer defines the relationship among randomized [mutable attributes](Synthetic_Data_Generation.md).

If the key-value pair in the description file has a key of `harmonizer_type`, it defines a harmonizer. A harmonizer constrains how a mutable attribute randomizes.

**Permutation harmonizer**

If `harmonizer_type` is `permutate`, it is a permutation harmonizer. When you [free randomize](Synthetic_Data_Generation.md) a `harmonized` mutable attribute, you can specify a `pitch` as the input to the permutation harmonizer. Then the permutation harmonizer shuffles these inputs and sends back the value to the harmonized attribute, which in turn can be used in a transform operator in the [harmonized randomize stage](Synthetic_Data_Generation.md).

For example, to define three OROs, facing in three directions:

```python
oro:
  count: 3
  type: geometry
  subtype: mesh
  usd_path: [PATH_TO_ORO]
  transform_operators:
  - translate:
    - ($[../index] % $[../count] - 1) * 600
    - 0
    - 0
  - rotateY: ($[../index] - 1) * 60
```

These three OROs have X-axis position `-600, 0, 600`; and they are rotated around Y-axis by `-60, 0, 60` degrees.

To shuffle the positions of these OROs, so that the ORO that rotates `-60` degrees can appear to the right:

Define a mutable attribute `permutated_index` as `harmonized`. During the free randomize stage, it submits its index as its `pitch` to the harmonizer `permutate_H`, which is a permutation harmonizer.

```python
oro:
  ...
  permutated_index:
    distribution_type: harmonized
    harmonizer_name: permutate_H
    pitch: $[index]
permutate_H:
  harmonizer_type: permutate
```

During the harmonize stage, `permutate_H` shuffles the received pitches from all relevant harmonized mutable attributes and resonates them back to each of them.

During the harmonized randomize stage, `permutated_index` gets the shuffled value back. You can use it in transform operators, like using an index.

```python
oro:
  ...
  transform_operators:
  - translate:
    - ($[permutated_index] % $[../count] - 1) * 600
    - 0
    - 0
  - rotateY: ($[../index] - 1) * 60
```

This feature can be used with any values within scope.

**Bin pack harmonizer**

If `harmonizer_type` is `bin_pack`, it is a bin pack harmonizer that packs objects into a cuboid space according to their axis-aligned bounding boxes. You can define a cuboid with custom dimensions like this:

```python
bin_pack_H:
  harmonizer_type: bin_pack
  bin_size:
  - 480
  - 260
  - 700
```

You can define many OROs, and pack them into this cuboid:

```python
oro:
  count: 200
  physics: rigidbody
  type: geometry
  subtype: mesh
  tracked: true
  transform_operators:
  - translate:
    - 0
    - 300
    - 0
  - transform:
      distribution_type: harmonized
      harmonizer_name: bin_pack_H
      pitch: local_aabb
  - scale:
    - 30
    - 30
    - 30
  usd_path: PATH_TO_ORO
```

For example, with many OROs densely packed together:

In this example, `200` OROs are spawned during initialization.

Here are some of the examples of randomized scenes generated using the bin pack harmonizer:

More insights can be found in the [harmonization example](Synthetic_Data_Generation.md).

---

# Macro

Macros can be used in [settings](Synthetic_Data_Generation.md) and [mutable attributes](Synthetic_Data_Generation.md) in certain ways to retrieve a value from another setting or mutable attribute. They are defined like `$[...]`. Macros are used everywhere to describe relationships among values to simulate complex scenes with compact descriptions.

**$[/absolute\_reference]**

Absolute references refer to values by their absolute paths.

> ```python
> bright_light:
>   type: light
>   subtype: dome
>   intensity:
>     distribution_type: range
>     start:
>       distribution_type: range
>       start: $[/dark_light/intensity]
>       end: $[/dark_light/intensity] + 200
>     end: $[/dark_light/intensity] + 1000
>   texture_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/NVIDIA/Assets/Skies/2022_1/Skies/Clear/lakeside.hdr
>   transform_operators:
>   - rotateX: 270
> dark_light:
>   subtype: dome
>   intensity:
>     distribution_type: range
>     start: 100
>     end: 1000
> ```

In this example, the mutable attribute, `/bright_light/intensity`, is a range that ranges from `$[/dark_light/intensity]` to `$[/dark_light/intensity] + 200`. These limits depend on the resolution of another mutable attribute, `/dark_light/intensity`. Thus, in every frame `/dark_light/intensity` is resolved before `bright_light/intensity` is resolved.

**$[relative\_reference]** and **$[../relative\_reference]**

Relative references refer to values with the same parent. In the example below, `$[a1]` is the same as `$[/a/a1]`:

> ```python
> a:
>   a1: x
>   a2: $[a1]
> ```

You can also go to parenting attribute using `..`. In the example below, `$[../a1]` is the same as `$[/a/a1]`:

> ```python
> a:
>   a1: x
>   a2:
>     a21: $[../a1]
> ```

**$[reference\_to\_list\_element~index]**

References to list elements refer to values in lists. In the example below, `/bins` will be expanded to `/bins_0` to `/bins_7`, with `/bins_X/dimension` resolved to a three-element list. `$[/bins_$[index]/dimension~0]` in `/transform_global_X/pitch` will retrieve the resolved value from index zero.

> ```python
> bins: # dimensions of eight small bins
>   count: 8
>   dimension:
>     distribution_type: range
>     start:
>     - 100
>     - 200
>     - 300
>     end:
>     - 400
>     - 200
>     - 300
> transform_global: # transforms of eight small bins
>   count: 8
>   distribution_type: harmonized
>   harmonizer_name: bin_pack_global_H
>   pitch:
>   - - -$[/bins_$[index]/dimension~0] / 2 * 1.5
>     - -$[/bins_$[index]/dimension~1] / 2
>     - -$[/bins_$[index]/dimension~2] / 2 * 1.5
>   - - $[/bins_$[index]/dimension~0] / 2 * 1.5
>     - $[/bins_$[index]/dimension~1] / 2
>     - $[/bins_$[index]/dimension~2] / 2 * 1.5
> ```

**$[/as\_is\_reference]**

> As-is reference macro substitutes the whole value, supporting references to dictionaries. For example:
>
> ```python
> screen_width: 960
> screen_height: 544
> camera_parameters:
>   far_clip: 100000
>   focal_length: 14.228393962367306
>   horizontal_aperture: 20.955
>   near_clip: 0.1
>   screen_height: $[/screen_height]
>   screen_width: $[/screen_width]
> default_camera:
>   type: camera
>   camera_parameters: $[/camera_parameters]
> ```
>
> Evaluates to:
>
> ```python
> screen_width: 960
> screen_height: 544
> camera_parameters:
>   focal_length: 14.228393962367306
>   horizontal_aperture: 20.955
>   near_clip: 0.1
>   far_clip: 100000
>   screen_width: 960
>   screen_height: 544
> default_camera:
>   type: camera
>   camera_parameters:
>     focal_length: 14.228393962367306
>     horizontal_aperture: 20.955
>     near_clip: 0.1
>     far_clip: 100000
>     screen_width: 960
>     screen_height: 544
> ```

**$[special\_macros]**

`$[seed]` resolves to the current frame’s seed number, and `$[frame]` resolves to the frame index.

Note

An error is triggered if a cyclic reference is detected.

Some other examples are listed below:

> You can define a macro for the path to load a USD file:
>
> ```python
> resources_root: [PATH_TO_MAIN_OBJECTS]
> main_object:
>   ...
>   usd_path:
>     distribution_type: folder
>     suffix: usd
>     value: $[/resources_root]/main_objects
> ```
>
> At runtime, the folder to sample from is resolved as `[PATH_TO_MAIN_OBJECTS]/main_objects`, so that `usd_path` is `[PATH_TO_MAIN_OBJECTS]/main_objects/[SAMPLED_FILE].usd`.
>
> ```python
> seed: 3
> penguin:
>   ...
>   count: 2
>   transform_operators:
>   - rotateY: ($[../index] + $[seed]) % $[../count] * 60
> ```
>
> At frame two, this is equivalent to:
>
> ```python
> seed: 5
> penguin_0:
>   ...
>   count: 2
>   index: 0
>   transform_operators:
>   - rotateY: (0 + 5) % 2 * 60
> penguin_1:
>   ...
>   count: 2
>   index: 1
>   transform_operators:
>   - rotateY: (1 + 5) % 2 * 60
> ```
>
> Here, `$[../index]` and `$[../count]` retrieve values from the local scope of the mutable they are in, while `$[seed]` retrieves values from the global settings.
>
> Using macros, you can describe complex scenes that have a combination of randomized transform operators for each mutable.

---

# Distribution Visualizer

The distribution visualizer is a tool that allows you to visualize the distribution of a mutable attribute. It gives a visual clue through a dynamic point cloud, showing how possible an object is to be generated at a particular pose.

## Concept

A prim has its scene graph described by a list of `xformOps`. It can be a rotation followed by a translation, and then another rotation, for example. In IRO, each `xformOp` can be a mutable attribute. By controlling the distribution of each `xformOp`, we obtain an understanding of the global spatial probability distribution of the prim by visualization.

## To Run

Here is a step-by-step guide to using the distribution visualizer on a basic prim.

1. Click **Tools** > **Action and Event Data Generation** > **Distribution Visualizer** to open the distribution visualizer as shown below.

   > 
2. Create a torus, a dome light; focus on the torus by pressing “F”; and switch to path tracing mode, as shown below.

   > 
3. Click on blank space to deselect.
4. Click on the torus again so that the distribution visualizer is in sync with the selected prim and its `xformOps` will be visible in the distribution visualizer. By default, it has translate, `rotateZYX`, and scale.
5. Apply preset `xformOps` to the torus, by clicking on `Apply Preset xformOps`. This step is not needed for an ordinary prim; this is only to demonstrate the concept. You can observe the torus is now transformed to a new pose.

   > Note
   >
   > If the torus is not visible, press “F” on the keyboard to focus the active camera to look at it. If it’s still not visible, go to the stage tab and click on the torus to make sure it’s selected.
6. Click on blank space to deselect, and then click on the torus again. From global to local, the new `xformOps` are `rotateY`, `rotateX`, and `translate`.

   > Each `xformOp` has three lines:
   >
   > * value
   > * start
   > * end
   >
   > The value is the current value of the `xformOp` and the start and end are the range of the value.
7. Change the value of rotateY and rotateX to observe how the torus rotates. More information about the scene graph can be found in [Transformation](Synthetic_Data_Generation.md).

   > So far, the steps are shown below:
   >
   > 
8. Adjust the range by changing the start and end of the `xformOps`: rotateY, rotateX, and the Z-component of translate.
9. Observe an animated shell that shows the distribution range of the torus:

   > 
10. To randomize a prim in IRO this way, insert a section like this in our description file:

```python
basic_shape:
  type: geometry
  subtype: torus
  transform_operators:
  - rotateY:
      distribution_type: range
      start: -120
      end: 120
  - rotateX:
      distribution_type: range
      start: -30
      end: 30
  - translate:
      distribution_type: range
      start:
      - 0
      - 0
      - 200
      end:
      - 0
      - 0
      - 500
```

---

# Randomization Dependency: Incremental Examples

IRO aims to provide flexible while accurate description of randomization and relationship among randomized values using fundamental building blocks:

* [mutable attributes](Synthetic_Data_Generation.md)
* [harmonizers](Synthetic_Data_Generation.md)

These elements can be wired up with macros to form a DAG-like dependency tree, such that a randomized element can depend on another randomization.

Note

The images in the following examples are generated using the [embedded interface](Synthetic_Data_Generation.md). In the viewport, you can focus on a selected prim by pressing “F”; and then you can press “Alt + Left Mouse Button” to rotate the active camera around the selected prim.

## A Basic Example

Let’s start with a basic example: “Randomly scatter ten randomly colored cubes on a plane”. The corresponding description file is:

```python
isaacsim.replicator.object:
  version: 0.x.y
  num_frames: 3
  seed: 0
  output_path: PATH_TO_OUTPUT
  simulation_time: 1
  gravity: 981

  dome_light:
    intensity: 3000
    subtype: dome
    type: light

  cube_size: 0.5
  basic_shape:
    count: 10
    type: geometry
    subtype: cube
    tracked: true
    physics: rigidbody
    color:
      distribution_type: range
      start:
        - 0.0
        - 0.0
        - 0.0
      end:
        - 1.0
        - 1.0
        - 1.0
    transform_operators:
      - translate:
          distribution_type: range
          start:
            - -300
            - $[/cube_size] / 2 * 100
            - -300
          end:
            - 300
            - $[/cube_size] / 2 * 100
            - 300
      - rotateY:
          distribution_type: range
          start: -180
          end: 180
      - scale:
        - $[/cube_size]
        - $[/cube_size]
        - $[/cube_size]

  plane:
    type: geometry
    subtype: plane
    tracked: true
    physics: collision
    color:
      - 0.5
      - 0.7
      - 0.7
    transform_operators:
      - scale:
        - 10
        - 10
        - 10

  screen_height: 2160
  screen_width: 3840
  focal_length: 14.228393962367306
  horizontal_aperture: 20.955
  camera_parameters:
    screen_width: $[/screen_width]
    screen_height: $[/screen_height]
    focal_length: $[/focal_length]
    horizontal_aperture: $[/horizontal_aperture]
    near_clip: 0.001
    far_clip: 100000
  default_camera:
    camera_parameters: $[/camera_parameters]
    transform_operators:
      - rotateY: 30
      - rotateX: -30
      - translate:
        - 0
        - 0
        - 500
    type: camera
```

By using the [embedded interface](Synthetic_Data_Generation.md), you can create such a scene:

## Randomization Dependency

To take a step further, to “Randomly scatter 10 randomly colored cubes on a plane, with varying sizes from 0.5 to 1.5, and varying color from red to blue, the bigger the size, the redder it is while the smaller the size, the bluer it is”, you can do:

```python
isaacsim.replicator.object:
  version: 0.x.y
  num_frames: 3
  seed: 0
  output_path: PATH_TO_OUTPUT
  simulation_time: 1
  gravity: 981

  dome_light:
    intensity: 3000
    subtype: dome
    type: light

  size_coef:
    count: 10
    distribution_type: range
    start: 0.0
    end: 1.0
  size_min: 0.5
  size_max: 1.5
  basic_shape:
    count: 10
    type: geometry
    subtype: cube
    tracked: true
    physics: rigidbody
    color:
    - 0.0 + $[/size_coef_$[index]] * 1.0
    - 0.0 + $[/size_coef_$[index]] * 0.0
    - 1.0 + $[/size_coef_$[index]] * -1.0
    size: $[/size_min] + $[/size_coef_$[index]] * ($[/size_max] - $[/size_min])
    transform_operators:
    - translate:
        distribution_type: range
        start:
        - -300
        - $[../size] / 2 * 100
        - -300
        end:
        - 300
        - $[../size] / 2 * 100
        - 300
    - rotateY:
        distribution_type: range
        start: -180
        end: 180
    - scale:
      - $[../size]
      - $[../size]
      - $[../size]

  plane:
    type: geometry
    subtype: plane
    tracked: true
    physics: collision
    color:
      - 0.5
      - 0.7
      - 0.7
    transform_operators:
    - scale:
      - 10
      - 10
      - 10

  screen_height: 2160
  screen_width: 3840
  focal_length: 14.228393962367306
  horizontal_aperture: 20.955
  camera_parameters:
    screen_width: $[/screen_width]
    screen_height: $[/screen_height]
    focal_length: $[/focal_length]
    horizontal_aperture: $[/horizontal_aperture]
    near_clip: 0.001
    far_clip: 100000
  default_camera:
    camera_parameters: $[/camera_parameters]
    transform_operators:
    - rotateY: 30
    - rotateX: -30
    - translate:
      - 0
      - 0
      - 1000
    type: camera
```

And we get:

The bigger the redder, the smaller the bluer. This is achieved by dependent mutable attributes. The color is determined by linear interpolation between red and blue:

```python
color:
- 0.0 + $[/size_coef_$[index]] * 1.0
- 0.0 + $[/size_coef_$[index]] * 0.0
- 1.0 + $[/size_coef_$[index]] * -1.0
```

Here:

```python
basic_shape:
  count: 10
```

Resolves to:

```python
basic_shape_0:
  index: 0
basic_shape_1:
  index: 1
basic_shape_2:
  index: 2
...
```

And that goes similarly for:

```python
size_coef:
  count: 10
  distribution_type: range
  start: 0.0
  end: 1.0
```

And then for `basic_shape_0`, for example, the R channel of color, `0.0 + $[/size_coef_$[index]] * 1.0`, will resolve to `0.0 + $[/size_coef_0] * 1.0` and `$/size_coef_0` will be replaced with a randomized value between `0` and `1`. Here is a DAG chart that shows the symbol resolution process:

## Harmonization

Try to “pack the above cubes into a big box and randomly place and rotate this big box around the up axis”:

The corresponding description file:

```python
isaacsim.replicator.object:
  version: 0.x.y
  num_frames: 3
  seed: 0
  output_path: PATH_TO_OUTPUT
  simulation_time: 1
  gravity: 981

  dome_light:
    intensity: 3000
    subtype: dome
    type: light

  bin_pack_H:
    harmonizer_type: bin_pack
    bin_size:
    - 400
    - 300
    - 400
  size_coef:
    count: 50
    distribution_type: range
    start: 0.0
    end: 1.0
  size_min: 0.5
  size_max: 1.5
  bin_translate:
    distribution_type: range
    start:
    - -100
    - 150
    - -100
    end:
    - 100
    - 150
    - 100
  bin_rotate_Y:
    distribution_type: range
    start: -180
    end: 180
  basic_shape:
    count: 50
    type: geometry
    subtype: cube
    tracked: true
    physics: rigidbody
    color:
    - 0.0 + $[/size_coef_$[index]] * 1.0
    - 0.0 + $[/size_coef_$[index]] * 0.0
    - 1.0 + $[/size_coef_$[index]] * -1.0
    size: $[/size_min] + $[/size_coef_$[index]] * ($[/size_max] - $[/size_min])
    transform_operators:
    - translate: $[/bin_translate]
    - rotateY: $[/bin_rotate_Y]
    - transform:
        distribution_type: harmonized
        harmonizer_name: bin_pack_H
        pitch:
        - - -$[../../size] / 2 * 100
          - -$[../../size] / 2 * 100
          - -$[../../size] / 2 * 100
        - - $[../../size] / 2 * 100
          - $[../../size] / 2 * 100
          - $[../../size] / 2 * 100
    - scale:
      - $[../size]
      - $[../size]
      - $[../size]

  plane:
    type: geometry
    subtype: plane
    tracked: true
    physics: collision
    color:
      - 0.5
      - 0.7
      - 0.7
    transform_operators:
    - scale:
      - 10
      - 10
      - 10

  screen_height: 2160
  screen_width: 3840
  focal_length: 14.228393962367306
  horizontal_aperture: 20.955
  camera_parameters:
    screen_width: $[/screen_width]
    screen_height: $[/screen_height]
    focal_length: $[/focal_length]
    horizontal_aperture: $[/horizontal_aperture]
    near_clip: 0.001
    far_clip: 100000
  default_camera:
    camera_parameters: $[/camera_parameters]
    transform_operators:
    - rotateY: 30
    - rotateX: -30
    - translate:
      - 0
      - 0
      - 1500
    type: camera
```

Here, `translate` and `rotateY` defines the global movement of the big box (the bin), and `transform` is a harmonized mutable attribute. The global `translate` and `rotateY` has the same value for all basic shapes, though randomized per frame. This is why the mutable attributes are defined outside of the basic shapes and then referenced through macros. Had it been defined inside the `xformOps` list, each basic shape would have a different randomized value.

## Insight into the Simulation Workflow

During initialization, mutable attributes and harmonizers are initialized, and a dependency tree with mutable elements (such as mutable attributes with different distribution types, expressions with macros, and more) is created based on the description file, and then the USD runtime scene is initialized, loading all the prims that are about to be randomized.

Each frame, all the mutable attributes resolve for their values. Mutable attributes with macros, like channels of color and size in our examples resolve their dependent mutable elements (like macro expressions) recursively. The symbol resolution procedures are totally in description level, so it’s as if we are doing randomization on text; in this stage, the USD environment is not involved.

### Harmonization Process

A harmonized mutable attribute is a special mutable attribute that can’t be resolved by running resolution one time, because it needs information from other mutable attributes sharing the same harmonizer. Run it the first time to resolve the symbols, the attribute gets into an `AWAITING_HARMONIZATION` state, and then the harmonizer absorbs its pitch (in this case, the size of the cube):

All non-harmonized attributes are resolved, which is necessary because harmonized attributes may depend on them. For example, an object can be randomized to use a different USD model with a different size bounding box, which can be the pitch to be absorbed by the harmonizer. The USD runtime is then updated based on non-harmonized attributes.

After getting all the information from the harmonized attributes, the harmonizer harmonizes. It now knows where each cube’s local transformations in the big box.

The system is in `AWAITING_HARMONIZATION` state if there is at least one attribute in this state, which means you need to resolve the whole description again. Now the corresponding values are propagated back to respective harmonized attributes.

All the numbers are fixed numbers, so you can use them to update the scene again. So, the whole workflow looks like:

---

# VLM Scene Captioning

Vision-language models (VLMs) rely on paired
image-caption datasets to learn the complex
relationships between visual content and textual
descriptions. Captions provide the semantic
grounding necessary for models to understand
objects, actions, and contexts within images.
High-quality captions are essential for training
VLMs capable of nuanced scene understanding and reasoning.

Leveraging 3D ground truth from NVIDIA
Omniverse transforms the captioning process by
enabling detailed, accurate, and scalable annotations.
These captions include overall scene descriptions,
object relationships, and spatial reasoning, such as
relative positions and interactions between elements in a camera view.
With 3D metadata, captions can describe not just what
is visible but how elements are arranged and interact,
offering richer contextual understanding.

This approach ensures more consistent and diverse
datasets, allowing VLMs to excel in complex tasks like
spatial reasoning and scene analysis, ultimately
bridging the gap between visual and linguistic comprehension.

`Isaacsim.Replicator.Caption.Core` (IRC) has the following features:

* Generate image-caption pairs for loaded scenes in Omniverse.
* Plug in to other `Isaacsim.Replicator` modules, including
  `Isaacsim.replicator.object (IRO)` and `Isaacsim.replicator.agent (IRA)` to
  generate captions for each frame at their runtime.
* Export scene graphs alongside caption outputs for customized postprocessing
  and caption preparation.

## Workflow

`Isaacsim.Replicator.Caption.Core` uses the following workflow to generate captions:

### Scene Graph

A scene graph is an intermediate output for caption generation. It is
a structured representation of a visual scene,
where nodes represent objects and edges denote spatial relationships
between them. It captures how elements are arranged in space,
such as relative positions and orientations. For example, in
an image of a person sitting on a bench under a tree, the graph
would include nodes for “person,” “bench,” and “tree,” with edges
like “sitting on” and “under.” This spatial focus makes scene graphs
valuable for tasks requiring detailed spatial reasoning and scene analysis.

You can export scene graphs alongside caption outputs to
enable flexible and customizable management of scene graph data
for your specific requirements.

## Enable Isaacsim.Replicator.Caption.Core Extension

1. Follow the [Omniverse Extension Manager guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html) to enable the `isaacsim.replicator.caption.core` extension.

   > * The extension fetches sample assets from Isaac Sim Assets during start. Refer to [Isaac Sim Assets](Isaac_Sim_Assets.md) if you encounter issues for loading assets.
   > * If loading the UI appears to be hanging, try starting Isaac Sim with the flag `--/persistent/isaac/asset_root/timeout=1.0`.
2. The IRC UI panel is accessible by **Tools > Action and Event Data Generation > VLM Scene Captioning** and it opens on the right side of the screen.

IRC can be invoked using the following methods:

* [Using the UI panel](#using-ui-panel)
* [Using the IRA extension](#using-ira-extension)
* [Using the IRO extension](#using-iro-extension)

### Using the UI Panel

To launch scene caption generation with the UI panel:

1. After enabling, the extension will appear in the UI panel:

   
2. To load the stage USD file, open up the `Caption Settings` panel, and then click on the file selector icon.

   
3. Select the USD file you want to caption. There is a default USD file for demonstration.

   Note

   We include an example USD. You can find it in `[Isaac Sim Assets Path]/Samples/Replicator/Captioning/test_caption.usda`.

   `[Isaac Sim Assets Path]` is the path to [Isaac Sim Assets](Isaac_Sim_Assets.md)

   Refer to [Isaac Sim Assets Check](Installation.md) for how to verify the assets access and how to retrieve the asset path.

   
4. Click on the **Load Scene** button to load the scene.

   

   The stage will be loaded in the stage view. If prompted to enable script execution, click **Yes**.

   
5. Enter the LLM model credentials in the [API key](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#generate-an-api-key) field of the **Model Settings** panel; click **Accept** to continue.

   
6. Under the **Caption Settings** panel, enter the desired caption level – **Brief Caption** for short and **Full Caption** for a more elaborate description. Enter the camera prim path in the **Input Camera Prim Path** field.
   Input the **Output Path** to specify where to save the generated captions, the associated scene graphs, and metadata. Ensure the output path is a valid directory. Click **Generate Scene Graph**.

   

   Note

   The default service URL and model name are provided as a convenience. The services are hosted by NVIDIA and provided free of charge on a trial basis.
   If the service associated with the default model is not reachable, a different model can be selected. Examples include:

   * `meta/llama3-8b-instruct`
   * `meta/llama3-70b-instruct`
   * `meta/llama-3.1-405b-instruct`

   It’s also possible to obtain the NVIDIA NIMs listed on the [LLM API reference page](https://docs.api.nvidia.com/nim/reference/llm-apis) and host them locally.
   Visit [NVIDIA’s NIM page](https://build.nvidia.com) for more details.
7. The scene graph, the caption, and the corresponding images are generated and saved in the output directory.

   

### Using the IRA Extension

To launch scene caption generation with IRA, load the a YAML configuration file.
Or use the default configuration file that comes with the extension and
follow the steps below to prepare some environment variables.

The anatomy of an IRC configuration file, used to run the extension
under IRO and IRA, is explained.

1. Prepare the [NIM API key](https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html#generate-an-api-key)
   for the extension to use.

   The extension requires NIM AI to generate captions.
   The credentials must be stored in the environment variables.

   **Linux/Mac:**

   Add to `~/.bashrc` or `~/.bash_profile`:

   ```python
   export NIM_API_KEY=<API_KEY>
   ```

   **Windows:**

   Command Prompt:

   ```python
   set NIM_API_KEY=<API_KEY>
   ```

   Note

   * The NIM API key has a limited lifetime. The number of free credits is limited and is accessible through the account associated with the API key. After the credits are exhausted, you can apply for more credits through the developer portal. Refer to [the developer forum](https://forums.developer.nvidia.com/t/nim-pricing/290144) for more details.
   * If you only need to generate scene graphs without captions, the AI credentials are not required.

## Example `Isaacsim.Replicator.Caption.Core` Configuration File

For example, a configuration file is similar to the following:

```python
isaacsim.replicator.caption.core:
   version: 0.6.6
   camera_prim_path: /World/Cameras/Camera
   scene_path: USD_FILE
   caption_configs:
      save_full_scene_graph: true
      save_pruned_scene_graph: true
      attach_label_to_usd: false
      use_ai_label: false
      visualize_caption: true
      max_object_capacity: 100
      export_edges: true
      global_caption: true
      qa_caption: false
      brief_caption: true
      pruning_ratio: 1.0
      verbose: true
      random_seed: 0
      caption_only: false
      export_world: true
   output_path: OUTPUT_PATH
```

### Global Properties

version

The version of IRC extension. If version does not match, the extension will not work.

camera\_prim\_path

The path to the camera prim in the scene. If not provided, the extension uses the default camera path defined in
the `default_config.yaml` file. However, if there is no camera in the scene, the extension will not work.
You must guarantee that the camera is available in the scene.

scene\_path

The path to the scene USD file. The extension can load the scene from this path. However, if the `scene_path` is
not provided, the extension uses whatever scene is loaded in the app. If no scene is loaded, the extension will not work.

output\_path

The path to the output directory where the generated captions will be saved. If not provided, the extension will use the default output path.

### Caption Configurations

save\_full\_scene\_graph

If True, it will save the full scene graph in the output directory.

The file will be saved as `<output_path>/<Camera Prim Name>/Captions/full_scene_graph.json`.

save\_pruned\_scene\_graph

If True, it will save the pruned scene graph in the output directory. The full scene graph includes
the edges between any two objects at the same level in the Support Tree.

The file will be saved as `<output_path>/<Camera Prim Name>/Captions/pruned_scene_graph.json`.

Note

**Support Tree:** A tree that represents the spatial relationships between objects in the scene.
The root of the tree is the floor (0th level). The direct children of the root are the objects on the floor, which is considered the 1st level.
The objects on the 2nd level are the objects supported by the objects on the 1st level, and so on.

pruning\_ratio

The ratio of the scene graph to be pruned. The scene graph will be pruned to a **Minimum Spanning Tree** (MST).
The pruning ratio determines the percentage of the MST edges to be kept. For example, if `pruning_ratio` is set to `0.5`,
the scene graph is pruned to 50% of the MST edges.

By default, `pruning_ratio` is set to `1.0`, which means the scene graph will not be further pruned after the MST is generated.

random\_seed

An integer for the random process. When `pruning_ratio` is less than `1.0`, the edges will be
randomly removed from the MST. The random seed is used to control the randomness of this process.

attach\_label\_to\_usd

If True, it will attach the automatically generated semantic labels to all prims with an USD address in the scene,
if the prim does not have a semantic label pre-attached.
The automatic semantic label is based on the prim path basename. For example, if the prim path is `/World/Objects/Chair`,
the semantic label will be `Chair`.

With semantic label attached, Omniverse [annotators](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html#annotators-information)
are able to capture the prim for the annotation defined. This is critical for captioning tasks, because prims not
captured by annotators cannot be included in the scene graph and therefore will not be captioned.

use\_ai\_label

If True, it will use the AI-generated labels for the prims with semantic labels in the scene. The AI-generated labels
are preprocessed and stored in the database, and they will be pulled from the database at runtime. This function can
be combined with `attach_label_to_usd: true` to handle the case when target prims does not have semantic labels pre-stored
in the scene file.

visualize\_caption

If True, it will visualize the scene graph on the output images. The visualization will be saved as
<output\_path>/<Camera Prim Name>/Captions/vis\_camera\_scene\_graph.jpg.

max\_object\_capacity

The maximum number of objects that the scene graph can contain. The objects are selected by their 2D bounding
box sizes in the camera view in a reverse order.

export\_edges

If True, the edges of the scene graph will be exported to scene graph files. The edges represent the spatial
relationships between objects.

export\_world

If True, the extension will export 3D World locations of the prims in the scene graph, and save them in the scene
graph files. The 3D World locations are the 3D coordinates of the prims in the world space. If not mentioned, all
other locations are in the camera space.

global\_caption

If True, the extension will generate a global caption for the scene. The global caption describes the overall
scene content and context. This will be saved in the output file
`<output_path>/<Camera Prim Name>/Captions/scene_graph_caption.json`.

qa\_caption

If True, the extension will generate QA captions for the scene. The QA captions are questions and answers
that test the model’s understanding of the scene.

This will be saved in the output file
`<output_path>/<Camera Prim Name>/Captions/scene_graph_caption.json`.

brief\_caption

If True, the extension will generate brief captions for the scene. The brief captions are the short version of
the global caption. This will be saved in the output file
`<output_path>/<Camera Prim Name>/Captions/scene_graph_caption.json`.

verbose

If True, the extension will print the detailed information of the scene graph generation process, such as the `support tree`,
and the number of nodes and edges in the scene graph.

caption\_only

If True, only the prims whose corresponding USD files have their object caption preprocessed and stored in the database
will be included in the scene graph and following caption generation process.

## Use IRC in `Isaacsim.Replicator.Agent`

[Isaacsim.replicator.agent](Synthetic_Data_Generation.md) (IRA) is a module that generates
synthetic data on human characters and robots across a variety of 3D environments. With the IRC extension enabled
in IRA, you can generate captions for each frame at the same time.

To enable IRC in IRA:

1. In the IRA configuration file, use IRC’s `SceneGraphWriter` to write the captions to the output directory.

   Example:

   ```python
   isaacsim.replicator.agent:
   version: 1.0.1
   simulation_duration: 5
   environment:
      base_stage_asset_path: "Isaac/Samples/Replicator/Captioning/test_caption.usda"
   sensor:
      groups:
         ceiling_cameras:
         num: 2
         aim_at_targets:
            distance_range: [5, 10]
            height_range: [7, 10]
            focal_length_range: [10, 15]
            look_down_angle_range: [30, 45]
   character:
      groups:
         warehouse_workers:
         asset_path: "Isaac/People/Characters/"
         num: 10
         routines:
            - wander:
               weight: 1
               repeat: 1
               walk:
                  speed_range: [0.8, 1.5]
                  distance_range: [5.0, 10.0]
               idle:
                  - animation: idle
                     weight: 1
                     time_range: [2.0, 5.0]
   replicator:
      writers:
         SceneGraphWriter:
         semantic_filter_predicate: "class:*"
         rgb: true
         camera_params: true
         object_info_bounding_box_2d_tight: true
         object_info_bounding_box_2d_loose: true
         object_info_bounding_box_3d: true
         pruning_ratio: 1.0
         global_caption: true
         qa_caption: false
         brief_caption: true
         visualize_caption: true
         max_object_capacity: 100
         export_edges: true
         save_full_scene_graph: true
         save_pruned_scene_graph: true
         export_world: false
         attach_label_to_usd: false
         use_ai_label: false
         verbose: false
         random_seed: 0
         caption_only: false
         scene_graph_interval: 10
         caption_interval: 10
   ```

   The caption output will be stored in the output directory as:

   * pruned scene graph: `<output_dir>/<Camera Prim Name>/caption_pruned_json/scene_graph_pruned_<frame id>.json`
   * full scene graph: `<output_dir>/<Camera Prim Name>/caption_full_json/scene_graph_full_<frame id>.json`
   * captions: `<output_dir>/<Camera Prim Name>/caption/scene_graph_caption_<frame id>.json`

   Below are the other parameters in the `SceneGraphWriter`:

   output\_dir

   The path to the output directory where the generated captions as well as IRA outputs will be saved.
   If not provided, the extension will use the default output path.

   caption\_interval

   The interval of the caption generation process. The caption will be generated every `caption_interval` frames.
   By default, `caption_interval` is set to `1000`.

   scene\_graph\_interval

   The interval of the scene graph generation process. The scene graph will be generated every `scene_graph_interval` frames.
   By default, `scene_graph_interval` is set to `1`.

   skip\_frames

   The number of frames to skip before starting the caption generation process.
   By default, `skip_frames` is set to `0`.

   writer\_interval

   The interval of the writer process. The writer will write the IRA outputs to the output directory every `writer_interval` frames.
   By default, `writer_interval` is set to `1`.

   export\_point\_cloud

   If True, the extension will export the point cloud of the frame. The point cloud will be saved in the output directory because `<output_dir>/<Camera Prim Name>/pointcloud/pointcloud_<frame id>.npy`. By default, `export_point_cloud` is set to False.

   export\_depth

   If True, the extension will export the depth map of the frame. The depth map will be saved in the output directory as
   `<output_dir>/<Camera Prim Name>/depth/depth_<frame id>.npy`. By default, `export_depth` is set to False.
2. Follow the steps in the [Isaacsim.replicator.agent](Synthetic_Data_Generation.md) tutorial to start the data generation process.

## Use IRC in `Isaacsim.Replicator.Object`

[Isaacsim.replicator.object](Synthetic_Data_Generation.md) (IRO) is a module that composes scenes that are
uniquely domain randomized. With the IRC extension enabled in IRC, you can generate captions for each frame at the same time.

To enable IRC in IRO:

1. In the IRO configuration file, use IRC’s `CombinedIROSceneGraphWriter` to write the IRO output together with captions
   to the output directory.

   Example:

   ```python
   isaacsim.replicator.object:
      version: 0.x.y
      camera_parameters: ...
      caption_configs:
         save_full_scene_graph: true
         save_pruned_scene_graph: true
         attach_label_to_usd: false
         use_ai_label: false
         visualize_caption: true
         max_object_capacity: 100
         export_edges: true
         caption_only: false
         global_caption: true
         qa_caption: true
         brief_caption: true
         pruning_ratio: 1.0
         verbose: true
         random_seed: 0
         caption_writer: CombinedIROSceneGraphWriter
      output_switches:
         caption: True
         ...
   ```

   In the `caption_configs` field, the configurations are the same as in the IRC configuration file, with
   one additional field `caption_writer`.

   caption\_writer

   The writer to write the captions to the output directory. The available writers are:

   * `CombinedIROSceneGraphWriter`: This writer combines the IRO outputs with the captions.
   * `IROSceneGraphWriter`: This writer only writes the captions to the output directory while suppressing other
     :   IRO outputs, such as `labels` (The 2D detection labels). However, it can generate `images`, `distance_to_image_plane` and `pointcloud`.

   The caption output will be stored in the output directory as:

   * pruned scene graph: `<output_dir>/caption/caption_pruned_json/<seed>_<camera_name>.json`
   * full scene graph: `<output_dir>/caption/caption_full_json/<seed>_<camera_name>.json`
   * visualized scene graph: `<output_dir>/caption_rgb/<seed>_<camera_name>.jpg`
   * captions: `<output_dir>/<Camera Prim Name>/caption_dict/<seed>_<camera_name>.json`
2. Follow the steps in the [Isaacsim.replicator.object](Synthetic_Data_Generation.md) tutorial to start the data generation process.

---

# Physical Space Event Generation

## Overview

`Isaacsim.Replicator.Incident` (IRI) is an extension that allows you to generate events
in urban simulation scenes.

Currently, IRI supports the following spontaneous event types,

* Box toppling events
* Fire and smoke events
* Liquid spills

To use IRI in a scene, follow this workflow:

1. Tag items in the scene with an appropriate event type using the property dropdown menu **+ Add > Incident
Tagging**.
Items can be tagged, for instance, as ‘loose items’ that can be knocked
over in a topple event, ‘spillable items’
that can leak or spill liquid in a spill event, or ‘flammable items’ that can catch fire in a fire event.

2. Save the scene to save the tagging information if you wish to save your progress.
A sample scene with tags already applied is provided in the Content Browser

`[Isaac Sim Assets Path]/Isaac/Samples/Replicator/Incidents/full_warehouse_with_incident_tags.usd`.

Note

* `[Isaac Sim Assets Path]` is the path to [Isaac Sim Assets](Isaac_Sim_Assets.md).
* Refer to [Isaac Sim Assets Check](Installation.md) for how to verify the assets access and how to retrieve the asset path.

3. (IRI standalone) Set up an event configuration file which defines what events will occur in the scene by using the **Event Config File** window
located in the menu **Tools > Action and Event Data Generation > Event Config File**.
This configuration can also be saved and loaded later.
Press **Set Up Events** to load the demons that will trigger the events at the specified times.

4. Run the simulation with the play button to preview the scene. To generate SDG data you can also use the **Record Events** button in the **Event Config File** window
Event items are given semantic labels as the simulation runs to support replicator’s SDG collection. A separate event log is also generated
to record the event details.

Note

No adjustment is made to the viewport camera during an event, so the you must manually find the event in the scene and move the viewport camera there to view it.

## IRI Standalone UI Example

This example shows how to use the standalone IRI UI to set up boxes falling off a shelf at a specific time.
It starts with the warehouse scene from the isaac assets folder:

`[Isaac Sim Assets Path]/Environments/Simple_Warehouse/full_warehouse.usd`.

1. Open the warehouse scene and ensuring that the navmesh has been baked. This example
uses the navmesh to determine the direction to topple the items.

1. Select boxes on a shelf and use the **IncidentTagging > LooseItem > Navmesh** button to tag them as loose items. When toppled, these boxes will fall off the shelf towards the nearest navmesh point, which will automatically make them fall towards the walkable area of the scene.
2. Optionally, you can save the scene to save your progress.
3. Open the **Event Config File** window located in the menu **Tools > Action and Event Data Generation > Event Config File**.
4. Remove the default **Spill** and **Fire** events, and examine the remaining default topple event settings.

   > The topple item is set to `$random_loose_item$`, which will randomly select a loose item in the scene to topple. The trigger is a time based trigger, and the time is set to `3` seconds.
5. Press **Set Up Events** to load the topple demon that will topple the item at the specified time.
6. Play the scene and collect event data with the **Record Events** button in the **Event Config File** window. Press **Stop Record** to stop the recording.

An event report will be generated in the specified output directory.

## Scene Tagging

To begin using IRI in a scene, tag the desired possible event items using the custom UI and then save the scene.
Right-click a prim in the stage window or viewport and select **+ Add > Incident
Tagging** and select either `loose items`, `spillable items`, or `flammable items`.
This menu is also accessible in the Property tab under the `+ Add`
button.

Currently tagged items in the scene may be visualized by enabling the Incident Scene Tags visualizer under
the eye icon on top of the viewport. Click **Show By Type > Incident Scene
Tags** and toggle the category of tagged items you wish to view.

### Loose Items

To topple items in a scene, forces are applied in a particular direction that depends
on the type of tag the loose item was given.

#### Random Direction

Items tagged as ‘random direction’ will have a force applied in a random direction.

#### NavMesh Direction

Items tagged as ‘navmesh direction’ are expected to be outside of the walkable area of
the agents in the scene. A force will be applied in the direction of the nearest navmesh edge,
useful for items on a warehouse shelf, or on a table.

#### Closest Waypoint Direction

The UI allows you to add ‘Waypoints’ to the scene. Waypoints are modeled as boxes that can be
placed anywhere in the scene and resized to outline walking paths or aisles.
Items tagged as ‘closest waypoint direction’ will have a force applied in the direction of the nearest point on the nearest waypoint.

#### Create Waypoint Prim

To add a waypoint to the scene, use the property dropdown menu and select **Create > Incident/Topple > Topple Destination**.
This button will add a waypoint to the scene for use with closest waypoint loose items.
The prim may be resized and duplicated to create
more complex structures like walking paths.

### Flammable Items

Flammable items are any items that can catch fire. When a flammable item is tagged as such,
it can be a target for a pyro event. The item’s prim must have a visible mesh under it’s hierarchy to act as the fuel source.

### Spillable Items

Spillable items are any items that can leak or spill liquid. When a spillable item is tagged as such,
it can be a target for a spill event. Item’s currently leak by instantiating a flat liquid surface onto
prims in the scene marked as ‘spillable area’ and which reside underneath the spillable item.

#### Spillable Area Floor

Spillable areas are prims that liquid may spill onto. When a spill event occurs, the liquid will be
instantiated on a prim below the spilling item with this tag. If no such prim exists, the liquid will be
instantiated on the ground at height 0.0.

**Untagging**:
Tagged items may be untagged in the Properties panel and removing any properties in the **Raw Usd Properties** section that begin with ‘isaacsim\_replicator\_incident\_attr:’.

## Event Configuration in IRI UI

IRI has a standalone UI for configuring events. This UI is accessed by navigating to **Tools > Action and Event Data Generation > Event Config File**.
Here, you can add and configure events in the scene and record them.

After adding an event, you must select and configure a trigger that will initiate the event.
The currently supported triggers are

* `time`: Begin the event at the designated time
* `carb_event`: Begin the event whenever the provided carb event happens. Carb events are the main way to integrate IRI events with other extensions.
* `physical_event`: Use the beginning of another IRI event to trigger this event.

The commands are generated as a YAML file, which can be saved and loaded later, or edited directly to change the events configuration.

## Event Configuration in IRI Script

IRI saves the event configuration to the script file, which can be edited directly to change the event configuration.

```python
isaacsim.replicator.incident:
version: 0.1.0
global:
    report_dir:
    seed: 654321
event:
    event_list:
    - ToppleEvent:
        name: my topple event
        topple_item:
            item: $random_loose_item$
            topple_nearby_radius: 1.5
        trigger:
            type: time
            time: 3
    - FireEvent:
        name: my fire event
        flammable_item:
            item: $random_flammable_item$
        trigger:
            type: time
            time: 6
    - SpillEvent:
        name: my spill event
        leakable_item:
            item: $random_leakable_item$
            target_size: 1.5
            leak_duration: 5.0
        trigger:
            type: time
            time: 9
```

In this example, three events are defined: a topple event, a fire event, and a spill event.
Each event has a name, and a simple time based trigger that will be trigger the event at the specified time.

The next few sections will go over the various event types and the parameters available for each.

### Topple Event

A topple event has the following required fields:

> * name: the name of the event
> * topple\_item: the item to topple. Can be a specific tagged item prim path, or a random tagged item given by $random\_loose\_item$
>   :   + topple\_nearby\_radius: Other loose items within this radius will also be toppled.
> * trigger: the trigger for the event. Can be a time based trigger. Triggers are defined in the trigger section [Trigger Fields](#iri-trigger-section).

```python
- ToppleEvent:
    name: my topple event
    topple_item:
        item: $random_loose_item$
        topple_nearby_radius: 1.5
    trigger:
        type: time
        time: 1.0
```

Toppled items in the scene will be given the semantic label ‘incident\_toppled\_item’.

### Fire Event

A fire event has the following required fields:

> * name: the name of the event
> * flammable\_item: the item to catch fire. Can be a specific tagged item prim path, or a random tagged item given by `$random_flammable_item$`
> * trigger: the trigger for the event. Can be a time based trigger. Triggers are defined in the trigger section [Trigger Fields](#iri-trigger-section).

```python
- FireEvent:
    name: my fire event
    flammable_item:
        item: $random_flammable_item$
    trigger:
        type: time
        time: 2.0
```

Flammable items in the scene will be given the semantic label ‘incident\_flaming\_item’. The flame itself will require a custom replicator writer to be written.

### Spill Event

A spill event has the following required fields:

> * name: the name of the event
> * leakable\_item: the item to spill. Can be a specific tagged item prim path, or a random tagged item given by `$random_leakable_item$`
>   :   + target\_size: the size of the spill area.
>       + leak\_duration: the duration of the spill.
> * trigger: the trigger for the event. Can be a time based trigger. Triggers are defined in the trigger section [Trigger Fields](#iri-trigger-section).

```python
- SpillEvent:
    name: my spill event
    leakable_item:
        item: $random_leakable_item$
        target_size: 3.0
        leak_duration: 5.0
    trigger:
        type: time
        time: 1.5
```

Leaking items in the scene will be given the semantic label ‘incident\_leaking\_item’. The liquid itself is given a separate semantic label,
‘incident\_liquid\_spill’.

## Triggers

Each event type has a trigger field, which is used to specify when the event should occur.
Here are the parameters for the various trigger types currently supported

**time**

```python
trigger:
    type: time
    # time: the time in seconds
    time: 1.5
```

**carb\_event**

```python
trigger:
    type: carb_event
    # event_name: the name associated to the desired carb event
    event_name: my_extension_custom_event
```

**physical\_event**

```python
trigger:
    type: physical_event
    # incident_name: Each physical event in IRI has a unique name.
    # This triggers at the beginning of the provided IRI event
    incident_name: MyFireEvent
```

## SDG Collection

SDG collection is handled by the replicator’s SDG writers based on the semantic labels of the event items. Additional information
is collected in the event log, which is a yaml file in the output directory.

---

# RTX Sensors Placement and Calibration

Optimizing camera placement is a crucial technique, particularly in indoor or enclosed spaces such as warehouses, retail stores, hospitals, and other similar environments, to ensure comprehensive coverage while minimizing camera deployment costs.

Isaac Sim provides two separate extensions to help you optimize camera placement and extract calibration data:

* **Camera Placement** (`isaacsim.sensors.rtx.placement`): Automatically determines optimal camera locations based on scene layout and coverage requirements.

  >   > - Camera Placement
  > 
* **Camera Calibration** (`isaacsim.sensors.rtx.calibration`): Extracts and manages camera calibration data, including position, orientation, and field of view information.

  >   > - Camera Calibration
  >

---

# Camera Placement

The Camera Placement Tool (`isaacsim.sensors.rtx.placement` extension) automatically optimizes camera placement in a stage according to user customized requirements.

## Enable the Extension

Follow the [Omniverse Extension Manager guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html) to enable the `isaacsim.sensors.rtx.placement` extension.

## Open the Camera Placement Tool Panel

The Camera Placement Tool is accessible by **Tools > Sensors > Camera Placement**.

### Input Fields

**Camera Placement Output Path**:
The folder path where the generated camera placement data will be saved.
The file `camera_info_payload.json` will be output to this folder, containing information about all cameras relevant to the current camera placement task.

**Cached data would include**:

> * Camera Path
> * Camera Position
> * Focus Point Position
>
> * Example
>   :   + Output: in `camera_info_payload.json`
>         :   ```python
>             {
>                 "X_Positive": [
>                     {
>                     "camera_path": "/World/Cameras/Camera",
>                     "camera_position": [
>                         -25.48988914489746,
>                         -14.219901084899902,
>                         3.319734811782837
>                     ],
>                     "focus_point": [
>                         -20.801025390625,
>                         -18.900035858154297,
>                         0.5
>                     ]
>                     }
>                 ]
>             }
>             ```

**Total Camera Number**:
The total number of cameras to be placed in the scene.

**Camera Range Parameters**:

> * **Camera Height Range**:
>   :   + Define the allowable height range for camera placement above the ground.
> * **Camera Distance Range**:
>   :   + Set the distance range within which a camera can be placed from any point P on the stage.
>       + Ensures:
>         :   - For any point P, there exists a camera C such that the distance between C and P is within this range.
> * **Camera Look Down Angle Range**:
>   :   + Define the downward tilt angle range for the cameras.
>
>         > - Zero degrees means the camera is parallel to the ground (horizontal view).
>         > - 90 degrees means the camera is pointing straight down (top-down view, perpendicular to the ground).

**Stage Processing Parameters**:

> * **Patch Size**:
>   :   The stage is divided into patches of this size for estimating coverage.
>
>       > + The smaller size means more detailed stage analysis when calculate camera coverage.
>       > + On the other hand, It also means more computation time.
> * **Ground Height**:
>   :   Height of the ground surface in the stage.
> * **Stage Scope**:
>   :   Defines the spatial boundaries of the stage for camera placement when navmesh is unavailable.
>
>       > + **X Scope**: Minimum and maximum X-axis boundaries.
>       > + **Y Scope**: Minimum and maximum Y-axis boundaries.
>
>       Warning
>
>       This parameter is not recommended for normal use. Only use it in edge cases when a valid navmesh cannot be built for the stage.

**Other Tuning Parameters**:

> * **Random Seed**:
>   :   Controls the random seed for the camera placement process. For a given random seed, the camera placement result will be deterministic.
> * **Border Checking Index**:
>   :   Controls how close cameras can be placed to the boundary of the stage.
>       Prevents invalid placements that can result from proximity to obstacles or being outside the stage bounds.
> * **Camera On Navmesh**:
>   :   Whether cameras must be placed only on the navigation mesh.
> * **Minimum Coverage Increase**:
>   :   The minimum additional patch a camera must cover for it to be considered valid.
>       If the new camera increases coverage less than this value, placement will stop.
> * **Limit FOV by Distance**:
>   :   Determines whether the camera’s field of view should be restricted based on its **Camera Distance Range**.
>       - If enabled, the estimated camera coverage will be further limited according to the distance between each visible area and the target camera.
> * **Coverage Density**:
>   :   Specifies how many cameras must cover each patch at a minimum.
> * **Target Coverage Ratio**:
>   :   The desired overall ratio of the stage that must be covered by cameras **according to the requirements**.
>       Placement stops if this target is not met.

#### Buttons and Functions

* **Place Cameras**:
  :   Begin the automated camera placement process using the parameters defined above.

      Note

      + After clicking the **Place Cameras** button, the process can take some time to complete. The duration depends on the number of cameras to be placed and the complexity of the stage.
      + At the end of the placement the number of the camera number of the camera in each direction would summarized and output in the console as a carb warning message.
* **Show Selected Camera Coverage**:
  :   Visualize the coverage area of the currently selected camera.

      Note

      + The **Show Selected Camera Coverage** button displays the coverage areas of all *selected* cameras.
      + Points with different levels of coverage will be shown in distinct colors.

        > - If the required **Coverage Density** is set to `N`, then `N` distinct colors will be used to represent coverage levels.
      + Example:

        > - [Camera Coverage Visualization Example](#camera-coverage-visualization-example)

* **Show all Camera Coverage**:
  :   Visualize the coverage area of all cameras in the scene, regardless of selection status.

      Note

      + This button displays the combined coverage areas of all generated cameras.
      + Use this to quickly verify overall scene coverage without manually selecting individual cameras.
      + Points with different levels of coverage will be shown in distinct colors based on the **Coverage Density** setting.
* **Hide Coverage**:
  :   Hide the camera coverage visualization from the stage view.

## Camera Placement Tool Tutorial

To use the **Camera Placement Tool**. Ensure the scene has valid navmesh baked before proceeding. The tutorial uses the [Isaac Sim Full Warehouse](Isaac_Sim_Assets.md) for demonstration.

Note

* Stage unit must be in meters.
* A valid [NavMesh](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh.html "(in Omniverse Extensions)") is required.
* Z axis is up.

### Enable the Extension

1. [Enable the isaacsim.sensors.rtx.placement extension](#enabling-camera-placement-extension).
2. [Open the camera placement tool panel](#activate-camera-placement-tool-panel).

### Open the Target Stage

Open the [Isaac Sim Full Warehouse](Isaac_Sim_Assets.md)

> Note
>
> * Verify that the navmesh is baked successfully
> * Access **Window > Navigation > Navmesh** and click on **Bake** button if you need to rebake the navmesh.
>
>   > + Before proceeding, ensure that the `omni.anim.navigation.bundle extension` is enabled according to [instruction](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh/installation.html "(in Omniverse Extensions)").

### Configure Camera Placement

In the **Camera Placement** section of the UI:

1. Set the **Camera Placement Output Path**, by entering your cache folder path.
2. Set the **Total Camera Number**. Use -1 to auto-compute the minimum number of cameras needed.

   > 

### (Optional) Adjust Camera Range

If needed, configure the **Camera Range Parameters** such as height, look-down angles, and target distance.
Refer to the [Camera Range Input Fields](#camera-range-parameters) for more details. This example uses the default values.

### (Optional) Adjust Stage Processing

**Stage Processing Parameters** allows you to configure the camera placement method according to the stage’s size, height, and complexity.
Tune **Stage Processing Parameters** to set patch size or ground height, if applicable.
Refer to the [Stage Processing Parameters Field](#stage-processing-parameters) for more details. This example uses the default values.

### Fine-tune Placement

Multiple configurable parameters have been added to help user check and refine the camera placement logic.

In this case, modify these two parameters:

> * Set **Coverage Density** to `2`, which means for each patch in the stage, you need two cameras to cover it.
> * Set **Target Coverage Ratio** to `0.99`, which means 99 percent of the patch needs to be covered according to the set requirements.
>
> 
>
> * In this example, use the default values for other parameters.
> * You are free to modify more **Other Tuning Parameters** to adjust placement logic if finer control is needed.
> * Refer to the [Fine Tuning Parameters Field](#fine-tuning-processing-parameters) for more details.

### Run Camera Placement

Click the **Place Cameras** button to begin automatic placement. Wait for the process to complete.

> * The process can take some time to complete. The duration depends on the number of cameras to be placed and the complexity of the stage.

### Check Coverage

1. Get a top view of the stage to make the camera coverage visualization more clear.

   > * [Create Top View Camera With Camera Calibration Panel](Synthetic_Data_Generation.md).
   > * Switch you view port camera to the created top view camera.
   > * Visualize Navmesh by clicking on **Visibility Menu (eye icon on viewport) > Show By Type > Navmesh**.
   >
   >   > 
2. In the **Camera Placement Tool** panel, click **Show all Camera Coverage** to visualize the coverage of all generated cameras.

   > * [How Camera Coverage Visualization Works](#show-all-camera-coverage)

> * In this example, points covered once are shown in red, while points covered twice are shown in green.
>
>   > + From the visualization result, most points are covered as our expectation.

### Hide Coverage

Click the **Hide Coverage** button to remove the coverage overlay.

### (Optional)Save Your Work

**Save** or **Save as** the updated USD file to preserve camera placements for further SDG workflows.

---

# Camera Calibration

The Camera Calibration Tool (`isaacsim.sensors.rtx.calibration` extension) generates camera calibration data for deployed cameras in the scene.

## Enable the Extension

Follow the [Omniverse Extension Manager guide](https://docs.omniverse.nvidia.com/extensions/latest/ext_core/ext_extension-manager.html) to enable the `isaacsim.sensors.rtx.calibration` extension.

## Open the Calibration Tool Panel

The **Calibration Tool** UI will automatically be opened on the right side of the screen. It is accessible by **Tools > Sensors > Camera Calibration**.

### Input Fields

**Place Info**: A string that describes the location of the scene, including city, building, and room. This information is converted and stored in `calibration.json`. Review the output example for more details.

> * Input Format: `city=[city name]/building=[building name]/room=[room name]`
> * Example:
>
>   > + Input: `city=Santa Clara/building=NVIDIA Voyager/room=Visitor Lobby`
>   > + Output: in `calibration.json`:
>   >
>   > ```python
>   > {
>   >     "place": [
>   >         {
>   >         "name": "city",
>   >         "value": "Santa Clara"
>   >         },
>   >         {
>   >             "name": "building",
>   >             "value": "NVIDIA Voyager"
>   >         },
>   >         {
>   >             "name": "room",
>   >             "value": "Visitor Lobby"
>   >         }
>   >     ],
>   > }
>   > ```

**Scene Root Prim Path**: The path to the root prim of the scene. This is used to approximate the top view camera’s position. The top view camera will look at the scene root prim’s center.

**Floor & Ceiling Height**: The floor and ceiling height values for the scene.

> Note
>
> * The ceiling height adjusts the clipping range of the top view camera, making it easier to create accurate top views.
> * By default, the ceiling height is set to `-1`, which means the top view camera’s clipping range will use its default value.
> * By customizing the ceiling height, you can clip out prims or objects above the specified value when creating a top view camera.

**Top View Camera**: This camera will be used to render the `top_view` images.

> * **Create**: After **Scene Root Prim Path** is set, clicking this button will automatically generate a top view camera that looks at the scene root’s center. The top view camera will be generated under `/World/Top_Camera`.
> * **Path**: After clicking **Create**, the top view camera’s path will be shown here. You can also use this field to select existing top view cameras in the stage.

> Note
>
> * The **Top View Camera** must be vertical to the ground and it must cover the position of all the calibration dots under `World/Calibration_Dots` and cameras under the `World/Cameras`.
> * The **Top View Camera** must have a rotation of `[0,0,0]` with the projection type set to `orthographic`.

**Raycast Density**: The density of the raycast. The higher this value, the more detailed the FOV contour will be. A density value of `N` indicates that `N * N` rays will be cast and they are uniformly distributed for each camera.

> * Default value: 100

**Minimum FOV Polygon Edge Length (meter)**: The minimum length of edges in the polygon’s contour. Edges shorter than this length are ignored and the vertices are connected to the next point that meets this criteria. The unit is meter.

> * Default value: 0 (no simplification in drawing the contour)

**Minimum Area of FOV Polygon Hole to Ignore**: When generating data, holes in the FOV polygon that are smaller than this threshold value are ignored. Holes are the areas that are not included in the FOV polygon.

> * Default value: 0 (don’t ignore any holes in FOV polygons)

**Create Camera View Images**: Whether to include camera view images in the output folder.

**Create FOV Polygon Images**: Whether to render top view images with FOV polygons in the debug data folder.

**Show FOV Polygon**: Whether to show FOV polygons from the currently selected camera.

**Output Folder Path**: The path to the output folder. Click on the folder icon to select the output folder path.

#### Buttons and Functions

Note

Before starting to generate the calibration file, the following prerequisites must be met:

> * **Top View Camera Path** field is set up with a valid camera prim path to a [valid top view camera](#valid-topview-camera-path).
> * The output folder path value must be valid.
> * The **Place Info** must have the correct format and input.
> * The cameras must be under `/World/Cameras`.

* **Create Dot Prims**: Generate calibration dot prims for each camera. Calibration dots will be randomly generated and they are used to sample the polygons contour that each camera can view.
* **Generate Calibration File**: Create `calibration.json` that stores all the camera calibration data.

  > Note
  >
  > You must run **Create Dot Prims** before generating the calibration file.
* **Generate Top View Image**: Generates the top view image and stores the image in the output folder. An `imageMetadata.json` file will be generated to store the image metadata.

  > Note
  >
  > If the `Create FOV Polygon Images` is checked, the FOV polygon is visualized on the top view layout. The FOV images are generated in a debug data folder within the output folder.

## Using the Camera Calibration Tool Tutorial

To use the **Camera Calibration** tool. This tutorial makes use of the [Isaac Sim Full Warehouse](Isaac_Sim_Assets.md) for demonstration.

Note

* Stage unit must be in meters.
* A valid [NavMesh](https://docs.omniverse.nvidia.com/extensions/latest/ext_navigation-mesh.html "(in Omniverse Extensions)") is required.

### Enable the Extension

1. [Enable the isaacsim.sensors.rtx.calibration extension](#enabling-camera-calibration-extension).
2. [Open the camera calibration tool panel](#activate-camera-calibration-tool-panel).

### Create Cameras

Cameras under `/World/Cameras` are used to generate the calibration file. Ideally, the cameras are able to view the walkable area of the scene.

> Tip
>
> To add cameras to the stage, follow the [Isaac Sim Camera tutorial](Robot_Setup.md).
>
> Alternatively, you can [use IRA to spawn cameras](Synthetic_Data_Generation.md).

### Create Top View Camera

The **Top View Camera** supports the following features:

* Capture top view images of the scene.
* Generate FOV polygons of the scene.
* Generate 2D camera locations of each camera within the top view image.

The extension provides a UI to help you create the top view camera:

1. Set the **Scene Root Prim Path**. The top view camera generated by this tool will look at this scene root. In this case, set it to `/Root`.
2. Set the **Floor & Ceiling Height** values to clip the ceiling from the top view camera’s view.

   > Note
   >
   > * In this tutorial, set **Ceiling Height** to `6` to clip the warehouse ceiling.
   > * Because the warehouse floor height is `0`, there’s no need to change the **Floor Height**.
3. Click **Create**. The top view camera will be generated and its path will be shown in the text field.

   > 
4. Switch the viewport to the new top view camera to verify that it covers the floorplan.

   > 

Tip

To switch the viewport to the top-view camera, click the Camera icon, then click **Cameras > Calibration\_Top\_Camera**.

### Set Up the Calibration Tool Attributes

This step is to enter the information needed for camera calibration.

1. Enter the place information in **Place Info**. In this case, it’s `city=Santa Clara/building=Isaac Sim Warehouse/room=Warehouse`.
2. Set **Raycast Density**, **Minimum FOV Polygon Edge Length**, and **Minimum Area of FOV Polygon Hole to Ignore**. See the [Input Field](#ira-calibration-attribute) for more details. In this case, use the default values.
3. Check the **Create Camera View Images**, **Create FOV Polygon Images**, and **Show FOV Polygon** boxes if these [additional data](#ira-calibration-additional-attribute) are needed.
4. Set **Output Folder Path** by either entering the path or clicking the folder picker icon.

### Generate Calibration Dots

Generate calibration dots for each camera by clicking the **Create Dot Prims** button.

Note

* Calibration dot prims are generated under `/World/Calibration_Dots/[Camera Name]/`, where `[Camera Name]` is the name of the camera.
* For each camera prim under `/World/Cameras`, six calibration dots are generated. The dot prims are used to calculate the projection matrix for each camera.
* You can switch your viewport to any camera’s view to check whether all calibration dots are visible.

### Generate the Calibration File

1. Generate the calibration file by clicking on the **Generate Calibration File** button. This generates a `calibration.json` file to **Output Folder Path**.
2. After the `calibration.json` file is generated. You can visualize the FOV in the stage by selecting the target camera.

   > 

Note

Your result might look different because it depends on the camera parameters. In this tutorial, the translate of the camera is `(-13.02311, 7.20828, 5.0)`, the orient is `(-55.253, -56.035, -150.088)`, and the camera focal length is `20.94`.

### Generate Top View Image

To visualize the generated FOV polygon top view image, generate the image by clicking on **Generate Top View Image** button.

> The Top View Camera’s view will be rendered and output to the `[Output Folder Path]`.
> An `imageMetadata.json` file is also generated to store image metadata.

Note

If **Create FOV Polygon Images** is checked, for each camera there will be an image with a white-shaded FOV polygon from the top view.
The FOV polygon images will be generated under `[Output Folder Path]/Debug/fieldOfViewPolygon`.

---

# Isaac Agent Planner (IAP)

The **Isaac Agent Planner (IAP)** is an advanced AI-powered system that automatically generates behavior trees for actors (characters and cameras) in simulation environments based on natural language scenario descriptions. By leveraging Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG), IAP transforms high-level scenario descriptions like *“Alice picks up the mug and places it on the table”* into executable behavior trees that drive actor behaviors in NVIDIA Omniverse Isaac Sim.

The IAP system consists of two complementary extensions:

| Extension | Purpose |
| --- | --- |
| `isaacsim.agent.planner.core` | Core API for behavior tree generation using LLMs and RAG |
| `isaacsim.agent.planner.bridge` | Bridge layer connecting the core API with Kit/Omniverse environment |

## Key Features

### Natural Language Scenario Processing

IAP accepts plain English scenario descriptions and automatically:

* Parses the scenario to identify actors and their actions
* Segments complex scenarios into per-actor behavior sequences
* Maps actions to available behavior tree node types
* Generates parameters using context-aware RAG retrieval

### Context-Aware Generation

The system understands your simulation environment through:

* **Actor Context**: Information about characters, cameras, and their capabilities
* **Object Context**: Details about interactable objects in the scene
* **Node Catalog**: Available behavior tree actions and their parameter schemas
* **Blackboard Variables**: Shared state accessible to all behavior trees

### RAG-Enhanced Parameter Generation

Using Retrieval-Augmented Generation, IAP enriches behavior tree parameters with:

* Semantic matching of scenario elements to scene objects
* Elaboration of actor metadata for richer context
* Intelligent parameter extraction based on node schemas

### Event-Driven Architecture

IAP uses a sophisticated event system for:

* Dispatching tree construction completion events
* Triggering automatic behavior tree switching at runtime
* Responding to blackboard value changes and timeline events

## Architecture

The IAP system is organized into three main layers:

**User Interface Layer**

* *Context Cache Panel*: Configure data sources for generation
* *Network Config Panel*: Configure LLM settings
* *Scenario Execution Panel*: Enter scenarios and trigger generation

**IAP Bridge Layer** (`isaacsim.agent.planner.bridge`)

* `BridgePipeline`: Orchestrates the generation workflow
* Path Mapping: Translates IDs to USD prim paths
* Event-Driven Tree Switcher: Manages runtime behavior switching

**IAP Core Layer** (`isaacsim.agent.planner.core`)

* Context Cache Manager: Loads and manages actor/object data
* RAG Systems: Context and event retrieval for parameter generation
* LLM Network: Tree construction using language models

**Output**

* Behavior Tree JSON Files compatible with OBC
* Blackboard Metadata for shared state
* Simulation Event Cache for event sequencing

## Generation Pipeline

The IAP pipeline processes scenarios through these phases:

1. **Configuration Validation**: Verify all required inputs and paths
2. **Context Loading**: Load actor, object, and blackboard data
3. **Node Catalog Loading**: Load available behavior tree node definitions
4. **RAG Construction**: Build vector stores for context retrieval
5. **Elaboration Building**: Create enriched parameter lookups
6. **Network Construction**: Build the LLM processing network
7. **Generation**: Invoke the network to generate behavior trees
8. **Output Saving**: Save trees and dispatch completion events

## Core Concepts

### Actors

Actors are entities in your scene that can execute behavior trees. IAP supports:

* **Human Actors**: Characters that can perform physical actions (walk, pick up, sit, etc.)
* **Camera Actors**: Cameras that can perform cinematic behaviors (pan, zoom, follow, etc.)

Actor data is provided as JSON with required fields:

Actor definition example

```python
{
    "id": "Alice",
    "semantic_description": "A character who can interact with objects",
    "metadata": {
        "actor_type": "human",
        "prim_path": "/World/Characters/Alice",
        "role": "protagonist"
    },
    "entity_type": "actor"
}
```

### Objects

Objects are scene elements that actors can interact with:

Object definition example

```python
{
    "id": "Mug_01",
    "semantic_description": "A ceramic coffee mug on the counter",
    "metadata": {
        "prim_path": "/World/Props/Mug_01",
        "object_type": "pickup_item",
        "location": {"x": 1.5, "y": 2.0, "z": 0.8}
    },
    "entity_type": "object"
}
```

### Node Catalog

The node catalog defines available behavior tree actions. Each node includes:

* Node type identifier
* Human-readable description
* Parameter schema (JSON Schema format)
* Semantic keywords for RAG matching

### Behavior Trees

Generated behavior trees are saved as JSON files compatible with the Omni Behavior Composer (OBC). Each tree contains:

* Root node structure
* Composite nodes (Sequence, Selector, Parallel)
* Action/Condition leaf nodes with parameters
* Blackboard references

## UI Components

The IAP Bridge Window provides a unified interface with four main panels:

### Context Cache Panel

Configure the data sources for generation:

* **Context Files**: JSON files containing actor and object definitions
* **Node Catalog**: Available behavior tree node types
* **Blackboard File**: Shared variable definitions
* **Metadata Schemas**: Optional JSON schemas for elaboration RAG

### Network Config Panel

Configure the LLM generation settings:

* **Model Configuration**: LLM model settings and endpoints
* **Node-to-Model Map**: Map specific nodes to specialized models
* **API Key**: Authentication for LLM services

### Output Settings Panel

Configure where generated outputs are saved:

* **Output Folder**: Root directory for all generated files
* **RAG Cache Folder**: Location for RAG vector store caches

### Scenario Execution Panel

Trigger behavior tree generation:

* **Scenario Description**: Natural language input describing the scene
* **Run Pipeline**: Execute the generation pipeline
* **Load Example Scene**: Set up a demo environment for testing

## Integration with Omni Behavior Composer

IAP integrates seamlessly with the [Omni Behavior Composer (OBC)](../ext_replicator-agent/ext_omni_behavior_composer.html) system:

1. **Node Libraries**: Action nodes are registered in OBC node libraries
2. **Blackboard**: Shared state between all actors in the scene
3. **Tree Execution**: Generated trees execute through OBC runtime
4. **Event System**: IAP events can trigger tree switching during simulation

## Use Cases

### Synthetic Data Generation (SDG)

Generate diverse actor behaviors for training data collection:

* Vary scenarios programmatically
* Create realistic motion sequences
* Capture multi-actor interactions

### Digital Twin Simulation

Model real-world scenarios with accurate agent behaviors:

* Warehouse worker activities
* Retail store customer flows
* Manufacturing line operations

### Rapid Prototyping

Quickly iterate on character behaviors without manual tree authoring:

* Test scenario variations
* Explore behavior possibilities
* Validate scene setups

## Requirements

### System Requirements

* NVIDIA Omniverse Isaac Sim 4.5+
* RTX GPU (4080+ or datacenter GPU recommended)
* Python 3.10+

### Dependencies

* `omni.behavior.composer` - Behavior tree runtime
* `omni.behavior.composer.models` - Pydantic models for BT nodes
* `omni.behavior.composer.schema` - USD schema types
* `omni.metropolis.core` - Metropolis utilities
* `omni.metropolis.agent_registry` - LLM agent management

### API Key Requirements

IAP requires an API key for LLM services. Configure via:

* UI: *Network Config Panel* > **API Key** field
* Environment variable: Set in your shell configuration
* Agent Registry: Configure in `omni.metropolis.agent_registry`

## Getting Started

* For a quick start guide using the example scene, see the [IAP Example Walkthrough](Synthetic_Data_Generation.md).
* For manual configuration and Python API usage, see the [IAP Configuration and API Reference](Synthetic_Data_Generation.md).

## API Reference

For programmatic usage, see the core API documentation:

Basic IAP API usage

```python
from isaacsim.agent.planner.core.api import (
    BehaviorTreeGenerationConfig,
    generate_behavior_trees,
    GenerationPhase,
)

# Configure and run generation
config = BehaviorTreeGenerationConfig(
    scenario="Alice picks up the mug.",
    actor_data_path="/path/to/actors.json",
    output_folder_path="/path/to/output",
)

result = await generate_behavior_trees(config)
```

## Additional Resources

* [IAP Example Walkthrough](Synthetic_Data_Generation.md) - Quick start tutorial
* [IAP Configuration and API Reference](Synthetic_Data_Generation.md) - Manual configuration and Python API
* [Omni Behavior Composer](../ext_replicator-agent/ext_omni_behavior_composer.html) - OBC reference documentation

---

# IAP Example Walkthrough

This guide provides a step-by-step walkthrough for using the Isaac Agent Planner (IAP) to generate behavior trees from natural language scenario descriptions using the built-in example scene.

For manual configuration and Python API usage, see the [IAP Configuration and API Reference](Synthetic_Data_Generation.md).

## Prerequisites

Before starting, ensure you have:

1. **Isaac Sim Installed**: NVIDIA Omniverse Isaac Sim 4.5 or later
2. **Extensions Enabled**: Both `isaacsim.agent.planner.core` and `isaacsim.agent.planner.bridge` extensions
3. **API Key Configured**: An API key for LLM services (NVIDIA NIM or compatible endpoint)
4. **GPU**: RTX 4080+ or datacenter GPU (A40, L40S) for optimal performance

## Quick Start with Example Scene

The fastest way to try IAP is using the built-in example scene.

### Step 1: Enable the Extensions

1. Open Isaac Sim
2. Navigate to **Window** > **Extensions**
3. Search for `isaacsim.agent.planner`
4. Enable both:

   * `isaacsim.agent.planner.core`
   * `isaacsim.agent.planner.bridge`

### Step 2: Open the IAP Bridge Window

After enabling the extensions, the *IAP Bridge* window opens automatically. If not visible:

1. Navigate to **Tools** > **IAP Bridge**
2. The *IAP Bridge* window will appear on the right side of the screen

### Step 3: Load the Example Scene

1. In the *Scenario Execution* panel at the bottom of the *IAP Bridge* window
2. Click the **Load Example Scene** button
3. Wait for the status message: *“Example scene loaded: X actors configured”*

This action will:

* Open a pre-configured test stage with actors and objects
* Load example context files (actors, objects, node catalog)
* Set up the OBC environment (motion library, node libraries, blackboard)
* Configure all actors with default idle behavior trees

### Step 4: Configure NVIDIA API Key

Before running the pipeline, configure your NVIDIA API key for LLM access:

1. In the *IAP Bridge* window, locate the *Network Configuration* panel
2. Find the **API Key** field
3. Enter your NVIDIA API key (format: `nvapi-XXXX...`)

Note

If you do not have an NVIDIA API key, visit the [NVIDIA API portal](https://build.nvidia.com) to obtain one. See the [NVIDIA API reference page](https://docs.api.nvidia.com/nim/reference/llm-apis) for more details on API usage and credits.

Alternatively, you can set the API key as an environment variable before launching Isaac Sim:

Setting NVIDIA API key via environment variable

```python
export NVIDIA_API_KEY="nvapi-YOUR-KEY-HERE"
```

### Step 5: Enter a Scenario Description

In the *Scenario Execution* panel:

1. Find the **Scenario Description** text area
2. Enter a natural language description of what you want actors to do

Example scenarios:

Simple pick and place scenario

```python
Anna picks up the box and places it on the table.
```

scenario with semantic description

```python
Anna walks to the black chair.
```

### Step 6: Run the Pipeline

1. Click the **Run Pipeline** button
2. Watch the status messages as the pipeline progresses:

   * *“Validating configuration…”*
   * *“Loading context data…”*
   * *“Generating behavior trees…”*
   * *“Pipeline completed!”*

### Step 7: View the Results

When the pipeline completes successfully:

1. **Status Message**: Shows how many actors have generated behavior trees
2. **Console Log**: Detailed information about generated files
3. **Output Folder**: Generated behavior tree JSON files saved to the configured output path

### Step 8: Run the Simulation

1. Click **Play** in the timeline toolbar
2. Actors will execute their generated behavior trees
3. Click **Stop** to end the simulation

### Step 9: Generate Synthetic Data (Optional)

Once your actors are executing their generated behavior trees, you can capture synthetic data for training computer vision models or other AI applications.

The [Synthetic Data Recorder](Synthetic_Data_Generation.md) provides a GUI extension for recording synthetic data from your simulation. It supports:

* Multiple camera render products with configurable resolutions
* Various annotators including RGB, depth, semantic segmentation, and bounding boxes
* Custom writers for specialized data formats
* Timeline-synchronized recording

To record synthetic data from your IAP-generated scene:

1. Open the Synthetic Data Recorder from **Tools** > **Replicator** > **Synthetic Data Recorder**
2. Add render products for the cameras in your scene
3. Select the annotators you need (RGB, bounding boxes, segmentation, etc.)
4. Configure the output directory
5. Click **Start** to begin recording while the simulation runs

For detailed instructions on using the recorder, see the [Synthetic Data Recorder Tutorial](Synthetic_Data_Generation.md).

## Next Steps

* Read the [IAP Introduction](Synthetic_Data_Generation.md) for architecture details
* See the [IAP Configuration and API Reference](Synthetic_Data_Generation.md) for manual configuration, Python API usage, troubleshooting, and best practices
* Explore the [Omni Behavior Composer](../ext_replicator-agent/ext_omni_behavior_composer.html) for behavior tree fundamentals

---

# IAP Configuration and API Reference

This document provides detailed configuration instructions and API reference for the Isaac Agent Planner (IAP). For a quick start guide, see the [IAP Example Walkthrough](Synthetic_Data_Generation.md).

## Manual Configuration

For production use or custom scenes, configure IAP manually using the steps below.

### Prepare Your Data Files

Create the required JSON data files for your scene.

#### Actor Data File

Define the actors in your scene:

actors.json

```python
[
    {
        "id": "Alice",
        "semantic_description": "A female character who can perform various actions",
        "metadata": {
            "actor_type": "human",
            "role": "protagonist",
            "prim_path": "/World/Characters/Alice",
            "semantic_label": "woman"
        },
        "entity_type": "actor"
    },
    {
        "id": "MainCamera",
        "semantic_description": "The primary filming camera",
        "metadata": {
            "actor_type": "camera",
            "role": "main_camera",
            "prim_path": "/World/Cameras/MainCamera",
            "semantic_label": "camera"
        },
        "entity_type": "actor"
    }
]
```

#### Object Data File

Define interactable objects:

objects.json

```python
[
    {
        "id": "CoffeeMug",
        "semantic_description": "A ceramic coffee mug on the counter",
        "metadata": {
            "prim_path": "/World/Props/CoffeeMug",
            "object_type": "pickup_item",
            "location": {"x": 1.5, "y": 2.0, "z": 0.8}
        },
        "entity_type": "object"
    },
    {
        "id": "DiningTable",
        "semantic_description": "A wooden dining table in the center of the room",
        "metadata": {
            "prim_path": "/World/Props/DiningTable",
            "object_type": "surface",
            "location": {"x": 0.0, "y": 0.0, "z": 0.0}
        },
        "entity_type": "object"
    }
]
```

### Configure Context Cache Files

In the *Context Cache Files* panel:

1. **Context Files (Actors & Objects)**:

   * Click **+ Add Context File**
   * Browse to your `actors.json` file
   * Click **+ Add Context File** again
   * Browse to your `objects.json` file
2. **Node Catalog** (optional):

   * Browse to a node catalog JSON file if you have custom nodes
   * Or leave empty to use default node types
3. **Blackboard File** (optional):

   * Browse to a blackboard definition JSON file
   * Or leave empty for default blackboard setup

### Configure Network Settings

In the *Network Configuration* panel:

1. **Model Config File**:

   * Browse to your `model_configs.json` file (defines LLM endpoints)
2. **Node-to-Model Map**:

   * Browse to your `node_to_model_map.json` file (maps nodes to specific models)
3. **API Key**:

   * Enter your LLM service API key
   * Or leave empty if using environment variable / agent registry

### Configure Output Settings

In the *Output Settings* panel:

1. **Output Folder**:

   * Set the root folder where generated files will be saved
   * Example: `/home/user/iap_output`
2. **RAG Cache Folder** (optional):

   * Set a folder for RAG vector store caches
   * Useful for faster subsequent runs with same context

### Run the Pipeline

1. In the *Scenario Execution* panel, enter your scenario
2. Click **Run Pipeline**
3. Monitor the progress through status messages

## Understanding the Output

### Generated Files

After a successful run, the output folder contains:

Output folder structure

```python
output_folder/
├── behavior_tree_folder/
│   └── ...                           # Behavior tree JSON files for each actor
├── bb_cache/
│   └── ...                           # Blackboard variable definitions
├── simulation_event_cache/
│   └── ...                           # Event sequence information
└── rag_storage/
    └── ...                           # RAG vector store files
```

### Behavior Tree JSON Structure

Generated trees follow the OBC format:

Example generated behavior tree

```python
{
    "name": "Alice_PickUpAndPlace",
    "root": {
        "type": "Sequence",
        "children": [
            {
                "type": "MoveToObject",
                "parameters": {
                    "target_path": "/World/Props/CoffeeMug"
                }
            },
            {
                "type": "PickUpObject",
                "parameters": {
                    "object_path": "/World/Props/CoffeeMug"
                }
            },
            {
                "type": "MoveToObject",
                "parameters": {
                    "target_path": "/World/Props/DiningTable"
                }
            },
            {
                "type": "PlaceObject",
                "parameters": {
                    "target_path": "/World/Props/DiningTable"
                }
            }
        ]
    }
}
```

## Python API Reference

For automated workflows, use the Python API directly.

### Basic Usage

Basic behavior tree generation

```python
from isaacsim.agent.planner.core.api import (
    BehaviorTreeGenerationConfig,
    generate_behavior_trees,
)

# Configure the generation
config = BehaviorTreeGenerationConfig(
    scenario="Alice picks up the mug and places it on the table.",
    actor_data_path="/path/to/actors.json",
    object_data_path="/path/to/objects.json",
    node_catalog_path="/path/to/node_catalog.json",
    output_folder_path="/path/to/output",
)

# Generate behavior trees
result = await generate_behavior_trees(config)

if result.success:
    print(f"Generated trees for: {result.actor_ids}")
    print(f"Output folder: {result.behavior_tree_folder_path}")
else:
    print(f"Error: {result.error}")
```

### With Progress Callback

Generation with progress tracking

```python
from isaacsim.agent.planner.core.api import (
    BehaviorTreeGenerationConfig,
    generate_behavior_trees,
    GenerationPhase,
)

def on_progress(phase: GenerationPhase, message: str):
    print(f"[{phase.value}] {message}")

config = BehaviorTreeGenerationConfig(
    scenario="The camera pans across the room.",
    actor_data_path="/path/to/actors.json",
    output_folder_path="/path/to/output",
)

result = await generate_behavior_trees(config, progress_callback=on_progress)
```

### Using the Bridge Pipeline

For Kit-integrated workflows:

Using BridgePipeline for Kit integration

```python
from isaacsim.agent.planner.bridge.pipeline import (
    BridgePipeline,
    PipelineConfig,
    ContextFilesConfig,
    OutputConfig,
)

# Create configuration
config = PipelineConfig(
    scenario="Bob walks to the chair and sits down.",
    context_files=ContextFilesConfig(
        context_file_paths=["/path/to/actors.json", "/path/to/objects.json"],
        node_catalog_path="/path/to/node_catalog.json",
    ),
    output=OutputConfig(
        output_folder_path="/path/to/output",
    ),
)

# Run the pipeline
bridge = BridgePipeline()
result = await bridge.run(config)
```

## Example Scene Setup API

For setting up example scenes programmatically:

### One-Call Setup

Quick example scene setup

```python
from isaacsim.agent.planner.bridge.examples.setup_stage.setup_example_stage import (
    setup_example_stage,
)

# Setup with all defaults
actors = await setup_example_stage()
print(f"Configured actors: {list(actors.keys())}")
```

### Step-by-Step Setup

Detailed example scene setup

```python
from isaacsim.agent.planner.bridge.examples.setup_stage.setup_example_stage import (
    ExampleStageSetup,
)

setup = ExampleStageSetup()

# Step 1: Load context information
context = setup.load_context_info(
    actor_files=["/path/to/actors.json"],
    object_files=["/path/to/objects.json"],
)

# Step 2: Setup OBC environment
await setup.setup_obc_environment()

# Step 3: Configure actors with OBC
actors = setup.configure_actors()

# Step 4: Assign default behavior trees
setup.assign_default_trees(switch_immediately=True)
```

## Troubleshooting

### Common Issues

#### No context files configured

**Cause**: No actor or object JSON files provided.

**Solution**: Add at least one context file in the *Context Cache Files* panel.

#### Pipeline initialization failed

**Cause**: Missing or invalid model configuration.

**Solution**:

* Ensure model config and node-to-model map files exist and are valid JSON
* Verify API key is configured correctly

#### Failed to load example scene

**Cause**: Test stage URL not accessible or OBC not initialized.

**Solution**:

* Ensure you have network access to the test stage location
* Wait for stage to fully load before retrying
* Check the console for specific error messages

#### Empty or partial behavior trees

**Cause**: Actors or objects mentioned in scenario not found in context files.

**Solution**:

* Ensure actor/object IDs in scenario match IDs in JSON files
* Check semantic descriptions for better matching
* Verify prim paths exist in the USD stage

### Debug Tips

1. **Check Console Output**: Detailed logs are written with `[IAP]` or `[IAP Bridge]` prefix
2. **Validate JSON Files**: Ensure all JSON files are valid and have required fields
3. **Verify Prim Paths**: Confirm that prim paths in context files exist in the stage
4. **Test with Simple Scenarios**: Start with single-actor, single-action scenarios

## Best Practices

### Writing Effective Scenarios

1. **Be Specific**: Use actor IDs and object names from your context files

   * Good: *“Alice picks up the CoffeeMug”*
   * Less effective: *“Someone picks up the cup”*
2. **Keep It Simple**: Break complex scenarios into clear steps

   * Good: *“Alice walks to the table. Alice picks up the mug. Alice places the mug on the shelf.”*
   * Less effective: *“Alice does various things with objects around the room”*
3. **Use Natural Language**: Write scenarios as you would describe them to a person

### Organizing Data Files

1. **Separate Actors and Objects**: Keep actor and object definitions in separate files
2. **Use Descriptive IDs**: Choose IDs that reflect the entity’s role or appearance
3. **Include Semantic Descriptions**: Rich descriptions help RAG matching

### Performance Optimization

1. **Use RAG Caching**: Set a persistent RAG cache folder for repeated runs
2. **Limit Context Size**: Include only relevant actors and objects for each scenario
3. **Batch Similar Scenarios**: Process scenarios with similar context together

## Additional Resources

* [IAP Introduction](Synthetic_Data_Generation.md) - Overview and architecture
* [IAP Example Walkthrough](Synthetic_Data_Generation.md) - Quick start tutorial
* [Omni Behavior Composer](../ext_replicator-agent/ext_omni_behavior_composer.html) - OBC reference documentation

---

# Grasping Synthetic Data Generation

This tutorial introduces the `isaacsim.replicator.grasping` extension and its associated UI, `isaacsim.replicator.grasping.ui`. These tools provide a comprehensive workflow for generating synthetic grasping datasets in Isaac Sim.

## Learning Objectives

After completing this tutorial, you will be able to:

* Understand the core components and data flow of the Grasping SDG extension.
* Navigate and utilize the Grasping SDG UI to configure and run grasp generation workflows.
* Define gripper properties, joint states, and multi-step grasp phases.
* Configure object properties and grasp pose sampling parameters.
* Execute and interpret the results of physics-based grasp evaluations.
* Manage grasping configurations using YAML files for saving, loading, and sharing setups.

The extensions are automatically loaded in Isaac Sim, and the UI window can be opened from the main menu using **Tools** > **Replicator** > **Grasping**.

## Getting Started

Before proceeding, it is recommended that you familiarize yourself with:

* [Simulation Fundamentals](Physics.md): For understanding physics simulation concepts and gripper rigging (for example, drive joints).
* [Grasp Editor](Robot_Simulation.md): This tutorial covers related concepts and provides a foundation for grasp definition.

Note

The grasp sampler requires the `libspatialindex` library. If you see related warnings, install it (for example, on Ubuntu: `sudo apt-get install libspatialindex-dev`).

This tutorial utilizes an example stage that includes a pre-configured gripper and objects suitable for grasping exercises. You can find this stage at:

```python
https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0
/Isaac/Samples/Replicator/Stage/sdg_grasping_xarm.usd
```

The stage asset can be found in the **Content Browser** under **Isaac Sim** > **Samples** > **Replicator** > **Stage** > **sdg\_grasping\_xarm.usd**, or can be loaded using by inserting the whole URL in the path field.

The example stage features a gripper with drive joints and three objects equipped with rigid body physics and colliders. Gravity is disabled for these objects to simplify initial interactions. The Grasping UI window typically docks in the **Property** panel upon opening.

## Overview

The extension is designed to automate the process of finding and evaluating potential grasp poses for a given gripper-object pair. At its core, the workflow revolves around several key components and stages:

1. **Configuration**: Defining the specific gripper, the target object, and the parameters that govern how grasps are found and tested.
2. **Grasp Pose Sampling**: Algorithms (for example, antipodal samplers) generate a set of candidate grasp poses around the target object. These poses represent potential ways the gripper might hold the object.
3. **Grasp Execution Phases**: For each candidate grasp, a sequence of actions, termed “Grasp Phases” (for example, move to pre-grasp, close fingers, lift), is simulated. This allows for defining complex, multi-step grasping behaviors analogous to real-world robot actions.
4. **Physics-Based Evaluation**: Each phase of the grasp is simulated in the physics engine. The success or failure of the grasp attempt, along with other metrics (like contact forces, object displacement), can be recorded. In its current state the extension saves the gripper state as result from which the grasps can be evaluated.
5. **Data Logging and Management**: Successful grasps and their associated parameters are logged. The entire setup can be saved to and loaded from configuration files (YAML format), ensuring reproducibility and facilitating batch processing.

The `GraspingManager` class is the central Python API orchestrating these steps, while the UI provides an intuitive way to configure and run this pipeline.

## UI Window Overview

The Grasping UI window provides the interface for setting up and running the grasping simulations workflows. It is organized into several sections, each addressing a specific part of the process. The general workflow involves configuring these sections, typically starting with the gripper and object, then defining the evaluation workflow and simulation parameters, and finally managing the overall configuration.

### Gripper Section

This section is dedicated to defining the properties and behavior of the gripper, which is fundamental for any grasp attempt.

* **Path**: Specify the USD path to the root prim of your gripper (for example, `/World/Robot/gripper_base`).
* **Joints**: After a gripper is selected, its articulated joints are listed. Here you can:

  + **Include/Exclude**: Select the joints that are actively controlled during the grasp phases. These joints have to be drive joints.
  + Set **pre-grasp positions**: Define the initial state for each joint, typically an open configuration, before the grasp sequence begins.
  + Toggle visibility between all joints or of type drive (non-mimic) joints.
* **Grasp Phases**: This powerful feature allows you to define a sequence of discrete actions that constitute a complete grasp attempt. This is analogous to defining a state machine or a sequence of motion primitives for the gripper.

  

  For each phase (for example, “Open”, “Close”), you specify:

  + Target joint positions for the active gripper joints.
  + Simulation step delta time (`dt`) for the physics steps within this phase.
  + Number of simulation steps to execute for this phase.

  Phases can be reordered, deleted, or simulated individually for debugging. If pre-grasp joint positions adequately prepare the gripper (for example, fully open), an explicit “Open” phase might be unnecessary.

### Object Section

This section focuses on specifying the target object and configuring how potential grasp poses are generated for it.

* **Path**: The USD path to the target object prim (for example, `/World/MyObject`).
* **Grasp Pose Sampler**: This configures the algorithm used to find potential grasp poses. This tutorial primarily uses an **antipodal grasp sampler** (implemented in `sampler_utils.py`). An antipodal grasp is typically stable for parallel-jaw grippers, involving two contact points on opposite sides of the object. Key parameters include:

  + **Number of orientations per grasp axis**: How many rotational variations around the primary grasp axis to sample.
  + **Gripper standoff distance**: The distance from the gripper’s Tool Center Point (TCP) or fingertips to the object surface during the approach phase, crucial for avoiding premature collision.
  + **Maximum gripper aperture**: The widest opening of the gripper jaws, filtering out grasps that are too wide for the object.
  + **Alignment axes for the grasp**: Defines local gripper axes to align with object features or the grasp line.
  + **Gripper approach direction**: The vector along which the gripper moves towards the object.
  + **Lateral perturbation (sigma)**: Adds randomness to the grasp point location along the grasp axis, allowing for exploration around nominal contact points.
  + **Random seed**: For ensuring reproducible sampling results.
* **Grasp Poses**: Manages the set of candidate grasp poses generated by the sampler.

  + Specify the desired number of candidate poses.
  + Clear previously generated poses.
  + Visualize the poses in the viewport (either in world or object-local frames) and cycle through them to inspect their placement.

  The following image shows example grasp poses generated by the antipodal sampler on various objects:

  
* **Trimesh**: Provides options for debug visualization of the object’s triangle mesh, which is used internally by the sampler for geometric calculations and collision checks.

Note

The [Measure Tool](https://docs.omniverse.nvidia.com/extensions/latest/ext_measure-tool.html) can be useful for determining values like gripper aperture or standoff distance.

### Workflow Section

The Workflow section is where you orchestrate the actual grasp evaluation process using the configurations defined in the Gripper and Object sections.

The system first saves the gripper’s initial pose. Then, for each generated grasp pose selected for evaluation, it sequentially executes the defined grasp phases within the physics simulation. After all phases for a given pose are completed, the outcome (for example, success based on object stability, contact with target) and other relevant metrics are recorded.

* **Number of Grasps Samples**: Specify how many of the generated grasp poses should be evaluated. Use -1 to evaluate all available poses.
* **Output Path**: Define the directory and base file name for saving the evaluation results. The results are typically saved in a structured format like YAML, detailing each evaluated grasp and its outcome.
* **Overwrite Results**: If enabled, existing result files at the output path will be overwritten. Otherwise, new files will be created (for example, with incremental numbering) to avoid data loss.
* **Start Workflow**: Initiates the grasp evaluation process. The UI will often provide feedback on the progress.

### Simulation Section

This section allows you to fine-tune global parameters that affect how the physics simulation is run during the grasp evaluation.

* **Render each simulation step**: Control whether the viewport updates after each individual physics step within a grasp phase. Disabling this can speed up the evaluation process significantly for large datasets, with rendering potentially only occurring after each full grasp attempt or phase.
* **Simulate using timeline**: Choose between advancing the simulation by stepping the main Isaac Sim timeline or by directly stepping the physics scene. Direct physics steps can offer more precise control for rapid evaluations, while timeline-based simulation might be closer to how a full robot application would run.
* **Isolated physics scene**: Optionally specify a path to a **Physics Scene** prim. If provided, the grasping simulation can be run within this dedicated physics scene, preventing interference from other dynamic objects or physics settings in the main stage. This is useful for ensuring consistent and repeatable grasp evaluations.

### Config Section

The Config section provides the crucial functionality for saving your entire grasping setup to a YAML file and loading it back later. This is essential for reproducibility, sharing configurations, and running batch experiments.

* **File Path**: Specify the path to the YAML configuration file for saving or loading.
* **Config Includes**: Selectively choose which components of the setup are included in the save/load operation. This allows for modular configurations. Options typically include:

  + Gripper Path
  + Joint Pregrasp States
  + Grasp Phases
  + Object Path
  + Sampler Parameters
  + Generated Grasp Poses (if you wish to save a specific set of poses)
* **Overwrite Existing File**: When saving, this option determines if an existing file at the specified path should be overwritten.
* **Load/Save Buttons**: Execute the respective file operations.

This structured UI and configuration system offers detailed control and flexibility for generating diverse grasping datasets.

### Configuration File Example

Below is a snippet illustrating the structure of a YAML configuration file. It can store settings for the gripper, object, sampler, and defined grasp phases. The specific content will depend on which components were selected for inclusion through the ‘Config Includes’ UI options.

xarm\_antipodal.yaml

```python
grasp_phases:
- joint_drive_targets:
    /World/Grippers/xarm_gripper/joints/drive_joint: 48.0
  name: Close
  simulation_step_dt: 0.016666666666666666
  simulation_steps: 32
gripper_path: /World/Grippers/xarm_gripper
joint_pregrasp_states:
  /World/Grippers/xarm_gripper/joints/drive_joint: 0.0
  /World/Grippers/xarm_gripper/joints/left_finger_joint: 0.0
  /World/Grippers/xarm_gripper/joints/left_inner_knuckle_joint: 0.0
  /World/Grippers/xarm_gripper/joints/right_finger_joint: 0.0
  /World/Grippers/xarm_gripper/joints/right_inner_knuckle_joint: 0.0
  /World/Grippers/xarm_gripper/joints/right_outer_knuckle_joint: 0.0
  /World/Grippers/xarm_gripper/root_joint: 0.0
sampler_config:
  grasp_align_axis:
  - 0
  - 1
  - 0
  gripper_approach_direction:
  - 0
  - 0
  - 1
  gripper_maximum_aperture: 0.08
  gripper_standoff_fingertips: 0.17000000178813934
  lateral_sigma: 0.0
  num_candidates: 100
  num_orientations: 1
  orientation_sample_axis:
  - 0
  - 1
  - 0
  random_seed: -1
  sampler_type: antipodal
  verbose: false
```

## Code Example

The following scripts demonstrates a complete workflow for generating a grasping dataset using the `GraspingManager` API. This script programmatically performs the steps configurable through the UI:

* opening a stage
* setting up the `GraspingManager` (potentially by loading a configuration file)
* generating grasp Poses
* evaluating these poses using physics simulation
* saving the results

This approach is highly suitable for batch processing or integration into larger robotics workflows. The script can be run directly from the [Script Editor](Development_Tools.md) or as a [Standalone Application](Workflows.md).

To run the standalone example from the terminal (on Windows, use `python.bat` instead of `python.sh`):

```python
./python.sh standalone_examples/api/isaacsim.replicator.grasping/grasping_workflow_sdg.py
```

Script Editor

Grasping Synthetic Data Generation Workflow

```python
import asyncio
import os

import omni.kit.app
import omni.usd
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.replicator.grasping.grasping_manager import GraspingManager
from isaacsim.storage.native import get_assets_root_path_async

async def run_example_async(
    stage_path,
    config_path=None,
    sampler_config=None,
    physics_scene_path=None,
    output_dir=None,
    gripper_path=None,
    object_prim_path=None,
):
    assets_root_path = await get_assets_root_path_async()
    print(f"Assets root path: {assets_root_path}")
    stage_url = assets_root_path + stage_path
    print(f"Opening stage: {stage_url}")
    await omni.usd.get_context().open_stage_async(stage_url)
    stage = omni.usd.get_context().get_stage()

    grasping_manager = GraspingManager()

    if config_path is not None:
        load_status = grasping_manager.load_config(config_path)
        print(f"Config load status: {load_status}")

    # Make sure the object to grasp is set (either from the config file or from the argument)
    if not grasping_manager.get_object_prim_path() and object_prim_path:
        grasping_manager.object_path = object_prim_path

    if not grasping_manager.get_object_prim_path():
        print("Warning: Object to grasp is not set (missing in config and argument). Aborting.")
        return

    # Make sure the gripper is set (either from the config file or from the argument)
    if not grasping_manager.gripper_path and gripper_path:
        grasping_manager.gripper_path = gripper_path

    if not grasping_manager.gripper_path:
        print("Warning: Gripper path is not set (missing in config and argument). Aborting.")
        return

    # If there are already grasp poses in the configuration, don't generate new ones
    if grasping_manager.grasp_locations:
        print(
            f"Found {len(grasping_manager.grasp_locations)} grasp poses in the configuration file. No new poses will be generated."
        )
    else:
        print("No grasp poses found in configuration, generating new ones...")

        # Determine Sampler Configuration
        if not (grasping_manager.sampler_config and grasping_manager.sampler_config.get("sampler_type")):
            if sampler_config:
                grasping_manager.sampler_config = sampler_config.copy()
            else:
                print(
                    "Warning: Sampler configuration is missing or invalid (not in config file and not provided as argument). Aborting pose generation."
                )
                return

        # Generate the grasp poses
        success_generation = grasping_manager.generate_grasp_poses()
        if not success_generation or not grasping_manager.grasp_locations:
            print("Failed to generate grasp poses or no poses were generated.")
            return
        print(f"Generated {len(grasping_manager.grasp_locations)} new grasp poses.")

    # Store the initial gripper pose to be able to restore it after the evaluation
    grasping_manager.store_initial_gripper_pose()

    print("Evaluating grasp poses...")
    poses_to_evaluate = grasping_manager.get_grasp_poses(in_world_frame=True)
    if not poses_to_evaluate:
        print("No poses available to evaluate..")
        return

    # Determine Output Path
    if not output_dir:
        print("Warning: Output path is not defined data will not be saved.")

    # Set the output path and overwrite flag
    grasping_manager.set_results_output_dir(output_dir)
    grasping_manager.set_overwrite_results_output(True)

    # Determine Physics Scene Path
    physics_scene_path_for_eval = None
    if physics_scene_path and stage.GetPrimAtPath(physics_scene_path):
        physics_scene_path_for_eval = physics_scene_path
    print(f"Physics scene path for evaluation: {physics_scene_path_for_eval}")

    await grasping_manager.evaluate_grasp_poses(
        grasp_poses=poses_to_evaluate,
        render=True,
        physics_scene_path=physics_scene_path_for_eval,
        simulate_using_timeline=False,
    )

    print("Grasping workflow example finished.")
    grasping_manager.clear()

stage_path = "/Isaac/Samples/Replicator/Stage/sdg_grasping_xarm.usd"
ext_path = get_extension_path_from_name("isaacsim.replicator.grasping")
config_path = os.path.join(ext_path, "data/gripper_configs/xarm_antipodal_soup_can.yaml")
output_dir = os.path.join(os.getcwd(), "xarm_antipodal")

asyncio.ensure_future(run_example_async(stage_path=stage_path, config_path=config_path, output_dir=output_dir))
```

Standalone Application

Grasping Synthetic Data Generation Workflow

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import asyncio
import os

import omni.kit.app
import omni.usd
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.storage.native import get_assets_root_path

# Make sure the grasping extension is loaded and enabled
ext_manager = omni.kit.app.get_app().get_extension_manager()
if not ext_manager.is_extension_enabled("isaacsim.replicator.grasping"):
    ext_manager.set_extension_enabled_immediate("isaacsim.replicator.grasping", True)
from isaacsim.replicator.grasping.grasping_manager import GraspingManager

def run_example(
    stage_path,
    config_path=None,
    sampler_config=None,
    physics_scene_path=None,
    output_dir=None,
    gripper_path=None,
    object_prim_path=None,
):
    assets_root_path = get_assets_root_path()
    print(f"Assets root path: {assets_root_path}")
    stage_url = assets_root_path + stage_path
    print(f"Opening stage: {stage_url}")
    omni.usd.get_context().open_stage(stage_url)
    stage = omni.usd.get_context().get_stage()

    grasping_manager = GraspingManager()

    if config_path is not None:
        load_status = grasping_manager.load_config(config_path)
        print(f"Config load status: {load_status}")

    # Make sure the object to grasp is set (either from the config file or from the argument)
    if not grasping_manager.get_object_prim_path() and object_prim_path:
        grasping_manager.object_path = object_prim_path

    if not grasping_manager.get_object_prim_path():
        print("Warning: Object to grasp is not set (missing in config and argument). Aborting.")
        return

    # Make sure the gripper is set (either from the config file or from the argument)
    if not grasping_manager.gripper_path and gripper_path:
        grasping_manager.gripper_path = gripper_path

    if not grasping_manager.gripper_path:
        print("Warning: Gripper path is not set (missing in config and argument). Aborting.")
        return

    # If there are already grasp poses in the configuration, don't generate new ones
    if grasping_manager.grasp_locations:
        print(
            f"Found {len(grasping_manager.grasp_locations)} grasp poses in the configuration file. No new poses will be generated."
        )
    else:
        print("No grasp poses found in configuration, generating new ones...")

        # Determine Sampler Configuration
        if not (grasping_manager.sampler_config and grasping_manager.sampler_config.get("sampler_type")):
            if sampler_config:
                grasping_manager.sampler_config = sampler_config.copy()
            else:
                print(
                    "Warning: Sampler configuration is missing or invalid (not in config file and not provided as argument). Aborting pose generation."
                )
                return

        # Generate the grasp poses
        success_generation = grasping_manager.generate_grasp_poses()
        if not success_generation or not grasping_manager.grasp_locations:
            print("Failed to generate grasp poses or no poses were generated.")
            return
        print(f"Generated {len(grasping_manager.grasp_locations)} new grasp poses.")

    # Store the initial gripper pose to be able to restore it after the evaluation
    grasping_manager.store_initial_gripper_pose()

    print("Evaluating grasp poses...")
    poses_to_evaluate = grasping_manager.get_grasp_poses(in_world_frame=True)
    if not poses_to_evaluate:
        print("No poses available to evaluate..")
        return

    # Determine Output Path
    if not output_dir:
        print("Warning: Output path is not defined data will not be saved.")

    # Set the output path and overwrite flag
    grasping_manager.set_results_output_dir(output_dir)
    grasping_manager.set_overwrite_results_output(True)

    # Determine Physics Scene Path
    physics_scene_path_for_eval = None
    if physics_scene_path and stage.GetPrimAtPath(physics_scene_path):
        physics_scene_path_for_eval = physics_scene_path
    print(f"Physics scene path for evaluation: {physics_scene_path_for_eval}")

    grasping_workflow_task = asyncio.ensure_future(
        grasping_manager.evaluate_grasp_poses(
            grasp_poses=poses_to_evaluate,
            render=True,
            physics_scene_path=physics_scene_path_for_eval,
            simulate_using_timeline=False,
        )
    )

    while not grasping_workflow_task.done():
        simulation_app.update()

    print("Grasping workflow example finished.")
    grasping_manager.clear()

stage_path = "/Isaac/Samples/Replicator/Stage/sdg_grasping_xarm.usd"
ext_path = get_extension_path_from_name("isaacsim.replicator.grasping")
config_path = os.path.join(ext_path, "data/gripper_configs/xarm_antipodal_soup_can.yaml")
output_dir = os.path.join(os.getcwd(), "xarm_antipodal")

run_example(stage_path=stage_path, config_path=config_path, output_dir=output_dir)

simulation_app.close()
```

---

# Data Generation with MobilityGen

MobilityGen is a toolset built on NVIDIA Isaac Sim
that enables you to generate and collect data for mobile robots.

MobilityGen supports:

* Many robot types
  :   + Differential drive - Jetbot, Carter
      + Quadruped - Spot
      + Humanoid - H1
* Many data collection methods
  :   + Manual - Keyboard Teleoperation, Gamepad Teleoperation
      + Automated - Random Accelerations, Random Path Following

## Generate Data with MobilityGen

### Build an Occupancy Map

You must create an occupancy map of your environment.

This tutorial uses an example warehouse scene.

1. Load the warehouse stage:

   1. Open Content Browser if it’s not already open (**Window > Browsers > Content**).
   2. Load the warehouse USD file in Isaac Sim/Environments/Simple\_Warehouse/warehouse\_multiple\_shelves.usd.
2. Create the occupancy map:

   1. Select **Tools > Robotics > Occupancy Map** to open the Occupancy Map extension.
   2. In the **Occupancy Map** window set **Origin** to:

      * `X`: `2.0`
      * `Y`: `0.0`
      * `Z`: `0.0`

      To input a value in the text box, `ctrl + left click` to activate the input mode.
   3. In the **Occupancy Map** window set **Upper Bound** to:

      * `X`: `10.0`
      * `Y`: `20.0`
      * `Z`: `2.0` (Assumes the robot can move under two meter overpasses)
   4. In the **Occupancy Map** window set **Lower Bound** to:

      * `X`: `-14.0`
      * `Y`: `-18.0`
      * `Z`: `0.1` (Assume the robot can move over `5cm` bumps)

      Please note, the coordinates specified for the occupancy upper and lower bound define a bounding box within the warehouse\_multiple\_shelves.usd scene that we want the robot to be able to navigate. We’ve pre-selected values that cover the main floor area.
      When using a different scene, you may adjust these bounds to cover the area suitable for your USD scene.
   5. Click **Calculate** to generate the occupancy map.
   6. Click **Visualize Image** to view the occupancy map.
   7. In the **Visualization** window under **Rotate Image** select **180**.
   8. In the **Visualization** window under **Coordinate Type** select **ROS Occupancy Map Parameters File YAML**.

      Please note, we must rotate the image and generate a ROS formatted YAML file, because MobilityGen expects Occupancy Maps saved in this format, and will not work with other formats.
   9. Click **Regenerate Image**.
   10. Copy the YAML text generated to your clipboard.
   11. In a text editor of your choice, create a new file named `~/MobilityGenData/maps/warehouse_multiple_shelves/map.yaml`.

       On Windows replace ~ with the directory of your choice.
   12. Paste the YAML text copied from the **Visualization** window into the created file.
   13. Edit the line `image: warehouse_multiple_shelves.png` to read `image: map.png`.
   14. Save the file.
   15. Back in the **Visualization** window, click **Save Image**.
   16. In the tree explorer, open the folder `~/MobilityGenData/maps/warehouse_multiple_shelves`.
   17. Under the file name enter `map.png`.
   18. Click save.

Verify that you now have a folder named `~/MobilityGenData/maps/warehouse_multiple_shelves/` with
a file named `map.yaml` and `map.png` inside.

### Record a Trajectory

After creating a map of the environment, you can generate data with MobilityGen:

1. Enable the MobilityGen UI extension.

   1. Navigate to **Window** > **Extensions** and search for **MobilityGen UI**.
   2. Click the toggle switch for the **MobilityGen UI** extension.
   3. Verify that two windows open. One window is the MobilityGen UI, the other is to display the Occupancy Map and visualizations. One window might be hiding behind the other when they first appear, so we recommend dragging them into a window pane to view both at the same time.

1. Build the scenario:

   1. In the **MobilityGen** window under **Stage** paste the following USD:

      <http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd>
   2. In the **MobilityGen** window under **Occupancy Map** enter the path to the `map.yaml` file created previously.

      ~/MobilityGenData/maps/warehouse\_multiple\_shelves/map.yaml
   3. Under the **Robot** dropdown select **H1Robot**.
   4. Under the **Scenario** dropdown select **KeyboardTeleoperationScenario**.
   5. Click **Build**.

      After a few seconds, verify that the scene and occupancy map appear.

1. Test drive the robot using the following keys:

   * `W` - Move forward
   * `A` - Turn left
   * `S` - Move backwards
   * `D` - Turn right
2. Start recording:

   1. Click **Start recording** to start recording a log.
   2. Move the robot around.
   3. Click **Stop recording** to stop recording.

The data is now recorded to `~/MobilityGenData/recordings` by default.

### Replay and Render

After recording a trajectory, which includes
data, like robot poses, you can *replay* the scenario.

To do this, use the `replay_directory.py` Python script that ships with Isaac Sim.

To run the script call the following from inside the Isaac Sim directory:

```python
./python.sh standalone_examples/replicator/mobility_gen/replay_directory.py --render_interval 40 --enable isaacsim.replicator.mobility_gen.examples
```

The arguments to this script are

* –input: The path to the input recordings.
* –output: The path to output the recordings with rendered sensor data.
* –rgb\_enabled: Set true to enable RGB image rendering.
* –segmentation\_enabled: Set true to enable semantic segmentation image rendering.
* –depth\_enabled: Set true to enable depth image rendering.
* –instance\_id\_segmentation\_enabled: Set true to enable instance segmentation image rendering.
* –normals\_enabled: Set true to enable surface normal image rendering.
* –render\_rt\_subframes: The number of subframes for RT rendering. Increase this number to improve rendering quality at the cost of speed.
* –render\_interval: The number of physics steps per rendering. For example, setting this value to 2 will render only once every 2 physics timesteps.

After the script finishes, verify that you have a folder `~/MobilityGenData/replays`, which contains
the rendered sensor data.

You can open this folder to explore the data. Some data (like segmentation masks) can be difficult to visualize using the file browser alone.

Fortunately, there are many examples on how to load and work with the recorded data in the open source [MobilityGen GitHub Repository](https://github.com/NVlabs/MobilityGen/tree/dev-external-occupancy-map-generation/examples). We recommend visualizing your recorded data by running the [Gradio Visualization Script](https://github.com/NVlabs/MobilityGen/blob/main/examples/04_visualize_gradio.py).

To run this example you would clone the above repository and run the following command from a Python interpreter with Gradio installed

```python
python examples/04_visualize_gradio.py --input_dir ~/MobilityGenData/replays
```

You can also check the [reader.py](https://github.com/NVlabs/MobilityGen/blob/main/examples/reader.py) file for a helper class for reading the data in Python.

## Tips

### Generate Procedural Data

Generating procedural mobility data with MobilityGen is done very similar to the basic teleoperation workflow above.

To generate procedural data:

1. Follow `Build an Occupancy Map` above to create an occupancy map of the environment.
2. Follow `Record a Trajectory` above, but select `RandomPathFollowingScenario` instead of `KeyboardTeleoperationScenario`.
   - You no longer need to manually teleoperate the robot. When the scenario is built, it will run and reset automatically.
   - You do need to hit “start recording” to enable recording to disk. However, when the scenario resets, a new recording will be created automatically.
   - Verify that you have recordings collected in the `~/MobilityGenData` folder the same as above.
3. Follow `Replay and render` above to render the sensor data from the recorded trajectories.

The process for other procedural scenarios (like `RandomAccelerationScenario`) is similar.

### Add a Custom Robot

You can implement a new robot for use with MobilityGen. This involves editing the `robots.py` file in the MobilityGen Examples extension.

The general workflow is as follows:

1. Open the `robots.py` file in an editor of choice. This is located at `<isaac sim path>/exts/isaacsim.replicator.mobility_gen.examples/isaacsim/replicator/mobility_gen/examples/robots.py`.
2. Create a new class that subclasses the `MobilityGenRobot` class. Alternatively, if your robot fits one of the existing implementations (like `WheeledMobilityGenRobot`), you can subclass that.

   * We recommend starting by reviewing an existing robot implementation in `robots.py`, to get started. A good way to start is by customizing an existing robot.
3. If you are starting from scratch, implement the required abstract methods of `MobilityGenRobot` class:

   * Implement the `build()` method. This method is responsible for adding the robot to the USD stage.
   * Implement the `write_action()` method. This method takes as input a linear and angular velocity command and performs any control logic.
   * Overwrite common class parameters (like physics\_dt).
4. Register the robot class by using the `ROBOT.register()` decorator. This makes the custom robot discoverable by MobilityGen.

After implementing this in the file above, save the file.

When you restart Isaac Sim, verify that the new robot is registered, in the MobilityGen UI, and ready for data collection.

Because the registration of a new robot requires editing the Isaac Sim build file, make a copy of your `robot.py` externally so you do not lose it.

When defining your robot, you may find the following list of common parameters and their descriptions helpful

* physics\_dt: The physics timestep to use for simulating the robot.
* z\_offset: The Z-axis offset height to spawn the robot.
* chase\_camera\_base\_path: The relative USD path which will be used to spawn the third person view camera. This is typically set to the robot base frame.
* chase\_camera\_x\_offset: The relative X-axis offset to spawn the third person view camera.
* chase\_camera\_z\_offset: The relative Z-axis offset to spawn the third person view camera.
* chase\_camera\_tilt\_angle: The tilt angle to apply to the third person view camera.
* occupancy\_map\_radius: The robot footprint radius to use for spawning and path planning.
* occupancy\_map\_collision\_radius: The robot footprint radius to use for collision based episode termination.
* front\_camera\_type: The static class representing the front camera.
* front\_camera\_base\_path: The relative USD path to spawn the front camera.
* front\_camera\_rotation: The relative XYZ rotation used when spawning the front camera.
* front\_camera\_translation: The relative XYZ translation used when spawning the front camera.
* keyboard\_linear\_velocity\_gain: The gain used to map keyboard button presses to the robot’s linear velocity. A larger gain results in faster movement.
* keyboard\_angular\_velocity\_gain: The gain used to map keyboard button presses to the robot’s angular velocity. A larger gain results in faster movement.
* gamepad\_linear\_velocity\_gain: The gain used to map gamepad axis movement to the robot’s linear velocity. A larger gain results in faster movement.
* gamepad\_angular\_velocity\_gain: The gain used to map gamepad axis movement to the robot’s angular velocity. A larger gain results in faster movement.
* random\_action\_linear\_velocity\_range: The robot linear velocity limits for the random acceleration scenario.
* random\_action\_angular\_velocity\_range: The robot angular velocity limits for the random acceleration scenario.
* random\_action\_linear\_acceleration\_std: The standard deviation used for sampling the robot linear acceleration each timestep during the random acceleration scenario.
* random\_action\_angular\_acceleration\_std: The standard deviation used for sampling the robot angular acceleration each timestep during the random acceleration scenario.
* random\_action\_grid\_pose\_sampler\_grid\_size: The grid size to use for spawning the robot during the random acceleration scenario.
* path\_following\_speed: The constant linear speed to use for the path following scenario.
* path\_following\_angular\_gain: The gain used for the proportional steering control in the path following scenario. A larger gain results in quicker turning, but potential overshoot and wobbling.
* path\_following\_stop\_distance\_threshold: The distance threshold at which point the robot will stop. Applies to the path following scenario.
* path\_following\_forward\_angle\_threshold: The angle threshold at which point the robot will move forward. Applies to the path following scenario.
* path\_following\_target\_point\_offset\_meters: The offset distance used to generate the ‘target point’ that the robot will follow in the path following scenario. A larger offset results in smoother motion, but too large may cause the robot to cut corners during turns.

### Visualize Trajectory with Gradio

python scripts/gradio\_visualization.py –log\_dir ~/MobilityGenData/replays/<your\_recording\_folder>

## Next Steps

In this tutorial, you:

1. Built an occupancy map for use with MobilityGen.
2. Recorded a MobilityGen trajectory using the H1 robot with keyboard Teleoperation.
3. Rendered sensor data based on the recorded trajectory.

As next steps, try recording data:

* for a different robot (for example: Spot)
* using a different scenario (for example: Random Path Following)