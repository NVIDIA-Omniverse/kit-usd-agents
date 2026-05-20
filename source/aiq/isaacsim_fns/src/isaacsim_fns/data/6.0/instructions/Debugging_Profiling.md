# Debugging & Profiling

## Debugging

Isaac Sim supports several debugging extensions for visualization and inspection, including:

- Debug Drawing Extension API
- Omniverse Commands Tool Extension
- Debugging With Visual Studio Code

## Profiling

- Profiling Performance Using Tracy

---

# Debug Drawing Extension API

## About

This [Debug Drawing Extension API](#isaac-debug-draw) API is used to coordinate groups of lines and points on the screen.
Use this API instead of Omniverse’s built-in debug drawing API to have greater control over how the geometry is drawn.
The 3D geometry drawn by this remains persistent across frames and is only cleared when desired (unlike the built-in debug drawer).

## API Documentation

See the [API Documentation](../../py/docs/extsbuild/isaacsim.util.debug_draw/docs/index.html) for complete usage information.

## Tutorials & Examples

The following screenshots showcase how the different geometries are drawn:

### Points

Drawing batches of points with different RGBA and radius values:

> ```python
> import random
>
> from isaacsim.util.debug_draw import _debug_draw
>
> draw = _debug_draw.acquire_debug_draw_interface()
>
> N = 10000
> point_list_1 = [(random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(N)]
> point_list_2 = [(random.uniform(-10, 10), random.uniform(10, 30), random.uniform(-10, 10)) for _ in range(N)]
> point_list_3 = [(random.uniform(-10, 10), random.uniform(-30, -10), random.uniform(-10, 10)) for _ in range(N)]
> colors = [(random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), 1) for _ in range(N)]
> sizes = [random.randint(1, 50) for _ in range(N)]
> draw.draw_points(point_list_1, [(1, 0, 0, 1)] * N, [10] * N)
> draw.draw_points(point_list_2, [(0, 1, 0, 1)] * N, [10] * N)
> draw.draw_points(point_list_3, colors, sizes)
> ```

### Lines

Drawing batches of lines with different RGBA and width values:

> ```python
> import random
>
> from isaacsim.util.debug_draw import _debug_draw
>
> draw = _debug_draw.acquire_debug_draw_interface()
>
> N = 10000
> point_list_1 = [(random.uniform(10, 30), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(N)]
> point_list_2 = [(random.uniform(10, 30), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(N)]
> colors = [(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), 1) for _ in range(N)]
> sizes = [random.randint(1, 25) for _ in range(N)]
> draw.draw_lines(point_list_1, point_list_2, colors, sizes)
> ```

### Splines

Drawing splines as filled or dashed between a set of points:

> ```python
> import random
>
> from isaacsim.util.debug_draw import _debug_draw
>
> draw = _debug_draw.acquire_debug_draw_interface()
>
> point_list_1 = [(random.uniform(-30, -10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(10)]
> draw.draw_lines_spline(point_list_1, (1, 1, 1, 1), 10, False)
> point_list_2 = [(random.uniform(-30, -10), random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(10)]
> draw.draw_lines_spline(point_list_2, (1, 1, 1, 1), 1, True)
> ```

---

# Omniverse Commands Tool Extension

## About

The [Omniverse Commands Tool Extension](#isaac-sim-command-tool) provides an interface that connects the UI operations in Omniverse and Isaac Sim
to their corresponding Python commands.

To access this extension, go to the top menu bar and click Window > Commands.

## User Interface

### Configuration Options

Below are the options that are supported:

* **Search commands**: Search for all the commands that can be executed.
* **Clear History**: Clears history for all the commands that have been executed and show up in history.
* **Top-level commands**: Generate Python scripts corresponding to all top-level commands in history and copy to clipboard.
* **Selected commands**: Generate Python scripts corresponding to selected commands in history and copy to clipboard.

## Tutorials & Examples

The following example demonstrates a simple scenario of creating and transforming a cube followed by changing via the UI.
It then shows how to use the [Omniverse Commands Tool Extension](#isaac-sim-command-tool) to generate the corresponding Python command to replicate the scenario.

---

# Debugging With Visual Studio Code

## Learning Objectives

In this tutorial, we will go over

* Debugging a standalone Python script
* Debugging [Python Scripts Running in Docker](#isaac-sim-app-tutorial-advanced-python-debugging-docker).
* Attaching to the `omni.kit.debug.vscode_debugger` extension to debug a running instance of Isaac Sim

## Standalone Python Scripts

Note

Debugging standalone Python scripts is only supported on Linux currently

1. Open a terminal in the Isaac Sim installation root folder, then execute the following command: `code .` This launches a new VS Code window and opens the current folder. You can also launch VS Code and open the folder.
2. Let’s try debugging a simple script, open `standalone_examples/api/isaacsim.simulation_app/hello_world.py` and place a breakpoint.
3. Select the “Run” icon from the toolbar on the left, and ensure “Current File” is selected from the configuration dropdown menu.
4. Click “Start Debugging” or press F5 to launch the debugger. Pressing F10 will step line by line. You can mouse over to examine variable values.

   
5. Stop the current debugging session and let’s try passing a command-line argument to our code in the “args” field of the .vscode/launch.json file. For example, here we change the default nucleus server

   > ```python
   >     {
   >         "name": "Python: Current File",
   >         "type": "python",
   >         "request": "launch",
   >         "program": "${file}",
   >         "console": "integratedTerminal",
   >         "env": {
   >             "EXP_PATH": "${workspaceFolder}/apps",
   >             "RESOURCE_NAME": "IsaacSim"
   >         },
   >         "python": "${workspaceFolder}/kit/python/bin/python3",
   >         "envFile": "${workspaceFolder}/.vscode/.standalone_examples.env",
   >         "preLaunchTask": "setup_python_env",
   >         "args": ["--/persistent/isaac/asset_root/default=\"omniverse://my_server\""]
   >     }
   > ```
6. Add the following lines to `hello_world.py` and place a breakpoint on the `print(server_check)` line.

   > ```python
   > # The most basic usage for creating a simulation app
   > from isaacsim import SimulationApp
   >
   > kit = SimulationApp()
   > import carb
   >
   > server_check = carb.settings.get_settings().get_as_string("/persistent/isaac/asset_root/default")
   > print(server_check)
   > for i in range(100):
   >     kit.update()
   > kit.close()  # Cleanup application
   > ```
7. After modifying and saving the launch.json, press F5 to launch the debugger.
8. Verify that the variable contains the server set in the `args` in `launch.json`

   

## Python Scripts Running in Docker

You can debug a Python script running headless in a docker container.

1. [Deploy the container](Installation.md) and run it with an interactive Bash session.
2. In the running container, install `debugpy`:

   > ```python
   > # ./python.sh -m pip install debugpy
   > ```
3. Create a new debugging configuration in VS Code with (“Run” menu > “Add Configuration…” > “Python Debugger” > “Remote Attach”, choose: host “localhost” and port “5678”).
4. Make sure the pathMappings are correct with `/isaac-sim` in the container mapping to the folder where you have Isaac Sim installed locally. These paths should match the configuration in your vscode `launch.json`:

   > ```python
   > {
   >     "name": "Python Debugger: Docker Attach",
   >     "type": "debugpy",
   >     "request": "attach",
   >     "connect": {"host": "localhost", "port": 5678},
   >     "pathMappings": [{"localRoot": "${workspaceFolder}/_build/linux-x86_64/release", "remoteRoot": "/isaac-sim"}],
   > },
   > ```
5. You must still use `./python.sh` to run Python scripts, but to debug them you have to add `-m debugpy --wait-for-client --listen 0.0.0.0:5678` after `./python.sh` and before the Python file.
6. As an example, open `standalone_examples/api/isaacsim.core.api/time_stepping.py` in VS Code and set a breakpoint by clicking on the margin to the left of a line of code.
7. Now start run `time_stepping.py` in the docker container with the complete debugging command:

   > ```python
   > # ./python.sh -m debugpy --wait-for-client --listen 0.0.0.0:5678 standalone_examples/api/isaacsim.core.api/time_stepping.py
   > ```
8. Because of the `--wait-for-client` flag, the script will not start right away. You must attach the debugger first by selecting it in VS Code’s debug window and pressing the Play button.
9. The script should start in the docker window, and stop at the breakpoint inside VS Code.

Note

If the path mappings are incorrect you will not be able to set breakpoints or step through code.

## Attaching the Debugger to a Running App

To debug a script you are already running, use the VS Code Debugger extension.

1. Launch Isaac Sim, and from the top toolbar, select Window > Extensions. Then search for “vscode” and click the Enable button for the `omni.kit.debug.vscode` extension.
   By default, the status will show “VS Code Debugger Unattached” in red text.

   
2. Then launch VS Code, and select the “Run” icon from the toolbar on the left.
3. From the configuration menu, select “Python: Attach (windows-x86\_64/linux-x86\_64) and click the green arrow to start debugging.
4. Notice that the status in Isaac Sim changes to “VS Code Debugger Attached” in blue text.

   
5. You can now return to your Python file in VS Code and add breakpoints to debug, as described above.

Note

To configure the host and port used for debugging, the following command-line arguments can be provided

```python
--/exts/omni.kit.debug.python/host="127.0.0.1"
--/exts/omni.kit.debug.python/port=3000
```

These should match the configuration in your vscode `launch.json`

> ```python
> {
>     "name": "Python: Attach (windows-x86_64/linux-x86_64)",
>     "type": "python",
>     "request": "attach",
>     "port": 3000,
>     "host": "127.0.0.1",
> },
> ```

## Summary

In this tutorial, we covered
#. Debugging a standalone Python script
#. Attaching the vscode debugger to a running instance of Isaac Sim

### Further Learning

For more details about how the vscode integration works, refer to [Visual Studio Code (VS Code)](Development_Tools.md)

---

# Profiling Performance Using Tracy

## Learning Objectives

This tutorial shows how to get a high-level live CPU/GPU performance overview of NVIDIA Isaac Sim using the Tracy profiler.

After this tutorial, you will be able to gauge the performance of various components of the application, add profiling zones and understand the relative significance of each zone.

Tracy also has a lot of other useful features for performance analysis, not covered in this tutorial. Refer to the Tracy documentation for more details on analyzing zone stats, filtering, and other features.

Note

Profiling the application can add some overhead to the simulation. When evaluating performance, a good workflow is to profile the application, try optimizations, and then profile again to see the impact. However, when evaluating final performance, disable profiling to get the most accurate results.

With already fast code, sometimes profiling itself is the bottleneck. Disabling profiling and running the application without it can help identify if this is the case.

*15-20 Minutes Tutorial*

## Getting Started

**Prerequisites**

* Review the Core API [Hello World](Python_Scripting_and_Tutorials.md) and GUI Tutorial series [Troubleshooting](Help_FAQ.md) prior to beginning this tutorial.
* Have an understanding of various workflows. Refer to [Workflows](Workflows.md) for details.

## Launching Tracy Profiler

The first step of profiling the application is to open the Tracy profiler. There are a few different ways to do this.

* It is recommended to use the Tracy binary that comes with NVIDIA Isaac Sim. To do this, you need to enable the `omni.kit.profiler.tracy` extension from the registry which contains the currently supported version of Tracy. To do this, navigate to **Windows > Extensions**, search for `omni.kit.profiler.tracy` extension and enable it. If you need the extension to be enabled by default, you can check the **AUTOLOAD** box as well. This will add a new **Profiler** menu item from where you can **Launch** the profiler or **Launch and Connect** an instance of the Tracy UI and stream the output of the NVIDIA Isaac Sim to it.
* Another convenient approach to open the Tracy applicaton is to use the binary that is used by the `omni.kit.profiler.tracy` extension manually. The binary is located inside the extension folder, e.g.

  > ```python
  > ./extscache/omni.kit.profiler.tracy-1.2.0+lx64/bin/Tracy
  > ```

Note

You can keep using the same instance of the profiler even if you close the NVIDIA Isaac Sim application. This is useful to keep Tracy profiler ready when you profile the Application in the standalone workflow.
However the same Isaac Sim instance can only be connected to Tracy once. If you need to connect to Tracy again, you will need to restart the Isaac Sim instance.

## Using Tracy Profiler

### GUI Workflow

1. Enable/Open Tracy based on the instructions above.
2. **Launch and Connect** the Tracy profiler will open the profiler windows.
3. Let the simulation run for a few seconds to collect some data.
4. Press **Stop** in the profiler window to stop the profiler. You can also press **Pause** if you wish to continue profiling later.You can press **Save trace** from the “net icon” to save the trace file.

Note

Pressing **Stop** will end the profiling session. In order to continue profiling, you will need to launch a new instance of Isaac Sim. Using **Pause** and **Resume** will allow you to continue profiling from the same session.

### Standalone Workflow

1. Enable/Open Tracy based on the instructions above. You can then close the Isaac Sim application, the Tracy profiler window will remain open.
2. Note that you need to change the `SimulationApp` parameters of your standalone script to include the *profiler\_backend* parameter as follows. This adds some useful capture options for the profiler.

   > ```python
   > from isaacsim import SimulationApp
   >
   > simulation_app = SimulationApp({"headless": False, "profiler_backend": ["tracy"]})
   > # Add your standalone script here
   > ```
3. Launch your standalone script with `--enable omni.kit.profiler.tracy` as follows:

   > ```python
   > python.sh PATH_TO_STANDALONE_EXAMPLE --enable omni.kit.profiler.tracy
   > ```
4. Once the application is running you can **Connect** the Tracy profiler instance to get the live performance data.

Note

If you are running in *non-headless* mode you can use the Tracy profiler by following the GUI Workflow instructions above.

For more fine-grained control, it is possible to customize the profiler output by adding additional command line arguments which works similarly for any kit-based app. For instance, you can run a standalone example with Tracy enabled as follows:

```python
python.sh PATH_TO_STANDALONE_EXAMPLE --enable omni.kit.profiler.tracy \
    --/profiler/enabled=true \
    --/app/profilerBackend=tracy \
    --/privacy/externalBuild=0 \
    --/app/profileFromStart=true \
    --/profiler/gpu=true \
    --/profiler/gpu/tracyInject/enabled=true
    --/profiler/gpu/tracyInject/msBetweenClockCalibration=0 \
    --/app/profilerMask=1 \
    --/profiler/channels/carb.tasking/enabled=false \
    --/profiler/channels/carb.events/enabled=false \
    --/plugins/carb.profiler-tracy.plugin/fibersAsThreads=false
```

Note

Above `--/profiler/enabled` and `--/app/profilerBackend` are the only necessary command line arguments, and the rest are optional parameters to customize the profiler output. `--/privacy/externalBuild=0` is necessary to capture traces with **F5** key.

All these parameters (with the default values mentioned above) are already included in the first method described above.

## Adding Profiling Zones

### Python

To add profiler zones in your Python script, you can use the `@carb.profiler.profile` function decorator to add a zone for a specific function.

> ```python
> import carb
>
>
> @carb.profiler.profile
> def some_function():
>     # function code here
>     return
> ```

More fine-grained control can be achieved by encapsulating the required code in a pair of `carb.profiler.begin` and `carb.profiler.end` calls to manually start and stop a zone as follows:

> ```python
> import carb
>
>
> @carb.profiler.profile
> def some_function():
>     # function code here
>     return
> ```

Python profiles can be enabled by adding the `export CARB_PROFILING_PYTHON=1` environment variable before launching the Isaac Sim application. This will enable capture of Python code at the cost of increased overhead.

### C++

To add profiler zones in your C++ code, you can use the `CARB_PROFILE_ZONE` macro to add a zone for a specific scope as follows:

```python
#include <carb/logging/Log.h>
void some_function() {
    // mask is a integer mask that can be used to filter the profiler capture. It's recommended to use 0 (default) or 1 for the mask.
    CARB_PROFILE_ZONE(mask, "zone title");
    {
    // code to profile
    }
}
```

Note

The `--/app/profilerMask` command line argument can be used to filter the profiler capture based on the mask value. Modify this setting to filter for specific zones of interest and avoid capturing unnecessary zones.

## Understanding Tracy Profiler Output

The Tracy profiler outputs a hierarchical view of captured zones, split across threads and fibers for CPU work and on specific GPU contexts for GPU work.

### App Main loop

In Isaac Sim, the top level zone indicates one frame of the simulation, denoted as `App Main loop`. The duration of this zone determines the simulation frame rate.

Note

The App Update zone is one level lower but effectively equivalent to the App Main loop zone.

A broad, high-level view of the hierarchy is shown below:

**App Update**
:   * Pre-Update Events
    * Update Events
      :   + ExecutionController: Definition
            :   - Post-Process Graphs
          + Timeline Update
            :   - Physics Step
                - Transform Updates + Synchronizations
          + Compute Graphs
            :   - onPlaybackTick Node Executions
          + Rendering
            :   - Render Launch For All Render Products
    * Post-Update Events

Note

Generally, the **Pre-Update** and **Post-Update** events contain things like viewport updates, setup/teardown operations, etc.
The **Update** event contains the main simulation logic and is usually the main focus of performance profiling.

One App Main loop zone contains one frame of the simulation. The GPU work is shown above the main thread in an individual zone hierarchy. Multi-GPU systems will display a separate zone hierarchy for each GPU.

Zooming into a single frame shows the breakdown the simulation work. Selection 1 shows the post-process graphs, selection 2 shows the timeline update step, and selection 3 shows the CPU-side rendering work.

#### ExecutionController

This zone generally contains a lot of node executions for post-processing graphs, often part of Replicator logic for processing rendering data. It also contains the main processing logic for the RTX Lidar sensor if being used.

#### Timeline Update Step

The ITimeline::update zone is inclusive of the main simulation work for the frame. This includes the physics step (or multiple steps given a smaller physics step size), transform updates, and the writes to USD/fabric.

The typical structure of this zone is show below:

1. USD Update
2. Physics Step(s)
3. Post-Step Update (Physics-based sensors: IMU, Contact, etc.)
4. Update Render Transforms (USD writes and Fabric synchronization)

#### Compute Graphs

This zone contains the execution of nodes that are dependent on the physics step. Most notably, this includes nodes that use *onPlaybackTick* to execute.

#### Rendering

This zone contains the execution of CPU-side rendering logic. This includes preparing views, updating render product prims, and launching the rendering pipeline on the GPU. The rendering work on the GPU can be visualized in Tracy’s GPU view by setting the `--/profiler/gpu=true` and `--/profiler/gpu/tracyInject/enabled=true` command line arguments documented above.

### Analyzing Bottlenecks

The most common bottlenecks to look for are in the physics computation and rendering execution.

#### Physics Bottlenecks

Physics bottlenecks are typically indicated by a long duration in the **Thread waiting…** zone under the **PhysXUpdateNonRender** zone. Looking through the many threads Tracy displays, you can find the thread that is completing the physics work (whether a GPU callback or CPU-side computation). The physics compute zone returns to the main thread when complete, allowing the **Thread waiting…** to complete.

This can be caused by a variety of factors, including:

1. Physics Backend
2. Physics Objects
3. Physics Step Size

Please refer to [Physics Simulation Optimizations](Isaac_Sim_Performance_Optimization_Handbook.md) for recommendations on optimizing physics performance.

#### Rendering Bottlenecks

Rendering bottlenecks are often characterized by a `waitUntilDone` zone on the main CPU thread. This zone indicates that the next render step is waiting on the previous render step to complete. This presents as a CPU stall, waiting for the GPU to complete the previous render step.

This can be mitigated by increasing the number of GPUs used, assuming a multi-GPU system is available.

Other optimizations to reduce rendering load can be found in [Scene and Rendering Optimizations](Isaac_Sim_Performance_Optimization_Handbook.md). For example, simplifying textures/materials, reducing lighting, or disabling unneeded effects like translucency or reflections.

Example of a GPU-bound case where the main thread is waiting for the GPU to complete the previous render step before beginning the next render step.

## Summary

This tutorial covered the following topics:

1. How to use the Tracy profiler in NVIDIA Isaac Sim in both GUI and standalone workflows.
2. How to add new profiling zones to gauge the performance of your code Python and C++.
3. How to identify common bottlenecks in the simulation execution.