# Isaac Sim Benchmarks

Attention

Benchmark KPIs will be updated with the official Isaac Sim 6.0.0 GA release.

This page contains key performance indicators (KPIs) for Isaac Sim, captured across
different reference hardware and measured using the `isaacsim.benchmark.services` extension. It also
contains a guide on how to collect the same KPIs on your hardware, to compare to our published
performance specs.

## GPU-Independent KPIs

These KPIs measure Isaac Sim performance independent of the GPU on which Isaac Sim is running.

Note

These KPIs were measured on a standardized reference machine using an Intel i9-14900k CPU and 32GB of DDR5 RAM.

GPU-Independent KPIs

| Name | Definition | Units | Value |
| --- | --- | --- | --- |
| Binary package size (Windows) | Size of Windows binary package | GB | 7.37 |
| Binary package size (Linux) | Size of Linux binary package | GB | 8.17 |
| Docker container size | Size of Docker container before extraction on [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:isaac-sim) | GB |  |
| `pip` package size | Size of `pip` package as downloaded | GB |  |
| Startup time (async) | Time from launching Isaac Sim executable to `app ready` appearing in logs | seconds | 31.472 [[1]](#id5)   6.31 [[2]](#id6) |
| Startup time (non-async) | Time from initializing `SimulationApp` in standalone Python to `app ready` appearing in logs | seconds | 263 [[3]](#id7)   4.43 [[4]](#id8) |

[[1](#id1)]

Includes shader installation, which is typically one-time when shaders are cached.

[[2](#id2)]

Startup time (async) using cached shaders.

[[3](#id3)]

Includes shader installation, which is typically one-time when shaders are cached.

[[4](#id4)]

Startup time (non-async) using cached shaders.

## GPU-Dependent KPIs

These KPIs measure Isaac Sim performance on reference hardware, including
frame rate for benchmark scenes and render rate for specific sensor combinations.
KPIs are reported as the average KPI value across 600 frames.

Note

For detailed explanations of each KPI, see [Measuring KPIs on Local Hardware](#isaac-sim-benchmarks-measuring-kpis). Instructions on how to measure the KPIs on local hardware as well as relevant optimization tips for similar workflows are provided.

### Workstation GPUs

GeForce RTX 4080 Super

Note

These KPIs were measured on a standardized reference machine using an Intel Core Ultra 9 285K CPU and 32GB of DDR5 RAM.

Hardware-Dependent KPIs

| Name | Definition | Units | Windows | Ubuntu |
| --- | --- | --- | --- | --- |
| Full Warehouse Sample Scene Load Time | Wall-clock time to load Full Warehouse Sample Scene | Seconds | 40.7 | 49.6 |
| Full Warehouse Sample Scene FPS | Frame rate of Full Warehouse Sample Scene | Frames per second | 229.36 | 210.53 |
| Physics steps per second | Number of physics steps executed per wall-clock second with 10 O3dyn robots | Hz | 52.91 | 59.28 |
| Isaac ROS Sample Scene FPS | Frame rate of Isaac ROS Sample Scene | Frames per second | 41.72 | 42.97 |
| ROS2 render & publishing speed | Frame rate rendered and published via [ROS2 bridge](ROS_2.md) from Nova Carter ROS asset, per wall-clock second | Frames per Second | 14.41 | 12.45 |
| SDG images per second (simple) | Images rendered by SDG per second, with only RGBD annotators enabled, per wall-clock second | Images per second | 6.20 | 8.59 |
| SDG images per second (complex) | Images rendered by SDG per second, with all annotators enabled, per wall-clock second | Images per second | 4.14 | 6.70 |

GeForce RTX 5080

Note

These KPIs were measured on a standardized reference machine using an Intel i9-14900k CPU and 32GB of DDR5 RAM.

Hardware-Dependent KPIs

| Name | Definition | Units | Windows | Ubuntu |
| --- | --- | --- | --- | --- |
| Full Warehouse Sample Scene Load Time | Wall-clock time to load Full Warehouse Sample Scene | Seconds | 35.58 | 34.28 |
| Full Warehouse Sample Scene FPS | Frame rate of Full Warehouse Sample Scene | Frames per second | 259.07 | 241.55 |
| Physics steps per second | Number of physics steps executed per wall-clock second with 10 O3dyn robots | Hz | 42.11 | 44.76 |
| Isaac ROS Sample Scene FPS | Frame rate of Isaac ROS Sample Scene | Frames per second | 32.85 | 49.70 |
| ROS2 render & publishing speed | Frame rate rendered and published via [ROS2 bridge](ROS_2.md) from Nova Carter ROS asset, per wall-clock second | Frames per second | 16.25 | 17.24 |
| SDG images per second (simple) | Images rendered by SDG per second, with only RGBD annotators enabled, per wall-clock second | Images per second | 6.29 | 8.52 |
| SDG images per second (complex) | Images rendered by SDG per second, with all annotators enabled, per wall-clock second | Images per second | 4.93 | 6.76 |

RTX PRO 6000 Blackwell

Note

These KPIs were measured on a standardized reference machine using an Intel i9-14900k CPU and 32GB of DDR5 RAM.

Hardware-Dependent KPIs

| Name | Definition | Units | Windows | Ubuntu |
| --- | --- | --- | --- | --- |
| Full Warehouse Sample Scene Load Time | Wall-clock time to load Full Warehouse Sample Scene | Seconds | 37.82 | 32.84 |
| Full Warehouse Sample Scene FPS | Frame rate of Full Warehouse Sample Scene | Frames per second | 259.07 | 333.33 |
| Physics steps per second | Number of physics steps executed per wall-clock second with 10 O3dyn robots | Hz | 41.20 | 45.54 |
| Isaac ROS Sample Scene FPS | Frame rate of Isaac ROS Sample Scene | Frames per second | 47.87 | 72.28 |
| ROS2 render & publishing speed | Frame rate rendered and published via [ROS2 bridge](ROS_2.md) from Nova Carter ROS asset, per wall-clock second | Frames per second | 17.65 | 21.43 |
| SDG images per second (simple) | Images rendered by SDG per second, with only RGBD annotators enabled, per wall-clock second | Images per second | 6.71 | 8.52 |
| SDG images per second (complex) | Images rendered by SDG per second, with all annotators enabled, per wall-clock second | Images per second | 4.98 | 8.47 |

### Server GPUs

A40

Note

These KPIs were measured on a standardized OVX machine using 2x Intel 8362 CPU and 1024GB of DDR4 RAM, on Ubuntu 24.04.
Some KPIs are measured on multi-GPU configurations, typically for 2, 4, or 8 GPUs.

Hardware-Dependent KPIs by GPU Count

| Name | Definition | Units | x1 | x2 | x4 | x8 |
| --- | --- | --- | --- | --- | --- | --- |
| Full Warehouse Sample Scene Load Time | Wall-clock time to load Full Warehouse Sample Scene | Seconds | 89.4 | 91.23 | 87.61 | 113.25 |
| Full Warehouse Sample Scene FPS | Frame rate of Full Warehouse Sample Scene | Frames per second | 117.40 | 116.82 | 112.36 | 121.82 |
| Physics steps per second | Number of physics steps executed per wall-clock second with 10 O3dyn robots | Hz | 31.3 | 32.7 | 31.6 | 31.9 |
| Isaac ROS Sample Scene FPS | Frame rate of Isaac ROS Sample Scene | Frames per second | 25.33 | 42.14 | 59.95 | 59.92 |
| ROS2 render & publishing speed | Frame rate rendered and published via [ROS2 bridge](ROS_2.md) from Nova Carter ROS asset, per wall-clock second | Frames per second | 7.28 | 14.23 | 23.05 | 26.51 |
| SDG images per second (simple) | Images rendered by SDG per second, with only RGBD annotators enabled, per wall-clock second | Images per second | 3.91 | 3.85 | 3.82 | 3.74 |
| SDG images per second (complex) | Images rendered by SDG per second, with all annotators enabled, per wall-clock second | Images per second | 3.18 | 3.21 | 3.16 | 3.12 |

L40

Note

These KPIs were measured on a standardized OVX machine using 2x Intel 8362 CPU and 1024GB of DDR4 RAM, on Ubuntu 24.04.
Some KPIs are measured on multi-GPU configurations, typically for 2, 4, or 8 GPUs.

Hardware-Dependent KPIs by GPU Count

| Name | Definition | Units | x1 | x2 | x4 | x8 |
| --- | --- | --- | --- | --- | --- | --- |
| Full Warehouse Sample Scene Load Time | Wall-clock time to load Full Warehouse Sample Scene | Seconds | 98.96 | 95.81 | 89.36 | 87.71 |
| Full Warehouse Sample Scene FPS | Frame rate of Full Warehouse Sample Scene | Frames per second | 236.41 | 237.53 | 225.73 | 175.75 |
| Physics steps per second | Number of physics steps executed per wall-clock second with 10 O3dyn robots | Hz | 47.7 | 46.8 | 47.6 | 46.8 |
| Isaac ROS Sample Scene FPS | Frame rate of Isaac ROS Sample Scene | Frames per second | 50.08 | 78.86 | 79.62 | 70.67 |
| ROS2 render & publishing speed | Frame rate rendered and published via [ROS2 bridge](ROS_2.md) from Nova Carter ROS asset, per wall-clock second | Frames per second | 15.21 | 23.84 | 28.71 | 29.35 |
| SDG images per second (simple) | Images rendered by SDG per second, with only RGBD annotators enabled, per wall-clock second | Images per second | 3.94 | 3.83 | 3.79 | 3.77 |
| SDG images per second (complex) | Images rendered by SDG per second, with all annotators enabled, per wall-clock second | Images per second | 3.2 | 3.15 | 3.15 | 3.12 |

RTX PRO 6000 Blackwell Server Edition

Hardware-Dependent KPIs by GPU Count

| Name | Definition | Units | x1 | x2 | x4 | x8 |
| --- | --- | --- | --- | --- | --- | --- |
| Full Warehouse Sample Scene Load Time | Wall-clock time to load Full Warehouse Sample Scene | Seconds | 94.23 | 92.51 | 95.57 | 91.75 |
| Full Warehouse Sample Scene FPS | Frame rate of Full Warehouse Sample Scene | Frames per second | 177.94 | 139.86 | 161.29 | 146.20 |
| Physics steps per second | Number of physics steps executed per wall-clock second with 10 O3dyn robots | Hz | 44.63 | 45.17 | 46.62 | 46.21 |
| Isaac ROS Sample Scene FPS | Frame rate of Isaac ROS Sample Scene | Frames per second | 50.08 | 78.86 | 79.62 | 70.67 |
| ROS2 render & publishing speed | Frame rate rendered and published via [ROS2 bridge](ROS_2.md) from Nova Carter ROS asset, per wall-clock second | Frames per second | 16.77 | 26.49 | 28.94 | 30.08 |
| SDG images per second (simple) | Images rendered by SDG per second, with only RGBD annotators enabled, per wall-clock second | Images per second | 6.41 | 6.36 | 6.36 | 6.36 |
| SDG images per second (complex) | Images rendered by SDG per second, with all annotators enabled, per wall-clock second | Images per second | 5.21 | 5.08 | 4.95 | 4.86 |

## Measuring KPIs on Local Hardware

Isaac Sim KPIs can be measured using the Python scripts provided in `standalone_examples/benchmarks`. Select a category below to see benchmark details, commands, and configuration options as well as optimization tips for similar workflows.

More specific optimization guidance can be found in the [Isaac Sim Performance Optimization Handbook](Isaac_Sim_Performance_Optimization_Handbook.md).

Note

Commands are provided in `bash` syntax (for Ubuntu). For Windows, replace `.sh` with `.bat` and `\` for multiline commands to `` ` ``.

Startup & Loading

Benchmarks for measuring application initialization and scene loading performance.

Startup Time (Async)

**Purpose:** Measure Isaac Sim initialization time in headless mode without blocking operations.

**What it measures:** Time from application launch to ready state, measured as `Runtime` for `phase: startup` in the logs.

**Command:**

```python
./isaac-sim.sh --no-window --/app/quitAfter=200 --/app/file/ignoreUnsavedOnExit=1 \
  --enable isaacsim.benchmark.services
```

**Interpreting Results:** Look for the following in the console output:

```python
[INFO] Runtime for phase: startup = 15234 ms
```

**Typical Values:** 10-30 seconds depending on hardware and system configuration.

Startup Time (Non-Async)

**Purpose:** Measure Isaac Sim initialization time with synchronous loading using the Python API.

**What it measures:** Time for complete application initialization through the Python API.

**Command:**

```python
./python.sh standalone_examples/api/isaacsim.simulation_app/hello_world.py \
  --enable isaacsim.benchmark.services
```

**Interpreting Results:** Look for `Runtime` for `phase: startup` in the logs.

**Comparison:** Non-async startup is typically slower than async due to synchronous loading.

Full Warehouse Load Time + FPS

**Purpose:** Measure scene loading performance and rendering FPS for complex warehouse environment.

**What it measures:** Duration of stage loading phase and FPS at runtime for the given stage.

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_scene_loading.py \
  --env-url /Isaac/Environments/Simple_Warehouse/full_warehouse.usd
```

**Configuration:**

* Environment: full warehouse sample scene

**Interpreting Results:**

```python
[INFO] Runtime for phase: loading = 8123 ms
[INFO] Mean FPS for phase: benchmark = 45.2
```

**Performance Notes:** Loading time depends on asset complexity and storage speed. FPS varies with CPU and GPU capability.

**Optimization Tips:**

1. Use a simpler scene with fewer materials and textures.
2. Disable material loading to reduce initial loading time (`--/app/renderer/skipMaterialLoading=1`).
3. Reduce rendering quality to increase runtime FPS.

Isaac ROS Sample Scene Load Time + FPS

**Purpose:** Measure load time and runtime performance in stages with the ROS2 bridge enabled.

**What it measures:** Duration of stage loading phase and FPS at runtime for the given stage with the ROS2 bridge enabled. The stage uses the Nova Carter robot in a warehouse environment with animated human workers.

**Measurement:** Loading time is measure by `Runtime` for `phase: loading`. Runtime FPS is measured as `Mean FPS` for `phase: benchmark`.

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_scene_loading.py \
  --env-url /Isaac/Samples/ROS2/Scenario/carter_warehouse_apriltags_worker.usd
```

**Interpreting Results:**

```python
[INFO] Runtime for phase: loading = 8556 ms
[INFO] Mean FPS for phase: benchmark = 38.7
```

**Optimization Tips:**

1. Disable material loading to reduce initial loading time (`--/app/renderer/skipMaterialLoading=1`).
2. Reduce rendering quality to increase runtime FPS.
3. Use a simpler scene with fewer materials, textures, and lighting. This will simplify the rendering work done by each render product.

**Multi-GPU:** Loading time is not impacted by the number of GPUs. Runtime FPS for this benchmark scales with GPU count - optimal GPU count is hardware dependent but typically 4 or 8 GPUs.

Workflow Performance

Benchmarks for measuring physics computation, rendering speed, and overall simulation performance.

Physics Steps per Second

**Purpose:** Measure physics simulation performance and compare CPU vs GPU physics backends for a complex robot.

**What it measures:** How many physics steps are executed per wall-clock second given a fixed step size, robot count, and Physics backend for the O3dyn robot in the full warehouse sample scene.

**Measurement:** Measured as `Mean FPS` for `phase: benchmark` given a physics dt of 1/60s.

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_robots_o3dyn.py \
  --num-robots 10 --num-gpus 1
```

**Configurations:**

* Robot Count
* Physics Backend (CPU: numpy, GPU: torch, warp)

```python
# CPU Physics
./python.sh standalone_examples/benchmarks/benchmark_robots_o3dyn.py \
  --num-robots 2 --physics numpy

# GPU Physics (default: torch)
./python.sh standalone_examples/benchmarks/benchmark_robots_o3dyn.py \
  --num-robots 10 --physics warp
```

**Interpreting Results:**

```python
Mean FPS: 51.706 FPS
```

Given a physics dt of `1/60`, the physics steps per second is equivalent to the FPS. A smaller physics dt will result in multiple physics steps per frame, changing the computation to be `FPS * physics steps per frame`.

**Performance Notes:** The O3dyn robot is very complex, particularly due to the simulation of the highly articulated wheels. Simpler robots will achieve faster framerates due to reduced physics computation work. Higher-spec GPUs will enable higher throughput as robot count or physics object count increases.

**Optimization Tips:**

1. Select the appropriate physics backend for the workload. It’s recommended to test with both backends to determine the optimal choice.

> * CPU Physics: Low robot count and/or low complexity robots + scenes
> * GPU Physics: Higher robot counts and/or higher complexity robots + scenes

2. Reduce the complexity of the robot by disabling unnecessary colliders, joints, and other components. Similarly decrease the complexity of the scene.

**Performance Scaling:** The O3dyn robot is a good example to see how CPU and GPU physics performance scales with the number of robots and the complexity of the robots.

* 1-4 robots: CPU physics is faster
* ~5 robots: CPU and GPU physics are comparable (hardware-dependent)
* 6+ robots: GPU physics is faster

**Multi-GPU:** GPU physics performance does not scale with GPU count as PhysX runs on a single GPU.

Rendering Speed

**Purpose:** Measure pure rendering performance with no additional physics computation.

**What it measures:** The framerate of the simulation when rendering the full warehouse sample scene with a variable number of cameras.

**Measurement:** Measured as `Mean FPS` for `phase: benchmark`

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_camera.py \
  --num-cameras 2 --resolution 1280 720 --num-gpus 1
```

**Configurations:**

* Camera count
* Camera resolution (default: 1280x720)
* GPU count (default: all available GPUs)

**Interpreting Results:**

```python
Mean FPS: 45.36 FPS
```

**Performance Notes:** Faster GPUs will achieve better performance as camera count and/or resolution increases. GPUs with lower VRAM may struggle to render multiple high resolution cameras or high counts of lower resolution cameras.

**Optimization Tips:**

1. Use minimum number of cameras and resolution to reduce rendering work.
2. Use as many GPUs as cameras to maximize throughput. Very high resolution cameras will also benefit from multiple GPUs due to tiling.
3. If visual quality is not critical, modify render settings to reduce realism of rendered images.
4. Use a simpler scene with fewer materials, textures, and lighting. This will simplify the rendering work done by each render product.

**Multi-GPU:** Camera rendering performance most effectively scales with the number of GPUs. The more GPUs, the more cameras can be rendered in parallel, improving throughput.

ROS 2 Render & Publishing Speed (Rendering + Physics + ROS2 Workflow)

**Purpose:** Measure full SIL workflow performance - combining rendering, physics, ROS2 message publishing, and robot control.

**What it measures:** Simulation framerate when publishing via ROS2 bridge using Nova Carter ROS asset, per wall-clock second. A total of 11 sensors are enabled: 3 lidars + 4 stereo camera pairs

**Measurement:** Overall speed is measured as `Mean FPS` for `phase: benchmark`.

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_robots_nova_carter_ros2.py \
  --num-robots 1 --enable-3d-lidar 1 --enable-2d-lidar 2 --enable-hawks 4
```

**Configuration:**

* 1x Nova Carter Robot

  + 1x 3D LiDAR sensor
  + 2x 2D LiDAR sensors
  + 4x Hawk stereo cameras (8x render products at 1920x1200p each)

**Interpreting Results:**

```python
[INFO] Mean FPS for phase: benchmark = 25.3
```

**Performance Notes:** This benchmarks uses a heavy sensor suite by default, reducing the number or resolution of sensors will improve performance. Lower VRAM GPUs (under 12GB) may not be able to render all sensors. Performance with fast CPUs will be limited by rendering speed, performance benefits will be observed with higher-spec GPUs or multi-GPU configurations.

**Optimization Tips:**

1. Reduce the camera count (`--enable-hawks 2`). This command runs 8 render products at 1920x1200p each. Reducing the camera count will reduce the number of render products and improve performance.
2. If visual quality is not critical, modify render settings to reduce accuracy of rendered images.
3. Use a simpler scene with fewer materials, textures, and lighting. This will simplify the rendering work done by each render product.

**Multi-GPU:** Performance scales with the sensor count. The more sensors, the more GPUs will help improve throughput. For server-grade hardware, simulating 4 Nova Carters with full sensor suites is feasible with 4x or 8x GPUs.

Synthetic Data Generation

Benchmarks for measuring synthetic data generation performance and throughput.

SDG Images per Second (Simple)

**Purpose:** Measure synthetic data generation performance with basic annotations

**What it measures:** Image generation rate with RGB and depth annotations for 500 prims, randomizing pose/orientation/scale/color per frame.

**Measurement:** Overall speed is measured as `Mean FPS` for `phase: benchmark`. Images generated per second is measured as `Mean FPS * number of cameras`.

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_sdg.py \
  --num-cameras 2 --resolution 1280 720 --asset-count 100 \
  --annotators rgb distance_to_image_plane --skip-write
```

**Configuration:**

* 2 cameras at 1280x720 resolution
* 100 count per asset type (5 types for total of 500 prims)
* RGB + depth annotations only
* Skip disk write for pure generation speed

**Interpreting Results:**

```python
[INFO] Mean FPS for phase: benchmark = 15.8
```

The throughput can be calculated as `Mean FPS * number of cameras` to yield the total number of images generated per second.

**Performance Notes:** The usage of the –skip-write flag improves performance by skipping the disk write step which can cause a bottleneck due to IO operations. Randomization of pose/orientation/material are CPU-intensive operations currently.

**Optimization Tips:**

1. If saving to disk, see the I/O Optimization Guide in the Replicator documentation to optimize throughput.
2. Decrease total number of assets in the scene.
3. Minimize randomization operations which are CPU-intensive.

**Multi-GPU:** Performance scales most effectively based on camera count and resolution. The more cameras, or higher the resolution, in the scene, the more GPUs will help improve throughput. This default benchmark with 2 720p cameras does not scale well with more GPUs as it’s limited by randomization operations.

SDG Images per Second (Complex)

**Purpose:** Measure synthetic data generation performance with full suite of annotators enabled.

**What it measures:** Image generation rate with all annotators enabled for 500 prims, randomizing pose/orientation/scale/color per frame.

**Measurement:** Overall speed is measured as `Mean FPS` for `phase: benchmark`. Images generated per second is measured as `Mean FPS * number of cameras`.

**Command:**

```python
./python.sh standalone_examples/benchmarks/benchmark_sdg.py \
  --num-cameras 2 --resolution 1280 720 --asset-count 100 \
  --annotators all --skip-write
```

**Configuration:**

* 2 cameras at 1280x720 resolution
* 100 count per asset type (5 types for total of 500 prims)
* All available annotators enabled
* Skip disk write for pure generation speed

**Annotators Available:**

* RGB
* Distance to Image Plane
* Distance to Camera
* Bounding Box 2D Tight
* Bounding Box 2D Loose
* Bounding Box 3D
* Semantic Segmentation
* Instance Segmentation
* Occlusion
* Normals
* Motion vectors
* Camera Parameters
* Point Cloud
* Skeleton Data

**Interpreting Results:**

```python
[INFO] Mean FPS for phase: benchmark = 4.2
```

The throughput can be calculated as `Mean FPS * number of cameras` to yield the total number of images generated per second.

**Performance Notes:** The usage of the –skip-write flag improves performance by skipping the disk write step which can cause a bottleneck due to IO operations. Randomization of pose/orientation/material are CPU-intensive operations, limiting GPU scaling.

**Optimization Tips:**

1. Disable unneeded annotators to improve performance for specific use cases.
2. If saving to disk, see the I/O Optimization Guide in the Replicator documentation to optimize throughput.
3. Decrease total number of assets in the scene.
4. Minimize randomization operations which are CPU-intensive.

**Multi-GPU:** Performance scales most effectively based on camera count and resolution. The more cameras, or higher the resolution, in the scene, the more GPUs will help improve throughput. This default benchmark with 2 720p cameras does not scale with more GPUs as it’s limited by randomization operations rather than rendering.

## Understanding Benchmark Outputs

This section walks through the outputs of the benchmark script to explain the different metrics and how to interpret them.

The benchmark script outputs a summary report and a raw metric file. The summary report is a concise summary of the benchmark results. The metrics file contains the raw metrics that are parsed into the summary report. The log indicates where the metrics file is stored.

### Summary Report

The summary report is output to the console for every benchmark script. It provides a concise summary of the benchmark results.

**Example Output:**

```python
|----------------------------------------------------|
|                   Summary Report                   |
|----------------------------------------------------|
| workflow_name: benchmark_robots_nova_carter_ros2   |
| num_robots: 2                                      |
| num_gpus: 1                                        |
| num_3d_lidar: 1                                    |
| num_2d_lidar: 2                                    |
| num_hawks: 4                                       |
| num_cpus: 32                                       |
| gpu_device_name: NVIDIA GeForce RTX 4090           |
|----------------------------------------------------|
| Phase: loading                                     |
| System Memory RSS: 17.021 GB                       |
| System Memory VMS: 145.177 GB                      |
| System Memory USS: 16.997 GB                       |
| GPU Memory Tracked: 1.124 GB                       |
| Runtime: 5549.776 ms                               |
|----------------------------------------------------|
| Phase: benchmark                                   |
| System Memory RSS: 17.021 GB                       |
| System Memory VMS: 145.177 GB                      |
| System Memory USS: 16.997 GB                       |
| GPU Memory Tracked: 1.124 GB                       |
| Mean FPS: 51.706 FPS                               |
| Real Time Factor: 0.849                            |
| Runtime: 11772.105 ms                              |
| Frametimes (ms):    mean |  stdev |   min |   max  |
| App_Update         19.34 |   0.39 | 18.92 | 20.42  |
| Physics            17.61 |   0.08 | 17.52 | 17.99  |
|----------------------------------------------------|
```

#### Configuration Section

The first section shows the benchmark configuration and system information.

```python
|----------------------------------------------------|
| workflow_name: benchmark_robots_nova_carter_ros2   |
| num_robots: 2                                      |
| num_gpus: 1                                        |
| num_3d_lidar: 1                                    |
| num_2d_lidar: 2                                    |
| num_hawks: 4                                       |
| num_cpus: 32                                       |
| gpu_device_name: NVIDIA GeForce RTX 4090           |
|----------------------------------------------------|
```

It’s populated with the `workflow_metadata` dictionary passed into the `BaseIsaacBenchmark` object defined in each benchmark script.

#### Loading Phase Metrics

The loading phase measures resource usage during scene loading and other setup steps:

* **System Memory RSS:** Resident Set Size of the process in GB
* **System Memory VMS:** Virtual Memory Size of the process in GB
* **System Memory USS:** Unique Set Size of the process in GB
* **GPU Memory Tracked:** VRAM utilized by the GPU in GB
* **Runtime:** Wall-clock time in milliseconds

#### Benchmark Phase Metrics

The benchmark phase measures performance during active simulation:

**Performance Metrics:**

* **Mean FPS:** Computed as `1000/mean_app_update_frametime` where `mean_app_update_frametime` is the average frametime of the app update phase in milliseconds.
* **Real Time Factor:** A ratio of how close simulation time is to wall-clock time. Computed as `simulation_time / wall_clock_time` where `simulation_time` is the total time simulated and `wall_clock_time` is the real-world time elapsed.
* **Runtime:** The wall-clock duration in milliseconds of the benchmark phase.

**Frametime Breakdown:**

The frametimes section shows detailed timing for different simulation components:

* **App\_Update:** One app update represents one frame of the simulation. In default configurations, this typically involves one physics step and one render step.
* **Physics:** The duration of the physics step. This is a component of the total app\_update frametime, representing the duration of physics computation work.
* **GPU:** The duration of GPU work. This is a component of the total app\_update frametime, representing the duration of rendering work. This is only collected when the `--gpu-frametime` flag is enabled.

For further insight into how the frametime breaks down for a specific workflow, refer to [Profiling Performance Using Tracy](Debugging_Profiling.md) for details on using the Tracy profiler to profile the simulation.

Note

One app update is characterized by some amount of physics compute and some amount of rendering work for the given frame. The sum of these two components are not expected to equal the app\_update frametime due to parallelization, other overhead, and any dedicated per frame compute.

### Interpreting Results

This section details how to interpret some of the key results explained in the previous sections, specifically as they relate to hardware selection.

**Mean FPS:**

The Mean FPS is the key metric to consider when selecting hardware. It is the average frame rate of the simulation over the course of the benchmark. It is a good indicator of the overall performance of the hardware for a given workflow.

**GPU Memory Tracked:**

The GPU Memory Tracked metric indicates the amount of VRAM needed by the workflow. Workflows that involve large scenes, high resolution cameras, or large amounts of sensors will require more VRAM.

**Physics Frametime:**

A Physics Frametime very close to the App Update frametime indicates that the physics computation may be bottlenecking the performance. With GPU Physics, higher-spec GPUs will scale better with more physics objects and/or higher complexity robots.

**GPU Frametime:**

With a GPU frametime very close to the App Update frametime, it indicates that the GPU rendering might be bottlenecking the performance. Adding additional GPUs or using a higher-spec GPU will help improve performance. Otherwise, if the GPU frametime is much lower than the App\_Update frametime, it indicates that CPU performance might be the bottleneck.

## Benchmark Methodology Changes

This section tracks changes to benchmark methodologies, measurement scripts, and hardware configurations across Isaac Sim versions to enable accurate version-to-version comparisons.

Note

When comparing benchmark results between versions, ensure you account for any methodology or hardware changes listed below.

### Version 6.0.0

**Measurement Changes:**

* Updated reference hardware CPU for workstation hardware from Intel i9-14900k to Intel Core Ultra 9 285K for workstation GPU KPIs

**Script Changes:**

* No changes to benchmark scripts in this version

### Version 5.1.0

**Measurement Changes:**

* Motion BVH disabled by default (previously enabled) - decreases rendering accuracy for motion-related sensor effects but improves rendering performance

**Script Changes:**

* Disabled default collection of GPU frametime due to slight performance impact on overall benchmark performance. Can be enabled with `--gpu-frametime` flag.

### Version 5.0.0

**Measurement Changes:**

* KPIs measured with Motion BVH (enabled by default in Isaac Sim 5.0.0) - increases rendering accuracy for motion-related sensor effects but decreases overall rendering performance

**Script Changes:**

* Disabled viewport updates by default in headless mode to improve performance (can be enabled with `--viewport-updates`)
* Physics Steps per Second (`benchmark_robots_o3dyn.py`): Added support for both CPU and GPU physics backends (previously CPU only).

  + Backend default changed from CPU to GPU (torch) physics backend
  + Robot count default changed from 2 to 10

### Version 4.5.0

**Measurement Changes:**

* Initial baseline measurements

**Script Changes:**

* Benchmark scripts introduced in `standalone_examples/benchmarks/`