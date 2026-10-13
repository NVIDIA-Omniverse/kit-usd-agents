# Sensors

Isaac Sim sensor simulation extensions can simulate ground truth perception and physics based sensors and provide a library of realistic sensors models.

- Camera Sensors
- GUI
- Standalone Python
- Calibration and Camera Lens Distortion Models
- Creating Camera Sensor Rigs
- Exposing the Pre-ISP Camera Pipeline
- Camera Inspector Extension
- Depth Sensors
- Stereoscopic Depth Cameras
- RTX Sensors
- Getting Started
- Sensor Types
- Data Collection and Materials
- Advanced Topics
- Extension Architecture
- Important Settings
- Motion BVH
- Troubleshooting and Known Issues
- Related Tutorials
- Physics-Based Sensors
- Articulation Joint Sensors
- Contact Sensor
- Effort Sensor
- IMU Sensor
- Proximity Sensor
- PhysX SDK Sensors
- PhysX SDK Generic Sensor
- PhysX SDK Lidar
- PhysX SDK Lightbeam Sensor
- Camera and Depth Sensors
- Cameras
- Depth Sensors
- Non-Visual Sensors
- RTX Lidars
- Tactile Sensors
- Sensor Gizmo in Viewport

---

# Camera Sensors

Cameras are modeled using the Camera USD prim type. Camera data is acquired from camera prims using render products, which can be created by multiple different extensions in Omniverse,
including the [omni.replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator.html "(in Omniverse Extensions)") extension.

Note

Isaac Sim camera functionality is based on [Omniverse cameras](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html).

## GUI

### Creating and Modifying a Camera

1. Create a cube by selecting **Create > Shape > Cube** and change its location and scale through the property panel as indicated in the screenshot below.

   > 
2. Create a camera prim by selecting **Create > Camera** and then select it from the stage window to view its field of view as indicated below.

   > 
3. To render the frames from the camera, switch the default viewport (which is a render product itself) to the camera prim that you just created.
   Select the video icon at the top of the viewport window and then select the camera prim you just created under the `Cameras` menu.

   > 

## Standalone Python

There are multiple ways to retrieve data from a render product attached to a camera prim in Isaac Sim. One method is the `Camera` class
under the `isaacsim.sensors.camera` extension. You can run an example using the `Camera` class using `./python.sh standalone_examples/api/isaacsim.sensors.camera/camera.py`.
The code in that example is provided below, for reference.

```python
 1from isaacsim import SimulationApp
 2
 3simulation_app = SimulationApp({"headless": False})
 4
 5import isaacsim.core.utils.numpy.rotations as rot_utils
 6import matplotlib.pyplot as plt
 7import numpy as np
 8from isaacsim.core.api import World
 9from isaacsim.core.api.objects import DynamicCuboid
10from isaacsim.sensors.camera import Camera
11
12my_world = World(stage_units_in_meters=1.0)
13
14# Add two cubes to the scene
15cube_1 = my_world.scene.add(
16    DynamicCuboid(
17        prim_path="/new_cube_1",
18        name="cube_1",
19        position=np.array([5.0, 3, 1.0]),
20        scale=np.array([0.6, 0.5, 0.2]),
21        size=1.0,
22        color=np.array([255, 0, 0]),
23    )
24)
25
26cube_2 = my_world.scene.add(
27    DynamicCuboid(
28        prim_path="/new_cube_2",
29        name="cube_2",
30        position=np.array([-5, 1, 3.0]),
31        scale=np.array([0.1, 0.1, 0.1]),
32        size=1.0,
33        color=np.array([0, 0, 255]),
34        linear_velocity=np.array([0, 0, 0.4]),
35    )
36)
37
38# Add a camera to the scene, facing the cubes
39camera = Camera(
40    prim_path="/World/camera",
41    position=np.array([0.0, 0.0, 25.0]),
42    frequency=20,
43    resolution=(256, 256),
44    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
45)
46
47# Add a ground plane to the scene
48my_world.scene.add_default_ground_plane()
49
50# Reset the world and initialize the camera
51my_world.reset()
52camera.initialize()
53
54i = 0
55# Collect motion vectors for each object in view of the camera
56camera.add_motion_vectors_to_frame()
57
58# Run indefinitely, until the simulation is stopped (eg. via Ctrl+C)
59for _ in range(101):
60    my_world.step(render=True)
61    if i == 100:
62        # Find the 2D coordinates of the cubes in the image
63        points_2d = camera.get_image_coords_from_world_points(
64            np.array([cube_1.get_world_pose()[0], cube_2.get_world_pose()[0]])
65        )
66        # Project the 2D coordinates of the cubes in the image back to 3D world coordinates,
67        # taking depth as z-position of the camera
68        points_3d = camera.get_world_points_from_image_coords(points_2d, np.array([24.94, 24.9]))
69        # Print both sets, demonstrating reprojection errors when comparing points_3D to the
70        print(points_2d)
71        print(points_3d)
72        # Plot the RGB image
73        plt.imsave("camera.png", camera.get_rgba()[:, :, :3])
74        # Print the motion vectors collected by the camera
75        print(camera.get_current_frame()["motion_vectors"])
76    if my_world.is_playing():
77        if my_world.current_time_step_index == 0:
78            my_world.reset()
79    i += 1
80
81
82simulation_app.close()
```

## Calibration and Camera Lens Distortion Models

Omniverse cameras support a variety of lens distortion models, described [here](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#omniverse-cameras).
The `isaacsim.sensors.camera.Camera` class includes APIs to set lens distortion parameters for each Omniverse camera lens distortion model.

Calibration toolkits like OpenCV normally provide the calibration parameters as an intrinsic matrix and distortion coefficients. Omniverse includes native renderer support for the OpenCV pinhole and
OpenCV fisheye lens distortion models. Isaac Sim provides two standalone examples demonstrating the use of the `Camera` class with OpenCV lens distortion models,
located at `standalone_examples/api/isaacsim.sensors.camera/camera_opencv_pinhole.py` and `standalone_examples/api/isaacsim.sensors.camera/camera_opencv_fisheye.py`.

Portions of these examples are repeated below for reference, and can be run using using **Script Editor**, opened from **Window > Script Editor**.

Note

* Previously, the `Camera` class included APIs to approximate OpenCV pinhole and fisheye models distortion parameters by setting coefficients for the `fisheyePolynomial` distortion model. Now that OpenCV lens distortion models are natively supported, those APIs have been deprecated.
* [Omniverse RTX Camera Projection Attributes](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#rtx-camera-projection-attributes-deprecated) have been deprecated as of Isaac Sim 5.0, in favor of the `OmniLensDistortion` [schemata](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#omnilensdistortion-schemata). The deprecated attributes are still visible in the UI in the `Fisheye Lens` panel when selecting a Camera prim, but will be ignored if the you have set an `OmniLensDistortion` schema instead. Follow the instructions in [“How To Add Schemata to Cameras”](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#omnilensdistortion-schemata) to see how to update Camera prim attributes for the new schemata in the UI.

### OpenCV Fisheye

```python
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from PIL import Image, ImageDraw

# Desired image resolution, camera intrinsics matrix, and distortion coefficients
# These values were selected to estimate distortion for the Realsense D455 camera, and
# will vary for each individual camera.
width, height = 1920, 1200
camera_matrix = [[455.8, 0.0, 943.8], [0.0, 454.7, 602.3], [0.0, 0.0, 1.0]]
distortion_coefficients = [0.05, 0.01, -0.003, -0.0005]

# Camera sensor size and optical path parameters. These parameters are not the part of the
# OpenCV camera model, but they are nessesary to simulate the depth of field effect.
#
# Note: To disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
# Set pixel size (microns)
pixel_size = 3
# Set f-number, the ratio of the lens focal length to the diameter of the entrance pupil (unitless)
f_stop = 1.8
# Set focus distance (meters) - chosen as distance from camera to cube
focus_distance = 1.5

# Add a ground plane to the scene
usd_path = get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/ground_plane")

# Add some cubes and a Camera to the scene
cube_1 = DynamicCuboid(
    prim_path="/new_cube_1",
    name="cube_1",
    position=np.array([0, 0, 0.5]),
    scale=np.array([1.0, 1.0, 1.0]),
    size=1.0,
    color=np.array([255, 0, 0]),
)

cube_2 = DynamicCuboid(
    prim_path="/new_cube_2",
    name="cube_2",
    position=np.array([2, 0, 0.5]),
    scale=np.array([1.0, 1.0, 1.0]),
    size=1.0,
    color=np.array([0, 255, 0]),
)

cube_3 = DynamicCuboid(
    prim_path="/new_cube_3",
    name="cube_3",
    position=np.array([0, 4, 1]),
    scale=np.array([2.0, 2.0, 2.0]),
    size=1.0,
    color=np.array([0, 0, 255]),
)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 2.0]),  # 1 meter away from the side of the cube
    frequency=30,
    resolution=(width, height),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)
camera.initialize()

# Calculate the focal length and aperture size from the camera matrix
((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix  # fx, fy are in pixels, cx, cy are in pixels
horizontal_aperture = pixel_size * width * 1e-6  # convert to meters
vertical_aperture = pixel_size * height * 1e-6  # convert to meters
focal_length_x = pixel_size * fx * 1e-6  # convert to meters
focal_length_y = pixel_size * fy * 1e-6  # convert to meters
focal_length = (focal_length_x + focal_length_y) / 2  # convert to meters

# Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
camera.set_focal_length(focal_length)
camera.set_focus_distance(focus_distance)
camera.set_lens_aperture(f_stop)
camera.set_horizontal_aperture(horizontal_aperture)
camera.set_vertical_aperture(vertical_aperture)

camera.set_clipping_range(0.05, 1.0e5)

# Set the distortion coefficients
camera.set_opencv_fisheye_properties(cx=cx, cy=cy, fx=fx, fy=fy, fisheye=distortion_coefficients)
```

After running the snippet above and setting the viewport to the newly-created camera, validate that you see an image like the one below.

### OpenCV Pinhole

```python
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from PIL import Image, ImageDraw

# Desired image resolution, camera intrinsics matrix, and distortion coefficients
# These values were selected to estimate distortion for the Realsense D455 camera, and
# will vary for each individual camera.
width, height = 1920, 1200
camera_matrix = [[958.8, 0.0, 957.8], [0.0, 956.7, 589.5], [0.0, 0.0, 1.0]]
distortion_coefficients = [0.14, -0.03, -0.0002, -0.00003, 0.009, 0.5, -0.07, 0.017]

# Camera sensor size and optical path parameters. These parameters are not the part of the
# OpenCV camera model, but they are nessesary to simulate the depth of field effect.
#
# Note: To disable the depth of field effect, set the f_stop to 0.0. This is useful for debugging.
# Set pixel size (microns)
pixel_size = 3
# Set f-number, the ratio of the lens focal length to the diameter of the entrance pupil (unitless)
f_stop = 1.8
# Set focus distance (meters) - chosen as distance from camera to cube
focus_distance = 1.5

# Add a ground plane to the scene
usd_path = get_assets_root_path() + "/Isaac/Environments/Grid/default_environment.usd"
add_reference_to_stage(usd_path=usd_path, prim_path="/ground_plane")

# Add some cubes and a Camera to the scene
cube_1 = DynamicCuboid(
    prim_path="/new_cube_1",
    name="cube_1",
    position=np.array([0, 0, 0.5]),
    scale=np.array([1.0, 1.0, 1.0]),
    size=1.0,
    color=np.array([255, 0, 0]),
)

cube_2 = DynamicCuboid(
    prim_path="/new_cube_2",
    name="cube_2",
    position=np.array([2, 0, 0.5]),
    scale=np.array([1.0, 1.0, 1.0]),
    size=1.0,
    color=np.array([0, 255, 0]),
)

cube_3 = DynamicCuboid(
    prim_path="/new_cube_3",
    name="cube_3",
    position=np.array([0, 4, 1]),
    scale=np.array([2.0, 2.0, 2.0]),
    size=1.0,
    color=np.array([0, 0, 255]),
)

camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.0, 0.0, 2.0]),  # 1 meter away from the side of the cube
    frequency=30,
    resolution=(width, height),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
)
camera.initialize()

# Calculate the focal length and aperture size from the camera matrix
((fx, _, cx), (_, fy, cy), (_, _, _)) = camera_matrix  # fx, fy are in pixels, cx, cy are in pixels
horizontal_aperture = pixel_size * width * 1e-6  # convert to meters
vertical_aperture = pixel_size * height * 1e-6  # convert to meters
focal_length_x = pixel_size * fx * 1e-6  # convert to meters
focal_length_y = pixel_size * fy * 1e-6  # convert to meters
focal_length = (focal_length_x + focal_length_y) / 2  # convert to meters

# Set the camera parameters, note the unit conversion between Isaac Sim sensor and Kit
camera.set_focal_length(focal_length)
camera.set_focus_distance(focus_distance)
camera.set_lens_aperture(f_stop)
camera.set_horizontal_aperture(horizontal_aperture)
camera.set_vertical_aperture(vertical_aperture)

camera.set_clipping_range(0.05, 1.0e5)

# Set the distortion coefficients
camera.set_opencv_pinhole_properties(cx=cx, cy=cy, fx=fx, fy=fy, pinhole=distortion_coefficients)
```

After running the snippet above and setting the viewport to the newly-created camera, you should see an image like the one below.

### Extrinsic Calibration

Extrinsic calibration parameters are normally provided by the calibration toolkits in a form of a transformation matrix. The convention between the axis and rotation order is important and it varies between the toolkits.

To set the extrinsic parameters for the individual camera sensor, use the following example to convert the transformation matrix from the calibration toolkit to the Isaac Sim units:

```python
import isaacsim.core.utils.numpy.rotations as rot_utils  # convenience functions for quaternion operations
import numpy as np

dX, dY, dZ = _, _, _  # Extrinsics translation vector from the calibration toolkit
rW, rX, rY, rZ = _, _, _, _  # Note the order of the rotation parameters, it depends on the toolkit

Camera(
    prim_path="/rig/camera_color",
    position=np.array([-dZ, dX, dY]),  # Note, translation in the local frame of the prim
    orientation=np.array([rW, -rZ, rX, rY]),  # quaternion orientation in the world/ local frame of the prim
    # (depends if translation or position is specified)
)
```

As an alternative, the camera sensor can be attached to a prim. In that case, the camera sensor will inherit the position and orientation from the prim.

```python
import isaacsim.core.utils.prims as prim_utils
from isaacsim.sensors.camera import Camera

camera_prim = prim_utils.create_prim(
    prim_path="/World/camera",
    prim_type="Camera",
    # translation = ...
    # orientation = ...
)

camera = Camera(
    prim_path="/World/camera",
)
```

## Creating Camera Sensor Rigs

The camera sensor rig is a collection of camera sensors that are attached to a single prim. It can be assembled from the individual sensors, that are either created manually or derived from the calibration parameters.

This will be a short discussion on how we created a digital twin of the Intel® RealSense™ Depth Camera D455. The USD for the camera can be found in the content folder as: `` `/Isaac/Sensors/Intel/RealSense/rsd455.usd ``.

There are three visual sensors, and one IMU sensor on the RealSense. Their placement relative to the camera origin was taken from the layout diagram in
the [TechSpec document](https://www.intelrealsense.com/wp-content/uploads/2023/07/Intel-RealSense-D400-Series-Datasheet-July-2023.pdf) from [Intel’s web site](https://www.intelrealsense.com/depth-camera-d455/).

Most camera parameters were also found in the TechSpec, for example, the USD parameter `fStop` is the denominator of the F Number from the TechSpec; the `focalLength` is the Focal Length, and the `ftheatMaxFov`
is the Diagonal Field of View. However, some parameters, like the `focusDistance` were estimated by comparing real output and informed guesses.

The `horizontalAperture` and `verticalAperture` in that example are derived from the technical specification. From the TechSpec, the left, right, and color sensors are listed as a OmniVision Technologies OV9782, and
the [Tech Spec](https://www.ovt.com/products/ov09782-ga4a/) for that sensor lists the image area as 3896 x 2453 µm. We used that as the aperture sizes.

The resolution for the depth and color cameras are 1280 x 800, but it’s up to you to attach a render product of that size to the outputs.

The `Pseudo Depth` camera is a stand in for the depth image created by the camera’s firmware. We don’t attempt to copy the algorithms that create the image from stereo, but the `Camera_Pseudo_Depth` component
is a convenience camera that can return the scene depth as seen from that camera position between the left and right stereo cameras. It would be more accurate to create a depth image from stereo, and if
the same algorithm that is used in the RealSense was used then the same results (including artifacts) would be produced.

## Exposing the Pre-ISP Camera Pipeline

The `omni.sensors.nv.camera` extension [simulates the camera sensor and image signal processor (ISP) pipeline](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#omni-sensors-nv-camera-extension).

Isaac Sim 5.1 now includes a standalone example demonstrating how to render and save output from each step of the pre-ISP camera pipeline, including color correction, CFA encoding, and companding, for users who
would like to test their own ISP using images rendered in Omniverse, or compare the output of their ISP with the output of the Omniverse simulatedISP.

Refer to the [extension documentation](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html#omni-sensors-nv-camera-extension) for more details on the camera pipeline.

Run the example via:

```python
# ./python.sh standalone_examples/api/isaacsim.sensors.camera/camera_pre_isp_pipeline.py --draw-output
```

The example will render and save output from three pre-ISP steps, by default in the `pre_isp_camera_pipeline_outputs` directory. The HDR buffer, raw sensor output, and ISP output from the example are shown below:

## Camera Inspector Extension

The Camera Inspector Extension allows you to:

* Create multiple viewports for each camera
* Check camera coverage
* Get and set camera poses in the desired frames

### Launching Extension

To open the Camera Inspector extension:

1. Go to the Menu Bar.
2. Select **Tools > Sensors > Camera Inspector**.
3. After launching the extension, verify that you can see your camera in the dropdown.
4. When adding a new camera, you must click the Refresh button to ensure that the extension finds this new camera.
5. Select the camera you want to inspect.

### Camera State Textbox

The **Camera State** textbox near the top of the extension provides a convenient way to copy the position and orientation of your camera directly into code.
Click the copy icon on the right of the textbox to copy to your clipboard.

### Creating a Viewport

With the camera selected, you can create a new viewport for your camera.

1. Click on the **Create Viewport** button to the right of the camera dropdown menu.

   > By default, this creates a new viewport and assigns the current selected Camera to it.
2. Assign different cameras to different viewports using the two dropdown menus and buttons in the extension:

   > 
3. After launching your viewport, you can change the resolution using the menu in the top left and going to **Viewport**.

   > Note
   >
   > When changing the resolution, Omniverse Kit only supports square pixels. This means that the resolution aspect ratio must be the same as the aperture ratio.
   >
   > 

---

# Depth Sensors

## Stereoscopic Depth Cameras

### Single-View Post-Processing Pipeline

Isaac Sim models stereoscopic depth cameras using a single camera view through the `isaacsim.sensors.camera.SingleViewDepthSensor` class. This class wraps around `isaacsim.sensors.camera.Camera`, and
includes APIs for configuring a post-processing pipeline for stereoscopic depth estimation from a single Camera prim. The process by which the renderer models disparity and noise from a single camera view
is described in detail [here](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.camera/0.21.0-coreapi/camera_extension.html#single-view-depth-camera).

#### Standalone Python

Check out the standalone example located at `standalone_examples/api/isaacsim.sensors.camera/camera_stereoscopic_depth.py` for an example of how to use the `isaacsim.sensors.camera.SingleViewDepthSensor` class
and the [new Annotators provided in Replicator](https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html).

When running the standalone example, a basic set of colored shapes in the Black Grid environment are in the viewport, like below:

Now, examine the disparity map generated by the depth sensor as follows:

1. Select the camera render product in the viewport.
2. Click **Render Settings > Post Processing > Depth Sensor** to examine the depth sensor post-processing pipeline settings.
3. Tick the checkbox for **Depth Sensor**.
4. Select **Disparity** from the **RGB Depth Output Mode** dropdown.

The settings will look like the following:

Note

To learn more about these Post Processing settings, visit [Single View Depth Camera documentation](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.camera/0.21.0-coreapi/camera_extension.html#single-view-depth-camera).

Verify that you see the disparity map in the viewport, like below:

Note

Any settings under **Render Settings > Post Processing > Depth Sensor** will be applied to all render products in the scene (including the viewport). The `isaacsim.sensors.camera.SingleViewDepthSensor` class
enables configuration of individual render products as depth sensors.

Close the Isaac Sim UI and rerun the standalone example as follows:

```python
./python.sh standalone_examples/api/isaacsim.sensors.camera/camera_stereoscopic_depth.py --test
```

Isaac Sim will now run the standalone example in headless mode and generate the following output from Annotators
attached to the `camera` render product. The first image is output from the `DepthSensorDistance` Annotator (`depth_sensor_distance.png`), and the
second image is output from the `DistanceToImagePlane` Annotator (`distance_to_image_plane.png`).

Warning

When using any of the new depth AOVs, you might see the following (or similar) errors:

```python
[Error] [rtx.postprocessing.plugin] DepthSensor: Texture sizes do not match: inColorTexDesc 1920x1080x1:11@0 inDepthTexDesc 1500x843x1:33@0
[Error] [rtx.postprocessing.plugin] DepthSensor: Failed to allocate view resources for view 1 device 0
[Error] [carb.scenerenderer-rtx.plugin] Failed to export AOV 38 to render product. The renderer did not generate the AOV texture
```

These errors are expected for the first frame of the depth simulation and will be corrected in a future release.

### Depth Camera Asset Wrapper

Isaac Sim supports several official [Depth Sensors](Sensors.md). These can be automatically loaded as references on a stage
using the `isaacsim.sensors.camera.SingleViewDepthSensorAsset` class. This API will search the asset for ``` RenderProduct``prims specifying single-view depth
sensor characteristics, tailored for a specific camera in the asset, then wrap those ``Camera ``` prims as `SingleViewDepthSensor` instances. By loading the
asset in this manner, you will have full control over the post-processing pipeline for each depth sensor in the asset, and can attach any number of Annotators
to the `SingleViewDepthSensor` instances through its API.

Note

Attribute specification for `Camera` prims in the official assets linked above are tentative, and can change in future asset updates or releases.

#### Script Editor

As an example, you can load the Intel Realsense D455 depth camera asset and attach an annotator to the depth sensor by running the
following snippet in the Script Editor:

```python
from isaacsim.sensors.camera import SingleViewDepthSensorAsset
from isaacsim.storage.native import get_assets_root_path

# Add Realsense D455 to the stage
asset_path = get_assets_root_path() + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
realsense_d455 = SingleViewDepthSensorAsset(prim_path="/World/realsense_d455", asset_path=asset_path)

# Initialize all depth sensor prims in the asset, creating render products
# attached to HydraTextures for each.
realsense_d455.initialize()

# Print prim paths for all available depth sensors in the asset
print(realsense_d455.get_all_depth_sensor_paths())

# Get a specific depth sensor by camera prim path
depth_sensor = realsense_d455.get_child_depth_sensor("/World/realsense_d455/RSD455/Camera_Pseudo_Depth")

# Attach an Annotator to the depth sensor
depth_sensor.attach_annotator("DepthSensorDistance")
```

Observe the Stage window indicates the Realsense D455 depth camera asset has been loaded.

Next, observer the Layer window indicates the appropriate `RenderProduct` prim has been created, with a
HydraTexture and `DepthSensorDistance` `RenderVar` attached:

### Building a Depth Sensor Model in Isaac Sim

#### Updating Existing Assets to Use Depth Sensors

Isaac Sim provides a convenient API to update an existing asset to use depth sensors using the `isaacsim.sensors.camera.SingleViewDepthSensorAsset` class. The following example demonstrates how to update a new `Camera` prim
as a depth sensor, then export it as a USD file that can be loaded as a reference in other stages using `isaacsim.sensors.camera.SingleViewDepthSensorAsset`.

```python
./python.sh standalone_examples/api/isaacsim.sensors.camera/camera_add_depth_sensor.py
```

Running the example will create a new `example_camera_with_depth_sensor.usd` asset in the local directory.
After opening the new asset in Isaac Sim, observe the following in the **Stage** window:

Observe the new render product prim has been created and associated to the `Camera` prim, with the custom
value set for the `omni:rtx:post:depthSensor:baselineMM` attribute.

Open a new stage, and run the following snippet in the **Script Editor** to load the new asset as a reference:

```python
from isaacsim.sensors.camera import SingleViewDepthSensorAsset

asset_path = "example_camera_with_depth_sensor.usd"
example_depth_sensor = SingleViewDepthSensorAsset(prim_path="/example_depth_sensor", asset_path=asset_path)
example_depth_sensor.initialize()
```

Observe in the Layer window the new render product is appropriately created, with the custom value set for the `omni:rtx:post:depthSensor:baselineMM` attribute.

#### Creating a New Depth Sensor Asset

As noted earlier, the [Single-View Post-Processing Pipeline](#isaacsim-sensors-camera-depth-stereoscopic-pipeline) is intended to model stereoscopic depth cameras specifically, not (eg.) time-of-flight
sensors or structured light sensors. This section will link to other sections of Isaac Sim documentation to describe a general process for building a new stereoscopic
depth sensor model, but should not be used as a template for other types of depth sensors.

1. Use any of the supported Isaac Sim [Importers and Exporters](Importers_and_Exporters.md) to import an existing model of the depth sensor into USD.
2. Add `Camera` prims to appropriate locations in the model and save the asset.
3. Build a test environment in USD, positioning objects and the depth sensor in the environment to accurately model a real-world test rig.
4. If using OpenCV to calibrate the real-world cameras, apply the OpenCV lens distortion schemas to the `Camera` prims, as described in [Calibration and Camera Lens Distortion Models](Sensors.md).
5. Calibrate camera intrinsics and extrinsics for each Camera prim by comparing rendered images to real-world images and tuning Camera prim attributes.
6. When the camera intrinsics and extrinsics are calibrated, refer to examples in [Standalone Python](#isaacsim-sensors-camera-depth-stereoscopic-standalone) to script the post-processing pipeline: apply the depth sensor schema to a render product
   attached to the depth sensor `Camera` prim, set attributes, render a depth image, and compare the rendered depth image to the real-world depth image. Update depth sensor schema attributes, and repeat the process until the
   rendered depth image matches the real-world depth image within some acceptable threshold.

---

# RTX Sensors

RTX sensors in Isaac Sim use the Omniverse RTX Renderer’s RTX Sensor SDK to sense the environment, enabling interaction with materials in visual and non-visual spectra.
This means an RTX-based Lidar can model returns from light interaction with transparent or reflective surfaces, and an RTX-based Radar can model returns accounting for
material emissivity and reflectivity in the radio spectrum.

Isaac Sim organizes utilities supporting RTX sensors into the `isaacsim.sensors.rtx` extension.

## Getting Started

To quickly get started with RTX sensors:

1. **Add a sensor to your scene**: Use **Create** > **Isaac** > **Sensors** > **RTX Lidar** or **RTX Radar** from the menu, or use the Python APIs described in the sensor-specific pages below.
2. **Collect data**: Attach [annotators](Sensors.md) to the sensor to extract point cloud data, scan buffers, or raw `GenericModelOutput` data.
3. **Visualize output**: Use the [Debug Draw Extension](Debugging_Profiling.md) to visualize point clouds, or configure viewport debug views.
4. **Integrate with ROS2**: Follow the [RTX Lidar ROS2 Tutorial](ROS_2.md) to publish sensor data as `PointCloud2` or `LaserScan` messages.

## Sensor Types

- RTX Lidar Sensor
- RTX Radar Sensor

## Data Collection and Materials

- RTX Sensor Annotators
- RTX Sensor Non-Visual Materials

## Advanced Topics

- Creating Custom RTX Sensor Profiles

## Extension Architecture

RTX sensors are built using the `omni.sensors` extension suite. To understand more about how RTX sensors are modeled,
and how to build your own, review the following documentation:

* [Omniverse Common Extension](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.common/2.7.0-coreapi/common_extension.html)
* [Omniverse Lidar Extension](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.lidar/2.7.0-coreapi/lidar_extension.html)
* [Omniverse Radar Extension](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.radar/2.8.0-coreapi/radar_extension.html)
* [Omniverse Materials Extension](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.materials/1.6.0-coreapi/materials_extension.html)

## Important Settings

The following settings affect RTX sensor behavior and performance:

| Setting | Default | Description |
| --- | --- | --- |
| `--/app/sensors/nv/lidar/outputBufferOnGPU` | `true` | Output Lidar data on GPU. Must be `true` for annotators to work correctly. |
| `--/app/sensors/nv/radar/outputBufferOnGPU` | `true` | Output Radar data on GPU. Must be `true` for annotators to work correctly. |
| `--/app/sensors/nv/lidar/publishNormals` | `false` | Enable hit normal output. Increases VRAM usage. |
| `--/rtx-transient/stableIds/enabled` | `false` | Enable stable 128-bit object IDs for semantic segmentation. |
| `--/renderer/raytracingMotion/enabled` | `false` | Enable Motion BVH for motion compensation and Doppler effects. |

## Motion BVH

RTX sensors use Motion BVH to improve accuracy when modeling motion-related sensor effects, for example, the motion of objects during sensor exposure, or the motion of the sensor itself as it collects data.

By default, Motion BVH is disabled in Isaac Sim to improve performance. The following RTX Sensor features are affected by Motion BVH:

* RTX Lidar

  + Motion BVH must be enabled for RTX Lidar motion compensation to work correctly.
* RTX Radar

  + Motion BVH must be enabled for the Doppler effect, and therefore RTX Radar entirely, to be modeled correctly.

### How to Enable Motion BVH

Note

Enabling Motion BVH can significantly increase rendering time by increasing VRAM usage for all sensors and must be left disabled when not needed.

There are two ways to enable Motion BVH:

1. In standalone Python workflows, you can enable Motion BVH by specifying `enable_motion_bvh` as `True` in the `SimulationApp` constructor:

> ```python
> from isaacsim import SimulationApp
>
> simulation_app = SimulationApp({"enable_motion_bvh": True})
>
> simulation_app.close()
> ```

2. In all workflows, you can enable Motion BVH by specifying the following settings on the command line:

> ```python
> --/renderer/raytracingMotion/enabled=true \
> --/renderer/raytracingMotion/enableHydraEngineMasking=true \
> --/renderer/raytracingMotion/enabledForHydraEngines='0,1,2,3,4'
> ```

## Troubleshooting and Known Issues

### Common Issues

**Annotators return empty data**
:   Ensure the simulation timeline is playing. RTX Sensor Annotators rely on the timeline to collect data.
    Also verify that `--/app/sensors/nv/lidar/outputBufferOnGPU` or `--/app/sensors/nv/radar/outputBufferOnGPU` is set to `true`.

**Point cloud appears to “drag” behind moving objects**
:   If the Lidar rotation rate is slower than the frame rate, accumulated scan data may contain returns from multiple frames.
    This is expected behavior for rotating Lidars. Consider using per-frame output instead of accumulated scans.

**Radar simulation does not show Doppler effects**
:   Motion BVH must be enabled for Doppler effects to be modeled correctly. See [How to Enable Motion BVH](#isaac-sim-sensors-rtx-how-to-enable-motion-bvh).

**Timestamps are discontinuous after pause/resume**
:   The `GenericModelOutput` AOV timestamp is independent of the animation timeline and continues to increase even when paused.
    This is expected behavior.

### Performance Considerations

* **VRAM Usage**: Each RTX sensor requires GPU memory. Multiple sensors or high-resolution configurations increase VRAM usage.
* **Motion BVH**: Enabling Motion BVH significantly increases VRAM usage and rendering time.
* **Normal Output**: Enabling `--/app/sensors/nv/lidar/publishNormals=true` increases VRAM usage.
* **Stable IDs**: Enabling `--/rtx-transient/stableIds/enabled=true` has minimal performance impact but requires additional processing for object ID resolution.

### Hardware Requirements

RTX sensors require an NVIDIA RTX GPU with ray tracing support. Performance scales with GPU capabilities, particularly:

* VRAM capacity (affects number of sensors and resolution)
* Ray tracing cores (affects simulation speed)

## Related Tutorials

* [RTX Lidar Sensors](ROS_2.md) - Publishing RTX Lidar data to ROS2
* [Debug Drawing Extension API](Debugging_Profiling.md) - Visualizing point clouds and geometry
* [Util Snippets](Python_Scripting_and_Tutorials.md) - Rendering and visualization utilities

---

# RTX Lidar Sensor

RTX Lidar sensors are simulated at render time on the GPU with RTX hardware.
Their results are then copied to the `GenericModelOutput` AOV for use.

## Overview

Note

In Isaac Sim 4.5 and earlier, RTX sensors were based on `Camera` prims. If the `Camera` prim’s
`sensorModelPluginName` attribute was set to `omni.sensors.nv.lidar.lidar_core.plugin`, then the
`Camera` prim was used to render the Lidar. The Lidar was configured using a JSON file whose
filename (without extension) was set in the `Camera` prim’s `sensorModelConfig` attribute, assuming
the file was present in a folder specified by the `app.sensors.nv.lidar.profileBaseFolder` setting.
Support for `Camera` prims as RTX Lidars was deprecated in Isaac Sim 5.0.

See [Convert a JSON File to an OmniLidar USD File](#isaacsim-sensors-rtx-lidar-convert-json-to-omni-lidar) for details on how to convert a JSON file to a USD file containing an equivalent `OmniLidar` prim.

RTX Lidars are rendered using `OmniLidar` prims, with the `OmniSensorGenericLidarCoreAPI` schema applied,
as configured by attributes on the prim. After attaching a render product to the `OmniLidar` prim, and setting
the `GenericModelOutput` AOV on the render product, the RTXSensor renderer will write Lidar render results to the AOV.

The `OmniSensorGenericLidarCoreAPI` schema is defined in the `omni.usd.schema.omni_sensors` extension, documented [here](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.usd.schema.omni_sensors/107.3.0/omni_sensors_schema.html).

## How to Create an RTX Lidar

The `isaacsim.sensors.rtx` extension provides two APIs for creating RTX Lidars. In addition, the `omni.replicator.core`
extension provides even lower-level APIs for creating `OmniLidar` prims (including batch creation) and attaching render
products to them.

### Create an RTX Lidar through Command

The lower-level `IsaacSensorCreateRtxLidar` command creates a reference on the stage to a known Lidar USD or USDA asset,
a generic `OmniLidar` prim with the appropriate schemas applied, or a `Camera` prim with the appropriate attributes
to support deprecated workflows.

```python
import omni
from pxr import Gf

# Specify attributes to apply to the ``OmniLidar`` prim.
sensor_attributes = {"omni:sensor:Core:scanRateBaseHz": 20}

_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    translation=Gf.Vec3d(0, 0, 0),
    orientation=Gf.Quatd(
        1,
        0,
        0,
        0,
    ),
    path="/lidar",
    parent=None,
    config="Example_Rotary",
    visiblity=False,
    variant=None,
    force_camera_prim=False,
    **sensor_attributes,
)
```

The example command above creates a reference to `Example_Rotary.usda` as an `OmniLidar` prim in the stage at the
specified `translation` with the specified `orientation`, at path `/lidar`. The prim is set to be invisible
in the stage. The `Example_Rotary` config does not support variant sets, so `variant` is unused. The prim’s
`omni:sensor:Core:scanRateBaseHz` attribute is set from 10 Hz (default) to 20 Hz.

Setting `force_camera_prim` to `True` will instead create an invisible `Camera` prim at the specified `translation`
and `orientation`, with the `sensorModelConfig` attribute set to `Example_Rotary`.

Setting `config` to `None` will create a generic `OmniLidar` prim with the `OmniSensorGenericLidarCoreAPI` schema applied;
any additional keyword arguments will be passed through and set as attributes on the `OmniLidar` prim.

Review the [OmniSensorGenericLidarCoreAPI](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html#omnisensorgenericlidarcoreapi)
schema and [OmniSensorGenericLidarCoreEmitterStateAPI](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html#omnisensorgenericlidarcoreemitterstateapi)
schema in the `omni.usd.schema.omni_sensors` extension to learn which attributes can be set on the `OmniLidar` prim.

If you are specifying emitter state attributes, the attribute names must be prefixed with the appropriate emitter state count, for example,
`OmniSensorGenericLidarCoreEmitterStateAPI:s001:elevationDeg` or `OmniSensorGenericLidarCoreEmitterStateAPI:s002:azimuthDeg`.

### Create an RTX Lidar through the `LidarRtx` Class

The higher-level `LidarRtx` class provides a Python interface for creating and configuring RTX Lidars.
In addition to passing constructor arguments to the `IsaacSensorCreateRtxLidar` command, the `LidarRtx`
class automatically wraps around the resulting `OmniLidar` prim and attaches a render product to it.

It includes APIs to attach appropriate `isaacsim.sensors.rtx`, [annotators](Sensors.md), and any writers to the render product. It also includes APIs to read annotator and writer results each frame through a data dictionary returned by the
`get_data` method.

An example of creating an RTX Lidar through the `LidarRtx` class is shown below:

```python
import numpy as np
import omni
from isaacsim.sensors.rtx import LidarRtx

sensor_attributes = {"omni:sensor:Core:scanRateBaseHz": 20}

# Create the RTX Lidar with the specified attributes.
sensor = LidarRtx(
    prim_path="/lidar",
    translation=np.array([0.0, 0.0, 1.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    config_file_name="Example_Rotary",
    **sensor_attributes,
)
```

Similar to the command above, the specified call to `LidarRtx` creates a reference to `Example_Rotary.usda` as
an `OmniLidar` prim in the stage at the specified `translation` with the specified `orientation`, at path
`/lidar`. The prim is set to be invisible in the stage. The `Example_Rotary` config does not support variant sets,
so `variant` is unused. The prim’s `omni:sensor:Core:scanRateBaseHz` attribute is set from 10 Hz (default) to 20 Hz.

## How to Collect Data from an RTX Lidar

The recommended method for collecting data from an RTX Lidar is to use Replicator Annotators.

Isaac Sim offers multiple [RTX Sensor Annotators](Sensors.md). The `LidarRtx` class
described above offers APIs for attaching any of those annotators to the `OmniLidar` prim it wraps,
as well as the `GenericModelOutput` annotator. Refer to [Reading Data from the GenericModelOutput Buffer](Sensors.md) for
more details on how to use the `GenericModelOutput` annotator.

## Visualizing RTX Lidar Output

There are several ways to visualize RTX Lidar point cloud data in Isaac Sim:

### Debug Draw

The [Debug Draw Extension](Debugging_Profiling.md) provides a performance-efficient method for visualizing point clouds directly in the viewport.
The geometry drawn with Debug Draw remains persistent across frames and does not interact with the physics scene.

The standalone example `rtx_lidar.py` demonstrates using Debug Draw to visualize RTX Lidar output:

```python
./python.sh standalone_examples/api/isaacsim.util.debug_draw/rtx_lidar.py --config Example_Rotary
```

For more information on Debug Draw APIs, see [Debug Drawing Extension API](Debugging_Profiling.md) and [Util Snippets](Python_Scripting_and_Tutorials.md).

### Viewport Debug Views

You can visualize non-visual material IDs in the viewport by selecting **RTX - Real-Time** > **Debug View** > **Non-Visual Material ID**.
This shows how materials appear to RTX sensors, which is useful for debugging material configurations.
See [RTX Sensor Non-Visual Materials](Sensors.md) for details.

### RViz2 Visualization

When using ROS2, point cloud data can be visualized in RViz2. See the [ROS2 Integration](#isaacsim-sensors-rtx-lidar-ros2) section below.

## ROS2 Integration

Isaac Sim provides full support for publishing RTX Lidar data to ROS2 as standard message types.

### Supported Message Types

* `sensor_msgs/PointCloud2` - Full 3D point cloud data
* `sensor_msgs/LaserScan` - 2D laser scan data (for 2D Lidar configurations)

For a comprehensive guide on integrating RTX Lidar sensors with ROS2, including:

* Adding RTX Lidar ROS2 bridge nodes via OmniGraph
* Publishing LaserScan and PointCloud2 messages
* Using the menu shortcut to create RTX Lidar sensor publishers
* Visualizing multiple sensors in RViz2
* Exposing RTX Lidar metadata (intensity, object IDs) in PointCloud2 messages

See the [RTX Lidar ROS2 Tutorial](ROS_2.md).

### Quick Start

To quickly add ROS2 publishing for an RTX Lidar sensor:

1. Create an RTX Lidar sensor using the methods described above.
2. Go to **Tools** > **Robotics** > **ROS 2 OmniGraphs** > **RTX Lidar**.
3. Configure the graph path, Lidar prim, frame ID, and select the data types to publish.
4. Press **Play** to begin publishing.

## RTX Lidar Asset Library

Isaac Sim includes a library of [RTX Lidars](Sensors.md) that can be loaded
onto the stage by specifying the `config` and `variant` parameters of the `IsaacSensorCreateRtxLidar` command,
or the `config_file_name` parameter of the `LidarRtx` constructor. The `config` or `config_file_name` parameter can be the following:

* The exact name of a Lidar model USD file without extension, as provided in the *Content Browser* and noted in the [RTX Lidars](Sensors.md) library (for example, `HESAI_XT32_SD10`).
* The exact name of a Lidar model USD file as noted above, but with spaces replacing underscore (for example, `HESAI XT32 SD10`).
* The exact name of a Lidar model USD file as noted above, omitting the vendor name (for example, `XT32_SD10`).
* The exact name of a Lidar model USD file as noted above, omitting the vendor name and replacing underscores with spaces (for example, `XT32 SD10`). This option matches the name of the Lidar in the **Create** > **Isaac** > **Sensors** menu.

The optional `variant` will select the specific variant of the provided Lidar configuration, as noted in the model’s documentation. For example,
the snippet below will load a SICK picoScan150 Lidar with the `Normal_11` variant selected.

```python
import omni
from pxr import Gf

_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path="/lidar",
    config="picoScan150",
    variant="Profile_11",
)
```

## Sensor Materials

The material system for RTX Lidar allows content creators to assign sensor material types to partial material prim names on a USD stage. Lidar return behavior depends on material properties (for example, emissivity, reflectivity),
as described below.

- RTX Sensor Non-Visual Materials

## Convert a JSON File to an OmniLidar USD File

Isaac Sim includes a utility tool to automatically convert legacy JSON Lidar configuration files to USD file(s) containing OmniLidar prims.

The tool can be run as a standalone application using:

```python
./python.sh tools/isaacsim.sensors.rtx/convert_lidar_json_to_usda.py
```

Providing the `-h` or `--help` flag will display the usage information for the tool.

The tool will automatically convert multiple provided JSON files to corresponding USD files containing an equivalent `OmniLidar` prim,
and can compile JSON files associated with variant configurations or profiles of the same Lidar model into a single USD, using
[USD variant sets](https://docs.omniverse.nvidia.com/usd/latest/learn-openusd/terms/variant-set.html) to allow the user to select the appropriate profile when creating an `OmniLidar` prim.

## Standalone Examples

For examples of creating and/or collecting data from a RTX Lidar, refer to the following:

```python
./python.sh standalone_examples/api/isaacsim.ros2.bridge/rtx_lidar.py
```

```python
./python.sh standalone_examples/api/isaacsim.sensors.rtx/inspect_lidar_metadata.py
```

```python
./python.sh standalone_examples/api/isaacsim.sensors.rtx/resolve_object_ids_from_gmo.py
```

```python
./python.sh standalone_examples/api/isaacsim.sensors.rtx/rotating_lidar_rtx.py
```

```python
./python.sh standalone_examples/api/isaacsim.util.debug_draw/rtx_lidar.py --config Example_Rotary
```

```python
./python.sh standalone_examples/api/isaacsim.util.debug_draw/rtx_lidar.py --config Example_Solid_State
```

Note

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

---

# RTX Sensor Non-Visual Materials

The `omni.sensors.nv.materials` extension, documented [here](http://omniverse-docs.s3-website-us-east-1.amazonaws.com/omni.sensors.nv.materials/1.6.0-coreapi/materials_extension.html), provides support for rendering materials which are visible in non-visual spectra for RTX sensors. These materials
are referred to as “non-visual materials”.

As described in the extension documentation, non-visual materials are rendered via USD attributes, and can be specified in the USD file. Isaac Sim includes APIs in the `isaacsim.sensors.rtx` extension to simplify setting these attributes on `Material` prims. The renderer
will compute a material ID for each non-visual material, based on the combination of provided attributes. This material ID is provided by the `GenericModelOutput` AOV, and is exposed by multiple Annotators. See [RTX Sensor Annotators](Sensors.md) for more details.

## Specifying Non-Visual Material Attributes

Valid non-visual material attribute names and values are specified [in Omniverse Kit documentation](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.materials/latest/materials_extension.html#materials-coatings-and-attributes).

### User Interface

Attributes may be added to materials from the UI by right-clicking the material in the *Stage* window, then selecting **Add** > **Attribute**.
This will open a new window like the one below, enabling you to specify custom non-visual attributes.

After adding the new attribute, it will appear in the material’s properties, at which point it can be populated:

### Python

`isaacsim.sensors.rtx` includes several Python APIs to simplify setting non-visual material attributes on `Material` prims. The following standalone example
demonstrates how to use these APIs. Examine the source code to learn more.

```python
./python.sh standalone_examples/api/isaacsim.sensors.rtx/specify_non_visual_materials.py
```

Upon running this example, you should see the following:

Observe each cube is colored differently in the visual spectrum. Select the `Non-Visual Material ID` Debug View in the viewport by selecting **RTX - Real-Time** > **Debug View** > **Non-Visual Material ID**. The following image
shows the menu selection:

After selecting the Debug View, you should see the following:

The `Non-Visual Material ID` Debug View shows the material ID for each non-visual material as a color, which can be used to identify the material in the scene.
Observe each cube’s color changes compared to the default view to reflect the material ID, which is computed from the combination of non-visual material attributes applied to the visual material
applied to the cube.

## Mapping Visual Materials to RTX Sensor Non-Visual Materials (Deprecated)

Warning

Mapping Visual Materials to RTX Sensor non-visual materials via a CSV specification is deprecated as of Isaac Sim 5.1. By default, RTX Sensor non-visual materials will
now be specified and rendered via USD attributes (see above).

There are 21 sensor materials that are rendered in the visual spectrum, and more can not be added at this time. Their properties are stored in JSON files by the same name, located in
the `./data/material_files/` folder.

| Index | Sensor Material Type |
| --- | --- |
| 0 | Default |
| 1 | AsphaltStandard |
| 2 | AsphaltWeathered |
| 3 | VegetationGrass |
| 4 | WaterStandard |
| 5 | GlassStandard |
| 6 | FiberGlass |
| 7 | MetalAlloy |
| 8 | MetalAluminum |
| 9 | MetalAluminumOxidized |
| 10 | PlasticStandard |
| 11 | RetroMarkings |
| 12 | RetroSign |
| 13 | RubberStandard |
| 14 | SoilClay |
| 15 | ConcreteRough |
| 16 | ConcreteSmooth |
| 17 | OakTreeBark |
| 18 | FabricStandard |
| 19 | PlexiGlassStandard |
| 20 | MetalSilver |
| 31 | INVALID |

### Using Sensor Material Mapping

In the legacy system, Isaac Sim must know how to map material IDs to the sensor material type
in the table above. This is done by setting the following `carb` setting on the command line:

```python
--/rtx/materialDb/rtSensorNameToIdMap="DefaultMaterial:0;AsphaltStandardMaterial:1;AsphaltWeatheredMaterial:2;VegetationGrassMaterial:3;WaterStandardMaterial:4;GlassStandardMaterial:5;FiberGlassMaterial:6;MetalAlloyMaterial:7;MetalAluminumMaterial:8;MetalAluminumOxidizedMaterial:9;PlasticStandardMaterial:10;RetroMarkingsMaterial:11;RetroSignMaterial:12;RubberStandardMaterial:13;SoilClayMaterial:14;ConcreteRoughMaterial:15;ConcreteSmoothMaterial:16;OakTreeBarkMaterial:17;FabricStandardMaterial:18;PlexiGlassStandardMaterial:19;MetalSilverMaterial:20"
```

Having set `rtx.materialDb.rtSensorNameToIdMap`, edit `kit/rendering-data/runtime/RtxSensorMaterialMap.csv` to map exact material name tokens to sensor material types.

The `RtxSensorMaterialMap.csv` file contains a material prim partial names to sensor material type pairs. The ones that come with Isaac Sim by default can be deleted as they may clash with names you wish to set.
There is only one CSV file. It controls the material mapping for all of the content. It is read at Isaac Sim startup and any changes made during runtime will not appear until Isaac Sim is restarted.

As an example, consider this scene:

The `/Root/SM_floor29/SM_floor02/SM_floor02` prim has a material prim assigned to it whose path is `/Root/SM_floor29/Looks/MI_Floor_02b`. If you want to add
an entry to the CSV file so that the `SM_floor02` prim looks like rough concrete to the RTX sensors, you would add the entry:

```python
mi_floor_02b,ConcreteRoughMaterial
```

Note that in the CSV mapping file, the first token after the first appearance of `/Looks/` in the material prim name attached to the mesh is used, and it must
always be lowercase in the CSV file, no matter what the case is on the stage. Also note how the word Material is concatenated onto the sensor material type from
the table above.

### Debugging

The carb parameter:

```python
[settings]
rtx.materialDb.rtSensorMaterialLogs=true
```

can help. If set to true, it will output a list of all the materials in the scene that are NOT mapped to a sensor material. This
list outputs to the terminal and the log at Isaac Sim startup.

---

# RTX Radar Sensor

RTX Radar sensors are simulated at render time on the GPU with RTX hardware.
Their results are then copied to the `GenericModelOutput` AOV for use.

Warning

**Motion BVH Must Be Enabled for RTX Radar**

RTX Radar requires Motion BVH to be enabled for the Doppler effect—and therefore RTX Radar entirely—to be modeled correctly.
**Without Motion BVH enabled, RTX Radar will not produce accurate results.**

Motion BVH is disabled by default in Isaac Sim for performance reasons. You must explicitly enable it before using RTX Radar.

**To enable Motion BVH**, add the following command line arguments when launching Isaac Sim:

```python
--/renderer/raytracingMotion/enabled=true \
--/renderer/raytracingMotion/enableHydraEngineMasking=true \
--/renderer/raytracingMotion/enabledForHydraEngines='0,1,2,3,4'
```

Or in standalone Python, pass `enable_motion_bvh=True` to the `SimulationApp` constructor.

See [How to Enable Motion BVH](Sensors.md) for complete instructions.

## Overview

Note

In Isaac Sim 4.5 and earlier, RTX sensors were based on `Camera` prims. If the `Camera` prim’s
`sensorModelPluginName` attribute was set to `omni.sensors.nv.radar.wpm_dmatapprox.plugin`, then the
`Camera` prim was used to render the Radar. The Radar was configured using a JSON file whose
filename (without extension) was set in the `Camera` prim’s `sensorModelConfig` attribute, assuming
the file was present in a folder specified by the `app.sensors.nv.radar.profileBaseFolder` setting.
Support for `Camera` prims as RTX Radars was deprecated in Isaac Sim 5.0.

RTX Radars are rendered using `OmniRadar` prims, with the `OmniSensorGenericRadarWpmDmatAPI` schema applied,
as configured by attributes on the prim. After attaching a render product to the `OmniRadar` prim, and setting
the `GenericModelOutput` AOV on the render product, the RTXSensor renderer will write Radar render results to the AOV.

## How to Create an RTX Radar

The `isaacsim.sensors.rtx` extension provides one API for creating RTX Radars. In addition, the `omni.replicator.core`
extension provides even lower-level APIs for creating `OmnRadar` prims (including batch creation) and attaching render
products to them.

### Create an RTX Radar using Command

The `IsaacSensorCreateRtxRadar` command creates
a generic `OmniRadar` prim with the appropriate schemas applied, or a `Camera` prim with the appropriate attributes
to support deprecated workflows.

```python
import omni
from pxr import Gf

# Specify attributes to apply to the ``OmniRadar`` prim.
sensor_attributes = {"omni:sensor:tickRate": 10}

_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxRadar",
    translation=Gf.Vec3d(0, 0, 0),
    orientation=Gf.Quatd(
        1,
        0,
        0,
        0,
    ),
    path="/radar",
    parent=None,
    visibility=False,
    variant=None,
    force_camera_prim=False,
    **sensor_attributes,
)
```

The example command above creates an `OmniRadar` prim in the stage at the
specified `translation` with the specified `orientation`, at path `/radar`. The prim is set to be invisible
in the stage. The prim’s `omni:sensor:Core:tickRate` attribute is set to 10 Hz from 20 Hz (default).

Review the [OmniSensorGenericRadarWpmDmatAPI](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html#omnisensorgenericradarwpmdmatapi)
schema in the `omni.usd.schema.omni_sensors` extension to learn which attributes can be set on the `OmniRadar` prim.

Setting `force_camera_prim` to `True` will instead create an invisible `Camera` prim at the specified `translation`
and `orientation`.

Annotators can then be attached to the `OmniRadar` prim to collect and visualize the Radar results.
Details about available annotators can be explored [here](Sensors.md).

## How to Collect Data from an RTX Radar

The recommended method for collecting data from an RTX Radar is to use Replicator Annotators, similar to RTX Lidar.

The `IsaacExtractRTXSensorPointCloudNoAccumulator` annotator works with both `OmniLidar` and `OmniRadar` prims,
extracting point cloud data from the `GenericModelOutput` buffer every frame.

Refer to [RTX Sensor Annotators](Sensors.md) for the full list of available annotators.

## Visualizing RTX Radar Output

### Debug Draw

The [Debug Draw Extension](Debugging_Profiling.md) can be used to visualize RTX Radar point cloud output in the viewport.

The standalone example `rtx_radar.py` demonstrates using Debug Draw to visualize RTX Radar output:

```python
./python.sh standalone_examples/api/isaacsim.util.debug_draw/rtx_radar.py
```

For more information on Debug Draw APIs, see [Debug Drawing Extension API](Debugging_Profiling.md) and [Util Snippets](Python_Scripting_and_Tutorials.md).

### Doppler Effects

Important

Motion BVH must be enabled for the Doppler effect to be modeled correctly in RTX Radar simulations.
See [How to Enable Motion BVH](Sensors.md) for instructions on enabling Motion BVH.

## Sensor Materials

The material system for RTX Radar allows content creators to assign sensor material types to partial material prim names on a USD stage. Radar return behavior depends on material properties (for example, emissivity, reflectivity),
as described below.

- RTX Sensor Non-Visual Materials

## Standalone Examples

For examples of creating RTX Radar refer to the examples:

```python
./python.sh standalone_examples/api/isaacsim.util.debug_draw/rtx_radar.py
```

Note

Refer to the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

---

# RTX Sensor Annotators

The `isaacsim.sensors.rtx` extension uses Omniverse Replicator to provide Annotators for RTX Lidar and Radar data collection.
Annotators can be attached to render products, attached to `OmniSensor` prims (for example, `OmniLidar` or `OmniRadar`); for example,
when run in the *Script Editor*, the following snippet creates an `OmniLidar` prim at `/lidar`, a render product for the sensor,
and attaches an `IsaacExtractRTXSensorPointCloudNoAccumulator` annotator to the render product.

```python
import omni
import omni.replicator.core as rep
from pxr import Gf

# Create an OmniLidar prim at prim path /lidar
_, sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    translation=Gf.Vec3d(0.0, 0.0, 0.0),
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
    path="/lidar",
)

# Create a render product for the sensor.
render_product = rep.create.render_product(sensor.GetPath(), resolution=(1024, 1024))

# Create an annotator
annotator = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")

# Attach the render product after the annotator is initialized.
annotator.attach([render_product.path])
```

Alternatively, the `LidarRtx` class offers a single API for attaching any annotator to an `OmniLidar` prim
and collecting data. For example, in a standalone Python workflow, the following snippet creates an `OmniLidar` prim at `/lidar`,
creates a render product for the sensor, and attaches an `IsaacExtractRTXSensorPointCloudNoAccumulator` annotator to it, then collects
data from the annotator on each simulation frame. Note that this snippet will not run in the script editor window.

```python
from isaacsim import SimulationApp

kit = SimulationApp()

import numpy as np
import omni
from isaacsim.sensors.rtx import LidarRtx

# Create the RTX Lidar with the specified attributes.
sensor = LidarRtx(
    prim_path="/lidar",
    translation=np.array([0.0, 0.0, 1.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    config_file_name="Example_Rotary",
)

# Initialize the LidarRtx object, which creates a render product for the sensor.
sensor.initialize()

# Attach an annotator to the sensor.
sensor.attach_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")

# Play the timeline to initialize the OmniGraph associated with the annotator and render product,
# and begin collecting data.
timeline = omni.timeline.get_timeline_interface()
timeline.play()

# Collect data from the annotator on each simulation frame.
for _ in range(100):
    # Step the simulation
    kit.update()
    # Print data collected by each annotator attached to the sensor as a Python dict
    print(sensor.get_current_frame())

timeline.stop()
kit.close()
```

## Time Behavior of RTX Sensor Annotators

Warning

RTX Sensor Annotators rely on the simulation timeline to collect data. If the timeline is not playing (e.g. if the simulation is paused or stopped), the annotators will not collect data.

The `GenericModelOutput` AOV produced by RTX Sensors contains an internal timestamp, which increases monotonically starting when `App Ready` appears in the simulation logs. This timestamp is
independent of the animation timeline (`omni.timeline`), so the sensor timestamp will continue to increase even if the timeline is paused or stopped. This AOV feeds into all other RTX Sensor Annotators.

If the user pauses the timeline, then resumes, timestamps in the `GenericModelOutput` point cloud (eg. the `timestamp` field of `IsaacCreateRTXLidarScanBuffer` below) may be discontinuous. This also means the simulation must be
stepped using `omni.kit.app.get_app().update` or `omni.kit.app.get_app().next_update_async()` rather than `omni.replicator.core.orchestrator.step()` or `omni.replicator.core.orchestrator.step_async()`
when collecting data using these Annotators.

Note

Isaac Sim APIs for controlling the simulation state use the former two methods rather than the latter two.

## Annotators

Each `isaacsim.sensors.rtx` Annotator is associated with a specific `isaacsim.sensors.rtx` OmniGraph node, which is linked
in that Annotator’s subsection below. The inputs and outputs of the Annotator are the same as the inputs and outputs of the
corresponding OmniGraph node.

Note

In Isaac Sim 5.0, several existing `isaacsim.sensors.rtx` annotators were removed in favor of simpler
annotators that can handle output from the new `OmniLidar` or `OmniRadar` prims, in addition to the
deprecated `Camera`-prim-based workflows. See [Deprecated Annotators](#rtx-sensor-deprecated-annotators) for details.

If the Lidar rotation rate is slower than the frame rate, data from Annotators for accumulated Lidar scans will contain returns from multiple frames. If the Lidar prim moves between frames, or objects
move in the scene, the buffer might contain returns from before the Lidar or objects moved, causing points to appear as though they are “dragging” behind objects when
viewed with the `DebugDrawPointCloud` or `DebugDrawPointCloudBuffer` writers.

`isaacsim.sensors.rtx` annotators rely on the `GenericModelOutput` AOV from the `OmniLidar` prim being
provided on device. If `--/app/sensors/nv/lidar/outputBufferOnGPU` or `--/app/sensors/nv/radar/outputBufferOnGPU` is
set to `false`, the annotators will not function correctly.

### IsaacCreateRTXLidarScanBuffer

The `IsaacCreateRTXLidarScanBuffer` Annotator accumulates frames of data from an `OmniLidar` prim into a single scan,
and provides the accumulated scan data as outputs. It is associated with the [IsaacCreateRTXLidarScanBuffer](../py/source/extensions/isaacsim.sensors.rtx/docs/ogn/OgnIsaacCreateRTXLidarScanBuffer.html) node.

Warning

The `IsaacCreateRTXLidarScanBuffer` Annotator only works with `OmniLidar` prims (RTX Lidar). It does not work with `OmniRadar` prims (RTX Radar).

By default the node outputs a 3D Cartesian point cloud, and
can optionally output the following data if the user sets the corresponding input flag to `True` when initializing the Annotator.

If creating the Annotator directly using the Replicator API, this can be done as follows:

```python
import omni.replicator.core as rep

annotator = rep.AnnotatorRegistry.get_annotator("IsaacCreateRTXLidarScanBuffer")
# Initialize the Annotator with the desired outputs.
# Note: This must be done before attaching the Annotator to a render product.
annotator.initialize(outputTimestamp=True, outputMaterialId=True)
```

If creating the Annotator through the `LidarRtx` class, this can be done as follows:

```python
import numpy as np
import omni
from isaacsim.sensors.rtx import LidarRtx

sensor = LidarRtx(
    prim_path="/lidar",
    translation=np.array([0.0, 0.0, 1.0]),
    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    config_file_name="Example_Rotary",
)
sensor.initialize()
# Initialize the specified Annotator with the flags as keyword arguments, then attach it to the render product.
sensor.attach_annotator("IsaacCreateRTXLidarScanBuffer", outputTimestamp=True, outputMaterialId=True)
```

The node outputs data as pointers to buffers and the table below specifies the data type of each buffer.

| Output | Type | Description | Notes |
| --- | --- | --- | --- |
| `data` | `float3` | 3D Cartesian point cloud. | Always provided. |
| `azimuth` | `float` | Azimuth of each return, in degrees. | Provided if `outputAzimuth` is set to `true`. |
| `elevation` | `float` | Elevation of each return, in degrees. | Provided if `outputElevation` is set to `true`. |
| `distance` | `float` | Range of each return, in world units (by default, meters). | Provided if `outputDistance` is set to `true`. |
| `intensity` | `float` | Intensity of each return, normalized as described [here](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.lidar/latest/lidar_extension.html#intensity-defining-attributes). | Provided if `outputIntensity` is set to `true`. |
| `timestamp` | `uint64` | Timestamp of each return, in nanoseconds since the start of the simulation. | Provided if `outputTimestamp` is set to `true`. |
| `emitterId` | `uint32` | ID of the emitter that emitted the return. | Provided if `outputEmitterId` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `BASIC` (or higher) on the `OmniLidar` prim. |
| `channelId` | `uint32` | ID of the channel the return was generated on. | Provided if `outputChannelId` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `BASIC` (or higher) on the `OmniLidar` prim. |
| `materialId` | `uint32` | ID of the material of the object that generated the return. | Provided if `outputMaterialId` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `EXTRA` (or higher) on the `OmniLidar` prim. Refer to [RTX Sensor Non-Visual Materials](Sensors.md) for more details on how material IDs are computed. |
| `tickId` | `uint32` | ID of the tick the return was generated on. | Provided if `outputTickId` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `BASIC` (or higher) on the `OmniLidar` prim. |
| `hitNormal` | `float3` | Normal to the surface of the object that generated the return. | Provided if `outputHitNormal` is set to `true`, `omni:sensor:Core:auxOutputType` is set to `FULL` on the `OmniLidar` prim, and `--/app/sensors/nv/lidar/publishNormals=true` is set. |
| `velocity` | `float3` | Velocity of the object that generated the return. | Provided if `outputVelocity` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `FULL` on the `OmniLidar` prim. |
| `objectId` | `uint8` | ID of the object that generated the return. | Provided if `outputObjectId` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `EXTRA` (or higher) on the `OmniLidar` prim, and `--/rtx-transient/stableIds/enabled=true` is set. Object ID is a stable, unique 128-bit unsigned integer mapping to the prim path of the object that generated the corresponding return. See [Semantic Segmentation with RTX Sensor using Object IDs](#rtx-sensor-resolving-object-ids) for more details. |
| `echoId` | `uint8` | Indicates which echo the return represents in a multi-echo Lidar configuration. | Provided if `outputEchoId` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `BASIC` (or higher) on the `OmniLidar` prim. |
| `tickState` | `uint8` | Indicates the state of the tick the return was generated on. | Provided if `outputTickState` is set to `true`, and `omni:sensor:Core:auxOutputType` is set to `BASIC` (or higher) on the `OmniLidar` prim. |

Warning

Enabling nonzero `normal` output by setting `--/app/sensors/nv/lidar/publishNormals=true` will increase VRAM usage and might negatively impact performance.

### IsaacComputeRTXLidarFlatScan

The `IsaacComputeRTXLidarFlatScan` Annotator extracts depth and azimuth data from an accumulated 2D RTX Lidar scan.
It is associated with the [IsaacComputeRTXLidarFlatScan](../py/source/extensions/isaacsim.sensors.rtx/docs/ogn/OgnIsaacComputeRTXLidarFlatScan.html) node.

Warning

The `IsaacComputeRTXLidarFlatScan` Annotator only works with `OmniLidar` prims (RTX Lidar) configured as 2D lidars, defined as having emitters only at elevation angle zero (0). It does not work with `OmniRadar` prims (RTX Radar) or 3D Lidars.

Even if `--/app/sensors/nv/lidar/outputBufferOnGPU=true` is set, `IsaacComputeRTXLidarFlatScanSimulationTime` output data will be on host memory.

### IsaacExtractRTXSensorPointCloudNoAccumulator

The `IsaacExtractRTXSensorPointCloud` Annotator extracts the `GenericModelOutput` buffer’s point cloud data
into a Cartesian vector `data` buffer every frame. It is associated with the [IsaacCreateRTXLidarScanBuffer](../py/source/extensions/isaacsim.sensors.rtx/docs/ogn/OgnIsaacCreateRTXLidarScanBuffer.html) node, with `enablePerFrameOutput` set to `true`.

Note

The `IsaacExtractRTXSensorPointCloudNoAccumulator` Annotator works with `OmniLidar` prims (RTX Lidar) and `OmniRadar` prims (RTX Radar).

Note

Isaac Sim 5.0 previously used the `IsaacExtractRTXSensorPointCloud` node for this annotator, but that node was removed
after performance improvements were made to the `IsaacCreateRTXLidarScanBuffer` node.

## Reading Data from the `GenericModelOutput` Buffer

Note

Isaac Sim 4.5 included the `OgnIsaacReadRTXLidarData` node, which provided an
example of reading data from the `GenericModelOutput` buffer in Python. This node has been removed
as of Isaac Sim 5.0 and replaced by the utility module and functions described below.

The `isaacsim.sensors.rtx.generic_model_output` Python module provides APIs for inspecting the
`GenericModelOutput` buffer, generated by the `GenericModelOutput` annotator.

For more information on the `GenericModelOutput` buffer, see [the API documentation.](../py/docs/source/generic_model_output/generic_model_output.html).

For an example of reading data from the `GenericModelOutput` buffer from Isaac Sim, checkout the
standalone example located at `standalone_examples/api/isaacsim.sensors.rtx/inspect_lidar_metadata.py`.

### Semantic Segmentation with RTX Sensor using Object IDs

The `GenericModelOutput` struct includes a `objId` field and the `IsaacCreateRTXLidarScanBuffer` node outputs an optional `objectId` output.

In both cases, the data is provided as a `numpy` array of `dtype` `np.uint8`, and is only populated if `--/rtx-transient/stableIds/enabled=true` is set.
This data is meant to be interpreted as a sequence of 128-bit unsigned integers (effectively `stride` 16), which are stable, unique IDs corresponding to
unique prim paths in the scene. In other words, the `i`-th 128-bit unsigned integer in the array corresponds to prim generating the `i`-th return from the sensor.
This can be used for semantic segmentation of the scene, by mapping the object IDs to prim paths and then retrieving semantic labels from the prims.

The `isaacsim.sensors.rtx.LidarRtx` class provides two utility functions for resolving object IDs as prim paths.

First, `LidarRtx.decode_stable_id_mapping` resolves the output of the `StableIdMap` AOV (which can be generated from an `OmniLidar`, `OmniRadar`, or `Camera` prim)
as a Python `dict` mapping 128-bit unsigned integers to prim paths.

Second, `LidarRtx.get_object_ids` resolves the object ID array output from `GenericModelOutput` or `IsaacCreateRTXLidarScanBuffer` as 128-bit unsigned integers.

Refer to `standalone_examples/api/isaacsim.sensors.rtx/resolve_object_ids_from_gmo.py` for an example of using these functions to resolve object IDs as prim paths.

## Deprecated Annotators

Several annotators have been removed and or replaced by the annotators described above, as of Isaac Sim 5.0.

New annotator outputs are not guaranteed to be the same as the outputs of the deprecated annotators;
the table below describes affected annotators and how to replace them.

| Deprecated Isaac Sim 4.5 Annotator | Replacement | Details |
| --- | --- | --- |
| `IsaacComputeRTXLidarFlatScanSimulationTime` | `IsaacComputeRTXLidarFlatScan` | The new annotator outputs the same data as the old annotator. To get an associated timestamp, use the `IsaacReadSimulationTime` annotator. |
| `IsaacComputeRTXLidarFlatScanSystemTime` | `IsaacComputeRTXLidarFlatScan` | The new annotator outputs the same data as the old annotator. To get an associated timestamp, use the `IsaacReadSystemTime` annotator. |
| `RtxSensorCpuIsaacComputeRTXLidarPointCloud` | `IsaacExtractRTXSensorPointCloudNoAccumulator` | The new annotator outputs the same data as the old annotator, excluding `azimuth`, `elevation`, and `range`. These values can be computed from the Cartesian `data` buffer. The new annotator also automatically supports CPU or GPU output based on the `--/app/sensors/nv/lidar/outputBufferOnGPU` and `--/app/sensors/nv/radar/outputBufferOnGPU` settings, rather than Annotator type. |
| `RtxSensorGpuIsaacComputeRTXLidarPointCloud` | `IsaacExtractRTXSensorPointCloudNoAccumulator` | See above. |
| `RtxSensorCpuIsaacComputeRTXRadarPointCloud` | `IsaacExtractRTXSensorPointCloudNoAccumulator` | See above. |
| `RtxSensorGpuIsaacComputeRTXRadarPointCloud` | `IsaacExtractRTXSensorPointCloudNoAccumulator` | See above. |
| `IsaacReadRTXLidarData` | `isaacsim.sensors.rtx.read_gmo_data` utility. | See [Reading Data from the GenericModelOutput Buffer](#rtx-sensor-reading-gmo-buffer) for details. |

---

# Creating Custom RTX Sensor Profiles

Note

This section is under development. Additional content will be added in a future update.

This page covers how to create custom RTX sensor configurations by setting attributes on `OmniLidar` and `OmniRadar` prims.

## Getting Started

When creating custom RTX sensor profiles, it is recommended to start with an existing Lidar or Radar configuration shipped with Isaac Sim as a reference:

* [RTX Lidar Asset Library](Sensors.md) - Pre-configured Lidar sensors from various vendors
* [RTX Radar Sensor](Sensors.md) - RTX Radar documentation and examples

You can load an existing configuration, inspect its USD attributes in the *Property* panel, and modify them to suit your needs.

## Setting Lidar Attributes

RTX Lidar sensors are configured via USD attributes on `OmniLidar` prims using the `OmniSensorGenericLidarCoreAPI` schema.

Key configuration areas include:

* **Output configuration**: Setting coordinate systems, motion compensation, and auxiliary data detail levels
* **Scanning principle**: Configuring rotary vs. solid-state scanning
* **Firing pattern**: Defining scan rate, emitter patterns, and number of returns
* **Field of view**: Constraining azimuth and elevation ranges
* **Intensity modeling**: Configuring beam properties, detector sensitivity, and atmospheric effects

For complete documentation on all Lidar attributes and their values, see [Setting Lidar Attributes](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.lidar/latest/lidar_extension.html#setting-lidar-attributes) in the Omniverse Lidar Extension documentation.

## Setting Radar Attributes

RTX Radar sensors are configured via USD attributes on `OmniRadar` prims using the `OmniSensorGenericRadarWpmDmatAPI` schema.

For complete documentation on all Radar attributes and their values, see [Setting Radar Attributes](https://docs.omniverse.nvidia.com/kit/docs/omni.sensors.nv.radar/latest/radar_extension.html#setting-radar-attributes) in the Omniverse Radar Extension documentation.

## Schema Reference

For the full USD schema definitions, refer to:

* [OmniSensorGenericLidarCoreAPI Schema](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html#omnisensorgenericlidarcoreapi)
* [OmniSensorGenericLidarCoreEmitterStateAPI Schema](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html#omnisensorgenericlidarcoreemitterstateapi)
* [OmniSensorGenericRadarWpmDmatAPI Schema](https://docs.omniverse.nvidia.com/kit/docs/omni.usd.schema.omni_sensors/107.3.1/omni_sensors_schema.html#omnisensorgenericradarwpmdmatapi)

## Validating Your Configuration

After creating a custom sensor configuration, you can validate it by:

1. Adding the sensor to a scene using the methods described in [RTX Lidar Sensor](Sensors.md) or [RTX Radar Sensor](Sensors.md).
2. Visualizing the sensor output using the [Debug Draw Extension](Debugging_Profiling.md) or the techniques described in [Visualizing RTX Lidar Output](Sensors.md) and [Visualizing RTX Radar Output](Sensors.md).
3. Collecting data using [RTX Sensor Annotators](Sensors.md) to verify the output matches your expectations.

## Converting Legacy JSON Configurations

Isaac Sim includes a utility tool to automatically convert legacy JSON Lidar configuration files to USD files containing `OmniLidar` prims. See [Convert a JSON File to an OmniLidar USD File](Sensors.md) for details.

---

# Physics-Based Sensors

Isaac Sim’s physics-based sensors are based on CPU physics simulations and are run after the rendering is finished. They have access to a prim’s physics properties, like mass and velocity.

These sensors output the exact measurements from the physics engine and the sensor readings can be augmented in post processing.
By default, the highest rate that the sensors can output data is the physics rate and you must provide additional interpolation options to generate data beyond this rate. Furthermore, ground truth readings from the simulator might
already have some noise; additional noise can be augmented to the sensor readings in post process to make them more realistic.

The physics-based sensors are organized in the isaacsim.sensors.physics extension.

Isaac Sim supports the following physics-based ground truth sensors:

- Articulation Joint Sensors
- Contact Sensor
- Effort Sensor
- IMU Sensor
- Proximity Sensor

---

# Articulation Joint Sensors

Articulation sensors allow reading the active and passive components of the joint forces. To read articulation joint forces you can use [Articulation](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.prims.Articulation) or [ArticulationView](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.prims.ArticulationView) APIs.
See [Robot Simulation Snippets](Python_Scripting_and_Tutorials.md) for more details about the Articulation and the ArticulationView classes. Specifically,

* [get\_applied\_joint\_efforts](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.prims.ArticulationView.get_applied_joint_efforts) API will return a tensor that specifies the efforts set by the user through the [set\_joint\_efforts](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.prims.ArticulationView.set_joint_efforts).
* [get\_measured\_joint\_forces](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.prims.ArticulationView.get_measured_joint_forces) API will return a tensor that specifies 6-dimensional spatial forces per joints for all articulations (total overall joint forces). To mimic force-torque sensors, this API can be used to retrieve forces from a fixed joint.
* [get\_measured\_joint\_efforts](../py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.prims.ArticulationView.get_measured_joint_efforts) API will return a tensor which specifies the active components (the projection of the joint forces on the motion direction) of the joint forces for all the joints and articulations.

Note

In an articulation tree, each link can have a single parent link.
The joint forces reported by `get_measured_joint_forces` and `get_measured_joint_efforts` APIs correspond to the forces,
torques, or efforts exerted by the joint connecting the child link to the parent link.
In short, the forces reported by these API denote the link incoming joints forces.

## GUI

### Script Editor

This section describes how to add and customize the articulation sensor through the Script Editor, opened from **Window > Script Editor**.

```python
import asyncio

import omni
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage_async,
    get_current_stage,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics

async def joint_force():
    World.clear_instance()
    await create_new_stage_async()
    my_world = World(stage_units_in_meters=1.0, backend="torch", device="cpu")
    await my_world.initialize_simulation_context_async()
    await omni.kit.app.get_app().next_update_async()
    assets_root_path = get_assets_root_path()
    asset_path = assets_root_path + "/Isaac/Robots/IsaacSim/Ant/ant.usd"
    add_reference_to_stage(usd_path=asset_path, prim_path="/World/Ant")
    await omni.kit.app.get_app().next_update_async()
    my_world.scene.add_default_ground_plane()
    arti_view = SingleArticulation("/World/Ant/torso")
    my_world.scene.add(arti_view)
    await my_world.reset_async(soft=False)
    stage = get_current_stage()

    sensor_joint_forces = arti_view.get_measured_joint_forces()
    sensor_actuation_efforts = arti_view.get_measured_joint_efforts()
    # Iterates through the joint names in the articulation, retrieves information about the joints and their associated links,
    # and creates a mapping between joint names and their corresponding link indices.
    joint_link_id = dict()
    for joint_name in arti_view._articulation_view.joint_names:
        joint_path = "/World/Ant/joints/" + joint_name
        joint = UsdPhysics.Joint.Get(stage, joint_path)
        body_1_path = joint.GetBody1Rel().GetTargets()[0]
        body_1_name = stage.GetPrimAtPath(body_1_path).GetName()
        child_link_index = arti_view._articulation_view.get_link_index(body_1_name)
        joint_link_id[joint_name] = child_link_index

    print("joint link IDs", joint_link_id)
    print(sensor_joint_forces[joint_link_id["front_left_leg"]])
    print(sensor_actuation_efforts[joint_link_id["front_left_leg"]])

asyncio.ensure_future(joint_force())
```

---

# Contact Sensor

The Contact Sensor uses the PhysX Contact Report API to generate a sensor reading similar to what you would have with contact cells, or pressure based sensors placed on the surface of an object.
The Contact Sensor API builds on the Contact Report API by providing contact data filtered by the object it was placed in, along with an optional filter only consider contacts in a specific region of the object. For example, imagine a quadruped robot with sensors in its feet. While in the simulation the entire leg is treated as a rigid body, the only place you can measure contact are on the foot pads, so you can add a region filter that will discard any contacts outside of that boundary.
The Contact Sensor API also provides persistent contact data, even when the PhysX engine stops streaming contacts to preserve compute time. While the simulation provides full information about the contacts, such as contact pairs, normals and contact points, the Contact Sensor API was designed to match real-data obtained by single-cell contact pads. Ultimately, if full contact data is needed, the Contact Sensor API gets you the filtered contact information without any changes from what was acquired in PhysX.

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

**Contact Sensor Properties**

1. `radius` parameter specifies the distance of the contact force that it would `sensor_period`.
2. `enabled` parameter determines if the sensor is running or note.
3. `min threshold` parameter specifies the minimum amount of force to trigger a contact.
4. `max threshold` parameter specifies the maximum amount of force the sensor will output.
5. `sensor period` parameter specifies the time in between sensor measurement. A sensor period that’s lower than the physics downtime will always output the latest physics data. The sensor frequency cannot go beyond the physics frequency.

## GUI

### Creating and Modifying the Contact Sensor

Assuming there is a prim present in the scene to which you want to add a contact sensor, the following steps will let you create and modify a contact sensor.

1. To create a Physics Scene, go to the top Menu Bar and click **Create > Physics > Physics Scene**. Verify that there is now a `PhysicsScene` [Prim](Glossary.md) in the [Stage](Glossary.md) panel on the right.
2. To create a contact sensor, left click on the prim to attach the contact sensor on the stage, then go to the top Menu Bar and click **Create > Sensors > Contact\_sensor**.
3. To change the position and orientation of the contact sensor, use **Translate and Orientate** tab.
4. To change other contact sensor properties, click **Raw USD Properties** and properties such as min/max force threshold, enable/disable sensor, sensor period will be available to modify.

### Contact Sensor Example

To run the Contact Sensor Example:

1. Activate **Robotics Examples** tab from **Windows** > **Examples** > **Robotics Examples**.
2. Click **Robotics Examples** > **Sensors** > **Contact Sensor**.
3. Verify that you see a window containing the sensor’s force readings color coded by each ant’s arm.
4. Press the **Open Source Code** button to view the source code. The source code illustrates how to load an Ant body into the scene and then add sensors to it using the Python API.
5. Press the **PLAY** button to begin simulating.
6. Press `SHIFT + LEFT_CLICK` to drag the ant around and see changes in the readings.

### OmniGraph Workflow

The following is a tutorial on using OmniGraph to interact with and visualize the Contact Sensor’s readings.

#### Scene Setup

1. Add a cube to the stage by **Create > Mesh > Cube**, select the cube and drag it up. Then select the cube and right click **Add > Physics > Rigid Body with Colliders Preset**.
2. Add a physics scene by **Create > Physics > PhysicsScene**.
3. Add a ground plane by **Create> Physics > GroundPlane**.
4. Add a contact sensor by selecting the cube, and select on the top menu **Create > Sensors > Contact Sensor**.

#### OmniGraph Setup

To set up the OmniGraph to collect readings from this sensor:

1. Create the new action graph by navigating to **Window > Graph Editors > Action Graph**, and selecting New Action Graph in the new tab that opens.
2. Add the following nodes to the graph:

> * *On Playback Tick*: Executes the graph every simulation timestep.
> * *Isaac Read Contact Sensor*: Reads the contact sensor. In the **Property** tab, set Contact Sensor Prim to */World/Cube/Contact\_Sensor* to point to the location of the contact sensor prim.
> * *To String*: Converts the contact sensor readings to string format.
> * *Print Text*: Prints the string readings to console. In the **Property** tab, set Log Level to *Warning* so that messages are visible in the terminal or console by default.

1. Connect the above nodes as follows to print out the contact sensor reading:

   > 
2. Press the **Play** button on the GUI. If set up correctly, verify that the Isaac Sim internal *Console* reads out the contact sensor’s force output.

   > 

**Contact Sensor Visualization**

The Contact sensor position and radius can be visualized using the `Isaac xPrim Radius Visualizer Node`, connect the xPrim input to the Contact Sensor Prim, connect `Tick` to `Exec in`. Then insert the correct radius and configure the desired color and line thickness visualization, and the Contact sensor will be visible on **PLAY**.

Note

The spherical region only determines the boundary for contacts that will be accounted for. All contacts still only happen at the surface of the object bounded by the spherical region.

## Standalone Python

### Creating and Modifying the Contact Sensor

For the example snippets below, prepare the scene using the following snippet by adding a `PhysicsScene`, `GroundPlane`, and `DynamicCuboid` prim.
The contact sensor will be attached to the latter.

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.physics_context import PhysicsContext

PhysicsContext()
GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
DynamicCuboid(
    prim_path="/World/Cube",
    position=np.array([-0.5, -0.2, 1.0]),
    scale=np.array([0.5, 0.5, 0.5]),
    color=np.array([0.2, 0.3, 0.0]),
)
```

#### Using Python Command

Contact sensors can be created with Python using the `IsaacSensorCreateContactSensor` command, with available parameters to set, specified below, with default values. The only required parameter is the parent path.

```python
import omni.kit.commands
from pxr import Gf

success, _isaac_sensor_prim = omni.kit.commands.execute(
    "IsaacSensorCreateContactSensor",
    path="Contact_Sensor",
    parent="/World/Cube",
    sensor_period=1,
    min_threshold=0.0001,
    max_threshold=100000,
    translation=Gf.Vec3d(0, 0, 0),
)
```

#### Using Python Wrapper

The contact sensor can also be created using the `isaacsim.sensors.physics.ContactSensor` Python wrapper class. The benefit of using the wrapper class is that it comes with additional helper functions to set the contact sensor properties and retrieve sensor data.

```python
import numpy as np
from isaacsim.sensors.physics import ContactSensor

sensor = ContactSensor(
    prim_path="/World/Cube/Contact_Sensor",
    name="Contact_Sensor",
    frequency=60,
    translation=np.array([0, 0, 0]),
    min_threshold=0,
    max_threshold=10000000,
    radius=-1,
)
```

Note

Translation and position cannot both be defined, frequency, and `dt` also cannot both be defined.

Creating a contact sensor can only be done on a prim with a collider API, and it depends on a Contact Report API. Both the command and the wrapper class automatically add a Contact Report API to the parent prim. You can also manually add a Contact Report API to a prim through:

```python
import numpy as np
from isaacsim.sensors.physics import ContactSensor

sensor = ContactSensor(
    prim_path="/World/Cube/Contact_Sensor",
    name="Contact_Sensor",
    frequency=60,
    translation=np.array([0, 0, 0]),
    min_threshold=0,
    max_threshold=10000000,
    radius=-1,
)
```

To modify sensor parameters, you can use built-in class API calls such as `set_frequency`, `set_dt`, or USD attribute API calls.

### Reading Sensor Output

The contact sensors are created dynamically on **Play**. Moving the sensor prim while the simulation is running invalidates the sensor. If you need to make hierarchical changes to the contact sensor like changing its rigid body parent, stop the simulator, make the changes, and then restart the simulation.

There are also three methods for reading the sensor output:

* using `get_sensor_reading()` in the sensor interface (recommended)
* `get_current_frame()` in the contact sensor Python class
* the OmniGraph node `Isaac Read Contact Sensor`

The following snippets assume you have created a `/World/Cube` prim and contact sensor prim using one of the two snippets [above](#isaacsim-sensors-physics-contact-standalone-python-create-modify).

**get\_sensor\_reading(sensor\_path, use\_latest\_data = False)**

The get sensor reading function takes in two parameters, the `prim_path` to any contact sensor prim and it uses the latest data flag (optional) for retrieving the data point from the current physics step if the sensor is running at a slower rate than physics rate.
The function returns an `CsSensorReading` object, which contains `is_valid`, `time`, `value`, and `in_contact`.

Sample usage to get the reading from the current frame:

```python
from isaacsim.sensors.physics import _sensor

_contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
_contact_sensor_interface.get_sensor_reading("/World/Cube/Contact_Sensor", use_latest_data=True)
```

**get\_current\_frame()**

The `get_current_frame()` function is a wrapper around `get_sensor_reading(path_to_current_sensor)` function and `get_contact_sensor_raw_data`, and it is also a member function of the ContactSensor class. This function returns a dictionary with `in_contact`, `force`, `number_of_contacts`, `time`, `body0`, `body1`, `position`, `normal`, `impulse`, `contacts`, and `physics_step` as `keys` for the IMU measurement.
The `get_current_frame()` function uses the default parameters of `get_sensor_reading`, so it gives you the sensor measurement at reading time.

Sample usage:

```python
from isaacsim.sensors.physics import _sensor

_contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
_contact_sensor_interface.get_sensor_reading("/World/Cube/Contact_Sensor", use_latest_data=True)
```

**get\_contact\_sensor\_raw\_data()**

The contact sensor raw data will output a list of raw contact API data `CsRawData`, which contains `time`, `dt`, `body0`, `body1`, `position`, `normal`, and `impulse`. The raw data disregards sensor thresholds. Contacts with the parent body below the force threshold appear here even though they are discarded in the processed sensor reading `CsSensorReading`.

```python
from isaacsim.sensors.physics import _sensor

_contact_sensor_interface = _sensor.acquire_contact_sensor_interface()
_contact_sensor_interface.get_sensor_reading("/World/Cube/Contact_Sensor", use_latest_data=True)
```

Warning

This function is deprecated and will be replaced in a future release.

### API Documentation

See the [API Documentation](../py/source/extensions/isaacsim.sensors.physics/docs/index.html) for complete usage information.

---

# Effort Sensor

The effort sensor in Isaac Sim tracks the torque or force applied to individual joints. Torque is measured for revolute joints and magnitude of force is measured for linear joints.

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

## GUI

### Scene Setup

Begin by adding a Simple Articulation to the scene, which can be accessed in the Content Browser.

1. In the *Content Browser*, search for `simple_articulation` or navigate to `Isaac Sim/Robots/IsaacSim/SimpleArticulation/simple_articulation.usd`.
2. Drag `simple_articulation` onto the *World* prim in the **Stage** UI window on the right hand side to add an instance into the environment.
3. To drive the revolute joint, in the **Stage** window, select the RevoluteJoint prim at */World/simple\_articulation/Arm/RevoluteJoint*, and scroll down to **Drive** in the **Property** window. Set the target velocity to `90 deg/s`, and stiffness to `0`.

### Creating and Modifying the Effort Sensor

The following section describes how to create the effort sensor using the **Script Editor**, opened from **Window > Script Editor**.
The effort sensor can be created using the `isaacsim.sensors.physics.EffortSensor` Python wrapper class. The benefit of using the wrapper class is that it comes with additional helper functions to set the effort sensor properties and retrieve sensor data.

```python
import numpy as np
from isaacsim.sensors.physics import EffortSensor

sensor = EffortSensor(
    prim_path="/World/simple_articulation/Arm/RevoluteJoint", sensor_period=0.1, use_latest_data=False, enabled=True
)
```

To modify sensor parameters, you can change class member variables like `sensor_period`, `use_latest_data`, and `enabled` directly, and for changing the `dof_name` and `buffer_size` for the readings, use the corresponding member functions `update_dof_name` and `change_buffer_size`.

### Reading Sensor Output with Python

**get\_sensor\_reading(self, interpolation\_function = None, use\_latest\_data = False)**

The get sensor reading function takes in two parameters:

* an interpolation function (optional) to use in place of the default linear interpolation function
* a use latest data flag (optional) for retrieving the data point from the current physics step, if the sensor is running at a slower rate than physics rate

The function will return an `EsSensorReading` object which contains `is_valid`, `time`, and `value`.

After you created the effort sensor, press **PLAY** to start the simulation and call the function below to get the sensor reading for the current frame:

```python
from isaacsim.sensors.physics import EffortSensor

# get sensor reading
reading = sensor.get_sensor_reading(use_latest_data=True)
```

Sample usage with custom interpolation function:

```python
from isaacsim.sensors.physics import EffortSensor

# Input Param: List of past EsSensorReading, time of the expected sensor reading
def interpolation_function(data, time):
    interpolated_reading = EsSensorReading()
    # do interpolation
    return interpolated_reading

# get sensor readings
reading = sensor.get_sensor_reading(interpolation_function)
```

### OmniGraph Workflow

To set up the OmniGraph to create the effort sensor and collect readings from it.

1. Create the new action graph by navigating to **Window > Graph Editors > Action Graph**, and selecting **New Action Graph** in the new tab that opens.
2. Add the following nodes to the graph:

   > * **On Playback Tick**: Executes the graph nodes every simulation timestep.
   > * **Isaac Read Effort Node**: Reads the effort sensor. In the **Property** tab, set Effort Prim to the exact joint of measurement. For example */World/simple\_articulation/Arm/RevoluteJoint* in `simple_articulation.usd`.
   > * **To String**: Converts the effort sensor readings to string format.
   > * **Print Text**: Prints the string readings to console. In the **Property** tab, set Log Level to *Warning* so that messages are visible in the terminal/console by default. Additionally, check *To Screen* to print directly to screen.

Connect the above nodes as follows to print out the effort sensor reading:

Note

Configure the joints to the correct axis to get the expected readings.

## API Documentation

See the [API Documentation](../py/source/extensions/isaacsim.sensors.physics/docs/index.html) for complete usage information.

---

# IMU Sensor

The IMU sensor in Isaac Sim tracks the motion of the body and outputs simulated accelerometer and gyroscope readings.
Like real IMU sensors, simulated IMUs gives acceleration and angular velocity measurements in local `x, y, z` axis with stage units.

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

**IMU Sensor Properties**

1. `enabled` parameter determines if the sensor is running or note.
2. `sensor period` parameter specifies the time in between sensor measurement. A sensor period that’s lower than the physics delta time will always output the latest physics data. The sensor frequency cannot go beyond the physics frequency.
3. `angularVelocityFilterWidth` parameter species the size of the angular velocity rolling average. Increasing this parameter will result in smoother angular velocity output.
4. `linearAccelerationFilterWidth` parameter species the size of the linear acceleration rolling average. Increasing this parameter will result in smoother linear acceleration output.
5. `orientationFilterWidth` parameter species the size of the orientation rolling average. Increasing this parameter will result in smoother orientation output.

The size of the data buffer used in interpolation is two times the max of the filter width or 20, whichever is greater.

## GUI

### Creating and Modifying the IMU

Assuming there is a prim present in the scene to which you want to add an IMU sensor, the following steps will let you create and modify an IMU sensor:

1. To create a Physics Scene, go to the top Menu Bar and click **Create > Physics > Physics Scene**. Verify that you have a `PhysicsScene` [Prim](Glossary.md) in the [Stage](Glossary.md) panel on the right.
2. To create an IMU, left click on the prim to attach the IMU on the stage, then go to the top Menu Bar and click **Create > Sensors > Imu Sensor**.
3. To change the position and orientation of the IMU, left click on the `Imu_Sensor` prim, then modify the **Transform** properties under the **Property** tab.
4. To change other IMU properties, expand the **Raw USD Properties** section, and properties such as filter width, enable/disable sensor, and sensor period will be available to modify.

### IMU Example

To run the IMU example:

1. Activate **Robotics Examples** tab from **Windows** > **Examples** > **Robotics Examples**.
2. Click **Robotics Examples** > **Sensors** > **IMU Sensor** > **Load Scene**.
3. Verify that you have a window containing each axis of the accelerometer and gyro readings being displayed.
4. Press the **Open Source Code** button to view the source code. The source code illustrates how to load an Ant body into the scene and then add the sensor to it using the Python API.
5. Press the **PLAY** button to begin simulating.
6. Press `SHIFT + LEFT_CLICK` over the ant to drag it around and see changes in the readings.

### OmniGraph Workflow

The following is a tutorial on using OmniGraph to interact with the IMU Sensor.

#### Scene Setup

Begin by adding a Simple Articulation to the scene. The articulation file can be accessed through a [Omniverse Nucleus](Glossary.md) server in the content window.
Connecting to this server allows allows you to access the library of Isaac Sim robots, sensors, and environments.

After connecting to the server:

1. Navigate to `Robots/IsaacSim/SimpleArticulation/simple_articulation.usd` in the **Content Browser**.
2. Drag `simple_articulation` onto the *World* prim in the **Stage** UI window on the right hand side to add an instance into the environment.
3. To drive the revolute joint, in the **Stage** window, select the RevoluteJoint prim at */World/simple\_articulation/Arm/RevoluteJoint*, and scroll down to **Drive** in the **Property** window. Set the target velocity to `90 deg/s` and stiffness to `0`.

To add an IMU sensor to your robot and collect some data:

1. In the **Stage** tab, navigate to the */World/simple\_articulation/Arm* prim and select it.
2. Add the sensor to the prim by **Create > Sensors > Imu Sensor**.
3. The newly added IMU sensor can be viewed by hitting the **+** button next to the Arm prim.

Note

In general, sensors must be added to rigid body prims to correctly report data. The prims in this robot are already rigid bodies, so nothing must be done for this case.

#### OmniGraph Setup

To set up the OmniGraph to collect readings from this sensor:

1. Create the new action graph by navigating to **Window > Graph Editors > Action Graph**, and selecting **New Action Graph** in the new tab that opens.
2. Add the following nodes to the graph, and set their properties as follows:

> * **On Playback Tick**: Executes the graph nodes every simulation timestep.
> * **Isaac Read IMU Node**: Reads the IMU sensor. In the **Property** tab, set IMU Prim to */World/simple\_articulation/Arm/Imu\_Sensor*, to point to the location of the IMU sensor prim. Select **read gravity** to read gravitational acceleration.
> * **To String**: Converts the IMU readings to string format.
> * **Print Text**: Prints the string readings to console. In the **Property** tab, set **Log Level** to **Warning** so that messages are visible in the terminal/console by default.

1. Connect the above nodes as follows to print out the IMU sensor reading:

   > 
2. Press the **Play** button on the GUI. If set up correctly, verify that the Isaac Sim internal *Console* reads out the IMU sensor’s angular velocity.

   > 

## Standalone Python

### Creating and Modifying the IMU

There are two ways to create an IMU Sensor in Python:

* using a command
* using the wrapper class

This section provides snippets to be executed using standalone Python; these snippets are intended as a references, and must be modified to suit your purposes. The following snippet adds a ground plane, cube prim, and physics scene to an Isaac Sim scene, which are required for the reference snippets further below to work correctly.

```python
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.objects.ground_plane import GroundPlane
from isaacsim.core.api.physics_context import PhysicsContext

PhysicsContext()
GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
DynamicCuboid(
    prim_path="/World/Cube",
    position=np.array([-0.5, -0.2, 1.0]),
    scale=np.array([0.5, 0.5, 0.5]),
    color=np.array([0.2, 0.3, 0.0]),
)
```

#### Using Python Command

You can add an IMU to the cube prim created above using the `IsaacSensorCreateImuSensor` command, as demonstrated in the following snippet. The only required argument is the parent path; the remaining arguments are optional.

```python
import omni.kit.commands
from pxr import Gf

success, _isaac_sensor_prim = omni.kit.commands.execute(
    "IsaacSensorCreateImuSensor",
    path="imu_sensor",
    parent="/World/Cube",
    sensor_period=1,
    linear_acceleration_filter_size=10,
    angular_velocity_filter_size=10,
    orientation_filter_size=10,
    translation=Gf.Vec3d(0, 0, 0),
    orientation=Gf.Quatd(1, 0, 0, 0),
)
```

#### Using Python Wrapper

You can add an IMU to the cube prim, created above, using the `isaacsim.sensors.physics.IMUSensor` Python wrapper class, as demonstrated in the following snippet. The benefit of using the wrapper class over the command is that it comes with additional helper functions to set the IMU sensor properties and retrieve sensor data.

```python
import numpy as np
from isaacsim.sensors.physics import IMUSensor

IMUSensor(
    prim_path="/World/Cube/Imu",
    name="imu",
    frequency=60,  # or, dt=1./60
    translation=np.array([0, 0, 0]),  # or, position=np.array([0, 0, 0]),
    orientation=np.array([1, 0, 0, 0]),
    linear_acceleration_filter_size=10,
    angular_velocity_filter_size=10,
    orientation_filter_size=10,
)
```

Note

`translation` and `position` cannot both be provided as input arguments. `frequency` and `dt` also cannot both be provided as input arguments.
The `IMUSensor` Python API documentation specifies the usage of each input argument.

To modify sensor parameters, you can use built-in class API calls such as `set_frequency`, `set_dt`, or USD attribute API calls.

### Reading Sensor Output

The sensors are created dynamically on PLAY. Moving the sensor prim while the simulation is running will invalidate the sensor. If you need to make hierarchical changes to the IMU like changing its rigid body parent, stop the simulator, make the changes, and then restart the simulation.

There are also three methods for reading the sensor output:

* using `get_sensor_reading()` in the sensor interface
* `get_current_frame()` in the IMU Python class
* OmniGraph node `Isaac Read IMU Node`

The following snippets assume you have created a `/World/Cube` prim and IMU sensor prim using one of the two snippets [above](#isaacsim-sensors-physics-imu-standalone-python-create-modify).

**get\_sensor\_reading(sensor\_path, interpolation\_function = None, use\_latest\_data = False, read\_gravity = True)**

The `get_sensor_reading` function takes in three parameters:

* the `prim_path` to any IMU sensor prim
* an interpolation function (optional) to use in place of the default linear interpolation function
* the `useLatestValue` flag (optional) for retrieving the data point from the current physics step if the sensor is running at a slower rate than physics rate

The function will return an `IsSensorReading` object, which has `is_valid`, `time`, `lin_acc_x`, `lin_acc_y`, `lin_acc_z`, `ang_vel_x`, `ang_vel_y`, `ang_vel_z`, and `orientation` properties.

Sample usage to get the reading from the current physics step with gravitational effects:

```python
from isaacsim.sensors.physics import _sensor

_imu_sensor_interface = _sensor.acquire_imu_sensor_interface()
_imu_sensor_interface.get_sensor_reading("/World/Cube/Imu", use_latest_data=True, read_gravity=True)
```

Sample usage with custom interpolation function without gravitational effects:

```python
from isaacsim.sensors.physics import _sensor

_imu_sensor_interface = _sensor.acquire_imu_sensor_interface()
_imu_sensor_interface.get_sensor_reading("/World/Cube/Imu", use_latest_data=True, read_gravity=True)
```

Note

When custom interpolation is used and the read gravity flag is enabled, the sensor will pass raw acceleration measurements to the custom interpolation function and apply gravitational transforms after.

**get\_current\_frame(read\_gravity = True)**

The `get_current_frame()` function is a wrapper around `get_sensor_reading(path_to_current_sensor)` function and a member function of the IMU class. This function returns a dictionary with `lin_acc`, `ang_vel`, `orientation`, `time`, and `physics_step` as `keys` for the Contact measurement.
The `get_current_frame()` function uses the default parameters of `get_sensor_reading`, so it utilizes linear interpolation and last sensor reading at reading time.

Sample usage:

```python
import numpy as np
from isaacsim.sensors.physics import IMUSensor

sensor = IMUSensor(
    prim_path="/World/Cube/Imu",
    name="imu",
    frequency=60,
    translation=np.array([0, 0, 0]),
    orientation=np.array([1, 0, 0, 0]),
    linear_acceleration_filter_size=10,
    angular_velocity_filter_size=10,
    orientation_filter_size=10,
)

value = sensor.get_current_frame()
print(value)
```

### API Documentation

See the [API Documentation](../py/source/extensions/isaacsim.sensors.physics/docs/index.html) for complete usage information.

---

# Proximity Sensor

The Proximity Sensor is a wrapper around a physics callback that can be attached to any prim in the scene. During simulation execution,
the sensor will record collisions between the prim it’s attached to and other prims in the scene each frame; that data can be accessed
using a callback function.

## Standalone Python

Execute the following script using `python.sh`. This will create a scene with two cubes, attaching a proximity sensor to one of the cubes.
At the start of the simulation, the two cubes will overlap and then move apart; the callback function in the script will print the proximity
sensor’s output to the screen.

```python
import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
import omni
from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
from isaacsim.core.api.world import World
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import Sdf, UsdLux

# Set up scene
world = World()
ground_plane = GroundPlane("/World/GroundPlane")

# Add lighting
stage = omni.usd.get_context().get_stage()
distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
distantLight.CreateIntensityAttr(500)

# Add cubes
cube_1 = DynamicCuboid(
    prim_path="/cube_1",
    name="cube_1",
    position=np.array([0.4, 0, 5.0]),
    scale=np.array([1, 1, 1]),
    size=1.0,
    color=np.array([255, 0, 0]),
)

cube_2 = DynamicCuboid(
    prim_path="/cube_2",
    name="cube_2",
    position=np.array([-0.4, 0, 5.0]),
    scale=np.array([1, 1, 1]),
    size=1.0,
    color=np.array([0, 0, 255]),
)

# Enable isaacsim.sensors.physx extension
enable_extension("isaacsim.sensors.physx")
simulation_app.update()

# Attach sensor to cube 1
from isaacsim.sensors.physx import ProximitySensor, clear_sensors, register_sensor

s = ProximitySensor(cube_1.prim)
register_sensor(s)

# Add callback to print proximity sensor data
def print_proximity_sensor_data_on_update(_):
    data = s.get_data()
    if "/cube_2" in data:
        # /cube_1 is colliding with /cube_2
        distance = data["/cube_2"]["distance"]
        duration = data["/cube_2"]["duration"]
        carb.log_warn(f"distance: {distance}, duration: {duration}")

# Play simulation
world.add_physics_callback("print_sensor_data", print_proximity_sensor_data_on_update)
simulation_app.update()
simulation_app.update()
world.play()

for i in range(100):
    # Run with a fixed step size
    world.step(render=True)
```

Example proximity sensor output is shown below; there might be small numerical differences in your output run-to-run.

```python
distance: 0.8995118804137266, duration: 0.03952527046203613
distance: 0.9490971672498862, duration: 0.04244112968444824
distance: 0.9978315307718298, duration: 0.045195579528808594
distance: 1.0952793930211249, duration: 0.00010466575622558594
distance: 1.0952880909233123, duration: 0.004382610321044922
distance: 1.0952874949586842, duration: 0.008539199829101562
distance: 1.095288806188406, duration: 0.012722015380859375
```

After the cubes land, the scene will look like below:

---

# PhysX SDK Sensors

Isaac Sim’s PhysX SDK sensors use raycasts provided by [PhysX SDK](https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/) to measure the range between
objects in the simulation.

These sensors will output the exact measurements from PhysX SDK. By default, the highest rate that the sensors can output data is the render rate.

The PhysX SDK sensors are organized in the isaacsim.sensors.physx extension.

Isaac Sim supports the following PhysX SDK sensors:

- PhysX SDK Generic Sensor
- PhysX SDK Lidar
- PhysX SDK Lightbeam Sensor

---

# PhysX SDK Generic Sensor

The PhysX SDK generic sensor in Isaac Sim uses PhysX SDK raycasts to measure depth between two prims. It demonstrates
how to build a PhysX SDK-based sensor in Isaac Sim to measure ground truth depth.

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

## GUI

### PhysX SDK Generic Sensor Example

To run the PhysX SDK generic sensor example:

1. Activate **Robotics Examples** tab from **Windows** > **Examples** > **Robotics Examples**.
2. Click **Robotics Examples** > **Sensors** > **Custom Pattern Range Sensor**.
3. Press the **Load Sensor** button.
4. Press the **Load Scene** button.
5. Press the **Set Sensor Pattern** button to load the example sensor pattern.
6. Press the **Open Source Code** button to view the source code. The source code illustrates how to create, add, and control the sensor using the Python API.
7. Press the **PLAY** button to begin simulating.

1. To visualize the pattern, you can save the image imprinted on the wall from the rays that hit it. To do so, select or type out the desired output directory and press **Save Pattern Image**. Open the saved image file, verify that you have a zigzag pattern.

### Script Editor

The following sections describe how to customize the PhysX SDK generic sensor through the **Script Editor**, opened from **Window > Script Editor**.

**Customizing Scanning Pattern**

To customize scanning patterns, these are the parameters that need to be filled or modified:

* **streaming:** Set to `True` if streaming data continuously, `False` if sending a batch of data once in the beginning and repeating it.
* **sampling\_rate:** Number of scans per second.
* **batch\_size:** The number of scans each batch of data contains. The size needs be large enough to run a few rendering frames without running out. For example, if you want to scan at a sampling rate of 2400 scans per second, and your frame rendering rate is at 120 fps, then each frame will render 20 scans. If you send a batch size 12000, you must be able to render 600 frames or five seconds at 120 fps before you run out of data. If batch\_size is less than what is needed to satisfy the desired sampling rate (that is, `batch_size` less than `sampling_rate/fps`), then the sensor will scan at a rate that equals the `batch_size` per frame, which likely means you will be scanning slower than desired.
* **sensor\_pattern:** a Nx2 size NumPy array. N is `batch_size`, and the columns are [azimuth, zenith] angles of each scanning ray. Azimuth is the ray’s horizontal angle measured from the x-axis, and zenith angle is the vertical angle measured from the z-axis.
* **origin\_offsets:** (Optional) an Nx3 size NumPy array, N is the batch size, and each row is the individual ray’s offset from origin in [x,y,z] coordinates.

**Example Scanning Patterns**

Let’s take a closer look at our example code to see how to produce the zigzag scanning pattern.
The pattern in the example is generated programmatically inside the same script that runs the example. Click on the **Open Source Code** icon in the upper right-hand corner of the example window and open the Python source code for this example.

There are two test patterns in the script, one for testing continuous streaming data mode, the other one for testing a repeating pattern mode.

**Streaming Generated Pattern**

The pattern is sweeping horizontally 10 times for each round of up and down, resulting in the zigzag.

```python
def _test_streaming_data(self):
    # custom pattern generation
    # send data in a batch that are at least large enough to run a few rendering frames without running out of data.
    # if batch_size > (sampling rate/rendering rate), the sensor will process all of the batches and ask for the next batch right before it runs out.
    # if batch_size < (sampling rate/rendering_rate), the sensor will scan only the provided rays in a given frame, which means it will be scanning slower than intended
    batch_size = int(1e6)  # size of each batch of data being processed
    half_batch = int(batch_size / 2)
    # example scanning pattern is a zigzag
    # each ray specified by an azimuth (horizontal angle measured from x-axis) and a zenith angle (vertical angle measured from z-axis)
    frequency = 10
    N_pts = int(batch_size / frequency / 2)
    # azimuth angle zigzag between the limits (frequency) times every batch
    azimuth = np.tile(
        np.append(np.linspace(-np.pi / 4, np.pi / 4, N_pts), np.linspace(np.pi / 4, -np.pi / 4, N_pts)), frequency
    )
    # zenith angle goes up and down once every batch
    zenith = np.append(np.linspace(-np.pi / 4, np.pi / 4, half_batch), np.linspace(np.pi / 4, -np.pi / 4, half_batch))
    # custom pattern must be sent as an array of [azimuth, zenith] angles.
    self.sensor_pattern = np.stack((azimuth, zenith))
```

Origin offset is optional. For the example, a small random offset was added, as seen below. For no offsets, you can either use an array of zeros or skip setting the `origin_offsets` parameter.

```python
import numpy as np

# individual rays can have an offset at the origin
# adding random offsets to the origin for the example pattern
self.origin_offsets = 5 * np.random.random((batch_size, 3))
# self.origin_offsets = np.zeros((batch_size,3))                  # no offsets
```

**Streaming Pattern Through File**

If you do not have a programmatic way to generate the scanning pattern from scratch, or if you do not want to disclose the generation method of the scanning pattern, you can also import data from the file. The example below shows importing data from a `.csv` file and converting it to match the format of the **sensor\_pattern** parameter.

```python
import numpy as np

## import data from file
sensor_pattern = np.loadtxt("filename.csv", delimiter=",")
batch_size = np.shape(sensor_pattern)[0]
sensor_pattern = np.deg2rad(sensor_pattern).T.copy()  ##  MUST USE .copy()
```

**Repeating Pattern**

To better visualize the repetitiveness of the pattern, you use a zigzag motion, but this time instead a smooth movement going up and down, it is split into two modes, one set scanning high and the other set scanning low. If correctly executed, verify that it repeats itself without any additional data being pulled in.

To change the example to run in non-streaming mode, set variable `self._streaming = False` and save the change. Verify that it then automatically use the following code the generate the pattern. Wait for the example to restart and reload before trying to run it.

```python
def _test_repeating_data(self):

    batch_size = int(1e6)  # size of each batch of data being processed
    half_batch = int(batch_size / 2)
    frequency = 10
    N_pts = int(batch_size / frequency / 2)
    azimuth = np.tile(
        np.append(np.linspace(-np.pi / 4, np.pi / 4, N_pts), np.linspace(np.pi / 4, -np.pi / 4, N_pts)), frequency
    )
    zenith = np.append(-0.5 * np.ones(half_batch), 0.5 * np.ones(half_batch))
    sensor_pattern = np.stack((azimuth, zenith))

    origin_offsets = 0.05 * np.random.random((batch_size, 3))
```

**Setting Scanning Pattern**

When the sensor processes each batch of `[azimuth, zenith]` pairs, just before it is about to run out of data, it will set the variable `send_next_batch()` to `True`, at which point, you can send the next batch through `set_next_batch_rays(prim_path, sensor_pattern)`, plus `set_next_batch_offsets(prim_path, sensor_pattern)` if there are any origin offsets. Like shown below.

```python
def _test_repeating_data(self):

    batch_size = int(1e6)  # size of each batch of data being processed
    half_batch = int(batch_size / 2)
    frequency = 10
    N_pts = int(batch_size / frequency / 2)
    azimuth = np.tile(
        np.append(np.linspace(-np.pi / 4, np.pi / 4, N_pts), np.linspace(np.pi / 4, -np.pi / 4, N_pts)), frequency
    )
    zenith = np.append(-0.5 * np.ones(half_batch), 0.5 * np.ones(half_batch))
    sensor_pattern = np.stack((azimuth, zenith))

    origin_offsets = 0.05 * np.random.random((batch_size, 3))
```

---

# PhysX SDK Lidar

The PhysX SDK Lidar sensor in Isaac Sim uses PhysX SDK raycasts to simulate a Lidar.
You can set horizontal and vertical beam resolution, rotation rate, and other Lidar parameters; the
PhysX SDK Lidar will then report depth information from each beam. The PhysX SDK Lidar cannot interact with
non-visual materials, it will always report ground truth information. For example, the Lidar will measure depth
of a transparent object with respect to the Lidar, even if a beam would normally pass through the transparent
object in real life.

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

## GUI

### PhysX SDK Lidar Sensor Example

To run the example:

1. Activate `Robotics Examples` tab from **Windows** > **Examples** > **Robotics Examples**.
2. Click **Robotics Examples** > **Sensors** > **Physx Lidar Sensor**.
3. Press the **Load Sensor** button.
4. Press the **Load Scene** button.
5. Press the **Open Source Code** button to view the source code. The source code illustrates how to add and control the sensor using the Python API.
6. Press the **PLAY** button to begin simulating.

### Adding PhysX SDK Lidar Sensor to Simulation

#### Scene Setup

Let’s begin setting up the scene by creating a `PhysicsScene` and a `PhysX Lidar` in the environment:

1. To create a Physics Scene, go to the top Menu Bar and click **Create > Physics > Physics Scene**. Verify that there is now be a `PhysicsScene` [Prim](Glossary.md) in the [Stage](Glossary.md) panel on the right.
2. To create a LIDAR, go to the top Menu Bar and click **Create > Sensors > PhysX Lidar > Rotating**.
   Next, let’s set some of the LIDAR properties for rotation and visualization:
3. Select the newly created LIDAR prim from the [Stage](Glossary.md) panel.
4. After selected, the **Property** panel to the bottom left will populate with all the available properties of the LIDAR.
5. Scroll down in the **Property** panel to the **Raw USD Properties** section.
6. Enable the **drawLines** checkbox to enable line rendering.
7. Set the revolutions per second to `1 Hz` by setting `rotationRate` to `1.0`.

   * To fire LIDAR rays in all directions at once, set the `rotationRate` to `0.0`.

Note

You can update all of the Lidar parameters on the fly while the stage is running.
When the rotation rate reaches zero or less, the Lidar prim will cast rays in all directions based on your FOV and resolution.

#### Setup Collision Detection

The LIDAR can only detect objects with **Collisions Enabled**. Let’s add an object for the LIDAR to detect:

1. Go to the top Menu Bar and click **Create > Mesh > Cube**.
2. Translate the cube to `(2, 0, 0)`.

Next, add a Physics Collider to the Cube:

10. With the Cube selected, go to the **Property** panel and click the **+ Add** button.
11. Select **+ Add > Physics > Collider**.

* Use the mouse and move the Cube around the scene to see how the LIDAR rays interact with the geometry.

#### Attach a LIDAR to Geometry

For most use cases, LIDARs will be attached to other more complex assemblies — such as cars or robots.
Let’s learn how to attach a LIDAR to a parent geometry.
We are going to use a Cylinder as a placeholder for a more complex prim.
Add a Cylinder to the scene and nest the LIDAR prim under it:

1. Right click in the viewport and select **Create > Mesh > Cylinder**.
2. Set the translation of the Cylinder to `(0, 0, 0)`.
3. In the [Stage](Glossary.md) panel, drag-and-drop the `LIDAR` prim onto the `Cylinder`.
4. This makes the `Cylinder` the parent of the `LIDAR`. Now when the `Cylinder` moves, the `LIDAR` moves with it. Moreover, all information reported by the LIDAR is now relative to the `Cylinder`.
5. Add a offset to `LIDAR` to precisely position it relative to the `Cylinder`. Select the `LIDAR` prim from the [Stage](Glossary.md) and move it to `(0.5, 0.5, 0)`.
6. Now move the `Cylinder` around the environment. The LIDAR maintains this relative transform.
7. Re-select the `LIDAR` prim and reset its Translate value to its default setting `(0, 0, 0)`.

#### Attach a LIDAR to a Moving Robot

You can attach a LIDAR prim to a robot. You can use the Carter V1 robot as an example.

1. Open the Isaac Sim **Content Browser**, navigate to `Robots/NVIDIA/Carter/carter_v1.usd`, and open the `carter_v1.usd` file.
2. Open the left wheel joint at carter/chassis\_link/left\_wheel, scroll down on the property panel, and set the Target Velocity to 100.
3. Repeat the same process for the right wheel joint at carter/chassis\_link/right\_wheel.
4. Press **play** and the Carter robot should drive forward automatically.
5. Create a `LIDAR`, go to the top Menu Bar and click **Create > Sensors > PhysX LIDAR > Rotating**. The `LIDAR` prim will be created as a child of the selected prim.
6. In the [Stage](Glossary.md) panel, select your `LIDAR` prim and drag it onto `/carter/chassis_link`.
7. Set the translation of the PhysX lidar to -0.06, 0.0, 0.38 to move it to the correct location.
8. Enable draw lines and set the rotation rate to zero for easier debugging.

### Script Editor

The LIDAR Python API is used to interact programmatically with a LIDAR through scripts and extensions.
It can be used to create, control, and query the sensor through scripts and extensions.
Let’s use the **Script Editor** and Python API to retrieve the data from the LIDAR’s last sweep:

1. Go to the top menu bar and click **Window > Script Editor** to open the **Script Editor** window.
2. Add the necessary imports:

```python
import asyncio  # Used to run sample asynchronously to not block rendering thread

import omni  # Provides the core omniverse APIs
from isaacsim.sensors.physx import _range_sensor  # Imports the python bindings to interact with Lidar sensor
from pxr import Gf, UsdGeom, UsdPhysics  # pxr usd imports used to create the cube
```

3. Grab the Stage, Simulation Timeline, and LIDAR interface:

```python
import asyncio  # Used to run sample asynchronously to not block rendering thread

import omni  # Provides the core omniverse APIs
from isaacsim.sensors.physx import _range_sensor  # Imports the python bindings to interact with Lidar sensor
from pxr import Gf, UsdGeom, UsdPhysics  # pxr usd imports used to create the cube
```

4. Create an obstacle for the LIDAR:

```python
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics

stage = get_current_stage()
CubePath = "/World/CubeName"  # Create a Cube
cubeGeom = UsdGeom.Cube.Define(stage, CubePath)
cubePrim = stage.GetPrimAtPath(CubePath)
cubeGeom.AddTranslateOp().Set(Gf.Vec3f(2.0, 0.0, 0.0))  # Move it away from the LIDAR
cubeGeom.CreateSizeAttr(1)  # Scale it appropriately
collisionAPI = UsdPhysics.CollisionAPI.Apply(cubePrim)  # Add a Physics Collider to it
```

5. Get the LIDAR data:

   > The Lidar needs a frame of simulation to get data for the first frame, so start
   > the simulation by calling `timeline.play` and waiting for a frame to complete, and then pause simulation using `timeline.pause()` to populate the depth buffers in the Lidar.
   > Because the simulation is running asynchronously with our script, use `asyncio` and `ensure_future` to wait for our script to complete
   > calling `timeline.pause()` is optional, data from the sensor can be gathered anytime while simulating.
   >
   > ```python
   > from isaacsim.core.utils.stage import get_current_stage
   > from pxr import Gf, UsdGeom, UsdPhysics
   >
   > stage = get_current_stage()
   > CubePath = "/World/CubeName"  # Create a Cube
   > cubeGeom = UsdGeom.Cube.Define(stage, CubePath)
   > cubePrim = stage.GetPrimAtPath(CubePath)
   > cubeGeom.AddTranslateOp().Set(Gf.Vec3f(2.0, 0.0, 0.0))  # Move it away from the LIDAR
   > cubeGeom.CreateSizeAttr(1)  # Scale it appropriately
   > collisionAPI = UsdPhysics.CollisionAPI.Apply(cubePrim)  # Add a Physics Collider to it
   > ```
6. Run the full script:

Expand to display full code

```python
# provides the core omniverse APIs
# used to run sample asynchronously to not block rendering thread
import asyncio

import omni

# import the python bindings to interact with Lidar sensor
from isaacsim.sensors.physx import _range_sensor

# pxr usd imports used to create cube
from pxr import Gf, UsdGeom, UsdPhysics

stage = omni.usd.get_context().get_stage()
lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
timeline = omni.timeline.get_timeline_interface()
omni.kit.commands.execute("AddPhysicsSceneCommand", stage=stage, path="/World/PhysicsScene")
lidarPath = "/LidarName"
result, prim = omni.kit.commands.execute(
    "RangeSensorCreateLidar",
    path=lidarPath,
    parent="/World",
    min_range=0.4,
    max_range=100.0,
    draw_points=False,
    draw_lines=True,
    horizontal_fov=360.0,
    vertical_fov=30.0,
    horizontal_resolution=0.4,
    vertical_resolution=4.0,
    rotation_rate=0.0,
    high_lod=False,
    yaw_offset=0.0,
    enable_semantics=False,
)

CubePath = "/World/CubeName"
cubeGeom = UsdGeom.Cube.Define(stage, CubePath)
cubePrim = stage.GetPrimAtPath(CubePath)
cubeGeom.AddTranslateOp().Set(Gf.Vec3f(2.0, 0.0, 0.0))
cubeGeom.CreateSizeAttr(1)
collisionAPI = UsdPhysics.CollisionAPI.Apply(cubePrim)

async def get_lidar_param():
    await omni.kit.app.get_app().next_update_async()
    timeline.pause()
    depth = lidarInterface.get_linear_depth_data("/World" + lidarPath)
    zenith = lidarInterface.get_zenith_data("/World" + lidarPath)
    azimuth = lidarInterface.get_azimuth_data("/World" + lidarPath)
    print("depth", depth)
    print("zenith", zenith)
    print("azimuth", azimuth)

timeline.play()
asyncio.ensure_future(get_lidar_param())
```

Verify that you have the following:

#### Segment a Point Cloud

This code snippet shows how to add semantic labels to the depth data for segmenting its resulting point cloud.

```python
import asyncio  # Used to run sample asynchronously to not block rendering thread

import omni  # Provides the core omniverse APIs
from isaacsim.sensors.physx import _range_sensor  # Imports the python bindings to interact with Lidar sensor
from pxr import Gf, Semantics, UsdGeom, UsdPhysics  # pxr usd imports used to create cube

stage = omni.usd.get_context().get_stage()  # Used to access Geometry
timeline = omni.timeline.get_timeline_interface()  # Used to interact with simulation
lidarInterface = _range_sensor.acquire_lidar_sensor_interface()  # Used to interact with the LIDAR
# These commands are the Python-equivalent of the first half of this tutorial
omni.kit.commands.execute("AddPhysicsSceneCommand", stage=stage, path="/World/PhysicsScene")
lidarPath = "/LidarName"
# Create Lidar prim
result, prim = omni.kit.commands.execute(
    "RangeSensorCreateLidar",
    path=lidarPath,
    parent="/World",
    min_range=0.4,
    max_range=100.0,
    draw_points=True,
    draw_lines=False,
    horizontal_fov=360.0,
    vertical_fov=60.0,
    horizontal_resolution=0.4,
    vertical_resolution=0.4,
    rotation_rate=0.0,
    high_lod=True,
    yaw_offset=0.0,
    enable_semantics=True,
)
UsdGeom.XformCommonAPI(prim).SetTranslate((2.0, 0.0, 0.0))

# Create a cube, sphere, add collision and different semantic labels
primType = ["Cube", "Sphere"]
for i in range(2):
    prim = stage.DefinePrim("/World/" + primType[i], primType[i])
    UsdGeom.XformCommonAPI(prim).SetTranslate((-2.0, -2.0 + i * 4.0, 0.0))
    UsdGeom.XformCommonAPI(prim).SetScale((1, 1, 1))
    collisionAPI = UsdPhysics.CollisionAPI.Apply(prim)

    # Add semantic label
    sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
    sem.CreateSemanticTypeAttr()
    sem.CreateSemanticDataAttr()
    sem.GetSemanticTypeAttr().Set("class")
    sem.GetSemanticDataAttr().Set(primType[i])

# Get point cloud and semantic id for Lidar hit points
async def get_lidar_param():
    await asyncio.sleep(1.0)
    timeline.pause()
    pointcloud = lidarInterface.get_point_cloud_data("/World" + lidarPath)
    semantics = lidarInterface.get_prim_data("/World" + lidarPath)

    print("Point Cloud", pointcloud)
    print("Semantic ID", semantics)

timeline.play()  # Start the Simulation
asyncio.ensure_future(get_lidar_param())  # Only ask for data after sweep is complete
```

The main differences between this example and the previous are as follows:

1. The LIDAR’s `enable_semantics` flag is set to `True` on creation.
2. The Cube and Sphere prims are assigned different semantic labels.
3. `get_point_cloud_data` and `get_prim_data` are used to retrieve the Point Cloud data and Semantic IDs.

The segmented point cloud from Lidar sensor should look like the image below:

---

# PhysX SDK Lightbeam Sensor

The PhysX SDK Lightbeam sensor in Isaac Sim uses PhysX SDK raycasts to determine if an object has intersected a light beam.
You can specify the number of rays and height to create a safety light “curtain” of lightbeam sensors.

See the [Isaac Sim Conventions](Isaac_Sim_Conventions.md) documentation for a complete list of Isaac Sim conventions.

## Examples

* PhysX SDK Lightbeam Sensor example: **Robotics Examples > Sensors > Lightbeam**

To run the example:

1. Activate **Robotics Examples** tab from **Windows** > **Examples** > **Robotics Examples**.
2. Click **Robotics Examples > Sensors > Lightbeam**.
3. Verify that you have a window containing empty data for each lightbeam, which will be populated with data after you press **play**. It will show if each beam was hit, the linear depth of the hit, and the exact hit position in `xyz`.
4. Press the **PLAY** button to begin simulating.
5. Press `SHIFT + LEFT_CLICK` to drag the cube or sensor around and see changes in the readings.

---

# Camera and Depth Sensors

Isaac Sim supports camera and depth sensors, with digital twins found in the Content Browser
:   under `Isaac Sim/Sensors`, organized into subfolders by manufacturer.

## Cameras

For more information about camera modeling in Isaac Sim, see [here](Sensors.md).

### Leopard Imaging

#### Hawk Stereo Camera

The Hawk Stereo Camera ([LI-AR0234CS-STEREO-GMSL2-30](https://leopardimaging.com/product/platform-partners/qualcomm/iot-robotics-qualcomm/li-ar0234cs-stereo-gmsl2-qualcomm/li-ar0234cs-stereo-gmsl2-30/)) from Leopard Imaging consists of two OnSemi AR0234CS RGB image sensors and a 6-axis IMU, both are simulated in the NVIDIA Isaac Sim.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>LeopardImaging>Hawk*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>LeopardImaging>Hawk>hawk\_v1.1\_nominal.usd*

Features and Specification

Camera Features

| name | camera\_left | camera\_right |
| --- | --- | --- |
| focalLength | 2.8734347820281982 | 2.8779797554016113 |
| focusDistance | 0.6000000238418579 | 0.6000000238418579 |
| fStop | 240.0 | 240.0 |
| projection | perspective | perspective |
| stereoRole | left | right |
| horizontalAperture | 5.760000228881836 | 5.760000228881836 |
| verticalAperture | 3.5999999046325684 | 3.5999999046325684 |
| clippingRange | (0.076, 100000) | (0.076, 100000) |
| cameraProjectionType | fisheyePolynomial | fisheyePolynomial |
| nominalWidth | 1920.0 | 1920.0 |
| nominalHeight | 1200.0 | 1200.0 |
| opticalCenterX | 957.85107421875 | 954.709228515625 |
| opticalCenterY | 589.5376586914062 | 588.3735961914062 |
| maxFOV | 150.0 | 150.0 |
| polyK0 | 5.0055230531143025e-05 | 8.962746505858377e-05 |
| polyK1 | 0.0010426010703667998 | 0.001039923052303493 |
| polyK2 | 9.85131620723223e-09 | 1.502240820627776e-08 |
| polyK3 | 1.6426542417957712e-11 | 5.982795422271314e-12 |
| polyK4 | 2.9886398802796144e-14 | 3.6818906078281075e-14 |
| polyK5 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | [0.147811, -0.032313, -0.000194, -0.000035, 0.008823, 0.517913, -0.06708, 0.01695] | [6.815791, 5.172144, -0.000246, -0.000128, 0.353267, 7.180808, 7.640372, 1.596375] |
| physicalDistortionModel | rational\_polynomial | rational\_polynomial |

**Other Features**

* Waterproof: IP65
* Dimensions: 180 mm (length) by 44.33 mm (depth) by 25.0 mm (height)
* Operating Temperature: -20C to 50C

IMU to Hawk sensor (left camera) transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 90 | 0.0 |
| Translation (meters) | 0.0 | -0.0947 | 0.0061 |

Note

For the datasheet and full list of specifications, visit the [Hawk stereo camera product page](https://leopardimaging.com/leopard-imaging-hawk-stereo-camera/) and [purchase here](https://leopardimaging.com/product/platform-partners/qualcomm/iot-robotics-qualcomm/li-ar0234cs-stereo-gmsl2-qualcomm/li-ar0234cs-stereo-gmsl2-30/).

#### Owl Fisheye camera

The Owl camera ([LI-AR0234CS-GMSL2-OWL](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/maxim-gmsl-2-cameras/li-ar0234cs-gmsl2-owl/li-ar0234cs-gmsl2-owl/)) from Leopard Imaging consists of a 2.3MP OnSemi AR0234CS RGB image sensor, capable of producing crisp images in low-light and bright scenes.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>LeopardImaging>Owl*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>LeopardImaging>Owl>owl.usd*

Features and Specification

Camera Features

| name | camera |
| --- | --- |
| focalLength | 1.3646053075790405 |
| focusDistance | 0.6000000238418579 |
| fStop | 180.0 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 3.5999999046325684 |
| clippingRange | (0.076, 100000) |
| cameraProjectionType | fisheyePolynomial |
| nominalWidth | 1920.0 |
| nominalHeight | 1200.0 |
| opticalCenterX | 943.99462890625 |
| opticalCenterY | 602.3110961914062 |
| maxFOV | 235.0 |
| polyK0 | 0.0002725422091316432 |
| polyK1 | 0.0021866457536816597 |
| polyK2 | 1.2340817079348199e-07 |
| polyK3 | -1.079574096785052e-09 |
| polyK4 | 5.997452426180494e-13 |
| polyK5 | 0.0 |
| p0 | -0.00037 |
| p1 | -0.00074 |
| s0 | -0.00058 |
| s1 | -0.00022 |
| s2 | 0.00019 |
| s3 | -0.0002 |
| physicalDistortionCoefficients | [0.057225, 0.012671, -0.002978, -0.000472] |
| physicalDistortionModel | kannalaBrandt |

**Other Features**

* Dimensions: 50 mm (length) by 37.63 mm (depth) by 25.0 mm (height)
* Operating Temperature: -20C to 50C

Note

For full list of specifications, visit the [product page](https://leopardimaging.com/leopard-imaging-hawk-stereo-camera/) ,and the owl cameras can be [purchased here](https://leopardimaging.com/product/automotive-cameras/cameras-by-interface/maxim-gmsl-2-cameras/li-ar0234cs-gmsl2-owl/li-ar0234cs-gmsl2-owl/).

### Sensing

#### SG2-AR0233C-5200-G2A-H100F1A Camera (Certified by Sensing)

SG2-AR0233C-5200-G2A-H100F1A from the [SG2-AR0233C-5200-G2A-Hxxx Series](https://www.sensing-world.com/en/pd.jsp?id=18) is a megapixel high performance automotive camera module, primarily used for ADAS, HDR imaging functionalities.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG2-AR0233C-5200-G2A-H100F1A*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG2>H100F1A>SG2-AR0233C-5200-G2A-H100F1A.usd*

Features and Specification

Camera Features

| name | SG2\_AR0233C\_5200\_G2A\_H100F1A\_01 |
| --- | --- |
| focalLength | 3.549999952316284 |
| focusDistance | 270.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 3.240000009536743 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeOrthographic |
| nominalWidth | 1920.0 |
| nominalHeight | 1080.0 |
| opticalCenterX | 998.0842895507812 |
| opticalCenterY | 520.5062866210938 |
| maxFOV | 100.0 |
| polyK0 | 0.9293811321258545 |
| polyK1 | 0.15743136405944824 |
| polyK2 | 0.008131147362291813 |
| polyK3 | 1.358112096786499 |
| polyK4 | 0.4388065040111542 |
| polyK5 | 0.035474397242069244 |
| p0 | -1.8616799934534356e-05 |
| p1 | -0.000114203299744986 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG2-AR0233C-5200-G2A-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=18)

#### SG2-OX03CC-5200-GMSL2-H60YA Series Camera (Certified by Sensing)

SG2-OX03CC-5200-GMSL2-H60YA from the [SG2-OX03CC-5200-GMSL2-Hxxx Series](https://www.sensing-world.com/en/pd.jsp?id=106&id=106) is a megapixel high performance automotive camera module, primarily used for ADAS, HDR imaging functionalities.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG2-OX03CC-5200-GMSL2-H60YA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG2>H60YA>Camera\_SG2\_OX03CC\_5200\_GMSL2\_H60YA.usd*

Features and Specification

Camera Features

| name | Camera\_SG2\_OX03CC\_5200\_GMSL2\_H60YA |
| --- | --- |
| focalLength | 5.75 |
| focusDistance | 700.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 3.240000009536743 |
| clippingRange | (0.1, 1000000) |
| cameraProjectionType | fisheyeOrthographic |
| nominalWidth | 1920.0 |
| nominalHeight | 1080.0 |
| opticalCenterX | 959.595947265625 |
| opticalCenterY | 647.6140747070312 |
| maxFOV | 60.0 |
| polyK0 | 0.7182272672653198 |
| polyK1 | 60.113136291503906 |
| polyK2 | 2.598527431488037 |
| polyK3 | 1.1977670192718506 |
| polyK4 | 60.394771575927734 |
| polyK5 | 31.610383987426758 |
| p0 | 0.0004008802934549749 |
| p1 | -0.0013344850158318877 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG2-OX03CC-5200-GMSL2F-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=106&id=106)

#### SG3-ISX031C-GMSL2F-H190XA (Certified by Sensing)

[SG3-ISX031C-GMSL2F-H190XA](https://www.sensing-world.com/en/pd.jsp?id=23#_jcp=2) is a 3 megapixels automotive camera for automotive surround view.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG3-ISX031C-GMSL2F-H190XA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG3>H190XA>SG3S-ISX031C-GMSL2F-H190XA.usd*

Features and Specification

Camera Features

| name | SG3S\_ISX031C\_GMSL2F\_H190XA\_01 |
| --- | --- |
| focalLength | 1.5099999904632568 |
| focusDistance | 39.0 |
| fStop | 2.0 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 5.760000228881836 |
| verticalAperture | 4.607999801635742 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeEquidistant |
| nominalWidth | 1920.0 |
| nominalHeight | 1536.0 |
| opticalCenterX | 960.6082153320312 |
| opticalCenterY | 768.0 |
| maxFOV | 190.0 |
| polyK0 | 0.13215887546539307 |
| polyK1 | -0.031036589294672012 |
| polyK2 | -0.004391151946038008 |
| polyK3 | 0.0018116832943633199 |
| polyK4 | 0.0 |
| polyK5 | 0.0 |
| p0 | 0.0 |
| p1 | 0.0 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG3-ISX031C-GMSL2F-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=23#_jcp=2)

#### SG5-IMX490C-5300-GMSL2-H110SA (Certified by Sensing)

[SG5-IMX490C-5300-GMSL2-H110SA](https://www.sensing-world.com/en/pd.jsp?id=24#_jcp=2) is a 5 megapixels automotive camera for automotive surround view, ADAS and viewing fusion.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG5-IMX490C-5300-GMSL2-H110SA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG5>H100SA>SG5-IMX490C-5300-GMSL2-H110SA.usd*

Features and Specification

Camera Features

| name | Camera\_SG5\_IMX490C\_5300\_GMSL2\_H110SA |
| --- | --- |
| focalLength | 4.260000228881836 |
| focusDistance | 220.0 |
| fStop | 2.799999952316284 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.640000343322754 |
| verticalAperture | 5.579999923706055 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeOrthographic |
| nominalWidth | 2880.0 |
| nominalHeight | 1860.0 |
| opticalCenterX | 1442.3316650390625 |
| opticalCenterY | 926.6644287109375 |
| maxFOV | 110.0 |
| polyK0 | 0.6106576919555664 |
| polyK1 | -0.11334560066461563 |
| polyK2 | -0.014692608267068863 |
| polyK3 | 0.9237731099128723 |
| polyK4 | -0.011052233166992664 |
| polyK5 | -0.051484767347574234 |
| p0 | 2.2259799152379856e-05 |
| p1 | -7.929380080895498e-05 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Waterproof: IP67
* Dimensions: 30 mm (length) by 22.5 mm (depth) by 30 mm (height)
* Operating Temperature: -40C to 85C
* Multi camera synchronization support

Note

For the datasheet and full list of specifications, visit the [SG5-IMX490C-5300-GMSL2-Hxxx series product page.](https://www.sensing-world.com/en/pd.jsp?id=24#_jcp=2)

#### SG8S-AR0820C-5300-G2A-H30YA Camera (Certified by Sensing)

[SG8S-AR0820C-5300-G2A-H30YA](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2) is a 4k high performance automotive grade camera that supports advanced on sensor HDR and multi-camera synchronization.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG8S-AR0820C-5300-G2A-H30YA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG8>H30YA>SG8S-AR0820C-5300-G2A-H30YA.usd*

Features and Specification

Camera Features

| name | SG8S\_AR0820C\_5300\_G2A\_H30YA\_01 |
| --- | --- |
| focalLength | 15.300000190734863 |
| focusDistance | 7070.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.064000129699707 |
| verticalAperture | 4.535999774932861 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | pinhole |
| nominalWidth | 3840.0 |
| nominalHeight | 2160.0 |
| opticalCenterX | 1864.240478515625 |
| opticalCenterY | 986.3945922851562 |
| maxFOV | 30.0 |
| polyK0 | -0.6564998626708984 |
| polyK1 | -4.156541347503662 |
| polyK2 | 245.6761932373047 |
| polyK3 | -0.43839189410209656 |
| polyK4 | -4.5701212882995605 |
| polyK5 | 251.74969482421875 |
| p0 | -0.000658363220281899 |
| p1 | 8.901000114747148e-07 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Dimensions: 40 mm (length) by 23 mm (depth) by 40 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG8-AR820C-5300-G2A-Hxxx series cameras product page.](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2)

#### SG8S-AR0820C-5300-G2A-H60SA Camera (Certified by Sensing)

[SG8S-AR0820C-5300-G2A-H60SA](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2) is a 4k high performance automotive grade camera that supports advanced on sensor HDR and multi-camera synchronization.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG8S-AR0820C-5300-G2A-H60SA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG8>H60SA>SG8S-AR0820C-5300-G2A-H60SA.usd*

Features and Specification

Camera Features

| name | SG8S\_AR0820C\_5300\_G2A\_H60SA\_01 |
| --- | --- |
| focalLength | 7.869999885559082 |
| focusDistance | 1670.0 |
| fStop | 1.7999999523162842 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.064000129699707 |
| verticalAperture | 4.535999774932861 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeEquidistant |
| nominalWidth | 3840.0 |
| nominalHeight | 2160.0 |
| opticalCenterX | 1919.1090087890625 |
| opticalCenterY | 1087.7274169921875 |
| maxFOV | 60.0 |
| polyK0 | 0.8600332140922546 |
| polyK1 | -0.30780455470085144 |
| polyK2 | -0.05103735625743866 |
| polyK3 | 1.5231009721755981 |
| polyK4 | 0.0005489090108312666 |
| polyK5 | -0.25151902437210083 |
| p0 | 6.143400241853669e-05 |
| p1 | -4.332419848651625e-05 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Dimensions: 40 mm (length) by 23 mm (depth) by 40 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG8-AR820C-5300-G2A-Hxxx series cameras product page.](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2)

#### SG8S-AR0820C-5300-G2A-H120YA Camera (Certified by Sensing)

[SG8S-AR0820C-5300-G2A-H120YA](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2) is a 4k high performance automotive grade camera that supports advanced on sensor HDR and multi-camera synchronization.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Sensing>Sensing SG8S-AR0820C-5300-G2A-H120YA*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Sensing>SG8>H120YA>SG8S-AR0820C-5300-G2A-H120YA.usd*

Features and Specification

Camera Features

| name | SG8S\_AR0820C\_5300\_G2A\_H120YA\_01 |
| --- | --- |
| focalLength | 4.010000228881836 |
| focusDistance | 480.0 |
| fStop | 1.600000023841858 |
| projection | perspective |
| stereoRole | mono |
| horizontalAperture | 8.064000129699707 |
| verticalAperture | 4.535999774932861 |
| clippingRange | (1, 1000000) |
| cameraProjectionType | fisheyeEquidistant |
| nominalWidth | 3840.0 |
| nominalHeight | 2160.0 |
| opticalCenterX | 1919.1090087890625 |
| opticalCenterY | 1087.7274169921875 |
| maxFOV | 120.0 |
| polyK0 | 0.8600332140922546 |
| polyK1 | -0.30780455470085144 |
| polyK2 | -0.05103735625743866 |
| polyK3 | 1.5231009721755981 |
| polyK4 | 0.0005489090108312666 |
| polyK5 | -0.25151902437210083 |
| p0 | 6.143400241853669e-05 |
| p1 | -4.332419848651625e-05 |
| s0 | 0.0 |
| s1 | 0.0 |
| s2 | 0.0 |
| s3 | 0.0 |
| physicalDistortionCoefficients | Not Applicable |
| physicalDistortionModel | Not Applicable |

**Other Features**

* Dimensions: 40 mm (length) by 23 mm (depth) by 40 mm (height)
* Operating Temperature: -40C to 85C

Note

For the datasheet and full list of specifications, visit the [SG8-AR820C-5300-G2A-Hxxx series cameras product page.](https://www.sensing-world.com/en/pd.jsp?id=26#_jcp=2)

### SICK

#### Inspector83x (Certified by SICK)

##### Inspector83x (Certified)

The [SICK Inspector83x](https://www.sick.com/inspector83x) is a 2D camera, which helps to rapidly solve vision applications such as quality assurance, defect detection, and sorting.

###### Features and Specification

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [Inspector83x product page](https://www.sick.com/inspector83x).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>SICK>Inspector83x*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>Inspector83x>SICK\_Inspector83x.usd*

## Depth Sensors

For more information about depth sensor modeling in Isaac Sim, see [here](Sensors.md).

### Realsense

#### Realsense D455 (Certified by Realsense)

##### Realsense D455 (Certified)

The [Realsense D455](https://realsenseai.com/products/real-sense-depth-camera-d455f) consists of multiple RGB and depth image sensors and a 6-axis IMU.

###### Features and Specification

**Other Features**

* Dimensions: 124 mm (length) by 25 mm (depth) by 29 mm (height)
* IMU: Bosch BM1055
* Ideal Range: 0.6m to 6m
* Minimum Depth Distance at Max resolution: 52cm
* Depth accuracy: under 2% at 4m

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [D455 product page](https://realsenseai.com/products/real-sense-depth-camera-d455f).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Realsense>Realsense D455*

Camera Attributes

Camera Features

| name | Camera\_Pseudo\_Depth | Camera\_OmniVision\_OV9782\_Color | Camera\_OmniVision\_OV9782\_Left | Camera\_OmniVision\_OV9782\_Right |
| --- | --- | --- | --- | --- |
| focalLength | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 |
| focusDistance | 0.6000000238418579 | 0.5 | 0.5 | 0.5 |
| fStop | 2.0 | 2.0 | 2.0 | 2.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 |
| verticalAperture | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 |
| clippingRange | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) |
| cameraProjectionType | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 100.5999984741211 | 98.0 | 98.0 | 98.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

IMU to RealSense transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 0.0 | 0.0 |
| Translation (meters) | 0.016 | -0.01728 | 0.0074 |

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Realsense>D455>rsd455.usd*

#### Realsense D457 (Certified by Realsense)

##### Realsense D457

The [Realsense D457](https://www.realsenseai.com/products/d457-gmsl-fakra/) is a ruggedized, IP65-rated stereo depth camera featuring a GMSL/FAKRA interface for secure, long-distance high-bandwidth connectivity. It utilizes the same optical module as the D455 and is designed for autonomous mobile robots (AMRs) and automotive infotainment.

###### Features and Specification

**Other Features**

* Dimensions: 124 mm (length) by 36 mm (depth) by 29 mm (height)
* Environment: Indoor/Outdoor (IP65 Rated)
* Vision Processor: Vision Processor D4 Board V5
* Sensors: Global Shutter (Depth and RGB)
* Depth FOV: 87° × 58° (Resolution up to 1280 × 720 at 90 fps)
* RGB FOV: 90° × 65° (Resolution up to 1280 × 800 at 60 fps)
* Minimum Depth Distance: 26-52 cm
* Connectors: GMSL/FAKRA (Maxim Integrated), USB-C
* IMU: Built-in 6-axis IMU

> ℹ️ **Note**
> For the full datasheet, visit the [D457 Datasheet](https://www.realsenseai.com/wp-content/uploads/2023/01/Intel-RealSense-D457-Datasheet-January2023.pdf).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Realsense>Realsense D457*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Realsense>D457>rsd457.usd*

Camera Attributes

Camera Features

| name | Camera\_Pseudo\_Depth | Camera\_OmniVision\_OV9782\_Color | Camera\_OmniVision\_OV9782\_Left | Camera\_OmniVision\_OV9782\_Right |
| --- | --- | --- | --- | --- |
| focalLength | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 |
| focusDistance | 0.6000000238418579 | 0.5 | 0.5 | 0.5 |
| fStop | 2.0 | 2.0 | 2.0 | 2.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 |
| verticalAperture | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 |
| clippingRange | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) |
| cameraProjectionType | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 100.5999984741211 | 98.0 | 98.0 | 98.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

IMU to RealSense transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 0.0 | 0.0 |
| Translation (meters) | 0.016 | -0.01728 | 0.0074 |

#### Realsense D555 (Certified by Realsense)

##### Realsense D555

The [Realsense D555](https://www.realsenseai.com/products/d555-poe/) is a ruggedized, IP65-rated stereo depth camera designed for industrial and outdoor environments. It features the new RealSense Vision SoC V5, on-chip Power over Ethernet (PoE), and global shutter sensors for both RGB and Depth.

###### Features and Specification

**Other Features**

* Dimensions: 167 mm (length) by 42 mm (depth) by 48 mm (height)
* Environment: Indoor/Outdoor (IP65 Rated)
* Vision Processor: RealSense Vision SoC V5
* Sensors: Global Shutter (Depth and RGB)
* Depth FOV: 87° × 58° (Resolution up to 1280 × 720 at 90 fps)
* RGB FOV: 90° × 65° (Resolution up to 1280 × 800 at 60 fps)
* Minimum Depth Distance: 26-52 cm
* Connectors: PoE (RJ45), USB-C (Power/Data), GMSL/FAKRA, External HW Sync via USB
* Native ROS Support: Powered by SafeDDS (ISO 26262-certified) and interoperable with Fast DDS, enabling plug-and-play ROS 2 streaming over Ethernet without additional installation.

> ℹ️ **Note**
> For the full datasheet, visit the [D555 Datasheet](https://www.realsenseai.com/wp-content/uploads/2025/08/D555-Datasheet-v1.1.pdf).

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Realsense>Realsense D555*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Realsense>D555>rsd555.usd*

Camera Attributes

Camera Features

| name | Camera\_Pseudo\_Depth | Camera\_OmniVision\_OV9782\_Color | Camera\_OmniVision\_OV9782\_Left | Camera\_OmniVision\_OV9782\_Right |
| --- | --- | --- | --- | --- |
| focalLength | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 | 1.9299999475479126 |
| focusDistance | 0.6000000238418579 | 0.5 | 0.5 | 0.5 |
| fStop | 2.0 | 2.0 | 2.0 | 2.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 | 3.8959999084472656 |
| verticalAperture | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 | 2.453000068664551 |
| clippingRange | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) | (0.01, 1000000) |
| cameraProjectionType | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant | fisheyeEquidistant |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 100.5999984741211 | 98.0 | 98.0 | 98.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

IMU to RealSense transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | 0.0 | 0.0 | 0.0 |
| Translation (meters) | 0.016 | -0.01728 | 0.0074 |

### Orbbec

#### Orbbec Gemini 2 (Certified by Orbbec)

The [Orbbec Gemini 2](https://www.orbbec.com/products/stereo-vision-camera/gemini-2/) is a depth camera based on Active Stereo IR technology.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec Gemini 2*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>Gemini 2>orbbec\_gemini2\_V1.0.usd*

Features and Specification

Camera Features

| name | Stream\_rgb | Stream\_depth | Stream\_ir\_left | Stream\_ir\_right |
| --- | --- | --- | --- | --- |
| focalLength | 2.9700000286102295 | 1.809999942779541 | 1.809999942779541 | 1.809999942779541 |
| focusDistance | 80.0 | 45.0 | 45.0 | 45.0 |
| fStop | 0.0 | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective | perspective |
| stereoRole | mono | mono | left | right |
| horizontalAperture | 5.539000034332275 | 3.619999885559082 | 3.880000114440918 | 3.880000114440918 |
| verticalAperture | 3.0920000076293945 | 2.440000057220459 | 2.440000057220459 | 2.440000057220459 |
| clippingRange | (0.005, 1000) | (0.005, 1000) | (0.005, 1000) | (0.005, 1000) |
| cameraProjectionType | pinhole | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 90 mm (length) by 25 mm (depth) by 30 mm (height)
* IMU supported with multi camera synchronization
* Ideal Range: 0.15m to 10m
* Depth accuracy: under 2% at 2m

Note

For the datasheet and full list of specifications, visit the [Gemini 2 product page.](https://www.orbbec.com/products/stereo-vision-camera/gemini-2)

#### Orbbec Femto Mega (Certified by Orbbec)

The [Orbbec Femto Mega](https://www.orbbec.com/products/tof-camera/femto-mega/) is a programmable multi-mode Depth and RGB camera.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec FemtoMega*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>FemtoMega>orbbec\_femtomega\_v1.0.usd*

Features and Specification

Camera Features

| name | camera\_rgb | camera\_tof\_nfov | camera\_tof\_wfov |
| --- | --- | --- | --- |
| focalLength | 3.25 | 1.690000057220459 | 1.690000057220459 |
| focusDistance | 150.0 | 44.0 | 44.0 |
| fStop | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective |
| stereoRole | mono | mono | mono |
| horizontalAperture | 5.449999809265137 | 2.5899999141693115 | 5.849999904632568 |
| verticalAperture | 3.0999999046325684 | 2.1500000953674316 | 5.849999904632568 |
| clippingRange | (0.01, 1000000) | (0.01, 1000) | (0.01, 1000) |
| cameraProjectionType | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 115mm (length) by 145 mm (depth) by 40mm (height)
* IMU supported
* Ideal Range: 0.25m to 5.46m
* Depth accuracy: under 11mm + 0.1% distance

Note

For the datasheet and full list of specifications, visit the [Femto Mega product page.](https://www.orbbec.com/products/tof-camera/femto-mega/)

#### Orbbec Gemini 335 (Certified by Orbbec)

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec Gemini 335*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>Gemini335>orbbec\_gemini\_335.usd*

Features and Specification

Camera Features

| name | Stream\_rgb | Stream\_ir\_left | Stream\_ir\_right |
| --- | --- | --- | --- |
| focalLength | 2.9700000286102295 | 1.809999942779541 | 1.809999942779541 |
| focusDistance | 80.0 | 45.0 | 45.0 |
| fStop | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective |
| stereoRole | mono | right | left |
| horizontalAperture | 5.539000034332275 | 3.880000114440918 | 3.880000114440918 |
| verticalAperture | 3.0920000076293945 | 2.440000057220459 | 2.440000057220459 |
| clippingRange | (0.005, 1000) | (0.005, 1000) | (0.005, 1000) |
| cameraProjectionType | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable |

#### Orbbec Gemini 335L (Certified by Orbbec)

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Orbbec>Orbbec Gemini 335L*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Orbbec>Gemini335L>orbbec\_gemini\_335L.usd*

Features and Specification

Camera Features

| name | Camera\_ir\_left | Camera\_ir\_right | Camera\_rgb |
| --- | --- | --- | --- |
| focalLength | 1.809999942779541 | 1.809999942779541 | 1.809999942779541 |
| focusDistance | 45.0 | 45.0 | 0.44999998807907104 |
| fStop | 0.0 | 0.0 | 0.0 |
| projection | perspective | perspective | perspective |
| stereoRole | mono | mono | mono |
| horizontalAperture | 3.8399999141693115 | 3.8399999141693115 | 3.8399999141693115 |
| verticalAperture | 2.4000000953674316 | 2.4000000953674316 | 2.4000000953674316 |
| clippingRange | (0.005, 100000) | (0.005, 100000) | (0.005, 100000) |
| cameraProjectionType | pinhole | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable | Not Applicable |

### Stereolabs

#### ZED X (Certified by Stereolabs)

The [ZED X Stereo Camera](https://www.stereolabs.com/zed-x/) from Stereolabs consists of two 1200p 60fps RGB image sensors and a 6-axis IMU, all simulated in the NVIDIA Isaac Sim.

To create the camera from the menu: *Create>Sensors>Camera and Depth Sensors>Stereolabs>ZED\_X*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Stereolabs>ZED\_X>ZED\_X.usd*

Features and Specification

Camera Features

| name | CameraLeft | CameraRight |
| --- | --- | --- |
| focalLength | 2.2079999446868896 | 2.2079999446868896 |
| focusDistance | 28.0 | 28.0 |
| fStop | 0.0 | 0.0 |
| projection | perspective | perspective |
| stereoRole | left | right |
| horizontalAperture | 5.760000228881836 | 5.760000228881836 |
| verticalAperture | 3.240000009536743 | 3.240000009536743 |
| clippingRange | (0.01, 100000) | (0.01, 100000) |
| cameraProjectionType | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 163.4 mm (length) by 31.8 mm (depth) by 36.7 mm (height)
* Operating Temperature: -20C to 55C

IMU to ZED X transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | -90.0 | 0.0 | 0.0 |
| Translation (meters) | 0.06 | -0.0 | 0.00185 |

Note

For the datasheet and full list of specifications, visit the [ZED X datasheet](https://www.stereolabs.com/datasheets), for usage in Isaac Sim, see [Stereolabs Documentation](https://www.stereolabs.com/docs/isaac-sim).

#### ZED X Mini (Certified by Stereolabs)

The [ZED X Mini Stereo Camera](https://www.stereolabs.com/zed-x/) from Stereolabs consists of two 1200p 60fps RGB image sensors and a 6-axis IMU, all simulated in the NVIDIA Isaac Sim.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Stereolabs>ZED\_X\_mini>ZED\_X\_Mini.usd*

Features and Specification

Camera Features

| name | CameraRight | CameraLeft |
| --- | --- | --- |
| focalLength | 2.2079999446868896 | 2.2079999446868896 |
| focusDistance | 28.0 | 28.0 |
| fStop | 0.0 | 0.0 |
| projection | perspective | perspective |
| stereoRole | right | left |
| horizontalAperture | 5.760000228881836 | 5.760000228881836 |
| verticalAperture | 3.240000009536743 | 3.240000009536743 |
| clippingRange | (0.01 100000) | (0.01 100000) |
| cameraProjectionType | pinhole | pinhole |
| nominalWidth | 1936.0 | 1936.0 |
| nominalHeight | 1216.0 | 1216.0 |
| opticalCenterX | 970.94244 | 970.94244 |
| opticalCenterY | 600.37482 | 600.37482 |
| maxFOV | 200.0 | 200.0 |
| polyK0 | 0.0 | 0.0 |
| polyK1 | 0.00245 | 0.00245 |
| polyK2 | 0.0 | 0.0 |
| polyK3 | 0.0 | 0.0 |
| polyK4 | 0.0 | 0.0 |
| polyK5 | 0.0 | 0.0 |
| p0 | -0.00037 | -0.00037 |
| p1 | -0.00074 | -0.00074 |
| s0 | -0.00058 | -0.00058 |
| s1 | -0.00022 | -0.00022 |
| s2 | 0.00019 | 0.00019 |
| s3 | -0.0002 | -0.0002 |
| physicalDistortionCoefficients | Not Applicable | Not Applicable |
| physicalDistortionModel | Not Applicable | Not Applicable |

**Other Features**

* Dimensions: 93.6 mm (length) by 31.8 mm (depth) by 36.7 mm (height)
* Operating Temperature: -20C to 55C

IMU to ZED X Mini transformation in Isaac Sim

IMU Sensor transformation

| Transformation | x | y | z |
| --- | --- | --- | --- |
| Rotation (degrees) | -90.0 | 0.0 | 0.0 |
| Translation (meters) | 0.06 | -0.0 | 0.00185 |

Note

For the datasheet and full list of specifications, visit the [ZED X Mini datasheet](https://www.stereolabs.com/datasheets), for usage in Isaac Sim, see [Stereolabs Documentation](https://www.stereolabs.com/docs/isaac-sim).

---

# Non-Visual Sensors

Isaac Sim models many types of non-visual sensors models, with digital twins found in the Content Browser under `Isaac Sim/Sensors`, organized into subfolders by manufacturer.

Some non-visual sensor types do not have digital twins. For more information about these sensors,
including how to create them from the GUI, follow the links below:

* [Contact sensors](Sensors.md)
* [IMU sensors](Sensors.md)
* [Lightbeam sensors](Sensors.md)
* [PhysX Lidars](Sensors.md)
* [RTX Radars](Sensors.md)

## RTX Lidars

RTX Lidars marked as “certified” have Lidar configurations verified by the sensor manufacturer and tested before release.

Some Lidar models feature multiple configurations or profiles, which are implemented as [USD Variants](https://docs.omniverse.nvidia.com/workflows/latest/variant-workflows.html).
In those cases, the available variants and their characteristics will also be provided as tables in the appropriate section below.

### NVIDIA

There are several example Lidar configuration files that ship with Isaac Sim. Note none of these Lidars have a mesh,
so only a prim will appear in the Stage window when they are created. To create them via the UI, select the appropriate
option below from the menu: *Create>Sensors>RTX Lidar>NVIDIA*.

* **Example Rotary 2D** - a 10Hz rotary Lidar configuration with emitters in a single plane.
* **Example Rotary** - a 10Hz rotary Lidar configuration with emitters in a single plane.
* **Example Rotary Beams** - a 10Hz rotary Lidar configuration using a Gaussian beam ray type.
* **Example Solid State** - a solid state Lidar configuration.
* **Example Solid State Beams** - a solid state Lidar configuration using a Gaussian beam ray type.
* **Simple Example Solid State** - a simple 12-emitter solid state Lidar configuration, used to debug solid state Lidar issues.

### HESAI

#### XT32 SD10

[HESAI XT32 SD10](https://www.hesaitech.com/product/xt32/) is a high precision, 32 Channels 360 degrees spinning mid range Lidar.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>HESAI>XT32 SD10*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>HESAI>XT32\_SD10>HESAI\_XT32\_SD10.usd*

Features and Specification

XT32 SD10 Features

| name | XT-32 10hz |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20000 |
| numberOfEmitters | 32 |
| nearRangeM | 0.05 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.004 |
| rangeAccuracyM | 0.02 |
| minDistBetweenEchos | 0.05 |
| minReflectance | 0.1 |
| minReflectanceRange | 80.0 |
| wavelengthNm | 905.0 |
| pulseTimeNs | 10 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.015 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.0 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 100 mm (Top Diameter) by 103 mm (Bottom Diameter) by 76.0 mm (Height)

Note

For the datasheet and full list of specifications, visit the [XT32 SD10 product page.](https://www.hesaitech.com/product/xt32/)

### Ouster

#### OS0

[Ouster OS0](https://ouster.com/products/hardware/os0-lidar-sensor) is a high precision Lidar for autonomous vehicles, heavy machinery, robot, and mapping solutions. Isaac Sim has several pre-configured frequencies and resolutions that can be added to the stage easily.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>OS0*, then select the desired sensor configuration.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>OS0>OS0.usd*

Features and Specification

OS0 Rev6 Features

10 Hz

512 Resolution

Variant: OS0\_REV6\_128ch10hz512res

| name | OS0 REV6 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV6\_128ch10hz1024res

| name | OS0 REV6 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS0\_REV6\_128ch10hz2048res

| name | OS0 REV6 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS0\_REV6\_128ch20hz512res

| name | OS0 REV6 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV6\_128ch20hz1024res

| name | OS0 REV6 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 50.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 20.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

OS0 Rev7 Features

10 Hz

512 Resolution

Variant: OS0\_REV7\_128ch10hz512res

| name | OS0 REV7 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV7\_128ch10hz1024res

| name | OS0 REV7 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS0\_REV7\_128ch10hz2048res

| name | OS0 REV7 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS0\_REV7\_128ch20hz512res

| name | OS0 REV7 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS0\_REV7\_128ch20hz1024res

| name | OS0 REV7 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 75.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 35.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

**Other Features**

* Rotation Rate: 10 or 20 hz (configurable)
* Dimensions: 87 mm (Diameter) by 58.35 mm (Height). With thermal cap, height is 74.2 mm.
* IMU supported: [InvenSense IAM-20680HT](https://invensense.tdk.com/download-pdf/iam-20680ht-datasheet/)

Note

For the datasheet and full list of specifications, visit the [OS0 product page.](https://ouster.com/products/hardware/os0-lidar-sensor)

#### OS1

[Ouster OS1](https://ouster.com/products/hardware/os1-lidar-sensor) is a high precision Lidar for autonomous vehicles, heavy machinery, robot, and mapping solutions.
Isaac Sim has several pre-configured frequencies and resolutions that can be easily added to the stage.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>OS1*.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>OS1>OS1.usd*

Features and Specification

OS1 Rev6 Features

32 Channels 10 Hz

512 Resolution

Variant: OS1\_REV6\_32ch10hz512res

| name | OS1 REV6 32 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_32ch10hz1024res

| name | OS1 REV6 32 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS1\_REV6\_32ch10hz2048res

| name | OS1 REV6 32 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

32 Channels 20 Hz

512 Resolution

Variant: OS1\_REV6\_32ch20hz512res

| name | OS1 REV6 32 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_32ch20hz1024res

| name | OS1 REV6 32 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 32 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

128 Channels 10 Hz

512 Resolution

Variant: OS1\_REV6\_128ch10hz512res

| name | OS1 REV6 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_128ch10hz1024res

| name | OS1 REV6 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS1\_REV6\_128ch10hz2048res

| name | OS1 REV6 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

128 Channels 20 Hz

512 Resolution

Variant: OS1\_REV6\_128ch20hz512res

| name | OS1 REV6 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV6\_128ch20hz1024res

| name | OS1 REV6 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.3 |
| farRangeM | 120.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 55.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

OS1 Rev7 Features

10 Hz

512 Resolution

Variant: OS1\_REV7\_128ch10hz512res

| name | OS1 REV7 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV7\_128ch10hz1024res

| name | OS1 REV7 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS1\_REV7\_128ch10hz2048res

| name | OS1 REV7 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS1\_REV7\_128ch20hz512res

| name | OS1 REV7 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS1\_REV7\_128ch20hz1024res

| name | OS1 REV7 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.5 |
| farRangeM | 170.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.025 |
| minReflectance | 0.1 |
| minReflectanceRange | 90.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 87 mm (Diameter) by 58.35 mm (Height). With thermal cap, height is 74.2 mm.
* IMU supported: [InvenSense IAM-20680HT](https://invensense.tdk.com/download-pdf/iam-20680ht-datasheet/)

Note

For the datasheet and full list of specifications, visit the [OS1 product page.](https://ouster.com/products/hardware/os1-lidar-sensor)

#### OS2

[Ouster OS2](https://ouster.com/products/hardware/os2-lidar-sensor) is a high precision Lidar for autonomous vehicles, heavy machinery, robot, and mapping solutions.
Isaac Sim has several pre-configured frequencies and resolutions that can be easily added to the stage.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>OS2*.

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>OS2>OS2.usd*

Features and Specification

OS2 Rev6 Features

10 Hz

512 Resolution

Variant: OS2\_REV6\_128ch10hz512res

| name | OS2 REV6 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV6\_128ch10hz1024res

| name | OS2 REV6 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS2\_REV6\_128ch10hz2048res

| name | OS2 REV6 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS2\_REV6\_128ch20hz512res

| name | OS2 REV6 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV6\_128ch20hz1024res

| name | OS2 REV6 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 240.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.03 |
| minReflectance | 0.1 |
| minReflectanceRange | 100.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

OS2 Rev7 Features

10 Hz

512 Resolution

Variant: OS2\_REV7\_128ch10hz512res

| name | OS2 REV7 128 10hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 5120 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV7\_128ch10hz1024res

| name | OS2 REV7 128 10hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

2048 Resolution

Variant: OS2\_REV7\_128ch10hz2048res

| name | OS2 REV7 128 10hz @ 2048 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

20 Hz

512 Resolution

Variant: OS2\_REV7\_128ch20hz512res

| name | OS2 REV7 128 20hz @ 512 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 10240 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

1024 Resolution

Variant: OS2\_REV7\_128ch20hz1024res

| name | OS2 REV7 128 20hz @ 1024 resolution |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 20.0 |
| reportRateBaseHz | 20480 |
| numberOfEmitters | 128 |
| nearRangeM | 0.8 |
| farRangeM | 350.0 |
| rangeResolutionM | 0.001 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 200.0 |
| wavelengthNm | 865.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.01 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.01 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 87 mm (Diameter) by 58.35 mm (Height). With thermal cap, height is 74.2 mm.
* IMU supported: [InvenSense IAM-20680HT](https://invensense.tdk.com/download-pdf/iam-20680ht-datasheet/)

Note

For the datasheet and full list of specifications, visit the [OS2 product page.](https://ouster.com/products/hardware/os2-lidar-sensor)

#### VLS 128

[Ouster VLS 128](https://ouster.com/products/hardware/vls-128) is a long range, ultra high resolution 3D Lidar for autonomous vehicles.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Ouster>VLS 128*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Ouster>VLS\_128>Ouster\_VLS\_128.usd*

Features and Specification

VLS 128 Features

| name | Velodyne VLS-128 |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 18761 |
| numberOfEmitters | 128 |
| nearRangeM | 1.0 |
| farRangeM | 200.0 |
| rangeResolutionM | 0.004 |
| rangeAccuracyM | 0.02 |
| minReflectance | 0.1 |
| minReflectanceRange | 120.0 |
| wavelengthNm | 903.0 |
| pulseTimeNs | 6 |
| maxReturns | 2 |

**Other Features**

* Dimensions: 165.5 mm (Diameter) by 141.3 mm (Height)
* Operating Temperature: -20C to 60C

Note

[VLS 128 product page.](https://ouster.com/products/hardware/vls-128)

### SICK

#### LRS4581R (Certified)

The [SICK LRS4581R](https://www.sick.com/LRS4000) of the LRS4000 family is a 2D LiDAR sensor for large scanning ranges in outdoor applications or for localization tasks.

##### Features and Specification

| Profile | Scan frequency | Angular resolution |
| --- | --- | --- |
| Profile\_1 | 12.5 Hz | 0.02° |
| Profile\_2 | 12.5 Hz | 0.04° |
| Profile\_3 | 12.5 Hz | 0.06° |
| Profile\_4 | 12.5 Hz | 0.1° |
| Profile\_5 | 12.5 Hz | 0.12° |
| Profile\_6 | 25 Hz | 0.04° |
| Profile\_7 | 25 Hz | 0.08° |
| Profile\_8 | 25 Hz | 0.12° |
| Profile\_9 | 25 Hz | 0.2° |
| Profile\_10 | 25 Hz | 0.24° |
| Profile\_Extended\_1 | 12.5 Hz | 0.04° |
| Profile\_Extended\_2 | 12.5 Hz | 0.08° |
| Profile\_Extended\_3 | 12.5 Hz | 0.12° |
| Profile\_Extended\_4 | 12.5 Hz | 0.24° |
| Profile\_Extended\_5 | 25 Hz | 0.08° |
| Profile\_Extended\_6 | 25 Hz | 0.16° |
| Profile\_Extended\_7 | 25 Hz | 0.24° |
| Profile\_Extended\_8 | 25 Hz | 0.48° |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [LRS4581R product page](https://www.sick.com/LRS4000).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>LRS4581R*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>LRS4581R>SICK\_LRS4581R.usd*

#### microScan3 (Certified)

The [SICK microScan3](https://www.sick.com/microScan3) safety laser scanner stands for the protection of very different applications: from stationary to mobile, from simple to complex and delivers high-precision measurement data.

##### Features and Specification

| Profile | Protective field range | Scan frequency |
| --- | --- | --- |
| Profile\_1 | 4.0 m | 33.3 Hz |
| Profile\_2 | 4.0 m | 25.0 Hz |
| Profile\_3 | 5.5 m | 33.3 Hz |
| Profile\_4 | 5.5 m | 25.0 Hz |
| Profile\_5 | 9.0 m | 25.0 Hz |
| Profile\_6 | 9.0 m | 20.0 Hz |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [microScan3 product page](https://www.sick.com/microScan3).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>microScan3*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>microScan3>SICK\_microScan3.usd*

#### MRS1104C (Certified)

The [SICK MRS1104C](https://www.sick.com/MRS1000) of the MRS1000 family is a 3D LiDAR sensor for collision protection and assistance for all traveling objects in production facilities or reliable monitoring in traffic management and  building security.

##### Features and Specification

| Profile | Scan frequency | Angular resolution |
| --- | --- | --- |
| Profile\_1 | 12.5 Hz | 0.25° |
| Profile\_2\_Interlaced | 6.25 Hz | 0.125° |
| Profile\_3\_Interlaced | 3.125 Hz | 0.0625° |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [MRS1104C product page](https://www.sick.com/MRS1000).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>MRS1104C*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>MRS1104C>SICK\_MRS1104C.usd*

#### multiScan136 (Certified)

The SICK [multiScan136](http://www.sick.com/multiScan100) of the multiScan100 family is a 3D LiDAR sensor for mobile and stationary applications and reliably detects drop-off edges and obstacles ahead.

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [multiScan136 product page](http://www.sick.com/multiScan100).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>multiScan136*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>multiScan136>SICK\_multiScan136.usd*

#### multiScan165 (Certified)

The SICK [multiScan165](http://www.sick.com/multiScan100) of the multiScan100 family is a 3D LiDAR sensor for mobile and stationary applications and reliably detects drop-off edges and obstacles ahead.

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [multiScan165 product page](http://www.sick.com/multiScan100).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>multiScan165*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>multiScan165>SICK\_multiScan165.usd*

#### nanoScan3 (Certified)

The [SICK nanoScan3](https://www.sick.com/nanoScan3) is the smallest safety laser scanner, which is well suited for the protection and localization of mobile platforms.

##### Features and Specification

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [nanoScan3 product page](https://www.sick.com/nanoScan3).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>nanoScan3*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>nanoScan3>SICK\_nanoScan3.usd*

#### picoScan150 (Certified)

The [SICK picoScan150](http://www.sick.com/picoScan100) of the picoScan100 family is a 2D LiDAR sensor for solving demanding industrial applications such as collision avoidance or measurement and monitoring in indoor and outdoor areas.

##### Features and Specification

| Profile | Scan frequency | Angular resolution |
| --- | --- | --- |
| Profile\_1 | 15 Hz | 0.5° |
| Profile\_2 | 15 Hz | 0.33° |
| Profile\_3 | 20 Hz | 0.1° |
| Profile\_4 | 20 Hz | 0.25° |
| Profile\_5 | 25 Hz | 0.25° |
| Profile\_6 | 30 Hz | 0.1° |
| Profile\_7 | 40 Hz | 0.25° |
| Profile\_8 | 50 Hz | 0.25° |
| Profile\_9 | 15 Hz | 0.05° |
| Profile\_10 | 40 Hz | 0.125° |
| Profile\_11 | 15 Hz | 1.0° |

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [picoScan150 product page](http://www.sick.com/picoScan100).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>picoScan150*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>picoScan150>SICK\_picoScan150.usd*

#### TiM781 (Certified)

The [SICK TiM781](http://www.sick.com/TiM) of the TiM family is a 2D LiDAR sensor for collision protection for mobile applications, object measurement or monitoring of objects.

> ℹ️ **Note**
> For the datasheet and full list of specifications, visit the [TiM781 product page](http://www.sick.com/TiM).

To create the sensor from the menu: *Create>Sensors>RTX Lidar>SICK>TiM781*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>SICK>tim781.usd*

### SLAMTEC

#### RPLIDAR S2E

[SLAMTEC RPLIDAR S2E](https://download-en.slamtec.com/api/download/rplidar-s2m1-RxE-datasheet/1.8?lang=en) is a low cost 360 degrees 2D laser scanner Lidar from SLAMTEC.

To create the sensor from the menu: *Create>Sensors>RTX Lidar>Slamtec>RPLIDAR S2E*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Slamtec>RPLIDAR\_S2E.usd*

Features and Specification

RPLIDAR S2E Features

|  |  |
| --- | --- |
| name | RPLIDAR S2E |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 32000 |
| numberOfEmitters | 1 |
| nearRangeM | 0.05 |
| farRangeM | 30.0 |
| rangeResolutionM | 0.013 |
| rangeAccuracyM | 0.03 |
| minDistBetweenEchos | 0.05 |
| minReflectance | 0.1 |
| minReflectanceRange | 10.0 |
| wavelengthNm | 905.0 |
| pulseTimeNs | 5 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.0 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.0 |
| maxReturns | 1 |

Note

For the datasheet and full list of specifications, vist the [RPLIDAR S2 product page.](https://www.slamtec.com/en/support#rplidar-s2)

### ZVISION

#### ML-30s+ (Certified)

[ZVISION ML-30s+](http://zvision.xyz/en/h-col-262.html) is a short range automotive grade solid state Lidar. Note there is no mesh for this lidar, so
when it is created via the UI, only a prim will appear in the Stage window.

Features and Specification

ML-30s+ Features

| name | ML-30s+ |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10 |
| numberOfEmitters | 51200 |
| numberOfChannels | 51200 |
| nearRangeM | 0.2 |
| farRangeM | 45.0 |
| effectiveApertureSize | 0.01 |
| focusDistM | 0.12 |
| rangeResolutionM | 0.03 |
| rangeAccuracyM | 0.03 |
| minDistBetweenEchos | 0.2 |
| minReflectance | 0.1 |
| minReflectanceRange | 270.0 |
| wavelengthNm | 905.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.025 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.025 |
| maxReturns | 2 |

To create the sensor from the menu: *Create>Sensors>RTX Lidar>ZVISION>ML30S+*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>ZVISION>ZVISION\_ML30S.usda*

Note

For the datasheet and full list of specifications, visit the [ML-30s+ product page.](http://zvision.xyz/en/h-col-262.html)

#### ML-Xs (Certified)

[ZVISION ML-Xs](http://zvision.xyz/en/h-col-279.html) is a long range automotive high performance grade solid state Lidar. Note there is no mesh for this lidar, so
when it is created via the UI, only a prim will appear in the Stage window.

Features and Specification

ML-30s+ Features

| name | ML-Xs |
| --- | --- |
| type | lidar |
| scanRateBaseHz | 10.0 |
| reportRateBaseHz | 10 |
| numberOfEmitters | 108000 |
| numberOfChannels | 108000 |
| nearRangeM | 0.5 |
| farRangeM | 250 |
| effectiveApertureSize | 0.01 |
| focusDistM | 0.12 |
| rangeResolutionM | 0.03 |
| rangeAccuracyM | 0.05 |
| minReflectance | 0.1 |
| minReflectanceRange | 270.0 |
| wavelengthNm | 1550.0 |
| pulseTimeNs | 6 |
| azimuthErrorMean | 0.0 |
| azimuthErrorStd | 0.025 |
| elevationErrorMean | 0.0 |
| elevationErrorStd | 0.025 |
| maxReturns | 2 |

To create the Lidar prim: *Create>Sensors>RTX Lidar>ZVISION>MLXS*

To create the sensor from the Content Browser: *Isaac Sim>Sensors>ZVISION>ZVISION\_MLXS.usda*

Note

For the datasheet and full list of specifications, visit the [ML-Xs product page.](http://zvision.xyz/en/h-col-279.html)

## Tactile Sensors

### Tashan Technology

#### Universal Tactile Sensor TS-F-A (Certified)

[Tashan Technology Universal Tactile Sensor TS-F-A](https://github.com/TashanTec/Tashan-Isaac-Sim) is a tactile simulation model based on real products to advance research and innovation in robotic tactile perception technology and promote the development of embodied intelligent robots.

Features and Specification

Outputs 11 dimensional feature channels:
:   * Proximity sensing [1]
    * Tactile sensing [2-4]: Normal force, tangential force, tangential force direction
    * Raw capacitance values [5-11]: 7-channel raw capacitance data

To create the sensor from the Content Browser: *Isaac Sim>Sensors>Tashan>TS-F-A>TS-F-A.usd*

Note

For usage in Isaac Sim, visit the [Tashan Technology Tactile Simulation Platform User Manual.](https://github.com/TashanTec/Tashan-Isaac-Sim)

## Sensor Gizmo in Viewport

In Isaac Sim, the sensor functions are decoupled from physical meshes, and you can have sensors on stage without any mesh associated with the sensor. We use sensor gizmo to track the location of the actual sensing functions regardless of mesh. The gizmos are not visible by default, but you can toggle them on or off in the viewport.

To toggle the sensor gizmos, go to **Viewport Menu** >  > **Show By Type** > **Sensors**.

