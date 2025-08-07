## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import math
from typing import List, Optional, Sequence, Tuple, Union

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom

# Try to import viewport utilities, but make them optional
try:
    from omni.kit.viewport.utility import get_active_viewport_camera_path, get_active_viewport_window

    VIEWPORT_AVAILABLE = True
except ImportError:
    VIEWPORT_AVAILABLE = False

    # Define stub functions when viewport is not available
    def get_active_viewport_camera_path(*args, **kwargs):
        return None

    def get_active_viewport_window(*args, **kwargs):
        return None


def create_camera(stage: Usd.Stage, path: Optional[str] = None) -> str:
    """
    Create a camera prim in the stage.

    This is the recommended way to create a camera in your USD stage, and is equivalent to
    using the "Create > Camera" menu in Omniverse Kit. The camera will be created with the
    same default settings and transform as if created from the UI menu.

    Args:
        stage: The USD stage
        path: Optional path for the camera. If None, camera will be created at default prim level

    Returns:
        str: Path to the created camera
    """
    if path is None:
        camera_path = omni.usd.get_stage_next_free_path(stage, "/Camera", True)
    else:
        camera_path = f"{path}"

    omni.kit.commands.execute(
        "CreatePrimWithDefaultXform", prim_type="Camera", prim_path=camera_path, select_new_prim=False
    )

    return camera_path


def focus_camera(
    stage: Usd.Stage,
    camera_path: Union[str, Sdf.Path],
    prims_to_frame: Union[str, Sdf.Path, List[Union[str, Sdf.Path]]],
    zoom: float = 1.0,
    time_code: Union[Usd.TimeCode, int, float] = Usd.TimeCode.Default(),
    rotation: Optional[Union[Tuple[float, float, float], Sequence[float], Gf.Vec3d, Gf.Vec3f]] = None,
    align_with_up_vector: bool = True,
) -> None:
    """
    Focus the camera on specified prim(s) and optionally create camera animation keyframes.

    Quick Example:
        ```python
        # Frame multiple objects at once
        usdcode.focus_camera(stage, "/World/Camera", ["/World/Sphere", "/World/Cylinder"], zoom=1.2)
        ```

    Args:
        stage: The USD stage
        camera_path: Path to the camera prim (accepts string or Sdf.Path)
        prims_to_frame: Path(s) to prim(s) to frame (accepts string, Sdf.Path, or list of either)
                       When providing multiple prims, the camera will frame ALL prims at once,
                       not animate between them.
        zoom: Zoom factor (default: 1.0)
              * IMPORTANT: The object is ALWAYS kept in focus regardless of zoom value
              * Zoom IN animation: Start with LARGER zoom value, end with SMALLER zoom value
              * Zoom OUT animation: Start with SMALLER zoom value, end with LARGER zoom value
              * RECOMMENDED: For minor zoom adjustments, use values between 0.9-1.2
        time_code: The time code to use for the framing.
                  When specified, creates a single keyframe at that time.
                  USD automatically handles interpolation between keyframes.
        rotation: Optional rotation to apply (in degrees as [x,y,z])
                 * First value (x): Pitch
                   - Negative values (-20): Camera looks from top to bottom
                   - Positive values (20): Camera looks from bottom to top
                   - -90: Camera aligned with up axis looking straight down
                   - 90: Camera aligned with up axis looking straight up
                   - 0: Camera looking horizontally (perpendicular to up axis)
                 * Second value (y): Yaw - controls LEFT/RIGHT orbiting around object
                 * Third value (z): Roll - tilts camera sideways
                 * RECOMMENDED: For subtle orbiting effects, use small angles between
                   10-15 degrees for orbit (second value)
        align_with_up_vector: Whether to align camera with world up vector (default: True)

    ZOOM EXPLAINED:
        * The object is ALWAYS kept in focus regardless of zoom value
        * ZOOM IN animation (camera moves CLOSER to object):
          ```python
          usdcode.focus_camera(stage, camera_path, shelf_path, zoom=1.2, time_code=0)
          usdcode.focus_camera(stage, camera_path, shelf_path, zoom=0.9, time_code=150)
          ```
          Start with larger value (1.2), end with smaller value (0.9)

        * ZOOM OUT animation (camera moves AWAY from object):
          ```python
          usdcode.focus_camera(stage, camera_path, shelf_path, zoom=0.9, time_code=0)
          usdcode.focus_camera(stage, camera_path, shelf_path, zoom=1.2, time_code=150)
          ```
          Start with smaller value (0.9), end with larger value (1.2)

        * RECOMMENDATION: For natural-looking animations, use values between 0.9-1.2

    ORBIT EXPLAINED:
        * Orbiting means moving the camera in an arc AROUND the object
          while keeping it centered in view and ALWAYS focused
        * rotation = [pitch, yaw, roll] controls this orbit:
          - pitch (first value):
            * Negative (-20): Camera points from top to bottom
            * Positive (20): Camera points from bottom to top
            * -90: Camera aligned with up axis looking straight down
            * 90: Camera aligned with up axis looking straight up
            * 0: Camera looking horizontally
          - yaw (second value): Rotate camera LEFT/RIGHT for orbiting
          - roll (third value): Tilt camera sideways (rarely needed)

        * Example orbit animation (camera rotates from 15 to 30 degrees while pitched down 20 degrees):
          ```python
          usdcode.focus_camera(stage, "/World/Camera", "/World/Sphere", rotation=[-20, 15, 0], time_code=0)
          usdcode.focus_camera(stage, "/World/Camera", "/World/Sphere", rotation=[-20, 30, 0], time_code=100)
          ```
          This keeps the camera pitched 20 degrees downward while smoothly rotating it around the sphere

    IMPORTANT:
        - DO NOT compute or set the camera position manually. The camera position is automatically
          computed by focus_camera to frame the target object(s) according to the zoom and rotation.
        - The zoom parameter controls how close or far the camera is from the object(s):
            - Small zoom = camera is close
            - Large zoom = camera is far
        - The camera will always be positioned so the target(s) are in view, and will look at them.
        - You do NOT need to interpolate or animate the camera position yourself. Just call focus_camera
          at the desired keyframes and USD will interpolate everything automatically.
        - DO NOT add translate/rotate/transform ops to the camera manually if you use focus_camera.
          This will conflict and cause errors.

    Animation Details:
        Each call to focus_camera with a specific time_code creates ONE keyframe.
        To create animations, call focus_camera multiple times with different time_codes.

        USD automatically handles ALL interpolation between keyframes!

        DO NOT:
        - DO NOT Calculate intermediate positions between objects
        - DO NOT Create temporary targets for interpolation
        - DO NOT Manually interpolate anything
        - DO NOT Add extra keyframes beyond start and end (unless you want specific control points)

        Just two calls with different time_codes (start and end) are ALL you need for
        a complete, smooth animation. Any more is usually unnecessary and can cause problems.

    Animation Example (CORRECT USAGE):
        ```python
        # KEYFRAME 1: Frame the sphere at time 0
        usdcode.focus_camera(stage, "/World/Camera", "/World/Sphere", time_code=0)

        # KEYFRAME 2: Frame the cylinder at time 100
        usdcode.focus_camera(stage, "/World/Camera", "/World/Cylinder", time_code=100)

        # That's it! USD handles ALL interpolation between frames automatically!
        # No manual calculation of intermediate positions, no temporary targets needed!
        ```

        This creates a complete 100-frame animation where the camera smoothly
        transitions from focusing on the sphere to focusing on the cylinder.

    Multiple Object Example:
        ```python
        # This will frame BOTH objects at ONCE in a single keyframe at time=1
        usdcode.focus_camera(stage, "/World/Camera", ["/World/Sphere", "/World/Cylinder"],
                    zoom=1.2, time_code=1)
        ```

        The camera will position itself to include both objects in view.
        It does NOT animate between the objects.
    """

    def _euler_to_quat(euler: Gf.Vec3d) -> Gf.Quatd:
        """Utility: convert XYZ Euler (deg) to quaternion (world axes order X,Y,Z)."""
        rx = math.radians(euler[0])
        ry = math.radians(euler[1])
        rz = math.radians(euler[2])
        qx = Gf.Quatd(math.cos(rx / 2), math.sin(rx / 2), 0, 0)
        qy = Gf.Quatd(math.cos(ry / 2), 0, math.sin(ry / 2), 0)
        qz = Gf.Quatd(math.cos(rz / 2), 0, 0, math.sin(rz / 2))
        # Apply rotations in X (pitch) then Y (yaw) then Z (roll) order
        return qz * qy * qx

    def _quat_to_euler(q: Gf.Quatd) -> Gf.Vec3d:
        """Convert quaternion to XYZ Euler angles in degrees."""
        qw, qx, qy, qz = q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]
        # Roll (X)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # Pitch (Y)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)
        # Yaw (Z)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return Gf.Vec3d(math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

    def compute_aligned_rotation_from_euler(
        euler: Gf.Vec3d,
        world_up: Gf.Vec3d = Gf.Vec3d(0, 1, 0),
    ) -> Gf.Vec3d:
        """Return Euler that looks in same direction but horizon aligned.

        Supports arbitrary Euler order (default "YXZ").
        """
        # Build quaternion from input Euler with given order
        q0 = _euler_to_quat(euler)
        # Camera forward (-Z) and up (+Y)
        forward = q0.Transform(Gf.Vec3d(0, 0, -1)).GetNormalized()
        # If forward parallel world_up – undefined roll
        if abs(forward * world_up) > 0.999:
            return Gf.Vec3d(euler[0], euler[1], 0.0)

        # projected up (world_up projected onto plane orthogonal to forward)
        projected = Gf.Cross(Gf.Cross(forward, world_up), forward).GetNormalized()
        side = Gf.Cross(forward, projected).GetNormalized()

        # Build basis -> matrix4
        m = Gf.Matrix4d(
            side[0],
            side[1],
            side[2],
            0,
            projected[0],
            projected[1],
            projected[2],
            0,
            -forward[0],
            -forward[1],
            -forward[2],
            0,
            0,
            0,
            0,
            1,
        )
        quat = m.ExtractRotationQuat()
        result = _quat_to_euler(quat)

        return result

    def _ensure_rotation_op_order_xyz(stage: Usd.Stage, prim_path: str) -> None:
        """Ensure the given prim has a rotation XformOp in XYZ order.

        If the prim already contains a ``xformOp:rotateXYZ`` attribute nothing is done. If it
        instead contains another rotation order (e.g. ``xformOp:rotateYXZ``) the helper will
        convert it to XYZ using the *ChangeRotationOp* omni.kit command.
        """
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        # Find the first rotation op attribute on the prim.
        src_attr_name: Optional[str] = None
        for attr in prim.GetAttributes():
            name = attr.GetName()
            # Interested only in rotation ops that follow the pattern xformOp:rotate<ORDER>
            if name.startswith("xformOp:rotate"):
                src_attr_name = name  # keep the first one found
                break

        # Nothing to do if no rotation op at all
        if src_attr_name is None:
            return

        # Already XYZ – nothing to change
        if src_attr_name == "xformOp:rotateXYZ":
            return

        # Extract the order suffix (part after "xformOp:rotate") to build the attribute path.
        # Defensive guard in case the attribute name is malformed and shorter than expected
        if len(src_attr_name) <= len("xformOp:rotate"):
            return

        # Build full attribute path (prim path + "." + attribute name)
        src_op_attr_path = f"{prim_path}.{src_attr_name}"

        # Execute the command to convert the rotation op order to XYZ.
        try:
            omni.kit.commands.execute(
                "ChangeRotationOp",
                src_op_attr_path=src_op_attr_path,
                op_name=src_attr_name,
                dst_op_attr_name="xformOp:rotateXYZ",
                is_inverse_op=False,
                auto_target_layer=True,
            )
        except Exception:
            # Fail silently – the command might not exist in some runtimes.
            pass

    def _convert_to_gf_vec3d(rot: Union[Tuple[float, float, float], Sequence[float], Gf.Vec3d, Gf.Vec3f]) -> Gf.Vec3d:
        """
        Convert different rotation formats to Gf.Vec3d

        Args:
            rot: Rotation in degrees as tuple, list, or Gf.Vec3

        Returns:
            Gf.Vec3d: Rotation as Gf.Vec3d
        """
        if isinstance(rot, Gf.Vec3d):
            return rot
        elif isinstance(rot, Gf.Vec3f):
            return Gf.Vec3d(rot[0], rot[1], rot[2])
        elif isinstance(rot, (tuple, list)) and len(rot) >= 3:
            return Gf.Vec3d(rot[0], rot[1], rot[2])
        else:
            raise TypeError(f"Cannot convert {type(rot)} to Gf.Vec3d")

    # If we are working with a non-default time code make sure that the camera xform
    # already contains at least one time sample (at t = 0). Some Kit commands will
    # silently skip setting keyed values on an attribute that has no existing
    # time samples. By pre-seeding each xformOp with its default value at time 0
    # we guarantee that subsequent commands executed with a specific ``time_code``
    # properly author a new keyed sample.

    def _ensure_xform_sample_at_zero(stage: Usd.Stage, prim_path: str) -> None:
        """If the given prim has xformOps without time samples, write one at t=0.

        This helper iterates over all the xformOps of *prim_path* and, for any
        op that currently has **no** time samples, it authors a sample at
        ``Usd.TimeCode(0)`` with the op's default value.  This is required
        because a number of omni.kit transform commands will refuse to author
        a keyed value on an attribute that has never been time-sampled before.
        """

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return

        for op in xformable.GetOrderedXformOps():
            # Skip ops that already have samples
            if op.GetNumTimeSamples() > 0:
                continue

            attr = op.GetAttr()
            if not attr:
                continue

            default_val = attr.Get()
            # Author the default value at time 0 so that the attribute is
            # considered animated and additional samples can be set later.
            if default_val is not None:
                attr.Set(default_val, Usd.TimeCode(0))

    # Normalize camera_path to string
    if isinstance(camera_path, Sdf.Path):
        camera_path = str(camera_path)

    # Normalize prims_to_frame into a list of strings
    if isinstance(prims_to_frame, (str, Sdf.Path)):
        prims_to_frame = [prims_to_frame]
    # At this point it is guaranteed to be a list; convert all to strings
    prims_to_frame = [str(p) if isinstance(p, Sdf.Path) else p for p in prims_to_frame]

    # Throw exception if prims_to_frame is empty
    if not prims_to_frame:
        raise ValueError("prims_to_frame cannot be empty")

    if isinstance(time_code, int):
        time_code = Usd.TimeCode(time_code)
    elif isinstance(time_code, float):
        time_code = Usd.TimeCode(time_code)

    # Ensure the stage's time range encompasses the requested keyframe
    if not time_code.IsDefault():
        tc_val = time_code.GetValue()

        try:
            root_layer = stage.GetRootLayer()
            start_tc = getattr(root_layer, "startTimeCode", None)
            if start_tc < 0 and tc_val == 0:
                start_tc = None
            end_tc = getattr(root_layer, "endTimeCode", None)
            if end_tc > 100 and tc_val > 100:
                end_tc = None

            if start_tc is None or tc_val < start_tc:
                root_layer.startTimeCode = tc_val
            if end_tc is None or tc_val > end_tc:
                root_layer.endTimeCode = tc_val
        except Exception:
            pass

    # Determine the stage's up axis so camera alignment uses the correct world up vector
    up_axis_token = UsdGeom.GetStageUpAxis(stage)
    if up_axis_token == UsdGeom.Tokens.z:
        world_up_vec = Gf.Vec3d(0, 0, 1)
    else:  # default to Y-up
        world_up_vec = Gf.Vec3d(0, 1, 0)

    # Seed samples only when authoring to a specific time code (animation).
    if not time_code.IsDefault():
        _ensure_xform_sample_at_zero(stage, camera_path)

    # Determine final rotation BEFORE framing
    if rotation is not None:
        # Make sure the camera prim rotates in XYZ order before any further processing
        _ensure_rotation_op_order_xyz(stage, camera_path)

        rot_vec_user = _convert_to_gf_vec3d(rotation)

        if up_axis_token == UsdGeom.Tokens.z:
            rot_vec = Gf.Vec3d(rot_vec_user[1], rot_vec_user[0] + 90.0, rot_vec_user[2])
        else:
            rot_vec = rot_vec_user

        if align_with_up_vector:
            final_rotation = compute_aligned_rotation_from_euler(rot_vec, world_up_vec)
        else:
            final_rotation = rot_vec
        omni.kit.commands.execute(
            "TransformPrimSRT", path=camera_path, new_rotation_euler=final_rotation, time_code=time_code
        )

    # Now frame the prims (after orientation set)
    omni.kit.commands.execute(
        "FramePrimsCommand", prim_to_move=camera_path, prims_to_frame=prims_to_frame, zoom=zoom, time_code=time_code
    )

    set_current_viewport_camera(stage, camera_path)


# -----------------------------------------------------------------------------
# Viewport camera helpers
# -----------------------------------------------------------------------------


def get_current_viewport_camera_path(stage: Usd.Stage) -> Optional[str]:
    """Return the camera prim path currently used by the active viewport.

    If no viewport window is active or it has no camera assigned, returns ``None``.
    """
    if not VIEWPORT_AVAILABLE:
        return None

    try:
        return get_active_viewport_camera_path()
    except Exception:
        # Utility may raise if no viewport; just return None in that case
        return None


def set_current_viewport_camera(stage: Usd.Stage, camera_path: Union[str, Sdf.Path]) -> bool:
    """Set the given camera as the active camera in the current viewport.

    Args:
        camera_path: Path to the camera prim (str or Sdf.Path).

    Returns:
        bool: ``True`` if the camera was successfully set, ``False`` otherwise.
    """
    if not VIEWPORT_AVAILABLE:
        return False

    # Normalize to string
    if isinstance(camera_path, Sdf.Path):
        camera_path = str(camera_path)

    # Helper to fetch viewport api
    def _get_viewport_api():
        viewport_window = get_active_viewport_window()
        if viewport_window:
            return viewport_window.viewport_api
        return None

    viewport_api = _get_viewport_api()

    if viewport_api is None:
        return False

    try:
        omni.kit.commands.execute("SetViewportCamera", camera_path=camera_path, viewport_api=viewport_api)
        return True
    except Exception:
        return False
