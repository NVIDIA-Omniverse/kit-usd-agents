## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdShade, UsdVol

from .add_op import *


def change_cylinder_light_radius(stage: Usd.Stage, light_path: str, radius: float):
    """Change the radius of a cylinder light.

    Args:
        stage (Usd.Stage): The USD stage.
        light_path (str): The path to the cylinder light prim.
        radius (float): The new radius value.

    Raises:
        ValueError: If the prim at the given path is not a valid cylinder light.
    """
    prim = stage.GetPrimAtPath(light_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {light_path}")
    cylinder_light = UsdLux.CylinderLight(prim)
    if not cylinder_light:
        raise ValueError(f"Prim at path {light_path} is not a valid CylinderLight")
    radius_attr = cylinder_light.GetRadiusAttr()
    if radius_attr:
        radius_attr.Set(radius)
    else:
        cylinder_light.CreateRadiusAttr(radius)


def batch_create_cylinder_lights(
    stage: Usd.Stage,
    prim_paths: List[str],
    radii: List[float],
    lengths: List[float],
    colors: List[Tuple[float, float, float]],
) -> List[UsdLux.CylinderLight]:
    """
    Create multiple UsdLux.CylinderLight prims in a batch.

    Args:
        stage (Usd.Stage): The stage to create the lights on.
        prim_paths (List[str]): A list of prim paths where the lights should be created.
        radii (List[float]): A list of radius values for the lights.
        lengths (List[float]): A list of length values for the lights.
        colors (List[Tuple[float, float, float]]): A list of color tuples for the lights.

    Returns:
        List[UsdLux.CylinderLight]: A list of the created UsdLux.CylinderLight prims.

    Raises:
        ValueError: If the lengths of prim_paths, radii, lengths, and colors are not equal.
    """
    if not len(prim_paths) == len(radii) == len(lengths) == len(colors):
        raise ValueError("The lengths of prim_paths, radii, lengths, and colors must be equal.")
    cylinder_lights = []
    for prim_path, radius, length, color in zip(prim_paths, radii, lengths, colors):
        cylinder_light = UsdLux.CylinderLight.Define(stage, Sdf.Path(prim_path))
        cylinder_light.CreateRadiusAttr().Set(radius)
        cylinder_light.CreateLengthAttr().Set(length)
        cylinder_light.CreateColorAttr().Set(Gf.Vec3f(*color))
        cylinder_lights.append(cylinder_light)
    return cylinder_lights


def toggle_cylinder_light_treat_as_line(light: UsdLux.CylinderLight) -> None:
    """Toggle the treatAsLine attribute of a UsdLuxCylinderLight.

    Args:
        light (UsdLux.CylinderLight): The cylinder light to modify.

    Raises:
        ValueError: If the input is not a valid UsdLuxCylinderLight.
    """
    if not light or not isinstance(light, UsdLux.CylinderLight):
        raise ValueError("Input must be a valid UsdLuxCylinderLight")
    treat_as_line_attr = light.GetTreatAsLineAttr()
    if not treat_as_line_attr:
        treat_as_line_attr = light.CreateTreatAsLineAttr(False, True)
    current_value = treat_as_line_attr.Get()
    treat_as_line_attr.Set(not current_value)


def create_and_configure_cylinder_light(
    stage: Usd.Stage, prim_path: str, length: float, radius: float, treat_as_line: bool
) -> UsdLux.CylinderLight:
    """Create and configure a UsdLuxCylinderLight prim.

    Args:
        stage (Usd.Stage): The USD stage to create the light on.
        prim_path (str): The path where the light prim should be created.
        length (float): The length of the cylinder light.
        radius (float): The radius of the cylinder light.
        treat_as_line (bool): Whether to treat the cylinder light as a line light.

    Returns:
        UsdLux.CylinderLight: The created cylinder light prim.
    """
    cylinder_light = UsdLux.CylinderLight.Define(stage, prim_path)
    length_attr = cylinder_light.CreateLengthAttr(length, True)
    length_attr.Set(length)
    radius_attr = cylinder_light.CreateRadiusAttr(radius, True)
    radius_attr.Set(radius)
    treat_as_line_attr = cylinder_light.CreateTreatAsLineAttr(treat_as_line, True)
    treat_as_line_attr.Set(treat_as_line)
    return cylinder_light


def get_cylinder_light_parameters(stage: Usd.Stage, prim_path: str) -> Tuple[float, float, bool]:
    """Get the parameters of a cylinder light.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the cylinder light prim.

    Returns:
        A tuple containing (length, radius, treatAsLine).
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    light = UsdLux.CylinderLight(prim)
    if not light:
        raise ValueError(f"Prim at path {prim_path} is not a cylinder light.")
    length_attr = light.GetLengthAttr()
    if length_attr.HasAuthoredValue():
        length = length_attr.Get()
    else:
        length = 1.0
    radius_attr = light.GetRadiusAttr()
    if radius_attr.HasAuthoredValue():
        radius = radius_attr.Get()
    else:
        radius = 0.5
    treat_as_line_attr = light.GetTreatAsLineAttr()
    if treat_as_line_attr.HasAuthoredValue():
        treat_as_line = treat_as_line_attr.Get()
    else:
        treat_as_line = False
    return (length, radius, treat_as_line)


def copy_cylinder_light(stage: Usd.Stage, source_path: str, dest_path: str) -> UsdLux.CylinderLight:
    """Copy a UsdLuxCylinderLight from one path to another."""
    source_prim = stage.GetPrimAtPath(source_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim at path {source_path} does not exist.")
    if not UsdLux.CylinderLight(source_prim):
        raise TypeError(f"Prim at path {source_path} is not a UsdLuxCylinderLight.")
    dest_prim = stage.GetPrimAtPath(dest_path)
    if not dest_prim.IsValid():
        dest_prim = UsdLux.CylinderLight.Define(stage, dest_path).GetPrim()
    elif not UsdLux.CylinderLight(dest_prim):
        raise TypeError(f"Prim at path {dest_path} is not a UsdLuxCylinderLight.")
    for attr in source_prim.GetAttributes():
        if attr.HasValue():
            attr_name = attr.GetName()
            attr_value = attr.Get()
            dest_prim.GetAttribute(attr_name).Set(attr_value)
    return UsdLux.CylinderLight(dest_prim)


def set_cylinder_light_parameters(
    cylinder_light: UsdLux.CylinderLight, length: float, radius: float, treat_as_line: bool
) -> None:
    """Set the parameters of a cylinder light.

    Args:
        cylinder_light (UsdLux.CylinderLight): The cylinder light to set parameters for.
        length (float): The length of the cylinder light.
        radius (float): The radius of the cylinder light.
        treat_as_line (bool): Whether to treat the cylinder light as a line light.

    Raises:
        ValueError: If the input cylinder_light is not a valid UsdLux.CylinderLight prim.
    """
    if not cylinder_light or not isinstance(cylinder_light, UsdLux.CylinderLight):
        raise ValueError("Invalid UsdLux.CylinderLight prim.")
    length_attr = cylinder_light.GetLengthAttr()
    if length_attr:
        length_attr.Set(length)
    radius_attr = cylinder_light.GetRadiusAttr()
    if radius_attr:
        radius_attr.Set(radius)
    treat_as_line_attr = cylinder_light.GetTreatAsLineAttr()
    if treat_as_line_attr:
        treat_as_line_attr.Set(treat_as_line)


def align_cylinder_lights(lights: List[UsdLux.CylinderLight]):
    """Align cylinder lights to the Z-axis and scale them to unit length."""
    for light in lights:
        prim = light.GetPrim()
        length = light.GetLengthAttr().Get()
        scale_factor = 1.0 / length if length > 0 else 1.0
        xformable = UsdGeom.Xformable(prim)
        rotate_op = xformable.AddRotateXOp(opSuffix="align")
        rotate_op.Set(90.0)
        scale_op = add_scale_op(xformable, opSuffix="unitLength")
        scale_op.Set(Gf.Vec3f(scale_factor))


def visualize_cylinder_light_distribution(stage: Usd.Stage, light_path: str, num_samples: int = 10) -> None:
    """
    Visualize the light distribution of a cylinder light by creating sphere prims at sample points.

    Args:
        stage (Usd.Stage): The USD stage to add the visualization prims to.
        light_path (str): The path to the cylinder light prim.
        num_samples (int, optional): The number of sample points along the length of the cylinder. Defaults to 10.
    """
    light_prim = stage.GetPrimAtPath(light_path)
    if not light_prim.IsValid():
        raise ValueError(f"Invalid light prim path: {light_path}")
    cylinder_light = UsdLux.CylinderLight(light_prim)
    length = cylinder_light.GetLengthAttr().Get()
    radius = cylinder_light.GetRadiusAttr().Get()
    sample_positions = [Gf.Vec3f(length * (i / (num_samples - 1)) - length / 2, 0, 0) for i in range(num_samples)]
    for i, position in enumerate(sample_positions):
        sphere_path = Sdf.Path(light_path).AppendPath(f"VisSphere_{i}")
        sphere_prim = stage.DefinePrim(sphere_path, "Sphere")
        sphere_prim.GetAttribute("radius").Set(radius * 0.1)
        add_translate_op(UsdGeom.Xform(sphere_prim)).Set(position)


def find_and_replace_cylinder_lights(stage: Usd.Stage) -> int:
    """Find all CylinderLight prims on the stage and replace them with SphereLight prims.

    Args:
        stage (Usd.Stage): The stage to search for CylinderLight prims.

    Returns:
        int: The number of CylinderLight prims replaced.
    """
    cylinder_lights = [p for p in stage.Traverse() if p.IsA(UsdLux.CylinderLight)]
    num_replaced = 0
    for cylinder_light_prim in cylinder_lights:
        prim_path = cylinder_light_prim.GetPath()
        parent_path = prim_path.GetParentPath()
        prim_name = prim_path.name
        cylinder_light = UsdLux.CylinderLight(cylinder_light_prim)
        radius = cylinder_light.GetRadiusAttr().Get()
        treat_as_line = cylinder_light.GetTreatAsLineAttr().Get()
        stage.RemovePrim(prim_path)
        sphere_light = UsdLux.SphereLight.Define(stage, parent_path.AppendChild(prim_name))
        if radius is not None:
            sphere_light.CreateRadiusAttr(radius)
        if treat_as_line is not None:
            sphere_light.CreateTreatAsPointAttr(treat_as_line)
        num_replaced += 1
    return num_replaced


def get_disk_light_radius(disk_light: UsdLux.DiskLight) -> float:
    """Get the radius of a disk light.

    Args:
        disk_light (UsdLux.DiskLight): The disk light prim.

    Returns:
        float: The radius of the disk light. If no radius is authored, returns the default value of 0.5.

    Raises:
        ValueError: If the input prim is not a valid UsdLuxDiskLight.
    """
    if not disk_light or not disk_light.GetPrim().IsValid():
        raise ValueError("Input prim is not a valid UsdLuxDiskLight")
    radius_attr = disk_light.GetRadiusAttr()
    if radius_attr.HasValue():
        return radius_attr.Get()
    else:
        return radius_attr.GetDefaultValue()


def set_disk_light_radius(disk_light: UsdLux.DiskLight, radius: float):
    """Set the radius of a disk light.

    Args:
        disk_light (UsdLux.DiskLight): The disk light prim.
        radius (float): The radius value to set.

    Raises:
        ValueError: If the provided prim is not a valid UsdLux.DiskLight.
        TypeError: If the radius is not a valid float value.
    """
    if not disk_light or not isinstance(disk_light, UsdLux.DiskLight):
        raise ValueError("Invalid UsdLux.DiskLight prim.")
    if not isinstance(radius, float):
        raise TypeError("Radius must be a float value.")
    radius_attr = disk_light.GetRadiusAttr()
    radius_attr.Set(radius)


def find_disk_lights_by_radius_range(stage: Usd.Stage, min_radius: float, max_radius: float) -> List[UsdLux.DiskLight]:
    """
    Find all UsdLuxDiskLight prims on the given stage whose radius is within the specified range.

    Args:
        stage (Usd.Stage): The stage to search for disk lights.
        min_radius (float): The minimum radius value (inclusive).
        max_radius (float): The maximum radius value (inclusive).

    Returns:
        List[UsdLux.DiskLight]: A list of UsdLuxDiskLight prims whose radius is within the specified range.
    """
    if min_radius < 0:
        raise ValueError("min_radius must be non-negative")
    if max_radius < min_radius:
        raise ValueError("max_radius must be greater than or equal to min_radius")
    disk_lights = []
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DiskLight):
            disk_lights.append(UsdLux.DiskLight(prim))
    result = []
    for disk_light in disk_lights:
        radius_attr = disk_light.GetRadiusAttr()
        if radius_attr.HasValue():
            radius = radius_attr.Get()
            if min_radius <= radius <= max_radius:
                result.append(disk_light)
    return result


def copy_disk_light(
    source_stage: Usd.Stage, source_light_path: str, dest_stage: Usd.Stage, dest_light_path: str
) -> UsdLux.DiskLight:
    """Copy a DiskLight prim from one stage to another."""
    source_light = UsdLux.DiskLight(source_stage.GetPrimAtPath(source_light_path))
    if not source_light:
        raise ValueError(f"Prim at path {source_light_path} is not a valid DiskLight.")
    dest_light = UsdLux.DiskLight.Define(dest_stage, dest_light_path)
    attrs_to_copy = ["radius", "color", "intensity", "exposure", "diffuse", "specular", "normalize"]
    for attr_name in attrs_to_copy:
        source_attr = source_light.GetPrim().GetAttribute(attr_name)
        if source_attr.HasValue():
            attr_value = source_attr.Get()
            dest_attr = dest_light.CreateInput(attr_name, source_attr.GetTypeName())
            dest_attr.Set(attr_value)
    source_xform = UsdGeom.Xformable(source_light.GetPrim())
    dest_xform = UsdGeom.Xformable(dest_light.GetPrim())
    if source_xform and dest_xform:
        source_xform_ops = source_xform.GetOrderedXformOps()
        for op in source_xform_ops:
            op_type = op.GetOpType()
            op_value = op.Get()
            dest_xform.AddXformOp(op_type).Set(op_value)
    return dest_light


def set_light_color(
    light: UsdLux.NonboundableLightBase, color: Gf.Vec3f, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the color attribute of a light.

    Args:
        light (UsdLux.NonboundableLightBase): The light prim.
        color (Gf.Vec3f): The color value to set.
        time_code (Usd.TimeCode, optional): The time code at which to set the value. Defaults to Default.

    Raises:
        ValueError: If the provided prim is not a valid light prim.
    """
    if not light:
        raise ValueError("Invalid light prim.")
    color_attr = light.GetColorAttr()
    if not color_attr:
        color_attr = light.CreateColorAttr(color, time_code)
    else:
        color_attr.Set(color, time_code)


def set_light_intensity(light_prim: UsdLux.LightAPI, intensity: float) -> None:
    """Set the intensity of a light.

    Args:
        light_prim (UsdLux.LightAPI): The light prim to set intensity for.
        intensity (float): The intensity value to set.

    Raises:
        ValueError: If the provided prim is not a valid light prim.
    """
    if not light_prim:
        raise ValueError("Invalid light prim provided.")
    intensity_attr = light_prim.GetIntensityAttr()
    if not intensity_attr:
        intensity_attr = light_prim.CreateIntensityAttr(intensity, writeSparsely=True)
    else:
        intensity_attr.Set(intensity)


def set_light_exposure(light: UsdLux.NonboundableLightBase, exposure: float):
    """Set the exposure attribute value for a light.

    Args:
        light (UsdLux.NonboundableLightBase): The light prim.
        exposure (float): The exposure value to set.

    Raises:
        ValueError: If the input light prim is not valid.
    """
    if not light:
        raise ValueError("Invalid light prim.")
    exposure_attr = light.GetExposureAttr()
    if exposure_attr:
        exposure_attr.Set(exposure)
    else:
        exposure_attr = light.CreateExposureAttr(exposure, True)


def set_light_specular(light: UsdLux.NonboundableLightBase, specular: float):
    """Sets the specular attribute of a non-boundable light.

    Args:
        light (UsdLux.NonboundableLightBase): The non-boundable light prim.
        specular (float): The specular value to set.

    Raises:
        ValueError: If the input light prim is not valid or if the specular value is negative.
    """
    if not light.GetPrim().IsValid():
        raise ValueError("Invalid light prim.")
    if specular < 0.0:
        raise ValueError("Specular value must be non-negative.")
    specular_attr = light.GetSpecularAttr()
    specular_attr.Set(specular)


def set_light_normalize(light: UsdLux.BoundableLightBase, normalize: bool) -> None:
    """Set the normalize attribute on a light.

    Args:
        light (UsdLux.BoundableLightBase): The light to set the normalize attribute on.
        normalize (bool): The value to set the normalize attribute to.

    Raises:
        ValueError: If the given prim is not a valid UsdLux.BoundableLightBase.
    """
    if not light:
        raise ValueError(f"The given prim is not a valid UsdLux.BoundableLightBase")
    attr = light.GetNormalizeAttr()
    attr.Set(normalize)


def enable_light_color_temperature(stage: Usd.Stage, light_path: str, enable: bool) -> None:
    """Enable or disable color temperature on a light.

    Parameters:
        stage (Usd.Stage): The stage containing the light prim.
        light_path (str): The path to the light prim.
        enable (bool): Whether to enable or disable color temperature.

    Raises:
        ValueError: If the prim at light_path is not a valid light.
    """
    light_prim = stage.GetPrimAtPath(light_path)
    if not light_prim.IsValid():
        raise ValueError(f"No prim found at path {light_path}")
    light = UsdLux.LightAPI(light_prim)
    if not light:
        raise ValueError(f"Prim at {light_path} is not a valid Light")
    enable_attr = light.GetEnableColorTemperatureAttr()
    if not enable_attr:
        enable_attr = light.CreateEnableColorTemperatureAttr(False, True)
    enable_attr.Set(enable)


def get_light_color(stage: Usd.Stage, light_prim_path: str) -> Gf.Vec3f:
    """Get the color of a light prim.

    Args:
        stage (Usd.Stage): The USD stage.
        light_prim_path (str): The path to the light prim.

    Returns:
        Gf.Vec3f: The color of the light as a Vec3f. Returns (1.0, 1.0, 1.0) if no color is authored.

    Raises:
        ValueError: If the prim at the given path is not a valid light prim.
    """
    light_prim = stage.GetPrimAtPath(light_prim_path)
    if not light_prim:
        raise ValueError(f"No prim found at path {light_prim_path}")
    light = UsdLux.LightAPI(light_prim)
    if not light:
        raise ValueError(f"Prim at path {light_prim_path} is not a valid Light prim")
    color_attr = light.GetColorAttr()
    if color_attr.HasAuthoredValue():
        return color_attr.Get()
    else:
        return Gf.Vec3f(1.0)


def get_light_intensity(light: UsdLux.LightAPI) -> float:
    """Get the effective intensity of a light, considering exposure and normalize attributes."""
    intensity_attr = light.GetIntensityAttr()
    if not intensity_attr:
        return 1.0
    intensity = intensity_attr.Get(1.0)
    exposure_attr = light.GetExposureAttr()
    if exposure_attr:
        exposure = exposure_attr.Get(0.0)
        intensity *= 2.0**exposure
    normalize_attr = light.GetNormalizeAttr()
    if normalize_attr:
        normalize = normalize_attr.Get(False)
        if normalize:
            surface_area = 1.0
            intensity /= surface_area
    return intensity


def get_light_color_temperature(light_prim: UsdLux.LightAPI) -> float:
    """Get the color temperature value of a light.

    Args:
        light_prim (UsdLux.LightAPI): The light prim.

    Returns:
        float: The color temperature value of the light. If color temperature is
            not enabled or not authored, returns None.

    Raises:
        ValueError: If the input prim is not a valid UsdLux.LightAPI.
    """
    if not light_prim or not isinstance(light_prim, UsdLux.LightAPI):
        raise ValueError("Input prim must be a valid UsdLux.LightAPI")
    if not light_prim.GetEnableColorTemperatureAttr().Get():
        return None
    color_temp_attr = light_prim.GetColorTemperatureAttr()
    if color_temp_attr.HasAuthoredValue():
        return color_temp_attr.Get()
    else:
        return None


def get_light_specular(light: UsdLux.BoundableLightBase) -> float:
    """Get the specular value for a light.

    Args:
        light (UsdLux.BoundableLightBase): The light to get the specular value from.

    Returns:
        float: The specular value of the light, or 0.0 if not set.
    """
    specular_attr = light.GetSpecularAttr()
    if specular_attr.IsValid():
        specular_val = specular_attr.Get()
        if specular_val is not None:
            return specular_val
    return 0.0


def duplicate_light(
    light: UsdLux.BoundableLightBase, name: str, translate: Tuple[float, float, float] = (0, 0, 0)
) -> UsdLux.BoundableLightBase:
    """Duplicate a light with optional translation."""
    if not light or not light.GetPrim().IsValid():
        raise ValueError("Invalid light input.")
    if not name:
        raise ValueError("Name cannot be empty.")
    stage = light.GetPrim().GetStage()
    parent_path = light.GetPath().GetParentPath()
    new_light_path = parent_path.AppendChild(name)
    existing_prim = stage.GetPrimAtPath(new_light_path)
    if existing_prim.IsValid():
        raise ValueError(f"A prim already exists at path: {new_light_path}")
    new_light = light.Define(stage, new_light_path)
    attrs_to_copy = [
        "color",
        "colorTemperature",
        "diffuse",
        "enableColorTemperature",
        "exposure",
        "intensity",
        "normalize",
        "specular",
    ]
    for attr_name in attrs_to_copy:
        src_attr = light.GetPrim().GetAttribute(attr_name)
        if src_attr.IsValid():
            src_value = src_attr.Get()
            new_attr = new_light.GetPrim().CreateAttribute(attr_name, src_attr.GetTypeName())
            new_attr.Set(src_value)
    filters_rel = light.GetFiltersRel()
    if filters_rel.IsValid():
        new_filters_rel = new_light.GetFiltersRel()
        for target in filters_rel.GetTargets():
            new_filters_rel.AddTarget(target)
    UsdGeom.XformCommonAPI(new_light).SetTranslate(translate)
    return new_light


def batch_update_light_attributes(stage: Usd.Stage, light_paths: List[str], attributes: Dict[str, Any]) -> None:
    """Updates attributes on a batch of light prims.

    Args:
        stage (Usd.Stage): The USD stage.
        light_paths (List[str]): List of paths to light prims.
        attributes (Dict[str, Any]): Dictionary of attribute names and values to set.

    Raises:
        ValueError: If any light path is invalid or not of type UsdLuxNonboundableLightBase.
    """
    for light_path in light_paths:
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            raise ValueError(f"Invalid light prim path: {light_path}")
        if not UsdLux.NonboundableLightBase(light_prim):
            raise ValueError(f"Prim at path {light_path} is not of type UsdLuxNonboundableLightBase")
        for attr_name, attr_value in attributes.items():
            attr = light_prim.GetAttribute(attr_name)
            if attr.IsValid():
                attr.Set(attr_value)


def get_all_lights(stage: Usd.Stage) -> List[UsdLux.BoundableLightBase]:
    """
    Get all the lights in the given USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search for lights.

    Returns:
        List[UsdLux.BoundableLightBase]: A list of all the lights found in the stage.
    """
    lights = []
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.BoundableLightBase):
            light = UsdLux.BoundableLightBase(prim)
            lights.append(light)
    return lights


def is_light_color_temperature_enabled(light: UsdLux.BoundableLightBase) -> bool:
    """Check if color temperature is enabled for a light.

    Args:
        light (UsdLux.BoundableLightBase): The light to check.

    Returns:
        bool: True if color temperature is enabled, False otherwise.
    """
    enable_attr = light.GetEnableColorTemperatureAttr()
    if enable_attr.HasAuthoredValue():
        enabled = enable_attr.Get()
        return bool(enabled)
    elif enable_attr.HasFallbackValue():
        enabled = enable_attr.Get()
        return bool(enabled)
    else:
        return False


def create_light_with_attributes(
    stage: Usd.Stage,
    light_type: str,
    light_path: str,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity: float = 1.0,
    exposure: float = 0.0,
    normalize: bool = False,
    diffuse: float = 1.0,
    specular: float = 1.0,
    colorTemp: float = 6500.0,
    enableColorTemp: bool = False,
) -> UsdLux.LightAPI:
    """Create a light with the given attributes."""
    if light_type not in ["SphereLight", "DiskLight", "DistantLight", "DomeLight", "RectLight", "CylinderLight"]:
        raise ValueError(f"Invalid light type: {light_type}")
    light_prim = stage.DefinePrim(light_path, "Light")
    light_api = UsdLux.LightAPI(light_prim)
    light = eval(f"UsdLux.{light_type}(light_prim)")
    light_api.CreateColorAttr().Set(Gf.Vec3f(*color))
    light_api.CreateIntensityAttr().Set(intensity)
    light_api.CreateExposureAttr().Set(exposure)
    light_api.CreateNormalizeAttr().Set(normalize)
    light_api.CreateDiffuseAttr().Set(diffuse)
    light_api.CreateSpecularAttr().Set(specular)
    if enableColorTemp:
        light_api.CreateEnableColorTemperatureAttr().Set(True)
        light_api.CreateColorTemperatureAttr().Set(colorTemp)
    else:
        light_api.CreateEnableColorTemperatureAttr().Set(False)
    return light


def find_lights_by_type(stage: Usd.Stage, light_type: str) -> List[Usd.Prim]:
    """
    Find all lights of a specific type in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search for lights.
        light_type (str): The type of light to search for (e.g., "RectLight", "DiskLight").

    Returns:
        List[Usd.Prim]: A list of Usd.Prim objects representing the found lights.
    """
    light_schema = getattr(UsdLux, light_type, None)
    if light_schema is None:
        raise ValueError(f"Invalid light type: {light_type}")
    found_lights = []
    for prim in stage.Traverse():
        if prim.IsA(light_schema):
            found_lights.append(prim)
    return found_lights


def create_prim(stage: Usd.Stage, prim_type: str, prim_path: str) -> Usd.Prim:
    """Create a new prim at the given path with the specified type."""
    return stage.DefinePrim(prim_path, prim_type)


def align_lights_to_target(stage: Usd.Stage, light_paths: List[str], target_path: str):
    """Align the lights to point at the target prim.

    Args:
        stage (Usd.Stage): The USD stage.
        light_paths (List[str]): List of paths to the light prims.
        target_path (str): Path to the target prim.
    """
    target_prim = stage.GetPrimAtPath(target_path)
    if not target_prim.IsValid():
        raise ValueError(f"Target prim at path {target_path} does not exist.")
    target_xformable = UsdGeom.Xformable(target_prim)
    target_world_xform = target_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    target_world_pos = target_world_xform.ExtractTranslation()
    for light_path in light_paths:
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            print(f"Warning: Light prim at path {light_path} does not exist. Skipping.")
            continue
        light_xformable = UsdGeom.Xformable(light_prim)
        if not light_xformable:
            print(f"Warning: Light prim at path {light_path} is not transformable. Skipping.")
            continue
        light_world_xform = light_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        light_world_pos = light_world_xform.ExtractTranslation()
        dir_to_target = target_world_pos - light_world_pos
        dir_to_target.Normalize()
        rotation = Gf.Rotation()
        rotation.SetRotateInto((0, 0, -1), Gf.Vec3d(dir_to_target))
        add_rotate_xyz_op(light_xformable).Set(
            Gf.Vec3f(rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1)))
        )


def get_light_diffuse(light: UsdLux.LightAPI) -> float:
    """Get the diffuse value for a light, with error handling and default value."""
    if not light:
        raise ValueError("Invalid light provided.")
    diffuse_attr = light.GetDiffuseAttr()
    if not diffuse_attr:
        return 1.0
    diffuse_value = diffuse_attr.Get(Usd.TimeCode.Default())
    if diffuse_value is None:
        return 1.0
    return diffuse_value


def set_distant_light_angle(stage: Usd.Stage, prim_path: str, angle: float) -> None:
    """Set the angle attribute of a distant light.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the distant light prim.
        angle (float): The angle value to set in degrees.

    Raises:
        ValueError: If the prim at the given path is not a valid distant light.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {prim_path}")
    distant_light = UsdLux.DistantLight(prim)
    if not distant_light:
        raise ValueError(f"Prim at path {prim_path} is not a valid DistantLight")
    angle_attr = distant_light.GetAngleAttr()
    if not angle_attr:
        raise ValueError(f"Unable to get angle attribute on distant light at path: {prim_path}")
    angle_attr.Set(angle)


def get_distant_light_angle(stage: Usd.Stage, light_path: str) -> float:
    """Get the angle attribute value of a distant light.

    Args:
        stage (Usd.Stage): The USD stage.
        light_path (str): The path to the distant light prim.

    Returns:
        float: The angle attribute value in degrees.

    Raises:
        ValueError: If the prim at the given path is not a valid distant light.
    """
    prim = stage.GetPrimAtPath(light_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {light_path}")
    if not prim.IsA(UsdLux.DistantLight):
        raise ValueError(f"Prim at path {light_path} is not a UsdLuxDistantLight")
    distant_light = UsdLux.DistantLight(prim)
    angle_attr = distant_light.GetAngleAttr()
    if not angle_attr.HasValue():
        return angle_attr.GetDefaultValue()
    return angle_attr.Get()


def set_distant_light_intensity(stage: Usd.Stage, prim_path: str, intensity: float) -> None:
    """Set the intensity of a distant light.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the distant light prim.
        intensity (float): The intensity value to set.

    Raises:
        ValueError: If the prim at the given path is not a valid distant light.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {prim_path}")
    if not UsdLux.DistantLight(prim):
        raise ValueError(f"Prim at path {prim_path} is not a valid DistantLight")
    distant_light = UsdLux.DistantLight(prim)
    intensity_attr = distant_light.GetIntensityAttr()
    if intensity_attr:
        intensity_attr.Set(intensity)
    else:
        distant_light.CreateIntensityAttr(intensity)


def update_distant_light_properties(stage: Usd.Stage, prim_path: str, angle: float) -> None:
    """Update the properties of a distant light.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the distant light prim.
        angle (float): The angular size of the light in degrees.

    Raises:
        ValueError: If the prim at the given path is not a valid UsdLuxDistantLight.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path: {prim_path}")
    distant_light = UsdLux.DistantLight(prim)
    if not distant_light:
        raise ValueError(f"Prim at path {prim_path} is not a valid UsdLuxDistantLight")
    angle_attr = distant_light.GetAngleAttr()
    if not angle_attr:
        angle_attr = distant_light.CreateAngleAttr(Sdf.ValueTypeNames.Float)
    angle_attr.Set(angle)


def align_distant_light_to_target(distant_light: UsdLux.DistantLight, target_prim: UsdGeom.Imageable):
    """Aligns a distant light to point at a target prim.

    Args:
        distant_light (UsdLux.DistantLight): The distant light to align.
        target_prim (UsdGeom.Imageable): The target prim to align the light to.

    Raises:
        ValueError: If the distant_light or target_prim is not valid.
    """
    if not distant_light:
        raise ValueError("Invalid distant light.")
    if not target_prim:
        raise ValueError("Invalid target prim.")
    target_world_pos = target_prim.ComputeWorldBound(Usd.TimeCode.Default(), "default").ComputeCentroid()
    light_world_pos = (
        UsdGeom.Xformable(distant_light)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .Transform(Gf.Vec3d(0, 0, 0))
    )
    direction = target_world_pos - light_world_pos
    direction.Normalize()
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction)
    distant_light_xform = UsdGeom.Xformable(distant_light)
    rotate_op = add_rotate_xyz_op(distant_light_xform)
    rotate_op.Set(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))


def create_dome_light_with_texture(
    stage: Usd.Stage, prim_path: str, texture_file_path: str, texture_format: str = "latlong"
) -> UsdLux.DomeLight:
    """Create a UsdLuxDomeLight with a texture applied.

    Args:
        stage (Usd.Stage): The USD stage to create the dome light on.
        prim_path (str): The path where the dome light prim should be created.
        texture_file_path (str): The file path to the texture to apply to the dome light.
        texture_format (str, optional): The format of the texture file. Defaults to "latlong".

    Returns:
        UsdLux.DomeLight: The created dome light prim.
    """
    if texture_format not in ["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"]:
        raise ValueError(f"Invalid texture format: {texture_format}")
    dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path(prim_path))
    texture_file_attr = dome_light.CreateTextureFileAttr()
    texture_file_attr.Set(texture_file_path)
    texture_format_attr = dome_light.CreateTextureFormatAttr()
    texture_format_attr.Set(texture_format)
    dome_light.OrientToStageUpAxis()
    return dome_light


def get_all_dome_lights(stage: Usd.Stage) -> List[UsdLux.DomeLight]:
    """
    Get all the UsdLux.DomeLight prims in the given stage.

    Args:
        stage (Usd.Stage): The USD stage to search for dome lights.

    Returns:
        List[UsdLux.DomeLight]: A list of all UsdLux.DomeLight prims in the stage.
    """
    dome_lights = []
    for prim in stage.Traverse():
        if prim.IsA(UsdLux.DomeLight):
            dome_light = UsdLux.DomeLight(prim)
            dome_lights.append(dome_light)
    return dome_lights


def copy_dome_light(
    source_stage: Usd.Stage, source_path: str, dest_stage: Usd.Stage, dest_path: str
) -> UsdLux.DomeLight:
    """Copy a dome light from one stage to another."""
    source_prim = source_stage.GetPrimAtPath(source_path)
    if not source_prim.IsValid():
        raise ValueError(f"Source prim at path {source_path} does not exist.")
    if not UsdLux.DomeLight(source_prim):
        raise ValueError(f"Prim at path {source_path} is not a dome light.")
    dest_prim = dest_stage.GetPrimAtPath(dest_path)
    if not dest_prim.IsValid():
        dest_prim = UsdLux.DomeLight.Define(dest_stage, dest_path).GetPrim()
    elif not UsdLux.DomeLight(dest_prim):
        raise ValueError(f"Destination prim at path {dest_path} is not a dome light.")
    for attr in source_prim.GetAttributes():
        if attr.IsAuthored():
            attr_name = attr.GetName()
            attr_value = attr.Get()
            dest_attr = dest_prim.GetAttribute(attr_name)
            if dest_attr.IsValid():
                dest_attr.Set(attr_value)
    return UsdLux.DomeLight(dest_prim)


def add_portal_to_dome_light(dome_light: UsdLux.DomeLight, portal_prim: Usd.Prim) -> None:
    """Add a portal to a dome light by appending to the light's portals relationship."""
    if not dome_light:
        raise ValueError("Provided dome light is invalid.")
    if not portal_prim.IsValid():
        raise ValueError("Provided portal prim is invalid.")
    portals_rel = dome_light.GetPortalsRel()
    if not portals_rel:
        portals_rel = dome_light.CreatePortalsRel()
    existing_portal_paths = portals_rel.GetTargets()
    if portal_prim.GetPath() not in existing_portal_paths:
        portals_rel.AddTarget(portal_prim.GetPath())


def set_dome_light_texture(
    dome_light: UsdLux.DomeLight, texture_file_path: str, texture_format: str = "automatic"
) -> None:
    """Set the texture file and format for a dome light.

    Args:
        dome_light (UsdLux.DomeLight): The dome light to set the texture on.
        texture_file_path (str): The file path to the texture to use.
        texture_format (str, optional): The format of the texture file. Defaults to "automatic".

    Raises:
        ValueError: If the dome_light prim is not valid or if an invalid texture_format is provided.
    """
    if not dome_light:
        raise ValueError("Invalid dome light prim.")
    valid_texture_formats = ["automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross"]
    if texture_format not in valid_texture_formats:
        raise ValueError(f"Invalid texture format: {texture_format}. Must be one of {valid_texture_formats}")
    texture_file_attr = dome_light.GetTextureFileAttr()
    texture_file_attr.Set(texture_file_path)
    texture_format_attr = dome_light.GetTextureFormatAttr()
    texture_format_attr.Set(texture_format)


def set_dome_light_color(
    dome_light: UsdLux.DomeLight, color: Gf.Vec3f, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the color of a dome light.

    Args:
        dome_light (UsdLux.DomeLight): The dome light to set the color for.
        color (Gf.Vec3f): The color to set as an RGB vector.
        time_code (Usd.TimeCode): The time code to set the value at. Defaults to Default.

    Raises:
        ValueError: If the provided prim is not a valid UsdLux.DomeLight.
    """
    if not dome_light or not dome_light.GetPrim().IsValid():
        raise ValueError("Provided prim is not a valid UsdLux.DomeLight")
    color_attr = dome_light.GetColorAttr()
    if not color_attr:
        color_attr = dome_light.CreateColorAttr()
    color_attr.Set(color, time_code)


def create_dome_light_with_portals(
    stage: Usd.Stage, prim_path: str, texture_file_path: str, portals: List[Usd.Prim]
) -> UsdLux.DomeLight:
    """Create a dome light with portals.

    Args:
        stage (Usd.Stage): The USD stage to create the dome light on.
        prim_path (str): The path where the dome light should be created.
        texture_file_path (str): The path to the texture file for the dome light.
        portals (List[Usd.Prim]): A list of portal prims to guide the dome light.

    Returns:
        UsdLux.DomeLight: The created dome light prim.
    """
    prim_path = Sdf.Path(prim_path)
    if not prim_path.IsAbsolutePath():
        raise ValueError(f"Prim path must be an absolute path, got: {prim_path}")
    dome_light = UsdLux.DomeLight.Define(stage, prim_path)
    if not dome_light:
        raise RuntimeError(f"Failed to create dome light at path: {prim_path}")
    texture_file_attr = dome_light.CreateTextureFileAttr()
    texture_file_attr.Set(texture_file_path)
    texture_format_attr = dome_light.CreateTextureFormatAttr()
    texture_format_attr.Set("latlong")
    guide_radius_attr = dome_light.CreateGuideRadiusAttr()
    guide_radius_attr.Set(1000.0)
    portals_rel = dome_light.CreatePortalsRel()
    for portal in portals:
        portals_rel.AddTarget(portal.GetPath())
    return dome_light


def remove_dome_light(stage: Usd.Stage, prim_path: str) -> bool:
    """Remove a dome light prim from the stage."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not UsdLux.DomeLight(prim):
        raise ValueError(f"Prim at path {prim_path} is not a UsdLuxDomeLight.")
    stage.RemovePrim(prim.GetPath())
    return not stage.GetPrimAtPath(prim_path).IsValid()


def align_dome_light_to_prim(dome_light: UsdLux.DomeLight, prim: Usd.Prim) -> None:
    """Align a dome light to a prim.

    This function sets the dome light's transform to match the prim's world transform.
    The dome light's orientation is adjusted to match the prim's orientation.

    Args:
        dome_light (UsdLux.DomeLight): The dome light to align.
        prim (Usd.Prim): The prim to align the dome light to.

    Raises:
        ValueError: If the input dome_light is not a valid UsdLuxDomeLight.
        ValueError: If the input prim is not a valid UsdPrim.
    """
    if not dome_light:
        raise ValueError("Invalid dome light.")
    if not prim:
        raise ValueError("Invalid prim.")
    prim_xformable = UsdGeom.Xformable(prim)
    if not prim_xformable:
        raise ValueError("Prim is not transformable.")
    prim_transform = prim_xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    dome_light_prim = dome_light.GetPrim()
    dome_light_xformable = UsdGeom.Xformable(dome_light_prim)
    dome_light_xformable.AddTransformOp().Set(prim_transform)
    UsdGeom.SetStageUpAxis(dome_light.GetPrim().GetStage(), UsdGeom.Tokens.y)
    dome_light.OrientToStageUpAxis()


def create_and_assign_geometry_light(stage: Usd.Stage, light_path: str, geometry_path: str) -> UsdLux.GeometryLight:
    """Create a GeometryLight and assign a geometry to it.

    Args:
        stage (Usd.Stage): The USD stage.
        light_path (str): The path where the GeometryLight should be created.
        geometry_path (str): The path to the geometry that should be assigned to the GeometryLight.

    Returns:
        UsdLux.GeometryLight: The created GeometryLight prim.

    Raises:
        ValueError: If the geometry prim does not exist or is not a valid UsdGeomGprim.
    """
    light = UsdLux.GeometryLight.Define(stage, light_path)
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if not geometry_prim.IsValid() or not geometry_prim.IsA(UsdGeom.Gprim):
        raise ValueError(f"Geometry prim at path {geometry_path} is not a valid UsdGeomGprim.")
    geometry_rel = light.GetGeometryRel()
    geometry_rel.SetTargets([geometry_prim.GetPath()])
    return light


def update_geometry_light_intensity(stage: Usd.Stage, light_path: str, intensity: float) -> None:
    """Update the intensity of a geometry light.

    Args:
        stage (Usd.Stage): The stage containing the light.
        light_path (str): The path to the geometry light prim.
        intensity (float): The new intensity value.

    Raises:
        ValueError: If the prim at light_path is not a valid UsdLuxGeometryLight.
    """
    light_prim = stage.GetPrimAtPath(light_path)
    if not light_prim:
        raise ValueError(f"No prim found at path: {light_path}")
    geometry_light = UsdLux.GeometryLight(light_prim)
    if not geometry_light:
        raise ValueError(f"Prim at path {light_path} is not a valid UsdLuxGeometryLight")
    intensity_attr = geometry_light.GetIntensityAttr()
    if not intensity_attr:
        intensity_attr = geometry_light.CreateIntensityAttr()
    intensity_attr.Set(intensity)


def batch_update_geometry_light_attributes(stage: Usd.Stage, prim_paths: List[str], attributes: Dict[str, Any]):
    """
    Update attributes on multiple UsdLuxGeometryLight prims in a single transaction.

    Args:
        stage (Usd.Stage): The stage containing the prims to update.
        prim_paths (List[str]): A list of paths to UsdLuxGeometryLight prims.
        attributes (Dict[str, Any]): A dictionary mapping attribute names to their new values.

    Raises:
        ValueError: If any of the prim paths are invalid or not of type UsdLuxGeometryLight.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid prim path: {prim_path}")
        if not prim.IsA(UsdLux.GeometryLight):
            raise ValueError(f"Prim at path {prim_path} is not a UsdLuxGeometryLight")
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            geo_light = UsdLux.GeometryLight(prim)
            for attr_name, attr_value in attributes.items():
                attr = geo_light.GetPrim().GetAttribute(attr_name)
                if attr.IsValid():
                    attr.Set(attr_value)
                else:
                    if isinstance(attr_value, float):
                        attr = geo_light.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
                    elif isinstance(attr_value, tuple):
                        attr = geo_light.GetPrim().CreateAttribute(attr_name, Sdf.ValueTypeNames.Color3f)
                    else:
                        raise ValueError(f"Unsupported attribute type for {attr_name}")
                    attr.Set(attr_value)


def query_and_print_geometry_light_info(light: UsdLux.GeometryLight):
    """Print information about a UsdLuxGeometryLight."""
    if not light:
        raise ValueError("Invalid UsdLuxGeometryLight object")
    prim = light.GetPrim()
    print(f"Prim Path: {prim.GetPath()}")
    geom_rel = light.GetGeometryRel()
    if geom_rel.IsValid():
        geom_target = geom_rel.GetTargets()[0]
        print(f"Geometry Target: {geom_target}")
        stage = prim.GetStage()
        geom_prim = stage.GetPrimAtPath(geom_target)
        if geom_prim.IsValid():
            if UsdGeom.Mesh(geom_prim):
                print("Geometry Type: Mesh")
            else:
                print("Geometry Type: Unknown")
        else:
            print("Geometry Prim is invalid")
    else:
        print("Geometry Relationship is invalid")


def create_geometry_light_with_material(
    stage: Usd.Stage, light_path: str, geometry_path: str, material_path: str
) -> UsdLux.GeometryLight:
    """Create a geometry light with a material assigned to the geometry.

    Args:
        stage (Usd.Stage): The USD stage to create the light on.
        light_path (str): The path where the light should be created.
        geometry_path (str): The path to the geometry to use as the light source.
        material_path (str): The path to the material to assign to the geometry.

    Returns:
        UsdLux.GeometryLight: The created geometry light prim.

    Raises:
        ValueError: If the geometry or material prim does not exist.
    """
    light = UsdLux.GeometryLight.Define(stage, light_path)
    geometry_prim = stage.GetPrimAtPath(geometry_path)
    if not geometry_prim.IsValid():
        raise ValueError(f"Geometry prim at path {geometry_path} does not exist.")
    light.CreateGeometryRel().SetTargets([geometry_prim.GetPath()])
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    UsdShade.MaterialBindingAPI(geometry_prim).Bind(UsdShade.Material(material_prim))
    return light


def batch_update_lights(stage: Usd.Stage, light_paths: List[str], update_func) -> None:
    """
    Update properties of multiple lights in a batch.

    Args:
        stage (Usd.Stage): The USD stage containing the lights.
        light_paths (List[str]): List of paths to the light prims.
        update_func (function): A function that takes a LightAPI object and updates its properties.
    """
    for light_path in light_paths:
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            print(f"Warning: Light prim at path {light_path} does not exist. Skipping.")
            continue
        light = UsdLux.LightAPI(light_prim)
        if not light:
            print(f"Warning: Prim at path {light_path} is not a valid light. Skipping.")
            continue
        update_func(light)


def update_light(light: UsdLux.LightAPI):
    """Update function to set light parameters."""
    light.GetIntensityAttr().Set(500.0)
    light.GetExposureAttr().Set(2.0)
    light.GetDiffuseAttr().Set(0.8)
    light.GetSpecularAttr().Set(0.2)
    light.GetNormalizeAttr().Set(True)
    light.GetColorAttr().Set((1.0, 0.8, 0.6))


def set_light_color_temperature(light_api: UsdLux.LightAPI, color_temperature: float, enable: bool = True) -> None:
    """Set the color temperature of a light.

    Args:
        light_api (UsdLux.LightAPI): The light to set the color temperature on.
        color_temperature (float): The color temperature in degrees Kelvin. Must be between 1000 and 10000.
        enable (bool, optional): Whether to enable the color temperature. Defaults to True.

    Raises:
        ValueError: If the light_api is not valid or if the color_temperature is out of range.
    """
    if not light_api:
        raise ValueError("Invalid LightAPI object.")
    if color_temperature < 1000 or color_temperature > 10000:
        raise ValueError(f"Color temperature must be between 1000 and 10000. Got {color_temperature}.")
    color_temp_attr = light_api.GetColorTemperatureAttr()
    if not color_temp_attr:
        color_temp_attr = light_api.CreateColorTemperatureAttr()
    color_temp_attr.Set(color_temperature)
    enable_attr = light_api.GetEnableColorTemperatureAttr()
    if not enable_attr:
        enable_attr = light_api.CreateEnableColorTemperatureAttr()
    enable_attr.Set(enable)


def apply_light_api_to_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Apply the LightAPI to the prims at the given paths."""
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not prim.IsA(UsdGeom.Gprim):
            print(f"Warning: Prim at path {prim_path} is not a UsdGeomGprim. Skipping.")
            continue
        light_api = UsdLux.LightAPI.Apply(prim)
        if not light_api:
            print(f"Warning: Failed to apply LightAPI to prim at path {prim_path}.")


def get_light_exposure(light: UsdLux.LightAPI) -> float:
    """Get the exposure value for a light, considering the enable state."""
    exposure_attr = light.GetExposureAttr()
    if exposure_attr.IsAuthored():
        exposure = exposure_attr.Get()
        if isinstance(exposure, float):
            return exposure
        else:
            print(f"Warning: Exposure value for {light.GetPath()} is not a float. Returning default value of 0.0.")
            return 0.0
    else:
        return 0.0


def enable_color_temperature(light_api: UsdLux.LightAPI, enable: bool) -> None:
    """Enable or disable color temperature on the given light."""
    enable_attr = light_api.GetEnableColorTemperatureAttr()
    if not enable_attr:
        enable_attr = light_api.CreateEnableColorTemperatureAttr(enable, True)
    else:
        enable_attr.Set(enable)


def get_all_light_outputs(light: UsdLux.LightAPI) -> List[UsdShade.Output]:
    """Get all the outputs for a light."""
    output_names = light.GetOutputs(onlyAuthored=True)
    outputs = []
    for output_name in output_names:
        output = light.GetOutput(output_name.GetFullName())
        if not output:
            continue
        outputs.append(output)
    return outputs


def set_light_material_sync_mode(light_prim: Usd.Prim, sync_mode: str) -> None:
    """Set the material sync mode for a light prim.

    Args:
        light_prim (Usd.Prim): The light prim to set the sync mode for.
        sync_mode (str): The sync mode to set. Must be one of:
            "materialGlowTintsLight", "independent", "noMaterialResponse".

    Raises:
        ValueError: If the prim is not a valid light prim or if the sync mode is invalid.
    """
    if not light_prim.IsValid():
        raise ValueError(f"Invalid prim: {light_prim}")
    light = UsdLux.LightAPI(light_prim)
    if not light:
        raise ValueError(f"Prim {light_prim} is not a valid light")
    valid_modes = ["materialGlowTintsLight", "independent", "noMaterialResponse"]
    if sync_mode not in valid_modes:
        raise ValueError(f"Invalid sync mode: {sync_mode}. Must be one of: {valid_modes}")
    sync_mode_attr = light.GetMaterialSyncModeAttr()
    sync_mode_attr.Set(sync_mode)


def get_light_material_sync_mode(light: UsdLux.LightAPI) -> Tuple[bool, str]:
    """Get the material sync mode for a light.

    Returns:
        A tuple of (has_value, mode) where:
        - has_value is True if the attribute is authored and False otherwise.
        - mode is the authored value if has_value is True, or the empty string otherwise.
    """
    attr = light.GetMaterialSyncModeAttr()
    if attr.HasAuthoredValue():
        value = attr.Get()
        return (True, value)
    else:
        return (False, "")


def create_and_assign_light(
    stage: Usd.Stage, light_type: str, light_path: str, link_prims: List[str]
) -> UsdLux.LightAPI:
    """Create a light, assign it to the stage, and link it to the specified prims."""
    light_prim = stage.DefinePrim(light_path, light_type)
    if not light_prim.IsValid():
        raise ValueError(f"Failed to create light prim at path {light_path}")
    light = UsdLux.LightAPI(light_prim)
    if not light:
        raise ValueError(f"Failed to apply {light_type} light API to prim at path {light_path}")
    light_link_api = light.GetLightLinkCollectionAPI()
    for prim_path in link_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            light_link_api.CreateIncludesRel().AddTarget(prim_path)
        else:
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping light linking.")
    return light


def remove_light_api_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Remove the LightAPI from the prims at the given paths."""
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not prim.HasAPI(UsdLux.LightAPI):
            print(f"Warning: Prim at path {prim_path} does not have the LightAPI. Skipping.")
            continue
        prim.RemoveAPI(UsdLux.LightAPI)
        print(f"Removed LightAPI from prim at path {prim_path}")


def copy_light_with_attributes(source_prim: Usd.Prim, dest_stage: Usd.Stage, dest_path: str) -> UsdLux.LightAPI:
    """Copy a light prim with all its attributes to a new stage."""
    if not UsdLux.LightAPI(source_prim):
        raise ValueError(f"Prim at path {source_prim.GetPath()} is not a valid light.")
    if not dest_stage.GetRootLayer():
        raise ValueError("Destination stage is not valid.")
    dest_prim = dest_stage.DefinePrim(dest_path)
    dest_light = UsdLux.LightAPI(dest_prim)
    for attr in source_prim.GetAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        if attr_value is not None:
            dest_light.GetPrim().CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    return dest_light


def normalize_light(light: UsdLux.NonboundableLightBase, normalize: bool) -> None:
    """Normalize the intensity of a non-boundable light.

    Args:
        light (UsdLux.NonboundableLightBase): The light to normalize.
        normalize (bool): Whether to normalize the light intensity.

    Raises:
        ValueError: If the input light is not a valid UsdLux.NonboundableLightBase.
    """
    if not light:
        raise ValueError("Invalid light provided.")
    normalize_attr = light.GetNormalizeAttr()
    normalize_attr.Set(normalize)


def create_light_filter(stage: Usd.Stage, filter_type: str, prim_path: str) -> UsdLux.LightFilter:
    """Create a new light filter prim of the specified type at the given path."""
    if filter_type not in ["LightFilter", "ShapingAPI"]:
        raise ValueError(f"Invalid light filter type: {filter_type}")
    prim = stage.GetPrimAtPath(prim_path)
    if prim.IsValid():
        if prim.IsA(UsdLux.LightFilter):
            if filter_type == "ShapingAPI":
                return UsdLux.ShapingAPI(prim)
            else:
                return UsdLux.LightFilter(prim)
        raise ValueError(f"Prim at path {prim_path} already exists and is not a {filter_type}")
    elif filter_type == "ShapingAPI":
        light_filter = UsdLux.LightFilter.Define(stage, prim_path)
        UsdLux.ShapingAPI.Apply(light_filter.GetPrim())
        return UsdLux.ShapingAPI(light_filter.GetPrim())
    else:
        return UsdLux.LightFilter.Define(stage, prim_path)


def attach_filter_to_light(light_prim: Usd.Prim, filter_prim: Usd.Prim) -> None:
    """Attach a filter to a light prim.

    Args:
        light_prim (Usd.Prim): The light prim to attach the filter to.
        filter_prim (Usd.Prim): The filter prim to attach.

    Raises:
        ValueError: If the provided prims are not valid or of the correct type.
    """
    if not light_prim.IsValid():
        raise ValueError("Invalid light prim.")
    if not filter_prim.IsValid():
        raise ValueError("Invalid filter prim.")
    light_api = UsdLux.LightAPI(light_prim)
    if not light_api:
        raise ValueError("The provided prim does not have a UsdLux.LightAPI applied.")
    light_filter = UsdLux.LightFilter(filter_prim)
    if not light_filter:
        raise ValueError("The provided prim is not a valid UsdLux.LightFilter.")
    filters_rel = light_api.GetFiltersRel()
    filters_rel.AddTarget(filter_prim.GetPath())


def detach_filter_from_light(light: UsdLux.LightAPI, filter_path: str) -> None:
    """Detach a light filter from a light.

    Args:
        light (UsdLux.LightAPI): The light to detach the filter from.
        filter_path (str): The path to the light filter prim.

    Raises:
        ValueError: If the light or filter prim is invalid.
    """
    if not light:
        raise ValueError("Invalid light prim.")
    stage = light.GetPrim().GetStage()
    filter_prim = stage.GetPrimAtPath(filter_path)
    if not filter_prim.IsValid():
        raise ValueError(f"Invalid filter prim at path: {filter_path}")
    filters_rel = light.GetFiltersRel()
    if filter_prim.GetPath() in filters_rel.GetTargets():
        filters_rel.RemoveTarget(filter_prim.GetPath())


def create_light_input(light_prim: UsdLux.LightAPI, input_name: str, input_type: Sdf.ValueTypeName) -> UsdShade.Input:
    """Create an input on a light.

    Args:
        light_prim (UsdLux.LightAPI): The light prim to create the input on.
        input_name (str): The name of the input to create.
        input_type (Sdf.ValueTypeName): The value type of the input.

    Returns:
        UsdShade.Input: The created input object.

    Raises:
        ValueError: If the light prim is not valid.
        Tf.ErrorException: If an input with the same name already exists.
    """
    if not light_prim:
        raise ValueError("Invalid light prim.")
    connectable_api = light_prim.ConnectableAPI()
    existing_input = connectable_api.GetInput(input_name)
    if existing_input:
        raise Tf.ErrorException(f"Input '{input_name}' already exists on the light prim.")
    input_obj = connectable_api.CreateInput(input_name, input_type)
    return input_obj


def create_light_output(prim: Usd.Prim, output_name: str, output_type: Sdf.ValueTypeNames) -> UsdShade.Output:
    """Create a light output attribute on the given prim with the specified name and type.

    Args:
        prim (Usd.Prim): The prim to create the light output attribute on.
        output_name (str): The name of the output attribute.
        output_type (Sdf.ValueTypeNames): The value type of the output.

    Returns:
        UsdShade.Output: The created light output attribute.

    Raises:
        ValueError: If the prim is not valid or does not have a LightAPI applied.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    light_api = UsdLux.LightAPI(prim)
    if not light_api:
        raise ValueError("Prim does not have a LightAPI applied.")
    output = light_api.CreateOutput(output_name, output_type)
    if not output:
        raise ValueError(f"Failed to create light output '{output_name}' on prim '{prim.GetPath()}'.")
    return output


def get_light_input(light: UsdLux.LightAPI, input_name: str) -> Optional[Usd.Attribute]:
    """Get a specific input attribute from a light prim.

    Args:
        light (UsdLux.LightAPI): The light prim to retrieve the input from.
        input_name (str): The name of the input attribute to retrieve.

    Returns:
        Optional[Usd.Attribute]: The input attribute if found, None otherwise.
    """
    attr_name = f"inputs:{input_name}"
    if light.GetPrim().HasAttribute(attr_name):
        attr = light.GetPrim().GetAttribute(attr_name)
        return attr
    return None


def get_all_light_inputs(light: UsdLux.LightAPI, only_authored: bool = False) -> Dict[str, Usd.Attribute]:
    """Get a dictionary of all inputs on a light.

    Args:
        light (UsdLux.LightAPI): The light to get inputs for.
        only_authored (bool, optional): If True, only return authored inputs. Defaults to False.

    Returns:
        Dict[str, Usd.Attribute]: A dictionary mapping input names to their attributes.
    """
    inputs = light.GetInputs(only_authored)
    input_dict = {}
    for input in inputs:
        name = input.GetBaseName()
        attr = input.GetAttr()
        input_dict[name] = attr
    return input_dict


def set_light_shader_id(light: UsdLux.LightAPI, shader_id: str, render_context: str = "") -> None:
    """Set the shader ID for a light.

    Args:
        light (UsdLux.LightAPI): The light to set the shader ID for.
        shader_id (str): The shader ID to set.
        render_context (str, optional): The render context to set the shader ID for. Defaults to "".

    Raises:
        ValueError: If the light prim is invalid.
    """
    if not light.GetPrim().IsValid():
        raise ValueError("Invalid light prim.")
    if render_context:
        shader_id_attr = light.GetShaderIdAttrForRenderContext(render_context)
        if shader_id_attr.IsValid():
            shader_id_attr.Set(shader_id)
    else:
        light.CreateShaderIdAttr(shader_id)


def assign_light_shader(light_prim: UsdLux.LightAPI, shader_id: str, render_context: str = ""):
    """Assign a shader ID to a light prim for a specific render context.

    Args:
        light_prim (UsdLux.LightAPI): The light prim to assign the shader to.
        shader_id (str): The shader ID to assign.
        render_context (str, optional): The render context for which to assign the shader.
            If empty, the default shader ID is set. Defaults to "".

    Raises:
        ValueError: If the input prim is not a valid UsdLux.LightAPI prim.
    """
    if not light_prim:
        raise ValueError("Invalid UsdLux.LightAPI prim.")
    if render_context:
        shader_attr = light_prim.CreateShaderIdAttrForRenderContext(render_context, shader_id, True)
    else:
        shader_attr = light_prim.CreateShaderIdAttr(shader_id, True)
    shader_attr.Set(shader_id)


def get_light_shader_id(light_api: UsdLux.LightAPI, render_contexts: List[str]) -> str:
    """Get the shader ID for a light given a list of render contexts."""
    for render_context in render_contexts:
        shader_id_attr = light_api.GetShaderIdAttrForRenderContext(render_context)
        if shader_id_attr:
            shader_id = shader_id_attr.Get()
            if shader_id:
                return shader_id
    default_shader_id_attr = light_api.GetShaderIdAttr()
    if default_shader_id_attr:
        default_shader_id = default_shader_id_attr.Get()
        if default_shader_id:
            return default_shader_id
    return ""


def get_shadow_linked_geometry(light_prim: Usd.Prim) -> List[Usd.Prim]:
    """Get the list of geometry prims that are shadow-linked to the given light prim."""
    if not light_prim.IsValid():
        raise ValueError(f"Invalid prim: {light_prim}")
    light_api = UsdLux.LightAPI(light_prim)
    if not light_api:
        raise ValueError(f"Prim {light_prim.GetPath()} does not have the LightAPI applied")
    shadow_link_collection = light_api.GetShadowLinkCollectionAPI()
    includes_rel = shadow_link_collection.GetIncludesRel()
    target_paths = includes_rel.GetTargets() if includes_rel else []
    stage = light_prim.GetStage()
    shadow_linked_prims = []
    for path in target_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            shadow_linked_prims.append(prim)
    return shadow_linked_prims


def create_prim(stage, prim_type, path):
    prim = stage.DefinePrim(path, prim_type)
    return prim


def list_all_light_filters(stage: Usd.Stage) -> list[UsdLux.LightFilter]:
    """
    Returns a list of all light filters in the given USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search for light filters.

    Returns:
        list[UsdLux.LightFilter]: A list of UsdLux.LightFilter objects.
    """
    root_prim = stage.GetPseudoRoot()
    light_filters = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdLux.LightFilter):
            light_filter = UsdLux.LightFilter(prim)
            light_filters.append(light_filter)
    return light_filters


def get_light_filter_inputs(light_filter: UsdLux.LightFilter) -> List[Tuple[str, Sdf.ValueTypeName]]:
    """
    Get a list of tuples containing the name and value type of each input on the given light filter.

    Args:
        light_filter (UsdLux.LightFilter): The light filter to get inputs from.

    Returns:
        List[Tuple[str, Sdf.ValueTypeName]]: A list of tuples, each containing the name and value type of an input.
    """
    inputs = light_filter.GetInputs()
    input_data = []
    for input in inputs:
        input_name = input.GetBaseName()
        input_type = input.GetTypeName()
        input_data.append((input_name, input_type))
    return input_data


def get_light_filter_outputs(light_filter: UsdLux.LightFilter) -> List[Tuple[str, Sdf.ValueTypeName]]:
    """
    Get the list of output attributes for a light filter.

    Args:
        light_filter (UsdLux.LightFilter): The light filter to get the outputs from.

    Returns:
        List[Tuple[str, Sdf.ValueTypeName]]: A list of tuples containing the output name and value type.
    """
    output_attrs = light_filter.GetOutputs()
    outputs = []
    for output_attr in output_attrs:
        output_name = output_attr.GetBaseName()
        output_type = output_attr.GetTypeName()
        outputs.append((output_name, output_type))
    return outputs


def update_light_filter_shader(light_filter: UsdLux.LightFilter, shader_id: str, render_context: str = "") -> None:
    """Update the shader ID for a light filter.

    Args:
        light_filter (UsdLux.LightFilter): The light filter prim to update.
        shader_id (str): The new shader ID to set.
        render_context (str, optional): The render context to set the shader ID for.
            If empty, the default shader ID will be set. Defaults to "".

    Raises:
        ValueError: If the given prim is not a valid UsdLux.LightFilter.
    """
    if not light_filter or not light_filter.GetPrim().IsValid():
        raise ValueError("Invalid light filter prim")
    if render_context:
        shader_id_attr = light_filter.CreateShaderIdAttrForRenderContext(render_context, shader_id, True)
    else:
        shader_id_attr = light_filter.CreateShaderIdAttr(shader_id, True)
    shader_id_attr.Set(shader_id)


def set_light_filter_input(
    light_filter: UsdLux.LightFilter,
    input_name: str,
    input_value: Any,
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
) -> None:
    """Set the value of a light filter input.

    Args:
        light_filter (UsdLux.LightFilter): The light filter prim.
        input_name (str): The name of the input to set.
        input_value (Any): The value to set for the input.
        time_code (Usd.TimeCode, optional): The time code at which to set the input value. Defaults to Default().

    Raises:
        ValueError: If the input doesn't exist or if the input's type doesn't match the provided value type.
    """
    input_attr = light_filter.GetInput(input_name)
    if not input_attr:
        raise ValueError(f"Input '{input_name}' does not exist on the light filter.")
    input_type = input_attr.GetTypeName()
    if input_type == Sdf.ValueTypeNames.Float or input_type == Sdf.ValueTypeNames.Double:
        if not isinstance(input_value, (float, int)):
            raise ValueError(
                f"Input '{input_name}' expects type '{input_type}', but received type '{type(input_value)}'."
            )
    elif input_type == Sdf.ValueTypeNames.Color3f or input_type == Sdf.ValueTypeNames.Color3d:
        if not isinstance(input_value, Gf.Vec3f) and (not isinstance(input_value, Gf.Vec3d)):
            raise ValueError(
                f"Input '{input_name}' expects type '{input_type}', but received type '{type(input_value)}'."
            )
    elif input_type == Sdf.ValueTypeNames.Float3Array or input_type == Sdf.ValueTypeNames.Double3Array:
        if not isinstance(input_value, list) or not all(
            (isinstance(val, Gf.Vec3f) or isinstance(val, Gf.Vec3d) for val in input_value)
        ):
            raise ValueError(
                f"Input '{input_name}' expects type '{input_type}', but received type '{type(input_value)}'."
            )
    else:
        raise ValueError(f"Input '{input_name}' has an unsupported type '{input_type}'.")
    input_attr.Set(input_value, time_code)


def toggle_light_filter_effect(light_filter: UsdLux.LightFilter, enable: bool) -> None:
    """Toggle the effect of a light filter.

    Args:
        light_filter (UsdLux.LightFilter): The light filter to modify.
        enable (bool): Whether to enable or disable the light filter effect.

    Raises:
        ValueError: If the given prim is not a valid UsdLux.LightFilter.
    """
    if not light_filter or not isinstance(light_filter, UsdLux.LightFilter):
        raise ValueError("Invalid light filter prim.")
    collection = light_filter.GetFilterLinkCollectionAPI()
    if enable:
        collection.CreateIncludesRel().SetTargets([Sdf.Path("/Geometry")])
        collection.CreateExcludesRel().ClearTargets(removeSpec=True)
    else:
        collection.CreateIncludesRel().ClearTargets(removeSpec=True)
        collection.CreateExcludesRel().SetTargets([Sdf.Path("/Geometry")])


def create_and_link_light_filter(
    stage: Usd.Stage, light_filter_path: str, linked_prims: List[str]
) -> UsdLux.LightFilter:
    """Create a light filter and link it to the specified prims.

    Args:
        stage (Usd.Stage): The stage to create the light filter on.
        light_filter_path (str): The path where the light filter should be created.
        linked_prims (List[str]): A list of prim paths that the light filter should be linked to.

    Returns:
        UsdLux.LightFilter: The created light filter.

    Raises:
        ValueError: If the light filter path is invalid or if any of the linked prim paths are invalid.
    """
    if not Sdf.Path.IsValidPathString(light_filter_path):
        raise ValueError(f"Invalid light filter path: {light_filter_path}")
    light_filter = UsdLux.LightFilter.Define(stage, light_filter_path)
    filter_link_collection = light_filter.GetFilterLinkCollectionAPI()
    for prim_path in linked_prims:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Invalid linked prim path: {prim_path}")
        filter_link_collection.CreateIncludesRel().AddTarget(prim_path)
    return light_filter


def batch_update_light_filter_inputs(light_filter: UsdLux.LightFilter, input_values: Dict[str, object]) -> None:
    """Updates the inputs of a light filter with the provided values in batch.

    Args:
        light_filter (UsdLux.LightFilter): The light filter to update inputs for.
        input_values (Dict[str, object]): A dictionary mapping input names to their corresponding values.

    Raises:
        ValueError: If the provided light filter is invalid.
    """
    if not light_filter:
        raise ValueError("Invalid light filter provided.")
    for input_name, input_value in input_values.items():
        input_attr = light_filter.GetInput(input_name)
        if input_attr:
            input_attr.Set(input_value)
        else:
            value_type = Sdf.ValueTypeNames.Find(type(input_value).__name__)
            input_attr = light_filter.CreateInput(input_name, value_type)
            input_attr.Set(input_value)


def merge_light_filter_collections(light_filter: UsdLux.LightFilter, collection_names: List[str]) -> UsdLux.LightFilter:
    """Merge multiple light filter collections into a single collection.

    Args:
        light_filter (UsdLux.LightFilter): The light filter to merge collections for.
        collection_names (List[str]): List of collection names to merge.

    Returns:
        UsdLux.LightFilter: The input light filter for chaining.

    Raises:
        ValueError: If the input light filter is invalid.
    """
    if not light_filter or not light_filter.GetPrim().IsValid():
        raise ValueError("Invalid light filter.")
    light_filter_prim = light_filter.GetPrim()
    merged_collection_name = "merged_collection"
    merged_collection = Usd.CollectionAPI.Apply(light_filter_prim, merged_collection_name)
    for collection_name in collection_names:
        collection = Usd.CollectionAPI(light_filter_prim, collection_name)
        if not collection.GetPrim().IsValid():
            continue
        target_paths = collection.GetIncludesRel().GetTargets()
        for target_path in target_paths:
            merged_collection.CreateIncludesRel().AddTarget(target_path)
    return light_filter


def discover_and_cache_lights(stage: Usd.Stage, cache_prim_path: str) -> None:
    """Discover all lights in the stage and cache them on the given prim path."""
    cache_prim = stage.GetPrimAtPath(cache_prim_path)
    if not cache_prim.IsValid():
        raise ValueError(f"Prim at path {cache_prim_path} does not exist.")
    light_list_api = UsdLux.LightListAPI.Apply(cache_prim)
    if not light_list_api:
        raise RuntimeError(f"Failed to apply LightListAPI to prim at path {cache_prim_path}.")
    light_paths = light_list_api.ComputeLightList(UsdLux.LightListAPI.ComputeModeIgnoreCache)
    light_list_api.StoreLightList(light_paths)


def check_light_cache_validity(list_api: UsdLux.LightListAPI) -> bool:
    """Check if the light cache is valid for the given LightListAPI object."""
    prim = list_api.GetPrim()
    if not prim.IsValid():
        return False
    cache_behavior_attr = list_api.GetLightListCacheBehaviorAttr()
    if not cache_behavior_attr or not cache_behavior_attr.IsValid():
        return False
    cache_behavior = cache_behavior_attr.Get()
    if cache_behavior == UsdLux.Tokens.ignore:
        return False
    light_list_rel = list_api.GetLightListRel()
    if not light_list_rel or not light_list_rel.GetTargets():
        return False
    return True


def apply_light_list_api(prim: UsdLux.ListAPI) -> None:
    """Apply the UsdLuxListAPI to a prim and set its lightList attribute."""
    if not prim.GetPrim().IsValid():
        raise ValueError("Prim is not valid.")
    UsdLux.ListAPI.Apply(prim.GetPrim())
    if not prim.GetPrim().HasRelationship("lightList"):
        lightList_rel = prim.CreateLightListRel()
    else:
        lightList_rel = prim.GetLightListRel()
    light_paths = [Sdf.Path("/Light1"), Sdf.Path("/Light2")]
    lightList_rel.SetTargets(light_paths)
    cache_behavior_attr = prim.CreateLightListCacheBehaviorAttr(UsdLux.Tokens.consumeAndContinue, True)
    cache_behavior_attr.Set(UsdLux.Tokens.consumeAndContinue)


def invalidate_light_cache(light_list_api: UsdLux.LightListAPI) -> None:
    """Invalidate the light cache for the given LightListAPI prim."""
    cache_behavior_attr = light_list_api.GetLightListCacheBehaviorAttr()
    if cache_behavior_attr.IsValid():
        cache_behavior_attr.Set(UsdLux.Tokens.ignore)
    else:
        light_list_api.CreateLightListCacheBehaviorAttr(UsdLux.Tokens.ignore)


def remove_light_list_api(stage: Usd.Stage, prim_path: str) -> bool:
    """Remove the LightListAPI from the specified prim."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.HasAPI(UsdLux.LightListAPI):
        return False
    success = prim.RemoveAPI(UsdLux.LightListAPI)
    return success


def get_all_light_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Get all light prims in the given stage.

    This function traverses the stage and returns a list of all prims that
    have the UsdLuxLightAPI applied or are of type UsdLuxLightFilter.

    Parameters:
        stage (Usd.Stage): The USD stage to search for lights.

    Returns:
        List[Usd.Prim]: A list of light prims found in the stage.
    """
    light_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdLux.LightAPI):
            light_prims.append(prim)
        elif prim.IsA(UsdLux.LightFilter):
            light_prims.append(prim)
    return light_prims


def load_cached_lights(stage: Usd.Stage, prim_path: str) -> Sdf.PathListOp:
    """Load cached lights from the given prim path.

    This function retrieves the cached light list from the prim at the given path.
    It checks if the prim has a valid LightListAPI and if the cache behavior is set
    to consume the cache. If the cache is valid, it returns the cached light paths.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim containing the cached light list.

    Returns:
        Sdf.PathListOp: The list of cached light paths, or an empty list if the cache is invalid.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return Sdf.PathListOp()
    light_list_api = UsdLux.LightListAPI(prim)
    if not light_list_api:
        return Sdf.PathListOp()
    cache_behavior_attr = light_list_api.GetPrim().GetAttribute("lightList:cacheBehavior")
    cache_behavior = cache_behavior_attr.Get() if cache_behavior_attr.IsValid() else None
    if cache_behavior not in ["consumeAndHalt", "consumeAndContinue"]:
        return Sdf.PathListOp()
    light_list_rel = light_list_api.GetPrim().GetRelationship("lightList")
    if not light_list_rel.IsValid():
        return Sdf.PathListOp()
    cached_lights = light_list_rel.GetTargets()
    return cached_lights


def get_light_list_cache_behavior(light_list_api: UsdLux.LightListAPI) -> str:
    """Get the value of the lightList:cacheBehavior attribute.

    Args:
        light_list_api (UsdLux.LightListAPI): The LightListAPI object.

    Returns:
        str: The value of the lightList:cacheBehavior attribute.
        Returns "ignore" if the attribute is not authored.
    """
    cache_behavior_attr = light_list_api.GetLightListCacheBehaviorAttr()
    if cache_behavior_attr.IsAuthored():
        cache_behavior = cache_behavior_attr.Get()
        allowed_values = [UsdLux.Tokens.consumeAndHalt, UsdLux.Tokens.consumeAndContinue, UsdLux.Tokens.ignore]
        if cache_behavior in allowed_values:
            return str(cache_behavior)
        else:
            return str(UsdLux.Tokens.ignore)
    else:
        return str(UsdLux.Tokens.ignore)


def collect_light_paths(light_list_api: UsdLux.LightListAPI, mode: str) -> List[Sdf.Path]:
    """Collect light paths from a LightListAPI object based on the specified compute mode."""
    stage = light_list_api.GetPrim().GetStage()
    light_paths = []
    if mode == "ignoreCache":
        for prim in stage.Traverse():
            if prim.HasAPI(UsdLux.LightAPI):
                light_paths.append(prim.GetPath())
            elif prim.IsA(UsdLux.LightFilter):
                light_paths.append(prim.GetPath())
    elif mode == "consultModelHierarchyCache":
        for prim in light_list_api.GetPrim().GetChildren():
            if prim.HasAPI(UsdLux.LightAPI):
                light_paths.append(prim.GetPath())
            elif prim.IsA(UsdLux.LightFilter):
                light_paths.append(prim.GetPath())
            if prim.HasRelationship("lightList"):
                cache_behavior = prim.GetAttribute("lightList:cacheBehavior").Get()
                if cache_behavior == "consumeAndHalt":
                    light_paths.extend(prim.GetRelationship("lightList").GetTargets())
                    continue
                elif cache_behavior == "consumeAndContinue":
                    light_paths.extend(prim.GetRelationship("lightList").GetTargets())
                elif cache_behavior == "ignore":
                    pass
            light_paths.extend(collect_light_paths(UsdLux.LightListAPI(prim), mode))
    else:
        raise ValueError(f"Invalid compute mode: {mode}")
    return light_paths


def update_light_cache(stage: Usd.Stage, prim_path: str) -> None:
    """Updates the light cache for the given prim path."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    if not UsdLux.LightListAPI.CanApply(prim):
        UsdLux.LightListAPI.Apply(prim)
    light_list_api = UsdLux.LightListAPI(prim)
    light_paths = light_list_api.ComputeLightList(UsdLux.LightListAPI.ComputeModeIgnoreCache)
    light_list_api.StoreLightList(light_paths)


def configure_light_list_computation(
    stage: Usd.Stage, prim_path: str, mode: UsdLux.Tokens.lightListCacheBehavior
) -> None:
    """Configure the light list computation mode for a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        mode (UsdLux.Tokens.lightListCacheBehavior): The compute mode to set.

    Raises:
        ValueError: If the prim is not valid or does not have a LightListAPI applied.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    light_list_api = UsdLux.LightListAPI(prim)
    if not light_list_api:
        raise ValueError(f"Prim at path {prim_path} does not have a LightListAPI applied.")
    light_list_api.GetLightListCacheBehaviorAttr().Set(mode)


def get_or_create_light_list_cache_behavior(
    list_api: UsdLux.ListAPI, default_value: UsdLux.Tokens = UsdLux.Tokens.ignore
) -> UsdLux.Tokens:
    """
    Get or create the lightList:cacheBehavior attribute on the given ListAPI prim.

    If the attribute does not exist, it will be created with the specified default value.

    Args:
        list_api (UsdLux.ListAPI): The ListAPI prim to get or create the attribute on.
        default_value (UsdLux.Tokens): The default value to use if the attribute is created.

    Returns:
        UsdLux.Tokens: The value of the lightList:cacheBehavior attribute.
    """
    attr = list_api.GetLightListCacheBehaviorAttr()
    if attr.IsValid():
        return attr.Get()
    attr = list_api.CreateLightListCacheBehaviorAttr(default_value, writeSparsely=True)
    return attr.Get()


def get_or_create_light_list_relationship(prim: Usd.Prim) -> Usd.Relationship:
    """Get or create the lightList relationship on the given prim.

    Args:
        prim (Usd.Prim): The prim to get or create the lightList relationship on.

    Returns:
        Usd.Relationship: The lightList relationship on the prim.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    list_api = UsdLux.ListAPI(prim)
    if list_api.GetLightListRel().IsValid():
        return list_api.GetLightListRel()
    else:
        return list_api.CreateLightListRel()


def get_light_list_and_cache_behavior(prim: Usd.Prim) -> Tuple[List[Sdf.Path], UsdLux.Tokens]:
    """
    Get the light list and cache behavior for a given prim.

    Args:
        prim (Usd.Prim): The prim to get the light list and cache behavior for.

    Returns:
        Tuple[List[Sdf.Path], UsdLux.Tokens]: A tuple containing the light list and cache behavior.
    """
    list_api = UsdLux.ListAPI(prim)
    if not list_api:
        return ([], UsdLux.Tokens.ignore)
    light_list_rel = list_api.GetLightListRel()
    if light_list_rel:
        light_list = light_list_rel.GetTargets()
    else:
        light_list = []
    cache_behavior_attr = list_api.GetLightListCacheBehaviorAttr()
    if cache_behavior_attr:
        cache_behavior = cache_behavior_attr.Get()
        if not cache_behavior:
            cache_behavior = UsdLux.Tokens.ignore
    else:
        cache_behavior = UsdLux.Tokens.ignore
    return (light_list, cache_behavior)


def test_get_light_list_and_cache_behavior():
    stage = Usd.Stage.CreateInMemory()
    prim = stage.DefinePrim("/TestPrim")
    list_api = UsdLux.ListAPI.Apply(prim)
    light_list_rel = list_api.CreateLightListRel()
    light_list_rel.AddTarget(Sdf.Path("/Light1"))
    light_list_rel.AddTarget(Sdf.Path("/Light2"))
    cache_behavior_attr = list_api.CreateLightListCacheBehaviorAttr(UsdLux.Tokens.consumeAndContinue, True)
    (light_list, cache_behavior) = get_light_list_and_cache_behavior(prim)
    print(f"Light list: {light_list}")
    print(f"Cache behavior: {cache_behavior}")


def create_and_apply_light_list_cache_behavior(prim: Usd.Prim, behavior: str) -> None:
    """Create and apply the lightList:cacheBehavior attribute to the given prim.

    Args:
        prim (Usd.Prim): The prim to apply the attribute to.
        behavior (str): The cache behavior value to set.

    Raises:
        ValueError: If the provided behavior is not a valid value.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    list_api = UsdLux.ListAPI(prim)
    allowed_values = ["consumeAndHalt", "consumeAndContinue", "ignore"]
    if behavior not in allowed_values:
        raise ValueError(f"Invalid cache behavior. Allowed values are: {allowed_values}")
    attr = list_api.CreateLightListCacheBehaviorAttr(behavior, True)
    attr.Set(behavior)


def compute_and_store_light_list_with_cache_behavior(prim: Usd.Prim, cache_behavior: str) -> None:
    """Compute the light list for the given prim and store it with the specified cache behavior.

    Args:
        prim (Usd.Prim): The prim to compute and store the light list for.
        cache_behavior (str): The cache behavior to use when storing the light list.
                              Valid values are "consumeAndHalt", "consumeAndContinue", or "ignore".
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    list_api = UsdLux.ListAPI.Apply(prim)
    if not list_api:
        raise ValueError(f"Failed to apply ListAPI to prim: {prim}")
    if cache_behavior == "ignore":
        light_list = list_api.ComputeLightList(UsdLux.ListAPI.ComputeModeIgnoreCache)
    elif cache_behavior in ["consumeAndHalt", "consumeAndContinue"]:
        light_list = list_api.ComputeLightList(UsdLux.ListAPI.ComputeModeConsultModelHierarchyCache)
    else:
        raise ValueError(f"Invalid cache behavior: {cache_behavior}")
    list_api.StoreLightList(light_list)
    cache_behavior_attr = list_api.GetLightListCacheBehaviorAttr()
    cache_behavior_attr.Set(cache_behavior)


def apply_and_store_light_list(prim: Usd.Prim, light_paths: List[Sdf.Path]) -> None:
    """Apply the ListAPI schema to the prim and store the given light paths.

    Args:
        prim (Usd.Prim): The prim to apply the ListAPI schema to.
        light_paths (List[Sdf.Path]): The list of light paths to store.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    list_api = UsdLux.ListAPI.Apply(prim)
    list_api.StoreLightList(light_paths)
    list_api.GetLightListCacheBehaviorAttr().Set(UsdLux.Tokens.consumeAndContinue)


def apply_and_invalidate_light_list(prim: Usd.Prim, light_paths: Sequence[str]) -> None:
    """Apply the given light paths to the prim's light list and then invalidate the cache."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    list_api = UsdLux.ListAPI.Apply(prim)
    if not list_api:
        raise ValueError(f"Failed to apply ListAPI to prim {prim.GetPath()}")
    light_path_list = [Sdf.Path(p) for p in light_paths]
    list_api.GetLightListRel().SetTargets(light_path_list)
    list_api.InvalidateLightList()


def set_light_list_computation_mode(compute_mode: str):
    """Set the light list computation mode.

    Args:
        compute_mode (str): The computation mode to set. Must be one of "none", "light", or "light,shadow".

    Raises:
        ValueError: If the input compute_mode is not a valid computation mode string.
    """
    valid_modes = ["none", "light", "light,shadow"]
    if compute_mode not in valid_modes:
        raise ValueError(f"Invalid compute_mode: {compute_mode}. Must be one of {valid_modes}.")
    os.environ["USD_LIGHT_LIST_CACHE_COMPUTATION"] = compute_mode


def apply_mesh_light_to_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Apply the MeshLightAPI to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to apply the MeshLightAPI to.

    Raises:
        ValueError: If any prim path does not exist or is not a mesh.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        if not UsdLux.MeshLightAPI.CanApply(prim):
            raise ValueError(f"MeshLightAPI cannot be applied to prim at path {prim_path}.")
        mesh_light_api = UsdLux.MeshLightAPI.Apply(prim)
        if not mesh_light_api.GetPrim():
            raise ValueError(f"Failed to apply MeshLightAPI to prim at path {prim_path}.")


def can_apply_mesh_light_to_prims(prims: List[Usd.Prim]) -> List[bool]:
    """Check if MeshLightAPI can be applied to each prim in a list.

    Args:
        prims (List[Usd.Prim]): A list of Usd.Prim objects to check.

    Returns:
        List[bool]: A list of booleans indicating whether MeshLightAPI can be applied to each prim.
    """
    results = []
    for prim in prims:
        if not prim.IsValid():
            results.append(False)
            continue
        if prim.IsA(UsdGeom.Mesh):
            results.append(True)
            continue
        (can_apply, reason) = UsdLux.MeshLightAPI.CanApply(prim)
        if can_apply:
            results.append(True)
        else:
            print(f"Cannot apply MeshLightAPI to prim {prim.GetPath()}: {reason}")
            results.append(False)
    return results


def remove_mesh_light_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> None:
    """Remove the MeshLightAPI from the specified prims."""
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not prim.HasAPI(UsdLux.MeshLightAPI):
            print(f"Warning: Prim at path {prim_path} does not have MeshLightAPI applied. Skipping.")
            continue
        prim.RemoveAPI(UsdLux.MeshLightAPI)
        if prim.HasAPI(UsdLux.MeshLightAPI):
            print(f"Error: Failed to remove MeshLightAPI from prim at path {prim_path}.")
        else:
            print(f"Successfully removed MeshLightAPI from prim at path {prim_path}.")


def get_mesh_light_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Return a list of all prims on the stage that have the MeshLightAPI applied.

    Args:
        stage (Usd.Stage): The USD stage to search for mesh light prims.

    Returns:
        List[Usd.Prim]: A list of prims with the MeshLightAPI applied.
    """
    mesh_light_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if UsdLux.MeshLightAPI.Get(stage, prim.GetPath()):
            mesh_light_prims.append(prim)
    return mesh_light_prims


def get_light_attributes(light: UsdLux.NonboundableLightBase) -> dict:
    """
    Get the attribute values of a nonboundable light.

    Args:
        light (UsdLux.NonboundableLightBase): The input nonboundable light.

    Returns:
        dict: A dictionary containing the attribute values.
    """
    if not light:
        raise ValueError("Invalid nonboundable light.")
    color = light.GetColorAttr().Get()
    color_temperature = light.GetColorTemperatureAttr().Get()
    diffuse = light.GetDiffuseAttr().Get()
    specular = light.GetSpecularAttr().Get()
    normalize = light.GetNormalizeAttr().Get()
    intensity = light.GetIntensityAttr().Get()
    exposure = light.GetExposureAttr().Get()
    enable_color_temperature = light.GetEnableColorTemperatureAttr().Get()
    attributes = {
        "color": color,
        "color_temperature": color_temperature,
        "diffuse": diffuse,
        "specular": specular,
        "normalize": normalize,
        "intensity": intensity,
        "exposure": exposure,
        "enable_color_temperature": enable_color_temperature,
    }
    return attributes


def find_lights_by_intensity(
    stage: Usd.Stage, min_intensity: float, max_intensity: float
) -> List[UsdLux.NonboundableLightBase]:
    """
    Find all lights on the stage with intensity between min_intensity and max_intensity.

    Args:
        stage (Usd.Stage): The USD stage to search for lights.
        min_intensity (float): The minimum intensity value (inclusive).
        max_intensity (float): The maximum intensity value (inclusive).

    Returns:
        List[UsdLux.NonboundableLightBase]: A list of lights matching the intensity criteria.
    """
    matching_lights = []
    for prim in stage.Traverse():
        light = UsdLux.NonboundableLightBase(prim)
        if light:
            intensity_attr = light.GetIntensityAttr()
            if intensity_attr:
                intensity = intensity_attr.Get(Usd.TimeCode.Default())
                if min_intensity <= intensity <= max_intensity:
                    matching_lights.append(light)
    return matching_lights


def find_lights_by_color(stage: Usd.Stage, color: Gf.Vec3f, tolerance: float = 0.01) -> list[str]:
    """
    Find all UsdLuxNonboundableLightBase prims on the stage with a specific color.

    Args:
        stage (Usd.Stage): The USD stage to search for lights.
        color (Gf.Vec3f): The color to match against the light's color attribute.
        tolerance (float): The tolerance for color comparison. Default is 0.01.

    Returns:
        list[str]: A list of prim paths for lights matching the specified color.
    """
    lights = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdLux.NonboundableLightBase):
            lights.append(prim)
    matching_lights = []
    for light in lights:
        light_api = UsdLux.NonboundableLightBase(light)
        color_attr = light_api.GetColorAttr()
        if color_attr.HasValue():
            light_color = color_attr.Get()
            if Gf.IsClose(light_color, color, tolerance):
                matching_lights.append(light.GetPath())
    return matching_lights


def copy_light_attributes(src_light: UsdLux.NonboundableLightBase, dst_light: UsdLux.NonboundableLightBase):
    """Copy light attributes from one light to another."""
    if not src_light or not dst_light:
        raise ValueError("Both source and destination lights must be valid.")
    light_attr_names = UsdLux.NonboundableLightBase.GetSchemaAttributeNames(includeInherited=True)
    for attr_name in light_attr_names:
        src_attr = src_light.GetPrim().GetAttribute(attr_name)
        if src_attr.HasValue():
            attr_value = src_attr.Get()
            dst_attr = dst_light.GetPrim().CreateAttribute(attr_name, src_attr.GetTypeName())
            dst_attr.Set(attr_value)


def update_light_color_based_on_temperature(light: UsdLux.NonboundableLightBase, temperature: float) -> None:
    """Update the color of a light based on the color temperature.

    Args:
        light (UsdLux.NonboundableLightBase): The light to update.
        temperature (float): The color temperature in Kelvin.

    Raises:
        ValueError: If the input temperature is outside the valid range (1000K to 10000K).
    """
    if temperature < 1000 or temperature > 10000:
        raise ValueError(f"Invalid color temperature: {temperature}. Must be between 1000K and 10000K.")
    enable_attr = light.GetEnableColorTemperatureAttr()
    if not enable_attr:
        enable_attr = light.CreateEnableColorTemperatureAttr(True, True)
    enable_attr.Set(True)
    temp_attr = light.GetColorTemperatureAttr()
    if not temp_attr:
        temp_attr = light.CreateColorTemperatureAttr(temperature, True)
    temp_attr.Set(temperature)
    color_attr = light.GetColorAttr()
    if color_attr:
        color_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))


def set_light_shader_parameters(
    light: UsdLux.NonboundableLightBase, color: Gf.Vec3f, intensity: float, exposure: float
):
    """Set common light shader parameters on a non-boundable light.

    Args:
        light (UsdLux.NonboundableLightBase): The non-boundable light prim.
        color (Gf.Vec3f): The color of the light.
        intensity (float): The intensity of the light.
        exposure (float): The exposure of the light.
    """
    if not light:
        raise ValueError("Invalid light prim.")
    color_attr = light.GetColorAttr()
    if color_attr:
        color_attr.Set(color)
    else:
        light.CreateColorAttr(color, True)
    intensity_attr = light.GetIntensityAttr()
    if intensity_attr:
        intensity_attr.Set(intensity)
    else:
        light.CreateIntensityAttr(intensity, True)
    exposure_attr = light.GetExposureAttr()
    if exposure_attr:
        exposure_attr.Set(exposure)
    else:
        light.CreateExposureAttr(exposure, True)


def animate_light_intensity(
    stage: Usd.Stage, light_path: str, keyframes: Dict[float, float], time_code_range: Tuple[float, float]
) -> bool:
    """
    Animate the intensity of a light over a given range of time.

    Args:
        stage (Usd.Stage): The USD stage.
        light_path (str): The path to the light prim.
        keyframes (Dict[float, float]): A dictionary of time values (keys) and intensity values (values).
        time_code_range (Tuple[float, float]): The range of time codes over which to animate the intensity.

    Returns:
        bool: True if the intensity was successfully animated, False otherwise.
    """
    light_prim = stage.GetPrimAtPath(light_path)
    if not light_prim.IsValid():
        print(f"Error: Light prim '{light_path}' does not exist.")
        return False
    light = UsdLux.NonboundableLightBase(light_prim)
    if not light:
        print(f"Error: Prim '{light_path}' is not a valid light.")
        return False
    intensity_attr = light.GetIntensityAttr()
    if not intensity_attr:
        print(f"Error: Light '{light_path}' does not have an intensity attribute.")
        return False
    try:
        for time_code in range(int(time_code_range[0]), int(time_code_range[1]) + 1):
            intensity_value = None
            for keyframe_time, keyframe_value in keyframes.items():
                if keyframe_time <= time_code:
                    intensity_value = keyframe_value
                else:
                    break
            if intensity_value is not None:
                intensity_attr.Set(intensity_value, Usd.TimeCode(time_code))
    except Exception as e:
        print(f"Error: Failed to animate intensity - {str(e)}")
        return False
    return True


def get_active_lights(stage: Usd.Stage) -> List[UsdLux.LightAPI]:
    """
    Get a list of all active light prims on the stage.

    Args:
        stage (Usd.Stage): The USD stage to search for lights.

    Returns:
        List[UsdLux.LightAPI]: A list of active light prims.
    """
    prims = stage.Traverse()
    active_lights = []
    for prim in prims:
        if prim.IsA(UsdLux.LightAPI):
            if prim.IsActive():
                light_schema = UsdLux.LightAPI(prim)
                if light_schema.GetPrim().IsValid():
                    if light_schema.GetEnableColorTemperatureAttr().Get():
                        active_lights.append(light_schema)
    return active_lights


def create_nonboundable_light(stage: Usd.Stage, prim_path: str, light_type: str) -> UsdLux.NonboundableLightBase:
    """Create a nonboundable light of a specific type.

    Args:
        stage (Usd.Stage): The USD stage to create the light on.
        prim_path (str): The path where the light should be created.
        light_type (str): The type of nonboundable light to create (e.g., "DomeLight", "RectLight").

    Returns:
        UsdLux.NonboundableLightBase: The created nonboundable light prim.

    Raises:
        ValueError: If an invalid light type is provided.
    """
    if light_type not in ["DomeLight", "RectLight"]:
        raise ValueError(f"Invalid light type: {light_type}. Must be 'DomeLight' or 'RectLight'.")
    light_class = getattr(UsdLux, light_type)
    light_prim = light_class.Define(stage, prim_path)
    light_prim.CreateIntensityAttr(1000.0)
    light_prim.CreateExposureAttr(1.0)
    light_prim.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    light_prim.CreateEnableColorTemperatureAttr(False)
    light_prim.CreateColorTemperatureAttr(6500.0)
    return light_prim


def toggle_light_filters(light: UsdLux.NonboundableLightBase, filters: List[Usd.Prim], enable: bool) -> None:
    """Toggles the specified light filters on or off for the given light.

    Args:
        light (UsdLux.NonboundableLightBase): The light to modify.
        filters (List[Usd.Prim]): A list of filter prims to toggle.
        enable (bool): True to enable the filters, False to disable them.
    """
    filters_rel = light.GetFiltersRel()
    current_filters = filters_rel.GetTargets()
    new_filters = []
    for filter_path in current_filters:
        if not enable and filter_path in [f.GetPath() for f in filters]:
            continue
        new_filters.append(filter_path)
    for filter_prim in filters:
        if not filter_prim.IsValid():
            print(f"Warning: Invalid filter prim: {filter_prim.GetPath()}")
            continue
        filter_path = filter_prim.GetPath()
        if enable and filter_path not in new_filters:
            new_filters.append(filter_path)
    filters_rel.SetTargets(new_filters)


def optimize_light_distribution(
    light: UsdLux.NonboundableLightBase, intensity: float, exposure: float, normalize: bool
) -> None:
    """Optimize the light distribution by setting intensity, exposure, and normalize attributes.

    Args:
        light (UsdLux.NonboundableLightBase): The light to optimize.
        intensity (float): The intensity value to set.
        exposure (float): The exposure value to set.
        normalize (bool): Whether to normalize the light.

    Raises:
        ValueError: If the light is not a valid NonboundableLightBase.
    """
    if not light or not isinstance(light, UsdLux.NonboundableLightBase):
        raise ValueError("Invalid NonboundableLightBase object")
    intensity_attr = light.GetIntensityAttr()
    if intensity_attr:
        intensity_attr.Set(intensity)
    exposure_attr = light.GetExposureAttr()
    if exposure_attr:
        exposure_attr.Set(exposure)
    normalize_attr = light.GetNormalizeAttr()
    if normalize_attr:
        normalize_attr.Set(normalize)


def align_light_to_target(light_prim: Usd.Prim, target_prim: Usd.Prim):
    """Align a light prim to point at a target prim."""
    if not light_prim.IsValid():
        raise ValueError("Invalid light prim")
    if not target_prim.IsValid():
        raise ValueError("Invalid target prim")
    light_xform = UsdGeom.Xformable(light_prim)
    target_xform = UsdGeom.Xformable(target_prim)
    light_pos = light_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    target_pos = target_xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    direction = target_pos - light_pos
    direction.Normalize()
    rotation = Gf.Rotation()
    rotation.SetRotateInto(Gf.Vec3d(0, 0, -1), Gf.Vec3d(direction))
    add_rotate_xyz_op(light_xform).Set(rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis()))


def get_all_plugin_lights(stage: Usd.Stage) -> List[UsdLux.PluginLight]:
    """
    Get all the plugin lights in the stage.

    Args:
        stage (Usd.Stage): The USD stage to search for plugin lights.

    Returns:
        List[UsdLux.PluginLight]: A list of all plugin lights in the stage.
    """
    root_prim = stage.GetPseudoRoot()
    plugin_lights = []
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdLux.PluginLight):
            plugin_light = UsdLux.PluginLight(prim)
            plugin_lights.append(plugin_light)
    return plugin_lights


def update_plugin_light_attributes(plugin_light: UsdLux.PluginLight, params: Dict[str, Any]) -> None:
    """Update the attributes of a PluginLight prim based on the provided parameters.

    Args:
        plugin_light (UsdLux.PluginLight): The PluginLight prim to update.
        params (Dict[str, Any]): A dictionary containing attribute names and their corresponding values.

    Raises:
        ValueError: If the provided prim is not a valid PluginLight prim.
    """
    if not plugin_light or not plugin_light.GetPrim().IsValid():
        raise ValueError("Invalid PluginLight prim.")
    for attr_name, attr_value in params.items():
        attr = plugin_light.GetPrim().GetAttribute(attr_name)
        if attr.IsDefined():
            attr.Set(attr_value)
        else:
            if isinstance(attr_value, float):
                value_type = Sdf.ValueTypeNames.Float
            elif isinstance(attr_value, Gf.Vec3f):
                value_type = Sdf.ValueTypeNames.Color3f
            else:
                value_type = Sdf.ValueTypeNames.String
            attr = plugin_light.GetPrim().CreateAttribute(attr_name, value_type)
            attr.Set(attr_value)


def create_plugin_light_filter(stage: Usd.Stage, prim_path: str, shading_node_id: str) -> UsdLux.PluginLightFilter:
    """Create a PluginLightFilter prim with a given shading node ID.

    Args:
        stage (Usd.Stage): The stage on which to create the PluginLightFilter prim.
        prim_path (str): The path at which to create the PluginLightFilter prim.
        shading_node_id (str): The identifier of the external shading node definition.

    Returns:
        UsdLux.PluginLightFilter: The created PluginLightFilter prim.

    Raises:
        ValueError: If the prim at the given path is not a valid PluginLightFilter prim.
    """
    plugin_light_filter = UsdLux.PluginLightFilter.Define(stage, prim_path)
    node_def_api = UsdShade.NodeDefAPI(plugin_light_filter)
    node_def_api.SetSourceAsset(shading_node_id, "")
    if not plugin_light_filter.GetPrim().IsValid():
        raise ValueError(f"Failed to create a valid PluginLightFilter prim at path {prim_path}")
    return plugin_light_filter


def find_plugin_light_filters_by_shader(stage: Usd.Stage, shader_id: str) -> List[UsdLux.PluginLightFilter]:
    """
    Find all UsdLuxPluginLightFilter prims on the given stage that reference the specified shader ID.

    Args:
        stage (Usd.Stage): The USD stage to search for plugin light filters.
        shader_id (str): The identifier of the shader to match against.

    Returns:
        List[UsdLux.PluginLightFilter]: A list of UsdLuxPluginLightFilter prims that reference the specified shader.
    """
    matching_filters = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdLux.PluginLightFilter):
            plugin_filter = UsdLux.PluginLightFilter(prim)
            node_def_api = plugin_filter.GetNodeDefAPI()
            if node_def_api.GetShaderId():
                prim_shader_id = node_def_api.GetShaderId()
                if prim_shader_id == shader_id:
                    matching_filters.append(plugin_filter)
    return matching_filters


def get_plugin_light_filter_shader_params(stage: Usd.Stage, light_filter_path: str) -> dict:
    """Get the shader parameters for a plugin light filter.

    Args:
        stage (Usd.Stage): The USD stage.
        light_filter_path (str): The path to the plugin light filter prim.

    Returns:
        dict: A dictionary containing the shader parameters.

    Raises:
        ValueError: If the prim at the given path is not a valid plugin light filter.
    """
    light_filter_prim = stage.GetPrimAtPath(light_filter_path)
    if not light_filter_prim.IsA(UsdLux.PluginLightFilter):
        raise ValueError(f"Prim at path {light_filter_path} is not a valid PluginLightFilter.")
    node_def_api = UsdShade.NodeDefAPI(light_filter_prim)
    shader_id = node_def_api.GetImplementationSourceAttr().Get()
    if not shader_id:
        return {}
    shader_prim = UsdShade.Shader.Get(stage, shader_id)
    if not shader_prim:
        return {}
    shader_params = {}
    for param in shader_prim.GetInputs():
        param_name = param.GetBaseName()
        param_value = param.Get()
        if param_value is not None:
            shader_params[param_name] = param_value
    return shader_params


def link_portal_light_to_dome_light(stage: Usd.Stage, portal_light_path: str, dome_light_path: str) -> None:
    """Link a portal light to a dome light.

    The portal light will guide the sampling of the specified dome light.

    Args:
        stage (Usd.Stage): The stage containing the portal light and dome light prims.
        portal_light_path (str): The path to the portal light prim.
        dome_light_path (str): The path to the dome light prim.

    Raises:
        ValueError: If either the portal light or dome light prim does not exist or is not of the correct type.
    """
    portal_light_prim = stage.GetPrimAtPath(portal_light_path)
    if not portal_light_prim.IsValid():
        raise ValueError(f"Prim at path {portal_light_path} does not exist.")
    portal_light = UsdLux.PortalLight(portal_light_prim)
    if not portal_light:
        raise ValueError(f"Prim at path {portal_light_path} is not a PortalLight.")
    dome_light_prim = stage.GetPrimAtPath(dome_light_path)
    if not dome_light_prim.IsValid():
        raise ValueError(f"Prim at path {dome_light_path} does not exist.")
    dome_light = UsdLux.DomeLight(dome_light_prim)
    if not dome_light:
        raise ValueError(f"Prim at path {dome_light_path} is not a DomeLight.")
    portal_light_prim.CreateRelationship("guiding:domeLight").AddTarget(dome_light_prim.GetPath())


def batch_create_portal_lights(
    stage: Usd.Stage, parent_path: str, positions: List[Tuple[float, float, float]]
) -> List[UsdLux.PortalLight]:
    """
    Batch create multiple portal lights under a common parent prim.

    Args:
        stage (Usd.Stage): The USD stage to create the lights on.
        parent_path (str): The path of the parent prim to create the lights under.
        positions (List[Tuple[float, float, float]]): A list of positions for the portal lights.

    Returns:
        List[UsdLux.PortalLight]: A list of the created portal light prims.
    """
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise ValueError(f"Invalid parent prim path: {parent_path}")
    portal_lights = []
    for i, position in enumerate(positions):
        light_path = f"{parent_path}/PortalLight_{i}"
        portal_light = UsdLux.PortalLight.Define(stage, light_path)
        UsdGeom.XformCommonAPI(portal_light).SetTranslate(Gf.Vec3d(position))
        portal_lights.append(portal_light)
    return portal_lights


def create_portal_light_with_transform(
    stage: Usd.Stage, prim_path: str, translation: Gf.Vec3f, rotation_euler: Gf.Vec3f, scale: Gf.Vec3f
) -> UsdLux.PortalLight:
    """
    Create a UsdLuxPortalLight prim with the specified transform.

    Args:
        stage (Usd.Stage): The USD stage to create the prim on.
        prim_path (str): The path where the prim should be created.
        translation (Gf.Vec3f): The translation of the prim.
        rotation_euler (Gf.Vec3f): The rotation (in Euler angles) of the prim.
        scale (Gf.Vec3f): The scale of the prim.

    Returns:
        UsdLux.PortalLight: The created PortalLight prim.
    """
    if not Sdf.Path(prim_path).IsAbsolutePath():
        raise ValueError("Prim path must be an absolute path.")
    portal_light = UsdLux.PortalLight.Define(stage, prim_path)
    if not portal_light:
        raise RuntimeError(f"Failed to create PortalLight prim at path {prim_path}")
    xform = UsdGeom.Xformable(portal_light)
    add_translate_op(xform).Set(translation)
    add_rotate_xyz_op(xform).Set(rotation_euler)
    add_scale_op(xform).Set(scale)
    return portal_light


def create_rect_light_with_texture(
    stage: Usd.Stage, prim_path: str, texture_path: str, width: float = 1.0, height: float = 1.0
) -> UsdLux.RectLight:
    """Creates a RectLight prim with a texture applied.

    Args:
        stage (Usd.Stage): The USD stage to create the light in.
        prim_path (str): The path where the light should be created.
        texture_path (str): The path to the texture file to apply.
        width (float): The width of the light. Defaults to 1.0.
        height (float): The height of the light. Defaults to 1.0.

    Returns:
        UsdLux.RectLight: The created RectLight prim.

    Raises:
        ValueError: If the prim path is invalid or the texture file does not exist.
    """
    if not Sdf.Path(prim_path).IsAbsolutePath():
        raise ValueError(f"Prim path '{prim_path}' is not an absolute path.")
    light = UsdLux.RectLight.Define(stage, prim_path)
    if not light:
        raise ValueError(f"Failed to create RectLight prim at path '{prim_path}'.")
    light.CreateWidthAttr().Set(width)
    light.CreateHeightAttr().Set(height)
    texture_attr = light.CreateTextureFileAttr()
    texture_attr.Set(texture_path)
    return light


def set_rect_light_transform(
    stage: Usd.Stage, rect_light_path: str, translation: Gf.Vec3f, rotation: Gf.Vec3f, scale: Gf.Vec3f
) -> None:
    """Set the transform for a RectLight prim.

    Args:
        stage (Usd.Stage): The USD stage.
        rect_light_path (str): The path to the RectLight prim.
        translation (Gf.Vec3f): The translation vector.
        rotation (Gf.Vec3f): The rotation vector in degrees (Euler angles).
        scale (Gf.Vec3f): The scale vector.
    """
    rect_light_prim = stage.GetPrimAtPath(rect_light_path)
    if not rect_light_prim.IsValid():
        raise ValueError(f"RectLight prim at path {rect_light_path} does not exist.")
    if not UsdLux.RectLight(rect_light_prim):
        raise ValueError(f"Prim at path {rect_light_path} is not a RectLight.")
    xformable = UsdGeom.Xformable(rect_light_prim)
    xformable.ClearXformOpOrder()
    translate_op = add_translate_op(xformable)
    translate_op.Set(translation)
    rotate_op = add_rotate_xyz_op(xformable)
    rotation_rad = Gf.Vec3f(
        Gf.RadiansToDegrees(rotation[0]), Gf.RadiansToDegrees(rotation[1]), Gf.RadiansToDegrees(rotation[2])
    )
    rotate_op.Set(rotation_rad)
    scale_op = add_scale_op(xformable)
    scale_op.Set(scale)


def set_rect_light_intensity(stage: Usd.Stage, prim_path: str, intensity: float) -> None:
    """Set the intensity of a rectangular light.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the rectangular light prim.
        intensity (float): The intensity value to set.

    Raises:
        ValueError: If the prim at the given path is not a rectangular light.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    rect_light = UsdLux.RectLight(prim)
    if not rect_light:
        raise ValueError(f"Prim at path {prim_path} is not a rectangular light")
    intensity_attr = rect_light.GetIntensityAttr()
    if intensity_attr:
        intensity_attr.Set(intensity)
    else:
        rect_light.CreateIntensityAttr(intensity)


def get_rect_light_attributes(stage: Usd.Stage, prim_path: str) -> Tuple[float, float, str]:
    """Get the attributes of a UsdLuxRectLight prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the UsdLuxRectLight prim.

    Returns:
        Tuple[float, float, str]: A tuple containing the width, height, and texture file path.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    rect_light = UsdLux.RectLight(prim)
    if not rect_light:
        raise ValueError(f"Prim at path {prim_path} is not a UsdLuxRectLight.")
    width_attr = rect_light.GetWidthAttr()
    if width_attr.HasAuthoredValue():
        width = width_attr.Get()
    else:
        width = 1.0
    height_attr = rect_light.GetHeightAttr()
    if height_attr.HasAuthoredValue():
        height = height_attr.Get()
    else:
        height = 1.0
    texture_file_attr = rect_light.GetTextureFileAttr()
    if texture_file_attr.HasAuthoredValue():
        texture_file = texture_file_attr.Get().path
    else:
        texture_file = ""
    return (width, height, texture_file)


def create_and_assign_rect_light_material(
    stage: Usd.Stage, rect_light_path: str, texture_file: str, width: float, height: float
) -> UsdLux.RectLight:
    """Create a RectLight prim and assign a texture to it.

    Args:
        stage (Usd.Stage): The USD stage to create the RectLight on.
        rect_light_path (str): The path where the RectLight should be created.
        texture_file (str): The path to the texture file to assign to the RectLight.
        width (float): The width of the RectLight.
        height (float): The height of the RectLight.

    Returns:
        UsdLux.RectLight: The created RectLight prim.

    Raises:
        ValueError: If the RectLight prim cannot be created or the texture file cannot be assigned.
    """
    rect_light = UsdLux.RectLight.Define(stage, rect_light_path)
    if not rect_light:
        raise ValueError(f"Failed to create RectLight prim at path {rect_light_path}")
    width_attr = rect_light.CreateWidthAttr(width)
    height_attr = rect_light.CreateHeightAttr(height)
    texture_attr = rect_light.CreateTextureFileAttr(texture_file)
    if not texture_attr:
        raise ValueError(f"Failed to assign texture file {texture_file} to RectLight {rect_light_path}")
    return rect_light


def adjust_rect_light_brightness(rect_light: UsdLux.RectLight, brightness: float) -> None:
    """Adjusts the brightness of a RectLight by scaling its inputs:intensity.

    Args:
        rect_light (UsdLux.RectLight): The RectLight prim to adjust.
        brightness (float): The brightness multiplier. Must be non-negative.

    Raises:
        ValueError: If rect_light is not a valid RectLight prim or if brightness is negative.
    """
    if not rect_light or not rect_light.GetPrim().IsValid():
        raise ValueError("Invalid RectLight prim.")
    if brightness < 0:
        raise ValueError("Brightness must be non-negative.")
    intensity_attr = rect_light.GetIntensityAttr()
    if intensity_attr.HasAuthoredValue():
        current_intensity = intensity_attr.Get()
    else:
        current_intensity = 1.0
    new_intensity = current_intensity * brightness
    intensity_attr.Set(new_intensity)


def batch_update_rect_lights(
    stage: Usd.Stage, rect_light_paths: List[str], width: float, height: float, texture_file: str
) -> None:
    """Update multiple rect lights with the given parameters.

    Args:
        stage (Usd.Stage): The USD stage.
        rect_light_paths (List[str]): A list of paths to the rect lights to update.
        width (float): The width of the rect lights.
        height (float): The height of the rect lights.
        texture_file (str): The path to the texture file for the rect lights.
    """
    for rect_light_path in rect_light_paths:
        rect_light_prim = stage.GetPrimAtPath(rect_light_path)
        if not rect_light_prim.IsValid():
            print(f"Warning: Invalid prim at path {rect_light_path}. Skipping.")
            continue
        rect_light = UsdLux.RectLight(rect_light_prim)
        if not rect_light:
            print(f"Warning: Prim at path {rect_light_path} is not a RectLight. Skipping.")
            continue
        width_attr = rect_light.GetWidthAttr()
        if width_attr.IsValid():
            width_attr.Set(width)
        else:
            print(f"Warning: Failed to get width attribute for rect light at path {rect_light_path}.")
        height_attr = rect_light.GetHeightAttr()
        if height_attr.IsValid():
            height_attr.Set(height)
        else:
            print(f"Warning: Failed to get height attribute for rect light at path {rect_light_path}.")
        texture_file_attr = rect_light.GetTextureFileAttr()
        if texture_file_attr.IsValid():
            texture_file_attr.Set(texture_file)
        else:
            print(f"Warning: Failed to get texture file attribute for rect light at path {rect_light_path}.")


def apply_shadow_api_to_prim(prim: Usd.Prim) -> UsdLux.ShadowAPI:
    """Applies the ShadowAPI schema to a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not UsdLux.ShadowAPI.CanApply(prim):
        raise RuntimeError(f"ShadowAPI schema cannot be applied to prim {prim.GetPath()}.")
    shadow_api = UsdLux.ShadowAPI.Apply(prim)
    if not shadow_api.GetPrim():
        raise RuntimeError(f"Failed to apply ShadowAPI schema to prim {prim.GetPath()}.")
    return shadow_api


def can_apply_shadow_api_to_prim(prim: Usd.Prim) -> bool:
    """Check if the ShadowAPI schema can be applied to the given prim."""
    if not prim.IsValid():
        return False
    prim_type = prim.GetTypeName()
    if prim_type not in [
        "DistantLight",
        "DiskLight",
        "DomeLight",
        "RectLight",
        "SphereLight",
        "CylinderLight",
        "LightFilter",
    ]:
        return False
    if prim.HasAPI(UsdLux.ShadowAPI):
        return False
    return True


def create_shadow_distance_attr(
    shadow_api: UsdLux.ShadowAPI, default_value: float = -1.0, write_sparsely: bool = False
) -> Usd.Attribute:
    """Creates the shadow distance attribute on the given UsdLuxShadowAPI prim.

    Args:
        shadow_api (UsdLux.ShadowAPI): The shadow API prim to create the attribute on.
        default_value (float, optional): The default value for the attribute. Defaults to -1.0.
        write_sparsely (bool, optional): Whether to write the attribute sparsely. Defaults to False.

    Returns:
        Usd.Attribute: The created shadow distance attribute.

    Raises:
        ValueError: If the given prim is not a valid UsdLuxShadowAPI prim.
    """
    if not shadow_api.GetPrim().IsValid():
        raise ValueError("The given prim is not valid.")
    shadow_distance_attr = shadow_api.CreateShadowDistanceAttr(default_value, write_sparsely)
    if not shadow_distance_attr.IsValid():
        raise ValueError("Failed to create the shadow distance attribute.")
    return shadow_distance_attr


def create_shadow_enable_attr(
    shadow_api: UsdLux.ShadowAPI, default_value: bool = True, write_sparsely: bool = False
) -> Usd.Attribute:
    """Create the shadow enable attribute on the given shadow API prim.

    Args:
        shadow_api (UsdLux.ShadowAPI): The shadow API prim.
        default_value (bool): The default value for the attribute. Default is True.
        write_sparsely (bool): Whether to write the attribute sparsely. Default is False.

    Returns:
        Usd.Attribute: The created shadow enable attribute.

    Raises:
        ValueError: If the given prim is not a valid shadow API prim.
    """
    if not shadow_api.GetPrim().IsValid():
        raise ValueError("The given prim is not a valid shadow API prim.")
    shadow_enable_attr = shadow_api.CreateShadowEnableAttr(default_value, write_sparsely)
    if not shadow_enable_attr.IsValid():
        raise ValueError("Failed to create the shadow enable attribute.")
    return shadow_enable_attr


def create_shadow_falloff_attr(
    prim: Usd.Prim, defaultValue: float = -1.0, writeSparsely: bool = False
) -> Usd.Attribute:
    """Creates the shadow falloff attribute on the given prim.

    The near distance at which shadow falloff begins. The default value (-1) indicates no falloff.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        defaultValue (float): The default value for the attribute. Default is -1.0.
        writeSparsely (bool): Whether to write the attribute sparsely. Default is False.

    Returns:
        Usd.Attribute: The created attribute.
    """
    shadowAPI = UsdLux.ShadowAPI(prim)
    if not shadowAPI:
        raise ValueError(f"ShadowAPI schema not applied to prim {prim.GetPath()}")
    attr = shadowAPI.CreateShadowFalloffAttr(defaultValue, writeSparsely)
    if not attr:
        raise RuntimeError(f"Failed to create shadow falloff attribute on prim {prim.GetPath()}")
    return attr


def get_shadow_color_attr(shadow_api: UsdLux.ShadowAPI) -> Gf.Vec3f:
    """Get the shadow color attribute value for a ShadowAPI object."""
    color_attr = shadow_api.GetShadowColorAttr()
    if not color_attr.IsValid():
        return Gf.Vec3f(0.0, 0.0, 0.0)
    color = color_attr.Get()
    if color is None:
        return Gf.Vec3f(0.0, 0.0, 0.0)
    return color


def get_shadow_enable_attr(shadow_api: UsdLux.ShadowAPI) -> Usd.Attribute:
    """Get the shadow enable attribute from a ShadowAPI schema.

    Args:
        shadow_api (UsdLux.ShadowAPI): The ShadowAPI schema object.

    Returns:
        Usd.Attribute: The shadow enable attribute.

    Raises:
        ValueError: If the ShadowAPI schema is invalid.
    """
    if not shadow_api.GetPrim().IsValid():
        raise ValueError("Invalid ShadowAPI schema.")
    shadow_enable_attr = shadow_api.GetShadowEnableAttr()
    if not shadow_enable_attr.IsValid():
        shadow_enable_attr = shadow_api.CreateShadowEnableAttr(True, True)
    return shadow_enable_attr


def get_shadow_falloff_gamma_attr(shadow_api: UsdLux.ShadowAPI) -> Usd.Attribute:
    """Get the shadow falloff gamma attribute from a ShadowAPI object."""
    if not shadow_api:
        raise ValueError("Invalid ShadowAPI object")
    attr = shadow_api.GetShadowFalloffGammaAttr()
    if not attr:
        raise ValueError("Shadow falloff gamma attribute not found")
    return attr


def set_shadow_distance(prim: Usd.Prim, distance: float):
    """Set the shadow distance for a prim with ShadowAPI applied."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    if not prim.HasAPI(UsdLux.ShadowAPI):
        raise ValueError(f"Prim {prim.GetPath()} does not have ShadowAPI applied")
    shadow_api = UsdLux.ShadowAPI(prim)
    shadow_distance_attr = shadow_api.GetShadowDistanceAttr()
    if shadow_distance_attr.IsValid():
        shadow_distance_attr.Set(distance)
    else:
        shadow_api.CreateShadowDistanceAttr(distance, writeSparsely=True)


def get_shadow_api(stage: Usd.Stage, prim_path: str) -> UsdLux.ShadowAPI:
    """Get the ShadowAPI for a prim, if it exists."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.HasAPI(UsdLux.ShadowAPI):
        raise ValueError(f"Prim at path {prim_path} does not have ShadowAPI applied.")
    shadow_api = UsdLux.ShadowAPI(prim)
    return shadow_api


def create_shadow_falloff_gamma_attr(self, defaultValue=1.0, writeSparsely=False) -> Usd.Attribute:
    """Create the shadow falloff gamma attribute.

    A gamma (i.e., exponential) control over shadow strength with linear
    distance within the falloff zone.

    This requires the use of shadowDistance and shadowFalloff.

    Args:
        defaultValue (float): The default value for the attribute. Default is 1.0.
        writeSparsely (bool): Whether to write sparsely. Default is False.

    Returns:
        Usd.Attribute: The created attribute.
    """
    if not isinstance(defaultValue, float):
        raise TypeError("defaultValue must be a float")
    if not isinstance(writeSparsely, bool):
        raise TypeError("writeSparsely must be a bool")
    return self.CreateShadowFalloffGammaAttr(defaultValue, writeSparsely)


def get_shadow_output(shadow_api: UsdLux.ShadowAPI, name: str) -> UsdShade.Output:
    """
    Get the requested output from the shadow API.

    Args:
        shadow_api (UsdLux.ShadowAPI): The shadow API object.
        name (str): The name of the output to retrieve.

    Returns:
        UsdShade.Output: The requested output object.

    Raises:
        ValueError: If the shadow API is invalid or the requested output does not exist.
    """
    if not shadow_api:
        raise ValueError("Invalid shadow API object.")
    output = shadow_api.GetOutput(name)
    if not output:
        raise ValueError(f"Output '{name}' does not exist on the shadow API.")
    return output


def GetShadowDistanceAttr(self) -> Usd.Attribute:
    """Get the shadow distance attribute for this shadow API schema.

    Returns:
        Usd.Attribute: The shadow distance attribute.

    Raises:
        ValueError: If the attribute does not exist.
    """
    attr = self.GetPrim().GetAttribute("inputs:shadow:distance")
    if not attr.IsDefined():
        raise ValueError("Shadow distance attribute does not exist.")
    return attr


def get_shadow_falloff_attr(shadow_api: UsdLux.ShadowAPI) -> Usd.Attribute:
    """Get the attribute for shadow falloff distance."""
    attr = shadow_api.GetPrim().GetAttribute("inputs:shadow:falloff")
    if attr.IsDefined():
        return attr
    if shadow_api.GetPrim().HasAttribute("inputs:shadow:falloff"):
        return attr
    raise AttributeError("Attribute 'inputs:shadow:falloff' does not exist on prim.")


def set_shadow_enable(prim: Usd.Prim, enable: bool) -> None:
    """Set the shadow enable attribute on a prim."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not UsdLux.ShadowAPI.CanApply(prim):
        raise ValueError(f"ShadowAPI cannot be applied to prim: {prim}")
    if not prim.HasAPI(UsdLux.ShadowAPI):
        UsdLux.ShadowAPI.Apply(prim)
    shadow_enable_attr = UsdLux.ShadowAPI(prim).GetShadowEnableAttr()
    shadow_enable_attr.Set(enable)


def set_shadow_falloff(shadow_api: UsdLux.ShadowAPI, value: float) -> None:
    """Set the shadow falloff attribute on the given ShadowAPI prim.

    Args:
        shadow_api (UsdLux.ShadowAPI): The ShadowAPI prim to set the attribute on.
        value (float): The value to set for the shadow falloff attribute.

    Raises:
        ValueError: If the provided shadow_api is not valid.
    """
    if not shadow_api.GetPrim().IsValid():
        raise ValueError("Invalid ShadowAPI prim")
    falloff_attr = shadow_api.GetShadowFalloffAttr()
    falloff_attr.Set(value)


def create_shadow_output(self, name: str, typeName: Sdf.ValueTypeName) -> Usd.Attribute:
    """Create an output attribute for the shadow API.

    The attribute representing the output is created in the "outputs:" namespace.
    Outputs on a shadow API cannot be connected, as their value is assumed to be
    computed externally.

    Args:
        name (str): The name of the output attribute.
        typeName (Sdf.ValueTypeName): The value type of the output attribute.

    Returns:
        Usd.Attribute: The created output attribute.

    Raises:
        ValueError: If the attribute already exists or an invalid type is provided.
    """
    attr = self.GetPrim().GetAttribute(f"outputs:{name}")
    if attr:
        raise ValueError(f"Output attribute 'outputs:{name}' already exists.")
    if not typeName:
        raise ValueError("Invalid typeName provided for the output attribute.")
    attr = self.GetPrim().CreateAttribute(f"outputs:{name}", typeName, False)
    return attr


def create_shadow_input(
    shadow_api: UsdLux.ShadowAPI, name: str, value: Any, type_name: Sdf.ValueTypeName
) -> UsdShade.Input:
    """Create a shadow input attribute with the given name, value, and type."""
    if not Sdf.Path.IsValidIdentifier(name):
        raise ValueError(f"Invalid input name: {name}")
    input_attr = shadow_api.GetInput(name)
    if input_attr:
        input_attr.Set(value)
        return input_attr
    input_attr = shadow_api.CreateInput(name, type_name)
    input_attr.Set(value)
    return input_attr


def create_shadow_color_attr(
    self, default_value: Gf.Vec3f = Gf.Vec3f(0.0, 0.0, 0.0), write_sparsely: bool = False
) -> Usd.Attribute:
    """Create the shadow color attribute with the given default value and sparsity.

    Args:
        default_value (Gf.Vec3f): The default color value for the shadow. Defaults to black (0.0, 0.0, 0.0).
        write_sparsely (bool): Whether to write the attribute sparsely. Defaults to False.

    Returns:
        Usd.Attribute: The created shadow color attribute.
    """
    if not isinstance(default_value, Gf.Vec3f):
        raise TypeError(f"Default value must be a Gf.Vec3f, not {type(default_value)}")
    return self.CreateShadowColorAttr(default_value, write_sparsely)


def get_shadow_input(shadow_api: UsdLux.ShadowAPI, input_name: str) -> UsdShade.Input:
    """
    Get a specific shadow input from a UsdLuxShadowAPI object.

    Args:
        shadow_api (UsdLux.ShadowAPI): The ShadowAPI object to retrieve the input from.
        input_name (str): The name of the input to retrieve.

    Returns:
        UsdShade.Input: The requested input if it exists, otherwise None.
    """
    if not shadow_api.GetPrim().IsValid():
        raise ValueError("Invalid ShadowAPI object provided.")
    input_attr = shadow_api.GetInput(input_name)
    if not input_attr:
        print(f"Input '{input_name}' does not exist on the ShadowAPI.")
        return None
    return input_attr


def create_shaping_attributes(
    shaping_api: UsdLux.ShapingAPI,
    angle: float = 90.0,
    softness: float = 0.0,
    focus: float = 0.0,
    focus_tint: Gf.Vec3f = Gf.Vec3f(0, 0, 0),
    ies_file: str = "",
    ies_angle_scale: float = 0.0,
    ies_normalize: bool = False,
):
    """Create shaping attributes on a UsdLuxShapingAPI prim.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI prim to create attributes on.
        angle (float, optional): Angular limit off the primary axis to restrict the light spread. Defaults to 90.0.
        softness (float, optional): Controls the cutoff softness for cone angle. Defaults to 0.0.
        focus (float, optional): A control to shape the spread of light. Higher focus values pull light towards the center and narrow the spread. Defaults to 0.0.
        focus_tint (Gf.Vec3f, optional): Off-axis color tint. This tints the emission in the falloff region. Defaults to Gf.Vec3f(0, 0, 0).
        ies_file (str, optional): An IES light profile file path describing the angular distribution of light. Defaults to "".
        ies_angle_scale (float, optional): Rescales the angular distribution of the IES profile. Defaults to 0.0.
        ies_normalize (bool, optional): Normalizes the IES profile so that it affects the shaping of the light while preserving the overall energy output. Defaults to False.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI prim.")
    shaping_api.CreateShapingConeAngleAttr(angle, writeSparsely=True)
    shaping_api.CreateShapingConeSoftnessAttr(softness, writeSparsely=True)
    shaping_api.CreateShapingFocusAttr(focus, writeSparsely=True)
    shaping_api.CreateShapingFocusTintAttr(focus_tint, writeSparsely=True)
    if ies_file:
        shaping_api.CreateShapingIesFileAttr(ies_file, writeSparsely=True)
        shaping_api.CreateShapingIesAngleScaleAttr(ies_angle_scale, writeSparsely=True)
        shaping_api.CreateShapingIesNormalizeAttr(ies_normalize, writeSparsely=True)


def set_shaping_focus(prim: Usd.Prim, focus: float):
    """Set the shaping:focus attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the shaping:focus attribute on.
        focus (float): The focus value to set.

    Raises:
        ValueError: If the prim is not valid or does not have the ShapingAPI applied.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    shaping_api = UsdLux.ShapingAPI(prim)
    if not shaping_api:
        raise ValueError(f"Prim {prim} does not have the ShapingAPI applied")
    focus_attr = shaping_api.GetShapingFocusAttr()
    focus_attr.Set(focus)


def set_shaping_ies_angle_scale(prim: Usd.Prim, angle_scale: float) -> None:
    """Set the shaping:ies:angleScale attribute on a prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not UsdLux.ShapingAPI.CanApply(prim):
        raise ValueError(f"Prim {prim.GetPath()} does not support the ShapingAPI.")
    shaping_api = UsdLux.ShapingAPI.Apply(prim)
    shaping_api.CreateShapingIesAngleScaleAttr(angle_scale)


def get_shaping_cone_softness(shaping_api: UsdLux.ShapingAPI) -> float:
    """Get the shaping cone softness value from a ShapingAPI object.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI object to retrieve the value from.

    Returns:
        float: The shaping cone softness value, or 0.0 if not authored.

    Raises:
        ValueError: If the input shaping_api is not a valid UsdLux.ShapingAPI object.
    """
    if not isinstance(shaping_api, UsdLux.ShapingAPI):
        raise ValueError("Input must be a valid UsdLux.ShapingAPI object.")
    softness_attr = shaping_api.GetShapingConeSoftnessAttr()
    if softness_attr.IsDefined():
        softness_value = softness_attr.Get(Usd.TimeCode.Default())
        if softness_value is not None:
            return softness_value
    return 0.0


def get_shaping_focus(shaping_api: UsdLux.ShapingAPI, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> float:
    """
    Get the shaping focus value for a light with the given ShapingAPI at the specified time code.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI object for the light.
        time_code (Usd.TimeCode): The time code at which to retrieve the value. Defaults to Default.

    Returns:
        float: The shaping focus value, or 0.0 if the attribute is not authored or has no value.

    Raises:
        ValueError: If the given ShapingAPI is invalid.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI object.")
    focus_attr = shaping_api.GetShapingFocusAttr()
    if focus_attr.HasAuthoredValue():
        focus_value = focus_attr.Get(time_code)
        if focus_value is not None:
            return float(focus_value)
    return 0.0


def get_shaping_ies_angle_scale(shaping_api: UsdLux.ShapingAPI) -> float:
    """
    Get the value of the shaping:ies:angleScale attribute from a UsdLuxShapingAPI.

    :param shaping_api: The UsdLuxShapingAPI object.
    :return: The value of the shaping:ies:angleScale attribute, or 0.0 if not defined.
    :raises ValueError: If the input shaping_api is not a valid UsdLuxShapingAPI object.
    """
    if not isinstance(shaping_api, UsdLux.ShapingAPI):
        raise ValueError("Input argument must be a valid UsdLuxShapingAPI object.")
    angle_scale_attr = shaping_api.GetShapingIesAngleScaleAttr()
    if angle_scale_attr.HasAuthoredValue():
        return angle_scale_attr.Get()
    else:
        return 0.0


def get_shaping_ies_file(shaping_api: UsdLux.ShapingAPI) -> Optional[str]:
    """Get the IES file path from a UsdLuxShapingAPI object."""
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI object")
    ies_file_attr = shaping_api.GetShapingIesFileAttr()
    if ies_file_attr.IsAuthored():
        ies_file_path = ies_file_attr.Get().path
        return ies_file_path
    else:
        return None


def get_shaping_ies_normalize(shaping_api: UsdLux.ShapingAPI) -> bool:
    """Get the value of the shaping:ies:normalize attribute.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI object.

    Returns:
        bool: The value of the shaping:ies:normalize attribute. If the attribute is
            not defined, returns False.

    Raises:
        ValueError: If the shaping_api object is not valid.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI object.")
    attr = shaping_api.GetShapingIesNormalizeAttr()
    if attr.IsDefined():
        return attr.Get()
    else:
        return False


def batch_apply_shaping_api(stage: Usd.Stage, prim_paths: List[str]) -> List[UsdLux.ShapingAPI]:
    """Apply UsdLux.ShapingAPI to multiple prims in a stage.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to apply the ShapingAPI to.

    Returns:
        List[UsdLux.ShapingAPI]: List of ShapingAPI objects for each prim path.
    """
    shaping_apis = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        if not UsdLux.ShapingAPI.CanApply(prim):
            print(f"Warning: ShapingAPI cannot be applied to prim at path {prim_path}. Skipping.")
            continue
        shaping_api = UsdLux.ShapingAPI.Apply(prim)
        if not shaping_api.GetPrim():
            print(f"Warning: Failed to apply ShapingAPI to prim at path {prim_path}. Skipping.")
            continue
        shaping_apis.append(shaping_api)
    return shaping_apis


def create_shaping_input(
    shaping_api: UsdLux.ShapingAPI, input_name: str, input_type: Sdf.ValueTypeName, default_value=None
) -> UsdShade.Input:
    """Create a shaping input on the given ShapingAPI prim.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI prim to create the input on.
        input_name (str): The name of the input attribute.
        input_type (Sdf.ValueTypeName): The value type of the input attribute.
        default_value (Optional): The default value for the input attribute. Defaults to None.

    Returns:
        UsdShade.Input: The created input attribute.

    Raises:
        ValueError: If the ShapingAPI prim is invalid or if the input name is empty.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI prim.")
    if not input_name:
        raise ValueError("Input name cannot be empty.")
    input_attr = shaping_api.CreateInput(input_name, input_type)
    if default_value is not None:
        input_attr.Set(default_value)
    return input_attr


def create_shaping_output(shaping_api: UsdLux.ShapingAPI, name: str, type_name: Sdf.ValueTypeName) -> UsdShade.Output:
    """Create a shaping output on the given UsdLuxShapingAPI prim.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI schema object.
        name (str): The name of the output attribute.
        type_name (Sdf.ValueTypeName): The value type of the output attribute.

    Returns:
        UsdShade.Output: The created output attribute or an invalid output if creation fails.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI object.")
    existing_output = shaping_api.GetOutput(name)
    if existing_output:
        print(f"Warning: Output '{name}' already exists on prim '{shaping_api.GetPath()}'.")
        return existing_output
    output = shaping_api.CreateOutput(name, type_name)
    if not output:
        raise RuntimeError(f"Failed to create output '{name}' on prim '{shaping_api.GetPath()}'.")
    return output


def get_all_shaping_inputs(shaping_api: UsdLux.ShapingAPI) -> Dict[str, Usd.Attribute]:
    """
    Get all shaping inputs for a ShapingAPI prim.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI prim.

    Returns:
        Dict[str, Usd.Attribute]: A dictionary mapping input names to their corresponding attributes.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI prim.")
    input_attrs = shaping_api.GetInputs()
    inputs_dict = {}
    for input_attr in input_attrs:
        input_name = input_attr.GetBaseName()
        inputs_dict[input_name] = input_attr
    return inputs_dict


def get_all_shaping_outputs(shaping_api: UsdLux.ShapingAPI, only_authored: bool = True) -> Dict[str, UsdShade.Output]:
    """
    Get all shaping outputs from a UsdLux.ShapingAPI object.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI object to retrieve outputs from.
        only_authored (bool, optional): If True, only return authored outputs. Defaults to True.

    Returns:
        Dict[str, UsdShade.Output]: A dictionary mapping output names to UsdShade.Output objects.
    """
    prim = shaping_api.GetPrim()
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    output_attrs = prim.GetAttributes()
    if only_authored:
        output_attrs = [attr for attr in output_attrs if attr.IsAuthored()]
    outputs = {}
    for attr in output_attrs:
        attr_name = attr.GetName()
        if attr_name.startswith("outputs:"):
            output_name = attr_name[len("outputs:") :]
            output = UsdShade.Output(attr)
            outputs[output_name] = output
    return outputs


def set_shaping_cone_softness(
    shaping_api: UsdLux.ShapingAPI, softness: float, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Set the shaping cone softness value for a light using the ShapingAPI.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI object for the light.
        softness (float): The softness value to set (controls cutoff softness for cone angle).
        time_code (Usd.TimeCode, optional): The time code at which to set the value. Defaults to Default time code.

    Raises:
        ValueError: If the ShapingAPI object is invalid.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("The ShapingAPI object is not valid.")
    softness_attr = shaping_api.GetShapingConeSoftnessAttr()
    if not softness_attr:
        softness_attr = shaping_api.CreateShapingConeSoftnessAttr(softness, True)
    softness_attr.Set(softness, time_code)


def set_shaping_focus_tint(shaping_api: UsdLux.ShapingAPI, tint_value: Gf.Vec3f) -> None:
    """Set the shaping focus tint value for a light.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI object for the light.
        tint_value (Gf.Vec3f): The focus tint color value to set.

    Raises:
        ValueError: If the shaping API is invalid.
    """
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("Invalid ShapingAPI object.")
    focus_tint_attr = shaping_api.GetShapingFocusTintAttr()
    if focus_tint_attr:
        focus_tint_attr.Set(tint_value)
    else:
        focus_tint_attr = shaping_api.CreateShapingFocusTintAttr(tint_value, writeSparsely=True)


def set_shaping_ies_file(shaping_api: UsdLux.ShapingAPI, ies_file_path: str):
    """Set the IES file path for the shaping API."""
    if not shaping_api.GetPrim().IsValid():
        raise ValueError("ShapingAPI is not applied to a valid prim")
    ies_file_attr = shaping_api.GetShapingIesFileAttr()
    if not ies_file_attr:
        ies_file_attr = shaping_api.CreateShapingIesFileAttr()
    ies_file_attr.Set(ies_file_path)


def set_shaping_ies_normalize(
    shaping_api: UsdLux.ShapingAPI, value: bool, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the shaping:ies:normalize attribute value.

    Args:
        shaping_api (UsdLux.ShapingAPI): The shaping API schema instance.
        value (bool): The value to set for the normalize attribute.
        time_code (Usd.TimeCode, optional): The time code at which to set the value. Defaults to Default().
    """
    normalize_attr = shaping_api.GetShapingIesNormalizeAttr()
    if not normalize_attr:
        normalize_attr = shaping_api.CreateShapingIesNormalizeAttr(value, True)
    else:
        normalize_attr.Set(value, time_code)


def validate_shaping_attributes(shaping_api: UsdLux.ShapingAPI) -> bool:
    """Validate the shaping attributes on a UsdLux.ShapingAPI object.

    Returns True if all attributes are valid, False otherwise.
    """
    if not shaping_api.GetPrim().IsValid():
        print("Error: Invalid prim.")
        return False
    cone_angle_attr = shaping_api.GetShapingConeAngleAttr()
    if cone_angle_attr.IsAuthored():
        cone_angle = cone_angle_attr.Get()
        if cone_angle < 0.0 or cone_angle > 180.0:
            print(f"Error: Invalid cone angle value: {cone_angle}")
            return False
    cone_softness_attr = shaping_api.GetShapingConeSoftnessAttr()
    if cone_softness_attr.IsAuthored():
        cone_softness = cone_softness_attr.Get()
        if cone_softness < 0.0 or cone_softness > 1.0:
            print(f"Error: Invalid cone softness value: {cone_softness}")
            return False
    focus_attr = shaping_api.GetShapingFocusAttr()
    if focus_attr.IsAuthored():
        focus = focus_attr.Get()
        if focus < 0.0 or focus > 1.0:
            print(f"Error: Invalid focus value: {focus}")
            return False
    ies_angle_scale_attr = shaping_api.GetShapingIesAngleScaleAttr()
    if ies_angle_scale_attr.IsAuthored():
        ies_angle_scale = ies_angle_scale_attr.Get()
        if ies_angle_scale < 0.0:
            print(f"Error: Invalid IES angle scale value: {ies_angle_scale}")
            return False
    return True


def get_shaping_attributes(shaping_api: UsdLux.ShapingAPI) -> Dict[str, Any]:
    """
    Get all the shaping attributes for a given ShapingAPI prim.

    Args:
        shaping_api (UsdLux.ShapingAPI): The ShapingAPI prim to retrieve attributes from.

    Returns:
        Dict[str, Any]: A dictionary containing attribute names and their values.
    """
    attributes = {}
    if shaping_api.GetShapingConeAngleAttr().HasValue():
        attributes["cone_angle"] = shaping_api.GetShapingConeAngleAttr().Get()
    if shaping_api.GetShapingConeSoftnessAttr().HasValue():
        attributes["cone_softness"] = shaping_api.GetShapingConeSoftnessAttr().Get()
    if shaping_api.GetShapingFocusAttr().HasValue():
        attributes["focus"] = shaping_api.GetShapingFocusAttr().Get()
    if shaping_api.GetShapingFocusTintAttr().HasValue():
        attributes["focus_tint"] = shaping_api.GetShapingFocusTintAttr().Get()
    if shaping_api.GetShapingIesAngleScaleAttr().HasValue():
        attributes["ies_angle_scale"] = shaping_api.GetShapingIesAngleScaleAttr().Get()
    if shaping_api.GetShapingIesFileAttr().HasValue():
        attributes["ies_file"] = shaping_api.GetShapingIesFileAttr().Get()
    if shaping_api.GetShapingIesNormalizeAttr().HasValue():
        attributes["ies_normalize"] = shaping_api.GetShapingIesNormalizeAttr().Get()
    return attributes


def copy_sphere_light(
    source_stage: Usd.Stage, source_path: str, dest_stage: Usd.Stage, dest_path: str
) -> UsdLux.SphereLight:
    """
    Copy a UsdLuxSphereLight from one stage to another.

    Args:
        source_stage (Usd.Stage): The stage containing the source sphere light.
        source_path (str): The path to the source sphere light prim.
        dest_stage (Usd.Stage): The stage to copy the sphere light to.
        dest_path (str): The path to the destination sphere light prim.

    Returns:
        UsdLux.SphereLight: The copied sphere light prim in the destination stage.
    """
    source_prim = source_stage.GetPrimAtPath(source_path)
    if not source_prim.IsValid():
        raise ValueError(f"Invalid source prim path: {source_path}")
    if not UsdLux.SphereLight(source_prim):
        raise TypeError(f"Source prim at {source_path} is not a UsdLuxSphereLight")
    dest_sphere_light = UsdLux.SphereLight.Define(dest_stage, dest_path)
    for attr in source_prim.GetAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        if attr_value is not None:
            dest_attr = dest_sphere_light.GetPrim().GetAttribute(attr_name)
            if dest_attr.IsValid():
                dest_attr.Set(attr_value)
    return dest_sphere_light


def set_sphere_light_radius(stage: Usd.Stage, light_path: str, radius: float) -> None:
    """Set the radius attribute of a UsdLuxSphereLight.

    Args:
        stage (Usd.Stage): The USD stage.
        light_path (str): The path to the sphere light prim.
        radius (float): The radius value to set.

    Raises:
        ValueError: If the prim at the given path is not a valid UsdLuxSphereLight.
    """
    light_prim = stage.GetPrimAtPath(light_path)
    if not light_prim.IsValid():
        raise ValueError(f"No prim found at path: {light_path}")
    sphere_light = UsdLux.SphereLight(light_prim)
    if not sphere_light:
        raise ValueError(f"Prim at path {light_path} is not a valid UsdLuxSphereLight")
    radius_attr = sphere_light.GetRadiusAttr()
    radius_attr.Set(radius)


def batch_update_sphere_lights_radius(stage: Usd.Stage, light_paths: List[str], radius: float):
    """
    Updates the radius attribute for multiple UsdLuxSphereLight prims in a single transaction.

    Args:
        stage (Usd.Stage): The USD stage.
        light_paths (List[str]): A list of paths to UsdLuxSphereLight prims.
        radius (float): The new radius value to set.

    Raises:
        ValueError: If any of the provided paths are invalid or not of type UsdLuxSphereLight.
    """
    if not stage:
        raise ValueError("Invalid USD stage.")
    if not light_paths:
        raise ValueError("No light paths provided.")
    if radius < 0:
        raise ValueError("Radius must be non-negative.")
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for light_path in light_paths:
            prim = stage.GetPrimAtPath(light_path)
            if not prim.IsValid():
                raise ValueError(f"Invalid prim path: {light_path}")
            if not prim.IsA(UsdLux.SphereLight):
                raise ValueError(f"Prim at path {light_path} is not a UsdLuxSphereLight")
            radius_attr = UsdLux.SphereLight.Get(stage, light_path).GetRadiusAttr()
            radius_attr.Set(radius)


def optimize_sphere_light_intensity(sphere_light: UsdLux.SphereLight, target_lumens: float) -> float:
    """
    Optimize the intensity of a sphere light to reach a target lumen value.

    Args:
        sphere_light (UsdLux.SphereLight): The sphere light to optimize.
        target_lumens (float): The desired lumen value to achieve.

    Returns:
        float: The optimized intensity value.

    Raises:
        ValueError: If the input sphere_light is not a valid UsdLux.SphereLight prim.
    """
    if not sphere_light or not isinstance(sphere_light, UsdLux.SphereLight):
        raise ValueError("Invalid input: sphere_light must be a valid UsdLux.SphereLight prim.")
    radius_attr = sphere_light.GetRadiusAttr()
    if not radius_attr or not radius_attr.HasValue():
        radius = 0.5
    else:
        radius = radius_attr.Get()
    surface_area = 4 * math.pi * radius**2
    intensity = target_lumens / surface_area
    sphere_light.CreateIntensityAttr(intensity)
    return intensity


def align_sphere_lights(stage: Usd.Stage, light_paths: List[str], target_path: str):
    """
    Align a list of UsdLuxSphereLights to point at a target prim.

    Args:
        stage (Usd.Stage): The USD stage containing the lights and target.
        light_paths (List[str]): A list of paths to the UsdLuxSphereLights.
        target_path (str): The path to the target prim.

    Raises:
        ValueError: If any of the light paths or the target path are invalid.
    """
    target_prim = stage.GetPrimAtPath(target_path)
    if not target_prim.IsValid():
        raise ValueError(f"Invalid target prim path: {target_path}")
    target_xform = UsdGeom.Xformable(target_prim)
    UsdShade.MaterialBindingAPI(target_prim).Bind(UsdShade.Material.Define(stage, f"{target_path}/Material"))
    target_pos = target_xform.ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default").ComputeCentroid()
    for light_path in light_paths:
        light_prim = stage.GetPrimAtPath(light_path)
        if not light_prim.IsValid():
            raise ValueError(f"Invalid light prim path: {light_path}")
        light = UsdLux.SphereLight(light_prim)
        if not light:
            raise ValueError(f"Prim at {light_path} is not a UsdLuxSphereLight")
        light_xform = UsdGeom.Xformable(light_prim)
        UsdShade.MaterialBindingAPI(light_prim).Bind(UsdShade.Material.Define(stage, f"{light_path}/Material"))
        light_pos = light_xform.ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default").ComputeCentroid()
        direction = target_pos - light_pos
        direction.Normalize()
        rotation = Gf.Rotation(Gf.Vec3d(0, 0, -1), direction)
        add_rotate_xyz_op(UsdGeom.Xformable(light_prim)).Set(
            rotation.Decompose(Gf.Vec3d.XAxis(), Gf.Vec3d.YAxis(), Gf.Vec3d.ZAxis())
        )


def list_lights_in_stage(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Returns a list of all light prims in the given stage.

    Args:
        stage (Usd.Stage): The USD stage to search for lights.

    Returns:
        List[Usd.Prim]: A list of all light prims in the stage.
    """
    lights = []
    for prim in stage.Traverse():
        if any(
            (
                prim.IsA(light_type)
                for light_type in [
                    UsdLux.DistantLight,
                    UsdLux.SphereLight,
                    UsdLux.DomeLight,
                    UsdLux.RectLight,
                    UsdLux.DiskLight,
                    UsdLux.CylinderLight,
                    UsdLux.GeometryLight,
                ]
            )
        ):
            lights.append(prim)
    return lights


def update_light_intensity(
    stage: Usd.Stage, light_path: str, intensity: float, time_code: Usd.TimeCode = Usd.TimeCode.Default()
):
    """Update the intensity of a light in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage.
        light_path (str): The path to the light prim.
        intensity (float): The new intensity value.
        time_code (Usd.TimeCode): The time code to set the value at (default is Default()).

    Raises:
        ValueError: If the prim at light_path is not a valid light.
    """
    prim = stage.GetPrimAtPath(light_path)
    if not prim.IsValid():
        raise ValueError(f"No prim found at path '{light_path}'")
    if not UsdLux.LightAPI(prim).GetIntensityAttr().IsDefined():
        raise ValueError(f"Prim at '{light_path}' is not a valid light")
    intensity_attr = UsdLux.LightAPI(prim).GetIntensityAttr()
    intensity_attr.Set(intensity, time_code)


def apply_volume_light_to_prim(prim: Usd.Prim) -> UsdLux.VolumeLightAPI:
    """Apply VolumeLightAPI to a Prim.

    Args:
        prim (Usd.Prim): The Prim to apply the VolumeLightAPI to.

    Returns:
        UsdLux.VolumeLightAPI: The applied VolumeLightAPI object.

    Raises:
        ValueError: If the input Prim is not valid or cannot have VolumeLightAPI applied.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid Prim: {prim}")
    if not UsdLux.VolumeLightAPI.CanApply(prim):
        raise ValueError(f"VolumeLightAPI cannot be applied to Prim: {prim}")
    return UsdLux.VolumeLightAPI.Apply(prim)


def get_volume_light_attributes(prim: Usd.Prim) -> Dict[str, Any]:
    """
    Get the attributes of a volume light prim.

    Args:
        prim (Usd.Prim): The volume light prim.

    Returns:
        Dict[str, Any]: A dictionary containing the volume light attributes.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    if not UsdLux.VolumeLightAPI.Get(prim.GetStage(), prim.GetPath()):
        raise ValueError(f"Prim {prim.GetPath()} does not have VolumeLightAPI applied")
    volume_light = UsdLux.VolumeLightAPI(prim)
    attributes = {}
    for attr_name in UsdLux.VolumeLightAPI.GetSchemaAttributeNames(includeInherited=True):
        attr = volume_light.GetPrim().GetAttribute(attr_name)
        if attr.HasValue():
            attributes[attr_name] = attr.Get()
    return attributes


def is_volume_light_applicable(prim: Usd.Prim) -> bool:
    """Check if the VolumeLightAPI can be applied to the given prim."""
    if not prim.IsValid():
        return False
    if not prim.IsA(UsdGeom.Gprim):
        return False
    if UsdLux.VolumeLightAPI.CanApply(prim):
        return True
    else:
        return False


def configure_volume_light(prim: Usd.Prim) -> UsdLux.VolumeLightAPI:
    """Configure a prim to be a volume light."""
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    if not prim.IsA(UsdVol.Volume):
        raise TypeError(f"Prim {prim.GetPath()} is not a UsdVol.Volume")
    volume_light = UsdLux.VolumeLightAPI.Apply(prim)
    if not volume_light:
        raise RuntimeError(f"Failed to apply VolumeLightAPI to {prim.GetPath()}")
    return volume_light


def transfer_volume_light_api(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> None:
    """Transfer the VolumeLightAPI from one prim to another."""
    if not source_prim.HasAPI(UsdLux.VolumeLightAPI):
        raise ValueError(f"Source prim {source_prim.GetPath()} does not have VolumeLightAPI applied.")
    if not dest_prim.IsValid():
        raise ValueError(f"Destination prim {dest_prim.GetPath()} is not valid.")
    source_api = UsdLux.VolumeLightAPI(source_prim)
    dest_api = UsdLux.VolumeLightAPI.Apply(dest_prim)
    for attr_name in source_api.GetSchemaAttributeNames(includeInherited=True):
        source_attr = source_api.GetPrim().GetAttribute(attr_name)
        if source_attr.IsAuthored():
            attr_value = source_attr.Get()
            dest_api.GetPrim().CreateAttribute(attr_name, source_attr.GetTypeName()).Set(attr_value)
