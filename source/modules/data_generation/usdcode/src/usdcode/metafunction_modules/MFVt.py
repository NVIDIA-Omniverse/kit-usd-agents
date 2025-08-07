## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

import os
import random
from typing import Callable, List, Optional, Sequence, Tuple, Union

from pxr import Gf, Sdf, Tf, Usd, UsdGeom, Vt

from .add_op import *


def convert_prim_to_bool_array(prim: Usd.Prim) -> Vt.BoolArray:
    """Convert a prim's bool array attribute to a Vt.BoolArray.

    Args:
        prim (Usd.Prim): The prim to get the bool array attribute from.

    Returns:
        Vt.BoolArray: The bool array value of the attribute.

    Raises:
        ValueError: If the prim is invalid or has no bool array attribute.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    bool_array_attr = None
    for attr in prim.GetAttributes():
        if attr.GetTypeName() == Sdf.ValueTypeNames.BoolArray:
            bool_array_attr = attr
            break
    if bool_array_attr is None:
        raise ValueError("Prim has no bool array attribute.")
    bool_array_value = bool_array_attr.Get()
    if bool_array_value is None:
        return Vt.BoolArray()
    return bool_array_value


def filter_prims_with_bool_attribute(stage: Usd.Stage, attribute_name: str) -> List[Usd.Prim]:
    """
    Filter prims on the stage that have a specific bool attribute.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        attribute_name (str): The name of the bool attribute to filter by.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified bool attribute.
    """
    prims_with_bool_attr: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            attr = prim.GetAttribute(attribute_name)
            if attr.GetTypeName() == Sdf.ValueTypeNames.Bool:
                prims_with_bool_attr.append(prim)
    return prims_with_bool_attr


def create_bool_attribute_with_default(prim: Usd.Prim, attribute_name: str, default_value: bool) -> Usd.Attribute:
    """Create a bool attribute on a prim with a default value.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute to create.
        default_value (bool): The default value for the attribute.

    Returns:
        Usd.Attribute: The created attribute.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Bool)
    attr.Set(default_value)
    return attr


def merge_bool_arrays(arrays: List[Vt.BoolArray]) -> Vt.BoolArray:
    """Merges a list of BoolArrays into a single BoolArray.

    Args:
        arrays (List[Vt.BoolArray]): A list of BoolArrays to merge.

    Returns:
        Vt.BoolArray: A new BoolArray containing the merged values.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    total_size = sum((len(arr) for arr in arrays))
    merged_array = Vt.BoolArray(total_size)
    index = 0
    for arr in arrays:
        size = len(arr)
        for i in range(size):
            merged_array[index] = arr[i]
            index += 1
    return merged_array


def toggle_bool_array_elements(bool_array: Vt.BoolArray) -> Vt.BoolArray:
    """Toggle the boolean values in a BoolArray.

    Args:
        bool_array (Vt.BoolArray): The input BoolArray to toggle.

    Returns:
        Vt.BoolArray: A new BoolArray with the toggled values.
    """
    toggled_array = Vt.BoolArray(len(bool_array))
    for i in range(len(bool_array)):
        toggled_value = not bool_array[i]
        toggled_array[i] = toggled_value
    return toggled_array


def concatenate_char_arrays(arrays: Sequence[Vt.CharArray]) -> Vt.CharArray:
    """Concatenate a sequence of CharArrays into a single CharArray."""
    if not arrays:
        return Vt.CharArray()
    total_length = sum((len(arr) for arr in arrays))
    result = Vt.CharArray(total_length)
    offset = 0
    for arr in arrays:
        length = len(arr)
        result[offset : offset + length] = arr
        offset += length
    return result


def find_char_in_array(char_array: Vt.CharArray, target_char: str) -> int:
    """
    Find the index of the first occurrence of a character in a CharArray.

    Args:
        char_array (Vt.CharArray): The CharArray to search.
        target_char (str): The character to find in the array.

    Returns:
        int: The index of the first occurrence of the target character in the array.
             Returns -1 if the character is not found.

    Raises:
        ValueError: If the target_char is not a single character string.
    """
    if len(target_char) != 1:
        raise ValueError("target_char must be a single character string")
    for i in range(len(char_array)):
        if char_array[i] == target_char:
            return i
    return -1


def get_char_array_slice(char_array: Vt.CharArray, start: int, end: int) -> Vt.CharArray:
    """Get a slice of a CharArray.

    Args:
        char_array (Vt.CharArray): The input CharArray.
        start (int): The start index of the slice (inclusive).
        end (int): The end index of the slice (exclusive).

    Returns:
        Vt.CharArray: A new CharArray containing the slice.

    Raises:
        ValueError: If start or end are out of bounds, or if start > end.
    """
    if start < 0 or start >= len(char_array):
        raise ValueError(f"Start index {start} is out of bounds for CharArray of length {len(char_array)}")
    if end < 0 or end > len(char_array):
        raise ValueError(f"End index {end} is out of bounds for CharArray of length {len(char_array)}")
    if start > end:
        raise ValueError(f"Start index {start} is greater than end index {end}")
    slice_array = Vt.CharArray(end - start)
    for i in range(start, end):
        slice_array[i - start] = char_array[i]
    return slice_array


def apply_function_to_double_array(double_array: Vt.DoubleArray, func: Callable[[float], float]) -> Vt.DoubleArray:
    """Apply a function to each element of a Vt.DoubleArray.

    Args:
        double_array (Vt.DoubleArray): The input array of doubles.
        func (Callable[[float], float]): The function to apply to each element.

    Returns:
        Vt.DoubleArray: A new array with the function applied to each element.
    """
    result = Vt.DoubleArray(len(double_array))
    for i in range(len(double_array)):
        value = func(double_array[i])
        result[i] = value
    return result


def square(x: float) -> float:
    return x**2


def find_min_max_in_double_array(double_array: Vt.DoubleArray) -> Tuple[float, float]:
    """Find the minimum and maximum values in a Vt.DoubleArray.

    Args:
        double_array (Vt.DoubleArray): The input array to search.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum values.

    Raises:
        ValueError: If the input array is empty.
    """
    if len(double_array) == 0:
        raise ValueError("Input array is empty.")
    min_val = double_array[0]
    max_val = double_array[0]
    for val in double_array:
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    return (min_val, max_val)


def normalize_double_array(double_array: Vt.DoubleArray) -> Vt.DoubleArray:
    """Normalize a DoubleArray by dividing each element by the sum of all elements.

    Args:
        double_array (Vt.DoubleArray): The input DoubleArray to normalize.

    Returns:
        Vt.DoubleArray: The normalized DoubleArray.

    Raises:
        ValueError: If the input DoubleArray is empty or contains only zeros.
    """
    if not double_array:
        raise ValueError("Input DoubleArray is empty.")
    array_sum = sum(double_array)
    if array_sum == 0:
        raise ValueError("Input DoubleArray contains only zeros.")
    normalized_array = Vt.DoubleArray(len(double_array))
    for i in range(len(double_array)):
        normalized_array[i] = double_array[i] / array_sum
    return normalized_array


def filter_double_array_by_range(double_array: Vt.DoubleArray, min_value: float, max_value: float) -> Vt.DoubleArray:
    """
    Filter a Vt.DoubleArray to contain only values between min_value and max_value (inclusive).

    Args:
        double_array (Vt.DoubleArray): The input array to filter.
        min_value (float): The minimum value (inclusive) for the filtered array.
        max_value (float): The maximum value (inclusive) for the filtered array.

    Returns:
        Vt.DoubleArray: A new array containing only the filtered values.
    """
    if min_value > max_value:
        raise ValueError("min_value must be less than or equal to max_value")
    filtered_list = []
    for value in double_array:
        if min_value <= value <= max_value:
            filtered_list.append(value)
    return Vt.DoubleArray(filtered_list)


def merge_double_arrays(arrays: List[Vt.DoubleArray]) -> Vt.DoubleArray:
    """Merges a list of DoubleArrays into a single DoubleArray.

    Args:
        arrays (List[Vt.DoubleArray]): The list of DoubleArrays to merge.

    Returns:
        Vt.DoubleArray: The merged DoubleArray.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays is empty.")
    merged_values = []
    for array in arrays:
        merged_values.extend(array)
    merged_array = Vt.DoubleArray(merged_values)
    return merged_array


def apply_dual_quat_transforms(transforms: Vt.DualQuatdArray, points: Vt.Vec3dArray) -> Vt.Vec3dArray:
    """Apply an array of dual quaternion transforms to an array of 3D points.

    Args:
        transforms (Vt.DualQuatdArray): An array of dual quaternion transforms.
        points (Vt.Vec3dArray): An array of 3D points to transform.

    Returns:
        Vt.Vec3dArray: The transformed points.
    """
    if not transforms or not points:
        raise ValueError("Input arrays cannot be empty.")
    if len(transforms) != len(points):
        raise ValueError("Number of transforms must match the number of points.")
    transformed_points = Vt.Vec3dArray(len(points))
    for i, (transform, point) in enumerate(zip(transforms, points)):
        gf_point = Gf.Vec3d(point)
        transformed_point = transform.Transform(gf_point)
        transformed_points[i] = transformed_point
    return transformed_points


def extract_rotations_from_dual_quats(dual_quats: Vt.DualQuatdArray) -> List[Gf.Quatd]:
    """Extract rotations from an array of dual quaternions.

    Args:
        dual_quats (Vt.DualQuatdArray): An array of dual quaternions.

    Returns:
        List[Gf.Quatd]: A list of quaternions representing the rotations.
    """
    if not isinstance(dual_quats, Vt.DualQuatdArray):
        raise TypeError("Input must be a Vt.DualQuatdArray.")
    rotations = []
    for dual_quat in dual_quats:
        real_quat = dual_quat.real
        real_quat = real_quat.GetNormalized()
        rotations.append(real_quat)
    return rotations


def blend_dual_quat_arrays(dqa1: Vt.DualQuatdArray, dqa2: Vt.DualQuatdArray, weight: float) -> Vt.DualQuatdArray:
    """Blend two dual quaternion arrays using linear interpolation.

    Args:
        dqa1 (Vt.DualQuatdArray): The first dual quaternion array.
        dqa2 (Vt.DualQuatdArray): The second dual quaternion array.
        weight (float): The blend weight in the range [0, 1].

    Returns:
        Vt.DualQuatdArray: The blended dual quaternion array.

    Raises:
        ValueError: If the input arrays have different sizes or weight is out of range.
    """
    if len(dqa1) != len(dqa2):
        raise ValueError("Input arrays must have the same size.")
    if weight < 0 or weight > 1:
        raise ValueError("Blend weight must be in the range [0, 1].")
    blended_dqa = Vt.DualQuatdArray(len(dqa1))
    for i in range(len(dqa1)):
        real_part = Gf.Slerp(weight, dqa1[i].real, dqa2[i].real)
        dual_part = Gf.Slerp(weight, dqa1[i].dual, dqa2[i].dual)
        blended_dq = Gf.DualQuatd(real_part, dual_part)
        blended_dqa[i] = blended_dq
    return blended_dqa


def retarget_dual_quatf_animation(
    source_anim: Vt.DualQuatfArray, target_rest_pose: Vt.DualQuatfArray
) -> Vt.DualQuatfArray:
    """Retargets a dual quaternion animation from a source skeleton to a target skeleton.

    Args:
        source_anim (Vt.DualQuatfArray): The source animation as an array of dual quaternions.
        target_rest_pose (Vt.DualQuatfArray): The rest pose of the target skeleton as an array of dual quaternions.

    Returns:
        Vt.DualQuatfArray: The retargeted animation for the target skeleton.
    """
    if len(source_anim) != len(target_rest_pose):
        raise ValueError("Source animation and target rest pose must have the same number of joints.")
    if len(source_anim) == 0:
        raise ValueError("Source animation must have at least one joint.")
    target_anim = Vt.DualQuatfArray(len(source_anim))
    for i in range(len(source_anim)):
        source_rest_inv = source_anim[i].GetInverse()
        relative_transform = source_rest_inv * target_rest_pose[i]
        target_anim[i] = relative_transform * source_anim[i]
    return target_anim


def blend_dual_quatf_arrays(a: Vt.DualQuatfArray, b: Vt.DualQuatfArray, t: float) -> Vt.DualQuatfArray:
    """Linearly interpolate two DualQuatfArrays.

    Args:
        a (Vt.DualQuatfArray): The first DualQuatfArray.
        b (Vt.DualQuatfArray): The second DualQuatfArray.
        t (float): The interpolation parameter in the range [0, 1].

    Returns:
        Vt.DualQuatfArray: The interpolated DualQuatfArray.

    Raises:
        ValueError: If the input arrays have different lengths or t is outside the range [0, 1].
    """
    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")
    if t < 0 or t > 1:
        raise ValueError("Interpolation parameter t must be in the range [0, 1].")
    blended = Vt.DualQuatfArray(len(a))
    for i in range(len(a)):
        (real_a, dual_a) = (a[i].GetReal(), a[i].GetDual())
        (real_b, dual_b) = (b[i].GetReal(), b[i].GetDual())
        real_blended = Gf.Slerp(t, real_a, real_b)
        dual_blended = Gf.Lerp(t, Gf.Vec3f(dual_a.GetImaginary()), Gf.Vec3f(dual_b.GetImaginary()))
        blended[i] = Gf.DualQuatf(real_blended, Gf.Quatf(dual_blended[0], dual_blended[1], dual_blended[2], 0))
    return blended


def apply_dual_quatf_transform(points: Vt.Vec3fArray, dual_quat: Gf.DualQuatf) -> Vt.Vec3fArray:
    """Apply a dual quaternion transformation to an array of points.

    Args:
        points (Vt.Vec3fArray): An array of 3D points to transform.
        dual_quat (Gf.DualQuatf): The dual quaternion representing the transformation.

    Returns:
        Vt.Vec3fArray: The transformed points.
    """
    if not points:
        return points
    transformed_points = Vt.Vec3fArray(len(points))
    for i, point in enumerate(points):
        gf_point = Gf.Vec3f(point)
        transformed_point = dual_quat.Transform(gf_point)
        transformed_points[i] = transformed_point
    return transformed_points


def combine_dual_quatf_arrays(array1: Vt.QuatfArray, array2: Vt.QuatfArray) -> Vt.QuatfArray:
    """Combine two QuatfArrays into a single array."""
    if not array1 or not array2:
        raise ValueError("Input arrays cannot be empty.")
    if len(array1) != len(array2):
        raise ValueError("Input arrays must have the same length.")
    combined_array = Vt.QuatfArray(len(array1))
    for i in range(len(array1)):
        combined_quat = array1[i] * array2[i]
        combined_array[i] = combined_quat
    return combined_array


def extract_referenced_dual_quath_arrays(prim: Usd.Prim) -> List[Vt.QuathArray]:
    """Extract all referenced QuathArray attributes from a prim.

    Args:
        prim (Usd.Prim): The prim to search for QuathArray attributes.

    Returns:
        List[Vt.QuathArray]: A list of referenced QuathArray attributes.
    """
    quath_arrays: List[Vt.QuathArray] = []
    for attr in prim.GetAttributes():
        if attr.GetTypeName() == Sdf.ValueTypeNames.QuathArray:
            value = attr.Get()
            if value is not None:
                quath_arrays.append(value)
    return quath_arrays


def interpolate_dual_quath_arrays(dqa1: Vt.DualQuathArray, dqa2: Vt.DualQuathArray, weight: float) -> Vt.DualQuathArray:
    """Linearly interpolate between two DualQuathArrays.

    Args:
        dqa1 (Vt.DualQuathArray): The first DualQuathArray.
        dqa2 (Vt.DualQuathArray): The second DualQuathArray.
        weight (float): The interpolation weight in the range [0, 1].

    Returns:
        Vt.DualQuathArray: The interpolated DualQuathArray.

    Raises:
        ValueError: If the input arrays have different sizes or the weight is out of range.
    """
    if len(dqa1) != len(dqa2):
        raise ValueError("Input DualQuathArrays must have the same size.")
    if weight < 0 or weight > 1:
        raise ValueError("Interpolation weight must be in the range [0, 1].")
    result = Vt.DualQuathArray(len(dqa1))
    for i in range(len(dqa1)):
        real = dqa1[i].real * (1 - weight) + dqa2[i].real * weight
        dual = dqa1[i].dual * (1 - weight) + dqa2[i].dual * weight
        result[i] = Gf.DualQuath(real, dual)
    return result


def create_float_attribute_with_default(prim: Usd.Prim, attribute_name: str, default_value: float) -> Usd.Attribute:
    """Create a float attribute on a prim with a default value.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute to create.
        default_value (float): The default value for the attribute.

    Returns:
        Usd.Attribute: The created attribute.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if prim.HasAttribute(attribute_name):
        attr = prim.GetAttribute(attribute_name)
        if attr.GetTypeName() != Sdf.ValueTypeNames.Float:
            raise ValueError(
                f"Attribute {attribute_name} already exists on prim {prim.GetPath()} but is not of type float."
            )
        return attr
    attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float)
    attr.Set(default_value)
    return attr


def blend_float_attributes(prim: Usd.Prim, attr_name_a: str, attr_name_b: str, blend_amount: float) -> Vt.FloatArray:
    """Blends two float array attributes on a prim using a blend amount.

    Args:
        prim (Usd.Prim): The prim containing the attributes to blend.
        attr_name_a (str): The name of the first attribute to blend.
        attr_name_b (str): The name of the second attribute to blend.
        blend_amount (float): The amount to blend between the two attributes (0.0 to 1.0).

    Returns:
        Vt.FloatArray: The blended float array.

    Raises:
        ValueError: If the attributes are not found or have different sizes.
    """
    attr_a = prim.GetAttribute(attr_name_a)
    attr_b = prim.GetAttribute(attr_name_b)
    if not attr_a.IsValid() or not attr_b.IsValid():
        raise ValueError(
            f"One or both attributes '{attr_name_a}' and '{attr_name_b}' not found on prim '{prim.GetPath()}'"
        )
    value_a = attr_a.Get()
    value_b = attr_b.Get()
    if len(value_a) != len(value_b):
        raise ValueError(f"Attributes '{attr_name_a}' and '{attr_name_b}' have different sizes")
    blended_value = Vt.FloatArray(len(value_a))
    for i in range(len(value_a)):
        blended_value[i] = value_a[i] * (1.0 - blend_amount) + value_b[i] * blend_amount
    return blended_value


def aggregate_float_attributes(prim: Usd.Prim, attribute_names: List[str]) -> Optional[Vt.FloatArray]:
    """Aggregate float attributes into a Vt.FloatArray.

    Args:
        prim (Usd.Prim): The prim to get the attributes from.
        attribute_names (List[str]): The names of the attributes to aggregate.

    Returns:
        Optional[Vt.FloatArray]: The aggregated float array, or None if no valid attributes found.
    """
    float_values = []
    for attr_name in attribute_names:
        if prim.HasAttribute(attr_name):
            attr = prim.GetAttribute(attr_name)
            if attr.GetTypeName() == Sdf.ValueTypeNames.Float:
                float_value = attr.Get()
                if float_value is not None:
                    float_values.append(float_value)
    if len(float_values) == 0:
        return None
    float_array = Vt.FloatArray(float_values)
    return float_array


def get_animated_float_values(float_array: Vt.FloatArray, time_code: Optional[int] = None) -> List[float]:
    """Get the float values from a FloatArray at a specific time code.

    Args:
        float_array (Vt.FloatArray): The FloatArray to get values from.
        time_code (Optional[int], optional): The time code to get values at. Defaults to None.

    Returns:
        List[float]: The float values from the FloatArray at the specified time code.
    """
    if time_code is None:
        time_code = 0
    size = len(float_array)
    float_values = []
    for i in range(size):
        value = float_array[i]
        float_values.append(value)
    return float_values


def compute_average_halfarray_values(halfarray: Vt.HalfArray) -> float:
    """Compute the average value of a Vt.HalfArray.

    Args:
        halfarray (Vt.HalfArray): The input HalfArray.

    Returns:
        float: The average value of the HalfArray.

    Raises:
        ValueError: If the input HalfArray is empty.
    """
    if not halfarray:
        raise ValueError("Input HalfArray is empty.")
    total_sum = 0.0
    count = 0
    for value in halfarray:
        total_sum += float(value)
        count += 1
    average = total_sum / count
    return average


def normalize_halfarray_values(halfarray: Vt.HalfArray) -> Vt.HalfArray:
    """Normalize the values in a Vt.HalfArray to be between 0 and 1."""
    if not halfarray:
        return halfarray
    min_val = min(halfarray)
    max_val = max(halfarray)
    value_range = max_val - min_val
    if value_range == 0:
        return Vt.HalfArray(halfarray)
    normalized_values = [(v - min_val) / value_range for v in halfarray]
    return Vt.HalfArray(normalized_values)


def find_halfarray_extremes(half_array: Vt.HalfArray) -> Tuple[float, float]:
    """
    Find the minimum and maximum values in a Vt.HalfArray.

    Args:
        half_array (Vt.HalfArray): The input array of half-precision floats.

    Returns:
        Tuple[float, float]: A tuple containing the minimum and maximum values as regular floats.

    Raises:
        ValueError: If the input array is empty.
    """
    if not half_array:
        raise ValueError("Input array cannot be empty.")
    min_val = max_val = float(half_array[0])
    for half_val in half_array:
        float_val = float(half_val)
        if float_val < min_val:
            min_val = float_val
        if float_val > max_val:
            max_val = float_val
    return (min_val, max_val)


def concatenate_halfarrays(arrays: Sequence[Vt.HalfArray]) -> Vt.HalfArray:
    """Concatenate a sequence of Vt.HalfArray into a single Vt.HalfArray.

    Args:
        arrays (Sequence[Vt.HalfArray]): The sequence of Vt.HalfArray to concatenate.

    Returns:
        Vt.HalfArray: The concatenated Vt.HalfArray.

    Raises:
        ValueError: If the input sequence is empty.
    """
    if not arrays:
        raise ValueError("Input sequence of Vt.HalfArray is empty.")
    concatenated = []
    for array in arrays:
        concatenated.extend(array)
    result = Vt.HalfArray(concatenated)
    return result


def apply_transformation_to_halfarray(halfarray: Vt.HalfArray, scale: float, offset: float) -> Vt.HalfArray:
    """Apply a linear transformation to a HalfArray.

    The transformation is defined as: y = scale * x + offset
    where x is an input element and y is the corresponding output element.

    Args:
        halfarray (Vt.HalfArray): The input HalfArray to transform.
        scale (float): The scale factor for the linear transformation.
        offset (float): The offset for the linear transformation.

    Returns:
        Vt.HalfArray: A new HalfArray with the transformation applied.
    """
    if not isinstance(halfarray, Vt.HalfArray):
        raise TypeError(f"Expected Vt.HalfArray, got {type(halfarray)}")
    transformed_array = Vt.HalfArray(len(halfarray))
    for i in range(len(halfarray)):
        x = halfarray[i]
        y = scale * x + offset
        transformed_array[i] = y
    return transformed_array


def set_int64_array_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.Int64Array) -> None:
    """Set an Int64Array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute to set.
        value (Vt.Int64Array): The Int64Array value to set on the attribute.

    Raises:
        ValueError: If the prim is not valid.
        TypeError: If the value is not an instance of Vt.Int64Array.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not isinstance(value, Vt.Int64Array):
        raise TypeError("Value must be an instance of Vt.Int64Array.")
    if not prim.HasAttribute(attribute_name):
        prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Int64Array)
    attribute = prim.GetAttribute(attribute_name)
    attribute.Set(value)


def get_int64_array_attribute(prim: Usd.Prim, attribute_name: str) -> Optional[List[int]]:
    """
    Get the value of an Int64Array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to retrieve the attribute from.
        attribute_name (str): The name of the Int64Array attribute.

    Returns:
        Optional[List[int]]: The value of the Int64Array attribute, or None if the attribute doesn't exist or has no value.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not prim.HasAttribute(attribute_name):
        return None
    attribute = prim.GetAttribute(attribute_name)
    if attribute.GetTypeName() != Sdf.ValueTypeNames.Int64Array:
        raise ValueError(f"Attribute {attribute_name} is not of type Int64Array")
    value = attribute.Get(Usd.TimeCode.Default())
    return value if value is not None else None


def sum_int64_array_attributes(prim: Usd.Prim, attribute_names: List[str]) -> int:
    """
    Sums the values of multiple Int64Array attributes on a prim.

    Args:
        prim (Usd.Prim): The prim to retrieve the attributes from.
        attribute_names (List[str]): The names of the Int64Array attributes to sum.

    Returns:
        int: The sum of all values in the specified Int64Array attributes.

    Raises:
        ValueError: If the prim is not valid or any of the specified attributes do not exist or are not of type Int64Array.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    total_sum = 0
    for attr_name in attribute_names:
        if not prim.HasAttribute(attr_name):
            raise ValueError(f"Attribute {attr_name} does not exist on prim {prim.GetPath()}.")
        attr = prim.GetAttribute(attr_name)
        if attr.GetTypeName() != Sdf.ValueTypeNames.Int64Array:
            raise ValueError(f"Attribute {attr_name} is not of type Int64Array.")
        int64_array = attr.Get()
        if int64_array is not None:
            total_sum += sum(int64_array)
    return total_sum


def filter_prims_by_int64_array_value(stage: Usd.Stage, attribute_name: str, attribute_value: int) -> List[Usd.Prim]:
    """
    Filter prims on the stage by the value of an Int64Array attribute.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        attribute_name (str): The name of the Int64Array attribute to filter by.
        attribute_value (int): The value to match in the Int64Array attribute.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified attribute with the given value.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            attr = prim.GetAttribute(attribute_name)
            if attr.GetTypeName() == Sdf.ValueTypeNames.Int64Array:
                attr_value: Vt.Int64Array = attr.Get()
                if attribute_value in attr_value:
                    matching_prims.append(prim)
    return matching_prims


def append_to_int64_array_attribute(prim: Usd.Prim, attribute_name: str, value: int) -> None:
    """Append an integer value to an Int64Array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to add the attribute to.
        attribute_name (str): The name of the Int64Array attribute.
        value (int): The integer value to append to the array.

    Raises:
        ValueError: If the attribute does not exist or is not an Int64Array.
    """
    if not prim.HasAttribute(attribute_name):
        attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Int64Array)
    else:
        attr = prim.GetAttribute(attribute_name)
    if attr.GetTypeName() != Sdf.ValueTypeNames.Int64Array:
        raise ValueError(f"Attribute {attribute_name} is not an Int64Array")
    int64_array = attr.Get()
    if int64_array is None:
        int64_array = Vt.Int64Array()
    int64_list = list(int64_array)
    int64_list.append(value)
    updated_int64_array = Vt.Int64Array(int64_list)
    attr.Set(updated_int64_array)


def find_prims_with_int_array_attribute(stage: Usd.Stage, attribute_name: str) -> List[Usd.Prim]:
    """Find all prims on the stage that have an attribute of type Vt.IntArray.

    Args:
        stage (Usd.Stage): The USD stage to search.
        attribute_name (str): The name of the attribute to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified attribute of type Vt.IntArray.
    """
    prims_with_int_array = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.HasAttribute(attribute_name):
            attr = prim.GetAttribute(attribute_name)
            if attr.GetTypeName() == Sdf.ValueTypeNames.IntArray:
                prims_with_int_array.append(prim)
    return prims_with_int_array


def set_int_array_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.IntArray) -> None:
    """Set an integer array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute to set.
        value (Vt.IntArray): The integer array value to set.

    Raises:
        ValueError: If the prim is not valid or the attribute name is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not attribute_name:
        raise ValueError("Attribute name cannot be empty.")
    if not prim.HasAttribute(attribute_name):
        prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.IntArray)
    attr = prim.GetAttribute(attribute_name)
    attr.Set(value)


def get_int_array_attribute(prim: Usd.Prim, attribute_name: str) -> List[int]:
    """Get the value of an int array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to get the attribute from.
        attribute_name (str): The name of the attribute to get.

    Returns:
        List[int]: The value of the attribute as a list of ints.

    Raises:
        ValueError: If the attribute does not exist or is not an int array.
    """
    if not prim.HasAttribute(attribute_name):
        raise ValueError(f"Attribute {attribute_name} does not exist on prim {prim.GetPath()}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        raise ValueError(f"Attribute {attribute_name} is not valid")
    if not attribute.GetTypeName() == Sdf.ValueTypeNames.IntArray:
        raise ValueError(f"Attribute {attribute_name} is not an int array")
    return attribute.Get(Usd.TimeCode.Default()) or []


def copy_int_array_attributes(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> None:
    """Copy all int array attributes from source_prim to dest_prim."""
    for attr in source_prim.GetAttributes():
        if attr.GetTypeName() == Sdf.ValueTypeNames.IntArray:
            attr_name = attr.GetName()
            attr_value = attr.Get()
            if attr_value is not None:
                dest_attr = dest_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.IntArray)
                dest_attr.Set(attr_value)


def analyze_int_array_attributes(int_array: Vt.IntArray) -> Tuple[int, int, float]:
    """Analyze a Vt.IntArray and return statistics.

    Args:
        int_array (Vt.IntArray): The input array to analyze.

    Returns:
        Tuple[int, int, float]: A tuple containing the minimum value, maximum value,
            and average value of the array.

    Raises:
        ValueError: If the input array is empty.
    """
    if not int_array:
        raise ValueError("Input array is empty.")
    min_value = int_array[0]
    max_value = int_array[0]
    sum_values = 0
    for value in int_array:
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value
        sum_values += value
    avg_value = sum_values / len(int_array)
    return (min_value, max_value, avg_value)


def intervals_to_time_samples(intervals: Vt.IntervalArray) -> List[Tuple[float, float]]:
    """Convert an array of GfInterval to a list of (start, end) time samples."""
    if not isinstance(intervals, Vt.IntervalArray):
        raise TypeError(f"Expected Vt.IntervalArray, got {type(intervals)}")
    time_samples: List[Tuple[float, float]] = []
    for i in range(len(intervals)):
        interval = intervals[i]
        if not isinstance(interval, Gf.Interval):
            raise TypeError(f"Expected Gf.Interval, got {type(interval)}")
        start_time = float(interval.GetMin())
        end_time = float(interval.GetMax())
        time_samples.append((start_time, end_time))
    return time_samples


def find_overlapping_intervals(intervals: Vt.IntervalArray) -> List[Tuple[int, int]]:
    """
    Find all pairs of overlapping intervals in the given array.

    Args:
        intervals (Vt.IntervalArray): An array of intervals.

    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple contains the indices
            of two overlapping intervals in the input array.
    """
    overlaps = []
    num_intervals = len(intervals)
    for i in range(num_intervals):
        for j in range(i + 1, num_intervals):
            if intervals[i].GetMin() <= intervals[j].GetMax() and intervals[j].GetMin() <= intervals[i].GetMax():
                overlaps.append((i, j))
    return overlaps


def filter_intervals_by_duration(intervals: Vt.IntervalArray, min_duration: float) -> Vt.IntervalArray:
    """Filter intervals by a minimum duration.

    Args:
        intervals (Vt.IntervalArray): The input array of intervals.
        min_duration (float): The minimum duration for an interval to pass the filter.

    Returns:
        Vt.IntervalArray: A new array containing only the intervals that are longer than min_duration.
    """
    filtered_intervals = []
    for interval in intervals:
        duration = interval.GetSize()
        if duration >= min_duration:
            filtered_intervals.append(interval)
    return Vt.IntervalArray(filtered_intervals)


def merge_intervals(self) -> Vt.IntervalArray:
    """Merge overlapping intervals in the array.

    Returns:
        Vt.IntervalArray: A new IntervalArray with merged intervals.
    """
    if len(self) == 0:
        return Vt.IntervalArray()
    sorted_intervals = sorted(self, key=lambda x: x.GetMin())
    merged = []
    current_interval = sorted_intervals[0]
    for interval in sorted_intervals[1:]:
        if current_interval.GetMax() >= interval.GetMin():
            current_interval = Gf.Interval(
                min(current_interval.GetMin(), interval.GetMin()), max(current_interval.GetMax(), interval.GetMax())
            )
        else:
            merged.append(current_interval)
            current_interval = interval
    merged.append(current_interval)
    return Vt.IntervalArray(merged)


def split_intervals_by_value(intervals: Vt.IntervalArray, value: float) -> Tuple[Vt.IntervalArray, Vt.IntervalArray]:
    """
    Split an array of intervals into two arrays based on a value.

    Args:
        intervals (Vt.IntervalArray): The input array of intervals.
        value (float): The value to split the intervals at.

    Returns:
        Tuple[Vt.IntervalArray, Vt.IntervalArray]: A tuple of two interval arrays,
        the first containing intervals with min <= value, the second containing intervals with max > value.
    """
    lower_intervals = []
    upper_intervals = []
    for interval in intervals:
        if interval.GetMin() <= value:
            lower_intervals.append(interval)
        if interval.GetMax() > value:
            upper_intervals.append(interval)
    return (Vt.IntervalArray(lower_intervals), Vt.IntervalArray(upper_intervals))


def transform_matrix2d_array(matrices: Vt.Matrix2dArray, transform: Gf.Matrix2d) -> Vt.Matrix2dArray:
    """
    Transform an array of GfMatrix2d by a given GfMatrix2d.

    Args:
        matrices (Vt.Matrix2dArray): The input array of matrices to transform.
        transform (Gf.Matrix2d): The transformation matrix to apply.

    Returns:
        Vt.Matrix2dArray: The transformed array of matrices.
    """
    if not matrices:
        return matrices
    transformed_matrices = Vt.Matrix2dArray(len(matrices))
    for i, matrix in enumerate(matrices):
        transformed_matrix = matrix * transform
        transformed_matrices[i] = transformed_matrix
    return transformed_matrices


def concatenate_matrix2d_arrays(matrix_arrays: List[Vt.Matrix2dArray]) -> Vt.Matrix2dArray:
    """Concatenate multiple Vt.Matrix2dArray into a single Vt.Matrix2dArray.

    Args:
        matrix_arrays (List[Vt.Matrix2dArray]): A list of Vt.Matrix2dArray to concatenate.

    Returns:
        Vt.Matrix2dArray: The concatenated Vt.Matrix2dArray.

    Raises:
        ValueError: If the input list is empty.
    """
    if not matrix_arrays:
        raise ValueError("Input list of matrix arrays cannot be empty.")
    total_matrices = sum((len(arr) for arr in matrix_arrays))
    concatenated_array = Vt.Matrix2dArray(total_matrices)
    current_index = 0
    for matrix_array in matrix_arrays:
        num_matrices = len(matrix_array)
        for i in range(num_matrices):
            concatenated_array[current_index] = matrix_array[i]
            current_index += 1
    return concatenated_array


def apply_matrix2d_array_to_points(matrix_array: Vt.Matrix2dArray, points: Vt.Vec2fArray) -> Vt.Vec2fArray:
    """
    Apply an array of GfMatrix2d to an array of GfVec2f points.

    Args:
        matrix_array (Vt.Matrix2dArray): An array of 2x2 matrices.
        points (Vt.Vec2fArray): An array of 2D points.

    Returns:
        Vt.Vec2fArray: The transformed points after applying the corresponding matrix to each point.

    Raises:
        ValueError: If the number of matrices and points do not match.
    """
    if len(matrix_array) != len(points):
        raise ValueError("The number of matrices and points must be the same.")
    transformed_points = Vt.Vec2fArray(len(points))
    for i in range(len(matrix_array)):
        matrix = matrix_array[i]
        point = points[i]
        transformed_point = matrix * point
        transformed_points[i] = transformed_point
    return transformed_points


def compute_inverse_matrix2d_array(matrix_array: Vt.Matrix2dArray) -> Vt.Matrix2dArray:
    """Compute the inverse of each matrix in a Matrix2dArray.

    Args:
        matrix_array (Vt.Matrix2dArray): The input array of matrices.

    Returns:
        Vt.Matrix2dArray: The array of inverted matrices.

    Raises:
        ValueError: If the input array is empty.
    """
    if not matrix_array:
        raise ValueError("Input matrix array is empty.")
    result = Vt.Matrix2dArray(len(matrix_array))
    for i, matrix in enumerate(matrix_array):
        try:
            result[i] = matrix.GetInverse()
        except Gf.Error:
            result[i] = Gf.Matrix2d(1)
    return result


def concatenate_matrices(matrices: Vt.Matrix2fArray) -> Gf.Matrix2f:
    """
    Concatenate a list of GfMatrix2f matrices.

    Args:
        matrices (Vt.Matrix2fArray): An array of GfMatrix2f matrices.

    Returns:
        Gf.Matrix2f: The result of concatenating the input matrices.

    Raises:
        ValueError: If the input array is empty.
    """
    if not matrices:
        raise ValueError("Input array cannot be empty.")
    result = Gf.Matrix2f(1)
    for matrix in matrices:
        result *= matrix
    return result


def transform_matrices(matrices: Vt.Matrix2fArray, scale: float = 1.0) -> Vt.Matrix2fArray:
    """
    Transform a list of matrices by a uniform scale.

    Args:
        matrices (Vt.Matrix2fArray): The input list of 2x2 matrices.
        scale (float): The uniform scale to apply to each matrix. Defaults to 1.0.

    Returns:
        Vt.Matrix2fArray: The transformed matrices.
    """
    if not isinstance(matrices, Vt.Matrix2fArray):
        raise TypeError("Input must be a Vt.Matrix2fArray")
    if not isinstance(scale, (int, float)):
        raise TypeError("Scale must be a number")
    scale_matrix = Gf.Matrix2f(scale, 0.0, 0.0, scale)
    transformed_matrices = Vt.Matrix2fArray(len(matrices))
    for i, matrix in enumerate(matrices):
        transformed_matrices[i] = matrix * scale_matrix
    return transformed_matrices


def animate_matrices_over_time(
    stage: Usd.Stage, prim_path: str, matrices: Vt.Matrix4dArray, time_samples: Vt.FloatArray
) -> None:
    """Animate a prim's transform matrices over time.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to animate.
        matrices (Vt.Matrix4dArray): The array of matrices to set as keyframes.
        time_samples (Vt.FloatArray): The array of time samples corresponding to each matrix.

    Raises:
        ValueError: If the prim path is invalid, the prim is not transformable,
            or the lengths of matrices and time_samples arrays do not match.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Invalid prim path: {prim_path}")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"Prim at path {prim_path} is not transformable")
    if len(matrices) != len(time_samples):
        raise ValueError("Length mismatch between matrices and time_samples arrays")
    transform_op = xformable.AddTransformOp()
    for matrix, time_sample in zip(matrices, time_samples):
        transform_op.Set(matrix, Usd.TimeCode(time_sample))


def extract_matrices_from_stage(stage: Usd.Stage, prim_path: str) -> Vt.Matrix2dArray:
    """Extract an array of GfMatrix2d from a USD stage at the specified prim path."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    if not prim.HasAttribute("matrix2d"):
        raise ValueError(f"Prim at path {prim_path} does not have a 'matrix2d' attribute.")
    attr = prim.GetAttribute("matrix2d")
    if attr.GetTypeName() != Sdf.ValueTypeNames.Matrix2dArray:
        raise ValueError(f"Attribute 'matrix2d' on prim {prim_path} is not of type Matrix2dArray.")
    matrix_array = attr.Get(Usd.TimeCode.Default())
    if matrix_array is None:
        return Vt.Matrix2dArray()
    return matrix_array


def transform_prims_with_matrix_array(
    stage: Usd.Stage, prim_paths: Vt.StringArray, matrix_array: Vt.Matrix3dArray
) -> None:
    """Transform a list of prims with a corresponding array of matrices.

    The number of prim paths and matrices must match. Each prim will be transformed by
    the corresponding matrix in the matrix_array.

    Args:
        stage (Usd.Stage): The stage containing the prims.
        prim_paths (Vt.StringArray): An array of prim paths to transform.
        matrix_array (Vt.Matrix3dArray): An array of matrices to apply to the prims.

    Raises:
        ValueError: If the number of prim paths and matrices do not match.
    """
    num_prims = len(prim_paths)
    num_matrices = len(matrix_array)
    if num_prims != num_matrices:
        raise ValueError(f"Number of prim paths ({num_prims}) does not match number of matrices ({num_matrices}).")
    for i in range(num_prims):
        prim_path = prim_paths[i]
        matrix = matrix_array[i]
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim at path {prim_path} is not transformable. Skipping.")
            continue
        transform_op = xformable.MakeMatrixXform()
        transform_op.Set(Gf.Matrix4d(matrix, Gf.Vec3d(0, 0, 0)))


def extract_subarray_from_matrix_array(
    matrix_array: Vt.Matrix3dArray, start_index: int, end_index: Optional[int] = None
) -> Vt.Matrix3dArray:
    """Extract a subarray from a Vt.Matrix3dArray.

    Args:
        matrix_array (Vt.Matrix3dArray): The input matrix array.
        start_index (int): The starting index of the subarray (inclusive).
        end_index (Optional[int]): The ending index of the subarray (exclusive). If None, extracts till the end.

    Returns:
        Vt.Matrix3dArray: The extracted subarray.

    Raises:
        IndexError: If start_index is out of bounds or greater than end_index.
        ValueError: If end_index is out of bounds.
    """
    array_length = len(matrix_array)
    if start_index < 0 or start_index >= array_length:
        raise IndexError("start_index is out of bounds.")
    if end_index is None:
        end_index = array_length
    if end_index < 0 or end_index > array_length:
        raise ValueError("end_index is out of bounds.")
    if start_index > end_index:
        raise IndexError("start_index cannot be greater than end_index.")
    subarray = matrix_array[start_index:end_index]
    return subarray


def apply_matrix_array_to_prims(
    stage: Usd.Stage,
    matrix_array: Vt.Matrix3dArray,
    prim_paths: List[str],
    time_code: Usd.TimeCode = Usd.TimeCode.Default(),
):
    """Apply an array of matrices to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        matrix_array (Vt.Matrix3dArray): The array of matrices to apply.
        prim_paths (List[str]): The list of prim paths to apply the matrices to.
        time_code (Usd.TimeCode, optional): The time code to set the transform at. Defaults to Default time code.

    Raises:
        ValueError: If the number of matrices does not match the number of prim paths.
    """
    if len(matrix_array) != len(prim_paths):
        raise ValueError("The number of matrices must match the number of prim paths.")
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        matrix = Gf.Matrix4d(
            matrix_array[i][0][0],
            matrix_array[i][0][1],
            matrix_array[i][0][2],
            0.0,
            matrix_array[i][1][0],
            matrix_array[i][1][1],
            matrix_array[i][1][2],
            0.0,
            matrix_array[i][2][0],
            matrix_array[i][2][1],
            matrix_array[i][2][2],
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        op = xformable.AddTransformOp()
        op.Set(matrix, time_code)


def interpolate_matrix_arrays(a: Vt.Matrix3dArray, b: Vt.Matrix3dArray, t: float) -> Vt.Matrix3dArray:
    """Interpolate between two arrays of GfMatrix3d.

    Args:
        a (Vt.Matrix3dArray): The first array of matrices.
        b (Vt.Matrix3dArray): The second array of matrices.
        t (float): The interpolation parameter in the range [0, 1].

    Returns:
        Vt.Matrix3dArray: The interpolated array of matrices.

    Raises:
        ValueError: If the input arrays have different lengths or if t is outside the range [0, 1].
    """
    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")
    if t < 0 or t > 1:
        raise ValueError("Interpolation parameter t must be in the range [0, 1].")
    result = Vt.Matrix3dArray(len(a))
    for i in range(len(a)):
        result[i] = Gf.Matrix3d(
            Gf.Lerp(a[i][0, 0], b[i][0, 0], t),
            Gf.Lerp(a[i][0, 1], b[i][0, 1], t),
            Gf.Lerp(a[i][0, 2], b[i][0, 2], t),
            Gf.Lerp(a[i][1, 0], b[i][1, 0], t),
            Gf.Lerp(a[i][1, 1], b[i][1, 1], t),
            Gf.Lerp(a[i][1, 2], b[i][1, 2], t),
            Gf.Lerp(a[i][2, 0], b[i][2, 0], t),
            Gf.Lerp(a[i][2, 1], b[i][2, 1], t),
            Gf.Lerp(a[i][2, 2], b[i][2, 2], t),
        )
    return result


def apply_transformation_matrix(points: Vt.Vec3fArray, matrix: Gf.Matrix3f) -> Vt.Vec3fArray:
    """Apply a transformation matrix to an array of points.

    Args:
        points (Vt.Vec3fArray): Array of points to transform.
        matrix (Gf.Matrix3f): Transformation matrix to apply.

    Returns:
        Vt.Vec3fArray: Array of transformed points.
    """
    if not points:
        raise ValueError("Input points array is empty.")
    transformed_points = Vt.Vec3fArray(len(points))
    for i, point in enumerate(points):
        vec3f = Gf.Vec3f(point)
        transformed_vec3f = matrix * vec3f
        transformed_points[i] = transformed_vec3f
    return transformed_points


def interpolate_transformation_matrices(matrices: Vt.Matrix3fArray, t: float) -> Gf.Matrix3f:
    """Interpolate between transformation matrices.

    Args:
        matrices (Vt.Matrix3fArray): An array of transformation matrices.
        t (float): Interpolation parameter in the range [0, 1].

    Returns:
        Gf.Matrix3f: The interpolated transformation matrix.
    """
    if not isinstance(matrices, Vt.Matrix3fArray):
        raise TypeError("matrices must be of type Vt.Matrix3fArray")
    num_matrices = len(matrices)
    if num_matrices == 0:
        raise ValueError("matrices array is empty")
    elif num_matrices == 1:
        return Gf.Matrix3f(matrices[0])
    if not isinstance(t, float) or t < 0.0 or t > 1.0:
        raise ValueError("t must be a float in the range [0, 1]")
    index = int((num_matrices - 1) * t)
    weight = (num_matrices - 1) * t - index
    index = min(index, num_matrices - 2)
    matrix0 = Gf.Matrix3f(matrices[index])
    matrix1 = Gf.Matrix3f(matrices[index + 1])
    interpolated_matrix = matrix0 * (1.0 - weight) + matrix1 * weight
    return interpolated_matrix


def apply_inverse_transformation_matrix(matrix_array: Vt.Matrix3fArray, vector: Gf.Vec3f) -> Gf.Vec3f:
    """
    Apply the inverse of the first matrix in the array to the given vector.

    Args:
        matrix_array (Vt.Matrix3fArray): An array of matrices.
        vector (Gf.Vec3f): The vector to transform.

    Returns:
        Gf.Vec3f: The transformed vector.

    Raises:
        ValueError: If the matrix array is empty.
    """
    if not matrix_array:
        raise ValueError("Matrix array cannot be empty.")
    matrix = matrix_array[0]
    inverse_matrix = matrix.GetInverse()
    transformed_vector = inverse_matrix * vector
    return transformed_vector


def compute_average_transformation_matrix(matrix_array: Vt.Matrix3fArray) -> Gf.Matrix3f:
    """Compute the average transformation matrix from an array of matrices.

    Args:
        matrix_array (Vt.Matrix3fArray): An array of transformation matrices.

    Returns:
        Gf.Matrix3f: The average transformation matrix.

    Raises:
        ValueError: If the input matrix array is empty.
    """
    if not matrix_array:
        raise ValueError("Input matrix array is empty.")
    sum_matrix = Gf.Matrix3f(0)
    for matrix in matrix_array:
        sum_matrix += matrix
    num_matrices = len(matrix_array)
    avg_matrix = Gf.Matrix3f(
        sum_matrix[0][0] / num_matrices,
        sum_matrix[0][1] / num_matrices,
        sum_matrix[0][2] / num_matrices,
        sum_matrix[1][0] / num_matrices,
        sum_matrix[1][1] / num_matrices,
        sum_matrix[1][2] / num_matrices,
        sum_matrix[2][0] / num_matrices,
        sum_matrix[2][1] / num_matrices,
        sum_matrix[2][2] / num_matrices,
    )
    return avg_matrix


def apply_transform_to_prims(prims: Vt.Vec3dArray, transform_matrix: Gf.Matrix4d) -> Vt.Vec3dArray:
    """Apply a transformation matrix to an array of prim positions.

    Args:
        prims (Vt.Vec3dArray): An array of prim positions.
        transform_matrix (Gf.Matrix4d): The transformation matrix to apply.

    Returns:
        Vt.Vec3dArray: The transformed prim positions.
    """
    if not prims:
        return prims
    transformed_prims = Vt.Vec3dArray(len(prims))
    for i, prim_pos in enumerate(prims):
        vec3d = Gf.Vec3d(prim_pos)
        transformed_vec3d = transform_matrix.Transform(vec3d)
        transformed_prims[i] = transformed_vec3d
    return transformed_prims


def compute_combined_transform(transform_matrices: Vt.Matrix4dArray) -> Gf.Matrix4d:
    """Computes the combined transform matrix from an array of matrices.

    Args:
        transform_matrices (Vt.Matrix4dArray): An array of transform matrices.

    Returns:
        Gf.Matrix4d: The combined transform matrix.

    Raises:
        ValueError: If the input array is empty.
    """
    if not transform_matrices:
        raise ValueError("Input array of transform matrices is empty.")
    combined_transform = Gf.Matrix4d(1.0)
    for matrix in transform_matrices:
        combined_transform *= matrix
    return combined_transform


def combine_animations(animations: List[Vt.Matrix4dArray]) -> Vt.Matrix4dArray:
    """Combine multiple animations into a single animation.

    Args:
        animations (List[Vt.Matrix4dArray]): A list of animations to combine.

    Returns:
        Vt.Matrix4dArray: The combined animation.

    Raises:
        ValueError: If the input list is empty or the animations have different lengths.
    """
    if not animations:
        raise ValueError("The input list of animations is empty.")
    num_frames = len(animations[0])
    for anim in animations[1:]:
        if len(anim) != num_frames:
            raise ValueError("All animations must have the same number of frames.")
    combined_anim = Vt.Matrix4dArray(num_frames)
    for i in range(num_frames):
        combined_matrix = Gf.Matrix4d(1.0)
        for anim in animations:
            combined_matrix *= anim[i]
        combined_anim[i] = combined_matrix
    return combined_anim


def animate_prims_with_matrices(
    stage: Usd.Stage, prim_paths: List[str], matrix_array: Vt.Matrix4dArray, time_code: Usd.TimeCode
):
    """Animate multiple prims with a matrix array at a specific time code.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to animate.
        matrix_array (Vt.Matrix4dArray): Array of matrices to apply to the prims.
        time_code (Usd.TimeCode): The time code to set the animation.

    Raises:
        ValueError: If the number of prim paths and matrices do not match.
    """
    if len(prim_paths) != len(matrix_array):
        raise ValueError(
            f"Number of prim paths ({len(prim_paths)}) does not match number of matrices ({len(matrix_array)})."
        )
    for prim_path, matrix in zip(prim_paths, matrix_array):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        transform_op = xformable.AddTransformOp()
        transform_op.Set(Gf.Matrix4d(matrix), time_code)


def extract_world_transforms(stage: Usd.Stage) -> List[Gf.Matrix4d]:
    """Extract world transform matrices for all prims in the stage.

    Args:
        stage (Usd.Stage): The USD stage to extract transforms from.

    Returns:
        List[Gf.Matrix4d]: A list of world transform matrices, one for each prim.
    """
    world_transforms = []
    for prim in stage.Traverse():
        if prim.IsPseudoRoot() or not prim.IsA(UsdGeom.Xformable):
            continue
        xformable = UsdGeom.Xformable(prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        world_transforms.append(world_transform)
    return world_transforms


def interpolate_transforms(transforms: Vt.Matrix4fArray, t: float) -> Gf.Matrix4f:
    """
    Linearly interpolate between two transforms at a given parameter t.

    Args:
        transforms (Vt.Matrix4fArray): An array of two transforms to interpolate between.
        t (float): The interpolation parameter in the range [0, 1].

    Returns:
        Gf.Matrix4f: The interpolated transform at parameter t.

    Raises:
        ValueError: If the input array does not contain exactly two transforms or if t is outside the range [0, 1].
    """
    if len(transforms) != 2:
        raise ValueError("Expected an array of exactly two transforms.")
    if t < 0 or t > 1:
        raise ValueError("Interpolation parameter t must be in the range [0, 1].")
    transform1 = transforms[0]
    transform2 = transforms[1]
    interpolated_matrix = Gf.Matrix4f(transform1 * (1 - t) + transform2 * t)
    return interpolated_matrix


def batch_update_transforms(stage: Usd.Stage, prim_paths: List[str], transforms: Vt.Matrix4fArray) -> None:
    """Update the local transforms for a batch of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths of the prims to update.
        transforms (Vt.Matrix4fArray): The new local transform matrices.

    Raises:
        ValueError: If the number of prim paths and transforms do not match.
    """
    if len(prim_paths) != len(transforms):
        raise ValueError("Number of prim paths and transforms must match.")
    for prim_path, transform in zip(prim_paths, transforms):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        xformable.GetXformOpOrderAttr().Clear()
        matrix_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTransform, UsdGeom.XformOp.PrecisionFloat, "")
        matrix_op.Set(Gf.Matrix4d(transform))


def get_prim_orientations(stage: Usd.Stage, prim_paths: List[str]) -> Vt.QuatdArray:
    """
    Get the local space orientations for a list of prims.

    Args:
        stage (Usd.Stage): The USD stage to query.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Vt.QuatdArray: An array of orientations as quaternions, one per prim.

    Raises:
        ValueError: If any prim is invalid or not transformable.
    """
    orientations = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        orientation_op = xformable.GetOrderedXformOps()[-1]
        if orientation_op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            rotation_attr = orientation_op.GetAttr()
            if rotation_attr.HasValue():
                rotation = rotation_attr.Get()
                orientations.append(Gf.Quatd(rotation))
            else:
                orientations.append(Gf.Quatd(1.0))
        else:
            orientations.append(Gf.Quatd(1.0))
    return Vt.QuatdArray(orientations)


def set_prim_orientations(stage: Usd.Stage, prim_paths: List[str], orientations: Vt.QuatfArray) -> None:
    """Set the orientations for a list of prims using a QuatfArray.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.
        orientations (Vt.QuatfArray): An array of quaternions representing the orientations.

    Raises:
        ValueError: If the number of prim paths and orientations don't match.
    """
    if len(prim_paths) != len(orientations):
        raise ValueError("Number of prim paths and orientations must match.")
    for prim_path, orientation in zip(prim_paths, orientations):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim at path {prim_path} is not transformable. Skipping.")
            continue
        add_orient_op(xformable).Set(orientation)


def extract_subarray_from_quatdarray(quat_array: Vt.QuatdArray, start_index: int, end_index: int) -> Vt.QuatdArray:
    """Extract a subarray from a QuatdArray.

    Args:
        quat_array (Vt.QuatdArray): The input QuatdArray.
        start_index (int): The starting index of the subarray (inclusive).
        end_index (int): The ending index of the subarray (exclusive).

    Returns:
        Vt.QuatdArray: The extracted subarray.

    Raises:
        ValueError: If the start or end index is out of bounds or if the end index is less than the start index.
    """
    if start_index < 0 or start_index >= len(quat_array):
        raise ValueError("Start index is out of bounds.")
    if end_index <= start_index or end_index > len(quat_array):
        raise ValueError("End index is out of bounds or less than the start index.")
    subarray = Vt.QuatdArray(end_index - start_index)
    for i in range(start_index, end_index):
        subarray[i - start_index] = quat_array[i]
    return subarray


def interpolate_orientations(orientations: Vt.QuatdArray, t: float) -> Gf.Quatd:
    """Interpolate between orientations at a specific time.

    Args:
        orientations (Vt.QuatdArray): Array of orientations as quaternions.
        t (float): Interpolation time in the range [0, 1].

    Returns:
        Gf.Quatd: Interpolated orientation at time t.

    Raises:
        ValueError: If orientations array is empty or t is outside [0, 1] range.
    """
    if not orientations:
        raise ValueError("Orientations array is empty.")
    if t < 0 or t > 1:
        raise ValueError("Interpolation time must be in the range [0, 1].")
    if t == 0:
        return orientations[0]
    if t == 1:
        return orientations[-1]
    index = int(t * (len(orientations) - 1))
    quat1 = orientations[index]
    quat2 = orientations[index + 1]
    return Gf.Slerp(t - index / (len(orientations) - 1), quat1, quat2)


def blend_orientations(quats: Vt.QuatdArray, weights: Optional[List[float]] = None) -> Gf.Quatd:
    """Blend a list of orientations represented as quaternions.

    Args:
        quats (Vt.QuatdArray): An array of quaternions representing orientations.
        weights (Optional[List[float]], optional): Optional weights for each quaternion. Defaults to None.

    Returns:
        Gf.Quatd: The blended quaternion representing the combined orientation.
    """
    if not quats:
        raise ValueError("Quaternion array is empty.")
    num_quats = len(quats)
    if weights is None:
        weights = [1.0 / num_quats] * num_quats
    else:
        if len(weights) != num_quats:
            raise ValueError("Number of weights does not match the number of quaternions.")
        weight_sum = sum(weights)
        if weight_sum == 0.0:
            raise ValueError("Sum of weights is zero.")
        weights = [weight / weight_sum for weight in weights]
    result = Gf.Quatd(quats[0])
    for i in range(1, num_quats):
        result = result * (1.0 - weights[i]) + quats[i] * weights[i]
        result.Normalize()
    return result


def blend_quaternions(quaternions: Vt.QuaternionArray, weights: Vt.FloatArray) -> Gf.Quaternion:
    """Blend quaternions using weights.

    Args:
        quaternions (Vt.QuaternionArray): Array of quaternions to blend.
        weights (Vt.FloatArray): Array of weights for blending. Must be the same length as quaternions.

    Returns:
        Gf.Quaternion: The blended quaternion.

    Raises:
        ValueError: If the lengths of quaternions and weights arrays don't match.
    """
    if len(quaternions) != len(weights):
        raise ValueError("The lengths of quaternions and weights arrays must match.")
    total_weight = sum(weights)
    if total_weight == 0:
        return Gf.Quaternion(1, Gf.Vec3d(0, 0, 0))
    normalized_weights = [w / total_weight for w in weights]
    result = Gf.Quaternion(0, Gf.Vec3d(0, 0, 0))
    for q, w in zip(quaternions, normalized_weights):
        result += q * w
    result.Normalize()
    return result


def get_average_quaternion(quat_array: Vt.QuaternionArray) -> Gf.Quaternion:
    """Compute the average quaternion from the array.

    Args:
        quat_array (Vt.QuaternionArray): The input quaternion array.

    Returns:
        Gf.Quaternion: The average quaternion.

    Raises:
        ValueError: If the quaternion array is empty.
    """
    if not quat_array:
        raise ValueError("Cannot compute average of an empty QuaternionArray.")
    avg_quat = quat_array[0]
    for i in range(1, len(quat_array)):
        quat = quat_array[i]
        if Gf.Dot(avg_quat.GetImaginary(), quat.GetImaginary()) < 0:
            quat = -quat
        avg_quat += quat
    avg_quat.Normalize()
    return avg_quat


def extract_rotations_from_quatf_array(quatf_array: Vt.QuatfArray) -> List[Gf.Quatf]:
    """Extract the rotations from a QuatfArray.

    Args:
        quatf_array (Vt.QuatfArray): The input array of quaternions.

    Returns:
        List[Gf.Quatf]: The list of extracted rotations as Gf.Quatf objects.
    """
    if not isinstance(quatf_array, Vt.QuatfArray):
        raise TypeError(f"Expected Vt.QuatfArray, got {type(quatf_array)}")
    rotations: List[Gf.Quatf] = []
    for i in range(len(quatf_array)):
        quatf = quatf_array[i]
        rotation = Gf.Quatf(quatf.GetReal(), quatf.GetImaginary())
        rotations.append(rotation)
    return rotations


def create_quatf_array_from_rotations(rotations: List[Gf.Rotation]) -> Vt.QuatfArray:
    """Create a QuatfArray from a list of GfRotation objects.

    Args:
        rotations (List[Gf.Rotation]): A list of rotation objects.

    Returns:
        Vt.QuatfArray: An array of quaternions representing the rotations.

    Raises:
        ValueError: If the input list is empty.
    """
    if not rotations:
        raise ValueError("Input list of rotations cannot be empty.")
    quat_array = Vt.QuatfArray(len(rotations))
    for i, rotation in enumerate(rotations):
        quat = Gf.Quatf(rotation.GetQuat())
        quat_array[i] = quat
    return quat_array


def interpolate_quatf_arrays(a: Vt.QuatfArray, b: Vt.QuatfArray, t: float) -> Vt.QuatfArray:
    """Linearly interpolates between two arrays of quaternions.

    Args:
        a (Vt.QuatfArray): The first array of quaternions.
        b (Vt.QuatfArray): The second array of quaternions.
        t (float): The interpolation parameter in the range [0, 1].

    Returns:
        Vt.QuatfArray: The interpolated array of quaternions.

    Raises:
        ValueError: If the input arrays have different lengths or if t is outside the range [0, 1].
    """
    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")
    if t < 0 or t > 1:
        raise ValueError("Interpolation parameter t must be in the range [0, 1].")
    result = Vt.QuatfArray(len(a))
    for i in range(len(a)):
        result[i] = Gf.Slerp(t, a[i], b[i])
    return result


def set_quath_array_attribute(prim: Usd.Prim, attribute_name: str, quath_array: Vt.QuathArray) -> None:
    """Sets a QuathArray attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute to set.
        quath_array (Vt.QuathArray): The QuathArray value to set on the attribute.

    Raises:
        ValueError: If the prim is not valid or the attribute name is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not attribute_name:
        raise ValueError("Attribute name cannot be empty.")
    if not prim.HasAttribute(attribute_name):
        prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.QuathArray)
    attr = prim.GetAttribute(attribute_name)
    attr.Set(quath_array)


def bake_quath_array_animation(
    stage: Usd.Stage, prim_path: str, attribute_name: str, start_time: float, end_time: float, time_samples: int
) -> None:
    """Bake the animation of a Vt.QuathArray attribute on a prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        attribute_name (str): The name of the Vt.QuathArray attribute.
        start_time (float): The start time of the animation.
        end_time (float): The end time of the animation.
        time_samples (int): The number of time samples to bake.

    Raises:
        ValueError: If the prim or attribute does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        raise ValueError(f"Attribute {attribute_name} does not exist on prim at path {prim_path}.")
    if attribute.GetTypeName() != Sdf.ValueTypeNames.QuathArray:
        raise ValueError(f"Attribute {attribute_name} is not of type Vt.QuathArray.")
    time_step = (end_time - start_time) / (time_samples - 1)
    for i in range(time_samples):
        time = start_time + i * time_step
        value = attribute.Get(time)
        if value is not None:
            attribute.Set(value, Usd.TimeCode(time))


def get_combined_quath_array(quath_arrays: List[Vt.QuathArray]) -> Vt.QuathArray:
    """
    Combines multiple QuathArrays into a single QuathArray.

    Args:
        quath_arrays (List[Vt.QuathArray]): A list of QuathArrays to combine.

    Returns:
        Vt.QuathArray: The combined QuathArray.

    Raises:
        ValueError: If the input list is empty.
    """
    if not quath_arrays:
        raise ValueError("The input list of QuathArrays is empty.")
    combined_quaths = []
    for quath_array in quath_arrays:
        for quath in quath_array:
            combined_quaths.append(quath)
    combined_quath_array = Vt.QuathArray(combined_quaths)
    return combined_quath_array


def get_range1d_array_bounds(range1d_array: Vt.Range1dArray) -> Gf.Range1d:
    """Get the overall bounds of a Vt.Range1dArray.

    Args:
        range1d_array (Vt.Range1dArray): The input array of Gf.Range1d.

    Returns:
        Gf.Range1d: The overall bounds of the input array.

    Raises:
        ValueError: If the input array is empty.
    """
    if not range1d_array:
        raise ValueError("Input Range1dArray is empty")
    bounds = Gf.Range1d(range1d_array[0])
    for range1d in range1d_array[1:]:
        bounds.UnionWith(range1d)
    return bounds


def sample_range1d_array(ranges: Vt.Range1dArray, num_samples: int) -> List[float]:
    """Sample a list of ranges at evenly spaced intervals.

    Args:
        ranges (Vt.Range1dArray): An array of Gf.Range1d objects.
        num_samples (int): The number of samples to take from each range.

    Returns:
        List[float]: A list of sampled values from the ranges.
    """
    if not isinstance(ranges, Vt.Range1dArray):
        raise TypeError("ranges must be a Vt.Range1dArray")
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")
    samples = []
    for range1d in ranges:
        if range1d.IsEmpty():
            continue
        step_size = (range1d.max - range1d.min) / (num_samples - 1)
        for i in range(num_samples):
            sample = range1d.min + i * step_size
            samples.append(sample)
    return samples


def filter_range1d_array_by_value(range1d_array: Vt.Range1dArray, value: float) -> Vt.Range1dArray:
    """Filter a Range1dArray to only include ranges that contain a specified value.

    Args:
        range1d_array (Vt.Range1dArray): The input array of GfRange1d objects.
        value (float): The value to check for inclusion in each range.

    Returns:
        Vt.Range1dArray: A new Range1dArray containing only the ranges that include the specified value.
    """
    filtered_list = []
    for range1d in range1d_array:
        if range1d.Contains(value):
            filtered_list.append(range1d)
    filtered_array = Vt.Range1dArray(filtered_list)
    return filtered_array


def transform_range1d_array(range1d_array: Vt.Range1dArray, translation: float) -> Vt.Range1dArray:
    """
    Translate each range in the given Range1dArray by the specified translation amount.

    Args:
        range1d_array (Vt.Range1dArray): The input array of GfRange1d objects.
        translation (float): The amount to translate each range by.

    Returns:
        Vt.Range1dArray: A new Range1dArray with each range translated by the specified amount.
    """
    transformed_ranges = []
    for range1d in range1d_array:
        new_min = range1d.min + translation
        new_max = range1d.max + translation
        new_range = Gf.Range1d(new_min, new_max)
        transformed_ranges.append(new_range)
    transformed_array = Vt.Range1dArray(transformed_ranges)
    return transformed_array


def aggregate_bbox_ranges(ranges: Vt.Range1fArray) -> Tuple[Gf.Range1f, List[int]]:
    """Aggregate a list of Range1f objects into a single Range1f.

    Args:
        ranges (Vt.Range1fArray): An array of Range1f objects.

    Returns:
        A tuple containing:
        - The aggregated Range1f
        - A list of indices of the ranges that were successfully aggregated
    """
    if not ranges:
        return (Gf.Range1f(), [])
    aggregated_range = Gf.Range1f(ranges[0])
    valid_indices = [0]
    for i in range(1, len(ranges)):
        range_i = ranges[i]
        if range_i.IsEmpty():
            continue
        aggregated_range.UnionWith(range_i)
        valid_indices.append(i)
    return (aggregated_range, valid_indices)


def compute_combined_bbox(ranges: Vt.Range1fArray) -> Gf.Range1f:
    """Compute the combined bounding box from an array of ranges.

    Args:
        ranges (Vt.Range1fArray): Array of ranges to combine.

    Returns:
        Gf.Range1f: The combined bounding box range.

    Raises:
        ValueError: If the input array is empty.
    """
    if not ranges:
        raise ValueError("Input range array is empty")
    combined_bbox = Gf.Range1f(ranges[0])
    for range1f in ranges[1:]:
        combined_bbox.UnionWith(range1f)
    return combined_bbox


def get_animated_bbox_ranges(prim: Usd.Prim, time_samples: Optional[List[Usd.TimeCode]] = None) -> Vt.Vec2fArray:
    """
    Get the animated bounding box ranges for a prim over time.

    Args:
        prim (Usd.Prim): The prim to compute the animated bounding box ranges for.
        time_samples (Optional[List[Usd.TimeCode]], optional): The time samples to compute the bounding box at.
            If None, uses the time samples from the stage. Defaults to None.

    Returns:
        Vt.Vec2fArray: An array of GfVec2f representing the bounding box ranges at each time sample.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim provided.")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    if time_samples is None:
        time_samples = []
        stage = prim.GetStage()
        time_codes = stage.GetTimeCodesInInterval(Gf.Interval.GetFullInterval())
        for time_code in time_codes:
            time_samples.append(time_code)
    bbox_ranges = Vt.Vec2fArray()
    for time_code in time_samples:
        bbox = bbox_cache.ComputeWorldBound(prim, time_code)
        if bbox.GetRange().IsEmpty():
            continue
        bbox_range = Gf.Vec2f(bbox.GetRange().GetMin()[1], bbox.GetRange().GetMax()[1])
        bbox_ranges.append(bbox_range)
    return bbox_ranges


def compute_union_of_bounding_boxes(bboxes: Vt.Range3fArray) -> Gf.Range3f:
    """Compute the union of an array of bounding boxes.

    Args:
        bboxes (Vt.Range3fArray): An array of Gf.Range3f bounding boxes.

    Returns:
        Gf.Range3f: The union of all the bounding boxes in the input array.

    Raises:
        ValueError: If the input array is empty.
    """
    if not bboxes:
        raise ValueError("Input bounding box array is empty")
    union_bbox = Gf.Range3f(bboxes[0])
    for bbox in bboxes[1:]:
        union_bbox.UnionWith(bbox)
    return union_bbox


def filter_bounding_boxes_by_size(
    bounding_boxes: Vt.Range3dArray, min_size: Tuple[float, float, float], max_size: Tuple[float, float, float]
) -> Vt.Range3dArray:
    """Filter bounding boxes by minimum and maximum size.

    Args:
        bounding_boxes (Vt.Range3dArray): Array of bounding boxes to filter.
        min_size (Tuple[float, float, float]): Minimum size for each dimension (x, y, z).
        max_size (Tuple[float, float, float]): Maximum size for each dimension (x, y, z).

    Returns:
        Vt.Range3dArray: Filtered array of bounding boxes.
    """
    if any((min_val > max_val for (min_val, max_val) in zip(min_size, max_size))):
        raise ValueError("Minimum size cannot be greater than maximum size.")
    filtered_boxes = []
    for bbox in bounding_boxes:
        size = (bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2])
        if all((min_val <= size_val <= max_val for (min_val, size_val, max_val) in zip(min_size, size, max_size))):
            filtered_boxes.append(bbox)
    return Vt.Range3dArray(filtered_boxes)


def get_bounding_boxes_within_range(bounding_boxes: Vt.Range2dArray, range_to_check: Gf.Range2d) -> List[Gf.Range2d]:
    """
    Get a list of bounding boxes that overlap with the given range.

    Args:
        bounding_boxes (Vt.Range2dArray): An array of bounding boxes.
        range_to_check (Gf.Range2d): The range to check for overlap.

    Returns:
        List[Gf.Range2d]: A list of bounding boxes that overlap with the given range.
    """
    if not isinstance(bounding_boxes, Vt.Range2dArray):
        raise TypeError("bounding_boxes must be of type Vt.Range2dArray")
    if not isinstance(range_to_check, Gf.Range2d):
        raise TypeError("range_to_check must be of type Gf.Range2d")
    overlapping_boxes = []
    for bounding_box in bounding_boxes:
        if (
            range_to_check.Contains(bounding_box.GetMin())
            or range_to_check.Contains(bounding_box.GetMax())
            or bounding_box.Contains(range_to_check.GetMin())
            or bounding_box.Contains(range_to_check.GetMax())
        ):
            overlapping_boxes.append(bounding_box)
    return overlapping_boxes


def compute_intersection_of_bounding_boxes(bboxes: Vt.Range2dArray) -> Gf.Range2d:
    """Compute the intersection of a list of 2D bounding boxes.

    Args:
        bboxes (Vt.Range2dArray): An array of 2D bounding boxes.

    Returns:
        Gf.Range2d: The intersection of the input bounding boxes.
            If no intersection exists, an empty range is returned.
    """
    if not bboxes:
        return Gf.Range2d()
    intersection = Gf.Range2d(bboxes[0])
    for bbox in bboxes[1:]:
        intersection = Gf.Range2d(
            Gf.Vec2d(max(intersection.GetMin()[0], bbox.GetMin()[0]), max(intersection.GetMin()[1], bbox.GetMin()[1])),
            Gf.Vec2d(min(intersection.GetMax()[0], bbox.GetMax()[0]), min(intersection.GetMax()[1], bbox.GetMax()[1])),
        )
        if intersection.IsEmpty():
            break
    return intersection


def transform_ranges(ranges: Vt.Range2fArray, transform: Gf.Matrix3f) -> Vt.Range2fArray:
    """Transform an array of ranges by a matrix.

    Args:
        ranges (Vt.Range2fArray): The input array of ranges.
        transform (Gf.Matrix3f): The transformation matrix.

    Returns:
        Vt.Range2fArray: The transformed array of ranges.
    """
    if not ranges:
        return ranges
    transformed_ranges = Vt.Range2fArray(len(ranges))
    for i, range_ in enumerate(ranges):
        min_point = Gf.Vec3f(range_.GetMin()[0], range_.GetMin()[1], 1)
        max_point = Gf.Vec3f(range_.GetMax()[0], range_.GetMax()[1], 1)
        transformed_min = transform * min_point
        transformed_max = transform * max_point
        transformed_range = Gf.Range2f(
            Gf.Vec2f(min(transformed_min[0], transformed_max[0]), min(transformed_min[1], transformed_max[1])),
            Gf.Vec2f(max(transformed_min[0], transformed_max[0]), max(transformed_min[1], transformed_max[1])),
        )
        transformed_ranges[i] = transformed_range
    return transformed_ranges


def merge_range_arrays(range_arrays: Sequence[Vt.Range2fArray]) -> Vt.Range2fArray:
    """Merge multiple Range2fArray into a single Range2fArray.

    Args:
        range_arrays (Sequence[Vt.Range2fArray]): A sequence of Range2fArray objects to merge.

    Returns:
        Vt.Range2fArray: The merged Range2fArray containing all ranges from the input arrays.
    """
    if not range_arrays:
        return Vt.Range2fArray()
    merged_ranges = []
    for array in range_arrays:
        for i in range(len(array)):
            range_value = array[i]
            merged_ranges.append(range_value)
    merged_array = Vt.Range2fArray(merged_ranges)
    return merged_array


def filter_ranges_by_size(ranges: Vt.Range2fArray, min_size: Gf.Vec2f, max_size: Gf.Vec2f) -> Vt.Range2fArray:
    """Filter a Vt.Range2fArray by minimum and maximum range size.

    Args:
        ranges (Vt.Range2fArray): The input array of GfRange2f ranges.
        min_size (Gf.Vec2f): The minimum size of range to include.
        max_size (Gf.Vec2f): The maximum size of range to include.

    Returns:
        Vt.Range2fArray: A new array containing only ranges within the specified size.
    """
    if min_size[0] > max_size[0] or min_size[1] > max_size[1]:
        raise ValueError("min_size must be less than or equal to max_size")
    filtered_ranges = []
    for range_2f in ranges:
        range_size = range_2f.GetSize()
        if min_size[0] <= range_size[0] <= max_size[0] and min_size[1] <= range_size[1] <= max_size[1]:
            filtered_ranges.append(range_2f)
    return Vt.Range2fArray(filtered_ranges)


def compute_union_of_ranges(ranges: Vt.Range2fArray) -> List[Tuple[float, float]]:
    """Compute the union of a list of ranges.

    Args:
        ranges (Vt.Range2fArray): An array of Gf.Range2f ranges.

    Returns:
        List[Tuple[float, float]]: A list of tuples representing the union of the input ranges.
    """
    sorted_ranges = sorted(ranges, key=lambda r: r.GetMin()[0])
    union: List[Tuple[float, float]] = [(sorted_ranges[0].GetMin()[0], sorted_ranges[0].GetMax()[0])]
    for current_range in sorted_ranges[1:]:
        prev_range = union[-1]
        if current_range.GetMin()[0] <= prev_range[1]:
            union[-1] = (prev_range[0], max(prev_range[1], current_range.GetMax()[0]))
        else:
            union.append((current_range.GetMin()[0], current_range.GetMax()[0]))
    return union


def merge_bounding_boxes(bboxes: Vt.Range3fArray) -> Gf.Range3f:
    """Merge a list of bounding boxes into a single bounding box.

    Args:
        bboxes (Vt.Range3fArray): An array of bounding boxes to merge.

    Returns:
        Gf.Range3f: The merged bounding box.
    """
    if not bboxes:
        raise ValueError("Input bounding box array is empty.")
    merged_bbox = Gf.Range3f(bboxes[0])
    for bbox in bboxes[1:]:
        if bbox.IsEmpty():
            raise ValueError("Empty bounding box encountered.")
        merged_bbox.UnionWith(bbox)
    return merged_bbox


def transform_bounding_boxes(bounding_boxes: Vt.Range3fArray, matrix: Gf.Matrix4d) -> Vt.Range3fArray:
    """
    Transform an array of bounding boxes by a given matrix.

    Args:
        bounding_boxes (Vt.Range3fArray): The array of bounding boxes to transform.
        matrix (Gf.Matrix4d): The transformation matrix to apply.

    Returns:
        Vt.Range3fArray: The transformed bounding boxes.
    """
    if not isinstance(bounding_boxes, Vt.Range3fArray):
        raise TypeError("bounding_boxes must be of type Vt.Range3fArray")
    if not isinstance(matrix, Gf.Matrix4d):
        raise TypeError("matrix must be of type Gf.Matrix4d")
    transformed_boxes = Vt.Range3fArray(len(bounding_boxes))
    for i, bbox in enumerate(bounding_boxes):
        corners = [
            bbox.GetCorner(0),
            bbox.GetCorner(1),
            bbox.GetCorner(2),
            bbox.GetCorner(3),
            bbox.GetCorner(4),
            bbox.GetCorner(5),
            bbox.GetCorner(6),
            bbox.GetCorner(7),
        ]
        transformed_corners = [matrix.Transform(Gf.Vec3d(corner)) for corner in corners]
        min_point = Gf.Vec3f(
            min((corner[0] for corner in transformed_corners)),
            min((corner[1] for corner in transformed_corners)),
            min((corner[2] for corner in transformed_corners)),
        )
        max_point = Gf.Vec3f(
            max((corner[0] for corner in transformed_corners)),
            max((corner[1] for corner in transformed_corners)),
            max((corner[2] for corner in transformed_corners)),
        )
        transformed_boxes[i] = Gf.Range3f(min_point, max_point)
    return transformed_boxes


def filter_bounding_boxes_by_volume(bboxes: Vt.Range3fArray, min_volume: float) -> List[Gf.Range3f]:
    """Filter bounding boxes by minimum volume.

    Args:
        bboxes (Vt.Range3fArray): An array of bounding boxes.
        min_volume (float): The minimum volume threshold.

    Returns:
        List[Gf.Range3f]: A list of bounding boxes with volume greater than or equal to the minimum volume.
    """
    if not bboxes:
        return []
    filtered_bboxes = []
    for bbox in bboxes:
        dimensions = bbox.max - bbox.min
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        if volume >= min_volume:
            filtered_bboxes.append(bbox)
    return filtered_bboxes


def get_prim_bounding_boxes_at_times(prim: Usd.Prim, times: Vt.Vec3fArray) -> List[Tuple[Gf.Vec3f, Gf.Vec3f]]:
    """
    Get the bounding boxes of a prim at specified times.

    Args:
        prim (Usd.Prim): The prim to compute bounding boxes for.
        times (Vt.Vec3fArray): The times at which to compute bounding boxes.

    Returns:
        List[Tuple[Gf.Vec3f, Gf.Vec3f]]: A list of tuples containing the minimum and maximum
        points of the bounding box at each time.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    bboxes = []
    for time in times:
        time_code = Usd.TimeCode(time[0])
        bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])
        bbox = bbox_cache.ComputeWorldBound(prim)
        if bbox.GetRange().IsEmpty():
            raise ValueError(f"Empty bounding box at time {time} for prim {prim.GetPath()}")
        min_point = Gf.Vec3f(bbox.GetRange().GetMin())
        max_point = Gf.Vec3f(bbox.GetRange().GetMax())
        bboxes.append((min_point, max_point))
    return bboxes


def transform_rect2iarray(rect_array: Vt.Rect2iArray, translation: Gf.Vec2i) -> Vt.Rect2iArray:
    """
    Translate each rectangle in the given Vt.Rect2iArray by the specified translation vector.

    Args:
        rect_array (Vt.Rect2iArray): The input array of rectangles.
        translation (Gf.Vec2i): The translation vector to apply to each rectangle.

    Returns:
        Vt.Rect2iArray: A new array of rectangles with the translation applied.
    """
    transformed_list = []
    for rect in rect_array:
        new_min = rect.GetMin() + translation
        new_max = rect.GetMax() + translation
        new_rect = Gf.Rect2i(new_min, new_max)
        transformed_list.append(new_rect)
    transformed_array = Vt.Rect2iArray(transformed_list)
    return transformed_array


def filter_rect2iarray_by_area(rects: Vt.Rect2iArray, min_area: int) -> Vt.Rect2iArray:
    """Filter a Vt.Rect2iArray by minimum area.

    Args:
        rects (Vt.Rect2iArray): The input array of Gf.Rect2i rectangles.
        min_area (int): The minimum area for a rectangle to pass the filter.

    Returns:
        Vt.Rect2iArray: A new array containing only rectangles with area >= min_area.
    """
    if min_area < 0:
        raise ValueError(f"min_area must be non-negative, got {min_area}")
    output = []
    for rect in rects:
        area = (rect.GetMaxX() - rect.GetMinX()) * (rect.GetMaxY() - rect.GetMinY())
        if area >= min_area:
            output.append(rect)
    return Vt.Rect2iArray(output)


def subdivide_rect2iarray(rects: Vt.Rect2iArray, subdivision_levels: int) -> Vt.Rect2iArray:
    """Subdivide an array of rectangles into smaller rectangles.

    Args:
        rects (Vt.Rect2iArray): The input array of rectangles to subdivide.
        subdivision_levels (int): The number of levels to subdivide each rectangle.

    Returns:
        Vt.Rect2iArray: The subdivided rectangles.
    """
    if subdivision_levels < 1:
        raise ValueError("Subdivision levels must be greater than or equal to 1")
    result = []
    for rect in rects:
        if rect.IsEmpty():
            continue
        sub_width = rect.GetWidth() // 2**subdivision_levels
        sub_height = rect.GetHeight() // 2**subdivision_levels
        if sub_width < 1 or sub_height < 1:
            result.append(rect)
            continue
        for y in range(2**subdivision_levels):
            for x in range(2**subdivision_levels):
                sub_min = Gf.Vec2i(rect.GetMin()[0] + x * sub_width, rect.GetMin()[1] + y * sub_height)
                sub_max = Gf.Vec2i(sub_min[0] + sub_width - 1, sub_min[1] + sub_height - 1)
                sub_rect = Gf.Rect2i(sub_min, sub_max)
                result.append(sub_rect)
    return Vt.Rect2iArray(result)


def merge_rect2iarrays(arrays: List[Vt.Rect2iArray]) -> Vt.Rect2iArray:
    """Merge multiple Rect2iArray instances into a single Rect2iArray.

    Args:
        arrays: A list of Rect2iArray instances to merge.

    Returns:
        A new Rect2iArray containing all the rectangles from the input arrays.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    merged_rects = []
    for array in arrays:
        for rect in array:
            merged_rects.append(rect)
    merged_array = Vt.Rect2iArray(merged_rects)
    return merged_array


def get_rect2iarray_bounds(rect2iarray: Vt.Rect2iArray) -> Gf.Rect2i:
    """Get the bounding rectangle for a Rect2iArray."""
    if not rect2iarray:
        return Gf.Rect2i()
    bounds = rect2iarray[0]
    for rect in rect2iarray[1:]:
        bounds = Gf.Rect2i(
            Gf.Vec2i(min(bounds.GetMin()[0], rect.GetMin()[0]), min(bounds.GetMin()[1], rect.GetMin()[1])),
            Gf.Vec2i(max(bounds.GetMax()[0], rect.GetMax()[0]), max(bounds.GetMax()[1], rect.GetMax()[1])),
        )
    return bounds


def interpolate_short_array_attributes(attr_a: Vt.ShortArray, attr_b: Vt.ShortArray, t: float) -> Vt.ShortArray:
    """Interpolate between two Vt.ShortArray attributes.

    Args:
        attr_a (Vt.ShortArray): The first attribute to interpolate from.
        attr_b (Vt.ShortArray): The second attribute to interpolate to.
        t (float): The interpolation factor, in the range [0.0, 1.0].

    Returns:
        Vt.ShortArray: The interpolated attribute value.

    Raises:
        ValueError: If the input attributes have different lengths or if t is outside the valid range.
    """
    if len(attr_a) != len(attr_b):
        raise ValueError("Input attributes must have the same length.")
    if t < 0.0 or t > 1.0:
        raise ValueError("Interpolation factor t must be in the range [0.0, 1.0].")
    interpolated_attr = Vt.ShortArray(len(attr_a))
    for i in range(len(attr_a)):
        interpolated_value = int(attr_a[i] + (attr_b[i] - attr_a[i]) * t)
        interpolated_attr[i] = interpolated_value
    return interpolated_attr


def create_short_array_attribute(prim: Usd.Prim, attribute_name: str, short_array: Vt.ShortArray) -> Usd.Attribute:
    """Creates an attribute for a prim that holds a short array.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute to create.
        short_array (Vt.ShortArray): The initial value for the attribute.

    Returns:
        Usd.Attribute: The created attribute.

    Raises:
        ValueError: If the prim is not valid.
        TypeError: If the provided value is not a Vt.ShortArray.
    """
    if not prim.IsValid():
        raise ValueError("The provided prim is not valid.")
    if not isinstance(short_array, Vt.ShortArray):
        raise TypeError("The provided value must be a Vt.ShortArray.")
    int_array = Vt.IntArray(short_array)
    attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.IntArray)
    attribute.Set(int_array)
    return attribute


def set_short_array_attribute_values(prim: Usd.Prim, attribute_name: str, values: Vt.ShortArray) -> None:
    """Set the values of a short array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute to set.
        values (Vt.ShortArray): The short array values to set.

    Raises:
        ValueError: If the prim is not valid or the attribute does not exist.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsDefined():
        raise ValueError(f"Attribute '{attribute_name}' does not exist on prim '{prim.GetPath()}'.")
    int_array = Vt.IntArray(values)
    attribute.Set(int_array)


def get_short_array_attribute_values(prim: Usd.Prim, attribute_name: str) -> List[int]:
    """
    Get the values of a short array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to retrieve the attribute from.
        attribute_name (str): The name of the short array attribute.

    Returns:
        List[int]: The values of the short array attribute.

    Raises:
        ValueError: If the prim is not valid or if the attribute does not exist or is not of type short array.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not prim.HasAttribute(attribute_name):
        raise ValueError(f"Attribute {attribute_name} does not exist on prim {prim.GetPath()}.")
    attribute = prim.GetAttribute(attribute_name)
    if attribute.GetTypeName() != Sdf.ValueTypeNames.IntArray:
        raise ValueError(f"Attribute {attribute_name} on prim {prim.GetPath()} is not of type short array.")
    short_array = attribute.Get()
    values = [int(value) for value in short_array]
    return values


def append_to_short_array_attribute(prim: Usd.Prim, attribute_name: str, value: int):
    """Append a value to a short array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to add the attribute to.
        attribute_name (str): The name of the attribute.
        value (int): The value to append to the array.

    Raises:
        ValueError: If the prim is not valid.
        TypeError: If the attribute exists but is not a short array.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid")
    attr = prim.GetAttribute(attribute_name)
    if not attr:
        attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Int64Array)
    elif attr.GetTypeName() != Sdf.ValueTypeNames.Int64Array:
        raise TypeError(f"Attribute '{attribute_name}' is not a short array")
    short_array = attr.Get(time=Usd.TimeCode.Default())
    if short_array is None:
        short_array = Vt.Int64Array()
    new_array = list(short_array)
    new_array.append(value)
    attr.Set(Vt.Int64Array(new_array))


def get_prim_paths_with_attribute(stage: Usd.Stage, attribute_name: str) -> List[str]:
    """
    Return a list of prim paths that have the specified attribute.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        attribute_name (str): The name of the attribute to search for.

    Returns:
        List[str]: A list of prim paths that have the specified attribute.
    """
    prim_paths: List[str] = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            prim_paths.append(str(prim.GetPath()))
    return prim_paths


def filter_prims_by_type_and_name(
    stage: Usd.Stage, prims: Vt.StringArray, type_name: str, name_pattern: Optional[str] = None
) -> Vt.StringArray:
    """Filter prim paths by prim type name and name pattern.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        prims (Vt.StringArray): Array of prim paths.
        type_name (str): Prim type name to filter by.
        name_pattern (str, optional): Optional glob pattern to match prim name against. Defaults to None.

    Returns:
        Vt.StringArray: Filtered array of prim paths.
    """
    filtered_prims = Vt.StringArray()
    for prim_path in prims:
        prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
        if prim.IsValid() and prim.GetTypeName() == type_name:
            if name_pattern is None or prim.GetName().startswith(name_pattern.strip("*")):
                filtered_prims.append(prim_path)
    return filtered_prims


def find_prims_with_metadata(stage: Usd.Stage, metadata_key: str) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have the specified metadata key.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        metadata_key (str): The metadata key to search for on prims.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified metadata key.
    """
    prims_with_metadata: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasMetadata(metadata_key):
            prims_with_metadata.append(prim)
    return prims_with_metadata


def batch_set_prim_attributes(stage: Usd.Stage, prim_path_prefix: str, attribute_name: str, values: Vt.StringArray):
    """Batch set attributes on prims with a common prefix.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path_prefix (str): The common prefix for the prim paths.
        attribute_name (str): The name of the attribute to set.
        values (Vt.StringArray): The array of string values to set on the attributes.
    """
    prims = stage.TraverseAll()
    matching_prims = [prim for prim in prims if prim.GetPath().pathString.startswith(prim_path_prefix)]
    if not matching_prims:
        raise ValueError(f"No prims found with prefix {prim_path_prefix}")
    if len(values) != len(matching_prims):
        raise ValueError(f"Number of values ({len(values)}) does not match the number of prims ({len(matching_prims)})")
    for prim, value in zip(matching_prims, values):
        if not prim.HasAttribute(attribute_name):
            prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.String)
        attribute = prim.GetAttribute(attribute_name)
        attribute.Set(value)


def replace_string_in_prim_names(stage: Usd.Stage, old_string: str, new_string: str) -> None:
    """Replace a string in all prim names on the given stage.

    Args:
        stage (Usd.Stage): The USD stage to operate on.
        old_string (str): The string to replace in prim names.
        new_string (str): The string to replace old_string with in prim names.
    """
    prims_to_rename = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        if prim.IsPseudoRoot():
            continue
        prim_name = prim.GetName()
        if old_string in prim_name:
            new_prim_name = prim_name.replace(old_string, new_string)
            prims_to_rename.append((prim, new_prim_name))
    for prim, new_name in prims_to_rename:
        prim.GetPath().ReplacePrefix(prim.GetPath(), Sdf.Path(prim.GetPath().GetParentPath().AppendChild(new_name)))


def get_all_tokens_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> Vt.TokenArray:
    """Get all unique token values from token attributes on prims.

    Args:
        stage (Usd.Stage): The USD stage to query.
        prim_paths (List[str]): A list of prim paths to query for token attributes.

    Returns:
        Vt.TokenArray: An array of unique TfToken values found on the prims.
    """
    tokens = set()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        for attr in prim.GetAttributes():
            if attr.GetTypeName() == Sdf.ValueTypeNames.Token:
                token = attr.Get()
                if token is not None:
                    tokens.add(token)
            elif attr.GetTypeName() == Sdf.ValueTypeNames.TokenArray:
                token_array = attr.Get()
                if token_array is not None:
                    for token in token_array:
                        tokens.add(token)
    return Vt.TokenArray(sorted(tokens))


def filter_prims_by_token(prims: List[Usd.Prim], token: Vt.TokenArray) -> List[Usd.Prim]:
    """Filter a list of prims by a token array.

    Args:
        prims (List[Usd.Prim]): A list of USD prims to filter.
        token (Vt.TokenArray): A token array to filter the prims by.

    Returns:
        List[Usd.Prim]: A list of prims that match the token array.
    """
    if not prims:
        return []
    if not token:
        return []
    token_set = set(token)
    filtered_prims = []
    for prim in prims:
        if prim.GetTypeName() in token_set:
            filtered_prims.append(prim)
    return filtered_prims


def merge_token_arrays(token_arrays: List[Vt.TokenArray]) -> Vt.TokenArray:
    """
    Merge multiple TokenArrays into a single TokenArray, removing duplicates.

    Args:
        token_arrays: A list of TokenArrays to merge.

    Returns:
        A new TokenArray containing the merged tokens without duplicates.
    """
    unique_tokens = set()
    for token_array in token_arrays:
        if not isinstance(token_array, Vt.TokenArray):
            raise TypeError(f"Expected Vt.TokenArray, got {type(token_array)}")
        for token in token_array:
            unique_tokens.add(token)
    merged_array = Vt.TokenArray(list(unique_tokens))
    return merged_array


def add_tokens_to_prim(prim: Usd.Prim, attribute_name: str, tokens: Vt.TokenArray) -> None:
    """Add a token array attribute to a prim.

    Args:
        prim (Usd.Prim): The prim to add the attribute to.
        attribute_name (str): The name of the attribute to create.
        tokens (Vt.TokenArray): The token array to set as the attribute value.

    Raises:
        ValueError: If the prim is not valid or if the attribute name is empty.
    """
    if not prim.IsValid():
        raise ValueError("Prim is not valid.")
    if not attribute_name:
        raise ValueError("Attribute name cannot be empty.")
    attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.TokenArray)
    attribute.Set(tokens)


def replace_token_in_prims(
    stage: Usd.Stage, prim_paths: List[str], token_attr_name: str, old_token: str, new_token: str
) -> None:
    """Replace a token value in a token-valued attribute on multiple prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): The paths to the prims to update.
        token_attr_name (str): The name of the token-valued attribute.
        old_token (str): The token value to replace.
        new_token (str): The new token value.

    Raises:
        ValueError: If any prim path is invalid or if the attribute does not exist or is not token-valued on any prim.
    """
    if not stage or not prim_paths or (not token_attr_name) or (not old_token) or (not new_token):
        raise ValueError("All input arguments must be non-empty")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist")
        if not prim.HasAttribute(token_attr_name):
            raise ValueError(f"Attribute {token_attr_name} does not exist on prim at path {prim_path}")
        attr = prim.GetAttribute(token_attr_name)
        if attr.GetTypeName() != Sdf.ValueTypeNames.TokenArray:
            raise ValueError(f"Attribute {token_attr_name} on prim at path {prim_path} is not token array-valued")
        token_array = attr.Get()
        if old_token not in token_array:
            continue
        new_token_array = Vt.TokenArray([new_token if t == old_token else t for t in token_array])
        attr.Set(new_token_array)


def set_uchar_array_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.UCharArray) -> None:
    """Set a UCharArray attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute.
        value (Vt.UCharArray): The UCharArray value to set.

    Raises:
        ValueError: If the prim is not valid.
        RuntimeError: If the attribute creation fails.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not prim.HasAttribute(attribute_name):
        success = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.UCharArray)
        if not success:
            raise RuntimeError("Failed to create attribute.")
    attr = prim.GetAttribute(attribute_name)
    if not attr.IsValid():
        raise RuntimeError("Failed to get attribute.")
    attr.Set(value)


def concatenate_uchar_arrays(arrays: Sequence[Vt.UCharArray]) -> Vt.UCharArray:
    """Concatenate multiple UCharArrays into a single UCharArray.

    Args:
        arrays (Sequence[Vt.UCharArray]): A sequence of UCharArrays to concatenate.

    Returns:
        Vt.UCharArray: The concatenated UCharArray.

    Raises:
        ValueError: If the input sequence is empty.
    """
    if not arrays:
        raise ValueError("Input sequence of arrays is empty.")
    total_size = sum((len(arr) for arr in arrays))
    concatenated_array = Vt.UCharArray(total_size)
    offset = 0
    for arr in arrays:
        size = len(arr)
        concatenated_array[offset : offset + size] = arr
        offset += size
    return concatenated_array


def uchar_array_to_image(uchar_array: Vt.UCharArray, width: int, height: int) -> "Image.Image":
    """Convert a Vt.UCharArray to a PIL Image.

    Args:
        uchar_array (Vt.UCharArray): The input array of unsigned char values.
        width (int): The width of the image.
        height (int): The height of the image.

    Returns:
        Image.Image: The converted PIL Image.

    Raises:
        ValueError: If the input array size does not match the provided width and height.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is not installed. Please install it to use this function.")

    if len(uchar_array) != width * height * 4:
        raise ValueError("Input array size does not match the provided width and height.")
    image = Image.frombytes("RGBA", (width, height), uchar_array)
    return image


def create_uchar_array_from_image(image_path: str) -> Tuple[Vt.UCharArray, int, int]:
    """Create a UCharArray from an image file."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("PIL is not installed. Please install it to use this function.")

    if not os.path.isfile(image_path):
        raise ValueError(f"Image file not found: {image_path}")
    try:
        image = Image.open(image_path)
        image = image.convert("RGB")
        (width, height) = image.size
        pixel_data = list(image.getdata())
        uchar_array = Vt.UCharArray(len(pixel_data) * 3)
        for i, pixel in enumerate(pixel_data):
            uchar_array[i * 3] = pixel[0]
            uchar_array[i * 3 + 1] = pixel[1]
            uchar_array[i * 3 + 2] = pixel[2]
        return (uchar_array, width, height)
    except (IOError, ValueError) as e:
        raise ValueError(f"Error loading image: {str(e)}")


def create_uint64array_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.UInt64Array) -> Usd.Attribute:
    """Create a uint64 array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute.
        value (Vt.UInt64Array): The initial value for the attribute.

    Returns:
        Usd.Attribute: The created attribute.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    if prim.HasAttribute(attribute_name):
        raise ValueError(f"Attribute {attribute_name} already exists on prim {prim.GetPath()}")
    attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.UInt64Array)
    attr.Set(value)
    return attr


def get_uint64array_attribute_values(stage: Usd.Stage, prim_path: str, attribute_name: str) -> Vt.UInt64Array:
    """Get the value of a Vt.UInt64Array attribute on a prim.

    Args:
        stage (Usd.Stage): The stage containing the prim.
        prim_path (str): The path to the prim.
        attribute_name (str): The name of the Vt.UInt64Array attribute.

    Returns:
        Vt.UInt64Array: The value of the attribute, or an empty array if the attribute does not exist or has no value.

    Raises:
        ValueError: If the prim does not exist or the attribute is not of type Vt.UInt64Array.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute:
        return Vt.UInt64Array()
    if attribute.GetTypeName() != Sdf.ValueTypeNames.UInt64Array:
        raise ValueError(f"Attribute {attribute_name} is not of type Vt.UInt64Array.")
    value = attribute.Get()
    if value is None:
        return Vt.UInt64Array()
    return value


def filter_prims_by_uint64array_attribute(
    stage: Usd.Stage, attribute_name: str, attribute_value: int
) -> List[Usd.Prim]:
    """
    Filter prims on the stage by a specific UInt64Array attribute value.

    Args:
        stage (Usd.Stage): The USD stage to filter prims from.
        attribute_name (str): The name of the UInt64Array attribute to filter by.
        attribute_value (int): The value to match against the attribute.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified attribute with the given value.
    """
    prims_with_attribute = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            attribute = prim.GetAttribute(attribute_name)
            if attribute.GetTypeName() == Sdf.ValueTypeNames.UInt64Array:
                prims_with_attribute.append(prim)
    matching_prims = []
    for prim in prims_with_attribute:
        uint64_array = prim.GetAttribute(attribute_name).Get()
        if uint64_array is not None and attribute_value in uint64_array:
            matching_prims.append(prim)
    return matching_prims


def replace_uint64array_values(arr: Vt.UInt64Array, old_value: int, new_value: int) -> None:
    """Replace all occurrences of old_value with new_value in the given UInt64Array.

    Args:
        arr (Vt.UInt64Array): The array to modify.
        old_value (int): The value to replace.
        new_value (int): The new value to use as a replacement.

    Raises:
        TypeError: If arr is not a Vt.UInt64Array.
        ValueError: If old_value or new_value is negative.
    """
    if not isinstance(arr, Vt.UInt64Array):
        raise TypeError("Input must be a Vt.UInt64Array")
    if old_value < 0 or new_value < 0:
        raise ValueError("Old and new values must be non-negative")
    for i in range(len(arr)):
        if arr[i] == old_value:
            arr[i] = new_value


def append_to_uint64array_attribute(prim: Usd.Prim, attribute_name: str, value: int) -> bool:
    """Append a value to a uint64 array attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to add the attribute to.
        attribute_name (str): The name of the attribute.
        value (int): The value to append to the array.

    Returns:
        bool: True if the value was appended successfully, False otherwise.
    """
    if not prim.IsValid():
        return False
    if prim.HasAttribute(attribute_name):
        attribute = prim.GetAttribute(attribute_name)
    else:
        attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.UInt64Array)
    if not attribute.IsValid() or attribute.GetTypeName() != Sdf.ValueTypeNames.UInt64Array:
        return False
    array = attribute.Get()
    if array is None:
        array = Vt.UInt64Array([value])
    else:
        array = Vt.UInt64Array(list(array) + [value])
    success = attribute.Set(array)
    return success


def create_uint_array_attribute(
    prim: Usd.Prim, attribute_name: str, default_value: Vt.UIntArray = None
) -> Usd.Attribute:
    """Creates an attribute of type UIntArray on the given prim.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute to create.
        default_value (Vt.UIntArray, optional): The default value for the attribute. Defaults to None.

    Returns:
        Usd.Attribute: The created attribute.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if prim.HasAttribute(attribute_name):
        raise ValueError(f"Attribute '{attribute_name}' already exists on prim '{prim.GetPath()}'")
    attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.UIntArray)
    if default_value is not None:
        attr.Set(default_value)
    return attr


def interpolate_uint_array_attributes(
    prim: Usd.Prim, attr_name: str, time_samples: Sequence[float], attr_values: Sequence[Vt.UIntArray]
) -> bool:
    """Interpolate UIntArray attribute values over time.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attr_name (str): The name of the attribute.
        time_samples (Sequence[float]): Time samples for keyframes.
        attr_values (Sequence[Vt.UIntArray]): Attribute values at each time sample.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not prim.IsValid():
        return False
    if len(time_samples) != len(attr_values):
        return False
    if len(time_samples) < 2:
        return False
    attr = prim.GetAttribute(attr_name)
    if not attr:
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.UIntArray)
    try:
        for t, v in zip(time_samples, attr_values):
            attr.Set(v, Usd.TimeCode(t))
    except Exception:
        return False
    return True


def filter_prims_by_uint_array_attribute(stage: Usd.Stage, attribute_name: str, expected_value: int) -> List[Usd.Prim]:
    """
    Filter prims in the stage that have a uint array attribute with the specified value.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        attribute_name (str): The name of the uint array attribute to check.
        expected_value (int): The expected value to match in the uint array attribute.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified uint array attribute with the expected value.
    """
    matching_prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            attr = prim.GetAttribute(attribute_name)
            if attr.GetTypeName() == Sdf.ValueTypeNames.UIntArray:
                uint_array = attr.Get()
                if expected_value in uint_array:
                    matching_prims.append(prim)
    return matching_prims


def merge_uint_arrays(arrays: List[Vt.UIntArray]) -> Vt.UIntArray:
    """Merge multiple UIntArrays into a single UIntArray.

    Args:
        arrays (List[Vt.UIntArray]): A list of UIntArrays to merge.

    Returns:
        Vt.UIntArray: The merged UIntArray.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    total_elements = sum((len(array) for array in arrays))
    merged_array = Vt.UIntArray(total_elements)
    index = 0
    for array in arrays:
        for element in array:
            merged_array[index] = element
            index += 1
    return merged_array


def set_ushort_array_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.UShortArray) -> None:
    """Set a Vt.UShortArray attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute to set.
        value (Vt.UShortArray): The value to set for the attribute.

    Raises:
        ValueError: If the prim is not valid or the attribute name is empty.
    """
    if not prim.IsValid():
        raise ValueError("Invalid prim.")
    if not attribute_name:
        raise ValueError("Attribute name cannot be empty.")
    if not prim.HasAttribute(attribute_name):
        prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.IntArray)
    attribute = prim.GetAttribute(attribute_name)
    value_list = [int(x) for x in value]
    attribute.Set(value_list)


def interpolate_vec2d_arrays(arrays: List[Vt.Vec2dArray], weights: List[float]) -> Vt.Vec2dArray:
    """
    Interpolate between multiple Vec2dArray using weights.

    Args:
        arrays (List[Vt.Vec2dArray]): List of Vec2dArray to interpolate between.
        weights (List[float]): List of weights for each array. Should sum to 1.

    Returns:
        Vt.Vec2dArray: Interpolated Vec2dArray.

    Raises:
        ValueError: If the input arrays are empty, have different lengths, or if the weights don't sum to 1.
    """
    if not arrays:
        raise ValueError("Input arrays cannot be empty.")
    if any((len(arr) != len(arrays[0]) for arr in arrays)):
        raise ValueError("All input arrays must have the same length.")
    if abs(sum(weights) - 1.0) > 1e-06:
        raise ValueError("Weights must sum to 1.")
    result = Vt.Vec2dArray(len(arrays[0]))
    for i in range(len(arrays[0])):
        interpolated = Gf.Vec2d(0, 0)
        for arr, weight in zip(arrays, weights):
            interpolated += arr[i] * weight
        result[i] = interpolated
    return result


def transform_vec2d_array(vec2d_array: Vt.Vec2dArray, scale: float, offset: Gf.Vec2d) -> Vt.Vec2dArray:
    """Transform a Vec2dArray by scaling and offsetting each element.

    Args:
        vec2d_array (Vt.Vec2dArray): The input array of Vec2d elements.
        scale (float): The scale factor to apply to each element.
        offset (Gf.Vec2d): The offset to add to each element.

    Returns:
        Vt.Vec2dArray: The transformed array.
    """
    if not vec2d_array:
        return vec2d_array
    transformed_array = Vt.Vec2dArray(len(vec2d_array))
    for i, vec2d in enumerate(vec2d_array):
        scaled_vec2d = vec2d * scale
        transformed_vec2d = scaled_vec2d + offset
        transformed_array[i] = transformed_vec2d
    return transformed_array


def filter_vec2d_array(vec2d_array: Vt.Vec2dArray, min_length: float) -> Vt.Vec2dArray:
    """Filter a Vec2dArray to only include vectors with length greater than or equal to min_length.

    Args:
        vec2d_array (Vt.Vec2dArray): The input array of GfVec2d vectors.
        min_length (float): The minimum length threshold for filtering.

    Returns:
        Vt.Vec2dArray: A new Vec2dArray containing only the filtered vectors.
    """
    filtered_list = []
    for vec in vec2d_array:
        length = Gf.GetLength(Gf.Vec2d(vec))
        if length >= min_length:
            filtered_list.append(vec)
    return Vt.Vec2dArray(filtered_list)


def combine_vec2d_arrays(arrays: List[Vt.Vec2dArray]) -> Vt.Vec2dArray:
    """Combine multiple Vec2dArray into a single Vec2dArray.

    Args:
        arrays (List[Vt.Vec2dArray]): A list of Vec2dArray to combine.

    Returns:
        Vt.Vec2dArray: The combined Vec2dArray.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    combined_array = Vt.Vec2dArray()
    for array in arrays:
        combined_array += array
    return combined_array


def create_vec2f_array_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.Vec2fArray) -> Usd.Attribute:
    """Creates an attribute of type Vec2fArray on the given prim with the specified name and value.

    Args:
        prim (Usd.Prim): The prim to create the attribute on.
        attribute_name (str): The name of the attribute to create.
        value (Vt.Vec2fArray): The initial value to set for the attribute.

    Returns:
        Usd.Attribute: The created attribute.

    Raises:
        ValueError: If the prim is not valid.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    attr = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Float2Array)
    attr.Set(value)
    return attr


def transform_vec2f_array(vec2f_array: Vt.Vec2fArray, matrix: Gf.Matrix3f) -> Vt.Vec2fArray:
    """
    Transforms a Vec2fArray by applying a transformation matrix to each element.

    Args:
        vec2f_array (Vt.Vec2fArray): The input array of Vec2f elements.
        matrix (Gf.Matrix3f): The 3x3 transformation matrix.

    Returns:
        Vt.Vec2fArray: A new Vec2fArray with the transformed elements.
    """
    if not isinstance(vec2f_array, Vt.Vec2fArray):
        raise TypeError("Input must be a Vt.Vec2fArray")
    if not isinstance(matrix, Gf.Matrix3f):
        raise TypeError("Transformation matrix must be a Gf.Matrix3f")
    result = Vt.Vec2fArray(len(vec2f_array))
    for i, vec in enumerate(vec2f_array):
        vec3f = Gf.Vec3f(vec[0], vec[1], 1.0)
        transformed_vec3f = matrix * vec3f
        result[i] = Gf.Vec2f(transformed_vec3f[0], transformed_vec3f[1])
    return result


def sample_vec2f_array_over_time(attr: Usd.Attribute, time_samples: Sequence[float]) -> List[Vt.Vec2fArray]:
    """Sample a Vec2fArray attribute over a series of time samples.

    Args:
        attr (Usd.Attribute): The Vec2fArray attribute to sample.
        time_samples (Sequence[float]): A sequence of time samples to query the attribute at.

    Returns:
        List[Vt.Vec2fArray]: A list of Vec2fArrays, one for each time sample.

    Raises:
        ValueError: If the attribute is not a Vec2fArray or if it doesn't exist.
    """
    if not attr.IsValid():
        raise ValueError("Invalid attribute")
    if attr.GetTypeName() != Sdf.ValueTypeNames.Float2Array:
        raise ValueError("Attribute is not a Vec2fArray")
    result = []
    for time in time_samples:
        value = attr.Get(time)
        if value is not None:
            result.append(Vt.Vec2fArray(value))
        else:
            result.append(Vt.Vec2fArray())
    return result


def interpolate_vec2f_array_attributes(
    prim: Usd.Prim, attr_name: str, time_a: Usd.TimeCode, time_b: Usd.TimeCode, alpha: float
) -> Vt.Vec2fArray:
    """Linearly interpolate Vec2fArray attribute values between two times.

    Args:
        prim (Usd.Prim): The prim to get the attribute from.
        attr_name (str): The name of the Vec2fArray attribute.
        time_a (Usd.TimeCode): The first time to get the attribute value at.
        time_b (Usd.TimeCode): The second time to get the attribute value at.
        alpha (float): The interpolation factor between 0 and 1.

    Returns:
        Vt.Vec2fArray: The interpolated attribute value.
    """
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        raise ValueError(f"Attribute '{attr_name}' does not exist on prim '{prim.GetPath()}'")
    if attr.GetTypeName() != Sdf.ValueTypeNames.Float2Array:
        raise TypeError(f"Attribute '{attr_name}' is not of type Vec2fArray")
    value_a = attr.Get(time_a)
    value_b = attr.Get(time_b)
    if value_a is None or value_b is None:
        raise ValueError(f"Attribute '{attr_name}' has no value at times {time_a} or {time_b}")
    if len(value_a) != len(value_b):
        raise ValueError(f"Attribute '{attr_name}' has different array lengths at times {time_a} and {time_b}")
    result = Vt.Vec2fArray(len(value_a))
    for i in range(len(value_a)):
        result[i] = Gf.Lerp(alpha, Gf.Vec2f(value_a[i]), Gf.Vec2f(value_b[i]))
    return result


def interpolate_vec2harray(src_array: Vt.Vec2hArray, dst_array: Vt.Vec2hArray, weight: float) -> Vt.Vec2hArray:
    """Linearly interpolate between two Vec2hArray arrays.

    Args:
        src_array (Vt.Vec2hArray): The source array.
        dst_array (Vt.Vec2hArray): The destination array.
        weight (float): The interpolation weight (0.0 to 1.0).

    Returns:
        Vt.Vec2hArray: The interpolated array.

    Raises:
        ValueError: If the input arrays have different lengths or if the weight is outside the valid range.
    """
    if len(src_array) != len(dst_array):
        raise ValueError("Input arrays must have the same length.")
    if weight < 0.0 or weight > 1.0:
        raise ValueError("Interpolation weight must be between 0.0 and 1.0.")
    interpolated_array = Vt.Vec2hArray(len(src_array))
    for i in range(len(src_array)):
        interpolated_array[i] = src_array[i] * (1.0 - weight) + dst_array[i] * weight
    return interpolated_array


def find_extreme_values_vec2harray(vec2h_array: Vt.Vec2hArray) -> Tuple[Gf.Vec2h, Gf.Vec2h]:
    """
    Find the minimum and maximum values in a Vec2hArray.

    Args:
        vec2h_array (Vt.Vec2hArray): The input array of Vec2h values.

    Returns:
        Tuple[Gf.Vec2h, Gf.Vec2h]: A tuple containing the minimum and maximum Vec2h values.

    Raises:
        ValueError: If the input array is empty.
    """
    if not vec2h_array:
        raise ValueError("Input array cannot be empty.")
    min_value = vec2h_array[0]
    max_value = vec2h_array[0]
    for vec2h in vec2h_array:
        if vec2h[0] < min_value[0]:
            min_value = Gf.Vec2h(vec2h[0], min_value[1])
        elif vec2h[0] > max_value[0]:
            max_value = Gf.Vec2h(vec2h[0], max_value[1])
        if vec2h[1] < min_value[1]:
            min_value = Gf.Vec2h(min_value[0], vec2h[1])
        elif vec2h[1] > max_value[1]:
            max_value = Gf.Vec2h(max_value[0], vec2h[1])
    return (min_value, max_value)


def transform_vec2harray(vec2harray: Vt.Vec2hArray, transform_matrix: Gf.Matrix3d) -> Vt.Vec2hArray:
    """Transform a Vec2hArray by a 3x3 matrix.

    Args:
        vec2harray (Vt.Vec2hArray): The input Vec2hArray to transform.
        transform_matrix (Gf.Matrix3d): The 3x3 transformation matrix.

    Returns:
        Vt.Vec2hArray: The transformed Vec2hArray.
    """
    if not isinstance(vec2harray, Vt.Vec2hArray):
        raise TypeError("Input vec2harray must be of type Vt.Vec2hArray")
    if not isinstance(transform_matrix, Gf.Matrix3d):
        raise TypeError("Input transform_matrix must be of type Gf.Matrix3d")
    transformed_vec2harray = Vt.Vec2hArray(len(vec2harray))
    for i, vec2h in enumerate(vec2harray):
        vec3d = Gf.Vec3d(vec2h[0], vec2h[1], 1)
        transformed_vec3d = transform_matrix * vec3d
        transformed_vec2h = Gf.Vec2h(transformed_vec3d[0], transformed_vec3d[1])
        transformed_vec2harray[i] = transformed_vec2h
    return transformed_vec2harray


def filter_vec2harray_by_threshold(vec2h_array: Vt.Vec2hArray, threshold: float) -> Vt.Vec2hArray:
    """Filter a Vec2hArray by a threshold value.

    Args:
        vec2h_array (Vt.Vec2hArray): The input array of Vec2h values.
        threshold (float): The threshold value to filter by.

    Returns:
        Vt.Vec2hArray: A new Vec2hArray containing only values where both
                       components are greater than the threshold.
    """
    filtered_list = []
    for vec2h in vec2h_array:
        if vec2h[0] > threshold and vec2h[1] > threshold:
            filtered_list.append(vec2h)
    return Vt.Vec2hArray(filtered_list)


def merge_vec2harrays(arrays: List[Vt.Vec2hArray]) -> Vt.Vec2hArray:
    """Merge multiple Vec2hArray instances into a single Vec2hArray.

    Args:
        arrays (List[Vt.Vec2hArray]): A list of Vec2hArray instances to merge.

    Returns:
        Vt.Vec2hArray: A new Vec2hArray containing all elements from the input arrays.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    merged_elements = []
    for array in arrays:
        for element in array:
            merged_elements.append(element)
    merged_array = Vt.Vec2hArray(merged_elements)
    return merged_array


def transform_vec2i_array(vec2i_array: Vt.Vec2iArray, transform: Callable[[Gf.Vec2i], Gf.Vec2i]) -> Vt.Vec2iArray:
    """
    Apply a transformation function to each element of a Vec2iArray.

    Args:
        vec2i_array (Vt.Vec2iArray): The input array of Vec2i elements.
        transform (Callable[[Gf.Vec2i], Gf.Vec2i]): A function that takes a Vec2i and returns a transformed Vec2i.

    Returns:
        Vt.Vec2iArray: A new array with the transformed elements.
    """
    if not isinstance(vec2i_array, Vt.Vec2iArray):
        raise TypeError("Input must be a Vt.Vec2iArray")
    transformed_array = Vt.Vec2iArray(len(vec2i_array))
    for i, vec2i in enumerate(vec2i_array):
        transformed_vec2i = transform(vec2i)
        transformed_array[i] = transformed_vec2i
    return transformed_array


def double_vec2i(vec2i: Gf.Vec2i) -> Gf.Vec2i:
    return Gf.Vec2i(vec2i[0] * 2, vec2i[1] * 2)


def filter_vec2i_array(vec2i_array: Vt.Vec2iArray, min_value: int, max_value: int) -> Vt.Vec2iArray:
    """Filter a Vec2iArray to only include elements where both components are between min_value and max_value (inclusive)."""
    if not isinstance(vec2i_array, Vt.Vec2iArray):
        raise TypeError(f"Expected Vt.Vec2iArray, got {type(vec2i_array)}")
    if min_value > max_value:
        raise ValueError(f"min_value ({min_value}) must be less than or equal to max_value ({max_value})")
    filtered_array = []
    for vec2i in vec2i_array:
        if min_value <= vec2i[0] <= max_value and min_value <= vec2i[1] <= max_value:
            filtered_array.append(vec2i)
    return Vt.Vec2iArray(filtered_array)


def append_vec2i_array_attribute(prim: Usd.Prim, attr_name: str, value: Gf.Vec2i) -> None:
    """Append a Vec2i value to a Vec2iArray attribute on a prim.

    Args:
        prim (Usd.Prim): The prim to add the attribute to.
        attr_name (str): The name of the Vec2iArray attribute.
        value (Gf.Vec2i): The Vec2i value to append to the array.

    Raises:
        ValueError: If the specified attribute does not exist or is not a Vec2iArray.
    """
    if not prim.HasAttribute(attr_name):
        attr = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Int2Array)
    else:
        attr = prim.GetAttribute(attr_name)
    if attr.GetTypeName() != Sdf.ValueTypeNames.Int2Array:
        raise ValueError(f"Attribute '{attr_name}' is not a Vec2iArray.")
    vec2i_array = attr.Get()
    if vec2i_array is None:
        vec2i_array = Vt.Vec2iArray()
    vec2i_list = list(vec2i_array)
    vec2i_list.append(value)
    new_vec2i_array = Vt.Vec2iArray(vec2i_list)
    attr.Set(new_vec2i_array)


def merge_vec2i_array_attributes(prim: Usd.Prim, attr_names: List[str], new_attr_name: str) -> Usd.Attribute:
    """Merge multiple Vec2iArray attributes into a single Vec2iArray attribute.

    Args:
        prim (Usd.Prim): The prim containing the attributes to merge.
        attr_names (List[str]): List of attribute names to merge.
        new_attr_name (str): Name of the new merged attribute.

    Returns:
        Usd.Attribute: The newly created merged attribute.

    Raises:
        ValueError: If any of the specified attributes do not exist or are not of type Vec2iArray.
    """
    for attr_name in attr_names:
        attr = prim.GetAttribute(attr_name)
        if not attr or attr.GetTypeName() != Sdf.ValueTypeNames.Int2Array:
            raise ValueError(f"Attribute {attr_name} does not exist or is not of type Vec2iArray.")
    merged_attr = prim.CreateAttribute(new_attr_name, Sdf.ValueTypeNames.Int2Array)
    merged_values = []
    for attr_name in attr_names:
        attr = prim.GetAttribute(attr_name)
        values = attr.Get()
        if values:
            merged_values.extend(values)
    merged_attr.Set(merged_values)
    return merged_attr


def get_bounding_box_from_vec2i_array(points: Vt.Vec2iArray) -> Gf.Range2d:
    """
    Calculate the bounding box from an array of Vec2i points.

    Args:
        points (Vt.Vec2iArray): Array of Vec2i points.

    Returns:
        Gf.Range2d: Bounding box as a Range2d.

    Raises:
        ValueError: If the input array is empty.
    """
    if not points:
        raise ValueError("Input array is empty.")
    min_point = Gf.Vec2d(points[0])
    max_point = Gf.Vec2d(points[0])
    for point in points:
        min_point = Gf.Vec2d(min(min_point[0], point[0]), min(min_point[1], point[1]))
        max_point = Gf.Vec2d(max(max_point[0], point[0]), max(max_point[1], point[1]))
    return Gf.Range2d(min_point, max_point)


def transform_prims_to_origin(stage: Usd.Stage, prim_paths: Vt.StringArray) -> None:
    """
    Transform the given prims to the origin (0, 0, 0).

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (Vt.StringArray): An array of prim paths to transform.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            print(f"Warning: Prim at path {prim_path} is not transformable. Skipping.")
            continue
        ordered_ops = xformable.GetOrderedXformOps()
        translate_op = None
        for op in ordered_ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
                break
        if not translate_op:
            translate_op = add_translate_op(xformable)
        translate_op.Set(Gf.Vec3d(0, 0, 0))


def compute_bounding_box_of_prims(stage: Usd.Stage, prim_paths: List[str]) -> Gf.BBox3d:
    """Compute the bounding box of a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Gf.BBox3d: The computed bounding box.
    """
    bbox = Gf.BBox3d()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping...")
            continue
        prim_bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), purpose1="default")
        if prim_bbox.GetBox().IsEmpty():
            print(f"Warning: Prim at path {prim_path} has an empty bounding box. Skipping...")
            continue
        bbox = Gf.BBox3d.Combine(bbox, prim_bbox)
    return bbox


def transform_positions(positions: Vt.Vec3fArray, mat: Gf.Matrix4d) -> Vt.Vec3fArray:
    """Transform an array of positions by a matrix.

    Args:
        positions (Vt.Vec3fArray): The input array of positions.
        mat (Gf.Matrix4d): The transformation matrix.

    Returns:
        Vt.Vec3fArray: The transformed positions.
    """
    if not positions:
        raise ValueError("Input positions array is empty.")
    if not mat or mat == Gf.Matrix4d(0.0):
        raise ValueError("Input matrix is invalid.")
    transformed_positions = Vt.Vec3fArray(len(positions))
    for i, pos in enumerate(positions):
        gf_pos = Gf.Vec3f(pos)
        transformed_pos = mat.Transform(gf_pos)
        transformed_positions[i] = transformed_pos
    return transformed_positions


def average_positions(positions: Vt.Vec3fArray) -> Tuple[float, float, float]:
    """Compute the average of a list of positions.

    Args:
        positions (Vt.Vec3fArray): A list of positions.

    Returns:
        Tuple[float, float, float]: The average position as a tuple of floats (x, y, z).
    """
    if not positions:
        return (0.0, 0.0, 0.0)
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    for pos in positions:
        sum_x += pos[0]
        sum_y += pos[1]
        sum_z += pos[2]
    count = len(positions)
    avg_x = sum_x / count
    avg_y = sum_y / count
    avg_z = sum_z / count
    return (avg_x, avg_y, avg_z)


def get_vertex_normals(points: Vt.Vec3fArray, indices: Vt.IntArray) -> Vt.Vec3fArray:
    """
    Compute vertex normals from points and triangle indices.

    Args:
        points (Vt.Vec3fArray): Array of vertex positions.
        indices (Vt.IntArray): Array of triangle indices.

    Returns:
        Vt.Vec3fArray: Array of vertex normals.
    """
    if len(points) == 0:
        raise ValueError("Points array is empty.")
    if len(indices) == 0 or len(indices) % 3 != 0:
        raise ValueError("Indices array is empty or not a multiple of 3.")
    normals = Vt.Vec3fArray(len(points))
    for i in range(0, len(indices), 3):
        (i0, i1, i2) = (indices[i], indices[i + 1], indices[i + 2])
        (v0, v1, v2) = (points[i0], points[i1], points[i2])
        e1 = v1 - v0
        e2 = v2 - v0
        normal = Gf.Cross(e1, e2)
        normal.Normalize()
        normals[i0] += normal
        normals[i1] += normal
        normals[i2] += normal
    for i in range(len(normals)):
        normals[i].Normalize()
    return normals


def interpolate_positions(positions: Vt.Vec3fArray, num_steps: int) -> Vt.Vec3fArray:
    """
    Interpolate between positions to create a smooth curve.

    Args:
        positions (Vt.Vec3fArray): The input positions to interpolate.
        num_steps (int): The number of interpolation steps between each pair of positions.

    Returns:
        Vt.Vec3fArray: The interpolated positions.
    """
    if len(positions) < 2:
        raise ValueError("At least two positions are required for interpolation.")
    if num_steps < 1:
        raise ValueError("The number of interpolation steps must be greater than or equal to 1.")
    interpolated_positions = []
    for i in range(len(positions) - 1):
        start_pos = positions[i]
        end_pos = positions[i + 1]
        interpolated_positions.append(start_pos)
        for j in range(1, num_steps):
            t = j / num_steps
            interpolated_pos = start_pos * (1 - t) + end_pos * t
            interpolated_positions.append(interpolated_pos)
    interpolated_positions.append(positions[-1])
    return Vt.Vec3fArray(interpolated_positions)


def blend_vec3h_arrays(a: Vt.Vec3hArray, b: Vt.Vec3hArray, t: float) -> Vt.Vec3hArray:
    """Blend two Vec3hArray arrays by the given factor t.

    Args:
        a (Vt.Vec3hArray): The first input array.
        b (Vt.Vec3hArray): The second input array.
        t (float): The blending factor between 0 and 1.
                   0 means use only values from a.
                   1 means use only values from b.

    Returns:
        Vt.Vec3hArray: The blended array.

    Raises:
        ValueError: If the input arrays are of different sizes.
        TypeError: If the input arrays are not of type Vt.Vec3hArray.
        ValueError: If the blending factor t is outside the range [0, 1].
    """
    if not isinstance(a, Vt.Vec3hArray) or not isinstance(b, Vt.Vec3hArray):
        raise TypeError("Inputs must be of type Vt.Vec3hArray")
    if len(a) != len(b):
        raise ValueError("Input arrays must be of the same size")
    if t < 0 or t > 1:
        raise ValueError("Blending factor t must be in the range [0, 1]")
    result = Vt.Vec3hArray(len(a))
    for i in range(len(result)):
        result[i] = Gf.Vec3h(Gf.Lerp(a[i][0], b[i][0], t), Gf.Lerp(a[i][1], b[i][1], t), Gf.Lerp(a[i][2], b[i][2], t))
    return result


def apply_transformations_to_vec3h_array(
    vec3h_array: Vt.Vec3hArray, rotation: Gf.Rotation, scale: Gf.Vec3h
) -> Vt.Vec3hArray:
    """Apply rotation and scale transformations to a Vec3hArray.

    Args:
        vec3h_array (Vt.Vec3hArray): The input array of Vec3h values.
        rotation (Gf.Rotation): The rotation to apply to each vector.
        scale (Gf.Vec3h): The scale to apply to each vector.

    Returns:
        Vt.Vec3hArray: The transformed Vec3hArray.
    """
    if not isinstance(vec3h_array, Vt.Vec3hArray):
        raise TypeError("Input must be a Vt.Vec3hArray")
    transformed_array = Vt.Vec3hArray(len(vec3h_array))
    for i in range(len(vec3h_array)):
        vec = Gf.Vec3f(vec3h_array[i])
        vec = rotation.TransformDir(vec)
        vec = Gf.Vec3h(vec * Gf.Vec3f(scale))
        transformed_array[i] = vec
    return transformed_array


def set_vec3h_array_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> Vt.Vec3hArray:
    """
    Set a Vec3hArray from a list of prim paths.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        Vt.Vec3hArray: An array of Vec3h values from the prims.

    Raises:
        ValueError: If any prim is invalid or not a UsdGeom.Xformable.
    """
    vec3h_array = Vt.Vec3hArray(len(prim_paths))
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            raise ValueError(f"Prim at path {prim_path} is not transformable.")
        local_matrix = xformable.GetLocalTransformation(Usd.TimeCode.Default())
        translate = local_matrix.ExtractTranslation()
        vec3h = Gf.Vec3h(translate)
        vec3h_array[i] = vec3h
    return vec3h_array


def extract_bounding_box_from_vec3h_array(points: Vt.Vec3hArray) -> Gf.Range3f:
    """Extract the bounding box from a Vec3hArray.

    Args:
        points (Vt.Vec3hArray): The input array of points.

    Returns:
        Gf.Range3f: The bounding box of the points.
    """
    if not points:
        raise ValueError("Input Vec3hArray is empty.")
    points_float = Vt.Vec3fArray([Gf.Vec3f(p) for p in points])
    bbox = Gf.Range3f(points_float[0], points_float[0])
    for point in points_float[1:]:
        bbox.UnionWith(point)
    return bbox


def animate_vec3h_array(vec3h_array: Vt.Vec3hArray, keyframes: List[Gf.Vec3h]) -> Vt.Vec3hArray:
    """Animate a Vec3hArray by interpolating between keyframes."""
    if not isinstance(vec3h_array, Vt.Vec3hArray):
        raise TypeError(f"Expected Vt.Vec3hArray, got {type(vec3h_array)}")
    if not keyframes:
        raise ValueError("Keyframes list is empty")
    num_frames = len(vec3h_array)
    num_keyframes = len(keyframes)
    interval = num_frames // (num_keyframes - 1)
    for i in range(num_frames):
        keyframe_index = i // interval
        next_keyframe_index = min(keyframe_index + 1, num_keyframes - 1)
        t = i % interval / interval
        start_vec = Gf.Vec3h(keyframes[keyframe_index])
        end_vec = Gf.Vec3h(keyframes[next_keyframe_index])
        interpolated_vec = start_vec + (end_vec - start_vec) * t
        vec3h_array[i] = interpolated_vec
    return vec3h_array


def get_average_vertex_position(vertices: Vt.Vec3iArray) -> Gf.Vec3i:
    """
    Calculate the average vertex position from an array of vertices.

    Args:
        vertices (Vt.Vec3iArray): An array of vertices.

    Returns:
        Gf.Vec3i: The average vertex position.
    """
    if not vertices:
        raise ValueError("Vertices array is empty.")
    sum_position = Gf.Vec3i(0, 0, 0)
    for vertex in vertices:
        sum_position += vertex
    num_vertices = len(vertices)
    average_position = Gf.Vec3i(
        sum_position[0] // num_vertices, sum_position[1] // num_vertices, sum_position[2] // num_vertices
    )
    return average_position


def transform_prim_vertices(stage: Usd.Stage, prim_path: str, matrix: Gf.Matrix4d) -> None:
    """Transform the vertices of a prim using a transformation matrix.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        matrix (Gf.Matrix4d): The transformation matrix.

    Raises:
        ValueError: If the prim is not a mesh or if it has no points attribute.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"No prim found at path: {prim_path}")
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim at path {prim_path} is not a mesh")
    mesh = UsdGeom.Mesh(prim)
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        raise ValueError(f"Mesh at path {prim_path} has no points attribute")
    points = points_attr.Get()
    transformed_points = Vt.Vec3fArray(len(points))
    for i, point in enumerate(points):
        vec3f = Gf.Vec3f(point[0], point[1], point[2])
        transformed_vec3f = matrix.Transform(vec3f)
        transformed_points[i] = transformed_vec3f
    points_attr.Set(transformed_points)


def filter_visible_vertices(vertices: Vt.Vec3iArray, visibility: Vt.IntArray) -> Vt.Vec3iArray:
    """Filter visible vertices based on the visibility array.

    Args:
        vertices (Vt.Vec3iArray): Array of vertex indices.
        visibility (Vt.IntArray): Array of visibility flags (0 or 1) for each vertex.

    Returns:
        Vt.Vec3iArray: Filtered array of visible vertex indices.
    """
    if len(vertices) != len(visibility):
        raise ValueError("Length of vertices and visibility arrays must be the same.")
    filtered_vertices = []
    for vertex, vis in zip(vertices, visibility):
        if vis == 1:
            filtered_vertices.append(vertex)
    return Vt.Vec3iArray(filtered_vertices)


def set_vertex_colors(prim: Usd.Prim, colors: Vt.Vec3fArray) -> None:
    """Set vertex colors for a UsdGeomMesh prim.

    Args:
        prim (Usd.Prim): The UsdGeomMesh prim to set vertex colors for.
        colors (Vt.Vec3fArray): The array of vertex colors to set.

    Raises:
        ValueError: If the prim is not a valid UsdGeomMesh prim or if the length of colors does not match the number of points.
    """
    if not prim.IsA(UsdGeom.Mesh):
        raise ValueError(f"Prim {prim.GetPath()} is not a UsdGeomMesh")
    mesh = UsdGeom.Mesh(prim)
    num_points = len(mesh.GetPointsAttr().Get())
    if len(colors) != num_points:
        raise ValueError(f"Length of colors ({len(colors)}) does not match the number of points ({num_points})")
    primvar = mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    primvar.Set(colors)


def interpolate_vec4darray(src_array: Vt.Vec4dArray, dest_array: Vt.Vec4dArray, weight: float) -> Vt.Vec4dArray:
    """Linearly interpolate between two Vec4dArray arrays.

    Args:
        src_array (Vt.Vec4dArray): The source array.
        dest_array (Vt.Vec4dArray): The destination array.
        weight (float): The interpolation weight between 0 and 1.

    Returns:
        Vt.Vec4dArray: The interpolated array.

    Raises:
        ValueError: If the input arrays have different lengths or if the weight is outside the valid range.
    """
    if len(src_array) != len(dest_array):
        raise ValueError("Input arrays must have the same length.")
    if weight < 0 or weight > 1:
        raise ValueError("Interpolation weight must be between 0 and 1.")
    interpolated_array = Vt.Vec4dArray(len(src_array))
    for i in range(len(src_array)):
        interpolated_array[i] = Gf.Lerp(weight, src_array[i], dest_array[i])
    return interpolated_array


def filter_vec4darray_by_value(vec4d_array: Vt.Vec4dArray, value: float, epsilon: float = 1e-06) -> Vt.Vec4dArray:
    """
    Filter a Vec4dArray by a specific value, keeping only elements where any component is within epsilon of the value.

    Args:
        vec4d_array (Vt.Vec4dArray): The input array to filter.
        value (float): The value to filter by.
        epsilon (float, optional): The tolerance for considering a component equal to the value. Defaults to 1e-6.

    Returns:
        Vt.Vec4dArray: A new Vec4dArray containing only the filtered elements.
    """
    filtered_elements: List[Gf.Vec4d] = []
    for vec4d in vec4d_array:
        if any((abs(component - value) < epsilon for component in vec4d)):
            filtered_elements.append(vec4d)
    return Vt.Vec4dArray(filtered_elements)


def copy_vec4darray(src: Vt.Vec4dArray) -> Vt.Vec4dArray:
    """Create a copy of a Vec4dArray."""
    if not isinstance(src, Vt.Vec4dArray):
        raise TypeError(f"Expected Vt.Vec4dArray, got {type(src)}")
    dst = Vt.Vec4dArray(len(src))
    for i in range(len(src)):
        dst[i] = src[i]
    return dst


def transform_vec4darray(vec4darray: Vt.Vec4dArray, matrix: Gf.Matrix4d) -> Vt.Vec4dArray:
    """Transform a Vec4dArray by a matrix.

    Args:
        vec4darray (Vt.Vec4dArray): The input array of Vec4d vectors.
        matrix (Gf.Matrix4d): The transformation matrix.

    Returns:
        Vt.Vec4dArray: The transformed array of Vec4d vectors.
    """
    if not isinstance(vec4darray, Vt.Vec4dArray):
        raise TypeError("Input must be a Vt.Vec4dArray")
    if not isinstance(matrix, Gf.Matrix4d):
        raise TypeError("Transformation matrix must be a Gf.Matrix4d")
    transformed_array = Vt.Vec4dArray(len(vec4darray))
    for i, vec in enumerate(vec4darray):
        gf_vec = Gf.Vec4d(vec)
        vec3d = Gf.Vec3d(gf_vec[0], gf_vec[1], gf_vec[2])
        transformed_vec3d = matrix.Transform(vec3d)
        transformed_vec4d = Gf.Vec4d(transformed_vec3d[0], transformed_vec3d[1], transformed_vec3d[2], gf_vec[3])
        transformed_array[i] = transformed_vec4d
    return transformed_array


def merge_vec4darrays(arrays: List[Vt.Vec4dArray]) -> Vt.Vec4dArray:
    """Merge multiple Vec4dArray instances into a single Vec4dArray.

    Args:
        arrays (List[Vt.Vec4dArray]): A list of Vec4dArray instances to merge.

    Returns:
        Vt.Vec4dArray: The merged Vec4dArray containing all elements from the input arrays.

    Raises:
        ValueError: If the input list is empty.
    """
    if not arrays:
        raise ValueError("Input list of arrays cannot be empty.")
    merged_elements = []
    for array in arrays:
        merged_elements.extend(array)
    merged_array = Vt.Vec4dArray(merged_elements)
    return merged_array


def interpolate_vec4f_arrays(a: Vt.Vec4fArray, b: Vt.Vec4fArray, t: float) -> Vt.Vec4fArray:
    """Linearly interpolate between two Vec4fArray arrays.

    Args:
        a (Vt.Vec4fArray): The first array.
        b (Vt.Vec4fArray): The second array.
        t (float): The interpolation factor between 0 and 1.

    Returns:
        Vt.Vec4fArray: The interpolated array.

    Raises:
        ValueError: If the input arrays have different lengths or t is outside [0, 1].
    """
    if len(a) != len(b):
        raise ValueError("Input arrays must have the same length.")
    if t < 0 or t > 1:
        raise ValueError("Interpolation factor t must be between 0 and 1.")
    result = Vt.Vec4fArray(len(a))
    for i in range(len(a)):
        result[i] = Gf.Lerp(t, a[i], b[i])
    return result


def merge_vec4f_arrays(arrays: Sequence[Vt.Vec4fArray]) -> Vt.Vec4fArray:
    """Merge multiple Vec4fArray instances into a single Vec4fArray.

    Args:
        arrays (Sequence[Vt.Vec4fArray]): A sequence of Vec4fArray instances to merge.

    Returns:
        Vt.Vec4fArray: A new Vec4fArray containing the concatenated elements from all input arrays.
    """
    if not arrays:
        return Vt.Vec4fArray()
    total_size = sum((len(arr) for arr in arrays))
    result = Vt.Vec4fArray(total_size)
    offset = 0
    for arr in arrays:
        for i in range(len(arr)):
            result[offset + i] = arr[i]
        offset += len(arr)
    return result


def transform_vec4f_array(vec4f_array: Vt.Vec4fArray, matrix: Gf.Matrix4d) -> Vt.Vec4fArray:
    """
    Transform an array of Vec4f using a 4x4 matrix.

    Args:
        vec4f_array (Vt.Vec4fArray): The input array of Vec4f to transform.
        matrix (Gf.Matrix4d): The 4x4 transformation matrix.

    Returns:
        Vt.Vec4fArray: The transformed array of Vec4f.
    """
    if not vec4f_array:
        return vec4f_array
    transformed_array = Vt.Vec4fArray(len(vec4f_array))
    for i, vec4f in enumerate(vec4f_array):
        vec3f = Gf.Vec3f(vec4f[0], vec4f[1], vec4f[2])
        transformed_vec3f = matrix.Transform(vec3f)
        transformed_vec4f = Gf.Vec4f(transformed_vec3f[0], transformed_vec3f[1], transformed_vec3f[2], 1.0)
        transformed_array[i] = transformed_vec4f
    return transformed_array


def filter_vec4f_array(vec4f_array: Vt.Vec4fArray, filter_func: Callable[[Gf.Vec4f], bool]) -> Vt.Vec4fArray:
    """Filter a Vec4fArray based on a provided filter function.

    Args:
        vec4f_array (Vt.Vec4fArray): The input Vec4fArray to be filtered.
        filter_func (Callable[[Gf.Vec4f], bool]): The filter function that takes a Gf.Vec4f and returns a boolean.

    Returns:
        Vt.Vec4fArray: A new Vec4fArray containing only the elements that satisfy the filter function.
    """
    filtered_list = []
    for vec4f in vec4f_array:
        if filter_func(vec4f):
            filtered_list.append(vec4f)
    return Vt.Vec4fArray(filtered_list)


def sum_greater_than_20(vec4f: Gf.Vec4f) -> bool:
    return sum(vec4f) > 20


def compute_bounding_box_from_vec4f_array(points: Vt.Vec4fArray) -> Gf.Range3f:
    """Compute the bounding box from an array of Vec4f points.

    Args:
        points (Vt.Vec4fArray): An array of Vec4f points.

    Returns:
        Gf.Range3f: The computed bounding box.

    Raises:
        ValueError: If the input array is empty.
    """
    if not points:
        raise ValueError("Input array is empty.")
    bbox = Gf.Range3f(
        Gf.Vec3f(points[0][0], points[0][1], points[0][2]), Gf.Vec3f(points[0][0], points[0][1], points[0][2])
    )
    for point in points[1:]:
        bbox.UnionWith(Gf.Vec3f(point[0], point[1], point[2]))
    return bbox


def extract_colors_from_prims(stage: Usd.Stage, prim_paths: List[str]) -> List[Tuple[str, Vt.Vec3fArray]]:
    """Extract color information from a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths.

    Returns:
        List[Tuple[str, Vt.Vec3fArray]]: A list of tuples containing the prim path and its color array.
    """
    results = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        color_primvar = mesh.GetDisplayColorPrimvar()
        if not color_primvar or not color_primvar.HasAuthoredValue():
            continue
        color_array = color_primvar.Get()
        if not isinstance(color_array, Vt.Vec3fArray):
            continue
        results.append((prim_path, color_array))
    return results


def interpolate_colors_across_keyframes(colors: Vt.Vec4hArray, num_frames: int) -> Vt.Vec4hArray:
    """
    Interpolate colors across keyframes.

    Args:
        colors (Vt.Vec4hArray): Array of colors at keyframes.
        num_frames (int): Total number of frames to interpolate.

    Returns:
        Vt.Vec4hArray: Interpolated colors for each frame.
    """
    num_keyframes = len(colors)
    if num_keyframes < 2:
        raise ValueError("At least two keyframe colors are required for interpolation.")
    if num_frames < num_keyframes:
        raise ValueError("Number of frames must be greater than or equal to the number of keyframes.")
    interpolated_colors = Vt.Vec4hArray(num_frames)
    keyframe_interval = (num_frames - 1) / (num_keyframes - 1)
    for frame in range(num_frames):
        keyframe_index = int(frame / keyframe_interval)
        next_keyframe_index = min(keyframe_index + 1, num_keyframes - 1)
        t = (frame - keyframe_index * keyframe_interval) / keyframe_interval
        start_color = colors[keyframe_index]
        end_color = colors[next_keyframe_index]
        interpolated_color = start_color * (1 - t) + end_color * t
        interpolated_colors[frame] = interpolated_color
    return interpolated_colors


def apply_color_gradient_to_prims(stage: Usd.Stage, prim_paths: List[str], colors: Vt.Vec4hArray) -> None:
    """Apply a gradient of colors to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): A list of prim paths to apply colors to.
        colors (Vt.Vec4hArray): An array of colors to apply as a gradient.

    Raises:
        ValueError: If the number of prim paths and colors don't match.
    """
    num_prims = len(prim_paths)
    num_colors = len(colors)
    if num_prims != num_colors:
        raise ValueError(f"Number of prim paths ({num_prims}) must match number of colors ({num_colors}).")
    for i in range(num_prims):
        prim = stage.GetPrimAtPath(prim_paths[i])
        if not prim.IsValid():
            continue
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        gprim.GetDisplayColorAttr().Set([Gf.Vec3h(colors[i][0], colors[i][1], colors[i][2])])
        gprim.GetDisplayOpacityAttr().Set([float(colors[i][3])])


def blend_prim_colors(stage: Usd.Stage, prim_paths: List[str], blend_amounts: Vt.Vec4hArray) -> Vt.Vec4hArray:
    """Blend the display colors of a list of prims using the provided blend amounts.

    Args:
        stage (Usd.Stage): The USD stage containing the prims.
        prim_paths (List[str]): A list of paths to the prims to blend colors for.
        blend_amounts (Vt.Vec4hArray): An array of Vec4h values representing the blend amounts for each prim.

    Returns:
        Vt.Vec4hArray: The blended color for each prim.

    Raises:
        ValueError: If the number of prim paths and blend amounts do not match.
    """
    if len(prim_paths) != len(blend_amounts):
        raise ValueError("Number of prim paths and blend amounts must match.")
    blended_colors = Vt.Vec4hArray(len(prim_paths))
    for i, prim_path in enumerate(prim_paths):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} does not exist.")
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            raise ValueError(f"Prim at path {prim_path} is not a Gprim.")
        display_color = gprim.GetDisplayColorAttr().Get(Usd.TimeCode.Default())
        if not display_color:
            display_color = Vt.Vec3fArray(1, Gf.Vec3f(0.5, 0.5, 0.5))
        blended_color = Gf.Vec4h(
            display_color[0][0] * blend_amounts[i][0],
            display_color[0][1] * blend_amounts[i][1],
            display_color[0][2] * blend_amounts[i][2],
            blend_amounts[i][3],
        )
        blended_colors[i] = blended_color
    return blended_colors


def batch_update_prim_colors(stage: Usd.Stage, prim_paths: List[str], colors: Vt.Vec4hArray) -> None:
    """Batch update the display color for multiple prims.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_paths (List[str]): List of prim paths to update colors for.
        colors (Vt.Vec4hArray): Array of colors to assign to the prims.

    Raises:
        ValueError: If the number of prim paths and colors don't match.
    """
    if len(prim_paths) != len(colors):
        raise ValueError("Number of prim paths and colors must match.")
    for prim_path, color in zip(prim_paths, colors):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            continue
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        gprim.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(color[0], color[1], color[2])]))
        gprim.GetDisplayOpacityAttr().Set(Vt.FloatArray([color[3]]))


def get_prim_vec4iarray_attribute(prim: Usd.Prim, attribute_name: str) -> Vt.Vec4iArray:
    """Get a Vec4iArray attribute from a prim.

    Args:
        prim (Usd.Prim): The prim to retrieve the attribute from.
        attribute_name (str): The name of the Vec4iArray attribute.

    Returns:
        Vt.Vec4iArray: The value of the Vec4iArray attribute.

    Raises:
        ValueError: If the prim is not valid or the attribute does not exist.
    """
    if not prim.IsValid():
        raise ValueError(f"Prim '{prim.GetPath()}' is not valid.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute.IsValid():
        raise ValueError(f"Attribute '{attribute_name}' does not exist on prim '{prim.GetPath()}'.")
    if attribute.GetTypeName() != Sdf.ValueTypeNames.Int4Array:
        raise ValueError(f"Attribute '{attribute_name}' is not of type Vec4iArray.")
    return attribute.Get(Usd.TimeCode.Default()) or Vt.Vec4iArray()


def find_prims_with_vec4iarray_attribute(stage: Usd.Stage, attribute_name: str) -> List[Usd.Prim]:
    """
    Find all prims on the given stage that have an attribute of type Vec4iArray.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        attribute_name (str): The name of the Vec4iArray attribute to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have the specified Vec4iArray attribute.
    """
    prims_with_attribute = []
    for prim in stage.TraverseAll():
        if prim.HasAttribute(attribute_name):
            attribute = prim.GetAttribute(attribute_name)
            if attribute.GetTypeName() == Sdf.ValueTypeNames.Int4Array:
                prims_with_attribute.append(prim)
    return prims_with_attribute


def set_prim_vec4iarray_attribute(prim: Usd.Prim, attribute_name: str, value: Vt.Vec4iArray) -> None:
    """Sets a Vec4iArray attribute on a given prim.

    Args:
        prim (Usd.Prim): The prim to set the attribute on.
        attribute_name (str): The name of the attribute to set.
        value (Vt.Vec4iArray): The value to set for the attribute.

    Raises:
        ValueError: If the prim is not valid.
        RuntimeError: If the attribute cannot be created or set.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute:
        attribute = prim.CreateAttribute(attribute_name, Sdf.ValueTypeNames.Int4Array)
    try:
        attribute.Set(value)
    except Exception as e:
        raise RuntimeError(f"Failed to set attribute '{attribute_name}' on prim '{prim.GetPath()}': {str(e)}") from e


def transform_vec4iarray_attribute(
    stage: Usd.Stage, prim_path: str, attribute_name: str, transform_matrix: Gf.Matrix4d
) -> None:
    """Transform a Vec4iArray attribute on a prim using a matrix.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim.
        attribute_name (str): The name of the Vec4iArray attribute.
        transform_matrix (Gf.Matrix4d): The transformation matrix.

    Raises:
        ValueError: If the prim or attribute does not exist.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Prim '{prim_path}' does not exist.")
    attribute = prim.GetAttribute(attribute_name)
    if not attribute:
        raise ValueError(f"Attribute '{attribute_name}' does not exist on prim '{prim_path}'.")
    vec4iarray = attribute.Get()
    if not vec4iarray:
        raise ValueError(f"Attribute '{attribute_name}' on prim '{prim_path}' has no value.")
    transformed_vec4iarray = Vt.Vec4iArray(
        [
            Gf.Vec4i(int(v[0]), int(v[1]), int(v[2]), 1)
            for v in [transform_matrix.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in vec4iarray]
        ]
    )
    attribute.Set(transformed_vec4iarray)


def merge_vec4iarray_attributes(attributes: List[Vt.Vec4iArray]) -> Vt.Vec4iArray:
    """Merges multiple Vec4iArray attributes into a single Vec4iArray.

    Args:
        attributes (List[Vt.Vec4iArray]): A list of Vec4iArray attributes to merge.

    Returns:
        Vt.Vec4iArray: The merged Vec4iArray attribute.

    Raises:
        ValueError: If the input list is empty.
    """
    if not attributes:
        raise ValueError("Input list of attributes cannot be empty.")
    merged_values = []
    for attribute in attributes:
        merged_values.extend(attribute)
    merged_attribute = Vt.Vec4iArray(merged_values)
    return merged_attribute
