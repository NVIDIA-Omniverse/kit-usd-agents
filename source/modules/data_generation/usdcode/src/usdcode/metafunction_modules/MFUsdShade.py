## Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
##
## NVIDIA CORPORATION and its licensors retain all intellectual property
## and proprietary rights in and to this software, related documentation
## and any modifications thereto.  Any use, reproduction, disclosure or
## distribution of this software and related documentation without an express
## license agreement from NVIDIA CORPORATION is strictly prohibited.
##

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pxr import Gf, Ndr, Sdf, Tf, Usd, UsdGeom, UsdShade


def validate_shade_attributes(prim: Usd.Prim, attributes: Dict[str, UsdShade.AttributeType]) -> List[str]:
    """Validate if the given prim has the specified shade attributes.

    Args:
        prim (Usd.Prim): The prim to validate the shade attributes on.
        attributes (Dict[str, UsdShade.AttributeType]): A dictionary mapping attribute names to their expected types.

    Returns:
        List[str]: A list of error messages for any missing or invalid shade attributes.
    """
    errors = []
    if not prim.IsValid():
        errors.append(f"Prim '{prim.GetPath()}' is not valid.")
        return errors
    if not UsdShade.Shader(prim):
        errors.append(f"Prim '{prim.GetPath()}' does not have a UsdShade schema.")
        return errors
    for attr_name, attr_type in attributes.items():
        if not prim.HasAttribute(attr_name):
            errors.append(f"Attribute '{attr_name}' is missing on prim '{prim.GetPath()}'.")
            continue
        attr = prim.GetAttribute(attr_name)
        if attr.GetTypeName() != UsdShade.AttributeType(attr_type).name:
            errors.append(
                f"Attribute '{attr_name}' on prim '{prim.GetPath()}' has the wrong type. Expected '{UsdShade.AttributeType(attr_type).name}', got '{attr.GetTypeName()}'."
            )
    return errors


def convert_attribute_type(attribute_type: UsdShade.AttributeType) -> str:
    """Convert UsdShade.AttributeType enum to a human-readable string."""
    if attribute_type == UsdShade.AttributeType.Input:
        return "input"
    elif attribute_type == UsdShade.AttributeType.Output:
        return "output"
    else:
        return "unknown"


def find_prims_by_attribute_type(stage: Usd.Stage, attribute_type: str) -> List[Usd.Prim]:
    """Find all prims on the stage that have an attribute of the specified type.

    Args:
        stage (Usd.Stage): The USD stage to search.
        attribute_type (str): The attribute type to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have an attribute of the specified type.
    """
    prims: List[Usd.Prim] = []
    for prim in stage.TraverseAll():
        attributes = prim.GetAttributes()
        for attr in attributes:
            attr_type_name = attr.GetTypeName()
            if attr_type_name == attribute_type:
                prims.append(prim)
                break
    return prims


def get_all_shade_attributes(prim: Usd.Prim) -> List[UsdShade.Input]:
    """Get all UsdShade attributes on a prim.

    Args:
        prim (Usd.Prim): The prim to query for shade attributes.

    Returns:
        List[UsdShade.Input]: A list of all UsdShade attributes on the prim.
    """
    shade_attributes: List[UsdShade.Input] = []
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    shader = UsdShade.Shader(prim)
    if not shader:
        return shade_attributes
    for input_attr in shader.GetInputs():
        shade_attributes.append(input_attr)
    for output_attr in shader.GetOutputs():
        shade_attributes.append(output_attr)
    return shade_attributes


def batch_update_attributes(stage: Usd.Stage, attributes: Dict[str, List[Dict]]) -> None:
    """Update multiple attributes on multiple prims in a single transaction.

    Args:
        stage (Usd.Stage): The stage to update attributes on.
        attributes (Dict[str, List[Dict]]): A dictionary mapping prim paths to a list of attribute updates.
            Each attribute update is a dictionary with keys "name", "type", and "value".

    Raises:
        ValueError: If an invalid attribute type is provided.
    """
    with Usd.EditContext(stage, stage.GetEditTarget()):
        for prim_path, attr_updates in attributes.items():
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
            for attr_update in attr_updates:
                attr_name = attr_update["name"]
                attr_type = attr_update["type"]
                attr_value = attr_update["value"]
                if attr_type == UsdShade.AttributeType.Input:
                    attribute = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Float)
                elif attr_type == UsdShade.AttributeType.Output:
                    attribute = prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.Color3f)
                else:
                    raise ValueError(f"Invalid attribute type: {attr_type}")
                attribute.Set(attr_value)


def create_shader_output(shader: UsdShade.Shader, output_name: str, type_name: str) -> UsdShade.Output:
    """Create a shader output attribute with the given name and type."""
    if not shader:
        raise ValueError("Invalid shader prim")
    existing_output = shader.GetOutput(output_name)
    if existing_output:
        return existing_output
    output = shader.CreateOutput(output_name, Sdf.ValueTypeNames.Find(type_name))
    if not output:
        raise RuntimeError(f"Failed to create output '{output_name}' of type '{type_name}'")
    return output


def get_shader_input_value(input: UsdShade.Input) -> Tuple[bool, Any]:
    """
    Get the value of a shader input.

    Args:
        input (UsdShade.Input): The shader input to get the value from.

    Returns:
        Tuple[bool, Any]: A tuple containing a boolean indicating if the value was
        successfully retrieved and the value itself. If the boolean is False,
        the value will be None.
    """
    if not input:
        return (False, None)
    attr = input.GetAttr()
    if not attr:
        return (False, None)
    type_name = attr.GetTypeName()
    try:
        value = attr.Get(Usd.TimeCode.Default())
    except:
        return (False, None)
    if type_name == Sdf.ValueTypeNames.Float:
        return (True, value)
    elif type_name == Sdf.ValueTypeNames.String:
        return (True, value)
    elif type_name == Sdf.ValueTypeNames.Int:
        return (True, value)
    elif type_name == Sdf.ValueTypeNames.Bool:
        return (True, value)
    elif type_name == Sdf.ValueTypeNames.Vector3f:
        return (True, Gf.Vec3f(value))
    elif type_name == Sdf.ValueTypeNames.Color3f:
        return (True, Gf.Vec3f(value))
    else:
        return (False, None)


def get_all_shader_inputs(shader: UsdShade.Shader) -> Dict[str, UsdShade.Input]:
    """
    Get all inputs for a given shader, including authored and un-authored builtin inputs.

    Args:
        shader (UsdShade.Shader): The shader to get inputs for.

    Returns:
        Dict[str, UsdShade.Input]: A dictionary mapping input names to UsdShade.Input objects.
    """
    authored_inputs = shader.GetInputs()
    builtin_inputs = shader.GetInputs(onlyAuthored=False)
    all_inputs = {}
    for input in authored_inputs + builtin_inputs:
        input_name = input.GetBaseName()
        all_inputs[input_name] = input
    return all_inputs


def get_all_shader_outputs(shader: UsdShade.Shader) -> List[Tuple[str, Sdf.ValueTypeName]]:
    """
    Get all outputs of a shader.

    Args:
        shader (UsdShade.Shader): The shader to get outputs from.

    Returns:
        List[Tuple[str, Sdf.ValueTypeName]]: A list of tuples containing the
        name and value type of each output.
    """
    output_attrs = shader.GetOutputs()
    outputs = []
    for output_attr in output_attrs:
        name = output_attr.GetBaseName()
        typename = output_attr.GetTypeName()
        outputs.append((name, typename))
    return outputs


def clear_shader_connections(shader: UsdShade.Shader) -> bool:
    """Clear all shader input connections on the given shader.

    Args:
        shader (UsdShade.Shader): The shader to clear connections on.

    Returns:
        bool: True if successful, False otherwise.
    """
    shader_inputs = shader.GetInputs()
    for input in shader_inputs:
        if input.HasConnectedSource():
            input_attr = input.GetAttr()
            success = UsdShade.ConnectableAPI.ClearSource(input_attr)
            if not success:
                return False
    return True


def create_material_with_shaders(stage: Usd.Stage, material_path: str, shader_paths: List[str]) -> UsdShade.Material:
    """Create a material with the given shader prims as inputs.

    Args:
        stage (Usd.Stage): The USD stage to create the material on.
        material_path (str): The path where the material should be created.
        shader_paths (List[str]): A list of paths to shader prims to use as inputs for the material.

    Returns:
        UsdShade.Material: The created material prim.

    Raises:
        ValueError: If the material path is invalid or any of the shader paths are invalid.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if material_prim.IsValid():
        raise ValueError(f"A prim already exists at path {material_path}")
    material = UsdShade.Material.Define(stage, material_path)
    for shader_path in shader_paths:
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            raise ValueError(f"Invalid shader prim path: {shader_path}")
        shader = UsdShade.Shader(shader_prim)
        output_attr = shader.GetOutput(UsdShade.Tokens.surface)
        material_input = material.CreateInput(UsdShade.Tokens.surface, Sdf.ValueTypeNames.Token)
        material_input.ConnectToSource(output_attr)
    return material


def connect_material_shaders(material: UsdShade.Material, shader_prims: List[UsdShade.Shader]) -> bool:
    """Connects a list of shader prims to a material.

    Args:
        material (UsdShade.Material): The material prim to connect shaders to.
        shader_prims (List[UsdShade.Shader]): A list of shader prims to connect.

    Returns:
        bool: True if all connections were successful, False otherwise.
    """
    if not material or not material.GetPrim().IsA(UsdShade.Material):
        return False
    connectable_material = UsdShade.ConnectableAPI(material)
    all_connected = True
    for shader_prim in shader_prims:
        if not shader_prim or not shader_prim.GetPrim().IsA(UsdShade.Shader):
            all_connected = False
            continue
        connectable_shader = UsdShade.ConnectableAPI(shader_prim)
        shader_surface_output = connectable_shader.GetOutput("surface")
        if not shader_surface_output:
            all_connected = False
            continue
        material_surface_input = connectable_material.GetInput("surface")
        if not material_surface_input:
            all_connected = False
            continue
        connect_result = material_surface_input.ConnectToSource(shader_surface_output)
        if not connect_result:
            all_connected = False
    return all_connected


def disconnect_material_shaders(material: UsdShade.Material) -> List[UsdShade.Shader]:
    """Disconnects all shader prims connected to the given material.

    Returns a list of the disconnected shader prims.
    """
    disconnected_shaders = []
    for input in material.GetInputs():
        if input.HasConnectedSource():
            shader_source = input.GetConnectedSource()[0]
            input.DisconnectSource()
            disconnected_shaders.append(shader_source)
    return disconnected_shaders


def clear_material_connections(material: UsdShade.Material) -> bool:
    """Clear all the connections on the inputs of a material."""
    inputs = material.GetInputs()
    all_cleared = True
    for input in inputs:
        if input.HasConnectedSource():
            success = input.ClearSource()
            if not success:
                all_cleared = False
    return all_cleared


def copy_shader_parameters(source_shader: UsdShade.Shader, dest_shader: UsdShade.Shader) -> None:
    """Copy shader parameters from one shader to another."""
    if not source_shader:
        raise ValueError("Invalid source shader")
    if not dest_shader:
        raise ValueError("Invalid destination shader")
    source_inputs = source_shader.GetInputs()
    for source_input in source_inputs:
        input_name = source_input.GetBaseName()
        source_attr = source_input.GetAttr()
        if source_attr.IsValid() and source_attr.HasValue():
            value = source_attr.Get()
            dest_input = dest_shader.CreateInput(input_name, source_attr.GetTypeName())
            dest_attr = dest_input.GetAttr()
            dest_attr.Set(value)


def merge_materials(material1: UsdShade.Material, material2: UsdShade.Material) -> UsdShade.Material:
    """Merge two materials into a new material, connecting their outputs."""
    if not material1.GetPrim().IsValid() or not material2.GetPrim().IsValid():
        raise ValueError("Invalid input materials")
    stage = material1.GetPrim().GetStage()
    merged_material_path = material1.GetPath().GetParentPath().AppendChild("MergedMaterial")
    merged_material = UsdShade.Material.Define(stage, merged_material_path)
    surface_output1 = material1.GetOutput(UsdShade.Tokens.surface)
    surface_output2 = material2.GetOutput(UsdShade.Tokens.surface)
    merged_surface_output = merged_material.CreateOutput(UsdShade.Tokens.surface, Sdf.ValueTypeNames.Token)
    add_shader_path = merged_material_path.AppendChild("AddShader")
    add_shader = UsdShade.Shader.Define(stage, add_shader_path)
    add_shader.CreateInput("input1", Sdf.ValueTypeNames.Float3)
    add_shader.CreateInput("input2", Sdf.ValueTypeNames.Float3)
    add_shader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    add_shader.GetInput("input1").ConnectToSource(surface_output1)
    add_shader.GetInput("input2").ConnectToSource(surface_output2)
    merged_surface_output.ConnectToSource(add_shader.GetOutput("result"))
    return merged_material


def create_node_graph(stage: Usd.Stage, parent_path: str, name: str) -> UsdShade.NodeGraph:
    """Create a UsdShade.NodeGraph prim under the given parent path."""
    if not stage.GetPrimAtPath(parent_path):
        raise ValueError(f"Invalid parent path: {parent_path}")
    node_graph_path = f"{parent_path}/{name}"
    existing_prim = stage.GetPrimAtPath(node_graph_path)
    if existing_prim:
        if existing_prim.IsA(UsdShade.NodeGraph):
            return UsdShade.NodeGraph(existing_prim)
        else:
            raise ValueError(f"Prim at path {node_graph_path} is not a UsdShade.NodeGraph")
    node_graph_prim = UsdShade.NodeGraph.Define(stage, node_graph_path)
    return node_graph_prim


def get_node_graph_shaders(node_graph: UsdShade.NodeGraph) -> List[UsdShade.Shader]:
    """
    Get all shader prims directly under a node graph.

    Args:
        node_graph (UsdShade.NodeGraph): The node graph to query.

    Returns:
        List[UsdShade.Shader]: A list of shader prims.
    """
    if not node_graph or not isinstance(node_graph, UsdShade.NodeGraph):
        raise ValueError("Invalid node graph input.")
    node_graph_prim = node_graph.GetPrim()
    shader_prims = []
    for child_prim in node_graph_prim.GetChildren():
        if UsdShade.Shader.Get(child_prim.GetStage(), child_prim.GetPath()):
            shader_prims.append(child_prim)
    shaders = [UsdShade.Shader(prim) for prim in shader_prims]
    return shaders


def connect_node_graph_shaders(
    node_graph: UsdShade.NodeGraph,
    upstream_shader: UsdShade.Shader,
    downstream_shader: UsdShade.Shader,
    upstream_output: str,
    downstream_input: str,
) -> bool:
    """Connect two shaders within a node graph.

    Args:
        node_graph (UsdShade.NodeGraph): The node graph containing the shaders.
        upstream_shader (UsdShade.Shader): The shader providing the output.
        downstream_shader (UsdShade.Shader): The shader receiving the input.
        upstream_output (str): The name of the output on the upstream shader.
        downstream_input (str): The name of the input on the downstream shader.

    Returns:
        bool: True if the connection was made successfully, False otherwise.
    """
    if not node_graph:
        return False
    if not upstream_shader or upstream_shader.GetPrim().GetParent() != node_graph.GetPrim():
        return False
    if not downstream_shader or downstream_shader.GetPrim().GetParent() != node_graph.GetPrim():
        return False
    upstream_connectable = UsdShade.ConnectableAPI(upstream_shader)
    upstream_output_attr = upstream_connectable.GetOutput(upstream_output)
    if not upstream_output_attr:
        return False
    downstream_connectable = UsdShade.ConnectableAPI(downstream_shader)
    downstream_input_attr = downstream_connectable.GetInput(downstream_input)
    if not downstream_input_attr:
        return False
    success = downstream_input_attr.ConnectToSource(upstream_output_attr)
    return success


def find_shaders_by_type(stage: Usd.Stage, shader_type: str) -> List[UsdShade.Shader]:
    """
    Find all shaders of a specific type in a USD stage.

    Args:
        stage (Usd.Stage): The USD stage to search for shaders.
        shader_type (str): The type of shader to search for (e.g., "UsdPreviewSurface").

    Returns:
        List[UsdShade.Shader]: A list of shader prims of the specified type.
    """
    shaders = []
    for prim in stage.TraverseAll():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if shader.GetShaderId() == shader_type:
                shaders.append(shader)
    return shaders


def find_materials_with_shader(stage: Usd.Stage, shader_path: str) -> List[Sdf.Path]:
    """
    Find all materials that have a specific shader assigned.

    Args:
        stage (Usd.Stage): The USD stage to search.
        shader_path (str): The path to the shader to search for.

    Returns:
        List[Sdf.Path]: A list of material prim paths that have the specified shader assigned.
    """
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not shader_prim.IsValid():
        raise ValueError(f"Shader prim at path {shader_path} does not exist.")
    material_paths = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material = UsdShade.Material(prim)
            surface_output = material.GetSurfaceOutput()
            if not surface_output:
                continue
            shader_source = surface_output.GetConnectedSource()
            if not shader_source:
                continue
            if shader_source[0].GetPrim() == shader_prim:
                material_paths.append(prim.GetPath())
    return material_paths


def create_material_network(
    stage: Usd.Stage, material_path: str, diffuse_texture_path: str, roughness_texture_path: str
) -> UsdShade.Material:
    """Create a material network with textures for diffuse color and roughness.

    Args:
        stage (Usd.Stage): The USD stage to create the material network on.
        material_path (str): The path where the material will be created.
        diffuse_texture_path (str): The file path to the diffuse texture.
        roughness_texture_path (str): The file path to the roughness texture.

    Returns:
        UsdShade.Material: The created material prim.
    """
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    diffuse_texture = UsdShade.Shader.Define(stage, shader_path + "/DiffuseTexture")
    diffuse_texture.CreateIdAttr("UsdUVTexture")
    diffuse_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture_path)
    roughness_texture = UsdShade.Shader.Define(stage, shader_path + "/RoughnessTexture")
    roughness_texture.CreateIdAttr("UsdUVTexture")
    roughness_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(roughness_texture_path)
    diffuse_input = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    diffuse_input.ConnectToSource(diffuse_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
    roughness_input = shader.CreateInput("roughness", Sdf.ValueTypeNames.Float)
    roughness_input.ConnectToSource(roughness_texture.CreateOutput("r", Sdf.ValueTypeNames.Float))
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def batch_update_shader_parameters(shader: UsdShade.Shader, parameter_values: Dict[str, Any]) -> None:
    """
    Update multiple parameters on a shader at once.

    Args:
        shader (UsdShade.Shader): The shader to update parameters on.
        parameter_values (Dict[str, Any]): A dictionary mapping parameter names to their new values.

    Raises:
        ValueError: If the shader is invalid or any of the parameter names are invalid.
    """
    if not shader or not shader.GetPrim().IsValid():
        raise ValueError("Invalid shader provided.")
    for param_name, param_value in parameter_values.items():
        param_input = shader.GetInput(param_name)
        if not param_input:
            raise ValueError(f"Invalid parameter name: {param_name}")
        param_input.Set(param_value)


def find_shaders_with_parameters(stage: Usd.Stage, param_names: List[str]) -> List[UsdShade.Shader]:
    """
    Find all shader prims on the given stage that have all the specified parameter names.

    Args:
        stage (Usd.Stage): The USD stage to search for shaders.
        param_names (List[str]): The list of parameter names to match.

    Returns:
        List[UsdShade.Shader]: A list of shader prims that have all the specified parameters.
    """
    shaders = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            has_all_params = True
            for param_name in param_names:
                if not shader.GetInput(param_name):
                    has_all_params = False
                    break
            if has_all_params:
                shaders.append(shader)
    return shaders


def find_shaders_by_name(stage: Usd.Stage, name: str) -> List[UsdShade.Shader]:
    """Find all shader prims on the stage with the given name.

    Args:
        stage (Usd.Stage): The USD stage to search for shaders.
        name (str): The name of the shader to find.

    Returns:
        List[UsdShade.Shader]: A list of shader prims with the given name.
    """
    shaders: List[UsdShade.Shader] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if shader.GetPrim().GetName() == name:
                shaders.append(shader)
    return shaders


def get_shader_usage(shader_prim: UsdShade.Shader) -> UsdShade.Tokens:
    """Get the shader usage for a given shader prim.

    Args:
        shader_prim (UsdShade.Shader): The shader prim to get the usage for.

    Returns:
        UsdShade.Tokens: The shader usage token, or an empty token if not set.

    Raises:
        ValueError: If the input prim is not a valid shader prim.
    """
    if not shader_prim or not shader_prim.GetPrim().IsA(UsdShade.Shader):
        raise ValueError("Invalid shader prim")
    usage_attr = shader_prim.GetIdAttr()
    if usage_attr.HasAuthoredValue():
        usage = usage_attr.Get()
        return usage
    else:
        return UsdShade.Tokens.universalRenderContext


def connect_shaders(
    connectable: UsdShade.ConnectableAPI, source_shader: UsdShade.Shader, source_output: str, dest_input: str
) -> bool:
    """Connect the output of one shader to the input of another.

    Args:
        connectable (UsdShade.ConnectableAPI): The connectable prim.
        source_shader (UsdShade.Shader): The source shader prim.
        source_output (str): The name of the source shader's output.
        dest_input (str): The name of the destination shader's input.

    Returns:
        bool: True if the connection was made successfully, False otherwise.
    """
    if not connectable:
        return False
    dest_input_attr = connectable.GetInput(dest_input)
    if not dest_input_attr:
        dest_input_attr = connectable.CreateInput(dest_input, Sdf.ValueTypeNames.Float)
        if not dest_input_attr:
            return False
    source_output_attr = source_shader.GetOutput(source_output)
    if not source_output_attr:
        source_output_attr = source_shader.CreateOutput(source_output, Sdf.ValueTypeNames.Float)
        if not source_output_attr:
            return False
    success = dest_input_attr.ConnectToSource(source_output_attr)
    return success


def disconnect_shader_input(shader: UsdShade.Shader, input_name: str) -> bool:
    """Disconnect a shader input from its upstream connection.

    Args:
        shader (UsdShade.Shader): The shader prim.
        input_name (str): The name of the input to disconnect.

    Returns:
        bool: True if the input was successfully disconnected, False otherwise.
    """
    connectable_api = UsdShade.ConnectableAPI(shader)
    input_attr = connectable_api.GetInput(input_name)
    if not input_attr:
        return False
    if not input_attr.HasConnectedSource():
        return True
    success = connectable_api.DisconnectSource(input_attr)
    return success


def set_shader_input_value(input: UsdShade.Input, value: Any, time_code: Usd.TimeCode = Usd.TimeCode.Default()) -> None:
    """
    Set the value of a shader input.

    Args:
        input (UsdShade.Input): The shader input to set the value for.
        value (Any): The value to set on the shader input.
        time_code (Usd.TimeCode, optional): The time code to set the value at. Defaults to Default time code.

    Raises:
        ValueError: If the input is not a valid UsdShade.Input.
    """
    if not input:
        raise ValueError("Invalid shader input.")
    attr = input.GetAttr()
    if not attr:
        raise ValueError("Failed to get attribute for shader input.")
    success = attr.Set(value, time_code)
    if not success:
        raise ValueError("Failed to set value on shader input.")


def assign_material_to_prims(stage: Usd.Stage, material_path: str, prim_paths: List[str]) -> None:
    """Assign a material to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.
        prim_paths (List[str]): A list of paths to the prims to assign the material to.

    Raises:
        ValueError: If the material prim does not exist or is not a Material.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a Material.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            binding_api = UsdShade.MaterialBindingAPI(prim)
            binding_api.Bind(material)
        else:
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping material assignment.")


def clear_node_graph_connections(node_graph: UsdShade.NodeGraph) -> bool:
    """Clear all connections on a node graph."""
    for input in node_graph.GetInputs():
        if input.HasConnectedSource():
            success = input.ClearSource()
            if not success:
                return False
    for output in node_graph.GetOutputs():
        output_attr = output.GetAttr()
        connections = output_attr.GetConnections()
        if connections:
            for conn in connections:
                output_attr.RemoveConnection(conn)
    return True


def create_shader_network(
    stage: Usd.Stage, material_path: str, diffuse_texture_path: str, specular_texture_path: str
) -> UsdShade.Material:
    """Create a shader network with a material, shader, and textures.

    Args:
        stage (Usd.Stage): The USD stage to create the shader network on.
        material_path (str): The path to create the material at.
        diffuse_texture_path (str): The file path to the diffuse texture.
        specular_texture_path (str): The file path to the specular texture.

    Returns:
        UsdShade.Material: The created material prim.
    """
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    diffuse_texture_path = shader_path + "/DiffuseTexture"
    diffuse_texture = UsdShade.Shader.Define(stage, diffuse_texture_path)
    diffuse_texture.CreateIdAttr("UsdUVTexture")
    diffuse_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture_path)
    diffuse_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    specular_texture_path = shader_path + "/SpecularTexture"
    specular_texture = UsdShade.Shader.Define(stage, specular_texture_path)
    specular_texture.CreateIdAttr("UsdUVTexture")
    specular_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(specular_texture_path)
    specular_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    diffuse_input = shader.GetInput("diffuseColor")
    UsdShade.ConnectableAPI.ConnectToSource(diffuse_input, diffuse_texture.GetOutput("rgb"))
    specular_input = shader.GetInput("specularColor")
    UsdShade.ConnectableAPI.ConnectToSource(specular_input, specular_texture.GetOutput("rgb"))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def duplicate_node_graph(source_node_graph: UsdShade.NodeGraph, dest_node_graph: UsdShade.NodeGraph) -> None:
    """Duplicate a UsdShade.NodeGraph to another UsdShade.NodeGraph."""
    if not source_node_graph.GetPrim().IsValid():
        raise ValueError("Invalid source node graph.")
    if not dest_node_graph.GetPrim().IsValid():
        raise ValueError("Invalid destination node graph.")
    old_to_new_shader_map = {}
    for old_shader in source_node_graph.GetOutputs():
        shader_type = old_shader.GetPrim().GetTypeName()
        new_shader = UsdShade.Shader.Define(
            dest_node_graph.GetPrim().GetStage(), dest_node_graph.GetPath().AppendChild(old_shader.GetPrim().GetName())
        )
        new_shader.GetPrim().SetTypeName(shader_type)
        old_to_new_shader_map[old_shader.GetPrim().GetPath()] = new_shader
        for old_input in old_shader.GetPrim().GetInputs():
            new_input = new_shader.CreateInput(old_input.GetBaseName(), old_input.GetTypeName())
            new_input.Set(old_input.Get())
    for old_shader_prim in source_node_graph.GetOutputs():
        new_shader = old_to_new_shader_map[old_shader_prim.GetPrim().GetPath()]
        for old_input in old_shader_prim.GetPrim().GetInputs():
            if old_input.HasConnectedSource():
                (old_connected_shader, old_connected_output) = old_input.GetConnectedSource()
                new_connected_shader = old_to_new_shader_map[old_connected_shader.GetPrim().GetPath()]
                new_connected_output = new_connected_shader.GetOutput(old_connected_output.GetBaseName())
                new_shader.GetInput(old_input.GetBaseName()).ConnectToSource(new_connected_output)


def get_shader_parameters(shader: UsdShade.Shader) -> List[UsdShade.Input]:
    """
    Get all the parameters of a given shader.

    Args:
        shader (UsdShade.Shader): The shader to get parameters from.

    Returns:
        List[UsdShade.Input]: A list of shader parameters.
    """
    if not shader or not shader.GetPrim().IsA(UsdShade.Shader):
        raise ValueError("Invalid shader provided.")
    connectable_api = UsdShade.ConnectableAPI(shader)
    shader_inputs = connectable_api.GetInputs()
    shader_parameters = [input for input in shader_inputs if not input.HasConnectedSource()]
    return shader_parameters


def add_shader_to_node_graph(node_graph: UsdShade.NodeGraph, shader_name: str, shader_type: str) -> UsdShade.Shader:
    """Add a shader to a node graph.

    Args:
        node_graph (UsdShade.NodeGraph): The node graph to add the shader to.
        shader_name (str): The name of the shader to create.
        shader_type (str): The type of shader to create (e.g. "UsdPreviewSurface").

    Returns:
        UsdShade.Shader: The created shader.

    Raises:
        ValueError: If the node graph is invalid.
    """
    if not node_graph.GetPrim().IsValid():
        raise ValueError("Invalid node graph.")
    node_graph_path = node_graph.GetPath()
    shader_path = node_graph_path.AppendChild(shader_name)
    shader = UsdShade.Shader.Define(node_graph.GetPrim().GetStage(), shader_path)
    shader.CreateIdAttr(UsdShade.Tokens.sourceAsset)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.SetShaderId(shader_type)
    return shader


def duplicate_shader(source_shader: UsdShade.Shader, destination_path: str) -> UsdShade.Shader:
    """
    Duplicates a shader to a new path in the same stage.

    Args:
        source_shader (UsdShade.Shader): The shader to duplicate.
        destination_path (str): The destination path for the duplicated shader.

    Returns:
        UsdShade.Shader: The duplicated shader.

    Raises:
        ValueError: If the source shader is invalid or the destination path is invalid.
    """
    if not source_shader.GetPrim().IsValid():
        raise ValueError("Invalid source shader.")
    stage = source_shader.GetPrim().GetStage()
    destination_path = Sdf.Path(destination_path)
    if not destination_path.IsAbsolutePath():
        raise ValueError("Destination path must be an absolute path.")
    existing_prim = stage.GetPrimAtPath(destination_path)
    if existing_prim.IsValid():
        raise ValueError(f"A prim already exists at the destination path: {destination_path}")
    dest_shader = UsdShade.Shader.Define(stage, destination_path)
    source_shader_id = source_shader.GetIdAttr().Get()
    dest_shader.CreateIdAttr(source_shader_id)
    for input in source_shader.GetInputs():
        input_name = input.GetBaseName()
        dest_input = dest_shader.CreateInput(input_name, input.GetTypeName())
        if input.Get() is not None:
            dest_input.Set(input.Get())
    return dest_shader


def test_duplicate_shader():
    stage = Usd.Stage.CreateInMemory()
    material_path = "/Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader_path = material_path + "/Shader"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("TestShader")
    shader.CreateInput("input1", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("input2", Sdf.ValueTypeNames.Color3f).Set((0.0, 0.5, 1.0))
    duplicated_shader_path = "/DuplicatedShader"
    duplicated_shader = duplicate_shader(shader, duplicated_shader_path)
    assert duplicated_shader.GetIdAttr().Get() == "TestShader"
    assert duplicated_shader.GetInput("input1").Get() == 1.0
    assert duplicated_shader.GetInput("input2").Get() == (0.0, 0.5, 1.0)
    print(f"Duplicated shader path: {duplicated_shader.GetPath()}")
    print(f"Duplicated shader ID: {duplicated_shader.GetIdAttr().Get()}")
    print("Duplicated shader inputs:")
    for input in duplicated_shader.GetInputs():
        print(f"  {input.GetBaseName()}: {input.Get()}")


def get_connected_shader(output: UsdShade.Output) -> Optional[UsdShade.Shader]:
    """
    Get the shader connected to the given output, if any.

    Args:
        output (UsdShade.Output): The output to get the connected shader for.

    Returns:
        Optional[UsdShade.Shader]: The connected shader, or None if no shader is connected.
    """
    if not output.HasConnectedSource():
        return None
    source_attr = output.GetConnectedSource()[0]
    source_prim = source_attr.GetPrim()
    if not UsdShade.Shader(source_prim):
        return None
    return UsdShade.Shader(source_prim)


def get_shader_connections(shader: UsdShade.Shader) -> List[Tuple[str, str]]:
    """
    Get a list of tuples representing the input connections to the given shader.

    Each tuple contains (inputName, sourceOutputName).
    """
    connections = []
    connectableAPI = UsdShade.ConnectableAPI(shader)
    for input in connectableAPI.GetInputs():
        inputAttr = input.GetAttr()
        if connectableAPI.HasConnectedSource(inputAttr):
            source = connectableAPI.GetConnectedSource(inputAttr)
            sourceOutputName = source[1]
            inputName = input.GetBaseName()
            connections.append((inputName, sourceOutputName))
    return connections


def get_material_shaders(material: UsdShade.Material) -> List[UsdShade.Shader]:
    """
    Get all shaders directly connected to the given material.

    Args:
        material (UsdShade.Material): The material to get shaders for.

    Returns:
        List[UsdShade.Shader]: A list of shaders connected to the material.
    """
    conn_api = UsdShade.ConnectableAPI(material)
    inputs = conn_api.GetInputs()
    outputs = conn_api.GetOutputs()
    inputs_and_outputs = []
    if inputs:
        inputs_and_outputs.extend(inputs)
    if outputs:
        inputs_and_outputs.extend(outputs)
    shaders = []
    for input in inputs_and_outputs:
        if input.HasConnectedSource():
            source = input.GetConnectedSource()
            source_attr = source[0]
            source_prim = source_attr.GetPrim()
            if source_prim.IsA(UsdShade.Shader):
                if UsdShade.Shader(source_prim) not in shaders:
                    shaders.append(UsdShade.Shader(source_prim))
    return shaders


def modify_shader_input_connection(
    shader: UsdShade.Shader,
    input_name: str,
    source: UsdShade.ConnectableAPI,
    output_name: str,
    mod: UsdShade.ConnectionModification,
) -> bool:
    """Modify the connection of a shader's input.

    Args:
        shader (UsdShade.Shader): The shader to modify.
        input_name (str): The name of the input to modify.
        source (UsdShade.ConnectableAPI): The source to connect to the input.
        output_name (str): The name of the output on the source to connect.
        mod (UsdShade.ConnectionModification): The modification type.

    Returns:
        bool: True if the connection was modified successfully, False otherwise.
    """
    input_attr = shader.GetInput(input_name)
    if not input_attr:
        return False
    if not source or not source.GetOutput(output_name):
        return False
    source_attr = source.GetOutput(output_name)
    if mod == UsdShade.ConnectionModification.Replace:
        input_attr.ClearConnections()
        input_attr.AddConnection(source_attr)
    elif mod == UsdShade.ConnectionModification.Prepend:
        input_attr.AddConnection(source_attr, Usd.ListPositionFrontOfPrependList)
    elif mod == UsdShade.ConnectionModification.Append:
        input_attr.AddConnection(source_attr, Usd.ListPositionBackOfAppendList)
    else:
        return False
    return True


def connect_prims_with_material(stage: Usd.Stage, prims: List[Usd.Prim], material: UsdShade.Material) -> None:
    """Connect a list of prims to a material.

    Args:
        stage (Usd.Stage): The USD stage.
        prims (List[Usd.Prim]): The list of prims to connect to the material.
        material (UsdShade.Material): The material to connect the prims to.
    """
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material provided.")
    for prim in prims:
        if not prim.IsValid():
            print(f"Warning: Prim {prim.GetPath()} is invalid. Skipping.")
            continue
        if not UsdShade.MaterialBindingAPI(prim):
            print(f"Warning: Prim {prim.GetPath()} does not have the MaterialBindingAPI. Skipping.")
            continue
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        material_path = material.GetPath()
        material_binding_api.Bind(material)
        print(f"Connected prim {prim.GetPath()} to material {material_path}")


def disconnect_upstream_source(material: UsdShade.Material, input_name: str) -> bool:
    """Disconnect the upstream source of a Material input.

    Args:
        material (UsdShade.Material): The Material prim.
        input_name (str): The name of the input to disconnect.

    Returns:
        bool: True if the input was successfully disconnected, False otherwise.
    """
    if not material or not material.GetPrim().IsValid():
        return False
    input_attr = material.GetInput(input_name)
    if not input_attr:
        return False
    if not input_attr.HasConnectedSource():
        return True
    source_info = input_attr.GetConnectedSources()[0]
    success = input_attr.ClearSources()
    return success


def list_all_connections(prim: UsdShade.Shader) -> List[UsdShade.ConnectionSourceInfo]:
    """
    Get a list of all connections on the given shader prim.

    Args:
        prim (UsdShade.Shader): The shader prim to query connections from.

    Returns:
        List[UsdShade.ConnectionSourceInfo]: A list of ConnectionSourceInfo objects
        representing all connections on the shader prim.
    """
    if not prim.GetPrim().IsA(UsdShade.Shader):
        raise ValueError(f"Prim '{prim.GetPath()}' is not a valid shader.")
    shader_inputs = prim.GetInputs()
    shader_outputs = prim.GetOutputs()
    shader_attributes = list(shader_inputs) + list(shader_outputs)
    connections = []
    for attr in shader_attributes:
        source = UsdShade.ConnectableAPI.GetConnectedSource(attr)
        if source and source.IsValid():
            connections.append(source)
    return connections


def has_local_coord_sys_bindings(prim: Usd.Prim) -> bool:
    """Check if the prim has local coordinate system binding opinions.

    Note that the resulting binding list may still be empty.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    properties = prim.GetProperties()
    for prop in properties:
        prop_name = prop.GetName()
        if prop_name.startswith("coordSys:"):
            targets = prop.GetTargets()
            if len(targets) > 0:
                return True
    return False


def disconnect_multiple_shader_inputs(shader: UsdShade.Shader, input_names: List[str]) -> int:
    """Disconnect multiple inputs on a shader.

    Args:
        shader (UsdShade.Shader): The shader to disconnect inputs from.
        input_names (List[str]): A list of input names to disconnect.

    Returns:
        int: The number of inputs successfully disconnected.
    """
    num_disconnected = 0
    if not shader:
        raise ValueError("Invalid shader provided.")
    for input_name in input_names:
        input_attr = shader.GetInput(input_name)
        if input_attr:
            if input_attr.HasConnectedSource():
                input_attr.ClearSources()
                num_disconnected += 1
        else:
            print(f"Warning: Input '{input_name}' not found on shader '{shader.GetPath()}'.")
    return num_disconnected


def connect_shader_input(
    input: UsdShade.Input,
    source: UsdShade.ConnectableAPI,
    sourceName: str,
    sourceType: UsdShade.AttributeType = UsdShade.AttributeType.Output,
) -> bool:
    """Connect a shader input to a source.

    Args:
        input (UsdShade.Input): The input to connect.
        source (UsdShade.ConnectableAPI): The source connectable prim.
        sourceName (str): The name of the source shader's output.
        sourceType (UsdShade.AttributeType, optional): The type of the source attribute. Defaults to UsdShade.AttributeType.Output.

    Returns:
        bool: True if the connection was successful, False otherwise.
    """
    if not input:
        return False
    if not source:
        return False
    sourceAttr = (
        source.GetOutput(sourceName) if sourceType == UsdShade.AttributeType.Output else source.GetInput(sourceName)
    )
    if not sourceAttr:
        return False
    success = input.ConnectToSource(sourceAttr)
    return success


def get_shader_input_connectability(input_attribute: UsdShade.Input) -> str:
    """
    Get the connectability of a shader input attribute.

    Args:
        input_attribute (UsdShade.Input): The input attribute to query.

    Returns:
        str: The connectability value, either "full" or "interfaceOnly".
    """
    if not input_attribute:
        raise ValueError("Invalid input attribute")
    connectability = input_attribute.GetConnectability()
    if not connectability:
        return UsdShade.Tokens.full
    return connectability


def set_shader_input_connectability(input_: UsdShade.Input, connectability: str) -> bool:
    """Set the connectability of the given shader input.

    Args:
        input_ (UsdShade.Input): The shader input to set the connectability for.
        connectability (str): The connectability value to set. Must be either
            "full" or "interfaceOnly".

    Returns:
        bool: True if the connectability was set successfully, False otherwise.
    """
    if not input_:
        raise ValueError("Invalid shader input")
    if connectability not in ["full", "interfaceOnly"]:
        raise ValueError("Invalid connectability value. Must be 'full' or 'interfaceOnly'")
    success = input_.SetConnectability(connectability)
    return success


def set_shader_input_sdr_metadata(input_: UsdShade.Input, key: str, value: str) -> None:
    """Set a key-value pair in the sdrMetadata dictionary of a shader input.

    Args:
        input_ (UsdShade.Input): The shader input to set metadata on.
        key (str): The key in the sdrMetadata dictionary.
        value (str): The value to set for the given key.

    Raises:
        ValueError: If the input is invalid.
    """
    if not input_:
        raise ValueError("Invalid shader input")
    if input_.HasSdrMetadataByKey(key):
        input_.SetSdrMetadataByKey(key, value)
    else:
        sdr_metadata = input_.GetSdrMetadata()
        sdr_metadata[key] = value
        input_.SetSdrMetadata(sdr_metadata)


def clear_shader_input_sdr_metadata(input: UsdShade.Input) -> bool:
    """Clear all sdrMetadata authored on the given shader input.

    Args:
        input (UsdShade.Input): The shader input to clear sdrMetadata from.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not input:
        return False
    if not input.HasSdrMetadata():
        return True
    input.ClearSdrMetadata()
    if input.HasSdrMetadata():
        return False
    return True


def clear_shader_input_sdr_metadata_by_key(input_: UsdShade.Input, key: str) -> bool:
    """Clear a specific key from an Input's sdrMetadata dictionary.

    Args:
        input_ (UsdShade.Input): The Input prim to clear sdrMetadata from.
        key (str): The key in the sdrMetadata to clear.

    Returns:
        bool: True if the value was cleared successfully, False otherwise.
    """
    if input_.HasSdrMetadataByKey(key):
        input_.ClearSdrMetadataByKey(key)
        return True
    else:
        return False


def get_shader_input_attr(shader: UsdShade.Shader, input_name: str) -> UsdShade.Input:
    """
    Get the UsdShade.Input object for a given input name on a shader.

    Args:
        shader (UsdShade.Shader): The shader to get the input attribute from.
        input_name (str): The name of the input attribute to get.

    Returns:
        UsdShade.Input: The UsdShade.Input object for the specified input name.

    Raises:
        ValueError: If the shader is invalid or if the input attribute doesn't exist.
    """
    if not shader:
        raise ValueError("Invalid shader")
    input_attr = shader.GetInput(input_name)
    if not input_attr:
        raise ValueError(f"Input attribute '{input_name}' does not exist on the shader")
    return input_attr


def get_shader_input_base_name(input: UsdShade.Input) -> str:
    """Get the base name of a shader input, stripping off the 'inputs:' namespace prefix."""
    full_name = input.GetFullName()
    name_parts = full_name.split(":")
    if len(name_parts) < 2 or name_parts[0] != "inputs":
        return full_name
    base_name = ":".join(name_parts[1:])
    return base_name


def get_shader_input_full_name(shader_input: UsdShade.Input) -> str:
    """Get the full name of a shader input, including the 'inputs:' prefix."""
    if not shader_input:
        raise ValueError("Invalid shader input")
    base_name = shader_input.GetBaseName()
    full_name = f"inputs:{base_name}"
    return full_name


def get_shader_input_display_group(input: UsdShade.Input) -> str:
    """Get the display group for a shader input.

    Args:
        input (UsdShade.Input): The shader input to get the display group for.

    Returns:
        str: The display group for the shader input, or an empty string if not set.
    """
    if not input:
        raise ValueError("Invalid shader input")
    display_group = input.GetDisplayGroup()
    return display_group if display_group else ""


def set_shader_input_display_group(input_: UsdShade.Input, display_group: str) -> None:
    """Set the display group metadata for a shader input.

    Args:
        input_ (UsdShade.Input): The shader input to set the display group for.
        display_group (str): The display group to set.
    """
    if not input_:
        raise ValueError("Invalid UsdShade.Input object")
    attr = input_.GetAttr()
    if not attr:
        raise ValueError("Input has no associated attribute")
    success = attr.SetMetadata("displayGroup", display_group)
    if not success:
        raise ValueError("Failed to set display group metadata")


def set_shader_input_documentation(input_: UsdShade.Input, docs: str) -> bool:
    """Set documentation string for a shader input.

    Args:
        input_ (UsdShade.Input): The shader input to set documentation for.
        docs (str): The documentation string to set.

    Returns:
        bool: True if the documentation was set successfully, False otherwise.
    """
    if not input_:
        print(f"Error: Invalid shader input")
        return False
    attr = input_.GetAttr()
    if not attr:
        print(f"Error: Failed to get attribute for shader input")
        return False
    success = attr.SetDocumentation(docs)
    if not success:
        print(f"Error: Failed to set documentation for shader input")
        return False
    return True


def set_shader_input_render_type(input_: UsdShade.Input, render_type: str) -> bool:
    """Set the render type for a shader input.

    Args:
        input_ (UsdShade.Input): The shader input to set the render type for.
        render_type (str): The render type to set.

    Returns:
        bool: True if the render type was set successfully, False otherwise.
    """
    if not input_:
        print(f"Error: Invalid shader input")
        return False
    if not isinstance(render_type, str) or not render_type:
        print(f"Error: Invalid render type '{render_type}'")
        return False
    success = input_.SetRenderType(render_type)
    if not success:
        print(f"Error: Failed to set render type '{render_type}' on shader input '{input_.GetFullName()}'")
        return False
    return True


def has_shader_input_render_type(input: UsdShade.Input) -> bool:
    """Check if a shader input has a non-empty render type."""
    if not input.GetAttr().IsDefined():
        raise ValueError("Invalid shader input")
    render_type = input.GetRenderType()
    return bool(render_type)


def has_shader_input_sdr_metadata(input_attr: UsdShade.Input, key: str) -> bool:
    """Check if a shader input has a specific sdrMetadata key."""
    if not input_attr:
        raise ValueError("Input attribute is invalid.")
    if not input_attr.HasSdrMetadata():
        return False
    return input_attr.HasSdrMetadataByKey(key)


def has_shader_input_sdr_metadata_by_key(input: UsdShade.Input, key: str) -> bool:
    """Check if a shader input has a specific sdrMetadata key."""
    if not input.GetAttr().IsDefined():
        raise ValueError("Invalid shader input")
    if not input.HasSdrMetadata():
        return False
    return input.HasSdrMetadataByKey(key)


def get_shader_input_sdr_metadata_by_key(shader: UsdShade.Shader, input_name: str, key: str) -> str:
    """Get the value of a specific key in the sdrMetadata dictionary for a shader input.

    Args:
        shader (UsdShade.Shader): The shader prim.
        input_name (str): The name of the input.
        key (str): The key in the sdrMetadata dictionary.

    Returns:
        str: The value corresponding to the key, or an empty string if not found.
    """
    input_attr = shader.GetInput(input_name)
    if not input_attr:
        return ""
    sdr_metadata = input_attr.GetSdrMetadata()
    if not sdr_metadata:
        return ""
    value = sdr_metadata.get(key, "")
    return value


def set_multiple_shader_input_values(shader: UsdShade.Shader, input_values: Dict[str, Any]) -> None:
    """Set multiple input values on a shader at once.

    Parameters
    ----------
    shader : UsdShade.Shader
        The shader to set input values on.
    input_values : Dict[str, Any]
        A dictionary mapping input names to their values.
    """
    if not shader or not isinstance(shader, UsdShade.Shader):
        raise ValueError("Invalid shader object")
    for input_name, input_value in input_values.items():
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            print(f"Warning: Shader input '{input_name}' does not exist. Skipping.")
            continue
        try:
            shader_input.Set(input_value)
        except Exception as e:
            print(f"Error setting value for input '{input_name}': {str(e)}. Skipping.")


def get_multiple_shader_input_connectabilities(shader_prim: UsdShade.Shader, input_names: List[str]) -> Dict[str, str]:
    """
    Get the connectability of multiple inputs on a shader prim.

    Args:
        shader_prim (UsdShade.Shader): The shader prim.
        input_names (List[str]): List of input names to query connectability for.

    Returns:
        Dict[str, str]: Dictionary mapping input name to its connectability.
    """
    connectabilities = {}
    for input_name in input_names:
        shader_input = shader_prim.GetInput(input_name)
        if shader_input:
            connectability = shader_input.GetConnectability()
            connectabilities[input_name] = connectability
        else:
            connectabilities[input_name] = None
    return connectabilities


def set_multiple_shader_input_connectabilities(
    shader: UsdShade.Shader, input_connectability_dict: Dict[str, str]
) -> None:
    """Set the connectability of multiple inputs on a shader.

    Args:
        shader (UsdShade.Shader): The shader prim.
        input_connectability_dict (Dict[str, str]): A dictionary mapping input names to their desired connectability.
    """
    for input_name, connectability in input_connectability_dict.items():
        shader_input = shader.GetInput(input_name)
        if not shader_input:
            print(f"Warning: Input '{input_name}' not found on shader '{shader.GetPath()}'.")
            continue
        success = shader_input.SetConnectability(connectability)
        if not success:
            raise ValueError(
                f"Failed to set connectability '{connectability}' for input '{input_name}' on shader '{shader.GetPath()}'."
            )


def get_multiple_shader_input_sdr_metadata(inputs: List[UsdShade.Input]) -> Dict[str, Dict[str, str]]:
    """
    Retrieves the sdrMetadata for multiple shader inputs.

    Args:
        inputs (List[UsdShade.Input]): A list of shader inputs.

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping input names to their sdrMetadata dictionaries.
    """
    metadata_dict = {}
    for input in inputs:
        input_name = input.GetBaseName()
        if input.HasSdrMetadata():
            sdr_metadata = input.GetSdrMetadata()
            metadata_dict[input_name] = {key: sdr_metadata[key] for key in sdr_metadata.keys()}
        else:
            metadata_dict[input_name] = {}
    return metadata_dict


def clear_multiple_shader_input_sdr_metadata(shader: UsdShade.Shader, input_names: List[str]) -> None:
    """Clear the sdrMetadata for multiple shader inputs at once.

    Args:
        shader (UsdShade.Shader): The shader prim.
        input_names (List[str]): The list of input names to clear sdrMetadata for.

    Raises:
        ValueError: If the shader prim is invalid.
    """
    if not shader or not shader.GetPrim().IsValid():
        raise ValueError("Invalid shader prim.")
    for input_name in input_names:
        shader_input = shader.GetInput(input_name)
        if shader_input:
            shader_input.ClearSdrMetadata()
        else:
            print(f"Warning: Input '{input_name}' not found on shader '{shader.GetPath()}'.")


def get_shader_input_sdr_metadata(input: UsdShade.Input) -> dict:
    """Get the shader input's sdrMetadata as a dictionary."""
    if not input:
        raise ValueError("Invalid input object")
    if not input.HasSdrMetadata():
        return {}
    sdr_metadata_map = input.GetSdrMetadata()
    sdr_metadata_dict = {}
    for key, value in sdr_metadata_map.items():
        sdr_metadata_dict[key] = value
    return sdr_metadata_dict


def set_shader_input_sdr_metadata_by_key(input: UsdShade.Input, key: str, value: str):
    """Set a key-value pair in the sdrMetadata dictionary of a shader input."""
    if not input:
        raise ValueError("Invalid input")
    sdr_metadata = input.GetSdrMetadata()
    sdr_metadata[key] = value
    input.SetSdrMetadata(sdr_metadata)


def get_multiple_shader_input_values(
    shader: UsdShade.Shader, input_names: List[str], time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> Dict[str, Union[float, int, str, bool, Sdf.ValueBlock]]:
    """
    Get the values of multiple inputs on a shader at a specific time code.

    Args:
        shader (UsdShade.Shader): The shader to query input values from.
        input_names (List[str]): A list of input names to query values for.
        time_code (Usd.TimeCode): The time code at which to query the input values. Defaults to Default.

    Returns:
        Dict[str, Union[float, int, str, bool, Sdf.ValueBlock]]: A dictionary mapping input names to their values.
                                                                  If an input is not found, its value will be Sdf.ValueBlock.
    """
    input_values = {}
    for input_name in input_names:
        input_attr = shader.GetInput(input_name)
        if input_attr:
            input_value = input_attr.Get(time_code)
            if input_value is None:
                input_value = Sdf.ValueBlock()
            input_values[input_name] = input_value
        else:
            input_values[input_name] = Sdf.ValueBlock()
    return input_values


def set_multiple_shader_input_connected_sources(
    shader_prim: UsdShade.Shader, connections: List[Tuple[str, UsdShade.ConnectableAPI]]
) -> bool:
    """
    Sets the connected sources for multiple inputs on a shader prim.

    Args:
        shader_prim (UsdShade.Shader): The shader prim to set input connections on.
        connections (List[Tuple[str, UsdShade.ConnectableAPI]]): A list of tuples, where each tuple contains:
            - The name of the input to set the connection on.
            - The connectable prim to connect to the input.

    Returns:
        bool: True if all connections were set successfully, False otherwise.
    """
    if not shader_prim:
        raise ValueError("Invalid shader prim")
    for input_name, source in connections:
        input_attr = shader_prim.GetInput(input_name)
        if not input_attr:
            print(f"Input '{input_name}' does not exist on shader prim '{shader_prim.GetPath()}'")
            return False
        input_attr.ClearSources()
        if not source:
            print(f"Invalid source for input '{input_name}' on shader prim '{shader_prim.GetPath()}'")
            return False
        source_output = source.GetOutput(input_name)
        if not source_output:
            print(f"Output '{input_name}' does not exist on source prim '{source.GetPath()}'")
            return False
        success = input_attr.ConnectToSource(source_output)
        if not success:
            print(
                f"Failed to connect input '{input_name}' on shader prim '{shader_prim.GetPath()}' to source '{source.GetPath()}'"
            )
            return False
    return True


def get_shader_input_connected_sources(input: UsdShade.Input) -> Tuple[List[UsdShade.ConnectableAPI], List[Sdf.Path]]:
    """
    Get the connected sources for a shader input.

    Args:
        input (UsdShade.Input): The shader input to get connected sources for.

    Returns:
        A tuple containing two lists:
        - A list of valid connected sources (UsdShade.ConnectableAPI objects).
        - A list of invalid source paths (Sdf.Path objects).
    """
    if not input.GetAttr().IsValid():
        raise ValueError("Invalid shader input.")
    source_paths = input.GetAttr().GetConnections()
    valid_sources = []
    invalid_paths = []
    for path in source_paths:
        source_prim = input.GetPrim().GetStage().GetPrimAtPath(path.GetPrimPath())
        if source_prim.IsValid():
            source = UsdShade.ConnectableAPI(source_prim)
            if source:
                valid_sources.append(source)
            else:
                invalid_paths.append(path)
        else:
            invalid_paths.append(path)
    return (valid_sources, invalid_paths)


def get_shader_input_documentation(input: UsdShade.Input) -> str:
    """Get the documentation string for a shader input.

    Args:
        input (UsdShade.Input): The input to get documentation for.

    Returns:
        str: The documentation string for the input, or an empty string if not set.
    """
    if not input:
        raise ValueError("Invalid input")
    docs = input.GetDocumentation()
    return docs


def get_shader_input_type_name(input: UsdShade.Input) -> str:
    """Get the type name of a shader input attribute."""
    attr = input.GetAttr()
    if not attr:
        raise ValueError("Input has no associated attribute.")
    type_name = attr.GetTypeName()
    if not type_name:
        raise ValueError("Attribute has no type name.")
    return str(type_name)


def clear_shader_input_sources(input_: UsdShade.Input) -> bool:
    """Clear all connected sources for a shader input.

    Args:
        input_ (UsdShade.Input): The shader input to clear sources for.

    Returns:
        bool: True if the sources were successfully cleared, False otherwise.
    """
    if not input_:
        return False
    try:
        success = input_.ClearSources()
    except Tf.ErrorException:
        return False
    return success


def set_multiple_shader_input_sdr_metadata(shader: UsdShade.Shader, input_metadata: Dict[str, Dict[str, str]]) -> None:
    """
    Sets multiple sdrMetadata key-value pairs on shader inputs.

    Args:
        shader (UsdShade.Shader): The shader prim.
        input_metadata (Dict[str, Dict[str, str]]): A dictionary mapping input names to dictionaries of sdrMetadata key-value pairs.

    Raises:
        ValueError: If the shader prim is not valid or if any input name is not found on the shader.
    """
    if not shader.GetPrim().IsValid():
        raise ValueError("Invalid shader prim.")
    for input_name, metadata in input_metadata.items():
        input_attr = shader.GetInput(input_name)
        if not input_attr:
            raise ValueError(f"Input '{input_name}' not found on shader.")
        input_obj = UsdShade.Input(input_attr)
        for key, value in metadata.items():
            input_obj.SetSdrMetadataByKey(key, value)


def get_shader_input_prim(input: UsdShade.Input) -> Usd.Prim:
    """Get the prim that owns the shader input attribute."""
    attr = input.GetAttr()
    if not attr:
        raise ValueError("Input has no associated attribute.")
    prim = attr.GetPrim()
    if not prim:
        raise ValueError("Attribute does not belong to a valid prim.")
    return prim


def is_shader_input(stage: Usd.Stage, attribute_path: str) -> bool:
    """Check if an attribute is a valid shader input.

    Args:
        stage (Usd.Stage): The USD stage.
        attribute_path (str): The full path to the attribute.

    Returns:
        bool: True if the attribute is a valid shader input, False otherwise.
    """
    attribute = stage.GetAttributeAtPath(attribute_path)
    if not attribute:
        return False
    is_input = UsdShade.Input.IsInput(attribute)
    return is_input


def has_shader_input_connected_source(input: UsdShade.Input) -> bool:
    """Check if a shader input has a connected source."""
    if not input.GetAttr().IsDefined():
        raise ValueError("Input attribute is not defined.")
    prim = input.GetPrim()
    if not prim.IsValid():
        raise ValueError("Prim that owns the input is not valid.")
    if not UsdShade.Shader(prim):
        raise ValueError("Prim is not a shader.")
    return input.HasConnectedSource()


def get_shader_input_value_producing_attributes(
    input_: UsdShade.Input, shader_outputs_only: bool
) -> List[UsdShade.Input]:
    """
    Find the attributes that are connected to this Input recursively.

    Args:
        input_ (UsdShade.Input): The input to find the value producing attributes for.
        shader_outputs_only (bool): If True, only include outputs from UsdShadeShader prims.

    Returns:
        List[UsdShade.Input]: A list of UsdShade.Input objects that are connected to this Input.
    """
    attributes: List[UsdShade.Input] = []
    if not input_:
        return attributes
    sources = input_.GetConnectedSources()
    for source in sources[0]:
        source_prim = source.source.GetPrim()
        source_attr = source_prim.GetAttribute(source.sourceName)
        if UsdShade.Output.IsOutput(source_attr):
            source_output = UsdShade.Output(source_attr)
            if not shader_outputs_only or UsdShade.Shader(source_output.GetPrim()):
                attributes.append(UsdShade.Input(source_attr))
        else:
            source_input = UsdShade.Input(source_attr)
            attributes.extend(get_shader_input_value_producing_attributes(source_input, shader_outputs_only))
    return attributes


def connect_multiple_shader_inputs(shader: UsdShade.Shader, input_connections: Dict[str, UsdShade.Output]) -> None:
    """Connect multiple inputs on a shader to their corresponding sources.

    Args:
        shader (UsdShade.Shader): The shader to connect inputs for.
        input_connections (Dict[str, UsdShade.Output]): A dictionary mapping input names to their
            source outputs. The input names should not include the "inputs:" prefix.

    Raises:
        ValueError: If the shader prim is not valid, or if any of the input names or sources are invalid.
    """
    if not shader:
        raise ValueError("Invalid shader: The shader prim is not valid")
    for input_name, source_output in input_connections.items():
        if not Sdf.Path.IsValidNamespacedIdentifier(input_name):
            raise ValueError(f"Invalid input name '{input_name}': Input names must be valid identifiers")
        input_attr = shader.GetInput(input_name)
        if not input_attr:
            raise ValueError(f"Input '{input_name}' does not exist on the shader")
        if not source_output:
            raise ValueError(f"Invalid source for input '{input_name}': The source output is not valid")
        input_attr.ConnectToSource(source_output)


def get_multiple_shader_input_connected_sources(
    shader_prim: UsdShade.Shader, input_names: List[str]
) -> Dict[str, List[Tuple[UsdShade.ConnectableAPI, str]]]:
    """
    Get the connected sources for multiple inputs on a shader prim.

    Args:
        shader_prim (UsdShade.Shader): The shader prim to query.
        input_names (List[str]): The names of the inputs to get sources for.

    Returns:
        Dict[str, List[Tuple[UsdShade.ConnectableAPI, str]]]: A dictionary mapping input names to lists of connected source outputs.
    """
    result: Dict[str, List[Tuple[UsdShade.ConnectableAPI, str]]] = {}
    for input_name in input_names:
        shader_input: UsdShade.Input = shader_prim.GetInput(input_name)
        if not shader_input:
            continue
        connected_sources: List[Tuple[UsdShade.ConnectableAPI, str]] = []
        source_infos: Tuple[List[UsdShade.ConnectionSourceInfo], List[Sdf.Path]] = shader_input.GetConnectedSources()
        for source_info in source_infos[0]:
            connected_sources.append((source_info.source, source_info.sourceName))
        result[input_name] = connected_sources
    return result


def clear_shader_input_connectability(input: UsdShade.Input) -> bool:
    """Clear the connectability metadata for a shader input.

    Args:
        input (UsdShade.Input): The shader input to clear connectability for.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    if not input:
        return False
    success = input.ClearConnectability()
    return success


def set_shader_input_connected_sources(
    input_: UsdShade.Input, source_infos: List[UsdShade.ConnectionSourceInfo]
) -> bool:
    """
    Set the connected sources for a shader input.

    Args:
        input_ (UsdShade.Input): The shader input to set connected sources for.
        source_infos (List[UsdShade.ConnectionSourceInfo]): List of ConnectionSourceInfo objects specifying the sources to connect.

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    if not input_:
        return False
    if not source_infos:
        return input_.ClearSources()
    usd_source_infos = []
    for source_info in source_infos:
        if not source_info.source or not source_info.sourceName or source_info.sourceType is None:
            continue
        usd_source_info = UsdShade.ConnectionSourceInfo(
            UsdShade.ConnectableAPI(source_info.source),
            source_info.sourceName,
            source_info.sourceType,
            Sdf.ValueTypeNames.Token,
        )
        usd_source_infos.append(usd_source_info)
    return input_.SetConnectedSources(usd_source_infos)


def get_shader_parameter(shader: UsdShade.Shader, param_name: str):
    """Get the value of a shader parameter.

    Args:
        shader (UsdShade.Shader): The shader to get the parameter from.
        param_name (str): The name of the parameter to get.

    Returns:
        Any: The value of the parameter, or None if not found.
    """
    if not shader or not shader.GetPrim().IsValid():
        raise ValueError("Invalid shader prim.")
    param_attr = shader.GetPrim().GetAttribute(param_name)
    if not param_attr or not param_attr.IsValid():
        print(f"Parameter '{param_name}' not found on shader '{shader.GetPath()}'.")
        return None
    param_value = param_attr.Get()
    return param_value


def create_material_from_usd(stage: Usd.Stage, material_path: str) -> UsdShade.Material:
    """Create a new UsdShadeMaterial prim from an existing material path."""
    if not stage:
        raise ValueError("Invalid stage")
    if not material_path:
        raise ValueError("Invalid material path")
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim:
        raise ValueError(f"Material prim not found at path: {material_path}")
    if not material_prim.IsA(UsdShade.Material):
        raise TypeError(f"Prim at path {material_path} is not a UsdShadeMaterial")
    new_material = UsdShade.Material(material_prim)
    return new_material


def assign_material_variant_to_prim(
    prim: Usd.Prim, material: UsdShade.Material, variant_name: str, purpose: str = ""
) -> None:
    """Assigns a specific variant of a material to a prim.

    Args:
        prim (Usd.Prim): The prim to assign the material variant to.
        material (UsdShade.Material): The material containing the variant to assign.
        variant_name (str): The name of the material variant to assign.
        purpose (str, optional): The material purpose specifying the render context. Defaults to "".

    Raises:
        ValueError: If the prim or material is invalid, or if the variant does not exist on the material.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    if not material.GetPrim().IsValid():
        raise ValueError(f"Invalid material: {material}")
    variant_set = material.GetMaterialVariant()
    if not variant_set.HasAuthoredVariant(variant_name):
        raise ValueError(f"Material {material.GetPath()} does not have variant '{variant_name}'")
    variant_set.SetVariantSelection(variant_name)
    binding_api = UsdShade.MaterialBindingAPI(prim)
    binding_api.Bind(material, materialPurpose=purpose)


def get_bound_materials(prim: Usd.Prim) -> List[UsdShade.Material]:
    """
    Get the list of materials bound to the given prim.

    Args:
        prim (Usd.Prim): The prim to query for bound materials.

    Returns:
        List[UsdShade.Material]: The list of bound materials.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim}")
    binding_api = UsdShade.MaterialBindingAPI(prim)
    collection_bindings = binding_api.GetCollectionBindings()
    bound_materials = []
    for binding in collection_bindings:
        if not binding.IsValid():
            continue
        material = binding.GetMaterial()
        if material.IsValid():
            bound_materials.append(material)
    return bound_materials


def create_material(stage, path):
    material = UsdShade.Material.Define(stage, path)
    return material


def switch_material_variants(material: UsdShade.Material, variant_name: str, edit_layer: Sdf.Layer = None) -> None:
    """Switch the material variant for the given material to the specified variant name.

    Args:
        material (UsdShade.Material): The material to switch variants for.
        variant_name (str): The name of the variant to switch to.
        edit_layer (Sdf.Layer, optional): The layer to author the variant edit on. Defaults to the stage's current edit target.

    Raises:
        ValueError: If the material is invalid or the variant name is empty.
    """
    if not material:
        raise ValueError("Invalid material")
    if not variant_name:
        raise ValueError("Empty variant name")
    variant_set = material.GetMaterialVariant()
    if not variant_set:
        variant_set = material.GetPrim().GetVariantSets().AddVariantSet("MaterialVariant")
    if not variant_set.HasAuthoredVariant(variant_name):
        variant_set.AddVariant(variant_name)
    if edit_layer:
        with Usd.EditContext(material.GetStage(), edit_layer):
            variant_set.SetVariantSelection(variant_name)
    else:
        variant_set.SetVariantSelection(variant_name)


def get_material_variant_names(material: UsdShade.Material) -> List[str]:
    """Get the list of variant names for a material's MaterialVariant variantSet."""
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material prim.")
    variantSet = material.GetMaterialVariant()
    if not variantSet.IsValid():
        return []
    variant_names = variantSet.GetVariantNames()
    return variant_names


def set_material_base_material(material: UsdShade.Material, base_material: UsdShade.Material) -> None:
    """Set the base material for a given material.

    Args:
        material (UsdShade.Material): The material to set the base material for.
        base_material (UsdShade.Material): The base material to set.

    Raises:
        ValueError: If the given material or base material is invalid.
    """
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material")
    if base_material and (not base_material.GetPrim().IsValid()):
        raise ValueError("Invalid base material")
    if not base_material:
        material.ClearBaseMaterial()
        return
    material.SetBaseMaterial(base_material)


def remove_shader_from_material(material: UsdShade.Material, shader: UsdShade.Shader) -> None:
    """Remove a shader from a material by removing any relationship that targets it."""
    for rel in material.GetPrim().GetRelationships():
        targets = rel.GetTargets()
        if len(targets) > 0 and targets[0].GetPrimPath() == shader.GetPath():
            rel.RemoveTarget(shader.GetPath())
            if len(rel.GetTargets()) == 0:
                rel.ClearTargets(removeSpec=True)


def get_material_usd_attributes(material: UsdShade.Material) -> List[Usd.Attribute]:
    """Get all USD attributes on a material prim."""
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid UsdShadeMaterial object")
    prim = material.GetPrim()
    attributes = prim.GetAttributes()
    usd_attributes = [attr for attr in attributes if attr.IsAuthored()]
    return usd_attributes


def add_material_variant(
    material: UsdShade.Material, variant_name: str, variant_set_name: str = "MaterialVariant"
) -> None:
    """Add a new variant to the material's variant set.

    Args:
        material (UsdShade.Material): The material to add the variant to.
        variant_name (str): The name of the new variant.
        variant_set_name (str, optional): The name of the variant set. Defaults to "MaterialVariant".

    Raises:
        ValueError: If the material is not a valid UsdShade.Material.
        ValueError: If the variant name is empty or contains invalid characters.
    """
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material.")
    if not variant_name or not Sdf.Path.IsValidIdentifier(variant_name):
        raise ValueError("Invalid variant name.")
    variant_set = material.GetPrim().GetVariantSets().GetVariantSet(variant_set_name)
    if not variant_set.IsValid():
        variant_set = material.GetPrim().GetVariantSets().AddVariantSet(variant_set_name)
    if not variant_set.HasAuthoredVariant(variant_name):
        variant_set.AddVariant(variant_name)
    variant_set.SetVariantSelection(variant_name)


def get_material_shader_connections(material: UsdShade.Material) -> List[Tuple[str, str]]:
    """
    Get a list of (sourceShaderPath, inputName) connections for a material.

    Args:
        material (UsdShade.Material): The material to get shader connections for.

    Returns:
        List[Tuple[str, str]]: A list of (sourceShaderPath, inputName) connections.
    """
    connections = []
    for input in material.GetInputs():
        if input.HasConnectedSource():
            sourceShader = input.GetConnectedSource()[0]
            sourceShaderPath = sourceShader.GetPath().pathString
            inputName = input.GetBaseName()
            connections.append((sourceShaderPath, inputName))
    return connections


def list_material_prims(stage: Usd.Stage) -> List[Usd.Prim]:
    """
    Return a list of all Material prims in the stage.

    Args:
        stage (Usd.Stage): The stage to search for Material prims.

    Returns:
        List[Usd.Prim]: A list of all Material prims in the stage.
    """
    material_prims: List[Usd.Prim] = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            material_prims.append(prim)
    return material_prims


def create_surface_shader(
    stage: Usd.Stage, material: UsdShade.Material, shader_name: str, shader_id: str
) -> UsdShade.Shader:
    """Create a surface shader for a material.

    Args:
        stage (Usd.Stage): The USD stage.
        material (UsdShade.Material): The material to create the shader for.
        shader_name (str): The name of the shader.
        shader_id (str): The identifier of the shader.

    Returns:
        UsdShade.Shader: The created surface shader.

    Raises:
        ValueError: If the material prim is not valid.
    """
    if not material:
        raise ValueError("Invalid material prim")
    shader_path = material.GetPath().AppendChild(shader_name)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr(UsdShade.Tokens.sourceAsset).Set(shader_id)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return shader


def bind_material_to_prims(material: UsdShade.Material, prims: List[Usd.Prim]) -> None:
    """Binds a material to a list of prims.

    Args:
        material (UsdShade.Material): The material to bind.
        prims (List[Usd.Prim]): The list of prims to bind the material to.

    Raises:
        ValueError: If the material is not a valid UsdShadeMaterial.
        ValueError: If any prim in the list is not a valid UsdPrim.
    """
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material provided.")
    for prim in prims:
        if not prim.IsValid():
            raise ValueError(f"Invalid prim: {prim.GetPath()}")
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_api.Bind(material)
        print(f"Bound material '{material.GetPath()}' to prim '{prim.GetPath()}'")


def set_shader_parameter(
    shader: UsdShade.Shader, param_name: str, param_value: object, param_type: Sdf.ValueTypeName
) -> None:
    """Set a shader parameter value."""
    if not shader:
        raise ValueError("Invalid shader")
    shader_prim = shader.GetPrim()
    if not shader_prim.HasAttribute(param_name):
        attr = shader_prim.CreateAttribute(param_name, param_type)
    else:
        attr = shader_prim.GetAttribute(param_name)
    attr.Set(param_value)


def create_displacement_shader(
    stage: Usd.Stage, material: UsdShade.Material, shader_name: str = "myDisplacementShader"
) -> UsdShade.Shader:
    """Create a displacement shader under the given material.

    Args:
        stage (Usd.Stage): The USD stage.
        material (UsdShade.Material): The material to create the shader under.
        shader_name (str, optional): The name of the shader. Defaults to "myDisplacementShader".

    Returns:
        UsdShade.Shader: The created displacement shader.
    """
    if not material:
        raise ValueError("Invalid material")
    shader_path = material.GetPath().AppendChild(shader_name)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPrimvarReader_float")
    displacement_output = material.CreateDisplacementOutput()
    displacement_output.ConnectToSource(shader.ConnectableAPI(), "result", UsdShade.AttributeType.Output)
    return shader


def create_volume_shader(material: UsdShade.Material, shader_name: str, shader_id: str) -> UsdShade.Shader:
    """Create a volume shader for a material.

    Args:
        material (UsdShade.Material): The material to create the shader for.
        shader_name (str): The name of the shader.
        shader_id (str): The ID of the shader.

    Returns:
        UsdShade.Shader: The created volume shader.

    Raises:
        ValueError: If the material is not valid or if the shader name or ID is empty.
    """
    if not material:
        raise ValueError("Invalid material.")
    if not shader_name or not shader_id:
        raise ValueError("Shader name and ID cannot be empty.")
    shader_path = material.GetPath().AppendChild(shader_name)
    shader = UsdShade.Shader.Define(material.GetPrim().GetStage(), shader_path)
    shader.CreateIdAttr(UsdShade.Tokens.volume).Set(shader_id)
    shader_volume_output = shader.CreateOutput("volume", Sdf.ValueTypeNames.Token)
    volume_output = material.CreateVolumeOutput()
    volume_output.ConnectToSource(shader_volume_output)
    return shader


def create_material_graph(stage: Usd.Stage, material_path: str, shader_name: str, shader_id: str) -> UsdShade.Material:
    """Create a material graph with a single shader node.

    Args:
        stage (Usd.Stage): The USD stage to create the material graph on.
        material_path (str): The path where the material should be created.
        shader_name (str): The name of the shader node.
        shader_id (str): The identifier of the shader type (e.g., UsdPreviewSurface).

    Returns:
        UsdShade.Material: The created material prim.

    Raises:
        ValueError: If the material path is invalid or the shader type is not recognized.
    """
    material_prim_path = Sdf.Path(material_path)
    if not material_prim_path.IsPrimPath():
        raise ValueError(f"Invalid material path: {material_path}")
    material = UsdShade.Material.Define(stage, material_prim_path)
    if shader_id not in ["UsdPreviewSurface", "UsdPrimvarReader_float2"]:
        raise ValueError(f"Unrecognized shader ID: {shader_id}")
    shader_path = material_prim_path.AppendChild(shader_name)
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr().Set(shader_id)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_and_assign_material(stage: Usd.Stage, prim_path: str, material_path: str) -> UsdShade.Material:
    """Create a new material and assign it to the specified prim.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to assign the material to.
        material_path (str): The path where the material should be created.

    Returns:
        UsdShade.Material: The created material prim.

    Raises:
        ValueError: If the prim or material path is invalid.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise ValueError(f"Invalid prim path: {prim_path}")
    material = UsdShade.Material.Define(stage, material_path)
    if not material:
        raise ValueError(f"Failed to create material at path: {material_path}")
    binding_api = UsdShade.MaterialBindingAPI(prim)
    binding_api.Bind(material)
    return material


def reassign_material(prim: Usd.Prim, material: UsdShade.Material, purpose: str = "") -> bool:
    """Reassign a material to a prim, optionally for a specific purpose.

    Args:
        prim (Usd.Prim): The prim to assign the material to.
        material (UsdShade.Material): The material to assign.
        purpose (str, optional): The material purpose to assign for (if any). Defaults to "".

    Returns:
        bool: True if the material was successfully assigned, False otherwise.
    """
    if not prim.IsValid():
        print(f"Error: Invalid prim '{prim.GetPath()}'")
        return False
    if not material.GetPrim().IsValid():
        print(f"Error: Invalid material '{material.GetPath()}'")
        return False
    if not prim.IsA(UsdGeom.Imageable):
        print(f"Error: Prim '{prim.GetPath()}' is not a valid UsdGeomImageable")
        return False
    binding_api = UsdShade.MaterialBindingAPI(prim)
    direct_binding = binding_api.GetDirectBinding()
    if purpose == "":
        if direct_binding.GetMaterial() == material:
            print(f"Material '{material.GetPath()}' already bound to prim '{prim.GetPath()}' for all purposes")
            return True
        binding_api.Bind(material)
        print(f"Assigned material '{material.GetPath()}' to prim '{prim.GetPath()}' for all purposes")
    else:
        if direct_binding.GetMaterialPurpose() == purpose and direct_binding.GetMaterial() == material:
            print(f"Material '{material.GetPath()}' already bound to prim '{prim.GetPath()}' for purpose '{purpose}'")
            return True
        binding_api.Bind(material, materialPurpose=purpose)
        print(f"Assigned material '{material.GetPath()}' to prim '{prim.GetPath()}' for purpose '{purpose}'")
    return True


def list_collection_bindings(
    prim: Usd.Prim, material_purpose: str
) -> Tuple[List[UsdShade.Material], List[Usd.Relationship]]:
    """Returns all the collection-based bindings on the given prim for the specified material purpose."""
    binding_api = UsdShade.MaterialBindingAPI(prim)
    binding_rels = binding_api.GetCollectionBindingRels(material_purpose)
    materials = []
    relationships = []
    for rel in binding_rels:
        target_paths = rel.GetTargets()
        if len(target_paths) != 2:
            continue
        material_prim = prim.GetStage().GetPrimAtPath(target_paths[1])
        if material_prim.IsValid() and material_prim.IsA(UsdShade.Material):
            materials.append(UsdShade.Material(material_prim))
            relationships.append(rel)
    return (materials, relationships)


def batch_set_material_binding_strength(
    prim: Usd.Prim, binding_rels: List[Usd.Relationship], binding_strength: str
) -> bool:
    """Set the material binding strength for multiple binding relationships on a prim.

    Args:
        prim (Usd.Prim): The prim to set the binding strength on.
        binding_rels (List[Usd.Relationship]): A list of binding relationships to set the strength for.
        binding_strength (str): The binding strength to set. Must be either "strongerThanDescendants" or "weakerThanDescendants".

    Returns:
        bool: True if the operation succeeded, False otherwise.
    """
    if binding_strength not in [UsdShade.Tokens.strongerThanDescendants, UsdShade.Tokens.weakerThanDescendants]:
        print(
            f"Invalid binding strength: {binding_strength}. Must be either 'strongerThanDescendants' or 'weakerThanDescendants'."
        )
        return False
    for binding_rel in binding_rels:
        if not binding_rel.IsValid() or not UsdShade.MaterialBindingAPI.CanContainPropertyName(binding_rel.GetName()):
            continue
        if not UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(binding_rel, binding_strength):
            print(f"Failed to set binding strength for relationship: {binding_rel.GetPath()}")
            return False
    return True


def transfer_material_bindings(source_prim: Usd.Prim, dest_prim: Usd.Prim, purpose: Optional[str] = None) -> bool:
    """Transfer material bindings from source_prim to dest_prim.

    If purpose is None, transfer all material bindings. Otherwise, only transfer
    bindings with the specified purpose.

    Returns True if successful, False otherwise.
    """
    source_binding_api = UsdShade.MaterialBindingAPI(source_prim)
    if not source_binding_api:
        return False
    dest_binding_api = UsdShade.MaterialBindingAPI(dest_prim)
    if not dest_binding_api:
        return False
    if purpose is None:
        purposes = source_binding_api.GetMaterialPurposes()
    else:
        purposes = [purpose]
    for purpose in purposes:
        direct_binding = source_binding_api.GetDirectBinding(purpose)
        if direct_binding.GetMaterial():
            dest_binding_api.Bind(
                direct_binding.GetMaterial(),
                bindingStrength=direct_binding.GetBindingStrength(),
                materialPurpose=purpose,
            )
        collection_bindings = source_binding_api.GetCollectionBindings(purpose)
        for binding in collection_bindings:
            dest_binding_api.Bind(
                binding.GetCollection(),
                binding.GetMaterial(),
                binding.GetBindingName(),
                binding.GetBindingStrength(),
                materialPurpose=purpose,
            )
    return True


def resolve_material_hierarchy(prim: Usd.Prim, purpose: str = UsdShade.Tokens.allPurpose) -> Tuple[Usd.Prim, Sdf.Path]:
    """Resolves the Material hierarchy of the given prim and returns the bound Material prim and the binding path.

    Args:
        prim (Usd.Prim): The prim to resolve the Material hierarchy for.
        purpose (str, optional): The material purpose to resolve for. Defaults to UsdShade.Tokens.allPurpose.

    Returns:
        Tuple[Usd.Prim, Sdf.Path]: A tuple with the bound Material prim and the path of the binding relationship.
                                   If no Material is bound, returns (Usd.Prim(), Sdf.Path.emptyPath)
    """
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if not binding_api:
        return (Usd.Prim(), Sdf.Path.emptyPath)
    direct_binding = binding_api.GetDirectBinding(purpose)
    if direct_binding.GetMaterial():
        return (direct_binding.GetMaterial(), direct_binding.GetBindingRel().GetPath())
    collection_bindings = binding_api.GetCollectionBindings(purpose)
    for binding in collection_bindings:
        if binding.GetMaterial():
            return (binding.GetMaterial(), binding.GetBindingRel().GetPath())
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        return resolve_material_hierarchy(parent_prim, purpose)
    else:
        return (Usd.Prim(), Sdf.Path.emptyPath)


def duplicate_material_bindings(prim: Usd.Prim, dst_prim: Usd.Prim) -> bool:
    """Duplicate all material bindings from prim to dst_prim."""
    src_binding_api = UsdShade.MaterialBindingAPI(prim)
    if not src_binding_api:
        return False
    dst_binding_api = UsdShade.MaterialBindingAPI(dst_prim)
    for purpose in UsdShade.MaterialBindingAPI.GetMaterialPurposes():
        direct_binding = src_binding_api.GetDirectBinding(purpose)
        if direct_binding.GetMaterial():
            dst_binding_api.Bind(
                direct_binding.GetMaterial(), purpose=purpose, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
        collection_bindings = src_binding_api.GetCollectionBindings(purpose)
        for binding in collection_bindings:
            dst_binding_api.Bind(
                binding.GetCollection(),
                binding.GetMaterial(),
                binding.GetBindingName(),
                purpose=purpose,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
    return True


def get_material_binding_strength(binding_rel: Usd.Relationship) -> str:
    """Resolves the 'bindMaterialAs' token-valued metadata on the given binding relationship and returns it."""
    binding_strength = binding_rel.GetMetadata("bindMaterialAs")
    if not binding_strength:
        return UsdShade.Tokens.weakerThanDescendants
    allowed_values = [UsdShade.Tokens.weakerThanDescendants, UsdShade.Tokens.strongerThanDescendants]
    if binding_strength not in allowed_values:
        raise ValueError(f"Invalid value for 'bindMaterialAs' metadata: {binding_strength}")
    return binding_strength


def get_material_bind_subsets(prim: Usd.Prim) -> List[UsdGeom.Subset]:
    """Returns all the existing GeomSubsets with familyName=UsdShadeTokens->materialBind below this prim."""
    children = prim.GetChildren()
    material_bind_subsets = []
    for child in children:
        if child.IsA(UsdGeom.Subset):
            subset = UsdGeom.Subset(child)
            if subset.GetFamilyNameAttr().Get() == UsdShade.Tokens.materialBind:
                material_bind_subsets.append(subset)
    return material_bind_subsets


def find_prims_with_material(stage: Usd.Stage, material_path: str) -> List[Usd.Prim]:
    """
    Find all prims in the stage that have a direct binding to the material at the given path.

    Args:
        stage (Usd.Stage): The USD stage to search.
        material_path (str): The path to the material to search for.

    Returns:
        List[Usd.Prim]: A list of prims that have a direct binding to the specified material.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"No material found at path {material_path}")
    prims_with_material = []
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if not binding_api:
            continue
        direct_binding = binding_api.GetDirectBinding()
        if direct_binding.GetMaterial() == material_prim:
            prims_with_material.append(prim)
    return prims_with_material


def clear_material_bindings(prim: Usd.Prim, material_purpose: str = UsdShade.Tokens.allPurpose) -> bool:
    """Unbinds all direct and collection-based bindings on the given prim for the specified material purpose."""
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if not binding_api:
        return False
    direct_unbound = binding_api.UnbindDirectBinding(material_purpose)
    collection_binding_rels = binding_api.GetCollectionBindingRels(material_purpose)
    collection_unbound = True
    for binding_rel in collection_binding_rels:
        binding_name = binding_rel.GetName().split(":")[-1]
        success = binding_api.UnbindCollectionBinding(binding_name, material_purpose)
        collection_unbound = collection_unbound and success
    return direct_unbound and collection_unbound


def material_binding_exists(prim: Usd.Prim, material_purpose: str) -> bool:
    """Check if a material binding exists on the prim for the given material purpose."""
    material_binding_api = UsdShade.MaterialBindingAPI(prim)
    direct_binding_rel = material_binding_api.GetDirectBindingRel(material_purpose)
    if direct_binding_rel.GetTargets():
        return True
    collection_binding_rels = material_binding_api.GetCollectionBindingRels(material_purpose)
    for binding_rel in collection_binding_rels:
        if binding_rel.GetTargets():
            return True
    return False


def unbind_material_from_prims(stage: Usd.Stage, material_path: str, prim_paths: Union[str, List[str]]):
    """Unbinds a material from one or more prims.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.
        prim_paths (Union[str, List[str]]): The path or list of paths to the prims to unbind the material from.

    Raises:
        ValueError: If the material prim or any of the target prims are invalid.
    """
    if isinstance(prim_paths, str):
        prim_paths = [prim_paths]
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} is invalid.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim at path {prim_path} is invalid.")
        binding_api = UsdShade.MaterialBindingAPI(prim)
        bindings = binding_api.GetCollectionBindings()
        for binding in bindings:
            if binding.GetMaterial() == material_prim:
                binding.GetBindingRel().RemoveTargets([material_path])
                break


def unbind_all_materials_from_prim(prim: Usd.Prim) -> bool:
    """Unbind all direct and collection-based material bindings from the given prim."""
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if not binding_api:
        return False
    purposes = UsdShade.MaterialBindingAPI.GetMaterialPurposes()
    for purpose in purposes:
        if not binding_api.UnbindDirectBinding(purpose):
            return False
    collection_bindings = binding_api.GetCollectionBindingRels(UsdShade.Tokens.allPurpose)
    for binding_rel in collection_bindings:
        binding_name = binding_rel.GetName().split(":")[-1]
        for purpose in purposes:
            if not binding_api.UnbindCollectionBinding(binding_name, purpose):
                return False
    return True


def get_resolved_target_path_from_binding(binding_rel: Usd.Relationship) -> Sdf.Path:
    """
    Returns the path of the resolved target identified by the given binding relationship.

    Args:
        binding_rel (Usd.Relationship): The binding relationship to resolve the target path from.

    Returns:
        Sdf.Path: The resolved target path.

    Raises:
        ValueError: If the binding relationship is invalid or has no targets.
    """
    if not binding_rel.IsValid():
        raise ValueError("Invalid binding relationship.")
    targets = binding_rel.GetTargets()
    if not targets:
        raise ValueError("Binding relationship has no targets.")
    resolved_paths = binding_rel.GetForwardedTargets()
    resolved_target_path = resolved_paths[-1]
    return resolved_target_path


def get_material_bindings_for_prims(
    prims: List[Usd.Prim], material_purpose: str
) -> Tuple[List[UsdShade.Material], List[Usd.Relationship]]:
    """
    Get the resolved material bindings for a list of prims and a specific material purpose.

    Args:
        prims (List[Usd.Prim]): A list of prims to compute the resolved material bindings for.
        material_purpose (str): The specific material purpose to compute bindings for.

    Returns:
        A tuple with two elements:
        - A list of resolved material bindings, one for each prim. If a prim has no binding,
          the corresponding entry will be an invalid Material prim.
        - A list of binding relationships. If a prim has no binding, the corresponding entry
          will be an invalid relationship.
    """
    material_bindings = []
    binding_rels = []
    for prim in prims:
        if not prim.IsValid():
            material_bindings.append(UsdShade.Material())
            binding_rels.append(Usd.Relationship())
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        binding_rel = binding_api.GetDirectBindingRel(material_purpose)
        if binding_rel.IsValid():
            material = UsdShade.Material(binding_rel.GetTargets()[0])
            material_bindings.append(material)
            binding_rels.append(binding_rel)
        else:
            collection_binding_rels = binding_api.GetCollectionBindingRels(material_purpose)
            if collection_binding_rels:
                collection_rel = collection_binding_rels[0]
                material_path = collection_rel.GetTargets()[1]
                material = UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path))
                material_bindings.append(material)
                binding_rels.append(collection_rel)
            else:
                material_bindings.append(UsdShade.Material())
                binding_rels.append(Usd.Relationship())
    return (material_bindings, binding_rels)


def resolve_material_binding_conflicts(prim: Usd.Prim, purpose: str) -> Sdf.Path:
    """Resolve material binding conflicts for a prim and purpose.

    This function resolves material binding conflicts by considering the binding
    strength and ancestral bindings. It returns the path to the resolved material.

    Args:
        prim (Usd.Prim): The prim to resolve material bindings for.
        purpose (str): The specific material purpose to resolve bindings for.

    Returns:
        Sdf.Path: The path to the resolved material prim.
    """
    binding_api = UsdShade.MaterialBindingAPI(prim)
    direct_binding = binding_api.GetDirectBinding(purpose)
    if direct_binding.GetMaterial():
        binding_rel = binding_api.GetDirectBindingRel(purpose)
        binding_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(binding_rel)
        if binding_strength == UsdShade.Tokens.strongerThanDescendants:
            return direct_binding.GetMaterial().GetPath()
    collection_bindings = binding_api.GetCollectionBindings(purpose)
    strongest_collection_binding = None
    strongest_binding_strength = None
    for binding in collection_bindings:
        binding_rel = binding.GetBindingRel()
        binding_strength = UsdShade.MaterialBindingAPI.GetMaterialBindingStrength(binding_rel)
        if strongest_collection_binding is None or binding_strength == UsdShade.Tokens.strongerThanDescendants:
            strongest_collection_binding = binding
            strongest_binding_strength = binding_strength
    if strongest_collection_binding:
        return strongest_collection_binding.GetMaterial().GetPath()
    parent_prim = prim.GetParent()
    if parent_prim.IsValid():
        return resolve_material_binding_conflicts(parent_prim, purpose)
    return Sdf.Path()


def test_resolve_material_binding_conflicts():
    stage = Usd.Stage.CreateInMemory()
    world = stage.DefinePrim("/World")
    geo = stage.DefinePrim("/World/Geo")
    sphere = stage.DefinePrim("/World/Geo/Sphere")
    cube = stage.DefinePrim("/World/Geo/Cube")
    material1 = UsdShade.Material.Define(stage, "/Materials/Material1")
    material2 = UsdShade.Material.Define(stage, "/Materials/Material2")
    material3 = UsdShade.Material.Define(stage, "/Materials/Material3")
    geo_binding_api = UsdShade.MaterialBindingAPI(geo)
    geo_binding_api.Bind(material1, UsdShade.Tokens.weakerThanDescendants, UsdShade.Tokens.allPurpose)
    sphere_binding_api = UsdShade.MaterialBindingAPI(sphere)
    sphere_binding_api.Bind(material2, UsdShade.Tokens.strongerThanDescendants, UsdShade.Tokens.allPurpose)
    cube_binding_api = UsdShade.MaterialBindingAPI(cube)
    cube_binding_api.Bind(material3)
    sphere_resolved_material = resolve_material_binding_conflicts(sphere, UsdShade.Tokens.allPurpose)
    cube_resolved_material = resolve_material_binding_conflicts(cube, UsdShade.Tokens.allPurpose)
    world_resolved_material = resolve_material_binding_conflicts(world, UsdShade.Tokens.allPurpose)
    print(f"Sphere resolved material: {sphere_resolved_material}")
    print(f"Cube resolved material: {cube_resolved_material}")
    print(f"World resolved material: {world_resolved_material}")


def batch_unbind_materials(prims: List[Usd.Prim], material_purpose: str) -> List[Tuple[Usd.Prim, bool]]:
    """Unbind direct and collection-based bindings for multiple prims.

    Args:
        prims (List[Usd.Prim]): List of prims to unbind materials from.
        material_purpose (str): The specific material purpose to unbind.

    Returns:
        List[Tuple[Usd.Prim, bool]]: List of tuples containing each prim and
        a boolean indicating if unbinding was successful for that prim.
    """
    results = []
    for prim in prims:
        if not prim.IsValid():
            results.append((prim, False))
            continue
        mat_binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_unbind_success = mat_binding_api.UnbindDirectBinding(material_purpose)
        collection_unbind_success = True
        collection_binding_rels = mat_binding_api.GetCollectionBindingRels(material_purpose)
        for binding_rel in collection_binding_rels:
            binding_name = binding_rel.GetName()
            collection_unbind_success &= mat_binding_api.UnbindCollectionBinding(binding_name, material_purpose)
        unbind_success = direct_unbind_success and collection_unbind_success
        results.append((prim, unbind_success))
    return results


def get_material_purposes() -> List[str]:
    """Return a list of the possible values for the 'material purpose'."""
    purpose_tokens = UsdShade.MaterialBindingAPI.GetMaterialPurposes()
    purposes = [str(token) for token in purpose_tokens]
    return purposes


def add_prim_to_collection_binding(prim: Usd.Prim, bindingName: str, materialPurpose: str) -> bool:
    """
    Adds the specified prim to the collection targeted by the binding relationship
    corresponding to given bindingName and materialPurpose.
    """
    bindingAPI = UsdShade.MaterialBindingAPI(prim)
    if not bindingAPI:
        bindingAPI = UsdShade.MaterialBindingAPI.Apply(prim)
    collBindingRel = bindingAPI.GetCollectionBindingRel(bindingName, materialPurpose)
    if not collBindingRel:
        return True
    collectionPath = collBindingRel.GetTargets()[0]
    collection = Usd.CollectionAPI.Get(prim.GetStage(), collectionPath)
    if not collection:
        raise ValueError(f"Invalid collection targeted by binding {collBindingRel.GetName()}")
    if collection.HasPrim(prim.GetPath()):
        return True
    collection.AddPrim(prim.GetPath())
    return True


def rebind_material_for_prims(
    prims: Sequence[Usd.Prim],
    material: UsdShade.Material,
    binding_name: str = "",
    binding_strength: str = UsdShade.Tokens.fallbackStrength,
    material_purpose: str = UsdShade.Tokens.allPurpose,
) -> bool:
    """Rebind a material to the given prims using UsdShadeMaterialBindingAPI.

    If binding_name is provided, it will do collection based binding using that name.
    Otherwise it will do direct binding.
    """
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Provided material prim is invalid.")
    for prim in prims:
        if not prim.IsValid():
            print(f"Warning: Prim {prim.GetPath()} is invalid. Skipping.")
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if binding_name:
            collection = Usd.CollectionAPI.Apply(prim, binding_name)
            collection.CreateIncludesRel().AddTarget(prim.GetPath())
            success = binding_api.Bind(collection, material, binding_name, binding_strength, material_purpose)
            if not success:
                print(
                    f"Warning: Failed to bind material {material.GetPath()} to collection {binding_name} on prim {prim.GetPath()}."
                )
        else:
            success = binding_api.Bind(material, binding_strength, material_purpose)
            if not success:
                print(f"Warning: Failed to directly bind material {material.GetPath()} to prim {prim.GetPath()}.")
    return True


def add_and_bind_material_to_prims(
    stage: Usd.Stage, prim_paths: List[str], material_path: str, purpose: Optional[str] = None
) -> None:
    """Add a material to the stage and bind it to the specified prims.

    Args:
        stage (Usd.Stage): The stage to add the material to and bind it to the prims.
        prim_paths (List[str]): The paths of the prims to bind the material to.
        material_path (str): The path where the material should be created.
        purpose (Optional[str], optional): The material purpose to use for the binding. Defaults to None.

    Raises:
        ValueError: If any of the prim paths are invalid or if the material path is invalid.
    """
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"Prim path {prim_path} is not valid.")
    material = UsdShade.Material.Define(stage, material_path)
    if not material:
        raise ValueError(f"Failed to create material at path {material_path}.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if purpose is None:
            binding_api.Bind(material)
        else:
            binding_api.Bind(material, materialPurpose=purpose)


def consolidate_materials(stage: Usd.Stage) -> Dict[str, List[str]]:
    """
    Consolidate materials in a USD stage.

    This function finds all material bindings in the stage and consolidates them
    based on the material path. It returns a dictionary where the keys are the
    unique material paths and the values are lists of prim paths that are bound
    to each material.

    Args:
        stage (Usd.Stage): The USD stage to consolidate materials in.

    Returns:
        Dict[str, List[str]]: A dictionary mapping material paths to lists of
        prim paths bound to each material.
    """
    consolidated: Dict[str, List[str]] = {}
    for prim in stage.TraverseAll():
        binding_api = UsdShade.MaterialBindingAPI(prim)
        direct_binding = binding_api.GetDirectBinding()
        if direct_binding:
            material_path = direct_binding.GetMaterialPath()
            prim_path = prim.GetPath().pathString
            if material_path in consolidated:
                consolidated[material_path].append(prim_path)
            else:
                consolidated[material_path] = [prim_path]
    return consolidated


def assign_material_to_hierarchy(stage: Usd.Stage, prim_path: str, material: UsdShade.Material) -> None:
    """Assign a material to a prim and its descendants.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the prim to assign the material to.
        material (UsdShade.Material): The material to assign.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    binding_api = UsdShade.MaterialBindingAPI(prim)
    if not material.GetPrim().IsValid():
        raise ValueError("Invalid material.")
    binding_api.Bind(material)
    for child in prim.GetAllChildren():
        assign_material_to_hierarchy(stage, child.GetPath(), material)


def get_material_assignments(prim: Usd.Prim) -> List[Tuple[str, str, str]]:
    """
    Get the material assignments for a given prim.

    Args:
        prim (Usd.Prim): The prim to retrieve material assignments for.

    Returns:
        List[Tuple[str, str, str]]: A list of tuples containing the material path,
                                    the bound prim path, and the material purpose.
    """
    assignments = []
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    binding_api = UsdShade.MaterialBindingAPI(prim)
    direct_binding = binding_api.GetDirectBinding()
    if direct_binding.GetMaterial():
        material_prim = direct_binding.GetMaterial()
        material_path = material_prim.GetPath().pathString
        bound_prim_path = prim.GetPath().pathString
        material_purpose = direct_binding.GetMaterialPurpose()
        assignments.append((material_path, bound_prim_path, material_purpose))
    return assignments


def reassign_material_based_on_purpose(prim: Usd.Prim, purpose: str, material_path: str) -> bool:
    """Reassign a material to a prim based on the material purpose.

    Args:
        prim (Usd.Prim): The prim to reassign the material on.
        purpose (str): The material purpose to match against.
        material_path (str): The path to the material to assign.

    Returns:
        bool: True if the material was successfully reassigned, False otherwise.
    """
    if not prim.IsValid():
        return False
    binding_api = UsdShade.MaterialBindingAPI(prim)
    direct_binding = binding_api.GetDirectBinding()
    if direct_binding:
        current_material = direct_binding.GetMaterial()
        if (
            current_material
            and current_material.GetPrim().HasAttribute("purpose")
            and (current_material.GetPrim().GetAttribute("purpose").Get() == purpose)
        ):
            return True
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        return False
    material = UsdShade.Material(material_prim)
    material_binding = binding_api.Bind(material)
    if not material_binding:
        return False
    return True


def apply_node_def_api_if_possible(prim: Usd.Prim) -> bool:
    """Apply the UsdShade.NodeDefAPI to the prim if it can be applied.

    Args:
        prim (Usd.Prim): The prim to apply the API to.

    Returns:
        bool: True if the API was applied, False otherwise.
    """
    if not prim.IsValid():
        return False
    if not UsdShade.NodeDefAPI.CanApply(prim):
        return False
    UsdShade.NodeDefAPI.Apply(prim)
    return True


def create_shader_with_id(stage: Usd.Stage, shader_id: str, shader_path: str) -> UsdShade.Shader:
    """Create a shader prim with the given ID and return the shader prim"""
    if not stage:
        raise ValueError("Invalid stage")
    if not shader_id:
        raise ValueError("Invalid shader ID")
    if not shader_path:
        raise ValueError("Invalid shader path")
    shader_prim = UsdShade.Shader.Define(stage, shader_path)
    node_def_api = UsdShade.NodeDefAPI(shader_prim)
    node_def_api.SetShaderId(shader_id)
    return shader_prim


def get_all_shader_ids(node_def_prim: Usd.Prim) -> List[str]:
    """Get all shader IDs for a given NodeDefAPI prim."""
    if not node_def_prim.IsValid():
        raise ValueError("Invalid prim provided.")
    if not UsdShade.NodeDefAPI.CanApply(node_def_prim):
        raise ValueError("NodeDefAPI cannot be applied to the given prim.")
    node_def = UsdShade.NodeDefAPI(node_def_prim)
    impl_source_attr = node_def.GetImplementationSourceAttr()
    if not impl_source_attr or impl_source_attr.Get() != UsdShade.Tokens.id:
        return []
    id_attr = node_def.GetIdAttr()
    if not id_attr:
        return []
    shader_id = id_attr.Get()
    if not shader_id:
        return []
    return [shader_id]


def set_shader_source_code(
    shader_prim: UsdShade.Shader, source_code: str, source_type: str = UsdShade.Tokens.universalSourceType
) -> bool:
    """
    Set the shader's source code for a given source type.

    Args:
        shader_prim (UsdShade.Shader): The shader prim to set the source code for.
        source_code (str): The source code to set.
        source_type (str): The source type to set the source code for. Defaults to UsdShade.Tokens.universalSourceType.

    Returns:
        bool: True if the source code was set successfully, False otherwise.
    """
    if not shader_prim or not shader_prim.GetPrim().IsA(UsdShade.Shader):
        return False
    node_def_api = UsdShade.NodeDefAPI(shader_prim)
    if not node_def_api:
        return False
    impl_source_attr = node_def_api.GetImplementationSourceAttr()
    impl_source_attr.Set(UsdShade.Tokens.sourceCode)
    source_code_attr_name = f"info:{source_type}:sourceCode"
    source_code_attr = shader_prim.CreateInput(source_code_attr_name, Sdf.ValueTypeNames.String)
    success = source_code_attr.Set(source_code)
    return success


def list_all_shader_prim_paths(stage: Usd.Stage) -> List[Sdf.Path]:
    """Return a list of paths to all shader prims in the stage."""
    shader_paths: List[Sdf.Path] = []
    for prim in stage.TraverseAll():
        if UsdShade.NodeDefAPI.CanApply(prim):
            shader_paths.append(prim.GetPath())
    return shader_paths


def fetch_and_validate_shader_id(shader: UsdShade.Shader) -> str:
    """Fetch and validate the shader ID from the given shader.

    Args:
        shader (UsdShade.Shader): The shader to fetch the ID from.

    Returns:
        str: The validated shader ID.

    Raises:
        ValueError: If the shader is invalid or has an invalid implementation source.
    """
    if not shader:
        raise ValueError("Invalid shader.")
    shader_prim = shader.GetPrim()
    shader_def = UsdShade.NodeDefAPI(shader_prim)
    impl_source = shader_def.GetImplementationSource()
    if impl_source != UsdShade.Tokens.id:
        raise ValueError(f"Invalid implementation source: {impl_source}. Expected 'id'.")
    shader_id = shader_def.GetIdAttr().Get()
    if not shader_id:
        raise ValueError("Empty shader ID.")
    return shader_id


def set_shader_source_asset_sub_identifier(shader_prim: UsdShade.Shader, sub_identifier: str, source_type: str) -> bool:
    """
    Set a sub-identifier to be used with a source asset of the given source type.

    This sets the info:<sourceType>:sourceAsset:subIdentifier attribute.
    This also sets the info:implementationSource attribute on the shader to UsdShadeTokens->sourceAsset.

    Args:
        shader_prim (UsdShade.Shader): The shader prim.
        sub_identifier (str): The sub-identifier value.
        source_type (str): The source type token.

    Returns:
        bool: True if the sub-identifier was set successfully, False otherwise.
    """
    if not shader_prim or not shader_prim.GetPrim().IsA(UsdShade.Shader):
        return False
    sub_id_attr_name = f"info:{source_type}:sourceAsset:subIdentifier"
    sub_id_attr = shader_prim.CreateInput(sub_id_attr_name, Sdf.ValueTypeNames.String)
    if not sub_id_attr:
        return False
    sub_id_attr.Set(sub_identifier)
    impl_source_attr = shader_prim.CreateInput(UsdShade.Tokens.infoImplementationSource, Sdf.ValueTypeNames.Token)
    if not impl_source_attr:
        return False
    impl_source_attr.Set(UsdShade.Tokens.sourceAsset)
    return True


def update_shader_id_if_exists(shader_prim: UsdShade.Shader, shader_id: str) -> bool:
    """Update the shader ID if the shader prim has a NodeDefAPI applied."""
    if not shader_prim.GetPrim().IsValid():
        return False
    shader_node_def = UsdShade.NodeDefAPI(shader_prim)
    if not shader_node_def:
        return False
    current_shader_id = ""
    if shader_node_def.GetIdAttr().HasValue():
        current_shader_id = shader_node_def.GetIdAttr().Get()
    if current_shader_id != shader_id:
        shader_node_def.SetShaderId(shader_id)
        return True
    return False


def remove_shader_node_def_api(prim: Usd.Prim) -> bool:
    """Remove the NodeDefAPI from the given prim."""
    if not prim.IsValid():
        raise ValueError(f"Prim {prim.GetPath()} is not valid.")
    if not UsdShade.NodeDefAPI.Get(prim.GetStage(), prim.GetPath()):
        return False
    removed = prim.RemoveAPI(UsdShade.NodeDefAPI)
    return removed


def export_shader_to_file(shader: UsdShade.Shader, file_path: str) -> bool:
    """Export a USD shader to a file.

    Args:
        shader (UsdShade.Shader): The shader to export.
        file_path (str): The file path to export the shader to.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    if not shader.GetPrim().IsValid():
        return False
    stage = Usd.Stage.CreateNew(file_path)
    if not stage:
        return False
    shader_prim_path = shader.GetPath()
    try:
        dest_shader_prim = stage.DefinePrim(shader_prim_path, shader.GetPrim().GetTypeName())
        for attr in shader.GetPrim().GetAttributes():
            dest_shader_prim.CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(attr.Get())
        for rel in shader.GetPrim().GetRelationships():
            dest_shader_prim.CreateRelationship(rel.GetName()).CopyTargets(rel)
        stage.SetDefaultPrim(dest_shader_prim)
        stage.GetRootLayer().Save()
    except Tf.ErrorException as e:
        print(f"Error exporting shader: {e}")
        return False
    return True


def import_shader_from_file(
    stage: Usd.Stage, shader_file_path: str, shader_name: str, shader_prim_path: str
) -> UsdShade.Shader:
    """Import a shader from a USD file and add it to the stage.

    Args:
        stage (Usd.Stage): The stage to add the shader to.
        shader_file_path (str): The file path to the USD file containing the shader.
        shader_name (str): The name of the shader prim in the USD file.
        shader_prim_path (str): The path where the shader should be added in the stage.

    Returns:
        UsdShade.Shader: The imported shader prim.

    Raises:
        ValueError: If the shader file or shader prim does not exist.
    """
    try:
        shader_layer = Sdf.Layer.FindOrOpen(shader_file_path)
    except Tf.ErrorException as e:
        raise ValueError(f"Could not open USD file: {shader_file_path}") from e
    shader_prim_spec = shader_layer.GetPrimAtPath(shader_name)
    if not shader_prim_spec:
        raise ValueError(f"Shader prim '{shader_name}' does not exist in file: {shader_file_path}")
    shader_prim = stage.DefinePrim(shader_prim_path, "Shader")
    shader_prim.GetReferences().AddReference(shader_layer.identifier, shader_name)
    return UsdShade.Shader(shader_prim)


def connect_nodegraph_to_material(
    nodegraph: UsdShade.NodeGraph, material: UsdShade.Material, material_input: str, nodegraph_output: str
) -> None:
    """Connect a nodegraph output to a material input."""
    if not nodegraph:
        raise ValueError("Invalid nodegraph.")
    if not material:
        raise ValueError("Invalid material.")
    material_input_attr = material.GetInput(material_input)
    if not material_input_attr:
        raise ValueError(f"Material input '{material_input}' does not exist.")
    nodegraph_output_attr = nodegraph.GetOutput(nodegraph_output)
    if not nodegraph_output_attr:
        raise ValueError(f"Nodegraph output '{nodegraph_output}' does not exist.")
    success = material_input_attr.ConnectToSource(nodegraph_output_attr)
    if not success:
        raise RuntimeError(
            f"Failed to connect nodegraph output '{nodegraph_output}' to material input '{material_input}'."
        )


def validate_nodegraph(node_graph: UsdShade.NodeGraph) -> bool:
    """Validate a UsdShade.NodeGraph for common issues.

    Checks performed:
    - The node graph has at least one input.
    - The node graph has at least one output.
    - Each input has a connected source or a authored value.
    - Each output has a connected source.

    Returns:
        bool: True if the node graph passes validation, False otherwise.
    """
    inputs = node_graph.GetInputs()
    if len(inputs) == 0:
        print(f"NodeGraph {node_graph.GetPath()} has no inputs")
        return False
    outputs = node_graph.GetOutputs()
    if len(outputs) == 0:
        print(f"NodeGraph {node_graph.GetPath()} has no outputs")
        return False
    for input in inputs:
        if input.HasConnectedSource():
            continue
        if input.Get() is not None:
            continue
        print(f"Input {input.GetFullName()} has no connected source or authored value")
        return False
    for output in outputs:
        if output.HasConnectedSource():
            continue
        print(f"Output {output.GetFullName()} has no connected source")
        return False
    return True


def update_nodegraph_shader(nodegraph: UsdShade.NodeGraph, shader_path: str) -> bool:
    """Update the shader reference on all shader nodes in the node graph.

    Args:
        nodegraph (UsdShade.NodeGraph): The node graph to update.
        shader_path (str): The new shader path to set on all shader nodes.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    if not nodegraph:
        return False
    nodegraph_prim = nodegraph.GetPrim()
    for child_prim in nodegraph_prim.GetChildren():
        shader = UsdShade.Shader(child_prim)
        if shader:
            if not shader.SetShaderId(shader_path):
                return False
    return True


def disconnect_nodegraph_input(node_graph: UsdShade.NodeGraph, input_name: str):
    """Disconnects the specified input on a node graph.

    Args:
        node_graph (UsdShade.NodeGraph): The node graph to disconnect input on.
        input_name (str): Name of the input to disconnect.

    Raises:
        ValueError: If the input does not exist or is not connected.
    """
    if not node_graph:
        raise ValueError("Invalid node graph")
    input_attr = node_graph.GetInput(input_name)
    if not input_attr:
        raise ValueError(f"Input '{input_name}' does not exist on the node graph")
    if not input_attr.HasConnectedSource():
        raise ValueError(f"Input '{input_name}' is not connected")
    UsdShade.ConnectableAPI.DisconnectSource(input_attr)


def create_nodegraph_with_shaders(stage: Usd.Stage, nodegraph_path: str, shader_names: List[str]) -> UsdShade.NodeGraph:
    """Create a NodeGraph prim with connected shader prims.

    Args:
        stage (Usd.Stage): The USD stage to create the prims on.
        nodegraph_path (str): The path for the NodeGraph prim.
        shader_names (List[str]): A list of names for the shader prims to create under the NodeGraph.

    Returns:
        UsdShade.NodeGraph: The created NodeGraph prim.
    """
    nodegraph = UsdShade.NodeGraph.Define(stage, nodegraph_path)
    shaders = []
    for shader_name in shader_names:
        shader_path = f"{nodegraph_path}/{shader_name}"
        shader = UsdShade.Shader.Define(stage, shader_path)
        shaders.append(shader)
    for i, shader in enumerate(shaders):
        output_name = f"out_{i}"
        output = nodegraph.CreateOutput(output_name, Sdf.ValueTypeNames.Color3f)
        shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
        UsdShade.ConnectableAPI.ConnectToSource(output, shader_output)
    return nodegraph


def disconnect_nodegraph_output(nodegraph: UsdShade.NodeGraph, output_name: str) -> bool:
    """Disconnect the specified output on the given NodeGraph.

    Args:
        nodegraph (UsdShade.NodeGraph): The NodeGraph to disconnect the output on.
        output_name (str): The name of the output to disconnect.

    Returns:
        bool: True if the output was successfully disconnected, False otherwise.
    """
    if not nodegraph:
        return False
    output_attr = nodegraph.GetOutput(output_name)
    if not output_attr:
        return False
    if not output_attr.HasConnectedSource():
        return True
    success = output_attr.DisconnectSource()
    return success


def get_nodegraph_shader_connections(nodegraph: UsdShade.NodeGraph) -> Dict[str, List[UsdShade.Shader]]:
    """
    Get a mapping of nodegraph output names to connected shader sources.

    Args:
        nodegraph (UsdShade.NodeGraph): The input nodegraph.

    Returns:
        Dict[str, List[UsdShade.Shader]]: Dictionary mapping each nodegraph
        output name to a list of connected shader sources. Outputs with no
        connections will be excluded from the dictionary.
    """
    result = {}
    for output in nodegraph.GetOutputs():
        output_name = output.GetBaseName()
        shader_sources = []
        if output.HasConnectedSource():
            source = output.GetConnectedSource()
            source_attr = source[0]
            source_prim = source_attr.GetPrim()
            if source_prim.IsA(UsdShade.Shader):
                shader_sources.append(UsdShade.Shader(source_prim))
        if shader_sources:
            result[output_name] = shader_sources
    return result


def connect_nodegraph_inputs(nodegraph: UsdShade.NodeGraph, input_connections: dict) -> None:
    """Connect inputs on a node graph to sources.

    Args:
        nodegraph (UsdShade.NodeGraph): The node graph to connect inputs for.
        input_connections (dict): A dictionary mapping input names to
            (source, outputName) tuples.

    Raises:
        ValueError: If nodegraph is not a valid UsdShade.NodeGraph.
        ValueError: If an input or source in input_connections is not valid.
    """
    if not nodegraph or not isinstance(nodegraph, UsdShade.NodeGraph):
        raise ValueError("Invalid UsdShade.NodeGraph.")
    for input_name, (source_prim, source_output) in input_connections.items():
        input_attr = nodegraph.GetInput(input_name)
        if not input_attr:
            input_attr = nodegraph.CreateInput(input_name, Sdf.ValueTypeNames.Float)
        source_shader = UsdShade.Shader(source_prim)
        if not source_shader:
            raise ValueError(f"Source prim {source_prim.GetPath()} is not a valid UsdShade.Shader.")
        source_output_attr = source_shader.GetOutput(source_output)
        if not source_output_attr:
            raise ValueError(f"Output {source_output} not found on source {source_prim.GetPath()}.")
        input_attr.ConnectToSource(source_output_attr)


def set_shader_output_sdr_metadata_by_key(output: UsdShade.Output, key: str, value: str):
    """Set a key-value pair in the sdrMetadata dictionary of a shader output."""
    if not output:
        raise ValueError("Invalid shader output.")
    if output.HasSdrMetadata():
        sdr_metadata = output.GetSdrMetadata()
    else:
        sdr_metadata = {}
    sdr_metadata[key] = value
    output.SetSdrMetadata(sdr_metadata)


def get_shader_output_sdr_metadata_by_key(output: UsdShade.Output, key: str) -> str:
    """
    Get the value of a specific key in the sdrMetadata dictionary of a shader output.

    Args:
        output (UsdShade.Output): The shader output to retrieve the metadata from.
        key (str): The key to retrieve the value for.

    Returns:
        str: The value associated with the specified key, or an empty string if the key is not found.
    """
    if not output:
        raise ValueError("Invalid shader output")
    if not output.HasSdrMetadata():
        return ""
    value = output.GetSdrMetadataByKey(key)
    return value if value is not None else ""


def connect_shader_output_to_material(
    shader_output: UsdShade.Output, material: UsdShade.Material, output_name: str = "surface"
) -> bool:
    """Connect a shader output to a material input.

    Args:
        shader_output (UsdShade.Output): The shader output to connect.
        material (UsdShade.Material): The material to connect to.
        output_name (str, optional): The name of the material input to connect to. Defaults to "surface".

    Returns:
        bool: True if the connection was made successfully, False otherwise.
    """
    if not shader_output:
        return False
    if not material:
        return False
    material_input = material.GetInput(output_name)
    if not material_input:
        material_input = material.CreateInput(output_name, shader_output.GetTypeName())
    if not material_input:
        return False
    success = material_input.ConnectToSource(shader_output)
    return success


def clear_shader_output_metadata(output: UsdShade.Output, key: Optional[str] = None) -> None:
    """Clear the shader output metadata for the given output.

    If key is provided, only clear the specific metadata entry.
    If key is None, clear all shader output metadata.
    """
    if not output:
        raise ValueError("Invalid output")
    if key is None:
        output.ClearSdrMetadata()
    elif output.HasSdrMetadataByKey(key):
        output.ClearSdrMetadataByKey(key)


def set_shader_output_render_type(output: UsdShade.Output, render_type: str) -> bool:
    """Set the render type for a shader output.

    Args:
        output (UsdShade.Output): The shader output to set the render type for.
        render_type (str): The render type to set.

    Returns:
        bool: True if the render type was set successfully, False otherwise.
    """
    if not output:
        return False
    success = output.SetRenderType(render_type)
    return success


def list_all_shader_outputs(shader: UsdShade.Shader) -> List[UsdShade.Output]:
    """Return a list of all outputs on the given shader."""
    if not shader or not shader.GetPrim().IsA(UsdShade.Shader):
        raise ValueError("Invalid shader input.")
    shader_prim = shader.GetPrim()
    attrs = shader_prim.GetAttributes()
    output_attrs = [attr for attr in attrs if UsdShade.Output.IsOutput(attr)]
    outputs = [UsdShade.Output(attr) for attr in output_attrs]
    return outputs


def set_shader_output_metadata(output: UsdShade.Output, key: str, value: str) -> None:
    """Set metadata key-value pair on a shader output.

    Args:
        output (UsdShade.Output): The shader output to set metadata on.
        key (str): The metadata key.
        value (str): The metadata value.

    Raises:
        ValueError: If the output is invalid.
    """
    if not output:
        raise ValueError("Invalid shader output.")
    output.SetSdrMetadataByKey(key, value)
    if not output.HasSdrMetadataByKey(key):
        raise ValueError(f"Failed to set metadata key '{key}' on the shader output.")


def get_shader_output_metadata(output: UsdShade.Output) -> Dict[str, Any]:
    """Get the metadata dictionary for a shader output."""
    if not output:
        raise ValueError("Invalid shader output.")
    if not output.HasSdrMetadata():
        return {}
    metadata = output.GetSdrMetadata()
    metadata_dict = {key: value for (key, value) in metadata.items()}
    return metadata_dict


def get_shader_output_base_name(output: UsdShade.Output) -> str:
    """Get the base name of a shader output, stripping off the 'outputs:' namespace prefix."""
    full_name = output.GetFullName()
    name_parts = full_name.split(":")
    if len(name_parts) == 2 and name_parts[0] == "outputs":
        return name_parts[1]
    else:
        return full_name


def get_shader_output_full_name(output: UsdShade.Output) -> str:
    """Get the full name of a shader output, including the 'outputs:' prefix."""
    if not output:
        raise ValueError("Invalid shader output")
    output_attr = output.GetAttr()
    if not output_attr:
        raise ValueError("Output attribute is invalid")
    full_name = output_attr.GetName()
    return full_name


def get_shader_output_source(output: UsdShade.Output) -> Optional[Union[UsdShade.Shader, UsdShade.NodeGraph]]:
    """
    Get the source shader or node graph for a shader output.

    Args:
        output (UsdShade.Output): The shader output to query.

    Returns:
        Optional[Union[UsdShade.Shader, UsdShade.NodeGraph]]: The source shader or node graph, or None if not found.
    """
    source_attr = output.GetConnectedSource()
    if not source_attr:
        return None
    source_prim = source_attr.GetPrim()
    if UsdShade.Shader.Get(source_prim.GetStage(), source_prim.GetPath()):
        return UsdShade.Shader(source_prim)
    if UsdShade.NodeGraph.Get(source_prim.GetStage(), source_prim.GetPath()):
        return UsdShade.NodeGraph(source_prim)
    return None


def find_shader_outputs_with_metadata(stage: Usd.Stage, metadata_key: str) -> List[UsdShade.Output]:
    """
    Find all shader outputs on the given stage that have the specified metadata key.

    Args:
        stage (Usd.Stage): The USD stage to search.
        metadata_key (str): The metadata key to search for.

    Returns:
        List[UsdShade.Output]: A list of shader outputs that have the specified metadata key.
    """
    outputs_with_metadata: List[UsdShade.Output] = []
    for prim in stage.Traverse():
        if UsdShade.Shader(prim):
            shader = UsdShade.Shader(prim)
            outputs = shader.GetOutputs()
            for output in outputs:
                if output.HasSdrMetadataByKey(metadata_key):
                    outputs_with_metadata.append(output)
    return outputs_with_metadata


def batch_connect_shader_outputs(
    stage: Usd.Stage, material_path: str, shader_path: str, output_connections: List[Tuple[str, str]]
) -> bool:
    """
    Connect multiple shader outputs to material inputs in a single operation.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.
        shader_path (str): The path to the shader prim.
        output_connections (List[Tuple[str, str]]): A list of tuples representing output-input connections.
                                                     Each tuple contains (shader_output_name, material_input_name).

    Returns:
        bool: True if all connections were successful, False otherwise.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not material_prim.IsValid() or not shader_prim.IsValid():
        print(f"Error: Invalid prim path(s) - material: {material_path}, shader: {shader_path}")
        return False
    if not material_prim.IsA(UsdShade.Material) or not shader_prim.IsA(UsdShade.Shader):
        print(f"Error: Invalid prim type(s) - material: {material_path}, shader: {shader_path}")
        return False
    material = UsdShade.Material(material_prim)
    shader = UsdShade.Shader(shader_prim)
    for shader_output_name, material_input_name in output_connections:
        shader_output = shader.GetOutput(shader_output_name)
        material_input = material.GetInput(material_input_name)
        if not shader_output or not material_input:
            print(
                f"Error: Invalid shader output or material input - output: {shader_output_name}, input: {material_input_name}"
            )
            continue
        success = material_input.ConnectToSource(shader_output)
        if not success:
            print(
                f"Error: Failed to connect shader output {shader_output_name} to material input {material_input_name}"
            )
    return True


def merge_shader_output_metadata(output: UsdShade.Output, metadata: Dict[str, str]) -> None:
    """Merges the given metadata dictionary into the output's existing sdrMetadata.

    Args:
        output (UsdShade.Output): The output to merge metadata into.
        metadata (Dict[str, str]): A dictionary of key-value pairs to merge.

    Raises:
        ValueError: If the given output is not valid.
    """
    if not output.GetAttr().IsValid():
        raise ValueError("Invalid UsdShade.Output object")
    existing_metadata = output.GetSdrMetadata()
    for key, value in metadata.items():
        existing_metadata[key] = value
    output.SetSdrMetadata(existing_metadata)


def set_shader_output_value(
    output: UsdShade.Output, value: Any, time_code: Usd.TimeCode = Usd.TimeCode.Default()
) -> None:
    """Set the value of a shader output attribute at a specific time.

    Args:
        output (UsdShade.Output): The shader output object.
        value (Any): The value to set on the output attribute.
        time_code (Usd.TimeCode, optional): The time code at which to set the value. Defaults to Default time code.

    Raises:
        ValueError: If the output is invalid or if setting the value fails.
    """
    if not output:
        raise ValueError("Invalid shader output.")
    attr = output.GetAttr()
    if not attr:
        raise ValueError("Failed to get attribute from shader output.")
    success = output.Set(value, time_code)
    if not success:
        raise ValueError("Failed to set value on shader output.")


def is_shader_output_connected(output: UsdShade.Output) -> bool:
    """Check if a shader output is connected to a source.

    Args:
        output (UsdShade.Output): The shader output to check.

    Returns:
        bool: True if the output is connected, False otherwise.
    """
    if not output:
        raise ValueError("Invalid shader output.")
    if output.HasConnectedSource():
        return True
    else:
        return False


def set_shader_output_animation(output: UsdShade.Output, keyframes: Dict[float, float]) -> bool:
    """Set keyframe animation on a shader output.

    Args:
        output (UsdShade.Output): The shader output to animate.
        keyframes (Dict[float, float]): A dictionary mapping time values (in frames) to output values.

    Returns:
        bool: True if the animation was set successfully, False otherwise.
    """
    if not output.GetAttr().IsValid():
        return False
    attr = output.GetAttr()
    attr.Clear()
    try:
        for time, value in keyframes.items():
            attr.Set(value, Usd.TimeCode(time))
    except Tf.ErrorException as e:
        print(f"Error setting keyframe animation: {e}")
        return False
    return True


def set_shader_output_visibility(output: UsdShade.Output, visibility: bool) -> bool:
    """Set the visibility attribute of a shader output.

    Args:
        output (UsdShade.Output): The shader output to set visibility for.
        visibility (bool): The visibility value to set.

    Returns:
        bool: True if the visibility was set successfully, False otherwise.
    """
    if not output:
        print(f"Error: Invalid shader output")
        return False
    output_attr = output.GetAttr()
    if not output_attr:
        print(f"Error: Failed to get output attribute for {output.GetFullName()}")
        return False
    try:
        output_attr.SetMetadata("hidden", not visibility)
        return True
    except Tf.ErrorException as e:
        print(f"Error: Failed to set visibility for {output.GetFullName()}: {e}")
        return False


def get_all_shader_output_values(shader: UsdShade.Shader) -> dict:
    """Get all output values for a given shader.

    Args:
        shader (UsdShade.Shader): The shader to get output values from.

    Returns:
        dict: A dictionary mapping output names to their values.
    """
    if not shader or not shader.GetPrim().IsA(UsdShade.Shader):
        raise ValueError("Invalid shader provided.")
    outputs = shader.GetOutputs()
    output_values = {}
    for output in outputs:
        output_name = output.GetBaseName()
        output_attr = output.GetAttr()
        output_value = output_attr.Get()
        output_values[output_name] = output_value
    return output_values


def get_shader_output_type(output: UsdShade.Output) -> str:
    """Get the type of a shader output, considering the renderType if set."""
    if not output:
        raise ValueError("Invalid shader output")
    render_type = output.GetRenderType()
    if render_type:
        return render_type
    type_name = output.GetTypeName()
    if not type_name:
        return "float"
    return str(type_name)


def replace_shader_output_connection(
    output: UsdShade.Output, old_source: UsdShade.ConnectableAPI, new_source: UsdShade.ConnectableAPI
) -> bool:
    """Replace a connection to a shader output with a new source.

    Args:
        output (UsdShade.Output): The output to replace the connection for.
        old_source (UsdShade.ConnectableAPI): The old source to be replaced.
        new_source (UsdShade.ConnectableAPI): The new source to connect to the output.

    Returns:
        bool: True if the connection was successfully replaced, False otherwise.
    """
    if not output:
        return False
    connected_sources = output.GetConnectedSources()
    if old_source not in [src.source for src in connected_sources[0]]:
        return False
    output.DisconnectSource(old_source)
    success = output.ConnectToSource(new_source)
    return success


def clear_shader_output_sdr_metadata_by_key(output: UsdShade.Output, key: str) -> bool:
    """Clear the sdrMetadata value for a specific key on a shader output.

    Args:
        output (UsdShade.Output): The shader output to clear the sdrMetadata from.
        key (str): The key in the sdrMetadata to clear.

    Returns:
        bool: True if the sdrMetadata was successfully cleared, False otherwise.
    """
    if not output:
        return False
    if not output.HasSdrMetadataByKey(key):
        return False
    output.ClearSdrMetadataByKey(key)
    return True


def disconnect_all_shader_outputs(shader: UsdShade.Shader) -> None:
    """Disconnect all outputs from a shader.

    Args:
        shader (UsdShade.Shader): The shader to disconnect outputs from.
    """
    for output in shader.GetOutputs():
        if output.HasConnectedSource():
            output.DisconnectSource()


def get_shader_output_value_at_time(output: UsdShade.Output, time_code: Usd.TimeCode = Usd.TimeCode.Default()):
    """
    Get the value of a shader output at a specific time.

    Args:
        output (UsdShade.Output): The shader output to retrieve the value from.
        time_code (Usd.TimeCode, optional): The time code at which to retrieve the value. Defaults to Usd.TimeCode.Default().

    Returns:
        The value of the shader output at the specified time, or None if the output has no value.

    Raises:
        ValueError: If the provided output is not a valid UsdShade.Output.
    """
    if not output:
        raise ValueError("Invalid shader output provided.")
    attr = output.GetAttr()
    if attr.HasValue():
        value = attr.Get(time_code)
        return value
    return None


def get_connected_shaders(output: UsdShade.Output) -> List[UsdShade.Shader]:
    """
    Get the list of shaders connected to the given output.

    Args:
        output (UsdShade.Output): The output to query for connected shaders.

    Returns:
        List[UsdShade.Shader]: The list of connected shaders.
    """
    connected_shaders = []
    output_attr = output.GetAttr()
    if not output_attr:
        return connected_shaders
    connected_sources = output.GetConnectedSources()
    for source_info in connected_sources[0]:
        source_attr = source_info.source
        source_prim = source_attr.GetPrim()
        if UsdShade.Shader(source_prim):
            shader = UsdShade.Shader(source_prim)
            connected_shaders.append(shader)
        if UsdShade.ConnectableAPI(source_prim).GetOutputs():
            source_outputs = UsdShade.ConnectableAPI(source_prim).GetOutputs()
            for source_output in source_outputs:
                connected_shaders.extend(get_connected_shaders(source_output))
    return connected_shaders


def copy_shader_output(src_output: UsdShade.Output, dst_material: UsdShade.Material) -> UsdShade.Output:
    """Copy a shader output to a new material.

    Args:
        src_output (UsdShade.Output): The source output to copy.
        dst_material (UsdShade.Material): The destination material to copy the output to.

    Returns:
        UsdShade.Output: The newly created output on the destination material.
    """
    if not src_output.GetAttr().IsDefined():
        raise ValueError("Source output attribute is not defined.")
    if not dst_material.GetPrim().IsValid():
        raise ValueError("Destination material prim is not valid.")
    src_attr = src_output.GetAttr()
    src_type = src_attr.GetTypeName()
    src_name = src_output.GetBaseName()
    dst_output = dst_material.CreateOutput(src_name, src_type)
    src_sources = src_output.GetConnectedSources()
    if src_sources:
        dst_output.SetConnectedSources(src_sources[0])
    if src_output.HasRenderType():
        dst_output.SetRenderType(src_output.GetRenderType())
    if src_output.HasSdrMetadata():
        dst_output.SetSdrMetadata(src_output.GetSdrMetadata())
    src_val = src_attr.Get()
    if src_val is not None:
        dst_output.Set(src_val)
    return dst_output


def connect_shader_output_to_multiple_sources(output: UsdShade.Output, sources: List[UsdShade.Output]) -> bool:
    """Connect a shader output to multiple sources."""
    if not output:
        raise ValueError("Invalid output")
    if not sources:
        raise ValueError("Empty sources list")
    source_infos = []
    for source in sources:
        if not source:
            continue
        source_info = UsdShade.ConnectionSourceInfo(source)
        source_infos.append(source_info)
    success = output.SetConnectedSources(source_infos)
    return success


def find_unused_shader_outputs(material: UsdShade.Material) -> list[UsdShade.Output]:
    """
    Find all shader outputs on the given material that are not connected to any inputs.

    Args:
        material (UsdShade.Material): The material to search for unused outputs.

    Returns:
        list[UsdShade.Output]: A list of unused shader outputs.
    """
    unused_outputs = []
    surface_output = material.GetSurfaceOutput()
    if not surface_output:
        return unused_outputs
    shader_prim = surface_output.GetConnectedSource()[0]
    if not shader_prim:
        return unused_outputs
    for output in UsdShade.Shader(shader_prim).GetOutputs():
        if not output.HasConnectedSource():
            unused_outputs.append(output)
    return unused_outputs


def copy_shader_output_connections(source_shader: UsdShade.Shader, dest_shader: UsdShade.Shader) -> None:
    """Copy shader output connections from one shader to another.

    Args:
        source_shader (UsdShade.Shader): The source shader to copy from.
        dest_shader (UsdShade.Shader): The destination shader to copy to.
    """
    for output in source_shader.GetOutputs():
        (sources, invalid_paths) = output.GetConnectedSources()
        dest_output = dest_shader.GetOutput(output.GetBaseName())
        if not dest_output:
            continue
        dest_output.ClearSources()
        for source in sources:
            source_attr = UsdShade.Output(source.source.GetPrim().GetAttribute(source.sourceName))
            dest_output.ConnectToSource(source_attr)


def create_texture_shader(
    stage: Usd.Stage, shader_name: str, texture_path: str, shader_id: str = "UsdUVTexture"
) -> UsdShade.Shader:
    """Create a texture shader prim.

    Args:
        stage (Usd.Stage): The stage to create the shader on.
        shader_name (str): The name of the shader prim.
        texture_path (str): The path to the texture file.
        shader_id (str, optional): The shader ID. Defaults to "UsdUVTexture".

    Returns:
        UsdShade.Shader: The created texture shader.
    """
    if not stage:
        raise ValueError("Invalid stage")
    if not shader_name:
        raise ValueError("Invalid shader name")
    if not texture_path:
        raise ValueError("Invalid texture path")
    shader_path = f"/Looks/{shader_name}"
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr(shader_id)
    file_input = shader.CreateInput("file", Sdf.ValueTypeNames.Asset)
    file_input.Set(texture_path)
    fallback_input = shader.CreateInput("fallback", Sdf.ValueTypeNames.Float4)
    fallback_input.Set((0.0, 0.0, 0.0, 1.0))
    shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader.CreateOutput("r", Sdf.ValueTypeNames.Float)
    shader.CreateOutput("g", Sdf.ValueTypeNames.Float)
    shader.CreateOutput("b", Sdf.ValueTypeNames.Float)
    shader.CreateOutput("a", Sdf.ValueTypeNames.Float)
    return shader


def batch_update_shader_params(shader: UsdShade.Shader, params: Dict[str, float]) -> None:
    """Update multiple shader parameters in a single transaction.

    Args:
        shader (UsdShade.Shader): The shader to update parameters for.
        params (Dict[str, float]): A dictionary mapping parameter names to their new values.

    Raises:
        ValueError: If the shader is invalid or any of the parameter names are invalid.
    """
    if not shader:
        raise ValueError("Invalid shader provided.")
    with Sdf.ChangeBlock():
        for param_name, param_value in params.items():
            param_attr = shader.GetInput(param_name)
            if not param_attr:
                raise ValueError(f"Invalid shader parameter: {param_name}")
            param_attr.Set(param_value)


def convert_shader_to_mdl(shader: UsdShade.Shader) -> str:
    """Converts a USD shader to an MDL material definition."""
    if not shader or not shader.GetPrim().IsValid():
        raise ValueError("Invalid UsdShade.Shader object")
    shader_id = shader.GetIdAttr().Get()
    if not shader_id:
        raise ValueError("Shader ID not found")
    mdl_material = f"mdl {shader_id} {{"
    for input in shader.GetInputs():
        input_name = input.GetBaseName()
        input_type = input.GetTypeName().cppTypeName
        input_value = input.Get()
        if input_value is not None:
            input_value_str = str(input_value)
        elif input_type == "float":
            input_value_str = "0.0"
        elif input_type == "color":
            input_value_str = "color(0.0, 0.0, 0.0)"
        elif input_type == "vector":
            input_value_str = "vector(0.0, 0.0, 0.0)"
        else:
            input_value_str = ""
        mdl_material += f"    {input_type} {input_name} = {input_value_str};\n"
    mdl_material += "}\n"
    return mdl_material


def create_shader_with_inputs(
    stage: Usd.Stage, shader_path: str, shader_id: str, input_names_and_types: Dict[str, Sdf.ValueTypeName]
) -> UsdShade.Shader:
    """
    Create a shader prim with the given inputs.

    Args:
        stage (Usd.Stage): The stage to create the shader on.
        shader_path (str): The path to create the shader at.
        shader_id (str): The identifier of the shader.
        input_names_and_types (Dict[str, Sdf.ValueTypeName]): A dictionary mapping input names to their value types.

    Returns:
        UsdShade.Shader: The created shader prim.
    """
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.SetShaderId(shader_id)
    for input_name, input_type in input_names_and_types.items():
        shader.CreateInput(input_name, input_type)
    return shader


def update_shader_metadata(shader: UsdShade.Shader, metadata: Dict[str, str]) -> None:
    """Update the sdrMetadata on a USD shader with the provided key-value pairs."""
    if not shader or not shader.GetPrim().IsValid():
        raise ValueError("Invalid shader object provided.")
    current_metadata = shader.GetSdrMetadata()
    for key, value in metadata.items():
        current_metadata[key] = value
    shader.SetSdrMetadata(current_metadata)


def find_shaders_with_metadata(stage: Usd.Stage, metadata_key: str) -> List[UsdShade.Shader]:
    """
    Find all shader prims on the given stage that have a specific metadata key.

    Args:
        stage (Usd.Stage): The USD stage to search for shaders.
        metadata_key (str): The metadata key to search for on shaders.

    Returns:
        List[UsdShade.Shader]: A list of shader prims that have the specified metadata key.
    """
    shaders = []
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            if shader.HasSdrMetadataByKey(metadata_key):
                shaders.append(shader)
    return shaders


def replace_shader_source_code(shader: UsdShade.Shader, source_code: str, source_type: str = "glslfx") -> None:
    """Replace the source code of a UsdShadeShader.

    Args:
        shader (UsdShade.Shader): The shader to update.
        source_code (str): The new source code.
        source_type (str, optional): The type of the source code. Defaults to "glslfx".
    """
    if not shader:
        raise ValueError("Invalid shader")
    if not source_code:
        raise ValueError("Source code cannot be empty")
    if source_type not in ["glslfx", "osl"]:
        raise ValueError(f"Unsupported source type: {source_type}")
    if not shader.SetSourceCode(sourceCode=source_code, sourceType=source_type):
        raise RuntimeError(f"Failed to set source code on shader: {shader.GetPath()}")


def reassign_shader_to_prims(stage: Usd.Stage, shader_path: str, prim_paths: List[str]) -> None:
    """Reassigns a shader to a list of prims.

    Args:
        stage (Usd.Stage): The USD stage.
        shader_path (str): The path to the shader prim.
        prim_paths (List[str]): The paths to the prims to reassign the shader to.
    """
    shader_prim = stage.GetPrimAtPath(shader_path)
    if not shader_prim.IsValid():
        raise ValueError(f"Shader prim at path {shader_path} does not exist.")
    shader = UsdShade.Shader(shader_prim)
    if not shader:
        raise ValueError(f"Prim at path {shader_path} is not a valid shader.")
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"Warning: Prim at path {prim_path} does not exist. Skipping.")
            continue
        binding_api = UsdShade.MaterialBindingAPI(prim)
        if not binding_api:
            print(f"Warning: Prim at path {prim_path} does not have a material binding API. Skipping.")
            continue
        binding_api.Bind(shader)


def find_unconnected_shader_outputs(stage: Usd.Stage) -> Dict[Sdf.Path, List[str]]:
    """
    Find all shader prims on the given stage that have unconnected outputs.

    Returns a dictionary mapping shader prim paths to lists of unconnected output names.
    """
    unconnected_outputs = {}
    for prim in stage.TraverseAll():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            outputs = shader.GetOutputs()
            unconnected_shader_outputs = []
            for output in outputs:
                if not output.HasConnectedSource():
                    unconnected_shader_outputs.append(output.GetBaseName())
            if unconnected_shader_outputs:
                unconnected_outputs[prim.GetPath()] = unconnected_shader_outputs
    return unconnected_outputs


def transfer_shader_params(source_shader: UsdShade.Shader, dest_shader: UsdShade.Shader):
    """
    Transfers shader parameters from the source shader to the destination shader.

    Args:
        source_shader (UsdShade.Shader): The source shader to transfer parameters from.
        dest_shader (UsdShade.Shader): The destination shader to transfer parameters to.

    Returns:
        None
    """
    source_inputs = source_shader.GetInputs()
    for source_input in source_inputs:
        input_name = source_input.GetBaseName()
        input_type = source_input.GetTypeName()
        dest_input = dest_shader.GetInput(input_name)
        if dest_input:
            if dest_input.GetTypeName() == input_type:
                input_value = source_input.Get()
                if input_value is not None:
                    dest_input.Set(input_value)
            else:
                print(
                    f"Warning: Input '{input_name}' type mismatch. Source type: {input_type}, Destination type: {dest_input.GetTypeName()}"
                )
        else:
            dest_shader.CreateInput(input_name, input_type)
            input_value = source_input.Get()
            if input_value is not None:
                dest_shader.GetInput(input_name).Set(input_value)


def animate_shader_param(shader: UsdShade.Shader, param_name: str, keyframes: Dict[float, Any]):
    """Animate a shader parameter over time using the given keyframes.

    Args:
        shader (UsdShade.Shader): The shader prim to animate a parameter on.
        param_name (str): The name of the parameter to animate.
        keyframes (Dict[float, Any]): A dictionary mapping time values (in seconds) to parameter values.

    Returns:
        None
    """
    param_attr = shader.GetInput(param_name)
    if not param_attr:
        raise ValueError(f"Shader parameter '{param_name}' not found on shader prim '{shader.GetPath()}'")
    attr = param_attr.GetAttr()
    if attr.HasAuthoredValue():
        attr.Clear()
    for time, value in keyframes.items():
        param_attr.Set(value, Usd.TimeCode(time))


def create_shader_from_preset(stage: Usd.Stage, prim_path: str, shader_id: str) -> UsdShade.Shader:
    """Create a shader prim from a preset shader ID.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path where the shader prim should be created.
        shader_id (str): The ID of the preset shader.

    Returns:
        UsdShade.Shader: The created shader prim.

    Raises:
        ValueError: If the prim path is invalid or the shader ID is not set.
    """
    if not Sdf.Path.IsValidPathString(prim_path):
        raise ValueError(f"Invalid prim path: {prim_path}")
    shader_prim = UsdShade.Shader.Define(stage, Sdf.Path(prim_path))
    if not shader_prim.SetShaderId(shader_id):
        raise ValueError(f"Failed to set shader ID: {shader_id}")
    return shader_prim


def create_material_with_shader(
    stage: Usd.Stage, material_path: str, shader_id: str, shader_inputs: Dict[str, Any]
) -> UsdShade.Material:
    """Create a new material with a shader and inputs.

    Args:
        stage (Usd.Stage): The USD stage to create the material on.
        material_path (str): The path where the material will be created.
        shader_id (str): The identifier of the shader to use, e.g. "UsdPreviewSurface".
        shader_inputs (Dict[str, Any]): A dictionary of input names and values for the shader.

    Returns:
        UsdShade.Material: The created material prim.

    Raises:
        ValueError: If the material path is invalid or the shader ID is not recognized.
    """
    material_prim_path = Sdf.Path(material_path)
    if not material_prim_path.IsPrimPath():
        raise ValueError(f"Invalid material path: {material_path}")
    material = UsdShade.Material.Define(stage, material_prim_path)
    shader_prim = UsdShade.Shader.Define(stage, material_prim_path.AppendChild("Shader"))
    shader_prim.CreateIdAttr(shader_id)
    for input_name, input_value in shader_inputs.items():
        shader_input = shader_prim.CreateInput(input_name, Sdf.ValueTypeNames.Float3)
        shader_input.Set(input_value)
    material.CreateSurfaceOutput().ConnectToSource(shader_prim.ConnectableAPI(), "surface")
    return material


def create_shader_network_from_material(stage: Usd.Stage, material_path: str) -> UsdShade.Material:
    """Create a shader network from a material path."""
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    material = UsdShade.Material(material_prim)
    if not material:
        raise ValueError(f"Prim at path {material_path} is not a valid Material.")
    surface_output = material.GetOutput(UsdShade.Tokens.surface)
    if not surface_output:
        raise ValueError(f"Material at path {material_path} does not have a surface output.")
    shader_prim = surface_output.GetConnectedSource()[0]
    if not shader_prim:
        raise ValueError(f"Surface output of material at path {material_path} is not connected to a shader.")
    shader_network_material = UsdShade.Material.Define(stage, Sdf.Path(f"{material_path}_ShaderNetwork"))
    shader = UsdShade.Shader(shader_prim)
    shader_network_material.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput(UsdShade.Tokens.surface, Sdf.ValueTypeNames.Token)
    )
    return shader_network_material


def convert_shader_to_custom_type(shader: UsdShade.Shader, custom_type_name: str) -> bool:
    """Convert a shader to a custom type.

    Args:
        shader (UsdShade.Shader): The shader to convert.
        custom_type_name (str): The name of the custom type to convert to.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not shader.GetPrim().IsValid():
        return False
    if not Sdf.Path.IsValidIdentifier(custom_type_name):
        return False
    prim_path = shader.GetPath()
    stage = shader.GetPrim().GetStage()
    typed_prim = stage.DefinePrim(prim_path)
    typed_prim.SetTypeName(custom_type_name)
    for attr in shader.GetPrim().GetAttributes():
        attr_name = attr.GetName()
        attr_value = attr.Get()
        typed_prim.CreateAttribute(attr_name, attr.GetTypeName()).Set(attr_value)
    for rel in shader.GetPrim().GetRelationships():
        rel_name = rel.GetName()
        rel_targets = rel.GetTargets()
        typed_prim.CreateRelationship(rel_name).SetTargets(rel_targets)
    return True


def merge_shaders(shaders: List[UsdShade.Shader]) -> UsdShade.Shader:
    """Merges multiple UsdShade.Shader prims into a single shader prim.

    The merged shader will have all the inputs and outputs from the source shaders.
    If there are name conflicts in inputs or outputs, they will be uniquified by
    appending a number suffix. The source shaders are left unmodified.

    Args:
        shaders (List[UsdShade.Shader]): The list of shader prims to merge.

    Returns:
        UsdShade.Shader: The merged shader prim.
    """
    if not shaders:
        raise ValueError("At least one shader must be provided for merging.")
    stage = shaders[0].GetPrim().GetStage()
    merged_shader = UsdShade.Shader.Define(stage, "/MergedShader")
    input_names = []
    output_names = []
    for shader in shaders:
        input_names.extend([i.GetBaseName() for i in shader.GetInputs()])
        output_names.extend([o.GetBaseName() for o in shader.GetOutputs()])
    for name in set(input_names):
        uniquified_name = name
        suffix = 1
        while merged_shader.GetInput(uniquified_name):
            uniquified_name = f"{name}_{suffix}"
            suffix += 1
        input_type = None
        for shader in shaders:
            if shader.GetInput(name):
                input_type = shader.GetInput(name).GetTypeName()
                break
        if input_type:
            merged_shader.CreateInput(uniquified_name, input_type)
    for name in set(output_names):
        uniquified_name = name
        suffix = 1
        while merged_shader.GetOutput(uniquified_name):
            uniquified_name = f"{name}_{suffix}"
            suffix += 1
        output_type = None
        for shader in shaders:
            if shader.GetOutput(name):
                output_type = shader.GetOutput(name).GetTypeName()
                break
        if output_type:
            merged_shader.CreateOutput(uniquified_name, output_type)
    return merged_shader


def consolidate_shader_inputs(shader: UsdShade.Shader) -> Dict[str, UsdShade.Input]:
    """
    Consolidates all the inputs of a shader into a dictionary.

    This function retrieves all the authored inputs of a shader, including
    inherited inputs, and returns them as a dictionary mapping input names
    to Input objects.

    Parameters:
        - shader (UsdShade.Shader): The shader to consolidate inputs for.

    Returns:
        - Dict[str, UsdShade.Input]: A dictionary mapping input names to Input objects.
    """
    authored_input_names = [input.GetBaseName() for input in shader.GetInputs()]
    consolidated_inputs = {}
    for input_name in authored_input_names:
        input_obj = shader.GetInput(input_name)
        if input_obj:
            consolidated_inputs[input_name] = input_obj
        else:
            continue
    parent = shader.GetPrim().GetParent()
    if parent.IsValid():
        parent_shader = UsdShade.Shader(parent)
        parent_inputs = consolidate_shader_inputs(parent_shader)
        consolidated_inputs.update(parent_inputs)
    return consolidated_inputs


def list_shader_variations(shader: UsdShade.Shader) -> List[Dict[str, str]]:
    """
    Returns a list of dictionaries representing the shader's variations.

    Each dictionary contains the following keys:
    - "id": The unique identifier of the shader variation.
    - "info:id": The identifier of the shader variation's info dictionary.
    - "info:sourceAsset": The source asset path of the shader variation.
    - "info:sourceCode": The source code of the shader variation.
    """
    variations = []
    if not shader or not shader.GetPrim().IsValid():
        return variations
    info_id_attr = shader.GetPrim().GetAttribute("info:id")
    if info_id_attr.IsValid():
        variant_sets = shader.GetPrim().GetVariantSets()
        if variant_sets.GetNames():
            for variant_name in variant_sets.GetNames():
                variant_set = shader.GetPrim().GetVariantSet(variant_name)
                variant_names = variant_set.GetVariantNames()
                for variant in variant_names:
                    variant_set.SetVariantSelection(variant)
                    shader_id = info_id_attr.Get()
                    source_asset = shader.GetPrim().GetAttribute("info:sourceAsset").Get()
                    source_code = shader.GetPrim().GetAttribute("info:sourceCode").Get()
                    variation_dict = {
                        "id": shader_id,
                        "info:id": variant,
                        "info:sourceAsset": source_asset.path if source_asset else "",
                        "info:sourceCode": source_code,
                    }
                    variations.append(variation_dict)
    return variations


def organize_shaders_by_type(stage: Usd.Stage) -> Dict[str, List[UsdShade.Shader]]:
    """Organizes all shaders in the stage by their type.

    Returns a dictionary where the keys are shader type names and the values
    are lists of UsdShade.Shader prims of that type.
    """
    shader_prims = [p for p in stage.Traverse() if p.IsA(UsdShade.Shader)]
    shader_dict: Dict[str, List[UsdShade.Shader]] = {}
    for shader_prim in shader_prims:
        shader = UsdShade.Shader(shader_prim)
        type_name = shader.GetImplementationSource()
        if type_name not in shader_dict:
            shader_dict[type_name] = []
        shader_dict[type_name].append(shader)
    return shader_dict


def get_shader_network_bounding_box(shader: UsdShade.Shader) -> Tuple[Gf.Vec3f, Gf.Vec3f]:
    """
    Get the bounding box of the geometry that the given shader is bound to.

    Args:
        shader (UsdShade.Shader): The shader to get the bounding box for.

    Returns:
        A tuple containing two Gf.Vec3f representing the minimum and maximum points of the bounding box.
        If no geometry is found, returns ((0, 0, 0), (0, 0, 0)).
    """
    material = shader.GetPrim().GetParent()
    if not material:
        raise ValueError("Shader is not contained in a material.")
    bindings = []
    for prim in material.GetPrim().GetStage().TraverseAll():
        if prim.IsA(UsdGeom.Gprim):
            binding_api = UsdShade.MaterialBindingAPI(prim)
            if binding_api.GetDirectBinding() == material:
                bindings.append(prim)
    if not bindings:
        return (Gf.Vec3f(0, 0, 0), Gf.Vec3f(0, 0, 0))
    bbox_min = Gf.Vec3f(float("inf"))
    bbox_max = Gf.Vec3f(float("-inf"))
    for prim in bindings:
        gprim = UsdGeom.Gprim(prim)
        extent = gprim.ComputeExtent(Usd.TimeCode.Default())
        if extent.IsEmpty():
            continue
        bbox_min = Gf.Min(bbox_min, extent.GetMin())
        bbox_max = Gf.Max(bbox_max, extent.GetMax())
    return (bbox_min, bbox_max)


def connect_shader_to_material(
    material: UsdShade.Material,
    shader: UsdShade.Shader,
    shader_output: str = "surface",
    material_input: str = "surface",
) -> None:
    """Connect a shader's output to a material's input.

    Args:
        material (UsdShade.Material): The material to connect the shader to.
        shader (UsdShade.Shader): The shader to connect to the material.
        shader_output (str, optional): The name of the shader's output. Defaults to "surface".
        material_input (str, optional): The name of the material's input. Defaults to "surface".

    Raises:
        ValueError: If the shader or material is invalid.
    """
    if not shader or not shader.GetPrim().IsValid():
        raise ValueError("Invalid shader.")
    if not material or not material.GetPrim().IsValid():
        raise ValueError("Invalid material.")
    shader_output_attr = shader.GetOutput(shader_output)
    if not shader_output_attr:
        shader_output_attr = shader.CreateOutput(shader_output, Sdf.ValueTypeNames.Token)
    material_input_attr = material.GetInput(material_input)
    if not material_input_attr:
        material_input_attr = material.CreateInput(material_input, Sdf.ValueTypeNames.Token)
    material_input_attr.ConnectToSource(shader_output_attr)


def optimize_shader_network(shader_network: Usd.Stage) -> Usd.Stage:
    """Optimize a shader network by removing unconnected nodes.

    Args:
        shader_network (Usd.Stage): The input shader network stage to optimize.

    Returns:
        Usd.Stage: The optimized shader network stage.
    """
    optimized_stage = Usd.Stage.CreateInMemory()
    optimized_stage.GetRootLayer().TransferContent(shader_network.GetRootLayer())
    prims_to_remove = []
    for prim in optimized_stage.TraverseAll():
        if prim.IsA(UsdShade.Shader):
            shader = UsdShade.Shader(prim)
            has_connections = False
            for input in shader.GetInputs():
                if input.HasConnectedSource():
                    has_connections = True
                    break
            if not has_connections:
                for output in shader.GetOutputs():
                    if output.HasConnectedSource():
                        has_connections = True
                        break
            if not has_connections:
                prims_to_remove.append(prim)
    for prim in prims_to_remove:
        prim.SetActive(False)
        for child in prim.GetAllChildren():
            child.SetActive(False)
        optimized_stage.RemovePrim(prim.GetPath())
    return optimized_stage


def parse_shader_definitions(stage: Usd.Stage, shader_def_paths: List[str]) -> List[Ndr.Node]:
    """Parse shader definitions from USD stage.

    Args:
        stage (Usd.Stage): The USD stage to parse shader definitions from.
        shader_def_paths (List[str]): The paths to the shader definitions to parse.

    Returns:
        List[Ndr.Node]: The parsed shader nodes.
    """
    parser = UsdShade.ShaderDefParserPlugin()
    parsed_nodes = []
    for shader_def_path in shader_def_paths:
        shader_def_prim = stage.GetPrimAtPath(shader_def_path)
        if not shader_def_prim.IsValid():
            continue
        discovery_result = Ndr.NodeDiscoveryResult(
            shader_def_prim.GetName(),
            shader_def_prim.GetTypeName(),
            shader_def_prim.GetPath().pathString,
            parser.GetSourceType(),
            parser.GetDiscoveryTypes()[0],
        )
        shader_node = parser.Parse(discovery_result)
        if shader_node:
            parsed_nodes.append(shader_node)
    return parsed_nodes


def update_shader_properties(shader_def: UsdShade.Shader, properties: Dict[str, Sdf.ValueTypeName]) -> None:
    """Updates the shader properties of the given shader definition.

    Args:
        shader_def (UsdShade.Shader): The shader definition to update.
        properties (Dict[str, Sdf.ValueTypeName]): A dictionary mapping property names to their value types.
    """
    existing_properties = shader_def.GetInputs()
    existing_property_dict = {prop.GetBaseName(): prop.GetTypeName() for prop in existing_properties}
    for prop_name, prop_type in properties.items():
        if prop_name in existing_property_dict:
            if existing_property_dict[prop_name] != prop_type:
                raise ValueError(f"Property '{prop_name}' already exists with a different value type.")
        else:
            shader_def.CreateInput(prop_name, prop_type)
    for prop_name in existing_property_dict:
        if prop_name not in properties:
            shader_def.RemoveInput(prop_name)


def list_shader_primvars(shader_def: UsdShade.Shader) -> List[str]:
    """List the primvar names for a shader definition."""
    if not shader_def or not shader_def.GetPrim().IsValid():
        raise ValueError("Invalid shader definition.")
    metadata = shader_def.GetSdrMetadata()
    if not metadata:
        raise ValueError("Failed to get shader metadata.")
    primvar_names_str = UsdShade.ShaderDefUtils.GetPrimvarNamesMetadataString(metadata, shader_def)
    primvar_names = primvar_names_str.split("|") if primvar_names_str else []
    primvar_names = [name for name in primvar_names if name]
    return primvar_names


def create_shader_definition(stage: Usd.Stage, shader_name: str, shader_id: str, shader_path: str) -> UsdShade.Shader:
    """Create a shader definition in the given stage.

    Args:
        stage (Usd.Stage): The USD stage to create the shader definition in.
        shader_name (str): The name of the shader.
        shader_id (str): The unique identifier of the shader.
        shader_path (str): The path in the stage where the shader definition will be created.

    Returns:
        UsdShade.Shader: The created shader definition.
    """
    if not Sdf.Path(shader_path).IsPrimPath():
        raise ValueError(f"Invalid shader path: {shader_path}")
    shader_def = UsdShade.Shader.Define(stage, shader_path)
    shader_def.CreateIdAttr(shader_id)
    shader_def.SetSourceAsset(shader_name, "")
    shader_def.SetSourceAssetSubIdentifier(shader_id)
    return shader_def


def unassign_material_from_hierarchy(stage: Usd.Stage, prim_path: str) -> List[Usd.Prim]:
    """Unassigns any material assignments from prims in the hierarchy.

    Args:
        stage (Usd.Stage): The USD stage.
        prim_path (str): The path to the root prim of the hierarchy.

    Returns:
        List[Usd.Prim]: A list of prims that had materials unassigned.
    """
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"Prim at path {prim_path} does not exist.")
    unassigned_prims = []
    for descendant in Usd.PrimRange(prim):
        if descendant.HasAPI(UsdShade.MaterialBindingAPI):
            binding_api = UsdShade.MaterialBindingAPI(descendant)
            if binding_api.GetDirectBindingRel().GetTargets():
                binding_api.GetDirectBindingRel().BlockTargets()
                unassigned_prims.append(descendant)
    return unassigned_prims


def list_prims_with_material(stage: Usd.Stage, material_path: str) -> List[Usd.Prim]:
    """Return a list of prims bound to the specified material.

    Args:
        stage (Usd.Stage): The USD stage to search for prims.
        material_path (str): The path to the material to search for.

    Returns:
        List[Usd.Prim]: A list of prims bound to the specified material.
    """
    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim.IsValid():
        raise ValueError(f"Material prim at path {material_path} does not exist.")
    if not UsdShade.Material(material_prim):
        raise ValueError(f"Prim at path {material_path} is not a material.")
    bound_prims = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdShade.MaterialBindingAPI):
            binding_api = UsdShade.MaterialBindingAPI(prim)
            bound_material = binding_api.ComputeBoundMaterial()[0]
            if bound_material and bound_material.GetPath() == material_prim.GetPath():
                bound_prims.append(prim)
    return bound_prims


def copy_material_from_prim(source_prim: Usd.Prim, dest_prim: Usd.Prim) -> bool:
    """Copy material from one prim to another.

    Args:
        source_prim (Usd.Prim): The prim to copy the material from.
        dest_prim (Usd.Prim): The prim to copy the material to.

    Returns:
        bool: True if the material was successfully copied, False otherwise.
    """
    if not source_prim.IsValid():
        return False
    if not dest_prim.IsValid():
        return False
    source_binding = UsdShade.MaterialBindingAPI(source_prim)
    source_material_paths = source_binding.GetDirectBindingRel().GetTargets()
    if not source_material_paths:
        return False
    material_prim = UsdShade.Material(source_prim.GetStage().GetPrimAtPath(source_material_paths[0]))
    dest_binding = UsdShade.MaterialBindingAPI(dest_prim)
    dest_binding.Bind(material_prim)
    return True


def apply_material_to_hierarchy(
    prim: Usd.Prim, material: UsdShade.Material, binding_api: UsdShade.MaterialBindingAPI = UsdShade.Tokens.preview
):
    """Apply a material to a prim and all its descendants.

    Args:
        prim (Usd.Prim): The root prim of the hierarchy to apply the material to.
        material (UsdShade.Material): The material to apply.
        binding_api (UsdShade.MaterialBindingAPI, optional): The material binding API to use. Defaults to UsdShade.Tokens.preview.
    """
    if not prim.IsValid():
        raise ValueError(f"Invalid prim: {prim.GetPath()}")
    if not material.GetPrim().IsValid():
        raise ValueError(f"Invalid material: {material.GetPath()}")
    for descendant in Usd.PrimRange(prim):
        if not descendant.IsA(UsdGeom.Imageable):
            continue
        binding_api = UsdShade.MaterialBindingAPI(descendant)
        binding_api.Bind(material)


def get_full_shading_attribute_names(base_names: List[str], attribute_type: UsdShade.AttributeType) -> List[str]:
    """
    Get the full shading attribute names given a list of base names and the attribute type.

    Args:
        base_names (List[str]): List of base attribute names.
        attribute_type (UsdShade.AttributeType): The shading attribute type.

    Returns:
        List[str]: List of full shading attribute names.
    """
    full_names = []
    for base_name in base_names:
        if not isinstance(base_name, str) or not base_name:
            raise ValueError(f"Invalid base name: {base_name}")
        full_name = UsdShade.Utils.GetFullName(base_name, attribute_type)
        full_names.append(full_name)
    return full_names


def get_shading_base_names_and_types(stage: Usd.Stage, material_path: str) -> List[Tuple[str, UsdShade.AttributeType]]:
    """
    Get the base names and types of all shading attributes on a material.

    Args:
        stage (Usd.Stage): The USD stage.
        material_path (str): The path to the material prim.

    Returns:
        List[Tuple[str, UsdShade.AttributeType]]: A list of tuples containing the base name and type of each shading attribute.
    """
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        raise ValueError(f"No material found at path: {material_path}")
    result = []
    for attr in material.GetInputs() + material.GetOutputs():
        full_name = attr.GetFullName()
        (base_name, attr_type) = UsdShade.Utils.GetBaseNameAndType(full_name)
        result.append((base_name, attr_type))
    return result
