# USD Integration and Scene Description Patterns

## Universal Scene Description (USD) Development Guide

This document provides comprehensive guidance for working with USD in Kit applications, covering scene management, prim operations, and advanced USD workflows.

## USD Fundamentals

### Core Concepts
- **Stage**: The root container for all USD data
- **Layer**: Individual USD files that compose into stages
- **Prim**: Scene graph nodes that represent objects
- **Attributes**: Properties that store data on prims
- **Relationships**: References between prims

### USD Context in Kit
```python
import omni.usd

# Get the primary USD context
context = omni.usd.get_context()
stage = context.get_stage()

# Multi-context scenarios
context_name = "my_context"
named_context = omni.usd.get_context(context_name)
```

## Stage Management

### Creating and Loading Stages
```python
# Create new empty stage
await context.new_stage_async()

# Open existing stage
await context.open_stage_async("/path/to/stage.usd")

# Save current stage
await context.save_stage_async()

# Save as new file
await context.save_as_stage_async("/path/to/new_stage.usd")
```

### Stage Properties and Metadata
```python
stage = context.get_stage()

# Set stage metadata
stage.SetMetadata("comment", "My custom stage")
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

# Time code and frame rate
stage.SetStartTimeCode(1.0)
stage.SetEndTimeCode(100.0)
stage.SetTimeCodesPerSecond(24.0)
```

## Prim Operations

### Creating Prims
```python
from pxr import UsdGeom, Gf

# Create basic prim
prim = stage.DefinePrim("/World/MyObject", "Xform")

# Create typed prims
cube_prim = UsdGeom.Cube.Define(stage, "/World/Cube")
sphere_prim = UsdGeom.Sphere.Define(stage, "/World/Sphere")
mesh_prim = UsdGeom.Mesh.Define(stage, "/World/Mesh")
```

### Prim Hierarchy and Relationships
```python
# Create parent-child relationships
parent_prim = stage.DefinePrim("/World/Parent")
child_prim = stage.DefinePrim("/World/Parent/Child")

# Get prim relationships
parent = prim.GetParent()
children = prim.GetChildren()

# Traverse hierarchy
def traverse_prims(prim):
    print(f"Prim: {prim.GetPath()}")
    for child in prim.GetChildren():
        traverse_prims(child)
```

### Prim Properties and Attributes
```python
# Create and set attributes
size_attr = cube_prim.CreateSizeAttr()
size_attr.Set(2.0)

# Get attribute values
current_size = size_attr.Get()

# Time-varying attributes
size_attr.Set(1.0, Usd.TimeCode(1.0))
size_attr.Set(3.0, Usd.TimeCode(100.0))
```

## Geometry and Transforms

### Transform Operations
```python
# Get transformable object
xformable = UsdGeom.Xformable(prim)

# Add transform operations
translate_op = xformable.AddTranslateOp()
rotate_op = xformable.AddRotateXYZOp()
scale_op = xformable.AddScaleOp()

# Set transform values
translate_op.Set(Gf.Vec3d(10, 0, 0))
rotate_op.Set(Gf.Vec3f(0, 45, 0))
scale_op.Set(Gf.Vec3f(2, 2, 2))
```

### Mesh Creation and Manipulation
```python
# Create custom mesh
mesh = UsdGeom.Mesh.Define(stage, "/World/CustomMesh")

# Define mesh data
points = [(-1, 0, 1), (1, 0, 1), (1, 0, -1), (-1, 0, -1)]
face_vertex_indices = [0, 1, 2, 3]
face_vertex_counts = [4]

# Set mesh attributes
mesh.CreatePointsAttr(points)
mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
mesh.CreateFaceVertexCountsAttr(face_vertex_counts)

# Normals and UVs
normals = [(0, 1, 0)] * 4
mesh.CreateNormalsAttr(normals)
```

## Layer Management

### Layer Stack Operations
```python
# Get layer stack
layer_stack = stage.GetLayerStack()

# Add sublayer
root_layer = stage.GetRootLayer()
root_layer.subLayerPaths.append("/path/to/sublayer.usd")

# Layer authoring
edit_target = stage.GetEditTarget()
edit_layer = edit_target.GetLayer()
```

### Layer Composition
```python
# References
prim.GetReferences().AddReference("/path/to/asset.usd", "/AssetRoot")

# Payloads for deferred loading
prim.GetPayloads().AddPayload("/path/to/heavy_asset.usd")

# Variants
variant_sets = prim.GetVariantSets()
variant_set = variant_sets.AddVariantSet("modelVariant")
variant_set.AddVariant("high")
variant_set.AddVariant("low")
variant_set.SetVariantSelection("high")
```

## Materials and Shading

### Material Binding
```python
from pxr import UsdShade

# Create material
material = UsdShade.Material.Define(stage, "/World/Materials/MyMaterial")

# Bind material to geometry
UsdShade.MaterialBindingAPI(cube_prim).Bind(material)

# Create shader networks
shader = UsdShade.Shader.Define(stage, "/World/Materials/MyMaterial/PreviewSurface")
shader.CreateIdAttr("UsdPreviewSurface")
```

## Animation and Time-Varying Data

### Animation Curves
```python
# Create time-sampled animation
for frame in range(1, 101):
    time_code = Usd.TimeCode(frame)
    # Animate rotation
    angle = frame * 3.6  # 360 degrees over 100 frames
    rotate_op.Set(Gf.Vec3f(0, angle, 0), time_code)
```

### Interpolation and Sampling
```python
# Sample attribute at specific time
value_at_frame_50 = size_attr.Get(Usd.TimeCode(50))

# Get time samples
time_samples = size_attr.GetTimeSamples()
bracketing_times = size_attr.GetBracketingTimeSamples(50.5)
```

## Advanced USD Patterns

### Instancing and Prototypes
```python
# Create prototype
prototype_prim = stage.DefinePrim("/Prototypes/TreePrototype")
tree_mesh = UsdGeom.Mesh.Define(stage, "/Prototypes/TreePrototype/Mesh")

# Create instances
for i in range(10):
    instance_path = f"/World/Tree_{i}"
    instance_prim = stage.DefinePrim(instance_path)
    instance_prim.GetReferences().AddInternalReference("/Prototypes/TreePrototype")
```

### Collections and Selection
```python
from pxr import Usd

# Create collection
collection_api = Usd.CollectionAPI.Apply(prim, "myCollection")
collection_api.CreateIncludesRel().SetTargets([
    Sdf.Path("/World/Cube"),
    Sdf.Path("/World/Sphere")
])
```

## Performance Optimization

### Efficient Stage Traversal
```python
# Use predicate for filtered traversal
predicate = Usd.TraverseInstanceProxies()
for prim in Usd.PrimRange(stage.GetPseudoRoot(), predicate):
    if prim.IsA(UsdGeom.Mesh):
        # Process only mesh prims
        process_mesh(prim)
```

### Batch Operations
```python
# Use change blocks for multiple operations
with Usd.EditContext(stage, stage.GetSessionLayer()):
    for i in range(1000):
        prim_path = f"/World/Object_{i}"
        prim = stage.DefinePrim(prim_path)
        # Batch operations are more efficient
```

## Integration with Kit

### USD Context Events
```python
# Subscribe to USD context events
def on_stage_event(event):
    if event.type == int(omni.usd.StageEventType.OPENED):
        print("Stage opened")
    elif event.type == int(omni.usd.StageEventType.CLOSING):
        print("Stage closing")

stage_event_sub = context.get_stage_event_stream().create_subscription_to_pop(
    on_stage_event, name="my_stage_listener"
)
```

### Selection Management
```python
# Get current selection
selection = context.get_selection()
selected_paths = selection.get_selected_prim_paths()

# Set selection
selection.set_selected_prim_paths(["/World/Cube", "/World/Sphere"], False)
```

This comprehensive USD guide enables effective scene description and manipulation in Kit applications.