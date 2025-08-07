Example 1:

> Create a chair next to the table.

```python
print("Found the following chairs and tables:")
for prim_path in usdcode.search_visible_prims_by_name(stage, ["chair", "table"]):
    translate = usdcode.get_translate(stage, prim_path)
    bbox_world = usdcode.get_bbox_world(stage, prim_path)
    bbox_local = usdcode.get_bbox_local(stage, prim_path)
    print(f"Prim: {prim_path}, Position: {translate}; World bound {bbox_world}; Local bound {bbox_local}")
```

Example 2:

> Delete the smallest mesh in the current scene.

```python
print("Found the following meshes in the scene:")
for prim_path in usdcode.search_visible_prims_by_type(stage, ["Mesh"]):
    bbox_world = usdcode.get_bbox_world(stage, prim_path)
    bbox_local = usdcode.get_bbox_local(stage, prim_path)
    print(f"{prim_path}: World bound {bbox_world}; Local bound {bbox_local}")
```

Example 3:

> Set the closest camera to the sphere as the active camera.

```python
# The user is asking about objects and doesn't provide the exact path. It's very important to find them by BOTH type and name.
print("Found the following cameras and spheres:")
prim_paths = usdcode.search_visible_prims_by_type(stage, ["camera", "sphere"]) + usdcode.search_visible_prims_by_name(stage, ["sphere"])
for prim_path in prim_paths:
    prim = stage.GetPrimAtPath(prim_path)
    xformable = UsdGeom.Xformable(prim)
    transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    translation = Gf.Vec3d(transform.ExtractTranslation())
    rotation_quat = transform.ExtractRotationQuat()
    rotation = Gf.Rotation(rotation_quat)
    angles = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
    rotation_vec3 = Gf.Vec3d(angles[0], angles[1], angles[2])
    print(f"Type: {prim.GetTypeName()}, Path: {prim.GetPath()}, World Space Position: {translation}, Rotation: {rotation_vec3}")
```

Example 4:

> Select all the tables

```python
print("Found the following tables:")
print(usdcode.search_visible_prims_by_name(stage, ["table"]))
```

Example 5:

> There is a cube in the scene. Create a copy of this cube.

```python
# The user is asking about objects and doesn't provide the exact path. It's very important to find them by BOTH type and name.
print("Found the following cubes in the scene:")
print(usdcode.search_visible_prims_by_type(stage, ["cube"]))
print(usdcode.search_visible_prims_by_name(stage, ["cube"]))
```

Example 6:

> Place a table on the floor

```python
# Search BOTH table and floor
print("Found the following tables and floors:")
for prim_path in usdcode.search_visible_prims_by_name(stage, ["table", "floor"]):
    translate = usdcode.get_translate(stage, prim_path)
    bbox_world = usdcode.get_bbox_world(stage, prim_path)
    bbox_local = usdcode.get_bbox_local(stage, prim_path)
    print(f"Prim: {prim_path}, Position: {translate}; World bound {bbox_world}; Local bound {bbox_local}")
```

Example 7:

> Place a kitchen table on the living room floor

```python
# Search all the possible combination. "kitchen table" cannot be a name because USD doesn't allow spaces in paths
print("Found the following kitchen, table, living room and floor objects:")
for prim_path in usdcode.search_visible_prims_by_name(stage, ["kitchen", "table", "living", "room", "floor"]):
    translate = usdcode.get_translate(stage, prim_path)
    bbox_world = usdcode.get_bbox_world(stage, prim_path)
    bbox_local = usdcode.get_bbox_local(stage, prim_path)
    print(f"Prim: {prim_path}, Position: {translate}; World bound {bbox_world}; Local bound {bbox_local}")
```

Example 8:

> Move the sphere up

```python
# The user is asking about objects and doesn't provide the exact path. It's very important to find them by BOTH type and name.
print("Found the following spheres:")
for prim_path in usdcode.search_visible_prims_by_type(stage, ["sphere"]) + usdcode.search_visible_prims_by_name(stage, ["sphere"]):
    translate = usdcode.get_translate(stage, prim_path)
    bbox_world = usdcode.get_bbox_world(stage, prim_path)
    bbox_local = usdcode.get_bbox_local(stage, prim_path)
    print(f"Prim: {prim_path}, Position: {translate}; World bound {bbox_world}; Local bound {bbox_local}")
```

Example 9:

> Stack the selected boxes on the selected shelf

```python
# Get selected prims
print("The selection has the following objects:")
selected_prims = usdcode.get_selection()
for prim_path in selected_prims:
    prim = stage.GetPrimAtPath(prim_path)
    translate = usdcode.get_translate(stage, prim_path)
    bbox_world = usdcode.get_bbox_world(stage, prim_path)
    bbox_local = usdcode.get_bbox_local(stage, prim_path)
    print(f"Type: {prim.GetTypeName()}, Path: {prim_path}, Position: {translate}; World bound {bbox_world}; Local bound {bbox_local}")
```

Example 10:

> Get the full name of the intensity attribute of all rect lights in the scene.

```python
print("The intensity attributes of all rect lights in the scene are:")
for attribute_full_name in usdcode.get_visible_prim_attributes_by_type(stage, ["rectlight"], "intensity"):
    print(attribute_full_name)
```

IMPORTANT: NEVER create anything related to the UI. Even if the user asks.

> Get the paths of all rect lights in the scene and the name of the intensity attribute of such lights

```python
print("The intensity attributes of all rect lights in the scene are:")
for attribute_full_name in usdcode.get_visible_prim_attributes_by_type(stage, ["rectlight"], "intensity"):
    print(attribute_full_name)
```

IMPORTANT: NEVER create anything related to the UI. Even if the user asks.

Example 11:

> What is the sphere's position attribute?

```python
print("Found the following spheres and their position-related attributes:")
found = []
found += usdcode.get_visible_prim_attributes_by_type(stage, ["sphere"], "translate")
found += usdcode.get_visible_prim_attributes_by_type(stage, ["sphere"], "transform")
found += usdcode.get_visible_prim_attributes_by_name(stage, ["sphere"], "translate")
found += usdcode.get_visible_prim_attributes_by_name(stage, ["sphere"], "transform")

for path in set(found):
    print(f"- {path}")
```

IMPORTANT NOTES FOR THIS EXAMPLE:
1. NEVER invent attribute names like 'translateY' or 'position'
2. ALWAYS use usdcode.get_visible_prim_attributes_by_type and usdcode.get_visible_prim_attributes_by_name
3. Search for BOTH type and name
4. Search for BOTH translate and transform
5. Use set() to remove duplicates
6. Print ALL found attributes

BAD EXAMPLE (NEVER DO THIS):
```python
print("Found sphere with attribute: translateY")  # WRONG! Never invent attribute names
```

ANOTHER BAD EXAMPLE (NEVER DO THIS):
```python
print("Position attribute: position")  # WRONG! Never invent attribute names
```

Example 12:

> Find the palette closest to the floor center (0,0,0).

```python
floor_center = (0, 0, 0)  # Calculated earlier from floor bounds

closest_palette = None
min_distance = float('inf')
closest_palette_info = None

# Search for all likely palette prims (different capitalizations)
for prim_path in usdcode.search_visible_prims_by_name(stage, ["pallet", "palette", "Pallet", "Palette"]):
    # Get the world-space bounding box at default time
    bbox = usdcode.get_bbox_world(stage, prim_path, Usd.TimeCode.Default())
    bbox_min = bbox.GetMin()
    bbox_max = bbox.GetMax()

    # Skip zero-size bounding boxes (invalid)
    # IMPORTANT!!! This is the most important thing in filtering
    if (bbox_max[0] - bbox_min[0]) <= 0 or (bbox_max[1] - bbox_min[1]) <= 0 or (bbox_max[2] - bbox_min[2]) <= 0:
        continue

    # Skip broken bounding boxes (inf values)
    if any(v in (float('inf'), float('-inf')) for v in bbox_min) or \
       any(v in (float('inf'), float('-inf')) for v in bbox_max):
        continue

    # Skip improbable gigantic bounding boxes (likely invalid)
    if (bbox_max[0] - bbox_min[0]) > 1e8 or (bbox_max[1] - bbox_min[1]) > 1e8 or (bbox_max[2] - bbox_min[2]) > 1e8:
        continue

    # Compute bounding-box CENTER (not the prim pivot)
    center = [0.5 * (bbox_min[i] + bbox_max[i]) for i in range(3)]

    # Distance in X-Y plane from floor center
    distance = ((center[0] - floor_center[0])**2 + (center[1] - floor_center[1])**2) ** 0.5

    if distance < min_distance:
        min_distance = distance
        closest_palette = prim_path
        closest_palette_info = {
            "path": prim_path,
            "center": center,
            "bbox": bbox,
            "distance": distance
        }

print("Closest palette to floor center:")
if closest_palette_info:
    print(f"Path: {closest_palette_info['path']}")
    print(f"Center (bbox): {closest_palette_info['center']}")
    print(f"Distance to floor center: {closest_palette_info['distance']}")
    print(f"Bounds: {closest_palette_info['bbox']}")
else:
    print("No valid palette found")
```

IMPORTANT NOTES FOR THIS EXAMPLE:
1. **Bounding-box center vs. prim position** – `usdcode.get_translate` ("position") returns the prim's pivot, which may be off-center.  For geometric proximity queries ("closest to floor center") you must compute the **bounding-box center** as `(min+max)/2`.
2. **Filtering invalid bounds** – Always ignore bounding boxes that contain `inf/-inf`, are unrealistically huge, or have zero or negative size. These often indicate hidden or corrupt geometry.
3. **Distance metric** – For warehouse scenes the vertical axis (Y-up or Z-up) is height; the palette choice is usually based on X-Y distance in the floor plane, so use only X and Y (or X and Z in Z-up) for distance.
4. **Name search strategy** – Search multiple case variations ("pallet", "palette") since asset naming is inconsistent.
5. **Result reporting** – Print the chosen palette's path, bounding-box center, distance, and full bounds so the planner can decide whether it satisfies the criteria.

Example 13:

> Find all empty shelves in the scene (have no objects on them).

```python
print("Empty shelf prims:")
for prim_path in usdcode.search_visible_prims_by_name(stage, ["shelf"]):
    # list_prims_within_vertical_zone returns child prims within the vertical bounds of the shelf's world-space bbox
    if not usdcode.list_prims_within_vertical_zone(stage, prim_path):
        bbox = usdcode.get_bbox_world(stage, prim_path, Usd.TimeCode.Default())
        print(f"{prim_path}: World bound {bbox}")
```

Example 14:

> List all prims and their types in the USD file at C:/foo/bar.usd without switching the current stage.

```python
layer = Sdf.Layer.FindOrOpen("C:/foo/bar.usd")
stage_ref = Usd.Stage.Open(layer)
for prim in stage_ref.Traverse():
    print(f"{prim.GetPath()} - {prim.GetTypeName()}")
```
