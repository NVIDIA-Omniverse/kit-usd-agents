Example 1:

> Create an orthographic camera

```python
camera_path = "{default_prim}/OrthoCam"
usdcode.create_prim(stage, "Camera", camera_path, projection="orthographic")
```

Example 2:

> Put the asset http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Residential/Furniture/DiningSets/DesPeres/DesPeres_Table.usd to the scene

```python
url = "http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Residential/Furniture/DiningSets/DesPeres/DesPeres_Table.usd"
usdcode.create_payload(stage, "{default_prim}/Table", url)
```

Example 3:

> Create a default prim with float and vector attributes

```python
prim_path = "{default_prim}"
usdcode.create_prim(stage, "Xform", prim_path)
usdcode.create_usd_attribute(stage, prim_path, "my_float_attr", Sdf.ValueTypeNames.Float, 0.1)
usdcode.create_usd_attribute(stage, prim_path, "my_vector_attr", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.1, 0.2, 0.3))
```

Example 4:

> Create a cube

```python
usdcode.create_prim(stage, "Cube", "{default_prim}/Cube")
```

> Make it bigger

```python
usdcode.change_property(stage, "{default_prim}/Cube.size", 50)
```

> I don't like the cube. Create a sphere instead

```python
usdcode.remove_prim(stage, "{default_prim}/Cube")
usdcode.create_prim(stage, "Sphere", "{default_prim}/Sphere", radius=50)
```

Note:
`Define` of Usd doesn't have such a syntax to set attributes. Never use it in `Define`. Define doesn't accept attributes.
UsdGeom.Sphere.Define(stage, "{default_prim}/Sphere", radius=100)


Example 5:

> Create a sphere

```python
usdcode.create_prim(stage, "Sphere", "{default_prim}/Sphere")
```

> What is the radius of this sphere?

```python
radius = stage.GetPrimAtPath("{default_prim}/Sphere").GetRadiusAttr().Get()
print("The sphere radius is", radius)
```

Example 6:

> Create a sphere

```python
usdcode.create_prim(stage, "Sphere", "{default_prim}/Sphere")
```

> I would like to have two spheres.

```python
usdcode.create_prim(stage, "Sphere", "{default_prim}/SecondSphere")
```

Example 7:

> Move the selected object up 100 units.

```python
up = usdcode.get_direction_up(stage)
up = (up[0] * 100, up[1] * 100, up[2] * 100)
selection = usdcode.get_selection()
for prim_path in selection:
    translate = usdcode.get_translate(stage, prim_path)
    usdcode.set_translate(stage, prim_path, (translate[0] + up[0], translate[1] + up[1], translate[2] + up[2]))
```

WRONG: `usdcode.set_translate(stage, prim_path, (translate[0] + up[0], translate[1] + up[2], translate[2] + up[1]))`

Example 8:

> Select all the spheres

```python
usdcode.set_selection(usdcode.search_prims_by_type(stage, ["Sphere"]))
```

Example 9:

> Make the selected object red

```python
material_path = usdcode.create_material(stage, "{default_prim}/Looks/RedMaterial", diffuse_reflection_color=(1, 0, 0)).GetPath()
selection = usdcode.get_selection()
for prim_path in selection:
    usdcode.assign_material(stage, prim_path, material_path)
```

Example 10:

> Create a two revolutions spiral of 25 references cones

```python
import math

cone_path = "{default_prim}/Cone_0"
usdcode.create_prim(stage, "Cone", cone_path)
usdcode.set_translate(stage, cone_path, (20, 0, 0))
usdcode.set_rotate(stage, cone_path, (0, 0, 0))

for i in range(1, 25):
    angle = (i / 24) * (4 * math.pi)
    radius = 20 + (380 * i / 24)
    size = 1 + (9 * i / 24)
    x, z = radius * math.cos(angle), radius * math.sin(angle)
    y = 10 * i
    ref_prim_path = f"{default_prim}/Cone_{i}"
    usdcode.copy_prim(stage, cone_path, ref_prim_path)
    usdcode.set_translate(stage, ref_prim_path, (x, y, z))
    usdcode.set_rotate(stage, ref_prim_path, (0, Gf.RadiansToDegrees(-angle - math.pi / 2), 0))
    usdcode.set_scale(stage, ref_prim_path, (size, size, size))
```

Example 11:

> Put the cup on top of the box

```python
usdcode.align_objects(stage, "{default_prim}/Cup", "{default_prim}/Box", axes=['x', 'z'], alignment_type="center_to_center")
usdcode.align_objects(stage, "{default_prim}/Cup", "{default_prim}/Box", axes=['y'], alignment_type='min_to_max')
```

Example 12:

> Put 10 copies of this cup on top of the box
> This is some scene information:
> Prim: /World/Box, Position: (0, 0, 0); World bound [(-100, 0, -100)...(100, 100, 100)]; Local bound [(-100, 0, -100)...(100, 100, 100)]
> Prim: /World/Cup, Position: (0, 5, 0); World bound [(-10, -5, -10)...(10, 15, 10)]; Local bound [(-10, -10, -10)...(10, 10, 10)]

```python
cup_length = 20  # Width in X direction
cup_width = 20   # Width in Z direction
# The layout of cups (3 rows, 4 columns)
cups_rows = 3
cups_columns = 4
# The available space on the box
box_length = 200  # Total length of the box (X axis)
box_width = 200   # Total width of the box (Z axis)
# The spacing between cups
spacing_x = (box_length - cups_rows * cup_length) / (cups_rows + 1)
spacing_z = (box_width - cups_columns * cup_width) / (cups_columns + 1)

for i in range(10):
    cup_name = f"{default_prim}/Cup_{i}"
    # Copy the existing cup
    usdcode.copy_prim(stage, "{default_prim}/Cup", cup_name)

    row = i // cups_columns
    col = i % cups_columns

    x = -box_length/2 + spacing_x/2 + (spacing_x + cup_length) * row + cup_length/2
    z = -box_width/2 + spacing_z/2 + (spacing_z + cup_width) * col + cup_width/2
    y = 5

    translate = (x, y, z)
    usdcode.set_translate(stage, cup_name, translate)

    # Put the cup on top of the box
    usdcode.align_objects(stage, cup_name, "{default_prim}/Box", axes=['{up_axis}'], alignment_type='min_to_max')
```

Example 12:

> Randomly put 10 copies of this cup on the ground surface
> This is some scene information:
> Prim: /World/Ground, Position: (0, 0, 0); World bound [(-100, 0, -100)...(100, 100, 100)]; Local bound [(-100, 0, -100)...(100, 100, 100)]
> Prim: /World/Cup, Position: (0, 5, 0); World bound [(-10, -5, -10)...(10, 15, 10)]; Local bound [(-10, -10, -10)...(10, 10, 10)]

```python
usdcode.scatter_prims(
    stage=stage,
    point_instancer_path="/World/CupPointInstancer",
    prototype_path="/World/Cup",
    surface_path="/World/Ground",
    num_instances=10,
    rotation=(0, 360),
    align_to_normals=1.0
)
```

Example 13:

> Add 30 instances of the cup from the scene to the shelf
> This is some scene information:
> Prim: /World/Shelf, Position: (0, 0, 0); World bound [(-100, 0, -100)...(100, 100, 100)]; Local bound [(-100, 0, -100)...(100, 100, 100)]
> Prim: /World/Cup, Position: (0, 5, 0); World bound [(-10, -5, -10)...(10, 15, 10)]; Local bound [(-10, -10, -10)...(10, 10, 10)]

```python
# IMPORTANT for placing objects!!!
# 1. Set `max_normal_deviation_angle=10.0` to make sure the objects are on the horizontal surface only and not on the bottom.
# 2. Set `align_to_normals=0.0` to make sure the cups are placed vertically and not inclined.
usdcode.scatter_prims(
    stage=stage,
    point_instancer_path="/World/CupPointInstancer",
    prototype_path="/World/Cup",
    surface_path="/World/Shelf",
    num_instances=30,
    rotation=(0, 360),
    align_to_normals=0.0,
    max_normal_deviation_angle=10.0
)
```

Example 14:

> Move the cup on the table left 10 units
> This is some scene information:
> /World/Cup, Position: (0.0, 105.0, -25.0); World bound [(-5, 100, -30)...(5, 110, -20)]; Local bound [(-50, -50, -50)...(50, 50, 50)]
> /World/Table, Position: (0.0, 50.0, 0.0); World bound [(-50, 0, -50)...(50, 100, 50)]; Local bound [(-50, -50, -50)...(50, 50, 50)]

```python
direction_left = usdcode.get_camera_direction_left(stage)
position_cup = usdcode.get_translate(stage, "/World/Cup")
vec_to_move = Gf.Vec3d(*direction_left) * 10.0
# Move to the left relative to the camera
usdcode.set_translate(stage, "/World/Cup", (position_cup[0] + vec_to_move[0],
                                       position_cup[1] + vec_to_move[1],
                                       position_cup[2] + vec_to_move[2]))
# Make sure it's still placed on the table
usdcode.align_objects(stage, "/World/Cup", "/World/Table", axes=["{up_axis}"], alignment_type="min_to_max")
```

Example 15:

> Move the cup on the table left to the glass
> This is some scene information:
> /World/Cup, Position: (0.0, 105.0, -25.0); World bound [(-5, 100, -30)...(5, 110, -20)]; Local bound [(-50, -50, -50)...(50, 50, 50)]
> /World/Glass, Position: (0.0, 105.0, 25.0); World bound [(-5, 100, 20)...(5, 110, 30)]; Local bound [(-50, -50, -50)...(50, 50, 50)]
> /World/Table, Position: (0.0, 50.0, 0.0); World bound [(-50, 0, -50)...(50, 100, 50)]; Local bound [(-50, -50, -50)...(50, 50, 50)]

```python
direction_left = usdcode.get_camera_direction_left(stage)
direction_left = Gf.Vec3d(*direction_left)
position_cup = Gf.Vec3d(0.0, 105.0, -25.0)
position_glass = Gf.Vec3d(0.0, 105.0, 25.0)
cup_to_glass = position_glass - position_cup
# Dot product is the length of projection of vector to another vector
dot_product = direction_left * cup_to_glass
# Projection of cup_to_glass onto direction_left
projection = direction_left * dot_product
# Move to the left relative to the camera
usdcode.set_translate(stage, "/World/Cup", (position_cup[0] + projection[0],
                                       position_cup[1] + projection[1],
                                       position_cup[2] + projection[2]))
# Make sure it's still placed on the table
usdcode.align_objects(stage, "/World/Cup", "/World/Table", axes=["{up_axis}"], alignment_type="min_to_max")
```

Example 16:

> Place boxes at the shelf
> This is some scene information:
> /World/Box_01, Position: (0, 25, 0); World bound [(-25, 0, -25)...(25, 50, 25)]; Local bound [(-50, -50, -50)...(50, 50, 50)]
> /World/Box_02, Position: (-60, 30, 0); World bound [(-90, 0, -30)...(-29, 60, 30)]; Local bound [(-50, -50, -50)...(50, 50, 50)]
> /World/Shelf, Position: (200, 100, 0.0); World bound [(150, 95, -50)...(250, 105, 50)]; Local bound [(-50, -50, -50)...(50, 50, 50)]

```python
usdcode.stack_objects(stage, "/World/Shelf", ["/World/Box_01", "/World/Box_02"])
```

Example 17:

> Stack the three largest boxes on the selected palette
> This is some scene information:
> /World/Box_01, Position: (0, 0, 0); World bound [(-5, 0, -5)...(5, 5, 5)]; Local bound [(-5, -5, -5)...(5, 5, 5)]
> /World/Box_02, Position: (10, 0, 0); World bound [(5, 0, -10)...(15, 10, 10)]; Local bound [(-10, -10, -10)...(10, 10, 10)]
> /World/Box_03, Position: (-10, 0, 0); World bound [(-15, 0, -15)...(-5, 15, 15)]; Local bound [(-15, -15, -15)...(15, 15, 15)]
> /World/Box_04, Position: (20, 0, 0); World bound [(15, 0, -7)...(25, 7, 7)]; Local bound [(-7, -7, -7)...(7, 7, 7)]
> /World/Palette, Position: (0, 0, 0); World bound [(-25, 0, -25)...(25, 1, 25)]; Local bound [(-25, -1, -25)...(25, 1, 25)]

```python
boxes = usdcode.search_visible_prims_by_name(stage, ["Box"])
box_sizes = []
for box in boxes:
    bound = usdcode.get_bbox_world(stage, box)
    if not bound:
        continue
    size = bound.GetSize()
    box_sizes.append((size[0] * size[1] * size[2], box))

largest_boxes = sorted(box_sizes, reverse=True)[:3]
largest_box_paths = [box[1] for box in largest_boxes]
usdcode.stack_objects(stage, "/World/Palette", largest_box_paths)
```

Example 18:

> Stack the three largest objects from selection on the selected shelf. Both objects and shelf are in the selection.

```python
selection = usdcode.get_selection()
# Find the shelf in selection (assuming it has "Shelf" in its name)
shelf_paths = [p for p in selection if "Shelf" in p]
if not shelf_paths:
    raise RuntimeError("No shelf found in selection")
target_shelf = shelf_paths[0]

# Process remaining objects from the selection (excluding shelf).
object_sizes = []
for prim_path in selection:
    # IMPORTANT: The user asked to stack objects from selection, not including the shelf
    if prim_path == target_shelf:
        continue
    bound = usdcode.get_bbox_world(stage, prim_path)
    if not bound:
        continue
    size = bound.GetSize()
    object_sizes.append((size[0] * size[1] * size[2], prim_path))

# IMPORTANT: The user asked only for the three largest objects
largest_objects = sorted(object_sizes, reverse=True)[:3]
largest_object_paths = [obj[1] for obj in largest_objects]
usdcode.stack_objects(stage, target_shelf, largest_object_paths)
```

Example 19:

> Create a room of size 5x4x3 using the prims /RoomParts/FloorPart, /RoomParts/CeilingPart, and /RoomParts/WallPart.

```python
usdcode.construct_room(stage, "{default_prim}/Room", (50, 50, 10), "/RoomParts/FloorPart", "/RoomParts/WallPart", "/RoomParts/CeilingPart")
```

Example 20:

> Create warehouse storage rows by arranging copies of a shelf unit prim at `/Assets/ShelfUnit`. Create 5 parallel rows, each made of 10 shelf units in length and stacked 3 shelf units high, with 2 units of space between rows for aisles.

````python
usdcode.construct_parallel_arrays(stage, "{default_prim}/StorageRows", "/Assets/ShelfUnit", 5, 10, 3, 2.0)
````

Example 21:

> Create the code that creates 10 references of the selected prim with name containing "box" under the same parent, and stack the created references on the selected prim with a name containing "shelf". You need to create 10 references of the initial box and after that stack the created references on the shelf.

````python
selection = usdcode.get_selection()
box_prim_path = next((prim for prim in selection if "box" in prim.lower()), None)
shelf_prim_path = next((prim for prim in selection if "shelf" in prim.lower()), None)

if box_prim_path and shelf_prim_path:
    parent_path = box_prim_path.rsplit("/", 1)[0]
    
    refs = []
    i = 0
    for _ in range(10):
        while True:
            ref_path = f"{parent_path}/box_ref_{i}"
            exists = stage.GetPrimAtPath(ref_path)
            if exists:
                i += 1
            else:
                break

        ref_prim = usdcode.create_reference(stage, ref_path, "", box_prim_path)
        refs.append(ref_path)

    usdcode.stack_objects(stage, shelf_prim_path, refs)
````
