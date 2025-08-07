Omniverse Kit has the following methods in the module `usdcode`. Module `usdcode` is available, don't import it. Use them everywhere it's possible.

Don't rewrite existing helper functions if they already exist.

- ALWAYS use `usdcode` where it's possible
- ALWAYS create lights using `usdcode.create_light`
- NEVER use `UsdLux.RectLight.Define` because we have `usdcode.create_light`. ALWAYS use `usdcode.create_light` to create lights
- NEVER use `xform.AddTranslateOp` because we have `usdcode.set_translate`
- NEVER use `usdcode.create_prim` to create a light. ALWAYS use `usdcode.create_light` to create a light.
- NEVER import `usdcode`. It's already imported.

Example:

> Create the code that creates a room of size 5x4x3 using the prims /RoomParts/FloorPart, /RoomParts/CeilingPart, and /RoomParts/WallPart.

IMPORTANT: We already have such a function in usdcode. The user doesn't need the implementation details. Don't create a new one. Just call the existing one.

```python
usdcode.construct_room(stage, "{default_prim}/Room", (50, 50, 10), "/RoomParts/FloorPart", "/RoomParts/WallPart", "/RoomParts/CeilingPart")
```

Example:

> Create the code that creates rack rows using the rack section prim and user-specified dimensions.

IMPORTANT: Since we already have such a function in usdcode: construct_parallel_arrays, don't create a new one. Just call the usdcode one.

```python
usdcode.construct_parallel_arrays(stage, "{default_prim}/StorageRows", "/Assets/ShelfUnit", 5, 10, 3, 2.0)
```

Example:

> Create a 10x10 rect light directed to negative Z direction.

```python
usdcode.create_light(stage, "{default_prim}/RectLight01", "rect", 1E-3, (1, 1, 1), (0, 0, -1), (0, 0, 10), (10, 10))
```

ALWAYS use the functions from usdcode when available.
ALWAYS use the functions from usdcode when available.
