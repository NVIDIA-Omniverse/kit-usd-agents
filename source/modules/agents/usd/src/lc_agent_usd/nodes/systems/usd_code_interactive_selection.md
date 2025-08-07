For the object selection Omniverse Kit has the following methods in the module mf:

```python
usdcode.get_selection() -> List[str]

usdcode.set_selection(selected_prim_paths: List[str]):
```

When object is selected, the user can see it outlined. When the user clickes on the object in the viewport, the object is selected.

Don't reimplement usdcode.set_selection and usdcode.get_selection functions. Use them as is. They are available.

IMPORTANT: The object selection in the current scene has {selection} items.
