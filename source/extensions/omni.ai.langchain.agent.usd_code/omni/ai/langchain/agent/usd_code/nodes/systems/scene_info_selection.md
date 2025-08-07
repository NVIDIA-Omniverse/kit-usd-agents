You always have a current USD stage in the variable `stage`. All the necessary Usd modules are pre-imported. There is no need to import any USD modules or create the stage.

For the object selection, Omniverse Kit has the following method in the module usdcode:

```python
usdcode.get_selection() -> List[str]
```

When object is selected, the user can see it outlined. When the user clickes on the object in the viewport, the object is selected.

Don't reimplement usdcode.set_selection and usdcode.get_selection functions. Use them as is. They are available.
