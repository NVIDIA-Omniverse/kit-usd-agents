# omni.ui.MultiFloatField

Use MultiFloatField to construct a matrix field:
```execute 200
args = [1.0 if i % 5 == 0 else 0.0 for i in range(16)]
ui.MultiFloatField(*args, width=ui.Percent(50), h_spacing=5, v_spacing=2)
```

### omni.ui.MultiFloatDragField
Each of the field value could be changed by dragging
```execute 200
ui.MultiFloatDragField(0.0, 0.0, 0.0, 0.0)
```

