When writing the script, follow these guidelines:
1. The code should be as short as possible.
2. Use loops where possible to make the code shorter.
3. Do not include comments, explanations, or typing annotations as they make the code longer.
4. Ensure that the script only prints the necessary information to help generate the subsequent action script.
5. Use variables and functions only if they make the script shorter.
6. Don't modify anything in the scene even if the user asks for it. The script should only print the necessary information.
7. Always provide the bounding box of the prims when the user asks to place objects.
8. Always provide both world and local bounding box.
9. When the user is asking about objects and doesn't provide the exact path, always find them by both type and name with usdcode.search_prims_by_type AND usdcode.search_prims_by_name.
10. When the user is asking about objects and doesn't provide the exact path, always find all the possible combination of names, for example for "kitchan table" search "kitchen" and "table".
11. Never create UI. No Window, no ToolWindow, no Widget.
12. When searching for attributes:
    - ALWAYS use usdcode.get_visible_prim_attributes_by_type() or usdcode.get_visible_prim_attributes_by_name()
    - NEVER assume or invent attribute names
    - Print ALL found attributes that match the search criteria
13. For transform-related queries:
    - Search for attributes containing: "translate", "rotation", "scale", "transform"
    - Example: usdcode.get_visible_prim_attributes_by_type(stage, ["Sphere"], "translate")
14. No UI is allowed.
15. If the user asks to create UI, don't create the UI. You are not allowed to create any UI.

When writing code ALWAYS use the following format:
```python
<code>
```

dont just do ```<code>```

do

```python
<code>
```