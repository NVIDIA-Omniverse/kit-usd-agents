# Extension Database API Format - Based on Actual Code Atlas

## Overview

After analyzing the actual Code Atlas implementation in the LC Agent system, the extension API details format should match the proven structure used for USD and other library analysis.

## Actual Code Atlas JSON Structure

The API details files should follow this exact format, matching the structure in `source/modules/lc_agent/src/lc_agent/code_atlas/`:

### Example: extensions_apis/omni.ui-2.27.2-api.json

```json
{
  "modules": {
    "omni.ui": {
      "name": "ui",
      "full_name": "omni.ui",
      "docstring": "Core UI framework for creating graphical interfaces in Omniverse Kit applications.",
      "line_number": null,
      "file_path": "omni.ui/__init__.py",
      "class_names": ["Window", "Button", "Label", "Frame", "VStack", "HStack"],
      "function_names": ["get_main_window", "create_workspace", "set_style"],
      "equivelant_modules": ["omni.ui_scene"],
      "extension_name": "omni.ui"
    },
    "omni.ui._ui": {
      "name": "_ui",
      "full_name": "omni.ui._ui",
      "docstring": "Internal UI implementation module.",
      "line_number": null,
      "file_path": "omni.ui/_ui.py",
      "class_names": ["Window", "Button", "Container"],
      "function_names": ["_internal_create"],
      "equivelant_modules": [],
      "extension_name": "omni.ui"
    }
  },

  "classes": {
    "omni.ui.Window": {
      "name": "Window",
      "full_name": "omni.ui.Window",
      "docstring": "Creates a new UI window that can contain other UI elements.\n\nThe Window class provides the foundation for creating graphical interfaces\nin Omniverse Kit applications. It supports various layouts, styling options,\nand event handling mechanisms.",
      "line_number": 1250,
      "module_name": "omni.ui",
      "methods": ["__init__", "show", "hide", "set_title", "set_visibility"],
      "class_variables": ["_instances", "_default_flags"],
      "parent_classes": ["omni.ui.Container", "omni.ui.Widget"],
      "decorators": []
    },
    "omni.ui.Button": {
      "name": "Button",
      "full_name": "omni.ui.Button",
      "docstring": "A clickable button widget that can trigger actions.\n\nButtons support text, icons, and custom styling. They emit clicked\nevents when activated by user interaction.",
      "line_number": 890,
      "module_name": "omni.ui",
      "methods": ["__init__", "clicked", "set_text", "set_enabled"],
      "class_variables": ["_button_count"],
      "parent_classes": ["omni.ui.Widget"],
      "decorators": []
    },
    "omni.ui.Container": {
      "name": "Container",
      "full_name": "omni.ui.Container",
      "docstring": "Base class for UI elements that can contain other widgets.",
      "line_number": 450,
      "module_name": "omni.ui",
      "methods": ["__init__", "add_child", "remove_child", "clear"],
      "class_variables": [],
      "parent_classes": ["omni.ui.Widget"],
      "decorators": []
    }
  },

  "methods": {
    "omni.ui.Window.__init__": {
      "name": "__init__",
      "full_name": "omni.ui.Window.__init__",
      "docstring": "Initialize a new window with specified dimensions and title.\n\nArgs:\n    title: The window title displayed in the title bar\n    width: Initial window width in pixels\n    height: Initial window height in pixels\n    flags: Window behavior flags",
      "line_number": 1289,
      "module_name": "omni.ui",
      "parent_class": "omni.ui.Window",
      "return_type": null,
      "arguments": [
        {
          "name": "self",
          "full_name": "omni.ui.Window.__init__.self",
          "docstring": null,
          "line_number": null,
          "type_annotation": null,
          "default_value": null,
          "is_variadic": false,
          "parent_method": "omni.ui.Window.__init__"
        },
        {
          "name": "title",
          "full_name": "omni.ui.Window.__init__.title",
          "docstring": null,
          "line_number": null,
          "type_annotation": "str",
          "default_value": null,
          "is_variadic": false,
          "parent_method": "omni.ui.Window.__init__"
        },
        {
          "name": "width",
          "full_name": "omni.ui.Window.__init__.width",
          "docstring": null,
          "line_number": null,
          "type_annotation": "int",
          "default_value": "400",
          "is_variadic": false,
          "parent_method": "omni.ui.Window.__init__"
        },
        {
          "name": "height",
          "full_name": "omni.ui.Window.__init__.height",
          "docstring": null,
          "line_number": null,
          "type_annotation": "int",
          "default_value": "300",
          "is_variadic": false,
          "parent_method": "omni.ui.Window.__init__"
        },
        {
          "name": "**kwargs",
          "full_name": "omni.ui.Window.__init__.**kwargs",
          "docstring": null,
          "line_number": null,
          "type_annotation": null,
          "default_value": null,
          "is_variadic": true,
          "parent_method": "omni.ui.Window.__init__"
        }
      ],
      "is_class_method": false,
      "is_static_method": false,
      "is_async_method": false,
      "decorators": [],
      "source_code": "if not title:\n    title = 'Window'\nsuper().__init__(**kwargs)\nself._title = title\nself._width = width\nself._height = height\nself._visible = False\nself._children = []\nWindow._instances.append(self)",
      "class_usages": ["omni.ui.Widget", "omni.ui.Container"]
    },
    "omni.ui.Window.show": {
      "name": "show",
      "full_name": "omni.ui.Window.show",
      "docstring": "Make the window visible on screen.\n\nThis method will display the window if it was previously hidden\nand bring it to the front of other windows. The window becomes\nthe active window and can receive user input.",
      "line_number": 1340,
      "module_name": "omni.ui",
      "parent_class": "omni.ui.Window",
      "return_type": "None",
      "arguments": [
        {
          "name": "self",
          "full_name": "omni.ui.Window.show.self",
          "docstring": null,
          "line_number": null,
          "type_annotation": null,
          "default_value": null,
          "is_variadic": false,
          "parent_method": "omni.ui.Window.show"
        }
      ],
      "is_class_method": false,
      "is_static_method": false,
      "is_async_method": false,
      "decorators": [],
      "source_code": "self._visible = True\nself._bring_to_front()\nself._update_layout()\nself._render()",
      "class_usages": []
    },
    "omni.ui.Window.hide": {
      "name": "hide",
      "full_name": "omni.ui.Window.hide",
      "docstring": "Hide the window from view.\n\nThe window remains in memory but is not visible to the user.\nAll child widgets are also hidden but maintain their state.",
      "line_number": 1355,
      "module_name": "omni.ui",
      "parent_class": "omni.ui.Window",
      "return_type": "None",
      "arguments": [
        {
          "name": "self",
          "full_name": "omni.ui.Window.hide.self",
          "docstring": null,
          "line_number": null,
          "type_annotation": null,
          "default_value": null,
          "is_variadic": false,
          "parent_method": "omni.ui.Window.hide"
        }
      ],
      "is_class_method": false,
      "is_static_method": false,
      "is_async_method": false,
      "decorators": [],
      "source_code": "self._visible = False\nself._clear_focus()\nself._notify_hidden()",
      "class_usages": []
    },
    "omni.ui.get_main_window": {
      "name": "get_main_window",
      "full_name": "omni.ui.get_main_window",
      "docstring": "Returns the main application window instance.\n\nThis function provides access to the primary Kit application window\nwhich hosts the main UI elements and viewport. Returns None if no\nmain window has been created yet.",
      "line_number": 156,
      "module_name": "omni.ui",
      "parent_class": null,
      "return_type": "Optional[omni.ui.Window]",
      "arguments": [],
      "is_class_method": false,
      "is_static_method": false,
      "is_async_method": false,
      "decorators": [],
      "source_code": "return _main_window_instance if _main_window_instance else None",
      "class_usages": ["omni.ui.Window"]
    },
    "omni.ui.Button.clicked": {
      "name": "clicked",
      "full_name": "omni.ui.Button.clicked",
      "docstring": "Event triggered when button is clicked.\n\nThis method can be overridden to handle click events, or connected\nto using the signal/slot mechanism.",
      "line_number": 920,
      "module_name": "omni.ui",
      "parent_class": "omni.ui.Button",
      "return_type": "None",
      "arguments": [
        {
          "name": "self",
          "full_name": "omni.ui.Button.clicked.self",
          "docstring": null,
          "line_number": null,
          "type_annotation": null,
          "default_value": null,
          "is_variadic": false,
          "parent_method": "omni.ui.Button.clicked"
        }
      ],
      "is_class_method": false,
      "is_static_method": false,
      "is_async_method": false,
      "decorators": ["property"],
      "source_code": "self._emit_clicked_signal()\nif self._click_callback:\n    self._click_callback()",
      "class_usages": []
    }
  },

  "used_classes": {
    "omni.ui.Window": [
      "omni.kit.app.create_main_window",
      "omni.kit.window.manager.show_window",
      "omni.ui.get_main_window",
      "omni.ui.workspace.create_workspace_window",
      "omni.kit.ui.dialog.MessageDialog.__init__"
    ],
    "omni.ui.Button": [
      "omni.ui.Window.__init__",
      "omni.kit.toolbar.create_button",
      "omni.ui.dialog.ButtonDialog.add_button",
      "omni.kit.menu.create_menu_item"
    ],
    "omni.ui.Container": [
      "omni.ui.Window.__init__",
      "omni.ui.VStack.__init__",
      "omni.ui.HStack.__init__",
      "omni.ui.Frame.__init__"
    ]
  }
}
```

## Processing Pipeline Alignment

The extension API extraction should follow the same process as the Code Atlas:

### 1. AST Parsing
- Use `ast.parse()` to analyze Python source code
- Extract docstrings using `ast.get_docstring()`
- Parse type annotations from AST nodes
- Track import statements and class usage

### 2. Pydantic Models
- Use the same Pydantic models: `CodeAtlasModuleInfo`, `CodeAtlasClassInfo`, `CodeAtlasMethodInfo`, `CodeAtlasArgumentInfo`
- Serialize using `model_dump(by_alias=True, exclude_defaults=True)`

### 3. Extension Detection
- Parse `extension.toml` files to identify extension boundaries
- Set `extension_name` field in root modules
- Handle equivalent modules through wildcard imports

### 4. Source Code Storage
- Store full method source code in `source_code` field
- Enable "USED_WITH" queries by tracking `class_usages`
- Support reconstruction of complete class definitions

## Benefits of Code Atlas Alignment

### 1. Proven Architecture
- Battle-tested structure used for USD library analysis
- Handles complex inheritance and module relationships
- Supports sophisticated queries (MODULE, CLASS, USED_WITH)

### 2. Tool Compatibility
- Extensions can reuse existing Code Atlas query tools
- Same JSON structure enables shared processing code
- Consistent developer experience across USD and Extensions

### 3. Rich Query Capabilities
- Find all methods that use a specific class
- Reconstruct complete class definitions with source
- Support partial name matching and fuzzy searches
- Enable batch queries for efficiency

### 4. Extensibility
- Easy to add new fields to existing Pydantic models
- Support for topics and documentation sections
- Handles equivalent modules and aliases

This format ensures the extension database leverages the proven Code Atlas architecture while providing comprehensive API access for Omniverse Kit extensions.