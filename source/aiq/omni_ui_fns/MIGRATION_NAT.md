# OmniUI-fns Migration to NAT (v0.6.0)

## Summary

OmniUI-fns has been upgraded from AIQ toolkit to NAT (NeMo Agent Toolkit) to enable compatibility with MaaS and the latest NVIDIA agent frameworks.

## Changes Made

### 1. Dependencies Updated (pyproject.toml)

**Before (v0.5.0):**
```toml
aiqtoolkit = "1.1.0"
aiqtoolkit-langchain = "1.1.0"
```

**After (v0.6.0):**
```toml
nvidia-nat = ">=1.3.0"
nvidia-nat-langchain = ">=1.3.0"
```

### 2. Import Statements Updated (all register_*.py files - 10 total)

**Before:**
```python
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
```

**After:**
```python
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
```

**Files Updated:**
- register_search_ui_code_examples.py
- register_search_ui_window_examples.py
- register_get_classes.py
- register_get_modules.py
- register_get_class_detail.py
- register_get_module_detail.py
- register_get_method_detail.py
- register_get_instructions.py
- register_get_class_instructions.py
- register_get_style_docs.py

### 3. Dual Entry Point Registration

OmniUI-fns now registers functions with BOTH entry point groups for maximum compatibility:

```toml
# AIQ entry points (backwards compatibility)
[tool.poetry.plugins."aiq.components"]
omni_ui_fns_search_ui_code_examples = "omni_ui_fns.register_search_ui_code_examples"
# ... (all 10 functions)

# NAT entry points (native)
[tool.poetry.plugins."nat.components"]
omni_ui_fns_search_ui_code_examples = "omni_ui_fns.register_search_ui_code_examples"
# ... (all 10 functions)
```

## Compatibility

### ✅ Works With:
- **NAT-based MCP servers** (like omni_ui_mcp_maas)
- **AIQ-based MCP servers** (like omni_ui_mcp) - backwards compatible
- **nvidia-nat >= 1.3.0**
- **maas-sdk[nat] >= 2.2.0.17**

### ❌ No Longer Compatible With:
- **aiqtoolkit 1.1.0** (replaced by nvidia-nat)

## Function Implementation

All 10 OmniUI functions remain unchanged:
- `search_ui_code_examples`
- `search_ui_window_examples`
- `list_ui_classes`
- `list_ui_modules`
- `get_ui_class_detail`
- `get_ui_module_detail`
- `get_ui_method_detail`
- `get_ui_instructions`
- `get_ui_class_instructions`
- `get_ui_style_docs`

Only the registration system was updated, not the actual function logic.

## Testing

After migration, test with:

```bash
cd source/aiq/omni_ui_fns
poetry install
```

Then use in NAT workflows:

```yaml
functions:
  search_ui_code_examples:
    _type: omni_ui_fns/search_ui_code_examples
    verbose: false
```

## Important Note

Package `nvidia-nat` provides the Python module `nat`, NOT `nvidia_nat`.

## Version History

- **v0.6.0** - Migrated to NAT from AIQ
- **v0.5.0** - Last AIQ-based version
