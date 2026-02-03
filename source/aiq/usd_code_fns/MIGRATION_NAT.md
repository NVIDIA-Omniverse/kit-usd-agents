# USD Code Functions Migration to NAT (v0.3.0)

## Summary

USD Code functions package has been upgraded from AIQ toolkit to NAT (NeMo Agent Toolkit) to enable compatibility with MaaS and the latest NVIDIA agent frameworks.

## Changes Made

### 1. Dependencies Updated (pyproject.toml)

**Before (v0.2.4):**
```toml
aiqtoolkit = "1.1.0"
aiqtoolkit-langchain = "1.1.0"
```

**After (v0.3.0):**
```toml
nvidia-nat = ">=1.3.0"
nvidia-nat-langchain = ">=1.3.0"
```

### 2. Import Statements Updated (all register_*.py files - 8 total)

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
- register_search_usd_code_examples.py
- register_search_usd_knowledge.py
- register_list_usd_modules.py
- register_list_usd_classes.py
- register_get_usd_module_detail.py
- register_get_usd_class_detail.py
- register_get_usd_method_detail.py
- register_give_usd_feedback.py

### 3. Dual Entry Point Registration

USD Code functions now registers with BOTH entry point groups for maximum compatibility:

```toml
# AIQ entry points (backwards compatibility)
[tool.poetry.plugins."aiq.components"]
omni_aiq_usd_code_search_usd_code_examples = "omni_aiq_usd_code.register_search_usd_code_examples"
# ... (all 8 functions)

# NAT entry points (native)
[tool.poetry.plugins."nat.components"]
omni_aiq_usd_code_search_usd_code_examples = "omni_aiq_usd_code.register_search_usd_code_examples"
# ... (all 8 functions)
```

## Compatibility

### ✅ Works With:
- **NAT-based MCP servers** (like usd_code_mcp_maas)
- **AIQ-based MCP servers** (like usd_code_mcp) - backwards compatible
- **nvidia-nat >= 1.3.0**
- **maas-sdk[nat] >= 2.2.0.17**

### ❌ No Longer Compatible With:
- **aiqtoolkit 1.1.0** (replaced by nvidia-nat)

## Function Implementation

All 8 USD Code functions remain unchanged:
- `search_usd_code_examples`
- `search_usd_knowledge`
- `list_usd_modules`
- `list_usd_classes`
- `get_usd_module_detail`
- `get_usd_class_detail`
- `get_usd_method_detail`
- `give_usd_feedback`

Only the registration system was updated, not the actual function logic.

## Testing

After migration, test with:

```bash
cd source/aiq/usd_code_fns
poetry install
```

Then use in NAT workflows:

```yaml
functions:
  search_usd_code_examples:
    _type: omni_aiq_usd_code/search_usd_code_examples
    verbose: false
```

## Important Note

Package `nvidia-nat` provides the Python module `nat`, NOT `nvidia_nat`.

## Version History

- **v0.3.0** - Migrated to NAT from AIQ
- **v0.2.4** - Last AIQ-based version
