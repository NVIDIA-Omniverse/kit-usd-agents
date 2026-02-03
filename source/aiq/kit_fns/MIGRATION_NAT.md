# Kit-fns Migration to NAT (v0.6.0)

## Summary

Kit-fns has been upgraded from AIQ toolkit to NAT (NeMo Agent Toolkit) to enable compatibility with MaaS and the latest NVIDIA agent frameworks.

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

### 2. Import Statements Updated (all register_*.py files)

**Before:**
```python
from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig
from aiq.builder.framework_enum import LLMFrameworkEnum
```

**After:**
```python
from nvidia_nat.builder.builder import Builder
from nvidia_nat.builder.function_info import FunctionInfo
from nvidia_nat.cli.register_workflow import register_function
from nvidia_nat.data_models.function import FunctionBaseConfig
from nvidia_nat.builder.framework_enum import LLMFrameworkEnum
```

### 3. Dual Entry Point Registration

Kit-fns now registers functions with BOTH entry point groups for maximum compatibility:

```toml
# AIQ entry points (backwards compatibility)
[tool.poetry.plugins."aiq.components"]
kit_fns_get_instructions = "kit_fns.register_get_instructions"
# ... (all 11 functions)

# NAT entry points (native)
[tool.poetry.plugins."nat.components"]
kit_fns_get_instructions = "kit_fns.register_get_instructions"
# ... (all 11 functions)
```

## Compatibility

### ✅ Works With:
- **NAT-based MCP servers** (like kit_mcp_maas)
- **AIQ-based MCP servers** (like kit_mcp) - backwards compatible
- **nvidia-nat >= 1.3.0**
- **maas-sdk[nat] >= 2.2.0.17**

### ❌ No Longer Compatible With:
- **aiqtoolkit 1.1.0** (replaced by nvidia-nat)

## Function Implementation

All 11 Kit functions remain unchanged:
- `get_kit_instructions`
- `search_kit_extensions`
- `get_kit_extension_details`
- `get_kit_extension_dependencies`
- `get_kit_extension_apis`
- `get_kit_api_details`
- `search_kit_code_examples`
- `search_kit_test_examples`
- `search_kit_settings`
- `search_kit_app_templates`
- `get_kit_app_template_details`

Only the registration system was updated, not the actual function logic.

## Testing

After migration, test with:

```bash
cd source/aiq/kit_fns
poetry install
```

Then use in NAT workflows:

```yaml
functions:
  get_kit_instructions:
    _type: kit_fns/get_kit_instructions
    verbose: false
```

## Rollback

If you need to rollback to AIQ:

1. Checkout v0.5.0:
   ```bash
   git checkout <commit-before-migration> -- source/aiq/kit_fns/
   ```

2. Or manually revert:
   - Change `nvidia-nat` back to `aiqtoolkit`
   - Change imports from `nvidia_nat` back to `aiq`
   - Remove `[tool.poetry.plugins."nat.components"]` section

## Version History

- **v0.6.0** - Migrated to NAT from AIQ
- **v0.5.0** - Last AIQ-based version
