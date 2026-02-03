# OmniUI Functions Tests

This directory contains tests for the `omni_ui_fns` package.

## Running Tests

### Run with Python directly

```bash
# Set your NVIDIA API key
export NVIDIA_API_KEY="your-api-key-here"  # Linux/macOS
set NVIDIA_API_KEY=your-api-key-here       # Windows

# Run the test
cd source/aiq/omni_ui_fns/src/omni_ui_fns/tests
python test_search_ui_code_examples.py
```

### Run with pytest

```bash
cd source/aiq/omni_ui_fns
pytest src/omni_ui_fns/tests/test_search_ui_code_examples.py -v -s
```

## Test Coverage

### `test_search_ui_code_examples.py`

Tests the code examples retrieval functionality:

1. **test_faiss_index_exists**: Verifies FAISS index is present
2. **test_retriever_initialization**: Tests retriever can load the index
3. **test_retriever_search**: Tests search returns documents with correct metadata
4. **test_rag_context_formatting**: Tests output formatting is correct
5. **test_search_ui_code_examples_async**: Tests the full async function end-to-end

## What These Tests Validate

The tests verify that:
- ✅ FAISS index exists and can be loaded
- ✅ Retrieved documents have correct metadata structure
- ✅ Metadata contains: `description`, `code`, `file_path`, `function_name`, `class_name`
- ✅ Output formatting includes proper file names, paths, and method names
- ✅ Code blocks contain actual code (not descriptions)
- ✅ Method names are formatted as `ClassName.function_name`
- ✅ Minimal "unknown" values appear in output

## Expected Output

When tests pass, you should see formatted code examples like:

```
### Example 1: Description of what the code does
File: example_file.py
Path: /path/to/example_file.py
Method: ClassName.function_name

```python
def function_name(self):
    # Actual function code here
    pass
```
```

## Troubleshooting

**FAISS index not found:**
- Run the data generation pipeline first
- Check `FAISS_CODE_INDEX_PATH` in config

**API key errors:**
- Ensure `NVIDIA_API_KEY` environment variable is set
- Verify the key is valid

**"unknown" values in output:**
- This indicates metadata mismatch
- Check FAISS database was built with correct schema
- Verify retrieval.py is reading correct metadata keys
