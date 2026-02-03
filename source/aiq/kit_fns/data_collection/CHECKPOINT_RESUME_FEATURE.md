# Checkpoint/Resume Feature Documentation

## Overview

The data collection pipeline now includes a robust checkpoint/resume system that allows it to recover from failures without restarting from the beginning. This is critical for long-running pipelines that process hundreds of extensions.

## Key Features

1. **Automatic Checkpointing**: After each stage completes successfully, the pipeline saves a checkpoint
2. **Easy Recovery**: Use `--resume` to continue from the last successful stage
3. **Auto-Resume Mode**: Set `resume_on_failure = true` in config for automatic checkpoint detection
4. **Config Persistence**: Important config values (kit cache path, kit version) are saved in checkpoint
5. **Force Restart**: Use `--force` to ignore checkpoints and start fresh
6. **No Manual Intervention**: The checkpoint system works transparently in the background

## Files Modified

### 1. `data_collection_pipeline.py`
Added checkpoint/resume functionality to the main pipeline orchestrator:

- Added `checkpoint_file` attribute to store checkpoint location
- Added `_save_checkpoint()` method to save pipeline state
- Added `_load_checkpoint()` method to load previous state
- Added `_restore_config_from_checkpoint()` to restore configuration
- Added `_clear_checkpoint()` to clean up after successful completion
- Updated `run()` method to support `resume` and `force` parameters
- Added checkpoint save after each successful stage
- Added checkpoint preservation when a stage fails
- Updated command-line arguments to support `--resume` and `--force`

### 2. `run_pipeline.py`
Extended the easy runner script to support checkpoint/resume:

- Added `--resume` argument
- Added `--force` argument
- Updated help text with resume/force examples
- Updated pipeline.run() call to pass resume/force flags
- Added helpful hint when pipeline fails to suggest using `--resume`

### 3. `README.md`
Added comprehensive documentation section:

- Section 3: "Pipeline Recovery & Resume"
- Explains how checkpoint system works
- Provides usage examples
- Documents checkpoint file contents
- Shows example workflow for recovery

### 4. `test_checkpoint_system.py` (New File)
Created test script to verify checkpoint functionality:

- Tests checkpoint creation after successful stages
- Tests resume from checkpoint after failure
- Tests checkpoint clearing on success
- Validates config restoration from checkpoint

## Checkpoint File Format

The checkpoint file (`.pipeline_checkpoint.json`) is stored in the work directory and contains:

```json
{
  "timestamp": "2025-10-28T23:46:10.577567",
  "completed_stages": [
    "pull_repo_exts",
    "preparation",
    "extension_data"
  ],
  "config_snapshot": {
    "kit_cache_path": "/path/to/extscache",
    "kit_version": "107.3.0",
    "work_dir": "/path/to/work"
  }
}
```

## Usage Examples

### Basic Resume After Failure

```bash
# Start pipeline
python data_collection_pipeline.py

# If it fails at stage 3/8:
# "Pipeline failed at stage: code_examples"
# "You can resume from this point using: --resume"

# Resume from checkpoint
python data_collection_pipeline.py --resume
```

### Force Restart

```bash
# Ignore any existing checkpoint and start fresh
python data_collection_pipeline.py --force
```

### Using with run_pipeline.py

```bash
# Start pipeline with auto-detection
python run_pipeline.py

# If it fails, resume
python run_pipeline.py --resume

# Force restart
python run_pipeline.py --force
```

### Combining with Other Options

```bash
# Resume and run only up to a specific stage
python data_collection_pipeline.py --resume --end settings

# Force restart with limited extensions
python run_pipeline.py --force --max-extensions 25
```

### Using Config-Based Auto-Resume

Edit `pipeline_config.toml`:

```toml
[advanced]
resume_on_failure = true  # Enable automatic resume
```

Then run the pipeline normally:

```bash
# With auto-resume enabled, pipeline will automatically check for checkpoints
python data_collection_pipeline.py

# If checkpoint exists, it will resume automatically
# No need to pass --resume flag

# To override auto-resume and start fresh
python data_collection_pipeline.py --force
```

This is useful for automated/scheduled runs where you want automatic recovery without manual intervention.

## Implementation Details

### Stage Execution Flow with Checkpoints

1. Pipeline starts, loads checkpoint (if `--resume`)
2. Determines which stages to run based on completed stages
3. For each stage:
   - Run the stage
   - If successful: add to completed_stages, save checkpoint
   - If failed: save checkpoint with current completed stages, exit
4. After all stages complete: clear checkpoint

### Pre-stages Handling

Pre-stages (pull_repo_exts, preparation) are handled specially:
- They only run once at the beginning
- If any pre-stage failed, resume will re-run all pre-stages
- Once all pre-stages complete, they won't run again on resume

### Config Restoration

The checkpoint saves critical config values that may be set dynamically:
- `kit_cache_path`: Path to extensions cache (set by pull_repo_exts)
- `kit_version`: Kit version detected from build (set by pull_repo_exts)
- `work_dir`: Pipeline work directory

These values are restored when resuming to ensure consistency.

## Testing

Run the test script to verify the checkpoint system:

```bash
python test_checkpoint_system.py
```

Expected output:
```
Testing Checkpoint/Resume System
...
✅ All tests passed!

Checkpoint/resume system working correctly:
  - Checkpoints saved after each stage
  - Resume skips completed stages
  - Checkpoint cleared on success
```

## Benefits

1. **Time Savings**: No need to re-run hours of processing after a failure
2. **Resource Efficiency**: Only process what's needed
3. **Reliability**: Handles transient failures (network, API limits, etc.)
4. **User Experience**: Simple `--resume` flag, no manual state tracking
5. **Debugging**: Can inspect checkpoint file to see what completed

## Edge Cases Handled

1. **No checkpoint exists**: Resume gracefully falls back to full run
2. **Checkpoint corrupted**: Warning logged, starts from beginning
3. **Config changes**: Force flag available to ignore checkpoint
4. **Stage selection conflicts**: Resume overrides start_stage parameter
5. **All stages completed**: Detects and exits early with success message

## Future Enhancements (Potential)

- Add checkpoint versioning to handle schema changes
- Store more detailed error information when stages fail
- Add checkpoint inspection command (--show-checkpoint)
- Support for partial stage restarts (e.g., resume within a stage)
- Checkpoint cleanup command (--clear-checkpoint)

## Backward Compatibility

The checkpoint system is fully backward compatible:
- Existing scripts without `--resume` work exactly as before
- Checkpoints are optional and don't interfere with normal operation
- No changes required to individual pipeline stages
- No changes to data formats or outputs
