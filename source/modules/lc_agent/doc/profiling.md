# LC Agent Profiling System

The LC Agent profiling system provides comprehensive performance monitoring and visualization for language model applications. It tracks execution times across the entire network hierarchy and generates interactive visualizations for performance analysis.

## Overview

The profiling system captures detailed timing information for:
- Network execution (`network`)
- Node processing (`node`)
- Modifier hooks (`modifier`)
- Streaming chunks (`chunk`)
- Retriever operations (`retriever`)
- Parent processing (`process_parents`)
- Input combination (`combine_inputs`)
- Custom operations (`custom`)

## Quick Start

### Enable Profiling

There are two ways to enable profiling:

1. **Environment Variable** (recommended for development):
```bash
export LC_AGENT_PROFILING=1
python your_script.py
```

2. **Programmatically**:
```python
from lc_agent.utils.profiling_utils import enable_profiling
enable_profiling()
```

### Basic Usage

Once profiling is enabled, all network executions are automatically profiled:

```python
from lc_agent import RunnableNetwork, RunnableNode

# Profiling happens automatically when enabled
with RunnableNetwork() as network:
    node = RunnableNode()
    result = await network.astream({"query": "Hello"})

# Access profiling data
profiling_data = network.profiling
```

### Visualizing Results

Generate an interactive HTML visualization:

```python
from lc_agent.utils.profiling_html import create_profiling_html

# From a network object
create_profiling_html(network, "profiling_results.html")

# From a saved JSON file (no need to import node types)
import json
with open("saved_network.json", "r") as f:
    network_data = json.load(f)
create_profiling_html(network_data, "profiling_results.html")
```

## Profiler API

### Context Manager Usage

```python
from lc_agent.utils.profiling_utils import Profiler
from lc_agent import RunnableNetwork

network = RunnableNetwork.get_active_network()

with Profiler("operation_name", "custom", network=network, custom_data="value"):
    # Your code here
    expensive_operation()
```

### Auto-stop on Destruction

The profiler automatically stops when the object is destroyed:

```python
def process_data():
    p = Profiler("data_processing", "custom", network=network)
    # Profiling starts automatically
    
    result = complex_calculation()
    
    # Profiling stops automatically when function exits
    return result
```

### Manual Control

```python
p = Profiler("manual_operation", "custom", network=network, auto_start=False)
p.start()
# Your code here
p.stop()

# Update metadata after creation
p.update_metadata(result_count=42, status="success")
```

## Data Structure

### ProfilingFrame

Each profiling frame contains:
- `name`: Descriptive name of the operation
- `frame_type`: Type of operation (network, node, modifier, chunk, etc.)
- `start_time`: Start timestamp (from `time.perf_counter()`)
- `end_time`: End timestamp
- `duration`: Calculated duration in seconds
- `metadata`: Additional context information
- `children`: Nested profiling frames

### ProfilingData

The root container stored on `RunnableNetwork`:
- `enabled`: Whether profiling was active
- `frames`: List of root-level frames
- `total_duration`: Total execution time

## Built-in Profiling Points

### Network Level
- `network_astream`: Overall streaming execution time

### Node Level
- `node_stream_*`: Time for each node's streaming execution
- `process_parents_*`: Time to process parent nodes
- `combine_inputs_*`: Time to combine inputs from parents

### Modifier Level
- `pre_invoke_*`: Pre-invocation modifier hooks
- `post_invoke_*`: Post-invocation modifier hooks

### Chunk Level
- `chunk_0`, `chunk_1`, etc.: Time between streaming chunks
- Includes chunk content in metadata when available

### Retriever Level
- `retriever_message_execute`: Overall retriever execution
- `retriever_invoke`: Time to query the retriever
- `retriever_format_message`: Time to format results

## HTML Visualization Features

The interactive HTML visualization provides:

### Timeline View
- Hierarchical display of all profiling frames
- Color-coded by frame type
- Nested network sections with separators

### Interactive Controls
- **Zoom**: Mouse wheel or buttons (maintains position under cursor)
- **Pan**: Click and drag
- **Reset**: Return to default view

### Dynamic Time Ruler
- Automatically adjusts scale (nanoseconds to hours)
- Shows appropriate intervals based on zoom level

### Frame Details
- Hover tooltips with complete frame information
- Duration, start/end times, metadata
- Truncated chunk content display

### Visual Indicators
- Frame width represents duration
- Color indicates frame type
- Text only shown when frame is wide enough
- Sticky section labels when scrolling

## Advanced Usage

### Custom Profiling in Your Code

```python
class CustomProcessor:
    def process(self, data):
        network = RunnableNetwork.get_active_network()
        
        # Profile data validation
        p_validate = Profiler(
            "validate_input",
            "custom",
            network=network,
            data_size=len(data)
        )
        self.validate(data)
        p_validate.stop()
        
        # Profile main processing
        p_process = Profiler(
            "process_data",
            "custom",
            network=network
        )
        result = self.transform(data)
        # p_process auto-stops when method exits
        
        return result
```

### Profiling Nested Operations

```python
def complex_operation():
    p_outer = Profiler("outer_operation", "custom")
    
    for i in range(10):
        p_inner = Profiler(f"iteration_{i}", "custom", iteration=i)
        process_item(i)
        # p_inner auto-stops at end of loop iteration
    
    # p_outer auto-stops at function exit
```

### Conditional Profiling

```python
from lc_agent.utils.profiling_utils import is_profiling_enabled

if is_profiling_enabled():
    # Only create profiler objects when profiling is active
    p = Profiler("expensive_diagnostics", "custom")
    gather_detailed_metrics()
```
