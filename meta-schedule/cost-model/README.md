# Cost Model Analysis

This document provides a detailed analysis of TileLang's cost modeling framework, which is implemented primarily in `tilelang/tools/Analyzer.py` and integrates with the scheduling policies in `tilelang/carver/roller/policy/`.

## Overview

TileLang's cost model is a hardware-aware performance analysis framework that provides:
- **Accurate FLOP counting** for arithmetic operations
- **Memory traffic analysis** for global memory transfers  
- **Roofline modeling** for performance estimation
- **Architecture-specific optimization** for different GPU architectures

## Core Components

### 1. Analyzer Class (`tilelang/tools/Analyzer.py`)

The `Analyzer` class is the main entry point for performance analysis:

```python
class Analyzer:
    def __init__(self, fn, device):
        self.fn = fn  # TVM IRModule or PrimFunc
        self.device = device  # Target device information
        self.total_flops = 0
        self.total_global_bytes = 0
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1}
        self.global_buffers = set()
```

#### Key Methods:

**`ir_pass()`**: Traverses the IR module to extract performance metrics
- Uses TVM's `ir_transform` to visit all nodes
- Identifies operation types (`tl.copy`, `tl.gemm`)
- Accumulates FLOP counts and memory traffic

**`calculate()`**: Computes final performance metrics using roofline model
- Estimates execution time based on compute and memory bounds
- Returns `AnalysisResult` with TFLOPS and bandwidth utilization

### 2. Memory Traffic Analysis

The cost model tracks global memory transfers through `_analyze_copy()`:

```python
def _analyze_copy(self, call):
    # Determine if source or destination is global buffer
    if src_buffer in self.global_buffers:
        buffer_region = call.args[0]
    elif dst_buffer in self.global_buffers:
        buffer_region = call.args[1]
    
    # Calculate elements and bytes transferred
    elements = 1
    for r in range(2, len(buffer_region.args)):
        elements *= buffer_region.args[r]
    
    dtype_size = np.dtype(buffer_region.args[0].buffer.dtype).itemsize
    bytes_transferred = elements * dtype_size
    
    # Account for loop and block dimensions
    loop_product = 1
    for extent in self.loop_stack:
        loop_product *= extent.value if hasattr(extent, 'value') else extent
    total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
    total_bytes = bytes_transferred * loop_product * total_blocks
```

### 3. FLOP Counting

GEMM operations are analyzed through `_analyze_gemm()`:

```python
def _analyze_gemm(self, call):
    M = call.args[5].value
    N = call.args[6].value  
    K = call.args[7].value
    flops_per_call = 2 * M * N * K  # Standard GEMM FLOP count
    
    # Scale by loop iterations and block parallelism
    loop_product = 1
    for extent in self.loop_stack:
        loop_product *= extent.value if hasattr(extent, 'value') else extent
    total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
    self.total_flops += flops_per_call * loop_product * total_blocks
```

## Roofline Performance Model

### Architecture Configuration

Performance limits are defined per architecture in `ARCH_CONFIGS`:

```python
ARCH_CONFIGS = {
    "80": (128, 1.41, 2, 108),  # A100: cores/SM, clock_GHz, flops/cycle, max_SMs
    "86": (128, 1.70, 2, 84),   # RTX 3080
    "89": (128, 2.52, 2, 128)   # RTX 4090
}
```

### Performance Calculation

The roofline model considers both compute and memory bounds:

```python
def calculate(self) -> AnalysisResult:
    # Get peak TFLOPS capability
    arch_key = device.compute_capability[:2]
    cores_per_sm, default_clock, flops_per_cycle, compute_max_core = ARCH_CONFIGS[arch_key]
    total_cores = compute_max_core * cores_per_sm
    peak_tflops = (total_cores * default_clock * flops_per_cycle) / 1e3
    
    # Memory bandwidth limit
    bandwidth_GBps = self.device.bandwidth[1] / 1000
    
    # Compute time bounds
    compute_time = self.total_flops / (peak_tflops * 1e12)
    mem_time = self.total_global_bytes / (bandwidth_GBps * 1e9)
    
    # Roofline: max of compute and memory bound
    estimated_time = max(mem_time, compute_time)
```

## Integration with Scheduling Policies

### Shared Memory Usage Computation

The cost model integrates with scheduling policies through shared memory analysis in `DefaultPolicy._compute_shared_memory_usage()`:

```python
def _compute_shared_memory_usage(self, td: TileDict):
    self._compute_stride_map(td)
    allocator = BestFit()  # Memory allocator simulation
    block_map = {}
    cached_tensors_map = {}
    
    for node in self.ordered_nodes:
        # Compute node's memory footprint
        node_internal_bytes, cached_tensors_map[node] = self.infer_node_smem_usage(td, node)
        
        # Simulate allocation/deallocation
        block = allocator.malloc(node_internal_bytes)
        allocator.free(block)
        
        # Track tensor lifetimes for memory reuse
        for edge in node.inputs:
            if can_free(edge.src_node, edge.src_id):
                allocator.free(block_map.pop((edge.src_node, edge.src_id)))
    
    return allocator.limit, cached_tensors_map
```

### Memory Footprint Calculation

Node memory usage is computed via `infer_node_smem_usage()`:

```python
def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
    return node.footprint(
        td.get_tile(node), 
        td.get_rstep(node), 
        td.tensor_strides_map[node]
    )
```

## Memory Allocator Simulation

### BestFit Allocator

TileLang uses a best-fit memory allocation strategy implemented in `tilelang/carver/roller/bestfit.py`:

```python
class BestFit:
    def __init__(self, align=32):
        self.limit = 0  # Total allocated memory
        self.list = []  # List of memory blocks
        self.align = align  # Memory alignment
    
    def malloc(self, size) -> Block:
        size = (size + self.align - 1) // self.align * self.align
        # Find best fitting free block
        found = None
        for block in self.list:
            if (block.is_free and block.size() >= size and 
                (not found or found.size() > block.size())):
                found = block
        
        if found:
            found.is_free = False
            # Split block if larger than needed
            remain = found.size() - size
            if remain != 0:
                found.end -= remain
                self.list.insert(
                    self.list.index(found) + 1, 
                    Block(found.end, found.end + remain, True)
                )
            return found
        else:
            # Allocate new block
            block = Block(self.limit, self.limit + size, False)
            self.list.append(block)
            self.limit += size
            return block
```

## Cost Model Applications

### 1. Tile Size Optimization

The cost model guides tile size selection by:
- Estimating shared memory requirements for different tile configurations
- Balancing memory usage with parallelism
- Avoiding configurations that exceed hardware limits

### 2. Memory Coalescing Analysis

Memory access patterns are analyzed through the `coalesced_factor` function:

```python
def coalesced_factor(subtensor: List[int], tensor: List[int]) -> int:
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])

def coalesced_tensor_shape(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    bytes = int(np.prod(subtensor))
    factor = int(coalesced_factor(subtensor, tensor))
    return transaction_size * bytes / min(transaction_size, factor)
```

### 3. Performance Prediction Accuracy

The cost model provides realistic performance estimates by:
- Accounting for hardware constraints (memory bandwidth, compute throughput)
- Modeling memory hierarchy effects (cache, shared memory)
- Considering parallelization overheads

## Key Metrics and Formulas

| Metric | Formula | Description |
|--------|---------|-------------|
| FLOPs per GEMM | `2 × M × N × K` | Standard matrix multiplication operations |
| Memory Traffic per Copy | `elements × dtype_size × loop_product` | Bytes transferred in copy operations |
| Achieved TFLOPS | `total_flops / estimated_time / 1e12` | Actual computational throughput |
| Memory Bandwidth | `total_global_bytes / estimated_time` | Memory subsystem utilization |
| Compute Time | `Total FLOPs / (SM Count × Cores/SM × Clock × FLOPs/Cycle)` | Compute-bound execution time |
| Memory Time | `Memory Bytes / (Bandwidth × Utilization)` | Memory-bound execution time |

## Limitations and Considerations

1. **Perfect Memory Coalescing Assumption**: The model assumes optimal memory access patterns
2. **No Bank Conflict Modeling**: Shared memory bank conflicts are not explicitly modeled
3. **Static Analysis**: Runtime behavior variations are not captured
4. **Architecture Coverage**: Limited to specific GPU architectures in `ARCH_CONFIGS`

## Future Enhancements

1. **Dynamic Profiling Integration**: Incorporate runtime measurements for model calibration
2. **Cache Modeling**: Add L1/L2 cache behavior modeling
3. **Memory Latency Modeling**: Include memory access latency effects
4. **Multi-GPU Scaling**: Extend model for distributed workloads