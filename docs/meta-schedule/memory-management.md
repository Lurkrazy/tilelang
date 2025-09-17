# Memory Management Documentation

## Overview

TileLang's memory management system provides sophisticated allocation strategies and optimization techniques for shared memory, registers, and global memory. The system is designed to maximize memory efficiency while respecting hardware constraints.

## Memory Hierarchy

### Global Memory
- **Characteristics**: High latency, high bandwidth, large capacity
- **Optimization**: Coalesced access patterns, minimal transfers
- **Usage**: Input/output tensor storage, large intermediate results

### Shared Memory
- **Characteristics**: Low latency, limited capacity (48KB-164KB per block)
- **Optimization**: Bank conflict avoidance, efficient allocation
- **Usage**: Tile data caching, inter-thread communication

### Registers
- **Characteristics**: Fastest access, very limited capacity (32K-64K per SM)
- **Optimization**: Minimal usage, efficient data reuse
- **Usage**: Temporary variables, accumulation buffers

## BestFit Memory Allocator

### Core Implementation

```python
class BestFit:
    """
    Best-fit memory allocator for shared memory optimization.
    Minimizes fragmentation and maximizes memory utilization.
    """
    
    def __init__(self, align=32):
        self.limit = 0          # Total allocated memory
        self.list = []          # List of memory blocks
        self.align = align      # Memory alignment requirement
```

#### Memory Block Structure

```python
class Block:
    def __init__(self, start, end, is_free):
        self.start = start      # Block start address
        self.end = end         # Block end address  
        self.is_free = is_free # Allocation status
    
    def size(self) -> int:
        return self.end - self.start
```

### Allocation Strategy

#### Best-Fit Algorithm

```python
def malloc(self, size) -> Block:
    """
    Allocates memory using best-fit strategy:
    1. Align size to hardware requirements
    2. Find smallest suitable free block
    3. Split block if necessary
    4. Create new block if no suitable block exists
    """
    size = (size + self.align - 1) // self.align * self.align
    
    # Find best-fit block
    found = None
    for block in self.list:
        if block.is_free and block.size() >= size:
            if not found or found.size() > block.size():
                found = block
```

#### Benefits of Best-Fit
1. **Minimal Fragmentation**: Chooses smallest suitable block
2. **Memory Efficiency**: Maximizes utilization of available memory
3. **Predictable Behavior**: Consistent allocation patterns

### Deallocation and Merging

```python
def free(self, block: Block) -> None:
    """
    Frees memory block and merges adjacent free blocks:
    1. Mark block as free
    2. Merge with adjacent free blocks
    3. Reduce fragmentation
    """
    # Merge with next block if free
    if idx + 1 < len(self.list) and self.list[idx + 1].is_free:
        self.list[idx].merge(self.list[idx + 1])
        self.list.pop(idx + 1)
    
    # Merge with previous block if free
    if idx - 1 >= 0 and self.list[idx - 1].is_free:
        self.list[idx].merge(self.list[idx - 1])
        self.list.pop(idx - 1)
```

## Shared Memory Usage Analysis

### Usage Computation

```python
def _compute_shared_memory_usage(self, td: TileDict):
    """
    Comprehensive shared memory analysis:
    1. Initialize BestFit allocator
    2. Process nodes in topological order
    3. Track tensor lifetimes
    4. Calculate peak memory usage
    """
    allocator = BestFit()
    block_map = {}
    processed = set()
    cached_tensors_map = {}
```

#### Tensor Lifetime Management

```python
def can_free(node, out_id):
    """
    Determines when tensors can be safely freed:
    - All consumers have been processed
    - No future references exist
    """
    for edge in node.outputs:
        if edge.src_id == out_id and edge.dst_node not in processed:
            return False
    return True
```

### Node-Specific Usage

```python
def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
    """
    Calculates shared memory usage for individual nodes:
    - Considers tile dimensions
    - Accounts for data types
    - Includes stride information
    """
    return node.footprint(
        td.get_tile(node), 
        td.get_rstep(node), 
        td.tensor_strides_map[node]
    )
```

## Memory Traffic Optimization

### Traffic Analysis

```python
def _compute_memory_traffic(self, output_tile):
    """
    Analyzes global memory access patterns:
    1. Propagate tile sizes through computation graph
    2. Calculate input/output tensor accesses
    3. Consider coalescing efficiency
    4. Account for transaction alignment
    """
    op_tile_map = self._get_output_tile_map(output_tile)
    traffic = 0
    
    for node in reversed(self.ordered_nodes):
        # Analyze input tensor accesses
        input_shapes = node.propagate_inputs(tile)
        for i, edge in enumerate(node.inputs):
            if edge.src_node.is_placeholder():
                nbytes = (edge.src_node.get_dtype().bits + 7) // 8
                read_elements = self.arch.transaction_size[1] // nbytes
                traffic += coalesced_tensor_shape(
                    input_shapes[i], 
                    edge.src_node.get_shape(),
                    read_elements
                ) * nbytes
```

#### Coalescing Optimization

```python
def coalesced_tensor_shape(tile_shape, tensor_shape, transaction_elements):
    """
    Calculates effective memory accesses considering coalescing:
    - Aligns accesses to transaction boundaries
    - Minimizes partial transactions
    - Optimizes for hardware memory controllers
    """
```

### Transaction Alignment

#### Read Transactions
```python
read_transaction_elements = self.arch.transaction_size[1] // nbytes
```

#### Write Transactions
```python
write_transaction_elements = self.arch.transaction_size[0] // nbytes
```

## Memory Stride Optimization

### Stride Calculation

```python
class Stride:
    """
    Manages memory stride information for efficient access patterns.
    """
    
    def __init__(self, stride: int = 1, ax: int = -1):
        self._ax = int(ax)       # Stride axis
        self._stride = int(stride)  # Stride value
    
    def compute_strides_from_shape(self, shape: List[int]) -> List[int]:
        """
        Computes memory strides for given tensor shape:
        - Ensures proper memory layout
        - Avoids bank conflicts
        - Optimizes cache usage
        """
```

#### Bank Conflict Avoidance

For TensorCore operations:
```python
# Apply padding to avoid shared memory bank conflicts
offset = 8
A_stride = Stride(
    stride=np.prod(AS_shape[A_high_ax + 1:]) + offset, 
    ax=A_high_ax
)
```

### Multi-Dimensional Stride Patterns

```python
def compute_strides_from_shape(self, shape: List[int]) -> List[int]:
    """
    Computes multi-dimensional stride patterns:
    1. Standard row-major layout
    2. Custom stride for specified axis
    3. Maintains memory alignment
    """
    ndim = len(shape)
    strides = [1 for _ in shape]
    
    for i in range(ndim - 2, -1, -1):
        if i == self.ax:
            strides[i] = self.stride
        else:
            strides[i] = int(strides[i + 1] * shape[i + 1])
```

## Register Usage Optimization

### Register Pressure Analysis

```python
# Estimate register usage per thread block
reg_usage = int(2 * max([
    np.prod(td.get_tile(node)) * node.get_dtype().bits / 32 
    for node in self.ordered_nodes
]))

# Validate against hardware limits
if reg_usage > self.arch.reg_cap:
    td.valid = False
    return td
```

#### Register Usage Factors

1. **Tile Size**: Larger tiles require more registers
2. **Data Type**: Higher precision increases register pressure
3. **Operation Count**: Multiple operations compound register usage
4. **Temporary Storage**: Intermediate calculations require registers

### Occupancy Optimization

```python
td.block_per_SM = min(
    self.arch.max_smem_usage // max(td.smem_cost, 1),    # Shared memory limit
    self.arch.reg_cap // max(reg_usage, 1),              # Register limit
    self.arch.sm_partition,                              # Hardware partition limit
)
```

## Memory Reuse Strategies

### Tensor Lifetime Analysis

```python
def _compute_shared_memory_usage(self, td: TileDict):
    """
    Optimizes memory reuse through lifetime analysis:
    1. Track tensor creation and destruction
    2. Identify reuse opportunities
    3. Minimize peak memory usage
    """
    for node in self.ordered_nodes:
        # Allocate memory for node operations
        node_bytes, cached_tensors = self.infer_node_smem_usage(td, node)
        block = allocator.malloc(node_bytes)
        
        # Free input tensors when no longer needed
        for edge in node.inputs:
            if can_free(edge.src_node, edge.src_id):
                allocator.free(block_map[edge.src_node])
```

### Cache-Aware Optimization

#### Temporal Locality
- **Principle**: Reuse recently accessed data
- **Implementation**: Schedule operations to maximize data reuse
- **Benefit**: Reduces memory bandwidth requirements

#### Spatial Locality
- **Principle**: Access nearby memory locations
- **Implementation**: Optimize tile layouts for sequential access
- **Benefit**: Improves cache hit rates

## Architecture-Specific Memory Features

### CUDA Memory Hierarchy

#### Shared Memory Configuration
```python
# Architecture-specific shared memory capacities
if self.arch.compute_capability >= "sm_80":
    max_shared_memory = 164 * 1024  # 164KB for Ampere+
elif self.arch.compute_capability >= "sm_70":
    max_shared_memory = 96 * 1024   # 96KB for Volta
else:
    max_shared_memory = 48 * 1024   # 48KB for older architectures
```

#### Memory Transaction Sizes
```python
# Transaction size optimization
self.transaction_size = [128, 128]  # Write, Read transaction sizes in bytes
```

### Memory Bandwidth Utilization

```python
def _score(node, warp_tile):
    """
    Scores memory access patterns based on bandwidth utilization:
    - Calculates effective bandwidth usage
    - Considers memory access patterns
    - Optimizes for hardware characteristics
    """
    score = 0
    shapes = node.propagate_inputs_on_reduction(warp_tile)
    for i, input_buffer in enumerate(node.input_buffers):
        score += np.prod(shapes[i]) / self.arch.bandwidth[1]
    return score
```

## Memory Optimization Techniques

### Dynamic Memory Allocation

```python
# Adaptive shared memory scope selection
if td.smem_cost > self.arch.smem_cap:
    # Use dynamic shared memory for large allocations
    codegen_dict.shared_scope = "shared.dyn"
    codegen_dict.max_smem_usage = td.smem_cost
else:
    # Use static shared memory for smaller allocations
    codegen_dict.shared_scope = "shared"
```

### Memory Access Pattern Optimization

#### Coalesced Access Patterns
```python
def coalesced_factor(tile_shape, tensor_shape):
    """
    Measures memory access coalescing efficiency:
    - Calculates wasted bandwidth
    - Identifies optimization opportunities
    - Guides tile size selection
    """
    # Implementation considers:
    # - Memory transaction alignment
    # - Access pattern regularity
    # - Hardware memory controller efficiency
```

#### Memory Interleaving
- **Technique**: Distribute memory accesses across banks
- **Implementation**: Use appropriate stride patterns
- **Benefit**: Eliminates memory bank conflicts

## Advanced Memory Features

### Asynchronous Memory Operations

For Ampere+ architectures:
```python
if self.use_async_copy:
    # Enable hardware-accelerated memory transfers
    # Overlap computation with memory operations
    # Improve overall throughput
    codegen_dict.use_async = True
```

### Memory Prefetching

```python
# Pipeline configuration for memory prefetching
codegen_dict.pipeline_stage = self.pipeline_stage

# Benefits:
# - Hides memory latency
# - Improves memory bandwidth utilization
# - Enables computation-memory overlap
```

## Performance Analysis and Tuning

### Memory Performance Metrics

#### Bandwidth Utilization
```python
effective_bandwidth = actual_bytes_transferred / theoretical_peak_bandwidth
```

#### Cache Hit Rates
```python
cache_efficiency = cache_hits / total_memory_accesses
```

#### Memory Latency
```python
average_latency = total_memory_latency / memory_operation_count
```

### Optimization Guidelines

1. **Maximize Coalescing**: Ensure memory accesses are aligned and sequential
2. **Minimize Transfers**: Reuse data in faster memory hierarchies
3. **Balance Resources**: Optimize trade-offs between different memory types
4. **Profile Real Workloads**: Validate optimizations with actual performance data

### Common Memory Issues

#### Bank Conflicts
- **Symptom**: Reduced shared memory bandwidth
- **Solution**: Apply stride padding and layout optimization

#### Register Spilling
- **Symptom**: Unexpected global memory accesses
- **Solution**: Reduce tile sizes or operation complexity

#### Poor Coalescing
- **Symptom**: Low memory bandwidth utilization
- **Solution**: Adjust access patterns and tile layouts

## Integration with Code Generation

The memory management system provides essential information for code generation:

```python
# Memory configuration for code generation
config.cached_tensors = td.cached_tensors_map[node]  # Shared memory allocations
config.shared_scope = "shared.dyn"                   # Memory scope selection
config.vectorize = vectorization_plan                # Memory access vectorization
config.rasterization_plan = rasterization_strategy   # Memory access ordering
```

This comprehensive memory management system enables efficient utilization of the GPU memory hierarchy while maintaining optimal performance across different architectures and workloads.