# Cost Model Documentation

## Overview

The cost model in TileLang's meta-scheduling system provides a comprehensive framework for analyzing and optimizing computational kernels. It estimates performance by modeling memory traffic, register usage, shared memory consumption, and parallelism efficiency.

## Architecture

### Core Components

#### 1. DefaultPolicy Class
The `DefaultPolicy` class implements the base cost model with the following key responsibilities:

- **Memory Traffic Analysis**: Estimates global memory access patterns
- **Shared Memory Usage**: Models shared memory allocation and usage
- **Register Usage**: Estimates register pressure for different tile configurations
- **Parallelism Analysis**: Evaluates thread block scheduling and occupancy

#### 2. Key Cost Metrics

##### Memory Traffic Cost
```python
def _compute_memory_traffic(self, output_tile):
    """
    Computes the memory traffic for a given output tile configuration.
    
    Returns:
        Tuple[int, Dict]: The total memory traffic and operation tile map
    """
```

The memory traffic calculation considers:
- **Coalesced Access Patterns**: Optimizes for efficient memory transactions
- **Transaction Size Alignment**: Aligns accesses to hardware transaction boundaries
- **Read/Write Patterns**: Separate analysis for input and output tensors

##### Shared Memory Cost
```python
def _compute_shared_memory_usage(self, td: TileDict):
    """
    Computes shared memory usage using best-fit allocation strategy.
    
    Uses BestFit allocator to model actual memory layout and fragmentation.
    """
```

Key features:
- **Best-Fit Allocation**: Models realistic memory allocation patterns
- **Memory Reuse**: Tracks tensor lifetime for optimal memory reuse
- **Fragmentation Analysis**: Accounts for memory fragmentation overhead

##### Register Usage Cost
```python
# Estimated register usage calculation
reg_usage = int(2 * max([
    np.prod(td.get_tile(node)) * node.get_dtype().bits / 32 
    for node in self.ordered_nodes
]))
```

Register cost modeling includes:
- **Data Type Considerations**: Accounts for different precision requirements
- **Tile Size Impact**: Larger tiles require more registers
- **Multiple Node Analysis**: Considers all operations in the kernel

## Cost Model Algorithms

### 1. Tile Configuration Evaluation

The cost model evaluates tile configurations using the `compute_tile_dict` method:

```python
def compute_tile_dict(self, output_tile: List[int], rstep_map) -> TileDict:
    """
    Core cost evaluation function that:
    1. Computes memory traffic
    2. Analyzes shared memory usage
    3. Estimates register pressure
    4. Calculates occupancy metrics
    """
```

#### Process Flow:
1. **Traffic Analysis**: Calculate memory access patterns
2. **Shared Memory Check**: Validate shared memory constraints
3. **Register Validation**: Ensure register usage is within limits
4. **Occupancy Calculation**: Determine block-per-SM ratio

### 2. Reduction Axis Optimization

The `_expand_reduce_axis` method optimizes reduction dimensions:

```python
def _expand_reduce_axis(self, td: TileDict):
    """
    Expands reduction axis based on shared memory limits to improve
    compute efficiency while maintaining memory constraints.
    """
```

#### Optimization Strategy:
- **Coalescing Score**: Evaluates memory access efficiency
- **Memory Limit Enforcement**: Respects shared memory constraints  
- **Iterative Improvement**: Progressively improves tile configuration

### 3. Block Size Assignment

The cost model determines optimal thread block configurations:

```python
def assign_block_size(self, td: TileDict, topk=1):
    """
    Determines optimal thread block sizes considering:
    - Warp efficiency
    - Memory access patterns
    - Hardware constraints
    """
```

## Hardware-Aware Optimizations

### Memory Hierarchy Modeling

#### Transaction Size Optimization
```python
# Calculate optimal transaction alignment
read_transaction_elements = self.arch.transaction_size[1] // nbytes
write_transaction_elements = self.arch.transaction_size[0] // nbytes
```

#### Bandwidth Considerations
The cost model considers different bandwidth characteristics:
- **Global Memory Bandwidth**: Used for tile size recommendations
- **Shared Memory Bandwidth**: Influences memory reuse strategies

### Architecture-Specific Optimizations

#### CUDA Architecture Support
- **Compute Capability**: Different optimization strategies for different SM versions
- **Warp Size**: 32-thread warp optimization
- **Shared Memory Capacity**: Varies by architecture (48KB-164KB)

#### Register Pressure Management
```python
td.block_per_SM = min(
    self.arch.max_smem_usage // max(td.smem_cost, 1),
    self.arch.reg_cap // max(reg_usage, 1),
    self.arch.sm_partition,
)
```

## Cost Model Metrics

### Performance Indicators

#### 1. Memory Traffic Score
- **Lower is Better**: Minimizes global memory accesses
- **Coalescing Factor**: Measures memory access efficiency
- **Transaction Alignment**: Optimizes for hardware memory controllers

#### 2. Occupancy Score
- **Blocks per SM**: Number of concurrent thread blocks
- **Wave Count**: Number of kernel launches needed
- **Resource Utilization**: Balance between memory and compute resources

#### 3. Efficiency Metrics
- **Register Efficiency**: Register usage vs. capacity ratio
- **Memory Efficiency**: Shared memory usage optimization
- **Compute Efficiency**: Arithmetic intensity analysis

## Advanced Features

### Dynamic Memory Management

The cost model includes sophisticated memory management:

```python
def can_free(node, out_id):
    """Determines when tensors can be freed based on usage analysis"""
    for edge in node.outputs:
        if edge.src_id == out_id and edge.dst_node not in processed:
            return False
    return True
```

### Multi-Node Optimization

For kernels with multiple operations:
- **Dependency Analysis**: Respects operation dependencies
- **Memory Reuse**: Optimizes tensor lifetime management
- **Pipeline Optimization**: Considers operation ordering

### Validation and Constraints

#### Tile Shape Validation
```python
def check_tile_shape_isvalid(self, td: TileDict) -> bool:
    """
    Validates tile configurations against:
    - Hardware constraints
    - Memory limitations
    - Alignment requirements
    """
```

## Usage Examples

### Basic Cost Evaluation
```python
# Create policy for target architecture
policy = DefaultPolicy(arch=cuda_arch)

# Evaluate tile configuration
output_tile = [128, 128]
rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}
tile_dict = policy.compute_tile_dict(output_tile, rstep_map)

# Access cost metrics
print(f"Memory Traffic: {tile_dict.traffic}")
print(f"Shared Memory Cost: {tile_dict.smem_cost}")
print(f"Blocks per SM: {tile_dict.block_per_SM}")
```

### Configuration Generation
```python
# Generate optimized configurations
configs = policy.emit_config(topk=10)
for config in configs:
    print(f"Block: {config.block}")
    print(f"Warp: {config.warp}")
    print(f"Memory Usage: {config.smem_cost}")
```

## Implementation Details

### Data Structures

#### TileDict
Central configuration container:
- `output_tile`: Output tensor tiling configuration
- `tile_map`: Per-operation tile configurations
- `rstep_map`: Reduction step configurations
- `traffic`: Computed memory traffic cost
- `smem_cost`: Shared memory usage
- `block_per_SM`: Computed occupancy

#### Hardware Device Abstraction
- `reg_cap`: Register capacity
- `smem_cap`: Shared memory capacity
- `compute_max_core`: Maximum compute cores
- `transaction_size`: Memory transaction sizes
- `bandwidth`: Memory bandwidth specifications

## Best Practices

### Optimization Guidelines

1. **Balance Memory and Compute**: Optimize for both memory efficiency and compute utilization
2. **Consider Hardware Limits**: Respect register, shared memory, and occupancy constraints
3. **Profile Real Workloads**: Validate cost model predictions with actual performance
4. **Iterative Refinement**: Use multiple configurations to find optimal solutions

### Common Pitfalls

1. **Ignoring Memory Coalescing**: Poor memory access patterns can severely impact performance
2. **Excessive Register Usage**: High register pressure reduces occupancy
3. **Shared Memory Overuse**: Exceeding shared memory limits invalidates configurations
4. **Inadequate Reduction Optimization**: Missing reduction axis optimization opportunities

This cost model provides a comprehensive framework for automated kernel optimization, enabling efficient scheduling and resource utilization across different hardware architectures.