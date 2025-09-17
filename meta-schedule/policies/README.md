# Scheduling Policies

This document provides a comprehensive analysis of TileLang's scheduling policy framework, which is the core of the meta-scheduling system for automatic kernel optimization.

## Overview

TileLang's scheduling policy framework provides a systematic approach to generating and evaluating kernel configurations. The framework includes:

- **Default Policy**: General-purpose heuristic-based scheduling
- **TensorCore Policy**: Specialized scheduling for mixed-precision tensor operations
- **Extensible Framework**: Support for custom policies and domain-specific optimizations

## Policy Architecture

### Base Policy Structure

All scheduling policies inherit from the `DefaultPolicy` class in `tilelang/carver/roller/policy/default.py`:

```python
class DefaultPolicy:
    """
    Default Policy for fastdlight, a heuristic plan that tries to
    minimize memory traffic and maximize parallelism.
    """
    
    func: tvm.tir.PrimFunc
    nodes: List[PrimFuncNode] = []
    arch: TileDevice
    tags: Dict
    
    def __init__(self, arch: TileDevice, tags: Optional[Dict] = None):
        self.arch = arch
        self.tags = tags if tags else {}
        self.rasterization = NoRasterization()
```

### Policy Initialization

Policies can be created from different input sources:

```python
@classmethod
def from_prim_func(cls, func: tvm.tir.PrimFunc, arch: TileDevice, tags: Optional[Dict] = None):
    return cls(arch, tags)._init_with_prim_func(func, name)

@classmethod  
def from_output_nodes(cls, nodes: List[OutputNode], arch: TileDevice, tags: Optional[Dict] = None):
    return cls(arch, tags)._init_with_output_nodes(nodes)
```

## Default Policy Implementation

### Core Workflow

The default policy follows this optimization workflow:

1. **Base Tile Generation**: `get_base_tile()`
2. **Reduce Step Assignment**: `_assign_reduce_step()`
3. **Shared Memory Tile Search**: `dfs_smem_tile()`
4. **Tile Validation**: `check_tile_shape_isvalid()`
5. **Reduce Axis Expansion**: `_expand_reduce_axis()`
6. **Block Size Assignment**: `assign_block_size()`

### Configuration Generation

```python
def emit_config(self, topk: int) -> List[Hint]:
    base_tile = self.get_base_tile()
    if base_tile is None:
        return []
    
    # Assign reduce steps for all nodes
    rstep_map = {node: self._assign_reduce_step(node) for node in self.ordered_nodes}
    
    # Generate tile candidates through DFS
    smem_tile_candidates = self.dfs_smem_tile(base_tile, rstep_map)
    
    results = []
    for td in smem_tile_candidates:
        if not self.check_tile_shape_isvalid(td):
            continue
        
        # Optimize reduce axis expansion
        self._expand_reduce_axis(td)
        
        # Generate block size configurations
        for codegen_dicts in self.assign_block_size(td):
            if isinstance(codegen_dicts, dict) and len(codegen_dicts) == 1:
                results.append(list(codegen_dicts.values())[0])
            else:
                results.append(codegen_dicts)
            
            if len(results) >= topk:
                break
                
    return results[:topk]
```

## Tile Dictionary (TileDict) Management

### TileDict Structure

The `TileDict` class manages tiling information and configurations:

```python
class TileDict:
    def __init__(self, output_tile):
        self.output_tile = output_tile
        
        # Schedule configuration
        self.tile_map = {}          # Node -> tile mapping
        self.rstep_map = {}         # Node -> reduce step mapping
        self.cached_tensors_map = {} # Node -> cached tensors
        self.output_strides_map = {} # Node -> output strides
        self.tensor_strides_map = {} # Node -> tensor strides
        
        # Analysis results
        self.traffic = -1           # Memory traffic cost
        self.smem_cost = -1         # Shared memory usage
        self.block_per_SM = -1      # Blocks per SM
        self.num_wave = -1          # Number of waves
        self.grid_size = -1         # Grid dimensions
        self.valid = True           # Configuration validity
```

### Tile Configuration Computation

```python
def compute_tile_dict(self, output_tile: List[int], rstep_map) -> TileDict:
    td = TileDict(output_tile)
    td.rstep_map = rstep_map
    
    # Compute memory traffic
    td.traffic, td.tile_map = self._compute_memory_traffic(output_tile)
    
    # Compute shared memory usage
    td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
    
    if td.smem_cost > self.arch.smem_cap:
        td.valid = False
        return td
    
    # Compute grid configuration
    output_shape = self.output_nodes[0].get_space_dim()
    td.grid_size = 1
    for i, dim_size in enumerate(output_shape):
        td.grid_size *= (dim_size + output_tile[i] - 1) // output_tile[i]
    
    td.block_per_SM = self.arch.compute_max_core // td.grid_size
    td.block_per_SM = max(1, min(td.block_per_SM, self.arch.sm_partition))
    td.num_wave = (td.grid_size + self.arch.compute_max_core - 1) // self.arch.compute_max_core
    
    return td
```

## Memory Management and Optimization

### Shared Memory Usage Analysis

The policy computes shared memory requirements using a sophisticated allocator simulation:

```python
def _compute_shared_memory_usage(self, td: TileDict):
    self._compute_stride_map(td)
    allocator = BestFit(align=32)  # Memory allocator with 32-byte alignment
    block_map = {}
    processed = set()
    cached_tensors_map = {}
    
    def can_free(node, out_id):
        # Check if output tensor can be freed
        for edge in node.outputs:
            if edge.src_id == out_id and edge.dst_node not in processed:
                return False
        return True
    
    # Simulate execution order and memory allocation
    for node in self.ordered_nodes:
        # Allocate internal node memory
        node_internal_bytes, cached_tensors_map[node] = self.infer_node_smem_usage(td, node)
        block = allocator.malloc(node_internal_bytes)
        allocator.free(block)
        
        processed.add(node)
        
        # Free input tensors when no longer needed
        for edge in node.inputs:
            if not edge.src_node.is_placeholder() and can_free(edge.src_node, edge.src_id):
                allocator.free(block_map.pop((edge.src_node, edge.src_id)))
        
        # Allocate output tensors
        for edge in node.outputs:
            if not edge.dst_node.is_output() and (node, edge.src_id) not in block_map:
                dtype_bytes = (node.get_dtype(edge.src_id).bits + 7) // 8
                stride = td.output_strides_map[node][len(node.inputs) + edge.src_id]
                output_elem = stride.compute_elements_from_shape(td.get_tile(node))
                block_map[(node, edge.src_id)] = allocator.malloc(output_elem * dtype_bytes)
    
    assert len(block_map) == 0  # All tensors should be freed
    return allocator.limit, cached_tensors_map
```

### Stride Map Computation

```python
def _compute_stride_map(self, td: TileDict):
    output_strides_map = {}
    tensor_strides_map = {}
    
    for node in self.ordered_nodes:
        output_strides_map[node], tensor_strides_map[node] = self.compute_node_stride_map(node, td)
    
    td.output_strides_map = output_strides_map
    td.tensor_strides_map = tensor_strides_map

def compute_node_stride_map(self, node: PrimFuncNode, td: TileDict):
    # Default implementation creates identity strides
    output_strides = {
        int(i + len(node.input_buffers)): Stride() 
        for i, _ in enumerate(node.output_buffers)
    }
    tensor_strides = {}
    return output_strides, tensor_strides
```

## Block Size Assignment Strategy

### Scoring Function

The default policy uses a sophisticated scoring function to evaluate thread block configurations:

```python
def score_block_size(self, n):
    """
    Scores a block size based on efficiency and fit relative to architecture.
    Lower scores are better.
    """
    num_wrap = (n + self.arch.warp_size - 1) // self.arch.warp_size
    
    # Efficiency score: how well block size fits SM partition
    r1 = max(num_wrap / self.arch.sm_partition, self.arch.sm_partition / num_wrap)
    
    # Utilization score: wasted threads within warps
    r2 = (num_wrap * self.arch.warp_size - n) / n
    
    return (r1, r2)

def get_block_size(self, n):
    """
    Determines optimal block size for given constraint.
    """
    factors = get_all_factors(n)
    factors = list(filter(lambda x: x <= 1024, factors))  # Max threads per block
    factor_ordered = sorted(factors, key=self.score_block_size)
    return factor_ordered[0]
```

### Thread Assignment Algorithm

```python
def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
    tile, rsteps = td.get_tile(node), td.get_rstep(node)
    factors = factorize(block_size)
    cur_threads = [1 for _ in tile]
    reduce_thread = {k: 1 for k in rsteps}
    ndim = len(tile)
    
    def _score(node, thread):  # Lower is better
        score = 0
        block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
        shape = node.propagate_inputs(block_tile)
        
        # Memory bandwidth scoring
        for i, _ in enumerate(node.input_buffers):
            score += np.prod(shape[i]) / self.arch.bandwidth[1]
        
        # Memory coalescing scoring
        for buffer in node.output_buffers:
            score += coalesced_tensor_shape(thread, buffer.shape, 8) / self.arch.bandwidth[0]
        
        return score
    
    # Distribute factors across dimensions to minimize score
    for factor in reversed(factors):
        score_map = {}
        
        # Try assigning to spatial dimensions
        for i in range(ndim):
            if tile[i] % (cur_threads[i] * factor) == 0:
                test_threads = cur_threads.copy()
                test_threads[i] *= factor
                score_map[i] = _score(node, test_threads)
        
        # Try assigning to reduce dimensions
        for k in rsteps:
            if rsteps[k] % (reduce_thread[k] * factor) == 0:
                test_reduce_thread = reduce_thread.copy()
                test_reduce_thread[k] *= factor
                score_map[f"reduce_{k}"] = _score(node, cur_threads)
        
        # Select best assignment
        if score_map:
            best_key = min(score_map.items(), key=lambda x: x[1])[0]
            if isinstance(best_key, int):
                cur_threads[best_key] *= factor
            else:
                k = best_key.replace("reduce_", "")
                reduce_thread[k] *= factor
    
    # Create configuration hint
    hint = Hint()
    hint.block = tile
    hint.thread = cur_threads
    hint.rstep = list(rsteps.values())
    hint.reduce_thread = list(reduce_thread.values())
    hint.arch = self.arch
    
    return hint
```

## Reduce Axis Optimization

### Reduce Step Assignment

```python
def _assign_reduce_step(self, node: PrimFuncNode):
    target_bytes = self.arch.transaction_size[0] * 2  # Target 512 bytes
    input_dtype = node.get_buffer_dtype(node.input_buffers[0])
    basic_elements = (target_bytes * 8) // input_dtype.bits
    
    result = {}
    for axis_info in node.raxis:
        axis_name = axis_info.var.name
        axis_extent = axis_info.dom.extent
        
        # Find optimal reduce step size
        candidates = get_all_factors(axis_extent)
        candidates = [x for x in candidates if x <= basic_elements]
        
        if candidates:
            result[axis_name] = candidates[-1]  # Largest factor <= basic_elements
        else:
            result[axis_name] = min(basic_elements, axis_extent)
    
    return result
```

### Reduce Axis Expansion

```python
def _expand_reduce_axis(self, td: TileDict):
    """
    Expands reduce axes to improve compute efficiency when tiles are small.
    """
    def _check_small_tile(td: TileDict):
        for node in self.ordered_nodes:
            tile = td.get_tile(node)
            if np.prod(tile) < 256:  # Small tile threshold
                return True
        return False
    
    if _check_small_tile(td):
        smem_limit = min(
            self.arch.max_smem_usage // td.block_per_SM, 
            self.arch.smem_cap
        )
        
        rstep_map = td.rstep_map.copy()
        
        def _optimize(node, rstep):
            # Generate larger reduce step candidates
            all_steps = {}
            for axis in node.raxis:
                axis_name = axis.var.name
                axis_extent = axis.dom.extent
                candidates = get_all_factors(axis_extent)
                candidates = [x for x in candidates if x >= rstep[axis_name]]
                all_steps[axis_name] = candidates
            
            # Binary search for optimal configuration
            cur_rstep_id = {k: 0 for k in all_steps}
            
            for axis in node.raxis:
                axis_name = axis.var.name
                for step_id in range(len(all_steps[axis_name])):
                    new_rstep_id = cur_rstep_id.copy()
                    new_rstep_id[axis_name] = step_id
                    
                    # Test memory usage with new configuration
                    new_rstep_map = rstep_map.copy()
                    new_rstep_map[node] = {
                        k: all_steps[k][new_rstep_id[k]] 
                        for k in new_rstep_id
                    }
                    
                    old_rstep_map = td.rstep_map
                    td.rstep_map = new_rstep_map
                    smem_usage, _ = self._compute_shared_memory_usage(td)
                    td.rstep_map = old_rstep_map
                    
                    if smem_usage > smem_limit:
                        break
                    else:
                        cur_rstep_id = new_rstep_id
            
            return {k: all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        
        # Apply optimization to all nodes with reduce axes
        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map[node])
                rstep_map[node] = rstep
        
        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
```

## Memory Coalescing Analysis

### Coalescing Factor Computation

```python
def coalesced_factor(subtensor: List[int], tensor: List[int]) -> int:
    """
    Computes the coalescing factor for memory access patterns.
    """
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])

def coalesced_tensor_shape(subtensor: List[int], tensor: List[int], transaction_size: int) -> int:
    """
    Computes effective memory traffic considering coalescing.
    """
    bytes = int(np.prod(subtensor))
    if bytes == 0:
        return 0
    
    factor = int(coalesced_factor(subtensor, tensor))
    return transaction_size * bytes / min(transaction_size, factor)
```

## Workload Analysis

### Memory Traffic Computation

```python
def compute_workload_per_item(self, output_tile) -> float:
    """
    Computes memory traffic per output element.
    """
    def _compute_item_traffic(node_tile):
        total_input_size = 0
        for input_buffer in node.input_buffers:
            input_shape = node.propagate_input_shape(node_tile, input_buffer)
            total_input_size += np.prod(input_shape)
        
        output_size = np.prod(node_tile)
        return (total_input_size + output_size) / output_size
    
    total_traffic = 0
    for node in self.ordered_nodes:
        node_tile = self._get_output_tile_map(output_tile)[node]
        total_traffic += _compute_item_traffic(node_tile)
    
    return total_traffic
```

## Policy Extension Framework

### Custom Policy Implementation

To create a custom policy, extend the `DefaultPolicy` class:

```python
class CustomPolicy(DefaultPolicy):
    def __init__(self, arch: TileDevice, tags: Optional[Dict] = None):
        super().__init__(arch, tags)
        # Custom initialization
    
    def _assign_reduce_step(self, node):
        # Custom reduce step assignment logic
        return super()._assign_reduce_step(node)
    
    def score_block_size(self, n):
        # Custom block size scoring
        return super().score_block_size(n)
    
    def check_tile_shape_isvalid(self, td: TileDict):
        # Custom tile validation logic
        return super().check_tile_shape_isvalid(td)
```

### Policy Registration

Policies are typically registered through the factory pattern:

```python
def create_policy(policy_type: str, arch: TileDevice, tags: Dict):
    if policy_type == "default":
        return DefaultPolicy(arch, tags)
    elif policy_type == "tensorcore":
        return TensorCorePolicy(arch, tags)
    elif policy_type == "custom":
        return CustomPolicy(arch, tags)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
```

## Performance Optimization Strategies

### Heuristic Rules

The default policy employs several heuristic rules:

1. **Memory Coalescing**: Prefer configurations that maximize coalesced memory access
2. **Occupancy Optimization**: Balance shared memory usage with thread block count
3. **Warp Efficiency**: Minimize warp divergence and maximize instruction throughput
4. **Cache Utilization**: Optimize for L1/L2 cache hit rates

### Search Space Pruning

The policy framework includes mechanisms to prune the search space:

1. **Early Termination**: Stop search when memory limits are exceeded
2. **Dominance Pruning**: Skip configurations dominated by better alternatives
3. **Heuristic Filtering**: Use domain knowledge to filter unlikely candidates

## Integration with Code Generation

### Hint Generation

Policies generate `Hint` objects that contain all necessary information for code generation:

```python
class Hint:
    def __init__(self):
        self.arch = None
        self.use_tc = None  # TensorCore usage flag
        
        # Tiling information
        self.block = []     # Block dimensions
        self.thread = []    # Thread dimensions (for CUDA cores)
        self.warp = []      # Warp dimensions (for TensorCores)
        self.rstep = []     # Reduce step sizes
        self.reduce_thread = []  # Reduce thread configuration
        
        # Advanced configuration
        self.rasterization_plan = NoRasterization()
        self.cached_tensors = []
        self.output_strides = {}
        self.pipeline_stage = 1
        self.use_async = False
        self.vectorize = {}
        self.intrin_info = IntrinInfo("float16", "float16", True)
    
    def to_dict(self) -> Dict:
        """Convert hint to dictionary for code generation."""
        dic = {}
        dic["block"] = self.block
        if self.use_tc:
            dic["warp"] = self.warp
        else:
            dic["thread"] = self.thread
        dic["rstep"] = self.rstep
        if np.prod(self.reduce_thread) > 1:
            dic["reduce_thread"] = self.reduce_thread
        if self.use_tc:
            dic["use_tc"] = self.use_tc
        dic["pipeline_stage"] = self.pipeline_stage
        dic["use_async"] = self.use_async
        return dic
```

## Future Enhancements

### Planned Features

1. **Machine Learning Integration**: Use ML models to predict optimal configurations
2. **Dynamic Adaptation**: Runtime adaptation based on profiling feedback
3. **Multi-Objective Optimization**: Balance performance, energy, and accuracy
4. **Cross-Platform Support**: Unified policies for different hardware backends

### Research Directions

1. **Automated Policy Generation**: Generate policies from hardware specifications
2. **Transfer Learning**: Apply learned policies across similar architectures
3. **Compositional Optimization**: Optimize sequences of operations jointly
4. **Uncertainty Quantification**: Handle variability in hardware performance