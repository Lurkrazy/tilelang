# TensorCore Implementation

This document provides a comprehensive analysis of TileLang's TensorCore support, implemented primarily in `tilelang/carver/roller/policy/tensorcore.py` and related components.

## Overview

TileLang's TensorCore implementation provides high-performance mixed-precision computing support for NVIDIA GPUs with TensorCore capabilities (Volta, Ampere, Ada Lovelace, and Hopper architectures).

## Key Features

- **Multi-Precision Support**: FP16, BF16, INT8, INT4, INT2, INT1, and FP8 formats
- **Automatic Layout Optimization**: Bank conflict avoidance and memory coalescing
- **Pipeline Scheduling**: Multi-stage pipelines with asynchronous memory operations
- **Warp-Level Optimization**: Efficient thread block and warp configuration
- **Architecture-Aware Scheduling**: Optimizations specific to different GPU generations

## Architecture Support

### Supported Precision Formats by Architecture

#### Volta (SM 7.0-7.5)
```python
volta_tensorcore_supported = [
    ("float16", "float32"),
    ("float16", "float16"),
]
```

#### Ampere (SM 8.0-8.9)
```python
ampere_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("int4", "int32"),
    ("int2", "int32"),
    ("int1", "int32"),
]
```

#### Ada Lovelace (SM 8.9)
```python
ada_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("float8_e5m2", "float32"),
    ("float8_e4m3", "float32"),
]
```

#### Hopper (SM 9.0)
```python
hopper_tensorcore_supported = ada_tensorcore_supported  # Same as Ada Lovelace
```

### Architecture Detection

```python
def is_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    if is_volta_arch(arch):
        return (in_dtype, accum_dtype) in volta_tensorcore_supported
    elif is_ampere_arch(arch):
        return (in_dtype, accum_dtype) in ampere_tensorcore_supported
    elif is_ada_arch(arch):
        return (in_dtype, accum_dtype) in ada_tensorcore_supported
    elif is_hopper_arch(arch):
        return (in_dtype, accum_dtype) in hopper_tensorcore_supported
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
```

## TensorCore Policy Implementation

### Core Policy Class

The `TensorCorePolicy` extends `DefaultPolicy` with TensorCore-specific optimizations:

```python
class TensorCorePolicy(DefaultPolicy):
    wmma_k: int = 16  # Default WMMA K dimension (32 for INT8)
    pipeline_stage: int = 1  # Number of pipeline stages
    use_async_copy: bool = False  # Asynchronous memory copy
    block_reduction_depth: Optional[int] = None  # Block reduction optimization
```

### Configuration Legalization

The policy automatically configures TensorCore-specific parameters based on architecture:

```python
def _legalize_info(self):
    # Set pipeline stages based on architecture
    if self.arch.compute_capability in {"sm_80", "sm_90", "sm_90a"}:
        self.pipeline_stage = 2
    else:
        self.pipeline_stage = 1
    
    # Enable async copy for modern architectures
    if self.arch.compute_capability in {"sm_80", "sm_90", "sm_90a"}:
        self.use_async_copy = True
    else:
        self.use_async_copy = False
```

## Memory Layout Optimization

### Stride Computation for Bank Conflict Avoidance

TensorCore operations require careful memory layout to avoid shared memory bank conflicts:

```python
def _compute_tc_strides(self, node: PrimFuncNode, tile: List[int], rstep: Optional[Dict[str, int]] = None) -> Tuple[Stride, Stride, Stride]:
    # Get input shapes after reduction propagation
    shapes = node.propagate_reduction_inputs(tile, rstep)
    AS_shape, BS_shape = shapes.values()
    CS_shape = tile
    
    # Infer TensorCore axis mapping
    A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
    
    # Apply memory padding to avoid bank conflicts
    offset = 8  # Padding offset
    A_high_ax = min(A_ax_m, A_ax_k)
    B_high_ax = min(B_ax_n, B_ax_k)
    C_high_ax = min(C_ax_m, C_ax_n)
    
    # Compute strides with padding
    A_stride = Stride(stride=np.prod(AS_shape[A_high_ax + 1:]) + offset, ax=A_high_ax)
    B_stride = Stride(stride=np.prod(BS_shape[B_high_ax + 1:]) + offset, ax=B_high_ax)
    C_stride = Stride(stride=np.prod(CS_shape[C_high_ax + 1:]) + offset, ax=C_high_ax)
    
    return A_stride, B_stride, C_stride
```

### Memory Usage Calculation

TensorCore policy accounts for pipeline stages in memory calculations:

```python
def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
    value, cached_tensors = super().infer_node_smem_usage(td, node)
    value *= self.pipeline_stage  # Multiply by pipeline depth
    return value, cached_tensors
```

## Reduction Step Assignment

### TensorCore-Specific Reduce Step Optimization

```python
def _assign_reduce_step(self, node):
    if not node.get_tag("tensorcore_config"):
        return super()._assign_reduce_step(node)
    
    # Calculate optimal reduce step for TensorCore
    target_transaction = self.arch.transaction_size[0] * 2  # 512 bytes
    reduce_input_dtype = node.get_buffer_dtype(
        node.block_analyzer.get_input_buffers(node.reduction_block)[0]
    )
    basic = (target_transaction * 8) // reduce_input_dtype.bits
    
    result = {}
    for iter_info in node.raxis:
        iter_name = iter_info.var.name
        iter_dom = iter_info.dom.extent
        
        if iter_dom % 16 > 0:
            result[iter_name] = (16 if iter_dom < basic else basic)
        elif iter_dom % basic == 0:
            result[iter_name] = basic
        elif iter_dom % 32 == 0:
            result[iter_name] = 32
        else:
            candidates = get_all_factors(iter_dom)
            candidates = list(filter(lambda x: x >= 16 and x <= basic, candidates))
            result[iter_name] = candidates[-1] if candidates else basic
    
    return result
```

## Reduce Axis Expansion

### Small Tile Optimization

The TensorCore policy includes logic to expand reduce axes for better compute efficiency:

```python
def _expand_reduce_axis(self, td: TileDict):
    def _check_small_tile(td: TileDict):
        # Check if tile size is too small for efficient TensorCore usage
        for node in self.ordered_nodes:
            if not node.get_tag("tensorcore_config"):
                continue
            tile = td.get_tile(node)
            if np.prod(tile) < 256:  # Threshold for small tiles
                return True
        return False
    
    if _check_small_tile(td):
        smem_limit = min(
            self.arch.max_smem_usage // td.block_per_SM, 
            self.arch.smem_cap
        )
        
        rstep_map = td.rstep_map.copy()
        
        def _optimize(node, rstep):
            # Try to expand reduce steps while staying within memory limits
            all_steps = {}
            for axis in node.raxis:
                candidates = get_all_factors(axis.dom.extent)
                candidates = [x for x in candidates if x >= rstep[axis.var.name]]
                all_steps[axis.var.name] = candidates
            
            # Binary search for optimal reduce step
            cur_rstep_id = {k: 0 for k in all_steps}
            for axis in node.raxis:
                for step_id in range(len(all_steps[axis.var.name])):
                    new_rstep_id = cur_rstep_id.copy()
                    new_rstep_id[axis.var.name] = step_id
                    
                    new_rstep_map = rstep_map.copy()
                    new_rstep_map[node] = {
                        k: all_steps[k][new_rstep_id[k]] 
                        for k in new_rstep_id
                    }
                    
                    # Check if new configuration exceeds memory limit
                    old_rstep_map = td.rstep_map
                    td.rstep_map = new_rstep_map
                    smem_usage, _ = self._compute_shared_memory_usage(td)
                    td.rstep_map = old_rstep_map
                    
                    if smem_usage > smem_limit:
                        break
                    else:
                        cur_rstep_id = new_rstep_id
            
            return {k: all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        
        # Apply optimization to all nodes
        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map[node])
                rstep_map[node] = rstep
        
        td.rstep_map = rstep_map
        td.smem_cost, td.cached_tensors_map = self._compute_shared_memory_usage(td)
```

## Block Size Assignment

### TensorCore-Aware Thread Block Configuration

The TensorCore policy includes specialized logic for assigning thread block sizes:

```python
def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
    tile, rsteps = td.get_tile(node), td.get_rstep(node)
    factors = factorize(block_size)
    cur_threads = [1 for _ in tile]
    reduce_thread = {k: 1 for k in rsteps}
    ndim = len(tile)
    
    def _score(node, warp_tile):  # Lower score is better
        score = 0
        shape = node.propagate_inputs_on_reduction(warp_tile)
        input_buffers = node.block_analyzer.get_input_buffers(node.reduction_block)
        
        # Score based on memory bandwidth utilization
        for i, _ in enumerate(input_buffers):
            score += np.prod(shape[i]) / self.arch.bandwidth[1]
        
        return score
    
    # Distribute factors across spatial dimensions
    for factor in reversed(factors):
        score_map = {}
        
        # Try assigning factor to each spatial dimension
        for i in range(ndim):
            if tile[i] % (cur_threads[i] * factor) == 0:
                test_threads = cur_threads.copy()
                test_threads[i] *= factor
                
                warp_tile = [
                    int(np.ceil(tile[j] / test_threads[j])) 
                    for j in range(ndim)
                ]
                score_map[i] = _score(node, warp_tile)
        
        # Try assigning factor to reduce dimensions
        for ax in rsteps:
            if rsteps[ax] % (reduce_thread[ax] * factor) == 0:
                test_reduce_thread = reduce_thread.copy()
                test_reduce_thread[ax] *= factor
                # Score reduce thread assignment
                score_map[f"reduce_{ax}"] = 1.0 / factor  # Prefer larger reduce threading
        
        # Select best assignment
        if score_map:
            best_assignment = min(score_map.items(), key=lambda x: x[1])[0]
            if isinstance(best_assignment, int):
                cur_threads[best_assignment] *= factor
            else:
                ax = best_assignment.replace("reduce_", "")
                reduce_thread[ax] *= factor
    
    # Create hint configuration
    hint = Hint()
    hint.block = tile
    hint.warp = cur_threads  # For TensorCore, use warp instead of thread
    hint.rstep = list(rsteps.values())
    hint.reduce_thread = list(reduce_thread.values())
    hint.use_tc = True
    hint.arch = self.arch
    
    return hint
```

## TensorCore Layout Support

### Instruction Configuration

TensorCore instructions are configured through `IntrinInfo`:

```python
class IntrinInfo:
    def __init__(
        self,
        in_dtype: str,
        out_dtype: str,
        trans_b: bool,
        input_transform_kind: int = 0,
        weight_transform_kind: int = 0,
    ):
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.trans_a = False
        self.trans_b = trans_b
        self.input_transform_kind = input_transform_kind
        self.weight_transform_kind = weight_transform_kind
    
    def is_input_8bit(self) -> bool:
        return DataType(self.in_dtype).bits == 8
    
    @property
    def smooth_a(self) -> bool:
        return self.input_transform_kind >= 2
    
    @property
    def smooth_b(self) -> bool:
        return self.weight_transform_kind >= 2
```

### Extra Configuration for TensorCore

```python
class TensorCoreExtraConfig:
    def __init__(
        self,
        AS_shape: Tuple[int],  # A matrix shared memory shape
        BS_shape: Tuple[int],  # B matrix shared memory shape
        AF_shape: Tuple[int],  # A matrix fragment shape
        BF_shape: Tuple[int],  # B matrix fragment shape
        tc_axis: Tuple[int],   # TensorCore axis mapping
    ):
        self.AS_shape = AS_shape
        self.BS_shape = BS_shape
        self.AF_shape = AF_shape
        self.BF_shape = BF_shape
        self.tc_axis = tc_axis
```

## Pipeline Optimization

### Multi-Stage Pipeline Support

The TensorCore policy supports multi-stage pipelines for hiding memory latency:

1. **Stage 1**: Memory copy operations (global to shared memory)
2. **Stage 2**: TensorCore compute operations
3. **Stage 3**: Output operations (shared to global memory)

Pipeline stages are configured based on architecture capabilities:
- **Volta/Turing**: Single stage pipeline
- **Ampere/Ada/Hopper**: Multi-stage pipeline with async copy

## Performance Optimizations

### Memory Coalescing

TensorCore operations ensure coalesced memory access through:
- Proper thread-to-memory mapping
- Vectorized memory operations
- Bank conflict avoidance in shared memory

### Occupancy Optimization

The policy optimizes GPU occupancy by:
- Balancing shared memory usage across thread blocks
- Optimizing register usage per thread
- Maximizing SM utilization

### Warp Efficiency

TensorCore scheduling ensures efficient warp utilization through:
- Proper warp-level tile sizes
- Minimizing warp divergence
- Optimizing instruction scheduling

## Integration with Code Generation

The TensorCore policy generates configuration hints that are consumed by the code generation backend:

```python
def to_dict(self) -> Dict:
    dic = {}
    dic["block"] = self.block
    if self.use_tc:
        dic["warp"] = self.warp  # TensorCore uses warp-level tiling
    else:
        dic["thread"] = self.thread
    dic["rstep"] = self.rstep
    if np.prod(self.reduce_thread) > 1:
        dic["reduce_thread"] = self.reduce_thread
    dic["use_tc"] = self.use_tc
    dic["pipeline_stage"] = self.pipeline_stage
    dic["use_async"] = self.use_async
    return dic
```

## Usage Examples

### Basic TensorCore GEMM Configuration

```python
# Configure TensorCore policy for GEMM
policy = TensorCorePolicy(arch=cuda_arch)
policy = policy._init_with_prim_func(gemm_func)

# Generate optimized configurations
hints = policy.emit_config(topk=10)

# Example hint output:
{
    'block': [128, 128],
    'warp': [16, 16], 
    'rstep': [32],
    'use_tc': True,
    'pipeline_stage': 2,
    'use_async': True
}
```

### TensorCore with Mixed Precision

```python
# Configure for FP16 input, FP32 accumulation
intrin_info = IntrinInfo("float16", "float32", True)
hint.intrin_info = intrin_info

# The policy will automatically select appropriate TensorCore instructions
```

## Limitations and Future Work

### Current Limitations
1. **Architecture Support**: Limited to NVIDIA TensorCore architectures
2. **Precision Coverage**: Not all mixed-precision combinations are supported
3. **Memory Model**: Simplified shared memory bank conflict model

### Future Enhancements
1. **AMD Matrix Core Support**: Extend to AMD CDNA architectures
2. **Intel XMX Support**: Add Intel GPU TensorCore equivalent support
3. **Dynamic Precision**: Runtime precision selection based on accuracy requirements
4. **Advanced Pipeline**: More sophisticated pipeline scheduling algorithms