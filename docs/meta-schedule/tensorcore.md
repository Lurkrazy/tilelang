# TensorCore Implementation Documentation

## Overview

The TensorCore implementation in TileLang provides specialized optimization strategies for NVIDIA TensorCore operations, enabling efficient mixed-precision matrix computations. The `TensorCorePolicy` extends the base `DefaultPolicy` with TensorCore-specific optimizations and constraints.

## TensorCore Architecture

### Supported Hardware

#### Volta Architecture (SM 7.0-7.5)
```python
volta_tensorcore_supported = [
    ("float16", "float32"),
    ("float16", "float16"),
]
```
- **WMMA Instructions**: Warp Matrix Multiply-Accumulate
- **Matrix Shapes**: 16x16x16 operations
- **Precision Support**: FP16 input, FP32/FP16 accumulation

#### Ampere Architecture (SM 8.0-8.6)
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
- **Enhanced Precision**: Support for bfloat16, int8, and sub-byte integers
- **Async Copy**: Hardware-accelerated memory transfers
- **Pipeline Stages**: Multi-stage computation pipelines

#### Ada Lovelace Architecture (SM 8.9)
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
- **FP8 Support**: New 8-bit floating-point formats
- **Improved Efficiency**: Enhanced sparse operations

#### Hopper Architecture (SM 9.0)
- **Fourth-Generation TensorCores**: Maximum performance and efficiency
- **Enhanced Sparsity**: 2:4 structured sparsity acceleration
- **Thread Block Clusters**: New thread hierarchy for better scaling

## TensorCorePolicy Implementation

### Core Configuration

```python
class TensorCorePolicy(DefaultPolicy):
    wmma_k: int = 16              # WMMA K-dimension (32 for int8)
    pipeline_stage: int = 1       # Number of pipeline stages
    use_async_copy: bool = False  # Hardware async copy usage
    block_reduction_depth: Optional[int] = None  # Block reduction factor
```

### Architecture-Specific Initialization

```python
def _legalize_info(self):
    """
    Configures TensorCore parameters based on compute capability:
    - SM 8.0+: 2-stage pipeline with async copy
    - SM 7.x: Single-stage pipeline without async copy
    """
    if self.arch.compute_capability in {"sm_80", "sm_90", "sm_90a"}:
        self.pipeline_stage = 2
        self.use_async_copy = True
    else:
        self.pipeline_stage = 1
        self.use_async_copy = False
```

## Memory Layout Optimization

### TensorCore Stride Computation

```python
def _compute_tc_strides(self, node: PrimFuncNode, tile: List[int], rstep: Optional[Dict[str, int]] = None):
    """
    Computes optimal memory strides for TensorCore operations to avoid
    shared memory bank conflicts and optimize data layout.
    """
    # Extract matrix dimensions
    A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = node.infer_tensorcore_axis()
    
    # Apply padding offset to avoid bank conflicts
    offset = 8
    A_stride = Stride(stride=np.prod(AS_shape[A_high_ax + 1:]) + offset, ax=A_high_ax)
    B_stride = Stride(stride=np.prod(BS_shape[B_high_ax + 1:]) + offset, ax=B_high_ax)
    C_stride = Stride(stride=np.prod(CS_shape[C_high_ax + 1:]) + offset, ax=C_high_ax)
```

#### Bank Conflict Avoidance
- **Padding Strategy**: Adds offset to memory strides
- **Layout Optimization**: Ensures efficient shared memory access patterns
- **Matrix Dimension Analysis**: Considers A, B, and C matrix layouts

### Shared Memory Usage Optimization

```python
def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode):
    """
    TensorCore operations require additional shared memory for pipelining.
    Usage = base_usage * pipeline_stages
    """
    value, cached_tensors = super().infer_node_smem_usage(td, node)
    value *= self.pipeline_stage
    return value, cached_tensors
```

## Reduction Axis Optimization

### Small Tile Detection and Expansion

```python
def _expand_reduce_axis(self, td: TileDict):
    """
    TensorCore-specific reduction axis optimization for improved compute efficiency.
    """
    def _check_small_tile(td: TileDict):
        minimal_threshold = 32
        for node in self.ordered_nodes:
            tile = td.get_tile(node)
            if any([t <= minimal_threshold for t in tile]):
                return True
        return False
```

#### Optimization Strategy
1. **Small Tile Detection**: Identifies suboptimal tile sizes (≤32 elements)
2. **Memory-Constrained Expansion**: Increases reduction dimensions within shared memory limits
3. **Efficiency Scoring**: Evaluates memory access patterns for optimization decisions

### Reduction Step Candidates

```python
def get_node_reduce_step_candidates(self, node):
    """
    TensorCore operations must use reduction steps that are multiples of wmma_k.
    For most operations: wmma_k = 16
    For int8 operations: wmma_k = 32
    """
    if node.get_tag("tensorcore_config"):
        return {
            k.var.name: [
                x * self.wmma_k for x in get_all_factors(int(k.dom.extent) // self.wmma_k)
            ] for k in node.raxis
        }
```

## Tile Shape Validation

### TensorCore Constraints

```python
def check_tile_shape_isvalid(self, td: TileDict):
    """
    Validates tile shapes against TensorCore hardware constraints:
    - Minimum tile dimensions for TensorCore operations
    - Alignment with WMMA instruction requirements
    - Matrix dimension compatibility
    """
    for node in self.ordered_nodes:
        if node.get_tag("tensorcore_config"):
            ax_m, ax_n = node.get_tag("tensorcore_config")
            block_m, block_n = td.tile_map[node][ax_m], td.tile_map[node][ax_n]
            
            # Validate against available TensorCore shapes
            wmma_invalid = [
                block_m < wmma_m or block_n < wmma_n
                for wmma_m, wmma_n in self.arch.get_available_tensorintrin_shapes()
            ]
            if all(wmma_invalid):
                return False
```

#### Validation Criteria
- **Minimum Dimensions**: Tiles must meet minimum TensorCore operation sizes
- **Shape Alignment**: Matrix dimensions must align with WMMA requirements
- **Hardware Compatibility**: Verification against device-specific TensorCore capabilities

## Block Size Assignment

### Warp-Level Optimization

```python
def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
    """
    Assigns optimal block and warp tile sizes for TensorCore operations.
    """
    if block_size % self.arch.warp_size != 0:
        return None
        
    warps = block_size // self.arch.warp_size
    
    # Get largest available WMMA shape
    wmma = self.arch.get_available_tensorintrin_shapes()[-1]
    wmma_tile = [1 for _ in range(ndim)]
    wmma_tile[ax_m] = wmma[0]
    wmma_tile[ax_n] = wmma[1]
```

#### Optimization Process
1. **Warp Count Calculation**: Determines number of warps per thread block
2. **WMMA Tile Configuration**: Sets base tile sizes based on largest available WMMA shape
3. **Factor Distribution**: Distributes remaining computation across spatial dimensions
4. **Performance Scoring**: Optimizes based on memory bandwidth utilization

### Dynamic Shared Memory Management

```python
# Adaptive shared memory scope selection
if td.smem_cost > self.arch.smem_cap:
    codegen_dict.shared_scope = "shared.dyn"
else:
    codegen_dict.shared_scope = "shared"
```

## Advanced TensorCore Features

### Asynchronous Copy Operations

For Ampere+ architectures:
```python
if self.use_async_copy:
    # Enable hardware-accelerated memory transfers
    codegen_dict.use_async = True
    # Configure pipeline stages for overlapped computation
    codegen_dict.pipeline_stage = self.pipeline_stage
```

#### Benefits
- **Latency Hiding**: Overlaps memory transfers with computation
- **Bandwidth Utilization**: Maximizes memory throughput
- **Pipeline Efficiency**: Enables multi-stage computation pipelines

### Rasterization Optimization

```python
def plan_rasterization(self, td: TileDict):
    """
    Determines optimal thread block execution order for TensorCore kernels.
    """
    conditions = []
    conditions.append(len(self.ordered_nodes) > 1)  # Single node optimization
    conditions.append(self.arch.compute_capability < "80")  # Ampere+ only
    
    def _check_memory_size():
        overall_gmem_size = sum(buffer memory sizes)
        return overall_gmem_size < self.arch.l2_cache_size_bytes
    
    if any(conditions):
        return NoRasterization()
    
    # 2D column rasterization for cache optimization
    raster_factor = int(self.arch.compute_max_core**0.5)
    return Rasterization2DColumn(raster_factor)
```

#### Rasterization Strategy
- **Cache Optimization**: Improves L2 cache utilization
- **Memory Locality**: Enhances spatial locality of memory accesses
- **Architecture-Specific**: Optimized for Ampere+ architectures

## Configuration Generation

### Complete Configuration Assembly

```python
def _assign_block_size(self, node: PrimFuncNode, td: TileDict, block_size: int):
    """
    Generates complete TensorCore configuration including:
    - Block and warp tile sizes
    - Pipeline configuration
    - Memory management settings
    - TensorCore-specific parameters
    """
    codegen_dict = Hint()
    codegen_dict.block = tile
    codegen_dict.warp = warp_tile
    codegen_dict.use_tc = True  # Enable TensorCore usage
    codegen_dict.pipeline_stage = self.pipeline_stage
    codegen_dict.use_async = self.use_async_copy
    codegen_dict.rstep = [int(rsteps[ax.var.name]) for ax in node.raxis]
    
    # TensorCore intrinsic information
    intrin_info = node.get_tag("intrin_info")
    if intrin_info:
        codegen_dict.intrin_info = IntrinInfo(**intrin_info)
        if intrin_info["out_dtype"] in ["float32"]:
            codegen_dict.shared_scope = "shared.dyn"
    
    # Complete configuration with TensorCore legalization
    codegen_dict.complete_config(node)
    codegen_dict.tensorcore_legalization()
```

## Performance Optimization Strategies

### Memory Access Optimization

1. **Coalesced Access**: Ensures memory accesses are aligned and coalesced
2. **Bank Conflict Avoidance**: Uses padding to prevent shared memory bank conflicts
3. **Pipeline Optimization**: Overlaps memory transfers with computation

### Compute Efficiency

1. **WMMA Utilization**: Maximizes TensorCore instruction usage
2. **Warp Efficiency**: Optimizes thread block organization
3. **Register Management**: Balances register usage with occupancy

### Architecture-Specific Tuning

1. **Volta Optimization**: Single-stage pipeline, WMMA-based computation
2. **Ampere Enhancement**: Multi-stage pipeline, async copy operations
3. **Ada/Hopper Features**: FP8 support, enhanced sparsity handling

## Usage Examples

### Basic TensorCore GEMM

```python
# Configure TensorCore policy for Ampere architecture
arch = CUDA("sm_80")
policy = TensorCorePolicy(arch)

# Generate TensorCore-optimized configurations
configs = policy.emit_config(topk=5)

for config in configs:
    print(f"TensorCore Config:")
    print(f"  Block: {config.block}")
    print(f"  Warp: {config.warp}")
    print(f"  Pipeline Stages: {config.pipeline_stage}")
    print(f"  Async Copy: {config.use_async}")
    print(f"  Shared Memory: {config.shared_scope}")
```

### Precision-Specific Optimization

```python
# Check TensorCore support for specific precision
if is_tensorcore_supported_precision("float16", "float32", arch):
    print("FP16 TensorCore operations supported")
    
    # Configure for mixed-precision computation
    policy.wmma_k = 16  # Standard WMMA K dimension
    
elif is_tensorcore_supported_precision("int8", "int32", arch):
    print("INT8 TensorCore operations supported")
    
    # Configure for integer TensorCore operations
    policy.wmma_k = 32  # Larger K dimension for int8
```

## Best Practices

### Optimization Guidelines

1. **Architecture Awareness**: Configure pipeline stages and async copy based on target architecture
2. **Memory Layout**: Use proper stride calculations to avoid bank conflicts
3. **Tile Size Selection**: Ensure tile dimensions meet TensorCore minimum requirements
4. **Reduction Optimization**: Expand reduction axes for small tiles to improve efficiency

### Common Issues and Solutions

#### Bank Conflicts
- **Problem**: Poor shared memory access patterns
- **Solution**: Use stride padding and proper matrix layout

#### Insufficient Occupancy
- **Problem**: High register usage or excessive shared memory
- **Solution**: Adjust tile sizes and pipeline configuration

#### Suboptimal TensorCore Utilization
- **Problem**: Tile sizes don't align with WMMA requirements
- **Solution**: Validate tile shapes and use proper factorization

## Integration with Code Generation

The TensorCore policy integrates seamlessly with TileLang's code generation system:

```python
# The generated configuration includes all necessary parameters
# for TensorCore code generation:
config.use_tc = True                    # Enable TensorCore intrinsics
config.intrin_info = IntrinInfo(...)    # Intrinsic function information
config.tensorcore_legalization()       # Apply TensorCore-specific transformations
```

This comprehensive TensorCore implementation provides efficient mixed-precision matrix operations with hardware-specific optimizations for different NVIDIA GPU architectures.