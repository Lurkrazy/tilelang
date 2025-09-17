# API Reference Documentation

## Core Classes and Interfaces

### DefaultPolicy

The base policy class that implements the fundamental cost model and optimization strategies.

#### Constructor

```python
class DefaultPolicy:
    def __init__(self, arch: TileDevice, tags: Optional[Dict] = None) -> None
```

**Parameters:**
- `arch`: Target hardware architecture (TileDevice instance)
- `tags`: Optional configuration tags for customization

#### Factory Methods

```python
@classmethod
def from_prim_func(cls, func: tvm.tir.PrimFunc, arch: TileDevice, 
                   tags: Optional[Dict] = None, name: str = "PrimFuncNode"):
    """Create policy from TVM PrimFunc"""

@classmethod  
def from_output_nodes(cls, nodes: List[OutputNode], arch: TileDevice,
                      tags: Optional[Dict] = None):
    """Create policy from output node list"""
```

#### Core Methods

##### Configuration Generation

```python
def emit_config(self, topk: int) -> List[Hint]:
    """
    Generate optimized kernel configurations.
    
    Args:
        topk: Number of top configurations to return
        
    Returns:
        List of Hint objects containing optimized configurations
    """
```

##### Cost Evaluation

```python
def compute_tile_dict(self, output_tile: List[int], rstep_map) -> TileDict:
    """
    Compute comprehensive cost analysis for tile configuration.
    
    Args:
        output_tile: Output tensor tiling configuration
        rstep_map: Reduction step mapping for each node
        
    Returns:
        TileDict containing cost metrics and configuration details
    """
```

##### Memory Analysis

```python
def _compute_memory_traffic(self, output_tile) -> Tuple[int, Dict]:
    """
    Analyze global memory access patterns.
    
    Args:
        output_tile: Output tensor tiling configuration
        
    Returns:
        Tuple of (total_traffic, operation_tile_map)
    """

def _compute_shared_memory_usage(self, td: TileDict) -> Tuple[int, Dict]:
    """
    Compute shared memory usage using best-fit allocation.
    
    Args:
        td: TileDict configuration
        
    Returns:
        Tuple of (peak_usage, cached_tensors_map)
    """
```

##### Block Size Optimization

```python
def assign_block_size(self, td: TileDict, topk=1) -> List[Hint]:
    """
    Determine optimal thread block configurations.
    
    Args:
        td: TileDict configuration
        topk: Number of configurations to return
        
    Returns:
        List of optimized Hint configurations
    """
```

#### Properties

```python
# Core attributes
func: tvm.tir.PrimFunc          # Source function
nodes: List[PrimFuncNode]       # Computation nodes
arch: TileDevice                # Target architecture
tags: Dict                      # Configuration tags
ordered_nodes: List[PrimFuncNode]  # Topologically sorted nodes
output_nodes: List[PrimFuncNode]   # Output nodes
```

### TensorCorePolicy

Specialized policy for TensorCore operations, extending DefaultPolicy.

#### Constructor

```python
class TensorCorePolicy(DefaultPolicy):
    # TensorCore-specific attributes
    wmma_k: int = 16                        # WMMA K dimension
    pipeline_stage: int = 1                 # Pipeline stages
    use_async_copy: bool = False            # Async copy usage
    block_reduction_depth: Optional[int] = None  # Block reduction factor
```

#### TensorCore-Specific Methods

```python
def get_node_reduce_step_candidates(self, node) -> Dict[str, List[int]]:
    """
    Get TensorCore-compatible reduction step candidates.
    
    Args:
        node: PrimFuncNode to analyze
        
    Returns:
        Dictionary mapping axis names to valid step sizes
    """

def check_tile_shape_isvalid(self, td: TileDict) -> bool:
    """
    Validate tile shapes against TensorCore constraints.
    
    Args:
        td: TileDict configuration
        
    Returns:
        Boolean indicating validity
    """

def plan_rasterization(self, td: TileDict) -> Rasterization:
    """
    Determine optimal thread block execution order.
    
    Args:
        td: TileDict configuration
        
    Returns:
        Rasterization strategy object
    """
```

#### Memory Layout Optimization

```python
def _compute_tc_strides(self, node: PrimFuncNode, tile: List[int], 
                        rstep: Optional[Dict[str, int]] = None) -> Tuple[Stride, Stride, Stride]:
    """
    Compute TensorCore-optimized memory strides.
    
    Args:
        node: PrimFuncNode for analysis
        tile: Tile configuration
        rstep: Reduction step configuration
        
    Returns:
        Tuple of (A_stride, B_stride, C_stride)
    """

def infer_node_smem_usage(self, td: TileDict, node: PrimFuncNode) -> Tuple[int, Dict]:
    """
    Infer shared memory usage with TensorCore pipeline considerations.
    
    Args:
        td: TileDict configuration
        node: PrimFuncNode to analyze
        
    Returns:
        Tuple of (memory_usage, cached_tensors)
    """
```

### TileDict

Configuration container for tiling strategies and cost metrics.

#### Constructor

```python
class TileDict:
    def __init__(self, output_tile) -> None
```

#### Attributes

```python
# Configuration
output_tile: List[int]              # Output tensor tile size
tile_map: Dict                      # Per-node tile configurations
rstep_map: Dict                     # Reduction step configurations
cached_tensors_map: Dict            # Shared memory tensor mapping
output_strides_map: Dict            # Output stride configurations
tensor_strides_map: Dict            # Tensor stride configurations

# Cost Metrics
traffic: int                        # Global memory traffic
smem_cost: int                      # Shared memory usage
block_per_SM: int                   # Thread blocks per SM
num_wave: int                       # Number of kernel waves
grid_size: int                      # Grid size for kernel launch
valid: bool                         # Configuration validity
```

#### Methods

```python
def get_tile(self, func) -> List[int]:
    """Get tile configuration for specific function"""

def get_rstep(self, node) -> Dict[str, int]:
    """Get reduction step configuration for node"""
```

### Hint

Configuration hint containing complete kernel optimization parameters.

#### Attributes

```python
class Hint:
    # Basic Configuration
    block: List[int]                    # Thread block tile size
    warp: List[int]                     # Warp tile size
    rstep: List[int]                    # Reduction steps
    
    # TensorCore Configuration
    use_tc: bool = False                # Enable TensorCore
    intrin_info: Optional[IntrinInfo] = None  # Intrinsic information
    
    # Memory Configuration
    cached_tensors: Dict = {}           # Shared memory allocations
    shared_scope: str = "shared"        # Memory scope ("shared" or "shared.dyn")
    
    # Pipeline Configuration
    pipeline_stage: int = 1             # Number of pipeline stages
    use_async: bool = False             # Async copy usage
    block_reduction_depth: Optional[int] = None  # Block reduction factor
    
    # Optimization Features
    vectorize: Dict = {}                # Vectorization configuration
    rasterization_plan: Rasterization = None  # Thread block ordering
    arch: TileDevice = None             # Target architecture
    opt_shapes: Dict = {}               # Shape optimization information
```

#### Methods

```python
def complete_config(self, node: PrimFuncNode):
    """Complete configuration with node-specific parameters"""

def tensorcore_legalization(self):
    """Apply TensorCore-specific transformations"""
```

### TileDevice

Base hardware architecture abstraction.

#### Attributes

```python
class TileDevice:
    # Compute Resources
    reg_cap: int = 0                    # Register capacity per SM
    compute_max_core: int = 0           # Number of SMs
    warp_size: int = 0                  # Threads per warp
    sm_partition: int = 0               # SM partition count
    
    # Memory Hierarchy
    smem_cap: int = 0                   # Shared memory per block
    max_smem_usage: int = 0             # Maximum shared memory usage
    l2_cache_size_bytes: int = 0        # L2 cache size
    
    # Memory Characteristics
    transaction_size: List[int] = [0, 0]  # [write, read] transaction sizes
    bandwidth: List[int] = [0, 0]       # Memory bandwidth [peak, sustained]
    
    # Architecture Information
    platform: str = "unknown"          # Platform name
    compute_capability: str = "unknown" # Compute capability version
```

#### Abstract Methods

```python
def get_available_tensorintrin_shapes(self) -> List[List[int]]:
    """
    Get available TensorCore instruction shapes.
    
    Returns:
        List of [M, N, K] shapes for TensorCore operations
    """
```

### CUDA

CUDA-specific hardware implementation.

#### Constructor

```python
class CUDA(TileDevice):
    def __init__(self, target: Union[Target, str])
```

#### Additional Attributes

```python
target: tvm.target.Target           # TVM target specification
sm_version: int                     # SM version number
device: tvm.runtime.Device          # Runtime device handle
name: str                           # Device name
```

#### Methods

```python
def get_available_tensorintrin_shapes(self) -> List[List[int]]:
    """
    Returns CUDA TensorCore shapes based on architecture:
    - Volta: [[16, 16, 16]]
    - Ampere+: [[16, 16, 16], [32, 8, 16], [8, 32, 16]]
    """
```

## Architecture Detection Functions

### CUDA Architecture Detection

```python
def check_sm_version(arch: str) -> int:
    """Extract SM version from architecture string"""

def is_cuda_arch(arch: TileDevice) -> bool:
    """Check if architecture is CUDA-based"""

def is_volta_arch(arch: TileDevice) -> bool:
    """Check for Volta architecture (SM 7.0-7.5)"""

def is_ampere_arch(arch: TileDevice) -> bool:
    """Check for Ampere architecture (SM 8.0-8.6)"""

def is_ada_arch(arch: TileDevice) -> bool:
    """Check for Ada Lovelace architecture (SM 8.9)"""

def is_hopper_arch(arch: TileDevice) -> bool:
    """Check for Hopper architecture (SM 9.0)"""

def has_mma_support(arch: TileDevice) -> bool:
    """Check for MMA instruction support (Ampere+)"""
```

### TensorCore Capability Functions

```python
def is_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    """
    Check TensorCore support for specific precision combination.
    
    Args:
        in_dtype: Input data type ("float16", "int8", etc.)
        accum_dtype: Accumulation data type ("float32", "int32", etc.)
        arch: Target architecture
        
    Returns:
        Boolean indicating support for the precision combination
    """
```

## Memory Management Classes

### BestFit

Memory allocation optimizer using best-fit strategy.

#### Constructor

```python
class BestFit:
    def __init__(self, align=32)
```

#### Methods

```python
def malloc(self, size) -> Block:
    """
    Allocate memory block using best-fit algorithm.
    
    Args:
        size: Requested memory size in bytes
        
    Returns:
        Allocated Block object
    """

def free(self, block: Block) -> None:
    """
    Free memory block and merge adjacent free blocks.
    
    Args:
        block: Block object to free
    """
```

### Block

Memory block representation for allocation tracking.

#### Constructor

```python
class Block:
    def __init__(self, start, end, is_free)
```

#### Attributes

```python
start: int      # Block start address
end: int        # Block end address
is_free: bool   # Allocation status
```

#### Methods

```python
def size(self) -> int:
    """Return block size in bytes"""

def merge(self, other):
    """Merge with adjacent block"""
```

### Stride

Memory stride configuration for optimized access patterns.

#### Constructor

```python
class Stride:
    def __init__(self, stride: int = 1, ax: int = -1)
```

#### Attributes

```python
ax: int         # Stride axis
stride: int     # Stride value
```

#### Methods

```python
def compute_strides_from_shape(self, shape: List[int]) -> List[int]:
    """
    Compute memory strides for tensor shape.
    
    Args:
        shape: Tensor shape
        
    Returns:
        List of stride values for each dimension
    """

def compute_elements_from_shape(self, shape: List[int]) -> int:
    """
    Compute total elements including stride padding.
    
    Args:
        shape: Tensor shape
        
    Returns:
        Total number of elements
    """

def is_valid(self) -> bool:
    """Check if stride configuration is valid"""
```

## Rasterization Strategies

### NoRasterization

Default rasterization strategy with standard block ordering.

```python
class NoRasterization:
    """Standard thread block execution order"""
```

### Rasterization2DColumn

2D column-wise rasterization for improved cache locality.

```python
class Rasterization2DColumn:
    def __init__(self, raster_factor: int)
    
    # Attributes
    raster_factor: int  # Rasterization factor for 2D blocking
```

## Utility Functions

### Factorization Utilities

```python
def factorize(n: int) -> List[int]:
    """
    Compute all factors of integer n.
    
    Args:
        n: Integer to factorize
        
    Returns:
        List of factors in ascending order
    """

def get_all_factors(n: int) -> List[int]:
    """
    Get all factors including divisors.
    
    Args:
        n: Integer to analyze
        
    Returns:
        Complete list of factors
    """
```

### Memory Access Optimization

```python
def coalesced_factor(tile_shape: List[int], tensor_shape: List[int]) -> float:
    """
    Compute memory coalescing efficiency factor.
    
    Args:
        tile_shape: Tile dimensions
        tensor_shape: Full tensor dimensions
        
    Returns:
        Coalescing efficiency score (higher is better)
    """

def coalesced_tensor_shape(tile_shape: List[int], tensor_shape: List[int], 
                          transaction_elements: int) -> int:
    """
    Compute effective memory accesses considering coalescing.
    
    Args:
        tile_shape: Tile dimensions
        tensor_shape: Full tensor dimensions
        transaction_elements: Elements per memory transaction
        
    Returns:
        Number of effective memory transactions
    """
```

## Usage Examples

### Basic Policy Usage

```python
import tilelang
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.policy import DefaultPolicy, TensorCorePolicy

# Create CUDA architecture
arch = CUDA("cuda -arch=sm_80")

# Create policy
policy = TensorCorePolicy(arch)

# Initialize with PrimFunc
policy = policy.from_prim_func(func, arch, tags={"pipeline_stage": 2})

# Generate configurations
configs = policy.emit_config(topk=10)

for config in configs:
    print(f"Block: {config.block}")
    print(f"Warp: {config.warp}")
    print(f"TensorCore: {config.use_tc}")
```

### Custom Configuration

```python
# Custom tags for specialized optimization
tags = {
    "pipeline_stage": 3,
    "use_async_copy": True,
    "block_reduction_depth": 2,
    "tensorcore_config": (0, 1),  # (M_axis, N_axis)
    "intrin_info": {
        "in_dtype": "float16",
        "out_dtype": "float32",
        "trans_a": False,
        "trans_b": True
    }
}

policy = TensorCorePolicy(arch, tags=tags)
```

### Cost Analysis

```python
# Analyze specific tile configuration
output_tile = [128, 128]
rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}

td = policy.compute_tile_dict(output_tile, rstep_map)

print(f"Valid: {td.valid}")
print(f"Memory Traffic: {td.traffic} bytes")
print(f"Shared Memory: {td.smem_cost} bytes")
print(f"Blocks per SM: {td.block_per_SM}")
print(f"Grid Size: {td.grid_size}")
print(f"Waves: {td.num_wave}")
```

### Architecture-Specific Optimization

```python
from tilelang.carver.arch.cuda import *

# Check architecture capabilities
if is_ampere_arch(arch):
    print("Ampere architecture detected")
    print(f"Async copy support: {arch.sm_version >= 80}")
    print(f"Available shapes: {arch.get_available_tensorintrin_shapes()}")

# Check TensorCore precision support
if is_tensorcore_supported_precision("float16", "float32", arch):
    print("FP16 TensorCore operations supported")
    
    # Configure for mixed-precision
    policy.wmma_k = 16
    policy.use_async_copy = True
    policy.pipeline_stage = 2
```

This API reference provides comprehensive documentation for all public interfaces and classes in the TileLang meta-scheduling system, enabling developers to effectively utilize the cost model and TensorCore optimization capabilities.