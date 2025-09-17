# Hardware Abstraction Documentation

## Overview

TileLang's hardware abstraction layer provides a unified interface for different computing devices and architectures. The `TileDevice` base class defines common hardware characteristics, while specialized implementations provide architecture-specific optimizations.

## Base Hardware Abstraction

### TileDevice Class

```python
class TileDevice:
    """
    Represents the architecture of a computing device, capturing various hardware specifications.
    """
    
    def __init__(self) -> None:
        self.reg_cap: int = 0                    # Register capacity per SM
        self.smem_cap: int = 0                   # Shared memory capacity per block
        self.compute_max_core: int = 0           # Maximum number of SMs
        self.warp_size: int = 0                  # Warp size (threads per warp)
        self.sm_partition: int = 0               # SM partition count
        self.transaction_size: List[int] = [0, 0] # [write, read] transaction sizes
        self.max_smem_usage: int = 0             # Maximum shared memory usage
        self.bandwidth: List[int] = [0, 0]       # Memory bandwidth specifications
        self.platform: str = "unknown"          # Platform identifier
        self.compute_capability: str = "unknown" # Compute capability version
        self.l2_cache_size_bytes: int = 0        # L2 cache size
```

#### Core Hardware Characteristics

##### Register Resources
- **reg_cap**: Total register capacity per streaming multiprocessor
- **Usage**: Determines maximum register pressure for kernels
- **Impact**: Higher register usage reduces occupancy

##### Memory Hierarchy
- **smem_cap**: Shared memory capacity per thread block
- **max_smem_usage**: Maximum usable shared memory (may exceed static limit)
- **l2_cache_size_bytes**: L2 cache size for cache-aware optimizations

##### Compute Resources
- **compute_max_core**: Number of streaming multiprocessors
- **warp_size**: Number of threads per warp (typically 32)
- **sm_partition**: SM partitioning factor for occupancy calculations

##### Memory Characteristics
- **transaction_size**: Memory transaction sizes [write, read] in bytes
- **bandwidth**: Memory bandwidth specifications [peak, sustained] in MB/s

## CUDA Architecture Implementation

### CUDA Device Class

```python
class CUDA(TileDevice):
    """
    CUDA-specific hardware abstraction with dynamic device detection and configuration.
    """
    
    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        
        self.target = target
        self.sm_version = check_sm_version(self.target.arch)
        device = tvm.runtime.cuda(0)
        
        # Runtime device information
        self.name = cuda_driver.get_device_name()
        self.device: tvm.runtime.Device = device
        self.platform: str = "CUDA"
        self.compute_capability = device.compute_version.replace(".", "")
```

#### Device Detection and Configuration

```python
def __init__(self, target: Union[Target, str]):
    """
    Initialize CUDA device with runtime detection:
    1. Parse target specification
    2. Query hardware capabilities
    3. Configure architecture-specific parameters
    """
    # Hardware capability detection
    self.smem_cap = cuda_driver.get_shared_memory_per_block()
    self.compute_max_core = device.multi_processor_count
    self.warp_size = device.warp_size
    self.reg_cap: int = 65536  # 64K registers per SM
    self.max_smem_usage: int = 2 * self.smem_cap  # Allow dynamic shared memory
    self.sm_partition: int = 4  # Typical SM partition count
```

### Architecture Detection Functions

#### Compute Capability Analysis

```python
def check_sm_version(arch: str) -> int:
    """Extract numeric SM version from architecture string"""
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1

def is_cuda_arch(arch: TileDevice) -> bool:
    """Verify CUDA architecture"""
    return isinstance(arch, CUDA)
```

#### Architecture Generation Detection

```python
def is_volta_arch(arch: TileDevice) -> bool:
    """Volta architecture: SM 7.0-7.5"""
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 70)
    conditions.append(arch.sm_version < 80)
    return all(conditions)

def is_ampere_arch(arch: TileDevice) -> bool:
    """Ampere architecture: SM 8.0-8.6"""
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80 and arch.sm_version < 89)
    return all(conditions)

def is_ada_arch(arch: TileDevice) -> bool:
    """Ada Lovelace architecture: SM 8.9"""
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version == 89)
    return all(conditions)

def is_hopper_arch(arch: TileDevice) -> bool:
    """Hopper architecture: SM 9.0"""
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version == 90)
    return all(conditions)
```

### TensorCore Capability Detection

#### Architecture-Specific TensorCore Support

```python
def has_mma_support(arch: TileDevice) -> bool:
    """Check for Matrix Multiply-Accumulate (MMA) instruction support"""
    conditions = [True]
    conditions.append(is_cuda_arch(arch))
    conditions.append(arch.sm_version >= 80)  # Ampere+ architectures
    return all(conditions)
```

#### Precision Support Matrix

```python
# Volta TensorCore capabilities
volta_tensorcore_supported = [
    ("float16", "float32"),
    ("float16", "float16"),
]

# Ampere TensorCore capabilities  
ampere_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("int4", "int32"),
    ("int2", "int32"),
    ("int1", "int32"),
]

# Ada Lovelace TensorCore capabilities
ada_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("float8_e5m2", "float32"),
    ("float8_e4m3", "float32"),
]

# Hopper TensorCore capabilities (same as Ada)
hopper_tensorcore_supported = ada_tensorcore_supported
```

#### Dynamic Precision Support Checking

```python
def is_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    """
    Determines TensorCore support for specific input and accumulation data types.
    
    Args:
        in_dtype: Input tensor data type
        accum_dtype: Accumulation data type
        arch: Target hardware architecture
        
    Returns:
        Boolean indicating TensorCore support for the precision combination
    """
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

### TensorCore Instruction Abstraction

#### Instruction Definition

```python
class TensorInstruction(object):
    """
    Represents a TensorCore instruction with specific shape and precision characteristics.
    """
    
    def __init__(self, name: str, shape: List[int]):
        self.name = name          # Instruction name (e.g., "wmma.m16n16k16")
        self.shape = shape        # Matrix dimensions [M, N, K]
        self.input_dtype = None   # Input data type
        self.output_dtype = None  # Output/accumulation data type
```

#### Available TensorCore Shapes

```python
def get_available_tensorintrin_shapes(self):
    """
    Returns available TensorCore instruction shapes for the architecture.
    
    Common shapes:
    - Volta: [16, 16, 16] for WMMA operations
    - Ampere: [16, 16, 16], [32, 8, 16], [8, 32, 16] for MMA operations
    - Ada/Hopper: Enhanced shapes with FP8 support
    """
    if is_volta_arch(self):
        return [[16, 16, 16]]
    elif is_ampere_arch(self) or is_ada_arch(self) or is_hopper_arch(self):
        return [[16, 16, 16], [32, 8, 16], [8, 32, 16]]
    else:
        return []
```

## Alternative Architecture Support

### CDNA Architecture (AMD)

```python
class CDNA(TileDevice):
    """
    AMD CDNA architecture abstraction for ROCm-based systems.
    """
    
    def __init__(self, target: Union[Target, str]):
        super().__init__()
        self.platform = "ROCm"
        # CDNA-specific configuration
        self.warp_size = 64        # AMD wavefront size
        self.compute_capability = "gfx90a"  # Example CDNA capability
```

### CPU Architecture

```python
class CPU(TileDevice):
    """
    CPU architecture abstraction for host-based execution.
    """
    
    def __init__(self, target: Union[Target, str]):
        super().__init__()
        self.platform = "CPU"
        self.warp_size = 1         # No SIMT execution
        self.smem_cap = 0          # No shared memory
        # CPU-specific optimizations focus on cache hierarchy
```

## Architecture-Specific Optimizations

### Memory Configuration

#### CUDA Memory Hierarchy
```python
# Volta architecture (SM 7.0-7.5)
if is_volta_arch(self):
    self.smem_cap = 96 * 1024           # 96KB shared memory
    self.transaction_size = [128, 128]   # Standard transaction sizes
    self.bandwidth = [900000, 900000]    # HBM2 bandwidth (~900 GB/s)

# Ampere architecture (SM 8.0-8.6)  
elif is_ampere_arch(self):
    self.smem_cap = 164 * 1024          # 164KB shared memory
    self.transaction_size = [128, 128]   # Enhanced transaction handling
    self.bandwidth = [1555000, 1555000]  # HBM2e bandwidth (~1.5 TB/s)

# Ada Lovelace architecture (SM 8.9)
elif is_ada_arch(self):
    self.smem_cap = 128 * 1024          # 128KB shared memory
    self.transaction_size = [128, 128]   # Optimized for gaming workloads
    self.bandwidth = [1000000, 1000000]  # GDDR6X bandwidth (~1 TB/s)
```

#### L2 Cache Configuration
```python
# Architecture-specific L2 cache sizes
if is_hopper_arch(self):
    self.l2_cache_size_bytes = 60 * 1024 * 1024  # 60MB L2 cache
elif is_ampere_arch(self):
    self.l2_cache_size_bytes = 6 * 1024 * 1024   # 6MB L2 cache
elif is_volta_arch(self):
    self.l2_cache_size_bytes = 6 * 1024 * 1024   # 6MB L2 cache
```

### Compute Capability Features

#### Feature Detection
```python
def supports_async_copy(arch: TileDevice) -> bool:
    """Check for hardware async copy support (Ampere+)"""
    return is_cuda_arch(arch) and arch.sm_version >= 80

def supports_fp8(arch: TileDevice) -> bool:
    """Check for FP8 TensorCore support (Ada+)"""
    return is_cuda_arch(arch) and arch.sm_version >= 89

def supports_sparse_tensors(arch: TileDevice) -> bool:
    """Check for 2:4 structured sparsity (Ampere+)"""
    return is_cuda_arch(arch) and arch.sm_version >= 80
```

## Driver Interface

### CUDA Driver Integration

```python
from .driver import cuda_driver

class CUDA(TileDevice):
    def __init__(self, target: Union[Target, str]):
        # Runtime hardware detection through driver interface
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        
        self.name = cuda_driver.get_device_name()
        self.smem_cap = cuda_driver.get_shared_memory_per_block()
        self.compute_max_core = device.multi_processor_count
```

#### Driver Functions
- **get_device_name()**: Retrieves GPU device name
- **get_shared_memory_per_block()**: Queries shared memory capacity
- **Runtime device properties**: Accesses hardware characteristics

## Architecture Selection and Configuration

### Target Specification

```python
# String-based target specification
arch = CUDA("cuda -arch=sm_80")

# TVM Target object specification  
target = tvm.target.Target("cuda -arch=sm_80")
arch = CUDA(target)

# Automatic detection (uses default device)
arch = CUDA("cuda")
```

### Multi-Architecture Support

```python
def create_device(platform: str, target: str) -> TileDevice:
    """Factory function for creating architecture-specific devices"""
    if platform.lower() == "cuda":
        return CUDA(target)
    elif platform.lower() == "rocm":
        return CDNA(target)
    elif platform.lower() == "cpu":
        return CPU(target)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
```

## Performance Characteristics

### Architecture Comparison

#### Volta vs Ampere vs Hopper
| Feature | Volta | Ampere | Hopper |
|---------|-------|--------|--------|
| Shared Memory | 96KB | 164KB | 228KB |
| TensorCore Precision | FP16 | FP16, BF16, INT8, INT4 | + FP8 |
| Async Copy | No | Yes | Yes |
| MMA Instructions | WMMA | MMA | Enhanced MMA |
| L2 Cache | 6MB | 6-40MB | 60MB |

#### Memory Bandwidth Evolution
- **Volta**: ~900 GB/s (HBM2)
- **Ampere**: ~1.5 TB/s (HBM2e)  
- **Ada**: ~1 TB/s (GDDR6X)
- **Hopper**: ~3 TB/s (HBM3)

### Optimization Guidelines

#### Architecture-Specific Strategies

##### Volta Optimization
- Focus on WMMA instruction utilization
- Optimize for 96KB shared memory limit
- Single-stage computation pipelines

##### Ampere Optimization
- Leverage async copy operations
- Utilize enhanced TensorCore precisions
- Multi-stage computation pipelines
- Take advantage of larger shared memory

##### Hopper Optimization
- Maximize FP8 TensorCore utilization
- Leverage thread block clusters
- Optimize for massive L2 cache
- Utilize enhanced sparsity support

## Integration with Cost Model

### Hardware-Aware Cost Modeling

```python
# Architecture-specific cost adjustments
if is_ampere_arch(self.arch):
    # Ampere benefits from pipelining
    pipeline_factor = 0.8
    async_copy_benefit = 0.9
    
elif is_volta_arch(self.arch):
    # Volta requires careful shared memory management
    smem_pressure_factor = 1.2
    pipeline_factor = 1.0
```

### Resource Constraint Modeling

```python
def validate_configuration(self, config, arch: TileDevice):
    """
    Validates kernel configuration against hardware constraints:
    - Shared memory limits
    - Register capacity
    - TensorCore requirements
    - Occupancy constraints
    """
    # Shared memory validation
    if config.smem_usage > arch.smem_cap:
        return False, "Shared memory limit exceeded"
    
    # Register validation
    if config.reg_usage > arch.reg_cap:
        return False, "Register limit exceeded"
        
    # TensorCore validation
    if config.use_tensorcore and not arch.supports_tensorcore():
        return False, "TensorCore not supported"
```

This comprehensive hardware abstraction layer enables TileLang to efficiently target diverse computing architectures while providing optimal performance through architecture-specific optimizations.