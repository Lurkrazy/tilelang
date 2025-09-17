# Architecture Support

This document provides comprehensive documentation of TileLang's hardware abstraction layer and device-specific optimizations, implemented primarily in `tilelang/carver/arch/`.

## Overview

TileLang's architecture support framework provides:

- **Hardware Abstraction**: Unified interface for different GPU architectures
- **Device-Specific Optimizations**: Architecture-aware scheduling and optimization
- **Extensible Design**: Support for adding new architectures and backends
- **Performance Modeling**: Architecture-specific performance characteristics

## Architecture Hierarchy

### Base Architecture Class

The `TileDevice` class in `tilelang/carver/arch/arch_base.py` provides the foundation:

```python
class TileDevice:
    """
    Represents the architecture of a computing device, capturing various hardware specifications.
    """
    
    def __init__(self) -> None:
        # Core compute specifications
        self.reg_cap: int = 0                    # Register capacity (bytes)
        self.smem_cap: int = 0                   # Shared memory capacity (bytes)
        self.compute_max_core: int = 0           # Maximum number of compute units
        self.warp_size: int = 0                  # Warp/wavefront size
        self.sm_partition: int = 0               # Streaming multiprocessor partitions
        
        # Memory specifications
        self.max_smem_usage: int = 0             # Maximum shared memory usage
        self.l2_cache_size_bytes: int = 0        # L2 cache size
        self.transaction_size: List[int] = [0, 0]  # Memory transaction sizes (bytes)
        self.bandwidth: List[int] = [0, 0]       # Memory bandwidth [write, read] (MB/s)
        
        # Platform identification
        self.platform: str = "unknown"          # Platform name (CUDA, ROCm, etc.)
        self.compute_capability: str = "unknown" # Compute capability version
    
    def get_avaliable_tensorintrin_shapes(self):
        """Get available tensor instruction shapes for this architecture."""
        raise NotImplementedError()
```

## CUDA Architecture Support

### CUDA Device Implementation

The `CUDA` class in `tilelang/carver/arch/cuda.py` implements NVIDIA GPU support:

```python
class CUDA(TileDevice):
    """
    CUDA architecture implementation for NVIDIA GPUs.
    """
    
    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        
        self.target = target
        self.sm_version = check_sm_version(self.target.arch)
        
        # Initialize CUDA device
        device = tvm.runtime.cuda(0)
        if not device.exist:
            raise RuntimeError("Cannot find cuda device 0.")
        
        # Device properties from CUDA runtime
        self.name = cuda_driver.get_device_name()
        self.device = device
        self.platform = "CUDA"
        self.smem_cap = cuda_driver.get_shared_memory_per_block()
        self.compute_max_core = device.multi_processor_count
        self.warp_size = device.warp_size
        self.compute_capability = device.compute_version.replace(".", "")
        
        # Architecture-specific configurations
        self.reg_cap = 65536                     # 64KB register file per SM
        self.max_smem_usage = 2 * self.smem_cap  # Dynamic shared memory limit
        self.sm_partition = 4                    # Warp schedulers per SM
        self.l2_cache_size_bytes = target.l2_cache_size_bytes
        
        # Memory subsystem characteristics
        self.transaction_size = [32, 128]        # L1/L2 transaction sizes (bytes)
        self.bandwidth = [750, 12080]            # Approximate bandwidth [write, read] (MB/s)
        
        # TensorCore instruction support
        self.available_tensor_instructions = None
    
    def get_avaliable_tensorintrin_shapes(self):
        """Get available TensorCore instruction shapes."""
        self.available_tensor_instructions = (
            TensorInstruction("mma", [16, 16]),   # Matrix-Multiply-Accumulate
            TensorInstruction("wmma", [16, 16]),  # Warp Matrix-Multiply-Accumulate
        )
        return [t.shape for t in self.available_tensor_instructions]
```

### Architecture Detection Functions

```python
def check_sm_version(arch: str) -> int:
    """Extract SM version from architecture string."""
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1

def is_cuda_arch(arch: TileDevice) -> bool:
    """Check if architecture is CUDA-based."""
    return isinstance(arch, CUDA)

def is_volta_arch(arch: TileDevice) -> bool:
    """Check if architecture is Volta (SM 7.x)."""
    return (is_cuda_arch(arch) and 
            70 <= arch.sm_version < 80)

def is_ampere_arch(arch: TileDevice) -> bool:
    """Check if architecture is Ampere (SM 8.0-8.9)."""
    return (is_cuda_arch(arch) and 
            80 <= arch.sm_version < 89)

def is_ada_arch(arch: TileDevice) -> bool:
    """Check if architecture is Ada Lovelace (SM 8.9)."""
    return (is_cuda_arch(arch) and 
            arch.sm_version == 89)

def is_hopper_arch(arch: TileDevice) -> bool:
    """Check if architecture is Hopper (SM 9.0)."""
    return (is_cuda_arch(arch) and 
            arch.sm_version == 90)

def has_mma_support(arch: TileDevice) -> bool:
    """Check if architecture supports MMA (Matrix-Multiply-Accumulate) instructions."""
    return (is_cuda_arch(arch) and 
            arch.sm_version >= 80)
```

### TensorCore Precision Support

Architecture-specific precision support is defined for different GPU generations:

```python
# Volta TensorCore support (SM 7.0-7.5)
volta_tensorcore_supported = [
    ("float16", "float32"),
    ("float16", "float16"),
]

# Ampere TensorCore support (SM 8.0-8.9)
ampere_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("int4", "int32"),
    ("int2", "int32"),
    ("int1", "int32"),
]

# Ada Lovelace TensorCore support (SM 8.9)
ada_tensorcore_supported = [
    ("bfloat16", "float32"),
    ("float16", "float32"),
    ("float16", "float16"),
    ("int8", "int32"),
    ("float8_e5m2", "float32"),  # FP8 formats
    ("float8_e4m3", "float32"),
]

# Hopper TensorCore support (SM 9.0)
hopper_tensorcore_supported = ada_tensorcore_supported

def is_tensorcore_supported_precision(in_dtype: str, accum_dtype: str, arch: TileDevice) -> bool:
    """
    Check if TensorCore supports the specified precision combination.
    
    Args:
        in_dtype: Input data type
        accum_dtype: Accumulation data type
        arch: Target architecture
        
    Returns:
        bool: True if precision combination is supported
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

### TensorCore Instruction Configuration

```python
class TensorInstruction:
    """
    Represents a TensorCore instruction with specific shape characteristics.
    """
    
    def __init__(self, name: str, shape: List[int]):
        self.name = name        # Instruction name (mma, wmma, etc.)
        self.shape = shape      # Instruction shape [M, N] dimensions
```

## AMD Architecture Support

### CDNA Implementation

The `CDNA` class in `tilelang/carver/arch/cdna.py` implements AMD GPU support:

```python
class CDNA(TileDevice):
    """
    CDNA architecture implementation for AMD GPUs.
    """
    
    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        
        self.target = target
        self.platform = "ROCm"
        
        # AMD-specific device properties
        self.compute_max_core = 104             # CUs for MI250X
        self.warp_size = 64                     # Wavefront size
        self.smem_cap = 65536                   # Local Data Share (LDS) capacity
        self.reg_cap = 65536                    # Vector register file
        self.max_smem_usage = self.smem_cap
        self.sm_partition = 4                   # SIMD units per CU
        
        # Memory characteristics
        self.transaction_size = [64, 128]       # L1/L2 transaction sizes
        self.bandwidth = [1600, 3200]           # HBM bandwidth (higher than CUDA)
        self.l2_cache_size_bytes = 8 * 1024 * 1024  # 8MB L2 cache
        
        # Matrix Core support
        self.available_tensor_instructions = None
    
    def get_avaliable_tensorintrin_shapes(self):
        """Get available Matrix Core instruction shapes."""
        self.available_tensor_instructions = (
            TensorInstruction("mfma", [16, 16]), # Matrix Fused Multiply-Add
            TensorInstruction("mfma", [32, 32]), # Larger tile size
        )
        return [t.shape for t in self.available_tensor_instructions]
```

### ROCm Architecture Detection

```python
def is_cdna_arch(arch: TileDevice) -> bool:
    """Check if architecture is AMD CDNA-based."""
    return isinstance(arch, CDNA)

def has_mfma_support(arch: TileDevice) -> bool:
    """Check if architecture supports MFMA (Matrix Fused Multiply-Add) instructions."""
    return is_cdna_arch(arch)
```

## CPU Architecture Support

### CPU Implementation

```python
class CPU(TileDevice):
    """
    CPU architecture implementation for x86/ARM processors.
    """
    
    def __init__(self, target: Union[Target, str]):
        if isinstance(target, str):
            target = tvm.target.Target(target)
        
        self.target = target
        self.platform = "CPU"
        
        # CPU-specific properties
        self.compute_max_core = 16              # Number of CPU cores
        self.warp_size = 1                      # No SIMT execution
        self.smem_cap = 32 * 1024               # L1 cache size (approximate)
        self.reg_cap = 4 * 1024                 # Register file size
        self.max_smem_usage = self.smem_cap
        self.sm_partition = 1                   # No partition concept
        
        # Memory hierarchy
        self.transaction_size = [64, 64]        # Cache line size
        self.bandwidth = [50, 100]              # DDR4/DDR5 bandwidth (GB/s)
        self.l2_cache_size_bytes = 256 * 1024   # 256KB L2 per core
        
        # Vector instruction support
        self.available_tensor_instructions = None
    
    def get_avaliable_tensorintrin_shapes(self):
        """Get available vector instruction shapes."""
        self.available_tensor_instructions = (
            TensorInstruction("vnni", [1, 16]),  # Vector Neural Network Instructions
            TensorInstruction("amx", [16, 16]),  # Advanced Matrix Extensions
        )
        return [t.shape for t in self.available_tensor_instructions]
```

## Device Driver Integration

### CUDA Driver Interface

```python
# tilelang/carver/arch/driver/cuda_driver.py
class CUDADriver:
    """
    Interface to CUDA driver for device information retrieval.
    """
    
    @staticmethod
    def get_device_name() -> str:
        """Get CUDA device name."""
        # Implementation uses CUDA runtime API
        pass
    
    @staticmethod
    def get_shared_memory_per_block() -> int:
        """Get maximum shared memory per block."""
        # Implementation queries device properties
        pass
    
    @staticmethod
    def get_compute_capability() -> str:
        """Get device compute capability."""
        # Implementation queries device properties
        pass

cuda_driver = CUDADriver()
```

### ROCm Driver Interface

```python
# tilelang/carver/arch/driver/rocm_driver.py
class ROCmDriver:
    """
    Interface to ROCm driver for device information retrieval.
    """
    
    @staticmethod
    def get_device_name() -> str:
        """Get ROCm device name."""
        # Implementation uses HIP runtime API
        pass
    
    @staticmethod
    def get_local_memory_size() -> int:
        """Get maximum local memory (LDS) size."""
        # Implementation queries device properties
        pass

rocm_driver = ROCmDriver()
```

## Architecture-Specific Optimizations

### Memory Layout Optimization

Different architectures require different memory layout strategies:

```python
class ArchitectureOptimizer:
    """
    Architecture-specific optimization strategies.
    """
    
    @staticmethod
    def get_optimal_memory_layout(arch: TileDevice, tensor_shape: List[int]) -> str:
        """
        Get optimal memory layout for tensor on target architecture.
        
        Args:
            arch: Target architecture
            tensor_shape: Tensor dimensions
            
        Returns:
            str: Layout specification
        """
        if is_cuda_arch(arch):
            if is_hopper_arch(arch):
                return "swizzled_128b"  # Hopper swizzling
            elif is_ampere_arch(arch):
                return "swizzled_64b"   # Ampere swizzling
            else:
                return "row_major"      # Default row-major
        elif is_cdna_arch(arch):
            return "lds_optimized"      # AMD LDS-optimized layout
        else:
            return "row_major"          # CPU default
    
    @staticmethod
    def get_optimal_tile_size(arch: TileDevice, operation: str) -> List[int]:
        """
        Get architecture-optimal tile size for operation.
        
        Args:
            arch: Target architecture
            operation: Operation type (gemm, conv, etc.)
            
        Returns:
            List[int]: Optimal tile dimensions
        """
        if operation == "gemm":
            if is_hopper_arch(arch):
                return [128, 256, 64]   # Large tiles for Hopper
            elif is_ampere_arch(arch):
                return [128, 128, 32]   # Medium tiles for Ampere
            elif is_volta_arch(arch):
                return [64, 64, 16]     # Smaller tiles for Volta
            elif is_cdna_arch(arch):
                return [128, 128, 16]   # AMD-optimized tiles
            else:
                return [32, 32, 32]     # Conservative CPU tiles
        else:
            # Default tile sizes for other operations
            return [64, 64]
```

### Architecture-Aware Scheduling

```python
class ArchitectureScheduler:
    """
    Generates architecture-specific scheduling strategies.
    """
    
    @staticmethod
    def get_thread_block_size(arch: TileDevice, tile_size: List[int]) -> int:
        """
        Calculate optimal thread block size for architecture.
        
        Args:
            arch: Target architecture
            tile_size: Tile dimensions
            
        Returns:
            int: Optimal thread block size
        """
        if is_cuda_arch(arch):
            # CUDA: maximize occupancy while staying within limits
            max_threads = 1024
            warp_size = arch.warp_size
            
            # Calculate threads needed for tile
            threads_needed = np.prod(tile_size)
            
            # Round up to next warp boundary
            threads_aligned = ((threads_needed + warp_size - 1) // warp_size) * warp_size
            
            return min(threads_aligned, max_threads)
            
        elif is_cdna_arch(arch):
            # AMD: optimize for wavefront size (64)
            wavefront_size = arch.warp_size
            threads_needed = np.prod(tile_size)
            
            # Round up to next wavefront boundary
            threads_aligned = ((threads_needed + wavefront_size - 1) // wavefront_size) * wavefront_size
            
            return min(threads_aligned, 1024)  # AMD limit
            
        else:
            # CPU: use number of cores
            return min(arch.compute_max_core, np.prod(tile_size))
    
    @staticmethod
    def get_pipeline_stages(arch: TileDevice) -> int:
        """
        Get optimal pipeline stage count for architecture.
        
        Args:
            arch: Target architecture
            
        Returns:
            int: Number of pipeline stages
        """
        if is_hopper_arch(arch):
            return 4    # Deep pipeline for Hopper
        elif is_ampere_arch(arch):
            return 3    # Medium pipeline for Ampere
        elif is_volta_arch(arch):
            return 2    # Shallow pipeline for Volta
        elif is_cdna_arch(arch):
            return 2    # AMD pipeline depth
        else:
            return 1    # CPU: no pipeline
```

## Cross-Platform Compatibility

### Unified Architecture Interface

```python
class UnifiedArchitecture:
    """
    Provides unified interface across different architectures.
    """
    
    def __init__(self, arch: TileDevice):
        self.arch = arch
        self.capabilities = self._analyze_capabilities()
    
    def _analyze_capabilities(self) -> Dict[str, bool]:
        """Analyze architecture capabilities."""
        capabilities = {
            'tensor_cores': False,
            'async_copy': False,
            'shared_memory': False,
            'vector_instructions': False,
        }
        
        if is_cuda_arch(self.arch):
            capabilities['tensor_cores'] = has_mma_support(self.arch)
            capabilities['async_copy'] = is_ampere_arch(self.arch) or is_hopper_arch(self.arch)
            capabilities['shared_memory'] = True
            
        elif is_cdna_arch(self.arch):
            capabilities['tensor_cores'] = has_mfma_support(self.arch)
            capabilities['async_copy'] = True  # AMD async copy
            capabilities['shared_memory'] = True  # LDS
            
        # CPU capabilities would be set here
        
        return capabilities
    
    def get_optimal_configuration(self, operation: str) -> Dict[str, Any]:
        """
        Get optimal configuration for operation on this architecture.
        
        Args:
            operation: Operation type
            
        Returns:
            Dict: Optimal configuration parameters
        """
        config = {}
        
        # Architecture-specific optimizations
        if self.capabilities['tensor_cores']:
            config['use_tensor_cores'] = True
            config['precision'] = 'mixed'
        else:
            config['use_tensor_cores'] = False
            config['precision'] = 'float32'
        
        if self.capabilities['async_copy']:
            config['async_copy'] = True
            config['pipeline_stages'] = ArchitectureScheduler.get_pipeline_stages(self.arch)
        else:
            config['async_copy'] = False
            config['pipeline_stages'] = 1
        
        return config
```

## Performance Characteristics Database

### Architecture Performance Models

```python
# Performance characteristics by architecture
PERFORMANCE_DATABASE = {
    # NVIDIA Architectures
    "sm_70": {  # Volta
        "peak_flops": {"fp32": 7.8e12, "fp16": 15.6e12},
        "memory_bandwidth": 900e9,  # bytes/sec
        "cache_sizes": {"l1": 128*1024, "l2": 6*1024*1024},
        "tensor_performance": {"fp16": 125e12},
    },
    "sm_80": {  # Ampere
        "peak_flops": {"fp32": 19.5e12, "fp16": 39e12, "bf16": 39e12},
        "memory_bandwidth": 1555e9,
        "cache_sizes": {"l1": 192*1024, "l2": 8*1024*1024},
        "tensor_performance": {"fp16": 312e12, "bf16": 312e12, "int8": 624e12},
    },
    "sm_89": {  # Ada Lovelace
        "peak_flops": {"fp32": 42e12, "fp16": 84e12, "bf16": 84e12},
        "memory_bandwidth": 1000e9,  # RTX 4090
        "cache_sizes": {"l1": 128*1024, "l2": 72*1024*1024},
        "tensor_performance": {"fp16": 165e12, "bf16": 165e12, "fp8": 330e12},
    },
    "sm_90": {  # Hopper
        "peak_flops": {"fp32": 60e12, "fp16": 120e12, "bf16": 120e12},
        "memory_bandwidth": 3350e9,  # H100
        "cache_sizes": {"l1": 256*1024, "l2": 50*1024*1024},
        "tensor_performance": {"fp16": 990e12, "bf16": 990e12, "fp8": 1980e12},
    },
    
    # AMD Architectures
    "gfx90a": {  # MI250X
        "peak_flops": {"fp32": 47.9e12, "fp16": 95.8e12, "bf16": 95.8e12},
        "memory_bandwidth": 3277e9,
        "cache_sizes": {"l1": 16*1024, "l2": 8*1024*1024},
        "tensor_performance": {"fp16": 383e12, "bf16": 383e12},
    },
}

def get_architecture_performance(arch: TileDevice, dtype: str = "fp32") -> Dict[str, float]:
    """
    Get performance characteristics for architecture and data type.
    
    Args:
        arch: Target architecture
        dtype: Data type
        
    Returns:
        Dict: Performance characteristics
    """
    arch_key = arch.compute_capability.replace(".", "_")
    if arch_key.startswith("sm_"):
        arch_key = arch_key  # NVIDIA format
    else:
        arch_key = f"gfx{arch_key}"  # AMD format
    
    if arch_key not in PERFORMANCE_DATABASE:
        raise ValueError(f"Unsupported architecture: {arch_key}")
    
    perf_data = PERFORMANCE_DATABASE[arch_key]
    
    return {
        "peak_flops": perf_data["peak_flops"].get(dtype, perf_data["peak_flops"]["fp32"]),
        "memory_bandwidth": perf_data["memory_bandwidth"],
        "l1_cache_size": perf_data["cache_sizes"]["l1"],
        "l2_cache_size": perf_data["cache_sizes"]["l2"],
        "tensor_flops": perf_data.get("tensor_performance", {}).get(dtype, 0)
    }
```

## Future Architecture Support

### Extensibility Framework

```python
class FutureArchitecture(TileDevice):
    """
    Template for adding new architecture support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Load configuration from specification
        self.platform = config["platform"]
        self.compute_capability = config["compute_capability"]
        self.compute_max_core = config["compute_units"]
        self.warp_size = config["simd_width"]
        self.smem_cap = config["local_memory_size"]
        self.bandwidth = config["memory_bandwidth"]
        
        # Architecture-specific features
        self.features = config.get("features", {})
        
    def get_avaliable_tensorintrin_shapes(self):
        """Define available tensor instructions."""
        instructions = []
        if "matrix_instructions" in self.features:
            for instr in self.features["matrix_instructions"]:
                instructions.append(TensorInstruction(instr["name"], instr["shape"]))
        return [t.shape for t in instructions]
    
    def supports_feature(self, feature: str) -> bool:
        """Check if architecture supports specific feature."""
        return feature in self.features
```

### Configuration-Driven Architecture Definition

```python
# Example architecture configuration file (YAML/JSON)
EXAMPLE_ARCH_CONFIG = {
    "platform": "Intel XPU",
    "compute_capability": "xe_hpg",
    "compute_units": 512,
    "simd_width": 16,
    "local_memory_size": 65536,
    "memory_bandwidth": [800, 1600],
    "features": {
        "matrix_instructions": [
            {"name": "xe_mma", "shape": [8, 16]},
            {"name": "xe_dot", "shape": [1, 16]}
        ],
        "async_copy": True,
        "fp16_support": True,
        "int8_support": True
    }
}

def load_architecture_from_config(config_path: str) -> TileDevice:
    """Load architecture from configuration file."""
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return FutureArchitecture(config)
```

This comprehensive architecture support framework enables TileLang to efficiently target multiple hardware platforms while providing architecture-specific optimizations for maximum performance.