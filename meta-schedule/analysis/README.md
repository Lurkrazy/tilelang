# Performance Analysis Framework

This document provides comprehensive documentation of TileLang's performance analysis framework, which combines IR analysis, roofline modeling, and architecture-aware performance estimation.

## Overview

TileLang's performance analysis framework provides accurate performance modeling through:

- **IR-Level Analysis**: Direct analysis of TVM Intermediate Representation
- **Operation Counting**: Precise FLOP and memory traffic analysis
- **Roofline Modeling**: Hardware-aware performance bound estimation
- **Architecture Integration**: Device-specific performance characteristics

## Core Analysis Components

### 1. Analyzer Class

The main `Analyzer` class in `tilelang/tools/Analyzer.py` orchestrates the analysis process:

```python
class Analyzer:
    """
    A class to analyze the performance of a TVM IR module.
    Calculates metrics such as FLOPs, memory bandwidth, and estimated execution time.
    """
    
    def __init__(self, fn, device):
        if isinstance(fn, tvm.tir.function.PrimFunc):
            self.fn = tvm.IRModule({"main": fn})
        else:
            self.fn = fn
        
        self.device = device
        self.total_flops = 0
        self.total_global_bytes = 0
        self.block_counts = {"blockIdx.x": 1, "blockIdx.y": 1}
        self.loop_stack = []
        self.global_buffers = set()
```

### 2. Analysis Result Structure

Results are encapsulated in the `AnalysisResult` dataclass:

```python
@dataclass(frozen=True)
class AnalysisResult:
    """
    Results of performance analysis.
    """
    total_flops: int                    # Total floating-point operations
    total_global_bytes: int             # Total bytes transferred to/from global memory
    estimated_time: float               # Estimated execution time (seconds)
    expected_tflops: float              # Theoretical peak TFLOPS
    expected_bandwidth_GBps: float      # Theoretical peak bandwidth (GB/s)
```

## IR Analysis Pass Implementation

### IR Transformation Framework

The analyzer uses TVM's IR transformation infrastructure:

```python
def ir_pass(self):
    """
    Traverse and transform the IR module to extract performance information.
    """
    
    def _ftransform(f, mod, ctx):
        # Initialize global buffer set
        self.global_buffers = set(f.buffer_map.values())
        
        def _pre_visit(stmt):
            """Pre-visit callback for IR nodes."""
            if isinstance(stmt, tvm.tir.AttrStmt):
                # Handle thread extent attributes for block dimensions
                if stmt.attr_key == "thread_extent":
                    iter_var = stmt.node
                    thread_tag = iter_var.thread_tag
                    if thread_tag in self.block_counts:
                        extent = stmt.value.value if hasattr(stmt.value, 'value') else stmt.value
                        self.block_counts[thread_tag] = extent
                        
            elif isinstance(stmt, tvm.tir.For):
                # Track loop nesting for iteration counting
                self.loop_stack.append(stmt.extent)
                
            elif isinstance(stmt, tvm.tir.Evaluate):
                # Analyze operation calls
                value = stmt.value
                if isinstance(value, tvm.tir.Call):
                    if value.op.name == "tl.copy":
                        self._analyze_copy(value)
                    elif value.op.name == "tl.gemm":
                        self._analyze_gemm(value)
            
            return None
        
        def _post_visit(stmt):
            """Post-visit callback for cleanup."""
            if isinstance(stmt, tvm.tir.For) and self.loop_stack:
                self.loop_stack.pop()
            return None
        
        # Apply IR transformation
        new_body = ir_transform(f.body, _pre_visit, _post_visit)
        return f.with_body(new_body)
    
    # Apply custom PrimFunc pass
    tvm.tir.transform.prim_func_pass(_ftransform, opt_level=0)(self.fn)
    return self
```

### Memory Copy Analysis

Memory traffic analysis handles global memory transfers:

```python
def _analyze_copy(self, call):
    """
    Analyze memory copy operations (e.g., tl.copy).
    
    Args:
        call: TVM Call node representing the copy operation
    """
    src_buffer = call.args[0].args[0].buffer
    dst_buffer = call.args[1].args[0].buffer
    
    # Determine if source or destination is global buffer
    if src_buffer in self.global_buffers:
        buffer_region = call.args[0]
    elif dst_buffer in self.global_buffers:
        buffer_region = call.args[1]
    else:
        return  # No global memory access
    
    # Calculate number of elements being copied
    elements = 1
    for r in range(2, len(buffer_region.args)):
        elements *= buffer_region.args[r]
    
    # Compute bytes transferred
    dtype_size = np.dtype(buffer_region.args[0].buffer.dtype).itemsize
    bytes_transferred = elements * dtype_size
    
    # Account for loop nesting and parallelism
    loop_product = 1
    for extent in self.loop_stack:
        loop_product *= extent.value if hasattr(extent, 'value') else extent
    
    total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
    total_bytes = bytes_transferred * loop_product * total_blocks
    
    self.total_global_bytes += total_bytes
```

### GEMM Operation Analysis

GEMM operations are analyzed for FLOP counting:

```python
def _analyze_gemm(self, call):
    """
    Analyze matrix multiplication (GEMM) operations.
    
    Args:
        call: TVM Call node representing the GEMM operation
    """
    # Extract GEMM dimensions from call arguments
    M = call.args[5].value
    N = call.args[6].value
    K = call.args[7].value
    
    # Standard GEMM FLOP formula: 2*M*N*K
    flops_per_call = 2 * M * N * K
    
    # Account for loop iterations and block parallelism
    loop_product = 1
    for extent in self.loop_stack:
        loop_product *= extent.value if hasattr(extent, 'value') else extent
    
    total_blocks = self.block_counts["blockIdx.x"] * self.block_counts["blockIdx.y"]
    total_flops = flops_per_call * loop_product * total_blocks
    
    self.total_flops += total_flops
```

## Architecture-Aware Performance Modeling

### Hardware Configuration Database

Performance characteristics are defined per architecture:

```python
# Configuration: (cores_per_SM, default_clock_GHz, flops_per_cycle, max_SM_count)
ARCH_CONFIGS = {
    "80": (128, 1.41, 2, 108),  # A100: 128 cores/SM, 1.41GHz, 2 FLOP/cycle, 108 SMs
    "86": (128, 1.70, 2, 84),   # RTX 3080: 128 cores/SM, 1.70GHz, 2 FLOP/cycle, 84 SMs  
    "89": (128, 2.52, 2, 128)   # RTX 4090: 128 cores/SM, 2.52GHz, 2 FLOP/cycle, 128 SMs
}
```

### Peak Performance Calculation

```python
def get_peak_tflops(device) -> Optional[float]:
    """
    Calculate theoretical peak TFLOPS for the target device.
    
    Args:
        device: Target device information
        
    Returns:
        float: Peak TFLOPS capability
    """
    arch_key = device.compute_capability[:2]
    if arch_key not in ARCH_CONFIGS:
        logger.info(f"Unsupported compute capability: {device.compute_capability}")
        return None
    
    cores_per_sm, default_clock, flops_per_cycle, compute_max_core = ARCH_CONFIGS[arch_key]
    
    # Total compute capability
    total_cores = compute_max_core * cores_per_sm
    tflops = (total_cores * default_clock * flops_per_cycle) / 1e3
    
    return round(tflops, 1)
```

## Roofline Performance Model

### Dual-Bound Analysis

The roofline model considers both compute and memory constraints:

```python
def calculate(self) -> AnalysisResult:
    """
    Calculate performance metrics using roofline model.
    
    Returns:
        AnalysisResult: Comprehensive performance analysis
    """
    
    # Get hardware capabilities
    bandwidth_GBps = self.device.bandwidth[1] / 1000  # Convert MB/s to GB/s
    peak_tflops = get_peak_tflops(self.device)
    
    # Compute-bound time estimate
    compute_time = None
    if peak_tflops:
        compute_time = self.total_flops / (peak_tflops * 1e12)
    
    # Memory-bound time estimate  
    mem_time = self.total_global_bytes / (bandwidth_GBps * 1e9)
    
    # Roofline model: performance limited by max of compute and memory bounds
    if compute_time is not None:
        estimated_time = max(mem_time, compute_time)
    else:
        estimated_time = mem_time
    
    return AnalysisResult(
        total_flops=self.total_flops,
        total_global_bytes=self.total_global_bytes,
        estimated_time=estimated_time,
        expected_tflops=peak_tflops,
        expected_bandwidth_GBps=bandwidth_GBps
    )
```

### Performance Bottleneck Identification

The model identifies whether workloads are compute-bound or memory-bound:

```python
def analyze_bottleneck(result: AnalysisResult) -> str:
    """
    Identify performance bottleneck from analysis results.
    """
    if result.expected_tflops is None:
        return "memory_bound"
    
    compute_time = result.total_flops / (result.expected_tflops * 1e12)
    memory_time = result.total_global_bytes / (result.expected_bandwidth_GBps * 1e9)
    
    if compute_time > memory_time * 1.1:  # 10% threshold
        return "compute_bound"
    elif memory_time > compute_time * 1.1:
        return "memory_bound"
    else:
        return "balanced"
```

## Integration with Scheduling Framework

### Performance-Guided Optimization

The analysis framework integrates with scheduling policies for optimization guidance:

```python
class PerformanceGuidedPolicy(DefaultPolicy):
    """
    Policy that uses performance analysis to guide optimization decisions.
    """
    
    def __init__(self, arch: TileDevice, tags: Optional[Dict] = None):
        super().__init__(arch, tags)
        self.analyzer = None
    
    def evaluate_configuration(self, hint: Hint) -> float:
        """
        Evaluate configuration using performance analysis.
        
        Args:
            hint: Configuration to evaluate
            
        Returns:
            float: Performance score (higher is better)
        """
        # Generate kernel with hint configuration
        kernel = self.generate_kernel(hint)
        
        # Analyze performance
        result = Analyzer.analysis(kernel, self.arch)
        
        # Compute performance score
        achieved_tflops = result.total_flops / (result.estimated_time * 1e12)
        utilization = achieved_tflops / result.expected_tflops if result.expected_tflops else 0
        
        return utilization
    
    def emit_config(self, topk: int) -> List[Hint]:
        """
        Generate configurations ranked by performance analysis.
        """
        base_configs = super().emit_config(topk * 3)  # Generate more candidates
        
        # Evaluate and rank configurations
        scored_configs = []
        for hint in base_configs:
            try:
                score = self.evaluate_configuration(hint)
                scored_configs.append((score, hint))
            except Exception as e:
                logger.warning(f"Failed to evaluate configuration: {e}")
        
        # Sort by score and return top-k
        scored_configs.sort(key=lambda x: x[0], reverse=True)
        return [hint for score, hint in scored_configs[:topk]]
```

## Advanced Analysis Features

### Memory Hierarchy Modeling

Extended analysis can model cache effects:

```python
class CacheAwareAnalyzer(Analyzer):
    """
    Extended analyzer with cache modeling capabilities.
    """
    
    def __init__(self, fn, device):
        super().__init__(fn, device)
        self.l1_cache_size = getattr(device, 'l1_cache_size_bytes', 64 * 1024)  # 64KB default
        self.l2_cache_size = getattr(device, 'l2_cache_size_bytes', 8 * 1024 * 1024)  # 8MB default
        self.l1_hits = 0
        self.l2_hits = 0
        self.l1_misses = 0
        self.l2_misses = 0
    
    def _analyze_cache_behavior(self, buffer_access):
        """
        Model cache hit/miss behavior for memory accesses.
        """
        access_size = buffer_access['size']
        access_pattern = buffer_access['pattern']  # sequential, random, etc.
        
        # Simple cache model based on access size and pattern
        if access_pattern == 'sequential' and access_size <= self.l1_cache_size:
            self.l1_hits += 1
        elif access_size <= self.l2_cache_size:
            self.l2_hits += 1
            self.l1_misses += 1
        else:
            self.l2_misses += 1
            self.l1_misses += 1
    
    def calculate_effective_bandwidth(self) -> float:
        """
        Calculate effective memory bandwidth considering cache effects.
        """
        total_accesses = self.l1_hits + self.l2_hits + self.l2_misses
        if total_accesses == 0:
            return self.device.bandwidth[1] / 1000
        
        # Weighted average based on cache hit rates and latencies
        l1_hit_rate = self.l1_hits / total_accesses
        l2_hit_rate = self.l2_hits / total_accesses
        miss_rate = self.l2_misses / total_accesses
        
        # Approximate latencies (cycles)
        l1_latency = 1
        l2_latency = 10
        memory_latency = 300
        
        effective_latency = (l1_hit_rate * l1_latency + 
                           l2_hit_rate * l2_latency + 
                           miss_rate * memory_latency)
        
        # Scale bandwidth by latency ratio
        baseline_latency = memory_latency
        bandwidth_scale = baseline_latency / effective_latency
        
        return (self.device.bandwidth[1] / 1000) * bandwidth_scale
```

### Workload Characterization

Analyze workload characteristics for optimization insights:

```python
def characterize_workload(result: AnalysisResult) -> Dict[str, float]:
    """
    Characterize workload based on analysis results.
    
    Returns:
        Dict with workload characteristics
    """
    characteristics = {}
    
    # Arithmetic intensity (FLOPs per byte)
    if result.total_global_bytes > 0:
        characteristics['arithmetic_intensity'] = result.total_flops / result.total_global_bytes
    else:
        characteristics['arithmetic_intensity'] = float('inf')
    
    # Compute utilization
    if result.expected_tflops and result.expected_tflops > 0:
        achieved_tflops = result.total_flops / (result.estimated_time * 1e12)
        characteristics['compute_utilization'] = achieved_tflops / result.expected_tflops
    else:
        characteristics['compute_utilization'] = 0.0
    
    # Memory utilization
    achieved_bandwidth = result.total_global_bytes / (result.estimated_time * 1e9)
    characteristics['memory_utilization'] = achieved_bandwidth / result.expected_bandwidth_GBps
    
    # Performance efficiency
    characteristics['performance_efficiency'] = min(
        characteristics['compute_utilization'],
        characteristics['memory_utilization']
    )
    
    return characteristics
```

## Validation and Benchmarking

### Analysis Accuracy Validation

Compare analysis predictions with actual measurements:

```python
class AnalysisValidator:
    """
    Validate analysis accuracy against empirical measurements.
    """
    
    def __init__(self, device):
        self.device = device
        self.measurements = []
    
    def validate_prediction(self, kernel, predicted_result: AnalysisResult) -> Dict[str, float]:
        """
        Validate prediction against measured performance.
        
        Returns:
            Dict with accuracy metrics
        """
        # Run actual kernel and measure performance
        measured_time = self.benchmark_kernel(kernel)
        measured_tflops = predicted_result.total_flops / (measured_time * 1e12)
        
        # Calculate prediction errors
        time_error = abs(predicted_result.estimated_time - measured_time) / measured_time
        tflops_error = abs(
            (predicted_result.total_flops / (predicted_result.estimated_time * 1e12)) - measured_tflops
        ) / measured_tflops
        
        validation_result = {
            'predicted_time': predicted_result.estimated_time,
            'measured_time': measured_time,
            'time_error': time_error,
            'predicted_tflops': predicted_result.total_flops / (predicted_result.estimated_time * 1e12),
            'measured_tflops': measured_tflops,
            'tflops_error': tflops_error
        }
        
        self.measurements.append(validation_result)
        return validation_result
    
    def benchmark_kernel(self, kernel) -> float:
        """
        Benchmark kernel execution time.
        """
        # Implementation would use actual GPU profiling
        # This is a simplified placeholder
        import time
        
        # Warm-up runs
        for _ in range(5):
            pass  # kernel execution
        
        # Timed runs
        start_time = time.time()
        for _ in range(10):
            pass  # kernel execution
        end_time = time.time()
        
        return (end_time - start_time) / 10
    
    def get_accuracy_statistics(self) -> Dict[str, float]:
        """
        Compute accuracy statistics across all measurements.
        """
        if not self.measurements:
            return {}
        
        time_errors = [m['time_error'] for m in self.measurements]
        tflops_errors = [m['tflops_error'] for m in self.measurements]
        
        return {
            'mean_time_error': np.mean(time_errors),
            'std_time_error': np.std(time_errors),
            'max_time_error': np.max(time_errors),
            'mean_tflops_error': np.mean(tflops_errors),
            'std_tflops_error': np.std(tflops_errors),
            'max_tflops_error': np.max(tflops_errors)
        }
```

## Usage Examples

### Basic Performance Analysis

```python
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA

# Define a simple GEMM kernel
M = N = K = 1024

@T.prim_func
def gemm_kernel(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((N, K), "float16"), 
    C: T.Tensor((M, N), "float16")
):
    # Kernel implementation
    pass

# Analyze performance
cuda_device = CUDA("cuda")
result = Analyzer.analysis(gemm_kernel, cuda_device)

print(f"Total FLOPs: {result.total_flops}")
print(f"Global Memory Traffic: {result.total_global_bytes} bytes")
print(f"Estimated Time: {result.estimated_time:.6f} seconds")
print(f"Theoretical Peak: {result.expected_tflops:.1f} TFLOPS")
print(f"Achieved Performance: {result.total_flops / (result.estimated_time * 1e12):.1f} TFLOPS")
```

### Comparative Analysis

```python
# Compare different kernel configurations
configurations = [
    {"block_M": 64, "block_N": 64, "block_K": 32},
    {"block_M": 128, "block_N": 128, "block_K": 32},
    {"block_M": 256, "block_N": 128, "block_K": 64}
]

results = []
for config in configurations:
    kernel = generate_kernel(**config)  # Generate kernel with configuration
    result = Analyzer.analysis(kernel, cuda_device)
    results.append((config, result))

# Rank by performance
results.sort(key=lambda x: x[1].total_flops / x[1].estimated_time, reverse=True)

print("Performance Ranking:")
for i, (config, result) in enumerate(results):
    tflops = result.total_flops / (result.estimated_time * 1e12)
    print(f"{i+1}. {config} -> {tflops:.1f} TFLOPS")
```

## Future Enhancements

### Planned Features

1. **Multi-Level Cache Modeling**: Detailed L1/L2/L3 cache behavior simulation
2. **Dynamic Analysis**: Runtime profiling integration for model calibration  
3. **Energy Modeling**: Power consumption analysis and optimization
4. **Cross-Architecture Prediction**: Transfer analysis results across devices

### Research Directions

1. **Machine Learning Integration**: ML-based performance prediction models
2. **Probabilistic Analysis**: Handle performance variability and uncertainty
3. **Multi-Objective Optimization**: Balance performance, power, and accuracy
4. **Real-Time Analysis**: Online performance monitoring and adaptation