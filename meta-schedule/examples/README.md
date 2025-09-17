# Examples and Implementation Patterns

This document provides practical examples and implementation patterns for using TileLang's meta-scheduling framework, cost model, and TensorCore support.

## Overview

This section demonstrates:
- **Basic Usage Patterns**: Getting started with the meta-scheduling framework
- **Performance Analysis Examples**: Using the cost model for optimization
- **TensorCore Implementation**: Mixed-precision computing examples
- **Custom Policy Development**: Extending the scheduling framework
- **Architecture-Specific Optimization**: Targeting different hardware platforms

## Basic Usage Examples

### 1. Simple GEMM Analysis

This example demonstrates basic performance analysis using the Analyzer:

```python
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA

# Define matrix dimensions
M = N = K = 1024

def create_gemm_kernel(block_M=128, block_N=128, block_K=32, num_stages=3, thread_num=128):
    """Create a GEMM kernel with specified tile sizes."""
    
    @T.prim_func
    def matmul(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            # Shared memory allocation
            A_shared = T.alloc_shared((block_M, block_K), "float16")
            B_shared = T.alloc_shared((block_N, block_K), "float16")
            C_local = T.alloc_fragment((block_M, block_N), "float16")
            C_shared = T.alloc_shared((block_M, block_N), "float16")
            
            # Clear accumulator
            T.clear(C_local)
            
            # Main computation loop with pipelining
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            
            # Write back results
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])
    
    return matmul

# Analyze performance
def analyze_gemm_performance():
    """Analyze GEMM performance with different configurations."""
    
    # Create CUDA device
    cuda_device = CUDA("cuda")
    
    # Test different configurations
    configurations = [
        {"block_M": 64, "block_N": 64, "block_K": 32},
        {"block_M": 128, "block_N": 128, "block_K": 32},
        {"block_M": 256, "block_N": 128, "block_K": 64},
    ]
    
    results = []
    for config in configurations:
        # Create kernel with configuration
        kernel = create_gemm_kernel(**config)
        
        # Analyze performance
        result = Analyzer.analysis(kernel, cuda_device)
        
        # Calculate metrics
        achieved_tflops = result.total_flops / (result.estimated_time * 1e12)
        efficiency = achieved_tflops / result.expected_tflops if result.expected_tflops else 0
        
        results.append({
            'config': config,
            'tflops': achieved_tflops,
            'efficiency': efficiency,
            'memory_gb': result.total_global_bytes / 1e9,
            'time_ms': result.estimated_time * 1000
        })
    
    # Sort by performance
    results.sort(key=lambda x: x['tflops'], reverse=True)
    
    print("GEMM Performance Analysis Results:")
    print("-" * 80)
    for i, r in enumerate(results):
        print(f"{i+1}. Config: {r['config']}")
        print(f"   Performance: {r['tflops']:.1f} TFLOPS ({r['efficiency']:.1%} efficiency)")
        print(f"   Memory: {r['memory_gb']:.2f} GB, Time: {r['time_ms']:.3f} ms")
        print()

if __name__ == "__main__":
    analyze_gemm_performance()
```

### 2. Convolution Analysis Example

```python
import tilelang.language as T
from tilelang.tools import Analyzer
from tilelang.carver.arch import CUDA
from tilelang.layout import make_swizzled_layout

def create_conv2d_kernel(N=64, C=256, H=512, W=512, F=512, K=3, 
                        block_M=64, block_N=128, block_K=32, num_stages=3, threads=256):
    """Create a 2D convolution kernel."""
    
    # Calculate output dimensions
    OH = H - K + 1  # Assuming no padding
    OW = W - K + 1
    
    @T.prim_func
    def conv2d(
        data: T.Tensor((N, H, W, C), "float16"),
        kernel: T.Tensor((K, K, C, F), "float16"),
        out: T.Tensor((N, OH, OW, F), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), 
            T.ceildiv(N * OH * OW, block_M),
            threads=threads
        ) as (bx, by):
            # Shared memory allocation
            data_shared = T.alloc_shared((block_M, block_K), "float16")
            kernel_shared = T.alloc_shared((block_K, block_N), "float16")
            out_local = T.alloc_fragment((block_M, block_N), "float16")
            out_shared = T.alloc_shared((block_M, block_N), "float16")
            
            # Flatten tensors for GEMM-like computation
            kernel_flat = T.Tensor((K * K * C, F), "float16", kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), "float16", out.data)
            
            # Apply swizzled layout for bank conflict avoidance
            T.annotate_layout({
                out_shared: make_swizzled_layout(out_shared),
                data_shared: make_swizzled_layout(data_shared),
                kernel_shared: make_swizzled_layout(kernel_shared),
            })
            
            T.clear(out_local)
            
            # Main computation loop
            for k_iter in T.Pipelined(T.ceildiv(K * K * C, block_K), num_stages=num_stages):
                # Im2col data loading with bounds checking
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i
                    
                    # Calculate input coordinates
                    n_idx = m // (OH * OW)
                    h_idx = (m % (OH * OW)) // OW
                    w_idx = m % OW
                    
                    k_h = k // (K * C)
                    k_w = (k // C) % K
                    k_c = k % C
                    
                    in_h = h_idx + k_h
                    in_w = w_idx + k_w
                    
                    # Bounds checking
                    in_bound = (in_h >= 0) and (in_w >= 0) and (in_h < H) and (in_w < W)
                    data_shared[i, j] = T.if_then_else(
                        in_bound, 
                        data[n_idx, in_h, in_w, k_c], 
                        0.0
                    )
                
                # Load kernel weights
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)
                
                # Perform GEMM
                T.gemm(data_shared, kernel_shared, out_local)
            
            # Write back results
            T.copy(out_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])
    
    return conv2d

def analyze_conv2d_performance():
    """Analyze convolution performance."""
    
    cuda_device = CUDA("cuda")
    
    # Create and analyze convolution kernel
    conv_kernel = create_conv2d_kernel()
    result = Analyzer.analysis(conv_kernel, cuda_device)
    
    print("Convolution Performance Analysis:")
    print(f"Total FLOPs: {result.total_flops:,}")
    print(f"Global Memory Traffic: {result.total_global_bytes / 1e9:.2f} GB")
    print(f"Estimated Time: {result.estimated_time * 1000:.3f} ms")
    print(f"Achieved Performance: {result.total_flops / (result.estimated_time * 1e12):.1f} TFLOPS")
    
    # Calculate convolution-specific metrics
    N, C, H, W, F, K = 64, 256, 512, 512, 512, 3
    OH, OW = H - K + 1, W - K + 1
    theoretical_flops = 2 * N * OH * OW * F * K * K * C
    
    print(f"Theoretical FLOPs: {theoretical_flops:,}")
    print(f"FLOP Analysis Accuracy: {result.total_flops / theoretical_flops:.3f}")

if __name__ == "__main__":
    analyze_conv2d_performance()
```

## TensorCore Implementation Examples

### 3. Mixed-Precision GEMM with TensorCores

```python
from tilelang.carver.roller.policy import TensorCorePolicy
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.hint import IntrinInfo

def create_tensorcore_gemm_policy():
    """Create a TensorCore-optimized GEMM policy."""
    
    # Create CUDA architecture with TensorCore support
    arch = CUDA("cuda")  # Assumes SM 8.0+ for TensorCore support
    
    # Create TensorCore policy
    policy = TensorCorePolicy(arch)
    
    # Configure TensorCore-specific settings
    policy.wmma_k = 16  # WMMA K dimension
    policy.pipeline_stage = 3  # Multi-stage pipeline
    policy.use_async_copy = True  # Asynchronous memory copy
    
    return policy

def generate_tensorcore_configurations():
    """Generate TensorCore-optimized configurations."""
    
    # Create policy
    policy = create_tensorcore_gemm_policy()
    
    # Generate configurations
    configs = policy.emit_config(topk=10)
    
    print("TensorCore GEMM Configurations:")
    print("-" * 50)
    
    for i, config in enumerate(configs):
        config_dict = config.to_dict()
        print(f"Configuration {i+1}:")
        print(f"  Block: {config_dict.get('block', [])}")
        print(f"  Warp: {config_dict.get('warp', [])}")
        print(f"  Reduce Step: {config_dict.get('rstep', [])}")
        print(f"  Use TensorCore: {config_dict.get('use_tc', False)}")
        print(f"  Pipeline Stages: {config_dict.get('pipeline_stage', 1)}")
        print(f"  Async Copy: {config_dict.get('use_async', False)}")
        print()
    
    return configs

def create_mixed_precision_kernel():
    """Create a mixed-precision GEMM kernel using TensorCores."""
    
    @T.prim_func  
    def mixed_precision_gemm(
        A: T.Tensor((1024, 1024), "float16"),
        B: T.Tensor((1024, 1024), "float16"),
        C: T.Tensor((1024, 1024), "float32"),  # FP32 accumulation
    ):
        with T.Kernel(T.ceildiv(1024, 128), T.ceildiv(1024, 128), threads=256) as (bx, by):
            # TensorCore-optimized shared memory layout
            A_shared = T.alloc_shared((128, 32), "float16")
            B_shared = T.alloc_shared((128, 32), "float16")
            C_local = T.alloc_fragment((128, 128), "float32")
            
            # Configure TensorCore intrinsic
            T.use_tensorcore(
                in_dtype="float16",
                out_dtype="float32",
                shape=[16, 16, 16]  # MMA shape
            )
            
            T.clear(C_local)
            
            # Multi-stage pipeline
            for k in T.Pipelined(T.ceildiv(1024, 32), num_stages=3):
                # Async copy for better latency hiding
                T.async_copy(A[by * 128, k * 32], A_shared)
                T.async_copy(B[bx * 128, k * 32], B_shared)
                T.wait_group(0)  # Wait for copies to complete
                
                # TensorCore GEMM
                T.mma(A_shared, B_shared, C_local)
            
            # Write back with type conversion
            T.copy(C_local, C[by * 128, bx * 128])
    
    return mixed_precision_gemm

if __name__ == "__main__":
    generate_tensorcore_configurations()
```

### 4. Custom Precision Support

```python
def create_int8_tensorcore_kernel():
    """Create an INT8 TensorCore kernel for quantized inference."""
    
    @T.prim_func
    def int8_gemm(
        A: T.Tensor((1024, 1024), "int8"),
        B: T.Tensor((1024, 1024), "int8"),
        C: T.Tensor((1024, 1024), "int32"),  # INT32 accumulation
        scale_A: T.Tensor((), "float32"),    # Quantization scale for A
        scale_B: T.Tensor((), "float32"),    # Quantization scale for B
    ):
        with T.Kernel(T.ceildiv(1024, 128), T.ceildiv(1024, 128), threads=256) as (bx, by):
            A_shared = T.alloc_shared((128, 32), "int8")
            B_shared = T.alloc_shared((128, 32), "int8")
            C_local = T.alloc_fragment((128, 128), "int32")
            
            # Configure INT8 TensorCore
            T.use_tensorcore(
                in_dtype="int8",
                out_dtype="int32",
                shape=[16, 16, 16]
            )
            
            T.clear(C_local)
            
            for k in T.Pipelined(T.ceildiv(1024, 32), num_stages=2):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[bx * 128, k * 32], B_shared)
                
                # INT8 TensorCore operation
                T.mma(A_shared, B_shared, C_local)
            
            # Dequantize and write back
            for i, j in T.Parallel(128, 128):
                dequantized = T.cast(C_local[i, j], "float32") * scale_A[()] * scale_B[()]
                C[by * 128 + i, bx * 128 + j] = T.cast(dequantized, "int32")
    
    return int8_gemm
```

## Custom Policy Development

### 5. Flash Attention Policy

```python
from tilelang.carver.roller.policy.default import DefaultPolicy
from tilelang.carver.roller.hint import Hint

class FlashAttentionPolicy(DefaultPolicy):
    """
    Custom policy for Flash Attention optimization.
    """
    
    def __init__(self, arch, tags=None):
        super().__init__(arch, tags)
        self.attention_specific_params = {
            'causal': tags.get('causal', False),
            'head_dim': tags.get('head_dim', 64),
            'block_q': tags.get('block_q', 64),
            'block_k': tags.get('block_k', 64),
        }
    
    def _assign_reduce_step(self, node):
        """Custom reduce step assignment for attention patterns."""
        
        # For attention, we want to process in blocks along sequence dimension
        if node.get_tag("attention_qk"):
            # QK^T computation - optimize for sequence length blocking
            seq_len = self.attention_specific_params['block_k']
            return {'k': min(seq_len, 32)}  # Process 32 elements at a time
            
        elif node.get_tag("attention_pv"):
            # PV computation - optimize for head dimension
            head_dim = self.attention_specific_params['head_dim']
            return {'d': min(head_dim, 16)}  # Process 16 dims at a time
            
        else:
            return super()._assign_reduce_step(node)
    
    def check_tile_shape_isvalid(self, td):
        """Custom validation for attention tile shapes."""
        
        # Check that tile sizes respect attention constraints
        for node in self.ordered_nodes:
            tile = td.get_tile(node)
            
            if node.get_tag("attention_qk"):
                # QK^T tiles should fit in shared memory for softmax
                q_block, k_block = tile[0], tile[1]
                softmax_memory = q_block * k_block * 4  # FP32 for intermediate
                
                if softmax_memory > self.arch.smem_cap // 4:  # Reserve 3/4 for other tensors
                    return False
        
        return super().check_tile_shape_isvalid(td)
    
    def emit_config(self, topk: int):
        """Generate Flash Attention specific configurations."""
        
        # Override base configuration generation
        configs = []
        
        # Generate configurations optimized for attention patterns
        q_blocks = [32, 64, 128]
        k_blocks = [32, 64, 128]
        
        for q_block in q_blocks:
            for k_block in k_blocks:
                if q_block * k_block * 4 <= self.arch.smem_cap // 4:  # Memory constraint
                    hint = Hint()
                    hint.block = [q_block, k_block]
                    hint.thread = [16, 8]  # 128 threads total
                    hint.rstep = [min(self.attention_specific_params['head_dim'], 16)]
                    hint.arch = self.arch
                    hint.use_tc = has_mma_support(self.arch)
                    
                    # Flash Attention specific configurations
                    hint.opt_shapes = {
                        'Q_block': q_block,
                        'K_block': k_block,
                        'causal': self.attention_specific_params['causal']
                    }
                    
                    configs.append(hint)
                    
                    if len(configs) >= topk:
                        break
            
            if len(configs) >= topk:
                break
        
        return configs[:topk]

def create_flash_attention_kernel():
    """Create a Flash Attention kernel with custom policy."""
    
    def flash_attention(seq_len=2048, head_dim=64, num_heads=16):
        
        @T.prim_func
        def flash_attn(
            Q: T.Tensor((seq_len, num_heads, head_dim), "float16"),
            K: T.Tensor((seq_len, num_heads, head_dim), "float16"),
            V: T.Tensor((seq_len, num_heads, head_dim), "float16"),
            O: T.Tensor((seq_len, num_heads, head_dim), "float16"),
        ):
            # Flash Attention implementation with tiling and online softmax
            for h in T.Parallel(num_heads):
                # Process attention head h
                with T.Kernel(T.ceildiv(seq_len, 64), threads=128) as (bx,):
                    Q_shared = T.alloc_shared((64, head_dim), "float16")
                    K_shared = T.alloc_shared((64, head_dim), "float16")
                    V_shared = T.alloc_shared((64, head_dim), "float16")
                    
                    S_local = T.alloc_fragment((64, 64), "float32")  # Attention scores
                    O_local = T.alloc_fragment((64, head_dim), "float32")  # Output accumulator
                    
                    # Load Q block
                    T.copy(Q[bx * 64:(bx + 1) * 64, h, :], Q_shared)
                    T.clear(O_local)
                    
                    max_val = T.alloc_fragment((64,), "float32")
                    sum_val = T.alloc_fragment((64,), "float32")
                    T.fill(max_val, -float('inf'))
                    T.fill(sum_val, 0.0)
                    
                    # Process K, V blocks for online softmax
                    for k_block in range(T.ceildiv(seq_len, 64)):
                        # Load K, V blocks
                        T.copy(K[k_block * 64:(k_block + 1) * 64, h, :], K_shared)
                        T.copy(V[k_block * 64:(k_block + 1) * 64, h, :], V_shared)
                        
                        # Compute QK^T
                        T.gemm(Q_shared, K_shared, S_local, transpose_B=True)
                        
                        # Online softmax update
                        for i in T.Parallel(64):
                            # Find max for numerical stability
                            row_max = T.max(S_local[i, :])
                            old_max = max_val[i]
                            new_max = T.max(old_max, row_max)
                            
                            # Update normalizer
                            alpha = T.exp(old_max - new_max)
                            beta = T.exp(row_max - new_max)
                            
                            new_sum = alpha * sum_val[i] + beta * T.sum(T.exp(S_local[i, :] - row_max))
                            
                            # Update output
                            for d in T.Parallel(head_dim):
                                O_local[i, d] = alpha * O_local[i, d] + beta * T.sum(
                                    T.exp(S_local[i, :] - row_max) * V_shared[:, d]
                                )
                            
                            max_val[i] = new_max
                            sum_val[i] = new_sum
                    
                    # Final normalization and output
                    for i, d in T.Parallel(64, head_dim):
                        O[bx * 64 + i, h, d] = T.cast(O_local[i, d] / sum_val[i], "float16")
        
        return flash_attn
    
    return flash_attention()
```

## Architecture-Specific Examples

### 6. Multi-Architecture Kernel Generation

```python
from tilelang.carver.arch import CUDA, CDNA, CPU

def create_multi_arch_gemm():
    """Create GEMM kernels optimized for different architectures."""
    
    def get_arch_specific_config(arch):
        """Get architecture-specific configuration."""
        
        if isinstance(arch, CUDA):
            if arch.sm_version >= 80:  # Ampere+
                return {
                    'block_size': [128, 128, 32],
                    'thread_config': [16, 16],
                    'use_tensorcore': True,
                    'pipeline_stages': 3,
                    'async_copy': True
                }
            else:  # Volta/Turing
                return {
                    'block_size': [64, 64, 16],
                    'thread_config': [8, 8],
                    'use_tensorcore': arch.sm_version >= 70,
                    'pipeline_stages': 2,
                    'async_copy': False
                }
                
        elif isinstance(arch, CDNA):
            return {
                'block_size': [128, 128, 16],
                'thread_config': [16, 16],  # Adapted for wavefront size 64
                'use_tensorcore': True,  # Use MFMA
                'pipeline_stages': 2,
                'async_copy': True
            }
            
        elif isinstance(arch, CPU):
            return {
                'block_size': [32, 32, 32],
                'thread_config': [1, 1],  # No SIMT
                'use_tensorcore': False,
                'pipeline_stages': 1,
                'async_copy': False
            }
    
    def generate_kernel(arch):
        """Generate architecture-specific kernel."""
        config = get_arch_specific_config(arch)
        
        block_M, block_N, block_K = config['block_size']
        thread_M, thread_N = config['thread_config']
        
        @T.prim_func
        def arch_optimized_gemm(
            A: T.Tensor((1024, 1024), "float16"),
            B: T.Tensor((1024, 1024), "float16"),
            C: T.Tensor((1024, 1024), "float16"),
        ):
            with T.Kernel(
                T.ceildiv(1024, block_N), 
                T.ceildiv(1024, block_M), 
                threads=thread_M * thread_N * (arch.warp_size if hasattr(arch, 'warp_size') else 1)
            ) as (bx, by):
                
                A_shared = T.alloc_shared((block_M, block_K), "float16")
                B_shared = T.alloc_shared((block_N, block_K), "float16")
                C_local = T.alloc_fragment((block_M, block_N), "float16")
                
                if config['use_tensorcore']:
                    # Configure TensorCore/MFMA
                    if isinstance(arch, CUDA):
                        T.use_tensorcore(in_dtype="float16", out_dtype="float16")
                    elif isinstance(arch, CDNA):
                        T.use_mfma(in_dtype="float16", out_dtype="float16")
                
                T.clear(C_local)
                
                # Architecture-specific memory and compute pattern
                if config['pipeline_stages'] > 1:
                    for k in T.Pipelined(T.ceildiv(1024, block_K), num_stages=config['pipeline_stages']):
                        if config['async_copy']:
                            T.async_copy(A[by * block_M, k * block_K], A_shared)
                            T.async_copy(B[bx * block_N, k * block_K], B_shared)
                            T.wait_group(0)
                        else:
                            T.copy(A[by * block_M, k * block_K], A_shared)
                            T.copy(B[bx * block_N, k * block_K], B_shared)
                        
                        if config['use_tensorcore']:
                            T.mma(A_shared, B_shared, C_local)
                        else:
                            T.gemm(A_shared, B_shared, C_local)
                else:
                    # Simple non-pipelined version for CPU
                    for k in range(T.ceildiv(1024, block_K)):
                        T.copy(A[by * block_M, k * block_K], A_shared)
                        T.copy(B[bx * block_N, k * block_K], B_shared)
                        T.gemm(A_shared, B_shared, C_local)
                
                T.copy(C_local, C[by * block_M, bx * block_N])
        
        return arch_optimized_gemm
    
    # Generate kernels for different architectures
    architectures = [
        CUDA("sm_80"),  # A100
        CUDA("sm_89"),  # RTX 4090
        CDNA("gfx90a"), # MI250X
        CPU("x86_64"),  # x86 CPU
    ]
    
    kernels = {}
    for arch in architectures:
        try:
            kernels[arch.platform] = generate_kernel(arch)
            print(f"Generated kernel for {arch.platform}")
        except Exception as e:
            print(f"Failed to generate kernel for {arch.platform}: {e}")
    
    return kernels

if __name__ == "__main__":
    multi_arch_kernels = create_multi_arch_gemm()
```

## Performance Optimization Patterns

### 7. Iterative Optimization Workflow

```python
class PerformanceOptimizer:
    """
    Iterative performance optimization workflow.
    """
    
    def __init__(self, arch, operation_type="gemm"):
        self.arch = arch
        self.operation_type = operation_type
        self.optimization_history = []
    
    def optimize_configuration(self, initial_config, target_metric="tflops", iterations=10):
        """
        Iteratively optimize configuration for target metric.
        
        Args:
            initial_config: Starting configuration
            target_metric: Optimization target (tflops, efficiency, etc.)
            iterations: Number of optimization iterations
        """
        
        current_config = initial_config.copy()
        best_config = current_config.copy()
        best_score = 0
        
        for iteration in range(iterations):
            print(f"Optimization Iteration {iteration + 1}/{iterations}")
            
            # Generate configuration variants
            variants = self._generate_variants(current_config)
            
            # Evaluate each variant
            variant_scores = []
            for variant in variants:
                try:
                    score = self._evaluate_config(variant, target_metric)
                    variant_scores.append((score, variant))
                except Exception as e:
                    print(f"Failed to evaluate variant: {e}")
                    variant_scores.append((0, variant))
            
            # Select best variant
            variant_scores.sort(key=lambda x: x[0], reverse=True)
            if variant_scores and variant_scores[0][0] > best_score:
                best_score, best_config = variant_scores[0]
                current_config = best_config.copy()
                
                print(f"  New best score: {best_score:.3f}")
                self.optimization_history.append({
                    'iteration': iteration,
                    'config': best_config,
                    'score': best_score
                })
            else:
                print(f"  No improvement found")
        
        return best_config, best_score
    
    def _generate_variants(self, config):
        """Generate configuration variants for exploration."""
        variants = []
        
        # Vary block sizes
        block_factors = [0.5, 0.75, 1.25, 1.5]
        for factor in block_factors:
            variant = config.copy()
            if 'block_M' in variant:
                variant['block_M'] = max(16, int(variant['block_M'] * factor))
                variant['block_N'] = max(16, int(variant['block_N'] * factor))
                variants.append(variant)
        
        # Vary thread configuration
        if 'thread_num' in config:
            for thread_num in [64, 128, 256, 512]:
                variant = config.copy()
                variant['thread_num'] = thread_num
                variants.append(variant)
        
        # Vary pipeline stages
        if 'num_stages' in config:
            for stages in [1, 2, 3, 4]:
                variant = config.copy()
                variant['num_stages'] = stages
                variants.append(variant)
        
        return variants
    
    def _evaluate_config(self, config, target_metric):
        """Evaluate configuration using specified metric."""
        
        # Create kernel with configuration
        if self.operation_type == "gemm":
            kernel = create_gemm_kernel(**config)
        else:
            raise ValueError(f"Unsupported operation type: {self.operation_type}")
        
        # Analyze performance
        result = Analyzer.analysis(kernel, self.arch)
        
        # Calculate target metric
        if target_metric == "tflops":
            return result.total_flops / (result.estimated_time * 1e12)
        elif target_metric == "efficiency":
            tflops = result.total_flops / (result.estimated_time * 1e12)
            return tflops / result.expected_tflops if result.expected_tflops else 0
        elif target_metric == "bandwidth":
            return result.total_global_bytes / (result.estimated_time * 1e9)
        else:
            raise ValueError(f"Unsupported target metric: {target_metric}")
    
    def plot_optimization_history(self):
        """Plot optimization progress over iterations."""
        if not self.optimization_history:
            print("No optimization history to plot")
            return
        
        import matplotlib.pyplot as plt
        
        iterations = [h['iteration'] for h in self.optimization_history]
        scores = [h['score'] for h in self.optimization_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, scores, 'b-o')
        plt.xlabel('Iteration')
        plt.ylabel('Performance Score')
        plt.title('Optimization Progress')
        plt.grid(True)
        plt.show()

# Example usage
def run_optimization_example():
    """Run iterative optimization example."""
    
    arch = CUDA("cuda")
    optimizer = PerformanceOptimizer(arch, "gemm")
    
    # Initial configuration
    initial_config = {
        'block_M': 128,
        'block_N': 128,
        'block_K': 32,
        'num_stages': 3,
        'thread_num': 128
    }
    
    # Optimize for TFLOPS
    best_config, best_score = optimizer.optimize_configuration(
        initial_config, 
        target_metric="tflops", 
        iterations=5
    )
    
    print(f"\nOptimization Results:")
    print(f"Best Configuration: {best_config}")
    print(f"Best Score: {best_score:.3f} TFLOPS")
    
    # Plot progress
    optimizer.plot_optimization_history()

if __name__ == "__main__":
    run_optimization_example()
```

## Integration Examples

### 8. End-to-End Workflow

```python
def complete_optimization_workflow():
    """Demonstrate complete optimization workflow from analysis to deployment."""
    
    print("TileLang Meta-Schedule Complete Workflow")
    print("=" * 50)
    
    # Step 1: Architecture Setup
    print("1. Setting up target architecture...")
    arch = CUDA("cuda")
    print(f"   Target: {arch.platform} {arch.compute_capability}")
    print(f"   Compute Cores: {arch.compute_max_core}")
    print(f"   Shared Memory: {arch.smem_cap / 1024:.0f} KB")
    print()
    
    # Step 2: Policy Configuration
    print("2. Configuring scheduling policy...")
    if has_mma_support(arch):
        policy = TensorCorePolicy(arch)
        print("   Using TensorCore-optimized policy")
    else:
        policy = DefaultPolicy(arch)
        print("   Using default policy")
    print()
    
    # Step 3: Configuration Generation
    print("3. Generating optimized configurations...")
    configs = policy.emit_config(topk=5)
    print(f"   Generated {len(configs)} configurations")
    print()
    
    # Step 4: Performance Analysis
    print("4. Analyzing configuration performance...")
    results = []
    for i, config in enumerate(configs):
        # Create kernel with configuration
        kernel = create_optimized_kernel(config)
        
        # Analyze performance
        result = Analyzer.analysis(kernel, arch)
        
        # Calculate metrics
        tflops = result.total_flops / (result.estimated_time * 1e12)
        efficiency = tflops / result.expected_tflops if result.expected_tflops else 0
        
        results.append({
            'config_id': i,
            'config': config.to_dict(),
            'tflops': tflops,
            'efficiency': efficiency,
            'memory_gb': result.total_global_bytes / 1e9,
            'result': result
        })
        
        print(f"   Config {i+1}: {tflops:.1f} TFLOPS ({efficiency:.1%} efficiency)")
    
    print()
    
    # Step 5: Configuration Selection
    print("5. Selecting optimal configuration...")
    best_result = max(results, key=lambda x: x['efficiency'])
    print(f"   Selected Config {best_result['config_id'] + 1}")
    print(f"   Performance: {best_result['tflops']:.1f} TFLOPS")
    print(f"   Efficiency: {best_result['efficiency']:.1%}")
    print()
    
    # Step 6: Code Generation
    print("6. Generating optimized code...")
    optimized_kernel = create_optimized_kernel(configs[best_result['config_id']])
    print("   Kernel generated successfully")
    print()
    
    # Step 7: Validation
    print("7. Validating performance prediction...")
    predicted_time = best_result['result'].estimated_time
    print(f"   Predicted execution time: {predicted_time * 1000:.3f} ms")
    print("   (Actual measurement would require GPU execution)")
    print()
    
    print("Workflow completed successfully!")
    return best_result

def create_optimized_kernel(config):
    """Create kernel from configuration hint."""
    config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
    
    # Extract configuration parameters
    block = config_dict.get('block', [128, 128])
    rstep = config_dict.get('rstep', [32])
    use_tc = config_dict.get('use_tc', False)
    pipeline_stages = config_dict.get('pipeline_stage', 3)
    
    return create_gemm_kernel(
        block_M=block[0],
        block_N=block[1] if len(block) > 1 else block[0],
        block_K=rstep[0] if rstep else 32,
        num_stages=pipeline_stages
    )

if __name__ == "__main__":
    complete_optimization_workflow()
```

These examples demonstrate the comprehensive capabilities of TileLang's meta-scheduling framework, from basic performance analysis to advanced TensorCore optimization and custom policy development. The framework provides the tools needed for systematic kernel optimization across different architectures and workloads.