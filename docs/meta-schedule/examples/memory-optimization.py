#!/usr/bin/env python3
"""
Memory Optimization Example

This example demonstrates advanced memory optimization techniques using TileLang's
memory management system, including shared memory allocation, stride optimization,
and memory traffic analysis.
"""

import numpy as np
import tilelang
import tilelang.language as T
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.policy import DefaultPolicy, TensorCorePolicy
from tilelang.carver.roller.bestfit import BestFit, Block
from tilelang.carver.roller.hint import Stride


def create_memory_intensive_kernel(M, N, K):
    """Create a kernel with complex memory access patterns"""
    @T.prim_func
    def complex_kernel(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
        D: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
            # Multiple shared memory allocations
            A_shared = T.alloc_shared((64, 32), "float16")
            B_shared = T.alloc_shared((32, 64), "float16")
            C_shared = T.alloc_shared((64, 64), "float16")
            
            # Fragment storage
            C_local = T.alloc_fragment((64, 64), "float")
            temp_local = T.alloc_fragment((64, 64), "float16")
            
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                T.copy(A[by * 64, k * 32], A_shared)
                T.copy(B[k * 32, bx * 64], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            # Additional operations requiring more memory
            T.copy(C_local, C_shared)
            T.copy(C_shared, temp_local)
            T.copy(temp_local, C[by * 64, bx * 64])
            T.copy(C[by * 64, bx * 64], D[by * 64, bx * 64])
    
    return complex_kernel


def demonstrate_bestfit_allocator():
    """Demonstrate BestFit memory allocator functionality"""
    print("=== BestFit Memory Allocator Demonstration ===")
    
    # Create allocator with 32-byte alignment
    allocator = BestFit(align=32)
    
    print("Simulating memory allocation patterns:")
    print(f"{'Operation':<15} {'Size (bytes)':<12} {'Address Range':<15} {'Total Used':<12}")
    print("-" * 65)
    
    # Simulate allocation pattern
    allocations = []
    
    # Allocate A_shared (64*32*2 = 4KB)
    block_a = allocator.malloc(4096)
    allocations.append(("A_shared", block_a))
    print(f"{'Alloc A_shared':<15} {4096:<12} {block_a.start}-{block_a.end:<6} {allocator.limit:<12}")
    
    # Allocate B_shared (32*64*2 = 4KB)
    block_b = allocator.malloc(4096)
    allocations.append(("B_shared", block_b))
    print(f"{'Alloc B_shared':<15} {4096:<12} {block_b.start}-{block_b.end:<6} {allocator.limit:<12}")
    
    # Allocate C_shared (64*64*2 = 8KB)
    block_c = allocator.malloc(8192)
    allocations.append(("C_shared", block_c))
    print(f"{'Alloc C_shared':<15} {8192:<12} {block_c.start}-{block_c.end:<6} {allocator.limit:<12}")
    
    # Free A_shared (simulate end of usage)
    allocator.free(block_a)
    print(f"{'Free A_shared':<15} {'-':<12} {'freed':<15} {allocator.limit:<12}")
    
    # Allocate smaller buffer that fits in freed space
    block_small = allocator.malloc(2048)
    print(f"{'Alloc small':<15} {2048:<12} {block_small.start}-{block_small.end:<6} {allocator.limit:<12}")
    
    print(f"\nMemory efficiency: {((block_b.size() + block_c.size() + block_small.size()) / allocator.limit) * 100:.1f}%")
    print()


def analyze_memory_stride_patterns():
    """Analyze memory stride optimization for different access patterns"""
    print("=== Memory Stride Pattern Analysis ===")
    
    # Test different stride configurations
    stride_configs = [
        (Stride(stride=1, ax=-1), "No stride (standard layout)"),
        (Stride(stride=65, ax=0), "Row-major stride + 1"),
        (Stride(stride=72, ax=0), "Row-major stride + 8"),
        (Stride(stride=80, ax=0), "Row-major stride + 16"),
    ]
    
    matrix_shape = [64, 64]
    
    print(f"{'Configuration':<25} {'Elements':<10} {'Wasted Bytes':<12} {'Bank Conflicts':<15}")
    print("-" * 70)
    
    for stride, desc in stride_configs:
        elements = stride.compute_elements_from_shape(matrix_shape)
        standard_elements = np.prod(matrix_shape)
        wasted_bytes = (elements - standard_elements) * 2  # Assuming float16
        
        # Simplified bank conflict analysis
        if stride.stride % 32 == 0:
            bank_conflicts = "High"
        elif stride.stride % 8 == 1:
            bank_conflicts = "Low"
        else:
            bank_conflicts = "Medium"
        
        print(f"{desc:<25} {elements:<10} {wasted_bytes:<12} {bank_conflicts:<15}")
    
    print()


def memory_traffic_optimization():
    """Demonstrate memory traffic optimization strategies"""
    print("=== Memory Traffic Optimization ===")
    
    arch = CUDA("cuda -arch=sm_80")
    kernel_func = create_memory_intensive_kernel(1024, 1024, 1024)
    policy = DefaultPolicy.from_prim_func(kernel_func, arch)
    
    # Test different tile configurations for memory efficiency
    tile_strategies = [
        ([64, 64], "Square tiles"),
        ([128, 64], "Rectangular (M-heavy)"),
        ([64, 128], "Rectangular (N-heavy)"),
        ([256, 32], "Extreme M-heavy"),
        ([32, 256], "Extreme N-heavy"),
    ]
    
    print(f"{'Strategy':<20} {'Tile Size':<12} {'Traffic (MB)':<12} {'Efficiency':<12}")
    print("-" * 60)
    
    baseline_traffic = None
    
    for tile, desc in tile_strategies:
        traffic, _ = policy._compute_memory_traffic(tile)
        traffic_mb = traffic / (1024 * 1024)
        
        if baseline_traffic is None:
            baseline_traffic = traffic
            efficiency = 1.0
        else:
            efficiency = baseline_traffic / traffic
        
        print(f"{desc:<20} {str(tile):<12} {traffic_mb:<12.2f} {efficiency:<12.2f}")
    
    print()


def shared_memory_optimization():
    """Demonstrate shared memory optimization techniques"""
    print("=== Shared Memory Optimization ===")
    
    arch = CUDA("cuda -arch=sm_80")
    
    # Test with and without TensorCore optimizations
    policies = [
        (DefaultPolicy, "Default Policy"),
        (TensorCorePolicy, "TensorCore Policy"),
    ]
    
    print(f"{'Policy':<20} {'Shared Mem (KB)':<15} {'Pipeline Factor':<15} {'Efficiency':<12}")
    print("-" * 70)
    
    for policy_class, desc in policies:
        kernel_func = create_memory_intensive_kernel(1024, 1024, 1024)
        
        if policy_class == TensorCorePolicy:
            tags = {"tensorcore_config": (0, 1), "pipeline_stage": 2}
            policy = policy_class.from_prim_func(kernel_func, arch, tags=tags)
        else:
            policy = policy_class.from_prim_func(kernel_func, arch)
        
        # Analyze shared memory usage
        tile = [128, 128]
        rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}
        td = policy.compute_tile_dict(tile, rstep_map)
        
        smem_kb = td.smem_cost / 1024
        pipeline_factor = getattr(policy, 'pipeline_stage', 1)
        efficiency = (td.smem_cost / arch.smem_cap) * 100
        
        print(f"{desc:<20} {smem_kb:<15.1f} {pipeline_factor:<15} {efficiency:<12.1f}%")
    
    print()


def memory_coalescing_analysis():
    """Analyze memory coalescing patterns"""
    print("=== Memory Coalescing Analysis ===")
    
    from tilelang.carver.roller.policy.common import coalesced_factor
    
    # Test different access patterns
    access_patterns = [
        ([64, 64], [1024, 1024], "Sequential access"),
        ([64, 1], [1024, 1024], "Strided access"),
        ([1, 64], [1024, 1024], "Column-wise access"),
        ([32, 32], [1024, 1024], "Small tiles"),
        ([256, 256], [1024, 1024], "Large tiles"),
    ]
    
    print(f"{'Pattern':<20} {'Tile Shape':<12} {'Tensor Shape':<15} {'Coalescing Score':<15}")
    print("-" * 70)
    
    for tile_shape, tensor_shape, desc in access_patterns:
        try:
            score = coalesced_factor(tile_shape, tensor_shape)
            print(f"{desc:<20} {str(tile_shape):<12} {str(tensor_shape):<15} {score:<15.3f}")
        except:
            print(f"{desc:<20} {str(tile_shape):<12} {str(tensor_shape):<15} {'N/A':<15}")
    
    print()


def memory_bandwidth_utilization():
    """Analyze memory bandwidth utilization"""
    print("=== Memory Bandwidth Utilization Analysis ===")
    
    arch = CUDA("cuda -arch=sm_80")
    kernel_func = create_memory_intensive_kernel(1024, 1024, 1024)
    policy = DefaultPolicy.from_prim_func(kernel_func, arch)
    
    # Test different reduction step sizes impact on bandwidth
    reduction_steps = [16, 32, 64, 128]
    tile = [128, 128]
    
    print(f"{'Reduction Step':<15} {'Memory Traffic':<15} {'Bandwidth Util %':<18} {'Shared Mem %':<15}")
    print("-" * 70)
    
    for rstep in reduction_steps:
        rstep_map = {node: {"k": rstep} for node in policy.ordered_nodes}
        
        # Compute memory metrics
        traffic, _ = policy._compute_memory_traffic(tile)
        td = policy.compute_tile_dict(tile, rstep_map)
        
        # Estimate bandwidth utilization (simplified)
        theoretical_bandwidth = arch.bandwidth[0] * 1024 * 1024  # Convert MB/s to bytes/s
        # Assume 1ms kernel execution time for calculation
        bandwidth_util = min((traffic / 0.001) / theoretical_bandwidth * 100, 100)
        
        smem_util = (td.smem_cost / arch.smem_cap) * 100
        
        print(f"{rstep:<15} {traffic/(1024*1024):<15.1f} {bandwidth_util:<18.1f} {smem_util:<15.1f}")
    
    print()


def advanced_memory_techniques():
    """Demonstrate advanced memory optimization techniques"""
    print("=== Advanced Memory Optimization Techniques ===")
    
    print("1. Dynamic Shared Memory Allocation:")
    print("   - Automatically switches to dynamic shared memory for large allocations")
    print("   - Enables kernels that exceed static shared memory limits")
    print("   - Configured via shared_scope parameter")
    print()
    
    print("2. Memory Bank Conflict Avoidance:")
    print("   - Applies padding to memory strides")
    print("   - Uses offset calculations for optimal access patterns") 
    print("   - Particularly important for TensorCore operations")
    print()
    
    print("3. Memory Reuse Optimization:")
    print("   - Tracks tensor lifetimes for optimal memory reuse")
    print("   - Uses best-fit allocation to minimize fragmentation")
    print("   - Frees memory as soon as tensors are no longer needed")
    print()
    
    print("4. Pipeline Memory Management:")
    print("   - Accounts for multi-stage pipeline memory requirements")
    print("   - Multiplies base memory usage by pipeline stages")
    print("   - Enables computation-memory overlap")
    print()
    
    print("5. Architecture-Specific Optimizations:")
    print("   - Adjusts transaction sizes based on hardware capabilities")
    print("   - Uses architecture-specific memory bandwidth values")
    print("   - Optimizes for different cache hierarchies")
    print()


def main():
    """Main execution function"""
    print("TileLang Memory Optimization Example")
    print("=" * 50)
    print()
    
    try:
        # Run memory optimization demonstrations
        demonstrate_bestfit_allocator()
        analyze_memory_stride_patterns()
        memory_traffic_optimization()
        shared_memory_optimization()
        memory_coalescing_analysis()
        memory_bandwidth_utilization()
        advanced_memory_techniques()
        
        print("=== Summary ===")
        print("Memory optimization analysis completed successfully!")
        print("\nKey insights:")
        print("- Best-fit allocation minimizes memory fragmentation")
        print("- Proper stride patterns avoid bank conflicts and improve performance")
        print("- Tile size selection significantly impacts memory traffic")
        print("- TensorCore optimizations require additional memory for pipelining")
        print("- Memory coalescing is critical for bandwidth utilization")
        print("- Architecture-specific optimizations maximize hardware efficiency")
        
    except Exception as e:
        print(f"Error during memory optimization analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()