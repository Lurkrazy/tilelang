#!/usr/bin/env python3
"""
Basic Cost Model Usage Example

This example demonstrates fundamental usage of TileLang's cost model system,
including configuration generation, cost analysis, and optimization strategies.
"""

import numpy as np
import tilelang
import tilelang.language as T
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.policy import DefaultPolicy


def create_sample_gemm(M, N, K):
    """Create a sample GEMM operation for analysis"""
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
            A_shared = T.alloc_shared((64, 32), "float16")
            B_shared = T.alloc_shared((32, 64), "float16")
            C_local = T.alloc_fragment((64, 64), "float")
            
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2):
                T.copy(A[by * 64, k * 32], A_shared)
                T.copy(B[k * 32, bx * 64], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * 64, bx * 64])
    
    return gemm


def analyze_cost_model():
    """Demonstrate basic cost model analysis"""
    print("=== Basic Cost Model Analysis ===")
    
    # Create target architecture
    arch = CUDA("cuda -arch=sm_80")
    print(f"Target Architecture: {arch.platform} {arch.compute_capability}")
    print(f"Shared Memory: {arch.smem_cap // 1024}KB")
    print(f"Register Capacity: {arch.reg_cap}")
    print(f"Max Cores: {arch.compute_max_core}")
    print()
    
    # Create sample GEMM operation
    gemm_func = create_sample_gemm(1024, 1024, 1024)
    
    # Create cost model policy
    policy = DefaultPolicy.from_prim_func(gemm_func, arch, name="GEMM")
    print(f"Nodes in computation: {len(policy.ordered_nodes)}")
    print(f"Output nodes: {len(policy.output_nodes)}")
    print()
    
    # Analyze different tile configurations
    tile_configs = [
        [64, 64],
        [128, 128], 
        [256, 128],
        [128, 256],
    ]
    
    print("=== Tile Configuration Analysis ===")
    print(f"{'Tile':<12} {'Traffic':<10} {'Shared Mem':<12} {'Blocks/SM':<10} {'Waves':<8} {'Valid':<6}")
    print("-" * 70)
    
    for tile in tile_configs:
        # Create reduction step mapping
        rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}
        
        # Compute cost metrics
        td = policy.compute_tile_dict(tile, rstep_map)
        
        print(f"{str(tile):<12} {td.traffic:<10} {td.smem_cost:<12} {td.block_per_SM:<10} {td.num_wave:<8} {td.valid:<6}")
    
    print()


def generate_optimized_configs():
    """Generate and analyze optimized configurations"""
    print("=== Optimized Configuration Generation ===")
    
    arch = CUDA("cuda -arch=sm_80")
    gemm_func = create_sample_gemm(1024, 1024, 1024)
    policy = DefaultPolicy.from_prim_func(gemm_func, arch)
    
    # Generate top configurations
    configs = policy.emit_config(topk=5)
    
    print(f"Generated {len(configs)} optimized configurations:")
    print()
    
    for i, config in enumerate(configs):
        print(f"Configuration {i+1}:")
        print(f"  Block Tile: {config.block}")
        print(f"  Warp Tile: {config.warp}")
        print(f"  Reduction Steps: {config.rstep}")
        print(f"  Vectorization: {config.vectorize}")
        print(f"  Cached Tensors: {len(config.cached_tensors)}")
        print()


def memory_traffic_analysis():
    """Analyze memory traffic patterns"""
    print("=== Memory Traffic Analysis ===")
    
    arch = CUDA("cuda -arch=sm_80")
    gemm_func = create_sample_gemm(1024, 1024, 1024)
    policy = DefaultPolicy.from_prim_func(gemm_func, arch)
    
    # Analyze traffic for different tile sizes
    tile_sizes = [32, 64, 128, 256]
    
    print(f"{'Tile Size':<12} {'Memory Traffic (MB)':<20} {'Traffic/Tile':<15}")
    print("-" * 50)
    
    for size in tile_sizes:
        tile = [size, size]
        traffic, _ = policy._compute_memory_traffic(tile)
        traffic_mb = traffic / (1024 * 1024)
        traffic_per_tile = traffic / (size * size)
        
        print(f"{size:<12} {traffic_mb:<20.2f} {traffic_per_tile:<15.2f}")
    
    print()


def shared_memory_analysis():
    """Analyze shared memory usage patterns"""
    print("=== Shared Memory Analysis ===")
    
    arch = CUDA("cuda -arch=sm_80")
    gemm_func = create_sample_gemm(1024, 1024, 1024)
    policy = DefaultPolicy.from_prim_func(gemm_func, arch)
    
    # Test different reduction step sizes
    reduction_steps = [16, 32, 64, 128]
    tile = [128, 128]
    
    print(f"{'Reduction Step':<15} {'Shared Memory (KB)':<20} {'Utilization %':<15}")
    print("-" * 55)
    
    for rstep in reduction_steps:
        rstep_map = {node: {"k": rstep} for node in policy.ordered_nodes}
        td = policy.compute_tile_dict(tile, rstep_map)
        
        smem_kb = td.smem_cost / 1024
        utilization = (td.smem_cost / arch.smem_cap) * 100
        
        print(f"{rstep:<15} {smem_kb:<20.2f} {utilization:<15.1f}")
    
    print()


def occupancy_analysis():
    """Analyze thread block occupancy"""
    print("=== Occupancy Analysis ===")
    
    arch = CUDA("cuda -arch=sm_80")
    gemm_func = create_sample_gemm(1024, 1024, 1024)
    policy = DefaultPolicy.from_prim_func(gemm_func, arch)
    
    # Test different block sizes
    block_sizes = [64, 128, 256, 512]
    tile = [128, 128]
    rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}
    
    print(f"{'Threads/Block':<15} {'Blocks/SM':<12} {'Occupancy %':<15} {'Waves':<8}")
    print("-" * 55)
    
    for block_size in block_sizes:
        td = policy.compute_tile_dict(tile, rstep_map)
        
        # Estimate occupancy (simplified)
        max_blocks_per_sm = arch.sm_partition
        theoretical_occupancy = min(td.block_per_SM / max_blocks_per_sm * 100, 100)
        
        print(f"{block_size:<15} {td.block_per_SM:<12} {theoretical_occupancy:<15.1f} {td.num_wave:<8}")
    
    print()


def main():
    """Main execution function"""
    print("TileLang Cost Model Basic Usage Example")
    print("=" * 50)
    print()
    
    try:
        # Run analysis functions
        analyze_cost_model()
        generate_optimized_configs()
        memory_traffic_analysis()
        shared_memory_analysis()
        occupancy_analysis()
        
        print("=== Summary ===")
        print("Cost model analysis completed successfully!")
        print("Key insights:")
        print("- Larger tiles generally reduce memory traffic but increase shared memory usage")
        print("- Optimal tile sizes balance memory efficiency with hardware constraints")
        print("- Reduction step size significantly impacts shared memory requirements")
        print("- Occupancy optimization requires balancing multiple resource constraints")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()