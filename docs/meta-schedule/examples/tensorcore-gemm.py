#!/usr/bin/env python3
"""
TensorCore GEMM Optimization Example

This example demonstrates advanced TensorCore optimization using TensorCorePolicy,
including precision selection, pipeline configuration, and performance analysis.
"""

import numpy as np
import tilelang
import tilelang.language as T
from tilelang.carver.arch import CUDA
from tilelang.carver.arch.cuda import (
    is_tensorcore_supported_precision, 
    is_ampere_arch, 
    is_volta_arch,
    is_ada_arch
)
from tilelang.carver.roller.policy import TensorCorePolicy


def create_tensorcore_gemm(M, N, K, in_dtype="float16", accum_dtype="float"):
    """Create a TensorCore-optimized GEMM operation"""
    @T.prim_func
    def tensorcore_gemm(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), in_dtype)
            B_shared = T.alloc_shared((32, 128), in_dtype)
            C_local = T.alloc_fragment((128, 128), accum_dtype)
            
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            
            T.copy(C_local, C[by * 128, bx * 128])
    
    return tensorcore_gemm


def analyze_tensorcore_capabilities():
    """Analyze TensorCore capabilities across different architectures"""
    print("=== TensorCore Capability Analysis ===")
    
    architectures = [
        ("sm_70", "Volta"),
        ("sm_80", "Ampere"),
        ("sm_89", "Ada Lovelace"),
        ("sm_90", "Hopper"),
    ]
    
    precisions = [
        ("float16", "float32"),
        ("float16", "float16"),
        ("bfloat16", "float32"),
        ("int8", "int32"),
        ("float8_e4m3", "float32"),
        ("float8_e5m2", "float32"),
    ]
    
    print(f"{'Architecture':<15} {'Precision':<20} {'Supported':<10}")
    print("-" * 50)
    
    for arch_name, arch_desc in architectures:
        try:
            arch = CUDA(f"cuda -arch={arch_name}")
            print(f"\n{arch_desc} ({arch_name}):")
            
            for in_dtype, accum_dtype in precisions:
                supported = False
                try:
                    supported = is_tensorcore_supported_precision(in_dtype, accum_dtype, arch)
                except:
                    supported = False
                
                precision_str = f"{in_dtype}->{accum_dtype}"
                print(f"{'':>15} {precision_str:<20} {'✓' if supported else '✗':<10}")
                
        except Exception as e:
            print(f"Could not analyze {arch_desc}: {e}")
    
    print()


def optimize_tensorcore_configuration():
    """Generate and analyze TensorCore-optimized configurations"""
    print("=== TensorCore Configuration Optimization ===")
    
    # Try different architectures if available
    target_archs = ["sm_80", "sm_70"]  # Ampere, Volta
    
    for arch_name in target_archs:
        try:
            arch = CUDA(f"cuda -arch={arch_name}")
            print(f"\nOptimizing for {arch.platform} {arch.compute_capability}:")
            
            # Create TensorCore policy with architecture-specific tags
            tags = {
                "tensorcore_config": (0, 1),  # M and N axes
                "intrin_info": {
                    "in_dtype": "float16",
                    "out_dtype": "float32",
                    "trans_a": False,
                    "trans_b": True
                }
            }
            
            # Add architecture-specific optimizations
            if is_ampere_arch(arch):
                tags["pipeline_stage"] = 3
                tags["use_async_copy"] = True
                print("  Ampere optimizations: 3-stage pipeline, async copy")
            elif is_volta_arch(arch):
                tags["pipeline_stage"] = 1
                tags["use_async_copy"] = False
                print("  Volta optimizations: single-stage pipeline")
            
            gemm_func = create_tensorcore_gemm(1024, 1024, 1024)
            policy = TensorCorePolicy.from_prim_func(gemm_func, arch, tags=tags)
            
            # Generate optimized configurations
            configs = policy.emit_config(topk=3)
            
            print(f"  Generated {len(configs)} configurations:")
            
            for i, config in enumerate(configs):
                print(f"\n  Configuration {i+1}:")
                print(f"    Block: {config.block}")
                print(f"    Warp: {config.warp}")
                print(f"    TensorCore: {config.use_tc}")
                print(f"    Pipeline Stages: {config.pipeline_stage}")
                print(f"    Async Copy: {config.use_async}")
                print(f"    Shared Memory Scope: {config.shared_scope}")
                
                if hasattr(config, 'intrin_info') and config.intrin_info:
                    print(f"    Intrinsic: {config.intrin_info.in_dtype} -> {config.intrin_info.out_dtype}")
        
        except Exception as e:
            print(f"Could not optimize for {arch_name}: {e}")
    
    print()


def tensorcore_memory_analysis():
    """Analyze memory usage patterns for TensorCore operations"""
    print("=== TensorCore Memory Analysis ===")
    
    arch = CUDA("cuda -arch=sm_80")
    
    # Test different tile sizes
    tile_configs = [
        ([64, 64], "Small tiles"),
        ([128, 128], "Medium tiles"),
        ([256, 128], "Large M tiles"),
        ([128, 256], "Large N tiles"),
    ]
    
    print(f"{'Configuration':<15} {'Tile Size':<12} {'Shared Mem (KB)':<16} {'Pipeline Factor':<16}")
    print("-" * 70)
    
    for tile, desc in tile_configs:
        tags = {
            "tensorcore_config": (0, 1),
            "pipeline_stage": 2,
            "use_async_copy": True
        }
        
        gemm_func = create_tensorcore_gemm(1024, 1024, 1024)
        policy = TensorCorePolicy.from_prim_func(gemm_func, arch, tags=tags)
        
        # Create reduction step mapping
        rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}
        
        # Compute TensorCore-specific costs
        td = policy.compute_tile_dict(tile, rstep_map)
        
        smem_kb = td.smem_cost / 1024
        pipeline_factor = policy.pipeline_stage
        
        print(f"{desc:<15} {str(tile):<12} {smem_kb:<16.1f} {pipeline_factor:<16}")
    
    print()


def precision_performance_comparison():
    """Compare performance characteristics of different TensorCore precisions"""
    print("=== TensorCore Precision Performance Comparison ===")
    
    arch = CUDA("cuda -arch=sm_80")
    
    # Test different precision combinations
    precision_configs = [
        ("float16", "float32", "Mixed FP16"),
        ("float16", "float16", "Pure FP16"),
        ("int8", "int32", "INT8"),
    ]
    
    print(f"{'Precision':<15} {'WMMA K':<8} {'Throughput Factor':<18} {'Memory Efficiency':<18}")
    print("-" * 70)
    
    for in_dtype, accum_dtype, desc in precision_configs:
        if is_tensorcore_supported_precision(in_dtype, accum_dtype, arch):
            tags = {
                "tensorcore_config": (0, 1),
                "intrin_info": {
                    "in_dtype": in_dtype,
                    "out_dtype": accum_dtype,
                    "trans_a": False,
                    "trans_b": True
                }
            }
            
            policy = TensorCorePolicy(arch, tags=tags)
            
            # Configure WMMA K dimension based on precision
            if in_dtype == "int8":
                wmma_k = 32
            else:
                wmma_k = 16
            
            policy.wmma_k = wmma_k
            
            # Calculate theoretical throughput factor
            bits_per_element = 16 if in_dtype == "float16" else 8
            throughput_factor = 16 / bits_per_element  # Relative to FP16
            
            # Memory efficiency (elements per byte)
            memory_efficiency = 8 / bits_per_element
            
            print(f"{desc:<15} {wmma_k:<8} {throughput_factor:<18.1f} {memory_efficiency:<18.1f}")
    
    print()


def advanced_tensorcore_features():
    """Demonstrate advanced TensorCore features"""
    print("=== Advanced TensorCore Features ===")
    
    arch = CUDA("cuda -arch=sm_80")
    
    if is_ampere_arch(arch):
        print("Ampere Architecture Features:")
        
        # Asynchronous copy operations
        print("  ✓ Asynchronous Copy Operations")
        print("    - Overlap memory transfers with computation")
        print("    - Hide memory latency")
        
        # Multi-stage pipelines
        print("  ✓ Multi-Stage Pipelines")
        print("    - Up to 3+ pipeline stages")
        print("    - Improved memory bandwidth utilization")
        
        # Enhanced precision support
        print("  ✓ Enhanced Precision Support")
        print("    - BFloat16 operations")
        print("    - INT8/INT4 quantized operations")
        print("    - Sparse tensor operations (2:4 sparsity)")
        
        # Memory bandwidth optimization
        print("  ✓ Memory Bandwidth Optimization")
        print(f"    - Memory bandwidth: {arch.bandwidth[0] / 1000:.1f} GB/s")
        print(f"    - L2 cache: {arch.l2_cache_size_bytes // (1024*1024)} MB")
        
    elif is_volta_arch(arch):
        print("Volta Architecture Features:")
        print("  ✓ WMMA Instructions")
        print("  ✓ FP16 TensorCore Operations")
        print("  ✓ Mixed Precision Training")
        
    # Rasterization strategies
    tags = {"tensorcore_config": (0, 1)}
    policy = TensorCorePolicy(arch, tags=tags)
    
    # Create a sample tile configuration
    rstep_map = {node: {"k": 32} for node in policy.ordered_nodes}
    td = policy.compute_tile_dict([128, 128], rstep_map)
    
    raster_plan = policy.plan_rasterization(td)
    print(f"\n  Rasterization Strategy: {type(raster_plan).__name__}")
    
    if hasattr(raster_plan, 'raster_factor'):
        print(f"    Raster Factor: {raster_plan.raster_factor}")
    
    print()


def main():
    """Main execution function"""
    print("TileLang TensorCore Optimization Example")
    print("=" * 50)
    print()
    
    try:
        # Run analysis functions
        analyze_tensorcore_capabilities()
        optimize_tensorcore_configuration()
        tensorcore_memory_analysis()
        precision_performance_comparison()
        advanced_tensorcore_features()
        
        print("=== Summary ===")
        print("TensorCore optimization analysis completed successfully!")
        print("\nKey insights:")
        print("- Architecture detection enables automatic optimization")
        print("- Precision selection impacts both performance and memory usage")
        print("- Pipeline stages significantly improve memory bandwidth utilization")
        print("- Async copy operations hide memory transfer latency")
        print("- Tile size optimization balances compute efficiency with memory constraints")
        
    except Exception as e:
        print(f"Error during TensorCore analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()