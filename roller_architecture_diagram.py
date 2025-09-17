#!/usr/bin/env python3
"""
Roller Features Architecture Diagram Generator

This script generates a visual representation of the TileLang roller features
architecture, showing the relationships between different components.
"""

def print_architecture_diagram():
    """Print a text-based architecture diagram of the roller system."""
    
    diagram = """
TileLang Roller Features Architecture
=====================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                              Template Layer                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ MatmulTmpl  │  │  ConvTmpl   │  │  GEMVTmpl   │  │ FlashAttTmpl│   ...  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │                 │              │
│         └─────────────────┼─────────────────┼─────────────────┘              │
│                           │                 │                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     BaseTemplate                                        │ │
│  │  - Architecture management                                              │ │
│  │  - Hardware capability checking                                        │ │
│  │  - Function initialization                                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               Policy Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────┐              ┌─────────────────────────────────┐ │
│  │    DefaultPolicy        │              │      TensorCorePolicy           │ │
│  │  ┌─────────────────────┐│              │  ┌─────────────────────────────┐│ │
│  │  │ Memory Traffic Opt  ││              │  │ WMMA Instruction Opt        ││ │
│  │  │ Tile Size DFS       ││              │  │ Pipeline Stage Config       ││ │
│  │  │ Block Size Assign   ││              │  │ Shared Memory Management    ││ │
│  │  │ Vectorization Plan  ││              │  │ Bank Conflict Avoidance     ││ │
│  │  └─────────────────────┘│              │  └─────────────────────────────┘│ │
│  └─────────────────────────┘              └─────────────────────────────────┘ │
│              │                                           │                    │
│              └───────────────┬───────────────────────────┘                    │
│                              │                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      Common Utilities                                   │ │
│  │  - Factorization algorithms                                            │ │
│  │  - Coalescing analysis                                                 │ │
│  │  - Memory transaction optimization                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Hint System                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    Hint     │  │   Stride    │  │ IntrinInfo  │  │ TensorCoreExtraConf │ │
│  │             │  │             │  │             │  │                     │ │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────────────┐ │ │
│  │ │ block   │ │  │ │stride   │ │  │ │ in_dtype│ │  │ │ AS_shape       │ │ │
│  │ │ thread  │ │  │ │ax       │ │  │ │out_dtype│ │  │ │ BS_shape       │ │ │
│  │ │ warp    │ │  │ └─────────┘ │  │ │trans_b  │ │  │ │ AF_shape       │ │ │
│  │ │ rstep   │ │  └─────────────┘  │ └─────────┘ │  │ │ BF_shape       │ │ │
│  │ │pipeline │ │                   └─────────────┘  │ │ tc_axis        │ │ │
│  │ │raster   │ │                                    │ └─────────────────┘ │ │
│  │ └─────────┘ │                                    └─────────────────────┘ │
│  └─────────────┘                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Rasterization System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐   │
│  │ NoRasterization │  │Rasterization2D  │  │   Rasterization2DColumn     │   │
│  │                 │  │     Row         │  │                             │   │
│  │ ┌─────────────┐ │  │                 │  │ ┌─────────────────────────┐ │   │
│  │ │return []    │ │  │ ┌─────────────┐ │  │ │ __device__ function     │ │   │
│  │ └─────────────┘ │  │ │panel_width=4│ │  │ │ Complex block mapping   │ │   │
│  └─────────────────┘  │ └─────────────┘ │  │ │ Alternating patterns    │ │   │
│                       └─────────────────┘  │ └─────────────────────────┘ │   │
│                                            └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Node System                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐      ┌─────────────┐      ┌─────────────────────────┐   │
│  │  PrimFuncNode   │ ───► │    Edge     │ ───► │     OutputNode          │   │
│  │                 │      │             │      │                         │   │
│  │ ┌─────────────┐ │      │ ┌─────────┐ │      │ ┌─────────────────────┐ │   │
│  │ │input_buffers│ │      │ │src_node │ │      │ │ is_output() = True  │ │   │
│  │ │output_buffs │ │      │ │dst_node │ │      │ └─────────────────────┘ │   │
│  │ │reduction_blk│ │      │ │src_id   │ │      └─────────────────────────┘   │
│  │ │space_dim    │ │      │ │dst_id   │ │                                    │
│  │ │raxis        │ │      │ └─────────┘ │                                    │
│  │ └─────────────┘ │      └─────────────┘                                    │
│  └─────────────────┘                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Architecture Support                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │    CUDA     │  │    CDNA     │  │     CPU     │  │    TileDevice       │ │
│  │             │  │             │  │             │  │     (Base)          │ │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────────────┐ │ │
│  │ │ Volta   │ │  │ │ MFMA    │ │  │ │ LLVM    │ │  │ │ warp_size       │ │ │
│  │ │ Ampere  │ │  │ │ HIP     │ │  │ │ OpenMP  │ │  │ │ sm_partition    │ │ │
│  │ │ Hopper  │ │  │ │ ROCM    │ │  │ │ AVX     │ │  │ │ max_smem_usage  │ │ │
│  │ │ Ada     │ │  │ └─────────┘ │  │ └─────────┘ │  │ │ compute_cap     │ │ │
│  │ └─────────┘ │  └─────────────┘  └─────────────┘  │ │ bandwidth       │ │ │
│  └─────────────┘                                    │ └─────────────────┘ │ │
│                                                     └─────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

Data Flow:
==========
1. Template Layer: High-level operation interfaces (MatMul, Conv, etc.)
2. Policy Layer: Optimization strategies (Default vs TensorCore specific)
3. Hint System: Configuration containers with optimization parameters
4. Rasterization: Memory access pattern optimization
5. Node System: Computational graph representation
6. Architecture: Hardware-specific parameters and capabilities

Key Interactions:
================
• Templates call Policies to generate optimization hints
• Policies analyze operations and produce Hint objects
• Hints contain rasterization plans for memory optimization
• Node system represents computational dependencies
• Architecture layer provides hardware-specific constraints
• Common utilities support factorization and memory analysis

Performance Optimizations:
=========================
• Memory traffic minimization through coalescing analysis
• Shared memory bank conflict avoidance via stride optimization
• L2 cache locality improvement through rasterization patterns
• Tensor core utilization via WMMA instruction optimization
• Pipeline staging for latency hiding in async operations
• Vectorization for increased memory and compute throughput
"""
    
    print(diagram)

def print_optimization_flow():
    """Print the optimization flow diagram."""
    
    flow = """
Optimization Flow in Roller System
==================================

Input: TVM PrimFunc + Target Architecture
│
▼
┌─────────────────────────────────────────┐
│         Function Analysis               │
│  • Extract operation type               │
│  • Identify reduction/spatial axes      │
│  • Analyze memory access patterns      │
│  • Check tensor core compatibility     │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│         Policy Selection                │
│  IF tensor core compatible:             │
│    → TensorCorePolicy                   │
│  ELSE:                                  │
│    → DefaultPolicy                      │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│      Configuration Generation           │
│  • DFS tile exploration                │
│  • Memory traffic analysis             │
│  • Shared memory optimization          │
│  • Thread block configuration          │
│  • Vectorization planning              │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│       Hint Object Creation              │
│  • Package optimization parameters     │
│  • Add rasterization plans             │
│  • Set pipeline configuration          │
│  • Include vectorization settings      │
└─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────┐
│        Ranking and Selection            │
│  • Score configurations by:            │
│    - Memory traffic                     │
│    - Compute utilization               │
│    - Architecture fit                  │
│  • Return top-k configurations         │
└─────────────────────────────────────────┘
│
▼
Output: List[Hint] with optimized configurations

Performance Metrics Used in Scoring:
====================================
• Memory Traffic = Global memory accesses × transaction overhead
• Compute Utilization = Operations / (SM count × compute capability)
• Wave Efficiency = Grid size / (Blocks per SM × SM count)
• Register Pressure = Estimated register usage / Register file size
• Shared Memory Efficiency = SMEM usage / SMEM capacity
• Cache Locality = Rasterization pattern benefit score
"""
    
    print(flow)

if __name__ == "__main__":
    print_architecture_diagram()
    print("\n" + "="*80 + "\n")
    print_optimization_flow()