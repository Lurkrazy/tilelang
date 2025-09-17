# TileLang Meta-Schedule Documentation

This documentation provides a comprehensive analysis of TileLang's cost model, scheduling policies, and TensorCore support implementation. It serves as a technical reference for understanding the performance optimization strategies and architectural considerations in TileLang.

## Table of Contents

1. [Overview](#overview)
2. [Cost Model Architecture](#cost-model-architecture)
3. [TensorCore Implementation](#tensorcore-implementation)  
4. [Scheduling Policies](#scheduling-policies)
5. [Performance Analysis Framework](#performance-analysis-framework)
6. [Architecture Support](#architecture-support)
7. [Examples and Use Cases](#examples-and-use-cases)

## Overview

TileLang employs a sophisticated meta-scheduling framework that combines:
- Hardware-aware cost modeling for accurate performance prediction
- Policy-driven scheduling for optimal kernel configuration
- TensorCore support for high-performance mixed-precision computing
- Comprehensive performance analysis tools

## Documentation Structure

- **[Cost Model Analysis](./cost-model/README.md)** - Detailed analysis of the performance modeling framework
- **[TensorCore Implementation](./tensorcore/README.md)** - Deep dive into TensorCore scheduling and optimization
- **[Scheduling Policies](./policies/README.md)** - Documentation of default and specialized scheduling policies
- **[Performance Analysis](./analysis/README.md)** - Performance analysis tools and roofline modeling
- **[Architecture Support](./architecture/README.md)** - Hardware abstraction and device-specific optimizations
- **[Examples](./examples/README.md)** - Practical examples and implementation patterns

## Key Features

### Cost Model
- **Memory Traffic Analysis**: Accurate modeling of global memory transfers
- **FLOP Counting**: Comprehensive floating-point operation analysis
- **Roofline Performance Modeling**: Compute-bound vs memory-bound analysis
- **Architecture-Aware Predictions**: Device-specific performance estimation

### TensorCore Support
- **Mixed-Precision Computing**: Support for FP16, BF16, INT8, and other precision formats
- **Automatic Layout Optimization**: Bank conflict avoidance and memory coalescing
- **Pipeline Scheduling**: Multi-stage pipeline with async memory operations
- **Warp-Level Optimization**: Efficient warp utilization and scheduling

### Policy Framework
- **Default Policy**: Heuristic-based scheduling for general workloads
- **TensorCore Policy**: Specialized scheduling for mixed-precision tensor operations
- **Extensible Design**: Support for custom policies and optimization strategies

## Getting Started

For a practical introduction to the cost model and features, see the [examples/analyze/README.md](../examples/analyze/README.md) in the main repository.

For detailed technical documentation, explore the individual sections linked above.