# Meta-Schedule Documentation

This directory contains comprehensive documentation for TileLang's meta-scheduling system, including cost models and TensorCore implementations.

## Documentation Structure

- [`cost-model.md`](./cost-model.md) - Detailed documentation of the cost model architecture
- [`tensorcore.md`](./tensorcore.md) - TensorCore implementation and optimization strategies
- [`memory-management.md`](./memory-management.md) - Memory allocation and optimization strategies
- [`hardware-abstraction.md`](./hardware-abstraction.md) - Hardware device abstractions
- [`api-reference.md`](./api-reference.md) - Complete API reference
- [`examples/`](./examples/) - Usage examples and tutorials

## Quick Start

The meta-scheduling system in TileLang provides automated optimization for tensor computations, particularly focusing on CUDA TensorCore acceleration. The system consists of:

1. **Cost Model**: Analyzes memory traffic, register usage, and shared memory usage to estimate performance
2. **TensorCore Policy**: Specialized optimization strategies for TensorCore operations
3. **Memory Management**: Efficient allocation and usage tracking for shared memory
4. **Hardware Abstraction**: Device-specific optimizations for different GPU architectures

## Key Components

- **DefaultPolicy**: Base policy implementing heuristic optimization strategies
- **TensorCorePolicy**: Specialized policy for TensorCore operations
- **TileDict**: Configuration management for tiling strategies
- **BestFit**: Memory allocation optimization
- **Hardware Architectures**: CUDA, CDNA, and CPU device abstractions