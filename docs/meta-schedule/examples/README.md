# Examples and Tutorials

This directory contains practical examples demonstrating the usage of TileLang's meta-scheduling system.

## Available Examples

### Basic Usage
- [`basic-cost-model.py`](./basic-cost-model.py) - Basic cost model usage and configuration analysis
- [`tensorcore-gemm.py`](./tensorcore-gemm.py) - TensorCore GEMM optimization example
- [`memory-optimization.py`](./memory-optimization.py) - Memory allocation and optimization strategies

### Advanced Features
- [`multi-architecture.py`](./multi-architecture.py) - Cross-architecture optimization
- [`custom-policy.py`](./custom-policy.py) - Creating custom optimization policies
- [`performance-analysis.py`](./performance-analysis.py) - Performance profiling and analysis

### Architecture-Specific Examples
- [`volta-tensorcore.py`](./volta-tensorcore.py) - Volta TensorCore optimization
- [`ampere-features.py`](./ampere-features.py) - Ampere-specific features (async copy, enhanced precision)
- [`ada-fp8.py`](./ada-fp8.py) - Ada Lovelace FP8 TensorCore operations

## Quick Start

```python
# Basic usage example
import tilelang
from tilelang.carver.arch import CUDA
from tilelang.carver.roller.policy import TensorCorePolicy

# Create target architecture
arch = CUDA("cuda -arch=sm_80")

# Create TensorCore policy
policy = TensorCorePolicy(arch)

# Generate optimized configurations
configs = policy.emit_config(topk=5)

# Analyze results
for i, config in enumerate(configs):
    print(f"Configuration {i+1}:")
    print(f"  Block: {config.block}")
    print(f"  Warp: {config.warp}")
    print(f"  TensorCore: {config.use_tc}")
    print(f"  Pipeline: {config.pipeline_stage}")
```

## Running Examples

Each example is self-contained and can be run independently:

```bash
# Run basic cost model example
python basic-cost-model.py

# Run TensorCore GEMM example
python tensorcore-gemm.py

# Run memory optimization example
python memory-optimization.py
```

## Example Categories

### Performance Optimization
Examples focusing on maximizing kernel performance through optimal tile configurations and memory access patterns.

### Memory Management
Examples demonstrating efficient memory allocation, reuse strategies, and shared memory optimization.

### TensorCore Utilization
Examples showcasing TensorCore optimization for different precisions and matrix operations.

### Cross-Architecture
Examples showing how to optimize kernels for multiple GPU architectures simultaneously.