# CudaNN-Lite

A comprehensive hands-on curriculum for learning GPU-accelerated computing and neural network primitives using CUDA. This four-module course takes you from CUDA basics through advanced optimization techniques for neural network operations.

## Overview

This repository provides a progressive learning path through GPU programming, structured as four interconnected modules that build upon each other. Each module includes starter code with TODOs, performance targets, profiling exercises, and reference implementations. Ideal for senior undergraduates, GPU computing courses, and self-study.

**Prerequisites:**
- CUDA Toolkit 12+ (with `nvcc` and `nvidia-smi`)
- NVIDIA GPU with compute capability 7.0+
- Linux environment (or WSL on Windows)
- Basic understanding of C/C++ and linear algebra

---

## Module 1: CUDA Fundamentals & Cluster Computing

**Directory:** `exp0_slurm_CUDA_toturial/`

Master the essentials of CUDA programming and learn to run GPU workloads on HPC clusters with Slurm workload manager.

**What You'll Build:**
- Vector addition kernels
- Matrix multiplication with tiling optimizations
- GPU information utilities

**Key Concepts:**
- CUDA thread hierarchy (grids, blocks, threads)
- Memory management (host ↔ device transfers)
- Kernel launch configurations
- Slurm job submission and cluster workflows
- Performance measurement basics

**Quick Start:**
```bash
cd exp0_slurm_CUDA_toturial
make
./build/vector_add
./build/matrix_multiply
```

---

## Module 2: Dense Matrix Multiplication (GEMM)

**Directory:** `exp1_dgemm/`

Implement dense General Matrix Multiplication with progressive optimizations, culminating in comparison against the industry-standard cuBLAS library.

**What You'll Build:**
- Naive element-per-thread GEMM kernel
- Tiled implementation with shared memory
- Performance benchmarking harness
- Roofline analysis pipeline

**Key Concepts:**
- Shared memory optimization
- Memory coalescing patterns
- Tiling strategies (blocking)
- cuBLAS integration and comparison
- FLOP/s calculations and roofline modeling

**Performance Targets:**
- Achieve >30% of cuBLAS performance
- Clear acceleration from naive → tiled (aim for 70% of cuBLAS)

**Quick Start:**
```bash
cd exp1_dgemm
make
./bin/dgemm --m 2048 --n 2048 --k 2048 --impl baseline
./scripts/measure.sh  # Sweep problem sizes
```

---

## Module 3: Neural Network Inference Engine

**Directory:** `exp2_mlp/`

Build a complete inference engine for fully-connected neural networks by composing cuBLAS calls with custom CUDA activation kernels.

**What You'll Build:**
- Batched GEMM orchestration for MLP layers
- ReLU and GELU activation kernels
- Bias-add operations
- Fused kernel optimizations
- End-to-end throughput benchmarking

**Key Concepts:**
- Layer-wise computation graphs
- Kernel fusion to reduce memory traffic
- Tensor layout management (row-major vs column-major)
- Batched operations for throughput
- CPU reference implementation for validation

**Performance Targets:**
- Compare layer-by-layer vs fused scheduling
- Analyze occupancy and memory bandwidth limits

**Quick Start:**
```bash
cd exp2_mlp
make
./bin/dmlp --layers 1024,2048,1024 --batch 128 --impl baseline
./scripts/measure.sh  # Sweep batch sizes and widths
```

---

## Module 4: Sparse Matrix Operations (SpMM)

**Directory:** `exp3_spmm/`

Optimize Sparse × Dense matrix multiplication, learning techniques critical for graph neural networks and sparse deep learning.

**What You'll Build:**
- CSR (Compressed Sparse Row) format handler
- Baseline SpMM kernel
- Warp-optimized implementation
- Performance comparison against dense operations

**Key Concepts:**
- Sparse matrix formats (CSR)
- Warp-level primitives and cooperation
- Memory coalescing for irregular access
- Load balancing for sparse workloads
- CPU reference for correctness checking

**Quick Start:**
```bash
cd exp3_spmm
make
./spmm_baseline
./spmm_opt
./run_experiments.sh
```

---

## Profiling & Performance Analysis

Each module includes profiling exercises. Use the provided tools to analyze kernel performance:

```bash
# Quick profile of all modules
./profile_quick.sh

# Detailed profiling with metrics
./profile_experiments.sh

# Analyze and visualize results
python analyze_results.py
```

**Key Metrics:**
- **SM Throughput:** Compute utilization (target >60%)
- **DRAM Throughput:** Memory bandwidth usage
- **Active Warps:** Occupancy levels (target >50%)

See [PROFILING_GUIDE.md](PROFILING_GUIDE.md) for detailed metrics interpretation and analysis templates.

---

## Repository Structure

```
CudaNN-Lite/
├── exp0_slurm_CUDA_toturial/   # Module 1: CUDA Fundamentals
├── exp1_dgemm/                  # Module 2: Dense GEMM
├── exp2_mlp/                    # Module 3: Neural Network Inference
├── exp3_spmm/                   # Module 4: Sparse Operations
├── profile_*.sh                 # Profiling automation scripts
├── analyze_results.py           # Performance analysis tool
└── PROFILING_GUIDE.md          # Profiling metrics guide
```

---

## Recommended Learning Path

**Complete the modules in order** for a coherent learning experience:

1. **Module 1** → Establish CUDA fundamentals and cluster workflow
2. **Module 2** → Master memory hierarchies and optimization patterns
3. **Module 3** → Apply techniques to real neural network workloads
4. **Module 4** → Handle irregular computation patterns

Each module builds on concepts from previous modules while introducing new optimization techniques.

---

## Deliverables & Assessment

Each module includes:
- ✅ Starter code with TODOs for completion
- ✅ Performance targets and rubrics
- ✅ Suggested timelines (typically 1-2 weeks per module)
- ✅ Report templates for analysis
- ✅ Reference implementations for validation

**Typical Assessment Criteria:**
- **Correctness** (30%): Numerical accuracy vs reference
- **Performance** (30%): Meeting optimization targets
- **Analysis** (20%): Thoughtful bottleneck discussion
- **Presentation** (20%): Code quality and report clarity
