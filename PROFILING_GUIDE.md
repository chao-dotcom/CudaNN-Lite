# Profiling Results Analysis Template
# Fill this in after running profile_quick.sh or profile_experiments.sh

## Key Metrics Explained

### 1. SM Throughput (Compute Utilization)
- **What it measures:** Percentage of peak compute capability used
- **Target:** > 60% is good, > 80% is excellent
- **Low values indicate:** Memory bottleneck or insufficient parallelism

### 2. DRAM Throughput (Memory Bandwidth)
- **What it measures:** Percentage of peak memory bandwidth used
- **Target:** Varies by kernel type
- **High values indicate:** Memory-bound kernel

### 3. Active Warps (Occupancy)
- **What it measures:** Percentage of maximum warps active
- **Target:** > 50% is good, > 75% is excellent
- **Low values indicate:** Register pressure or shared memory limits

---

## EXP1: Dense GEMM (Tiled Implementation)

### Profiling Results:
```
SM Throughput:     ____ %
DRAM Throughput:   ____ %
Active Warps:      ____ %
```

### Analysis:
- [ ] **Compute-bound** (SM Throughput > DRAM Throughput)
- [ ] **Memory-bound** (DRAM Throughput > SM Throughput)

### Report Claim:
> "Achieved **[X]%** SM occupancy with tiled GEMM kernel. Profiling identified 
> **[compute/memory]**-bound behavior, with **[Y]%** of peak memory bandwidth 
> utilized."

**Example filled:**
> "Achieved **68%** SM occupancy with tiled GEMM kernel. Profiling identified 
> **compute**-bound behavior, with **45%** of peak memory bandwidth utilized."

---

## EXP2: MLP Inference (Fused Kernels)

### Profiling Results:
```
Baseline Implementation:
  SM Throughput:     ____ %
  DRAM Throughput:   ____ %
  Active Warps:      ____ %

Fused Implementation:
  SM Throughput:     ____ %
  DRAM Throughput:   ____ %
  Active Warps:      ____ %
```

### Analysis:
**Fusion Benefits Observed:**
- [ ] Reduced memory traffic (lower DRAM throughput %)
- [ ] Improved compute utilization (higher SM throughput %)
- [ ] Better occupancy (more active warps)

### Report Claim:
> "Kernel fusion reduced global memory traffic from **[X]%** to **[Y]%** of 
> peak bandwidth, while increasing compute throughput by **[Z]** percentage points."

**Example filled:**
> "Kernel fusion reduced global memory traffic from **52%** to **38%** of 
> peak bandwidth, while increasing compute throughput by **8** percentage points."

---

## EXP3: Sparse Matrix Multiply (Warp-Collaborative)

### Profiling Results:
```
SM Throughput:     ____ %
DRAM Throughput:   ____ %
Active Warps:      ____ %
```

### Analysis:
- [ ] Good occupancy despite irregular memory access
- [ ] Memory throughput limited by sparse access pattern
- [ ] Compute underutilized due to sparsity

### Report Claim:
> "Warp-collaborative design maintained **[X]%** occupancy despite irregular 
> CSR access patterns. Memory throughput of **[Y]%** reflects expected overhead 
> from sparse data structures."

**Example filled:**
> "Warp-collaborative design maintained **72%** occupancy despite irregular 
> CSR access patterns. Memory throughput of **35%** reflects expected overhead 
> from sparse data structures."

---

## Overall Profiling Summary for Resume/Report

### Conservative Claim (No Actual Profiling Done):
> "Implemented performance-optimized CUDA kernels with consideration for occupancy 
> and memory bandwidth constraints. Used CUDA event timers to measure kernel 
> performance and identify optimization opportunities."

### Full Claim (With Actual Profiling):
> "Profiled SM occupancy and memory throughput using **Nsight Compute** to identify 
> and resolve hardware utilization bottlenecks. Achieved **[60-80]%** occupancy on 
> tiled GEMM kernels and analyzed compute vs. memory-bound behavior across all 
> implementations."

---

## Interpretation Guide

### If SM Throughput is HIGH (>70%) and DRAM is LOW (<40%):
✅ **Compute-bound** - Good! Your kernel is doing lots of math per memory access.
   This is ideal for GEMM operations.

### If DRAM Throughput is HIGH (>60%) and SM is LOW (<40%):
⚠️ **Memory-bound** - Your kernel is waiting for memory.
   Solutions: Increase data reuse (tiling), use shared memory, coalesce accesses.

### If BOTH are LOW (<40%):
❌ **Underutilized** - Hardware is idle.
   Solutions: Increase occupancy (more blocks), reduce divergence, check launch config.

### If Active Warps is LOW (<50%):
⚠️ **Low Occupancy** - Not enough parallelism.
   Causes: Too many registers, too much shared memory, small grid size.
   Solutions: Reduce resource usage, increase grid dimensions.

---

## Commands Reference

### Full profiling with GUI:
```bash
chmod +x profile_experiments.sh
./profile_experiments.sh
ncu-ui profiles/gemm_tiled.ncu-rep
```

### Quick metrics only:
```bash
chmod +x profile_quick.sh
./profile_quick.sh
# Check profile_*.csv files
```

### Manual single kernel:
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active \
./bin/dgemm --m 2048 --n 2048 --k 2048 --impl tiled
```

### Save full report:
```bash
ncu --set full --export profile.ncu-rep ./your_program
ncu --import profile.ncu-rep --page details > report.txt
```
