#!/bin/bash
# Nsight Compute Profiling Script for COMP468-568 Experiments
# Run this on a system with CUDA and Nsight Compute installed

set -e

echo "=========================================="
echo "Nsight Compute Profiling"
echo "=========================================="
echo ""

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "ERROR: ncu (Nsight Compute) not found."
    echo "Install it from: https://developer.nvidia.com/nsight-compute"
    echo "Or use: module load cuda (on HPC clusters)"
    exit 1
fi

# Create profiles directory
mkdir -p profiles

# ==========================================
# EXP1: Profile GEMM Kernels
# ==========================================
echo "=========================================="
echo "Profiling EXP1: Dense GEMM"
echo "=========================================="

cd exp1_dgemm
make

echo ""
echo "Profiling naive GEMM kernel..."
ncu --set full \
    --export ../profiles/gemm_naive \
    --force-overwrite \
    ./bin/dgemm --m 2048 --n 2048 --k 2048 --impl naive --no-verify

echo ""
echo "Profiling tiled GEMM kernel..."
ncu --set full \
    --export ../profiles/gemm_tiled \
    --force-overwrite \
    ./bin/dgemm --m 2048 --n 2048 --k 2048 --impl tiled --no-verify

echo ""
echo "Quick metrics for tiled GEMM:"
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active \
    ./bin/dgemm --m 2048 --n 2048 --k 2048 --impl tiled --no-verify

cd ..

# ==========================================
# EXP2: Profile MLP Kernels
# ==========================================
echo ""
echo "=========================================="
echo "Profiling EXP2: MLP Inference"
echo "=========================================="

cd exp2_mlp
make

echo ""
echo "Profiling baseline MLP (separate kernels)..."
ncu --set full \
    --export ../profiles/mlp_baseline \
    --force-overwrite \
    ./bin/dmlp --layers 1024,2048,1024 --batch 128 --impl baseline --no-verify

echo ""
echo "Profiling fused MLP (bias+activation)..."
ncu --set full \
    --export ../profiles/mlp_fused \
    --force-overwrite \
    ./bin/dmlp --layers 1024,2048,1024 --batch 128 --impl activation_fused --no-verify

echo ""
echo "Quick metrics for fused kernel:"
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed,\
launch__occupancy_per_register_count \
    ./bin/dmlp --layers 1024,2048,1024 --batch 128 --impl activation_fused --no-verify

cd ..

# ==========================================
# EXP3: Profile SpMM Kernels
# ==========================================
echo ""
echo "=========================================="
echo "Profiling EXP3: Sparse Matrix Multiply"
echo "=========================================="

cd exp3_spmm
make

echo ""
echo "Profiling baseline SpMM (row-per-thread)..."
ncu --set full \
    --export ../profiles/spmm_baseline \
    --force-overwrite \
    ./spmm_baseline

echo ""
echo "Profiling optimized SpMM (warp-collaborative)..."
ncu --set full \
    --export ../profiles/spmm_opt \
    --force-overwrite \
    ./spmm_opt

echo ""
echo "Quick metrics for optimized SpMM:"
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.pct \
    ./spmm_opt

cd ..

# ==========================================
# Summary
# ==========================================
echo ""
echo "=========================================="
echo "Profiling Complete!"
echo "=========================================="
echo ""
echo "Profile files saved in: ./profiles/"
echo ""
echo "To view detailed reports:"
echo "  ncu-ui profiles/gemm_tiled.ncu-rep"
echo "  ncu-ui profiles/mlp_fused.ncu-rep"
echo "  ncu-ui profiles/spmm_opt.ncu-rep"
echo ""
echo "To generate text reports:"
echo "  ncu --import profiles/gemm_tiled.ncu-rep --page details"
echo ""
echo "Key metrics to analyze:"
echo "  - SM Throughput (Compute Utilization)"
echo "  - DRAM Throughput (Memory Bandwidth)"
echo "  - Occupancy (Active Warps)"
echo "  - Stall Reasons (Memory/Instruction dependencies)"
echo ""
