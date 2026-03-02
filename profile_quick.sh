#!/bin/bash
# Quick Profiling - Essential Metrics Only
# For students who need basic occupancy and throughput data

echo "Quick Profiling - Essential Metrics"
echo "===================================="
echo ""

# Key metrics we care about:
# 1. Occupancy (how many warps are active)
# 2. Memory throughput (bandwidth utilization)
# 3. Compute throughput (SM utilization)

METRICS="sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__warps_active.avg.pct_of_peak_sustained_active"

# ==========================================
# EXP1: Tiled GEMM
# ==========================================
echo "Profiling GEMM Tiled Kernel..."
cd exp1_dgemm
ncu --metrics $METRICS \
    --csv \
    ./bin/dgemm --m 2048 --n 2048 --k 2048 --impl tiled --no-verify \
    | tee ../profile_gemm.csv
cd ..

# ==========================================
# EXP2: Fused MLP
# ==========================================
echo ""
echo "Profiling MLP Fused Kernel..."
cd exp2_mlp
ncu --metrics $METRICS \
    --csv \
    ./bin/dmlp --layers 1024,2048,1024 --batch 128 --impl activation_fused --no-verify \
    | tee ../profile_mlp.csv
cd ..

# ==========================================
# EXP3: Optimized SpMM
# ==========================================
echo ""
echo "Profiling SpMM Optimized Kernel..."
cd exp3_spmm
ncu --metrics $METRICS \
    --csv \
    ./spmm_opt \
    | tee ../profile_spmm.csv
cd ..

echo ""
echo "===================================="
echo "Results saved to:"
echo "  - profile_gemm.csv"
echo "  - profile_mlp.csv"
echo "  - profile_spmm.csv"
echo ""
echo "Use these metrics in your report!"
echo "===================================="
