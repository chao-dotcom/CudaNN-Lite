#!/bin/bash
# Comprehensive testing script for all experiments
# Run this on a system with CUDA installed

set -e  # Exit on error

echo "=========================================="
echo "Testing COMP468-568 Experiments"
echo "=========================================="
echo ""

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

nvidia-smi
echo ""

# ==========================================
# EXP1: Dense GEMM
# ==========================================
echo "=========================================="
echo "EXP1: Testing Dense GEMM"
echo "=========================================="
cd exp1_dgemm
make clean && make

echo ""
echo "--- Small matrix (512x512x512) ---"
./bin/dgemm --m 512 --n 512 --k 512 --impl naive
./bin/dgemm --m 512 --n 512 --k 512 --impl tiled
./bin/dgemm --m 512 --n 512 --k 512 --impl cublas

echo ""
echo "--- Medium matrix (1024x1024x1024) ---"
./bin/dgemm --m 1024 --n 1024 --k 1024 --impl naive
./bin/dgemm --m 1024 --n 1024 --k 1024 --impl tiled
./bin/dgemm --m 1024 --n 1024 --k 1024 --impl cublas

echo ""
echo "--- Large matrix (2048x2048x2048) ---"
./bin/dgemm --m 2048 --n 2048 --k 2048 --impl tiled
./bin/dgemm --m 2048 --n 2048 --k 2048 --impl cublas

echo ""
echo "TIP: Calculate percentage as (tiled_GFLOPS / cublas_GFLOPS) * 100"
echo ""

cd ..

# ==========================================
# EXP2: MLP Inference
# ==========================================
echo "=========================================="
echo "EXP2: Testing MLP Inference"
echo "=========================================="
cd exp2_mlp
make clean && make

echo ""
echo "--- ReLU activation (baseline vs fused) ---"
./bin/dmlp --layers 1024,2048,1024 --batch 128 --activation relu --impl baseline
./bin/dmlp --layers 1024,2048,1024 --batch 128 --activation relu --impl activation_fused

echo ""
echo "--- GELU activation (baseline vs fused) ---"
./bin/dmlp --layers 1024,2048,1024 --batch 256 --activation gelu --impl baseline
./bin/dmlp --layers 1024,2048,1024 --batch 256 --activation gelu --impl activation_fused

echo ""
echo "TIP: Calculate speedup as (fused_time / baseline_time) - this should be 1.2-1.4x"
echo ""

cd ..

# ==========================================
# EXP3: Sparse Matrix Multiplication
# ==========================================
echo "=========================================="
echo "EXP3: Testing Sparse Matrix Multiplication"
echo "=========================================="
cd exp3_spmm
make clean && make

echo ""
echo "--- Baseline (row-per-thread) ---"
./spmm_baseline

echo ""
echo "--- Optimized (warp-collaborative) ---"
./spmm_opt

echo ""
echo "TIP: Both should show 'Max error = 0' or very small (<1e-5)"
echo ""

cd ..

# ==========================================
# Summary
# ==========================================
echo "=========================================="
echo "Testing Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review the GFLOP/s and timing results above"
echo "2. Calculate performance percentages"
echo "3. Run Nsight Compute for detailed profiling:"
echo "   ncu -o profile ./bin/dgemm --m 2048 --n 2048 --k 2048 --impl tiled"
echo "4. Analyze with: ncu-ui profile.ncu-rep"
echo ""
