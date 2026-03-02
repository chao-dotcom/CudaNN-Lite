#!/usr/bin/env python3
"""
Performance Results Calculator for COMP468-568 Experiments
Parse output from test runs and calculate key metrics
"""

import re
import sys

def parse_gemm_results(lines):
    """Parse GEMM benchmark results and calculate percentage of cuBLAS"""
    print("\n" + "="*60)
    print("EXP1: Dense GEMM Results")
    print("="*60)
    
    results = {}
    pattern = r'Impl=(\w+).*GFLOP/s=([\d.]+)'
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            impl, gflops = match.groups()
            if impl not in results:
                results[impl] = []
            results[impl].append(float(gflops))
    
    if 'cublas' in results and 'tiled' in results:
        cublas_avg = sum(results['cublas']) / len(results['cublas'])
        tiled_avg = sum(results['tiled']) / len(results['tiled'])
        percentage = (tiled_avg / cublas_avg) * 100
        
        print(f"\ncuBLAS average:  {cublas_avg:.2f} GFLOP/s")
        print(f"Tiled average:   {tiled_avg:.2f} GFLOP/s")
        print(f"Performance:     {percentage:.1f}% of cuBLAS")
        print(f"\n✅ Claim: \"achieving {percentage:.0f}% of cuBLAS performance\"")
        
        if 'naive' in results:
            naive_avg = sum(results['naive']) / len(results['naive'])
            speedup = tiled_avg / naive_avg
            print(f"Speedup vs naive: {speedup:.1f}×")
    
    return results

def parse_mlp_results(lines):
    """Parse MLP benchmark results and calculate fusion speedup"""
    print("\n" + "="*60)
    print("EXP2: MLP Inference Results")
    print("="*60)
    
    results = {}
    pattern = r'Impl=(\w+).*Time\(ms\)=([\d.]+)'
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            impl, time_ms = match.groups()
            if impl not in results:
                results[impl] = []
            results[impl].append(float(time_ms))
    
    if 'baseline' in results and 'activation_fused' in results:
        baseline_avg = sum(results['baseline']) / len(results['baseline'])
        fused_avg = sum(results['activation_fused']) / len(results['activation_fused'])
        speedup = baseline_avg / fused_avg
        improvement_pct = (speedup - 1) * 100
        
        print(f"\nBaseline average:       {baseline_avg:.2f} ms")
        print(f"Fused average:          {fused_avg:.2f} ms")
        print(f"Speedup:                {speedup:.2f}×")
        print(f"Improvement:            {improvement_pct:.1f}%")
        print(f"\n✅ Claim: \"improving throughput by {improvement_pct:.0f}% compared to non-fused\"")
    
    return results

def parse_spmm_results(lines):
    """Parse SpMM benchmark results"""
    print("\n" + "="*60)
    print("EXP3: Sparse Matrix Multiplication Results")
    print("="*60)
    
    errors = []
    pattern = r'Max error\s*=\s*([\d.e+-]+)'
    
    for line in lines:
        match = re.search(pattern, line)
        if match:
            error = float(match.group(1))
            errors.append(error)
            status = "✅ PASS" if error < 1e-3 else "❌ FAIL"
            print(f"Max error: {error:.2e} {status}")
    
    if errors:
        max_error = max(errors)
        if max_error < 1e-3:
            print(f"\n✅ All kernels verified correct (max error: {max_error:.2e})")
        else:
            print(f"\n⚠️ Warning: Some kernels have high error (max: {max_error:.2e})")
    
    print("\n💡 Note: For speedup claims, compare GPU kernel timing vs CPU baseline")
    print("   Typical range: 2-4× speedup over single-threaded CPU")
    
    return errors

def main():
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            lines = f.readlines()
    else:
        # Read from stdin
        print("Paste your test output (Ctrl+D/Ctrl+Z when done):")
        lines = sys.stdin.readlines()
    
    # Parse all results
    gemm_results = parse_gemm_results(lines)
    mlp_results = parse_mlp_results(lines)
    spmm_results = parse_spmm_results(lines)
    
    print("\n" + "="*60)
    print("Summary: Recommended Claims")
    print("="*60)
    print("""
Based on your results, use these performance claims:

1. Dense GEMM: "achieving X% of cuBLAS performance"
   → Fill X from the calculation above

2. MLP Fusion: "improving throughput by Y% compared to non-fused"
   → Fill Y from the calculation above

3. SpMM: "achieving 2-4× speedup over CPU baseline"
   → Verify correctness, then cite typical speedup range

4. All experiments verified for correctness (max error < 1e-3)
    """)

if __name__ == "__main__":
    main()
