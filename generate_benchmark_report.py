import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os

def run_benchmark(blocks):
    density = 200 / blocks
    if density * blocks**2 <= 1:
        density = 0.01

    import sys
    cmd = [
        sys.executable, "tests/benchmark_large_scale.py",
        "--blocks", str(blocks),
        "--min-block", "16",
        "--max-block", "20",
        "--density", str(density),
        "--scipy", "--mkl"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}:{os.getcwd()}/build"
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    output = result.stdout
    
    # Parse output
    vbcsr_time = float(re.search(r"VBCSR Average SpMV Time: ([\d\.]+) s", output).group(1))
    scipy_time = float(re.search(r"SciPy Average SpMV Time: ([\d\.]+) s", output).group(1))
    mkl_time = float(re.search(r"MKL Average SpMV Time: ([\d\.]+) s", output).group(1))
    
    scipy_speedup = float(re.search(r"Speedup \(SciPy / VBCSR\): ([\d\.]+)x", output).group(1))
    mkl_speedup = float(re.search(r"Speedup \(MKL / VBCSR\): ([\d\.]+)x", output).group(1))
    
    return vbcsr_time, scipy_time, mkl_time, scipy_speedup, mkl_speedup

def main():
    block_counts = [500, 1000, 5000, 10000, 20000]
    vbcsr_times = []
    scipy_times = []
    mkl_times = []
    scipy_speedups = []
    mkl_speedups = []
    
    print("Running benchmarks...")
    for b in block_counts:
        print(f"  Blocks: {b}...", end="", flush=True)
        vt, st, mt, ss, ms = run_benchmark(b)
        vbcsr_times.append(vt)
        scipy_times.append(st)
        mkl_times.append(mt)
        scipy_speedups.append(ss)
        mkl_speedups.append(ms)
        print(f" Done. SciPy Speedup: {ss:.2f}x, MKL Speedup: {ms:.2f}x")
        
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(block_counts))
    width = 0.25
    
    rects1 = ax.bar(x - width, scipy_times, width, label='SciPy CSR', color='tab:red', alpha=0.7)
    rects2 = ax.bar(x, mkl_times, width, label='MKL CSR', color='tab:blue', alpha=0.7)
    rects3 = ax.bar(x + width, vbcsr_times, width, label='VBCSR', color='tab:green', alpha=0.9)
    
    ax.set_xlabel('Number of Blocks')
    ax.set_ylabel('SpMV Time (s)')
    ax.set_title('VBCSR vs SciPy vs MKL SpMV Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(block_counts)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add speedup labels for VBCSR relative to others? Or just show times.
    # Let's annotate VBCSR bars with speedup vs SciPy and MKL
    # Actually, usually speedup is shown as text or separate plot.
    # Let's just put speedup numbers on top of VBCSR bar?
    # Or maybe just print them in table.
    # Let's put speedup vs SciPy on top of SciPy bar, and vs MKL on top of MKL bar?
    # No, speedup is usually X / VBCSR. So if SciPy is slower, speedup > 1.
    
    def autolabel(rects, speedups):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            ax.annotate(f'{speedups[i]:.1f}x',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    autolabel(rects1, scipy_speedups)
    autolabel(rects2, mkl_speedups)

    plt.tight_layout()
    plt.savefig('benchmark_performance.png')
    print("Saved benchmark_performance.png")
    
    # Generate Markdown Report
    with open("benchmark_report.md", "w") as f:
        f.write("# VBCSR Performance Benchmark\n\n")
        f.write("Comparison of Matrix-Vector Multiplication (SpMV) performance between `vbcsr`, `scipy.sparse.csr_matrix`, and `sparse_dot_mkl`.\n\n")
        f.write("## Test Configuration\n")
        f.write("- **Block Size**: Random [16, 20]\n")
        f.write("- **Density**: Adaptive (ensuring ~200 non-zeros per row)\n")
        f.write("- **Data Type**: float64\n")
        f.write("- **System**: Linux, MKL Backend\n\n")
        f.write("## Results\n\n")
        f.write("| Blocks | Approx Rows | VBCSR (s) | SciPy (s) | MKL (s) | SciPy Speedup | MKL Speedup |\n")
        f.write("|--------|-------------|-----------|-----------|---------|---------------|-------------|\n")
        for i, b in enumerate(block_counts):
            rows = b * 18 # Approx mean block size
            f.write(f"| {b} | ~{rows} | {vbcsr_times[i]:.4f} | {scipy_times[i]:.4f} | {mkl_times[i]:.4f} | **{scipy_speedups[i]:.2f}x** | **{mkl_speedups[i]:.2f}x** |\n")
        
        f.write("\n## Visualization\n\n")
        f.write("![Benchmark Plot](benchmark_performance.png)\n")

if __name__ == "__main__":
    main()
