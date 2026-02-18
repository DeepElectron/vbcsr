import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Set style for publication quality
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'sans-serif', # or 'serif'
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'savefig.bbox': 'tight'
})

def run_benchmark(blocks, mode="spmv", num_vecs=32):
    density = 200 / blocks
    density = min(density, 1.0)
    if density * blocks**2 <= 1:
        density = 0.01

    cmd = [
        sys.executable, "tests/benchmark_large_scale.py",
        "--blocks", str(blocks),
        "--min-block", "16",
        "--max-block", "20",
        "--density", str(density),
        "--scipy", "--mkl",
        "--mode", mode
    ]
    
    if mode == "spmm":
        cmd.extend(["--num-vecs", str(num_vecs)])
        
    env = os.environ.copy()
    # Do not override PYTHONPATH to ensure we use the installed vbcsr package
    # env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}:{os.getcwd()}/build"
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, check=True)
        output = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(e.stdout)
        print(e.stderr)
        return None, None, None, None, None

    # Parse output based on mode
    # Regex patterns depend on mode strings in benchmark_large_scale.py
    # Example: "VBCSR SpMV Average Time: 0.000126 s"
    vbcsr_match = re.search(r"VBCSR .+ Average Time: ([\d\.]+) s", output)
    scipy_match = re.search(r"SciPy .+ Average Time: ([\d\.]+) s", output)
    mkl_match = re.search(r"MKL .+ Average Time: ([\d\.]+) s", output)

    vbcsr_time = float(vbcsr_match.group(1)) if vbcsr_match else None
    scipy_time = float(scipy_match.group(1)) if scipy_match else 0.0
    mkl_time = float(mkl_match.group(1)) if mkl_match else float('inf')
    
    if vbcsr_time is None:
        print(f"Failed to parse VBCSR time. Output:\n{output[:500]}...")
        return None, None, None, None, None

    # Re-calculate speedups from times to be sure
    scipy_speedup = scipy_time / vbcsr_time if vbcsr_time > 0 else 0
    mkl_speedup = mkl_time / vbcsr_time if (vbcsr_time > 0 and mkl_time != float('inf')) else 0
    
    return vbcsr_time, scipy_time, mkl_time, scipy_speedup, mkl_speedup

def plot_performance(data, title, filename, xlabel='Number of Blocks'):
    x_vals = data['x']
    vbcsr_times = data['vbcsr']
    scipy_times = data['scipy']
    mkl_times = data['mkl']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(x_vals))
    width = 0.25
    
    # Plot bars
    ax.bar(x - width, scipy_times, width, label='SciPy CSR', color='#E63946', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x, mkl_times, width, label='MKL CSR', color='#457B9D', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.bar(x + width, vbcsr_times, width, label='VBCSR (Ours)', color='#2A9D8F', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel('Execution Time (s)', fontweight='bold')
    ax.set_title(title, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(x_vals)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    
    # Add speedup annotations for VBCSR
    for i in range(len(x_vals)):
        # Calculate speedups
        s_scipy = scipy_times[i] / vbcsr_times[i] if vbcsr_times[i] > 0 else 0
        s_mkl = mkl_times[i] / vbcsr_times[i] if (vbcsr_times[i] > 0 and mkl_times[i] != float('inf')) else 0
        
        # Annotation text
        txt = f"{s_scipy:.1f}x vs SciPy"
        if s_mkl > 0:
            txt += f"\n{s_mkl:.1f}x vs MKL"
            
        # Place centered above the group of bars
        max_height = max(vbcsr_times[i], scipy_times[i], mkl_times[i] if mkl_times[i] != float('inf') else 0)
        ax.annotate(txt,
                    xy=(x[i], max_height),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")

def main():
    block_counts = [500, 1000, 2000] # Kept small for quick execution in this context, can be increased
    
    # 1. SpMV Benchmark (Scaling Blocks)
    print("\n--- Running SpMV Benchmark ---")
    spmv_data = {'x': block_counts, 'vbcsr': [], 'scipy': [], 'mkl': []}
    for b in block_counts:
        print(f"  Blocks: {b}...", end="", flush=True)
        vt, st, mt, _, _ = run_benchmark(b, mode="spmv")
        if vt is None: continue
        spmv_data['vbcsr'].append(vt)
        spmv_data['scipy'].append(st)
        spmv_data['mkl'].append(mt)
        print(f" Done.")
    plot_performance(spmv_data, 'SpMV Performance: VBCSR vs SciPy vs MKL', 'benchmark_spmv.png', xlabel='Number of Blocks')

    # 2. SpMM Benchmark (Scaling Vectors K, fixed blocks)
    print("\n--- Running SpMM Benchmark (Scaling K) ---")
    fixed_blocks = 1000
    k_values = [16, 32, 64]
    spmm_data = {'x': k_values, 'vbcsr': [], 'scipy': [], 'mkl': []}
    for k in k_values:
        print(f"  K: {k}...", end="", flush=True)
        vt, st, mt, _, _ = run_benchmark(fixed_blocks, mode="spmm", num_vecs=k)
        if vt is None: continue
        spmm_data['vbcsr'].append(vt)
        spmm_data['scipy'].append(st)
        spmm_data['mkl'].append(mt)
        print(f" Done.")
    plot_performance(spmm_data, f'SpMM Performance (Blocks={fixed_blocks})', 'benchmark_spmm_k.png', xlabel='Number of RHS Vectors (K)')

    # 3. SpGEMM Benchmark (Scaling Blocks)
    print("\n--- Running SpGEMM Benchmark ---")
    spgemm_block_counts = [100, 300, 500]
    spgemm_data = {'x': spgemm_block_counts, 'vbcsr': [], 'scipy': [], 'mkl': []}
    for b in spgemm_block_counts:
        print(f"  Blocks: {b}...", end="", flush=True)
        vt, st, mt, _, _ = run_benchmark(b, mode="spgemm")
        if vt is None: continue
        spgemm_data['vbcsr'].append(vt)
        spgemm_data['scipy'].append(st)
        spgemm_data['mkl'].append(mt)
        print(f" Done.")
    plot_performance(spgemm_data, 'SpGEMM Performance (A * A)', 'benchmark_spgemm.png', xlabel='Number of Blocks')

    # Generate Markdown Report
    with open("benchmark_report.md", "w") as f:
        f.write("# VBCSR Performance Benchmarks\n\n")
        f.write("Comprehensive performance comparison of `vbcsr` against `scipy.sparse` and `sparse_dot_mkl`.\n\n")
        
        f.write("## 1. SpMV Performance\n")
        f.write("Matrix-Vector Multiplication (A * x).\n\n")
        f.write("![SpMV Benchmark](benchmark_spmv.png)\n\n")
        
        f.write("## 2. SpMM Performance\n")
        f.write(f"Sparse Matrix-Dense Matrix Multiplication (A * X), scaling with number of RHS vectors (K). Blocks={fixed_blocks}.\n\n")
        f.write("![SpMM Benchmark](benchmark_spmm_k.png)\n\n")
        
        f.write("## 3. SpGEMM Performance\n")
        f.write("Sparse Matrix-Sparse Matrix Multiplication (A * A).\n\n")
        f.write("![SpGEMM Benchmark](benchmark_spgemm.png)\n\n")

if __name__ == "__main__":
    main()
