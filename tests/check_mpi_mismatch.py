import sys
import os
import subprocess

def get_shared_libs(module_path):
    try:
        if sys.platform.startswith('linux'):
            cmd = ['ldd', module_path]
        elif sys.platform == 'darwin':
            cmd = ['otool', '-L', module_path]
        else:
            return "Unknown platform"
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return str(e)

def check_mpi():
    print("=== Checking MPI Configuration ===")
    print(f"Python Executable: {sys.executable}")
    
    # Check mpi4py
    try:
        import mpi4py
        from mpi4py import MPI
        print(f"mpi4py version: {mpi4py.__version__}")
        print(f"mpi4py file: {mpi4py.__file__}")
        
        # Find the actual .so file for mpi4py
        mpi4py_dir = os.path.dirname(mpi4py.__file__)
        mpi4py_so = None
        for root, dirs, files in os.walk(mpi4py_dir):
            for f in files:
                if f.endswith('.so') and 'MPI' in f:
                    mpi4py_so = os.path.join(root, f)
                    break
            if mpi4py_so: break
            
        if mpi4py_so:
            print(f"mpi4py extension: {mpi4py_so}")
            print("--- mpi4py Dependencies ---")
            print(get_shared_libs(mpi4py_so))
        else:
            print("Could not find mpi4py extension .so")
            
    except ImportError:
        print("mpi4py not installed")

    # Check vbcsr
    try:
        import vbcsr
        import vbcsr_core
        print(f"vbcsr file: {vbcsr.__file__}")
        print(f"vbcsr_core file: {vbcsr_core.__file__}")
        
        print("--- vbcsr_core Dependencies ---")
        print(get_shared_libs(vbcsr_core.__file__))
        
    except ImportError as e:
        print(f"vbcsr import failed: {e}")

if __name__ == "__main__":
    check_mpi()
