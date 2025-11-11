"""
optimization_process.py
"""
import time
import numpy as np

# Simulated optimization process: writes progress to a file
def optimization_process(progress_file):
    with open(progress_file, "w") as f:
        for i in range(100):
            time.sleep(0.2)
            value = np.sin(i/10) + np.random.randn()*0.1
            f.write(f"{value}\n")
            f.flush()
            
if __name__== "__main__":
    import sys
    progress_file= sys.argv[1] if len(sys.argv) > 1 else "progress.txt"
    print("Starting optimization process, writing to", progress_file)
    optimization_process(progress_file)