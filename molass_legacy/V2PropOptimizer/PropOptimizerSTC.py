"""
    V2PropOptimizer.PropOptimizerSTC.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from time import sleep



def compute_optimal_proportion(progress_queue, job_args):
    for i in range(10):
        print([i])
        sleep(1)
        progress_queue.put([i])
    progress_queue.put([-1])
