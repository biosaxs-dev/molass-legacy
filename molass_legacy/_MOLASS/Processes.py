"""
    _MOLASS/Processes.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

processes = []

def register_process(process):
    processes.append(process)

def remove_process(process):
    process.join()
    processes.remove(process)

def terminate_all_processes():
    for process in processes:
        # task: do this more politely
        process.terminate()