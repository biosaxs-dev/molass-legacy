"""
    RunPython.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import sys
import subprocess

class RunPythonResult:
    def __init__(self, result):
        self.returncode = result.returncode
        self.stdout = result.stdout.decode()
        self.stderr = result.stderr.decode()

def run_this_python(*args):
    result = subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        )
    return RunPythonResult(result)

def run_this_python_stdout_lines(*args):
    # reference  https://qiita.com/megmogmog1965/items/5f95b35539ed6b3cfa17
    proc = subprocess.Popen([sys.executable, *args],
            shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while True:
        line = proc.stdout.readline()
        if line:
            yield line
        if not line and proc.poll() is not None:
            break
