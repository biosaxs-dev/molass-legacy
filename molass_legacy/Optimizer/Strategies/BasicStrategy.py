"""
    Optimizer.Strategies.BasicStrategy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

class BasicStrategy:
    def __init__(self, cycle=None):
        self.cycle = cycle

    def trust_initial_baseline(self):
        return True

    def baseline_first(self):
        return True

    def is_strategic(self, n):
        if self.cycle is None:
            return False
        if n % self.cycle == 1:
            return True
        return False