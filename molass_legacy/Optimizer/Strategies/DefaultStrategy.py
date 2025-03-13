"""
    Optimizer.Strategies.DefaultStrategy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

class DefaultStrategy:
    def __init__(self):
        pass

    def trust_initial_baseline(self):
        return True

    def baseline_first(self):
        return True

    def is_strategic(self, n):
        return False