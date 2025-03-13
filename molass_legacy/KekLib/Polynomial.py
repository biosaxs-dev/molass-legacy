"""
    Polynomial.py

    
"""
class poly1d:
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __call__(self, x):
        y = 0
        for c in self.coeffs:
            y = y*x + c
        return y
