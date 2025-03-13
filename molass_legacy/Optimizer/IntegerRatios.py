"""
    IntegerRatios.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np

def determine_integer_ratios(mws):
    ratios = mws/mws[-1]

    rounded_ratios = [round(v) for v in ratios]
    print("ratios=", ratios)
    print("rounded_ratios=", rounded_ratios)

    return np.array(rounded_ratios)
