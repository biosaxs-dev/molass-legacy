"""
    Simulative/SimulativeDemo.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import seaborn as sns
sns.set_theme()

def demo(parent=None):
    from importlib import reload
    import Simulative.LognormalPsd
    reload(Simulative.LognormalPsd)
    from Simulative.LognormalPsd import lognormalpore_model_interactive_impl
    """
    these parameter values were taken from 20190529_1
    """
    x = np.arange(330)
    N = 1000
    T = 0.7327
    t0 = -143.8
    me = 1.5
    mp = 1.5
    mu = 6.197
    sigma = 0.05
    scales = np.array([2.312, 10, 2.07, 0.6056])
    params = np.concatenate([[N, T, t0, me, mp, mu, sigma], scales])
    rgs = np.array([91.40615662, 67.23922297, 47.23719934, 35.73305675])
    lognormalpore_model_interactive_impl(x, None, params, rgs, use_ty_as_data=True, plot_mnp=True, window_title="Demo", parent=parent)

if __name__ == "__main__":
    import os
    import sys
    this_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(this_dir, '..'))
    demo()