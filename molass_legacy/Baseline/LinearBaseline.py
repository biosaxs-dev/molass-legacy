"""
    LinearBaseline.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

USE_END_PARAMS = False
if not USE_END_PARAMS:
    from .Constants import SLOPE_SCALE

class LinearBaseline:
    def __init__(self, x=None, y=None, debug=False):
        if debug:
            from importlib import reload
            import Baseline.Baseline
            reload(Baseline.Baseline)
        from .Baseline import compute_baseline

        if x is None:
            return

        yb, (a,b) = compute_baseline(y, x=x, return_params=True, debug=debug)
        self.yb = yb
        self.params = [a*SLOPE_SCALE, b]
        self.x1 = x[0]
        self.x2 = x[-1]
        self.end_params = yb[[0,-1]]

    def __call__(self, x, params, y_, cy_list):
        # y_ is not used in LinearBaseline
        # cy_list is not used in LinearBaseline

        if USE_END_PARAMS:
            """
            (y2 - y1) = k*(x2 - x1)
            k = (y2 - y1)/(x2 - x1)

            (y - y1) = k*(x - x1)
            y = y1 + k*(x - x1)
            """
            y1, y2 = params
            k = (y2 - y1)/(self.x2 - self.x1)
            return y1 + k*(x - self.x1)
        else:
            slope, intercept = params
            return x*slope/SLOPE_SCALE + intercept
