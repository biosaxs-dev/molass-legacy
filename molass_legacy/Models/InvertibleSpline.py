"""
    InvertibleSpline.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import UnivariateSpline

class InvertibleSpline(UnivariateSpline):
    def __init__(self, x, y, k=3, s=None, ext=0, debug=True):
        try:
            UnivariateSpline.__init__(self, x, y, k=k, s=s, ext=ext)
        except Exception as exc:
            # better to aviod this situation. namely, remove the cause which brought you here.

            import logging
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception

            logger = logging.getLogger(__name__)
            log_exception(logger, "InvertibleSpline: ", n=10)

            if len(x) in [2,3]:
                # as in 20161006/OA01/Backsub with cum_components=5
                k = len(x) - 1
                UnivariateSpline.__init__(self, x, y, k=k, s=s, ext=ext)
            else:
                if len(x) == 1:
                    raise ValueError("len(x) == 1")

                diff = x[1:] - x[:-1]
                illegal = np.where(diff <= 0)[0]
                print("x=", x)
                print("y=", y)
                print("illegal=", illegal)

                if debug:
                    with plt.Dp():
                        fig, ax = plt.subplots()
                        ax.set_title("InvertibleSpline debug plot")
                        ax.plot(x, y)
                        ax.plot(x[illegal], y[illegal], "o", color="red")
                        fig.tight_layout()
                        plt.show()
                raise exc
