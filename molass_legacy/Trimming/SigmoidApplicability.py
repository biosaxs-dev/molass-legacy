"""
    Trimming.SigmoidApplicability.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.BasicUtils import Struct

def check_applicability(x, y, debug=False):
    logger = logging.getLogger(__name__)

    # L ,x0, k, b, s1, s2
    safe_params = 0, 0, 1, 0, 0, 0

    if debug:
        from molass_legacy.Trimming.Sigmoid import ex_sigmoid
        y_ = ex_sigmoid(x, *safe_params)
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("check_applicability")
            ax.plot(x, y, label="data")
            ax.plot(x, y_, label="safe sigmoid")
            fig.tight_layout()
            plt.show()

    judge = False
    info = Struct(safe_params=safe_params)
    logger.warning("check_applicability: not implemented yet")
    return judge, info
