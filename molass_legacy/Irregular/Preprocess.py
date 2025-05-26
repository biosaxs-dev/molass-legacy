"""
    Preprocess.py

    Copyright (c) 2022,2025, SAXS Team, KEK-PF
"""
import numpy as np

NEGATIVE_RATIO_LIMIT = 0.7  # < 0.85 for 20220716/lowSN_BSA_01

def correct_flat_negative_baseline(y, debug=False):

    negetive_ratio = len(np.where(y < 0)[0])/len(y)
    if negetive_ratio > NEGATIVE_RATIO_LIMIT:
        import logging
        import molass_legacy.KekLib.DebugPlot as plt
        logger = logging.getLogger(__name__)
        logger.warning("temporarily correcting a flat negative_baseline due to negetive_ratio=%.3g", negetive_ratio)

        kth = int(len(y)*NEGATIVE_RATIO_LIMIT)
        kpp = np.argpartition(y, kth)
        base = np.average(y[kpp[0:kth]])

        if debug:
            from DataUtils import get_in_folder
            x = np.arange(len(y))
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("negative baseline fix for %s" % get_in_folder(), fontsize=16)
                ax.plot(x, y, label="data")
                xmin, xmax = ax.get_xlim()
                ax.set_xlim(xmin, xmax)
                ax.plot([xmin, xmax], [0, 0], ":", color="red")
                ax.plot([xmin, xmax], [base, base], color="red")
                fig.tight_layout()
                plt.show()

        y -= base

    return y
