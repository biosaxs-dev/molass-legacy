"""

    Optimize.ElutionComposer.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
import numpy as np
import logging
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import get_setting

COMPOSER_CB_TEXT = "try reducing number of components using composite models"

def make_composites_from_deviations(devs):
    allow = get_setting("identification_allowance")

    composites = []
    comp = []
    for j in range(len(devs)):
        comp.append(j)
        if devs[j] < allow:
            pass
        else:
            composites.append(comp)
            comp = []

    comp.append(j+1)
    composites.append(comp)
    return composites

class ElutionComposer:
    def __init__(self, optimizer, params, sd):
        self.logger = logging.getLogger(__name__)
        self.n_components = optimizer.n_components
        self.make_judge_info(optimizer, params, sd)

    def make_judge_info(self, optimizer, params, sd, debug=False):
        from Kratky.GuinierKratkyInfo import GuinierKratkyInfo

        lrf_info = optimizer.objective_func(params, return_lrf_info=True)

        gk_info = GuinierKratkyInfo(optimizer, params, lrf_info)
        devs = gk_info.compute_adjacent_deviation_ratios()

        if debug:
            print("devs=", devs)
            qrgnys = gk_info.qrgnys
            qrgs = gk_info.qrgs
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("Kratky Plot in ElutionComposer debug")

                for k, (qrg, qrgny) in enumerate(zip(qrgs, qrgnys)):
                    ax.plot(qrg, qrgny, 'o', markersize=1, label="component-%d" % (k+1))

                ax.legend()
                fig.tight_layout()
                plt.show()

        self.composites = make_composites_from_deviations(devs) + [[self.n_components - 1]]     # add baseline

    def reducible(self):
        return np.max([len(comp) for comp in self.composites]) > 1

    def make_composite(self):
        from .CompositeInfo import CompositeInfo
        composite = CompositeInfo(composites=self.composites)
        return composite
