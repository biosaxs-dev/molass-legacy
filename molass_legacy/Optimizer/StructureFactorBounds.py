"""
    StructureFactorBounds.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting

class StructureFactorBounds:
    def __init__(self, qv, lrf_info, gk_info):
        self.qv = qv
        rg = np.mean(gk_info.rgs)
        self.R = np.sqrt(5/3)*rg            # hard sphere raidius
        L = self.estimate_L(lrf_info)
        bound = 1/(self.qv*self.R*L)**2
        self.zv = np.zeros(len(qv))
        self.bounds = (-bound, bound)

    def get_bounds(self):
        return self.bounds

    def estimate_L(self, lrf_info):
        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices
        cv = np.max(Cxr[:-1], axis=1)
        n = np.argmax(cv)
        print("cv=", cv, "n=", n)
        self.n = n
        self.c = cv[n]
        self.eoii_pv = Pxr[-1]
        sf_bound_ratio = get_setting("sf_bound_ratio")
        if sf_bound_ratio < 0:
            # estimate any way
            sf_bound_ratio = 1      # temporarily estimated value
        return sf_bound_ratio

    def compute_penalty(self, Pxr, debug=False):
        penalty = 0
        bq = Pxr[:,-1]
        if debug:
            bq_bounds_list = []
        for aq in Pxr.T[:-1]:
            aq_ = np.abs(aq)
            penalty = np.mean(np.min([self.zv, bq - self.bounds[0]*aq_], axis=0)**2) + np.mean(np.max([self.zv, bq - self.bounds[1]*aq_], axis=0)**2)
            if debug:
                bq_bounds_list.append(self.bounds[0]*aq_)
                bq_bounds_list.append(self.bounds[1]*aq_)

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            qv = self.qv
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.plot(qv, bq)
                for bv in bq_bounds_list:
                    ax.plot(qv, bv)
                plt.show()

        return penalty

    def compute_bounded_bq(self, Pxr):
        bq = Pxr[:,-1].copy()
        aq = np.abs(Pxr[:,self.n])
        lower_bound = self.bounds[0]*aq
        where_exceeded = bq < lower_bound
        bq[where_exceeded] = lower_bound[where_exceeded]
        upper_bound = self.bounds[1]*aq
        where_exceeded = bq > upper_bound
        bq[where_exceeded] = upper_bound[where_exceeded]
        return bq
