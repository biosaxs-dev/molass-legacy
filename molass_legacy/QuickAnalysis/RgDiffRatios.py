"""
    QuickAnalysis.RgDiffRatios.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from SimpleGuinier import SimpleGuinier
from SvdDenoise import get_denoised_data

def compute_rgr_legacy_impl(qv, M, E, P1, A1, A2):
    # called from Conc.ConcDepend

    Minv = np.linalg.pinv(M)
    W = np.dot(Minv, P1)
    Pe  = np.sqrt(np.dot(E**2, W**2))
    Ae = Pe[:,0]

    sg1 = SimpleGuinier(np.array([qv, A1, Ae]).T)
    sg2 = SimpleGuinier(np.array([qv, A2, Ae]).T)
    Rg1 = sg1.Rg
    Rg2 = sg2.Rg

    if Rg1 is None or Rg1 == 0 or Rg2 is None or Rg2 == 0:
        rdr = 0
    else:
        rdr = (Rg1 - Rg2)*2/(Rg1 + Rg2)

    return rdr

RDR12_COMPUTABLE_RDRLR_LIMIT = 0.01     # < 

class RgDiffRatios:
    def __init__(self, sd, x_curve):
        self.logger = logging.getLogger(__name__)
        self.sd = sd
        self.paired_ranges = x_curve.get_default_editor_ranges()

    def compute_rg_quartets(self, debug=False):
        D, E, qv, xr_curve = self.sd.get_xr_data_separate_ly()

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.Elution.CurveUtils import simple_plot
            print("paired_ranges=", self.paired_ranges)
            with plt.Dp():
                fig, ax = plt.subplots()
                simple_plot(ax, xr_curve)
                fig.tight_layout()
                plt.show()

        x = xr_curve.x
        y = xr_curve.y
        pair_list = []
        for k, erec in enumerate(self.paired_ranges):
            print([k], erec)
            rg_pairs = []
            for f, t in erec:
                eslice = slice(f,t+1)
                D_ = D[:,eslice]
                E_ = E[:,eslice]
                c = y[eslice]
                c = c/np.max(c)
                M1 = get_denoised_data(D_, rank=1)
                M2 = get_denoised_data(D_, rank=2)
                C1 = np.array([c])
                C2 = np.array([c, c**2])
                C1inv = np.linalg.pinv(C1)
                C2inv = np.linalg.pinv(C2)
                P1 = M1 @ C1inv
                P2 = M2 @ C2inv

                Minv = np.linalg.pinv(D_)
                W = np.dot(Minv, P1)
                Pe  = np.sqrt(E_**2 @ W**2)
                Ae = Pe[:,0]

                A1 = P1[:,0]
                A2 = P2[:,0]

                sg1 = SimpleGuinier(np.array([qv, A1, Ae]).T)
                sg2 = SimpleGuinier(np.array([qv, A2, Ae]).T)
                Rg1 = sg1.Rg
                Rg2 = sg2.Rg
                if Rg1 is None:
                    Rg1 = np.nan
                if Rg2 is None:
                    Rg2 = np.nan
                rg_pairs.append((Rg1,Rg2))
            pair_list.append(np.array(rg_pairs).T)

        return pair_list

    def get_rank_hints(self):
        hints = []
        for k, quartet in enumerate(self.compute_rg_quartets()):
            if len(quartet[0,:]) == 1:
                # as in 20200630_5
                self.logger.info("%dth rdr is not computable due to a single range %s", k, str(self.paired_ranges))
                computable = False
                rdr12 = np.nan
            else:
                rgL, rgR = quartet[0,:]
                rdrLR = (rgL - rgR)*2/(rgL + rgR)
                print([k], quartet, rdrLR)
                computable = abs(rdrLR) < RDR12_COMPUTABLE_RDRLR_LIMIT
                if computable:
                    rg1 = np.average(quartet[0,:])
                    rg2 = np.average(quartet[1,:])
                    rdr12 = (rg1 - rg2)*2/(rg1 + rg2)
                else:
                    self.logger.info("%dth rdr is not computable due to rdrLR=%.3g >= %.3g", k, rdrLR, RDR12_COMPUTABLE_RDRLR_LIMIT)
                    rdr12 = np.nan
            hints.append((computable, rdr12))
        return hints

def spike(caller):
    from time import time

    t0 = time()

    sd = caller.judge_holder.sd
    paired_ranges = caller.mapper.x_curve.get_default_editor_ranges()

    rdrs = RgDiffRatios(sd, paired_ranges)
    hints = rdrs.get_rank_hints()
    print("hints=", hints)

    print("it took ", time() - t0)
