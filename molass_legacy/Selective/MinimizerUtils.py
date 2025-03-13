"""
    Selective/MinimizerUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from scipy.stats import linregress
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.GuinierAnalyzer.SimpleGuinierScore import compute_rg
from molass_legacy.DataStructure.AnalysisRangeInfo import shift_range_from_to_by_x
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ModelUtils import compute_cy_list

class RgComputer:
    def __init__(self, model, D, E, qv, x, y, compute_range, num_components, target_i):
        self.logger = logging.getLogger(__name__)
        self.logger.info("RgComputer has been created with model=%s, num_components=%d, target_i=%d" % (model.get_name(), num_components, target_i))
        self.model = model
        self.num_components = num_components
        self.M = get_denoised_data(D, rank=num_components)
        self.E = E
        self.qv = qv
        self.x = x
        self.y = y
        self.compute_range = compute_range
        self.target_i = target_i

    def compute_C(self, params_array):
        C = np.array(compute_cy_list(self.model, self.x, params_array))
        return C

    def compute_rg_list(self, C, return_sg=False, return_nI=False,
                        return_with_pos=False, discard_bad_rg=False,
                        gslice=None, debug=False):
        # C
        # P = MC⁺
        model = self.model
        M = self.M
        E = self.E
        qv = self.qv
        x = self.x
        compute_range = self.compute_range
        target_i = self.target_i
        
        if debug:
            y = self.y
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("compute_rg_list: C")
                ax.plot(x, y, color="orange")
                for k, cy in enumerate(C):
                    ax.plot(x, cy, ":", label="component-%d" % k)
                for k, (f, t) in enumerate(compute_range.get_fromto_list()):
                    print([k], f, t)
                    ax.axvspan(f, t, alpha=0.3, color="C%d" % k)
                ax.legend()
                fig.tight_layout()
                plt.show()

        if return_sg:
            sg_list = []
        else:
            assert gslice is not None
            qv2 = qv[gslice]**2

        rg_list = []
        if return_nI:
            assert gslice is not None
            nI_list = []
        for f, t in compute_range.get_fromto_list():
            f, t = shift_range_from_to_by_x(x, f, t)
            slice_ = slice(f,t+1)
            M_ = M[:,slice_]
            C_ = C[:,slice_]
            E_ = E[:,slice_]
            P_ = M_ @ np.linalg.pinv(C_)
            if return_sg:
                Minv = np.linalg.pinv(M_)
                W = Minv @ P_
                Pe = np.sqrt(E_**2 @ W**2)
                data = np.array([qv, P_[:,target_i], Pe[:,target_i]]).T
                if debug:
                    gslice = slice(0, len(qv)//4)
                    qv_ = qv[gslice]
                    iv_ = P_[gslice,target_i]
                    with plt.Dp():
                        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5))
                        ax1.set_title("Log Plot in %s" % ((f,t),))
                        ax1.set_yscale("log")
                        ax1.plot(qv_, iv_)
                        ax2.set_title("Guinier Plot in %s" % ((f,t),))
                        ax2.plot(qv_**2, np.log(iv_))
                        fig.tight_layout()
                        plt.show()
                sg = SimpleGuinier(data)
                if discard_bad_rg:
                    if sg.Rg is None or sg.Rg < 10:
                        continue
                sg_list.append(sg)
                rg = sg.Rg
            else:
                slope, intercept = linregress(qv2, np.log(P_[gslice,target_i]))[0:2]
                rg = compute_rg(slope)
                if return_nI:
                    nI_list.append(P_[:,target_i]/np.exp(intercept))
            if return_with_pos:
                pos = (t+f)/2   # be aware that this is a "shifted" position
                rg_list.append((pos, rg))
            else:
                rg_list.append(rg)

        if return_sg:
            return rg_list, sg_list
        else:
            if return_nI:
                return rg_list, nI_list
            else:
                return rg_list
