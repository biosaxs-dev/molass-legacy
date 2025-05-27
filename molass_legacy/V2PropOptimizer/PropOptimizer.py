"""
    V2PropOptimizer.PropOptimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from time import sleep
import numpy as np
from scipy import stats
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from molass_legacy.GuinierAnalyzer.SimpleGuinierScore import compute_rg

PROP_MIN_RATIO = 0.1
PROP_MIN_VALUE = 1e-5

def compute_range_rgs(qv, D, E, paired_ranges, C, return_sgs=False, region_info=None):
    assert C.shape[0] == 2
    assert len(paired_ranges) == 1
    cmax = np.max(C, axis=1)
    cprop = cmax/np.sum(cmax)
    i = np.argmin(cprop)        # minor component

    if return_sgs:
        ret_sgs = []
    else:
        ret_rgs = []

    for f, t in paired_ranges[0].get_fromto_list():
        slice_ = slice(f, t+1)
        D_ = D[:,slice_]
        E_ = E[:,slice_]
        C_ = C[:,slice_]
        cmax = np.max(C_, axis=1)
        cprop = cmax/np.sum(cmax)
        # print("cmax=", cmax)
        # print("cprop=", cprop)
        if cprop[i] < 1e-3:
            C_ = C_[1,:].reshape((1, C_.shape[1]))
            j = 0
        else:
            j = 1
        Cinv = np.linalg.pinv(C_)
        P_ = D_ @ Cinv
        Minv = np.linalg.pinv(D_)
        W = np.dot(Minv, P_)
        Pe  = np.sqrt(E_**2 @ W**2)
        data = np.array([qv, P_[:,j], Pe[:,j]]).T

        if region_info is None:
            sg = SimpleGuinier(data)
            if return_sgs:
                ret_sgs.append(sg)
            else:
                ret_rgs.append(sg.Rg)
        else:
            assert not return_sgs

            region, qv2 = region_info
            x_ = qv2[region]
            y_ = np.log(P_[region,j])
            slope = stats.linregress(x_, y_)[0]
            rg = compute_rg(slope)
            ret_rgs.append(rg)

    if return_sgs:
        return ret_sgs
    else:
        return ret_rgs

class RangeRgComputer:
    def __init__(self, qv, D, E, paired_ranges):
        self.return_sgs = True
        self.region_list = []
        self.fixed_params = (qv, D, E, paired_ranges)
        self.qv2 = qv**2

    def compute(self, C, debug=False):
        if self.return_sgs:
            ret_sgs = compute_range_rgs(*self.fixed_params, C, return_sgs=True)
            ret_rgs = [sg.Rg for sg in ret_sgs]
            regions = [(sg.guinier_start, sg.guinier_stop) for sg in ret_sgs]
            self.region_list.append(regions)
            if len(self.region_list) == 20:
                region_array = np.array(self.region_list)
                start = int(np.mean(region_array[:,1,0]))
                stop = int(np.mean(region_array[:,1,1]))
                if debug:
                    import molass_legacy.KekLib.DebugPlot as plt
                    with plt.Dp():
                        fig, ax = plt.subplots()
                        ax.set_title("region_list debug")
                        region0 = region_array[:,0,:]
                        region1 = region_array[:,1,:]
                        ax.plot(*region0.T, "o")
                        ax.plot(*region1.T, "o")
                        ax.plot(start, stop, "o", color="red")
                        fig.tight_layout()
                        plt.show()
                self.return_sgs = False
                self.region = slice(start, stop)
        else:
            ret_rgs = compute_range_rgs(*self.fixed_params, C, region_info=[self.region, self.qv2])
        return ret_rgs
