"""

    Rank.SRR.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF

"""
import numpy as np
from bisect import bisect_right
from SvdDenoise import get_denoised_data
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

class SRR:
    def __init__(self, sd):
        D, E, qv, ecurve = sd.get_xr_data_separate_ly()
        self.D = D
        self.E = E
        self.qv = qv
        self.i_smp = sd.xray_index
        self.ecurve = ecurve

    def compute_judge_info(self, figs_folder=None, analysis_name=None):
        ecurve = self.ecurve
        srr_list = []

        if figs_folder is not None:
            figs_prefix = analysis_name.replace("analysis", "fig")

        for k, peak_rec in enumerate(ecurve.get_major_peak_info()):
            print([k], peak_rec)
            start = peak_rec[0]
            stop = peak_rec[2]+1
            if start >= stop:
                # as in 20171226
                # or should be investigated for the cause
                continue

            eslice = slice(start, stop)
            D = self.D[:,eslice]
            D1_ = get_denoised_data(D, rank=1)
            D2_ = get_denoised_data(D, rank=2)
            srr1 = np.linalg.norm(D2_ - D)/np.linalg.norm(D1_ - D)

            c = self.D[self.i_smp,eslice]
            C1 = np.array([c])
            Cinv1 = np.linalg.pinv(C1)
            C2 = np.array([c, c**2])
            Cinv2 = np.linalg.pinv(C2)
            Dinv = np.linalg.pinv(D)
            E_ = self.E[:,eslice]

            sg_list = []
            rg_list = []
            qrg_list = []
            for D_, Cinv in [(D1_, Cinv1), (D2_, Cinv2)]:
                P = D_ @ Cinv
                W   = np.dot(Dinv, P)
                Pe  = np.sqrt(np.dot(E_**2, W**2))
                data = np.array([self.qv, P[:,0], Pe[:,0]]).T
                sg = SimpleGuinier(data)
                sg_list.append(sg)
                rg_list.append(max(10.0, sg.Rg))
                qrg_list.append(sg.min_qRg)

            # Q*rg = 2
            # Q = 2/rg
            rg_ = np.mean(rg_list)
            i = bisect_right(self.qv, 2.0/rg_)
            print("Rg=", rg_, "i=", i)
            aslice = slice(0, i)
            M = self.D[aslice,eslice]
            M1_ = get_denoised_data(M, rank=1)
            M2_ = get_denoised_data(M, rank=2)
            srr2 = np.linalg.norm(M2_ - M)/np.linalg.norm(M1_ - M)
            top_x = peak_rec[1]
            srr_list.append((top_x, srr1, srr2, *rg_list, *qrg_list))
            if figs_folder is not None:
                pass
                # to be implemented in SimpleGuinierDemo.py

        return srr_list
