"""
    QuickAnalysis.Homopeak.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import numpy as np

class Homopeakness:
    def __init__(self, predec):
        self.sd = predec.sd
        self.sd_copy = predec.sd_copy
        self.xr_curve = predec.xr_curve
        self.xr_peaks = predec.xr_peaks
        self.xr_proportions = predec.get_xr_proportions()
        self.recognize_peak_groups()

    def recognize_peak_groups(self):
        groups = []
        prop_indeces = []
        start = 0
        stop_points = self.xr_curve.boundaries + [len(self.xr_curve.x)]
        for i, sp in enumerate(stop_points):
            print([i], sp)
            members = []
            indeces = []
            for j, peak in enumerate(self.xr_peaks[start:]):
                # peal[1] : mu
                if peak[1] < sp:
                    members.append(peak)
                    indeces.append(start + j)
                else:
                    break
            groups.append(np.array(members))
            prop_indeces.append(indeces)
            start += len(members)
        self.groups = groups
        self.prop_indeces = prop_indeces

    def get_homopeakness_scores(self):
        from SimpleGuinier import SimpleGuinier
        D, E, qv, _ = self.sd_copy.get_xr_data_separate_ly()
        x0 = self.sd_copy.xr_j0
        print("sd.shape=", self.sd.conc_array.shape, "sd_copy.shape=", self.sd_copy.conc_array.shape,  "x0=", x0)
        for k, (pinfo, members, indeces) in enumerate(zip(self.xr_curve.peak_info, self.groups, self.prop_indeces)):
            props = self.xr_proportions[indeces]
            m = np.argmax(props)
            h, mu, sigma, tau = self.xr_peaks[indeces[m]]
            rgs = []
            mu_ = mu - x0
            for px in mu_, mu_-sigma, mu_+sigma:
                j = int(round(px))
                slice_ = slice(j-2, j+3)
                y = np.sum(D[:,slice_], axis=1)
                ye = np.sum(E[:,slice_], axis=1)
                sg = SimpleGuinier(np.array([qv,y,ye]).T)
                rgs.append(sg.Rg)
            print([k], (pinfo, members[:,1], indeces, props, m), rgs)
