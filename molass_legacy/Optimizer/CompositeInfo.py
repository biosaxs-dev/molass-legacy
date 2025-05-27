"""
    CompositeInfo.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Optimizer.OptLrfInfo import get_ratio_cy_list

def convert_for_display(composites):
    ret_list = []
    for comp in composites:
        ret_list.append([i+1 for i in comp])
    return ret_list

class CompositeInfo:
    def __init__(self, num_components=None, composites=None):
        # note that CompositeInfo does not incluse the baseline

        if composites is None:
            if num_components is None:
                pass
            else:
                assert type(num_components) is int
                composites = [[j] for j in range(num_components)]

        self.composites = composites    # note that this includes baseline
        if composites is None:
            # will be loaded later
            pass
        else:
            self.check_num_elements()
        self.separate_eoii = get_setting("separate_eoii")
        self.separate_eoii_type = get_setting("separate_eoii_type")
        self.separate_eoii_flags = get_setting("separate_eoii_flags")

    def check_num_elements(self):
        k = 0
        valid_indeces = []
        really_composite = False
        for comp in self.composites:
            if len(comp) > 1:
                really_composite = True
            for j, i in enumerate(comp):
                assert i == k
                if j == 0:
                    valid_indeces.append(k)
                k += 1
        self.num_elements = k 
        self.valid_indeces = np.array(valid_indeces[:-1])   # excluding baseline
        self.really_composite = really_composite

    def get_num_components(self):
        return len(self.composites)         # including the baseline

    def get_num_substantial_components(self):
        return len(self.composites) - 1     # excluding the baseline

    def get_valid_rgs(self, rgs):
        return rgs[self.valid_indeces]

    def get_composite_penalty(self):
        if self.really_composite:
            pass
        else:
            return 0

    def __str__(self):
        return str(self.composites)

    def get_extra_num_components(self):
        return 1 if self.separate_eoii else 0

    def save(self, path):
        with open(path, "w") as fh:
            fh.write(str(self.composites))

    def load(self, path):
        from molass_legacy.KekLib.EvalUtils import file_eval
        self.composites = file_eval(path)
        self.check_num_elements()

    def compute_C_matrix(self, y, cy_list, eoii=False, ratio_interpretation=False):
        # note that cy_list are non-composite curves

        work_list = []
        for comp in self.composites:
            for k, j in enumerate(comp):
                if k == 0:
                    cy = cy_list[j].copy()
                else:
                    cy += cy_list[j]
            work_list.append(cy)

        if ratio_interpretation:
            assert self.separate_eoii_type == 0
            cy_list = get_ratio_cy_list(y, cy_list)
            ty = y

        if eoii:
            ty = np.sum(cy_list[:-1], axis=0)   # excluding baseline
            if self.separate_eoii_type == 1:                
                work_list.append(ty**2)
            elif self.separate_eoii_type == 2:
                # not yet supported
                for k, flag in enumerate(self.separate_eoii_flags):
                    if flag:
                        work_list.append(cy_list[k]**2)

        C = np.array(work_list)
        return C

    def get_composite_ids(self):
        ids = []
        for k, comp in enumerate(self.composites):
            for i in comp:
                ids.append(k)
        return ids

    def get_composite_show_flags(self):
        flags = []
        for k, comp in enumerate(self.composites):
            for j, i in enumerate(comp):
                if j == 0:
                    flags.append(True)
                else:
                    flags.append(False)
        return flags

    def get_composite_proportions(self, proportions):
        ret_proportions = []
        for k, comp in enumerate(self.composites[:-1]):     # excluding baseline
            p = 0
            for j, i in enumerate(comp):
                if i < self.num_elements:
                    p += proportions[i]
            ret_proportions.append(p)
        return ret_proportions

    def get_elutions(self, cy_list, pv=None):
        if pv is None:
            pv = np.ones(len(cy_list))
        ret_list = []
        for comp in self.composites:
            for k, j in enumerate(comp):
                if k == 0:
                    cy = pv[j]*cy_list[j]
                else:
                    cy += pv[j]*cy_list[j]
            ret_list.append(cy)
        if len(ret_list) < len(cy_list):
            # temporary support for eoii components
            ret_list += cy_list[len(ret_list):]
        return ret_list

    def get_elements_rgs(self, rg_params):
        # this is deprecated replaced by get_valid_rgs

        ret_rgs = rg_params.copy()
        k = 0
        for comp in self.composites:
            n = len(comp)
            if n > 1:
                ret_rgs[k:k+n] = np.mean(rg_params[k:k+n])
            k += n
        return ret_rgs

    def estimate_peak_points(self, x, elutions):
        # use get_elutions method for elutions
        # do not use this method in optimization because it is slow

        n = self.get_num_substantial_components()
        peak_points = []
        for cy in elutions[:n]:
            j = np.argmax(cy)
            peak_points.append((x[j], cy[j]))
        peak_points = np.array(peak_points)
        return peak_points
