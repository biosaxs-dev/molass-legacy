"""
    Optimizer.Strategies.SdmStrategy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from .BasicStrategy import BasicStrategy

class SdmStrategy(BasicStrategy):
    def __init__(self, **kwargs):
        self.cycle = 7
        self.nc = nc = kwargs.pop("nc", 3)   # excluding baseline

        # cycles 1-2
        xr_start = 0
        uv_start = nc*2 + 2 + 2
        mapping_start = nc*2 + 2
        mapping_indeces = [mapping_start, mapping_start+1]
        self.indeces_list = [list(range(start, start+nc)) + mapping_indeces for start in [xr_start, uv_start]]

        # cycle 3
        rg_start = nc + 2
        self.indeces_list.append(list(range(rg_start, rg_start+nc)))

        # cycle 4
        uv_baseline_start = nc*3 + 2 + 2
        self.indeces_list.append(list(range(uv_baseline_start, uv_baseline_start+7)))

        # cycles 5-6
        # 0, 1, 2,  3,        4,  5
        # N, K, x0, poresize, N0, tI = sdmcol_params
        cp_start = nc*3 + 2 + 7 + 2 + 2
        for list_ in [2, 3, 5], [0, 1, 4]:
            self.indeces_list.append([cp_start + i for i in list_])

    def is_strategic(self, n):
        if self.cycle is None:
            return False
        if n % self.cycle == 0:
            return False
        return True

    def get_indeces_list(self, n):
        r = n % self.cycle
        assert r > 0
        return [self.indeces_list[r-1]]