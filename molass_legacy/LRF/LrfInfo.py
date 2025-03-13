"""
    LrfInfo.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from SimpleGuinier import SimpleGuinier

DEBUG = False
if DEBUG:
    import logging

class LrfInfo:
    def __init__(self, opt_info, A, lrfE):
        if DEBUG:
            self.logger = logging.getLogger(__name__)
        self.num_iterations = opt_info[0]
        nth, cdl_list = opt_info[3]
        self.need_bq_ = cdl_list[nth] > 1
        qv = opt_info[4]
        self.boundary = opt_info[5]
        self.boundary_j = None if self.boundary is None else bisect_right(qv, self.boundary)
        self.data = opt_info[6]

        A_data = np.array( [qv, A, lrfE[0]] ).T

        sg = SimpleGuinier(A_data)
        self.Rg = Rg = sg.Rg
        self.sg = sg
        self.basic_quality = sg.basic_quality
        if DEBUG:
            Rg_ = str(Rg) if Rg is None else "%.3g" % Rg
            self.logger.info("Rg=%s with basic_quality=%.3g", Rg_, self.basic_quality)

    def need_bq(self):
        return self.need_bq_
