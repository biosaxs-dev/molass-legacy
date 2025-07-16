"""
    LrfInfo.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF
"""
import numpy as np
from bisect import bisect_right
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier

class LrfInfo:
    def __init__(self, opt_info, A, lrfE, debug=False):
        self.num_iterations = opt_info[0]
        nth, cdl_list = opt_info[3]
        self.need_bq_ = cdl_list[nth] > 1
        qv = opt_info[4]
        if debug:
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
            def log_shape(name, data):
                if isinstance(data, np.ndarray):
                    return f"{name}.shape={data.shape}"
                else:
                    return f"len({name})={len(data)}"
            qv_shape = log_shape("qv", qv)
            A_shape = log_shape("A", A)
            lrfE0_shape = log_shape("lrfE0", lrfE[0])
            self.logger.info("LrfInfo.__init__() with %s, %s, %s", qv_shape, A_shape, lrfE0_shape)
            print(f"LrfInfo.__init__() with {qv_shape}, {A_shape}, {lrfE0_shape}") # for jupyter output
        self.boundary = opt_info[5]
        self.boundary_j = None if self.boundary is None else bisect_right(qv, self.boundary)
        self.data = opt_info[6]

        A_data = np.array( [qv, A, lrfE[0]] ).T

        sg = SimpleGuinier(A_data)
        self.Rg = Rg = sg.Rg
        self.sg = sg
        self.basic_quality = sg.basic_quality
        if debug:
            Rg_ = str(Rg) if Rg is None else "%.3g" % Rg
            self.logger.info("Rg=%s with basic_quality=%.3g", Rg_, self.basic_quality)

    def need_bq(self):
        return self.need_bq_
