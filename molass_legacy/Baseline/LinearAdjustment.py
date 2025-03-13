# coding: utf-8
"""
    Baseline.LinearAdjustment.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import set_setting

LA_MIN_SCI = 80

class LaJudge:
    def __init__(self, mapper):
        self.logger = logging.getLogger(__name__)
        self.mapper = mapper
        self.min_sci = np.min(mapper.get_sci_list())
        if False:
            from molass_legacy.Mapping.MappingUtils import debug_plot_mapping
            debug_plot_mapping(mapper)

    def ok(self):
        return self.min_sci >= LA_MIN_SCI

    def modify_opt_params(self):
        if not self.ok():
            opt_params = self.mapper.opt_params
            if opt_params['uv_baseline_adjust']:
                opt_params['uv_baseline_adjust'] = 0
                self.logger.info("UV baseline adjustment has been suppressed due to a low SCI %d.", int(self.min_sci))
                set_setting('uv_adjust_suppressed', 1)
                self.mapper.optimize(opt_params=opt_params)
