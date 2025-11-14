"""
Optimizer.JobState.py
"""
import os
import logging
import numpy as np
import time
from .StateSequence import read_callback_txt_impl
from .OptimizerUtils import get_method_name

class JobState:
    def __init__(self, cb_file, niter=20):
        self.logger = logging.getLogger(__name__)
        self.cb_file = cb_file
        self.solver_name = get_method_name()
        self.niter = niter
        self.last_mod_time = None
        self._has_changed = False
        self.fv = np.array([])
        self.x = np.array([])
        self.xmax = None
        self.time_created = time.time()

    def has_changed(self):
        return self._has_changed

    def get_plot_info(self):
        return self.fv, self.xmax, self.x

    def estimate_xmax(self, fv_list):
        counter = fv_list[-1][0] if len(fv_list) > 0 else 0
        if counter == 0:
            # init state
            xmax = 500000*self.niter//100
        else:
            xmax = int(counter * (self.niter+1)/len(fv_list))
        return xmax

    def update(self, debug=True):
        if not os.path.exists(self.cb_file):
            t = time.time() - self.time_created
            if t > 10:
                self.logger.warning("callback.txt file not found: %s", self.cb_file)
            return
        
        mod_time = os.path.getmtime(self.cb_file)
        if self.last_mod_time is not None and mod_time == self.last_mod_time:
            self._has_changed = False
            return  # No changes
        
        self.last_mod_time = mod_time
        self._has_changed = True

        fv_list, x_list = read_callback_txt_impl(self.cb_file)

        if debug:
                self.logger.info("updating information from %s, len(x_list)=%d", self.cb_file, len(x_list))

        self.fv = np.array(fv_list)
        self.x = np.array(x_list)

        if self.solver_name == "ultranest":
            from molass_legacy.Solvers.UltraNest.SolverUltraNest import get_max_ncalls
            # task: unify this estimation
            xmax = get_max_ncalls(self.niter)
        else:
            xmax = self.estimate_xmax(fv_list)

        self.xmax = xmax