"""
    Optimizer.JobStateInfo.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import logging
import os
import numpy as np
from datetime import datetime

class JobStateInfo:
    def __init__(self, optinit_info, demo_index):
        self.logger = logging.getLogger(__name__)
        self.optinit_info = optinit_info
        self.dsets = optinit_info.dsets
        self.fullopt = optinit_info.fullopt
        self.demo_index = demo_index
        self.optlogfile = None

        if demo_index is not None:
            if demo_index == 0:
                # for run
                self.prepare_init_state()
            else:
                # for demo
                self.prepare_for_demo()
            self.curr_fv = self.demo_info[0]

    def get_ready_from_folder(self, work_folder):
        self.demo_index = -1
        self.dsets = self.get_dsets_from_work_folder(work_folder)
        self.prepare_init_state()
        self.curr_fv = self.demo_info[0]

    def get_dsets_from_work_folder(self, work_folder):
        from .TheUtils import get_work_info
        from .FullOptInput import FullOptInput
        # for demo to revive
        work_info = get_work_info(work_folder)
        fullopt_input = FullOptInput(in_folder=work_info.in_folder)
        return fullopt_input.get_dsets()

    def get_demo_info(self, work_folder):
        from .StateSequence import StateSequence
        sseq = StateSequence(work_folder=work_folder)
        return sseq.get_info()

    def prepare_for_demo(self):
        from .TheUtils import get_work_info_for_demo
        work_info = get_work_info_for_demo()
        self.logger.info("prepare_for_demo: real_bounds=%s" % work_info.real_bounds)
        self.set_init_info(init_params=work_info.init_params, real_bounds=work_info.real_bounds)
        self.demo_info = self.get_demo_info(work_info.work_folder)

    def get_optlogfile(self):
        if self.optlogfile is None:
            from molass_legacy._MOLASS.SerialSettings import get_setting
            from molass_legacy.Optimizer.OptLogFile import OptLogFile
            work_folder = get_setting("optworking_folder")
            path = os.path.join(work_folder, 'optimizer.log')
            self.optlogfile = OptLogFile(path)
        return self.optlogfile

    def get_active_indeces(self):
        optlogfile = self.get_optlogfile()
        return optlogfile.get_active_indeces()

    def prepare_init_state(self, init_params=None, real_bounds=None):
        self.set_init_info(init_params=init_params, real_bounds=real_bounds)
        self.demo_info = self.get_init_info()

    def set_init_info(self, init_params=None, real_bounds=None):
        if init_params is None:
            init_params = self.optinit_info.init_params

        self.fullopt.prepare_for_optimization(init_params, real_bounds=real_bounds)
        fv = self.fullopt.objective_func(init_params)
        fv_array = np.array([(0, fv, False, datetime.now())])   # should be compatible with .StateSequence.read_callback_txt_impl
        x_array = np.array([init_params])
        self.init_info = fv_array, x_array, 500000, 0

    def get_init_info(self):
        return self.init_info

    def set_demo_info(self, demo_info):
        self.demo_info = demo_info
        self.curr_fv = self.demo_info[0]
        self.draw_suptitle()
        i = self.get_best_index()
        self.best_index = i
        self.curr_index = i
        self.draw_state_impl(i)
        self.draw_progress()
