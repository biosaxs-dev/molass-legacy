# coding: utf-8
"""
    QmmController.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from .PairedDataSets import PairedDataSets, get_denoise_rank_impl, get_num_components
from .QuadMM import QuadMM
from .QmmDialog import QmmDialog

class QmmController:
    def __init__(self, qmm, in_folder, sd, separately=True, exe_queue=None):
        self.qmm = qmm
        self.logger = logging.getLogger()
        self.dialog = None
        self.dialog_error = None
        self.exe_queue = exe_queue
        try:
            self.prepare_datasets(in_folder, sd, separately)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            self.logger.error(etb)
            raise RuntimeError("Error in QmmController.prepare_datasets", etb)

    def put_step_info(self, stepno):
        if self.exe_queue is not None:
            print('put_step_info', [0, stepno])
            self.exe_queue.put([0, stepno])

    def prepare_datasets(self, in_folder, sd, separately):
        self.put_step_info(1)
        kwargs = {}
        kwargs['sd'] = sd
        kwargs['lpm_correct'] = True
        n_components = 10
        kwargs['n_components'] = n_components
        self.datasets = datasets = PairedDataSets(in_folder, kwargs)

        ecurve = datasets.pair[1].ecurve
        denoise_rank = get_denoise_rank_impl(ecurve)
        set_setting('last_denoise_rank', denoise_rank)

        self.put_step_info(2)
        y_list, data_list = datasets.generate_sample_datasets(quad=True, bubble_care=True)
        self.put_step_info(3)

        X_list = [np.expand_dims(data,1) for data in data_list]
        bins_list = [len(y) for y in y_list]
        if separately:
            self.qmm.separate_fit(X=X_list, bins=bins_list)
        else:
            self.qmm.unified_fit(X=X_list, bins=bins_list)
        self.y_list = y_list
        self.data_list = data_list

        self.put_step_info(4)

    def prepare_dialog(self, parent):
        # must be called in the main thread
        try:
            self.dialog = QmmDialog(parent, self)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            self.logger.error(etb)
            self.dialog_error = etb
            raise RuntimeError("Error in QmmController.show_dialog", etb)      

    def show_dialog(self, parent):
        self.dialog.show()
        return self.dialog.applied

    def dialog_ready(self):
        if self.dialog_error is None:
            return self.dialog is not None
        else:
            # avoid the above waiting forever in test
            raise RuntimeError("Error in QmmController.show_dialog", self.dialog_error)

def get_qmm_controller(parent, mm_type, in_folder, n_components=None, max_iterations=None, sd=None):
    from ProgressMinDialog import run_with_progress

    """
    start loggin copied from Analyzer.do_analysis
    """
    if parent.testing:
        tester_info_log = ' with test pattern ' + str( parent.tester_info.test_pattern )
    else:
        tester_info_log = ''

    parent.analyzer.app_logger.info( "start analysis for " + in_folder + tester_info_log )

    if n_components is None:
        auto_n_components = get_setting('auto_n_components')
        if auto_n_components:
            if sd is None:
                n_components = 10
            else:
                n_components = get_num_components(sd.get_xray_curve())
        else:
            n_components = get_setting('n_components')

    if max_iterations is None:
        max_iterations = get_setting('max_iterations')

    qmm_separately = get_setting('qmm_separately')
    fixed_random_seeds = get_setting('fixed_random_seeds')

    def get_controller(exe_queue):
        if fixed_random_seeds:
            states = fixed_random_seeds
        else:
            states = np.random.randint(1000, 9999, 4)
        set_setting('last_random_seeds', states)
        qmm = QuadMM(mm_type, n_components, max_iter=max_iterations, random_states=states, anim_data=True)
        controller = QmmController(qmm, in_folder, sd, qmm_separately, exe_queue)
        return controller

    def on_return(ret):
        # ret: controller
        ret.prepare_dialog(parent)  # must be called in the main thread

    try:
        controller, error_info = run_with_progress(parent, get_controller, max_iter=5,
                                    title="QmmController", on_return=on_return)
    except:
        controller = None
    return controller
