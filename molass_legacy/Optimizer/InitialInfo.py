"""
    InitialInfo.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from .OptimizerSettings import OptimizerSettings

class PeProxy:
    def __init__(self, optinit_info, optimizer, init_params):
        from molass_legacy.Peaks.PeakEditor import get_default_maxnum_trials
        self.previous_optimizer_folder = get_setting('optimizer_folder')
        self.previous_rg_folder = os.path.join(self.previous_optimizer_folder, 'rg-curve')
        self.sd = optinit_info.sd
        self.treat = optinit_info.treat
        self.n_components = optimizer.n_components
        self.fullopt = optimizer
        self.drift_type = optinit_info.drift_type
        self.init_params = init_params
        self.n_iterations = optinit_info.n_iterations
        self.maxnum_trials = get_default_maxnum_trials(self.n_components - 1)   # for consistency, see PeakEditor.get_n_components()
        self.param_init_type = optinit_info.param_init_type

    def load_settings(self):
        # creating logger here because this method will be called first
        # not in the __init__ which is too early.
        self.logger = logging.getLogger(__name__)
        settings = OptimizerSettings(param_init_type=self.param_init_type)
        settings.load(optimizer_folder=self.previous_optimizer_folder)
        self.logger.info("previous settings will be used through PeProxy.")

    def get_sd(self):
        return self.sd

    def get_treat(self):
        return self.treat

    def get_dsets(self):
        from .FullOptInput import FullOptInput
        # note that the analysis_folder should have been renewed by the time of this call

        """ this is required later in the show_peak_editor_impl.
            to clarify this necessity,
            find where the equivalent mkdirs is done in the non-proxy normal process
        """
        optimizer_folder = os.path.join(get_setting('analysis_folder'), 'optimized')
        if False:
            if not os.path.exists(optimizer_folder):
                from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
                mkdirs_with_retry(optimizer_folder)

        if os.path.exists(self.previous_rg_folder):
            set_setting("trust_rg_curve_folder", True)

        set_setting("rg_curve_folder", self.previous_rg_folder)
        rg_folder = os.path.join(optimizer_folder, 'rg-curve')

        self.logger.info("optimizer_folder=%s", optimizer_folder)
        self.logger.info("previous_rg_folder=%s", self.previous_rg_folder)
        self.logger.info("new rg_folder=%s", rg_folder)

        # rg_folder will be expected to be copied from the previous_rg_folder

        sd = self.sd
        self.fullopt_input = FullOptInput(sd=sd, corrected_sd=None, rg_folder=rg_folder)
        self.dsets = self.fullopt_input.get_dsets(compute_rg=False)
        return self.dsets

    def get_drift_type(self):
        return self.drift_type

    def get_n_components(self):
        return self.n_components

    def get_seed(self):
        return np.random.randint(100000, 999999)

    def get_init_params(self):
        return self.init_params

    def get_function_class(self):
        class_ = self.fullopt.__class__
        return class_, class_.__name__

    def get_param_init_type(self):
        return self.param_init_type

    def get_n_iterations(self):
        return self.n_iterations

    def get_maxnum_trials(self):
        return self.maxnum_trials

class InitialInfo:
    def __init__(self, sd_copy=None, treat=None, pe=None, settings=None, result=None):

        if result is None:
            assert sd_copy is not None
            assert treat is not None
            assert pe is not None

            dsets = pe.get_dsets()
            dsets.relocate_rg_folder()

            drift_type = pe.get_drift_type()
            n_components = pe.get_n_components()
            n_iterations = pe.get_n_iterations()
            seed = pe.get_seed()
            init_params = pe.get_init_params()
            fullopt_class_not_used, class_code = pe.get_function_class()
            fullopt = pe.fullopt
            param_init_type = pe.get_param_init_type()
            maxnum_trials = pe.get_maxnum_trials()

            treat.save()

            settings = OptimizerSettings(param_init_type=param_init_type)
            settings.save()
 
            job_list = []
            composite = None
        else:
            # result
            sd_copy = result.sd_copy
            dsets = result.dsets
            drift_type = result.drift_type
            n_components = result.n_components
            n_iterations = result.n_iterations
            seed = result.seed
            init_params = result.init_params
            class_code = result.class_code
            fullopt = result.fullopt
            maxnum_trials = result.get_num_jobs()
            job_list = result.get_job_list()
            treat = result.treat
            settings = result.settings          # OptimizerSettings must be loaded before fullopt construction
                                                # which is done in FullOptResult

            param_init_type = settings.get("param_init_type")
            composite = result.composite

        self.treat = treat
        self.sd = sd_copy
        self.dsets = dsets
        self.drift_type = drift_type
        self.n_components = n_components
        self.init_params = init_params
        self.n_iterations = n_iterations
        self.seed = seed
        self.class_code = class_code            # str used in the optimizer
        self.fullopt = fullopt
        self.param_init_type = param_init_type
        self.bounds_type = settings.get("bounds_type")
        self.maxnum_trials = maxnum_trials
        self.job_list = job_list
        self.composite = composite
