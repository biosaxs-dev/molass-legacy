"""
    Optimizer.ResultProxy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
from .BasicOptimizer import BasicOptimizer
from .FullOptResult import FullOptResult
from .OptJobInfo import OptJobInfo
from .FuncImporter import import_objective_function

class ResultProxy(FullOptResult):
    def __init__(self, folder, first_result):
        self.logger = logging.getLogger(__name__)
        self.folder = folder
        self.first_result = first_result
        self.optimizer = None
        self.first_optimizer = first_result.get_optimizer()
        self.dsets = first_result.dsets
        info = OptJobInfo()
        info.load(self.folder)
        self.n_components = info.nc

    def get_optimizer(self):
        if self.optimizer is None:
            composite = self.load_composite()
            log_file = self.get_log_file()
            class_code = self.get_class_code(log_file)
            optimizer_class = import_objective_function(class_code, self.logger)[0]
            first_optimizer = self.first_optimizer
            self.optimizer = optimizer_class(self.dsets, self.n_components,
                                uv_base_curve=first_optimizer.uv_base_curve,
                                xr_base_curve=first_optimizer.xr_base_curve,
                                qvector=first_optimizer.qvector,
                                wvector=first_optimizer.wvector,
                                composite=composite,
                                )
    
            self.fv_list, self.x_list, self.fv_array= self.load_callback_info()

            if class_code == 'G1100':
                from Estimators.SdmEstimatorProxy import SdmEstimatorProxy
                estimator = SdmEstimatorProxy(self.folder)
                self.optimizer.params_type.set_estimator(estimator)      # reconsider the neccesity of this line

            self.optimizer.debug_fv = False
            init_params = self.load_init_params()
            self.optimizer.prepare_for_optimization(init_params, minimally=True)

        return self.optimizer