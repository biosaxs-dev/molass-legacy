"""
    Selective.PropOptimizer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from time import sleep
from importlib import import_module, reload
from molass_legacy.Models.ElutionCurveModels import EGH, EMG, EGHA, EMGA
from molass_legacy.Models.RateTheory.EDM import EDM
from molass_legacy.Models.Stochastic.Tripore import Tripore
from .PropOptimizerImpl import AFFINE

IMPL_DICT = {
    'EGH' : 'Selective.PropOptimizerEGH',
    'EMG' : 'Selective.PropOptimizerEMG',
    'EDM' : 'Selective.PropOptimizerEDM',
    'STC' : 'Selective.PropOptimizerSTC',
    }

if AFFINE:
    # this is consistent with traditional V1 models
    MODEL_DICT = {
        'EGH' : EGHA(),
        'EMG' : EMGA(),
        'EDM' : EDM(delayed=True),
        'STC' : Tripore(delayed=True),
        }
else:
    MODEL_DICT = {
        'EGH' : EGH(),
        'EMG' : EMG(),
        'EDM' : EDM(),
        'STC' : Tripore(),
        }

def load_func(modelname, funcname, devel=True):
    classname = IMPL_DICT.get(modelname)
    assert classname is not None

    module = import_module(classname)
    if devel:
        reload(module)
    func = getattr(module, funcname)    # from Selective.PropOptimizerXXX import funcname
    return func

class PropOptimizer:
    def __init__(self, modelname, x, y):
        self.guess_init_params_impl = load_func(modelname, 'guess_init_params')
        self.compute_props_impl = load_func(modelname, 'compute_props')
        self.compute_cy_list_impl = load_func(modelname, 'compute_cy_list')
        self.optimize_to_props_impl = load_func(modelname, 'optimize_to_props')
        self.model = MODEL_DICT.get(modelname)
        self.x = x
        self.y = y
        self.init_params = self.guess_init_params_impl(x, y)

    def get_init_params(self):
        return self.init_params

    def compute_props(self, params):
        return self.compute_props_impl(self.x, params)

    def compute_cy_list(self, params):
        return self.compute_cy_list_impl(self.x, params)

    def optimize(self, *o_args, init_params=None):
        if init_params is None:
            init_params = self.init_params
        return self.optimize_to_props_impl(self.x, self.y, init_params, *o_args)

    def make_xr_opt_recs(self, fx, y, peaks):
        return self.model.make_xr_opt_recs(fx, y, peaks)

    def make_uv_opt_recs(self, fx, uv_y, peaks, scale):
        return self.model.make_uv_opt_recs(fx, uv_y, peaks, scale)

def compute_optimal_proportion_impl(progress_queue, job_args):
    func = load_func(job_args.modelname, 'compute_optimal_proportion')
    func(progress_queue, job_args)

def compute_optimal_proportion(progress_queue, job_args, debug=True):
    from molass_legacy.KekLib.DebugPlot import exec_in_threaded_mainloop, quit_threaded_mainloop

    if debug:
        def exec_closure():
            compute_optimal_proportion_impl(progress_queue, job_args)
            quit_threaded_mainloop()

        exec_in_threaded_mainloop(exec_closure)
    else:
        compute_optimal_proportion_impl(progress_queue, job_args)
