"""
    V2PropOptimizer.PropOptimizerUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from time import sleep
from importlib import import_module, reload
from molass_legacy.Models.ElutionCurveModels import EGH
from molass_legacy.Models.RateTheory.EDM import EDM

IMPL_DICT = {
    'EGH' : 'V2PropOptimizer.PropOptimizerEGH',
    'EDM' : 'V2PropOptimizer.PropOptimizerEDM',
    'STC' : 'V2PropOptimizer.PropOptimizerSTC',
    }

MODEL_DICT = {
    'EGH' : EGH(),      # or EGHA()?
    'EDM' : EDM(),
    }

def load_func(modelname, funcname):
    classname = IMPL_DICT.get(modelname)
    assert classname is not None

    module = import_module(classname)
    reload(module)
    func = getattr(module, funcname)    # from V2PropOptimizer.PropOptimizerXXX import funcname
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
