"""
    Optimizer.ParameterTransition.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
import queue
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy._MOLASS.SerialSettings import set_setting

USE_RESULT_PROXY = False

def show_parameter_transition_impl(selector_dialog, debug=True):
    if debug:
        from importlib import reload
        import molass_legacy.KekLib.OnTheFlyUtils
        reload(KekLib.OnTheFlyUtils)
        import Optimizer.ResultProxy
    from molass_legacy.KekLib.OnTheFlyUtils import show_progress
    from molass_legacy.Optimizer.ResultProxy import ResultProxy

    joblist_folder = selector_dialog.folder.get()
    print("show_parameter_transition_impl: joblist_folder=", joblist_folder)

    params_list = []
    bounds_list = []
    fv_list = []
    optimizer = None
    progress_queue = queue.Queue()

    num_steps = 100

    def prepare_data():
        nonlocal optimizer
        for i, node in enumerate(os.listdir(joblist_folder)):
            print([i], node)
            if i == num_steps:
                break

            jobpath = os.path.join(joblist_folder, node)
            set_setting('optworking_folder', jobpath)

            if USE_RESULT_PROXY:
                # this yet seems buggy
                if i == 0:
                    first_result = selector_dialog.get_result(folder=jobpath)
                    result = first_result
                else:
                    result = ResultProxy(jobpath, first_result)
            else:
                result = selector_dialog.get_result(folder=jobpath)

            optimizer = result.get_optimizer()
            bounds = optimizer.get_param_bounds(optimizer.init_params)
            bounds_list.append(bounds)
            for j, params in result.get_result_iterator(all=True):
                nparams = optimizer.to_norm_params(params)
                params_list.append(nparams)
            fv_list.append(result.fv_array)
            progress_queue.put(i+1)

    show_progress(prepare_data, progress_queue, num_steps)

    params_array = np.array(params_list)
    bounds_array = np.array(bounds_list)
    print("params_array.shape=", params_array.shape)

    def plot_transition():
        from importlib import reload
        import Optimizer.ParameterTransitionPlot as Plot
        reload(Plot)
        from molass_legacy.Optimizer.ParameterTransitionPlot import plot_transition_impl
        try:
            plot_transition_impl(optimizer, params_array, bounds_array, fv_list)
        except:
            import traceback
            traceback.print_exc()

    with plt.Dp(window_title="Parameter Transition",
                button_spec=["OK", "Cancel"],
                extra_button_specs=[("Plot", plot_transition)]):
        fig, ax = plt.subplots()
        fig.tight_layout()
        plt.show()