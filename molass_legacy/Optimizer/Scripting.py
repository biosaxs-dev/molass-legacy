"""
Optimizer.Scripting.py

- Menus/V2Menu.py
- Optimizer/OptimizerUtils.py
- Optimizer/OptStrategyDialog.py
- Peaks/PeakEditor.py
- Optimizer/FullOptDialog.py

- Optimizer/optimizer.py
    - Optimizer/OptimizerMain.py
"""
import os
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import set_setting

def prepare_data(in_folder, sd=None, clear_temp_settings=True, analysis_folder=None):
    from molass_legacy.Global.V2Init import update_sec_settings

    if clear_temp_settings:
        from molass_legacy._MOLASS.SerialSettings import set_setting, clear_v2_temporary_settings
        clear_v2_temporary_settings()

    set_setting('in_folder', in_folder)
    if analysis_folder is None:
        analysis_folder = 'temp_analysis'
        if not os.path.exists(analysis_folder):
            os.makedirs(analysis_folder, exist_ok=True)
    set_setting('analysis_folder', analysis_folder)
    update_sec_settings()

    optimizer_folder = os.path.join(analysis_folder, "optimized")
    if not os.path.exists(optimizer_folder):
        os.makedirs(optimizer_folder, exist_ok=True)
    rg_folder = os.path.join(optimizer_folder, "rg-curve")
    if not os.path.exists(rg_folder):
        os.makedirs(rg_folder, exist_ok=True)

    if sd is None:
        from molass_legacy.Batch.StandardProcedure import StandardProcedure
        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)

    if sd.pre_recog is None:
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        sd.pre_recog = PreliminaryRecognition(sd)

    from molass_legacy.Batch.OptDataSetsProxy import OptDataSetsProxy as OptDataSets
    from molass_legacy.SecSaxs.DataTreatment import DataTreatment

    trimming = 1
    correction = 1
    unified_baseline_type = 1
    treat = DataTreatment(route="v2", trimming=trimming, correction=correction, unified_baseline_type=unified_baseline_type)
    pre_recog = sd.pre_recog
    trimmed_sd = treat.get_trimmed_sd(sd, pre_recog)
    corrected_sd = treat.get_corrected_sd(sd, pre_recog, trimmed_sd)
    treat.save()

    from molass_legacy.Batch.FullBatch import FullBatch
    from molass_legacy.Optimizer.FullOptInput import FullOptInput
    # equivalent to PeakEditor.__init__
    batch = FullBatch()
    batch.logger = logging.getLogger(__name__)
    batch.sd = trimmed_sd
    batch.corrected_sd = corrected_sd
    batch.pre_recog = pre_recog
    batch.base_curve_info = treat.get_base_curve_info()     # not used?

    batch.strict_sec_penalty = False
    batch.fullopt_class, batch.class_code = None, None
    batch.fullopt_input = FullOptInput(sd=trimmed_sd, corrected_sd=corrected_sd, rg_folder=rg_folder)
    batch.dsets = batch.fullopt_input.get_dsets(progress_cb=None, compute_rg=True, possibly_relocated=False)
    return batch

def prepare_optimizer(batch, num_components=None, model='EGH', function_code=None, debug=False):
    from molass_legacy.Optimizer.FuncImporter import import_objective_function

    if function_code is None:
        assert model is not None, "Either model or function_code must be provided."
        from molass_legacy.Optimizer.OptimizerUtils import get_function_code
        function_code = get_function_code(model)
    else:
        assert model is None, "Either model or function_code must be provided, not both."
        from molass_legacy.Optimizer.OptimizerUtils import get_model_name       
        model_name = get_model_name(function_code)

    function_class = import_objective_function(function_code)

    if debug:
        print("Running optimizer with function:", function_class.__name__)

    batch.exact_num_peaks = num_components

    # equivalent to PeakEditor.body
    uv_x, uv_y, xr_x, xr_y, baselines = batch.get_curve_xy(return_baselines=True)
    uv_y_ = uv_y - baselines[0]
    xr_y_ = xr_y - baselines[1]
    uv_peaks, xr_peaks = batch.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_)
    batch.set_lrf_src_args1(uv_x, uv_y, xr_x, xr_y, baselines)

    batch.construct_optimizer(fullopt_class=function_class)
    return batch.optimizer

def estimate_init_params(batch, optimizer, developing=False, debug=True):
    batch.get_ready_for_progress_display()

    init_params = batch.compute_init_params(developing=developing, debug=debug)
    print("Initial Parameters:", len(init_params))
    optimizer.prepare_for_optimization(init_params)
    return init_params

def set_optimizer_settings(param_init_type=0, method="BH"):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from .OptimizerSettings import OptimizerSettings

    solver_name = method.upper()
    if solver_name == "BH":
        optimization_method = 0
    elif solver_name == "NS":
        optimization_method = 1
    else:
        assert False, f"Unknown method: {method}"
    set_setting("optimization_method", optimization_method)     # for backward compatibility 

    settings = OptimizerSettings(param_init_type=param_init_type, optimization_method=optimization_method)
    settings.save()

def run_optimizer(in_folder, optimizer, init_params, niter=20, clear_jobs=True, dummy=False, debug=True):
    if debug:
        from importlib import reload
        import molass_legacy.Optimizer.MplMonitor
        reload(molass_legacy.Optimizer.MplMonitor)
    from molass_legacy.Optimizer.MplMonitor import MplMonitor
    monitor = MplMonitor(optimizer.get_function_code())
    if clear_jobs:
        monitor.clear_jobs()  # equivalent to BackRunner.
    monitor.create_dashboard()
    monitor.run(optimizer, init_params, niter=niter, dummy=dummy, debug=debug)
    monitor.show(debug=debug)

    if dummy:
        import time
        time.sleep(10)
        monitor.terminate_job(None)
    else:
        monitor.start_watching()

def get_params(job_result_folder, index=None, debug=False):
    from .StateSequence import read_callback_txt_impl
    cb_file = os.path.join(job_result_folder, 'callback.txt')
    fv_list, x_list = read_callback_txt_impl(cb_file)
    fv = np.array(fv_list)
    x = np.array(x_list)
    if index is None:
        k = np.argmin(fv[:,1])
        params = x[k]
        if debug:
            print("Best parameters at index %d with fv=%g" % (k, fv[k,1]))
    else:
        params = x[index]
        if debug:
            print("Parameters at index %d with fv=%g" % (index, fv[index,1]))
    return params