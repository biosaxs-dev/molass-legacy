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
            os.makedirs(analysis_folder)
    set_setting('analysis_folder', analysis_folder)
    update_sec_settings()

    optimizer_folder = os.path.join(analysis_folder, "optimized")
    if not os.path.exists(optimizer_folder):
        os.makedirs(optimizer_folder)
    rg_folder = os.path.join(optimizer_folder, "rg-curve")
    if not os.path.exists(rg_folder):
        os.makedirs(rg_folder)

    if sd is None:
        from molass_legacy.Batch.StandardProcedure import StandardProcedure
        sp = StandardProcedure()
        sd = sp.load_old_way(in_folder)

    if sd.pre_recog is None:
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        sd.pre_recog = PreliminaryRecognition(sd)

    from molass_legacy.Batch.OptDataSetsProxy import OptDataSetsProxy as OptDataSets
    from molass_legacy.SecSaxs.DataTreatment import DataTreatment

    trimming = 2
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
    batch.pre_recog = PreliminaryRecognition(corrected_sd)  # to ensure better that mapping is available
    batch.base_curve_info = treat.get_base_curve_info()     # not used?

    batch.strict_sec_penalty = False
    batch.fullopt_class, batch.class_code = None, None
    batch.fullopt_input = FullOptInput(sd=trimmed_sd, corrected_sd=corrected_sd, rg_folder=rg_folder)
    batch.dsets = batch.fullopt_input.get_dsets(progress_cb=None, compute_rg=True, possibly_relocated=False)
    return batch

def prepare_optimizer(batch, num_components=3, model="EGH", method="BH", function_code=None, debug=False):
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
    set_optimizer_settings(num_components=num_components, model=model, method=method)

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

def set_optimizer_settings(num_components=3, model="EGH", method="BH", param_init_type=1, ns_narrow_bounds=True, ns_adaptive_nsteps=False, ns_nsteps=None, **solver_kwargs):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from .OptimizerSettings import OptimizerSettings

    # ── Model registry ────────────────────────────────────────────────────────
    # This is the canonical registration point for new elution models.
    # Add a new elif branch here and assign an unused elution_model integer.
    # Current assignments: EGH=0, SDM=1, EDM/NEDM/CEDM=5, LKM=6, GRM=7
    # (These match the GUI MODEL_LIST indices in OptStrategyDialog.py)
    elution_model = 0
    model = model.upper()
    if model == "EGH":
        elution_model = 0
    elif model == "SDM":
        elution_model = 2
    elif model in ("EDM", "NEDM", "CEDM"):
        elution_model = 5   # G2000–G2999 range covers both NEDM (G2010) and EDM/CEDM (G2020)
    elif model == "LKM":
        elution_model = 6   # G1400 (matches OptStrategyDialog MODEL_LIST index 6)
    elif model == "GRM":
        elution_model = 7   # G1500 (matches OptStrategyDialog MODEL_LIST index 7)
    else:
        raise ValueError(
            f"Unknown model: {model!r}. "
            "Add it to set_optimizer_settings() in Optimizer/Scripting.py "
            "and assign an unused elution_model integer."
        )

    solver_name = method.upper()
    try:
        from molass_legacy.Solvers.Registry import SOLVER_REGISTRY
        if solver_name not in SOLVER_REGISTRY:
            raise ValueError(
                f"Unknown method: {method!r}. "
                "Add it to SOLVER_REGISTRY in molass_legacy/Solvers/Registry.py."
            )
        optimization_method = SOLVER_REGISTRY[solver_name].int_code
    except ImportError:
        # Fallback (Registry not available)
        if solver_name == "BH":     optimization_method = 0
        elif solver_name == "NS":   optimization_method = 1
        elif solver_name == "CMA":  optimization_method = 5
        elif solver_name == "DE":   optimization_method = 6
        elif solver_name == "NSGA2":optimization_method = 7
        else: raise ValueError(f"Unknown method: {method!r}.")
    set_setting("optimization_method", optimization_method)     # for backward compatibility
    for k, v in solver_kwargs.items():
        set_setting(k, v)

    separate_eoii_flags = [0] * num_components

    settings = OptimizerSettings(param_init_type=param_init_type,
                                 elution_model=elution_model,
                                 optimization_method=optimization_method,
                                 separate_eoii_flags=separate_eoii_flags,
                                 ns_narrow_bounds=ns_narrow_bounds,
                                 ns_adaptive_nsteps=ns_adaptive_nsteps,
                                 ns_nsteps=ns_nsteps)
    settings.save()

def estimate_init_params(batch, optimizer, developing=False, debug=False):
    batch.get_ready_for_progress_display()

    init_params = batch.compute_init_params(developing=developing, debug=debug)
    print("Initial Parameters:", len(init_params))
    optimizer.prepare_for_optimization(init_params)
    return init_params

def run_optimizer(optimizer, init_params, niter=20, clear_jobs=True, dummy=False, x_shifts=None, debug=True):
    """ Run the optimizer with a monitoring dashboard.

    Parameters
    ----------
    optimizer : Optimizer
        The optimizer instance to run.
    init_params : array-like
        Initial parameters for the optimization.
    niter : int, optional
        Number of iterations to run. Default is 20.
    clear_jobs : bool, optional
        Whether to clear previous jobs. Default is True.
    dummy : bool, optional
        If True, runs in dummy mode without actual optimization. Default is False.
    x_shifts : array-like, optional
        Shifts to apply to the parameters. Default is None.
        x_shifts can be obtained from dsets.get_x_shifts().
        See also OptDataSets.get_x_shifts().
    debug : bool, optional
        If True, enables debug mode. Default is True.

    Returns
    -------
    monitor : MplMonitor
        The monitoring dashboard instance.
    """
    if debug:
        from importlib import reload
        import molass_legacy.Optimizer.MplMonitor
        reload(molass_legacy.Optimizer.MplMonitor)
    from molass_legacy.Optimizer.MplMonitor import MplMonitor
    monitor = MplMonitor(function_code=optimizer.get_function_code(), clear_jobs=clear_jobs, xr_only=optimizer.get_xr_only())
    monitor.create_dashboard()
    monitor.run(optimizer, init_params, niter=niter, dummy=dummy, x_shifts=x_shifts, debug=debug)
    monitor.show(debug=debug)

    if dummy:
        import time
        time.sleep(10)
        monitor.terminate_job(None)
    else:
        monitor.start_watching()
    return monitor

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