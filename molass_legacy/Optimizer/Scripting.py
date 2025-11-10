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
import  os

def prepare_optimizer(in_folder, sd=None, num_components=None, function_code='G0346', analysis_folder=None, clear_temp_settings=True, debug=False):
    import logging
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

    if function_code is not None:
        from molass_legacy.Optimizer.FuncImporter import import_objective_function
        function_class = import_objective_function(function_code)

    if debug:
        print("Running optimizer with function:", function_class.__name__)

    optimizer_folder = os.path.join(analysis_folder, "optimized")
    if not os.path.exists(optimizer_folder):
        os.makedirs(optimizer_folder, exist_ok=True)
    rg_folder = os.path.join(optimizer_folder, "rg-curve")
    if not os.path.exists(rg_folder):
        os.makedirs(rg_folder, exist_ok=True)

    from molass_legacy.Batch.FullBatch import FullBatch
    from molass_legacy.Optimizer.FullOptInput import FullOptInput
    # equivalent to PeakEditor.__init__
    batch = FullBatch()
    batch.logger = logging.getLogger(__name__)
    batch.sd = sd
    batch.corrected_sd = corrected_sd
    batch.pre_recog = pre_recog
    batch.base_curve_info = treat.get_base_curve_info()     # not used?

    batch.exact_num_peaks = num_components
    batch.strict_sec_penalty = False
    batch.fullopt_class, batch.class_code = None, None
    batch.fullopt_input = FullOptInput(sd=sd, corrected_sd=corrected_sd, rg_folder=rg_folder)
    batch.dsets = batch.fullopt_input.get_dsets(progress_cb=None, compute_rg=True, possibly_relocated=False)

    # equivalent to PeakEditor.body
    uv_x, uv_y, xr_x, xr_y, baselines = batch.get_curve_xy(return_baselines=True)
    uv_y_ = uv_y - baselines[0]
    xr_y_ = xr_y - baselines[1]
    uv_peaks, xr_peaks = batch.get_modeled_peaks(uv_x, uv_y_, xr_x, xr_y_)
    batch.set_lrf_src_args1(uv_x, uv_y, xr_x, xr_y, baselines)

    batch.construct_optimizer(fullopt_class=function_class)
    optimizer = batch.optimizer

    init_params = batch.compute_init_params()
    optimizer.prepare_for_optimization(init_params)
    return optimizer, init_params


def run_optimizer(in_folder, optimizer, init_params, debug=False):
    if debug:
        from importlib import reload
        import molass_legacy.Optimizer.MplMonitor
        reload(molass_legacy.Optimizer.MplMonitor)
    from molass_legacy.Optimizer.MplMonitor import MplMonitor
    from molass_legacy.Optimizer.BackRunner import BackRunner