"""
    EmbedCushion.py

    edit this code while keeping the caller code unchanged

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import os

def embed_cushion(caller):
    try:
        embed_cushion_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "embed_cushion_impl: ", n=10)
        raise RuntimeError("embed_cushion_impl")

def embed_cushion_impl_UnifiedDecompResultTest(caller):
    from importlib import reload
    import Decomposer.UnifiedDecompResultTest
    reload(Decomposer.UnifiedDecompResultTest)
    from molass_legacy.Decomposer.UnifiedDecompResultTest import unit_test
    unit_test(caller)

def embed_cushion_impl_OptRecsUtils(caller):
    import copy
    from importlib import reload
    import Decomposer.OptRecsUtils
    reload(Decomposer.OptRecsUtils)
    from molass_legacy.Decomposer.OptRecsUtils import eoii_correct_opt_recs
    frame = caller.editor
    opt_recs_ = copy.deepcopy(frame.opt_recs)
    eoii_correct_opt_recs(frame, opt_recs_, debug=True)

def embed_cushion_impl_(caller):
    from importlib import reload

    import Tools.EmbedCushionUtils
    reload(Tools.EmbedCushionUtils)
    from molass_legacy.Tools.EmbedCushionUtils import get_caller_attr

    import SimTools.ErrorModel
    reload(SimTools.ErrorModel)
    from SimTools.ErrorModel import demo

    gi_in_folder = r"E:\PyTools\Data\20180526\GI"
    if not os.path.exists(gi_in_folder):
        gi_in_folder = r"D:\PyTools\Data\20180526\GI"
    cushion_dict = get_caller_attr(caller, "cushion_dict", {})
    gi_sd = cushion_dict.get("gi_sd")
    if gi_sd is None:
        from molass_legacy.Tools.SdUtils import get_sd
        cushion_dict["gi_sd"] = gi_sd = get_sd(gi_in_folder)

    demo(gi_in_folder, gi_sd)

def embed_cushion_impl_(caller):
    from importlib import reload

    import BoundedLRF.HardSphereDemo
    reload(BoundedLRF.HardSphereDemo)
    from BoundedLRF.HardSphereDemo import demo

    demo()

def embed_cushion_impl_IterativeLrfSolverDemo(caller):
    from importlib import reload

    import Tools.EmbedCushionUtils
    reload(Tools.EmbedCushionUtils)
    from molass_legacy.Tools.EmbedCushionUtils import get_caller_attr

    import Trials.BoundedLRF.IterativeLrfSolverDemo
    reload(Trials.BoundedLRF.IterativeLrfSolverDemo)
    from Trials.BoundedLRF.IterativeLrfSolverDemo import demo, demo2, demo3

    print("embed_cushion")

    pdata, popts = caller.get_preview_data(with_update=False)
    try:
        # demo(caller.dialog, pdata, popts)
        # demo2(caller)
        demo3(caller.dialog, pdata, popts)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_Recon(caller):
    from importlib import reload
    import BoundedLRF.ReconsiderDemo
    reload(BoundedLRF.ReconsiderDemo)
    from BoundedLRF.ReconsiderDemo import demo

    try:
        demo(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_IFT(caller):
    from importlib import reload
    import BoundedLRF.IftDemo
    reload(BoundedLRF.IftDemo)
    from BoundedLRF.IftDemo import demo

    try:
        demo(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_ErrorMonteCarlo(caller):
    from importlib import reload
    import BoundedLRF.ErrorMonteCarlo
    reload(BoundedLRF.ErrorMonteCarlo)
    from BoundedLRF.ErrorMonteCarlo import demo

    try:
        demo(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_Error(caller):
    from importlib import reload
    import BoundedLRF.ErrorCorrectionDemo
    reload(BoundedLRF.ErrorCorrectionDemo)
    from BoundedLRF.ErrorCorrectionDemo import demo

    try:
        demo(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_EdmParamInspect(caller):
    from importlib import reload
    import Models.RateTheory.EdmParamInspect
    reload(Models.RateTheory.EdmParamInspect)
    from molass_legacy.Models.RateTheory.EdmParamInspect import demo

    try:
        demo(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_DecompDummy(caller):
    from importlib import reload
    import RangeEditors.DecompDummyDialog
    reload(RangeEditors.DecompDummyDialog)
    from RangeEditors.DecompDummyDialog import decompose

    try:
        decompose(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "decompose: ", n=10)

def embed_cushion_impl_Models(caller):
    from importlib import reload
    import Models.ElutionModelDemo
    reload(Models.ElutionModelDemo)
    from molass_legacy.Models.ElutionModelDemo import demo

    try:
        demo(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "do_devel_test: ", n=10)

def embed_cushion_impl_DispersiveUtils(caller):
    from importlib import reload
    import Models.Stochastic.DispersiveUtils
    reload(Models.Stochastic.DispersiveUtils)
    from molass_legacy.Models.Stochastic.DispersiveUtils import investigate_sdm_params_from_v2_result

    try:
        investigate_sdm_params_from_v2_result(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.dialog.logger, "investigate_sdm_params_from_v2_result: ", n=10)

def embed_cushion_impl_SdmEstimator(caller):
    from importlib import reload
    import Estimators.SdmEstimator
    reload(Estimators.SdmEstimator)
    from Estimators.SdmEstimator import onthefly_test

    try:
        onthefly_test(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "onthefly_test: ", n=10)

def embed_cushion_impl_ParamsIterator(caller):
    from importlib import reload
    import Optimizer.ParamsIterator
    reload(Optimizer.ParamsIterator)
    from molass_legacy.Optimizer.ParamsIterator import iterator_test_from_dialog

    try:
        iterator_test_from_dialog(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "iterator_test_from_dialog: ", n=10)

def embed_cushion_impl_EdmEstimator(caller):
    from importlib import reload
    import Estimators.EdmEstimator
    reload(Estimators.EdmEstimator)
    from Estimators.EdmEstimator import onthefly_test

    try:
        onthefly_test(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "EdmEstimator onthefly_test: ", n=10)

def embed_cushion_impl_RatioInterpretIllust(caller):
    from importlib import reload
    import Optimizer.RatioInterpretIllust
    reload(Optimizer.RatioInterpretIllust)
    from molass_legacy.Optimizer.RatioInterpretIllust import ratio_interpret_illust

    try:
        ratio_interpret_illust(caller.canvas)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "ratio_interpret_illust: ", n=10)

def ScaleAdjustInspect_impl(caller):
    from importlib import reload
    import Optimizer.ScaleAdjustInspect
    reload(Optimizer.ScaleAdjustInspect)
    from molass_legacy.Optimizer.ScaleAdjustInspect import scale_adjust_inspect

    try:
        scale_adjust_inspect(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "scale_adjust_inspect: ", n=10)

def RgCurveInspect_impl(caller):
    from importlib import reload
    import Optimizer.RgCurveInspect
    reload(Optimizer.RgCurveInspect)
    from molass_legacy.Optimizer.RgCurveInspect import rg_curve_inspect

    try:
        rg_curve_inspect(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "rg_curve_inspect: ", n=10)

def RestartPatcher_impl(caller):
    from importlib import reload
    import Optimizer.RestartPatcher
    reload(Optimizer.RestartPatcher)
    from molass_legacy.Optimizer.RestartPatcher import patch_and_restart

    try:
        patch_and_restart(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "patch_and_restart_test: ", n=10)

def BoundsInspect_impl(caller):
    from importlib import reload
    import Optimizer.Devel.BoundsInspect
    reload(Optimizer.Devel.BoundsInspect)
    from molass_legacy.Optimizer.Devel.BoundsInspect import bounds_inspect_impl

    try:
        bounds_inspect_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "bounds_inspect_impl: ", n=10)

def McmcTrial_impl(caller):
    from importlib import reload
    import Solvers.MCMC.TrialBridge
    reload(Solvers.MCMC.TrialBridge)
    from Solvers.MCMC.TrialBridge import mcmc_trial_impl

    try:
        mcmc_trial_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "mcmc_trial_impl: ", n=10)

def SmcTrial_impl(caller):
    from importlib import reload
    import Solvers.SMC.TrialBridge
    reload(Solvers.SMC.TrialBridge)
    from Solvers.SMC.TrialBridge import smc_trial_impl

    try:
        smc_trial_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "smc_trial_impl: ", n=10)

def AbcTrial_impl(caller):
    from importlib import reload
    import Solvers.ABC.TrialBridge
    reload(Solvers.ABC.TrialBridge)
    from Solvers.ABC.TrialBridge import abc_trial_impl

    try:
        abc_trial_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "abc_trial_impl: ", n=10)

def NestTrial_impl(caller):
    from importlib import reload
    import Solvers.UltraNest.TrialBridge
    reload(Solvers.UltraNest.TrialBridge)
    from Solvers.UltraNest.TrialBridge import nest_trial_impl

    try:
        nest_trial_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "nest_trial_impl: ", n=10)

def Plot_callback_txt(caller):
    from importlib import reload
    import Solvers.UltraNest.CallbacktxtPlot
    reload(Solvers.UltraNest.CallbacktxtPlot)
    from Solvers.UltraNest.CallbacktxtPlot import plot_callback_txt_impl

    try:
        plot_callback_txt_impl(caller)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(caller.logger, "plot_callback_txt_impl: ", n=10)

def debug_objective_function(caller):
    from importlib import reload
    import Optimizer.SimpleDebugUtils
    reload(Optimizer.SimpleDebugUtils)
    from molass_legacy.Optimizer.SimpleDebugUtils import debug_optimizer
    debug_optimizer(caller.fullopt)

def embed_cushion_impl(caller):
    import molass_legacy.KekLib.DebugPlot as plt

    extra_button_specs = [
        ("ScaleAdjust Inspect", lambda: ScaleAdjustInspect_impl(caller)),
        ("RgCurve Inspect", lambda: RgCurveInspect_impl(caller)),
        ("Restart Patcher", lambda: RestartPatcher_impl(caller)),
        ("Bounds Inspect", lambda: BoundsInspect_impl(caller)),
        ("MCMC Trial", lambda: McmcTrial_impl(caller)),
        ("SMC Trial", lambda: SmcTrial_impl(caller)),
        ("ABC Trial", lambda: AbcTrial_impl(caller)),
        ("Nest Trial", lambda: NestTrial_impl(caller)),
        ("Plot callback.txt", lambda: Plot_callback_txt(caller)),
        ("Objective Function Debug", lambda: debug_objective_function(caller)),
    ]

    with plt.Dp(button_spec=["OK", "Cancel"], extra_button_specs=extra_button_specs):
        fig, ax = plt.subplots()
        plt.show()