"""
    Peaks.PeakDevel.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt

def devel_test_impl(self):
    print("devel_test_impl")

    def exec_embed_cushion():
        import molass_legacy.Tools.EmbedCushion
        reload(molass_legacy.Tools.EmbedCushion)
        from molass_legacy.Tools.EmbedCushion import embed_cushion
        embed_cushion(self)

    def cpd_spike():
        import molass_legacy.GuinierTools.CpdDecompIndirect
        reload(molass_legacy.GuinierTools.CpdDecompIndirect)
        from molass_legacy.GuinierTools.CpdDecompIndirect import cpd_spike_impl
        cpd_spike_impl(self)

    def estimator_test():
        import molass_legacy.Estimators.TestTools
        reload(molass_legacy.Estimators.TestTools)
        from molass_legacy.Estimators.TestTools import estimator_test_impl
        estimator_test_impl(self)

    def show_restart_patcher():
        import molass_legacy.Optimizer.RestartPatcher
        reload(molass_legacy.Optimizer.RestartPatcher)
        from molass_legacy.Optimizer.RestartPatcher import patch_and_restart_from_editor
        patch_and_restart_from_editor(self)

    def test_fixedbaseline_optimizer():
        import molass_legacy.Optimizer.FixedBaselineOptimizer
        reload(molass_legacy.Optimizer.FixedBaselineOptimizer)
        from molass_legacy.Optimizer.FixedBaselineOptimizer import test_optimizer
        test_optimizer(self)

    def debug_objective_function():
        import molass_legacy.Optimizer.SimpleDebugUtils
        reload(molass_legacy.Optimizer.SimpleDebugUtils)
        from molass_legacy.Optimizer.SimpleDebugUtils import debug_optimizer
        debug_optimizer(self.fullopt, self.init_params)

    def test_estimate_uvbaseline():
        import molass_legacy.Optimizer.UvBaselineEstimator
        reload(molass_legacy.Optimizer.UvBaselineEstimator)
        from molass_legacy.Optimizer.UvBaselineEstimator import test_estimate_uvbaseline_impl
        test_estimate_uvbaseline_impl(self.fullopt, self.init_params)

    def test_func_loader():
        import logging
        import molass_legacy.Optimizer.FuncImporter
        reload(molass_legacy.Optimizer.FuncImporter)
        from molass_legacy.Optimizer.FuncImporter import get_objective_function_info
        logger = logging.getLogger(__name__)
        func_info = get_objective_function_info(logger, default_func_code='G0346')
        print("Objective Functions:", func_info.func_dict)

    extra_button_specs = [
        ("Embed Cushion", exec_embed_cushion),
        ("CPD Spike", cpd_spike),
        ("Estimator Test", estimator_test),
        ("Restart Patcher", show_restart_patcher),
        ("Fixed Baseline Optimizer Test", test_fixedbaseline_optimizer),
        ("Objective Function Debug", debug_objective_function),
        ("Estimate UV Baseline", test_estimate_uvbaseline),
        ("Function Loader Test", test_func_loader),
    ]

    with plt.Dp(button_spec=["OK", "Cancel"], extra_button_specs=extra_button_specs):
        fig, ax = plt.subplots()
        plt.show()
