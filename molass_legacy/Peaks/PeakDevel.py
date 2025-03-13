"""
    Peaks.PeakDevel.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt

def devel_test_impl(self):
    print("devel_test_impl")

    def exec_embed_cushion():
        import Tools.EmbedCushion
        reload(Tools.EmbedCushion)
        from molass_legacy.Tools.EmbedCushion import embed_cushion
        embed_cushion(self)

    def cpd_spike():
        import GuinierTools.CpdDecompIndirect
        reload(GuinierTools.CpdDecompIndirect)
        from GuinierTools.CpdDecompIndirect import cpd_spike_impl
        cpd_spike_impl(self)

    def estimator_test():
        import Estimators.TestTools
        reload(Estimators.TestTools)
        from Estimators.TestTools import estimator_test_impl
        estimator_test_impl(self)

    def show_restart_patcher():
        import Optimizer.RestartPatcher
        reload(Optimizer.RestartPatcher)
        from Optimizer.RestartPatcher import patch_and_restart_from_editor
        patch_and_restart_from_editor(self)

    def test_fixedbaseline_optimizer():
        import Optimizer.FixedBaselineOptimizer
        reload(Optimizer.FixedBaselineOptimizer)
        from Optimizer.FixedBaselineOptimizer import test_optimizer
        test_optimizer(self)

    def debug_objective_function():
        import Optimizer.SimpleDebugUtils
        reload(Optimizer.SimpleDebugUtils)
        from Optimizer.SimpleDebugUtils import debug_optimizer
        debug_optimizer(self.fullopt, self.init_params)

    def test_estimate_uvbaseline():
        import Optimizer.UvBaselineEstimator
        reload(Optimizer.UvBaselineEstimator)
        from Optimizer.UvBaselineEstimator import test_estimate_uvbaseline_impl
        test_estimate_uvbaseline_impl(self.fullopt, self.init_params)

    extra_button_specs = [
        ("Embed Cushion", exec_embed_cushion),
        ("CPD Spike", cpd_spike),
        ("Estimator Test", estimator_test),
        ("Restart Patcher", show_restart_patcher),
        ("Fixed Baseline Optimizer Test", test_fixedbaseline_optimizer),
        ("Objective Function Debug", debug_objective_function),
        ("Estimate UV Baseline", test_estimate_uvbaseline),
    ]

    with plt.Dp(button_spec=["OK", "Cancel"], extra_button_specs=extra_button_specs):
        fig, ax = plt.subplots()
        plt.show()
