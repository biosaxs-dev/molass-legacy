"""
    DevTrial.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt

def dev_trial_entry(caller):

   corrected_sd = None
   comparison_result = None

   def prepare_data():
      import Trimming.MomentTrimming
      reload(Trimming.MomentTrimming)
      from molass_legacy.Trimming.MomentTrimming import set_moment_trimming_info
      import Batch.LiteBatch
      reload(Batch.LiteBatch)
      from molass_legacy.Batch.LiteBatch import create_corrected_sd
      nonlocal corrected_sd

      set_moment_trimming_info(caller.serial_data)
      corrected_sd = create_corrected_sd(caller.serial_data)
      ax.cla()
      ax.text(0.5, 0.5, "Corrected Data Prepared", ha="center", va="center", fontsize=30, color="gray")
      fig.canvas.draw()

   def spike_test():
      import Models.MomentFitting
      reload(Models.MomentFitting)
      from molass_legacy.Models.MomentFitting import spike
      spike(caller)
 
   def bridge_test():
      import Alsaker.Bridge
      reload(Alsaker.Bridge)
      from molass_legacy.Alsaker.Bridge import bridge_test_impl
      bridge_test_impl(caller)

   def compare_bridge():
      import Alsaker.Compare
      reload(Alsaker.Compare)
      from molass_legacy.Alsaker.Compare import compare_bridge_impl
      nonlocal comparison_result
      comparison_result = compare_bridge_impl(caller, corrected_sd=corrected_sd)

   def plot_comparison_result():
      import Alsaker.ComparisonPlot
      reload(Alsaker.ComparisonPlot)
      from molass_legacy.Alsaker.ComparisonPlot import plot_comparison_result_impl
      plot_comparison_result_impl(corrected_sd, comparison_result)

   def v2_baseline_inspect():
      import Baseline.Inspect
      reload(Baseline.Inspect)
      from molass_legacy.Baseline.Inspect import baseline_inspect_impl
      baseline_inspect_impl(caller)

   def bo_baseline_trial():
      import Baseline.BoBaseline
      reload(Baseline.BoBaseline)
      from molass_legacy.Baseline.BoBaseline import bo_baseline_trial_impl
      bo_baseline_trial_impl(caller)

   def debug_modeled_peaks():
      import QuickAnalysis.PeakUtils
      reload(QuickAnalysis.PeakUtils)
      from molass_legacy.QuickAnalysis.PeakUtils import demo_modeled_peaks_dialog
      in_folder = caller.in_folder.get()
      demo_modeled_peaks_dialog(caller, in_folder)

   def opt_view_range():
      import Trimming.OptViewRange
      reload(Trimming.OptViewRange)
      from molass_legacy.Trimming.OptViewRange import get_opt_view_range
      
      get_opt_view_range(caller)

   extra_button_specs = [
      ("Prepare Dara", prepare_data),
      ("MomentFitting", spike_test),
      ("Alsaker Bridge Test", bridge_test),
      ("Compare Bridge to SimpleGuinier", compare_bridge),
      ("Plot Comparison Result", plot_comparison_result),
      ("V2 Baseline", v2_baseline_inspect),
      ("BO Baseline", bo_baseline_trial),
      ("Debug Modeled Peaks", debug_modeled_peaks),
      ("Opt View Range", opt_view_range),
      ]

   with plt.Dp(button_spec=["Close"],
               extra_button_specs=extra_button_specs):
      fig, ax = plt.subplots()
      plt.show()