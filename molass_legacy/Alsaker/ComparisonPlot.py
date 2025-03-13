"""
    Alsaker.ComparisonPlot.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

def plot_comparison_result_impl(corrected_sd, comparison_result):
    print("plot_comparison_result_impl")

    xr_curve = corrected_sd.get_xr_curve()
    x = xr_curve.x
    y = xr_curve.y

    sg_array, at_array, al_array = comparison_result
    in_folder = get_in_folder()

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Comparison of Rg values of %s" % in_folder, fontsize=16)
        ax.plot(x, y)
        axt = ax.twinx()
        axt.grid(False)
        axt.plot(x, sg_array[:,0], color="cyan", label="_MOLASS")
        axt.plot(x, at_array[:,0], color="red", label="ATSAS")
        axt.plot(x, al_array[:,0], color="green", label="Alsaker")
        axt.legend()
        fig.tight_layout()
        plt.show()    