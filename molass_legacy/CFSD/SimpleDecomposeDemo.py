"""
    SimpleDecomposeDemo.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
from molass_legacy.KekLib.BasicUtils import Struct

def demo(in_folder, sd_orig, extra_peaks=0, pdf_only=False, modelname="EDM", num_peaks=3):
    from importlib import reload
    import CFSD.SimpleDecompose
    reload(CFSD.SimpleDecompose)    
    from CFSD.SimpleDecompose import decompose_elution_simply
    from molass_legacy.Tools.MapperSingleton import get_mapper

    mapper, sd = get_mapper(sd_orig)
    uv_y = mapper.make_uniformly_scaled_vector(scale=1)

    print(in_folder)
    ecurve = sd.get_xray_curve()
    x = ecurve.x
    y = ecurve.y

    # num_peaks = 3
    if modelname == "EDM":
        import Models.RateTheory.EDM
        reload(Models.RateTheory.EDM)
        from molass_legacy.Models.RateTheory.EDM import EDM, save_reg_data_fh
        model = EDM()
    else:
        import Models.Stochastic.Tripore
        reload(Models.Stochastic.Tripore)
        from molass_legacy.Models.Stochastic.Tripore import Tripore
        save_reg_data_fh = None
        model = Tripore()

    traditional_info = Struct(mapper=mapper, num_peaks=num_peaks)
    decomp_result_non_cfs = decompose_elution_simply(x, y, uv_y, model, traditional_info, using_cfs=False, debug=True)
    if not pdf_only:
        decomp_result_cfs = decompose_elution_simply(x, y, uv_y, model, traditional_info, using_cfs=True)

    if save_reg_data_fh is not None:
        save_reg_data_fh.close()

    def plot_results(ax, x, y, opt_recs):
        ax.plot(x, y)
        cy_list = []
        for rec in opt_recs:
            cy = rec.evaluator(x)
            ax.plot(x, cy, ":", lw=3)
            cy_list.append(cy)
        ty = np.sum(cy_list, axis=0)
        ax.plot(x, ty, ":", color="red", lw=3)

    in_folder_ = get_in_folder(in_folder)
    with plt.Dp():
        fig = plt.figure(figsize=(15,10))
        fig.suptitle("Comparison of %s-decomposition in Different Fitting Spaces with %s" % (model.get_name(), in_folder_), fontsize=20)

        gs = GridSpec(2,5)
        ax10 = fig.add_subplot(gs[0,0])
        ax20 = fig.add_subplot(gs[1,0])
        for ax in ax10, ax20:
            ax.set_axis_off()

        ax10.text(0.5, 0.5, "PDF\nfitting", ha="center", va="center", fontsize=16)
        ax20.text(0.5, 0.5, "CF\nfitting", ha="center", va="center", fontsize=16)

        ax11 = fig.add_subplot(gs[0,1:3])
        ax12 = fig.add_subplot(gs[0,3:5])
        ax21 = fig.add_subplot(gs[1,1:3])
        ax22 = fig.add_subplot(gs[1,3:5])

        ax11.set_title("UV Decomposition", fontsize=16)
        ax12.set_title("XR Decomposition", fontsize=16)

        plot_results(ax11, x, uv_y, decomp_result_non_cfs.opt_recs_uv)
        plot_results(ax12, x, y, decomp_result_non_cfs.opt_recs)

        if not pdf_only:
            plot_results(ax21, x, uv_y, decomp_result_cfs.opt_recs_uv)
            plot_results(ax22, x, y, decomp_result_cfs.opt_recs)

        fig.tight_layout()
        plt.show()
