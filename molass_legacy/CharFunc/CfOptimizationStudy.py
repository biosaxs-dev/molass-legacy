"""
    Models/Stochastic/CfOptimizationStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props
from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
from SecTheory.SecPDF import BidirectFft

def cf_optimization_study_impl(advanced_frame, devel=True):
    if devel:
        from importlib import reload
        import Models.Stochastic.LognormalPoreColumn as LognormalPoreColumn
        reload(LognormalPoreColumn)
    from molass_legacy.Models.Stochastic.LognormalPoreColumn import LognormalPoreColumn
    editor = advanced_frame.editor
    editor_frame = editor.get_current_frame()
    model = editor_frame.model
    print("estimate_psd_impl", model.get_name(), model.__class__)
    params_array = editor.get_current_params_array()

    fx = editor_frame.fx
    x = editor_frame.x
    y = editor_frame.y
    uv_y = editor_frame.uv_y

    D, E, qv, ecurve = editor.sd.get_xr_data_separate_ly()
    peak_region = ecurve.get_peak_region(sigma_scale=5)
    print("peak_region=", peak_region)

    def plot_cy_list(cy_list, title, test_y=None):
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.plot(x, y)
            for cy in cy_list:
                ax.plot(x, cy, ":", label="component-%d" % (len(cy_list)))
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", label="model total")
            for x_ in peak_region:
                ax.axvline(x=x_+x[0], color="green")
            if test_y is not None:
                ax.plot(x, test_y, ":", color="cyan", label="test curve")
            fig.tight_layout()
            plt.show()

    slice_ = slice(*[int(x_) for x_ in peak_region])
    num_components = len(params_array)
    print("num_components=", num_components)
    M_ = get_denoised_data(D[:,slice_], rank=num_components)
    E_ = E[:,slice_]
    cy_list = compute_cy_list(model, fx, params_array)
    plot_cy_list(cy_list, "initial state")

    min_msd = None
    column = LognormalPoreColumn()
    init_scales = None
    last_rgs = None
    for k in range(5):
        C = np.array(cy_list)
        C_ = C[:,slice_]
        Cinv = np.linalg.pinv(C_)
        P_ = M_ @ Cinv
        Minv = np.linalg.pinv(M_)
        W = Minv @ P_
        Pe = np.sqrt(E_**2 @ W**2)

        rg_list = []
        for j, p_ in enumerate(P_.T):
            data = np.array([qv, p_, Pe[:,j]]).T
            sg = SimpleGuinier(data)
            rg_list.append(sg.Rg)

        x0 = 300
        scale = 3
        print("rg_list=", rg_list)
        cy_list = []
        for rg in rg_list:
            cy = column.compute_curve(fx, x0, scale, rg)
            cy_list.append(cy)

        bifft = BidirectFft(fx)
        z = bifft.compute_z(y)
        w = bifft.get_w()
        u = bifft.compute_y(z)

        with plt.Dp():
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
            ax1.set_title("PDF Domain")
            ax1.plot(x, y)
            for cy in cy_list:
                ax1.plot(x, cy, ":")
            for x_ in peak_region:
                ax1.axvline(x=x_+x[0], color="green")
            ax2.set_title("CF Real Domain")
            ax2.plot(w, np.real(z))
            ax3.set_title("CF Imaginary Domain")
            ax3.plot(w, np.imag(z))
            ax1.plot(x, u, ":", color="cyan")
            fig.tight_layout()
            plt.show()
        break