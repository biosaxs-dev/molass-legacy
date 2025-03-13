"""
    Models.Stochastic.RoughGuess.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from matplotlib.widgets import Slider, CheckButtons
import molass_legacy.KekLib.DebugPlot as plt
from Experiment.ColumnTypes import get_all_poresizes
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.ModelUtils import compute_raw_moments1, compute_area_props
from molass_legacy.Models.Stochastic.ParamLimits import MAX_PORESIZE
from importlib import reload
import Models.Stochastic.MonoporeUtils
reload(Models.Stochastic.MonoporeUtils)
from molass_legacy.Models.Stochastic.MonoporeUtils import compute_monopore_curves, plot_monopore_curves, adjust_scales_to_egh_params_array

NUM_STAYS_LIST = [1000, 2000, 3000, 4000, 5000]
M1_WEIGHT = 9
M2_WEIGHT = 0.5
M3_WEIGHT = 0.5

def guess_monopore_params_roughtly(x, y, model, peaks, peak_rgs, props, egh_moments_list, debug=False):
    if debug:
        print("props=", props)
        for k, moments in enumerate(egh_moments_list):
            print([k], moments)

    if False:
        from molass_legacy.Models.ModelUtils import data_dump
        data_dump(x, y, peaks, prefix="roughtly-")

    me = 1.5
    mp = 1.5
    max_rg = np.max(peak_rgs)
    poresizes = np.asarray(get_all_poresizes())
    possible_poresizes = poresizes[poresizes > max_rg + 20]
    dev_list = []
    for poresize in possible_poresizes:
        for N in NUM_STAYS_LIST:
            def rough_objective(p, debug=False):
                T_, x0_ = p
                dev_list = []
                cy_list = []
                for k, (rg, M) in enumerate(zip(peak_rgs, egh_moments_list)):
                    rho = min(1, rg/poresize)
                    M1_ = x0_ + N * T_ * (1 - rho)**(me+mp)
                    M2_ = 2 * N * T_**2 * (1 - rho)**(me+2*mp)
                    dev = M1_WEIGHT*(M1_ - M[0])**2 + M2_WEIGHT*(M2_ - M[1])**2
                    dev_list.append(dev)
                    ni_ = N * (1 - rho)**me
                    ti_ = T_ * (1 - rho)**mp
                    cy = props[k] * robust_single_pore_pdf(x - x0_, ni_, ti_)
                    cy_list.append(cy)
                mdev = np.sum(np.asarray(dev_list)*props)
                if debug:
                    print("poresize, N, T, x0=", poresize, N, T_, x0_)
                return mdev
            init_T = 0.5
            init_x0 = 0
            bounds = ((0, 10), (-500, 200))
            res = minimize(rough_objective, [init_T, init_x0], bounds=bounds, method="Nelder-Mead")
            dev_list.append((poresize, N, *res.x, res.fun))
    dev_list = np.asarray(dev_list)

    # remove outliers so that we can get a better view in 3D debug plot
    fv_temp = dev_list[:,-1]
    k8 = int(0.8*len(fv_temp))
    pp = np.argpartition(fv_temp, k8)
    p8 = sorted(pp[0:k8])
    fv_x = dev_list[p8,0]
    fv_y = dev_list[p8,1]
    fv_T = dev_list[p8,2]
    fv_x0 = dev_list[p8,3]
    fv_z = dev_list[p8,-1]
    k = np.argmin(fv_z)

    if debug:
        def get_nth_params(k):
            poresize = fv_x[k]
            N = fv_y[k]
            T = fv_T[k]
            x0 = fv_x0[k]
            print([k], "N, T, x0=", N, T, x0)
            return np.concatenate([[N, T, x0, me, mp, poresize], props])

        with plt.Dp():
            fig = plt.figure(figsize=(12, 5))
            fig.suptitle("guess_monopore_params_roughtly Debug & Proof", fontsize=20)
            ax1 = fig.add_subplot(121, projection="3d")
            ax2 = fig.add_subplot(122)
            ax1.set_title("fv vs. poresize vs. num_stays", fontsize=16)
            ax1.set_xlabel("poresize")
            ax1.set_ylabel("num_stays")
            ax1.set_zlabel("fv")
            ax1.plot(fv_x, fv_y, fv_z, "o")
            ax1.plot(fv_x[k], fv_y[k], fv_z[k], "o", color="red")
            # zmin, zmax = ax1.get_zlim()
            # ax.set_zlim(zmin, 1000)
            ax2.set_title("curves with selected params", fontsize=16)
            plot_params = get_nth_params(k)
            poresize, N = plot_params[[5,0]]
            rough_objective(plot_params[1:3], debug=True)
            ax2.plot(x, y, label="data")
            artists = plot_monopore_curves(ax2, x, plot_params, peak_rgs, return_artists=True)

            rax = ax1.inset_axes([1.05, 0.0, 0.3, 0.1])
            selecting = False
            check = CheckButtons(
                ax=rax,
                labels=["Selecting"],
                actives=[selecting],
                )
            def check_on_click(label):
                nonlocal selecting
                if label == "Selecting":
                    selecting = not selecting
                    fig.canvas.draw_idle()
            check.on_clicked(check_on_click)

            def on_click_to_select(event):
                if not selecting:
                    return
                if event.inaxes != ax1:
                    return

                ix, iy = event.xdata, event.ydata
                print("ix, iy=", ix, iy)
                p1, pane_idx = ax1._calc_coord(ix, iy)  # see matplotlib/lib/mpl_toolkits/mplot3d/axes3d.py as of matplotlib 3.8.3
                ix_, iy_, iz_ = p1
                # Find nearest point
                points = np.c_[fv_x, fv_y, fv_z]
                tree = cKDTree(points)
                dist, idx = tree.query([ix_, iy_, iz_])
                N_, poresize_, fv_ = fv_x[idx], fv_y[idx], fv_z[idx]
                print("idx=", idx, N_, poresize_, fv_)
                # Highlight nearest point
                ax1.scatter(fv_x[idx], fv_y[idx], fv_z[idx], c='r')     # task: should update the artist instead of adding new one

                plot_params = get_nth_params(idx)
                cy_list, ty = compute_monopore_curves(x, plot_params[0:6], peak_rgs, plot_params[6:])
                for k, y_ in enumerate(cy_list + [ty]):
                    artists[k].set_ydata(y_)            
                # plt.draw()
                fig.canvas.draw_idle()

            fig.canvas.mpl_connect('button_press_event', on_click_to_select)
            fig.tight_layout()
            # fig.subplots_adjust(right=0.7)
            ret = plt.show()
        if not ret:
            return

    cy_list = []
    for k, params in enumerate(peaks):
        cy = model(x, params)
        cy_list.append(cy)
    ty = np.sum(cy_list, axis=0)

    def plot_curves(ax, cy_list, ty, return_artists=False):
        ax.plot(x, y, label='data')
        artists = []
        for k, cy in enumerate(cy_list):
            curve, = ax.plot(x, cy, ":", label='component-%d' % k)
            artists.append(curve)
        curve, = ax.plot(x, ty, ":", color="red", label='model total')
        artists.append(curve)
        if return_artists:
            return artists

    # N, T, x0, me, mp, poresize
    poresize, N, T, x0 = dev_list[k,0:4]
    if debug:
        print("poresize, N, T, x0=", poresize, N, T, x0)
        print("peak_rgs=", peak_rgs)
        # print("rhov=", rhov)

    abort = False
    use_multi_factor_fv = False
    if use_multi_factor_fv:
        raw_moments = np.array([M[0] for M in egh_moments_list])
    optimize_poresize = False
    if not optimize_poresize:
        rhov = peak_rgs/poresize
        rhov[rhov > 1] = 1

    def scales_objective(p, title=None, return_curves=False, with_sliders=False):
        nonlocal abort, rhov
        if optimize_poresize:
            rp, N_, T_, x0_ = p[0:4]
            rhov = peak_rgs/rp
            rhov[rhov > 1] = 1
            scales_ = p[4:]
        else:
            N_, T_, x0_ = p[0:3]
            scales_ = p[3:]
        cy_list = []
        for k, (rho, scale) in enumerate(zip(rhov, scales_)):
            ni_ = N_*(1 - rho)**me
            ti_ = T_*(1 - rho)**mp
            cy = scale*robust_single_pore_pdf(x - x0_, ni_, ti_)
            cy_list.append(cy)
            # print([k],  x0_ + ni_*ti_ - m)    # m == x0_ + ni_*ti_, approximately
        ty_ = np.sum(cy_list, axis=0)
        if return_curves:
            return cy_list, ty_

        if use_multi_factor_fv:
            raw_moments_ = compute_raw_moments1(x, cy_list)
            props_ = compute_area_props(cy_list)

        if title is not None:
            figsize = (12, 5) if with_sliders else None
            print("params=", p)
            with plt.Dp():
                fig, ax = plt.subplots(figsize=figsize)
                ax.set_title(title)
                artists = plot_curves(ax, cy_list, ty_, return_artists=with_sliders)
                ax.legend()
                if with_sliders:
                    assert not optimize_poresize
                    update_params = p.copy()
                    slider_specs = [    ("N", 300, 4000, update_params[0]),
                                        ("T", 0, 3, update_params[1]),
                                        ("x0",  -500, 500, update_params[2]),
                                        ]
                    for k, scale in enumerate(scales_):
                        slider_specs.append(("Scale-%d" % k, 0, 2, update_params[3+k]))
                        
                    def slider_update(k, val):
                        print([k], "slider_update", val)
                        update_params[k] = val
                        redraw_params = np.concatenate([update_params[0:3], [me, mp, poresize], update_params[3:]])
                        cy_list, ty_ = compute_monopore_curves(x, redraw_params[0:6], peak_rgs, redraw_params[6:])
                        for k, (curve, y_) in enumerate(zip(artists, cy_list + [ty_])):
                            print([k], np.max(y_))
                            curve.set_ydata(y_)
                        fig.canvas.draw_idle()

                    slider_axes = []
                    sliders = []
                    for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
                        ax_ = fig.add_axes([0.8, 0.7 - 0.08*k, 0.11, 0.03])
                        slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
                        slider.on_changed(lambda val, k_=k: slider_update(k_, val))
                        slider_axes.append(ax_)
                        sliders.append(slider)                   

                    fig.tight_layout()
                    fig.subplots_adjust(right=0.6)
                else:
                    fig.tight_layout()
                ret = plt.show()
            if not ret:
                abort = True

        if use_multi_factor_fv:
            return np.sum((ty_ - ty)**2)**2 + 0.0001*np.sum((raw_moments_ - raw_moments)**2)**2 + 0.0001*np.sum((props_ - props)**2)**2
        else:
            return np.sum((ty_ - ty)**2)

    num_components = len(peak_rgs)

    init_x0 = x0
    column_params = [N, T, init_x0]
    adjust_params = column_params + [me, mp, poresize]
    adjusted_scales = adjust_scales_to_egh_params_array(x, y, adjust_params, peak_rgs, peaks)

    bounds = [(300, 5000), (0.01, 10), (init_x0-500, init_x0+500)] + [(0, 100)]*num_components
    if optimize_poresize:
        column_params.insert(0, poresize)
        bounds.insert(0, (20, MAX_PORESIZE))
    init_params = np.concatenate([column_params, adjusted_scales])
    if debug:
        scales_objective(init_params, title="before scales_objective minimize", with_sliders=True)
        if abort:
            return

    res = minimize(scales_objective, init_params, bounds=bounds, method="Nelder-Mead")
    if debug:
        scales_objective(res.x, title="after scales_objective minimize")
        if abort:
            return

        with plt.Dp(button_spec=["OK", "Cancel"]):
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
            fig.suptitle("guess_monopore_params_roughtly")
            plot_curves(ax1, cy_list, ty)
            plot_curves(ax2, *scales_objective(res.x, return_curves=True))
            fig.tight_layout()
            ret = plt.show()
        if not ret:
            return

    if optimize_poresize:
        poresize, N, T, x0 = res.x[0:4]
        scales = res.x[4:]
    else:
        N, T, x0 = res.x[0:3]
        scales = res.x[3:]

    return np.concatenate([[N, T, x0, me, mp, poresize], scales])