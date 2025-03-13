"""
    Models.Stochastic.MonoporeUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
import molass_legacy.KekLib.DebugPlot as plt
from SecTheory.BasicModels import robust_single_pore_pdf
from molass_legacy.Models.Stochastic.MonoporeUtilsImpl import compute_monopore_curves

def plot_monopore_curves(ax, x, monopore_params, rgs, return_artists=False):
    cy_list, ty = compute_monopore_curves(x, monopore_params[0:6], rgs, monopore_params[6:])
    artists = []
    for k, cy in enumerate(cy_list):
        curve, = ax.plot(x, cy, ":", label='component-%d' % k)
        artists.append(curve)
    curve, = ax.plot(x, ty, ":", color="red", label='model total')
    artists.append(curve)
    if return_artists:
        return artists

def adjust_scales_to_egh_params_array(x, y, adjust_params, rgs, peaks, debug=False):
    N, T, x0, me, mp, poresize = adjust_params
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    scales = []
    if debug:
        cy_list = []
    for k, (rho, params) in enumerate(zip(rhov, peaks)):
        ni_ = N*(1 - rho)**me
        ti_ = T*(1 - rho)**mp
        cy = robust_single_pore_pdf(x - x0, ni_, ti_)
        if debug:
            cy_list.append(cy)
        scale = params[0]/np.max(cy)
        print([k], params, scale)
        scales.append(scale)

    if debug:
        def plot_curves(ax, cy_list, scales=None):
            ax.plot(x, y, label='data') 
            for k, cy in enumerate(cy_list):
                if scales is not None:
                    cy *= scales[k]
                ax.plot(x, cy, ":", label='component-%d' % k)
            ty = np.sum(cy_list, axis=0)
            ax.plot(x, ty, ":", color="red", label='model total')
            ax.legend()

        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            fig.suptitle("PDF vs. Scaled", fontsize=20)
            ax1.set_title("PDF", fontsize=16)
            plot_curves(ax1, cy_list)
            ax2.set_title("Scaled", fontsize=16)
            plot_curves(ax2, cy_list, scales=scales)
            fig.tight_layout()
            plt.show()

    return scales

def draw_exclusion_cuve(axt, monopore_columnparams, trs, rgs):
    N, T, x0, me, mp, poresize = monopore_columnparams
    if np.isscalar(poresize):
        rv = np.linspace(poresize, 10, 100)
        rhov = rv/poresize
        tv = x0 + N*T*(1 - rhov)**(me+mp)
        axt.plot(tv, rv, lw=2, color="yellow")
        axt.plot(trs, rgs, "o", color="red")
        axt.axvline(x=x0, color="red")

def plot_monopore_moments_state(title, x, y, peak_rgs, mnp_params, x0_upper, cy_list, ty, moments, egh_moments_list):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder
    import Models.Stochastic.MonoporeUtilsImpl as module

    def plot_monopore_component_with_sliders():
        from importlib import reload
        import Models.Stochastic.MonoporeUtilsImpl as module
        reload(module)
        from molass_legacy.Models.Stochastic.MonoporeUtilsImpl import plot_monopore_component_with_sliders_impl
        plot_monopore_component_with_sliders_impl(title, x, y, peak_rgs, mnp_params, x0_upper, cy_list, ty, moments, egh_moments_list)

    print("plot_monopore_moments_state: peak_rgs=", peak_rgs)

    N, T, x0_, me, mp, poresize = mnp_params[0:6]
    num_peaks = len(peak_rgs)
    extra_button_specs = [("Plot Component with Sliders", plot_monopore_component_with_sliders)]
    with plt.Dp(button_spec=["OK", "Cancel"],
                extra_button_specs=extra_button_specs):
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle(title)
        ax2.set_title("Moments Fitting for %s" % get_in_folder())
        ax2.set_xlabel("Time (frames)")
        ax2.set_ylabel("Intensity")
        for ax in (ax1,ax2):
            ax.plot(x, y, color='orange', label='data')
            ax.plot(x, ty, ':', color="red", label='model total')
            for k, (cy, M, eghM) in enumerate(zip(cy_list, moments, egh_moments_list)):
                ax.plot(x, cy, ':', label='component-%d' % k)
                label = None if k < num_peaks - 1 else "Monopore Moments"
                ax.axvline(x=M[0], color="green", label=label)
                label = None if k < num_peaks - 1 else "EGH Moments"
                ax.axvline(x=eghM[0], ls=":", lw=3, color="cyan", label=label)
            ax.axvline(x=x0_upper, color="pink")
            axt = ax.twinx()
            axt.grid(False)
            trs = [M[0] for M in moments]
            draw_exclusion_cuve(axt, (N, T, x0_, me, mp, poresize), trs, peak_rgs)
        # xmin, xmax = ax1.get_xlim()
        # ymin, ymax = ax1.get_ylim()
        m1, m2 = moments[0]
        xmin = m1 - 10*np.sqrt(m2)
        m1, m2 = moments[-1]
        xmax = m1 + 10*np.sqrt(m2)                
        ax2.set_xlim(xmin, xmax)
        ax2.legend()
        fig.tight_layout()
        ret = plt.show()

    return ret

def plot_monopore_conformance(K, t0, poresize, plot_info, rg_info=None, confrg_info=None, plot_cuvatures=False):
    from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

    rv = np.linspace(10, 100, 100)
    rhov = rv/poresize
    rhov[rhov > 1] = 1
    tv = t0 + K*np.power(1 - rhov, 3)
    x = plot_info.x
    y = plot_info.y
    model = plot_info.model
    cy_list = []
    for params in plot_info.peaks:
        cy = model(x, params)
        cy_list.append(cy)

    with plt.Dp():
        if plot_cuvatures:
            fig, (ax,ax2) = plt.subplots(ncols=2, figsize=(18,5))
        else:
            fig, ax = plt.subplots(figsize=(9,5))

        fig.suptitle("Exclusion Curve Estimate on %s" % get_in_folder(), fontsize=20)

        ax.set_title("Data Exclusion Points against Estimated Exclusion Curve", fontsize=16)
        ax.set_xlabel("Time (Frames)")
        ax.set_ylabel("Intensity")
        ax.plot(x, y, color="orange")
        for cy in cy_list:
            ax.plot(x, cy, ":")

        ax.axvline(x=t0, color="red", label="$t_0$")
        ax.legend(loc="upper left")

        axt = ax.twinx()
        axt.grid(False)
        axt.set_ylabel(r"$R_g (\AA_)$")
        axt.plot(tv, rv, color='yellow', lw=3, label="Estimated Exclusion Curve: K=%.4g, Rp=%.3g" % (K, poresize))
        ry = None
        if rg_info is not None:
            rgs, trs = rg_info[0:2]
            axt.plot(trs, rgs, "o")
            if len(rg_info) > 2:
                from scipy.interpolate import UnivariateSpline
                if rg_info[2]:  # draw spline
                    spline = UnivariateSpline(trs, rgs)
                ry = spline(tv)
                axt.plot(tv, ry, color="cyan", alpha=0.5, lw=3, label="Spline of Exclusion Points")
                
        if confrg_info is not None:
            rgs, trs = confrg_info
            axt.plot(trs, rgs, "o")

        ymin, ymax = axt.get_ylim()
        axt.set_ylim(max(0, ymin), ymax)

        axt.legend()

        if plot_cuvatures:
            from molass_legacy.KekLib.OurCurvature import curvature_curve
            ax2.set_title("Corresponding Curvatures", fontsize=16)
            ax2.set_xlabel("Time (Frames)")
            ax2.set_ylabel("Curvature")

            # to avoid ValueError: x must be strictly increasing if s = 0
            t = np.flip(tv)     
            diff_t = np.diff(t)
            diff_t_pos = np.where(diff_t > 0)[0]
            p = diff_t_pos[0]
            t_ = t[p:]
            try:
                cy = curvature_curve(t_, np.flip(rv)[p:])

                # to avoid anomalies
                m = np.argmax(cy)
                # there can be anomalies beyond m as in 20191006_OA_Ald_Ferr
                diff_cy = np.diff(cy[m:])
                diff_cy_pos = np.where(diff_cy > 0)[0]
                if len(diff_cy_pos) > 0:
                    m_ = m + diff_cy_pos[-1] + 1
                else:
                    # as in 20210727/data02 
                    m_ = m

                ax2.plot(t_[m_:], cy[m_:], color="yellow", lw=3, label="Estimated Exclusion Curve Curvature")
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(None, "curvature_curve rv failure: ")
            if ry is not None:
                try:
                    cy_ = curvature_curve(t_, np.flip(ry)[p:])
                    ax2.plot(t_, cy_, color="cyan", alpha=0.5, lw=3, label="Spline Curvature of Exclusion Points")
                except:
                    from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                    log_exception(None, "curvature_curve ry failure: ")
            ax2.legend()

        fig.tight_layout()
        ret = plt.show()

    return ret

def guess_monopore_colparams(rgs, qualities, trs, init_params, debug=False, plot_info=None):

    def objective(p):
        K, t0, poresize = p
        rhov = rgs/poresize
        rhov[rhov > 1] = 1
        trs_ = t0 + K*(1 - rhov)**3
        return np.sum((trs_ - trs)**2 * qualities)

    init_params_ = init_params[1:]      # optimize except for N
    res = minimize(objective, init_params_)

    if debug and plot_info is not None:
        ret = plot_monopore_conformance(*res.x, plot_info, rg_info=(rgs, trs, True), plot_cuvatures=False)
        if not ret:
            return

    return np.concatenate([[1000], res.x])      # always return N=1000
