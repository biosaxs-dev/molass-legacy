# coding: utf-8
"""
    Rgg.Spike.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time, sleep
import logging
import asyncio
import numpy as np
from scipy.stats import multivariate_normal
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from SliceUtils import slice_consecutives
from pomegranate import *
from molass_legacy.Peaks.ElutionModels import egh
from Prob.GaussianMixture import gaussian_pdf
import molass_legacy.KekLib.DebugPlot as plt
from .RggMixtureModel import RggMixtureModel
from .RggUtils import plot_histogram_2d
from molass_legacy.Peaks.ElutionModels import egh_pdf

def distributions_demo():
    x = np.arange(500)

    plt.push()
    fig, ax = plt.subplots()

    d = GaussianKernelDensity([200], bandwidth=10)
    ax.plot(x, d.probability(x))

    fig.tight_layout()
    plt.show()
    plt.pop()

    y = np.arange(100)
    xx, yy = np.meshgrid(x, y)
    plt.push()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cov = np.array([[900, 0], [0, 100]])
    rv = multivariate_normal([200, 50], cov)
    ax.plot_surface(xx, yy, rv.pdf(np.vstack([xx.flatten(), yy.flatten()]).T).reshape(xx.shape))

    fig.tight_layout()
    plt.show()
    plt.pop()

def generate_demo_data():
    x = np.arange(500)
    y = np.zeros(len(x))
    rg_list = []
    cy_list = []
    for rg, params in [
            # [0.6, 200, 40, 10],
            [40, [0.6, 200, 30,  5]],
            [30, [1.0, 300, 30, 10]],
            # [25, [0.4, 370, 30, 20]],
            ]:
        if True:
            cy = egh(x, *params)
        else:
            h, mu, sigma, tau = params
            cy = h*gaussian_pdf(x, mu, sigma)
        cy_list.append(cy)
        y += cy
        rg_list.append(rg)

    rg = np.sum(np.array(rg_list)[:,np.newaxis]*np.array(cy_list)/y, axis=0)
    return x, y, rg, len(cy_list)

def rgg_demo():
    from .RggEghDistribution import RggEghDistribution
    from RgProcess.RgCurve import convert_to_probabilitic_data
    x, y, rg, num_components = generate_demo_data()

    X = convert_to_probabilitic_data(x, y, rg)
    model = GeneralMixtureModel.from_samples(RggEghDistribution, num_components, X)
    print(model)

def spike_demo():
    x, y, rg, num_components = generate_demo_data()
    spike_demo_impl([(x, y, rg, None)], num_components=num_components)

def make_availability_slices(y, min_ratio):
    max_y = np.max(y)
    pairs = slice_consecutives(np.where(y/max_y >= min_ratio)[0])
    print("pairs=", pairs)
    slices = []
    states = []
    start = 0
    for f, t in pairs:
        if start < f:
            slices.append(slice(start, f))
            states.append(0)

        start = t+1
        slices.append(slice(f, start))
        states.append(1)
    if start < len(y):
        slices.append(slice(start, len(y)))
        states.append(0)
    return slices, states

PLOT_AS_SURFACE = True

def spike_demo_impl(data_list, num_components=None, use_peaks=False, seed=None, axes=None, plot_all=True, threed=True):
    from DataUtils import get_in_folder

    assert len(data_list) == 1

    if use_peaks:
        from molass_legacy.Peaks.RobustPeaks import RobustPeaks

    num_datasets = len(data_list)

    if axes is None:
        plt.push()
        fig = plt.figure(figsize=(21,6))
        gs = GridSpec(num_datasets, 3)
    else:
        fig = axes[0].figure

    fig.suptitle("2D EGH Mixture Model Demo for %s" % get_in_folder(), fontsize=20)

    fit_result = None
    for k, (x, y, rgc, X) in enumerate(data_list):
        max_y = np.max(y)

        model = RggMixtureModel(X=X, num_components=num_components)
        fit_result = model.fit()
        cy_list, ty = model.get_components(fit_result, x, y)

        if axes is None:
            ax1 = fig.add_subplot(gs[k,0])
            ax2 = fig.add_subplot(gs[k,1], projection='3d')
            projection = '3d' if threed else None
            ax3 = fig.add_subplot(gs[k,2], projection=projection)
        else:
            if plot_all:
                ax1, ax2, ax3 = axes
            else:
                ax3 = axes

        if plot_all:

            axt = ax1.twinx()
            axt.grid(False)

            ax1.set_title("Xray Elution and Rg Variation", fontsize=16)
            ax2.set_title("2D Histogram of Probability Variables", fontsize=16)
            random_text = "" if seed is None else " with seed=%d" % seed

            ax3.set_title("Decomposition Result%s" % random_text, fontsize=16)

            ax1.set_ylabel('Xray Intensity')
            ax1.set_xlabel('Eno')
            axt.set_ylabel('Rg')

            ax1.plot(x, y, color='orange')

            ax2.set_xlabel('Eno')
            ax2.set_ylabel('Rg')
            ax2.set_zlabel('Counts')
            ymin, ymax = ax1.get_ylim()
            segments = rgc.get_curve_segments()
            labeled = False
            for x_, y_, rg_ in segments:
                rg_label = None if labeled else 'Rg (smoothed)'
                axt.plot(x_, rg_, ':', color='C1', label=rg_label)
                rec_label = None if labeled else 'Rg-available range'
                p = Rectangle(
                        (x_[0], ymin),  # (x,y)
                        x_[-1] - x_[0], # width
                        ymax - ymin,    # height
                        facecolor   = 'cyan',
                        alpha       = 0.2,
                        label=rec_label,
                    )
                ax1.add_patch(p)
                labeled = True

            ax1.legend()
            axt.legend()
            ymin, ymax = axt.get_ylim()
            axt.set_ylim(0, ymax*1.2)

            k = 0
            for slice_, state in zip(rgc.slices, rgc.states):
                if state == 0:
                    continue

                x_ = x[slice_]
                rg = segments[k][2]     # Rg
                if PLOT_AS_SURFACE:
                    plot_histogram_2d(ax2, x_, y[slice_], rg, max_y)
                else:
                    z_ = y[slice_]/max_y*100
                    b_ = np.zeros(len(z_))
                    ax2.bar3d(x_, rg, b_, 0.5, 0.05, z_/max_y*100, shade=True, edgecolor='green')
                k += 1

            ymin, ymax = ax2.get_ylim()
            ax2.set_ylim(10, ymax)

        if threed:
            ax3.set_xlabel('Eno')
            ax3.set_ylabel('Rg')
            ax3.set_zlabel('Intensity')
            for w, m, s, t, rg in fit_result.get_params_for_refiner():
                y_ =  w * egh_pdf(x, m, s, t)
                rg_ = np.ones(len(x))*rg
                plot_histogram_2d(ax3, x, y_, rg_, max_y)
            ax3.set_ylim(10, ymax)
        else:
            ax3.set_ylabel('Xray Intensity')
            ax3.set_xlabel('Eno')
            ax3.plot(x, y, color='orange')

            for k, cy in enumerate(cy_list):
                ax3.plot(x, cy, ':', label='component-%d' % (k+1))
            ax3.plot(x, ty, ':', color='red', label='total')
            ax3.legend()

    if axes is None:
        fig.tight_layout()
        plt.show()
        plt.pop()

    return fit_result

def get_data_info(in_folder, correction, both=False):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from RgProcess.RgCurve import RgCurve

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    if correction:
        sd = sp.get_corrected_sd()
    else:
        sd = sp.get_sd()
    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    rgc = RgCurve(qv, xr_curve, D, E)
    # rgc.proof_plot()
    X = rgc.get_probabilistic_data()
    if both:
        return [D, E, qv, xr_curve, rgc, X], [sd.conc_array, sd.lvector, sd.get_uv_curve()]
    else:
        return D, E, qv, xr_curve, rgc, X

def spike_demo_real(in_folder, data_info=None, num_components=3, correction=True, use_peaks=False, seed=None, axes=None, refiner=True, plot_all=True):
    from molass_legacy._MOLASS.SerialSettings import set_setting

    logger = logging.getLogger(__name__)

    if data_info is None:
        D, E, qv, xr_curve, rgc, X = get_data_info(in_folder, correction)
    else:
        D, E, qv, xr_curve, rgc, X = data_info

    if seed is None:
        seed = np.random.randint(1000, 9999)
    np.random.seed(seed)
    print("seed=", seed)

    set_setting('in_folder', in_folder)
    fit_result = spike_demo_impl([[xr_curve.x, xr_curve.y, rgc, X]], num_components=num_components, use_peaks=use_peaks, seed=seed,
                                    axes=axes, plot_all=plot_all)

    logger.info("Mixture(%d) params=%s", seed, str(fit_result.get_params_for_refiner()))

    if refiner:
        from .RgRefiner import RgRefiner, plot_rgrefiner_result
        init_params = fit_result.get_params_for_refiner()
        print("init_params=", init_params)

        rr = RgRefiner(init_params)
        x_, y_, rg_ = rgc.get_valid_curves()
        rr.fit(x_, y_, rg_, debug=False)
        logger.info("Refiner(%d, %d) params=%s, func_value=%g", seed, rr.seed, str(rr.params), rr.func_value)

        plt.push()
        fig, ax = plt.subplots()
        x = xr_curve.x
        y = xr_curve.y
        plot_rgrefiner_result(ax, x, y, rgc, rr, seed)
        fig.tight_layout()
        plt.show()
        plt.pop()
    else:
        return rgc, fit_result

def spike_demo_many_trials(in_folder, num_components=3, plot_all=True, num_trials=1, mm_seed=None,rr_seeds=None, pause_at_turn=False):
    global finished

    print("num_trials=", num_trials)

    data_info, uv_info = get_data_info(in_folder, True, both=True)
    if False:
        from molass_legacy.Elution.CurveUtils import simple_plot
        D, E, qv, xr_curve, rgc, X = data_info
        U, wlvec, uv_curve = uv_info
        plt.push()
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))
        simple_plot(ax1, uv_curve)
        simple_plot(ax2, xr_curve, color='orange')
        fig.tight_layout()
        plt.show()
        plt.pop()

    for n in range(num_trials):

        dp = plt.push()
        fig  = plt.figure(figsize=(21,11))

        axes = np.empty((2,3), dtype=object)
        gs = GridSpec(2,3)
        for i in range(2):
            for j in range(3):
                projection = '3d'if plot_all and (i, j) in [(0,1), (0,2)] else None
                axes[i,j] = fig.add_subplot(gs[i,j], projection=projection)

        fig.tight_layout()
        fig.subplots_adjust(top=0.9, hspace=0.2)
        fig.canvas.draw()
        dp.update()

        if plot_all:
            mm_seed_ = np.random.randint(1000, 9999) if mm_seed is None else mm_seed
            rgc, fit_result = spike_demo_real(in_folder, data_info, num_components, seed=mm_seed_, axes=axes[0,:], refiner=False, plot_all=plot_all)
            j = 0
            axes_ = axes[1,:]
            dp.after(100, lambda: refiner_loop(dp, j, axes_, mm_seed_, rr_seeds, rgc, fit_result, plot_all, pause_at_turn))
        else:
            assert False

        plt.show()
        plt.pop()

def refiner_loop(dp, j, axes, mm_seed, rr_seeds, rgc, fit_result, plot_all, pause_at_turn):
    from .RgRefiner import RgRefiner, plot_rgrefiner_result
    global finished

    logger = logging.getLogger(__name__)
    init_params = fit_result.get_params_for_refiner()
    print("init_params=", init_params)
    rr = RgRefiner(init_params)
    x_, y_, rg_ = rgc.get_valid_curves()
    min_overlap = True
    seed = None if rr_seeds is None else rr_seeds[j]
    rr.fit(x_, y_, rg_, seed=seed, min_overlap=min_overlap, debug=False)
    logger.info("Refiner(%d, %d) params=%s, func_value=%g", mm_seed, rr.seed, str(rr.params), rr.func_value)

    ax = axes[j] if plot_all else axes
    ecurve = rgc.ecurve
    plot_rgrefiner_result(ax, ecurve.x, ecurve.y, rgc, rr, mm_seed, min_overlap=min_overlap)
    dp.fig.canvas.draw()
    if j < 2:
        dp.after(100, lambda: refiner_loop(dp, j+1, axes, mm_seed, rr_seeds, rgc, fit_result, plot_all, pause_at_turn))
    else:
        sleep(1)
        path = get_figure_path()
        dp.fig.savefig(path)
        if pause_at_turn:
            pass
        else:
            dp.cancel()

fig_no = -1
def get_figure_path():
    from molass_legacy._MOLASS.SerialSettings import get_setting
    global fig_no
    while True:
        fig_no += 1
        path  = os.path.join(get_setting("temp_folder"), "fig-%03d.jpg" % fig_no)
        if os.path.exists(path):
            continue
        else:
            break
    return path

def mm_refiner_combined_loop():
    pass

async def get_content(n):
    await asyncio.sleep(3)
    return n + 1

async def f(n):
    content = await get_content(n)
    return content

def coroutine_spike():
    loop = asyncio.get_event_loop()
    v = loop.run_until_complete(f(1))
    print(v)
