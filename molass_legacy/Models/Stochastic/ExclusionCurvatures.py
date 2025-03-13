"""
    Models.Stochastic.ExclusionCurvatures.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

def tripore_simulation():
    from molass_legacy.Models.Stochastic.LognormalPoreFunc import distr_func
    from molass_legacy.Models.Stochastic.ParamLimits import MAX_PORESIZE
    from molass_legacy.Models.Stochastic.LognormalUtils import compute_mu_sigma
    N = 1000
    T = 1
    t0 = 50
    me = 1.5
    mp = 1.5
    mnp_sizes = np.array([150, 150, 150])
    mnp_props = np.array([1, 1, 1])
    mnp_props = mnp_props/np.sum(mnp_props)
    sizes = np.array([100, 140, 200])
    props = np.array([2, 1, 3])
    props = props/np.sum(props)

    rv = np.linspace(10, 120, 100)
    def compute_tripore_tv(N, T, t0, me, mp, sizes, props):
        tv = []
        for rg in rv:
            rhov = rg/sizes
            rhov[rhov > 1] = 1
            tr = t0 + N * T * np.sum(props * (1 - rhov)**(me + mp))
            tv.append(tr)
        return np.asarray(tv)

    def compute_lnpore_tv(N, T, t0, me, mp, mu, sigma):
        tv = []
        for rg in rv:
            rhov = rg/sizes
            rhov[rhov > 1] = 1
            tr = t0 + N * T * quad(lambda r : distr_func(r, mu, sigma) * (1 - min(1, rg/r))**(me +   mp), rg, MAX_PORESIZE)[0]
            tv.append(tr)
        return np.asarray(tv)

    tv_mnp = compute_tripore_tv(N, T, t0, me, mp, mnp_sizes, mnp_props)
    tv = compute_tripore_tv(N, T, t0, me, mp, sizes, props)
    mu, sigma = compute_mu_sigma(150, 10)
    tv_lnp = compute_lnpore_tv(N, T, t0, me, mp, mu, sigma)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18,5))
    fig.suptitle("Exclusion Curve Simulations of Monopore, Tripore, and Lognormalpore PSDs", fontsize=20)
    ax1.set_title("Exclusion Curves", fontsize=16)
    mnp_curve, = ax1.plot(tv_mnp, rv, alpha=0.5, label="monopore")
    trp_curve, = ax1.plot(tv, rv, alpha=0.5, label="tripore")
    lnp_curve, = ax1.plot(tv_lnp, rv, alpha=0.5, label="lognormalpore")
    ax1.axvline(x=t0, color="red", label="$t_0$")
    ax1.legend()
    ax1.set_xlabel("Retension Time (Frames)")
    ax1.set_ylabel(r"Rg ($\AA$)")

    ax2.set_title("Pore Size Distributions", fontsize=16)
    ax2.bar(mnp_sizes, mnp_props, width=5, alpha=0.5, label="monopore")
    ax2.bar(sizes, props, width=5, alpha=0.5, label="tripore")
    pv = np.linspace(10, 400, 100)   
    ax2.fill_between(pv, np.zeros(len(pv)), distr_func(pv, mu, sigma), alpha=0.5, label="lognormalpore")
    ax2.legend()
    ax2.set_xlabel(r"Pore Size ($\AA$)")
    ax2.set_ylabel("Density")

    # https://stackoverflow.com/questions/22263807/how-is-order-of-items-in-matplotlib-legend-determined
    handles, labels = ax2.get_legend_handles_labels()
    order = [1, 2, 0]
    ax2.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    ax2.set_xlim(0, 400)
    ax2.set_ylim(0, 0.6)

    patch_list = []
    for c in ax2.containers:
        print(len(c.patches))
        patch_list.append(c.patches)
    print(len(patch_list))

    plot_params = np.concatenate([[N, T, t0, me, mp, 150], sizes, props, [150, 10]])

    slider_specs = [    ("N", 0, 6000, plot_params[0]),
                        ("T", 0, 3, plot_params[1]),
                        ("t0",  -1000, 1000, plot_params[2]),
                        ("me", 0, 3, plot_params[3]),
                        ("mp", 0, 3, plot_params[4]),
                        ("mono poresize", 20, 400, plot_params[5]),
                        ("tri poresize1", 20, 400, plot_params[6]),
                        ("tri poresize2", 20, 400, plot_params[7]),
                        ("tri poresize3", 20, 400, plot_params[8]),                        
                        ("tri proportion1", 0, 1, plot_params[9]),
                        ("tri proportion2", 0, 1, plot_params[10]),
                        ("tri proportion3", 0, 1, plot_params[11]),       
                        ("lognormal mode", 20, 400, plot_params[12]),
                        ("lognormal stdev", 1, 100, plot_params[13]),       
                        ]

    def slider_update(k, val):
        print([k], "slider_update", val)
        plot_params[k] = val
        N, T, t0, me, mp = plot_params[0:5]

        if k == 5:
            i = 0
            for patch in patch_list[i]:
                patch.set_x(val)
            mnp_sizes = np.ones(3)*val
            tv_mnp = compute_tripore_tv(N, T, t0, me, mp, mnp_sizes, mnp_props)
            mnp_curve.set_xdata(tv_mnp)

        if 6 <= k and k <= 11:
            i = 1
            if k < 9:
                j = k - 6
                patch = patch_list[i][j]
                patch.set_x(val)
            elif k < 12:
                j = k - 9
                patch = patch_list[i][j]
                patch.set_height(val)

                indeces = np.setdiff1d(np.arange(3), j)
                props = plot_params[9 + indeces]   
                props /= np.sum(props)       
                print("indeces", indeces, "props=", props)
                for m, p in zip(indeces, props):
                    v = p*(1-val)
                    patch_list[i][m].set_height(v)
                    k_ = 9 + m
                    plot_params[k_] = v
                    slider = sliders[k_]
                    slider.eventson = False
                    slider.set_val(v)
                    slider.eventson = True
            else:
                # to be implemented
                pass

            sizes = plot_params[6:9]
            props = plot_params[9:12]
            tv = compute_tripore_tv(N, T, t0, me, mp, sizes, props)
            trp_curve.set_xdata(tv)

        fig.canvas.draw_idle()

    slider_axes = []
    sliders = []
    for k, (label, valmin, valmax, valinit) in enumerate(slider_specs, start=0):
        ax_ = fig.add_axes([0.75, 0.8 - 0.05*k, 0.2, 0.03])
        slider  = Slider(ax_, label=label, valmin=valmin, valmax=valmax, valinit=valinit)
        slider.on_changed(lambda val, k_=k: slider_update(k_, val))
        slider_axes.append(ax_)
        sliders.append(slider)

    def reset(event):
        print("reset")
        for k, slider in enumerate(sliders):
            slider.reset()

    button_ax = fig.add_axes([0.85, 0.07, 0.12, 0.05])
    debug_btn = Button(button_ax, 'Reset', hovercolor='0.975')
    debug_btn.on_clicked(reset)

    fig.tight_layout()
    fig.subplots_adjust(left=0.06, right=0.67, wspace=0.2)
    plt.show()

if __name__ == "__main__":
    import sys
    import seaborn as sns
    sns.set_theme()
    sys.path.append("../lib")

    tripore_simulation()