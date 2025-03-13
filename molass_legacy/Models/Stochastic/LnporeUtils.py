"""
    Models.Stochastic.LnporeUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def plot_lognormal_fitting_state(x, y, lnp_params, rgs, plot_boundary=False, title=None, save_fig_as=None):
    from matplotlib.gridspec import GridSpec
    import molass_legacy.KekLib.DebugPlot as plt
    from importlib import reload
    import Models.Stochastic.LnporeUtilsImpl as module
    reload(module)
    from molass_legacy.Models.Stochastic.LnporeUtilsImpl import plot_elution_exclusion_impl, plot_lognormal_psd_impl

    def plot_lnpore_component_with_sliders():
        reload(module)
        from molass_legacy.Models.Stochastic.LnporeUtilsImpl import plot_lnpore_component_with_sliders_impl
        plot_lnpore_component_with_sliders_impl(x, y, lnp_params, rgs)

    extra_button_specs = [("Plot Component with Sliders", plot_lnpore_component_with_sliders)]

    with plt.Dp(button_spec=["OK", "Cancel"],
                extra_button_specs=extra_button_specs):
        if plot_boundary:
            fig = plt.figure(figsize=(20,5))
            gs = GridSpec(1,7)
            ax1 = fig.add_subplot(gs[0,0:3])
            ax2 = fig.add_subplot(gs[0,3:5])
            ax3 = fig.add_subplot(gs[0,5:7])
        else:
            fig = plt.figure(figsize=(14,5))
            gs = GridSpec(1,5)
            ax1 = fig.add_subplot(gs[0,0:3])
            ax2 = fig.add_subplot(gs[0,3:5])
        if title is None:
            title = "Lognormal Fitting"
        fig.suptitle(title, fontsize=20)
        plot_elution_exclusion_impl(ax1, x, y, lnp_params, rgs)
        mu, sigma = lnp_params[5:7]
        plot_lognormal_psd_impl(ax2, mu, sigma)
        if plot_boundary:
            from Simulative.ApproxBoundary import compute_approx_boundary, plot_boundary_impl
            rv, bv = compute_approx_boundary(x, y, lnp_params, rgs, quickly=True)
            plot_boundary_impl(ax3, rv, bv, mu, sigma)

        fig.tight_layout()
        if save_fig_as is None:
            ret = plt.show()
        else:
            plt.show(block=False)
            fig.savefig(save_fig_as)
            ret = True            
    return ret