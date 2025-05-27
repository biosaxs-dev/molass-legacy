"""
    MappingUtils.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""

import molass_legacy.KekLib.DebugPlot as plt

def debug_plot_mapping(mapper, title="debug_plot_mapping"):
    from scipy.interpolate import UnivariateSpline

    a_curve = mapper.a_curve
    a_base = mapper.a_base
    x_curve = mapper.x_curve
    x_base = mapper.x_base

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21, 7))
        fig.suptitle(title, fontsize=20)
        ax1.set_title("UV Elution", fontsize=16)
        ax2.set_title("Xray Elution", fontsize=16)
        ax3.set_title("Mapped Elutions", fontsize=16)

        for ax, curve, color, base in [(ax1, a_curve, None, a_base), (ax2, x_curve, 'orange', x_base)]:
            x = curve.x
            y = curve.y
            yb = y + base
            ax.plot(x, yb, color=color, label='data')
            ax.plot(x, base, ':', color='red')

            top = curve.peak_top_x
            bnd = curve.boundaries
            d_spline = UnivariateSpline(x, yb, s=0, ext=3)
            ax.plot(top, d_spline(top), 'o', color='red', label='peak tops')
            if len(bnd) > 0:
                b_spline = UnivariateSpline(x, base, s=0, ext=3)
                ax.plot(bnd, b_spline(bnd), 'o', color='green', label='valley bottoms')
            ax.legend()

        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
    plt.show()

def save_mapped_curves(mapper, plot=False):
    import os
    import numpy as np
    from molass_legacy._MOLASS.SerialSettings import get_setting
    analysis_folder = get_setting('analysis_folder')

    x = mapper.x_curve.x
    mapped_uv_vector = mapper.make_uniformly_scaled_vector(scale=1)

    if plot:
        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
            for k, (name, curve) in enumerate([("uv_curve", mapper.a_curve), ("xr_curve", mapper.x_curve)]):
                ax = ax1 if k == 0 else ax2
                ax.plot(curve.x, curve.y, label=name)

            ax2.plot(x, mapped_uv_vector, label="mapped uv")
            ax2.legend()
            fig.tight_layout()
            plt.show()

    for name, curve in [("uv_curve", mapper.a_curve), ("xr_curve", mapper.x_curve)]:
        file = os.path.join(analysis_folder, name + ".dat")
        np.savetxt(file, np.array([curve.x, curve.y]).T)

    file = os.path.join(analysis_folder, "mapped_uv_curve.dat")
    np.savetxt(file, np.array([x, mapped_uv_vector]).T)
