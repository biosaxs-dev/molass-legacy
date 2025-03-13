"""
    Models.Stochastic.SdmShapeStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import seaborn as sns

def study():
    from molass_legacy.Models.Stochastic.SdmGuessSingle import guess_sdm_params
    from molass_legacy.Models.Stochastic.DispersivePdf import dispersive_monopore_pdf
    import molass_legacy.KekLib.DebugPlot as plt
    from molass_legacy.Models.RateTheory.RobustEDM import edm_impl

    sns.set_theme()

    t0 = 50
    u = 1
    a = 1.0
    # b = -4.0
    b = 1.0
    e = 0.3
    Dz = 0.2
    cinj = 1

    x = np.arange(500)

    cinj = 1
    Dz = 0.5
    timescale = 0.25

    with plt.Dp():
        fig, axes = plt.subplots(ncols=3, figsize=(18, 5))
        fig.suptitle("SDM-EDM Correspondence Study", fontsize=20)
        for ax, b, title in zip(axes, [2, 0.0001, -2], ["Tailing Peak", "Symmetic Peak", "Fronting Peak"]):
            ax.set_title(title, fontsize=16)
            y = edm_impl(x, t0, u, a, b, e, Dz, cinj)
            ax.plot(x, y, label="EDM, b=%g" % b)
            K, N, t0, N0, rho, me, mp, scale = guess_sdm_params(x, y, timescale=timescale)
            T = K/N
            n1 = N*(1 - rho)**me
            t1 = T*(1 - rho)**mp
            y_ = scale*dispersive_monopore_pdf(x, n1, t1, N0, t0, timescale=timescale)
            ax.plot(x, y_, ":", lw=3, label="SDM, $t_0$=%.0f" % t0)
            if t0 > 0:
                ax.axvline(t0, color="red", label="$t_0$")
            ax.legend()
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("..\\lib")
    study()