# coding: utf-8
"""
    SecTheory.Distinction.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def hard_sphere_curves():
    from bisect import bisect_right
    from Theory.SolidSphere import phi
    from Theory.Rg import compute_Rg
    from Error.ErrorModel import error_model

    qv = np.linspace(0.005, 0.5, 500)

    R50, R30 = [np.sqrt(5/3)*r for r in [50, 30]]

    y50 = phi(qv, R50)**2
    y30 = phi(qv, R30)**2
    ymix = (y50+y30)/2

    i = bisect_right(qv, 0.01)
    for y in [y50, y30, ymix]:
        rg = compute_Rg(qv[0:i], y[0:i])
        print("Rg=", rg)

    k = 1e4
    c = 0.05
    e50, e30, emix = [error_model(qv, y, k, c, y[0]) for y in [y50, y30, ymix]]

    with plt.Dp():
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18,5))

        ax1.set_title("Individual Curves", fontsize=20)
        ax2.set_title("Mixture Curve", fontsize=20)
        ax3.set_title("Error Curves", fontsize=20)

        for ax in [ax1, ax2, ax3]:
            ax.set_yscale("log")

        ax1.plot(qv, y50, label="Sphere Rg=50")
        ax1.plot(qv, y30, label="Sphere Rg=30")
        ax2.plot(qv, ymix, label="1:1 Mixture")

        ax3.plot(qv, e50, label="Rg=50")
        ax3.plot(qv, e30, label="Rg=30")
        ax3.plot(qv, emix, label="1:1 Mixture")

        for ax in [ax1, ax2, ax3]:
            ax.legend(fontsize=16)

        fig.tight_layout()
        plt.show()

    for name, rec in [
            ("rg50.dat", (qv, y50, e50)),
            ("rg30.dat", (qv, y30, e30)),
            ("mix.dat", (qv, ymix, emix))]:
        np.savetxt(name, np.array(rec).T)

def sec_elution_curves():
    from .BasicModels import SimpleSec

    rp0 = 200
    sec = SimpleSec(rp0=rp0, nperm=200, tperm=1)

    # x = np.arange(300)
    x = np.linspace(0.001, 300, 300)

    curves = []
    for rg in [50, 41, 30]:
        y  = sec.get_elution(x, rg)
        curves.append(y)

    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))

        fig.suptitle("SEC Theory Prediction with Pore Size=%s" % rp0, fontsize=24)
        ax1.set_title("Separate Elutions", fontsize=20)
        ax2.set_title("Poly vs. Mono", fontsize=20)

        for y, name in zip(curves, ["rg50", "rg41", "rg30"]):
            ax1.plot(x, y, label=name)

        y1 = (curves[0] + curves[2])/2
        y2 = curves[1]
        ax2.plot(x, y1, label="Polydisperse (rg50, rg30)")
        ax2.plot(x, y2, label="Monodisperse (rg41)")

        for ax in [ax1, ax2]:
            ax.legend(fontsize=14)
        fig.tight_layout()
        plt.show()
