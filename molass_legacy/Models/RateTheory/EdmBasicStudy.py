"""
    Models.RateTheory.EdmBasicStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

def study():
    import seaborn as sns
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
   
    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Varying e: Total Porosity", fontsize=16)
        for e in [0.5, 0.4, 0.3, 0.2, 0.1]:
            y = edm_impl(x, t0, u, a, b, e, Dz, cinj)
            ax.plot(x, y, label="e=%g" % e)
        ax.legend()
        fig.tight_layout()
        plt.show()

    e = 0.2

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Varying cinj: Concentrationn at Injection", fontsize=16)
        for cinj in [1, 2, 3, 4, 5]:
            y = edm_impl(x, t0, u, a, b, e, Dz, cinj)
            ax.plot(x, y, label="cinj=%g" % cinj)
        ax.legend()
        fig.tight_layout()
        plt.show()
        
    cinj = 1

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Varying Dz: Dispersion Coefficient", fontsize=16)
        for Dz in [0.2, 0.3, 0.5, 0.7, 0.9]:
            y = edm_impl(x, t0, u, a, b, e, Dz, cinj)
            ax.plot(x, y, label="Dz=%g" % Dz)
        ax.legend()
        fig.tight_layout()
        plt.show()

    cinj = 1
    Dz = 0.5

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title("Varying b: Isotherm Parameter", fontsize=16)
        for b in [-1, -0.5, 0.001, 0.5, 1]:
            y = edm_impl(x, t0, u, a, b, e, Dz, cinj)
            ax.plot(x, y, label="b=%g" % b)
        ax.legend()
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    import sys
    sys.path.append("..\\lib")
    study()