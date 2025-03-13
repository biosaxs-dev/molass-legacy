"""
    HdcTheory.ElutionCurve.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

NUM_PLATES_PC = 14400
C = 2.698   # 

def retention_time(tp, reff, r0, C):
    lamb = reff/r0
    return tp / (1 + 2*lamb - C*lamb**2)

def compute_elution_curve(x, t0, reff, r0, sigma, Npc=NUM_PLATES_PC, C=C):
    from molass_legacy.Peaks.ElutionModels import gaussian
    tp = np.sqrt(Npc)*sigma
    tI = t0 - tp
    tR = tI + retention_time(tp, reff, r0, C)
    print("tR=", tR)
    return gaussian(x, 1, tR, sigma)

def demo():
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme()
    x = np.arange(500)
    t0 = 200
    sigma = 10
    r0 = 5000
    tp = np.sqrt(NUM_PLATES_PC)*sigma
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    for rg in [200, 150, 75, 20]:
        tR = retention_time(tp, rg, r0, C)
        ax1.plot(tR, rg, "o", label=r'$R_g$=%g' % rg)
        y = compute_elution_curve(x, t0, rg, r0, 10)
        ax2.plot(x, y, label=r'$R_g$=%g' % rg)
    ax2.axvline(x=t0, color='red', label=r"$t_0$")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    # import seaborn as sns
    # sns.set_theme()
    sys.path.append("../lib")
    demo()
