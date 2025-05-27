# coding: utf-8
"""
    SecTheory.Dondi1986.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib.gridspec import GridSpec
import molass_legacy.KekLib.DebugPlot as plt

def demo():
    lam = 40
    size = 1000
    Np = np.random.poisson(lam, size)

    if False:
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.plot(Np)
            plt.show()

    fig, ax = plt.subplots()
    scale = 1.0
    traj_list = []
    for n in Np:
        y = np.linspace(0, 1, n)
        Tm = np.random.exponential(scale, n)
        Ts = np.random.exponential(scale, n)
        tx = [0.0]
        ty = [0.0]
        t = 0
        my = 0
        for k, (tm, ts) in enumerate(zip(Tm, Ts)):
            t += tm
            my += tm
            tx.append(t)
            ty.append(my)
            if k < n-1:
                t += ts
                tx.append(t)
                ty.append(my)
        tx = np.array(tx)
        ty = np.array(ty)/my
        traj_list.append((tx, ty))

    with plt.Dp():
        fig = plt.figure(figsize=(10,10))
        fig.suptitle("Stochastic Theory Illustration with size=%d" % size, fontsize=20)

        gs = GridSpec(5,1)

        ax1 = fig.add_subplot(gs[0,:])
        ax2 = fig.add_subplot(gs[1:,:])

        ax2.set_xlabel("Time", fontsize=16)
        ax2.set_ylabel("Length (Displacement)", fontsize=16)

        tr = [tx[-1] for tx, ty in traj_list]
        ax1.hist(tr)

        for tx, ty in traj_list:
            ax2.plot(tx, ty)

        xmin, xmax = ax2.get_xlim()
        ax.set_xlim(xmin, xmax)

        fig.tight_layout()
        plt.show()
