"""
    SecTheory.EdmDemo.py

    EDM - Equilibrium Dispersive Model - Demo

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import sys
import os

# execute this file with python this-path demo
if len(sys.argv) > 1 and sys.argv[1].find("demo") >= 0:
    this_dir = os.path.dirname(os.path.abspath( __file__ ))
    sys.path.append(this_dir + '/..')

import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn
seaborn.set()

from SecTheory.Edm import Edm

def demo_2d(title, a=1.5, b=0.5, Dz=0.02):
    t = np.linspace(2, 400, 400)
    edm = Edm(z=30, L=30, a=a, b=b, u=1.2, Dz=Dz)

    y = edm(t)
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.plot(t, y)
    fig.tight_layout()
    plt.show()

def peak_tailing_demo():
    a = 5
    b = 1.0
    Dz = 0.5
    demo_2d("Peak Tailing Simulation using EDM; a=%g, b=%g, Dz=%g" % (a, b, Dz), a=a, b=b, Dz=Dz)

def peak_fronting_demo():
    a = 5
    b = -0.5
    Dz = 0.05
    demo_2d("Peak Fronting Simulation using EDM; a=%g, b=%g, Dz=%g" % (a, b, Dz), a=a, b=b, Dz=Dz)

def demo_3d(title, a=1.5, b=0.5, Dz=0.02):
    t = np.linspace(2, 400, 100)
    z = np.linspace(0.5, 30, 30)
    zz, tt = np.meshgrid(z, t)
    edm = Edm(z=zz, L=30, a=a, b=b, u=1.2, Dz=Dz)
    cc = edm(tt)

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("t")
    ax.set_zlabel("c")
    ax.plot_surface(zz, tt, cc)
    fig.tight_layout()
    plt.show()

def peak_tailing_demo_3d():
    a = 5
    b = 1.0
    Dz = 0.5
    demo_3d("Peak Tailing Simulation using EDM; a=%g, b=%g, Dz=%g" % (a, b, Dz), a=a, b=b, Dz=Dz)

def peak_fronting_demo_3d():
    a = 5
    b = -0.5
    Dz = 0.05
    demo_3d("Peak Fronting Simulation using EDM; a=%g, b=%g, Dz=%g" % (a, b, Dz), a=a, b=b, Dz=Dz)

def animation(title, a=1.5, b=0.5, Dz=0.02, save=False):
    import matplotlib.image as mpimg
    import matplotlib.animation as animation

    P = 2000
    N = 120
    M = 90
    L = 30
    t = np.linspace(2, 400, N)

    edm = Edm(z=L, L=L, a=a, b=b, u=1.2, Dz=Dz)
    cmax = np.max(edm(t))

    z = np.linspace(1, L, M)
    zz, tt = np.meshgrid(z, t)
    edm = Edm(z=zz, L=L, a=a, b=b, u=1.2, Dz=Dz)
    cc = edm(tt)

    print("cc.shape=", cc.shape)

    fig = plt.figure(figsize=(10,5))
    fig.suptitle(title, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, wspace=1.0)

    gs = GridSpec(1,4)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1:4])

    ax2.set_title("Concentration at the End of Column", fontsize=16)
    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(-0.01, cmax*1.2)

    images = []
    el_x = []
    el_y = []
    line, = ax2.plot(el_x, el_y, lw=5)

    LM_ratio = L/M

    for i in range(N):
        ax1.cla()
        ax1.set_title("Column State", fontsize=16)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(L, 0)
        ax1.set_xticks([])

        xmin, xmax = ax1.get_xlim()
        ymin, ymax = ax1.get_ylim()

        c = cc[i,:]
        """
        r = c/np.sum(c)     # this is not appropriate for very small np.sum(c)
        n = int(r*P)
        """

        ey = 0
        for k, c in enumerate(c):
            n_ = int(P*c*LM_ratio)
            if k == M - 1:
                ey += n_
            if n_ > 0:
                pos = np.random.uniform(0, 1, (n_,2))
                x_ = pos[:,0]
                y_ = (k + pos[:,1])*LM_ratio
                ax1.plot(x_, y_, "o", color="red", markersize=1)

        el_x.append(t[i])
        el_y.append(ey/P/LM_ratio)
        line.set_xdata(el_x)
        line.set_ydata(el_y)

        plt.pause(0.2)

        buffer = io.BytesIO()
        fig.savefig(buffer)
        buffer.seek(0)
        img = mpimg.imread(buffer)
        images.append(img)

    plt.show()
    plt.close()

    if not save:
        return

    print("saving the animation ...")

    fig, ax = plt.subplots(figsize=(9,5))
    ax.set_axis_off()
    fig.tight_layout()

    ims = []
    for img in images:
        print(".")
        im = ax.imshow(img, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
    ani.save("animation.gif")

    plt.show()

def peak_taling_animation(save=False):
    animation("Peak Tailing Simulation using EDM", a=5, b=1.0, Dz=0.5, save=save)

def peak_fronting_animation(save=False):
    animation("Peak Fronting Simulation using EDM", a=5, b=-0.5, Dz=0.05, save=save)

def paper_proof_1():
    t = np.linspace(0, 40, 100)
    z = 10
    L = 10
    u = 1.2
    Pe = 600
    Dz = L*u/Pe
    print("Dz=", Dz)

    fig, ax = plt.subplots()

    for b in [0.5, 0.2, 1.0, 2.0]:
        edm = Edm(z=z, b=b, Dz=Dz)
        c = edm(t)
        ax.plot(t, c, label="b=%.1f" % b)

    ax.legend()
    fig.tight_layout()
    plt.show()

def emg_vs_edm():
    from scipy.optimize import minimize

    for name in ["KekLib", "DataStructure", "SerialAnalyzer", "Decomposer"]:
        sys.path.append(this_dir + '/../' + name)

    from molass_legacy.QuickAnalysis.ModeledPeaks import recognize_peaks
    from molass_legacy.QuickAnalysis.ModeledPeaks import egh

    title = "EGH vs. EDM"
    a = 5
    b = -0.5
    Dz = 0.05
    x = np.linspace(2, 400, 400)
    edm = Edm(z=30, L=30, a=a, b=b, u=1.2, Dz=Dz)

    y = edm(x)

    fig, axes = plt.subplots(ncols=3, figsize=(18,5))
    fig.suptitle(title, fontsize=20)

    num_peaks = 1

    def objective(p):
        peaks = p.reshape(num_peaks, 4)
        ty_ = np.zeros(len(x))
        for h, m, s, t in peaks:
            y_ = egh(x, h, m, s, t)
            ty_ += y_
        return np.sum((ty_ - y)**2)

    for n, ax in enumerate(axes):
        num_peaks = n + 1
        ax.set_title("Number of components = %d" % num_peaks, fontsize=16)

        peaks = recognize_peaks(x, y, exact_num_peaks=num_peaks)

        ax.plot(x, y, color="orange", label="EDM")

        params = np.array(peaks)

        ret = minimize(objective, params.flatten())
        peaks = ret.x.reshape(num_peaks, 4)

        ty_ = np.zeros(len(x))
        for k, (h, m, s, t) in enumerate(peaks):
            y_ = egh(x, h, m, s, t)
            ty_ += y_
            ax.plot(x, y_, ":", label="EGH-%d" % (k+1))

        ax.plot(x, ty_, ":", color="red", lw=2, label="EGH total")
        ax.legend(loc="upper left")

        xmin, xmax = ax.get_xlim()
        ylim = ax.get_ylim()

        rmsd = np.sqrt(np.mean((ty_ - y)))
        tx = xmin*0.7 + xmax*0.3
        ty = np.mean(ylim)
        ax.text(tx, ty, "RMSD=%.2g" % rmsd, alpha=0.5, fontsize=20, ha="center", va="center")

    fig.tight_layout()
    plt.show()

def quadratic_isotherm():

    def q(a, b, c):
        return (a*c)/(1 + b*c)

    def q_(a, b, c, c0):
        denominator = (1 + b*c0)**3
        gam1 = (a * b**2 * c0**3) / denominator
        gam2 = (a + 3*a*b*c0) / denominator
        gam3 = (-a*b) / denominator
        return gam1 + gam2*c + gam3*c**2

    a = 1
    b = 0.5
    c0 = 0.0001
    c = np.linspace(0, 0.6, 100)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5))
    fig.suptitle("Quadratic isotherm")
    ymin, ymax = 1, 0
    for ax, a_, b_, c0_ in [(ax1, a, b, c0), (ax2, a, -b, c0)]:
        ax.set_title("a, b, c0 = (%g, %g, %g)" % (a_, b_, c0_))
        ax.set_xlabel("mobile phase concentration")
        ax.set_ylabel("adsorbed phase concentration")
        ax.plot(c, q(a_, b_, c), label="Langmuir isotherm")
        ax.plot(c, q_(a_, b_, c, c0_), label="Quadratic isotherm")
        ymin_, ymax_ = ax.get_ylim()
        ymin = min(ymin, ymin_)
        ymax = max(ymax, ymax_)
        ax.legend(loc="upper left")

    for ax in ax1, ax2:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    # peak_tailing_demo()
    # peak_tailing_demo_3d()
    # peak_fronting_demo()
    peak_fronting_demo_3d()
    # peak_taling_animation(save=True)
    # peak_fronting_animation(save=True)
    # paper_proof_1()
    # emg_vs_edm()
    # quadratic_isotherm()
