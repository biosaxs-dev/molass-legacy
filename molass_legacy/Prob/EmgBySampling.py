# coding: utf-8
"""
    EmgBySampling.py

    Copyright (c) 2020,2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as mplt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from .AnimatedHistogram import AnimatedHistogram
import molass_legacy.KekLib.DebugPlot as plt

FONTSIZE = 16

class EmgBySampling:
    def __init__(self, num_bins=100, num_frames=200):
        self.num_bins = num_bins
        self.num_frames = num_frames

    def play(self, save_only=False):
        fig = plt.figure(figsize=(18, 9))
        gs = GridSpec(3,17)
        axes = []
        for i in range(3):
            ax1 = fig.add_subplot(gs[i,0:10])
            ax2 = fig.add_subplot(gs[i,10])
            ax3 = fig.add_subplot(gs[i,11:17])
            axes.append([ax1, ax2, ax3])
        self.axes = axes = np.array(axes)

        N = 10000
        self.cycle = N//self.num_frames
        X = np.random.normal(0, 1, N)
        Y = np.random.exponential(1, N)
        Z = X + Y
        rect1, bar1, _, ah1 = self.plot_data(*axes[0,:], X, "X: Gaussian")
        rect2, bar2, _, ah2 = self.plot_data(*axes[1,:], Y, "Y: Exponential", bar_color="pink")
        rect3, bar3, bar3_, ah3 = self.plot_data(*axes[2,:], Z, "X+Y: EMG", y=Y)
        rects = [rect1, rect2, rect3]
        bars = [bar1, bar2, bar3, bar3_]
        a_hists = [ah1, ah2, ah3]
        hist_patches = [ah.reset()[0] for ah in a_hists]

        plt.dp.mpl_canvas.draw()

        u_ymin, u_ymax = 999, -999
        for ax1 in axes[:,0]:
            ymin, ymax = ax1.get_ylim()
            u_ymin = min(u_ymin, ymin)
            u_ymax = max(u_ymax, ymax)
        for ax_ in axes[:,0:2]:
            for ax in ax_:
                ax.set_ylim(u_ymin, u_ymax)

        u_xmin, u_xmax = 999, -999
        for ax3 in axes[:,2]:
            xmin, xmax = ax3.get_xlim()
            u_xmin = min(u_xmin, xmin)
            u_xmax = max(u_xmax, xmax)
        for ax3 in axes[:,2]:
            ax3.set_xlim(u_xmin, u_xmax)

        fig.tight_layout()

        def reset():
            return rects + bars + hist_patches

        def animate(i):
            i_ = i * self.cycle
            for rect, x in zip(rects, [X, Y, Z]):
                new_vertices = [(0,u_ymin), (i_, u_ymin), (i_, u_ymax), (0, u_ymax)]
                rect.set_xy(new_vertices)

            for bar, x in zip(bars, [X, Y, Y]):
                bar.set_data([0.5, 0.5], [0, x[i_]])

            bar3_.set_data([0.5, 0.5], [Y[i_], Z[i_]])

            for ah in a_hists:
                ah.animate(i)

            return rects + bars + hist_patches

        self.anim = animation.FuncAnimation(fig, animate, frames=self.num_frames, blit=True, init_func=reset, interval=100)

        if save_only:
            pass
        else:
            plt.show()

    def plot_data(self, ax1, ax2, ax3, x, dtype_title, y=None, bar_color=None):
        N = len(x)
        title_with_num = dtype_title.replace(':', ': ' + str(N) )
        ax1.set_title(title_with_num + " Random Values", fontsize=FONTSIZE)
        ax2.set_title("Bar", fontsize=FONTSIZE)
        ax3.set_title(dtype_title + " Histogram", fontsize=FONTSIZE)
        ax1.plot(x, 'o', markersize=1)
        m = np.mean(x)
        s = np.std(x)
        px = [0, N-1]
        line_stype = '-'
        ax1.plot(px, [m, m], line_stype, color='green', linewidth=3, label='mean')
        ax1.plot(px, [m-s, m-s], line_stype, color='orange', linewidth=3, label='mean-stdev')
        ax1.plot(px, [m+s, m+s], line_stype, color='orange', linewidth=3, label='mean+stdev')
        ax1.plot(px, [0, 0], ':', color='red', linewidth=3, label='zero')
        ax1.legend(loc='upper right')

        vertices = [(0,0), (1,0), (1,1), (0,1)]
        rect = Polygon(vertices, color='yellow', alpha=0.2)
        ax1.add_patch(rect)
        bar, = ax2.plot([0.5, 0.5], [0, 0], solid_capstyle='butt', linewidth=10, color=bar_color)
        if y is None:
            bar_ = None
        else:
            bar_, = ax2.plot([0.5, 0.5], [0, 0], solid_capstyle='butt', linewidth=10, color='pink')
        ax2.plot([0, 1], [0, 0], ':', color='red', linewidth=3)
        ax2.set_xlim(0, 1)
        ax2.get_xaxis().set_visible(False)

        patch_options = [{'facecolor':None}, {'facecolor':'yellow', 'alpha':0.3}]
        ah = AnimatedHistogram(x, self.num_bins, self.num_frames, patch_options=patch_options)
        ah.prepare(ax3)
        ymin, ymax = ax3.get_ylim()
        ax3.set_ylim(ymin, ymax)
        ax3.plot([m,m], [ymin, ymax], line_stype, color='green', linewidth=3, label='mean')
        ax3.plot([m-s,m-s], [ymin, ymax], line_stype, color='orange', linewidth=3, label='mean-stdev')
        ax3.plot([m+s,m+s], [ymin, ymax], line_stype, color='orange', linewidth=3, label='mean+stdev')
        ax3.plot([0,0], [ymin, ymax], ':', color='red', linewidth=3, label='zero')
        ax3.legend(loc='upper right')

        return rect, bar, bar_, ah

    def save(self, file, **kwargs):
        self.anim.save(file, **kwargs)

def moments_proof():
    N = 1000000

    m1_list = []
    m2_list = []
    m3_list = []
    x = np.arange(0.1, 3, 0.1)
    for tau in x:
        print('tau=', tau)
        X = np.random.normal(0, 1, N)
        Y = np.random.exponential(tau, N)
        Z = X + Y

        M1 = np.mean(Z)
        print('M1=', M1, tau)
        m1_list.append((M1, tau))
        M2 = np.sum((Z - M1)**2)/N
        print('M2=', M2, 1+tau**2)
        m2_list.append((M2, 1+tau**2))

        M3 = np.sum((Z - M1)**3)/N
        print('M3=', M3, 2*tau**3)
        m3_list.append((M3, 2*tau**3))

    m1_array = np.array(m1_list)
    m2_array = np.array(m2_list)
    m3_array = np.array(m3_list)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_title("Proof of EMG Moment Formulas with %d Sampling" % N, fontsize=16)
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel('Moments')
    ax.plot(x, m1_array[:,0], color='red', label=r'$M_1$ (raw)')
    ax.plot(x, m1_array[:,1], ':', color='red', label=r'$\mu + \tau$')
    ax.plot(x, m2_array[:,0], color='green', label=r'$M_2$ (central)')
    ax.plot(x, m2_array[:,1], ':', color='green', label=r'$\mu^2 + \tau^2$')
    ax.plot(x, m3_array[:,0], color='blue', label=r'$M_3$ (central)')
    ax.plot(x, m3_array[:,1], ':', color='blue', label=r'$2\tau^3$')
    ax.legend()

    fig.tight_layout()
    plt.show()
