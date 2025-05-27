# coding: utf-8
"""
    MinimumSurface.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from XrayData import XrayData
from MicrofluidicElution import get_mtd_elution
from SimpleUnfolding import proportion_folded
from SvdDenoise import get_denoised_data
import molass_legacy.KekLib.DebugPlot as plt

def compute_min_norm(G, m, x, M):
    pf = proportion_folded(G, m, x)
    pu = 1 - pf
    C = np.array([pf, x*pf, pu, x*pu])
    Cpinv = np.linalg.pinv(C)
    P = M@Cpinv
    z = np.linalg.norm(P@C - M)
    return z

def plot_surface(title, ax, x, GG, mm, M):
    zz = np.zeros(GG.shape)
    for i in range(GG.shape[0]):
        for j in range(GG.shape[1]):
            G_ = GG[i,j]
            m_ = mm[i,j]
            zz[i,j] = compute_min_norm(G_, m_, x, M)

    # demo_xy = [(1.74, 0.052), (7.89, 0.168), (54.2, 3.29)]
    # demo_xy = [(1.53e-09, 0.0519), (7.59e-10, 0.0524), (5.16, 0.107)]
    demo_xy = [     (0.,         0.05796411),
                    (0.99928439, 0.05584393),
                    (8.35120496, 0.09850686),
                    (8.72237253, 0.09875249),
                    (7.05725645, 0.10006241),
                    (5.95324888, 0.11418208),
                    (5.99807986, 0.1143187),
                    (6.99719508, 0.1000397),
                    (7.99302308, 0.18208686),
                    (9.00542046, 0.09865848),
                    ]
    demo_points = []
    for x_, y_ in demo_xy:
        z = compute_min_norm(x_, y_, x, M)
        demo_points.append((x_, y_, z))

    ax.set_title(title, fontsize=16)
    ax.set_xlabel('G')
    ax.set_ylabel('m')
    ax.set_zlabel('min norm')
    ax.plot_surface(GG, mm, zz, alpha=0.3)

    for x_, y_, z_ in demo_points:
        ax.plot([x_], [y_], [z_], 'o', markersize=10, label='(G,m)=(%.3g,%.3g)' % (x_,y_))

    ax.legend()

class MinimumSurface:
    def __init__(self, in_folder, gmax=12, mmax=0.3, auto=False, allq_only=True, save_no=None):
        xdata = XrayData(in_folder)
        num_files = xdata.data.shape[1]
        mtd_elution = get_mtd_elution(in_folder, num_files)
        slice_ = mtd_elution.guess_range(num_files)
        D = xdata.data[:,slice_]
        M = get_denoised_data(D, rank=4)

        """
        P = M@Cpinv
        """
        x = np.arange(M.shape[1])

        G = np.hstack([np.linspace(0.001, 0.09, 9), np.linspace(0.1, gmax, 50)])
        m = np.linspace(0.01, mmax, 50)

        GG, mm = np.meshgrid(G, m)
        print(GG.shape, mm.shape)

        if allq_only:
            fig = plt.figure(figsize=(15,9))
            ax2 = fig.add_subplot(111, projection='3d')
        else:
            fig = plt.figure(figsize=(18,8))
            ax1 = fig.add_subplot(121, projection='3d')
            ax2 = fig.add_subplot(122, projection='3d')

        if not allq_only:
            i = xdata.e_index
            M_ = M[i,:].reshape((1,M.shape[1]))
            title = r'$ min \parallel P \cdot C_{G,m} - \tilde{M} \parallel $ Surface at Q=0.02'
            plot_surface(title, ax1, x, GG, mm, M_)
        title = r'$ min \parallel P \cdot C_{G,m} - \tilde{M} \parallel $ Surface for all Q'
        plot_surface(title, ax2, x, GG, mm, M)

        fig.tight_layout()
        plt.show(block=not auto)
        if auto:
            from time import sleep
            sleep(0.5)
        if save_no is not None:
            import os
            image_folder = r'.\temp\images'
            if not os.path.exists(image_folder) or save_no == 0:
                from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry
                clear_dirs_with_retry([image_folder])
            path = image_folder + r'\image%03d.png' % save_no
            print([save_no], 'saving...')
            fig.savefig( path )

def make_it_into_a_movie(in_folder, G_range=(100, 12), m_range=(50, 0.3), num_frames=200):
    from subprocess import run
    gf, gt = G_range
    mf, mt = m_range

    gs = np.logspace(np.log10(gt), np.log10(gf), num_frames)
    ms = np.logspace(np.log10(mt), np.log10(mf), num_frames)

    k = 0
    for g, m in zip(reversed(gs), reversed(ms)):
        surface = MinimumSurface(in_folder, gmax=g, mmax=m, auto=True, allq_only=True, save_no=k)
        k += 1

    run(['ffmpeg', '-r', '8', '-i', r'temp\images\image%03d.png', '-vcodec', 'mpeg4', '-y', 'movie.mp4'])
