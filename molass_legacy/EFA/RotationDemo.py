# coding: utf-8
"""
    EFA.RotationDemo.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.animation as animation
from CorrectedData import CorrectedXray
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.OurMatplotlib import get_color
from MatrixData import simple_plot_3d
from ModeledData import ModeledData
from Tutorials.MatrixFactorization import draw_PQ, generate_random_points
from Tutorials.SvdPrecision import plot_text

FONTSIZE = 200
A_SIZE = 600

def model_demo(bogus_init=False, random_init=False, symmetric=True, save_anim=False):
    fig = plt.figure(figsize=(21,11))

    proof_type = 'Proof' if symmetric else 'Weakness Proof'
    fig.suptitle("Iterative Rotation Method %s" % proof_type, fontsize=40)

    pause = False

    def on_click(event):
        nonlocal pause
        print('on_click')
        if event.inaxes is None:
            return
        pause ^= True

    plt.dp.mpl_canvas.mpl_connect('button_press_event', on_click)

    q = np.linspace(0.01, 0.6, A_SIZE)
    rg_list = [50, 24]
    d_list = [1, 3]
    if symmetric:
        E_SIZE = 300
        md = ModeledData(q, E_SIZE, rg_list=rg_list, d_list=d_list)
        f1, t1 = 0, 200
        f2, t2 = 100, E_SIZE
        num_frames = 20
        interval = 1000
    else:
        E_SIZE = 500
        md = ModeledData(q, E_SIZE, rg_list=rg_list, d_list=d_list,
                            h_list=[1, 0.5], mu_list=[130, 220], sigma_list=[20, 30], tau_list=[40, 20])
        f1, t1 = 0, 350
        f2, t2 = 120, E_SIZE
        num_frames = 1000
        interval = 50

    if False:
        dplt.push()
        md.plot_components()
        dplt.pop()
    M = md.get_data()

    rank = len(rg_list)

    U, s, VT = np.linalg.svd(M)

    Us_ = np.dot( U[:,0:rank], np.diag( s[0:rank] ) )
    M_  = np.dot( Us_, VT[0:rank,:] )

    gs = GridSpec(2,4)
    axes = []
    for i in range(2):
        arow = []
        for k in range(4):
            proj = '3d' if i==0 and k == 0 else None
            arow.append(fig.add_subplot(gs[i,k], projection=proj))
        axes.append(arow)
    ax1, ax2, ax3, ax4  = axes[0]
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    for i in range(rank):
        ax2.plot(q, U[:,i])
        ax4.plot(VT[i,:])

    simple_plot_3d(ax1, M_, x=q)
    ns = 5
    ax3.plot(s[0:ns], ':', color='gray')
    for k in range(ns):
        ax3.plot(k, s[k], 'o', color=get_color(k))

    plot_text(ax2, "$U$", fontsize=FONTSIZE)
    plot_text(ax3, "$\Sigma$", fontsize=FONTSIZE)
    plot_text(ax4, "$V^T$", fontsize=FONTSIZE)

    ax5, ax6, ax7, ax8  = axes[1]

    n = md.num_components
    for k in range(n):
        ax5.plot(md.i, md.P[:,k])
        ax6.plot(md.C[k,:])

    poly1 = Polygon([(f1,0), (f1,1), (t1,1), (t1,0)], color='cyan', alpha=0.2)
    ax6.add_patch(poly1)
    poly2 = Polygon([(f2,0), (f2,1), (t2,1), (t2,0)], color='pink', alpha=0.2)
    ax6.add_patch(poly2)

    def make_C():
        if bogus_init:
            if random_init:
                c1 = np.random.uniform(0,1, E_SIZE)
                c2 = 1 - c1
            else:
                c1 = np.zeros(E_SIZE)
                c1[f1:f2] = 1
                c1[f2:t1] = 0.5
                c2 = np.zeros(E_SIZE)
                c2[f2:t1] = 0.5
                c2[t1:t2] = 1
            C = np.vstack([c1, c2])
        else:
            C = VT[0:rank,:].copy()
        return C

    C = make_C()
    R = np.zeros(C.shape)
    R[0,f1:t1] = 1
    R[1,f2:t2] = 1

    if bogus_init:
        ax4.plot(C[0,:], ':')
        ax4.plot(C[1,:], ':')

    x = np.arange(E_SIZE)

    def draw_C_lines(C):
        line_c1, = ax8.plot(x, C[0,:])
        line_c2, = ax8.plot(x, C[1,:])
        return line_c1, line_c2

    line_c1, line_c2 = draw_C_lines(C)

    P = np.dot(M_, np.linalg.pinv(C))

    def draw_P_lines(P):
        line_p1, = ax7.plot(q, P[:,0])
        line_p2, = ax7.plot(q, P[:,1])
        return line_p1, line_p2

    line_p1, line_p2 = draw_P_lines(P)

    def clear_axes():
        for ax in [ax7, ax8]:
            ax.cla()

    def draw_texts(text):
        ret = []
        for ax in [ax7, ax8]:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            t = ax.text(tx, ty, text, ha="center", va="center", alpha=0.1, fontsize=200)
            ret.append(t)
        return ret

    text7, text8 = draw_texts("0")

    blit = False        # it is difficult to update axis labels with blit=True

    def update_text(text):
        for ax, t in [(ax7, text7), (ax8, text8)]:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            t.set_position((tx, ty))
            t.set_text(text)

    artists = [line_c1, line_c2, line_p1, line_p2, text7, text8]

    def update_C_lines(C):
        line_c1.set_data(x,C[0,:])
        line_c2.set_data(x,C[1,:])

    def update_P_lines(P):
        line_p1.set_data(q,P[:,0])
        line_p2.set_data(q,P[:,1])

    def reset():
        nonlocal P, C
        C = make_C()
        P = np.dot(M_, np.linalg.pinv(C))

        if blit:
            update_C_lines(C)
            update_P_lines(P)
            update_text("0")
            return artists
        else:
            clear_axes
            draw_C_lines(C)
            draw_P_lines(P)
            draw_texts("0")

    def normalize_C(C):
        C = R*C
        for k in range(C.shape[0]):
            C[k,:] /= np.sum(C[k,:])
        return C

    def animate(i):
        nonlocal P, C
        print([i])
        if i % 2 == 0:
            P = np.dot(M_, np.linalg.pinv(C))
            C = np.dot(np.linalg.pinv(P), M_)
        else:
            C = normalize_C(C)

        num_text = str(i//2)

        if blit:
            update_C_lines(C)
            update_P_lines(P)
            ax7.set_ylim(P.min(), P.max())
            ax8.set_ylim(C.min(), C.max())
            update_text(num_text)
            return artists
        else:
            clear_axes()
            draw_C_lines(C)
            draw_P_lines(P)
            draw_texts(num_text)

    def index_generator():
        i = 0
        while i < num_frames:
            i_ = i
            if pause:
                pass
            else:
                i += 1
            yield i_

    save_count = 1500 if save_anim else None
    anim = animation.FuncAnimation(fig, animate, index_generator, blit=blit, init_func=reset, interval=interval
                                    , save_count=save_count)
    if save_anim:
        anim.save("anim.mp4", writer="ffmpeg")

    plt.show()
