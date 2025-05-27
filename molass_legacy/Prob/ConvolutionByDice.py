# coding: utf-8
"""
    ConvolutionByDice.py

    Copyright (c) 2020,2024, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import matplotlib.animation as animation
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.KekLib.OurMatplotlib import get_color
from .DiscretePdf import DiscretePdf

def create_vertices(i, j):
    i_ = i + 0.5
    j_ = j + 0.5
    return [(i_,j_), (i_+1,j_), (i_+1,j_+1), (i_,j_+1)]

def demo(save=False):
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3,4)
    ax0 = fig.add_subplot(gs[:,0:3])
    ax1 = fig.add_subplot(gs[0,3])
    ax2 = fig.add_subplot(gs[1,3])
    ax3 = fig.add_subplot(gs[2,3])
    axes = [ax0, ax3]

    ax0.set_title("Simultaneous Probability Density of (X,Y) as Die Roll Pairs ", fontsize=16)
    ax1.set_title("Probability Density of X as Die Rolls", fontsize=16)
    ax2.set_title("Probability Density of Y as Die Rolls", fontsize=16)
    ax3.set_title("Probability Density of the Sum X + Y", fontsize=16)

    rects = []
    texts = []
    for i in range(6):
        row = []
        for j in range(6):
            vertices = create_vertices(i, j)
            rect = Polygon(vertices, alpha=0.5)
            ax0.add_patch(rect)
            t = ax0.text(i+1, j+1, "(%d,%d)" % (i+1,j+1), ha='center', va='center', fontsize=16)
            texts.append(t)
            row.append(rect)
        rects.append(row)

    vertices = create_vertices(0, 0)
    active_rect = Polygon(vertices, facecolor='red', alpha=0.5)
    ax0.add_patch(active_rect)

    ax0.set_xlim(0,7)
    ax0.set_ylim(0,7)
    ax0.set_aspect('equal')
    ax0.set_xlabel('X', fontsize=16)
    ax0.set_ylabel('Y', fontsize=16)
    ax0.grid(False)

    rects_array = np.array(rects)
    rects_list = list(rects_array.flatten())

    default_color = get_color(0)

    data_list = []
    num_frames = 0
    control_list = []
    for k in range(0,11):
        start = max(0, k-5)
        stop = min(k+1, 6)
        num_diag = stop - start
        for m in range(num_diag):
            control_list.append([m, start, stop, num_diag, k])
            data_list.append(k+2)

    control_array = np.array(control_list)
    num_frames = control_array.shape[0]
    data = np.array(data_list)

    die_rolls = np.arange(1,7)
    pdf1 = DiscretePdf(die_rolls, num_bins=6, num_frames=num_frames)
    pdf1.prepare(ax1)
    vertices = pdf1.get_nth_bar_vertices(0)
    pdf1_pnt = Polygon(vertices, facecolor='red', alpha=0.5)
    ax1.add_patch(pdf1_pnt)

    pdf2 = DiscretePdf(die_rolls, num_bins=6, num_frames=num_frames)
    pdf2.prepare(ax2)
    pdf2_pnt = Polygon(vertices, facecolor='red', alpha=0.5)
    ax2.add_patch(pdf2_pnt)

    pdf3 = DiscretePdf(data, num_bins=11, num_frames=num_frames)
    pdf3.prepare(ax3)

    for ax in [ax1, ax2]:
        ax.set_ylim(0, 0.3)

    vertices = pdf3.get_nth_bar_vertices(0)
    pdf3_bar = Polygon(vertices, facecolor='yellow')
    ax3.add_patch(pdf3_bar)
    pdf_bar_vertices = vertices
    pdf3_pnt = Polygon(pdf_bar_vertices, facecolor='red', alpha=0.5)
    ax3.add_patch(pdf3_pnt)

    ax3.set_ylim(0, 0.2)

    show_artists = [active_rect, pdf1_pnt, pdf2_pnt, pdf3_bar, pdf3_pnt]

    pdf_bar_pos = 0
    actual_n = 0
    pause = False

    def on_click(event):
        nonlocal pause
        pause ^= True

    fig.canvas.mpl_connect('button_press_event', on_click)

    def reset():
        if actual_n == 0:
            rects_array[5,5].set_facecolor(default_color)
            return rects_list + texts + show_artists
        else:
            return []

    def set_diag_color(n, color, alt_color=None):
        m, start, stop, num_diag, k = control_array[n]
        if alt_color is not None:
            if k % 2 == 1:
                color = alt_color
        for i in range(start, stop):
            j = k - i
            rects_array[i,j].set_facecolor(color)

    def move_active_rect(n):
        m, start, stop, num_diag, k = control_array[n]
        i = start + m
        j = k - i
        vertices = create_vertices(i, j)
        active_rect.set_xy(vertices)
        return i,j

    def pdf_pointer_vertices(vertices, n):
        m, num_diag = control_array[n][[0,3]]
        left_, bottom_ = vertices[0]
        right_, top_ = vertices[2]
        delta_height = (top_ - bottom_)/num_diag
        new_bottom = bottom_ + delta_height*m
        new_top = new_bottom + delta_height
        return [(left_, new_bottom), (right_, new_bottom), (right_, new_top), (left_, new_top)]

    def animate(n_):
        nonlocal pdf_bar_vertices, actual_n
        n = actual_n
        m, k = control_array[n][[0, -1]]
        if m == 0:
            if n > 0:
                set_diag_color(n-1, default_color)
            set_diag_color(n, 'cyan', alt_color='yellow')
            pdf_bar_vertices = pdf3.get_nth_bar_vertices(k)
            pdf3_bar.set_xy(pdf_bar_vertices)
            color = 'cyan' if k % 2 == 0 else 'yellow'
            pdf3_bar.set_facecolor(color)
        i, j = move_active_rect(n)
        vertices = pdf1.get_nth_bar_vertices(i)
        pdf1_pnt.set_xy(vertices)
        vertices = pdf2.get_nth_bar_vertices(j)
        pdf2_pnt.set_xy(vertices)
        vertices = pdf_pointer_vertices(pdf_bar_vertices, n)
        pdf3_pnt.set_xy(vertices)

        if not pause:
            actual_n += 1
            if actual_n == num_frames:
                actual_n = 0
                reset()

        return rects_list + texts + show_artists

    anim = animation.FuncAnimation(fig, animate, frames=num_frames, blit=True, init_func=reset, interval=500)

    fig.subplots_adjust(left=-0.05, right=0.95, bottom=0.05, top=0.95, wspace=-0.1, hspace=0.3)

    if save:
        anim.save("anim.mp4", writer="ffmpeg")

    plt.show()

def demo3d():
    pass
