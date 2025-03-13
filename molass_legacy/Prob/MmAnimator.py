# coding: utf-8
"""
    MmAnimator.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .GammaVisualizer import GammaVisualizer

class MmAnimator:
    def __init__(self, fig, axes, ylims, xy_list, mm_list, datasets,
                    step_mode=False, show_extrapolated=False):
        self.fig = fig
        self.axes = axes[1,:]
        self.axes_anim = axes[2,:]
        self.ylims = ylims
        self.xy_list = xy_list
        self.mm_list = mm_list
        self.datasets = datasets
        self.step_mode = step_mode
        self.show_extrapolated = show_extrapolated
        self.prepare()

        self.num_frames = mm_list[0].max_iter + 1
        if step_mode:
            self.current_step = 0
        else:
            self.anim = animation.FuncAnimation(fig, self.animate, frames=self.num_frames, blit=True, init_func=self.reset, interval=100)

    def prepare(self):
        def add_number_text(ax):
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            tx = (xmin + xmax)/2
            ty = (ymin + ymax)/2
            text = ax.text(tx, ty, "_", ha='center', va='center', alpha=0.05, fontsize=180)
            return text

        artist_list = []
        for ax, ylim, xy, mm in zip(self.axes, self.ylims, self.xy_list, self.mm_list):
            ax.set_ylim(ylim)
            x, y = xy
            ax.plot(x, y, label='input data')
            text = add_number_text(ax)
            ys = mm.get_anim_components(x, y, 0)
            line, = ax.plot(x, ys[0], label='eghmm-fit')
            artist_rec = [text, line] + [ax.plot(x, y_, ':', label='component-%d' % k)[0] for k, y_ in enumerate(ys[1:])]
            artist_list.append( artist_rec )
            ax.legend()

        if self.show_extrapolated:
            curves_list = []
            for ano, (mm, xy, ax) in enumerate(zip(self.mm_list, self.xy_list, self.axes_anim)):
                C = mm.get_anim_C(*xy, -1)
                lines = self.datasets.draw_exprapolated(ano, ax, C)[0]
                curves_list.append(lines)
            self.curves_matrix = np.array(curves_list)
        else:
            self.gv_list = []
            for ax, mm in zip(self.axes_anim, self.mm_list):
                gv = GammaVisualizer()
                poly, verts = gv.draw_gamma_3d(ax, mm.gamma_array[1,:,:])
                self.gv_list.append(gv)

        self.artist_matrix = np.array(artist_list)

        # self.artists = np.hstack([self.artist_matrix.flatten(), np.array(self.poly_list)])
        self.artists = self.artist_matrix.flatten()
        if self.show_extrapolated:
            self.artists_ex = np.hstack([self.artists, self.curves_matrix.flatten()])

        if self.step_mode:
            self.visual = True
            self.current_step = 0
            self.animate(0)
        else:
            artists_ = self.artists_ex if self.show_extrapolated else self.artists
            for art in artists_:
                art.set_visible(False)
            self.visual = False

    def reset(self):
        if not self.visual:
            artists_ = self.artists_ex if self.show_extrapolated else self.artists
            for art in artists_:
                art.set_visible(True)
            self.visual = True
        return self.animate(0)

    def animate(self, n):
        for i, (ax, xy, mm) in enumerate(zip(self.axes, self.xy_list, self.mm_list)):
            x, y = xy
            ys = mm.get_anim_components(x, y, n)
            artists = self.artist_matrix[i,:]
            artists[0].set_text(str(n))
            for art, y_ in zip(artists[1:], ys):
                art.set_data(x, y_)
            if self.show_extrapolated:
                ax2d = self.axes_anim[i]
                lines = self.curves_matrix[i,:]
                C = mm.get_anim_C(*xy, n)
                self.datasets.update_exprapolated(i, lines, C)
            else:
                ax3d = self.axes_anim[i]
                ax3d.cla()
                gv = self.gv_list[i]
                gamma = mm.gamma_array[n,:,:]
                gv.draw_gamma_3d(ax3d, gamma)

        if self.show_extrapolated:
            return self.artists_ex
        else:
            return self.artists


    """
    for the following cf.
    stop / start / pause in python matplotlib animation
    https://stackoverflow.com/questions/16732379/stop-start-pause-in-python-matplotlib-animation
    """
    def stop(self):
        self.anim.event_source.stop()

    def pause(self):
        self.anim.event_source.stop()

    def start(self):
        self.anim.event_source.start()

    def step(self, steps):
        self.current_step = i = (self.current_step + steps) % self.num_frames
        self.animate(i)

    def save(self, file, **kwargs):
        self.anim.save(file, **kwargs)
