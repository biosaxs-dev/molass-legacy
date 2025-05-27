# coding: utf-8
"""
    DiscretePdf.py

    revised and made into a class from the example at
    https://matplotlib.org/3.1.1/gallery/animation/animated_histogram.html

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import molass_legacy.KekLib.DebugPlot as plt

class DiscretePdf:
    def __init__(self, data, num_bins, num_frames=None, update_data=False, patch_options=None, scale=None):
        self.data = data
        self.num_bins = num_bins
        self.update_data = update_data
        if update_data:
            self.increment = None
        else:
            self.increment = len(data)//num_frames
        self.patch_options = patch_options

        hist, bin_edges_ = np.histogram(data, num_bins)
        # modify so that the xticks come to the center of each bar
        bin_edges = np.linspace(bin_edges_[0]-0.5, bin_edges_[-1]+0.5, len(bin_edges_))

        left = np.array(bin_edges[:-1])
        right = np.array(bin_edges[1:])
        bottom = np.zeros(len(left))
        if scale is None:
            scale = 1/len(data)
        top = bottom + hist*scale
        nrects = len(left)

        nverts = nrects * (1 + 3 + 1)
        verts = np.zeros((nverts, 2))
        codes = np.ones(nverts, int) * path.Path.LINETO
        codes[0::5] = path.Path.MOVETO
        codes[4::5] = path.Path.CLOSEPOLY
        verts[0::5, 0] = left
        verts[0::5, 1] = bottom
        verts[1::5, 0] = left
        verts[1::5, 1] = top
        verts[2::5, 0] = right
        verts[2::5, 1] = top
        verts[3::5, 0] = right
        verts[3::5, 1] = bottom

        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.verts = verts
        self.codes = codes

    def get_nth_bar_vertices(self, n):
        left_ = self.left[n]
        right_ = self.right[n]
        bottom_ = self.bottom[n]
        top_ = self.top[n]
        return [(left_, bottom_), (right_, bottom_), (right_, top_), (left_, top_)]

    def prepare(self, ax):

        num_patches = 1 if self.update_data else 2
        verts_list = []
        if num_patches == 2:
            verts_list.append(self.verts.copy())    # copy the initial values
        verts_list.append(self.verts)

        for i in range(num_patches):
            """
            create two patches in the non self.update_data mode.
            note that you can't use copy.deepcopy for patches
            because "TransformNode instances can not be copied."
            """
            if self.patch_options:
                kwargs = self.patch_options[i]
            else:
                kwargs = {'alpha':0.5}

            barpath = path.Path(verts_list[i], self.codes)
            patch = patches.PathPatch(
                        barpath, **kwargs)
            ax.add_patch(patch)

        self.patch = patch

        ax.set_xlim(self.left[0], self.right[-1])
        ax.set_ylim(self.bottom.min(), self.top.max())
        ax.grid(False)

    def reset(self):
        return [self.patch, ]

    def animate(self, i):
        # simulate new data coming in
        if self.update_data:
            data = np.random.randn(len(self.data))
        else:
            data = self.data[0:(i+1)*self.increment]
        hist, bin_edges = np.histogram(data, self.num_bins)
        self.top = self.bottom + hist
        self.verts[1::5, 1] = self.top
        self.verts[2::5, 1] = self.top
        return [self.patch, ]

def demo():
    data = np.random.randn(1000)
    patch_options = [{'facecolor':None}, {'facecolor':'yellow', 'alpha':0.3}]
    ah = AnimatedHistogram(data, 100, 100, patch_options=patch_options)
    # fig, ax = plt.subplots()  # does not work for DebugPlot
    fig = plt.figure()
    ax = fig.gca()
    ah.prepare(ax)
    ani = animation.FuncAnimation(fig, ah.animate, 100, repeat=True, blit=True, init_func=ah.reset, interval=100)
    fig.tight_layout()
    plt.show()
