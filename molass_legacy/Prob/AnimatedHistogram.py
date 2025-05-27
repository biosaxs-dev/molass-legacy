# coding: utf-8
"""
    AnimatedHistogram.py

    revised and made into a class from the example at
    https://matplotlib.org/3.1.1/gallery/animation/animated_histogram.html

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation
import molass_legacy.KekLib.DebugPlot as plt

class AnimatedHistogram:
    def __init__(self, data, num_bins, num_frames, update_data=False, patch_options=None):
        self.data = data
        self.num_bins = num_bins
        self.update_data = update_data
        if update_data:
            self.increment = None
        else:
            self.increment = len(data)//num_frames
        self.patch_options = patch_options

        hist, bin_edges = np.histogram(data, num_bins)

        left = np.array(bin_edges[:-1])
        right = np.array(bin_edges[1:])
        bottom = np.zeros(len(left))
        top = bottom + hist
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
                kwargs = {'facecolor':'green', 'edgecolor':'yellow', 'alpha':0.5}

            barpath = path.Path(verts_list[i], self.codes)
            patch = patches.PathPatch(
                        barpath, **kwargs)
            ax.add_patch(patch)

        self.patch = patch

        ax.set_xlim(self.left[0], self.right[-1])
        ax.set_ylim(self.bottom.min(), self.top.max())

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

def demo_():
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

def demo():
    data = np.random.randn(1000)
    patch_options = [{'facecolor':None}, {'facecolor':'yellow', 'alpha':0.3}]
    # fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,6))
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    axes = [ax1, ax2]

    a_hists = []
    for ax in axes:
        ah = AnimatedHistogram(data, 100, 100, patch_options=patch_options)
        ah.prepare(ax)
        a_hists.append(ah)

    hist_patches = [ah.reset()[0] for ah in a_hists]

    def reset():
        return hist_patches

    def animate(i):
        for ah in a_hists:
            ah.animate(i)
        return hist_patches

    ani = animation.FuncAnimation(fig, animate, 100, repeat=True, blit=True, init_func=reset, interval=100)
    fig.tight_layout()
    plt.show()
