"""
    mplclofa.py

    inspired by https://github.com/jwkvam/celluloid

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import copy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


class MplClone:
    def __init__(self, camera, obj):
        self.camera = camera
        self.obj = obj
        self.legend_obj = None
        self.legend_labels = []

    def set_title(self, title):
        # using text method instead of set_title, since the latter seems not ok for animation
        artist = self.obj.text(0.5, 1.05, title, transform=self.obj.transAxes, ha="center")
        self._append(artist)

    def plot(self, *args, **kwargs):
        label = kwargs.get("label", None)
        self.legend_labels.append(label)
        artist, = self.obj.plot(*args, **kwargs)
        self._append(artist)

    def legend(self):
        if self.legend_obj is None:
            self.legend_obj = self.obj.legend()

    def _append(self, artist):
        self.camera.frame_artists.append(artist)

    def _update_legend(self):
        if self.legend_obj is None:
            return

        texts = self.legend_obj.get_texts()
        for text, label in zip(texts, self.legend_labels):
            print("label=", label)
            text.set_text(label)
            self._append(text)

        self.legend_labels = []

class Camera:
    def __init__(self, fig):
        self.fig = fig
        self.photos = []
        self.frame_artists = []
        self.legends = []

    def clone(self, *args):
        ret_objects = []
        for obj in args:
            ret_objects.append(MplClone(self, obj))
        self.cloned = ret_objects
        return ret_objects

    def snap(self):
        for clone in self.cloned:
            clone._update_legend()
        self.photos.append(self.frame_artists)
        self.frame_artists = []
        self.legends = []

    def animate(self, **kwargs):
        anim = animation.ArtistAnimation(self.fig, self.photos, **kwargs)
        return anim

def demo():

    fig, axes_ = plt.subplots(nrows=2)
    fig.subplots_adjust(hspace=0.5)

    camera = Camera(fig)

    ax1, ax2 = camera.clone(*axes_)

    fig.suptitle("mplclofa demo")

    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    for i in t:

        ax1.set_title("ax1: %.3g" % i)
        ax2.set_title("ax2: %.3g" % i)

        ax1.plot(t, np.sin(t + i), color='blue', label="np.sin(t + %.3g)" % i)
        ax2.plot(t, np.sin(t - i), color='blue', label="np.sin(t - %.3g)" % i)

        ax1.legend()    # not yet updating labels
        ax2.legend()    # not yet updating labels

        camera.snap()

    anim = camera.animate()
    plt.show()

if __name__ == '__main__':
    demo()
