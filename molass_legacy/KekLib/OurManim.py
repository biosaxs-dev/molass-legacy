"""
    OurManim.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import animation
from matplotlib.patches import Polygon, Circle
from molass_legacy.KekLib.OurMatplotlib import mpl_1_5_backward_compatible_init

def latex_init():
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def manim_init():
    plt.close()
    plt.style.use('dark_background')
    latex_init()

def use_default_style():
    plt.close()
    plt.style.use('default')
    mpl_1_5_backward_compatible_init()
    latex_init()

def rotation(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

def angle(v0, v1):
    """
    Angles between two n-dimensional vectors in Python
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
    """
    return np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))

def rotate(theta, vertices):
    R = rotation(theta)
    return list(np.dot(R, np.array(vertices).T).T)

class Animation:
    def __init__(self, start, stop, on_stop=None):
        self.start = start
        self.stop = stop
        self.on_stop = on_stop
        self.count = stop - start
        self.slice_ = slice(start, stop)
        self.anim_objects = []
        self.artists = []

    def close(self):
        print('close: ', self.stop)
        if self.on_stop is not None:
            print('on_stop: ', self.stop)
            self.on_stop()
        self.set_visible(False)
        # return self.artists + self.close_artists
        return self.artists

    def append(self, anim_object):
        anim_object.divide(self.count)
        self.anim_objects.append(anim_object)
        self.artists += anim_object.get_artists()

    def get_artists(self):
        return self.artists

    def animate(self, i):
        assert i >= self.start
        # print([i], "- animate")
        for aobj in self.anim_objects:
            if i < self.stop:
                aobj.animate(i - self.start)
            else:
                if i == self.stop:
                    self.close()
                break
        return self.artists

    def set_visible(self, torf):
        for a in self.artists:
            a.set_visible(torf)

class Collection:
    def __init__(self, num_frames, ret_artists=[], on_reset=None):
        self.num_frames = num_frames
        self.ret_artists = ret_artists
        self.on_reset = on_reset
        self.anim_dict = {}
        """
        indeces = {
                    0 : [a0, a1],
                    1 : [a0, a1],
                    2 : [a0,    a2, ...],
                    3 : [       a2, ...],
                    ...
                  }
        """
        self.last_anims = None

    def append(self, anim_object):
        for i in range(anim_object.slice_.start, anim_object.slice_.stop):
            anims = self.anim_dict.get(i)
            if anims is None:
                self.anim_dict[i] = anims = []
            anims.append(anim_object)
        anim_object.set_visible(False)

    def make_animation(self, fig):
        anim = animation.FuncAnimation(fig, self.animate, frames=self.num_frames, blit=True, init_func=self.reset)
        return anim

    def reset(self):
        print('reset ------------------------------')

        if self.on_reset is not None:
            self.on_reset()

        hidden_artists = []
        if self.last_anims is not None:
            for anim in self.last_anims:
                anim.set_visible(False)
                hidden_artists += anim.get_artists()

        init_artists = []
        init_anims = self.anim_dict.get(0)
        if init_anims is not None:
            for anim in init_anims:
                anim.set_visible(True)
                init_artists += anim.get_artists()

        return self.ret_artists + hidden_artists + init_artists

    def animate(self, i):
        # sleep(1)
        # print([i], "animate")
        anims = self.anim_dict.get(i)
        if anims is None:
            return self.ret_artists

        active_artists = []
        for anim in anims:
            anim.set_visible(True)
            active_artists += anim.animate(i)

        close_artists = []
        if self.last_anims is not None:
            for anim in self.last_anims:
                if i == anim.slice_.stop:
                    close_artists += anim.close()

        self.last_anims = anims
        return self.ret_artists + active_artists + close_artists

class TextGroup:
    def __init__(self, ax, texts, positions, **kwargs):
        self.ax = ax
        self.texts = texts
        self.init_positions = np.array(positions)
        self.artists = []
        for t, p in zip(texts, positions):
            self.artists.append(ax.text(*p, t, **kwargs))

    def get_artists(self):
        return self.artists

    def set_target_positions(self, positions):
        self.target_positions = np.array(positions)

    def divide(self, count):
        self.deltas = (self.target_positions - self.init_positions)/count

    def set_visible(self, torf):
        for a in self.artists:
            a.set_visible(torf)

    def reset_positions(self):
        for k, a in enumerate(self.artists):
            a.set_position(self.init_positions[k,:])

    def animate(self, i):
        # print([i], "-- animate")
        new_positions = self.init_positions + (i+1)*self.deltas
        for k, a in enumerate(self.artists ):
            a.set_position(new_positions[k,:])

class Parallelogram:
    def __init__(self, ax, vertices, motion='linear', **kwargs):
        self.init_vertices = vertices
        self.polygon = Polygon(vertices, **kwargs)
        self.motion = motion
        ax.add_patch(self.polygon)

    def set_target_vertices(self, vertices):
        self.target_vertices = np.array(vertices)

    def divide(self, count):
        if self.motion == 'rotation':
            v0 = self.init_vertices[1]
            v1 = self.target_vertices[1]
            a = angle(v0, v1)
            print('angle=', np.degrees(a))
            self.theta = a/count
        else:
            self.deltas = (self.target_vertices - self.init_vertices)/count

    def get_artists(self):
        return [self.polygon]

    def animate(self, i):
        if self.motion == 'rotation':
            theta = (i+1)*self.theta
            new_vertices = rotate(theta, self.init_vertices)
        else:
            new_vertices = self.init_vertices + (i+1)*self.deltas
        self.polygon.set_xy(new_vertices)

    def make_reversed(self, ax, **kwargs):
        para = Parallelogram(ax, self.target_vertices, motion=self.motion, **kwargs)
        para.set_target_vertices(self.init_vertices)
        return para

class Arrow:
    """
    this class is incomplete,
    because the head shape won't be consistent with the move direction.
    """
    def __init__(self, ax, pos, dpos, **kwargs):
        self.init_position = (pos, dpos)
        self.arrow = ax.arrow(*pos, *dpos, **kwargs)
        ax.add_patch(self.arrow)
        self.ax = ax
        self.init_vertices = self.arrow.get_xy()
        self.init_kwartgs = kwargs

    def set_target_position(self, pos, dpos):
        self.target_position = (pos, dpos)
        temp_arrow = self.ax.arrow(*pos, *dpos, visible=False, **self.init_kwartgs)
        self.target_vertices = temp_arrow.get_xy()

    def divide(self, count):
        self.deltas = (self.target_vertices - self.init_vertices)/count

    def get_artists(self):
        return [self.arrow]

    def animate(self, i):
        new_vertices = self.init_vertices + (i+1)*self.deltas
        self.arrow.set_xy(new_vertices)

class Circles:
    def __init__(self, ax, pos, **kwargs):
        self.init_positions = np.array(pos)
        self.circles = [Circle(xy=p, **kwargs) for p in pos]
        for c in self.circles:
            ax.add_patch(c)

    def set_target_positions(self, pos):
        self.target_positions = np.array(pos)

    def divide(self, count):
        self.deltas = (self.target_positions - self.init_positions)/count

    def get_artists(self):
        return self.circles

    def animate(self, i):
        new_positions = self.init_positions + (i+1)*self.deltas
        for c, p in zip(self.circles, new_positions):
            c.set_center(p)
