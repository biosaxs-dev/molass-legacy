"""
    DraggableCurves.py

    ref:
        https://stackoverflow.com/questions/63561034/arrows-with-draggable-end-in-matplotlib

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import ArrowStyle

class DraggableCurves:
    def __init__(self, ax, x_only=False, button_release_user_callback=None, debug=False):
        self.ax = ax
        self.fig = ax.figure
        self.x_only = x_only
        self.button_release_user_callback = button_release_user_callback
        self.debug = debug
        self.clear()

    def clear(self):
        self.picked_artist = None
        self.last_moveevent = None
        self.curve = None
        self.cursor = None
        self.moving = False
        self.button_release_callback(None)
        self.curve_list = []
        self.displ_list = []

    def add(self, x, y, *args, **kwargs):
        curve, = self.ax.plot(x, y, *args, **kwargs, picker=True)
        self.curve_list.append(curve)
        self.displ_list.append(np.zeros(2))

    def create_cursor(self, artist):
        x, y = artist.get_data()
        self.cursor, = self.ax.plot(x, y, ":", color="gray", lw=3)
        self.moving = False

    def remove_cursor(self):
        if self.debug:
            print("remove_cursor:", self.cursor)
        if self.cursor is None:
            # in the initial call
            pass
        else:
            self.cursor.remove()
        self.cursor = None
        self.moving = False

    def update_cursor(self, artist):
        if self.cursor is None:
            self.create_cursor(artist)
        else:
            if artist == self.picked_artist:
                # picking twice is regarded to be moving for the cursor to be able to be removed
                self.moving = True
            else:
                self.remove_cursor()
                self.create_cursor(artist)

    def update_displ_list(self):
        i = None
        for k, artist in enumerate(self.curve_list):
            if artist == self.picked_artist:
                i = k
                break
        assert i is not None
        event = self.last_moveevent
        target_point = np.array([event.xdata, event.ydata])
        self.displ_list[i] += target_point - self.picked_point
        self.last_displacement = self.displ_list[i]

    def get_displacements(self):
        return self.displ_list

    def get_last_displacement(self):
        return self.last_displacement

    def move(self, event, artist):
        if event is None or event.xdata is None or artist is None:
            return

        self.moving = True
        target_point = np.array([event.xdata, event.ydata])
        dx, dy = target_point - self.picked_point

        if self.x_only:
            dy = 0

        if self.debug:
            print("move: (%.2g, %.2g) => (%.2g, %.2g)" % (*self.picked_point, *target_point))
            print("move: %.2g, %.2g" % (dx, dy))

        x, y = self.picked_artist.get_data()
        artist.set_data(x+dx, y+dy)
        self.last_moveevent = event

    def pick_callback(self, event):
        'called when an element is picked'
        if self.debug:
            print("pick_callback")
        if event.mouseevent.button == MouseButton.LEFT:
            self.picked_artist = event.artist
            self.update_cursor(self.picked_artist)

            mouseevent = event.mouseevent
            self.picked_point = np.array([mouseevent.xdata, mouseevent.ydata])
            self.fig.canvas.draw_idle()

    def button_release_callback(self, event):
        'called when a mouse button is released'
        if self.debug:
            print("button_release_callback")
        if event is None or event.button == MouseButton.LEFT:
            if self.moving:
                self.move(self.last_moveevent, self.picked_artist)
                self.update_displ_list()
                if self.button_release_user_callback is not None:
                    self.button_release_user_callback(self)
                self.picked_artist = None
                self.picked_point = None
                self.last_moveevent = None
                self.remove_cursor()
                self.fig.canvas.draw_idle()

    def motion_notify_callback(self, event):
        'called when the mouse moves'
        if self.picked_artist is not None and event.inaxes is not None and event.button == MouseButton.LEFT:
            if self.debug:
                print("motion_notify_callback: move")
            self.move(event, self.cursor)
            self.fig.canvas.draw_idle()

def demo():
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
    ax1.set_title("DraggableCurves Demo")

    x = np.linspace(-np.pi, +np.pi, 100)

    curves = DraggableCurves(ax1, x_only=True, debug=False)
    curves.add(x, np.sin(x))
    curves.add(x, np.sin(2*x))

    ax1.set_xlim(-np.pi, +np.pi)
    ax1.set_ylim(-2, 2)

    fig.canvas.mpl_connect('button_release_event', curves.button_release_callback)
    fig.canvas.mpl_connect('pick_event', curves.pick_callback)
    fig.canvas.mpl_connect('motion_notify_event', curves.motion_notify_callback)

    plt.show()

if __name__ == '__main__':
    demo()
