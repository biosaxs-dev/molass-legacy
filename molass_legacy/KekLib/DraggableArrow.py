"""
    DraggableArrow.py

    ref:
        https://stackoverflow.com/questions/63561034/arrows-with-draggable-end-in-matplotlib

    Copyright (c) 2022, SAXS Team, KEK-PF
"""

from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.patches import ArrowStyle

class DraggableArrow:
    def __init__(self, event):
        ax = event.inaxes
        self.ax = ax
        x = event.xdata
        y = event.ydata
        end_x = x+0.5
        end_y = y
        color = "C0"
        arrowstyle = ArrowStyle('<|-', head_width=0.4, head_length=0.6)
        self.vector = ax.annotate("", xy=(x, y), xytext=(end_x, end_y),
                                    arrowprops=dict(arrowstyle=arrowstyle,
                                                    fc=color, ec=color,
                                                    shrinkA=0, shrinkB=0)
                                    )
        self.start_point, = ax.plot(x, y, "o", color=color)
        self.end_point, = ax.plot(end_x, end_y, ".", color='none', picker=True)

    def update_position(self):
        ex, ey = self.end_point.get_data()
        self.vector.set_position([ex[0], ey[0]])

arrow = None
picked_artist = None

def button_press_callback(event):
    global picked_artist, arrow
    print("button_press_callback")
    if arrow is None:
        arrow = DraggableArrow(event)
        event.inaxes.figure.canvas.draw()
        print("created an arrow")

def pick_callback(event):
    'called when an element is picked'
    global picked_artist
    print("pick_callback")
    if event.mouseevent.button == MouseButton.LEFT:
        picked_artist = event.artist

def button_release_callback(event):
    'called when a mouse button is released'
    global picked_artist
    print("button_release_callback")
    if event.button == MouseButton.LEFT:
        picked_artist = None

def motion_notify_callback(event):
    'called when the mouse moves'
    global picked_artist
    # print("motion_notify_callback")
    if picked_artist is not None and event.inaxes is not None and event.button == MouseButton.LEFT:
        print("motion_notify_callback: move")
        picked_artist.set_data([event.xdata], [event.ydata])
        arrow.update_position()
        event.inaxes.figure.canvas.draw_idle()

def demo():
    fig, ax = plt.subplots()

    ax.plot(0, 0, "o")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    fig.canvas.mpl_connect('button_press_event', button_press_callback)
    fig.canvas.mpl_connect('button_release_event', button_release_callback)
    fig.canvas.mpl_connect('pick_event', pick_callback)
    fig.canvas.mpl_connect('motion_notify_event', motion_notify_callback)

    plt.show()

if __name__ == '__main__':
    demo()
