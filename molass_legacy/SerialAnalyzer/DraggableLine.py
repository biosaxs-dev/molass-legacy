"""
    DraggableLine.py

    borrowed, adapted and renamed from
        Interactive Line in matplotlib
        http://stackoverflow.com/questions/34855074/interactive-line-in-matplotlib
"""
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

class DraggableLine(object):
    def __init__(self, line, dual=None, epsilon=1.0, select_cb=None, update_cb=None, vertical_only=False,
                    sync_widgets=None ):
        canvas = line.figure.canvas
        self.canvas = canvas
        self.line = line
        self.dual = dual
        self.epsilon = epsilon
        self.select_cb = select_cb
        self.update_cb = update_cb
        self.vertical_only  = vertical_only
        self.sync_widgets   = sync_widgets
        self.axes = line.axes
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())

        self.ind = None
        self.has_moved = False

        canvas.mpl_connect('button_press_event', self.button_press_callback)
        canvas.mpl_connect('button_release_event', self.button_release_callback)
        canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)

    def get_ind(self, event):
        if event.inaxes !=self.axes or event.xdata is None or event.ydata is None:
            return None

        x = np.array(self.line.get_xdata())
        y = np.array(self.line.get_ydata())
        d = np.sqrt((x-event.xdata)**2 + (y - event.ydata)**2)
        if min(d) > self.epsilon:
            return None
        if d[0] < d[1]:
            return 0
        else:
            return 1

    def button_press_callback(self, event, level=0 ):
        if event.button != 1:
            return
        self.ind = self.get_ind(event)
        if self.ind is None: return

        print( 'button_press_callback: ind=', self.ind, 'level=', level )

        if self.select_cb is not None:
            self.select_cb( self.ind )

        self.line.set_animated(True)
        if self.dual is not None:
            self.dual.set( markevery=[ self.ind ] )
            self.dual.set_animated(True)

        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.line.axes.bbox)

        self.axes.draw_artist(self.line)
        self.canvas.blit(self.axes.bbox)
        self.has_moved = False
        if level == 0:
            print( 'len(sync_widgets)=', len(self.sync_widgets) )
            for w in self.sync_widgets:
                if w != self:
                    w.button_press_callback( event, level=1 )

    def button_release_callback(self, event):
        if event.button != 1:
            return
        self.ind = None
        self.line.set_animated(False)
        if self.dual is not None:
            self.dual.set_animated(False)
        self.background = None
        if self.update_cb is not None and self.has_moved:
            self.update_cb( self )
        self.line.figure.canvas.draw()

    def motion_notify_callback(self, event):
        if event.inaxes != self.line.axes:
            return
        if event.button != 1:
            return
        if self.ind is None:
            return

        if not self.vertical_only:
            self.xs[self.ind] = int( event.xdata + 0.5 )
        self.ys[self.ind] = event.ydata
        self.line.set_data(self.xs, self.ys)
        if self.dual is not None:
            self.dual.set_data(self.xs, self.ys)

        self.canvas.restore_region(self.background)
        self.axes.draw_artist(self.line)
        if self.dual is not None:
            self.axes.draw_artist(self.dual)
        self.canvas.blit(self.axes.bbox)
        self.has_moved = True

    def unselect( self ):
        if self.dual is not None:
            self.dual.set( markevery=[] )

if __name__ == '__main__':

    fig, ax = plt.subplots()
    line1 = Line2D([0, 1], [0, 1], marker = 'o', markersize=7, markerfacecolor = 'pink')
    line2 = Line2D([0, 1], [0, 1], linestyle='None', marker = 'o', markersize=7, markerfacecolor = 'red', markevery=[] )
    ax.add_line(line1)
    ax.add_line(line2)

    def select_cb( ind ):
        print( 'select_cb: ind=', ind )

    def update_cb( obj ):
        print( 'update_cb' )

    linebuilder = DraggableLine(line1, dual=line2, select_cb=select_cb, update_cb=update_cb )
    # linebuilder = DraggableLine(line1 )

    ax.set_title('click to create lines')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    plt.show()