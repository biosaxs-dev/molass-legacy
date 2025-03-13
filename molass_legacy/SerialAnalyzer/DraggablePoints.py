"""
    DraggablePoints.py

    borrowed and adapted from
        Matplotlib drag overlapping points interactively
        http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DraggablePoints(object):
    def __init__( self, artists, tolerance=5, y_only=False,
                    select_cb=None, update_cb=None, select_fc=None ):
        for artist in artists:
            artist.set_picker(tolerance)
        self.artists = artists
        self.axes   = artists[0].axes
        self.artists_dict = {}
        for i, artist in enumerate( artists ):
            self.artists_dict[artist] = [ i, artist.get_facecolor() ]
        self.currently_dragging = False
        self.has_moved = False
        self.current_artist = None
        self.offset = (0, 0)
        self.y_only = y_only
        self.select_cb = select_cb
        self.update_cb = update_cb
        self.select_fc = select_fc

        for canvas in set(artist.figure.canvas for artist in self.artists):
            canvas.mpl_connect('button_press_event', self.on_press)
            canvas.mpl_connect('button_release_event', self.on_release)
            canvas.mpl_connect('pick_event', self.on_pick)
            canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.axes:
            return

        self.currently_dragging = True
        self.has_moved = False

    def on_release(self, event):
        self.currently_dragging = False
        self.current_artist = None
        if self.update_cb is not None and self.has_moved:
            self.update_cb( self )

    def on_pick(self, event):
        if self.current_artist is None:
            self.current_artist = event.artist
            # self.current_artist.set( alpha=1 )
            if self.select_cb is not None:
                self.select_cb( self.artists_dict[event.artist][0] )
            x0, y0 = event.artist.center
            x1, y1 = event.mouseevent.xdata, event.mouseevent.ydata
            self.offset = (x0 - x1), (y0 - y1)
            if self.select_fc is not None:
                for a in self.artists:
                    if a == self.current_artist:
                        a.set( fc=self.select_fc )
                    else:
                        a.set( fc=self.artists_dict[a][1] )
                self.current_artist.figure.canvas.draw()

    def on_motion(self, event):
        if not self.currently_dragging:
            return
        if self.current_artist is None:
            return
        dx, dy = self.offset
        if self.y_only:
            x_, y_ = self.current_artist.center
            self.current_artist.center = x_, event.ydata + dy
        else:
            self.current_artist.center = event.xdata + dx, event.ydata + dy
        self.current_artist.figure.canvas.draw()
        self.has_moved = True

    def unselect( self ):
        if self.select_fc is not None:
            for a in self.artists:
                    a.set( fc=self.artists_dict[a][1] )

    def select( self, ind ):
        for i, a in enumerate( self.artists ):
            if i == ind:
                fc_ = self.select_fc
            else:
                fc_ = self.artists_dict[a][1]
            a.set( fc=fc_ )

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.set(xlim=[-1, 2], ylim=[-1, 2])
    # ax.set(xlim=[ 0, 400 ], ylim=[-1, 2])

    circles = [patches.Circle((0.32, 0.3), 0.2, fc='r', alpha=0.5),
               patches.Circle((0.3, 0.3), 0.2, fc='b', alpha=0.5) ]
    for circ in circles:
        ax.add_patch(circ)

    def select_cb( ind ):
        print( 'select_cb: ind=', ind )

    def update_cb( obj ):
        print( 'update_cb' )

    dr = DraggablePoints( circles, select_cb=select_cb, update_cb=update_cb, select_fc='yellow' )
    plt.show()