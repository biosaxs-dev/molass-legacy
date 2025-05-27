"""

    ファイル名：   TkUtils.py

    処理内容：

    Copyright (c) 2016-2024, Masatsuyo Takahashi, KEK-PF

"""
import os
import sys
import re
from MultiMonitor import get_selected_monitor

DO_NO_ADJUST_GEOMETRY   = 'DO_NO_ADJUST_GEOMETRY'

def split_geometry( geometry ):
    n = geometry.split('+')
    s = n[0].split('x')
    w = int( s[0] )
    h = int( s[1] )
    x = 0 if len(n) < 2 else int( n[1] )
    y = 0 if len(n) < 3 else int( n[2] )
    return [ w, h, x, y ]

def join_geometry( w, h, x, y ):
    return '%dx%d+%d+%d' % ( w, h, x, y )

def geometry_fix( top, x, y ):
    W, H, X, Y = split_geometry( top.geometry() )
    top.geometry( join_geometry( W, H, x, y ) )
    top.update()

"""
    URL: http://effbot.org/tkinterbook/wm.htm
"""
def parsegeometry(geometry):
    m = re.match(r"(\d+)x(\d+)([-+]\d+)([-+]\d+)", geometry)
    if not m:
        raise ValueError("failed to parse geometry string")
    return map(int, m.groups())

do_not_adjust_geometry = os.environ.get( DO_NO_ADJUST_GEOMETRY )
"""
    TODO:
    some tests seem to go wrong with adjusted geometry.
    it may have been improved by self.udate() before applying adjusted geometry.
"""

def adjusted_geometry( geometry, monitor=None, width_margin=0, height_margin=0.15, loc=None, debug=False ):
    global max_monitor, monitors

    if do_not_adjust_geometry:
        return geometry

    if monitor is None:
        monitor = get_selected_monitor()

    try:
        w0, h0, x0, y0 = split_geometry( geometry )
        w1, h1, x1, y1 = monitor.width, monitor.height, monitor.x,  monitor.y
        if debug:
            print(w0, h0, x0, y0)
            print(w1, h1, x1, y1)

        w1_ = int(w1 * (1-width_margin) )
        h1_ = int( h1 * (1-height_margin) )

        if loc is None:
            pass
        elif loc == "center":
            x1 += w1//2 - w0//2
            y1 += h1//2 - h0//2
        else:
            assert False

        return join_geometry( min(w0, w1_), min(h0, h1_), x1+x0, y1+y0 )
    except:
        return geometry

def geometry_move( toplevel, parent, x=50, y=50 ):
    x_, y_ = (parent.winfo_rootx()+x, parent.winfo_rooty()+y)
    # print( 'geometry_move: x_, y_=',  x_, y_  )
    toplevel.geometry("+%d+%d" % ( x_, y_ ) )

def rational_geometry( self, parent, w_ratio=0.5, h_ratio=0.5 ):
    w, h, x, y = split_geometry( parent.geometry() )
    x_, y_ = (parent.winfo_rootx() + int(w*w_ratio), parent.winfo_rooty() + int(h*h_ratio))
    self.geometry("+%d+%d" % ( x_, y_ ) )

def is_low_resolution():
    try:
        monitor = get_selected_monitor()
        # print( monitor.width, monitor.height )
        return monitor.height < 800
    except:
        # it seems this can occur
        return False

def get_widget_geometry(widget):
    w = widget.winfo_width()
    h = widget.winfo_height()
    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    return (w, h, x, y)

def get_tk_root(loc=None, withdraw=True, debug=False):
    if debug:
        import logging
        logging.basicConfig(filename="get_tk_root-debug.log", level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info("get_tk_root entry")
    import OurTk as Tk
    from molass_legacy.KekLib.DebugPlot import set_plot_env
    root = Tk.Tk()
    if debug:
        logger.info("root ok")
    adj_geometry = adjusted_geometry(root.geometry(), loc=loc)
    if debug:
        logger.info("adjusted_geometry ok")
    root.geometry(adj_geometry)
    if debug:
        logger.info("geometry ok")
    if withdraw:
        root.withdraw()
    if debug:
        logger.info("withdraw ok")
    root.update()
    if debug:
        logger.info("update ok")
    set_plot_env(root)
    if debug:
        logger.info("set_plot_env ok")
    return root

class PositionSynchronizer:
    # modifeid from https://stackoverflow.com/questions/45183914/tkinter-detecting-a-window-drag-event
    def __init__(self, leader, follower, debug=False):
        self.leader = leader
        self.follower = follower
        self.geometry_re = re.compile(r"^(.+)(\+\-?\d+\+\-?\d+)")
        self.drag_id = None
        self.debug = debug
        self.leader.bind('<Configure>', self.on_configure)

    def on_configure(self, event):
        # print("on_configure")

        if event.widget is self.leader: # do nothing if the event is triggered by one of root's children
            if self.drag_id is None:
                # action on drag start
                if self.debug:
                    print('start drag')
            else:
                # cancel scheduled call to stop_drag
                self.leader.after_cancel(self.drag_id)
                # print('dragging')
            # schedule stop_drag
            self.drag_id = self.leader.after(100, self.stop_drag)

    def stop_drag(self):
        if self.debug:
            print('stop drag')
        self.synchronize_position()
        self.drag_id = None

    def synchronize_position(self):
        g1 = self.leader.geometry()
        g2 = self.follower.geometry()
        """
        g1: WWWxHHH+AAA+BBB
        g2: ___x___+CCC+DDD
        g2: ___x___+AAA+BBB <=== new geometry
        """
        m = self.geometry_re.match(g1)
        if m:
            position = m.group(2)
            new_g = re.sub(self.geometry_re, lambda m_: m_.group(1) + position, g2)
            if self.debug:
                print("g1=", g1)
                print("g2=", g2)
                print("new_g=", new_g)
            self.follower.geometry(new_g)
