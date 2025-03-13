# coding: utf-8
"""

    Canvas3D.py

        Canvas for 3D-plot with ayx-span selectors

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.widgets import Button, SpanSelector
from matplotlib import colors as mcolors

SPAN_STAYS  = False

class OurSpanSelector(SpanSelector):
    def __init__(*args, **kwargs):
        self = args[0]
        self.onpress_callback = kwargs.pop('onpress_callback', None)
        self.onrelease_callback = kwargs.pop('onrelease_callback', None)
        self.onhover_callback = kwargs.pop('onhover_callback', None)
        SpanSelector.__init__(*args, **kwargs)

    def _press(self, event):
        """on button press event"""
        if self.onpress_callback is not None:
            self.onpress_callback(event)
        return SpanSelector._press(self, event)

    def _release(self, event):
        """on button release event"""
        if self.onrelease_callback is not None:
            self.onrelease_callback(event)
        return SpanSelector._release(self, event)

    def onmove(self, event):
        """ overriding _SelectorWidget.onmove """
        if self.onhover_callback is not None:
            if not self.ignore(event):
                if self.pressv is None:
                    self.onhover_callback(event)
        return super(SpanSelector, self).onmove(event)

axis_names = [ "X", "Y", "Z" ]

# https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
dc = default_colors[0]
facecolors = [ mcolors.to_rgba(arg, alpha=0.1) for arg in [dc, dc] ]

class Canvas3D:
    def __init__(self, fig, grid_rows, grid_cols, hspace=0.3):
        self.fig = fig
        gs = gridspec.GridSpec( grid_rows, grid_cols )
        self.ax = fig.add_subplot(gs[0:grid_rows-3,:], projection='3d')
        axx = fig.add_subplot( gs[grid_rows-3,1:-1] )
        axy = fig.add_subplot( gs[grid_rows-2,1:-1] )
        axz = fig.add_subplot( gs[grid_rows-1,1:-1] )
        self.axes = [axx, axy, axz]

        fig.subplots_adjust( top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=hspace )

        for k, ax_ in enumerate(self.axes):
            ax_.tick_params(
                which='both',
                bottom=False,
                top=False,
                left=False,
                labelbottom=False,
                labelleft=False)
            ax_.set_xlim(0, 1)
            ax_.text( -0.03, 0, axis_names[k] + " range selector", ha='right' )

        resetax = plt.axes([0.82, 0.18, 0.12, 0.04])
        self.v_reset_btn = Button(resetax, 'view reset', hovercolor='0.975')
        self.v_reset_btn.on_clicked(self.view_reset)

        resetax = plt.axes([0.82, 0.05, 0.12, 0.04])
        self.r_reset_btn = Button(resetax, 'range reset', hovercolor='0.975')
        self.r_reset_btn.on_clicked(self.range_reset)

        props = dict(alpha=0.5, facecolor=default_colors[0])
        minspan = 0.05

        spanx = OurSpanSelector(axx, self.on_select_x,
                    'horizontal', useblit=True, span_stays=SPAN_STAYS,
                    onpress_callback=self.on_press_or_hover_x,
                    onmove_callback=self.on_move_x,
                    onrelease_callback=self.on_release_x,
                    onhover_callback=self.on_press_or_hover_x,
                    props=props,
                    minspan=minspan)

        spany = OurSpanSelector(axy, self.on_select_y,
                    'horizontal', useblit=True, span_stays=SPAN_STAYS,
                    onpress_callback=self.on_press_or_hover_y,
                    onmove_callback=self.on_move_y,
                    onrelease_callback=self.on_release_y,
                    onhover_callback=self.on_press_or_hover_y,
                    props=props,
                    minspan=minspan)

        spanz = OurSpanSelector(axz, self.on_select_z,
                    'horizontal', useblit=True, span_stays=SPAN_STAYS,
                    onpress_callback=self.on_press_or_hover_z,
                    onmove_callback=self.on_move_z,
                    onrelease_callback=self.on_release_z,
                    onhover_callback=self.on_press_or_hover_z,
                    props=props,
                    minspan=minspan)

        fig.canvas.mpl_connect('axes_leave_event', self.on_leave_selector)
        fig.canvas.mpl_connect('motion_notify_event', self.on_leave_selector)
        fig.canvas.mpl_connect('figure_leave_event', self.on_leave_selector)

        self.selectors = [ spanx, spany, spanz ]
        self.init_limits = None
        self.poly_x = None
        self.poly_y = None
        self.poly_z = None

    def get_axis(self):
        return self.ax

    def set_init_lims(self):
        if self.init_limits is None:
            ax = self.ax
            self.init_limits = copy.deepcopy( [ ax.get_xlim(), ax.get_ylim(), ax.get_zlim() ] )
            print( 'init_limits=', self.init_limits )

    def view_reset(self, event):
        self.ax.view_init()
        self.fig.canvas.draw()

    def range_reset(self, event):
        # print( 'range_reset' )
        for sel in self.selectors:
            if SPAN_STAYS:
                sel.stay_rect.set_visible(False)
            sel.canvas.draw()

        if self.init_limits is not None:
            # print('reset limits')
            ax = self.ax
            ax.set_xlim( self.init_limits[0] )
            ax.set_ylim( self.init_limits[1] )
            ax.set_zlim( self.init_limits[2] )
            self.fig.canvas.draw()

    def on_select_x(self, min_, max_):
        # print( 'on_select_x', min_, max_ )
        xmin, xmax = self.ax.get_xlim()
        xmin_ = xmin * (1 - min_) + xmax * min_
        xmax_ = xmin * (1 - max_) + xmax * max_
        self.ax.set_xlim(xmin_, xmax_)
        self.fig.canvas.draw()

    def on_select_y(self, min_, max_):
        # print( 'on_select_y', min_, max_ )
        ymin, ymax = self.ax.get_ylim()
        ymin_ = ymin * (1 - min_) + ymax * min_
        ymax_ = ymin * (1 - max_) + ymax * max_
        self.ax.set_ylim(ymin_, ymax_)
        self.fig.canvas.draw()

    def on_select_z(self, min_, max_):
        # print( 'on_select_z', min_, max_ )
        zmin, zmax = self.ax.get_zlim()
        zmin_ = zmin * (1 - min_) + zmax * min_
        zmax_ = zmin * (1 - max_) + zmax * max_
        self.ax.set_zlim(zmin_, zmax_)
        self.fig.canvas.draw()

    def compute_zs(self, u, w, xminmax):
        self.set_init_lims()

        xmin, xmax = xminmax
        px = xmin * (1 - u) + xmax * u
        if w is None:
            w = u + 0.01
        qx = xmin * (1 - w) + xmax * w
        xs = [px, qx]
        return xs

    def compute_verts(self, yminmax, zminmax):
        ymin, ymax = yminmax
        zmin, zmax = zminmax
        y = [ymin, ymax, ymax, ymin]
        z = [zmin, zmin, zmax, zmax]
        yz = list(zip(y, z))
        verts = [ yz, yz ]
        return verts

    def on_press_or_hover_x(self, event):
        # print('on_press_x: ', event.xdata, event.ydata)
        ax = self.ax
        self.clear_poly_any(except_poly=0)
        if self.poly_x is not None:
            self.poly_x.remove()
        self.verts_x = self.compute_verts(ax.get_ylim(), ax.get_zlim())
        self.poly_x = PolyCollection(self.verts_x, facecolors=facecolors)
        xs = self.compute_zs(event.xdata, None, ax.get_xlim())
        ax.add_collection3d(self.poly_x, zs=xs, zdir='x')
        self.fig.canvas.draw()

    def on_press_or_hover_y(self, event):
        # print('on_press_y: ', event.xdata, event.ydata)
        ax = self.ax
        self.clear_poly_any(except_poly=1)
        if self.poly_y is not None:
            self.poly_y.remove()
        self.verts_y = self.compute_verts(ax.get_xlim(), ax.get_zlim())
        self.poly_y = PolyCollection(self.verts_y, facecolors=facecolors)
        ys = self.compute_zs(event.xdata, None, ax.get_ylim())
        ax.add_collection3d(self.poly_y, zs=ys, zdir='y')
        self.fig.canvas.draw()

    def on_press_or_hover_z(self, event):
        # print('on_press_z: ', event.xdata, event.ydata)
        ax = self.ax
        self.clear_poly_any(except_poly=2)
        if self.poly_z is not None:
            self.poly_z.remove()
        self.verts_z = self.compute_verts(ax.get_xlim(), ax.get_ylim())
        self.poly_z = PolyCollection(self.verts_z, facecolors=facecolors)
        zs = self.compute_zs(event.xdata, None, ax.get_zlim())
        ax.add_collection3d(self.poly_z, zs=zs, zdir='z')
        self.fig.canvas.draw()

    def on_move_x(self, min_, max_):
        ax = self.ax
        xs = self.compute_zs(min_, max_, ax.get_xlim())
        if self.poly_x is not None:
            self.poly_x.remove()
        self.verts_x = self.compute_verts(ax.get_ylim(), ax.get_zlim())
        self.poly_x = PolyCollection(self.verts_x, facecolors=facecolors)
        ax.add_collection3d(self.poly_x, zs=xs, zdir='x')
        self.fig.canvas.draw()

    def on_move_y(self, min_, max_):
        ax = self.ax
        ys = self.compute_zs(min_, max_, ax.get_ylim())
        if self.poly_y is not None:
            self.poly_y.remove()
        self.verts_y = self.compute_verts(ax.get_xlim(), ax.get_zlim())
        self.poly_y = PolyCollection(self.verts_y, facecolors=facecolors)
        ax.add_collection3d(self.poly_y, zs=ys, zdir='y')
        self.fig.canvas.draw()

    def on_move_z(self, min_, max_):
        ax = self.ax
        zs = self.compute_zs(min_, max_, ax.get_zlim())
        if self.poly_z is not None:
            self.poly_z.remove()
        self.verts_z = self.compute_verts(ax.get_xlim(), ax.get_ylim())
        self.poly_z = PolyCollection(self.verts_z, facecolors=facecolors)
        ax.add_collection3d(self.poly_z, zs=zs, zdir='z')
        self.fig.canvas.draw()

    def on_release_x(self, event):
        # print('on_release_x')
        if self.poly_x is not None:
            self.poly_x.remove()
        self.poly_x = None

    def on_release_y(self, event):
        # print('on_release_y')
        if self.poly_y is not None:
            self.poly_y.remove()
        self.poly_y = None

    def on_release_z(self, event):
        # print('on_release_z')
        if self.poly_z is not None:
            self.poly_z.remove()
        self.poly_z = None

    def clear_poly_any(self, except_poly=-1):
        removed = False
        if self.poly_x is not None and except_poly != 0:
            self.poly_x.remove()
            self.poly_x = None
            removed = True
        if self.poly_y is not None and except_poly != 1:
            self.poly_y.remove()
            self.poly_y = None
            removed = True
        if self.poly_z is not None and except_poly != 2:
            self.poly_z.remove()
            self.poly_z = None
            removed = True
        if removed:
            self.fig.canvas.draw()

    def on_leave_selector(self, event):
        if event.xdata is None or event.inaxes == self.ax:
            self.clear_poly_any()
