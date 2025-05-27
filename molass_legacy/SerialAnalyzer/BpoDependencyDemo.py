# coding: utf-8
"""
    BpoDependencyDemo.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy.stats import norm
from bisect import bisect_right
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

N  = 301
NUM_ITERATIONS  = 100
MIN_SIGMA = 3
MAX_SIGMA = 7

class MyFormatter:
    def __init__(self, x):
        self.x  = x

    def x_formatter( self, val, pos):
        if pos is None: return ''
        # print( 'val=', val )
        ival = int(val)
        if ival >= 0 and ival <= N:
            return '%g\n(%.2gσ)' % ( val, self.x[ival] )
        else:
            return '%g' % ( val )

def demo_func(fig):
    ym  = norm.pdf( 0, 0, 1 )

    rec_list = []
    params_list = []

    for w in np.linspace( MIN_SIGMA, MAX_SIGMA, 21 ):

        x   = np.linspace(-w, w, N )

        if w == 3:
            i   = 0
            wi  = w
            xi  = x
        elif abs( w - 5 ) < 1e-5:
            i   = 1
            wi  = w
            xi  = x
        elif w == 7:
            i   = 2
            wi  = w
            xi  = x
        else:
            i   = None
            wi  = None
            xi  = None

        y0  = norm.pdf( x, 0, 1 )
        scale = 1/ym
        y_  = y0 * scale

        x_array = []
        p_array = []
        s_array = []

        params_row = []
        for n in range(50):
            if n == 0:
                j = 0
            elif n == 24:
                j = 1
            elif n == 49:
                j = 2
            else:
                j = None

            nn = 0.01 * (n+1)
            ys_array = []
            zi_array = []
            yi  = None
            yi_ = None
            for k in range(NUM_ITERATIONS):
                noise = np.random.normal(0, nn, N)
                y   = y_ + noise
                ys  = sorted( y )
                zi  = bisect_right( ys, 0 )
                zi_array.append( zi )
                ys_array.append( ys[zi] )
                if k == 0  and i is not None and j is not None:
                    yi_ = y_
                    yi  = y

            p = np.average( zi_array ) * 100 / N
            x_array.append( nn )
            p_array.append( p )
            s_array.append( np.std( zi_array ) )

            if i is not None and j is not None:
                params_row.append( [ wi, nn, p ] )

        rec_list.append( [ w, x_array, p_array, s_array ] )
        if i is not None:
            params_list.append( params_row )

    gs = gridspec.GridSpec( 3, 6 )
    ax3d  = fig.add_subplot( gs[:,0:3], projection='3d' )
    ax3d.set_title( 'Noise and Sigma-width Dependency of Base Percentile Offset' )
    ax3d.set_xlabel( 'noise' )
    ax3d.set_ylabel( 'width(σ)' )
    ax3d.set_zlabel( 'base percentile offset' )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    # print( 'colors=', colors )

    for rec in rec_list:
        w, x_array, p_array, s_array = rec
        y_array = np.ones( len(x_array) ) * w
        ax3d.plot( x_array, y_array, s_array, color='yellow' )
        ax3d.plot( x_array, y_array, p_array, color='orange' )

    params = np.array( params_list )


    for i in range(3):
        for j in range(3):
            wi, nn, p = params[i,j]
            x   = np.linspace(-wi, wi, N )
            y0  = norm.pdf( x, 0, 1 )
            scale = 1/ym
            y_  = y0 * scale
            noise = np.random.normal(0, nn, N)
            y   = y_ + noise

            sy  = sorted( y )
            m   = bisect_right( sy, nn )
            pn  = m/N
            print( (i, j), pn )

            ax  = fig.add_subplot( gs[i,3+j] )
            ax.set_ylim(-1, 2)

            fmt = MyFormatter(x)
            formatter   = FuncFormatter( fmt.x_formatter )
            ax.xaxis.set_major_formatter( formatter )
            # TODO: pn ?
            # title = 'width=%g; noise=%g; bpo=%.2g; %.2g' % ( wi, nn, p, pn*0.5 )
            title = 'width=%g; noise=%g; bpo=%.2g' % ( wi, nn, p )
            ax.set_title( title )
            ax.plot( y )
            ax.plot( y_ )
            ax.plot( [0, 300], [0, 0], ':', color='red' )
            # ax.plot( x, y )
            # ax.plot( x, y_ )
            fwhm = 2 * np.sqrt( 2*np.log(2) ) * 150/wi
            print( 'fwhm[%d,%d]=%.3g' % (i, j, fwhm) )

            ax3d.scatter( nn, wi, p, color='red' )

    fig.tight_layout()

class BpoDependencyDemo(Dialog):
    def __init__(self, parent):
        self.parent = parent
        self.log_scale = 1
        Dialog.__init__(self, parent, "Noise and Sigma-width Dependency of BPO", visible=False)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        bframe = Tk.Frame(body_frame)
        bframe.pack(fill=Tk.X)
        tbframe = Tk.Frame(bframe)
        tbframe.pack(side=Tk.LEFT)

        self.fig = plt.figure( figsize=(18, 9)  )
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        demo_func(self.fig)
        self.fig.tight_layout()
        self.mpl_canvas.draw()

        # self.toolbar = NavigationToolbar( self.mpl_canvas, tbframe, show_mode=False )
        self.toolbar = NavigationToolbar( self.mpl_canvas, tbframe )
        self.toolbar.update()

    def show(self):
        self._show()
