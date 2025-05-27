# coding: utf-8
"""
    GuinierDiffViewer.py

    Copyright (c) 2018-2019, SAXS-Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from CanvasFrame            import CanvasFrame
from molass_legacy.KekLib.OurMatplotlib          import get_default_colors
from SimpleGuinier          import SimpleGuinier
from molass_legacy._MOLASS.SerialSettings         import get_setting, get_xray_picking

class GuinierDiffFrame( Tk.Frame ):
    def __init__( self, parent, viewer ):
        Tk.Frame.__init__( self, parent)
        self.viewer = viewer

        figsize = ( 15, 5 ) if is_low_resolution() else ( 18, 6 )
        self.canvas_frame = canvas_frame = CanvasFrame( self, figsize=figsize )
        canvas_frame.pack()

        fig = canvas_frame.fig
        ax1  = fig.add_subplot( 131 )
        ax2  = fig.add_subplot( 132 )
        ax3  = fig.add_subplot( 133 )
        fig.tight_layout()
        self.axes = [ax1, ax2, ax3]

    def draw( self, xx, ly1, ly2, zoom=False, colors=None, gcolors=None, guinier_list=None ):
        for ax in self.axes:
            ax.cla()

        for ax in self.axes:
            ax.set_xlabel( 'Q²' )
            ax.set_ylabel( 'Ln(I)' )

        ax1, ax2, ax3 = self.axes

        elultion_no = self.viewer.j.get()
        ax1.set_title( "Guinier Plot of the original %d-th elution" % elultion_no )
        ax2.set_title( "Guinier Plot of the corrected %d-th elution" % elultion_no )
        ax3.set_title( "Rg difference between the original and the corrected" )

        if colors is None:
            color1, color2 = None, None
        else:
            color1, color2 = colors

        if gcolors is None:
            gcolor1, gcolor2 = 'cyan', 'red'
        else:
            gcolor1, gcolor2 = gcolors

        ax1.plot( xx, ly1, 'o', markersize=3, color=color1 )
        ax2.plot( xx, ly2, 'o', markersize=3, color=color2 )

        ax3.plot( xx, ly1, 'o', markersize=3, color=color1 )
        ax3.plot( xx, ly2, 'o', markersize=3, color=color2 )

        if zoom:
            xmin1, xmax1 = ax1.get_xlim()
            ymin1, ymax1 = ax1.get_ylim()
            xmin_   = xmin1 * 0.98 + xmax1 * 0.02
            xmax_   = xmin1 * 0.80 + xmax1 * 0.2
            # ymin_   = ymin1 * 0.3  + ymax1 * 0.7
            # ymax_   = ymin1 * 0.05 + ymax1 * 0.95
            ymin_, ymax_ = -9.0, -3.0
            for ax in self.axes:
                ax.set_xlim( xmin_, xmax_ )
                ax.set_ylim( ymin_, ymax_ )

        if guinier_list is not None:
            guinier1 = guinier_list[0]
            x_  = guinier1.guinier_x
            y_  = guinier1.guinier_y
            for ax in [ax1, ax3]:
                ax.plot( x_, y_, color=gcolor1, marker='o', markersize=8 )

            guinier2 = guinier_list[1]
            x_  = guinier2.guinier_x
            y_  = guinier2.guinier_y
            for ax in [ax2, ax3]:
                ax.plot( x_, y_, color=gcolor2, marker='o', markersize=8 )

            tx  = xmin_*0.6 + xmax_*0.4
            ty  = ymin_*0.2 + ymax_*0.8
            text = [ "Original", "Corrected", "Difference" ]
            for k, ax in enumerate(self.axes):
                ax.text( tx, ty, text[k], alpha=0.2, fontsize=30 )

            ty  = ymin_*0.4 + ymax_*0.6
            ax1.text( tx, ty, "Rg=%.3g" % guinier1.Rg, alpha=0.2, fontsize=30 )
            ax2.text( tx, ty, "Rg=%.3g" % guinier2.Rg, alpha=0.2, fontsize=30 )
            ax3.text( tx, ty, "Rg diff=%.3g" % ( guinier2.Rg - guinier1.Rg ), alpha=0.2, fontsize=30 )

        self.canvas_frame.show()


class ElutionCurveFrame( Tk.Frame ):
    def __init__( self, parent, viewer ):
        Tk.Frame.__init__( self, parent)
        self.viewer = viewer
        self.Q  = get_xray_picking()
        self.x  = np.arange( len(viewer.ecurve_y) )
        self.y  = viewer.ecurve_y

        figsize = ( 10, 2 ) if is_low_resolution() else ( 12, 2 )
        self.canvas_frame = canvas_frame = CanvasFrame( self, figsize=figsize )
        canvas_frame.pack()

        fig = canvas_frame.fig
        ax1  = fig.add_subplot( 111 )
        fig.tight_layout()
        self.ax = ax1
        self.canvas_frame.mpl_canvas.mpl_connect( 'button_press_event', self.move )

    def draw( self, j ):
        ax  = self.ax
        ax.cla()
        ax.set_title( 'Elution Curve at Q=%.3g' % self.Q )
        ax.plot( self.y, color='orange' )
        ymin, ymax = ax.get_ylim()
        ax.set_ylim( ymin, ymax )
        ax.plot( [j, j], [ymin, ymax], color='red' )

        xmin, xmax = ax.get_xlim()
        tx  = xmin * 0.7 + xmax * 0.3
        ty  = ymin * 0.5 + ymax * 0.5
        ax.text( tx, ty, "Double click to move", alpha=0.2, fontsize=30 )

        self.canvas_frame.show()

    def move( self, event ):
        if not event.dblclick:
            return

        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        n = np.argmin( np.abs( self.x - event.xdata) + np.abs(self.y - event.ydata) )
        self.viewer.j.set( n )

class GuinierDiffViewer( Tk.Toplevel ):
    def __init__( self, parent, qvector, ecurve_y, data1, data2, j, redraw_on_spin=True ):
        self.parent = parent
        Tk.Toplevel.__init__( self, parent )
        self.title( "GuinierDiffViewer" )

        self.ecurve_y   = ecurve_y
        self.data1  = data1
        self.data2  = data2

        vsize   = len(qvector)//2
        slice_  = slice(0,vsize)
        self.slice_ = slice_

        self.xx  = qvector[slice_]**2

        plot_frame = Tk.Frame( self )
        plot_frame.pack()

        self.gframe1 = GuinierDiffFrame( plot_frame, self )
        self.gframe1.pack()

        lower_frame = Tk.Frame( self )
        lower_frame.pack( fill=Tk.BOTH, expand=Tk.Y )

        self.ecurve_frame = ElutionCurveFrame( lower_frame, self )
        self.ecurve_frame.pack( side=Tk.LEFT )

        panel_frame = Tk.Frame( lower_frame )
        panel_frame.pack( side=Tk.RIGHT, fill=Tk.BOTH, expand=Tk.Y )

        panel_center = Tk.Frame( panel_frame )
        panel_center.pack( side=Tk.BOTTOM, anchor=Tk.CENTER, pady=40 )

        jsize   = data1.shape[0]
        self.j  = Tk.IntVar()
        self.j.set( j )

        redraw_frame = Tk.Frame( panel_center )
        redraw_frame.pack(side=Tk.LEFT, padx=10 )

        label = Tk.Label( redraw_frame, text="Elution No." )
        label.pack( side=Tk.LEFT )

        sinbox = Tk.Spinbox( redraw_frame, textvariable=self.j,
                                            from_=0, to=jsize-1, increment=1, 
                                            justify=Tk.CENTER, width=6, state=Tk.NORMAL )
        sinbox.pack( side=Tk.LEFT, padx=5, pady=10 )

        close_button = Tk.Button( panel_center, text="Close", command=self.close )
        close_button.pack( side=Tk.LEFT, padx=20, pady=10 )

        if redraw_on_spin:
            self.j.trace( 'w', lambda *args: self.draw() )
        else:
            redraw_button = Tk.Button( redraw_frame, text="Redraw", command=self.draw )
            redraw_button.pack( side=Tk.LEFT )

        self.draw()

        X   = parent.winfo_rootx()
        Y   = parent.winfo_rooty()
        print( 'X,Y=', X, Y )

        self.geometry("+%d+%d" % (X+50, Y+50))
        self.update()

    def get_guinier_data( self, j ):

        y1  = self.data1[j,self.slice_,1]
        y2  = self.data2[j,self.slice_,1]

        ly1 = np.log( y1 )
        ly2 = np.log( y2 )

        guinier_list = []
        for data in [ self.data1, self.data2 ]:
            data_ = data[j,:,:]
            sg = SimpleGuinier( data_ )
            guinier_list.append( sg )

        return ly1, ly2, guinier_list

    def draw( self ):
        try:
            j   = self.j.get()
        except:
            return

        colors  = get_default_colors()
        gcolors = [ 'cyan', 'yellow', 'red' ]

        ly1, ly2, guinier_list = self.get_guinier_data( j )

        self.gframe1.draw( self.xx, ly1, ly2, zoom=True, colors=colors[0:2], gcolors=gcolors[0:2], guinier_list=guinier_list[0:2] )
        self.ecurve_frame.draw( j )

    def close( self ):
        self.destroy()
        