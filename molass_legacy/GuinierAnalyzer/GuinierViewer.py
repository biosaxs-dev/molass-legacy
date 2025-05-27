# coding: utf-8
"""
    GuinierViewer.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from CanvasFrame            import CanvasFrame
from molass_legacy.KekLib.OurMatplotlib          import get_default_colors
from SimpleGuinier          import SimpleGuinier

class GuinierFrame( Tk.Frame ):
    def __init__( self, parent ):
        Tk.Frame.__init__( self, parent)

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

        ax1, ax2, ax3 = self.axes

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
            ty  = ymin_*0.5 + ymax_*0.5
            ax1.text( tx, ty, "Rg=%.3g" % guinier1.Rg, alpha=0.2, fontsize=30 )
            ax2.text( tx, ty, "Rg=%.3g" % guinier2.Rg, alpha=0.2, fontsize=30 )
            ax3.text( tx, ty, "Rg diff=%.3g" % ( guinier2.Rg - guinier1.Rg ), alpha=0.2, fontsize=30 )

        self.canvas_frame.show()

class GuinierViewer( Tk.Toplevel ):
    def __init__( self, parent, qvector, data1, data2, data3, j ):
        self.parent = parent
        Tk.Toplevel.__init__( self, parent )
        self.title( "GuinierViewer" )

        self.data1  = data1
        self.data2  = data2
        self.data3  = data3

        vsize   = len(qvector)//2
        slice_  = slice(0,vsize)
        self.slice_ = slice_

        self.xx  = qvector[slice_]**2

        plot_frame = Tk.Frame( self )
        plot_frame.pack( side=Tk.LEFT )
        panel_frame = Tk.Frame( self )
        panel_frame.pack( side=Tk.LEFT, padx=20, pady=20 )

        self.gframe1 = GuinierFrame( plot_frame )
        self.gframe1.pack()
        self.gframe2 = GuinierFrame( plot_frame )
        self.gframe2.pack()

        jsize   = data1.shape[0]
        self.j  = Tk.IntVar()
        self.j.set( j )
        sinbox = Tk.Spinbox( panel_frame, textvariable=self.j,
                                            from_=0, to=jsize-1, increment=1, 
                                            justify=Tk.CENTER, width=6, state=Tk.NORMAL )
        sinbox.pack()

        redraw_button = Tk.Button( panel_frame, text="Redraw", command=self.draw )
        redraw_button.pack()

        self.draw()

        X   = parent.winfo_rootx()
        Y   = parent.winfo_rooty()
        print( 'X,Y=', X, Y )

        self.geometry("+%d+%d" % (X+50, Y+50))
        self.update()

    def get_guinier_data( self, j ):

        y1  = self.data1[j,self.slice_,1]
        y2  = self.data2[j,self.slice_,1]
        y3  = self.data3[j,self.slice_,1]

        ly1 = np.log( y1 )
        ly2 = np.log( y2 )
        ly3 = np.log( y3 )

        guinier_list = []
        for data in [ self.data1, self.data2, self.data3 ]:
            data_ = data[j,:,:]
            sg = SimpleGuinier( data_ )
            guinier_list.append( sg )

        return ly1, ly2, ly3, guinier_list

    def draw( self ):
        try:
            j   = self.j.get()
        except:
            return

        colors  = get_default_colors()
        gcolors = [ 'cyan', 'yellow', 'red' ]

        ly1, ly2, ly3, guinier_list = self.get_guinier_data( j )

        self.gframe1.draw( self.xx, ly1, ly2, zoom=True, colors=colors[0:2], gcolors=gcolors[0:2], guinier_list=guinier_list[0:2] )
        self.gframe2.draw( self.xx, ly2, ly3, zoom=True, colors=colors[1:3], gcolors=gcolors[1:3], guinier_list=guinier_list[1:3] )
        