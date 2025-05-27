# coding: utf-8
"""

    ファイル名：   PilatusImageViewer.py

    処理内容：

       Pilatus 画像データの表示

    Copyright (c) 2015-2018, Masatsuyo Takahashi, KEK-PF
"""
from __future__ import division, print_function, unicode_literals
import os
import sys
import numpy as np
import pylab as pl
import copy

from molass_legacy.KekLib.OurTkinter         import Tk, ToolTip
from Preferences        import get_preference
from OurColorMaps       import CmapAlbulaLikeDynamic, Diverging
# from matplotlib.colors import SymLogNorm
# from PilatusColors      import LogNorm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from ZoomPan            import ZoomPan
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure  import Figure
from molass_legacy.KekLib.TkSupplements      import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib      import NavigationToolbar, CoordinateFormatter, DataCursor, ColorBar
from ControlKeyState    import set_shift_key_state, set_ctrl_key_state

value_shift = 10

class PilatusImageViewer( Tk.Toplevel ):
    def __init__( self, action, sample_id, exec_params ):
        Tk.Toplevel.__init__( self )
        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'

        # ここでアイコン設定すると、
        # 一瞬小さい Toplevel 画面が表示されるため、
        # 最後に設定する。
        # tk_set_icon_portable( self, 'synthesizer' )

        if action == 1:
            fig_title   = '%s (original)' % ( sample_id )
        else:
            fig_title   = '%s (adjusted)' % ( sample_id )

        self.title( fig_title )

        num_images = len( exec_params )

        fig = Figure( figsize=( num_images * 4, 5 ), dpi=100 )
        gs  = gridspec.GridSpec( 1, num_images )

        im_array_list   = []
        im_shift_list   = []
        ax_list         = []
        im_list         = []

        i = -1
        for param in exec_params:
            # print( '[%d] param=' % (i), param )

            i += 1
            file, im_array_orig = param[0:2]

            im_array = im_array_orig + value_shift
            im_array_list.append( im_array )
            if len( param ) >= 4:
                im_shift_list.append( param[2:4] )
            else:
                im_shift_list.append( [] )

            ax = fig.add_subplot( gs[i] )

            ax.set_title( file )

            if i ==0:
                numrows, numcols = im_array.shape
                self.formatter = CoordinateFormatter( numrows, numcols, im_array_list, value_shift )

            ax.format_coord = self.formatter
            ax.invert_yaxis()
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

            cmap_preference = get_preference( 'color_map' )
            if cmap_preference == 'Diverging':
                cmap_ = Diverging()
            else:
                vmin = np.min( im_array )
                vmax = np.max( im_array )
                v60  = np.percentile( im_array, 60 )
                logvmin, logvmax, logv60 = np.log(vmin), np.log(vmax), np.log( v60 )
                # cmap_mean = 0.2352 - 0.0590 * logvmin - 0.0393 * logvmax + 0.1480 * logv60
                cmap_mean = 1.1120 - 0.2301 * logvmin - 0.0162 * logvmax + 0.0176 * logv60
                # print( 'cmap_mean=%g' % ( cmap_mean ) )
                cmap_ = CmapAlbulaLikeDynamic( mean=cmap_mean )
            im = ax.imshow( im_array, cmap=cmap_, norm=LogNorm(), interpolation="nearest" )
            cb = ColorBar( im, ax )
            ax_list.append( ax )
            im_list.append( [ im, cmap_ ] )

        # fig.tight_layout()
        fig.set_tight_layout(True)

        canvas = FigureCanvasTkAgg( fig, master=self )

        if self.mplt_ge_2_2:
            canvas.draw()
        else:
            canvas.show()

        ToolTip( self, "<mouse-wheel> to zoom images; <drag> to move them.  <shift-click> to show pixel annotations; press <escape> to hide them." )

        scale = 2.0

        # Zoom 機能の設定
        for i in range( len(ax_list) ):
            # i 番目が先頭になるように並べ替えた ax_list を作る
            ax_list_copy = []
            ax_list_copy.append( ax_list[i] )
            for j in range( len(ax_list) ):
                if j != i:
                    ax_list_copy.append( ax_list[j] )
            # 並べ替えた ax_list_copy を使って Zoom 制御オブジェクトを作る
            zp      = ZoomPan( copy.copy(i), canvas, ax_list_copy[0], im_list[i] )
            figZoom = zp.zoom_factory( ax_list_copy, base_scale = scale )
            figPan  = zp.pan_factory( ax_list_copy )

        toolbar = NavigationToolbar( canvas, self )
        toolbar.update()

        canvas.get_tk_widget().pack( side=Tk.TOP, fill=Tk.BOTH, expand=1 )

        self.data_cursor = DataCursor( ax_list, action, value_shift )

        def on_key_press( event ):
            # print('you pressed %s' % event.key)

            # TODO:
            # key_press_handler(event, canvas, toolbar)

            if event.key == 'shift':
                set_shift_key_state( True )
            elif event.key == 'control':
                set_ctrl_key_state( True )

            self.data_cursor( 1, event, im_array_list, im_shift_list )

        def on_key_release( event ):
            if event.key == 'shift':
                set_shift_key_state( False )
            elif event.key == 'control':
                set_ctrl_key_state( False )

        def on_button_press( event ):
            self.data_cursor( 3, event, im_array_list, im_shift_list )

        canvas.mpl_connect( 'key_press_event',      on_key_press )
        canvas.mpl_connect( 'key_release_event',    on_key_release )
        canvas.mpl_connect( 'button_press_event',   on_button_press )

        def _quit():
            self.destroy()  # this is necessary on Windows to prevent
                            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        self.quit_button = Tk.Button(master=self, text='Quit', command=_quit)
        self.quit_button.pack(side=Tk.BOTTOM)

        tk_set_icon_portable( self, 'synthesizer' )

        self.update()   # 必要？

        return
