# coding: utf-8
"""
    ScatteringPlot.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np
import copy
import re
from bisect                 import bisect_right
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from mpl_toolkits.mplot3d   import Axes3D
from molass_legacy._MOLASS.SerialSettings         import get_setting, get_xray_picking
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.TkUtils                import split_geometry
from ResultGui              import ResultGui
from ScatteringBaseUtil     import apply_baseline_correction_impl
from ScatteringBasesurface  import ScatteringBasesurface, is_to_plot

DEBUG           = False
ZLIM_EXPAND     = 1.25

class ScatteringPlot:
    def __init__( self, qvector, curve_y, intensity_array, xray_slice, affine_info, parent=None ):
        self.qvector    = qvector
        self.index      = bisect_right( qvector, get_xray_picking() )
        self.curve_y    = curve_y
        self.data       = intensity_array
        self.xray_slice = xray_slice
        self.affine_info    = affine_info
        self.parent     = parent
        self.basesurfaces = []
        self.ax2        = None
        self.ax3        = None
        self.very_small_angle_only  = False

    def draw_3d_scattering( self, ax, title, data, curve_y, zlim=None, zlim_expand=None ):

        ax.set_title( title )
        ax.set_xlabel( '\nQ(Å⁻¹)' )
        ax.set_ylabel( '\nsequential №' )
        ax.set_zlabel( '\nintensity' )
        # ax.zaxis._set_scale('log')

        if zlim is None:
            if zlim_expand is None:
                zlim_expand = ZLIM_EXPAND
            zmin    = np.min( self.curve_y )
            zmax    = np.max( self.curve_y )
            # zmin, zmax = ax.get_zlim()
            zmin_   = zlim_expand * zmin + ( 1-zlim_expand ) * zmax
            zmax_   = ( 1-zlim_expand ) * zmin + zlim_expand * zmax
            zlim    = ( zmin_, zmax_ )

        ax.set_zlim( zlim )

        print( 'data.shape=', data.shape )

        size = data.shape[0]

        peak_j = None
        for i, q in enumerate( self.qvector ):
            if not is_to_plot( self, i, q, very_small_angle_only=self.very_small_angle_only ):
                continue

            X = np.ones( size ) * q
            Y = np.arange( size )

            if i == self.index:
                alpha   = 1
                color   = 'orange'
                Z = curve_y
            else:
                alpha   = 0.2
                color   = '#1f77b4'
                Z = data[:,i,1]

            ax.plot( X, Y, Z, color=color, alpha=alpha )

    def draw_3d_for_gui( self, parent, title, do_correction, baseline_degree ):
        self.left_title  = title
        self.parent = parent
        from DebugCanvas            import DebugCanvas
        from molass_legacy.KekLib.TkUtils                import is_low_resolution

        def draw_func( fig ):
            # fig.set_size_inches( 20, 10 )
            gs  = gridspec.GridSpec( 1, 2 )
            ax1 = fig.add_subplot( gs[0,0], projection='3d' )
            # ax2 = fig.add_subplot( gs[0,1] )
            self.fig    = fig
            self.gs     = gs
            self.ax1    = ax1
            # self.ax2    = ax2
            self.draw_3d_scattering( ax1, title, self.data, self.curve_y )
            # self.draw_3d_scatterring_detail( ax1, ax2 )
            if do_correction:
                self.add_basesurface( self.data )
                self.basesurfaces[0].plot( self.ax1 )
            fig.tight_layout()

        figsize = ( 16, 8 ) if is_low_resolution() else ( 20, 10 )
        self.canvas = DebugCanvas( "Scattering Data 3D Plot", draw_func,
                        parent=parent, figsize=figsize )

        def connect_popup():
            # this function must be called after self.canvas.show
            self.create_popup_menu()
            # pick_event does not seem to work in this case
            self.canvas.mpl_canvas.mpl_connect( 'button_press_event', self.popup )

        self.parent.after( 100, connect_popup )
        self.do_correction  = do_correction
        self.baseline_degree = baseline_degree
        self.canvas.show( cunstom_button_cb=self.cunstom_button_box, cursor_update=False )

    def create_popup_menu( self ):
        self.popup_menu = Tk.Menu( self.canvas, tearoff=0 )
        self.popup_menu.add_command( label='Guinier Analysis',     command=self.guinier_analysis )
        self.popup_menu_count = 1

    def popup( self, event ):
        if event.button != 3:
            return

        self.popup_event = event
        # print( 'popup: x=%g, y=%g' % ( event.x, event.y ) )

        self.file_index = self.get_file_index( event )
        if self.file_index is None:
            return

        base_widget = self.canvas
        w, h, x, y = split_geometry( base_widget.mpl_canvas_widget.winfo_geometry() )
        h_ = h
        # print( 'self.mpl_canvas_widget.winfo_geometry()=', (w, h, x, y) )
        w, h, x, y = split_geometry( base_widget.winfo_geometry() )
        # print( 'self.winfo_geometry()=', (w, h, x, y) )
        # TODO: better way to get the mouse cursor position
        file_index_str = str(self.file_index)
        print( 'file_index_str=', file_index_str )
        self.popup_menu.entryconfig( 0, label="Guinier Analysis on y=" + file_index_str )
        if self.popup_menu_count == 2:
            self.popup_menu.entryconfig( 1, label="Guinier Analysis for both on y=" + file_index_str )
        self.popup_menu.post( int( x + event.x + 10 ), int( y + h_ - event.y + 30 ) )

    def get_file_index( self, event ):
        if event.ydata is None:
            return None

        # print( "guinier_analysis", (event.x, event.y) )
        # print( "guinier_analysis", (event.xdata, event.ydata) )
        """
            learned from stackoverflow article
            https://stackoverflow.com/questions/6748184/matplotlib-plot-surface-get-the-x-y-z-values-written-in-the-bottom-right-cor
        """

        y_value_re = re.compile( r'y=(-?\d+\.?\d*)' )

        ax = event.inaxes
        if ax == self.ax1:
            print( 'ax1' )
            self.current_data   = self.data
        elif ax == self.ax3:
            print( 'ax3' )
            self.current_data   = self.corrected_data
        else:
            print( 'ax2' )
            return None

        coord3d = ax.format_coord( event.xdata, event.ydata )
        print( 'coord3d=', coord3d )

        m   = y_value_re.search( coord3d )
        if not m:   return None

        file_index = int( float( m.group(1) ) )
        if file_index >= 0 and file_index < self.data.shape[0]:
            return file_index
        else:
            return None

    def cunstom_button_box( self, frame ):
        box1 = Tk.Frame( frame )
        box1.pack( fill=Tk.X, anchor=Tk.CENTER )
        self.button_frame = Tk.Frame( box1 )
        self.button_frame.grid( row=0, column=0 )
        box1.columnconfigure( 0, weight=1 )
        # box1.columnconfigure( 1, weight=1 )
        box2 = Tk.Frame( frame )
        box2.pack( fill=Tk.X )

        w = Tk.Button(box2, text="Close", width=10, command=self.canvas.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.RIGHT, padx=10, pady=10)
        self.ok_button = w

        self.cunstom_button_box_add()

        self.canvas.bind("<Return>", self.canvas.ok)

    def add_basesurface( self, data ):
        print( 'add_basesurface' )
        basesurface = ScatteringBasesurface( self.qvector, self.index, data,
                                inty_curve_y=self.curve_y,
                                parent=self.parent, baseline_degree=self.baseline_degree )

        self.basesurfaces.append( basesurface )

    def cunstom_button_box_add( self ):
        self.entire = Tk.IntVar()
        self.entire.set( 1 )
        cb = Tk.Checkbutton( self.button_frame, text="Entire Scattering", variable=self.entire, state=Tk.DISABLED )
        cb.grid( row=0, column=0, padx=5 )

        self.base = Tk.IntVar()
        self.base.set( 1 )
        cb = Tk.Checkbutton( self.button_frame, text="Base Surface", variable=self.base, state=Tk.DISABLED )
        cb.grid( row=0, column=1, padx=5 )

        self.vsa_only = Tk.IntVar()
        self.vsa_only.set( 0 )
        self.vsa_only_cb = Tk.Checkbutton( self.button_frame, text="Very Small Angle only", variable=self.vsa_only )
        self.vsa_only_cb.grid( row=0, column=2, padx=5 )

        space = Tk.Frame( self.button_frame, width=10 )
        space.grid( row=0, column=3, padx=10 )

        label = Tk.Label( self.button_frame, text="Zlim expand ratio:" )
        label.grid( row=0, column=4 )
        self.zlim_expand = Tk.DoubleVar()
        self.zlim_expand.set( 1.25 )
        entry = Tk.Entry( self.button_frame, textvariable=self.zlim_expand, width=6, justify=Tk.RIGHT )
        entry.grid( row=0, column=5 )

        space = Tk.Frame( self.button_frame, width=10 )
        space.grid( row=0, column=6, padx=10 )

        self.redraw = Tk.Button( self.button_frame, text="Redraw", command=self.redraw )
        self.redraw.grid( row=0, column=7, padx=10 )

        view_init = Tk.Button( self.button_frame, text="View Init", command=self.view_init )
        view_init.grid( row=0, column=8, padx=10 )

        self.change_to_corrected_3d()
        self.canvas.update()

    def redraw( self ):
        self.canvas.config( cursor='wait' )
        self.canvas.update()

        self.very_small_angle_only = self.vsa_only.get() == 1

        zlim_expand = self.zlim_expand.get()
        self.ax1.cla()
        self.draw_3d_scattering( self.ax1, self.left_title, self.data, self.curve_y, zlim_expand=zlim_expand )
        self.basesurfaces[0].plot( self.ax1, very_small_angle_only=self.very_small_angle_only )

        self.ax3.cla()
        self.draw_3d_scattering( self.ax3, self.right_title, self.corrected_data, self.corrected_curve_y, zlim_expand=zlim_expand )
        self.basesurfaces[1].plot( self.ax3, very_small_angle_only=self.very_small_angle_only )
        self.canvas.config( cursor='' )
        self.canvas.mpl_canvas.draw()

    def view_init( self ):
        for ax in [self.ax1, self.ax3]:
            ax.view_init()
        self.canvas.mpl_canvas.draw()

    def guinier_analysis( self, both=False ):
        in_folder   = get_setting( 'in_folder' )
        serial_data = self.parent.serial_data       # only for filename and serial_data.q_limit_index
        filename    = serial_data.datafiles[self.file_index].split( '\\' )[-1]
        print( 'filename=', filename )

        if both:
            data_list   = [ self.data, self.corrected_data ]
        else:
            data_list   = [ self.current_data ]

        result_array = []
        for data in data_list:
            np_array    = data[self.file_index,:,:]

            if data is self.data:
                filename_       = filename
                atsas_np_array  = None
            else:
                filename_       = filename + '(corrected)'
                atsas_np_array  = np_array
            result_gui  = ResultGui( self.canvas, in_folder, filename,
                                        np_array=np_array, serial_data=serial_data,
                                        title_filename=filename_,
                                        atsas_np_array=atsas_np_array )
            result_array.append( result_gui )

    def create_corrected_data( self ):
        jvector = np.arange( self.data.shape[0] )
        qvector = self.data[0,:,0]
        self.corrected_data = copy.deepcopy( self.data )

        print( 'create_corrected_data: baseline_degree=', self.baseline_degree )

        try:
            assert False
            # incompatible with revised apply_baseline_correction_impl
            apply_baseline_correction_impl( jvector, qvector, self.corrected_data,
                                                affine_info=self.affine_info,
                                                inty_curve_y=self.curve_y,
                                                baseline_opt=1,     # force this even in case of 'No correction'
                                                baseline_degree=self.baseline_degree,
                                                parent=self.parent
                                                )
        except RuntimeError as exc:
            # this is known to be harmless
            # because it seemts to occur at the (widest angle) end point.
            pass

        self.corrected_curve_y = np.average( self.corrected_data[:, self.xray_slice, 1], axis=1 )

    def change_to_corrected_3d( self ):
        self.canvas.config( cursor='wait' )
        self.canvas.update()

        self.create_corrected_data()

        if self.ax2 is not None:
            self.toolbar.destroy()
            self.change_button.pack_forget()

        if self.ax3 is None:
            self.ax3 = self.fig.add_subplot( self.gs[0,1], projection='3d' )

        self.right_title = "Corrected scattering made by subtracting the basesurface on the left figure"
        self.draw_3d_scattering( self.ax3, self.right_title, self.corrected_data, self.corrected_curve_y )

        if self.do_correction:
            self.add_basesurface( self.corrected_data )
            self.basesurfaces[1].plot( self.ax3 )

        self.canvas.mpl_canvas.draw()

        self.canvas.config( cursor='' )
        self.canvas.update()
        self.popup_menu.add_command( label='Guinier Analysis for both',     command=lambda: self.guinier_analysis( both=True ) )
        self.popup_menu_count = 2
