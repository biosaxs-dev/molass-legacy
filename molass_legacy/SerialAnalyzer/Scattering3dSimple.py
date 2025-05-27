# coding: utf-8
"""
    Scattering3dSimple.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import copy
import numpy                as np
from bisect                 import bisect_right
from mpl_toolkits.mplot3d   import Axes3D, proj3d
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.TkUtils                import is_low_resolution, split_geometry
from CanvasFrame            import CanvasFrame
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting, get_xray_picking

VERY_SMALL_ANGLE_LIMIT  = 0.03
NUM_SAMPLE_CURVES       = 20
ZLIM_EXPAND             = 3.0

def is_to_plot( index, i, q, very_small_angle_only=False ):
    # return q >= 0.01 and ( i == self.index or i % 8 == 0 )
    if very_small_angle_only:
        return q < VERY_SMALL_ANGLE_LIMIT or i == index
    else:
        return i == index or i % 8 == 0

class Scattering3dSimpleFrame( Tk.Frame ):
    def __init__( self, parent, dialog, serial_data, data, title, i ):
        experimental = i == 1
        self.plot_no        = i
        self.parent         = parent
        self.dialog         = dialog
        self.qvector        = serial_data.qvector
        self.jvector        = serial_data.jvector
        self.index          = bisect_right( self.qvector, get_xray_picking() )
        self.xray_curve_y   = serial_data.xray_curve.y
        self.xray_slice     = serial_data.xray_slice
        self.data           = data
        self.data_list      = [ data ]
        self.title          = title
        self.el_x   = np.ones( len(self.jvector) ) * self.qvector[self.index]
        self.el_y   = self.jvector
        self.el_z   = self.xray_curve_y
        self.suppress_on_draw   = False

        Tk.Frame.__init__( self, parent )

        self.body( self )
        self.buttonbox()
        self.draw()
        self.create_popup_menu()

    def compute_baselines( self, corrector ):

        vsa_base_list = []
        all_base_list = []
        for i, q  in enumerate( self.qvector ):
            i_select = ( i == self.index or i % 8 == 0 )
            if q < VERY_SMALL_ANGLE_LIMIT or i_select:
                baseline = corrector.correct_a_single_q_plane( i, return_baseline=True )
                if q < VERY_SMALL_ANGLE_LIMIT or i == self.index:
                    vsa_base_list.append( baseline )
                if i_select:
                    all_base_list.append( baseline )

        return vsa_base_list, all_base_list

    def body( self, body_frame ):   # overrides parent class method

        base_frame = Tk.Frame( body_frame );
        # base_frame.pack( expand=1, fill=Tk.BOTH, padx=20, pady=10 )
        base_frame.pack( expand=1, fill=Tk.BOTH )

        figsize = ( 8, 8 ) if is_low_resolution() else ( 10, 10 )

        self.canvas_frame = canvas_frame = CanvasFrame( base_frame, figsize=figsize )
        canvas_frame.pack()
        fig = self.canvas_frame.fig
        ax1 = fig.add_subplot( 111, projection='3d' )
        self.axes = [ ax1 ]

        fig.tight_layout()
        self.canvas_frame.fig.canvas.mpl_connect( 'draw_event', self.on_draw )
        self.canvas_frame.fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        frame = self

        box1 = Tk.Frame( frame )
        box1.pack( fill=Tk.X, anchor=Tk.CENTER )
        self.button_frame = Tk.Frame( box1 )
        self.button_frame.grid( row=0, column=0 )
        box1.columnconfigure( 0, weight=1 )
        # box1.columnconfigure( 1, weight=1 )

        self.cunstom_button_box_add( self.button_frame )

    def cunstom_button_box_add( self, frame ):
        self.vsa_only = Tk.IntVar()
        self.vsa_only.set( 1 )
        self.vsa_only_cb = Tk.Checkbutton( frame, text="Very Small Angle only", variable=self.vsa_only )
        self.vsa_only_cb.grid( row=0, column=0, padx=5 )

        space = Tk.Frame( frame, width=10 )
        space.grid( row=0, column=1, padx=10 )

        label = Tk.Label( frame, text="Zlim expand ratio:" )
        label.grid( row=0, column=2 )
        self.zlim_expand = Tk.DoubleVar()
        self.zlim_expand.set( ZLIM_EXPAND )
        entry = Tk.Entry( frame, textvariable=self.zlim_expand, width=6, justify=Tk.RIGHT )
        entry.grid( row=0, column=3 )

        space = Tk.Frame( frame, width=10 )
        space.grid( row=0, column=6, padx=10 )

        self.redraw_btn = Tk.Button( frame, text="Redraw", command=self.redraw )
        self.redraw_btn.grid( row=0, column=4, padx=10 )

        view_init = Tk.Button( frame, text="View Init", command=self.view_init )
        view_init.grid( row=0, column=5, padx=10 )

    def create_popup_menu( self ):
        self.popup_menu = Tk.Menu( self, tearoff=0 )
        self.popup_menu.add_command( label='Guinier Analysis', command=lambda: self.guinier_analysis() )
        self.popup_menu.add_command( label='Guinier Analysis Corrected', command=lambda: self.guinier_analysis(corrected=True) )

        # self.draw()
    def draw( self ):
        try:
            zlim_expand = self.zlim_expand.get()
        except:
            zlim_expand = None

        self.draw_3d_scattering( 0, self.title, zlim_expand=zlim_expand )
        self.canvas_frame.show()

    def redraw( self ):
        for ax in self.axes:
            ax.cla()
        self.draw()
        self.canvas_frame.show()

    def view_init( self ):
        for ax in self.axes:
            ax.view_init()
        self.canvas_frame.show()

    def draw_3d_scattering( self, n, title, zlim=None, zlim_expand=None ):
        ax      = self.axes[n]
        data    = self.data_list[n]

        very_small_angle_only = self.vsa_only.get() == 1

        ax.set_title( title )
        ax.set_xlabel( '\nQ(Å⁻¹)' )
        ax.set_ylabel( '\nElution №' )
        ax.set_zlabel( '\nIntensity' )
        # ax.zaxis._set_scale('log')

        if zlim is None:
            if zlim_expand is None:
                zlim_expand = ZLIM_EXPAND
            zmin    = np.min( self.xray_curve_y )
            zmax    = np.max( self.xray_curve_y )
            # zmin, zmax = ax.get_zlim()
            zmin_   = zlim_expand * zmin + ( 1-zlim_expand ) * zmax
            zmax_   = ( 1-zlim_expand ) * zmin + zlim_expand * zmax
            zlim    = ( zmin_, zmax_ )

        ax.set_zlim( zlim )

        print( 'data.shape=', data.shape )

        size = data.shape[0]
        Y = np.arange( size )

        peak_j = None
        i_ = -1
        i_list = []
        for i, q in enumerate( self.qvector ):
            if not is_to_plot( self.index, i, q, very_small_angle_only=very_small_angle_only ):
                continue

            i_list.append( i )

            X = np.ones( size ) * q

            if i == self.index:
                alpha   = 1
                color   = 'orange'
                Z = self.xray_curve_y
            else:
                alpha   = 0.2
                color   = '#1f77b4'
                Z = data[:,i,1]

            ax.plot( X, Y, Z, color=color, alpha=alpha )

            i_ += 1

            if False:
                if very_small_angle_only:
                    baseline = vsa_base_list[i_]
                else:
                    baseline = all_base_list[i_]
                ax.plot( X, Y, baseline, color='red', alpha=0.2 )

        return

    def on_draw( self, event ):
        # print( 'on_draw' )
        if self.dialog.sync_view.get() == 0:
            return

        if self.suppress_on_draw:
            return

        this_ax = self.axes[0]
        other_frame = self.dialog.plot_frames[1-self.plot_no]
        # https://stackoverflow.com/questions/47610614/get-viewing-camera-angles-in-matplotlib-3d-plot
        other_frame.suppress_on_draw = True

        zlim_expand_ = self.zlim_expand.get()
        if other_frame.zlim_expand.get() != zlim_expand_:
            other_frame.zlim_expand.set(zlim_expand_)
            other_frame.redraw()

        other_frame.axes[0].view_init( this_ax.elev, this_ax.azim )
        other_frame.canvas_frame.show()
        other_frame.update()

        other_frame.suppress_on_draw = False

    def compute_projected_xy( self ):
        ax = self.axes[0]
        # print( ax.get_proj() )
        projection = ax.get_proj()
        x, y, _ = proj3d.proj_transform( self.el_x, self.el_y, self.el_z, projection )
        xy = np.vstack( [x, y] ).T
        ret = ax.transData.transform( xy )
        print( 'ret.shape=', ret.shape )
        self.x_scr = ret[:,0]
        self.y_scr = ret[:,1]
        if False:
            for i in range(5):
                print( 'point[%d]=' % i, ( self.x[i], self.y[i], self.r[i] ) )
                # print( 'transformed point[%d]=' % i, (x[i], y[i]) )
                print( 'transformed screen point[%d]=' % i, (self.x_scr[i], self.y_scr[i]) )

    def on_button_press( self, event ):
        if event.button == 3:
            self.on_right_button_press( event )
            return

    def on_right_button_press( self, event ):
        print( 'on_right_button_press' )

        if event.inaxes not in self.axes:
            return

        self.compute_projected_xy()

        print( 'x=', event.x )
        print( 'y=', event.y )
        w, h, x, y = split_geometry( self.canvas_frame.mpl_canvas_widget.winfo_geometry() )
        h_ = h
        print( 'self.mpl_canvas_widget.winfo_geometry()=', (w, h, x, y) )

        w, h, x, y = split_geometry( self.dialog.winfo_geometry() )
        # print( 'self.winfo_geometry()=', (w, h, x, y) )
        # TODO: better way to get the mouse cursor position

        elution_no = self.guess_pointed_elution_no( event )
        if elution_no is None:
            return

        self.elution_no = elution_no
        x_  = x if self.plot_no == 0 else x + w//2

        # self.folder_no.set( folder_no )
        self.popup_menu.entryconfig( 0, label="Guinier analysis on %d-th elution" % elution_no )
        self.popup_menu.entryconfig( 1, label="Guinier analysis on %d-th elution corrected" % elution_no )
        self.popup_menu.post( int( x_ + event.x + 10 ), int( y + h_ - event.y + 50 ) )

    def guess_pointed_elution_no( self, event ):
        dist = ( self.x_scr - event.x )**2 + ( self.y_scr - event.y )**2
        i = np.argmin( dist )
        return i

    def update_cursor( self, cursor ):
        for w in [self.canvas_frame.mpl_canvas_widget]:
            w.config( cursor=cursor )

    def guinier_analysis( self, elution_no=None, corrected=False ):
        if elution_no is None:
            elution_no = self.elution_no

        print( 'guinier_analysis: ', elution_no )
        self.update_cursor( 'wait' )

        from SimpleGuinier          import SimpleGuinier
        from molass_legacy.AutorgKekAdapter       import AutorgKekAdapter
        from SimpleGuinierAnimation import SimpleGuinierAnimation
        from CanvasDialog           import CanvasDialog

        if corrected:
            data    = self.corrected_data
        else:
            data    = self.data

        s_data  = data[elution_no,]
        guinier = SimpleGuinier( s_data, anim_data=True )
        adapter = AutorgKekAdapter( None, guinier=guinier )
        adapter_result = adapter.run()

        dialog = CanvasDialog( "SimpleGuinier Animation", parent=self )
        anim_iter_max = 300
        title = 'Guinier analysis of the ' + str(elution_no) + '-th elution'

        def anim_func( fig  ):
            anim = SimpleGuinierAnimation( guinier, title=title, result_rg=adapter_result.Rg  )
            anim.draw( fig, anim_iter_max=anim_iter_max )
            dialog.mpl_canvas.draw()

        self.update_cursor( '' )

        dialog.show( anim_func, figsize=( 18, 5 ) )

class Scattering3dSimpleDialog( Dialog ):
    def __init__( self, parent, serial_data, data_list=None, title_list=None ):
        self.parent = parent
        self.sd     = serial_data
        if data_list is None:
            data_list   = [ serial_data.intensity_array ]
        self.data_list  = data_list
        if title_list is None:
            title_list  = []

            title = 'Xray Scattering from ' + get_setting( 'in_folder' )
            for n in range( len(self.data_list) ):
                sim_title = '' if n == 0 else 'Guinier-Porod-fitted '
                title_list.append( sim_title + title )

        self.title_list = title_list
        self.applied    = False

    def show( self ):
        title = "Scattering3dSimplePlot"
        Dialog.__init__(self, self.parent, title )

    def body( self, body_frame ):
        tk_set_icon_portable( self )

        plot_frame = Tk.Frame( body_frame )
        plot_frame.pack()

        self.sync_view = Tk.IntVar()
        self.sync_view.set(0)

        self.plot_frames = []

        for i, data in enumerate(self.data_list):
            # add ", sim_base=self.sim_base" when plotting simulated base
            plot3d_frame = Scattering3dSimpleFrame( plot_frame, self, self.sd, data, self.title_list[i], i )
            plot3d_frame.grid( row=0, column=i )
            self.plot_frames.append( plot3d_frame )

        if len(self.plot_frames) > 1:
            self.sync_view.set(1)

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        frame = self

        box2 = Tk.Frame( frame )
        box2.pack( fill=Tk.X, expand=1 )

        frames = []
        for k in range(3):
            f = Tk.Frame( box2 )
            f.pack( side=Tk.LEFT, fill=Tk.X, expand=1 )
            frames.append( f )

        w = Tk.Button(frames[2], text="Cancel", width=10, command=self.cancel, default=Tk.ACTIVE)
        w.pack(side=Tk.RIGHT, padx=10, pady=10)
        self.cancel_button = w

        w = Tk.Button(frames[2], text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.RIGHT, padx=10, pady=10)
        self.ok_button = w

        self.apply_fitted_base = Tk.IntVar()
        self.apply_fitted_base.set(0)

        if len(self.plot_frames) > 1:
            w = Tk.Checkbutton(frames[2], text="Apply Fitted Correction", variable=self.apply_fitted_base )
            w.pack(side=Tk.RIGHT, padx=10, pady=10)

            cb = Tk.Checkbutton( frames[1], text="Synchronize view", variable=self.sync_view  )
            cb.pack( side=Tk.RIGHT, padx=20 )

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def apply( self ):  # overrides parent class method
        self.applied        = True
