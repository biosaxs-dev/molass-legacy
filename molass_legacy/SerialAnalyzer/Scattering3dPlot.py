# coding: utf-8
"""
    Scattering3dPlot.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""
import os
import copy
import numpy                as np
from bisect                 import bisect_right
from mpl_toolkits.mplot3d   import Axes3D, proj3d
from molass_legacy.KekLib.BasicUtils             import get_caller_module, get_filename_extension
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.TkUtils                import is_low_resolution, split_geometry
from CanvasFrame            import CanvasFrame
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting, get_xray_picking
from ScatteringBaseCorrector    import ScatteringBaseCorrector
from molass_legacy.KekLib.TkCustomWidgets        import FolderEntry
from DevSettings            import get_dev_setting, set_dev_setting, ITEM_DEFAULTS as DEV_ITEM_DEFAULTS
from ScatteringViewUtils    import compute_baselines, draw_3d_scattering, ZLIM_EXPAND

class Scattering3dPlotFrame( Tk.Frame ):
    def __init__( self, parent, dialog, serial_data, mapper, data, title, i, sim_base=None ):
        # experimental = i == 1
        experimental = False
        self.plot_no        = i
        self.parent         = parent
        self.dialog         = dialog
        self.sd             = serial_data
        self.qvector        = serial_data.qvector
        self.jvector        = serial_data.jvector
        self.index          = bisect_right( self.qvector, get_xray_picking() )
        self.xray_curve_y   = serial_data.xray_curve.y
        self.xray_slice     = serial_data.xray_slice
        self.data           = data
        self.data_list      = [ data ]
        self.title          = title
        self.sim_base       = sim_base
        self.experimental   = experimental
        self.el_x   = np.ones( len(self.jvector) ) * self.qvector[self.index]
        self.el_y   = self.jvector
        self.el_z   = self.xray_curve_y
        self.suppress_on_draw   = False

        opt_params  = mapper.opt_params

        self.baseline_pair_list = []

        data_copy = copy.deepcopy( data )
        corrector = ScatteringBaseCorrector(
                                serial_data.jvector,
                                serial_data.qvector,
                                data_copy,
                                curve=serial_data.xray_curve,
                                affine_info=mapper.get_affine_info(),
                                inty_curve_y=self.xray_curve_y,
                                baseline_opt=opt_params['xray_baseline_opt'],
                                baseline_degree=opt_params['xray_baseline_const_opt'] + 1,
                                need_adjustment=opt_params['xray_baseline_adjust'] == 1,
                                parent=self )
        self.baseline_pair_list.append( compute_baselines( self.qvector, self.index, corrector ) )
        self.corrected_data = data_copy

        Tk.Frame.__init__( self, parent )

        self.body( self )
        self.buttonbox()
        self.draw()
        self.create_popup_menu()

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

        # fig.tight_layout()
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

        if False:
            show_save = Tk.Button( frame, text="Save Dialog", command=self.show_save_dialog )
            show_save.grid( row=0, column=6, padx=10 )

    def create_popup_menu( self ):
        self.popup_menu = Tk.Menu( self, tearoff=0 )
        self.popup_menu.add_command( label='Guinier analysis animation', command=lambda: self.guinier_analysis() )
        self.popup_menu.add_command( label='Guinier analysis animation Corrected', command=lambda: self.guinier_analysis(corrected=True) )
        self.popup_menu.add_command( label='Rg comparison (GuinierDiffViewer)', command=self.show_guinier_viewer )

        # self.draw()
    def draw( self ):
        try:
            zlim_expand = self.zlim_expand.get()
        except:
            zlim_expand = None

        n = 0
        ax  = self.axes[n]
        data    = self.data_list[n]
        vsa_base_list, all_base_list    = self.baseline_pair_list[n]
        very_small_angle_only = self.vsa_only.get() == 1

        draw_3d_scattering( ax, data, self.qvector, self.index, self.xray_curve_y,
                            self.title,
                            vsa_base_list, all_base_list,
                            zlim_expand=zlim_expand, experimental=self.experimental,
                            very_small_angle_only=very_small_angle_only,
                            sim_base=self.sim_base,
                            )

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
        self.popup_menu.entryconfig( 0, label="Guinier analysis animation on %d-th elution" % elution_no )
        self.popup_menu.entryconfig( 1, label="Guinier analysis animation on %d-th elution corrected" % elution_no )
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

    def show_guinier_viewer( self ):
        self.config( cursor='wait' )
        self.update()

        from ScatteringBaseUtil     import apply_baseline_correction_impl
        from GuinierDiffViewer      import GuinierDiffViewer

        sd  = self.sd

        usable_slice = sd.get_usable_slice()
        mapper = self.dialog.mapper
        # data1 = self.dialog.data_list[0]
        data1 = self.data
        data2 = copy.deepcopy( data1 )

        mapped_info = mapper.get_mapped_info()
        apply_baseline_correction_impl(
                sd.jvector, sd.qvector, data2,
                mapped_info,
                ecuve=sd.xray_curve,
                )

        self.config( cursor='' )
        viewer = GuinierDiffViewer( self.dialog, sd.qvector, sd.ivector, data1, data2, self.elution_no, usable_slice=usable_slice )
        self.dialog.update()

    def show_save_dialog( self ):
        dialog = ScatteringSaveDialog( self.dialog )
        dialog.show()

class ScatteringSaveDialog( Dialog ):
    def __init__( self, parent ):
        self.parent = parent
        self.caller_module = get_caller_module( level=2 )

    def show( self ):
        title = "Absorbance Data Save"
        Dialog.__init__( self, self.parent, title, auto_geometry=False, geometry_cb=self.adjust_geometry )

    def adjust_geometry( self ):
        w, h, x, y = split_geometry( self.parent.geometry() )
        self.geometry("+%d+%d" % (self.parent.winfo_rootx() + w//2,
                                  self.parent.winfo_rooty() + int(h*0.7) ))

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        label_frame = Tk.Frame(body_frame)
        label_frame.pack( padx=50, pady=10 )

        guide = Tk.Label(label_frame, text='Select what and where (file name) to save and press "OK"')
        guide.pack()

        detail_frame = Tk.Frame(body_frame)
        detail_frame.pack( padx=50, pady=10 )

        grid_row = 0
        self.xb_save = Tk.IntVar()

        cb = Tk.Checkbutton( detail_frame, text="enable X-ray Scattering Baseline Save",
                                variable=self.xb_save  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W   )

        self.xb_folder = Tk.StringVar()
        folder_path = os.path.join( get_setting( 'analysis_folder' ), 'xray_scattering_base' ).replace( '\\', '/' )
        self.xb_folder.set( folder_path )
        folder_entry = FolderEntry( detail_frame, textvariable=self.xb_folder, width=70,
                                            on_entry_cb=self.on_entry_xb_folder )
        folder_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        grid_row += 1
        self.filename_example   = os.path.split( self.parent.sd.datafiles[0] )[-1]
        self.filename_extention = '.' + get_filename_extension( self.filename_example )

        postfix_frame = Tk.Frame( detail_frame )
        postfix_frame.grid( row=grid_row, column=1, sticky=Tk.W )
        postfix_label = Tk.Label( postfix_frame, text="filename postfix" )
        postfix_label.grid( row=0, column=0 )
        self.base_file_postfix = Tk.StringVar()
        self.base_file_postfix.set( "_base" )

        self.base_file_postfix_entry = Tk.Entry( postfix_frame, textvariable=self.base_file_postfix, width=10 )
        self.base_file_postfix_entry.grid( row=0, column=1, padx=5 )

        as_in_label = Tk.Label( postfix_frame, text="as in" )
        as_in_label.grid( row=0, column=2 )

        self.postfix_eg = Tk.StringVar()
        self.postfix_eg_update()
        eg_label = Tk.Label( postfix_frame, textvariable=self.postfix_eg )
        eg_label.grid( row=0, column=3, padx=3 )

    def on_entry_xb_folder( self ):
        pass

    def postfix_eg_update( self ):
        filename = self.filename_example.replace( self.filename_extention, self.base_file_postfix.get() + self.filename_extention )
        self.postfix_eg.set( filename )

    def validate( self ):
        if self.xb_save.get() == 1:
            self.save_xray_scattering_base()
        return 1

    def save_xray_scattering_base( self ):
        for file in self.parent.sd.datafiles:
            d, f = os.path.split( file )
            print( f )

class Scattering3dPlotDialog( Dialog ):
    def __init__( self, parent, serial_data, mapper, data_list=None, title_list=None, sim_base=None ):
        self.parent = parent
        self.sd     = serial_data
        self.mapper = mapper
        if data_list is None:
            data_list   = [ serial_data.intensity_array ]
        self.data_list  = data_list
        if title_list is None:
            title_list  = []

            title = 'Xray Scattering from ' + get_setting( 'in_folder' )
            for n in range( len(self.data_list) ):
                sim_title = '' if n == 0 else 'Residual of '
                title_list.append( sim_title + title )

        self.title_list = title_list
        self.sim_base   = sim_base
        self.applied    = False

    def show( self ):
        title = "ScatteringThreeDimPlot"
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
            # plot3d_frame = Scattering3dPlotFrame( plot_frame, self, self.sd, self.mapper, data, self.title_list[i], i, sim_base=self.sim_base )
            plot3d_frame = Scattering3dPlotFrame( plot_frame, self, self.sd, self.mapper, data, self.title_list[i], i )
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
