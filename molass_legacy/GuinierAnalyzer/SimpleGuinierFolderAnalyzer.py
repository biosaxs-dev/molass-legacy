# coding: utf-8
"""
    SimpleGuinierFolderAnalyzer.py

    Copyright (c) 2017-2019, SAXS Team, KEK-PF
"""
import os
import re
import numpy as np
import glob
from molass_legacy.KekLib.NumpyUtils                 import np_loadtxt
from molass_legacy.KekLib.OurTkinter                 import Tk, Font, is_empty_val
from molass_legacy.KekLib.TkUtils                    import adjusted_geometry, split_geometry
from molass_legacy.KekLib.TkCustomWidgets            import FileEntry, FolderEntry
from CanvasFrame                import CanvasFrame
from molass_legacy.KekLib.OurMatplotlib              import NavigationToolbar
from CanvasDialog               import CanvasDialog
from DataUtils                  import get_pytools_folder
from SerialData                 import SerialData, find_conc_files
from molass_legacy._MOLASS.SerialSettings             import get_setting
from SimpleGuinierGuiUtils      import window_title, CheckbuttonFrame, TypeInfoPanel

NUM_EXTRAS  = 10
USE_SAVED_DATA  = True
if USE_SAVED_DATA:
    from SerialDataLoader       import SerialDataLoader

class SimpleGuinierFolderAnalyzer( Tk.Toplevel ):
    def __init__( self, parent, folder_no, folder_path, folder_info, x, z, r, type_names, report_folder=None ):
        self.parent = parent
        self.folder_no = folder_no
        self.folder_path = folder_path
        self.folder_info = folder_info
        if report_folder is None:
            report_folder = self.parent.report_folder.get() + '/analysis-%03d' % folder_no
        self.report_folder = report_folder
        self.num_zx_results = 0
        for left, peak, right in folder_info:
            self.num_zx_results += right - left + 2 + NUM_EXTRAS    # left-peak, peak-right
        self.num_sg_results = len(x) - self.num_zx_results
        if USE_SAVED_DATA:
            self.loader = SerialDataLoader()
            in_folder = report_folder + '/averaged'
            self.loader.load_xray_data_in_another_thread( in_folder )
        else:
            conc_folder, conc_file = self.get_conc_file( folder_path )
            self.serial_data = SerialData( conc_folder, folder_path, conc_file=conc_file )
        self.averaged_intensity_array = None
        self.init_x  = x
        self.init_z  = z
        self.init_r  = r
        self.type_names = type_names
        temp_dict = {}
        for v in z:
            temp_dict[v] = 1
        self.types = sorted( temp_dict.keys() )
        Tk.Toplevel.__init__( self, parent )
        self.title( window_title )

        frame1 = Tk.Frame( self )
        frame1.pack()

        self.canvas = CanvasFrame( frame1, figsize=(12, 8) )
        self.canvas.pack( side=Tk.LEFT )

        self.toolbar = NavigationToolbar( self.canvas.mpl_canvas, self.canvas )
        self.toolbar.update()
        self.create_popup_menu()

        frame12 = Tk.Frame( frame1 )
        frame12.pack( side=Tk.RIGHT, fill=Tk.BOTH  )
        label = Tk.Label( frame12, text='Control Panel' )
        label.pack( anchor=Tk.N )

        self.cb_frame = CheckbuttonFrame( frame12, types=self.types )
        self.cb_frame.pack()
        self.update_xzr( force=True )

        redraw_btn   = Tk.Button( frame12, text='redraw', command=self.redraw )
        redraw_btn.pack()

        typeinfo_btn   = Tk.Button( frame12, text='type help', command=self.show_type_info )
        typeinfo_btn.pack()

        if False:
            print( 'len(x)=', len(x) )
            print( 'x=', x )
            print( 'num_sg_results=', self.num_sg_results )
            print( 'folder_info=', self.folder_info )

        self.range_indicators = []
        for left, peak, right in self.folder_info:
            self.range_indicators.append( left )
            self.range_indicators.append( peak )
            self.range_indicators.append( right )

        zx_base = self.num_sg_results + 1       # there is one space before extrapolation starts
        for left, peak, right in self.folder_info:
            self.range_indicators.append( zx_base )
            zx_peak = zx_base + ( peak - left ) + 5
            self.range_indicators.append( zx_peak )
            zx_base = zx_peak + 2
            self.range_indicators.append( zx_base )
            zx_right = zx_base + ( right - peak ) + 5 
            self.range_indicators.append( zx_right )
            zx_base = zx_right + 2

    def get_conc_file( self, folder_path ):
        datafile_recs  = list( find_conc_files( folder_path ) )
        if len(datafile_recs) > 0:
            conc_file = datafile_recs[0][1]
            conc_folder = folder_path
        else:
            uv_folder_path = folder_path.replace( 'Backsub', 'spectra' )
            datafile_recs  = list( find_conc_files( uv_folder_path ) )
            if len(datafile_recs) > 0:
                conc_file = datafile_recs[0][1]
                conc_folder = folder_path
            else:
                # TODO:
                assert False
        return conc_folder, conc_file

    def show( self ):
        self.canvas.draw( self.draw )
        self.geometry( adjusted_geometry(self.geometry()) )
        self.fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )

    def create_popup_menu( self ):
        self.popup_menu = Tk.Menu( self, tearoff=0 )
        self.popup_menu.add_command( label='Animation', command=self.do_animation )

    def update_xzr( self, force=False ):
        if force or self.cb_frame.changed:
            index   = self.cb_frame.make_index( self.init_z )
            self.x  = self.init_x[index]
            self.z  = self.init_z[index]
            self.r  = self.init_r[index]

    def draw( self, fig ):
        self.fig = fig
        ax = fig.add_subplot( 111 )
        self.axes = [ax]
        self.plot( ax, get_lim=True )
        fig.tight_layout()

    def plot( self, ax, draw_flags=None, get_lim=False ):
        folderno = '' if self.folder_no < 0 else 'Foder No: %03d      ' % self.folder_no
        ax.set_title( folderno + 'Path: %s' % ( self.folder_path ) )
        ax.set_xlabel( 'File No' )
        ax.set_ylabel( 'Rg' )

        for i, n in enumerate(self.types):
            if draw_flags is not None:
                if draw_flags[i] == 0:
                    continue

            index = self.z == n
            x_ = self.x[index]
            if len( x_ ) == 0:
                continue
            r_ = self.r[index]
            ax.plot( x_, r_, 'o', markersize=3, label='type %d' % n )

        if get_lim:
            self.xlim = np.array( ax.get_xlim() )   # deep copy
            self.ylim = np.array( ax.get_ylim() )   # deep copy

        ax.set_xlim( self.xlim )
        ax.set_ylim( self.ylim )

        for x in self.range_indicators:
            ax.plot( [ x, x ], self.ylim, color='green', linewidth=1, alpha=0.2 )

        ax.legend()

    def redraw( self ):
        self.update_xzr()

        ax = self.axes[0]
        ax.cla()
        # draw_flags = [ var.get() for var in self.cb_frame.cb_vars ]
        def redraw_closure( fig ):
            self.plot( ax )
            ax.set_xlim( self.xlim )
            ax.set_ylim( self.ylim )

        self.canvas.draw( redraw_closure )

    def on_button_press( self, event ):
        if event.button == 3:
            self.on_right_button_press( event )
            return

    def on_right_button_press( self, event ):
        print( 'on_right_button_press' )

        if event.inaxes not in self.axes:
            return

        # https://stackoverflow.com/questions/17711099/programmatically-change-matplotlib-toolbar-mode-in-qt4
        if self.toolbar._active == 'ZOOM':
            # toggle zoom mode
            self.toolbar.zoom()

        print( 'x=', event.xdata )
        print( 'y=', event.ydata )

        dist = ( self.x - event.xdata )**2 + ( self.r - event.ydata )**2
        m = np.argmin( dist )
        self.file_index = m

        w, h, x, y = split_geometry( self.canvas.mpl_canvas_widget.winfo_geometry() )
        h_ = h
        # print( 'self.mpl_canvas_widget.winfo_geometry()=', (w, h, x, y) )
        w, h, x, y = split_geometry( self.winfo_geometry() )
        # print( 'self.winfo_geometry()=', (w, h, x, y) )
        # TODO: better way to get the mouse cursor position
        self.popup_menu.entryconfig( 0, label="Animation for the %d-th curve" % self.file_index )
        # can't post(event.x_root, event.y_root) for matplotlib events
        self.popup_menu.post( int( x + event.x + 10 ), int( y + h_ - event.y + 30 ) )

    def update_cursor( self, cursor ):
        for w in [self.canvas.mpl_canvas_widget]:
            w.config( cursor=cursor )

    def do_animation( self ):
        i = self.file_index
        print( 'do_animation', i )
        self.update_cursor( 'wait' )

        from molass_legacy.KekLib.NumpyUtils             import np_loadtxt
        from SimpleGuinier          import SimpleGuinier
        from molass_legacy.AutorgKekAdapter       import AutorgKekAdapter
        from SimpleGuinierAnimation import SimpleGuinierAnimation
        from CanvasDialog           import CanvasDialog

        if self.averaged_intensity_array is None:
            if USE_SAVED_DATA:
                self.loader.wait_for_xray_loading()
                self.averaged_intensity_array = self.loader.xray_array
                self.datafiles = self.loader.datafiles
            else:
                self.serial_data.wait_until_ready()
                self.serial_data.apply_data_reduction()
                num_curves_averaged = get_setting( 'num_curves_averaged' )
                self.averaged_intensity_array, _, _ = self.serial_data.get_averaged_data( num_curves_averaged )
                self.datafiles  = self.serial_data.datafiles

        if USE_SAVED_DATA:
            averaged_text = ''
        else:
            averaged_text = ' (averaged)'

        print( 'self.num_sg_results=', self.num_sg_results )
        print( 'len( self.averaged_intensity_array )=', len( self.averaged_intensity_array ) )
        assert len( self.averaged_intensity_array ) == self.num_sg_results
        if i < self.num_sg_results:
            data = self.averaged_intensity_array[i, : ]
            file = self.datafiles[i]
            file_for_title = file.replace('\\', '/').split('/')[-1] + averaged_text
        else:
            data, file = self.load_zx_data( i - self.num_sg_results )
            file_for_title = '/'.join( file.replace('\\', '/').split('/')[-2:] )
        guinier = SimpleGuinier( data, anim_data=True )
        adapter = AutorgKekAdapter( None, guinier=guinier )
        adapter_result = adapter.run()

        dialog = CanvasDialog( "SimpleGuinier Animation", parent=self )
        anim_iter_max = 300

        def anim_func( fig  ):
            anim = SimpleGuinierAnimation( guinier, title=file_for_title, result_rg=adapter_result.Rg  )
            anim.draw( fig, anim_iter_max=anim_iter_max )
            dialog.mpl_canvas.draw()

        self.update_cursor( '' )

        dialog.show( anim_func, figsize=( 18, 5 ) )

    def load_zx_data( self, k ):
        j = -1
        k_for_name = k
        found = False
        single_peak = len(self.folder_info) == 1
        half_num = NUM_EXTRAS//2
        for p, tuple_ in enumerate(self.folder_info):
            left, peak, right = tuple_
            left_size = peak - left + half_num + 1 
            for _ in range( left_size ):
                j += 1
                if j == k:
                    found = True
                    ad = 'asc' if single_peak else 'asc(%d)' % (p+1)
                    break
            if found:
                break
            k_for_name -= left_size
            right_size = right - peak + half_num + 1
            for _ in range( right_size ):
                j += 1
                if j == k:
                    found = True
                    ad = 'desc' if single_peak else 'desc(%d)' % (p+1)
                    k -= left_size
                    break
            if found:
                break
            k_for_name -= right_size
        if not found:
            return None, None

        folder = self.report_folder + '/extrapolated/%s' % ( ad )
        if not os.path.exists( folder ):
            return  None, None

        folder_glob_text = folder.replace( '/', '\\' ) + '\\extrapolated-%03d*.*' %  k_for_name
        print( 'folder_glob_text=', folder_glob_text )
        files = glob.glob( folder_glob_text )
        file_path = files[0]
        data, _ = np_loadtxt( file_path )
        return data, file_path

    def show_type_info( self ):
        dialog = TypeInfoPanel( self )
        dialog.show( self.type_names )
        