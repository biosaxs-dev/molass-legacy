# coding: utf-8
"""
    SimpleGuinierMain.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
import re
from mpl_toolkits.mplot3d       import proj3d
from molass_legacy.KekLib.OurTkinter                 import Tk
from molass_legacy.KekLib.TkUtils                    import adjusted_geometry, split_geometry
from molass_legacy.KekLib.TkCustomWidgets            import FileEntry, FolderEntry
from SimpleGuinierAnalyzer      import SimpleGuinierAnalyzer
from CanvasFrame                import CanvasFrame
from molass_legacy.KekLib.OurMatplotlib              import NavigationToolbar
from CanvasDialog               import CanvasDialog
from DataUtils                  import get_pytools_folder
from SerialData                 import SerialData, find_conc_files
from molass_legacy._MOLASS.SerialSettings             import get_setting
from SimpleGuinierGuiUtils      import window_title, CheckbuttonFrame, TypeInfoPanel, FolderInfoPanel
from SimpleGuinierFolderAnalyzer    import SimpleGuinierFolderAnalyzer

class SimpleGuinierMain( Tk.Toplevel ):
    def __init__( self, parent, tester_log=None ):
        self.parent = parent
        Tk.Toplevel.__init__( self, parent )
        self.withdraw()
        self.title( window_title )

        frame0 = Tk.Frame( self )
        frame0.pack()
        frame01 = Tk.Frame( frame0 )
        frame01.pack( side=Tk.LEFT, padx=10 )
        frame02 = Tk.Frame( frame0 )
        frame02.pack( side=Tk.LEFT, padx=10 )
        frame03 = Tk.Frame( frame0 )
        frame03.pack( side=Tk.LEFT, padx=10 )

        label = Tk.Label( frame01, text='Tester Log: ' )
        label.pack( side=Tk.LEFT )

        self.tester_log = Tk.StringVar()
        entry   = FileEntry( frame01, textvariable=self.tester_log, width=50, on_entry_cb=self.on_entry_tester_log )
        entry.pack( side=Tk.LEFT )

        label = Tk.Label( frame02, text='Data Folder: ' )
        label.pack( side=Tk.LEFT )

        self.pytools_folder = get_pytools_folder()
        self.data_folder = Tk.StringVar()
        self.data_folder.set( self.pytools_folder + '/Data' )
        entry   = FolderEntry( frame02, textvariable=self.data_folder, width=50 )
        entry.pack( side=Tk.LEFT )

        label = Tk.Label( frame03, text='Report Folder: ' )
        label.pack( side=Tk.LEFT )

        self.report_folder = Tk.StringVar()
        self.report_folder.set( self.pytools_folder + '/TODO/guinier-results/reports-1_1_5' )
        entry   = FolderEntry( frame03, textvariable=self.report_folder, width=50 )
        entry.pack( side=Tk.LEFT )

        frame1 = Tk.Frame( self )
        frame1.pack()
        self.canvas = CanvasFrame( frame1, figsize=(12, 8) )
        self.canvas.pack( side=Tk.LEFT )
        self.toolbar = NavigationToolbar( self.canvas.mpl_canvas, self.canvas )
        self.create_popup_menu()

        frame12 = Tk.Frame( frame1 )
        frame12.pack( side=Tk.RIGHT, fill=Tk.BOTH  )
        label = Tk.Label( frame12, text='Control Panel' )
        label.pack( anchor=Tk.N )

        view_init_btn = Tk.Button( frame12, text='view init', command=self.view_init )
        view_init_btn.pack()

        self.cb_frame = CheckbuttonFrame( frame12 )
        self.cb_frame.pack()

        redraw_btn   = Tk.Button( frame12, text='redraw', command=self.redraw )
        redraw_btn.pack()

        type_info_btn = Tk.Button( frame12, text='type help', command=self.show_type_info )
        type_info_btn.pack()

        folder_info_btn = Tk.Button( frame12, text='folder help', command=self.show_folder_info )
        folder_info_btn.pack()

        hist_btn = Tk.Button( frame12, text='histogram', command=self.show_hist )
        hist_btn.pack()

        folder_frame = Tk.Frame( frame12 )
        folder_frame.pack()
        folder_frame1 = Tk.Frame( folder_frame )
        folder_frame1.pack()
        label = Tk.Label( folder_frame1, text='Folder No:' )
        label.pack( side=Tk.LEFT )
        self.folder_no = Tk.IntVar()
        entry = Tk.Entry( folder_frame1, textvariable=self.folder_no, width=3, justify=Tk.CENTER )
        entry.pack( side=Tk.LEFT )

        show_folder_btn = Tk.Button( folder_frame, text='draw folder', command=self.draw_folder )
        show_folder_btn.pack()

        self.update()
        self.deiconify()
        self.geometry( adjusted_geometry(self.geometry()) )
        self.update()
        self.protocol( "WM_DELETE_WINDOW", self.quit )
        if tester_log is not None:
            self.tester_log.set( tester_log )
            self.on_entry_tester_log()

    def quit( self ):
        self.parent.quit()
        self.destroy()

    def on_entry_tester_log( self ):
        tester_log = self.tester_log.get()
        print( 'tester_log=', tester_log )
        self.analyzer = SimpleGuinierAnalyzer( tester_log )
        self.num_folders = len( self.analyzer.folder_infos )
        self.canvas.draw( self.analyzer.plot_results )
        self.canvas.fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )
        self.canvas.fig.canvas.mpl_connect( 'draw_event', self.on_draw )
        self.axes = [ self.analyzer.ax1 ]
        self.update_xyr( force=True )
        self.compute_projected_xy()

    def update_xyr( self, force=False ):
        if force or self.cb_frame.changed:
            index   = self.cb_frame.make_index( self.analyzer.z )
            self.x  = self.analyzer.x[index]
            self.y  = self.analyzer.y[index]
            self.r  = self.analyzer.r[index]
            print( 'len(analyzer.x)', len(self.analyzer.x),'len(self.x)=', len(self.x) )

    def compute_projected_xy( self ):
        self.update_xyr()

        ax = self.axes[0]
        # print( ax.get_proj() )
        projection = ax.get_proj()
        x, y, _ = proj3d.proj_transform( self.x, self.y, self.r, projection )
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

    def create_popup_menu( self ):
        self.popup_menu = Tk.Menu( self, tearoff=0 )
        self.popup_menu.add_command( label='Animation', command=self.draw_folder )

    def on_button_press( self, event ):
        if event.button == 3:
            self.on_right_button_press( event )
            return

    def on_right_button_press( self, event ):
        print( 'on_right_button_press' )

        if event.inaxes not in self.axes:
            return

        print( 'x=', event.x )
        print( 'y=', event.y )
        w, h, x, y = split_geometry( self.canvas.mpl_canvas_widget.winfo_geometry() )
        h_ = h
        # print( 'self.mpl_canvas_widget.winfo_geometry()=', (w, h, x, y) )
        w, h, x, y = split_geometry( self.winfo_geometry() )
        # print( 'self.winfo_geometry()=', (w, h, x, y) )
        # TODO: better way to get the mouse cursor position

        folder_no = self.guess_pointed_folder_no( event )
        if folder_no is None:
            return

        self.folder_no.set( folder_no )
        self.popup_menu.entryconfig( 0, label="draw folder %d" % folder_no )
        self.popup_menu.post( int( x + event.x + 10 ), int( y + h_ - event.y + 50 ) )

    def on_draw( self, event ):
        print( 'on_draw' )
        self.compute_projected_xy()

    def guess_pointed_folder_no( self, event ):
        dist = ( self.x_scr - event.x )**2 + ( self.y_scr - event.y )**2
        i = np.argmin( dist )
        folder_no = int( self.y[i] )
        print( 'y[%d]=%g' % ( i, folder_no ) )
        return folder_no

    def view_init( self ):
        self.analyzer.ax1.view_init()
        self.canvas.mpl_canvas.show()

    def redraw( self ):
        draw_flags = [ var.get() for var in self.cb_frame.cb_vars ]

        def redraw_closure( fig ):
            self.analyzer.plot_redraw( fig, draw_flags=draw_flags )

        self.canvas.draw( redraw_closure )
        # self.canvas.mpl_canvas.show()

    def draw_folder( self, folder_no=None ):
        if folder_no is None:
            folder_no = self.folder_no.get()
        print( 'draw_folder: folder_no=', folder_no )
        analyzer = self.analyzer
        folder_path = self.rename_folder_path( analyzer.folder_paths[folder_no] )
        folder_info = analyzer.folder_infos[folder_no]
        index = analyzer.y == folder_no
        x = analyzer.x[index]
        z = analyzer.z[index]
        r = analyzer.r[index]
        fa = SimpleGuinierFolderAnalyzer( self, folder_no, folder_path, folder_info, x, z, r, analyzer.type_names )
        fa.show()

    def rename_folder_path( self, path ):
        nodes = path.split( '/' )
        i = nodes.index( 'Data' )
        new_path = self.pytools_folder + '/' + '/'.join( nodes[i:] )
        return new_path

    def show_type_info( self ):
        dialog = TypeInfoPanel( self )
        dialog.show( self.analyzer.type_names )

    def show_folder_info( self ):
        dialog = FolderInfoPanel( self )
        dialog.show( self.analyzer.folder_paths )

    def show_hist( self ):
        dialog = CanvasDialog( "Debug", parent=self )
        dialog.show( self.draw_hist, figsize=(9,8), toolbar=True )

    def draw_hist( self, fig ):
        ax = fig.add_subplot(111)
        ax.set_title( 'Histogram of Peak Process Types' )
        bins = np.arange(18) - 0.5
        # print( 'bins=', bins )
        # bins = 17
        ax.hist( self.analyzer.z, bins=bins )
        ax.xaxis.set_ticks(np.arange(17))
        fig.tight_layout()
