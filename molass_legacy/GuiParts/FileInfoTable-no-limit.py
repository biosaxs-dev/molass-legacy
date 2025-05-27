"""

    FileInfoTable.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF

"""

import os
import glob
import re
import time
from datetime               import datetime as dt
from molass_legacy.KekLib.OurTkinter             import Tk, ToolTip, ttk
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import OurMessageBox as MessageBox
from TkMiniTable            import TkMiniTable
from molass_legacy._MOLASS.SerialSettings         import get_setting
from ResultGui              import ResultGui
from KekToolsGP             import autorg   as autorg_kekpf
from molass_legacy.KekLib.NumpyUtils             import np_loadtxt
from SerialDataUtils        import get_xray_files

class FileInfoTable( Tk.Frame ):
    def __init__( self, parent, dialog, loader, fixed_font, *args, **kw ):

        Tk.Frame.__init__( self, parent, *args, **kw )

        self.parent = parent
        self.dialog = dialog
        self.high_dpi = dialog.high_dpi
        self.loader = loader
        self.fixed_font = fixed_font
        self.create_table()
        self.datafiles  = None
        # self.refresh_impl()

        self.popup_menu_ready = False

        parent.update()

    def create_table( self ):
        self.averaged_intensity_array = None

        self.table = TkMiniTable( self, columns=[ '_', 'filename', 'datetime', 'checkbutton' ] )
        self.table.pack( fill=Tk.BOTH, expand=1 )

        self.table.add_mousewheel_bind( self.dialog )
        # it seemd that '<Control-c>' should be bound from the top-level.
        self.dialog.bind( '<Control-c>', self.table._on_control_c )

    def create_popup_menu( self, destroy=False ):
        if destroy:
            if self.popup_menu_ready:
                self.menu.destroy()

        atsas_exe_paths = get_setting( 'atsas_exe_paths' )
        atsas_versions = self.get_atsas_versions( atsas_exe_paths )

        # create a popup menu
        self.menu = Tk.Menu( self, tearoff=0 )
        num_atsas_exe = len(atsas_exe_paths)
        if num_atsas_exe > 0:
            for i in range( num_atsas_exe ):
                self.menu.add_command( label='Run with ' + atsas_versions[i],  command=lambda i_ = i: self.run_autorg( i_ ) )
        else:
            self.menu.add_command( label='Run AutoRg',  command=lambda: self.run_autorg() )

        self.menu.add_command( label='Run AutoGuinier Animation',  command=lambda: self.run_simple_guinier_animation() )

    def get_atsas_versions( self, atsas_exe_paths ):
        from molass_legacy.ATSAS.AtsasUtils import get_versions
        return ['ATSAS ' + v for v in get_versions()]

    def _on_right_click( self, event ):
        if not self.popup_menu_ready:
            self.create_popup_menu()
            self.popup_menu_ready = True

        self.popup_position = ( event.x, event.y )
        # print( 'popup_position=', self.popup_position )

        self.menu.post( event.x_root, event.y_root )

    def refresh(self):
        self.table.destroy()
        self.create_table()
        self.refresh_impl()

    def refresh_impl(self):

        self.testing = self.dialog.testing
        table = self.table

        table.heading( "_", text='No' )
        table.column(  "_", width=4, stretch=0  )
        table.heading( 'filename', text='File Name')
        width = 53 if self.high_dpi else 42
        table.column(  'filename', anchor='center', width=width )
        table.heading( 'datetime', text='Last Modified')
        width = 20 if self.high_dpi else 16
        table.column(  'datetime', anchor='center', width=width, stretch=0 )
        table.heading( 'checkbutton', text='exclude')
        table.column(  'checkbutton', anchor='center', width=7, stretch=0 )

        self.in_folder   = get_setting( 'in_folder' )
        self.datafiles  = get_xray_files( self.in_folder )

        n = 0
        if self.in_folder is not None:
            for path in self.datafiles:
                t   = os.path.getmtime( path )
                dt_ = dt.fromtimestamp( t )
                dir_, file = os.path.split( path )
                table.insert_row( ( n, file, dt_.strftime( '%Y-%m-%d %H:%M:%S' ), '' ) )
                n += 1
        self.num_rows = n

    def _on_click( self, event ):
        # print( '_on_click' )
        cell = self.table.active_cell
        if cell is None: return

        if cell.row < 1 or cell.row > self.num_rows:
                return

        self.dialog.draw_figure(selected=cell.row-1)

        if get_setting('data_exclusion'):
            if cell.col == 3:
                # print( 'cell.row=',  cell.row, 'cell.col=', cell.col )
                if cell.svar.get() == '':
                    if not self.check_too_many_excluded_neighbor_cells( cell.row ):
                        cell.svar.set( 'X' )
                else:
                    cell.svar.set( '' )

    def check_too_many_excluded_neighbor_cells( self, row ):
        return False

    def _on_double_click( self, event ):
        self.run_autorg( 0 )

    def run_autorg( self, exe_index=None ):
        self.config( cursor='wait' )
        self.update()

        data, file, _, n = self.get_data_array()

        print( 'run_autorg: data.shape=', data.shape )

        try:
            q_limit_index  = self.serial_data.get_usable_q_limit()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)
            q_limit_index  = None

        orig_info = [ self.serial_data, n ]
        result_gui = ResultGui( self, self.in_folder, file, np_array=data, atsas_exe_index=exe_index, orig_info=orig_info, q_limit_index=q_limit_index )

        self.config( cursor='' )
        self.update()

        result_gui.after( 0, result_gui.focus_force )       # workaround to raise to forefront

    def get_data_array( self, n=None ):

        if n is None:
            row = self.table.selected_rows[0]
            # file = self.table.cells_array[ row, 1 ].svar.get()
            n = row -1

        file = self.table.cells_array[ n+1, 1 ].svar.get()

        if self.loader.is_loadable:
            serial_data = self.loader.get_data_object()
            usable_slice = serial_data.get_usable_slice()
            if self.averaged_intensity_array is None:
                num_curves_averaged = get_setting( 'num_curves_averaged' )
                self.averaged_intensity_array, _, _ = serial_data.get_averaged_data( num_curves_averaged )
            data = self.averaged_intensity_array[n, usable_slice]
            title_averaged = ' (averaged)'
            self.serial_data = serial_data
        else:
            data, _ = np_loadtxt( self.in_folder + '/' + file )
            title_averaged = ''
            self.serial_data = None

        # print( 'file[%d]=%s%s' % ( n, file, title_averaged ) )

        return data, file, title_averaged, n

    def run_simple_guinier_animation( self ):
        self.config( cursor='wait' )
        self.update()

        from SimpleGuinier          import SimpleGuinier
        from molass_legacy.AutorgKekAdapter       import AutorgKekAdapter
        from SimpleGuinierAnimation import SimpleGuinierAnimation
        from CanvasDialog           import CanvasDialog

        data, file, title_averaged, n = self.get_data_array()

        guinier = SimpleGuinier( data, anim_data=True )
        adapter = AutorgKekAdapter( None, guinier=guinier )
        adapter_result = adapter.run()

        self.anim_dialog = CanvasDialog( "AutoGuinier Animation", parent=self.parent )
        anim_iter_max = 300

        def anim_func( fig  ):
            anim = SimpleGuinierAnimation( guinier, title=file+title_averaged, result_rg=adapter_result.Rg  )
            anim.draw( fig, anim_iter_max=anim_iter_max )
            self.anim_dialog.mpl_canvas.draw()

        self.anim_dialog.show( anim_func, figsize=( 18, 5 ) )

        self.config( cursor='' )
        self.update()

    def get_excluded_indeces( self ):
        indeces = []
        for i, row in enumerate( self.table.cells_array, start=-1 ):
            if row[3].svar.get() == 'X':
                indeces.append( i )
        return indeces

    def set_exclude( self, exclude ):
        for i in exclude:
            self.table.cells_array[ i+1, 3 ].svar.set( 'X' )

    def select_row(self, row):
        if row < self.num_rows:
            self.table.select_row(row)
