# -*- coding: utf-8 -*-
"""

    ファイル名：   PilatusImageInfoTable.py

    処理内容：

       Pilatus 画像データ情報の一覧

"""
from __future__ import division, print_function, unicode_literals

import sys
import copy
import logging
import traceback
from datetime               import datetime

from molass_legacy.KekLib.OurTkinter             import Tk, ToolTip
import OurMessageBox        as     MessageBox 
from CuntomizedPandasTable import PandasTable

from PilatusUtils           import get_data_info
from GuiPreferences         import PreferencesDialog
from PilatusImageProperty   import ImagePropertyWindow
from Preferences            import get_preference, temporary_preferences_begin, temporary_preferences_end, get_usual_preference
from SynthesizerSettings    import get_setting

# mapping displayed table col to internal array col
tif_dir_map = {
        0   :   'in_folder',
        1   :   'in_folder',
        4   :   'in_folder',
        7   :   'syn_folder',
        }

class PilatusImageInfoTable:
    def __init__( self, parent, synthesizer, suggestion ):
        self.parent     = parent
        self.var        = None
        self.table      = None
        self.scrollbar  = None
        self.current_row = 0
        self.synthesizer = synthesizer
        self.x_view     = None
        self.y_view     = None
        self.suggestion = suggestion
        self.temporary_i_ratio_array = None
        self.num_selected_rows  = 0
        self.current_data_end = 0

        self.refresh()

    def refresh( self, restore_view=False, pilatus_counter=None, bottom_view=False, log_file_path=None ):

        # print( 'refresh: bottom_view=', bottom_view )

        if not restore_view:
            self.temporary_i_ratio_array = None

        self.data_array = []

        in_folder   = get_setting( 'in_folder' )
        adj_folder  = get_setting( 'adj_folder' )
        syn_folder  = get_setting( 'syn_folder' )

        counter_id = get_preference( 'detection_counter' )
        # print( 'counter_id=', counter_id )

        try:
            _, _, data_array, pilatus_counter = get_data_info( in_folder, adj_folder, syn_folder, pilatus_counter, counter_id,
                                                        log_file_path=log_file_path
                                                        )
        except:
            logging.exception( 'Unexpected error' )
            MessageBox.showerror( 'Unexpected Error', traceback.format_exc() )
            return

        self.data_array = data_array
        self.pilatus_counter = pilatus_counter

        if self.current_data_end > len( data_array ):
            # 作動中に出力フォルダのデータを削除した場合などが該当する。
            self.current_data_end = 0

        columns = [
                [ 'Original Image', 140 ],
                [ 'Shited Image 1', 140 ],
                [ 'Relative Position 1', 100 ],
                [ 'Intensity Ratio 1 None', 110 ],
                [ 'Shited Image 2', 140 ],
                [ 'Relative Position 2', 100 ],
                [ 'Intensity Ratio 2 None', 110 ],
                [ 'Synthesized Image', 140 ],
            ]

        num_cols = 8

        row = 0
        self.var = []

        for row_rec in self.data_array:
            row += 1
            # print( '[%d]' % (row), row_rec )

            var_rec = []

            sample_id   = row_rec[0]
            i = 0
            for sub_rec in row_rec[1]:
                i += 1

                var_rec.append( sub_rec[0] )

                if i == 1:
                    continue

                disp_value = '%s,%s' % tuple( sub_rec[1] )
                var_rec.append( disp_value )

                i_ratio_value = sub_rec[2]
                if i_ratio_value:
                    changed_counter_id = ''
                    if self.temporary_i_ratio_array:
                        # TODO
                        pass
                    disp_value = '%.5f%s' % ( i_ratio_value, changed_counter_id )   # 表示用の桁数調整
                else:
                    disp_value = ''
                var_rec.append( disp_value )

                if i == len( row_rec[1] ):
                    for j in range( len( var_rec ), num_cols-1 ):
                        var_rec.append( '' )
                    var_rec.append( sub_rec[4] )

            self.var.append( var_rec )

        if self.table:
            # self.save_xy_views()
            # self.table.destroy()
            if self.scrollbar:
                self.scrollbar.destroy()

        created = False
        if self.table == None:
            self.table = PandasTable( self.parent, columns=columns )
            self.table.pack( expand=1, fill=Tk.BOTH )
            created = True

        self.table.import_array( self.var )

        if created:
            # create a popup menu
            self.menu = Tk.Menu( self.table, tearoff=0 )
            self.menu.add_command( label='Show Original Images',    command=lambda: self.do_action( 1 ) )
            self.menu.add_command( label='Show Adjusted Images',    command=lambda: self.do_action( 2 ) )
            self.menu.add_command( label='Make Synthsized Images',  command=lambda: self.do_action( 3 ) )
            self.menu_c = Tk.Menu( self.table, tearoff=0 )
            # ToolTip( self.menu_c, 'With the cascading submenus, you can temporarily change the preferences and execute.')
            # does not work expectedly
            self.menu.add_cascade( label='Execute with Temporary Preference Changes', menu=self.menu_c )
            self.menu_c.add_command( label='Show Original Simply',   command=lambda: self.do_action( 1, True ) )
            self.menu_c.add_command( label='Show Adjusted Images',   command=lambda: self.do_action( 2, True ) )
            self.menu_c.add_command( label='Make Synthsized Images', command=lambda: self.do_action( 3, True ) )
            self.menu.add_command( label='Tiff File Description',    command=self.show_properties )

            self.table.bindActionMenu( "<Button-3>", self.popup, corner=True )
            self.table.bindActionMenu( "<Double-Button-1>", lambda evet: self.do_action( 1 ), corner=False )

            ToolTip( self.table, '<Double-click> a row to show images. Select and <right-click> to show the action menu. Click top-left corner "No" to select all rows.')

        self.num_selectable_cols = num_cols

        if restore_view:
            # self.restore_xy_views()
            pass

        if bottom_view:
            self.set_view_to_bottom()
            self.on_resize()

        self.select_added_rows()

    def on_resize( self, event=None ):
        pass

    def set_view_to_bottom( self ):
        # TODO:
        pass

    def popup( self, event ):
        self.popup_position = ( event.x, event.y )
        # print 'popup_position=', self.popup_position

        # 改造した Tktable.dll ならば継ぎの work-around は不要
        # self.table.configure( state='normal' )

        self.menu.post( event.x_root, event.y_root )

    def popup_state_reset( self, event ):
        # 改造した Tktable.dll ならば継ぎの work-around は不要
        # self.table.configure( state='disabled' )
        pass

    def test_cmd( self, event ):
        if event.i == 0:
            return '%i, %i' % (event.r, event.c)
        else:
            return 'set'

    def browsecmd( self, event ):
        pass

    def select_all_rows( self ):
        self.table.selectAll()
        self.selected_rows = self.table.selectedRows()
        self.num_selected_rows = len( self.selected_rows )

    def select_added_rows( self ):
        if len( self.data_array ) == 0: return

        if self.current_data_end < len( self.data_array ):
            print( 'select_added_rows: selectRows %d, %d' % ( self.current_data_end, len( self.data_array ) ) )
            self.table.selectRows( self.current_data_end, len( self.data_array ) )
            self.selected_rows = self.table.selectedRows()
        else:
            self.table.selectionClear()
            self.selected_rows = []
            pass

        self.num_selected_rows = len( self.selected_rows )

    def get_selected_indices( self ):
        # 手動の選択もあるので、取り直す。
        self.selected_rows = self.table.selectedRows()
        self.num_selected_rows = len( self.selected_rows )
        return self.selected_rows

    def select_data_array( self, selected_indices ):
        array = []
        for i in selected_indices:
            array.append( self.data_array[ i ] )
        return array

    def clear_selection( self, indices ):
        # TODO: clear only indices
        self.table.selectionClear()

    def do_action( self, action, change=False ):
        self.popup_state_reset( 0 )

        if self.suggestion and self.suggestion.is_blinking():
            MessageBox.showwarning( "Not Allowed", "You can't do any action until you refresh the image data infomation table." )
            return

        selected_indices = self.get_selected_indices()
        exec_array = self.select_data_array( selected_indices )

        counter_id_is_changed = False
        if change:
            temporary_preferences_begin()
            dialog = PreferencesDialog( self.parent, "Temporary Preferences", self.pilatus_counter,
                            action=action )
            if not dialog.applied:
                return

            counter_id = get_preference( 'detection_counter' )
            if counter_id != get_usual_preference( 'detection_counter' ):
                counter_id_is_changed = True
                counter_dict = self.pilatus_counter.get_counter_dict( counter_id )
                if not self.temporary_i_ratio_array:
                    self.temporary_i_ratio_array = []
                    for rec in self.data_array:
                        self.temporary_i_ratio_array.append( None )

        start = datetime.now()
        # print '%s start.' % ( start.strftime( '%Y-%m-%d %H:%M:%S' ) )

        self.synthesizer.execute( action, exec_array )

        end = datetime.now()
        t = end - start
        # print '%s end. took %d seconds' % ( end.strftime( '%Y-%m-%d %H:%M:%S' ), t.seconds )

        if change:
            temporary_preferences_end()

        if action == 3:
            self.update_current( selected_indices )
            self.refresh( restore_view=True )
            # TODO: update in synthesizer.single_exec

        return

    def update_current( self, selected_indices ):
        if len( selected_indices ) > 0:
            end_index = selected_indices[-1] + 1
            if end_index > self.current_data_end:
                self.current_data_end = end_index
                print( 'new self.current_data_end=', self.current_data_end )

    def dummy( self ):
        pass

    def show_properties( self ):
        self.popup_state_reset( 0 )
        row, col = self.table.selectedCell()
        if row == None:
            # this case do happen.
            MessageBox.showinfo( 'No Row Selected Notification', 'No row is selected. Select a row and retry.' )
            return

        if col in tif_dir_map:
            dirkey = tif_dir_map[col]
            filename = self.var[row][col]
            dirpath = get_setting( dirkey )
            path = dirpath + '/' + filename
            title = filename + ' Description'
            ImagePropertyWindow( self.parent, title, path )
        else:
            MessageBox.showinfo( 'Select Tiff File Notice', 'Select tiff file cell and retry.' )
        return

    def save_xy_views( self ):
        self.x_view = self.table.xview()
        self.y_view = self.table.yview()

    def restore_xy_views( self ):
        if self.x_view and self.y_view:
            self.table.xview_moveto( self.x_view[0] )
            self.table.yview_moveto( self.y_view[0] )
