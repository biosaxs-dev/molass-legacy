"""

    ファイル名：   PilatusImageInfoTable.py

    処理内容：

       Pilatus 画像データ情報の一覧

    Copyright (c) 2020,2024 SAXS Team, KEK-PF
"""
import sys
import copy
import logging
import traceback
from datetime               import datetime

from molass_legacy.KekLib.OurTkinter             import Tk, ToolTip
import OurMessageBox        as     MessageBox
from TkTableWrapper         import ArrayVar, Table
from ChangeableLogger       import Logger
from PilatusUtils           import get_data_info
from GuiPreferences         import PreferencesDialog
from PilatusImageProperty   import ImagePropertyWindow
from Preferences            import get_preference, temporary_preferences_begin, temporary_preferences_end, get_usual_preference
from SynthesizerSettings    import get_setting

# mapping displayed table col to internal array col
tif_dir_map = {
        1   :   'in_folder',
        2   :   'in_folder',
        5   :   'in_folder',
        8   :   'syn_folder',
        }

class PilatusImageInfoTable:
    # TODO: widget にすべきか否か
    def __init__( self, parent, synthesizer, suggestion, geometry_getter=None ):
        self.parent     = parent
        self.var        = None
        self.table      = None
        self.init_geometry = None
        self.current_row = 0
        self.synthesizer = synthesizer
        self.x_view     = None
        self.y_view     = None
        self.suggestion = suggestion
        self.geometry_getter = geometry_getter
        self.temporary_i_ratio_array = None
        self.yscrollbar_frame = None
        self.xscrollbar_frame = None
        self.current_data_end = 0

        self.refresh()

    def get_geometry(self):
        parent = self.parent
        parent.update()
        return parent.winfo_width(), parent.winfo_height()

    def refresh( self, restore_view=False, pilatus_counter=None, bottom_view=False, log_file_path=None, logger=None, autorun=False ):

        print('refresh: restore_view=', restore_view)
        print('refresh: bottom_view=', bottom_view)
        print('refresh: logger=', logger)

        temp_logger = Logger()

        if not restore_view:
            self.temporary_i_ratio_array = None

        self.num_selected_rows  = 0

        in_folder   = get_setting( 'in_folder' )
        adj_folder  = get_setting( 'adj_folder' )
        syn_folder  = get_setting( 'syn_folder' )

        counter_id = get_preference( 'detection_counter' )
        # print( 'counter_id=', counter_id )

        try:
            _, _, data_array, pilatus_counter = get_data_info( in_folder, adj_folder, syn_folder, pilatus_counter, counter_id,
                                                        log_file_path=log_file_path,
                                                        logger=temp_logger,
                                                        )
        except:
            logging.exception( 'Unexpected error' )
            MessageBox.showerror( 'Unexpected Error', traceback.format_exc() )
            return

        buffer = temp_logger.get_stream_buffer()
        if buffer.find("ERROR") > 0:
            if autorun:
                from ErrorLogCheck import all_known_errors
                show_message = not all_known_errors(buffer)
            else:
                show_message = True
            if show_message:
                    MessageBox.showerror( 'Refresh Error',
                                            "Following errors have been detected during refresh.\n"
                                            "Please check them before continuing.\n"
                                            "(You can copy this info with <Ctrl+c> to the clipboard\n"
                                            "in case needed)\n\n"
                                            + buffer.replace(",root", "") )

        new_num_rows = len(data_array)
        print( 'refresh: new_num_rows=', new_num_rows)

        self.data_array = data_array
        self.pilatus_counter = pilatus_counter

        if self.current_data_end > len( data_array ):
            # 作動中に出力フォルダのデータを削除した場合などが該当する。
            self.current_data_end = 0

        self.var = ArrayVar( self.parent )

        self.var['0,0'] = 'No'
        # self.var['0,1'] = 'Sample ID'       # TODO: change names derived from Image ID => Sample ID
        self.var['0,1'] = 'Original Image'
        # self.var['0,2'] = 'Absolute Position'
        self.var['0,2'] = 'Shifted Image 1'
        self.var['0,3'] = 'Relative Position 1'
        self.var['0,4'] = 'Intensity Ratio 1 %s' % ( counter_id )
        # self.var['0,5'] = 'Adjusted Image'
        self.var['0,5'] = 'Shifted Image 2'
        self.var['0,6'] = 'Relative Position 2'
        self.var['0,7'] = 'Intensity Ratio 2 %s' % ( counter_id )
        self.var['0,8'] = 'Synthesized Image'

        num_cols = 8

        row = 0
        for row_rec in self.data_array:
            row += 1
            # print( '[%d]' % (row), row_rec )
            self.var["%d,%d" % (row, 0)] = row
            sample_id   = row_rec[0]
            i = 0
            col = 0
            for sub_rec in row_rec[1]:
                i += 1

                col += 1
                if sub_rec[0] == None:
                    disp_value = ''
                else:
                    disp_value = sub_rec[0]
                self.var["%d,%d" % (row, col)] = disp_value

                if i == 1:
                    continue

                disp_value = '%s,%s' % tuple( sub_rec[1] )
                col += 1
                self.var["%d,%d" % (row, col)] = disp_value

                i_ratio_value = sub_rec[2]
                if i_ratio_value:
                    changed_counter_id = ''
                    if self.temporary_i_ratio_array:
                        # TODO
                        pass
                    disp_value = '%.5f%s' % ( i_ratio_value, changed_counter_id )   # 表示用の桁数調整
                else:
                    disp_value = ''
                col += 1
                self.var["%d,%d" % (row, col)] = disp_value

                if i == len( row_rec[1] ):
                    col = num_cols
                    if sub_rec[4] == None:
                        disp_value = ''
                    else:
                        disp_value = sub_rec[4]
                    self.var["%d,%d" % (row, col)] = disp_value

        need_resize = False
        if self.table is None:
            self.init_num_rows = len(self.data_array)
        else:
            self.save_xy_views()
            self.table.destroy()
            print('table destroyed')
 
        if self.init_num_rows == 0:
            print('must show invisible rows')
            need_resize = True

        self.table = Table( self.parent,
                     rows       = row + 1,
                     cols       = num_cols + 1,
                     state      = 'disabled',
                     # state      ='normal',
                     width      = 0,
                     maxwidth   = 1600,
                     # height     = 10,
                     height     = 21,
                     titlerows  = 1,
                     titlecols  = 1,
                     roworigin  = 0,
                     colorigin  = 0,
                     # selectmode = 'browse',
                     selectmode = 'extended',
                     selecttype = 'row',
                     rowstretch = 'unset',
                     # colstretch = 'last',
                     colstretch = 'all',
                     browsecmd  = self.browsecmd,
                     flashmode  = 'on',
                     variable   = self.var,
                     usecommand = 0,
                     command    = self.test_cmd,
                     colwidth   = 20
                     )

        self.num_selectable_cols = num_cols - 1

        # thanks to sof
        # http://effbot.org/zone/tkinter-scrollbar-patterns.htm
        if self.yscrollbar_frame:
            self.yscrollbar_frame.destroy()
        self.yscrollbar_frame = Tk.Frame( self.parent )
        self.yscrollbar_frame.pack(side=Tk.RIGHT, fill=Tk.Y)
        # self.yscrollbar_frame.grid( row=0, column=1 )
        self.yscrollbar = Tk.Scrollbar( self.yscrollbar_frame, orient='vertical', command=self.table.yview_scroll )
        self.table.config( yscrollcommand=self.yscrollbar.set )
        self.yscrollbar_isvisible = False

        if self.xscrollbar_frame:
            self.xscrollbar_frame.destroy()
        self.xscrollbar_frame = Tk.Frame( self.parent )
        self.xscrollbar_frame.pack(side=Tk.BOTTOM, fill=Tk.X)
        # self.xscrollbar_frame.grid( row=1, column=0 )
        self.xscrollbar = Tk.Scrollbar( self.xscrollbar_frame, orient='horizontal', command=self.table.xview_scroll )
        self.table.config( xscrollcommand=self.xscrollbar.set )
        self.xscrollbar_isvisible = False

        self.on_resize()

        self.table.pack( expand=1, fill=Tk.BOTH )
        # self.table.grid( row=0, column=0 )

        self.table.tag_configure('sel', background = 'steel blue')
        self.table.tag_configure('active', background = 'blue')
        # self.table.tag_configure('title', anchor='w', bg='skyblue', relief='sunken')
        self.table.tag_configure('title', bg='gray25', relief='sunken')
        self.table.tk.call( self.table._w, 'width',
            0,  5,      # No
            # 1, 16,      # Image ID
            1, 36,      # Original Image
            # 2, 20,      # Absolute Position
            2, 36,      # Shifted Image 1
            3, 16,      # Relative Position 1
            4, 18,      # Intensity Ratio 1

            5, 36,      # Shifted Image 2
            6, 16,      # Relative Position 2
            7, 18,      # Intensity Ratio 2

            5, 36,      # Adjusted Image

            8, 36,      # Synthesized Image
            )


        if restore_view:
            self.restore_xy_views()

        if bottom_view:
            self.set_view_to_bottom()
            self.on_resize()

        self.select_added_rows()

        print( 'refresh: after select_added_rows y_view=', self.y_view )
        print( 'refresh: after select_added_rows yscrollbar_isvisible=', self.yscrollbar_isvisible )
        print( 'refresh: after select_added_rows height=',  self.table.cget( 'height' ))
        if need_resize:
            print('on_resize when need_resize')
            # self.table.see('topleft')
            self.on_resize()

        self.table.see('end')

        if self.init_geometry is None:
            # this does not get what is wanted.
            self.init_geometry = self.get_geometry()
            print('init_geometry=', self.init_geometry)

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

        self.table.bind( "<Button-3>", self.popup )
        self.table.bind( "<Button-1>", self.popup_state_reset )
        # self.table.bind( "<Triple-Button-1>", lambda evet: self.do_action( 1 ) )
        self.table.bind( "<Double-Button-1>", lambda evet: self.do_action( 1 ) )
        self.table.bind( "<Configure>", self.on_resize )

        ToolTip( self.table, '<Double-click> a row to show images. Select and <right-click> to show the action menu. Click top-left corner "No" to select all rows.')

        return need_resize

    def get_resize_height(self):
        num_rows = min(20, len(self.data_array))
        height = int(self.table.cget('height'))
        return num_rows * height

    def on_resize( self, event=None ):
        f, t = self.table.yview()
        if f > 0 or t < 1:
            if not self.yscrollbar_isvisible:
                self.yscrollbar.pack(side=Tk.RIGHT, fill=Tk.Y)
                self.yscrollbar_isvisible = True
        else:
            if self.yscrollbar_isvisible:
                self.yscrollbar.pack_forget()
                self.yscrollbar_isvisible = False

        f, t = self.table.xview()
        if f > 0 or t < 1:
            if not self.xscrollbar_isvisible:
                self.xscrollbar.pack(side=Tk.BOTTOM, fill=Tk.X)
                self.xscrollbar_isvisible = True
        else:
            if self.xscrollbar_isvisible:
                self.xscrollbar.pack_forget()
                self.xscrollbar_isvisible = False
                # self.xscrollbar_frame.configure( height=0 )

    def popup( self, event ):
        self.popup_position = ( event.x, event.y )
        # print( 'popup_position=', self.popup_position )

        # 改造した Tktable.dll ならば次の work-around は不要
        # self.table.configure( state='normal' )

        self.menu.post( event.x_root, event.y_root )

    def popup_state_reset( self, event ):
        # 改造した Tktable.dll ならば継ぎの work-around は不要
        # self.table.configure( state='disabled' )
        pass

    def set_view_to_bottom( self ):
        height  = self.table.cget( 'height' )
        rows    = self.table.cget( 'rows' )
        # caveate: silent exception without int()!
        ypos = max( 0, 1 - int(height) / int(rows)  )
        if ypos < 0:
            ypos = 0
        self.table.yview_moveto( ypos )

    def test_cmd( self, event ):
        if event.i == 0:
            return '%i, %i' % (event.r, event.c)
        else:
            return 'set'

    def test_select_row( self, row ):
        # TODO: remove this method
        self.num_selected_rows = 1
        self.current_row = row + 1

    def set_num_selected_rows( self ):
        self.num_selected_rows = len( self.table.curselection() ) // ( self.num_selectable_cols + 1 )

    def browsecmd( self, event ):
        self.selection_array = []
        self.set_num_selected_rows()
        # print( 'self.num_selected_rows=', self.num_selected_rows )
        active_cell_index = self.table.index('active')
        # print( 'active_cell_index=', active_cell_index )
        self.current_row = int( active_cell_index.split(',')[0] )
        # print( 'self.current_row=', self.current_row )

    def get_last_cell_index( self ):
        num_rows = len( self.data_array )
        return '%d,%d' % ( num_rows,  self.num_selectable_cols + 1 )

    def select_all_rows( self ):
        last_cell = self.get_last_cell_index()
        self.table.selection_set( '1,1', last_cell )
        self.set_num_selected_rows()

    def select_added_rows( self ):
        if len( self.data_array ) == 0: return

        last_cell   = self.get_last_cell_index()

        if self.current_data_end < len( self.data_array ):
            first_cell  = '%d,1' % ( self.current_data_end + 1 )
            self.table.selection_set( first_cell, last_cell )
            self.set_num_selected_rows()
        else:
            self.table.selection_clear( '0,0', last_cell )

    def get_selected_indices( self ):
        curselection = self.table.curselection()
        # print( 'curselection=', curselection )
        indices = []
        for c in curselection[ slice( 0, len(curselection), self.num_selectable_cols + 1 ) ]:
            i = int( c.split(',')[0] ) - 1
            indices.append( i )
        # print( 'indices=', indices )
        return indices

    def get_not_yet_done_indices( self ):
        indices = []
        for k, row in enumerate(self.data_array):
            key, recs = row[0:2]
            syn_file = recs[0][4]
            if syn_file is None:
                indices.append(k)
        return indices

    def clear_selection( self, indices ):
        for i in indices:
            begin   = '%d,0' % ( i + 1 )
            end     = '%d,%d' % ( i + 1, self.num_selectable_cols + 1 )
            self.table.selection_clear( begin, end )

    def select_data_array( self, selected_indices ):
        array = []
        for i in selected_indices:
            array.append( self.data_array[ i ] )
        return array

    def do_action( self, action, change=False ):
        self.popup_state_reset( 0 )

        if self.suggestion and self.suggestion.is_blinking():
            MessageBox.showwarning( "Not Allowed", "You can't do any action until you refresh the image data infomation table." )
            return

        selected_indices = self.get_selected_indices()
        exec_array = self.select_data_array( selected_indices )

        if len( exec_array ) == 0:
            # this case do happen.
            MessageBox.showinfo( 'No Rows Selected Notification', 'No rows are selected. Select row(s) and retry.' )
            return

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
        curselection = self.table.curselection()
        if len( curselection ) == 0:
            # this case do happen.
            MessageBox.showinfo( 'No Row Selected Notification', 'No row is selected. Select a row and retry.' )
            return

        # TODO: also consider right-click position
        active_cell_index = self.table.index('active')
        print( "active cell index:", active_cell_index )

        first_row = int( curselection[0].split( ',' )[0] )
        col = int( active_cell_index.split(',')[1] )
        if col in tif_dir_map:
            dirkey = tif_dir_map[ col ]
            filename = self.var[ '%d,%d' % (first_row, col) ]
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
