# -*- coding: utf-8 -*-
"""

    ファイル名：   GuiSettingInfo.py

    処理内容：

        DISyn GUI のパラメータ入力フレーム

        Copyright (c) 2016-2021, SAXS Team, KEK-PF

"""
from __future__ import division, print_function, unicode_literals

import sys
import os
import re

from molass_legacy.KekLib.OurTkinter         import Tk, FileDialog, ToolTip
import OurMessageBox        as MessageBox

import PilatusCounter
from PilatusUtils       import get_in_folder_info
import ChangeableLogger
from CreateFolderDialog import CreateFolderDialog
from molass_legacy.KekLib.BasicUtils         import exe_name
from Preferences        import get_preference
from Development        import get_devel_info
from SynthesizerSettings    import clear_settings, get_setting, set_setting, set_mask, get_mask
from molass_legacy.KekLib.TkSupplements      import SlimButton

DEFAULT_WATCH_INVERVAL = 180

def is_empty_val( val ):
    return val == None or val == '' or val[0] == '<' or val == 'None' 

class EntryFrame(Tk.Frame):

    def __init__(self, master, controller):
        Tk.Frame.__init__(self, master)

        self.during_init = True
        self.controller = controller
        self.entries    = []
        self.app_logger = None
        self.pilatus_counter    = None
        self.auto_set_asked = False
        self.auto_set_done  = False
        self.adj_output     = get_devel_info( 'adj_output' )
        self.data_entry_ready = False
        self.in_construction = True
        self.logfile_path   = ''
        self.error_notified = {}

        grid_row = -1

        grid_row += 1
        in_folder_name  = Tk.Label( self, text= 'Measured Image Folder: ' )
        in_folder_name.grid( row=grid_row, column=0, sticky=Tk.E )
        self.in_folder = Tk.StringVar()
        in_folder = get_setting( 'in_folder' )
        self.in_folder.set( in_folder )
        self.in_folder_entry = Tk.Entry( self, textvariable=self.in_folder, width=80 )
        self.entries.append( [ 'dir', self.in_folder_entry, self.on_entry_in_folder ] )
        self.in_folder_entry.grid( row=grid_row, column=1 )
        b1 = Tk.Button( self, text='...', command=self.select_in_folder )
        b1.grid( row=grid_row, column=2, sticky=Tk.W )
        ToolTip( b1, 'Select an appropriate folder with this button.' )
        cm1 = Tk.Label( self, text= ' must be entered manually' )
        cm1.grid( row=grid_row, column=3, sticky=Tk.W )

        grid_row += 1
        log_file_name = Tk.Label( self, text= 'Measurement Log File: ' )
        log_file_name.grid( row=grid_row, column=0, sticky=Tk.E )
        self.log_file = Tk.StringVar()
        log_file = get_setting( 'log_file' )
        self.log_file.set( log_file )
        self.log_file_entry = Tk.Entry( self, textvariable=self.log_file, width=80 )
        self.entries.append( [ 'file', self.log_file_entry, self.on_entry_log_file ] )
        self.log_file_entry.grid( row=grid_row, column=1 )
        b2 = Tk.Button( self, text='...', command=self.select_log_file )
        b2.grid( row=grid_row, column=2, sticky=Tk.W )
        ToolTip( b2, 'Select an appropriate file with this button.' )
        cm2 = Tk.Label( self, text= ' automatically set if exists in the above folder' )
        cm2.grid( row=grid_row, column=3, sticky=Tk.W )

        grid_row += 1
        mask_file_name = Tk.Label( self, text= 'SAngler Mask File: ' )
        mask_file_name.grid( row=grid_row, column=0, sticky=Tk.E  )
        self.mask_file = Tk.StringVar()
        mask_file = get_setting( 'mask_file' )
        self.mask_file.set( mask_file )
        self.mask_file_entry = Tk.Entry( self, textvariable=self.mask_file, width=80 )
        self.entries.append( [ 'file', self.mask_file_entry, self.on_entry_mask_file ] )
        self.mask_file_entry.grid( row=grid_row, column=1 )
        b3 = Tk.Button( self, text='...', command=self.select_mask_file )
        b3.grid( row=grid_row, column=2, sticky=Tk.W )
        ToolTip( b3, 'Select an appropriate file with this button.' )
        cm3 = Tk.Label( self, text= ' automatically set if exists in the above folder' )
        cm3.grid( row=grid_row, column=3, sticky=Tk.W )

        if self.adj_output == 'YES':
            grid_row += 1
            adj_folder_name = Tk.Label( self, text= 'Adjusted Image Folder: ' )
            adj_folder_name.grid( row=grid_row, column=0, sticky=Tk.E  )
            self.adj_folder = Tk.StringVar()
            adj_folder = get_setting( 'adj_folder' )
            self.adj_folder.set( adj_folder )
            self.adj_folder_entry = Tk.Entry( self, textvariable=self.adj_folder, width=80 )
            self.entries.append( [ 'dir', self.adj_folder_entry, self.on_entry_adj_folder ] )
            self.adj_folder_entry.grid( row=grid_row, column=1 )
            b4 = Tk.Button( self, text='...', command=self.select_adj_folder )
            b4.grid( row=grid_row, column=2, sticky=Tk.W )
            ToolTip( b4, 'Select an appropriate folder with this button.' )
            cm4 = Tk.Label( self, text= ' must be entered manually' )
            cm4.grid( row=grid_row, column=3, sticky=Tk.W )

        grid_row += 1
        syn_folder_name = Tk.Label( self, text= 'Synthesized Image Folder: ' )
        syn_folder_name.grid( row=grid_row, column=0, sticky=Tk.E  )
        self.syn_folder = Tk.StringVar()
        syn_folder = get_setting( 'syn_folder' )
        self.syn_folder.set( syn_folder )
        self.syn_folder_entry = Tk.Entry( self, textvariable=self.syn_folder, width=80 )
        self.entries.append( [ 'dir', self.syn_folder_entry, self.on_entry_syn_folder ] )
        self.syn_folder_entry.grid( row=grid_row, column=1 )
        b5 = Tk.Button( self, text='...', command=self.select_syn_folder )
        b5.grid( row=grid_row, column=2, sticky=Tk.W )
        ToolTip( b5, 'Select an appropriate folder with this button.' )
        cm5 = Tk.Label( self, text= ' must be entered manually' )
        cm5.grid( row=grid_row, column=3, sticky=Tk.W )

        grid_row += 1
        op_option_label = Tk.Label( self, text= 'Operation Mode: ' )
        op_option_label.grid( row=grid_row, column=0, sticky=Tk.E  )
        op_option_frame = Tk.Frame( self )
        op_option_frame.grid( row=grid_row, column=1, sticky=Tk.W  )
        self.op_option = Tk.StringVar()
        j = 0
        spaces = '    '
        self.op_option_buttons = []
        for t, v in ( [ 'Manual   ' + spaces, 'MANUAL' ], [ 'Automatic' + spaces, 'AUTOMATIC' ] ):
            b = Tk.Radiobutton( op_option_frame, text=t,
                            variable=self.op_option, value=v )
            self.op_option_buttons.append( b )
            b.grid( row=0, column=j, sticky=Tk.W )
            j += 1

        self.watch_interval_label = Tk.Label( op_option_frame, text= 'Watch Interval ' )
        self.watch_interval_label.grid( row=0, column=j, sticky=Tk.W )

        j += 1
        self.watch_interval = Tk.StringVar()
        watch_interval = get_setting( 'watch_interval' )
        self.watch_interval.set( str( watch_interval ) )
        self.watch_interval_box = Tk.Spinbox( op_option_frame, textvariable=self.watch_interval,
                                                from_=10, to=600, increment=10, width=4, justify=Tk.CENTER )
        self.watch_interval_box.grid( row=0, column=j, sticky=Tk.W )

        j += 1
        self.watch_interval_s = Tk.Label( op_option_frame, text= 's  ' )
        self.watch_interval_s.grid( row=0, column=j, sticky=Tk.W )

        j += 1
        self.autorun_button = SlimButton( op_option_frame, text='Auto-Run Start', command=self.controller.auto_start )
        self.autorun_button.grid( row=0, column=j, sticky=Tk.W )
        ToolTip( self.autorun_button, "Select 'Automatic' and press this button to start automatic operation." )

        self.previous_syn_folder    = None
        if not is_empty_val(syn_folder):
            self.on_entry_syn_folder()

        self.add_focus_event_binds()

        self.add_dnd_bind()
        self.during_init = False

    def after_construction_proc( self ):
        """
            このメソッドは常に __init__ に続けて実行するものであるが、
            テストの場合、mainloop に入った後に実行する必要が生じることがあるので
            分離している。
        """
        # 入力項目の初期値をチェックし、従属する処理があれば実行する。
        check_result = self.check_entries()
        print( 'check_result=', check_result )

        self.previous_in_folder     = None
        self.previous_log_file      = None
        self.previous_mask_file     = None
        self.previous_adj_folder    = None

        if check_result:
            # TODO: refactoring on_entry's
            self.on_entry_in_folder( event=None, check_only=True )
            self.on_entry_log_file()
            self.on_entry_mask_file()
            if self.adj_output == 'YES':
                self.on_entry_adj_folder()
            self.on_entry_syn_folder()
            # これらの入力値は妥当であり、
            # 初期には、それによってテーブルが初期化されるため、
            # Refresh の注意喚起は不要
            self.controller.refresh_button_suggestion.stop()

        # must be after on_entry's
        self.previous_in_folder     = self.in_folder.get()
        self.previous_log_file      = self.log_file.get()
        self.previous_mask_file     = self.mask_file.get()
        if self.adj_output == 'YES':
            self.previous_adj_folder    = self.adj_folder.get()
        self.previous_syn_folder    = self.syn_folder.get()

        # 初期値設定による変更は変更としない。
        self.changed_after_refresh  = False
        self.changed_ever           = False
        # print '__init__: changed_after_refresh=', self.changed_after_refresh

        # 以降、値の変更を監視する。
        def entry_tracer( *args ):
            # print 'entry_tracer:', 
            self.changed_after_refresh  = True
            self.changed_ever           = True

        self.in_folder.trace ( 'w', entry_tracer )
        self.log_file.trace  ( 'w', entry_tracer )
        self.mask_file.trace ( 'w', entry_tracer )
        if self.adj_output == 'YES':
            self.adj_folder.trace( 'w', entry_tracer )
        self.syn_folder.trace( 'w', entry_tracer )

        def op_option_tracer( *args ):
            op_option_ = self.op_option.get()
            set_setting( 'op_option', op_option_ )
            if op_option_ == 'MANUAL':
                self.auto_run_widgets_disable()
                self.controller.auto_disable( force=True )
            else:
                
                if self.check_entries():
                    self.auto_run_widgets_enable()
                    self.controller.auto_enable()
                else:
                    if not self.in_construction:
                        MessageBox.showinfo( "Not Allowed", "You can't select 'Automatic' until all required entries are filled." )
                        self.op_option.set( 'MANUAL' )
                        self.op_option_buttons[1].event_generate( "<Leave>" )
                        self.op_option_buttons[0].event_generate( "<Button-1>" )
                        self.update()
                        # self.error_entries[0].focus_force()

        self.op_option.trace( 'w', op_option_tracer )

        self.op_option.set( get_setting( 'op_option' ) )

        self.in_construction = False

    def __del__( self ):
        # print( 'EntryFrame.__del__' )
        self.app_logger = None

    def clear( self ):
        self.clear_entries()
        self.check_entries()        # neccesary as of 2015-12-09

        self.previous_in_folder     = ''
        self.previous_log_file      = ''
        self.previous_mask_file     = ''
        if self.adj_output == 'YES':
            self.previous_adj_folder    = ''
        self.previous_syn_folder    = ''
        set_setting( 'in_folder', '' )
        set_setting( 'log_file', '' )
        set_setting( 'mask_file', '' )
        if self.adj_output == 'YES':
            set_setting( 'adj_folder', '' )
        set_setting( 'syn_folder', '' )

    def auto_run_widgets_disable( self ):
        self.watch_interval_label.configure( state='disabled' )
        self.watch_interval_box.configure( state='disabled' )
        self.watch_interval_s.configure( state='disabled' )
        self.autorun_button.configure( state='disabled' )

    def auto_run_widgets_enable( self ):
        self.watch_interval_label.configure( state='normal' )
        self.watch_interval_box.configure( state='normal' )
        self.watch_interval_s.configure( state='normal' )
        self.autorun_button.configure( state='normal' )

    def add_focus_event_binds( self ):
        def suggest_text_and_check( event, text='', proc=None ):
            current = event.widget.get()
            # print 'suggest_text: current=', current, '; entry=', on_entry
            if current == '':
                event.widget.insert( 0, text )
                event.widget.configure( fg='red' )
            elif current[0] == '<':
                event.widget.delete( 0, Tk.END )

            # この proc には、self.entries に登録された on_entry が
            # 渡されている。
            if proc:
                proc()

        for obj_type, entry, on_entry in self.entries:
            current = entry.get()
            # print 'add_suggest_text_bind: current=', current

            if is_empty_val( current ):
                entry.delete( 0, Tk.END )
                if obj_type == 'dir':
                    entry.insert( 0, '<Folder>' )
                else:
                    entry.insert( 0, '<File>' )
            else:
                pass

            entry.bind( "<FocusIn>",    lambda event: suggest_text_and_check( event ) )

            # 以下の lambda では、on_entry を値渡しするために default argument を使用している。
            # 参照URL: http://stackoverflow.com/questions/10452770/python-lambdas-binding-to-local-values
            # また、lambda 表現の制限により、suggest_text_and_check に２つの処理を実装している。
            if obj_type == 'dir':
                entry.bind( "<FocusOut>",   lambda event, proc_value=on_entry:
                                                suggest_text_and_check( event, '<Folder>', proc_value ) )
            else:
                entry.bind( "<FocusOut>",   lambda event, proc_value=on_entry:
                                                suggest_text_and_check( event, '<File>', proc_value ) )

    def check_entries( self, index=-999 ):
        """
            Drag and Drop の場合はフォルダやファイルを指定できないので、
            （存在はしても）違ったタイプのパスが設定されることがある点に
            注意。
        """
        # print( 'check_entries: index=', index )
        msg = ''
        ret = True
        if index == -999:
            self.error_entries = []

        if index >= -1:
            entries_ = [ self.entries[index] ]
        else:
            entries_ = self.entries

        error_index = None
        i_ = -1
        for obj_type, entry, on_entry in entries_:
            i_ += 1
            path = entry.get()
            entry_ok = True
            # print 'Checking if %s is a %s' % ( path, obj_type )
            if path and os.path.exists( path ):
                if obj_type == 'dir':
                    if os.path.isdir( path ):
                        entry.configure( fg='black' )
                    else:
                        if msg == '':
                            error_index = i_
                            msg = '%s is not a folde.r' % ( path )
                        entry.configure( fg='red' )
                        ret = False
                        entry_ok = False
                elif obj_type == 'file':
                    if os.path.isfile( path ):
                        entry.configure( fg='black' )
                    else:
                        if msg == '':
                            error_index = i_
                            msg = '%s is not a file.' % ( path )
                        entry.configure( fg='red' )
                        ret = False
                        entry_ok = False
                else:
                    pass

            else:
                if not is_empty_val( path ):
                    if msg == '':
                        error_index = i_
                        msg = '%s does not exist.' % ( path )

                entry.configure( fg='red' )
                ret = False
                entry_ok = False

            if index == -999:
                if not entry_ok:
                    self.error_entries.append( entry )

        if msg:
            if index == -999:
                if error_index == len( entries_ ) - 1:
                    # Synthesized Image Folder is referred to as -1
                    error_index = -1
                self.error_notified[ ( error_index, msg ) ] = True
                reply = MessageBox.askyesno( 'Entry Error & Clear Info Question',
                                msg
                                + "\nLooks like setting environment has been changed."
                                + "\nDo you want to clear the previous setting info?"
                                )
                # print( reply )
                if reply:
                    clear_settings()
                    self.clear_entries()
                    self.check_entries()
                else:
                    pass
            else:
                if self.error_notified.get( ( index, msg ) ) is None:
                    MessageBox.showerror( 'Entry Error', msg )
                self.error_notified[ ( index, msg ) ] = True

        # print "check_entries: ret=", ret
        if ret:
            if index == -999:
                if get_mask() == None:
                    # mask が設定されていない場合がありうるので、
                    # 強制的に設定する。
                    print( 'mask is not set yet' )
                    self.on_entry_mask_file( set_force=True )
                self.data_entry_ready = True
                if self.op_option.get() == 'MANUAL':
                    if self.controller.op_is_manual:
                        print( '(1) calling controller.auto_disable()' )
                        self.controller.auto_disable()
                else:
                    if not self.controller.op_is_manual:
                        print( '(2) calling controller.auto_enable()' )
                        self.controller.auto_enable()
            else:
                pass

            # if get_setting( 'op_option' ) == 'MANUAL':
            # TODO: set_setting( 'op_option' )
            if self.op_option.get() == 'MANUAL':
                all_entries_are_filled = True
                for obj_type, entry, on_entry in self.entries:
                    if is_empty_val( entry.get() ):
                        all_entries_are_filled = False
                        break
                if all_entries_are_filled:
                    self.controller.refresh_button_enable()
                    if self.changed_after_refresh:
                        self.controller.refresh_button_suggestion.start()
        else:
            self.data_entry_ready = False
            if self.controller.op_is_manual:
                print( '(3) calling controller.auto_disable()' )
                if self.during_init:
                    pass
                else:
                    self.controller.auto_disable()
                    self.controller.manual_buttons_disable()
                    self.controller.refresh_button_suggestion.stop()

        print( 'check_entries: index=%d, ret=%s' % ( index, ret ) )

        return ret

    def clear_entries( self ):
        for obj_type, entry, on_entry in self.entries:
            entry.delete( 0, Tk.END )
            # TODO: consistency in the empty prompts
            if obj_type == 'dir':
                entry.insert( 0, '<Folder>' )
            else:
                entry.insert( 0, '<File>' )

    def add_dnd_bind( self ):
        self.in_folder_entry.register_drop_target("*")
        self.log_file_entry.register_drop_target("*")
        self.mask_file_entry.register_drop_target("*")
        self.syn_folder_entry.register_drop_target("*")

        def dnd_handler( event ):
            event.widget.delete( 0, Tk.END )
            data = re.sub( r'({|})', '', event.data )
            event.widget.insert( 0, data )

        def dnd_handler_in_folder( event ):
            dnd_handler( event )
            self.on_entry_in_folder()

        def dnd_handler_log_file( event ):
            dnd_handler( event )
            self.on_entry_log_file()

        def dnd_handler_mask_file( event ):
            dnd_handler( event )
            self.on_entry_mask_file()

        def dnd_handler_adj_folder( event ):
            dnd_handler( event )
            self.on_entry_adj_folder()

        def dnd_handler_syn_folder( event ):
            dnd_handler( event )
            self.on_entry_syn_folder()

        self.in_folder_entry.bind("<<Drop>>", dnd_handler_in_folder)
        self.log_file_entry.bind("<<Drop>>", dnd_handler_log_file)
        self.mask_file_entry.bind("<<Drop>>", dnd_handler_mask_file)

        if self.adj_output == 'YES':
            self.adj_folder_entry.register_drop_target("*")
            self.adj_folder_entry.bind("<<Drop>>", dnd_handler_adj_folder)
        self.syn_folder_entry.bind("<<Drop>>", dnd_handler_syn_folder)

        for obj_type, entry, on_entry in self.entries:
            ToolTip( entry, 'You can enter directly, or use the right file dialog button, otherwise, drag and drop here.')

    def askdirectory( self, entry_variable, suggest_name=False ):
        entered_path = entry_variable.get()
        # print 'entered_path=', entered_path
        dir_ = os.path.dirname( entered_path ).replace( '/', '\\' )
        if suggest_name:
            if is_empty_val( entered_path ):
                dialog = CreateFolderDialog( self, 'Folder Creation Dialog', self.suggested_folders( suggest_name ) )
                if dialog.created_folder:
                    return dialog.created_folder

        # print 'dir_=', dir_
        f = FileDialog.askdirectory( initialdir=dir_ )
        return f

    def suggested_folders( self, name ):
        folders = []
        in_folder = self.in_folder.get()
        if not is_empty_val( in_folder ):
            in_path = os.path.abspath( in_folder )
            f = os.path.join( in_path, name )
            folders.append( f )
            f = in_path + '-' + name
            folders.append( f )

        cwd_ = os.getcwd()

        drive_ = cwd_.split( '\\' )[0]
        f = os.path.join( drive_ + '\\', name )
        folders.append( f )

        ret_foleds = []
        for f in folders:
            ret_foleds.append( f.replace( '\\', '/' ) )
        return ret_foleds

    def select_in_folder( self ):
        f = self.askdirectory( self.in_folder )
        if not f:
            return

        self.in_folder.set( f )
        print( 'select_in_folder: self.on_entry_in_folder call' )
        self.on_entry_in_folder()

    def on_entry_in_folder( self, event=None, check_only=False ):
        # print( 'on_entry_in_folder: check_only=', check_only )
        self.auto_set_asked = False
        self.auto_set_done  = False

        in_folder = self.in_folder.get()

        if not self.check_entries( 0 ):
            self.previous_in_folder       = in_folder
            return

        if check_only:
            return

        if self.previous_in_folder and in_folder == self.previous_in_folder:
            # 値が同じならば以下の処理は不要
            return
        self.previous_in_folder       = in_folder
        set_setting( 'in_folder', in_folder )

        in_folder = self.in_folder.get()
        log_file, mask_file = get_in_folder_info( in_folder )

        do_auto_fill = True
        if ( log_file or mask_file ):
            if is_empty_val( self.log_file.get() ):
                action_ = 'Insert'
            else:
                action_ = 'Replace'
            self.auto_set_asked = True
            yes = MessageBox.askyesno( 'Quention', 'Log and/or mask files exist in the folder. %s the corresponding entries?' % (action_) )
            if not yes:
                do_auto_fill =False

        if do_auto_fill:
            if log_file:
                log_file_   = '%s/%s' % ( in_folder, log_file )
                self.log_file.set( log_file_ )
                self.auto_set_done  = True
                self.on_entry_log_file()
            if mask_file:
                mask_file_  = '%s/%s' % ( in_folder, mask_file)
                self.auto_set_done  = True
                self.mask_file.set( mask_file_ )
                self.on_entry_mask_file()

        if self.controller.image_info_table is not None:
            # print( 'on_entry_in_folder: current_data_end=', self.controller.image_info_table.current_data_end )
            # in_folder が更新されたので、カレントをリセットする。
            self.controller.image_info_table.current_data_end = 0

    def select_log_file(self):
        f = FileDialog.askopenfilename( initialdir=get_setting( 'in_folder' ) )
        if not f:
            return

        self.log_file.set( f )
        self.on_entry_log_file()

    def on_entry_log_file( self ):
        f = self.log_file.get()
        if not self.check_entries( 1 ):
            self.previous_log_file = f
            return

        if self.previous_log_file and f == self.previous_log_file:
            # 値が同じならば以下の処理は不要
            return

        self.previous_log_file = f
        self.pilatus_counter = PilatusCounter.Counter( self.in_folder.get() )
        set_setting( 'log_file', f )

    def select_mask_file(self):
        f = FileDialog.askopenfilename( initialdir=get_setting( 'in_folder' ) )
        if not f:
            return

        self.mask_file.set( f )
        self.on_entry_mask_file()

    def on_entry_mask_file( self, set_force=False ):
        f = self.mask_file.get()

        if not self.check_entries( 2 ):
            self.previous_mask_file = f
            return

        if not set_force and self.previous_mask_file and f == self.previous_mask_file:
            # 値が同じならば以下の処理は不要
            return

        self.previous_mask_file = f
        try:
            print( 'on_entry_mask_file: set_mask, set_force=', set_force )
            if set_mask( f ):
                self.mask_file_entry.configure( fg='black' )
                print( get_mask() )
            else:
                # TODO: get the error msg from SAnglerMask
                self.mask_file_entry.configure( fg='red' )
                MessageBox.showerror( 'SAngler Mask File Error', 'Not a well formatted SAngler mask file' )
                return False

            set_setting( 'mask_file', f )
        except Exception as err:
            print( 'on_entry_mask_file: unexpected exception', err.message )
            return False

        return True

    def select_adj_folder(self):
        f = self.askdirectory( self.adj_folder, suggest_name='Adjusted' )
        if not f:
            return
        self.adj_folder.set( f )
        self.on_entry_adj_folder()

    def on_entry_adj_folder( self ):
        f = self.adj_folder.get()
        if not self.check_entries( 3 ):
            self.previous_adj_folder    = f
            return

        if self.previous_adj_folder and f == self.previous_adj_folder:
            # 値が同じならば以下の処理は不要
            return

        self.previous_adj_folder = f
        set_setting( 'adj_folder', f )

    def select_syn_folder(self):
        f = self.askdirectory( self.syn_folder, suggest_name='Synthesized' )
        if not f:
            return

        self.syn_folder.set( f )
        self.on_entry_syn_folder()

    def on_entry_syn_folder( self ):
        f = self.syn_folder.get()
        if not self.check_entries( -1 ):
            self.previous_syn_folder = f
            return

        if self.previous_syn_folder and f == self.previous_syn_folder:
            # 値が同じならば以下の処理は不要
            return

        self.previous_syn_folder = f
        self.on_entry_syn_folder_sub()

    def on_entry_syn_folder_sub( self ):
        syn_folder = self.syn_folder.get()
        if is_empty_val( syn_folder ):
            return True

        set_setting( 'syn_folder', syn_folder )
        exe_name_ =  exe_name()
        self.logfile_path = '%s/%s.log' % ( syn_folder, exe_name_ )
        if self.app_logger:
            self.app_logger.changeto( self.logfile_path )
        else:
            self.app_logger = None
            self.app_logger = ChangeableLogger.Logger( self.logfile_path )

        print( 'on_entry_syn_folder_sub: return' )
        return True

    def change_reset( self ):
        self.changed_after_refresh   = False

    def has_been_changed_after_refresh( self ):
        # print 'has_been_changed_after_refresh: changed_after_refresh=', self.changed_after_refresh
        return self.changed_after_refresh

    def has_been_changed_ever( self ):
        return self.changed_ever

    """
        以下は、AutoRunController から起動した ActionWindow が、
        終了時（OK を押したとき）に grab が効かない状態になるための
        暫定措置として使用する。（ grab が有効ならば不要 ）
    """
    def disable( self ):
        for obj_type, entry, on_entry in self.entries:
            entry.configure( state='disabled' )

        for button in self.op_option_buttons:
            button.configure( state='disabled' )

        if self.op_option.get() == 'MANUAL':
            # manual の場合 grab が有効のため、ここは未使用。
            self.controller.refresh_button_disable()
            self.controller.run_button_disable()
        else:
            self.auto_run_widgets_disable()

    def enable( self ):
        print( 'EntryFrame: enable' )
        for obj_type, entry, on_entry in self.entries:
            entry.configure( state='normal' )

        for button in self.op_option_buttons:
            button.configure( state='normal' )

        if self.op_option.get() == 'MANUAL':
            # manual の場合 grab が有効のため、ここは未使用。
            self.controller.refresh_button_enable()
            self.controller.run_button_enable()
        else:
            self.auto_run_widgets_enable()
