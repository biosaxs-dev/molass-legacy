# -*- coding: utf-8 -*-
"""

    ファイル名：   GuiController.py

    処理内容：

        GUI 制御クラス

        Controller
            EntryFrame
            PilatusImageInfoTable
                PilatusImageViewer
            PreferencesDialog

        Copyright (c) 2016-2020, SAXS Team, KEK-PF

"""
import sys
import os
import re
import ExecutionWindow
import platform
from molass_legacy.KekLib.OurTkinter             import Tk, ToolTip
from molass_legacy.KekLib.TkUtils                import adjusted_geometry, split_geometry
import OurMessageBox        as     MessageBox
from GuiSettingInfo         import EntryFrame
from GuiPreferences         import PreferencesDialog
from Development            import DeveloperOptionsDialog
from molass_legacy.KekLib.TkSupplements          import BlinkingFrame, tk_set_icon_portable, SlimButton
from ImageSynthesizer       import ImageSynthesizer
from PersistentInfo         import PersistentInfo
from Preferences            import get_preference
from SynthesizerSettings    import save_settings, get_setting, set_setting
from molass_legacy.AutorgKek.AppVersion             import synthesizer_version_string
from AutoRunController      import AutoRunController

# label_bg_color = 'dim gray'
label_bg_color = 'gray25'
label_fg_color = 'white'
window_height = 600

class Controller( Tk.Toplevel ):
    def __init__( self, root, opts=None ):

        root.wm_title( synthesizer_version_string() )

        Tk.Toplevel.__init__( self, root )
        self.withdraw()

        """
            テスト制御の都合により、
            上で CommandLineOptions を import しないで、
            opts を引数で受け取る。
        """
        if opts and opts.pandastable or os.name != 'nt':
            from PilatusImageInfoTable2 import PilatusImageInfoTable
        else:
            from PilatusImageInfoTable  import PilatusImageInfoTable

        self.op_is_feasible = False
        self.op_is_manual   = True
        self.refresh_button_suggestion = None
        self.image_info_table = None
        self.geometry_resized = False

        # メイン画面を構築する。
        self.root = root
        # self.root.option_add( '*Font', get_preference( 'font' ) )
        # self.root.option_add( '*Font', ( 'TkDefaultFont', 10 ) )
        # self.root.option_add( '*Font', ( 'TkFixedFont', 9 ) )
        # self.root.option_add( '*ScrolledText.Font', ( 'DotumChe', 10 ) )
        # self.root.option_add( '*Font', ('FixedSys', 9) )
        # self.root.option_add( '*Font', ('Arial', 10) )
        # self.root.option_add( '*Font', ('DotumChe', 10) )
        # self.root.option_add( '*Font', ('Lucida Console', 9) )
        # self.root.option_add( '*Font', ('System', 10) )
        # self.root.option_add( '*Font', ('Arial Unicode MS', 10) )
        # self.root.option_add( '*Font', ('MS Sans Serif', 9) )
        # self.root.option_add( '*Font', ('Microsoft Sans Serif', 10) )
        # self.root.option_add( '*Font', ('ＭＳ Ｐゴシック', 10) )
        # self.root.option_add( '*Font', ('MS UI Gothic', 10) )
        # self.root.option_add( '*Font', ('Lucida Sans Unicode', 10) )

        tk_set_icon_portable( self, 'synthesizer' )

        menubar = Tk.Menu( self )
        self.config( menu=menubar )
        menu = Tk.Menu( menubar, tearoff=0 )
        menubar.add_cascade(label="Options", menu=menu )
        menu.add_command( label="Preferences", command=self.preferences )
        menu.add_command( label="Developer Options", command=self.developer_options )
        menu.add_command( label="Exit", command=self.quit )

        help = Tk.Menu( menubar, tearoff=0 )
#        menubar.add_cascade( label="Help", menu=help )
#        help.add_command( label="Help",     command=self.help )
#        help.add_command( label="About",    command=self.about )

        self.frame_inner = Tk.Frame( self )
        self.frame_inner.pack( side=Tk.TOP, fill=Tk.BOTH, expand=1, padx=10, pady=10 )

        setting_info_label_frame = Tk.Frame( self.frame_inner )
        setting_info_label_frame.pack( side=Tk.TOP, anchor=Tk.W )
        setting_info_label = Tk.Label( setting_info_label_frame, text= ' Setting Information Entries ', relief=Tk.FLAT, fg=label_fg_color, bg=label_bg_color )
        setting_info_label.pack( side=Tk.LEFT, anchor=Tk.W )
        self.clear_button = SlimButton( setting_info_label_frame, text='Clear', command=self.setting_clear )
        self.clear_button.pack( side=Tk.LEFT, anchor=Tk.W, padx=10 )
        clear_button_guide = Tk.Label( setting_info_label_frame, text= 'press to clear the follwing entries' )
        clear_button_guide.pack( side=Tk.LEFT, anchor=Tk.W )
        ToolTip( self.clear_button, 'Press this button to clear the setting infomation.' )

        self.entry_frame_frame = Tk.Frame( self.frame_inner )
        self.entry_frame_frame.pack()
        self.entry_frame = EntryFrame( self.entry_frame_frame, self )
        self.entry_frame.pack( side=Tk.TOP, padx=10, pady=10 )

        detector_image_info_frame = Tk.Frame( self.frame_inner )
        detector_image_info_frame.pack( side=Tk.TOP, anchor=Tk.W )

        detector_image_info_label = Tk.Label( detector_image_info_frame, text= ' Image Data Information Table ', relief=Tk.FLAT, fg=label_fg_color, bg=label_bg_color )
        detector_image_info_label.pack( side=Tk.LEFT, anchor=Tk.W )

        self.refresh_button = SlimButton( detector_image_info_frame, text='Refresh', command=self.refresh )
        self.refresh_button_guide = Tk.Label( detector_image_info_frame, text='' )

        ToolTip( self.refresh_button, 'Press this button to refresh the measured data infomation listed below.' )

        object_spec_array = []
        object_spec_array.append( [ self.refresh_button ,
                                     { "side" : Tk.LEFT, "anchor" : Tk.W, "padx" : 10 } ] )

        object_spec_array.append( [ self.refresh_button_guide,
                                     { "side" : Tk.LEFT, "anchor" : Tk.W} ] )

        self.refresh_button_suggestion = BlinkingFrame( detector_image_info_frame, object_spec_array,
                                            start_proc=self.run_button_disable,
                                            stop_proc =self.run_button_enable, debug=True )

        self.refresh_button_suggestion.pack( side=Tk.LEFT, anchor=Tk.W, padx=10 )

        self.run_button = SlimButton( detector_image_info_frame, text='Run', command=self.run, state='disabled' )
        self.run_button.pack( side=Tk.LEFT, padx=10 )
        self.run_button_guide    = Tk.Label( detector_image_info_frame, text='' )
        self.run_button_guide.pack( side=Tk.LEFT )
        ToolTip( self.run_button, 'Press this button to make synthesized images for the selected samples.' )

        table_frame = Tk.Frame( self.frame_inner )
        table_frame.pack( expand=1, fill=Tk.BOTH, padx=10, pady=10, anchor=Tk.N )

        self.synthesizer = ImageSynthesizer( table_frame )

        self.image_info_table = PilatusImageInfoTable( table_frame, self.synthesizer, self.refresh_button_suggestion )

        if self.image_info_table.num_selected_rows > 0:
            self.run_button_enable()

        # refresh_button や run_button の state を変更するかもしれないので、
        # ここで呼び出す。
        self.entry_frame.after_construction_proc()

        # getting ready to show
        self.update()
        self.deiconify()
        self.geometry( adjusted_geometry( self.geometry() ) )

        self.protocol( "WM_DELETE_WINDOW", self.quit )

    def __del__( self ):
        self.entry_frame.__del__()

    def setting_clear( self ):
        ok = MessageBox.askyesno( 
                        'Confirmation',
                        'Do you want to clear the follwing entries?'
                        )
        if not ok:
            return
        self.entry_frame.clear()

    def refresh_button_disable( self ):
        print( 'refresh_button_disable' )
        self.refresh_button.configure( state='disabled' )
        self.refresh_button_guide.configure( text='' )

    def refresh_button_enable( self ):
        print( 'refresh_button_enable' )
        self.refresh_button.configure( state='normal' )
        self.refresh_button_guide.configure( text='← be sure to press after you have changed above entries' )

    def refresh(self, autorun=False):
        if self.entry_frame.check_entries():
            need_resize = self.image_info_table.refresh( log_file_path=self.entry_frame.log_file.get(),
                                            restore_view = not self.entry_frame.has_been_changed_after_refresh(),
                                            logger=self.get_logger(), autorun=autorun )
            if need_resize:
                if not self.geometry_resized:
                    """
                    this is to fix the bug which hides all the rows that should have been shown.
                    """
                    curr_geometry = self.geometry()
                    w, h, x, y = split_geometry( curr_geometry )
                    height_increment = self.image_info_table.get_resize_height()
                    new_height = h + height_increment
                    height = min(new_height, window_height)
                    new_geometry = "%dx%d+%d+%d" % (w, height, x, y)
                    self.geometry(new_geometry)
                    self.geometry_resized = height == window_height

            self.refresh_button_suggestion.stop()
            self.entry_frame.change_reset()
        else:
            MessageBox.showerror( 'Required Entries', 'Please fill in required entries.' )

    def run_button_disable( self ):
        self.run_button.configure( state='disabled' )
        self.run_button_guide.configure( text='' )

    def run_button_enable( self ):
        self.run_button.configure( state='normal' )
        self.run_button_guide.configure( text='← press to make synthesized images' )

    def manual_buttons_disable( self ):
        self.refresh_button_disable()
        self.run_button_disable()

    def run( self ):
        self.image_info_table.do_action( 3 )

    def auto_start( self ):
        print( 'auto_start' )
        ok = MessageBox.askokcancel( 
                        'Confirmation',
                        'Do you want to start automatic control?'
                        )
        if not ok:
            return

        interval = int( self.entry_frame.watch_interval.get() )
        # 実行した watch_interval を永続化の候補とする。
        set_setting( 'watch_interval', interval )
        self.grab_set_alternative()
        self.ar_controller = AutoRunController( self, interval, self.image_info_table,
                                                log_file_path=self.entry_frame.log_file.get(),
                                                on_stop=self.grab_release_alternative )
        self.ar_controller.start()

    def grab_set_alternative( self ):
        # print( 'grab_set_alternative' )
        self.entry_frame.disable()

    def grab_release_alternative( self ):
        # print( 'grab_release_alternative' )
        self.entry_frame.enable()

    def auto_disable( self, force=False ):
        print( 'auto_disable' )
        if force or not self.op_is_manual:
            self.op_is_manual = True
            self.refresh_button_enable()
            self.run_button_enable()

    def auto_enable( self, force=False  ):
        print( 'auto_enable' )
        # TODO: check ok
        if force or self.op_is_manual:
            self.op_is_manual = False
            self.refresh()
            self.refresh_button_disable()
            self.run_button_disable()

    def quit( self ):
        if not MessageBox.askyesno("Quit", "Do you want to quit?"):
            return

        if self.entry_frame.has_been_changed_ever():
            save_policy = get_preference( 'save_policy' )
            do_save = False
            if save_policy == 'Ask':
                if MessageBox.askyesno("Entry Data Handling Question", "Do you want to save the folder/file entries?"):
                    do_save = True
            elif save_policy == 'Yes':
                do_save = True

            if do_save:
                save_settings()

        else:
            # 全く変更されていない場合は、保存しない。
            pass

        self.root.quit()    # stops mainloop
        self.destroy()

    def preferences( self ):
        if self.refresh_button_suggestion and self.refresh_button_suggestion.is_blinking():
            MessageBox.showwarning( "Not Allowed", "You can't change preferences until you refresh the image data infomation table." )
            return

        dialog = PreferencesDialog( self.root, "Usual Preferences", self.entry_frame.pilatus_counter )

        if dialog.adj_output_changed:
            self.entry_frame.__del__()
            self.entry_frame.destroy()
            self.entry_frame = EntryFrame( self.entry_frame_frame, self )
            self.entry_frame.pack( side=Tk.TOP, padx=10, pady=10 )
            self.entry_frame.after_construction_proc()

        if dialog.couter_id_changed:
            self.image_info_table.refresh(  log_file_path=self.entry_frame.log_file.get(),
                                            restore_view=True, pilatus_counter=self.entry_frame.pilatus_counter,
                                            logger=self.get_logger() )

        return

    def get_logger(self):
        return self.entry_frame.app_logger

    def developer_options( self ):
        DeveloperOptionsDialog( self.root, "Developer Options" )
        return
    """
    def help( self ):
        print( 'help' )
        return

    def about( self ):
        print( 'about' )
        return
    """
