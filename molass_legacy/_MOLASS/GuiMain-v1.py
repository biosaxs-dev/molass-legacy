"""

    GuiMain.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF

"""
import sys
import re
import os
import time
import warnings
from molass_legacy.KekLib.EnvironCheck import executables_check
from molass_legacy.KekLib.OurMatplotlib import mpl_1_5_backward_compatible_init, mpl_font_init
mpl_1_5_backward_compatible_init()
mpl_font_init()

from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker, log_exception
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry, is_almost_empty_dir
from molass_legacy.KekLib.ChangeableLogger import Logger
from molass_legacy.KekLib.OurTkinter import Tk, ttk, FileDialog, Font, is_empty_val
from molass_legacy.KekLib.TkUtils import adjusted_geometry, PositionSynchronizer
from molass_legacy.KekLib.TkSupplements import SlimButton, BlinkingFrame
from molass_legacy.KekLib.TkSizableWidgets import SizableLabel, SizableEntry, BlinkingLabel
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry, FileEntry
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import molass_legacy.KekLib.OurMessageBox as MessageBox
print( 'import molass_legacy.KekLib.CustomMessageBox ok' )
from molass_legacy.KekLib.RecentFolders import RecentFolders
import Decomposer, DataStructure, Extrapolation, GuinierAnalyzer
from Menus.GuiSettings import GuiSettingsMenu
from Menus.GuiSecTools import GuiSecToolsMenu

from Menus.GuiDenssTools import GuiDenssToolsMenu
from Menus.GuiTutorials import GuiTutorialsMenu
from Menus.GuiReferences import GuiReferencesMenu
from Menus.GuiDevelopment import GuiDevelopmentMenu

from molass_legacy.SerialAnalyzer.SerialDataLoader import SerialDataLoader
from molass_legacy.SerialAnalyzer.SerialDataUtils import get_uv_filename, get_mtd_filename
from molass_legacy.DataStructure.MeasuredData import MeasuredData
from molass_legacy.Trimming.OutlineFigure import OutlineFigure
from molass_legacy.SerialAnalyzer.AbnormalityCheck import exclude_abnormality
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.QuickAnalysis.Analyzer import Analyzer
from molass_legacy._MOLASS.Version import get_version_string
from Rank.RankView import RankViewMenu
from GuiParts.UvFolderEntry import UvFolderEntry
from molass_legacy._MOLASS.SerialSettings import ( initialize_settings,
                                    get_setting, set_setting, save_settings,
                                    clear_temporary_settings, restore_default_setting,
                                    get_settings_folder )
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting, set_dev_setting
import AutorgKek.Settings

analysis_name_re    = re.compile( r'^(.*?)(\d+)(.*)$' )
distance_symbol_re  = re.compile( r'^(\S+)\s' )

label_bg_color = 'gray40'
label_fg_color = 'white'
label_gb_color_var = 'white smoke'
relief_var = Tk.GROOVE
radio_button_text_length = 20

angular_unit_guide = "; this unit is used only for display purpose."

class GuiMain( Tk.Toplevel ):
    def __init__(self, parent, pid=None, debug=0):
        from molass_legacy._MOLASS.Debug import DebugInfo
        dpi_aware = False
        self.debug_info = DebugInfo(debug=debug)
        self.clear_test_mode()

        self.parent_pid = pid
        self.cleaner = None

        """
            dpi_aware is for compatibility with V2
            and not used
        """

        self.testing = False
        if not executables_check(parent):
            parent.after(100, parent.quit)
            return

        initialize_settings()

        self.single_instance = None     # this is required in case of exiting within the following check
        from molass_legacy._MOLASS.AppSingleInstance import single_instance_check
        app_lock_path = get_settings_folder() + '\\lock'
        self.single_instance = single_instance_check(app_lock_path)

        self.default_font = Font.nametofont("TkDefaultFont")
        self.hiresolution   = get_dev_setting( 'hiresolution' )
        self.high_dpi = False

        if self.hiresolution:
            self.default_font.configure(size=16)
            parent.option_add("*Font", self.default_font)
            self.slimbutton = False
            fixed_fontsize  = 16
        else:
            self.slimbutton = True
            fixed_fontsize  = 9

        self.fixed_font = Font.Font( family="Courier", size=fixed_fontsize )

        warnings.filterwarnings("ignore")

        parent.wm_title( get_version_string() + " pid:%d" % os.getpid() )
        parent.report_callback_exception = self.report_callback_exception

        if '-i' in sys.argv:
            from molass_legacy._MOLASS.SerialSettings   import clear_settings
            clear_settings()

        self.parent = parent
        self.tmp_logger = Logger()  # this logger logs into string stream
        clear_temporary_settings()
        self.serial_data = None
        self.measured_data = None

        self.loader     = SerialDataLoader(dialog=self)
        self.analyzer   = Analyzer( self, self.loader )
        self.intial_geometry = None
        self.use_mtd_conc = False
        self.is_executing   = False
        self.dataset_is_ready = False
        self.preproc_dialog = None
        self.keeping_setting_items = None

        Tk.Toplevel.__init__( self, parent )
        self.withdraw()

        if get_setting('enable_debug_plot'):
            from molass_legacy.KekLib.DebugPlot import set_plot_env
            set_plot_env(self)

        self.build_window()
        if '--close_immediately' in sys.argv:
            print( 'closing immediately' )
            self.quit( immediately=True )

        self.after(100, self.init_proc)
        self.pos_sync = PositionSynchronizer(self, self.parent)

    def __del__(self):
        if self.single_instance is not None:
            self.single_instance.clean_up()

    def set_test_mode(self, tester_info):
        self.testing = True
        self.tester_info = tester_info
        set_setting("test_pattern", tester_info.test_pattern)

    def clear_test_mode(self):
        self.testing = False
        self.tester_info = None
        set_setting("test_pattern", None)

    def init_proc(self):
        # initial proc done after GuiMain.__init__
        for k in range(2):
            try:
                self.check_environment()
                self.make_it_more_clever()
                break
            except:
                # ocurred once when D&D in a not ready state
                pass
            time.sleep(1)

        self.after(1000, self.update_menu_states)   # to ensure DENSS Tools menu states update

    def check_environment( self ):
        from molass_legacy.Env.EnvInfo import EnvInfo, set_global_env_info
        env_info = EnvInfo()
        env_info.show_and_log_if_unavailable(self, self.tmp_logger)
        self.env_info = env_info
        set_global_env_info(env_info)

    def make_it_more_clever(self):
        from molass_legacy.SerialAnalyzer.MakeItMoreClever import increase_menu_availability
        increase_menu_availability(self)

    def is_busy( self, auto_apply=True ):
        if auto_apply:
            try:
                if self.analyzer.cancel_reports:
                    # this case is for testing only
                    dialog = self.analyzer.dialog
                    if dialog is None:
                        pass
                else:
                    dialog = self.analyzer.progress_dialog
                    if dialog is None:
                        pass
                    else:
                        text = dialog.button.cget('text')
                        if text == 'OK':
                            self.analyzer.progress_dialog.button.invoke()
            except:
                etb = ExceptionTracebacker()
                print(etb)
                pass
        return self.is_executing

    def report_callback_exception(self, exc, val, tb):
        # This method is to override the Tk method to be able
        # to report to the spawned console in windows application mode.

        if self.tmp_logger is None:
            self.tmp_logger = Logger()
        logger = self.tmp_logger

        etb = ExceptionTracebacker()
        msg = ( 'Overridden report_callback_exception: ' + str( etb )
                + '\n---- %s ----\n' % get_version_string(cpuid=True)
                + 'This message is saved either to "%s"\n' % logger.get_final_log_path()
                + 'or to "molass.log" in the analysis report folder.'  )

        logger.error( msg )
        self.tmp_logger = logger = Logger()     # this line invokes Logger's destructor
                                                # and recovers another logger in case
                                                # the execution continues
        """
            this is necessary for
            some modules such as scipy.interpolate seem to exit
            without giving any chance to the destructor
        """

        if not self.analyzer.logger_visible:
            try:
                MessageBox.showerror( "Tkinter Error", msg, parent=self.parent )
            except:
                # _tkinter.TclError: bad window path name ".!guimain"
                pass

    def adjust_geometry( self ):
        self.update()
        self.deiconify()
        if self.intial_geometry is None:
            self.geometry( adjusted_geometry( self.geometry() ) )
            self.intial_geometry = self.geometry()

    def build_window( self ):
        from GuiParts.FileInfoTable import FileInfoTable    # must be imported after initialize_settings

        global radio_button_text_length

        lib_dir = os.path.dirname( os.path.dirname( os.path.abspath( __file__ ) ) )
        iconpath = lib_dir + '/_MOLASS/molamola.ico'
        self.iconbitmap(default=iconpath)           # this default argument key is for Windows only

        # TODO: recover the previous state
        # main_geometry = get_setting( 'main_geometry' )
        # print( main_geometry )
        # self.geometry( main_geometry )
        # self.geometry( default_main_geometry )

        # temp_label = DynamicLabel(self)
        # fixed_font = temp_label.get_fixed_font()
        # fixed_font = Font.Font(family="FixedSys", size=8 )
        # fixed_font = Font.Font( family="ＭＳ ゴシック", size=10 )
        # fixed_font = Font.Font( family="Lucida Console", size=10 )
        # fixed_font = Font.Font( family="DotumChe", size=10 )
        # fixed_font = Font.Font( family="Consolas", size=10 )
        self.fixed_font = Font.Font( family="Courier", size=9 )

        self.create_menus()

        # basic Layout
        top_space = Tk.Frame( self, height=20 )
        top_space.pack()

        upper_frame = Tk.Frame( self )
        upper_frame.pack( fill=Tk.BOTH, padx=20 )

        lower_frame = Tk.Frame( self )
        lower_frame.pack( fill=Tk.BOTH, expand=1, padx=20 )

        # layout params
        section_label_width = 20
        button_width = 12
        state_show_width = 23
        state_show_sizable_width    = 164
        state_show_sizable_height   = 20
        folder_entry_width          = 60

        grid_row = -1

        # Input Information --------------------------------------------------------
        grid_row += 1
        column_adjust  = Tk.Frame( upper_frame, width=160, height=0 )
        column_adjust.grid( row=grid_row, column=0 )

        input_label  = Tk.Label( upper_frame, text= 'Input', width=section_label_width,
                                    relief=Tk.FLAT, fg=label_fg_color, bg=label_bg_color )
        input_label.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        in_folder_label  = Tk.Label( upper_frame, text= 'Xray Scattering Data Folder: ' )
        in_folder_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.in_folder = Tk.StringVar()
        self.in_folder.set( get_setting( 'in_folder' ) )
        self.in_folder_entry = FolderEntry( upper_frame, textvariable=self.in_folder, width=folder_entry_width,
                                            slimbutton=self.slimbutton,
                                            on_entry_cb=self.on_entry_in_folder )
        self.in_folder_entry.grid( row=grid_row, column=1, sticky=Tk.W )
        self.previous_in_folder = None
        self.previous_uv_file = None
        self.uv_file_info_changed = False
        self.cleared_uv_folder  = False

        file_extension_label  = Tk.Label( upper_frame, text= '  File Ext.: ' )
        file_extension_label.grid( row=grid_row, column=2, sticky=Tk.E )
        self.file_extension  = Tk.StringVar()
        self.file_extension.set( get_setting( 'file_extension' ) )
        file_extension_box = ttk.Combobox( upper_frame, textvariable=self.file_extension,
                                            width=5, justify=Tk.CENTER )
        file_extension_box[ 'values' ] = [ '*.dat', '*.csv', '*.int', '*.*' ]
        file_extension_box.grid( row=grid_row, column=3, sticky=Tk.W )

        self.file_extension.trace( "w", self.file_extension_tracer )

        grid_row += 1
        uv_folder_label  = Tk.Label( upper_frame, text= 'UV Absorbance Data Folder: ' )
        uv_folder_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.uv_folder = Tk.StringVar()
        self.uv_folder.set( get_setting( 'uv_folder' ) )
        self.uv_folder_entry = UvFolderEntry( upper_frame, textvariable=self.uv_folder, width=folder_entry_width,
                                            slimbutton=self.slimbutton,
                                            on_entry_cb=self.on_entry_uv_folder )
        self.uv_folder_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        uv_file_name_label  = Tk.Label( upper_frame, text= '  File Name: ' )
        uv_file_name_label.grid( row=grid_row, column=2, sticky=Tk.E )
        self.uv_file = Tk.StringVar()

        uv_file_frame = Tk.Frame( upper_frame )
        uv_file_frame.grid( row=grid_row, column=3, sticky=Tk.W )

        # TODO: refactoring with FileEntry
        self.uv_file_entry = Tk.Entry( uv_file_frame, textvariable=self.uv_file, width=30 )
        self.uv_file_entry.grid( row=0, column=0, sticky=Tk.W )
        if self.slimbutton:
            b3 = SlimButton( uv_file_frame, text='...', command=self.select_uv_file, width=18, height=22 )
        else:
            b3 = Tk.Button( uv_file_frame, text='...', command=self.select_uv_file )

        b3.grid( row=0, column=1, sticky=Tk.W )

        # DND bind
        self.uv_file_entry.register_drop_target("*")

        def dnd_handler( event ):
            self.select_uv_file(path=event.data)

        self.uv_file_entry.bind("<<Drop>>", dnd_handler)

        # disable UV data
        grid_row += 1
        self.disable_uv_data = Tk.IntVar()
        self.disable_uv_data.set(get_setting('disable_uv_data'))
        self.disable_uv_data_cb = Tk.Checkbutton( upper_frame, variable=self.disable_uv_data,
                                text='disable UV data', state=Tk.DISABLED )
        self.disable_uv_data_cb.grid(row=grid_row, column=2, columnspan=2, sticky=Tk.W, padx=5)
        self.disable_uv_data_tracing = True
        self.disable_uv_data_tracer_busy = False
        self.disable_uv_data.trace('w', self.disable_uv_data_tracer)

        # Conc. Factors
        from GuiParts.ConcFactorsEntry import ConcFactorsEntry

        grid_row += 1
        self.cfs_entry = ConcFactorsEntry(upper_frame, grid_row)

        # pad
        grid_row += 2   # ConcFactorsEntry includes 2 rows
        """
        self.no_absorbance = Tk.IntVar()
        cb = Tk.Checkbutton( upper_frame, text='Use without Absorbance Data',
                    variable=self.no_absorbance )
        cb.grid( row=grid_row, column=1, sticky=Tk.W )
        """

        pad_frame = Tk.Label( upper_frame )
        pad_frame.grid( row=grid_row, column=0 )

        # Analysis Result Setting ----------------------------------------------
        grid_row += 1

        analysis_label  = Tk.Label( upper_frame, text= 'Output', width=section_label_width,
                                    relief=Tk.FLAT, fg=label_fg_color, bg=label_bg_color )
        analysis_label.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        an_folder_label  = Tk.Label( upper_frame, text= 'Analysis Result Folder: ' )
        an_folder_label.grid( row=grid_row, column=0, sticky=Tk.E )
        self.an_folder = Tk.StringVar()
        self.an_folder.set( get_setting( 'an_folder' ) )
        self.an_folder_frame = BlinkingFrame( upper_frame )
        self.an_folder_frame.grid( row=grid_row, column=1, sticky=Tk.W )
        self.an_folder_entry = FolderEntry( self.an_folder_frame, textvariable=self.an_folder, width=folder_entry_width,
                                            slimbutton=self.slimbutton,
                                            on_entry_cb=self.on_entry_an_folder )
        self.an_folder_entry.pack()
        self.an_folder_entry.bind('<FocusOut>', self.an_folder_focusout)
        self.an_folder_frame.objects = [self.an_folder_entry.entry]

        grid_row += 1
        analysis_name_label = Tk.Label( upper_frame, text='Subfolder: ' )
        analysis_name_label.grid( row=grid_row, column=0, sticky=Tk.E )
        analysis_frame = Tk.Frame( upper_frame )
        analysis_frame.grid( row=grid_row, column=1, sticky=Tk.W )
        self.analysis_name = Tk.StringVar()
        self.analysis_name.set( get_setting( 'analysis_name' ) )
        self.analysis_name_entry = Tk.Entry( analysis_frame, textvariable=self.analysis_name, width=20 )
        self.analysis_name_entry.grid( row=0, column=0, sticky=Tk.W )
        self.auto_number = Tk.IntVar()
        auto_number_ = get_setting( 'auto_number' )
        self.auto_number.set( auto_number_ )
        if False:
            self.auto_numbering_cb = Tk.Checkbutton( analysis_frame, variable=self.auto_number,
                                                text='Auto-number (un-check to reuse)' )
                                                # text='Auto-number (un-check this to reuse previous results)' )
            self.auto_numbering_cb.grid( row=0, column=1, sticky=Tk.W, padx=10 )

        if auto_number_ == 1 and not is_empty_val( self.an_folder.get() ):
            analysis_name, _ = self.make_analysis_folder( format_check_only=True )
            self.analysis_name.set( analysis_name )

        self.analysis_name_entry.bind( '<FocusOut>',  self.analysis_name_entry_focus_out )

        grid_row += 1
        result_book_label = Tk.Label( upper_frame, text='Book Name: ' )
        result_book_label.grid( row=grid_row, column=0, sticky=Tk.E )
        self.result_book = Tk.StringVar()
        result_book = get_setting( 'result_book' )
        self.result_book.set( result_book )
        self.result_book_entry = Tk.Entry( upper_frame, textvariable=self.result_book, width=20 )
        self.result_book_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        grid_row += 1
        pad_frame = Tk.Label( upper_frame )
        pad_frame.grid( row=grid_row, column=0 )

        # File Information Table -----------------------------------------------
        grid_row += 1
        grid_row_for_analysis_button = grid_row
        table_label  = Tk.Label( upper_frame, text= 'Xray Scattering Data Files', width=section_label_width,
                                    relief=Tk.FLAT, fg=label_fg_color, bg=label_bg_color )
        table_label.grid( row=grid_row, column=0, sticky=Tk.W )

        fig_table_frame = Tk.Frame( lower_frame )
        fig_table_frame.pack(fill=Tk.X, expand=1, pady=10)

        # navigation button frame
        nav_frame = Tk.Frame(fig_table_frame)
        nav_frame.pack( side=Tk.BOTTOM, fill=Tk.X, expand=1, pady=10)

        self.message_frame = Tk.Frame( fig_table_frame )
        self.message_frame.pack( side=Tk.BOTTOM, fill=Tk.X, expand=1)   # Tk.BOTTOM required!

        table_height = 150 if self.high_dpi else 100
        self.file_info_table = FileInfoTable( fig_table_frame, self, self.loader, self.fixed_font, height=table_height )
        self.file_info_table.pack( side=Tk.RIGHT, fill=Tk.BOTH, expand=1, anchor=Tk.N)

        figsize = (6,8) if self.high_dpi else (4,5)
        self.fig_frame = OutlineFigure(fig_table_frame, self, file_info_table=self.file_info_table, figsize=figsize)
        self.fig_frame.pack(side=Tk.LEFT, fill=Tk.BOTH)

        # fill=Tk.BOTH would make the Scrollbar too long in some cases

        self.btn_frame = Tk.Frame(upper_frame)
        self.btn_frame.grid(row=grid_row, column=1, columnspan=3, sticky=Tk.E)

        self.refresh_button = Tk.Button( self.btn_frame, text='Refresh', command=self.refresh )
        self.refresh_button.grid(row=0, column=0)
        self.presync_button = Tk.Button( self.btn_frame, text='3D View', command=self.show_threedim_dialog )
        self.presync_button.grid(row=0, column=1, padx=10)
        self.restrict_button = Tk.Button( self.btn_frame, text='Data Range', command=self.show_datarange_dialog )
        self.restrict_button.grid(row=0, column=2)
        self.rankview_menu = RankViewMenu(self.btn_frame, self)
        self.rankview_menu.grid(row=0, column=3, padx=10)
        self.baseline_button = Tk.Button( self.btn_frame, text='Baseline', command=self.show_lpm_inspector )
        self.baseline_button.grid(row=0, column=4)

        # Analysis start button ---------------------------------------------------
        nav_btn_frame = Tk.Frame(nav_frame)
        nav_btn_frame.pack(side=Tk.RIGHT)

        exit_button = Tk.Button(nav_frame, text="▼ Exit", command=self.quit)
        exit_button.pack(side=Tk.LEFT, padx=10)

        button_frame = Tk.Frame( nav_btn_frame )
        button_frame.pack()

        self.analysis_button = Tk.Button( button_frame, text="▶ Analysis start", command=self.analyze_all )
        self.analysis_button.grid( row=0, column=0, padx=5 )
        self.fully_automatic  = Tk.IntVar()
        self.fully_automatic.set(get_setting('fully_automatic'))
        self.fully_automatic_cb = Tk.Checkbutton( button_frame, text="fully automatic", variable=self.fully_automatic )
        self.fully_automatic_cb.grid( row=0, column=1, padx=5 )
        self.buttons = [self.refresh_button, self.restrict_button, self.presync_button, self.rankview_menu, self.baseline_button,
                            self.analysis_button, self.fully_automatic_cb,
                            self.disable_uv_data_cb, self.cfs_entry,
                            exit_button]

        self.enable_synthesized_lrf = get_setting('enable_synthesized_lrf')
        if self.enable_synthesized_lrf:
            self.synthesized_lrf  = Tk.IntVar()
            self.synthesized_lrf.set(get_setting('synthesized_lrf'))
            self.synthesized_lrf_cb = Tk.Checkbutton( button_frame, text="synthesized LRF", variable=self.synthesized_lrf )
            self.synthesized_lrf_cb.grid( row=0, column=2, padx=5 )
            self.buttons.append(self.synthesized_lrf_cb)

        self.update_button_states()
        self.update_analysis_button_state()

        # Plot results button ---------------------------------------------------
        self.plot_button = Tk.Button( button_frame, text="Plot Results", command=self.plot_results )
        self.plot_button.grid( row=0, column=1, padx=5 )
        self.update_plot_button_state()

        self.in_folder_has_been_changed = False
        self.settings_have_been_changed = False

        # it seemd that '<Control-c>' should be bound from the top-level.
        self.bind( '<Control-c>', self._on_control_c )
        self.bind( '<Button-1>', self._on_click )
        # self.bind( '<Return>', self._on_enter )
        # self.bind( '<Control-a>', self._on_enter )
        self.bind( '<Escape>', self._on_escape )
        self.in_folder_entry.bind( '<FocusIn>', self.on_in_folder_entry_focus_in )
        self.in_folder_entry.bind( '<FocusOut>',  self.on_in_folder_entry_focus_out )

        # message frame
        self.guide_message = Tk.Label(self.message_frame, bg='white')
        self.guide_message.pack(fill=Tk.X, pady=10)

        self.set_state_guide_message("Specify input/output folders")

        # getting ready to show
        self.adjust_geometry()

        invoke_on_entry = False
        if not is_empty_val( self.in_folder.get() ):
            invoke_on_entry = self.ask_resume()

        # file_info_table does not grow in size after geometry setting
        self.file_info_table.refresh()

        if invoke_on_entry:
            self.on_entry_in_folder()

        uv_folder_ = self.uv_folder.get()
        if not is_empty_val( uv_folder_ ):
            if is_empty_val( self.uv_file.get() ):
                uv_file = get_uv_filename( uv_folder_ )
                self.uv_file.set( uv_file )

        self.datarange_dialog = None
        self.protocol( "WM_DELETE_WINDOW", self.quit )

    def ask_resume(self):
        self.update()
        ret = MessageBox.askyesno("Reload Confirmation",
            'Previous input was from\n'
            '%s\n'
            'Would you like to have it loaded?\n'
            % self.in_folder.get(), parent=self,
            )
        if ret:
            invoke_on_entry = True
        else:
            self.in_folder.set(None)
            set_setting("in_folder", None)
            invoke_on_entry = False
        return invoke_on_entry

    def set_state_guide_message(self, guide_message, fg='black'):
        self.message_frame.update()
        try:
            self.guide_message.config(text=guide_message.replace('\n', ' '), fg=fg)
        except:
            # _tkinter.TclError: invalid command name ".!guimain.!frame3.!frame.!frame2.!label"
            log_exception(self.tmp_logger, "self.guide_message.config(...): ")

    def file_extension_tracer( self, *args ):
        set_setting( 'file_extension', self.file_extension.get() )
        self.on_entry_in_folder()

    def refresh( self, keeping_setting_items=None):
        print( 'refresh' )
        self.keeping_setting_items = keeping_setting_items
        self.previous_in_folder = None  # this will induce clear_temporary_settings()
        self.in_folder_entry.on_entry(refresh=True)
        # self.uv_folder_entry.on_entry()

        if self.pre_recog.restrict_info_changed():
            yn = MessageBox.askyesno("Question",
                            "Data Range info seems to have been changed mannually.\n"
                            "Do you wish reset it to the default?",
                            parent=self)
            if yn:
                self.pre_recog.reset_restrict_info()

    def show_datarange_dialog( self ):
        from molass_legacy.Trimming import DataRangeDialog
        range_editor_info = get_setting('range_editor_info')
        decomp_editor_info = get_setting('decomp_editor_info')

        if range_editor_info is not None or decomp_editor_info is not None:
            ok = MessageBox.askyesno( 
                            'Confirmation Question',
                            'Previously memorized analysis range info exists.\n'
                            'Applying a new data restriction will reset that info to default.\n'
                            'Are you sure to proceed?',
                            parent=self
                            )
            if not ok:
                return

        self.datarange_dialog = DataRangeDialog(self.parent, self.pre_recog)
        self.datarange_dialog.show()

        self.draw_figure()

    def has_datarange_dialog(self):
        return self.datarange_dialog is not None

    def create_menus( self ):
        import _MOLASS.SerialSettings
        menubar = Tk.Menu( self )
        self.config( menu=menubar )
        self.menubar = menubar

        menu0 = Tk.Menu( menubar, tearoff=0 )
        self.recent_menu = Tk.Menu( menubar, tearoff=0 )

        menubar.add_cascade( label="Folder", menu=menu0 )
        menu0.add_cascade( label="Recently used data folders", menu=self.recent_menu )
        self.recent_folders = RecentFolders( size=20, settings=_MOLASS.SerialSettings  )
        self.recent_menu_add()
        menu0.add_command( label="Exit", command=self.quit )

        menu1 = GuiSettingsMenu( self, menubar )
        self.menu3 = GuiSecToolsMenu( self, menubar )
        self.menu4 = GuiDenssToolsMenu( self, menubar )
        self.menu5 = GuiTutorialsMenu( self, menubar )
        menu6 = GuiReferencesMenu( self, menubar )
        menu7 = GuiDevelopmentMenu( self, menubar )

    def recent_menu_add( self ):
        for folder in self.recent_folders.get_sorted_list():
            path = folder[1]
            # print( 'add recent menu', path )
            self.recent_menu.add_command( label=path, command=lambda path_=path: self.set_in_folder( path_ ) )

        self.update()

    def recent_menu_delete( self ):
        for folder in self.recent_folders.get_sorted_list():
            path = folder[1]
            # print( 'deleting menu ', path )
            try:
                self.recent_menu.delete( path )
            except:
                print( 'error deleting menu ', path )

    def askdirectory( self, entry_variable ):
        entered_path = entry_variable.get()
        # print 'entered_path=', entered_path
        dir_ = os.path.dirname( entered_path ).replace( '/', '\\' )
        # print 'dir_=', dir_
        f = FileDialog.askdirectory( initialdir=dir_, parent=self )
        return f

    def set_in_folder( self, path ):
        # print( 'set_in_folder', path )
        self.in_folder.set( path )
        self.on_entry_in_folder()
        self.settings_have_been_changed = True

    def set_uv_folder( self, path, filename ):
        # print( 'set_uv_folder', path )
        self.uv_folder.set( path )
        self.on_entry_uv_folder( filename=filename )
        self.settings_have_been_changed = True

    def select_uv_file(self, path=None):
        if path is None:
            f = FileDialog.askopenfilename( initialdir=self.uv_folder.get(), parent=self )
            if not f:
                return
        else:
            f = path

        dir_, file = os.path.split( f )
        self.on_entry_uv_file( dir_, file  )
        self.settings_have_been_changed = True

    def update_button_states(self):
        state = Tk.NORMAL if (self.dataset_is_ready and not self.is_executing) else Tk.DISABLED
        for k, btn in enumerate(self.buttons):
            btn.config(state=state)

        disable_uv_data = get_setting('disable_uv_data')
        if not disable_uv_data:
            if is_empty_val(self.uv_file.get()):
                for w in [self.cfs_entry, self.disable_uv_data_cb]:
                    w.config(state=Tk.DISABLED)

        self.update_menu_states()
        self.update()

    def update_menu_states(self):
        self.menu3.update_states()
        self.menu4.update_states()
        self.menu5.update_states()

    def update_analysis_button_state( self ):
        if self.is_executing:
            state_ = Tk.DISABLED
        else:
            if self.dataset_is_ready:
                num_data_rows = self.file_info_table.table.cells_array.shape[0] - 1
                use_xray_conc = get_setting( 'use_xray_conc' ) == 1
                use_mtd_conc = get_setting( 'use_mtd_conc' ) == 1
                if use_xray_conc or use_mtd_conc:
                    state_ = Tk.NORMAL if num_data_rows > 0 else Tk.DISABLED
                else:
                    state_ = Tk.NORMAL if num_data_rows > 0 and self.uv_file_exists else Tk.DISABLED
            else:
                state_ = Tk.DISABLED
        try:
            self.analysis_button.config( state=state_ )
            self.update()
        except:
            # _tkinter.TclError: invalid command name ".!guimain.!frame3.!frame.!frame.!frame.!frame.!button"
            pass

    def update_all_button_states(self):
        self.update()
        self.update_button_states()
        self.update_analysis_button_state()
        self.update_plot_button_state()
        self.update()

    def on_entry_in_folder( self, refresh=False ):
        # refresh is currently not used

        self.uv_file_info_changed = False
        self.dataset_is_ready = False
        self.config( cursor='wait' )
        self.update_all_button_states()
        self.uv_file_exists = False

        self.conc_serie = None
        in_folder_ = self.in_folder.get()

        if in_folder_ != self.previous_in_folder:
            kept_items = None
            if refresh and self.keeping_setting_items is not None:
                kept_items = []
                for item in self.keeping_setting_items:
                    kept_items.append(get_setting(item))
            self.on_in_folder_change()
            self.load_start_time = time.time()
            if kept_items is not None:
                for item, value in zip(self.keeping_setting_items, kept_items):
                    set_setting(item, value)
            self.serial_data = None
            self.measured_data = None
            self.previous_in_folder = in_folder_
            self.cleared_uv_folder  = False

        self.recent_menu_delete()
        self.recent_folders.add( in_folder_ )
        self.recent_menu_add()

        set_setting( 'in_folder', in_folder_ )
        self.file_info_table.refresh()
        self.update()

        self.config( cursor='' )

        if self.file_info_table.num_rows == 0:
            self.set_in_folder_error()
            self.update()   # necessary for the file extension combobox to shrink
            MessageBox.showerror( "Xray Scattering Data Folder Error", "No Data File in '%s'" % in_folder_, parent=self )
            self.set_in_folder_error()  # call it again to focus
            return

        # need to restore fg if the case is after the error above
        self.in_folder_entry.config( fg='black' )

        self.loader.reset_input_status()
        if self.tmp_logger is None:
            self.tmp_logger = Logger()  # this logger logs into string stream
        else:
            # self.tmp_logger should have been created after do_analysis
            pass

        if self.parent_pid is not None:
            from molass_legacy.KekLib.CleanerThread import CleanerThread
            with_parent_pid = " with parent_pid=%d" % self.parent_pid
            self.cleaner = CleanerThread(self.parent_pid)
        else:
            with_parent_pid = ""
            self.cleaner = None
        self.tmp_logger.info('logging started on_entry_in_folder in %s%s', get_version_string(cpuid=True), with_parent_pid)

        if self.uv_file_info_changed:
            # keep the current uv file info
            self.loader.reset_current_status()
            self.after(500, self.prepare_serial_data)
        else:
            filename = get_uv_filename( in_folder_ )
            if filename is None:
                filename = get_mtd_filename( in_folder_ )
                if filename is None:
                    if not self.cleared_uv_folder:
                        # don't do this for the same in_folder_
                        self.uv_folder.set( '' )
                        self.uv_file.set( '' )
                        set_setting( 'uv_folder',   None )  # clear
                        set_setting( 'uv_file',     None )  # clear
                        self.cleared_uv_folder = True
                        set_setting( 'use_xray_conc', 1 )
                        from molass_legacy._MOLASS.SerialSettings import do_xray_conc_temporary_settings
                        do_xray_conc_temporary_settings()
                        self.after(500, self.prepare_serial_data)
                else:
                    # 
                    folder, file = os.path.split(filename)
                    self.uv_folder.set( folder )
                    self.uv_file.set( file )

                    set_setting( 'use_xray_conc', 0 )
                    set_setting( 'use_mtd_conc', 1 )
                    self.use_mtd_conc = True
                    set_setting( 'mtd_file_path', filename )
                    # self.fully_automatic.set(0)
                    from molass_legacy._MOLASS.SerialSettings import do_microfluidic_temporary_settings
                    do_microfluidic_temporary_settings()
                    self.after(500, self.prepare_serial_data)
            else:
                self.set_uv_folder( in_folder_, filename )
                set_setting( 'use_xray_conc', 0 )
                # verify the need to call self.prepare_serial_data
                # self.after(500, self.prepare_serial_data)

        self.lacking_q_values_warned = False
        self.fig_frame.clear_figures()
        self.set_state_guide_message("Waiting for the completion\nof data loading.")
        self.update()
        # update states
        self.update_uv_folder_entry_state()
        self.settings_have_been_changed = True

    def clear_entries(self):
        self.cfs_entry.reset_entries()
        clear_temporary_settings()
        self.update()

    def on_in_folder_change(self):
        self.clear_entries()

        if self.disable_uv_data_tracer_busy:
            """
            do nothing when called during the tracer process.
            """
            pass
        else:
            """
            this is the case when the input folder has been changed
            after disabling the previous folder.
            """
            self.disable_uv_data_tracing = False
            self.disable_uv_data.set(get_setting('disable_uv_data'))
            self.update()
            self.disable_uv_data_tracing = True

    def set_in_folder_error( self ):
        self.in_folder_entry.config( fg='red' )
        self.in_folder_entry.focus_force()

    def on_entry_uv_folder( self, filename=None ):
        if self.loader.is_busy:
            # TODO : better to get rid of this case
            if self.tmp_logger is not None:
                self.tmp_logger.info('on_entry_uv_folder is skipped because the loader is busy.')
                return

        uv_folder_ = self.uv_folder.get()

        if filename is None:
            filename = get_uv_filename( uv_folder_ )
            if filename is None:
                self.set_uv_folder_error()
                MessageBox.showerror( "Absorbance Folder Error", "No absorbance file in '%s'." % uv_folder_, parent=self )
                self.uv_file.set( "<Not found>" )
                return

        self.uv_file_info_changed = True
        self.uv_file.set( filename )
        set_setting( 'uv_folder', uv_folder_ )
        set_setting( 'uv_file', filename )
        self.uv_file_exists = True

        data_folder = get_setting( 'in_folder' ).replace( '\\', '/' )
        conc_folder = get_setting( 'uv_folder' ).replace( '\\', '/' )
        conc_file   = get_setting( 'uv_file'   )
        if conc_file.find( '*' ) >= 0:
            conc_file = None

        # self.analysis_button.config( state=Tk.NORMAL )

        if not is_empty_val(self.previous_uv_file):
            self.set_state_guide_message("Re-loading the data")

        self.previous_uv_file = filename
        self.update_analysis_button_state()
        self.loader.load_from_folders( data_folder, uv_folder=conc_folder, uv_file=conc_file )
        # serial_data construction will be delayed until self.analyze_all
        # to keep this thread running without waiting here

        self.settings_have_been_changed = True
        self.after(500, self.prepare_serial_data)

    def set_uv_folder_error( self ):
        self.uv_folder_entry.config( fg='red' )

    def on_entry_uv_file( self, dir_, file ):
        self.uv_file.set( file )
        self.uv_folder.set( dir_ )

        if not self.uv_folder_entry.check():
            return

        if not self.uv_file_check():
            return

        self.set_uv_folder( dir_, file )

    def uv_file_check( self ):
        # TODO: include these checks in a custom widget class
        ok_ = False
        uv_file = self.uv_file.get()
        uv_folder = self.uv_folder.get()
        if uv_file.find( '*' ) >= 0:
            file_ = get_uv_filename(uv_folder, glob=uv_file)
            if file_ is None:
                self.set_uv_file_error()
                MessageBox.showerror( "Absorbance File Error", "No Absorbance File in '%s'" % uv_folder, parent=self )
            else:
                self.uv_file_entry.config( fg='black' )
                ok_ = True
        else:
            uv_filepath = os.path.join( uv_folder, uv_file );
            if is_empty_val( uv_file ):
                self.set_uv_file_error()
                MessageBox.showerror( "Absorbance File Error", "The Absorbance File is required.", parent=self )
            elif not os.path.exists( uv_filepath ):
                self.set_uv_file_error()
                MessageBox.showerror( "Absorbance File Error", "'%s' does not exist." % uv_file, parent=self )
            elif not os.path.isfile( uv_filepath ):
                self.set_uv_file_error()
                MessageBox.showerror( "Absorbance File Error", "'%s' is not a file." % uv_file, parent=self )
            else:
                self.uv_file_entry.config( fg='black' )
                ok_ = True

        if not ok_:
            self.uv_file_entry.focus_force()

        return ok_

    def set_uv_file_error( self ):
        self.uv_file_entry.config( fg='red' )

    def on_in_folder_entry_focus_in( self, event ):
        self.in_folder_ext_state = self.in_folder.get() + self.file_extension.get()

    def on_in_folder_entry_focus_out( self, event ):
        # print( 'on_in_folder_entry_focus_out' )
        in_folder = self.in_folder.get()
        file_extension = self.file_extension.get()
        in_folder_ext_state = in_folder + file_extension
        if in_folder_ext_state == self.in_folder_ext_state:
            return

        uv_folder = self.uv_folder.get()
        if is_empty_val( in_folder ) or not is_empty_val( uv_folder ):
            return

        self.on_entry_in_folder()

    def on_entry_an_folder( self ):
        # print( 'on_entry_an_folder' )
        self.an_folder_frame.stop()
        self.update()

        if not self.an_folder_check():
            return

        # self.an_folder_entry.config(fg=None)
        self.config( cursor='wait' )    # doesn't seem to work
        self.update()
        an_folder_ = self.an_folder.get()
        set_setting( 'an_folder', an_folder_ )
        self.settings_have_been_changed = True
        self.update_plot_button_state()
        if self.serial_data is not None:
            self.draw_figure()
        self.config( cursor='' )
        self.make_it_more_clever()

    def an_folder_focusout( self, *args ):
        if not is_empty_val(self.an_folder.get()):
            if self.an_folder_frame.is_blinking():
                self.on_entry_an_folder()

    def an_folder_check( self ):
        # TODO: include these checks in a custom widget class
        ok_ = False
        an_folder = self.an_folder.get()
        if is_empty_val( an_folder ):
            self.set_an_folder_error()
            MessageBox.showerror( "Analysis Folder Error", "The Analysis Result Folder is required.", parent=self )
        else:
            from GuiParts.NonAsciiPaths import nonascii_path_check
            if not nonascii_path_check(an_folder, self):
                return ok_

            self.an_folder_frame.stop()
            if not os.path.exists( an_folder ):
                self.set_an_folder_error()
                yn = MessageBox.askyesno( "Analysis Folder Error",
                            "'%s' does not exist.\nWould you like to make the folder?" % an_folder, icon="warning", parent=self )
                if yn:
                    try:
                        mkdirs_with_retry( an_folder )
                    except:
                        pass
                    if os.path.exists( an_folder ):
                        ok_ = True
            elif not os.path.isdir( an_folder ):
                self.set_an_folder_error()
                MessageBox.showerror( "Analysis Folder Error", "'%s' is not a folder." % an_folder, parent=self )
            elif self.check_if_like_report_folder( an_folder ):
                self.set_an_folder_error()
                yn = MessageBox.askyesno( "Analysis Folder Error",
                                            ( "'%s' seems like a used sub-folder.\n"
                                             + "Are you sure this is an appropriate selection?" )
                                            % an_folder, parent=self )
                if yn:
                    ok_ = True
            else:
                ok_ = True

        if ok_:
            self.an_folder_entry.config( fg='black' )
        else:
            self.an_folder_entry.focus_force()

        return ok_

    def check_if_like_report_folder( self, an_folder ):
        logfile = an_folder + '/molass.log'
        repfile = an_folder + '/analysis_report.xlsx'
        return os.path.exists( logfile ) and  os.path.exists( repfile )

    def set_an_folder_error( self ):
        self.an_folder_entry.config( fg='red' )

    def new_analysis( self ):
        reuse = get_setting("reuse_analysis_folder")
        if reuse:
            analysis_name = get_setting('analysis_name')
            analysis_folder = get_setting('analysis_folder')
            if analysis_folder is None:
                pass
            else:
                return analysis_name, analysis_folder

        try:
            analysis_name, analysis_folder = self.make_analysis_folder()
            self.analysis_name.set( analysis_name )
            set_setting( 'analysis_name', analysis_name )
            set_setting( 'analysis_folder', analysis_folder )
            return analysis_name, analysis_folder
        except:
            print( sys.exc_info() )
            return

    def get_analysis_name(self):
        # this method is for use in testing
        an_folder = self.an_folder.get()
        analysis_name = self.analysis_name.get()
        return self.auto_number_analysis_name(an_folder, analysis_name)

    def make_analysis_folder( self, format_check_only=False ):
        an_folder = self.an_folder.get()
        analysis_name = self.analysis_name.get()
        self.auto_number_analysis_name( an_folder, analysis_name ) # this is just for check if it is propperly named

        def get_analysis_folder( out_folder, analysis_name ):
            if is_empty_val( out_folder ):
                return None
            else:
                analysis_folder = os.path.join( out_folder, analysis_name )
                return analysis_folder.replace( '\\', '/' )

        analysis_folder = get_analysis_folder( an_folder, analysis_name )

        if not format_check_only:

            if self.auto_number.get() == 1:
                if not is_empty_val( analysis_folder ):
                    while os.path.exists( analysis_folder ) and not is_almost_empty_dir( analysis_folder ):
                        analysis_name = self.auto_number_analysis_name( an_folder, analysis_name )
                        analysis_folder = get_analysis_folder( an_folder, analysis_name )

            # print( 'make_analysis_folder: analysis_folder=', analysis_folder, type(analysis_folder) )
            if not is_empty_val( analysis_folder ) and not os.path.exists( analysis_folder ):
                mkdirs_with_retry( analysis_folder )

        return analysis_name, analysis_folder

    def auto_number_analysis_name( self, an_folder, analysis_name ):
        m = analysis_name_re.match( analysis_name )
        if m:
            first   = m.group( 1 )
            number  = m.group( 2 )
            third   = m.group( 3 )
            len_ = len( number )
            name_ = first + '%0*d' % ( len_, int( number ) ) + third
            if not is_empty_val( an_folder ):
                folder = an_folder + '/' + name_
                if os.path.exists( folder ) and not is_almost_empty_dir( folder ):
                    name_ = first + '%0*d' % ( len_, int( number ) + 1 ) + third
            self.analysis_name_entry.config( fg='black' )
            # print( 'new name=', name_ )
            return name_
        else:
            self.analysis_name_entry.config( fg='red' )
            MessageBox.showerror(
                'Naming Error',
                "Analysis Name '%s' is not expected. Name it with a numeric postfix such as 'analysis-000'." % analysis_name,
                parent=self,
                )
            assert( False )

    def analysis_name_entry_focus_out( self, event ):
        # print( 'analysis_name_entry_focus_out' )
        self.update_plot_button_state()

    def update_analysis_folder(self):
        analysis_folder = get_setting('analysis_folder')
        _, analysis_folder_ = self.make_analysis_folder(format_check_only=True)
        if analysis_folder != analysis_folder_:
            self.new_analysis()

    def prepare_serial_data( self ):
        self.tmp_logger.info('preparing serial data.')
        self.prepare_start_time = time.time()
        self.serial_data = self.loader.get_current_object()
        # note that data are not yet deepcopied at this stage
        self.abnomality_dialog = None
        if not self.use_mtd_conc:
            exclude_abnormality( self.serial_data, self.file_info_table, self.tmp_logger, dialog=self )

        if self.serial_data.has_excluded_xray_elutions:
            self.loader.memorize_exclusion( self.serial_data )

        if self.serial_data.is_serial():
            self.pre_recog = PreliminaryRecognition(self.serial_data)
        else:
            self.pre_recog = None
            return

        try:
            self.fig_frame.set_data(self.pre_recog)
        except Exception as exc:
            if not self.use_mtd_conc:
                raise exc

        eno = self.serial_data.xray_curve.primary_peak_i

        try:
            self.file_info_table.select_row(eno)
        except:
            log_exception(self.tmp_logger, "prepare_serial_data: self.file_info_table.select_row(eno)")

        self.draw_figure(eno)
        self.cfs_entry.update_path_length()

        self.dataset_is_ready = True
        preparing_time = time.time() - self.prepare_start_time
        self.tmp_logger.info('data set is ready. it took %.3g seconds for preparing.', preparing_time)
        self.update_button_states()

        self.measured_data = MeasuredData(None, sd=self.serial_data, pre_recog=self.pre_recog)
        updated = self.measured_data.update_picking_params(self.tmp_logger)
        # TODO: update self.fig_frame
        if updated:
            self.fig_frame.update_elution_curve()
        self.tmp_logger.info('created measured_data.')

        if not self.lacking_q_values_warned:
            if get_setting('found_lacking_q_values'):
                self.lacking_q_values_warned = True
                MessageBox.showwarning( "Lacking Q-values",
                    "There have been found lacking Q-values\n"
                    "in some of the input files.\n"
                    "Make sure to confirm them in the molass.log",
                    parent=self )

    def re_prepare_serial_data( self ):
        save_items = ["uv_device_no", "path_length", "beamline_name", "xray_baseline_type"]   # any other?
        saved_values = []
        for name in save_items:
            saved_values.append(get_setting(name))      # save

        self.clear_entries()

        for name, value in zip(save_items, saved_values):
            set_setting(name, value)                    # restore

        if self.dataset_is_ready:
            self.dataset_is_ready = False
            self.update_all_button_states()
            self.tmp_logger.info("re-preparing serial data after user's settings change.")
            self.after(500, self.prepare_serial_data)

    def is_reloadable(self):
        return self.dataset_is_ready

    def draw_figure( self, selected=None ):
        if self.serial_data is None or not self.serial_data.is_serial():
            return

        self.update()   # seems to be required to be sure to get the current self.an_folder value

        if is_empty_val( self.an_folder.get() ):
            self.set_state_guide_message("Specify output folder.")
            self.an_folder_frame.start()
            # self.an_folder_entry.config(bg='yellow')
            return

        self.fig_frame.draw_figure(selected)
        self.set_state_guide_message('Press [▶ Analysis start] or click to select.', fg='green')
        self.an_folder_frame.stop()
        self.update()   # seems to be required sometimes (not always)

    def wait_until_the_data_is_ready( self ):
        while self.loader.is_busy:
            self.update()
            print('waiting for the completion of data loading')
            time.sleep(1)

    def detected_abnomality( self ):
        if self.abnomality_dialog is None:
            ret = False
        else:
            self.tmp_logger.warning( 'detected abnomality(bubles?) in the input data.' )
            ret = True
        return ret

    def get_range_type( self ):
        return get_setting( 'range_type' )

    def analyze_all( self ):
        from molass_legacy.SerialAnalyzer.AbnormalityCheck import update_abnormality_fix_state

        if not self.check_all_setting_info():
            self.analysis_button.grid( row=0, column=0, padx=5 )
            return

        try:
            analysis_name, analysis_folder = self.make_analysis_folder()
            if os.path.exists( analysis_folder ) and not is_almost_empty_dir( analysis_folder ):
                self.new_analysis()
        except:
            print( sys.exc_info() )
            return

        fully_automatic = self.fully_automatic.get()
        if fully_automatic:
            ret = MessageBox.askyesno("Fully Automatic Run Confirmation",
                    '"Fully automatic" execution may take several minutes.\n'
                    + 'Are you sure to proceed?',
                    parent=self)
            if not ret:
                return

        # TODO: if using the same in_folder, ask if the previous params should be retained or discarded

        self.analysis_name.set( analysis_name )

        set_setting( 'analysis_folder', analysis_folder )
        set_setting( 'analysis_name', analysis_name )
        set_setting( 'result_book', self.result_book.get() )
        set_setting( 'fully_automatic', fully_automatic )
        if self.enable_synthesized_lrf:
            set_setting( 'synthesized_lrf', self.synthesized_lrf.get() )

        self.update_setting_info()
        self.update_plot_button_state()

        self.analyzer.change_log_to(analysis_folder)

        update_abnormality_fix_state(self.pre_recog.get_pre_recog_copy(), self.file_info_table, self.analyzer.app_logger, dialog=self)

        def v1_exec_closure():
            self.set_state_guide_message("Preparing for Mapping Dialog.")
            self.analyzer.do_analysis(self.serial_data, self.pre_recog, self.measured_data, analysis_folder, analysis_name)
            self.set_state_guide_message("V1 manipulation done.")

        self.exec_wrapper(v1_exec_closure, "do_analysis")

    def exec_wrapper(self, exec_closure, exec_name):
        # this wrapper is intended to unify the button state control

        self.is_executing = True
        self.update_button_states()
        self.update_analysis_button_state()
        self.update()

        exec_closure()

        self.reset_tmp_logger(exec_name)
        self.is_executing = False
        try:
            self.update_button_states()
            self.update_analysis_button_state()
        except:
            """
            _tkinter.TclError: invalid command name ".!guimain.!frame3.!frame.!frame.!frame.!frame.!checkbutton"
            probabply harmless
            """
            pass

    def reset_tmp_logger(self, process_name):
        self.tmp_logger = Logger()
        self.tmp_logger.info("tmp_logger has been created after %s in " % process_name + get_version_string(cpuid=True))

    def check_all_setting_info( self ):
        # for method in [ self.in_folder_entry.check, self.uv_folder_entry.check, self.an_folder_check, self.check_atsas_executability ]:
        for method in [ self.in_folder_entry.check, self.uv_folder_entry.check, self.an_folder_check ]:
            ok_ = method()
            if not ok_:
                return False

        return True

    def check_atsas_executability(self):
        from molass_legacy.ATSAS.ExecCheck import atsas_exec_check
        return atsas_exec_check(self)

    def update_setting_info( self, clear_temp=False ):
        set_setting( 'in_folder', self.in_folder.get() )
        set_setting( 'uv_folder', self.uv_folder.get() )
        set_setting( 'uv_file',   self.uv_file.get() )
        set_setting( 'an_folder', self.an_folder.get() )
        set_setting( 'auto_number', self.auto_number.get() )
        self.cfs_entry.apply_entries()
        self.save_settings(clear_temp=clear_temp)

    def update_uv_folder_entry_state( self ):
        self.use_xray_conc = get_setting( 'use_xray_conc' ) == 1
        self.use_mtd_conc = get_setting( 'use_mtd_conc' ) == 1

        uv_folder_entry = self.uv_folder_entry.entry
        if self.use_xray_conc or self.use_mtd_conc:
            in_folder = self.in_folder.get()
            if not is_empty_val( in_folder ):
                set_setting( 'file_extension', self.file_extension.get() )
                # TODO: make this symmetric with load_from_folders
                self.loader.load_xray_data_only( in_folder )
                self.loader.wait_until_ready()
            if self.use_xray_conc:
                if self.loader.has_enough_num_files():
                    text = 'No UV data; using Xray-proportional data instead'
                else:
                    # self.fig_frame.clear_figures("Guinier analysis only is\navaiblable for non-serial data.")
                    text = 'No UV data; using non-serial data'
                uv_folder_entry.config( state=Tk.NORMAL )
                uv_folder_entry.insert(0, text)
                uv_folder_entry.config( state='readonly' )
                uv_folder_entry.config( fg='orange' )
                uv_folder_entry.config( justify='center' )
                self.uv_file_entry.config( state=Tk.DISABLED )
                normal_state = False
            else:
                normal_state = True
        else:
            normal_state = True

        if normal_state:
            uv_folder_entry.config( state=Tk.NORMAL )
            uv_folder_entry.config( fg='black' )
            uv_folder_entry.config( justify='left' )
            self.uv_file_entry.config( state=Tk.NORMAL )

        self.update_analysis_button_state()

    def doing_mfc(self):
        return self.use_mtd_conc == 1

    def plot_results( self ):
        print( 'plot_results' )
        from molass_legacy.GuinierAnalyzer.SimpleGuinierAnalyzer import SimpleGuinierAnalyzer
        from molass_legacy.GuinierAnalyzer.SimpleGuinierFolderAnalyzer import SimpleGuinierFolderAnalyzer
        in_folder = self.in_folder.get()
        sg_analyzer = SimpleGuinierAnalyzer( self.maintenance_log )
        report_folder = '/'.join( self.maintenance_log.split('/')[:-2]  )
        fo_analyzer = SimpleGuinierFolderAnalyzer( self, -1, in_folder,
                        sg_analyzer.folder_infos[0],
                        sg_analyzer.x, sg_analyzer.z, sg_analyzer.r,
                        sg_analyzer.type_names,
                        report_folder=report_folder )

        fo_analyzer.show()

    def update_plot_button_state( self ):
        maintenance_mode = get_setting( 'maintenance_mode' )
        if maintenance_mode == 0:
            self.plot_button.grid_forget()
            self.maintenance_log = None
        else:
            self.plot_button.grid( row=0, column=1, padx=5 )
            self.maintenance_log = self.make_maintenance_log_path()
        if self.same_folder_in_maintenance_log():
            state = Tk.NORMAL
        else:
            state = Tk.DISABLED
        self.plot_button.config( state=state )

    def same_folder_in_maintenance_log( self ):
        if self.maintenance_log is None:
            return False
        if not os.path.exists( self.maintenance_log ):
            return False

        fh = open( self.maintenance_log )
        line = fh.readline()
        fh.close()

        folder = line.replace( 'Doing ', '' )
        return folder[0:-1] == self.in_folder.get()

    def make_maintenance_log_path( self ):
        an_folder       = self.an_folder.get()
        analysis_name   = self.analysis_name.get()
        if is_empty_val(an_folder) or is_empty_val(analysis_name):
            return None

        return an_folder + '/' + analysis_name + '/.temp/maintenance.log'

    def save_settings( self, clear_temp=False ):
        restore_default_setting( 'file_extension' )
        if clear_temp:
            clear_temporary_settings()
        try:
            save_settings()
            AutorgKek.Settings.save_settings()    # for saving AutorgKek settings
        except:
            pass

    def set_dev_setting( self, item, value ):
        set_dev_setting( item, value )

    def set_setting( self, item, value ):
        set_setting( item, value )

    def _exit(self):
        if self.cleaner is not None:
            if self.tmp_logger is None:
                # this can be avoided; temp fix
                pass
            else:
                self.tmp_logger.info("os._exit(0)")
            os._exit(0)

    def quit( self, immediately=False ):
        # print( 'quit' )

        if immediately:
            self.parent.quit()
            self.destroy()
            self._exit()
            return

        reply = MessageBox.askokcancel(
                "Quit Confirmation",
                "Do you really want to quit?",
                parent=self,
                )
        if not reply:
            return

        from molass_legacy.DENSS.DenssManagerDialog import terminate_manager
        ret = terminate_manager(self)
        if not ret:
            return

        from molass_legacy._MOLASS.Processes import terminate_all_processes
        terminate_all_processes()

        # print( self.geometry() )
        # set_setting( 'main_geometry',   self.geometry()          )

        """
        if self.settings_have_been_changed:
            reply = MessageBox.askyesno(
                    "Setting Info Save Question",
                    "Save the setting information?",
                    parent=self,
                    )
            if reply:
                save_settings()
        """
        save_settings(clear_no_save_items=True)
        self.update_setting_info( clear_temp=True )
        self.parent.quit()
        self.destroy()
        self._exit()

    def is_placed_foremost( self ):
        # TODO
        # num_stacked = len( self.tk.eval( 'wm stackorder '+str(self.top_widget) ).split(' ') )
        # return num_stacked == 1
        return True

    def _on_control_c( self, event ):
        w = self.focus_get()
        # print( 'table=', self.file_info_table )
        # print( 'w=', w )
        if str(w).find( str(self.file_info_table) ) == 0:
            self.file_info_table.table._on_control_c( event )
        else:
            # TODO:  w.selection
            pass

    def _on_click( self, event ):
        # the follwing order is essential to keep exluded marks (i.e., 'X')
        # unchanged in case of non-cell clicks.
        self.file_info_table.table._on_click( event )
        self.file_info_table._on_click( event )

    def _on_enter( self, event ):
        # TODO: only when changed
        # self.on_entry_in_folder()
        if not self.is_placed_foremost(): return

        print( 'ENTER in main window' )
        state = self.analysis_button.cget( 'state' )
        if state == Tk.NORMAL:
            self.analyze_all()
        else:
            self.in_folder_entry.on_entry()

    def _on_escape( self, event ):
        if not self.is_placed_foremost(): return

        self.quit()

    def do_simple_guinier( self, i ):
        from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
        from molass_legacy.GuinierAnalyzer.AutorgKekAdapter import AutorgKekAdapter
        print( 'do_simple_guinier: i=', i )

        data, file, _, n = self.file_info_table.get_data_array( i )

        guinier = SimpleGuinier( data )
        adapter = AutorgKekAdapter( None, guinier=guinier )
        adapter_result = adapter.run()
        return adapter_result

    def show_threedim_dialog(self, debug=False):
        if debug:
            from importlib import reload
            import Tools.ThreeDimViewer
            reload(Tools.ThreeDimViewer)
        from molass_legacy.Tools.ThreeDimViewer import ThreeDimViewer
        self.threedim_dialog = ThreeDimViewer(self.parent, self.measured_data)
        self.threedim_dialog.show()

    def show_lpm_inspector(self):
        from molass_legacy.Baseline.LpmInspect import LpmInspector
        self.baseline_inspector = LpmInspector(self.parent, self.measured_data)
        self.baseline_inspector.show()

    def disable_uv_data_tracer(self, *args):
        if not self.disable_uv_data_tracing:
            print('skipping disable_uv_data_tracer')
            return

        self.disable_uv_data_tracer_busy = True

        disable_uv_data = self.disable_uv_data.get()
        set_setting('disable_uv_data', disable_uv_data)
        if disable_uv_data:
            self.uv_folder.set('')
            self.uv_file.set('')
        self.uv_file_info_changed = True
        self.refresh(keeping_setting_items=['disable_uv_data'])

        self.disable_uv_data_tracer_busy = False

    def get_setting_for_test(self, item):
        return get_setting(item)
