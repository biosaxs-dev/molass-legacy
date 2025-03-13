"""

    ファイル名：   TesterDialog.py

    処理内容：

        開発者用の設定変更ダイアログ

    Copyright (c) 2017-2024, SAXS Team, KEK-PF

"""

import sys
import os
import re
import warnings
import time
import glob

from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, Font, FileDialog, is_empty_val
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
import molass_legacy.KekLib.OurMessageBox as MessageBox
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting, set_dev_setting
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry, FileEntry
from molass_legacy.KekLib.BasicUtils import get_home_folder
from molass_legacy._MOLASS.SerialSettings import clear_settings, set_setting
from molass_legacy.KekLib.MultiMonitor import get_max_monitor
from molass_legacy.SerialAnalyzer.DataUtils import get_pytools_folder, get_next_subfolder

drive = __file__.split("\\")[0]

TEST_HOME = drive + r"\TODO\20240806\temp-reports"

def get_home_folder_for_test():
    return TEST_HOME
    # return get_home_folder()

class TesterDialog( Dialog ):
    def __init__( self, parent, title ):
        self.grab = 'local'     # used in grab_set
        self.parent             = parent
        self.title_             = title
        self.v2_analysis_folders = None
        self.applied            = False

    def show( self ):
        self.parent.config( cursor='wait' )
        self.parent.update()

        Dialog.__init__(self, self.parent, self.title_ )

        self.parent.config( cursor='' )

    def body( self, body_frame_ ):   # overrides parent class method

        body_frame = Tk.Frame( body_frame_ )
        body_frame.pack( padx=40, pady=10 )

        tk_set_icon_portable( self )

        clear_button = Tk.Button( body_frame, text="Clear Setting", command=self.clear_settings )
        clear_button.pack( anchor=Tk.NW )

        monitor = get_max_monitor()
        display_info = Tk.Label( body_frame, text="Display Info: " + str(monitor) )
        display_info.pack( anchor=Tk.W )

        # Tester Setting Frame
        test_all_frame = Tk.Frame( body_frame, bd=3, relief=Tk.RIDGE )
        test_all_frame.pack( anchor=Tk.W )

        # Data folder
        test_all_label = Tk.Label( test_all_frame, text="Test/Demo walking the entire folder: " )
        test_all_label.grid( row=0, column=0, pady=5 )

        entry_width = 70
        grid_row = 0

        data_folder = get_pytools_folder() + '/Data'
        self.test_folder = Tk.StringVar()
        self.test_folder.set( data_folder )
        folder_entry = FolderEntry( test_all_frame, textvariable=self.test_folder, width=entry_width )
        folder_entry.grid( row=grid_row, column=1 )

        grid_row += 1
        data_folder = get_pytools_folder() + '/Data_microfluidics'
        self.test_folder_mct = Tk.StringVar()
        self.test_folder_mct.set( data_folder )
        folder_entry = FolderEntry( test_all_frame, textvariable=self.test_folder_mct, width=entry_width )
        folder_entry.grid( row=grid_row, column=1 )

        # Result folder
        grid_row += 1
        label = Tk.Label( test_all_frame, text="Result folder: " )
        label.grid( row=grid_row, column=0, pady=5, sticky=Tk.E )
        self.result_folder = Tk.StringVar()
        self.result_folder.set(get_home_folder_for_test())
        folder_entry = FolderEntry( test_all_frame, textvariable=self.result_folder, width=entry_width )
        folder_entry.grid( row=grid_row, column=1 )

        # V2 result folder
        grid_row += 1
        test_all_label = Tk.Label( test_all_frame, text="V2 report folder: " )
        test_all_label.grid( row=grid_row, column=0, pady=5, sticky=Tk.E )
        self.v2_report_folder = Tk.StringVar()
        folder_entry = FolderEntry( test_all_frame, textvariable=self.v2_report_folder, width=entry_width )
        folder_entry.grid( row=grid_row, column=1 )
        self.v2_report_folder.trace("w", self.v2_result_folder_tracer)

        # Data Restriction
        grid_row += 1
        data_restriction_label = Tk.Label( test_all_frame, text="Data restriction: " )
        data_restriction_label.grid( row=grid_row, column=0, sticky=Tk.E)

        self.minimum_only = Tk.IntVar()
        self.minimum_only.set( 0 )
        minimum_only_cb = Tk.Checkbutton( test_all_frame, text="minimum only",
                                variable=self.minimum_only )
        minimum_only_cb.grid( row=grid_row, column=1, sticky=Tk.W )

        # Dataset Filter
        grid_row += 1
        dataset_filter_label = Tk.Label( test_all_frame, text="Dataset filter: " )
        dataset_filter_label.grid( row=grid_row, column=0, sticky=Tk.E)

        self.filter_book = Tk.StringVar()
        # book_path = os.environ["USERPROFILE"] + r"\Dropbox\_MOLASS\論文\analysis_results\selection.xlsx"
        # self.filter_book.set(book_path)
        self.filter_book_entry = FileEntry(test_all_frame, textvariable=self.filter_book, width=entry_width)
        self.filter_book_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        # Test patterns
        grid_row += 1
        test_patterns_label = Tk.Label( test_all_frame, text="Test patterns: " )
        test_patterns_label.grid( row=grid_row, column=0, sticky=Tk.E)

        test_patterns_frame = Tk.Frame( test_all_frame )
        test_patterns_frame.grid( row=grid_row, column=1, sticky=Tk.W)

        restart_str, restart_repo_dir, test_pattern, compare_folder = self.get_restart_info()
        if restart_str is None:
            restart_str = ""

        if test_pattern is None:
            test_pattern = '5'

        self.test_patterns = Tk.StringVar()
        self.test_patterns.set( '[' + test_pattern + ']')
        self.test_patterns_entry = Tk.Entry( test_patterns_frame, textvariable=self.test_patterns, width=10 )
        self.test_patterns_entry.grid( row=0, column=0, sticky=Tk.W)

        self.mapping_only = Tk.IntVar()
        self.mapping_only.set( 1 )
        mapping_only_cb = Tk.Checkbutton( test_patterns_frame, text="mapping only",
                                variable=self.mapping_only )
        mapping_only_cb.grid( row=0, column=1 )

        self.debug_trial = Tk.IntVar()
        self.debug_trial.set( 0 )
        debug_trial_cb = Tk.Checkbutton( test_patterns_frame, text="debug trial",
                                variable=self.debug_trial )
        debug_trial_cb.grid( row=0, column=2 )

        # Outline Options
        grid_row += 1
        outline_options_label = Tk.Label( test_all_frame, text='Outline Options: ' )
        outline_options_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.save_outline_figures = Tk.IntVar()
        self.save_outline_figures.set( 1 )

        cb = Tk.Checkbutton( test_all_frame, text="save outline figures",
                                variable=self.save_outline_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W )

        grid_row += 1
        self.save_3d_figures = Tk.IntVar()
        self.save_3d_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save 3d figures",
                                variable=self.save_3d_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        grid_row += 1
        self.save_baseline_figures = Tk.IntVar()
        self.save_baseline_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save baseline figures",
                                variable=self.save_baseline_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        # Mapping Options
        grid_row += 1
        mapping_options_label = Tk.Label( test_all_frame, text='Mapping Options: ' )
        mapping_options_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.save_mapping_figures = Tk.IntVar()
        self.save_mapping_figures.set( 1 )

        cb = Tk.Checkbutton( test_all_frame, text="save mapping figures",
                                variable=self.save_mapping_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        grid_row += 1
        self.save_cdi_figures = Tk.IntVar()
        self.save_cdi_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save cdi figures",
                                variable=self.save_cdi_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        grid_row += 1
        self.save_decomp_figures = Tk.IntVar()
        self.save_decomp_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save decomposition figures",
                                variable=self.save_decomp_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        # Preview Options
        grid_row += 1
        preview_options_label = Tk.Label( test_all_frame, text='Preview Options: ' )
        preview_options_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.save_preview_figures = Tk.IntVar()
        self.save_preview_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save preview figures",
                                variable=self.save_preview_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        grid_row += 1
        self.save_preview_results = Tk.IntVar()
        self.save_preview_results.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save preview results",
                                variable=self.save_preview_results  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        grid_row += 1
        self.save_preview_cdi_figures = Tk.IntVar()
        self.save_preview_cdi_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save preview cdi figures",
                                variable=self.save_preview_cdi_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        # V2 Options
        grid_row += 1
        v2_options_label = Tk.Label( test_all_frame, text='V2 Options: ' )
        v2_options_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.save_peakeditor_figures = Tk.IntVar()
        self.save_peakeditor_figures.set( 0 )

        cb = Tk.Checkbutton( test_all_frame, text="save peakeditor figures",
                                variable=self.save_peakeditor_figures  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )

        # Restart file string
        grid_row +=1
        start_infolder_string_label = Tk.Label( test_all_frame, text="Restart file string: " )
        start_infolder_string_label.grid( row=grid_row, column=0, sticky=Tk.E)

        self.start_infolder_string = Tk.StringVar()
        if restart_str is not None:
            self.start_infolder_string.set( restart_str )
        self.start_infolder_string_entry = Tk.Entry( test_all_frame, textvariable=self.start_infolder_string, width=20 )
        self.start_infolder_string_entry.grid( row=grid_row, column=1, sticky=Tk.W)

        # Restart report-subfolder
        grid_row +=1
        report_subfolder_label = Tk.Label( test_all_frame, text="Restart report-subfolder: " )
        report_subfolder_label.grid( row=grid_row, column=0, sticky=Tk.E)

        self.report_subfolder = Tk.StringVar()
        if restart_repo_dir is not None:
            self.report_subfolder.set( restart_repo_dir )
        self.report_subfolder_entry = Tk.Entry( test_all_frame, textvariable=self.report_subfolder, width=20 )
        self.report_subfolder_entry.grid( row=grid_row, column=1, sticky=Tk.W)

        grid_row +=1
        guinier_propgram_label = Tk.Label( test_all_frame, text="Guinier analysis program: " )
        guinier_propgram_label.grid( row=grid_row, column=0, sticky=Tk.E)

        guinier_propgram_frame = Tk.Frame( test_all_frame )
        guinier_propgram_frame.grid( row=grid_row, column=1, sticky=Tk.W)

        self.use_simpleguinier = Tk.IntVar()
        self.use_simpleguinier.set( get_dev_setting( 'use_simpleguinier' ) )
        for i, pname in enumerate([ 'SimpleGuinier', 'GP-Guinier' ]):
            rb = Tk.Radiobutton( guinier_propgram_frame, text=pname,
                        variable=self.use_simpleguinier, value=1-i,
                        )
            rb.grid( row=0, column=i, sticky=Tk.W )

        # Constant Term in Regression
        grid_row += 1
        extrapolation_options_label = Tk.Label( test_all_frame, text='Extrapolation Options: ' )
        extrapolation_options_label.grid( row=grid_row, column=0, sticky=Tk.E )

        # save extrapolated data
        self.tester_zx_save = Tk.IntVar()
        self.tester_zx_save.set( get_dev_setting( 'tester_zx_save' ) )

        zx_out_folder_cb = Tk.Checkbutton( test_all_frame,
                                        text='save extrapolated data',
                                        variable=self.tester_zx_save,
                                        state=Tk.NORMAL )
        zx_out_folder_cb.grid( row=grid_row, column=1, sticky=Tk.W )

        # 
        grid_row += 1
        shutdown_options_label = Tk.Label( test_all_frame, text='Shutdown Options: ' )
        shutdown_options_label.grid( row=grid_row, column=0, sticky=Tk.E )

        self.shutdown_machine = Tk.IntVar()
        self.shutdown_machine.set(0)

        cb = Tk.Checkbutton( test_all_frame, text="power off the machine when finished",
                                variable=self.shutdown_machine  )
        cb.grid( row=grid_row, column=1, sticky=Tk.W  )


        # Report folder to compare with
        grid_row += 1
        to_compare_label = Tk.Label( test_all_frame, text="Report folder to compare with: " )
        to_compare_label.grid( row=grid_row, column=0, pady=5 )

        self.to_compare_folder = Tk.StringVar()
        if compare_folder is None:
            compare_folder = ''
        self.to_compare_folder.set( compare_folder )
        folder_entry = FolderEntry( test_all_frame, textvariable=self.to_compare_folder, width=entry_width )
        folder_entry.grid( row=grid_row, column=1 )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

        self.update()

    def previous_results_exist(self):
        reports_folder = self.home + '/reports'
        if os.path.exists(reports_folder):
            nodes = os.listdir(reports_folder)
            ret = len(nodes) > 0
        else:
            ret = False
        return ret

    def rename_previous_results(self):
        ext_re = re.compile(r"-?(\d*)$")
        def new_postfix(m):
            # print('new_postfix: ', m.group(1))
            if m.group(1) == '':
                n = 0
            else:
                n = int(m.group(1))
            return '-' + str(n+1)

        def get_new_folder_name(folder):
            folder_ = folder
            while os.path.exists(folder_):
                folder_ = re.sub(ext_re, new_postfix, folder_, 1)   # the 4-th arg is required to let it replace only once
            return folder_

        for name in ['reports', 'images-outline', 'images-baseline',
                        'images-3d', 'images-mapping',
                        'images-cdi',
                        'images-decomp-uv', 'images-decomp-xray', 'images-preview',
                        'images-preview-cdi',
                        'images-peakeditor',
                        'qmm-images',
                        'images-fullopt_init']: # TODO: unifiy these names with those in Tester.py
            folder = os.path.join(self.home, name)
            if os.path.exists(folder):
                new_folder = get_new_folder_name(folder)
                os.rename(folder, new_folder)

    def get_last_report_folder(self):
        repo_folder = os.path.join(self.home, 'reports')
        n = 1
        while os.path.exists(repo_folder + '-' + str(n)):
            n += 1
        n -= 1
        return repo_folder + '-' + str(n)

    def to_be_conitued_reports_dir_exists( self ):
        self.to_be_continued_reports_dir = self.get_last_report_folder()
        return os.path.exists( self.to_be_continued_reports_dir )

    def get_restart_info( self ):
        self.home = get_home_folder_for_test()

        if self.previous_results_exist():
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            ok = MessageBox.askyesno('Previous Results',
                                ("Previous results seem to exist.\n"
                                "Would you like to rename them?"),
                                parent=self)
            if ok:
                self.rename_previous_results()
                MessageBox.showinfo('Previous Results',
                                "Previous results are successfully renamed.",
                                parent=self)

        self.restart = self.to_be_conitued_reports_dir_exists()
        if not self.restart:
            return [ None ] * 4

        repo_dirs = os.listdir( self.to_be_continued_reports_dir )
        # print( 'repo_dirs=', repo_dirs )
        last_successful_dir = self.to_be_continued_reports_dir + '/' + repo_dirs[-2]
        print( 'last_successful_dir=', last_successful_dir )
        last_successful_log = last_successful_dir + '/molass.log'
        start_line_re = re.compile( r'start analysis for (.+) with test pattern (\d+)' )
        fh = open( last_successful_log )
        restart_str, restart_repo_dir = None, None
        test_pattern = ''
        for line in fh:
            m = start_line_re.search( line )
            if m:
                next_subfolder = get_next_subfolder( self.test_folder.get(), m.group(1) )
                restart_str = '/'.join( next_subfolder.split( '/' )[-2:] )
                restart_repo_dir = repo_dirs[-1]
                test_pattern = m.group(2)
                break

        if test_pattern == '':
            restart_str, restart_repo_dir, test_pattern = self.get_restart_info_v2(last_successful_log, repo_dirs)

        fh.close()
        compare_folder = self.get_previous_compare_folder()
        print( 'compare_folder=', compare_folder )

        return restart_str, restart_repo_dir, test_pattern, compare_folder

    def get_restart_info_v2(self, last_successful_log, repo_dirs):
        start_line_re = re.compile( r"started loading (\S+)" )
        fh = open( last_successful_log )
        restart_str, restart_repo_dir = None, None
        test_pattern = ''
        for line in fh:
            m = start_line_re.search(line)
            if m:
                next_subfolder = get_next_subfolder( self.test_folder.get(), m.group(1) )
                restart_str = '/'.join( next_subfolder.split( '/' )[-2:] )
                restart_repo_dir = repo_dirs[-1]
                test_pattern = "20"
                break
        fh.close()
        return restart_str, restart_repo_dir, test_pattern

    def get_previous_compare_folder( self ):
        folder = None
        if False:
            # get_log_path was removed
            from molass_legacy.Test.TesterLogger import get_log_path
            log_path = get_log_path()
            if os.path.exists( log_path ):
                fh = open( log_path )
                previous_start_line = fh.readline()
                print( 'previous_start_line=', previous_start_line )
                fh.close()
                compare_folder_re = re.compile( r'with compare folder (.+)$' )
                m = compare_folder_re.search( previous_start_line )
                if m:
                    folder = m.group(1)
        return folder

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Run", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def clear_settings( self ):
        ok = MessageBox.askyesno( 'Warninig', 'This operations is not cancelable. OK?' )
        if ok:
            clear_settings()

    def v2_result_folder_tracer(self, *args):
        self.test_patterns.set("[20]")

    def validate(self):
        if is_empty_val(self.result_folder.get()):
            MessageBox.showerror('Invalid Test Patterns',
                                "Result folder must not be empty.\n",
                                parent=self)
            return 0

        try:
            test_patterns = eval(self.test_patterns.get())
            v2_report_folder = self.v2_report_folder.get()

            if (test_patterns[0] < 20 or test_patterns[0] >= 30) and is_empty_val(v2_report_folder):
                return 1

            assert not is_empty_val(v2_report_folder)
        except:
            # TODO: restore V2 result folder
            MessageBox.showerror('Invalid Test Patterns',
                                "Test Pattern must be in [20, 30) for V2 result trace\n"
                                "or V2 result folder must not be empty.",
                                parent=self)
            return 0

        try:
            self.v2_analysis_folders = glob.glob(v2_report_folder + r"\analysis-*")
            assert len(self.v2_analysis_folders) > 0
        except:
            MessageBox.showerror('Invalid V2 Result Folder',
                                "The folder does not seem to properly include V2 results.",
                                parent=self)
            return 0

        return 1

    def apply( self ):  # overrides parent class method
        self.applied    = True
        set_dev_setting( 'use_simpleguinier',       self.use_simpleguinier.get() )
        set_dev_setting( 'tester_zx_save',          self.tester_zx_save.get() )
        self.test_all()

    def test_all( self ):
        # withdraw/deiconify does not seem to be appropriate here

        # enable DebugPlot during test-execution
        from molass_legacy.KekLib.DebugPlot import set_plot_env
        set_plot_env(sub_parent=self.parent)

        set_dev_setting('running_with_tester', True)

        try:
            print( self.test_patterns.get() )
            test_patterns = eval( self.test_patterns.get() )
            assert type( test_patterns) == list, 'Not a list'
        except Exception as exc:
            print( exc )
            self.test_patterns_entry.config( fg='red' )
            MessageBox.showerror( 'Value Error', str(exc), parent=self )
            self.test_patterns_entry.focus_force()
            return

        filter_book = self.filter_book.get()
        if is_empty_val(filter_book):
            ds_filter = None
        else:
            try:
                from .DatasetFilter import DatasetFilter
                assert os.path.exists(filter_book)

                ds_filter = DatasetFilter(filter_book)
            except Exception as exc:
                print( exc )
                self.filter_book_entry.config( fg='red' )
                MessageBox.showerror( 'Value Error', str(exc), parent=self )
                self.filter_book_entry.focus_force()
                return

        self.test_patterns_entry.config( fg='black' )

        to_compare_logfolder = self.to_compare_folder.get()
        if is_empty_val( to_compare_logfolder ):
            to_compare_logfolder = None

        def test_all_closure():
            from molass_legacy.Test.Tester import Tester
            tester = Tester( self.parent )
            test_folder = self.test_folder.get()
            test_folder_mct = self.test_folder_mct.get()
            mapping_only = self.mapping_only.get() == 1
            debug_trial = self.debug_trial.get() == 1
            tester.test_all( [test_folder, test_folder_mct],
                                result_folder=self.result_folder.get(),
                                minimum_only=self.minimum_only.get(),
                                ds_filter=ds_filter,
                                mapping_only=mapping_only,
                                debug_trial=debug_trial,
                                save_outline_figures=self.save_outline_figures.get(),
                                save_3d_figures=self.save_3d_figures.get(),
                                save_baseline_figures=self.save_baseline_figures.get(),
                                save_mapping_figures=self.save_mapping_figures.get(),
                                save_cdi_figures=self.save_cdi_figures.get(),
                                save_decomp_figures=self.save_decomp_figures.get(),
                                save_preview_figures=self.save_preview_figures.get(),
                                save_preview_results=self.save_preview_results.get(),
                                save_preview_cdi_figures=self.save_preview_cdi_figures.get(),
                                save_peakeditor_figures=self.save_peakeditor_figures.get(),
                                shutdown_machine=self.shutdown_machine.get(),
                                test_patterns=test_patterns,
                                start_str=self.start_infolder_string.get(),
                                report_subfolder=self.report_subfolder.get(),
                                to_compare_logfolder=to_compare_logfolder,
                                v2_analysis_folders=self.v2_analysis_folders,
                                )

        self.parent.after( 100, test_all_closure )
        self.destroy()

class LogDiffDialog( Dialog ):
    def __init__( self, parent, title ):
        self.grab = 'local'     # used in grab_set
        self.parent             = parent
        self.title_             = title
        self.applied            = False

    def show( self ):
        self.parent.config( cursor='wait' )
        self.parent.update()

        Dialog.__init__(self, self.parent, self.title_ )

        self.parent.config( cursor='' )

    def body( self, body_frame_ ):   # overrides parent class method

        body_frame = Tk.Frame( body_frame_ )
        body_frame.pack( padx=40, pady=10 )

        tk_set_icon_portable( self )

        comming_label = Tk.Label( body_frame, text='Comming soon' )
        comming_label.pack()
