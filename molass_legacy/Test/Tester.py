"""

    SerialAnalyzer.Tester.py

    Copyright (c) 2017-2024, SAXS Team, KEK-PF

"""

import os
import re
import queue
import numpy        as np
from time           import sleep
from datetime import datetime
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.SerialAnalyzer.DataUtils import serial_folder_walk, mct_folder_walk
from molass_legacy.KekLib.TkTester import TestClient, messagebox_shown
from molass_legacy.KekLib.MessageBoxUtils import reply_messagebox, window_exists
from molass_legacy.KekLib.BasicUtils import clear_dirs_with_retry, Struct
from molass_legacy._MOLASS.SerialSettings import clear_settings, get_setting, set_setting, DF_PATH_LENGTH, DF_PATH_LENGTH_OLD
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass_legacy.Test.TesterLogger import *

success_rg_re   = re.compile( r'Rg from (\S+) is estimated to be (\S+) with quality (\S+)\.' )
failure_rg_re   = re.compile( r'Rg estimation failed in \S+\.' )
pytools_ver_rg  = re.compile( r'pytools-\w+-develop' )
final_std_diff_str  = 'final_std_diff='

INVESTIGATE_ABSORBANCE_ANOMALY  = False
MINIMUM_DATA_NAMES = [
    'pH6',          # peak consistency when noisy (one peak)
    'pH7',          # peak consistency when noisy (two peaks)
    '20190315_3',   # No decomposition
    '20191006_proteins5',   # many peaks, 10-sigma rule covering
    '20191031',     # use_xray_conc
    'sample_data',  # standard
    '20190522',     # MCT-SAXS
    ]

PATH_LENGTH_DICT = {
    'pH6':DF_PATH_LENGTH_OLD,
    'pH7':DF_PATH_LENGTH_OLD,
    '20190315_3':DF_PATH_LENGTH_OLD,
    '20191006_proteins5':DF_PATH_LENGTH,
    '20191031':DF_PATH_LENGTH,
    'sample_data':DF_PATH_LENGTH_OLD,
    '20190522':DF_PATH_LENGTH_OLD,
    }

fromlog_rg_list = None

def compare_tester_rg( i, rg, quality_boundary ):
    if fromlog_rg_list is None:
        return

    if i >= len( fromlog_rg_list ):
        put_to_tester_log_queue( 'No %d-th result to compare\n' % (i) )
        return

    from_rec = fromlog_rg_list[i]
    if from_rec is None:
        from_rg         = None
        from_quality    = None
    else:
        from_rg         = from_rec[0]
        from_quality    = from_rec[1]
    if from_rg is None:
        if rg is None:
            pass
        else:
            put_to_tester_log_queue( 'got new Rg(%.3g) from %d-th file\n' % (rg, i) )
    else:
        if rg is None:
            put_to_tester_log_queue( 'got None Rg from %d-th file\n' % i )
        else:
            if from_rg > 0:
                diff_ratio = abs( rg - from_rg ) / from_rg
                if from_quality > quality_boundary:
                    allow_diff  = 0.01
                else:
                    allow_diff  = 0.1
                if diff_ratio > allow_diff:
                    put_to_tester_log_queue( 'got different Rg(%.3g != %.3g) from %d-th file\n' % (rg, from_rg, i) )
            else:
                pass

class Tester:
    def __init__( self, app ):
        self.app   = app

    def test_all( self, test_folders, result_folder=None,
                    report_folder=None, report_subfolder=None, filter_str=None, start_str=None,
                    test_patterns=[0,1,2],
                    minimum_only=False,
                    ds_filter=None,
                    mapping_only=False,
                    debug_trial=False,
                    save_outline_figures=False,
                    save_3d_figures=False,
                    save_baseline_figures=False,
                    save_mapping_figures=False,
                    save_cdi_figures=False,
                    save_decomp_figures=False,
                    save_preview_figures=False,
                    save_preview_results=False,
                    save_preview_cdi_figures=False,
                    save_peakeditor_figures=False,
                    shutdown_machine=False,
                    to_compare_logfolder=None,
                    v2_analysis_folders=None,
                    ):
        global log_fh, tester_log_queue
        tester_log_queue = queue.Queue()
        self.test_folders   = test_folders
        self.filter_str     = filter_str
        if start_str is not None and len( start_str.replace( ' ', '' ) ) == 0:
            start_str = None
        self.start_str      = start_str
        # self.start_str      = '20161217'
        self.minimum_only   = minimum_only
        self.ds_filter      = ds_filter
        self.mapping_only   = mapping_only
        self.allow_mct_folder = 0
        self.debug_trial    = debug_trial
        self.save_outline_figures   = save_outline_figures
        self.save_3d_figures        = save_3d_figures
        self.save_baseline_figures  = save_baseline_figures
        self.save_mapping_figures   = save_mapping_figures
        self.save_cdi_figures       = save_cdi_figures
        self.save_decomp_figures    = save_decomp_figures
        self.save_preview_figures   = save_preview_figures
        self.save_preview_results   = save_preview_results
        self.save_preview_cdi_figures = save_preview_cdi_figures
        self.save_peakeditor_figures = save_peakeditor_figures
        self.shutdown_machine = shutdown_machine
        self.to_compare_logfolder = to_compare_logfolder
        self.v2_analysis_folders = v2_analysis_folders
        self.diff_count     = 0
        self.error_count    = 0
        self.mdiff_count    = 0
        self.test_patterns  = test_patterns
        if result_folder is None:
            this_dir = os.path.dirname( os.path.abspath( __file__ ) )
            home_dir = this_dir + '/../..'
        else:
            home_dir = result_folder
        self.log    = open_tester_log( home_dir )

        if report_folder is None:
            report_folder   = os.path.abspath( home_dir + '/reports' ).replace( '\\', '/' )

        self.outline_image_folder = None
        self.threed_image_folder = None
        self.baseline_image_folder = None
        self.mapping_image_folder = None
        self.decomp_image_folder_xray = None
        self.decomp_image_folder_uv = None
        self.preview_image_folder = None
        self.qmm_image_folder = None
        self.fulloptinit_image_folder = None
        self.result_folder = result_folder

        dirs_to_clear = [ report_folder ]

        if self.save_outline_figures:
            self.outline_image_folder   = os.path.abspath( home_dir + '/images-outline' ).replace( '\\', '/' )
            dirs_to_clear.append( self.outline_image_folder )
        if self.save_3d_figures:
            self.threed_image_folder   = os.path.abspath( home_dir + '/images-3d' ).replace( '\\', '/' )
            dirs_to_clear.append( self.threed_image_folder )
        if self.save_baseline_figures:
            self.baseline_image_folder   = os.path.abspath( home_dir + '/images-baseline' ).replace( '\\', '/' )
            dirs_to_clear.append( self.baseline_image_folder )
        if save_mapping_figures:
            self.mapping_image_folder   = os.path.abspath( home_dir + '/images-mapping' ).replace( '\\', '/' )
            dirs_to_clear.append( self.mapping_image_folder )
        if self.save_cdi_figures:
            self.cdi_image_folder   = os.path.abspath( home_dir + '/images-cdi' ).replace( '\\', '/' )
            dirs_to_clear.append( self.cdi_image_folder )
        if save_decomp_figures:
            self.decomp_image_folder_xray   = os.path.abspath( home_dir + '/images-decomp-xray' ).replace( '\\', '/' )
            dirs_to_clear.append( self.decomp_image_folder_xray )
            self.decomp_image_folder_uv = os.path.abspath( home_dir + '/images-decomp-uv' ).replace( '\\', '/' )
            dirs_to_clear.append( self.decomp_image_folder_uv )
        if save_preview_figures:
            self.preview_image_folder   = os.path.abspath( home_dir + '/images-preview' ).replace( '\\', '/' )
            dirs_to_clear.append( self.preview_image_folder )
        if self.save_preview_cdi_figures:
            self.preview_cdi_image_folder = os.path.abspath( home_dir + '/images-preview-cdi' ).replace( '\\', '/' )
            dirs_to_clear.append( self.preview_cdi_image_folder )
        if self.save_peakeditor_figures:
            self.peakeditor_image_folder = os.path.abspath( home_dir + '/images-peakeditor' ).replace( '\\', '/' )
            dirs_to_clear.append( self.peakeditor_image_folder )

        if self.test_patterns[0] in [33333]:
            self.srr_folder = os.path.abspath( home_dir + '/srr' ).replace( '\\', '/' )
            dirs_to_clear.append( self.srr_folder )

        if self.test_patterns[0] in [4]:
            print("home_dir=", home_dir)
            survey_file = os.path.abspath( home_dir + '/mapping_params.csv' )
            self.survey_fh = open(survey_file, "w")

        if self.test_patterns[0] in [9, 10]:
            self.fulloptinit_image_folder = os.path.abspath( home_dir + '/images-fullopt_init' ).replace( '\\', '/' )
            dirs_to_clear.append( self.fulloptinit_image_folder )

        if self.test_patterns[0] >= 20 and self.test_patterns[0] < 30:
            self.make_v2_results_dict()

        while True:
            clear_dirs_with_retry( dirs_to_clear )
            sleep( 1 )
            if os.path.exists( report_folder ):
                break
            print( "retrying to clear %s." % report_folder )

        allow_angular_slope_in_mf = get_setting( 'allow_angular_slope_in_mf' )
        maintenance_mode = get_setting( 'maintenance_mode' )

        clear_settings()

        set_setting('allow_angular_slope_in_mf', allow_angular_slope_in_mf)
        set_setting('maintenance_mode', maintenance_mode)
        set_setting('test_pattern', test_patterns[0])
        set_setting('mapping_image_folder', self.mapping_image_folder)
        set_setting('decomp_image_folder_xray', self.decomp_image_folder_xray)
        set_setting('decomp_image_folder_uv', self.decomp_image_folder_uv)
        set_setting('preview_image_folder', self.preview_image_folder)

        self.report_folder  = report_folder
        self.report_subfolder = report_subfolder
        self.client         = TestClient( self.app, self.test_all_entry, log_func=write_to_log )

    def test_all_entry( self, client, agent ):
        self.client     = client
        self.agent      = agent
        self.counter    = 0
        with_compare = '' if self.to_compare_logfolder is None else ' with compare folder ' + self.to_compare_logfolder
        with_shutdown = ' with shutdown ' if self.shutdown_machine else ''
        write_to_log( 'test start' + with_compare+ with_shutdown + '.\n' )

        for i in self.test_patterns:

            self.test_pattern = i
            # set_setting('test_pattern', self.test_pattern)
            while True:
                if os.path.exists( self.report_folder ):
                    break
                print( "retrying to clear %s." % self.report_folder )
                sleep( 1 )

            if self.test_pattern == 2:
                self.agent.set_setting( 'use_elution_models', 1 )

            agent.analysis_name_entry.focus_force()
            agent.analysis_name_entry.delete( 0, Tk.END)
            if self.counter == 0 and self.report_subfolder is not None and self.report_subfolder !='':
                subfolder = self.report_subfolder
            else:
                subfolder = 'analysis-000'
            agent.analysis_name_entry.insert( 0, subfolder )
            agent.update()

            if self.test_pattern == 33333:
                self.srr_fh = open(self.srr_folder + "/srr.csv", "w")
                if False:
                    self.srr_figs_folder = self.srr_folder + "/figs"
                    if not os.path.exists():
                        os.makedirs(self.srr_figs_folder)

            for k, folder in enumerate(self.test_folders):
                if k == 0:
                    serial_folder_walk(folder, self.test_a_folder)
                elif k == 1:
                    mct_folder_walk(folder, self.test_a_folder)

            if self.test_pattern == 33333:
                self.srr_fh.close()

            if self.test_pattern == -1:
                if self.counter == 0:
                    from importlib import reload
                    import Test.TempInspection
                    reload(Test.TempInspection)
                from molass_legacy.Test.TempInspection import inspection_close
                inspection_close(agent)

            save_folder = self.report_folder + '-' + str(i)
            os.rename( self.report_folder, save_folder )
            clear_dirs_with_retry( [ self.report_folder ] )

        write_to_log( 'done.\n' )
        if self.to_compare_logfolder is not None:
            write_to_log( 'diff  count: %d\n' % self.diff_count )
            write_to_log( 'error count: %d\n' % self.error_count )
            write_to_log( 'mdiff count: %d\n' % self.mdiff_count )
        self.log.close()
        if self.shutdown_machine:
            from molass_legacy.KekLib.Shutdown import shutdown_machine
            shutdown_machine()

    def test_a_folder( self, in_folder, uv_folder, plot=None, suppress_retry_call=False, analysis_name=None ):
        global fromlog_rg_list

        client  = self.client
        agent   = self.agent
        tester_info = Struct(test_pattern=self.test_pattern)
        agent.set_test_mode(tester_info)

        if self.start_str is not None:
            if self.counter == 0:
                self.start_str_found = False
            if in_folder.find( self.start_str ) > 0:
                self.start_str_found = True
            if not self.start_str_found:
                return True, None

        if self.minimum_only:
            self.data_name_key = None
            found = False
            for name_key in MINIMUM_DATA_NAMES:
                if in_folder.find(name_key) >= 0:
                    found = True
                    self.data_name_key = name_key
                    break

            print(in_folder, found)
            if not found:
                return True, None

        if not self.allow_mct_folder:
            from molass_legacy.SerialAnalyzer.SerialDataUtils import get_mtd_filename
            mtd_file = get_mtd_filename(in_folder)
            if mtd_file is not None:
                return True, None

        if self.ds_filter is not None:
            if self.ds_filter.is_in_the_selection(in_folder):
                pass
            else:
                return True, None

        """
        # modify and uncomment to restart
        if self.test_pattern == 5:
            if self.counter == 0:
                self.start_str_found = False
            if in_folder.find( 'Kosugi8' ) > 0:
                self.start_str_found = True
            if not self.start_str_found:
                return True, None
        """

        if self.filter_str is not None and in_folder.find( self.filter_str ) < 0:
            return True, None

        if self.counter > 0:
            sleep( 0.5 )    # to avoid hang?

        self.log.write( 'Doing ' + in_folder + '\n' )
        self.log.flush()

        self.counter += 1

        agent.in_folder_entry.focus_force()
        agent.in_folder_entry.delete( 0, Tk.END)
        agent.in_folder_entry.insert( 0, in_folder )
        agent.on_entry_in_folder()

        if self.test_pattern in [7]:
            agent.fully_automatic.set(1)
        else:
            agent.fully_automatic.set(0)

        if False:
            if uv_folder != in_folder:
                agent.uv_folder_entry.focus_force()
                agent.uv_folder_entry.delete( 0, Tk.END)
                agent.uv_folder_entry.insert( 0, uv_folder )
                agent.on_entry_uv_folder()

        if self.counter == 1:
            agent.an_folder_entry.focus_force()
            agent.an_folder_entry.delete( 0, Tk.END)
            agent.an_folder_entry.insert( 0, self.report_folder)
            agent.on_entry_an_folder()
            agent.update()      # for Windows 7
            sleep( 1 )

        if analysis_name is not None:
            agent.analysis_name.set( analysis_name )

        assert( agent.an_folder.get() == self.report_folder )
        agent.update(__wait__=False )   # for Windows 7

        agent.wait_until_the_data_is_ready()
        if agent.detected_abnomality():
            print( '---- detected_abnomality' )
            try:
                write_from_log_queue()
                write_to_log( '---- detected_abnomality\n' )
                agent.abnomality_dialog.ok_button.invoke()
                sleep( 1 )
                reply_messagebox('Exclusion Done', 'Y')
            except:
                # this occurs in safe mode
                print('assumed to have been replied manually.')
            sleep( 1 )

        if self.test_pattern == -1:
            if self.counter == 0:
                from importlib import reload
                import Test.TempInspection
                reload(Test.TempInspection)
            from molass_legacy.Test.TempInspection import datarange_problem_inspection
            datarange_problem_inspection(agent, self.result_folder)
            if self.debug_trial and self.counter > 2:
                return False, None
            else:
                return True, None

        agent.set_setting('suppress_low_quality_warning', 1)
        agent.set_test_mode( Struct( test_pattern=self.test_pattern ) )

        if self.test_pattern >= 20 and self.test_pattern < 30:
            self.do_v2_results(in_folder)
            return True, None

        if self.test_pattern in [0]:
            self.do_outline_only()
            return True, None

        if self.test_pattern >= 30:
            self.do_v2_thru_peak_editor()
            return True, None

        if self.test_pattern == 8:
            try:
                self.do_v2_demo()
                return True, None
            except:
                return False, None

        analyzer    = agent.analyzer

        if self.save_outline_figures:
            agent.fig_frame.save_the_figure( self.outline_image_folder, agent.get_analysis_name() )

        if self.test_pattern in [7]:
            return self.do_fully_automatic()

        if self.test_pattern in [5]:
            if self.minimum_only:
                path_length = agent.get_setting_for_test('path_length')
                assert path_length == PATH_LENGTH_DICT[self.data_name_key]

            agent.refresh_button.invoke()
            agent.wait_until_the_data_is_ready()

        if self.test_pattern in [4, 5, 6]:

            if not agent.doing_mfc():

                if self.save_3d_figures:
                    agent.presync_button.invoke(__wait__=False )
                    sleep(1)
                    threedim_dialog = agent.threedim_dialog
                    threedim_dialog.save_the_figure( self.threed_image_folder, agent.analysis_name.get() )
                    threedim_dialog.cancel()

                if self.test_pattern in [5]:
                    agent.restrict_button.invoke(__wait__=False)

                    sleep(1)
                    while True:
                        datarange_dialog = agent.datarange_dialog
                        if datarange_dialog is not None:
                            break
                        print("waiting for datarange_dialog being ready")
                        sleep(1)

                    if not agent.get_setting_for_test('use_xray_conc'):

                        for i in range(3):
                            sleep(1)
                            try:            
                                current_frame = datarange_dialog.get_current_frame()
                                current_frame.toggle_btn.invoke()   # note that current_frame is not an agent widget
                                sleep(1)
                                current_frame = datarange_dialog.get_current_frame()
                                current_frame.toggle_btn.invoke()   # note that current_frame is not an agent widget
                                break
                            except Exception as e:
                                print(e)
                                print([i], "retrying...")
                                continue

                        sleep(1)

                    # datarange_dialog.save_the_figure( self.restrictor_image_folder, agent.analysis_name.get() )
                    datarange_dialog.cancel()

                if self.save_baseline_figures:
                    agent.baseline_button.invoke(__wait__=False )
                    sleep(3)
                    baseline_inspector = agent.baseline_inspector
                    baseline_inspector.save_the_figure( self.baseline_image_folder, agent.analysis_name.get() )
                    baseline_inspector.cancel()
                    agent.update()

            sleep(1)
            # return True, None

        agent.analysis_button.invoke( __wait__=False )
        sleep( 1 )      # take some time to get analyzer.mapper_canvas cleared

        if agent.doing_mfc():
            self.do_mfc()
            return True, None

        while not analyzer.has_mapper_canvas():
            sleep( 1 )

        mapper_canvas = analyzer.mapper_canvas
        if not mapper_canvas.get_edit_mode():
            mapper_canvas.toggle_button.invoke()
        sleep( 1 )

        if self.test_pattern in [33333]:
            mapper_canvas.cancel_button.invoke()
            sleep(0.5)
            scd_list = mapper_canvas.get_cd_list()
            sd = mapper_canvas.get_sd()
            from Rank.SRR import SRR
            srr = SRR(sd)
            # srr_list = srr.compute_judge_info(figs_folder=self.srr_figs_folder, analysis_name=agent.analysis_name.get())
            srr_list = srr.compute_judge_info()
            for k, (scd, rec) in enumerate(zip(scd_list, srr_list)):
                self.srr_fh.write(','.join([in_folder] + ["%s" % v if v is None else "%g" % v for v in (k, scd, *rec)]) + "\n")
                self.srr_fh.flush()
            return True, None

        if self.test_pattern in [4]:
            mapper_canvas.mapper.make_uniformly_scaled_vector(scale=1, survey_fh=self.survey_fh)
            if self.save_mapping_figures:
                mapper_canvas.plotter.save_the_figure( self.mapping_image_folder, agent.analysis_name.get() )
            if True:
                mapper_canvas.cancel_button.invoke()
                sleep(0.5)
                return True, None
            else:
                sleep(0.5)

        if self.test_pattern in [9, 10]:
            try:
                self.do_fullopt_initialization(mapper_canvas)
                if self.test_pattern == 10:
                    self.do_fullopt_execution(mapper_canvas)

                if self.debug_trial:
                    ret_tuple = False, None
                else:
                    ret_tuple = True, None
            except:
                ret_tuple = False, None
            if self.save_mapping_figures:
                mapper_canvas.plotter.save_the_figure( self.mapping_image_folder, agent.analysis_name.get() )
            mapper_canvas.cancel_button.invoke()
            sleep(0.5)
            return ret_tuple

        adjuster = mapper_canvas.adjuster
        conc_type = adjuster.get_conc_type()

        # at this time, app_logfile should have already been set
        self.app_logfile = analyzer.get_logfile()
        if self.to_compare_logfolder is None:
            self.from_logfile   = None
        else:
            self.from_logfile   = '/'.join(  [self.to_compare_logfolder] + self.app_logfile.split( '/' )[-2:] )

        if INVESTIGATE_ABSORBANCE_ANOMALY:
            anomaly = adjuster.observe_base_curve_anomaly()
            print( 'anomaly=', str(anomaly) )

        if self.test_pattern in [1, 2]:
            adjuster.advanced_btn.invoke( __wait__=False )
            sleep( 0.5 )
            dialog = adjuster.advanced_settings_dialog
            dialog.range_type_buttons[1].invoke()
            sleep( 1 )
            dialog.ok_button.invoke()
            sleep( 0.5 )

        if self.test_pattern in [6] and conc_type < 2:
            if self.save_cdi_figures:
                plotter = mapper_canvas.plotter
                for pno in range(len(plotter.get_target_ranges())):
                    plotter.show_cdi_dialog(pno=pno, show_later=True)
                    plotter.cdi_dialog.show( __wait__=False )
                    sleep(1)
                    plotter.cdi_dialog.save_the_figure(self.cdi_image_folder, pno)
                    plotter.cdi_dialog.ok()
                    sleep(0.5)

        if self.test_pattern in [4, 5, 6] and conc_type < 2:
            if self.test_pattern in [4, 6]:
                need_decomp = mapper_canvas.guide_info.decomp_proc_needed()
            else:
                need_decomp = True

            def preview_control(need_decomp, post_fix=""):
                if need_decomp:
                    mapper_canvas.decomp_btn.invoke( __wait__=False )
                    sleep( 1 )
                    decomp_editor = mapper_canvas.decomp_editor
                    while not decomp_editor.is_ready():
                        print( 'waiting for the decomp_editor to be ready.' )
                        sleep( 1 )

                    if self.save_decomp_figures:
                        decomp_editor.save_the_figure( self.decomp_image_folder_xray, agent.analysis_name.get() )

                    if self.test_pattern != 4:

                        decomp_editor.get_current_frame().toggle_btn.invoke()

                        if self.save_decomp_figures:
                            decomp_editor.save_the_figure( self.decomp_image_folder_uv, agent.analysis_name.get() )

                    write_to_log( 'model_fit_error=' + str(mapper_canvas.decomp_editor.get_model_fit_errors()) + '\n' )
                    write_to_log( 'applied model=' + decomp_editor.get_applied_model().get_name() + '\n' )

                    if self.test_pattern != 4:
                        self.do_preview(decomp_editor, post_fix)

                    decomp_editor.ok_button.invoke()
                    print( 'decomp_editor ok' )
                    sleep( 0.5 )
                else:
                    mapper_canvas.reditor_btn.invoke( __wait__=False )
                    sleep( 1 )
                    range_editor = mapper_canvas.range_editor
                    while not range_editor.is_ready():
                        print( 'waiting for the range_editor to be ready.' )
                        sleep( 1 )

                    if self.test_pattern != 4:
                        self.do_preview(range_editor, post_fix)
                    range_editor.ok_button.invoke()
                    print( 'range_editor ok' )
                    sleep( 0.5 )

            if self.mapping_only:
                pass
            else:
                if self.test_pattern == 44444:
                    conc_type = adjuster.get_conc_type()
                    if conc_type == 0:
                        adjuster.uv_baseline_adjust.set(0)
                        adjuster.optimize_btn.invoke()
                        preview_control(need_decomp, post_fix="-bpa")
                        for bpa_var in [adjuster.uv_baseline_with_bpa, adjuster.xray_baseline_with_bpa]:
                            bpa_var.set(0)
                        adjuster.optimize_btn.invoke()
                        preview_control(need_decomp, post_fix="-nonbpa")
                else:
                    preview_control(need_decomp)

        num_peaks = mapper_canvas.mapper.get_num_peaks()
        write_to_log( 'num_peaks=' + str(num_peaks) + '\n' )

        if self.test_pattern in [0, 2, 5, 6]:
            mapper = mapper_canvas.mapper
            sci = mapper.get_sci_list()
            write_to_log( 'sci=' + str(sci) + '\n' )

            mapped_info = mapper.get_mapped_info()
            put_to_tester_log_queue( 'x_ranges=%s\n' % str(mapped_info.x_ranges) )

        if self.save_mapping_figures:
            mapper_canvas.plotter.save_the_figure( self.mapping_image_folder, agent.analysis_name.get() )

        if self.mapping_only:
            mapper_canvas.cancel_button.invoke()
            sleep(0.5)
        else:
            mapper_canvas.ok_button.invoke()
            sleep(0.5)
            self.do_whole()

        self.do_finish( in_folder )

        # if self.counter > 3: return False, None
        # TODO: return serial_data
        if self.debug_trial:
            # stop on return
            return False, None
        else:
            return True, None

    def do_preview(self, dialog, post_fix):
        preview_frame = dialog.preview_frame
        preview_frame.show_zx_preview_button.invoke( __wait__=False )
        sleep( 1 )
        pool = preview_frame.pool
        preview_wait_count = 0
        while not pool.ok() and preview_wait_count < 1200:
            print( 'preview_wait_count=', preview_wait_count )
            sleep( 0.5 )
            preview_wait_count += 1
        if pool.ok():
            if self.save_preview_figures:
                sleep( 0.5 )
                pool.dialog.save_the_figure( self.preview_image_folder, self.agent.analysis_name.get() )
                solver_results, peak_range_infos = pool.get_solver_ret_tuple()
            if self.save_preview_results:
                pool.dialog.save_results( __wait__=False )
                sleep(0.5)
                saver = pool.dialog.saver
                ret = saver.validate(notify=False, post_fix=post_fix)
                if ret:
                    print('preview results save ok.')
                else:
                    print('preview results save failed.')
                saver.cancel()
            if self.save_preview_cdi_figures:
                pool.dialog.auto_cdi_for_all(out_folder=self.preview_cdi_image_folder)
            pool.dialog.ok()
        else:
            put_to_tester_log_queue( 'failed to preview' )

    def do_whole( self ):
        global fromlog_rg_list

        agent           = self.agent
        from_logfile    = self.from_logfile
        analyzer        = agent.analyzer
 
        if self.to_compare_logfolder is None:
            fromlog_rg_list = None
        else:
            # print( 'from_logfile=', from_logfile )
            if os.path.exists( from_logfile ):
                fromlog_rg_list = []
                final_std_diff  = None
                try:
                    fh = open( from_logfile )
                    for line in fh:
                        if final_std_diff is None:
                            p = line.find( final_std_diff_str )
                            if p > 0:
                                p_ = p + len(final_std_diff_str)
                                # print( 'line[p_:]=-1', line[p_:-1] )
                                final_std_diff = line[p_:-1]
                                self.log_final_std_diff_check( final_std_diff )
                                continue

                        if line.find( 'Rg' ) < 0:
                            continue
                        m = success_rg_re.search( line )
                        if m:
                            fromlog_rg_list.append( [ float( m.group(i) ) for i in [2, 3] ] )
                            continue
                        m = failure_rg_re.search( line )
                        if m:
                            fromlog_rg_list.append( None )
                            continue
                    # print( 'fromlog_rg_list=', fromlog_rg_list )
                except:
                    from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                    etb = ExceptionTracebacker()
                    self.log.write( str( etb ) )
                    fromlog_rg_list = None
                # self.log.write( str( fromlog_rg_list ) + '\n' )
            else:
                fromlog_rg_list = None

        write_from_log_queue()

        analyzer.dialog.run_button.invoke( __wait__=False )

        counter = 0
        exception_break = False
        while True:
            counter += 1
            # if analyzer.progress_dialog.is_at_end(): # seems sometimes to fail on Windows 7
            try:
                if analyzer.progress_dialog.button.cget( 'text' ) == "OK":
                    break
            except:
                etb = ExceptionTracebacker()
                write_to_log( str(etb) )
                print( 'Canceled?' )
                exception_break = True
                break

            if not agent.__is_alive__():
                write_from_log_queue()
                write_to_log( 'agent is not alive\n' )
                return False, None

            sleep( 1 )
            print( 'counter=', counter )

        write_from_log_queue()

        if not exception_break:
            analyzer.progress_dialog.button.invoke()
        agent.update()
        sleep( 1 )

    def do_fully_automatic( self ):
        agent = self.agent
        if self.test_pattern == 6:
            agent.analyzer.set_cancel_mode()

        # agent.fully_automatic_cb.invoke() is not appropriate here because it toggles at the next time
        agent.fully_automatic.set(1)
        agent.analysis_button.invoke( __wait__=False )
        sleep(1)
        reply_messagebox('Fully Automatic Run Confirmation', 'Y')
        sleep(1)
        log_confirm = 'Log file removal confirmation'
        while agent.is_busy():
            print('waiting for the fully_automatic control to finish.')
            sleep(1)
            if self.test_pattern == 6:
                print('window_exists(log_confirm)')
                if window_exists(log_confirm):
                    sleep(1)
                    print('reply_messagebox')
                    reply_messagebox(log_confirm, 'N')
                    print('reply_messagebox end')
                print('test_pattern 6 continue')
        return True, None

    def do_mfc(self):
        write_to_log( 'skippig MFC data.\n' )
        while True:
            try:
                self.agent.analyzer.mfc_dialog.cancel()
                break
            except:
                sleep(1)

    def do_finish( self, in_folder ):
        app_logfile     = self.app_logfile
        from_logfile    = self.from_logfile

        write_from_log_queue()

        if self.to_compare_logfolder is not None:
            try:
                diff_count, error_count = self.make_log_diff( app_logfile, from_logfile )
                print( 'diff_count, error_count=', diff_count, error_count  )
                self.diff_count += diff_count
                self.error_count += error_count
            except:
                etb = ExceptionTracebacker()
                print( etb )
            sleep( 1 )

        self.log.write( 'Finished ' + in_folder + '\n' )

    def log_final_std_diff_check( self, final_std_diff ):
        if final_std_diff == self.final_std_diff:
            write_to_log( 'final_std_diff ok\n' )
        else:
            write_to_log( 'final_std_diff is different: %s != %s\n' % ( final_std_diff, self.final_std_diff ) )
            self.mdiff_count += 1

    def get_q_statistics( self ):
        sd = self.app.analyzer.dialog.serial_data
        qvector = sd.qvector
        qmin = np.min(qvector)
        qmax = np.max(qvector)
        qdelta = qvector[1] - qvector[0]
        size = len(qvector)
        imax = np.max( sd.intensity_array[:, :, 1] )
        return [qmin, qmax, qdelta, size, imax]

    def make_log_diff( self, app_logfile, from_logfile ):
        from molass_legacy.KekLib.LogDiff import make_log_diff, time_re_sub
        to_logfile   = app_logfile

        def my_re_sub( line ):
            line = time_re_sub( line )
            line = re.sub( success_rg_re, 'Rg from (ffff.dat) is estimated to be (00.0) with quality (0.00).', line )
            line = re.sub( pytools_ver_rg, 'pytools-x_x_x-develop', line )
            return line

        diff_count = 0
        error_count = 0
        try:
            diff = make_log_diff( from_logfile, to_logfile, re_sub=my_re_sub )
            if len(diff) > 0:
                self.log.write( '==== diff begin ====\n' )
                self.log.write( diff )
                self.log.write( '==== diff end ======\n' )
                diff_count = 1
            else:
                self.log.write( '==== no diff in the log file ====\n' )
        except Exception as exc:
            etb = ExceptionTracebacker()
            self.log.write( 'make_log_diff failed: ' + str(etb) + '\n' )
            error_count = 0

        return diff_count, error_count

    def do_outline_only(self):
        folder, _ = os.path.split(self.report_folder)

        if self.outline_image_folder is None:
            self.outline_image_folder = os.path.join(folder, 'images-outline').replace( '\\', '/' )
            clear_dirs_with_retry([self.outline_image_folder])

        agent = self.agent
        if agent.doing_mfc():
            put_to_tester_log_queue('Skipped MFC data')
            return

        agent.get_it_ready_for_qmm()
        agent.fig_frame.save_the_figure( self.outline_image_folder, agent.analysis_name.get() )
        return

    def do_v2_demo(self):
        folder, _ = os.path.split(self.report_folder)

        if self.qmm_image_folder is None:
            self.qmm_image_folder = os.path.join(folder, 'qmm-images').replace( '\\', '/' )
            clear_dirs_with_retry([self.qmm_image_folder])

        agent = self.agent
        if agent.doing_mfc():
            put_to_tester_log_queue('Skipped MFC data')
            return

        if True:    # add another arg to select between EGH and EMG
            agent.qmm_menu.show_egh_qmm_dialog(__wait__=False)
        else:
            agent.qmm_menu.show_emg_qmm_dialog(__wait__=False)

        # must do this after agent.get_it_ready_for_qmm() in the above call
        if self.outline_image_folder is not None:
            agent.fig_frame.save_the_figure( self.outline_image_folder, agent.analysis_name.get() )

        while agent.is_qmm_busy():
            print("waining for the qmm-controller's getting ready.")
            sleep(1)

        qmm_controller = agent.get_qmm_controller()
        if qmm_controller is None:
            put_to_tester_log_queue('Failed to get qmm_controller')
            assert False

        try:
            while not qmm_controller.dialog_ready():
                print("waining for the qmm-dialo's getting ready.")
                sleep(1)
        except:
            put_to_tester_log_queue('Failed in dialog_ready')
            assert False

        sleep(1)
        qmm_dialog = qmm_controller.dialog
        frame = qmm_dialog.get_current_frame()
        path = frame.save_the_figure(self.qmm_image_folder, agent.analysis_name.get())
        qmm_dialog.cancel()
        agent.clear_test_mode()
        write_from_log_queue()

        file = path + '.png'
        if not os.path.exists(file):
            put_to_tester_log_queue('Failed to save the figure' + file)
            write_from_log_queue()
            agent.quit(immediately=True)
            assert False

    def do_fullopt_initialization(self, mapper_canvas):
        print("start do_fullopt_initialization")
        mapper_canvas.full_opt_button.invoke(__wait__=False)
        sleep(3)

        while True:
            try:
                fullopt_init = mapper_canvas.fullopt_init
                while fullopt_init.is_busy():
                    print('waiting for the fullopt_init to be ready.')
                    sleep(1)

                fullopt_init.save_the_figure(self.fulloptinit_image_folder, self.agent.analysis_name.get())
                break
            except:
                print("May be we can't get fullopt_init")
                sleep(1)
                continue

        if self.test_pattern == 9:
            fullopt_init.user_cancel(ask=False)
        sleep(1)

    def do_fullopt_execution(self, mapper_canvas):
        print("start do_fullopt_execution")
        fullopt_init = mapper_canvas.fullopt_init
        fullopt_init.proceed_btn.invoke(__wait__=False)     # fullopt_init.ok(__wait__=False) makes it freeze
        sleep(1)

        while True:
            try:
                fullopt_dialog = mapper_canvas.fullopt_dialog
                while fullopt_dialog.is_busy():
                    t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print('%s waiting for the fullopt_dialog to be ready.' % t)
                    progress = fullopt_dialog.get_progress()
                    seconds = 60 if progress < 0.98 else 10
                    sleep(seconds)
                break
            except:
                print("May be we can't get fullopt_dialog")
                sleep(1)
                continue

        fullopt_dialog.cancel()
        sleep(1)

    def make_v2_results_dict(self):
        analysis_folder_re = re.compile("(analysis-\d+)")
        in_folder_re = re.compile("in_folder=(\S+)")
        self.results_dict = {}
        for folder in self.v2_analysis_folders:
            in_data_info_txt = os.path.join(folder, r"optimizer\000\in_data_info.txt")
            if os.path.exists(in_data_info_txt):
                with open(in_data_info_txt) as fh:
                    line = fh.read()
                m = in_folder_re.match(line)
                assert m
                in_folder = m.group(1)

                job_folder = os.path.join(folder, r"optimizer\000")
                print(in_folder, job_folder)
                self.results_dict[in_folder] = job_folder

    def do_v2_results(self, in_folder):
        job_folder = self.results_dict.get(in_folder)
        if job_folder is None:
            print("-------------------- No results exist for %s" % in_folder)
            return

        agent   = self.agent
        agent.v2_menu.show_result_trace(__wait__=False)
        sleep(1)
        trace_dialog = agent.v2_menu.get_trace_dialog()

        entry = trace_dialog.optjob_folder_entry
        entry.delete(0, Tk.END)
        entry.insert(0, job_folder)

        callback_txt = os.path.join(job_folder, "callback.txt")
        size = os.path.getsize(callback_txt)
        if size == 0:
            # as with 20160628
            print("It seems like a failed result. Skip.")
            trace_dialog.cancel()
            return

        try:
            trace_dialog.on_entry_optjob_folder()
            trace_dialog.proceed_btn.invoke(__wait__=False)

            i = 0
            while True:
                print([i], "waiting for FullOptDialog's being ready.")
                sleep(1)
                fullopt_dialog = agent.get_fullopt_dialog()
                if fullopt_dialog is not None:
                    break
                i += 1

            print([i], "FullOptDialog seems to be ready.")
            fullopt_dialog.runner.set_work_folder(job_folder)
            fullopt_dialog.save_the_result_figure()
            fullopt_dialog.cancel_button.invoke()   # fullopt_dialog.cancel() gets an error
        except Exception as e:
            print(e)

        sleep(1)

    def do_v2_thru_peak_editor(self):
        agent   = self.agent

        agent.v2_menu.show_peak_editor(__wait__=False)
        i = 0

        while True:
            print([i], "waiting for OptStrategyDialog's construction.")
            sleep(1)
            strategy_dialog = agent.v2_menu.get_strategy_dialog()
            if strategy_dialog is not None:
                break
            i += 1

        strategy_dialog.proceed_btn.invoke()

        while True:
            print([i], "waiting for PeakEditor's construction.")
            sleep(1)
            peak_editor = agent.v2_menu.get_peak_editor()
            if peak_editor is not None:
                break
            i += 1

        while peak_editor.is_busy():
            print([i], "waiting for PeakEditor's being ready.")
            sleep(1)
            i += 1

        print([i], "PeakEditor seems to be ready.")
        if self.save_peakeditor_figures:
            analysis_name = agent.analysis_name.get()
            filename = analysis_name.replace( 'analysis', 'figure' )
            file_path = os.path.join(self.peakeditor_image_folder, filename)
            peak_editor.save_the_figure(file_path)

        if self.test_pattern == 30:
            peak_editor.cancel_btn.invoke()      # peak_editor.user_cancel(ask=False) gets "RuntimeError: main thread is not in main loop"
        else:
            peak_editor.proceed_btn.invoke()
            while True:
                print([i], "waiting for FulloptDialog's construction.")
                sleep(1)
                fullopt_dialog = agent.v2_menu.get_fullopt_dialog()
                if fullopt_dialog is not None:
                    break
                i += 1
            self.wait_for_v2_completion(fullopt_dialog)

    def wait_for_v2_completion(self, fullopt_dialog):
        i = 0
        while fullopt_dialog.is_busy():
            print([i], "waiting for FulloptDialog's being complete.")
            sleep(60)
            i += 1

        fullopt_dialog.cancel_button.invoke()
