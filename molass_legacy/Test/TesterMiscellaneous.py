# coding: utf-8
"""

    SerialAnalyzer.TesterMiscellaneous.py

    Copyright (c) 2018-2020, SAXS Team, KEK-PF

"""

import os
import re
import queue
import threading
import numpy        as np
from time           import sleep
from molass_legacy.KekLib.OurTkinter     import Tk, is_empty_val
from DataUtils      import serial_folder_walk
from TkTester       import TestClient, reply_ok, reply_yes, reply_no, messagebox_shown
from molass_legacy.KekLib.BasicUtils     import clear_dirs_with_retry, mkdirs_with_retry, open_w_safely
from molass_legacy._MOLASS.SerialSettings import clear_settings, get_setting, set_setting
from molass_legacy.KekLib.ExceptionTracebacker       import ExceptionTracebacker
from LightObjects   import AnyObject

def normal_input( agent, self ):
        print( 'in_folder=', self.in_folder )

        agent.in_folder_entry.focus_force()
        agent.in_folder_entry.delete( 0, Tk.END)
        agent.in_folder_entry.insert( 0, self.in_folder )
        agent.on_entry_in_folder()

        while is_empty_val(agent.uv_folder.get()):
            print('waiting for uv_folder update')
            # agent.update()
            sleep(1)

        agent.an_folder_entry.focus_force()
        agent.an_folder_entry.delete( 0, Tk.END)
        agent.an_folder_entry.insert( 0, self.an_folder )
        agent.on_entry_an_folder()
        sleep(1)

class NormalInput:
    def __init__( self, app ):
        self.app    = app

    def test_folders( self, in_folders, an_folder ):
        self.in_folders  = in_folders
        self.an_folder  = an_folder
        self.client = TestClient( self.app, self.test_func )

    def test_func( self, client, agent ):
        self.agent  = agent

        for folder in self.in_folders:
            self.in_folder = folder
            normal_input( agent, self )
            sleep( 5 )

        agent.quit( immediately=True, __wait__=False )
        print( 'agent.quit' )

class VariationBase:
    def do_assert( self, assert_key, value ):
        assert_ = self.assert_dict.get( assert_key )
        if assert_ is None:
            print( 'no assert for ' + assert_key )
        else:
            try:
                assert_( value )
                print( assert_key + ' assert ok' )
            except:
                etb = ExceptionTracebacker()
                print( etb )
                print( assert_key + ' assert failed' )

class MapperCanvasVariation1(VariationBase):
    def __init__( self, app ):
        self.app    = app

    def test_a_folder( self, in_folder, an_folder, assert_dict ):
        self.in_folder  = in_folder
        self.an_folder  = an_folder
        self.assert_dict    = assert_dict
        self.client = TestClient( self.app, self.test_func )

    def test_func( self, client, agent ):
        self.agent  = agent

        normal_input( agent, self )

        # ---- guinier animation
        agent.file_info_table.table.select_row( 141 )
        agent.file_info_table.run_simple_guinier_animation( __wait__=False )
        sleep( 5 )
        agent.file_info_table.anim_dialog.ok_button.invoke()

        # ---- default manipulation
        agent.analysis_button.invoke( __wait__=False )
        sleep( 1 )
        analyzer_dialog = agent.analyzer.dialog
        mapper_canvas  = analyzer_dialog.mapper_canvas
        while True:
            try:
                mapper_canvas.ok_button.invoke()
                break
            except:
                pass
            sleep( 1 )

        analyzer_dialog.cancel_button.invoke()
        sleep( 0.5 )
        reply_ok()

        range_type = agent.get_range_type()
        self.do_assert( 'ranges1-0', range_type )

        ranges1 = agent.analyzer.dialog.get_analysis_ranges_old_style()
        self.do_assert( 'ranges1-1', ranges1 )

        # ---- range adjustment
        agent.analysis_button.invoke( __wait__=False )
        sleep( 1 )
        mapper_canvas  = analyzer_dialog.mapper_canvas
        mapper_canvas.reditor_btn.invoke()
        sleep( 1 )

        for k in range(2):
            print( k, 'buttonup' )
            mapper_canvas.range_list_entry.range_entries[0][0].invoke( 'buttonup' )
            sleep( 0.1 )
        sleep( 1 )

        mapper_canvas.ok_button.invoke()
        sleep( 1 )
        analyzer_dialog.cancel_button.invoke()
        sleep( 0.5 )
        reply_ok()

        range_type = agent.get_range_type()
        self.do_assert( 'ranges2-0', range_type )

        ranges2 = agent.analyzer.dialog.get_analysis_ranges_old_style()
        self.do_assert( 'ranges2-1', ranges2 )

        sleep( 3 )
        agent.quit( immediately=True, __wait__=False )
        print( 'agent.quit' )


class MapperCanvasVariation2(VariationBase):
    def __init__( self, app ):
        self.app    = app

    def test_a_folder( self, in_folder, an_folder, assert_dict ):
        self.in_folder  = in_folder
        self.an_folder  = an_folder
        self.assert_dict    = assert_dict
        self.client = TestClient( self.app, self.test_func )

    def test_func( self, client, agent ):
        self.agent  = agent

        normal_input( agent, self )

        agent.analysis_button.invoke( __wait__=False )
        sleep( 1 )

        analyzer_dialog = agent.analyzer.dialog
        mapper_canvas   = analyzer_dialog.mapper_canvas
        mapper_adjuster = None

        while mapper_adjuster is None:
            try:
                mapper_adjuster  = mapper_canvas.adjuster
            except Exception as exc:
                print( exc )
                pass
            sleep( 1 )

        # ---- mapping helper input
        if False:
            mapper_adjuster.helper_btn.invoke( __wait__=False )
            sleep( 1 )

            helper_dialog = analyzer_dialog.helper_dialog
            helper_dialog.flow_change_cb_vars[0].set( 1 )

            axes = helper_dialog.get_axes()

            event = AnyObject( dblclick=True, inaxes=axes[0], xdata=435.0, ydata=0.08 )
            helper_dialog.a_manipulator.modify_peaks( event )

            event = AnyObject( dblclick=True, inaxes=axes[1], xdata=217.0, ydata=0.044 )
            helper_dialog.x_manipulator.modify_peaks( event )

            sleep( 1 )
            helper_dialog.ok_button.invoke()

        # ---- set adjustment options
        mapper_adjuster.uv_baseline_adjust.set( 1 )
        mapper_adjuster.xray_baseline_opt.set( 1 )
        mapper_adjuster.xray_baseline_adjust.set( 1 )
        sleep( 1 )

        # ---- optimize
        mapper_adjuster.optimize_btn.invoke()

        # ---- show 3D view
        mapper_canvas.show_absorbance_button.invoke( __wait__=False  )
        sleep( 1 )
        mapper_canvas.viewer.ok_button.invoke()

        mapper_canvas.show_scattering_button.invoke( __wait__=False  )
        sleep( 2.5 )

        scatterning_plot = mapper_canvas.scatterning_plot
        scatterning_plot.plot_frames[0].vsa_only_cb.invoke()
        scatterning_plot.plot_frames[0].redraw_btn.invoke()

        sleep( 1 )
        scatterning_plot.ok_button.invoke()

        # ---- range adjustment
        mapper_canvas.reditor_btn.invoke()
        sleep( 1 )

        ranges1 = mapper_canvas.range_list_entry.get_ranges()
        self.do_assert( 'ranges1-1', ranges1 )

        mapper_canvas.range_list_entry.range_entries[1][0].invoke( 'buttonup' )
        mapper_canvas.range_list_entry.range_entries[1][2].invoke( 'buttondown' )

        ranges2 = mapper_canvas.range_list_entry.get_ranges()
        self.do_assert( 'ranges2-1', ranges2 )

        sleep( 1 )
        mapper_canvas.ok_button.invoke()
        sleep( 1 )
        analyzer_dialog.cancel_button.invoke()
        sleep( 0.5 )
        reply_ok()

        range_type = agent.get_range_type()
        self.do_assert( 'ranges2-0', range_type )

        ranges2 = agent.analyzer.dialog.get_analysis_ranges_old_style()
        self.do_assert( 'ranges2-1', ranges2 )

        sleep( 1 )
        agent.quit( immediately=True, __wait__=False )
