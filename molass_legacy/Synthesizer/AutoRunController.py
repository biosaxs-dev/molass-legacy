"""

    ファイル名：   AutoRunController.py

    処理内容：

        自動制御

    Copyright (c) 2020,2024, SAXS Team, KEK-PF
"""
import sys
import time
import logging

from molass_legacy.KekLib.OurTkinter     import Tk
import OurMessageBox  as MessageBox
from OurMock        import MagicMock

import ThreadsConnector
import ActionWindow
import threading

from SynthesizerSettings    import get_setting, get_mask
from Preferences            import get_preference
from ImageSynthesizer       import ImageSynthesizer
from DebugQueue             import debug_queue_get

import gettext
_ = gettext.gettext

class AutoRunController:

    def __init__( self, window, interval, image_info_table, log_file_path=None, on_stop=None ):
        self.window             = window
        self.interval           = interval
        self.image_info_table   = image_info_table
        self.log_file_path      = log_file_path
        self.synthesizer        = ImageSynthesizer()
        self.logger             = logging.getLogger( __name__ )
        self.on_stop            = on_stop

        print( '__init__:', threading.current_thread() )

        title       = _('Auto Run Thread Log')
        text        = _('Controlling syntheses' )
        geometry    = "900x400"
        self.action_window  = ActionWindow.ActionWindow(self.window, title, text, geometry, progressbarlabel=0, on_stop=self.on_stop )

    def start( self ):
        # テストでは self を GuiController.Controller から使用するため、
        # コンストラクタから分離しておく必要がある。
        conn = ThreadsConnector.ThreadsConnector( on_cancel=self.on_cancel )
        conn.runInGui( self.action_window, conn, None, self.control_syntheses, 'control' )

    def control_syntheses( self, connector, progress ):
        self.in_folder  = get_setting( 'in_folder' )
        self.adj_folder = get_setting( 'adj_folder' )
        self.syn_folder = get_setting( 'syn_folder' )
        # GUI から起動される場合、マスクは確定している。
        self.mask       = get_mask()

        print( 'control_syntheses:', threading.current_thread() )
        self.logger.info( 'Started auto-run with data end index %d.' % ( self.image_info_table.current_data_end) )

        while( True ):
            start_time  = time.time()
            progress.set( 0, self.interval )

            self.exec_thread = threading.Thread( None, self.exec_syntheses, 'exec', () )
            self.exec_thread.start()
            last_time   = time.time()
            # if self.exec_thread.is_alive():
            #    print( 'Looks like running.' )

            exec_thread_looks_like_done = False
            for i in range( self.interval ):
                # オーバヘッド分の遅れを調整する。
                sec_ = i + 1 - ( last_time - start_time )
                if sec_ < 0:
                    continue
                elif sec_ <= 1:
                    pass
                else:
                    sec_ = 1

                # print( '[%d] sec_=%g' % ( i, sec_ ) )

                time.sleep( sec_ )
                progress.tick()
                connector.ack()

                if not exec_thread_looks_like_done and not self.exec_thread.is_alive():
                    exec_thread_looks_like_done = True
                    # print( 'Looks like done.' )
                    # TODO: in case failed

                last_time   = time.time()

        self.logger.info(_('Unexpected! Never reach here.'))

    def exec_syntheses( self ):
        try:
            self.logger.info( 'Refreshing infomation.' )

            if debug_queue_get() == __name__ + '.die()': die()  # die(): undefined function

            selected_indices = self.image_info_table.get_not_yet_done_indices()
            num_selected_records = len( selected_indices )
            if num_selected_records > 0:
                self.logger.info( 'Begin sythesizing for %d samples.' % ( num_selected_records ) )

                # self.image_info_table.do_action( 3 )
                self.synthesizer.set_setting_info( 3 )

                exec_array = self.image_info_table.select_data_array( selected_indices )
                self.synthesizer.exec_array = exec_array

                mockconnector   = MagicMock()
                mockprogress    = MagicMock()

                self.synthesizer.exec_syntheses( mockconnector, mockprogress )

                self.image_info_table.update_current( selected_indices )
                self.image_info_table.clear_selection( selected_indices )
            else:
                self.logger.info( 'No new data since last refresh.' )
        except:
            logging.exception( 'Unexpected error in the execution thread:' )

        # self.image_info_table.refresh( log_file_path=self.log_file_path, bottom_view=True, logger=self.logger, autorun=True )
        self.window.refresh(autorun=True)

    def cancel( self ):
        self.action_window.cancel()

    def on_cancel( self ):
        # print( 'exec_cancel:', threading.current_thread() )

        if self.exec_thread.is_alive():
            self.exec_thread.join( 10 )

        self.exec_thread = None
        self.logger.info( 'exec_cancel' )
