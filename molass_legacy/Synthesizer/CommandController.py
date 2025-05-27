# -*- coding: utf-8 -*-
"""

    ファイル名：   CommandController.py

    処理内容：

        コマンドモードの実行制御

"""
from __future__ import division, print_function, unicode_literals
import sys
import os
sys.path.append( os.path.dirname( os.path.abspath( __file__ ) ) + '/ExecutionWindow/bin' )

import shutil
import logging
from OurException       import OurException
from SynthesizerSettings    import set_setting, set_mask
from molass_legacy.KekLib.BasicUtils         import mkdirs_with_retry, exe_name
from PilatusUtils       import get_data_info
from ImageSynthesizer   import ImageSynthesizer
from Preferences        import set_preference
from OurMock            import MagicMock
import ChangeableLogger

class Controller:

    def __init__( self, opts ):
        self.opts   = opts

        if opts.in_folder == None:
            raise OurException( '-i IN_FOLDER argument is required!' )

        if not os.path.exists( opts.in_folder ):
            raise OurException( '%s does not exist!' % ( opts.in_folder ) )

        if not os.path.isdir( opts.in_folder ):
            raise OurException( '%s is not a folder!' % ( opts.in_folder ) )

        self.new_out_folder_created = False
        if opts.out_folder == None:
            new_file_ok = False
            out_folder = opts.in_folder + '/Synthesized'

            if not opts.autonum_folders:
                if os.path.exists( out_folder ):
                    shutil.rmtree( out_folder )

            for i in range( 1, 100 ):
                if os.path.exists( out_folder ):
                    out_folder = opts.in_folder + '/Synthesized(%d)' % ( i )
                else:
                    new_file_ok = True
                    break
            assert( new_file_ok )
            mkdirs_with_retry( out_folder )
            opts.out_folder = out_folder
            self.new_out_folder_created = True

        if not os.path.exists( opts.out_folder ):
            raise OurException( '%s does not exist!' % ( opts.out_folder ) )

        if not os.path.isdir( opts.out_folder ):
            raise OurException( '%s is not a folder!' % ( opts.out_folder ) )

        set_setting( 'in_folder', opts.in_folder )
        set_setting( 'adj_folder', None )
        set_setting( 'syn_folder', opts.out_folder )

    def execute( self ):
        counter_id      = 'None'
        pilatus_counter = None
        in_folder       = self.opts.in_folder
        adj_folder      = self.opts.adj_folder
        out_folder      = self.opts.out_folder

        self.logfile_path   = '%s/%s.log' % ( out_folder, exe_name() )
        app_logger  = ChangeableLogger.Logger( self.logfile_path )
        logger      = logging.getLogger( __name__ )

        if self.new_out_folder_created:
            logger.info( "New folder '%s' has been created." % ( out_folder ) )

        try:
            log_file, mask_file, data_array, pilatus_counter = get_data_info( in_folder, adj_folder, out_folder, pilatus_counter, counter_id )

            set_mask( os.path.join( in_folder, mask_file) )

            synthesizer = ImageSynthesizer()
            synthesizer.set_setting_info( 3 )
            synthesizer.exec_array = data_array
            mockconnector   = MagicMock()
            mockprogress    = MagicMock()
            synthesizer.exec_syntheses( mockconnector, mockprogress )
        except:
            logging.exception( 'Unexpected error' )
