"""
    SerialDataLoader.py

    a) There is only one SerialDataLoader object, which belongs to the GuiMain.
    b) It holds a set of serial data which is currently processed.
    c) The current set of data never changes while the user operates on the same input folder.
    d) For such abnormal data as have been inccured by bubbles,
       fixing of them is done only once just after the initial reading.
    e) Flow change recongnition is performed only once, while it can be modified with the helper info.
    f) Baseline correction depends on the set of optimization parameters.
    g) 
            GuiMain -> Analyzer --(1)-> Mapper
                                <------
                                --(2)-> Executor
                                <------
    f) For each set of optimization parameters, 
    f) The current set is copied every time to be handed to the Anlyzer.
    g) While the 

    Copyright (c) 2018-2023, SAXS Team, KEK-PF
"""

import os
import copy
import threading
import logging
import time
from SerialDataUtils import get_uv_filename, load_intensity_files, load_uv_array
from SerialData import REQUIRED_SERIAL_NUM_FILES, SerialData
from molass_legacy._MOLASS.SerialSettings import get_setting

DEBUG_LOG       = True

def get_restrict_ranges():
        u_lower = get_setting( 'uv_restrict_lower' )
        u_upper = get_setting( 'uv_restrict_upper' )
        x_lower = get_setting( 'xray_restrict_lower' )
        x_upper = get_setting( 'xray_restrict_upper' )
        return [ (u_lower, u_upper), (x_lower, x_upper) ]

class SerialDataLoader:
    def __init__(self, dialog=None):
        self.logger         = logging.getLogger( __name__ )
        self.dialog = dialog
        self.in_folder      = None
        self.uv_folder      = None
        self.uv_file        = None
        self.load_thread    = None
        self.reset_input_status()
        self.reset_load_status()

    def reset_input_status( self ):
        self.lvector_       = None
        self.uv_array_      = None
        self.uv_file_       = None
        self.col_header     = None
        self.lvector_xp     = None
        self.uv_array_xp    = None
        self.uv_file_xp     = None
        self.is_loadable    = False
        self.is_busy        = False

    def reset_load_status( self ):
        self.is_loadable    = True
        self.ready          = False
        self.initial_data   = None
        self.current_data   = None
        self.is_restricted  = None
        self.is_busy        = True
        self.mtd_elution    = None

    def reset_current_status( self ):
        self.current_data   = None

    def load_from_folders( self, in_folder, uv_folder=None, uv_file=None ):

        if uv_folder is None:
            uv_folder   = in_folder

        assert uv_file is not None or uv_folder is not None

        if uv_file is None:
            uv_file     = get_uv_filename( uv_folder )

        if (    in_folder == self.in_folder
            and uv_folder == self.uv_folder
            and uv_file == self.uv_file
            and self.ready
            ):
            self.logger.info( 'skipped loading for the same input params.' )
            return

        self.in_folder  = in_folder
        self.uv_folder  = uv_folder
        self.uv_file    = uv_file
        self.reset_load_status()

        self.load_thread = threading.Thread(
                                target=self.load_data,
                                name='LoadDataThread',
                                args=[  in_folder,
                                        uv_folder,
                                        uv_file
                                     ],
                                )
        self.load_thread.start()
        self.start_time = time.time()
        self.logger.info( 'started loading ' + in_folder )

    def load_data( self, in_folder, uv_folder, uv_file ):
        self.logger.info( 'loading by SerialDataLoader' )
        self.load_xray_data( in_folder )
        self.load_uv_data( uv_folder, uv_file )
        self.ready = True
        self.is_busy = False    # added for the Tester to be robust

    def wait_until_ready( self ):
        assert self.load_thread is not None

        while not self.ready:
            if self.dialog is None:
                time.sleep(0.1)
            else:
                self.dialog.update()

        self.load_thread.join()
        load_time = time.time() - self.start_time
        self.logger.info( 'finished loading. xray_array.shape=%s. it took %.3g seconds for loading.', str(self.xray_array.shape), load_time )
        self.is_busy = False

    def load_xray_data_only( self, in_folder ):
        self.in_folder  = in_folder
        self.uv_folder  = None
        self.uv_file    = None
        self.reset_load_status()

        self.load_thread = threading.Thread(
                                target=self.load_xray_data_only_impl,
                                name='LoadDataThread',
                                args=[  in_folder,
                                     ],
                                )
        self.load_thread.start()
        self.start_time = time.time()
        self.logger.info('started loading (xray only) ' + in_folder )

    def load_xray_data_only_impl( self, in_folder ):
        self.load_xray_data( in_folder )
        if get_setting('use_xray_conc'):
            if self.has_enough_num_files():
                self.make_xray_proportinal_uv_data()
                self.logger.info('made xray proportinal uv data')
        elif get_setting('use_mtd_conc'):
            self.make_mtd_simulated_uv_data(len(self.datafiles))
            self.logger.info('made mtd simulated uv data')
        else:
            self.logger.error('unexpected state: no UV data')
        self.ready  = True
        self.is_busy = False    # added for the Tester to be robust

    def load_xray_data( self, in_folder ):
        if not os.path.exists( in_folder ):
            raise Exception( in_folder + ' does not exist!'  )

        data_array, datafiles = load_intensity_files( in_folder, logger=self.logger )
        self.datafiles  = datafiles
        self.xray_array = data_array
        self.excluded_xray_array = None

    def load_uv_data( self, uv_folder, uv_file ):
        data_array, lvector, uv_file_, col_header = load_uv_array( uv_folder, uv_file, column_header=True )
        self.uv_array_  = data_array
        self.lvector_   = lvector
        self.uv_file_   = uv_file_
        self.col_header = col_header

    def get_current_object( self ):
        if self.current_data is None:
            self.wait_until_ready()
            self.get_current_object_impl()

        return self.current_data 

    def get_current_object_impl( self ):
        xray_array = self.xray_array
        lvector, uv_array, uv_file, col_header = self.get_active_uv_data()
        data_info   = [ self.datafiles, xray_array, uv_array, lvector, col_header, self.mtd_elution ]
        self.current_data = SerialData( self.uv_folder, self.in_folder, conc_file=uv_file, data_info=data_info )

    def get_data_object( self ):
        if self.current_data is not None:
            if not self.is_restricted:
                return self.current_data
            else:
                pass
        else:
            pass

        self.get_current_object()
        self.initial_data = self.current_data

        self.is_restricted      = False
        self.restrict_ranges    = None

        return self.current_data

    def get_init_object( self ):
        if self.initial_data is None:
            self.get_data_object()
        return self.initial_data

    def get_active_uv_data( self ):
        if get_setting( 'use_xray_conc' ) == 0 and get_setting( 'use_mtd_conc' ) == 0:
            lvector     = self.lvector_
            uv_array    = self.uv_array_
            uv_file     = self.uv_file_
            col_header  = self.col_header
        else:
            lvector     = self.lvector_xp
            uv_array    = self.uv_array_xp
            uv_file     = self.uv_file_xp
            col_header  = None

        # TODO: verify this setting
        self.lvector    = lvector
        self.uv_array   = uv_array
        self.uv_file    = uv_file
        self.col_header = col_header

        return lvector, uv_array, uv_file, col_header

    def get_restricted_uv_data( self, lower, upper ):
        upper_ = None if upper is None else upper+1
        lower_ = None if lower is None else lower
        slice_  = slice( lower_, upper_ )
        lvector, uv_array, uv_file, col_header = self.get_active_uv_data()
        return uv_array[:, slice_], lvector, col_header[slice_]

    def get_restricted_xray_data( self, lower, upper ):
        print( 'get_restricted_xray_data' )
        upper_ = None if upper is None else upper+1
        lower_ = None if lower is None else lower
        slice_  = slice( lower_, upper_ )

        xray_array = self.xray_array
        if self.excluded_xray_array is not None:
            # print( 'getting restricted data from excluded_xray_array' )
            xray_array = self.excluded_xray_array

        return self.datafiles[slice_], xray_array[slice_, :, :]

    def load_xray_data_in_another_thread( self, in_folder ):
        if not os.path.exists( in_folder ):
            raise Exception( in_folder + ' does not exist!'  )

        self.load_xray_thread = threading.Thread(
                                target=self.load_xray_data,
                                name='LoadDataThread',
                                args=[  in_folder,
                                     ],
                                )
        self.load_xray_thread.start()
        self.start_time = time.time()
        print( 'started xray loading.' )

    def wait_for_xray_loading( self ):
        assert self.load_xray_thread is not None

        self.load_xray_thread.join()
        load_time = time.time() - self.start_time
        print( 'finished xray loading. it took %.3g seconds for loading.', load_time)

    def memorize_exclusion( self, serial_data ):
        # print( 'memorize_exclusion' )
        self.excluded_xray_array = serial_data.intensity_array

    def has_enough_num_files( self ):
        return self.xray_array.shape[0] >= REQUIRED_SERIAL_NUM_FILES

    def make_xray_proportinal_uv_data(self, ivector=None):
        from molass_legacy.UV.XrayProportional import make_xray_proportinal_uv_data_impl

        self.logger.info('making xray_proportinal_uv_data')

        lvector_xp, uv_array_xp, uv_file_xp = make_xray_proportinal_uv_data_impl(self.xray_array, ivector=ivector)

        self.logger.info('made uv_array_xp shape=%s from xray_array shape=%s', str(uv_array_xp.shape), str(self.xray_array.shape))

        self.lvector_xp = lvector_xp
        self.uv_array_xp = uv_array_xp
        self.uv_file_xp = uv_file_xp

    def make_mtd_simulated_uv_data( self, size ):
        self.logger.info('making mtd_simulated_uv_data')
        from Microfluidics.MicrofluidicElution import MicrofluidicElution
        mtd_file_path = get_setting( 'mtd_file_path' )
        self.mtd_elution = MicrofluidicElution(mtd_file_path, size)
        x, y = self.mtd_elution.get_elution_data(size)
        self.make_xray_proportinal_uv_data(ivector=y)
