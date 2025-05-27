# -*- coding: utf-8 -*-
"""

    ファイル名：   ImageSynthesizer.py

    adapted from ExecutionWindow/bin/MainWindowApp.py

    Copyright (c) 2015-2020, SAXS Team, KEK-PF
"""
import re
from molass_legacy.KekLib.OurTkinter             import Tk
import OurMessageBox            as MessageBox

import logging
import time
import ExecutionWindow
import ViewLog
import ThreadsConnector
import ActionWindow

import numpy as np
from PIL                    import Image
from PilatusImageViewer     import PilatusImageViewer
from PilatusImage           import PilatusImage, pixel_rounded_shift
from SAnglerMask            import SAnglerMask
from Preferences            import get_preference
from SynthesizerSettings    import get_setting, get_mask
from Development            import get_devel_info
from molass_legacy.AutorgKek.AppVersion             import synthesizer_version_string

import gettext
_ = gettext.gettext

UNEXPECTED_ERROR_TEST = False

class ImageSynthesizer():
    def __init__(self, window=None ):
        self.window         = window
        self.logger         = logging.getLogger( __name__ )

    def set_setting_info( self, action ):
        self.mask_array     = get_mask().mask_array
        self.action         = action
        self.syn_method     = get_preference( 'syn_method' )
        self.min_ratio      = get_devel_info( 'min_ratio'  )
        self.syn_flags      = get_preference( 'syn_flags'  )
        self.postfix_adj    = get_devel_info( 'postfix_adj' )
        self.postfix_syn    = get_preference( 'postfix_syn' )

    def execute( self, action, exec_array ):

        self.set_setting_info( action )

        num_selected_rows = len( exec_array )
        if num_selected_rows == 0:
            return

        if action < 3:
            if num_selected_rows > 1:
                MessageBox.showinfo( 'First Selection Notice', 'Only the first selection will be shown.' )
            self.show_images( exec_array[0] )
            return

        # what follows is the case when action == 3
        num_selected_samples = len( exec_array )

        s_ = ''
        if num_selected_samples > 1:
            s_ = 's'

        ok = MessageBox.askokcancel( 
                        'Confirmation',
                        'You are making sythesized images for %d sample%s. Ok?' % ( num_selected_samples, s_ )
                        )
        if not ok:
            return

        self.exec_array = exec_array

        self.logger.info(_('Preparing and starting calculations with %s'), synthesizer_version_string())

        title       = _('Execution Thread Log')
        text        = _('Making %d synthesized images from %s to %s' % ( num_selected_rows, exec_array[0][0], exec_array[num_selected_rows - 1][0] ) )
        geometry    = "1200x500"
        wnd     = ActionWindow.ActionWindow(self.window, title, text, geometry )

        conn = ThreadsConnector.ThreadsConnector()
        conn.runInGui(wnd, conn, None, self.exec_syntheses, 'exec')

    def show_images( self, exec_rec, last_exec_params=[] ):
        # print( 'exec_rec=', exec_rec )
        """
            [0]     Sample id           MAG2wk7210
            [1]     i                   1
            [2]     Original Image      MAG2wk7210_0_00000.tif
            [3]     Absolute Position   ['12.87201', '180.49000'],
            [4]     Shifted Image       MAG2wk7210_1_00000.tif
            [5]     Relative Position   ['5', '5']
            [6]     Intensity Ratio     0.99686741082288877
            [7]     Adjusted Image      MAG2wk7210_1_adj.tif
            [8]     Synthesized Image   MAG2wk7210_1_syn.tif
            [9]     fkey                MAG2wk7210_1
        """

        sample_id   = exec_rec[0]
        in_folder   = get_setting( 'in_folder' )
        adj_folder  = get_setting( 'adj_folder' )
        syn_folder  = get_setting( 'syn_folder' )

        exec_rec_array = exec_rec[1]
        # print( 'exec_rec_array=', exec_rec_array )

        exec_params   = []

        if self.action == 1:
            mask_array = None
        else:
            mask_array = self.mask_array

        o_file  = exec_rec_array[0][0]
        if o_file:
            o_path      = '%s/%s' % ( in_folder, o_file )
            o_pim       = PilatusImage( o_path, mask_array )
            oim_array   = o_pim.image_array()
            exec_params.append( [ o_file, oim_array ] )

        i = 0
        for sub_rec in exec_rec_array[1:]:
            i += 1

            s_delta = []
            i_delta = []
            for d in sub_rec[1]:
                ishift_ = int(float(d) * 1000)
                s_delta.append( ishift_ )
                i_delta.append( pixel_rounded_shift( ishift_ ) )

            if self.action == 1:
                a_file      = sub_rec[0]
                if a_file != None:
                    a_path      = '%s/%s' % ( in_folder, a_file )
                    a_pim       = PilatusImage( a_path )
                    a_im_array  = a_pim.image_array()
                else:
                    continue
            else:
                a_file      = '%s_%d%s.tif' % ( sample_id, i, self.postfix_adj )
                s_file      = sub_rec[0]
                if s_file != None:
                    s_path  = '%s/%s' % ( in_folder, s_file )
                    a_pim   = PilatusImage( s_path, self.mask_array, s_delta[0], s_delta[1] )
                    a_im_array  = a_pim.image_array()
                else:
                    a_im_array  = None

            if type(a_im_array) == np.ndarray:
                exec_params.append( [ a_file, a_im_array, i_delta[0], i_delta[1] ] )


            if i == len( exec_rec_array ) - 1:
                z_file  = sub_rec[4]
                if z_file:
                    z_path  = '%s/%s' % ( syn_folder, z_file )
                    z_pim   = PilatusImage( z_path )
                    z_im_array  = z_pim.image_array()
                    exec_params.append( [ z_file, z_im_array ] )

        self.viewer = PilatusImageViewer( self.action, sample_id, exec_params )

    def exec_syntheses( self, connector, progress ):

        num_rows_to_execute = 0
        for exec_rec in self.exec_array:
            num_rows_to_execute += len( exec_rec[1] ) -1
        # print( 'num_rows_to_execute=', num_rows_to_execute )

        progress.set(0, num_rows_to_execute)

        counter_id  = get_preference( 'detection_counter' )

        for exec_rec in self.exec_array:
            """
                exec_rec example:
                    case of 3 chenges:

                    [ 'AgBh008',
                        [
                            ['AgBh008_0_00000.tif', ['0.70000', '-0.35000'], 1.0, None, None],
                            ['AgBh008_1_00000.tif', ['5', '3'],              1.0, None, None],
                            ['AgBh008_2_00000.tif', ['-5', '-3'],            1.0, None, None]
                        ],
                        3   # num_changes
                    ]

                    case of 2 chenges:
                    [ 'SiGe64201', 
                        [
                            ['SiGe64201_0_00000.tif', ['50.28120', '-0.55000'], 1.0, None, None],
                            ['SiGe64201_1_00000.tif', ['-5', '3'], 1.0, None, None]
                        ],
                        3   # num_changes
                    ]

            """

            # print( 'exec_rec=', exec_rec )
            sample_id       = exec_rec[0]

            try:
                if UNEXPECTED_ERROR_TEST:
                    if sample_id == 'test003_2':
                        unexpected_error()

                exec_rec_array  = exec_rec[1]
                num_changes = exec_rec[2]

                org_name    = '%s_0'  % ( sample_id )
                org_file    = re.sub( r'\.\w+$', '', exec_rec_array[0][0] )
                base_rec    = exec_rec_array[0]

                last_fkey = None
                last_result = []
                last_syn_file = None
                i = 0
                for sub_rec in exec_rec_array[1:]:
                    i += 1
                    exec_seq_no = i
                    fkey        = '%s_%d' % ( sample_id, i )

                    if i > 1:
                        org_name    = last_fkey
                        org_file    = last_syn_file
                        base_rec    = last_result

                    isfinal = i == len( exec_rec_array ) - 1

                    sft_name    = fkey
                    sft_file    = re.sub( r'\.\w+$', '', sub_rec[0] )
                    if isfinal:
                        syn_file    = '%s%s'  % ( sample_id, self.postfix_syn )
                    else:
                        syn_file    = '%s%s'  % ( fkey, self.postfix_syn )

                    i_ratio     = sub_rec[2]

                    # if exec_seq_no < 3 and self.syn_flags[ exec_seq_no ]:
                    if (exec_seq_no >= 3 or self.syn_flags[ exec_seq_no ]) and len(exec_rec_array) == num_changes:
                        counter_id = get_preference( 'detection_counter' )
                        if counter_id == 'None':
                            with_phrase = ''
                        else:
                            with_phrase = ' with %s-ratio %.5f' % ( counter_id, i_ratio )
                        self.logger.info (_('Synthesizing %-20s and %-20s into %-20s%s'), org_file, sft_file, syn_file, with_phrase )

                        last_result = self.exec_single_synthesis( sample_id, base_rec, fkey, sub_rec, isfinal )
                        last_syn_file = syn_file
                    else:
                        self.logger.info (_('Skipping     %-20s and %-20s'), org_file, sft_file )

                    progress.tick()
                    connector.ack() # can be ommitted in this program

            except:
                logging.exception( "Unexpected error while processing '%s' in exec_syntheses" % sample_id )

        self.logger.info(_('Done!'))

    def exec_single_synthesis( self, sample_id, base_rec, fkey, sub_rec, isfinal ):

        in_folder   = get_setting( 'in_folder' )
        adj_folder  = get_setting( 'adj_folder' )
        syn_folder  = get_setting( 'syn_folder' )
        adj_output  = get_devel_info( 'adj_output' )

        o_file = base_rec[0]
        if type(o_file) != np.ndarray:
            o_path  = '%s/%s' % ( in_folder, o_file )
            o_pim   = PilatusImage( o_path, self.mask_array )
            o_ext   = o_pim.image.ext
            self.previous_im = o_pim

            if self.syn_method == 'average':
                # ここで画像のサイズが判明しているので、
                # 平均化に必要な行列を初期化する。
                oim_array = o_pim.image_array()

                # 強度集積用の行列と、有効値カウント用の行列を初期化する。
                self.valid_value_sum_array      = np.zeros( oim_array.shape, dtype='f8' )
                self.valid_pixel_counter_array  = np.zeros( oim_array.shape, dtype='i4' )

                # 正画像の有効値の pixel についてのみ、それぞれ加算する。
                valid_cond = oim_array >= 0
                self.valid_pixel_counter_array[ valid_cond ] += 1
                self.valid_value_sum_array[ valid_cond ] += oim_array[ valid_cond ]
        else:
            o_pim   = PilatusImage( o_file, original_image=self.previous_im )
            o_ext   = self.previous_im.image.ext

        s_file  = sub_rec[0]
        s_path  = '%s/%s' % ( in_folder, s_file )
        s_delta = []
        for d in sub_rec[1]:
            s_delta.append( int(float(d) * 1000) )

        # 副画像を調整する。
        i_ratio = sub_rec[2]
        a_ratio = self.min_ratio
        a_pim   = PilatusImage( s_path, self.mask_array, s_delta[0], s_delta[1], i_ratio, a_ratio )

        if adj_output=='YES' and adj_folder:
            a_file  = '%s%s.%s' % ( fkey, self.postfix_adj, o_ext )
            w_path = '%s/%s' % ( adj_folder, a_file )
            # w_image = Image.fromarray( a_pim.image_array() )
            # w_image.save( w_path, tiffinfo = a_pim.tiffinfo )
            a_pim.save(w_path  )

        if self.syn_method == 'cover':
            # result_im_array = o_pim.make_covered_array( a_pim )
            result_im_array = o_pim.fast_make_covered_array( a_pim )
        else:
            # 調整済副画像の有効値の pixel についてのみ、それぞれ加算する。
            a_im_array  = a_pim.image_array()
            valid_cond  = a_im_array >= 0
            self.valid_pixel_counter_array[ valid_cond ] += 1
            self.valid_value_sum_array[ valid_cond ] += a_im_array[ valid_cond ]
            # result_im_array = o_pim.make_average_array( a_pim )
            result_im_array = o_pim.fast_make_average_array( self.valid_pixel_counter_array, self.valid_value_sum_array )

        if isfinal:
            fkey_ = sample_id
        else:
            fkey_ = fkey

        if isfinal or get_devel_info( 'intermediate_results' ) == 'YES':
            z_file  = '%s%s.%s' % ( fkey_, self.postfix_syn, o_ext )
            w_path  = '%s/%s' % ( syn_folder, z_file )
            # w_image = Image.fromarray( result_im_array )
            # w_image.save( w_path, tiffinfo = a_pim.tiffinfo )
            o_pim.image.set_data( result_im_array, force=True )
            o_pim.save( w_path )

        result = [ result_im_array ]

        return result
