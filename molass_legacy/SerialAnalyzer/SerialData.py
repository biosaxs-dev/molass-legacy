"""
    SerialData.py

    連続測定データ

    Copyright (c) 2016-2025, SAXS Team, KEK-PF
"""

import os
import copy
from bisect import bisect_right
import numpy    as np
import re
import threading
import logging
import inspect
from molass_legacy.SerialAnalyzer.ElutionCurve import ElutionCurve
from molass_legacy._MOLASS.SerialSettings import get_setting
from DevSettings import get_dev_setting
from molass_legacy.Elution.CurveUtils import get_xray_elution_vector
from molass_legacy.Trimming import FlangeLimit
from molass_legacy.UV.Absorbance import Absorbance
from molass_legacy.Baseline.ScatteringBaseline import ScatteringBaseline
from ScatteringBaseUtil import apply_baseline_correction_impl
from InputSmootherAveraging import IntensitySmootherAveraging, ConcentrationSmootherAveraging
from molass_legacy.Test.TesterLogger import put_to_tester_log_queue, write_to_tester_log
from SerialDataUtils import load_intensity_files, load_uv_array
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker

DEBUG = False
DEBUG_PLOT = False

REQUIRED_MIN_QUALITY            = 0.5
NUM_POINTS_DISCOMFORT_DETECT    = 10
REQUIRED_SERIAL_NUM_FILES       = 20

class SerialData:
    def __init__( self, conc_folder, intensity_folder,
                    conc_file=None,
                    data_info=None,
                    orig_info=None,
                    copy_info=None,
                    pre_recog=None,
                    do_mapping=True,
                    debug=False,
                    ):
        self.debug  = debug
        self.do_mapping = do_mapping
        self.q_limit_index  = None
        self.usable_slice   = None          # to be removed
        self.num_curves_averaged = False
        self.intensity_reduced  = False
        self.absorbance = None
        self.ready      = False
        self.initial_std_diff   = None
        self.current_std_diff   = None
        self.has_adjusted_array = False
        self.baseline_corrected = False
        self.corrected_base = None
        self.scattering_baseline_discomfort  = None
        self.mc_vector = None
        self.conc_factor = None
        self.xray_transformed_conc = None
        self.logger         = logging.getLogger( __name__ )
        self.data_info      = data_info
        self.orig_info      = orig_info
        self.copy_info      = copy_info
        self.cd_info = None
        self.set_offsets()
        # self.logger.info('copy_info' + str(copy_info))
        self.logger.info('copy_info from %s', 'None' if copy_info is None else copy_info[0].get_id_info() )
        self.top_x_pair     = self.get_original_peak_top_x_pair()
        self.pre_recog      = pre_recog
        self.helper_info    = None
        self.has_excluded_xray_elutions = False
        self.excluded_set   = set()     # better control info
        self.elutin_slice   = None
        self.cd_slice       = None
        self.mtd_elution    = None

        if data_info is None:
            assert False
            self.read_thread = threading.Thread(
                                    target=self.read,
                                    name='ReadThread',
                                    args=[  conc_folder,
                                            intensity_folder,
                                            conc_file,
                                         ],
                                    )
            self.read_thread.start()
        else:
            self.set_data( data_info, conc_file )

    def __del__(self):
        print("SerialData.__del__ ok")

    def set_offsets(self):
        if self.copy_info is None:
            self.uv_i0 = 0
            self.uv_j0 = 0
            self.xr_i0 = 0
            self.xr_j0 = 0
        else:
            uv_restrict = self.copy_info[1]
            if uv_restrict is None:
                self.uv_i0 = 0
                self.uv_j0 = 0
            else:
                elution, wavelength = uv_restrict
                self.uv_i0 = 0 if wavelength is None else wavelength.start
                self.uv_j0 = 0 if elution is None else elution.start

            xray_restrict = self.copy_info[2]
            if xray_restrict is None:
                self.xr_i0 = 0
                self.xr_j0 = 0
            else:
                elution, angle = xray_restrict
                self.xr_i0 = 0 if angle is None else angle.start
                self.xr_j0 = 0 if elution is None else elution.start

    def set_offsets_from_md(self, md):
        uv = md.uv
        self.uv_i0 = uv.i_slice.start
        self.uv_j0 = uv.j_slice.start

        xr = md.xr
        self.xr_i0 = xr.i_slice.start
        self.xr_j0 = xr.j_slice.start

    def set_prerecog_proxy(self, md):
        from PrerecogProxy import PrerecogProxy
        self.pre_recog = PrerecogProxy(md)

    def read( self, conc_folder, intensity_folder, conc_file ):
        # print( 'reading: ', intensity_folder,  conc_file)
        self.logger.info( 'loading by SerialData' )
        self.load_conc_array( conc_folder, conc_file=conc_file )
        self.load_intensity_array(intensity_folder)
        # self.make_adjusted_array() should be call on button-press either by "Figure" or "Run"

    def wait_until_ready( self ):
        if self.data_info is None:
            assert False
            self.read_thread.join()
            if self.absorbance is None:
                error_msg = 'absorbance is not ready'
                write_to_tester_log( error_msg )
                raise RuntimeError( error_msg )
        else:
            pass
        self.ready  = True

    def load_conc_array( self, conc_folder, conc_file=None ):
        data_array, lvector, conc_file = load_uv_array( conc_folder, conc_file )
        self.conc_array = data_array
        self.lvector    = lvector
        self.conc_file  = conc_file

    def load_intensity_array(self, in_folder):
        if not os.path.exists( in_folder ):
            raise Exception( in_folder + ' does not exist!'  )

        data_array, datafiles = load_intensity_files( in_folder, self.logger )

        self.num_datafiles = len( datafiles )
        self.datafiles = datafiles
        self.set_intensity_array( data_array )

    def set_data( self, data_info, conc_file ):
        datafiles, xray_array, uv_array, lvector, col_header, mtd_elution = data_info

        # uv-data
        self.conc_array = uv_array
        self.lvector    = lvector
        self.conc_file  = conc_file
        self.col_header = col_header
        self.mtd_elution = mtd_elution

        # xray-data
        self.num_datafiles = len( datafiles )
        self.datafiles  = datafiles
        data_num_re = re.compile( r'_(\d+)' )
        _, first_file = os.path.split( self.datafiles[0] )
        m = data_num_re.search( first_file )
        if m:
            self.start_file_no  = int( m.group(1) )
        else:
            self.start_file_no  = 0
        self.set_intensity_array( xray_array )

        orig_top_x = None if self.top_x_pair else self.top_x_pair[0]
        self.absorbance = Absorbance( lvector, uv_array, self.xray_curve, col_header=col_header, orig_top_x=orig_top_x )

        self.ready  = True

    def get_data_shapes( self ):
        return self.intensity_array.shape, self.conc_array.shape

    def _get_analysis_copy_impl(self, given_pre_recog, return_also_new_prerecog=False, debug=False):
        datafiles   = self.datafiles
        xray_array  = self.intensity_array
        uv_array    = self.conc_array
        lvector     = self.lvector
        col_header  = self.col_header

        xr_restrict_list = get_setting('xr_restrict_list')
        if xr_restrict_list is None:
            # usable_slice = self.get_usable_slice()
            # xray_array_ = xray_array[:,usable_slice,:]
            xray_array_ = xray_array
            datafiles_ = datafiles
        else:
            # print('xr_restrict_list=', xr_restrict_list)
            elution_restrict = xr_restrict_list[0]
            if elution_restrict is None:
                eslice = slice(None, None)
            else:
                eslice = slice(elution_restrict.start, elution_restrict.stop)
            angle_restrict = xr_restrict_list[1]
            if angle_restrict is None:
                sslice = slice(None, None)
            else:
                sslice = slice(angle_restrict.start, angle_restrict.stop)
            xray_array_ = xray_array[eslice,sslice,:]
            datafiles_ = datafiles[eslice]
            self.logger.info('xray_array has been restricted from %s to %s' % (str(xray_array.shape), str(xray_array_.shape)))

        uv_restrict_list = get_setting('uv_restrict_list')
        col_header_ = col_header
        if uv_restrict_list is None:
            uv_array_   = uv_array
            lvector_    = lvector
        else:
            elution_restrict = uv_restrict_list[0]
            if elution_restrict is None:
                eslice = slice(None, None)
            else:
                eslice = slice(elution_restrict.start, elution_restrict.stop)
            wl_restrict = uv_restrict_list[1]
            if wl_restrict is None:
                wslice = slice(None, None)
            else:
                wslice = slice(wl_restrict.start, wl_restrict.stop+1)
            lvector_ = lvector[wslice]
            uv_array_ = uv_array[wslice, eslice]
            if col_header is not None:
                col_header_ = col_header[eslice]
            self.logger.info('uv_array has been restricted from %s to %s' % (str(uv_array.shape), str(uv_array_.shape)))

        # data_info = copy.deepcopy( [ datafiles_, xray_array_, uv_array_, lvector, col_header_, self.mtd_elution ] )
        data_info = copy.deepcopy( [ datafiles_, xray_array_, uv_array_, lvector_, col_header_] )
        data_info += [self.mtd_elution]
        analysis_copy = SerialData( None, None,
                            conc_file=self.conc_file,
                            data_info=data_info,
                            pre_recog=given_pre_recog,
                            copy_info=[self, uv_restrict_list, xr_restrict_list],
                            )
        self.logger.info("serialdata analysis copy has been made with trimming info UV %s and Xray %s", str(uv_restrict_list), str(xr_restrict_list))
        self.logger.info( 'serialdata analysis copy %s made with xray_array.shape=%s from %s', analysis_copy.get_id_info(), str(analysis_copy.intensity_array.shape), self.get_id_info() )

        if return_also_new_prerecog:
            from importlib import reload
            import molass.DataUtils.ForwardCompat
            reload(molass.DataUtils.ForwardCompat)
            from molass.DataUtils.ForwardCompat import convert_to_trimmed_prerecog
            new_pre_recog = convert_to_trimmed_prerecog(given_pre_recog, uv_restrict_list, xr_restrict_list, debug=debug)
            return analysis_copy, new_pre_recog
        else:
            return analysis_copy

    def get_original_peak_top_x_pair(self):
        if self.copy_info is None:
            return [None, None]

        orig_sd, uv_restrict_list, xr_restrict_list = self.copy_info

        ret_top_x_list = []
        a_curve = orig_sd.absorbance.a_curve
        x_curve = orig_sd.xray_curve

        for curve, list_ in zip([a_curve, x_curve], [uv_restrict_list, xr_restrict_list]):

            peak_top_x = curve.peak_top_x

            if list_ is None:
                ret_top_x = peak_top_x
            else:
                elution_restrict = list_[0]
                if elution_restrict is None:
                    delta_x = 0
                else:
                    delta_x = elution_restrict.start
                ret_top_x = peak_top_x - delta_x

            ret_top_x_list.append(ret_top_x)

        return ret_top_x_list

    def get_copy( self, pre_recog=None ):
        datafiles   = self.datafiles
        xray_array  = self.intensity_array
        uv_array    = self.conc_array
        lvector     = self.lvector
        col_header  = self.col_header
        # data_info = copy.deepcopy( [ datafiles, xray_array, uv_array, lvector, col_header, self.mtd_elution ] )
        data_info = copy.deepcopy( [ datafiles, xray_array, uv_array, lvector, col_header])
        data_info += [self.mtd_elution]
        sd_ = SerialData( None, None, conc_file=self.conc_file, data_info=data_info, pre_recog=pre_recog, copy_info=self.copy_info )
        sd_.baseline_corrected = self.baseline_corrected
        sd_.corrected_base = self.corrected_base
        sd_.cd_info = self.cd_info
        sd_.excluded_set = self.excluded_set
        return sd_

    def get_exec_copy( self, mapper, conc_factor=None ):
        if conc_factor is None:
            conc_factor = self.conc_factor
        # note that conc_factor must be specified for the first copy
        assert conc_factor is not None

        exec_copy = self.get_copy(pre_recog=self.pre_recog)
        exec_copy.set_mc_vector( mapper, conc_factor )

        self.logger.info( 'serialdata exec copy %s made with xray_array.shape=%s from %s', exec_copy.get_id_info(), str(exec_copy.intensity_array.shape), self.get_id_info() )
        return exec_copy

    def get_exec_copy_inside_of_flowchanges( self, mapper, conc_factor ):
        start, stop = mapper.flow_changes
        if start is None:
            start = 0
        if stop is None:
            stop  = start + self.num_datafiles

        elutin_slice = slice(start, stop)
        usable_slice = self.get_usable_slice()

        datafiles   = self.datafiles[elutin_slice]
        xray_array  = self.intensity_array[elutin_slice,usable_slice]
        uv_array    = self.conc_array
        lvector     = self.lvector
        col_header  = self.col_header
        mtd_elution = self.mtd_elution

        data_info   = copy.deepcopy( [ datafiles, xray_array, uv_array, lvector, col_header, mtd_elution ] )
        exec_copy = SerialData( None, None, conc_file=self.conc_file, data_info=data_info, copy_info=self.copy_info )
        exec_copy.set_mc_vector( mapper, conc_factor, elutin_slice=elutin_slice )
        exec_copy.elutin_slice = elutin_slice

        return exec_copy

    def get_v2_copy(self, pre_recog):
        start, stop = pre_recog.get_angle_range()
        datafiles   = self.datafiles
        xray_array  = self.intensity_array[:,start:stop,:]
        uv_array    = self.conc_array
        lvector     = self.lvector
        col_header  = self.col_header
        data_info = copy.deepcopy( [ datafiles, xray_array, uv_array, lvector, col_header])
        data_info += [self.mtd_elution]
        sd_ = SerialData( None, None, conc_file=self.conc_file, data_info=data_info, pre_recog=pre_recog, copy_info=self.copy_info )
        sd_.baseline_corrected = self.baseline_corrected
        sd_.corrected_base = self.corrected_base
        sd_.cd_info = self.cd_info
        sd_.excluded_set = self.excluded_set
        return sd_

    def get_xray_elution_vector( self ):
        ivector_, slice_ = get_xray_elution_vector( self.qvector, self.intensity_array )
        self.xray_slice = slice_
        self.xray_index = (slice_.start + slice_.stop)//2
        self.xr_index = self.xray_index     # for forward compatibility 
        return ivector_

    def is_serial( self ):
        return len(self.ivector) >= REQUIRED_SERIAL_NUM_FILES

    def set_ivector_etcetera(self, debug=False):
        self.ivector = self.get_xray_elution_vector()
        if not self.is_serial():
            return

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            with plt.Dp():
                fig, ax = plt.subplots()
                ax.set_title("set_ivector_etcetera")
                ax.plot(self.ivector, label="ivector")
                ax.legend()
                fig.tight_layout()
                plt.show()

        orig_top_x = None if self.top_x_pair is None else self.top_x_pair[1]
        self.xray_curve = ElutionCurve( self.ivector, orig_top_x=orig_top_x )

    def update_xray_elution_info(self, ivector, xray_slice):
        self.ivector = ivector
        self.xray_slice = xray_slice
        self.xray_index = (xray_slice.start + xray_slice.stop)//2
        self.xr_index = self.xray_index     # for forward compatibility 
        orig_top_x = None if self.top_x_pair is None else self.top_x_pair[1]
        self.xray_curve = ElutionCurve( self.ivector, orig_top_x=orig_top_x )

    def get_uv_elution_vector( self ):
        std_wvlen   = get_setting( 'absorbance_picking' )
        a_vector, i = self.absorbance.get_vector_at( std_wvlen )
        return a_vector

    def get_uv_curve( self ):
        return self.absorbance.a_curve

    def get_xray_curve( self ):
        return self.xray_curve

    def get_xr_curve( self ):
        return self.xray_curve

    def set_intensity_array( self, new_intensity_array ):
        self.intensity_array = new_intensity_array
        self.qvector = self.intensity_array[0, :, 0]
        self.jvector = np.arange( self.intensity_array.shape[0] )
        self.set_ivector_etcetera()

    def apply_data_reduction( self ):
        if self.intensity_reduced:
            # do not repeat reduction
            return

        if get_dev_setting( 'intensity_reduction' ) == 0:
            return

        reduction_cycle = get_dev_setting( 'reduction_cycle' )
        reduction_start = get_dev_setting( 'reduction_start' )
        reduced_array = self.intensity_array[ :, slice( reduction_start, None, reduction_cycle ), : ]
        print( 'Intesity data reduced by ', reduction_cycle, reduction_start )
        self.set_intensity_array( reduced_array )

        self.intensity_reduced = True

    def get_usable_slice( self ):
        self.get_usable_q_limit()
        return self.usable_slice

    def get_usable_q_limit( self, debug=False):
        self.logger.warning('get_usable_q_limit is deprecated. use FlangeLimit instead.')

        if self.usable_slice is not None:
            return self.q_limit_index

        if self.is_serial():
            if self.pre_recog is not None:
                return self.pre_recog.flange_limit

            matrix_data = self.intensity_array[:,:,1].T
            error_data = self.intensity_array[:,:,2].T
            fl = FlangeLimit(matrix_data, error_data, self.xray_curve, self.qvector)
            limit = fl.get_limit(debug=debug)
        else:
            limit = None

        q_limit = None if limit is None else self.qvector[limit]
        write_to_tester_log( 'q_limit_index=%s, q_limit=%s\n' % (str(limit), str(q_limit)) )

        self.q_limit_index = limit
        self.usable_slice = slice( 0, limit )
        return limit

    def get_around_slice( self, vector, val, num ):
        num_half    = num//2
        num_r       = num%2
        i = bisect_right( vector, val )
        return max( 0, i - num_half ), min( len(self.qvector), i + num_half + num_r )

    def update_mapping_info( self, mapper ):
        if self.initial_std_diff is None:
            self.initial_std_diff = mapper.std_diff
        self.current_std_diff = mapper.std_diff

    def set_mc_vector( self, mapper, conc_factor, elutin_slice=None ):
        # TODO: to be removed
        conc_vector = mapper.get_conc_vector( conc_factor )
        if elutin_slice is None:
            mc_vector = conc_vector
        else:
            mc_vector = conc_vector[elutin_slice]
        self.mc_vector  = mc_vector
        self.conc_factor = conc_factor      # TODO: better place to hold this info

    def exclude_intensities( self, to_be_excluded ):
        from molass.DataUtils.AnomalyHandlers import remove_bubbles_impl
        self.wait_until_ready()
        print( 'to_be_excluded=', to_be_excluded )

        remove_bubbles_impl(self.intensity_array, to_be_excluded, self.excluded_set)

        self.has_excluded_xray_elutions = True
        self.set_ivector_etcetera()
        self.logger.info( 'Xray data at elution points ' + str(to_be_excluded) + ' have been removed and interpolated.' )

    def apply_baseline_correction( self, mapped_info, basic_lpm=False, progress_cb=None, return_base=False, debug_obj=None  ):
        frame = inspect.stack()[1]
        self.logger.info("apply baseline_correction called from %s(%d) with basic_lpm=%s", frame.filename, frame.lineno, str(basic_lpm))
        self.logger.info("id:%s, code_context:%s", str(id(self)),''.join(frame.code_context)[:-1])

        debug = False
        if debug:
            xr_array = self.intensity_array.copy()

        base = apply_baseline_correction_impl(
            self.jvector, self.qvector, self.intensity_array,
            mapped_info, basic_lpm=basic_lpm,
            index=self.xray_index,
            ecurve=self.xray_curve,
            progress_cb=progress_cb,
            return_base=return_base, logger=self.logger, debug_obj=debug_obj )

        if debug:
            import molass_legacy.KekLib.DebugPlot as plt
            from molass_legacy.DataStructure.MatrixData import simple_plot_3d
            qv = self.qvector

            with plt.Dp():
                ey1 = get_xray_elution_vector(qv, xr_array)[0]
                ey2 = get_xray_elution_vector(qv, self.intensity_array)[0]
                fig, ax = plt.subplots()
                ax.set_title("apply_baseline_correction: elution curves")
                ax.plot(ey1, label="original")
                ax.plot(ey2, label="corrected")
                ax.legend()
                fig.tight_layout()
                plt.show()

            with plt.Dp():
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,5), subplot_kw=dict(projection="3d"))
                fig.suptitle("apply_baseline_correction: 3d views")
                simple_plot_3d(ax1, xr_array[:,:,1].T, x=qv)
                simple_plot_3d(ax2, self.intensity_array[:,:,1].T, x=qv)
                fig.tight_layout()
                plt.show()

        self.baseline_corrected = True
        self.corrected_base = base
        try:
            self.set_ivector_etcetera(debug=debug)
        except:
            # not fatal as in the degraded case of 20161124
            etb = ExceptionTracebacker()
            self.logger.warning( etb )

        return base

    def compute_scattering_baseline_discomfort( self ):
        if self.scattering_baseline_discomfort is not None:
            return self.scattering_baseline_discomfort

        size    = len(self.intensity_array)
        j       = self.jvector

        param_list = []
        i_end = (self.xray_slice.start + self.xray_slice.stop) // 2 # i.e., average(start, stop) * 2
        for i in np.linspace( 0, i_end, NUM_POINTS_DISCOMFORT_DETECT, dtype=int ):
            y = self.intensity_array[:,i,1]
            # giving curve=self.xray_curve to improve failed LPM cases such as 20181127
            sbl = ScatteringBaseline( y, curve=self.xray_curve )
            A, B = sbl.solve()
            # print( [ i, self.qvector[i], A, B ] )
            M = np.percentile( y, 95 )
            baseline = A*j + B
            integ   = np.sum( baseline )
            integ_a = np.sum( np.abs( baseline ) )
            param_list.append( [A, B, M, size, integ, integ_a] )

        param_array = np.array( param_list )
        average = np.average( param_array[:, 0:4], axis=0 )
        # print( 'average base params=', average )
        sum_    = np.sum( param_array[:, 4:], axis=0 ) / NUM_POINTS_DISCOMFORT_DETECT
        # print( 'sum_=', sum_ )
        # print( 'integrals=', param_array[:, 4] )
        record  = np.hstack( [ average, sum_ ] )
        recstr = '[ ' + ' '.join( [ str(v) for v in record ] ) + ' ]'
        """
            average( A, B, M, size ), sum( integ, integ_a )
        """
        put_to_tester_log_queue( 'baseline discomfort record=%s\n' % recstr  )

        integ_a_sum   = sum_[1]
        self.scattering_baseline_discomfort = integ_a_sum / average[2] / size

        return self.scattering_baseline_discomfort

    def get_averaged_data( self, num_curves_averaged ):

        # Intensity Smoothing
        smoother = IntensitySmootherAveraging( self.intensity_array, num_curves_averaged )
        indeces = np.arange( 0, self.intensity_array.shape[0] )     # these indeces start at 0, not at self.xr_j0
        averaged_intensity_array, average_slice_array = smoother( indeces, return_numpy_ndarray=True )
        self.num_curves_averaged        = num_curves_averaged
        self.averaged_intensity_array   = averaged_intensity_array
        self.average_slice_array        = average_slice_array

        # Concetration Smoothing
        if self.mc_vector is None:
            self.averaged_c_vector = None
        else:
            cvector_smoother = ConcentrationSmootherAveraging( self.mc_vector, num_curves_averaged )
            self.averaged_c_vector =  cvector_smoother( indeces )

        return self.averaged_intensity_array , self.average_slice_array, self.averaged_c_vector

    def get_id_info(self):
        return 'SD(id=%s, corrected=%s)' % (str(id(self)), str(self.baseline_corrected))

    def get_xray_data(self):
        q = self.intensity_array[0,:,0]
        data = self.intensity_array[:,:,1].T
        error = self.intensity_array[:,:,2].T
        return q, data, error

    def get_xr_data_separate_ly(self):
        D = self.intensity_array[:,:,1].T
        E = self.intensity_array[:,:,2].T
        return D, E, self.qvector, self.xray_curve

    def get_uv_data_separate_ly(self):
        D = self.conc_array
        uv_curve = self.get_uv_curve()
        return D, None, self.lvector, uv_curve

    def get_xray_scale(self):
        return self.xray_curve.max_y

    def get_cd_slice(self):
        if self.cd_slice is None:
            qmax = bisect_right(self.qvector, get_setting('cd_eval_qmax'))
            self.cd_slice = slice(0, qmax)
        return self.cd_slice

    def save_xray_data(self, out_folder):
        from molass_legacy.KekLib.NumpyUtils import np_savetxt

        ret_files = []
        for j, path in enumerate(self.datafiles):
            _, file = os.path.split(path)
            # out_path = os.path.join(out_folder, file)
            out_path = '/'.join([out_folder, file])
            print([j], out_path)
            ret_files.append(out_path)
            np_savetxt(out_path, self.intensity_array[j])

        return ret_files

    def log_id_values(self, label):
        self.logger.info("id values at %s are (%g, %g)", label, np.sum(self.intensity_array), np.sum(self.conc_array))

    def get_elution_curves(self):
        uv_curve = self.get_uv_curve()
        xr_curve = self.get_xray_curve()
        return uv_curve, xr_curve
