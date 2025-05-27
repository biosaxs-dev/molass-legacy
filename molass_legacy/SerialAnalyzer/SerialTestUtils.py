# coding: utf-8
"""

    SerialTestUtils.py

    Copyright (c) 2018-2021, SAXS Team, KEK-PF

"""
import numpy                as np
from molass_legacy.KekLib.BasicUtils             import clear_dirs_with_retry, Struct
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from ChangeableLogger       import Logger
from molass_legacy._MOLASS.SerialSettings         import set_setting
from SerialDataUtils        import get_mtd_filename
from molass_legacy.SerialAnalyzer.AbnormalityCheck       import bubble_check
from InputSmootherAveraging import ConcentrationSmootherAveraging
from molass_legacy.KekLib.OurTkinter             import Tk
from molass_legacy.KekLib.DebugPlot              import set_plot_env
from molass_legacy.KekLib.TkUtils                import adjusted_geometry, get_tk_root

def create_our_logger(temp_dir):
    # clear_dirs_with_retry( [ temp_dir ] )
    logger = Logger( temp_dir + '/test.log' )
    return logger

def standard_setup(temp_dir):
    clear_dirs_with_retry( [temp_dir] )
    logger = create_our_logger( temp_dir )
    set_setting( 'temp_folder', temp_dir )
    set_setting( 'analysis_folder', temp_dir )
    return logger

def debug_plot_runner(run_func):
    root = Tk.Tk()
    root.geometry( adjusted_geometry( root.geometry() ) )
    root.update()
    root.withdraw()
    set_plot_env(root)

    def run_closure():
        run_func()
        root.quit()

    root.after(0, run_closure)
    root.mainloop()
    root.destroy()

def get_serialdata_simply(in_folder):
    from SerialDataLoader import SerialDataLoader
    sdl = SerialDataLoader()
    sdl.load_from_folders(in_folder, in_folder)
    sd = sdl.get_data_object()
    return sd

def prepare_serialdata_env( in_folder, uv_folder=None, conc_factor=5, mapper_hook=None, enable_auto_helper=True,
                            xray_only=False, mtd_conc=False, adjust=False, root=None, logger=None, ret_pre_recog=False ):
    from SerialDataLoader import SerialDataLoader
    set_setting( 'in_folder', in_folder )   # for xray 3d plot
    if enable_auto_helper:
        set_setting( 'enable_auto_helper', 1 )
    else:
        set_setting( 'enable_auto_helper', 0 )

    if uv_folder is None:
        uv_folder = in_folder

    sdl = SerialDataLoader()
    if xray_only or mtd_conc:
        if mtd_conc:
            set_setting( 'use_mtd_conc', 1 )
            set_setting( 'use_xray_conc', 0 )
            filename = get_mtd_filename( in_folder )
            set_setting( 'mtd_file_path', filename )
        else:
            set_setting( 'use_mtd_conc', 0 )
            set_setting( 'use_xray_conc', 1 )
        sdl.load_xray_data_only( in_folder )
    else:
        sdl.load_from_folders( in_folder, uv_folder )
    sd  = sdl.get_data_object()

    exclude = bubble_check( sd )
    if len(exclude) > 0:
        sd.exclude_intensities( exclude )
        sdl.memorize_exclusion( sd )

    if ret_pre_recog or adjust:
        if sd.is_serial():
            from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
            try:
                pre_recog = PreliminaryRecognition(sd)
                sd.conc_factor = conc_factor
            except:
                etb = ExceptionTracebacker()
                print(etb.last_lines())
                pre_recog = None

    if adjust:
        # this will do the baseline correction
        sd = pre_recog.get_analysis_copy()
        mapper = get_mapper_for_the_test(root, sd, adjust=adjust, logger=logger)

    if ret_pre_recog:
        return sd, pre_recog
    else:
        return sd

global_info = None

def get_mapper_for_the_test(root, sd_, adjust=False, default_opt_params=False, opt_methods=None, logger=None):
    from molass_legacy.Mapping.MapperConstructor import create_mapper

    if logger is not None:
        logger.info('get_mapper_for_the_test: adjust=' + str(adjust))

    if default_opt_params:
        opt_params = None
    else:
        from molass_legacy.Mapping.MappingParams import MappingParams

        opt_params_default = MappingParams(
            ( 'uv_baseline_opt',            1 ),
            ( 'uv_baseline_type',           1 ),
            ( 'uv_baseline_adjust',         0 ),
            ( 'uv_baseline_with_bpa',       0 ),
            ( 'xray_baseline_opt',          0 ),
            ( 'xray_baseline_type',         0 ),
            ( 'xray_baseline_adjust',       0 ),
            ( 'xray_baseline_with_bpa',     0 ),
            ( 'dev_allow_ratio',            0.5 ),
            )

        opt_params_adjust = MappingParams(
            ( 'uv_baseline_opt',            1 ),
            ( 'uv_baseline_type',           1 ),
            ( 'uv_baseline_adjust',         1 ),
            ( 'uv_baseline_with_bpa',       0 ),
            ( 'xray_baseline_opt',          1 ),
            ( 'xray_baseline_type',         0 ),
            ( 'xray_baseline_adjust',       1 ),
            ( 'xray_baseline_with_bpa',     1 ),
            ( 'dev_allow_ratio',            0.5 ),
            )

        if adjust:
            opt_params = opt_params_adjust
        else:
            opt_params = opt_params_default

    mapper = create_mapper( root, sd_, opt_params=opt_params, opt_methods=opt_methods, logger=logger )
    return mapper

def get_tttt_data_objects( in_folder, uv_folder=None, xray_only=False, adjust=False,
                            exec_copy=False, analysis_copy=False,
                            default_opt_params=False, root=None, logger=None,
                            opt_methods=None, mtd_conc=False ):
    global global_info

    sd_orig = prepare_serialdata_env( in_folder, uv_folder, xray_only=xray_only, mtd_conc=mtd_conc )
    # sd.absorbance.solve_bottomplane_LPM()



    if root is None:
        root = Tk.Tk()
        root.geometry( adjusted_geometry( root.geometry() ) )
        root.withdraw()

    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    pre_recog = PreliminaryRecognition(sd_orig)

    if analysis_copy:
        # sd_ = pre_recog.get_analysis_copy()
        sd_ = sd_orig._get_analysis_copy_impl(pre_recog)
    else:
        sd_ = sd_orig

    mapper = get_mapper_for_the_test(root, sd_, adjust=adjust, default_opt_params=default_opt_params, opt_methods=opt_methods, logger=logger)

    conc_factor = 5
    if exec_copy:
        sd = sd_.get_exec_copy( mapper, conc_factor )
        sd.absorbance = sd_orig.absorbance    # exec_copy does not have absorbance
    else:
        sd = sd_
        sd.set_mc_vector(mapper, conc_factor)

    global_info = Struct( parent=root, sd=sd, mapper=mapper )

    return root, sd, mapper

def get_global_info():
    global global_info
    if global_info is None:
        raise RuntimeError( "call get_tttt_data_objects before getting global info" )
    return global_info

def set_setting_for_baseline_correction( opt_params ):
    set_setting( 'uv_baseline_opt',         opt_params[ 'uv_baseline_opt' ] )
    set_setting( 'uv_baseline_const_opt',   opt_params[ 'uv_baseline_const_opt' ] )
    set_setting( 'uv_baseline_adjust',      opt_params[ 'uv_baseline_adjust' ] )
    set_setting( 'xray_baseline_opt',       opt_params[ 'xray_baseline_opt' ] )
    set_setting( 'xray_baseline_const_opt', opt_params[ 'xray_baseline_const_opt' ] )
    set_setting( 'xray_baseline_adjust',    opt_params[ 'xray_baseline_adjust' ] )

class MockExecuter:
    def __init__( self, sd ):
        from molass_legacy.Mapping.ElutionMapper import ElutionMapper
        mapper  = ElutionMapper( sd )
        mapper.optimize()
        conc_factor = 5
        self.mapped_info    = mapper.get_mapped_info()
        sd.set_mc_vector( mapper, conc_factor )

        # Concetration Smoothing
        cvector_smoother = ConcentrationSmootherAveraging( sd.mc_vector, 5 )
        indeces = np.arange( 0, sd.intensity_array.shape[0] )
        c_vector = cvector_smoother( indeces )
        c_vector[ np.abs( c_vector ) < 1e-5 ] = 1e-5

        self.c_vector       = c_vector
        # self.quality_array  = 

def get_uv_elution_vector( sd, correct_base=True, use_LB=False, plot=False):
    from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
    from molass_legacy.UV.Absorbance import Absorbance

    absorbance = Absorbance( sd.lvector, sd.conc_array, sd.xray_curve, col_header=sd.col_header )
    a_vector    = absorbance.a_vector

    if correct_base:
        try:
            pre_recog = PreliminaryRecognition(sd)
            j_min, j_max = pre_recog.get_flow_changes()
            if use_LB:
                absorbance.solve_bottomplane_LB(j_min, j_max)
            else:
                absorbance.solve_bottomplane_LPM(j_min, j_max)
            base    = absorbance.get_standard_elution_base()
        except:
            # when mtd_conc==True
            base = 0
    else:
        base = 0

        if plot:
            import molass_legacy.KekLib.DebugPlot as plt
            plt.plot( a_vector )
            if type(base) != int:
                plt.plot( base, color='red' )
            plt.show()

    return a_vector - base
