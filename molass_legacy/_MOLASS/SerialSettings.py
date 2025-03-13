"""
    SerialSettings.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF
"""
import os
from molass_legacy.KekLib.PersistentPy import PersistentPy, get_py_file_path
try:
    from molass_legacy._MOLASS.Version import get_version_string
except:
    from molass_legacy.AutorgKek.AppVersion import get_com_version_string as get_version_string

# from MachineTypes       import get_display_resolution, get_chassistype_name, get_monitors

# TODO: import wmi into exe
# width, height = get_display_resolution()
# print( 'display width, height=', width, height )

# chassistype_name = get_chassistype_name()
# print( chassistype_name )

# TODO: threading because it takes a significant amountt of time
# for monitor in get_monitors():
#    print( 'monitor=', monitor )
"""
if chassistype_name == 'Desktop':
    default_main_geometry = '1080x750'
else:
    default_main_geometry = '1080x700'
"""

SHOW_TEXT_DICT = {
    'range_type'                :   {   0:'Required Minimum Quality',
                                        1:'Required Minimum Concentration',
                                        2:'Boundary Points from Seq No.',
                                        3:'Intelligent Strategy',
                                        4:'Decomposed Elution Range',
                                        5:'Microfluidic Elution Range',
                                        },
    'zx_boundary_method'        :   {   'AUTO':'Automatic', 'FIXED':'',         'NO':'No boundary'}
    }


default_main_geometry = '1080x500'
# 永続的な記憶をオブジェクト化する（メモリに取り込む）。
setting_file    = 'serial_settings.py'
setting_info    = None
settings_       = None
v1_flag = None

def get_locals_for_settings():
    from numpy import array, nan, inf
    from molass_legacy.DataStructure.PeakInfo import PeakInfo
    from molass_legacy.DataStructure.RangeInfo import DecompEditorInfo, RangeEditorInfo
    from molass_legacy.DataStructure.AnalysisRangeInfo  import PairedRange
    # from Extrapolation.PreviewData import PreviewData, PreviewOptions
    from molass_legacy.Mapping.MappingParams import MappingParams
    from molass_legacy.Trimming.TrimmingInfo import TrimmingInfo
    # from molass_legacy.Decomposer.ModelEvaluator import ModelEvaluator
    # from molass_legacy.Decomposer.DualEvaluator import DualEvaluator
    # from molass_legacy.Models.ElutionCurveModels import EGHA, EMGA     # add required models such as STC, EDM, etc. when ready
    # from molass_legacy.Models.EGH import EGHA
    from molass_legacy._MOLASS.DummyClasses import (EGHA,
                                     EMGA,
                                     DualEvaluator,
                                     ModelEvaluator,
                                     # TrimmingInfo,
                                     # MappingParams,
                                     PreviewData, PreviewOptions,
                                     # PairedRange,
                                     # DecompEditorInfo, RangeEditorInfo,
                                     # PeakInfo,
                                     DummySd, DummyMapper,
                                     DummyJudgeHolder)  # wildcard import not allowed here
    return locals()

def initialize_settings(warn_on_fail=True):
    from molass_legacy.AutorgKek.Settings import overriding_init
    from molass_legacy.SerialAnalyzer.DevSettings import dev_init
    """
    must not be called in non-main processes
    """
    global setting_info, settings_
    setting_info    = PersistentPy(setting_file, locals_=get_locals_for_settings(), warn_on_fail=warn_on_fail)
    settings_       = setting_info.get_dictionary()
    check_version_stamp()
    overriding_init(settings_)
    dev_init(alt_folder=setting_info.alt_folder)

def get_settings_folder():
    return os.path.split( get_py_file_path(setting_file) )[0]

mask_           = None

stamp_name = 'version_stamp'

RANGE_TYPE_POINTS   = 2     # Boundary Points from molass_legacy.Elution No.
UV_BASE_NO          = 0
UV_BASE_CONST       = 1
UV_BASE_STANDARD    = 0
UV_BASE_SHIFTED     = 1
XRAY_BASE_NO        = 0
XRAY_BASE_CONST     = 1
XARY_BASE_LINEAR    = 1
XARY_BASE_QUADRATIC = 2
XARY_BASE_SPLINED   = 3
INTEGRAL_BASELINE = 5
DF_PATH_LENGTH_OLD  = 5.0
PF_PATH_LENGTH      = 8.437
SPRING8_PATH_LENGTH = 8.155
DF_PATH_LENGTH      = 7.071
DF_EXTINCTION       = 1

from molass.DataUtils.Beamline import BEAMLINE_NAME

def set_path_length(measurement_date, uv_device_no):
    if measurement_date is None or measurement_date > 20190930:
        beamline_name = BEAMLINE_NAME.get(uv_device_no, "")
        if beamline_name[0:2] == "PF":
            path_length = PF_PATH_LENGTH
        elif beamline_name[0:2] == "SP":
            path_length = SPRING8_PATH_LENGTH
        else:
            path_length = DF_PATH_LENGTH
    else:
        path_length = DF_PATH_LENGTH_OLD
    set_setting('path_length', path_length)

def get_beamline_name(uv_device_no=None, formal=False):
    beamline_name = get_setting("beamline_name")    # set in GuiMain
    if beamline_name == "":
        if uv_device_no is None:
            uv_device_no = get_setting("uv_device_no")  # set in SerialDataUtils.load_uv_file()
        beamline_name = BEAMLINE_NAME.get(uv_device_no, "")

    if formal:
        beamline_name = beamline_name.replace(" ", ", ").replace("PF", "Photon Factory")

    return beamline_name

ITEM_DEFAULTS = {
    stamp_name              : None,

    'in_folder'             : None,
    'uv_folder'             : None,
    'uv_file'               : None,
    'disable_uv_data'       : 0,
    'an_folder'             : None,
    'analysis_name'         : 'analysis-000',
    'analysis_folder'       : None,
    'fully_automatic'       : 1,
    'fully_optimize'        : 0,
    'auto_navigated_dailog' : 1,

    'data_exclusion'        : 0,
    'enable_usable_limit'   : True,

    'enable_mapping_anim'   : 0,
    'enable_drift_simulation'   : 0,
    'use_xray_conc'         : 0,
    'use_mtd_conc'          : 0,
    'mtd_file_path'         : 0,
    'apply_backsub'         : 0,

    'temp_folder'           : None,
    'auto_number'           : 1,
    'save_averaged_data'    : 1,
    'averaged_data_folder'  : None,
    'file_extension'        : '*.dat',
    'atsas_exe_paths'       : [],
    'num_recent_folders'    : 20,
    'recent_folders'        : {},

    'bad_mapping_ratios_ok' : 0,
    'bad_std_diff_ok'       : 1,
    'enable_auto_helper'    : 1,

    'use_moment_trimming'   : 1,

    'measurement_date'      : None,
    'sangler_version'       : None,
    'uv_restrict_list'      : None,
    'xr_restrict_list'      : None,
    'manually_trimmed'      : False,
    'uv_restrict_copy'      : None,
    'xr_restrict_copy'    : None,
    'cut_before_guinier'    : 2,
    'use_bqlimit'           : 1,
    'force_bqlimit'         : 0,
    'acceptable_rg_consist' : 0.8,

    'conc_adjust'           : 0,    # to be removed
    'conc_adjust_amount'    : None, # to be removed

    'manual_time_shift'     : None,
    'manual_time_scale'     : None,
    'mapper_sync_options'   : None,

    'uv_lpm_option'         : 1,
    'uv_baseline_opt'       : UV_BASE_CONST,
    'uv_baseline_type'      : 0,        # 0: no-correction, 1: linear, 4: shifted, 5: integral
    'uv_baseline_adjust'    : 1,
    'uv_baseline_with_bpa'  : 1,
    'uv_adjust_suppressed'  : 0,
    'uv_wl_lower_bound'     : 250,
    'enable_lrf_baseline'   : 1,
    'uv_device_no'          : "",
    'beamline_name'         : "",

    'baseline_manually'     : 0,        # 0: auto  1: manual
    'xray_baseline_type'    : None,     # 0: no-correction, 1: linear, 5: integral
    'xr_bpa_option'         : 1,
    'manual_end_slices'     : None,
    'xray_baseline_opt'     : XRAY_BASE_CONST,
    'xray_baseline_adjust'  : 0,
    'disable_xray_adjust'   : 1,
    'xray_baseline_with_bpa' : 1,
    'allow_angular_slope_in_mf' : 0,

    'enable_separate_fouling' : 0,
    'unified_baseline_type' : 1,        # baseline_type

    'base_drift_params'     : None,

    'dev_allow_ratio'       : 0.5,      # valid only only when adjust_both; will not be changed when disable_xray_adjust==1

    'fixed_guinier_start'   : 0,
    'guinier_start_point'   : 0,

    'baseline_auto'         : 1,
    'absorbance_baseline_type'  : 0,

    'scattering_base'       : 1,
    'baseline_degree'       : 1,
    'scattering_correction' : 0,
    'correction_iteration'  : 2,
    'peak_mapping_only'     : 0,
    'lpm_variations'        : 0,
    'mapping_show_mode'     : 'locally',
    'oopt_fit_consistency'  : 1,
    'oopt_optimize_only'    : 0,
    'oopt_qrg_limits'       : 'Limited',
    'oopt_qrg_limits_vals'  : [ 0.0, 1.3 ],
    'oopt_qrg_limits_apply' : 1,
    'min_quality'           : 0.5,
    'min_conc'              : 0.5,
    'range_type'            : 3,        # Intelligent Strategy
    'manual_range_info'     : None,
    'analysis_range_info'   : None,
    'preview_params'        : None,
    'num_ranges'            : 1,
    'input_smoothing'       : 1,        # this is forced to be 1 in SerialController.py
    'num_partitions_gpr'    : 200,
    'num_curves_averaged'   : 1,
    'avg_file_postfix'      : '_avg',
    'result_book'           : 'analysis_report.xlsx',
    'path_length'           : DF_PATH_LENGTH,
    'extinction'            : DF_EXTINCTION,
    'conc_factor'           : None,
    'absorbance_picking'    : 280,
    'absorbance_picking_sub'    : 260,
    'zero_absorbance'       : 400,      # can be overridden by flat_wavelength. see UV.PlainCurveUtils.get_flat_wavelength
    'zero_absorbance_auto'  : 1,
    'flat_wavelength'       : None,
    'consider_scatter'      : 0,
    'scatter_picking'       : None,
    'num_points_absorbance' : 1,
    'intensity_picking'     : 0.020,
    'num_points_intensity'  : 11,
    'x_ecurve_pickmethod'  : 0,
    'x_ecurve_pickslice'   : None,
    'x_ecurve_picking_q'   : None,
    'quality_weighting'     : [
                                0.2,    # basic_quality
                                0.2,    # positive_score
                                0.2,    # end_consistency
                                0.2,    # rg_stdev_score
                                0.2,    # q_rg_score
                              ],
    'suppress_low_quality_warning' : 0,
    'locally_2d_regression' : True,
    'using_elution_models'  : 0,
    'use_elution_models'    : 0,
    'has_elution_models'    : False,
    'matrix_formulation'    : 1,

    'enable_synthesized_lrf'    : 0,
    'synthesized_lrf'       : 0,
    'narrow_trimming'       : 1,

    'aq_smoothness'         : 0,
    'aq_positivity'         : 0,
    'penalty_weighting'     : [ 0.1, 0.01, 0.05, 0.005 ],
    'ignore_all_bqs'        : 0,
    'ignore_bq_list'        : None,
    'decomp_editor_info'    : None,
    'range_editor_info'     : None,
    'editor_ranges'         : None,
    'editor_model'          : None,
    'preview_model'         : None,
    'tau_hints_dict'        : None,
    'known_info_list'       : None,
    'svd_reconstruct'       : 1,
    'rank_control'          : 1,
    'allow_rank_variation'  : 1,
    'rank_increment'        : 0,
    'extended_conc_dep'     : 0,
    'conc_dependence'       : None,
    'enable_new_features'   : 1,
    'lrf_bound_correction'  : 1,
    'mapper_cd_color_info'  : None,
    'cd_eval_qmax'          : 0.2,
    'denat_dependent'       : 1,
    'enable_conc_opts'      : 1,
    'conc_curve_type'       : 0,    # 0: mapped UV, 1: scaled Xray
    'weight_matrix_type'    : 0,
    'concentration_datatype' : None,    # 0: XR model, 1: XR data, 2: UV model, 3: UV data
    'enable_conctype_change' : 1,

    'ee_elements_matrix'    : None,

    'zx_penalty_matrix'     : None,
    'zx_num_q_points'       : 3,
    'zx_build_method'       : 'MAX',
    'zx_boundary_method'    : 'AUTO',
    'zx_boundary'           : 0.1,
    'zx_a_file'             : 'zx_A_file.dat',
    'zx_b_file'             : 'zx_B_file.dat',
    'almerge_analyzer'      : 0,
    'axis_direction_desc'   : 0,
    'advance_persist'       : 0,
    'maintenance_mode'      : 0,
    'enable_debug_plot'     : 1,    # 1 also for release versions
    'enable_affine_tran'    : 1,
    'enable_gr_analyzer'    : 1,
    'enable_edm_model'      : 1,
    'decomp_from_separation'    : 0,

    'suppress_warning'      : False,
    'test_pattern'          : None,
    'mapping_image_folder'  : None,
    'decomp_image_folder_xray' : None,
    'decomp_image_folder_uv'   : None,
    'preview_image_folder'  : None,
    'no_excel_warning'      : True,

    'revoke_atsas'          : 0,
    'revoke_excel'          : 0,
    'revoke_cuda'           : 0,

    'auto_n_components'     : 1,
    'n_components'          : 10,
    'max_iterations'        : 200,
    'qmm_separately'        : 0,
    'fixed_random_seeds'    : None,
    'last_random_seeds'     : None,
    'forced_denoise_rank'   : None,
    'last_denoise_rank'     : None,
    'qmm_window_slice'      : None,
    'qmm_window_slice_uv'   : None,
    'denss_fitted_rg'       : 1,
    'found_lacking_q_values' : 0,
    'keep_tempfolder_averaged'  : 0,

    'used_mapping_params'   : None,
    'mapping_canvas_debug'  : 0,
    'beta_release'          : True,
    'using_shared_memory'   : True,

    'default_columtype_id'  : 'ad200w',
    'test_columtype_id'     : None,
    'exclusion_limit'       : 1300,     # Kda
    'poresize'              : None,
    'poresize_bounds'       : None,
    'num_plates_pc'         : 14400,    # per column (30cm)
    'columntype_id'         : None,

    'rg_curve_folder'       : None,
    'reuse_analysis_folder' : 0,
    'elution_model'         : 0,        # 0: Free EGH, 1 : Monopore, 2: LJ EGH, 3: FD EMG, 4: RT EMG, 5: EDM
    'init_sec_params'       : None,

    'optimizer_folder'      : None,     # optimized
    'optjob_folder'         : None,     # optimized/jobs, or optimized/jobs/nnn
    'optworking_folder'     : None,     # optimized/jobs/nnn;  should be unified with the above optjob_folder

    'rg_buffer_file'        : None,
    'default_func_egh'      : 'G0346',
    'default_func_sdm'      : 'G1100',
    'default_func_lj_egh'   : 'G0525',
    'default_func_fd_emg'   : 'G0665',
    'default_func_rt_egh'   : 'G0705',
    'default_func_edm'      : 'G2010',
    'default_objective_func' : None,
    'TAU_BOUND_RATIO'       : 0.65,
    'RG_UPPER_BOUND'        : 150,
    'RATE_R_UPPER_BOUND'    : 300,  # > 228 for 20200623_1
    'recompute_rg_curve'    : 0,
    'optimization_method'   : 0,        # 0: Basin-Hopping, 1: Nested Sampling
    'optimization_strategy' : 0,        # 0: Standard, 1: Custom

    'ratio_interpretation'  : 0,
    'separate_eoii'         : 0,
    'separate_eoii_type'    : 0,
    'separate_eoii_flags'   : [],
    'apply_rg_discreteness' : 0,
    'rg_discreteness_unit'  : 1.0,
    'apply_mw_integrity'    : 0,
    'mw_integer_ratios'     : None,
    'avoid_peak_fronting'   : 0,
    'kratky_smoothness'     : 1,
    'ignore_secconformance' : 0,
    'try_model_composing'   : 0,
    'identification_allowance' : 0.1,
    'apply_sf_bounds'       : 1,
    'sf_bound_ratio'        : 1.0,
    'NUM_MAJOR_SCORES'      : 7,
    'trust_rg_curve_folder' : False,    # temporary fix to trimming inconsistency

    # See also Optimizer.OptimizerSettings.py on the items below
    't0_upper_bound'        : None,
    'uv_basemodel'          : 1,
    'poreexponent'          : None,

    'local_debug'           : False,
    'debug_path'            : None,
    'debug_fh'              : None,

    'report_default_font'   : "Arial",
    'suppress_numba_warning' : True,
   }

V2_TEMPORARY_ITEMS = [
    'uv_wl_lower_bound',    # to be refreshed always without clearing the permanent memory
    'using_shared_memory', 'reuse_analysis_folder', 'elution_model', 'unified_baseline_type', 'init_sec_params',
    'optimizer_folder', 'optjob_folder', 'optworking_folder',
    'rg_buffer_file', 'uv_basemodel', 'default_objective_func',
    'RATE_R_UPPER_BOUND',
    't0_upper_bound', 'poreexponent',
    'poresize', 'exclusion_limit', 'poresize_bounds',
    'optimization_method', 'ratio_interpretation',
    'separate_eoii', 'separate_eoii_type', 'separate_eoii_flags',
    'apply_rg_discreteness', 'rg_discreteness_unit',
    'apply_mw_integrity', 'mw_integer_ratios',
    'avoid_peak_fronting',
    'ignore_secconformance', 'try_model_composing', 'identification_allowance', 'apply_sf_bounds', 'sf_bound_ratio',
    'NUM_MAJOR_SCORES',
    'trust_rg_curve_folder',
    ]

TEMPORARY_ITEMS = [
    'revoke_atsas', 'revoke_excel', 'revoke_cuda',

    'uv_folder', 'uv_file', 'disable_uv_data',
    'averaged_data_folder',

    'fully_optimize',

    'bad_mapping_ratios_ok', 'bad_std_diff_ok',

    'fixed_guinier_start', 'guinier_start_point',

    'manual_time_shift', 'manual_time_scale', 'mapper_sync_options',
    'conc_adjust', 'conc_adjust_amount',
    'baseline_auto', 'absorbance_baseline_type',
    'scattering_base', 'baseline_degree', 'scattering_correction',
    'correction_iteration',
    'consider_scatter',
    'scatter_picking',
    'flat_wavelength',
    'x_ecurve_pickmethod', 'x_ecurve_pickslice', 'x_ecurve_picking_q',

    'enable_usable_limit',
    'use_xray_conc', 'use_mtd_conc',
    'measurement_date',
    'uv_wl_lower_bound',    # to be refreshed always without clearing the permanent memory
    'uv_restrict_list', 'xr_restrict_list', 'manually_trimmed',
    'uv_restrict_copy', 'xr_restrict_copy',

    'uv_lpm_option',
    'uv_baseline_opt', 'uv_baseline_type', 'uv_baseline_adjust', 'uv_adjust_suppressed',
    'uv_baseline_with_bpa', 'uv_device_no', 'beamline_name',
    'baseline_manually', 'xr_bpa_option', 'manual_end_slices',
    'xray_baseline_opt', 'xray_baseline_type', 'xray_baseline_adjust',
    'xray_baseline_with_bpa',
    'apply_backsub',
    'dev_allow_ratio',
    'base_drift_params',

    'conc_factor',
    'conc_adjust', 'conc_adjust_amount',

    'analysis_range_info', 'preview_params',
    'range_type', 'manual_range_info', 'num_ranges',

    'path_length', 'extinction',
    'zx_penalty_matrix',
    'ignore_all_bqs', 'ignore_bq_list',
    'decomp_editor_info', 'range_editor_info',
    'editor_ranges', 'editor_model', 'preview_model', 'tau_hints_dict',
    'known_info_list',
    'svd_reconstruct', 'rank_increment', 'conc_dependence', 'weight_matrix_type',
    'lrf_bound_correction', 'mapper_cd_color_info',
    'using_elution_models',
    'use_elution_models',
    'has_elution_models',
    'concentration_datatype',

    'columntype_id', 'test_columtype_id',

    'auto_n_components', 'n_components', 'max_iterations', 'qmm_separately',
    'fixed_random_seeds', 'last_random_seeds', 'forced_denoise_rank', 'last_denoise_rank',
    'qmm_window_slice', 'qmm_window_slice_uv', 'denss_fitted_rg',
    'found_lacking_q_values',
    'used_mapping_params', 'mapping_canvas_debug',

    'suppress_low_quality_warning',
    'rg_curve_folder',
    'local_debug',
    'debug_path',
    'debug_fh',
    *V2_TEMPORARY_ITEMS,
    ]

XRAY_CONC_ITEM_DEFAULTS = {
    'uv_baseline_opt'       : 0,
    'uv_baseline_type'      : 0,
    'uv_baseline_adjust'    : 0,
    'xray_baseline_opt'     : 1,
    'xray_baseline_type'    : 1,
    'xray_baseline_adjust'  : 0,
    }

MICROFLUIDIC_ITEM_DEFAULTS = {
    'uv_baseline_opt'       : 0,
    'uv_baseline_type'      : 0,
    'uv_baseline_adjust'    : 0,
    'xray_baseline_opt'     : 0,
    'xray_baseline_type'    : 0,
    'xray_baseline_adjust'  : 0,
    }

NO_SAVE_ITEMS = [
    'test_pattern',
    'mapping_image_folder',
    'decomp_image_folder_xray',
    'decomp_image_folder_uv',
    'preview_image_folder',
    ]

ALTERNATIVE_WEIGHTS = [ 0.2, 0.2, 0.3, 0.2, 0.1 ]

def reload_settings():
    global setting_info
    global settings_

    setting_info    = PersistentPy(setting_file, locals_=get_locals_for_settings())
    settings_       = setting_info.get_dictionary()

def clear_settings():
    global settings_
    settings_ = {}
    if setting_info is None:
        initialize_settings()
    setting_info.set_dictionary( settings_ )

def get_setting( item ):
    assert( item in ITEM_DEFAULTS )
    if settings_ is None:
        initialize_settings()

    value = settings_.get( item )
    if value is None:
        value = ITEM_DEFAULTS.get( item )
        set_setting( item, value )
    return value

def get_xray_picking():
    picking = get_setting('x_ecurve_picking_q')
    if picking is None:
        picking = get_setting('intensity_picking')
    return picking

def restore_default_setting( item ):
    set_setting( item, ITEM_DEFAULTS.get( item ) )

def set_setting( item, value ):
    assert( item in ITEM_DEFAULTS )
    if settings_ is None:
        initialize_settings()

    settings_[ item ] = value

def reset_setting( item ):
    assert( item in ITEM_DEFAULTS )
    settings_[item] = ITEM_DEFAULTS[item]

def temporary_settings_begin():
    global settings_save
    settings_save = settings_.copy()

def temporary_settings_end():
    global settings_save
    global settings_
    settings_ = settings_save
    setting_info.set_dictionary( settings_ )

def save_settings(file=None, clear_no_save_items=False, debug=False):
    if clear_no_save_items:
        for item in NO_SAVE_ITEMS:
            set_setting( item, ITEM_DEFAULTS.get( item ) )

    if debug:
        settings_ = setting_info.get_dictionary()
        with open(r"D:\TODO\20230724\temp\setting-debug.txt", "w") as fh:
            for k, v in settings_.items():
                fh.write(str((k, v)) + "\n")

    setting_info.save( file=file )

def load_settings_dict(file):
    from molass_legacy.KekLib.PersistentPy import load_py_file
    dict_ = load_py_file(file, locals_=get_locals_for_settings())
    # TODO: compatibillty check
    return dict_

def load_settings(file):
    initialize_settings()
    dict_ = load_settings_dict(file)
    setting_info.set_dictionary(dict_)

def clear_temporary_settings():
    for item in TEMPORARY_ITEMS:
        set_setting( item, ITEM_DEFAULTS[item] )

def clear_v2_temporary_settings():
    for item in V2_TEMPORARY_ITEMS:
        set_setting( item, ITEM_DEFAULTS[item] )

def do_xray_conc_temporary_settings():
    # care for the fact that defaults are not the same for microfluidic processing
    for k, v in XRAY_CONC_ITEM_DEFAULTS.items():
        set_setting(k, v)

def do_microfluidic_temporary_settings():
    # care for the fact that defaults are not the same for microfluidic processing
    for k, v in MICROFLUIDIC_ITEM_DEFAULTS.items():
        set_setting(k, v)

def check_version_stamp():
    version_stamp = get_setting( stamp_name )
    current_version = get_version_string()
    if current_version.find("_MOLASS 1.0") >= 0:
        modify_to_v1_0_setting()
    if version_stamp is None or version_stamp != current_version:
        # TODO: MessageBox
        clear_settings()
    set_setting( stamp_name, get_version_string() )
    # dump_settings()

def modify_to_v1_0_setting():
    ITEM_DEFAULTS['enable_synthesized_lrf'] = 0
    ITEM_DEFAULTS['synthesized_lrf'] = 0
    ITEM_DEFAULTS['narrow_trimming'] = 1

def dump_settings():
    fh = open("settings.txt", "w")
    sorted_list = sorted(settings_.items(), key=lambda x: x[0])
    fh.write('\n'.join([str(pair) for pair in sorted_list]))
    fh.close()
