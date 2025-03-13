# coding: utf-8
"""

    DevSettings.py

    Copyright (c) 2017-2022, SAXS Team, KEK-PF
"""
from PersistentInfo     import PersistentInfo

settings_ = None

def dev_init(alt_folder=None):
    # 永続的な記憶をオブジェクト化する（メモリに取り込む）。
    global setting_info, settings_
    setting_file    = 'dev_settings.dump'
    setting_info    = PersistentInfo( setting_file, alt_folder=alt_folder )
    settings_       = setting_info.get_dictionary()
    # print("dev_init done", settings_)

ITEM_DEFAULTS = {
    'hiresolution'          : False,
    'take_screenshots'      : 0,
    'screenshot_folder'     : None,
    'intensity_reduction'   : 0,
    'reduction_method'      : 'THIN-OUT',
    'reduction_cycle'       : 4,
    'reduction_start'       : 0,
    'no_usable_q_limit'     : 0,
    'log_memory_usage'      : 0,
    'log_xray_lpm_params'   : 0,
    'show_num_iterations'   : 0,
    'suppress_defer_test'   : 0,
    'use_datgnom'           : 1,
    'use_simpleguinier'     : 1,
    'recompute_regboundary' : 0,
    'zx_add_constant'       : 0,
    'add_conc_const'        : 0,
    'individual_bq_ingore'  : 0,
    'tester_zx_save'        : 0,
    'adopt_lom'             : 0,
    'smoothed_xray_conc'    : 0,
    'enable_xb_save'        : 0,
    'make_excel_visible'    : 0,
    'keep_remaining_excel'  : 0,
    'enable_dnd_debug'      : 0,
    'xb_folder'             : None,
    'base_file_postfix'     : None,
    'running_with_tester'   : False,
    'preview_rg_list'       : None,
    }

def get_dev_setting( item ):
    global settings_
    if settings_ is None:
        dev_init()
    # print("get_dev_setting", settings_)
    assert( item in ITEM_DEFAULTS )
    value = settings_.get( item )
    if value is None:
        value = ITEM_DEFAULTS.get( item )
        set_dev_setting( item, value )
    return value

def set_dev_setting( item, value ):
    global settings_
    assert( item in ITEM_DEFAULTS )
    settings_[ item ] = value
