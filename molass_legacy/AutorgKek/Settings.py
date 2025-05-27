# coding: utf-8
"""

    ファイル名：   Settings.py

    Copyright (c) 2016-2021, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.PersistentInfo import PersistentInfo
try:
    from molass_legacy._MOLASS.Version        import get_version_string
except:
    from molass_legacy.AutorgKek.AppVersion     import get_com_version_string as get_version_string

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
default_main_geometry = '1080x500'
mask_           = None

stamp_name = 'version_stamp'

settings_ = None

ITEM_DEFAULTS = {
    stamp_name              : None,
    'recent_folders'        : {},
    'atsas_exe_paths'       : [],
    'result_book'           : 'analysis_report.xlsx',
    'quality_weighting'     : [
                                0.2,    # basic_quality
                                0.2,    # positive_score
                                0.0,    # fit_consistency
                                0.3,    # rg_stdev_score
                                0.3,    # q_rg_score
                              ],
    }

ALTERNATIVE_WEIGHTS = [ 0.2, 0.2, 0.3, 0.2, 0.1 ]

def reload_settings():
    global setting_info
    global settings_

    setting_info    = PersistentInfo( setting_file )
    settings_       = setting_info.get_dictionary()

def clear_settings():
    global settings_
    settings_ = {}
    setting_info.set_dictionary( settings_ )

def get_setting( item ):
    assert( item in ITEM_DEFAULTS )
    if settings_ is None:
        initialize()
    value = settings_.get( item )
    if value is None:
        value = ITEM_DEFAULTS.get( item )
        set_setting( item, value )
    return value

def set_setting( item, value ):
    assert( item in ITEM_DEFAULTS )
    settings_[ item ] = value

def save_settings():
    setting_info.save()

def check_version_stamp():
    version_stamp = get_setting( stamp_name )
    if version_stamp is None or version_stamp != get_version_string():
        # TODO: MessageBox
        clear_settings()
    set_setting( stamp_name, get_version_string() )

def initialize(alt_folder=None):
    # 永続的な記憶をオブジェクト化する（メモリに取り込む）。
    global setting_info, settings_
    setting_file    = 'settings.dump'
    setting_info    = PersistentInfo( setting_file, alt_folder=alt_folder )
    settings_       = setting_info.get_dictionary()
    check_version_stamp()
    # print("initialize", settings_)

def overriding_init(settings):
    global settings_
    settings_ = settings
