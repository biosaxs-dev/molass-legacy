# -*- coding: utf-8 -*-
"""

    ファイル名：   settings.py

    処理内容：

        設定情報

"""
from __future__ import division, print_function, unicode_literals

from PersistentInfo     import PersistentInfo
from SAnglerMask        import SAnglerMask

# 永続的な記憶をオブジェクト化する（メモリに取り込む）。
setting_file    = 'settings.dump'
setting_info    = PersistentInfo( setting_file )
settings_       = setting_info.get_dictionary()
mask_           = None

ITEM_DEFAULTS = {
    'in_folder'         : None,
    'log_file'          : None,
    'mask_file'         : None,
    'adj_folder'        : None,
    'syn_folder'        : None,
    'op_option'         : 'MANUAL',
    'watch_interval'    : 180,
    }

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
    value = settings_.get( item )
    if value == None:
        value = ITEM_DEFAULTS.get( item )
        set_setting( item, value )
    return value

def set_setting( item, value ):
    assert( item in ITEM_DEFAULTS )
    settings_[ item ] = value

def temporary_settings_begin():
    global settings_save
    settings_save = settings_.copy()

def temporary_settings_end():
    global settings_save
    global settings_
    settings_ = settings_save
    setting_info.set_dictionary( settings_ )

def save_settings():
    setting_info.save()

def set_mask( filepath ):
    global mask_

    try:
        mask_ = SAnglerMask( filepath )
        return True
    except Exception as err:
        return False

def get_mask():
    return mask_
