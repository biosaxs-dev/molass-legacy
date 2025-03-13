# -*- coding: utf-8 -*-
"""

    ファイル名：   Preferences.py

    処理内容：

        オプション情報

"""
from __future__ import division, print_function, unicode_literals

from PersistentInfo         import PersistentInfo

DEFAULT_PREFERENCES = {
    'syn_method'        : 'cover',
    'detection_counter' : 'None',
    'postfix_syn'       : '_syn',
    'color_map'         : 'ALBULA',
    'save_policy'       : 'Ask',
    'syn_policy'        : 'all',
    'syn_flags'         : [ 1, 1, 1 ],
    }

preference_file = 'preferences.dump'
preference_info = PersistentInfo( preference_file, DEFAULT_PREFERENCES )
preferences_    = preference_info.get_dictionary()

def reload_preferences():
    global preference_info
    global preferences_

    preference_info    = PersistentInfo( preference_file, DEFAULT_PREFERENCES )
    preferences_       = preference_info.get_dictionary()

def clear_preferences():
    global preferences_
    preferences_ = {}
    preference_info.set_dictionary( preferences_ )

def get_preference( item ):
    assert( item in DEFAULT_PREFERENCES )
    return preferences_.get( item )

def set_preference( item, value ):
    assert( item in DEFAULT_PREFERENCES )
    preferences_[ item ] = value

def temporary_preferences_begin():
    global preferences_save
    preferences_save = preferences_.copy()

def temporary_preferences_end():
    global preferences_save
    global preferences_
    preferences_ = preferences_save
    preference_info.set_dictionary( preferences_ )

def get_usual_preference( item ):
    global preferences_save
    return preferences_save.get( item )

def save_preferences():
    preference_info.save()
