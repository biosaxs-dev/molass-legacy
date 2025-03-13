"""
    ATSAS.AutoRg.py

    extracted from SerialAtsasTools.py

    Copyright (c) 2016-2022, SAXS Team, KEK-PF
"""
import os
import glob

USED_IN_MOLASS = True
if USED_IN_MOLASS:
    from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
else:
    from molass_legacy.AutorgKek.Settings import get_setting, set_setting

autorg_exe_array = []

def get_dirs_in_later_order(pattern):
    paths = glob.glob(pattern)
    paths.sort(reverse=True)
    return paths

def get_autorg_exe_paths():
    dir_patterns = [r'C:\Program Files\ATSAS*', r'C:\Program Files (x86)\ATSAS*', r'C:\atsas*']
    atsas_dirs = []

    try:
        current_dir = os.environ["ATSAS"]
        if os.path.exists(current_dir):
            current_dir = current_dir.replace("/", "\\")
            atsas_dirs.append(current_dir)
    except:
        pass

    for pattern in dir_patterns:
        atsas_dirs += get_dirs_in_later_order(pattern)

    autorg_exe_array = []
    for dir_ in atsas_dirs:
        for sub_dir in [ r'\bin', '' ]:
            exe_path = dir_ + sub_dir + r'\autorg.exe'
            if os.path.exists(exe_path) and exe_path not in autorg_exe_array:
                autorg_exe_array.append( exe_path )
                break
    return autorg_exe_array

def set_exe_array(paths=None):
    if paths is None:
        temp_paths = get_setting( 'atsas_exe_paths' )
        if len(temp_paths) == 0:
            temp_paths = get_autorg_exe_paths()
    else:
        temp_paths = paths

    autorg_exe_array.clear()
    for path in temp_paths:
        autorg_exe_array.append(path)

    set_setting( 'atsas_exe_paths', autorg_exe_array )

set_exe_array()

def reset_exe_array():
    global autorg_exe_array
    autorg_exe_array = get_autorg_exe_paths()
