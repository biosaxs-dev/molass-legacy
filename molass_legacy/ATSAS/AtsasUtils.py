"""
    ATSAS.AtsasUtils.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF
"""
import os
import glob
from .AutoRg import autorg_exe_array
from .AtsasVersion import get_atsas_bin_path    # for backward (or SAngler/CorMap) compatibility

def get_version(i=0):
    if len(autorg_exe_array) == 0:
        return None

    from molass_legacy.KekLib.OurImporter import import_module_from_path
    exe_path = autorg_exe_array[i]
    # print('exe_path=', exe_path)
    bin_dir, _ = os.path.split(exe_path)
    packages_pattern = bin_dir + r'\python*\site-packages'
    path_list = glob.glob(packages_pattern)
    if len(path_list) > 0:
        packages_path = path_list[0]
    else:
        atsas_dir, _ = os.path.split(bin_dir)
        packages_path = atsas_dir + r'\python2.7\site-packages'
        if not os.path.exists(packages_path):
             packages_path = None

    if packages_path is None:
        ver = 'Unkown'
    else:
        try:
            file_path = os.path.join(packages_path, "atsas.py")
            atsas = import_module_from_path('atsas', file_path)
            ver = atsas.ATSAS_VERSION
        except:
            print("loading from %s failed." % packages_path)
            ver = 'Failed'
    return ver

def get_versions():
    n = len(autorg_exe_array)
    if n == 0:
        return []
    vers = []
    for i in range(n):
        vers.append(get_version(i))
    return vers
