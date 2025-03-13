"""
    ATSAS.AtsasVersion.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
import glob

def get_atsas_bin_path():
    path = os.path.join(os.environ["ATSAS"], "bin")
    path = path.replace('"', '')
    return path

def get_version():
    # implement another one separately because
    # importing from (or unifying) .AtsasUtils causes a peculiar problem
    from OurImporter import import_module_from_path
    packages_pattern = get_atsas_bin_path() + r'\python*\site-packages'
    path_list = glob.glob(packages_pattern)
    if len(path_list) > 0:
        packages_path = path_list[0]
        file_path = os.path.join(packages_path, "atsas.py")
        atsas = import_module_from_path('atsas', file_path)
        ver = atsas.ATSAS_VERSION
    else:
        ver = 'Unkown'
    return ver

def atsas_version_for_publication():
    return "ATSAS " + get_version()
