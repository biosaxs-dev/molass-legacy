"""

    Version.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF

"""
import platform
from molass_legacy import get_version

def get_version_string(cpuid=False):
    if cpuid:
        from molass_legacy.KekLib.MachineTypes import get_cpuid
        cpuid = ' cpuid:' + str(get_cpuid())
    else:
        cpuid = ''

    version = get_version()
    return 'MOLASS %s (python %s %s%s)' % (
                version, platform.python_version(), platform.architecture()[0], cpuid )

def molass_version_for_publication():
    import re
    version = get_version_string()
    return re.sub(r"\s+\(.+", "", version)

def is_developing_version():
    return get_version_string().find(":") > 0
