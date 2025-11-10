"""

    Version.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF

"""
import platform
from molass_legacy import get_version

def get_version_string(cpuid=False, with_date=False):
    """
    Get MOLASS version string.

    Parameters
    ----------
    cpuid : bool
        If True, append cpuid information to the version string.
    with_date : bool
        If True, append the build date to the version string.
        This option is for backward compatibility.
        To keep it as it was before, update this file's timestamp on each release.

    Returns
    -------
    str
        The version string.
    """
    if cpuid:
        from molass_legacy.KekLib.MachineTypes import get_cpuid
        cpuid = ' cpuid:' + str(get_cpuid())
    else:
        cpuid = ''

    version = get_version()
    if with_date:
        import os
        from datetime import datetime
        timestamp_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pyproject.toml')
        if not os.path.exists(timestamp_file):
            timestamp_file = __file__
        timestamp = os.path.getmtime(timestamp_file)
        version_date = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S ")
    else:
        version_date = ""

    return'MOLASS %s (%spython %s %s%s)' % (
                version, version_date, platform.python_version(), platform.architecture()[0], cpuid )

def molass_version_for_publication():
    import re
    version = get_version_string()
    return re.sub(r"\s+\(.+", "", version)

def is_developing_version():
    return get_version_string().find(":") > 0
