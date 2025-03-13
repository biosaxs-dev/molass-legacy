"""

    CormapVersion.py

"""
import platform

def get_version_string(cpuid=False):
    if cpuid:
        from MachineTypes import get_cpuid
        cpuid = ' cpuid:' + str(get_cpuid())
    else:
        cpuid = ''

    return 'Cormap Maker 0.0.5 (2021-12-22 python %s %s%s)' % (
                platform.python_version(), platform.architecture()[0], cpuid )
