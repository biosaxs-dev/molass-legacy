import os
import sys
import platform

thisdir_ = os.path.dirname( os.path.abspath( __file__ ) )
sys.path.append( thisdir_ )

def get_version_string():
    return 'KekLib 1.1.4 (2019-11-06 python %s %s)' % ( platform.python_version(), platform.architecture()[0] )
