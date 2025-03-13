import os
import sys

thisdir_ = os.path.dirname( os.path.abspath( __file__ ) )
if thisdir_ not in sys.path:
    sys.path.append( thisdir_ )
