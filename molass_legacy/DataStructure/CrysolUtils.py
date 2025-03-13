"""
    CrysolUtils.py

    Copyright (c) 2021-2023, SAXS Team, KEK-PF
"""
import os
import re
import numpy        as np

def np_loadtxt_crysol( filename, encoding='cp932' ):
    fh = open( filename, encoding=encoding )

    comment_lines = []

    def generator( fh ):
        for line in fh:
            if len( comment_lines ) < 1:
                comment_lines.append( line )
            else:
                yield line

    try:
        array = np.loadtxt( generator( fh ) )
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "np_loadtxt_crysol: ")
        array = None

    fh.close()
    return array, comment_lines
