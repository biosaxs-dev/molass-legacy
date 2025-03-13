"""
    KekLib.EvalUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np      # used in the eval, in cases where the value is like 'np.int64(140)'
from numpy import array, nan, inf   # used in the eval

def eval_file(filename, locals_={}, replacer=None):
    try:
        with open(filename) as fh:
            code = fh.read()
        if replacer is not None:
            code = replacer(code)
        return eval(code, globals(), locals_)
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None,  'Error in file_eval', n=10)