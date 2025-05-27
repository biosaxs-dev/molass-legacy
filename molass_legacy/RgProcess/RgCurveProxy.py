"""
    RgProcess.RgCurveProxy.py

    Copyright (c) 2022-2025, SAXS Team, KEK-PF
"""
import os
import logging
from time import time
import numpy as np  # used in the eval, in cases where the value is like 'np.int64(140)'
from numpy import array, int64      # array and int64 are referenced during the eval below
from molass_legacy.RgProcess.RgCurve import RgCurve
from molass_legacy.Trimming import TrimmingInfo   # used in eval_file

class RgCurveProxy(RgCurve):
    def __init__(self, ecurve, rg_folder, progress_cb=None):
        self.logger = logging.getLogger(__name__)

        t0 = time()

        if ecurve is not None:
            self.x = ecurve.x
            self.y = ecurve.y
        self.ecurve = ecurve
        self.X = None
        self.excl_info = None
        self.excl_spline = None

        objects = []
        for filename in ["slices.txt", "states.txt", "segments.txt", "qualities.txt", "rg_trimming.txt", "baseline_type.txt"]:
            path  = os.path.join(rg_folder, filename)
            if os.path.exists(path):
                from molass_legacy.KekLib.EvalUtils import eval_file
                objects.append(eval_file(path, locals_=globals()))
            else:
                objects.append(None)    # some files may not exist

        self.slices = objects[0]
        self.states = objects[1]
        self.segments = objects[2]
        self.qualities = objects[3] if len(objects) > 3 else None
        self.rg_trimming = objects[4] if len(objects) > 4 else None
        self.baseline_type = objects[5] if len(objects) > 5 else None

        self.logger.info("It took %.3g seconds for rg_curve proxy construction.", time()-t0)
