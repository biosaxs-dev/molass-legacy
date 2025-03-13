"""
    ATSAS.AutorgRunner.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.AutorgKek.AtsasTools import autorg
from molass_legacy._MOLASS.WorkUtils import get_temp_folder

class AutorgRunner:
    def __init__(self):
        self.temp_folder = get_temp_folder()

    def run_from_array(self, data, remove_tempfile=True):
        tempfile = os.path.join(self.temp_folder, 'autorg-%d.txt' % os.getpid())
        np.savetxt(tempfile, data)
        orig_result, eval_result = autorg(tempfile)
        if remove_tempfile:
            os.remove(tempfile)
        return orig_result, eval_result