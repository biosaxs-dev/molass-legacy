# coding: utf-8
"""
    SdDebugger.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import inspect
from collections import OrderedDict
from DataDebugger import DataDebugger

class SdDebugger(DataDebugger):
    def __init__(self):
        DataDebugger.__init__(self, "sd-debugger-00")

    def save_info(self, sd):
        sub_folder = self.get_sub_folder()
        np.savetxt(sub_folder + '/q.dat', sd.intensity_array[0,:,0])
        np.savetxt(sub_folder + '/M.dat', sd.intensity_array[:,:,1].T)

        param_info = OrderedDict()
        param_info['sd_id_info'] = sd.get_id_info()

        with open(sub_folder + '/param_info.txt', 'w') as fh:
            frame = inspect.stack()[1]
            fh.write("%s(%d)\n" % (frame.filename, frame.lineno))
            fh.write(''.join(frame.code_context) + "----\n")
            fh.write(str(param_info) + '\n')
