# coding: utf-8
"""
    ExtrapolationDebugger.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from DataDebugger import DataDebugger

class ExtrapolationDebugger(DataDebugger):
    def __init__(self):
        DataDebugger.__init__(self, "ex-debugger-00")

    def save_info(self, M, C, A, param_info):
        sub_folder = self.get_sub_folder()
        np.savetxt(sub_folder + '/M.dat', M)
        np.savetxt(sub_folder + '/C.dat', C)
        np.savetxt(sub_folder + '/A.dat', A)
        with open(sub_folder + '/param_info.txt', 'w') as fh:
            fh.write(str(param_info) + '\n')
