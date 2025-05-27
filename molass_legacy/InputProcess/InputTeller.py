# coding: utf-8
"""
    InputTeller.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.SharedArrays import SharedArrays

class InputInfo:
    def __init__(self, code, log=True, **kwargs):
        self.code = code
        self.log = log
        self.__dict__.update(kwargs)

    def __repr__(self):
        return 'TellerInfo<%s>' % self.code

class InputTeller:
    def __init__(self, log_folder=None, debug=False):
        import logging
        self.logger = logging.getLogger(__name__)
        self.log_folder = log_folder
        self.debug = debug
        self.invoke_input_manager()
        self.xr_data = None
        self.uv_data = None

    def invoke_input_manager(self):
        import multiprocessing as mp
        from .InputManager import InputManager
        self.logger.info('invoke_input_manager')
        self.input_manager = InputManager(debug=self.debug)
        self.job_queue = mp.Queue()
        self.ret_queue = mp.Queue()
        self.manager_process = mp.Process(target=self.input_manager.manage_loop, args=(self.job_queue, self.ret_queue, self.log_folder))
        self.manager_process.start()
        # print(__name__, 'invoke_input_manager', self.manager_process)

    def __del__(self):
        self.stop()

    def stop(self):
        # print(__name__, 'stop', self.manager_process)
        if self.manager_process is not None:
            print(__name__, 'sending stop')
            self.job_queue.put(InputInfo('stop'))
            self.manager_process.join()
            self.manager_process = None

    def tell(self, code, wait=True, **kwargs):
        self.job_queue.put(InputInfo(code, wait=wait, **kwargs))
        if wait:
            ret = self.ret_queue.get()
        else:
            ret = None
        if self.debug:
            print(code, 'ret=', ret)
        return ret

    def start_load_data(self, in_folder, for_sec=True, xray_only=True):
        self.sh_arrays = None
        self.in_folder = in_folder
        self.tell('start_load_data', args=(in_folder, for_sec, xray_only))

    def is_ready(self):
        return self.tell('is_ready', args=None, log=False)

    def get_sarray(self):
        if self.sh_arrays is None:
            name, tuples = self.tell('get_array_info', args=None)
            sarrays = SharedArrays(name=name, tuples=tuples)
            self.sh_arrays = sarrays.get_arrays()

    def get_redundant_xr_data(self):
        import copy
        self.get_sarray()
        return copy.deepcopy(self.sh_arrays[0])

    def get_xr_data(self):
        from XrayData import XrayData
        self.get_sarray()
        sh_array = self.sh_arrays[0]

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            from MatrixData import simple_plot_3d
            fig = plt.figure()
            M = sh_array[:,:,1].T
            if False:
                ax = fig.add_subplot(111, projection='3d')
                simple_plot_3d(ax, M)
            else:
                ax = fig.gca()
                ax.plot(M[100, :])
            plt.show()

        self.xr_data = XrayData(None, array=sh_array)
        return self.xr_data

    def get_uv_data(self):
        from molass_legacy.UVData import UvData
        self.get_sarray()
        self.uv_data = UvData(self.in_folder, array=self.sh_arrays[1], vector=self.sh_arrays[2])

        return self.uv_data

    def get_folder_info(self):
        return self.tell('get_folder_info', args=None)
