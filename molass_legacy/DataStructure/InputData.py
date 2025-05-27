# coding: utf-8
"""
    InputData.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import logging
import time
import numpy as np
from OurProfiler import take_time

class InputData:
    def __init__(self, in_folder, teller=None, debug=False):
        from SerialDataUtils import get_uv_filename, get_mtd_filename

        self.logger = logging.getLogger(__name__)
        self.in_folder = in_folder
        self.teller = teller
        self.debug = debug
        self.for_sec = True
        self.xray_only = False
        filename = get_uv_filename(in_folder)
        if filename is None:
            filename = get_mtd_filename(in_folder)
            if filename is None:
                self.xray_only = True
            else:
                self.for_sec = False

        self.filename = filename
        if self.teller is None:
            self.load_data(self.in_folder)
        else:
            self.start_prepare()

    def start_prepare(self):
        self.teller.start_load_data(self.in_folder, self.for_sec, self.xray_only)

    def wait_for_ready(self):
        teller = self.teller
        interval = 0.5 if self.debug else 0.1
        while not teller.is_ready():
            if self.debug:
                print('waiting for ready')
            time.sleep(interval)
        self.get_ready_for_prepare()

    def get_ready_for_prepare(self):
        self.xr_data = self.teller.get_xr_data()
        self.folder_info = self.teller.get_folder_info()

        if self.for_sec:
            self.uv_data = self.teller.get_uv_data()
        else:
            self.uv_data = None

    def prepare(self):
        if self.teller is not None:
            self.wait_for_ready()
        try:
            if self.for_sec:
                self.prepare_for_sec()
            else:
                self.prepare_for_mf()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            self.logger.error(str(etb))

    def prepare_for_sec(self):
        from SerialData import SerialData
        self.logger.info('preparing input for size-exclusive analysis.')

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            self.xr_data.plot()
            plt.show()

            self.uv_data.plot()
            plt.show()

        folder_info = self.folder_info

        print('folder_info.num_files=', folder_info.num_files)
        files = folder_info.get_files()

        xray_array = self.make_redundant_xray_data()
        uv_array = self.uv_data.data
        lvector = self.uv_data.vector
        col_header = None       # implement if needed
        mtd_elution = None
        data_info = [files, xray_array, uv_array, lvector, col_header, mtd_elution]
        in_folder = self.in_folder
        self.sd = SerialData(in_folder, in_folder, conc_file=folder_info.conc_file, data_info=data_info)
        self.files = files
        self.comments = folder_info.get_comments()

    def make_redundant_xray_data(self):
        # this is for backward compatibility and will be removed in the future
        return self.teller.get_redundant_xr_data()

    def prepare_for_mf(self):
        self.logger.info('preparing input for microfluidic analysis.')
        if self.teller is not None:
            xr_data = self.teller.get_xr_data()
            # print('xr_data.data.shape=', xr_data.data.shape)
            self.xr_data = xr_data

    def load_data(self, in_folder):
        from InputProcess.InputManager import InputManager
        from XrayData import XrayData
        from molass_legacy.UVData import UvData

        manager = InputManager(logger=self.logger)
        array_list = manager.load_data(in_folder, self.for_sec, self.xray_only, shared=False)
        self.xr_data = XrayData(None, array=array_list[0])
        self.folder_info = manager.get_folder_info()
        if self.for_sec:
            self.uv_data = UvData(in_folder, array=array_list[1], vector=array_list[2])
        else:
            self.uv_data = None
