# coding: utf-8
"""
    InputManager.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
from time import sleep
import numpy as np
import threading
from .FolderInfo import FolderInfo

class InputManager:
    def __init__(self, logger=None, debug=False):
        # logger is not None for single process uses
        self.logger = logger
        self.debug = debug

    def manage_loop(self, job_queue, ret_queue, log_folder):
        from ChangeableLogger import Logger
        if self.debug:
            print('manage_loop')
        self.stop = False
        self.interval = 0.5 if self.debug else 0.1
        self.job_queue = job_queue
        self.ret_queue = ret_queue
        self.logger = Logger(log_folder+'/input.log')
        self.logger.info('start')
        self.busy = False
        self.load_thread = None

        while not self.stop:
            if self.debug:
                print('managing')
            if self.job_queue.empty():
                sleep(self.interval)
            else:
                info = self.job_queue.get()
                if info.code == 'stop':
                    break
                self.execute(info)

        self.close()

    def close(self):
        self.logger.info('stop')

    def execute(self, info):
        if info.log:
            self.logger.info('executing %s', info.code)

        try:
            action = self.get_action(info)
            if action is None:
                raise RuntimeError(info)
            else:
                ret = action(info.args)
                if info.wait:
                    self.ret_queue.put(ret)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            self.logger.error(str(etb))

    def get_action(self, info):
        action = self.__getattribute__(info.code)
        return action

    def start_load_data(self, args):
        print('start_load_data',  args)
        self.busy = True
        self.load_thread = threading.Thread(
                                target=self.load_data,
                                name='LoadDataThread',
                                args=args,
                                )
        self.load_thread.start()
        self.logger.info( 'started loading ' + str(args) )

    def load_data(self, in_folder, for_sec, xray_only, shared=True):
        from SerialDataUtils import load_intensity_files, load_uv_array
        if shared:
            from molass_legacy.KekLib.SharedArrays import SharedArrays

        print('load_data',  in_folder)

        self.sa_list = []

        array, files, comments = load_intensity_files(in_folder, return_comments=True)
        self.logger.info( 'loaded ' + in_folder )
        array_list = [array]
        self.files = files
        self.comments = comments

        if for_sec and not xray_only:
            array, vector, conc_file = load_uv_array(in_folder)
            self.conc_file = conc_file
            array_list += [array, vector]
        else:
            self.conc_file = None

        if shared:
            self.sarrays = SharedArrays(array_list)
        else:
            return array_list

        self.busy = False

    def join_load_thread(self):
        if self.load_thread is not None:
            self.load_thread.join()
            self.load_thread = None

    def is_ready(self, args):
        return not self.busy

    def get_array_info(self, args):
        return self.sarrays.get_name(), self.sarrays.get_tuples()

    def get_folder_info(self, args=None):
        return FolderInfo(self.files, self.conc_file, self.comments)
