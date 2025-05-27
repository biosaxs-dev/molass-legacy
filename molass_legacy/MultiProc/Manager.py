# coding: utf-8
"""
    SubProcess.Manager.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
from time import time, sleep
from molass_legacy._MOLASS.SerialSettings import get_setting

class JobInfo:
    def __init__(self, code, **kwargs):
        self.code = code
        self.analysis_name = None
        self.__dict__.update(kwargs)
        if self.analysis_name is None:
            self.analysis_name = get_setting('analysis_name')

    def __repr__(self):
        return 'JobInfo<%s>' % self.code

class CustomInfo:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_outfolder(self, jobid):
        # override this method
        assert False

manager = None

def activate_manager(log_folder, use_gpu, custominfo):
    global manager
    if manager is None:
        manager = Manager(log_folder, use_gpu, custominfo)
    return manager.num_workers

def terminate_manager_impl():
    if manager is not None:
        manager.terminate()

def print_table():
    print(manager.table)

def get_table():
    return manager.table

def get_list():
    ret = []
    if manager is not None:
        for k, item in sorted(manager.table.items()):
            ret.append((k, item))
    return ret

def submit(job_info):
    job_id = manager.accept_job(job_info)
    return job_id

class Manager:
    def __init__(self, log_folder, use_gpu, custominfo):
        from multiprocessing import Pool, Manager
        from molass_legacy.KekLib.BasicUtils import rename_existing_file
        from ChangeableLogger import Logger
        log_path = log_folder+'/manager.log'
        rename_existing_file(log_path, ext=".log")
        self.logger = Logger(log_path)
        self.use_gpu = use_gpu
        self.custominfo = custominfo
        self.job_counter = 0
        self.manager = Manager()
        self.table = self.manager.dict({})
        self.num_workers = self.decide_num_workers()
        self.pool = Pool(processes=self.num_workers)
        self.logger.info('started with number of workers %d', self.num_workers)


    def decide_num_workers(self):
        import os
        cores = os.cpu_count()//2
        return max(1, cores//2)

    def accept_job(self, job_info):
        from .Driver import run_decomp
        job_id = self.job_counter
        self.job_counter += 1
        submitted = time()
        file_info = [job_info.analysis_name, job_info.infile_name]
        self.table[job_id] = [file_info, None, -2, -1, None, None, None, submitted, None, None, None, 0]
        def success_callback_(result):
            self.success_callback(job_id, result)

        def error_callback_(exc_obj):
            self.error_callback(job_id, exc_obj)

        out_folder = self.custominfo.get_outfolder(job_id)
        self.pool.apply_async(run_decomp, (job_id, out_folder, job_info, self.table, self.use_gpu), {}, success_callback_, error_callback_)
        self.logger.info('accepted %s as %d', str(job_info), job_id)
        return job_id

    def success_callback(self, job_id, result):
        self.logger.info('%03d %s', job_id, str(result))

    def error_callback(self, job_id, exc_obj):
        self.logger.error('%03d error %s', job_id, str(exc_obj))

    def terminate(self):
        self.pool.close()
        self.pool.join()
