"""
    ExcelTeller.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
from molass_legacy._MOLASS.Processes import register_process, remove_process

class TellerInfo:
    def __init__(self, code, **kwargs):
        self.code = code
        self.__dict__.update(kwargs)

    def __repr__(self):
        return 'TellerInfo<%s>' % self.code

class ExcelTeller:
    def __init__(self, log_folder=None):
        import logging
        self.logger = logging.getLogger(__name__)
        self.log_folder = log_folder
        self.invoke_excel_manager()

    def invoke_excel_manager(self):
        import multiprocessing as mp
        from .ExcelManager import ExcelManager
        self.logger.info('invoke_excel_manager')
        self.excel_manager = ExcelManager()
        self.job_queue = mp.Queue()
        self.manager_process = mp.Process(target=self.excel_manager.manage_loop, args=(self.job_queue, self.log_folder))
        register_process(self.manager_process)
        self.manager_process.start()

    def get_pid(self):
        return self.manager_process.pid

    def stop(self):
        self.job_queue.put(TellerInfo('stop'))
        remove_process(self.manager_process)
        self.manager_process = None

    def tell(self, code, **kwargs):
        self.job_queue.put(TellerInfo(code, **kwargs))
