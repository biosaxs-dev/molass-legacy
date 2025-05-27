"""
    ExcelManager.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import os
from time import sleep

class ExcelManager:
    def __init__(self):
        pass

    def manage_loop(self, job_queue, log_folder):
        from molass_legacy.KekLib.ExcelCOM import CoInitialize, CoUninitialize, ExcelComClient
        from molass_legacy.KekLib.ChangeableLogger import Logger
        from molass_legacy._MOLASS.SerialSettings import initialize_settings
        print('manage_loop')
        self.stop = False
        self.interval = 1
        self.job_queue = job_queue
        self.logger = Logger(log_folder+'/manager.log')
        self.logger.info("manager start in process %d", os.getppid())
        initialize_settings(warn_on_fail=False)     # suppress the warning message
        self.logger.info("initialize_settings() done")
        CoInitialize()
        self.excel_client = ExcelComClient()
        self.logger.info("excel_client created with pid=%d", self.excel_client.get_pid())

        while not self.stop:
            print('managing')
            if self.job_queue.empty():
                sleep(self.interval)
            else:
                info = self.job_queue.get()
                if info.code == 'stop':
                    break
                self.execute(info)

        self.close()
        CoUninitialize()

    def close(self):
        # self.excel_client.quit()
        pid = self.excel_client.get_pid()
        del self.excel_client
        self.excel_client = None
        self.logger.info("excel_client with pid=%d destroyed.", pid)

    def execute(self, info):
        self.logger.info('executing %s', info.code)

        try:
            action = self.get_action(info)
            if action is None:
                raise RuntimeError(info)
            else:
                action(info.args)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            self.logger.error(str(etb))

    def get_action(self, info):
        action = self.__getattribute__(info.code)
        return action

    def guinier_book(self, args):
        from molass_legacy.Reports.GuinierExcelFormatter import add_guinier_annonations
        self.logger.info('gunier_book: %s', str(args))
        add_guinier_annonations(self.excel_client, args, self.logger)

    def range_extrapolation_book(self, args):
        from molass_legacy.Reports.ZeroExExcelFormatter import add_result_format_setting
        self.logger.info('range_extrapolation_book: %s', str(args))
        add_result_format_setting(self.excel_client, args, self.logger)

    def range_datgnom_book(self, args):
        from molass_legacy.Reports.DatgnomExcelFormatter import add_datgnom_format_setting
        self.logger.info('range_datgnom_book: %s', str(args))
        add_datgnom_format_setting(self.excel_client, args, self.logger)

    def range_overlay_book(self, args):
        from molass_legacy.Reports.ZeroExExcelFormatter import add_overlay_format_setting
        self.logger.info('range_overlay_book: %s', str(args))
        add_overlay_format_setting(self.excel_client, args, self.logger)

    def summary_book(self, args):
        from molass_legacy.Reports.SummaryExcelFormatter import add_summary_format_setting
        self.logger.info('summary_book: %s', str(args))
        add_summary_format_setting(self.excel_client, args, self.logger)

    def merge_excel_books(self, args):
        from molass_legacy.SerialAnalyzer.StageSummary import merge_books_impl
        self.logger.info('merge_excel_books: %s', str(args))
        merge_books_impl(self.excel_client, args, self.logger)
