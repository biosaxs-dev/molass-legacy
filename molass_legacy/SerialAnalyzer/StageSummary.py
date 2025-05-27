"""

    StageSummary.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF

"""
import os
from openpyxl import Workbook
from molass_legacy.Reports.ReportUtils import make_summary_book
from molass_legacy.KekLib.ExcelCOM import merge_into_a_book
from molass_legacy.KekLib.ProgressInfo import put_info
from .ProgressInfoUtil import NUM_SHEET_TYPES, STREAM_ZERO_EX
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Reports.DefaultFont import set_default_font
set_default_font()

def do_summary_stage(self):
    do_summary(self)
    merge_books(self)

def do_summary( self ):
    # Summary Sheet
    summary_file = self.temp_folder + '/--summary.xlsx'

    if self.excel_is_available:
        wb = Workbook()
    else:
        wb = self.result_wb

    make_summary_book( self.excel_is_available, wb, summary_file, self )
    # self.temp_books.insert( 0, summary_file )
    self.temp_books.append( summary_file )
    self.logger.info( 'summary book done.' )

class MergeBooksArgs:
    def __init__(self, controller):
        self.book_path = os.path.abspath( controller.book_file )
        self.temp_books = controller.temp_books
        if controller.zx and controller.atsas_is_available:
            self.book_path_atsas = controller.atsas_folder + '/datgnom-result.xlsx'
            self.temp_books_atsas = controller.temp_books_atsas
        else:
            self.book_path_atsas = None
            self.temp_books_atsas = []

def merge_books(self, debug=False):
    # this is in order to avoid inconsistency in progressbar control
    # when then the notified number of steps becomes different
    # from the actual number of steps.
    max_num_progress = self.num_peaks_to_exec * NUM_SHEET_TYPES

    if self.excel_is_available:
        args = MergeBooksArgs(self)

        if self.more_multicore:
            self.teller.tell('merge_excel_books', args=args)
        else:
            def progress_callback(i):
                if i < max_num_progress:
                    if debug:
                        self.logger.info("put_info( (%d,1000), %d )", STREAM_ZERO_EX, i+1)
                    put_info( (STREAM_ZERO_EX,1000), i+1 )

            merge_books_impl(self.excel_client, args, self.logger, progress_callback)

        self.logger.info( 'books merge done.' )

    else:
        from molass_legacy.KekLib.OpenPyXlUtil import save_allowing_user_reply
        if len(self.temp_books) > 0:
            book_path = os.path.abspath( self.book_file )
            save_allowing_user_reply(self.result_wb, book_path)
        if len(self.temp_books_atsas) > 0:
            book_path_atsas = self.atsas_folder + '/datgnom-result.xlsx'
            save_allowing_user_reply(self.result_wb_datgnom, book_path_atsas)
        self.logger.info( 'books output complete.' )

    if debug:
        self.logger.info("put_info( (%d,1000), %d )", STREAM_ZERO_EX, max_num_progress)
    put_info( (STREAM_ZERO_EX,1000), max_num_progress )

def merge_books_impl(excel_client, args, logger, progress_cb=None):
    # logger is not yet used
    fontname = get_setting("report_default_font")

    if len(args.temp_books) > 0:
        merge_into_a_book( args.temp_books, args.book_path, excel_client=excel_client, progress_cb=progress_cb, delete_target="Sheet", default_font=fontname )

    if len(args.temp_books_atsas) > 0:
        merge_into_a_book( args.temp_books_atsas, args.book_path_atsas, excel_client=excel_client, progress_cb=progress_cb, default_font=fontname )
