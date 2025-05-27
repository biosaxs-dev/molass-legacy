"""

    ZeroExtrapolationOverlayBook.py

    Copyright (c) 2017-2022, SAXS Team, KEK-PF

"""
import os
import numpy                as np
import logging
# from openpyxl               import Workbook
from OpenPyXlUtil           import save_allowing_user_reply, LINE_WIDTH
from .ZeroExtrapolationResultBook import create_extrapolation_params_chart_
from molass_legacy.Reports.DefaultFont import set_default_font
set_default_font()

class ZeroExtrapolationOverlayBook:
    def __init__( self,
                wb, ws,
                mn, qvector, param_array_list, parent=None, lrf_info=None ):
        m, num_peaks_to_exec = mn

        self.logger  = logging.getLogger( __name__  )
        self.parent  = parent   # SerialController
        self.need_bq = lrf_info.need_bq()

        self.wb = wb
        self.ws = ws

        aq_asc_vector  = param_array_list[0][:,0]
        aq_desc_vector = param_array_list[1][:,0]
        bq_asc_vector  = param_array_list[0][:,1]
        bq_desc_vector = param_array_list[1][:,1]

        paren = '' if num_peaks_to_exec == 1 else '(%d)' % ( m + 1 )
        ws.title = 'Extrapolation Overlay%s' % ( paren )

        ws.append( [ 'Q', 'A(q) Asc', 'A(q) Desc', 'B(q) Asc', 'B(q) Desc' ] )

        num_rows = len(aq_asc_vector)       # <= len(qvector)
        print( 'num_rows=', num_rows )
        for i in range( num_rows ):
            a = aq_asc_vector[i]
            if a <= 0:
                a = None
            d = aq_desc_vector[i]
            if d <= 0:
                d = None
            ws.append( [ qvector[i], a, d, bq_asc_vector[i], bq_desc_vector[i] ] )

        no_orig_param_array = True
        extra_col   = 1
        col_offset  = 2

        row_offset  = 0
        j           = 0         # log scale
        param_name  = "A(q) Asc/Desc Overlayed (1.0 mg/ml)"
        aq_chart = create_extrapolation_params_chart_( ws, 2, num_rows, row_offset, col_offset, j,
                                        param_name, extra_col, no_orig_param_array, overlay=True )
        ws.add_chart( aq_chart, 'G2' )

        if self.need_bq:
            row_offset  = 0
            j           = 1         # linear scale
            param_name  = "B(q) Asc/Desc Overlayed"
            bq_chart = create_extrapolation_params_chart_( ws, 2, num_rows, row_offset, col_offset, j,
                                            param_name, extra_col, no_orig_param_array, overlay=True )
            ws.add_chart( bq_chart, 'G54' )

    def add_format_setting(self, book_file, boundary_indeces=[]):
        from .ZeroExExcelFormatter import ZeroExOverlayArgs
        args = ZeroExOverlayArgs(self.ws.title, self, book_file, boundary_indeces)  # make it picklable

        if self.parent.more_multicore:
            self.add_format_setting_more_multicore(args)
        else:
            self.add_format_setting_less_multicore(args)

    def add_format_setting_more_multicore(self, args):
        self.parent.teller.tell('range_overlay_book', args=args)

    def add_format_setting_less_multicore(self, args):
        from .ZeroExExcelFormatter import add_overlay_format_setting

        add_overlay_format_setting(self.parent.excel_client, args, self.logger)

    def save( self, xlsx_file ):
        save_allowing_user_reply( self.wb, xlsx_file )
