"""

    GuinierAnalysisResultBook.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF

"""
import numpy as np
import logging
from openpyxl.chart import BarChart, LineChart, ScatterChart, Reference, Series
from openpyxl.chart.series_factory  import SeriesFactory
from openpyxl.chart.error_bar import ErrorBars
from openpyxl.chart.data_source import NumDataSource, NumData, NumVal
from openpyxl.chart.layout import Layout, ManualLayout
from molass_legacy.KekLib.OpenPyXlUtil import save_allowing_user_reply, LINE_WIDTH
from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
from molass_legacy.KekLib.HtmlColorNames import *
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from molass_legacy.Reports.DefaultFont import set_default_font
set_default_font()

CHART_WIDTH     = 54
CHART_ADJUST    = 0.005

COLNAMES = [
    '№',               #  0 A
    'folder',           #  1 B
    'file',             #  2 C
    'concentration',    #  3 D
    'basic_quality',    #  4 E
    'positive_score',   #  5 F
    'stdev_score',      #  6 G      row[5] end_consistency
    'q_rg_score',       #  7 H      row[6] stdev_score
    'end_consistency',  #  8 I      row[7] q_rg_score
    'atsas_qualiity',   #  9 J      row[8] atsas_qualiity
    'molass.I(0)',      # 10 K
    'gp.I(0)',          # 11 L
    'atsas.I(0)',       # 12 M
    'molass.Rg',        # 13 N
    'gp.Rg',            # 14 O
    'atsas.Rg',         # 15 P
    'molass.I(0)/conc.',# 16 Q
    'gp.I(0)/conc.',    # 17 R
    'atsas.I(0)/conc.', # 18 S
    ]


class GuinierAnalysisResultBook:
    def __init__( self, wb, ws, array, j0, parent=None ):

        self.logger  = logging.getLogger( __name__  )
        self.parent  = parent   # SerialController
        self.wb_path = None
        self.wb = wb
        self.ws = ws
        self.j0 = j0
        ws.title = 'Guinier Analysis'
        self.num_points = len( array )
        self.excel_client_ = None
        self.include_end_consistency = True

        colnames_ = list( COLNAMES )
        if self.include_end_consistency:
            for k in [ 17, 14, 11 ]:
                del colnames_[k]
        else:
            colnames_[8] = COLNAMES[9]
            colnames_[9] = COLNAMES[8]

        ws.append( colnames_ )

        Iz_array = []
        Izc_array = []
        Rg_array = []
        i = 1
        for row in array:
            i += 1
            row_ = list(row)
            if self.include_end_consistency:
                row_[7] = row[5]        # move end_consistency
                row_[5:7] = row[6:8]    # shift stdev_score, q_rg_score
                for k in [ 13, 10 ]:
                    del row_[k]
                fmt_cols = [ 'K', 'L' ]
            else:
                row_[8] = row[5]        # move end_consistency
                row_[5:8] = row[6:9]    # shift stdev_score, q_rg_score, atsas_qualiity
                fmt_cols = [ 'K', 'L', 'M' ]

            fmt_rows =  [ '=IF(D{0}=0,0,{1}{0}/D{0})'.format( i, c ) for c in fmt_cols ]

            ws.append( [j0+i-2] + row_ + fmt_rows )
            # autorg_kek is excluded for it may give extraordinary values for some data
            Iz_array.append( max( row[9:11] ) )     # 9:11 => [ kek.I0, fit.I0 ]
            Izc_array.append( max( row[9:11] )/row[2]  )
            Rg_array.append( row[12] )

        self.iz_array   = np.array(Iz_array)
        self.Izc_array  = np.array(Izc_array)
        self.rg_array   = np.array(Rg_array)

        self.xvalues    = Reference(ws, min_col=1, min_row=2, max_row=len(array)+1)

        if self.include_end_consistency:
            chart_start_col = 'R'
        else:
            chart_start_col = 'U'

        if get_setting( 'use_xray_conc' ) == 0:
            xray_text = ''
        else:
            xray_text = ' (Xray-proportional'
            if get_dev_setting( 'smoothed_xray_conc' ) == 1:
                xray_text += ' & smoothed'
            xray_text += ')'
        self.ceate_ConcentrationChart( array, ws,       chart_start_col+'2', 'Concentration' + xray_text, 4 )
        self.create_QualityChart( array, ws,            chart_start_col+'19' )
        self.ceate_IzChart( array, ws,                  chart_start_col+'36' )
        self.ceate_Iz_devided_by_C_Chart( array, ws,    chart_start_col+'53' )
        self.ceate_RgChart( array, ws,                  chart_start_col+'70' )

    def ceate_ConcentrationChart( self, array, ws, pos, title, col ):

        c_ = LineChart()
        c_.title = title
        c_.style = 13
        c_.y_axis.title = 'Concentration'
        c_.x_axis.title = 'Elution №'

        num_rows = len(array)
        data = Reference(ws, min_col=col, max_col=col, min_row=1, max_row=num_rows+1 )
        c_.add_data(data, titles_from_data=True)
        c_.set_categories(self.xvalues)
        c_.width    = CHART_WIDTH
        c_.hight    = 15

        colors = [ steelblue ]

        # Style the lines

        for i, color in enumerate(colors):
            s_ = c_.series[i]
            s_.marker.symbol = "circle"
            s_.marker.graphicalProperties.solidFill         = color
            s_.marker.graphicalProperties.line.solidFill    = color
            s_.graphicalProperties.line.solidFill           = color
            s_.graphicalProperties.line.width               = LINE_WIDTH    # in EMUs

        self.set_layout( c_, adjust=0 )
        ws.add_chart(c_, pos)

    def create_QualityChart( self, array, ws, pos ):

        c_ = BarChart()
        c_.title = "Quality Factors"
        c_.type  = "col"
        c_.style = 10
        c_.grouping = "stacked"
        c_.overlap = 100
        # c_.gapWidth = 300
        c_.y_axis.title = 'Quality'
        c_.x_axis.title = 'Elution №'

        num_rows = len(array)
        max_col = 10 if self.include_end_consistency else 9
        data = Reference(ws, min_col=5, max_col=max_col, min_row=1, max_row=num_rows+1 )
        c_.add_data(data, titles_from_data=True)
        c_.set_categories(self.xvalues)
        c_.width    = CHART_WIDTH
        c_.hight    = 15

        # colors = [ lightskyblue, darkorange, gold, dodgerblue, limegreen, red ]
        # remove end_consistency
        colors = [ lightskyblue, darkorange, dodgerblue, limegreen, red ]
        if self.include_end_consistency:
            colors.insert( 4, gold )

        # Style the lines

        for i, color in enumerate(colors):
            s_ = c_.series[i]
            s_.graphicalProperties.solidFill                = color
            s_.graphicalProperties.line.solidFill           = color
            # the last series, i.e. atsas quality, which is the fitfh FullSeriesCollection,
            # will be modified using COM in self.add_annonations_impl method.

        self.set_layout( c_, adjust=0 )
        ws.add_chart(c_, pos)

    def ceate_IzChart( self, array, ws, pos ):
        c_ = LineChart()
        c_.title = 'I(0)'
        c_.style = 13
        c_.y_axis.title = 'I(0)'
        c_.x_axis.title = 'Elution №'

        if self.include_end_consistency:
            min_col=11
            max_col=12
            colors = [ steelblue, darkorange ]
        else:
            min_col=11
            max_col=13
            colors = [ steelblue, darkgray, darkorange ]

        num_rows = len(array)
        data = Reference(ws, min_col=min_col, max_col=max_col, min_row=1, max_row=num_rows+1 )
        c_.add_data(data, titles_from_data=True)
        c_.set_categories(self.xvalues)
        c_.width    = CHART_WIDTH
        c_.hight    = 15

        # Style the lines

        for i, color in enumerate(colors):
            s_ = c_.series[i]
            s_.marker.symbol = "circle"
            s_.marker.graphicalProperties.solidFill         = color
            s_.marker.graphicalProperties.line.solidFill    = color
            s_.graphicalProperties.line.solidFill           = color
            s_.graphicalProperties.line.width               = LINE_WIDTH    # in EMUs

        self.set_layout( c_, adjust=CHART_ADJUST )
        ws.add_chart(c_, pos)

    def ceate_Iz_devided_by_C_Chart( self, array, ws, pos ):

        c_ = LineChart()
        c_.title = "I(0) / Concentration"
        c_.style = 13
        c_.y_axis.title = 'I(0) / Conc.'
        c_.x_axis.title = 'Elution №'

        if self.include_end_consistency:
            min_col=15
            max_col=16
            colors = [ steelblue, darkorange ]
        else:
            min_col=17
            max_col=19
            colors = [ steelblue, darkgray, darkorange ]

        num_rows = len(array)
        data = Reference(ws, min_col=min_col, max_col=max_col, min_row=1, max_row=num_rows+1 )
        c_.add_data(data, titles_from_data=True)
        c_.set_categories(self.xvalues)
        c_.width    = CHART_WIDTH
        c_.hight    = 15

        # Style the lines

        for i, color in enumerate(colors):
            s_ = c_.series[i]
            s_.marker.symbol = "circle"
            s_.marker.graphicalProperties.solidFill         = color
            s_.marker.graphicalProperties.line.solidFill    = color
            s_.graphicalProperties.line.solidFill           = color
            s_.graphicalProperties.line.width               = LINE_WIDTH    # in EMUs

        self.set_layout( c_, adjust=CHART_ADJUST*4.6 )
        ws.add_chart(c_, pos)

    def ceate_RgChart( self, array, ws, pos ):

        c_ = LineChart()
        c_.title = 'Rg'
        c_.style = 13
        c_.y_axis.title = 'Rg'
        c_.x_axis.title = 'Elution №'

        if self.include_end_consistency:
            min_col=13
            max_col=14
            colors = [ steelblue, darkorange ]
        else:
            min_col=14
            max_col=16
            colors = [ steelblue, darkgray, darkorange ]

        num_rows = len(array)
        data = Reference(ws, min_col=min_col, max_col=max_col, min_row=1, max_row=num_rows+1 )
        c_.add_data(data, titles_from_data=True)
        c_.set_categories(self.xvalues)
        c_.width    = CHART_WIDTH
        c_.hight    = 15

        # Style the lines

        for i, color in enumerate(colors):
            s_ = c_.series[i]
            s_.marker.symbol = "circle"
            s_.marker.graphicalProperties.solidFill         = color
            s_.marker.graphicalProperties.line.solidFill    = color
            s_.graphicalProperties.line.solidFill           = color
            s_.graphicalProperties.line.width               = LINE_WIDTH    # in EMUs

        self.set_layout( c_, adjust=CHART_ADJUST*0.4 )
        ws.add_chart(c_, pos)

    def set_layout( self, chart, adjust=0 ):
        chart.layout = Layout(
            ManualLayout(
            x=0.04-adjust, y=0.1,
            h=0.8,  w=0.895+adjust,
            xMode="edge",
            yMode="edge",
            )
        )

    def add_annonations( self, book_file, ranges ):
        from .GuinierExcelFormatter import GuinierReportArgs
        args = GuinierReportArgs(self.ws.title, self, book_file, ranges)    # make it picklable

        if self.parent.more_multicore:
            self.add_annonations_more_multicore(args)
        else:
            self.add_annonations_less_multicore(args)

    def add_annonations_more_multicore(self, args):
        self.parent.teller.tell('guinier_book', args=args)

    def add_annonations_less_multicore(self, args):
        from .GuinierExcelFormatter import add_guinier_annonations

        try:
            add_guinier_annonations(self.parent.excel_client, args, self.logger)
        except Exception as exc:
            etb = ExceptionTracebacker()
            self.logger.warning( etb )

    def save( self, xlsx_file ):
        save_allowing_user_reply( self.wb, xlsx_file )
