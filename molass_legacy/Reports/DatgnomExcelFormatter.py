"""

    DatgnomExcelFormatter.py

    SummaryBook.py

    Copyright (c) 2018-2023, SAXS Team, KEK-PF

"""
from ExcelCOM import xlCategory, xlValue, xlLow, msoThemeColorText1
from ExcelLogChart import ExcelLogChart
from .DefaultFont import set_default_bookfont_com

class DatgnomResultArgs:
    def __init__(self, title, book_file):
        self.title = title
        self.book_file = book_file

def add_datgnom_format_setting(excel_client_, args, logger):
    # logger is not yet used

    ws = excel_client_.openWorksheet( args.book_file + '(' + args.title  + ')' )
    set_default_bookfont_com(ws.workbook)

    chart = ws.get_chart( 0 )
    # plot area frame lines
    line = chart.PlotArea.Format.Line
    line.Visible    = True
    line.ForeColor.ObjectThemeColor = msoThemeColorText1

    # grid lines
    chart.Axes(xlValue).MajorGridlines.Format.Line.Visible = False
    chart.Axes(xlCategory).MajorGridlines.Format.Line.Visible = False

    # text font size
    chart.ChartArea.Format.TextFrame2.TextRange.Font.Size = 16

    y_axis = chart.Axes(xlValue)
    # y_axis.TickLabels.NumberFormatLocal = "#,##0.000_)"
    # y_axis.TickLabels.NumberFormatLocal = "0.0E+00"
    y_axis.MajorUnit    = 10
    chart.Axes(xlCategory).TickLabelPosition = xlLow

    log_chart = ExcelLogChart( chart, ws.worksheet )
    log_chart.change_y_labels( superscripts=True )
    L, T, W, H = log_chart.get_plotarea()
    # print( 'L, T, W, H=', L, T, W, H )
    log_chart.change_plotarea( left=L+30, width=W-50 )
    log_chart.change_yaxis_title_area( left=25 )

    chart = ws.get_chart( 1 )
    # plot area frame lines
    line = chart.PlotArea.Format.Line
    line.Visible    = True
    line.ForeColor.ObjectThemeColor = msoThemeColorText1

    # grid lines
    chart.Axes(xlValue).MajorGridlines.Format.Line.Visible = False
    chart.Axes(xlCategory).MajorGridlines.Format.Line.Visible = False

    # text font size
    chart.ChartArea.Format.TextFrame2.TextRange.Font.Size = 16

    y_axis = chart.Axes(xlValue)
    # y_axis.TickLabels.NumberFormatLocal = "#,##0.000_)"
    y_axis.TickLabels.NumberFormatLocal = "0.0E+00"

    ws.zoom( 70 )
    # scroll_column=16
    ws.freeze_panes( "3:3", scroll_column=None )

    ws.workbook.Save()
    ws.workbook.Close()
