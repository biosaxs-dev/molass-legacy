"""

    ZeroExExcelFormatter.py

    Copyright (c) 2017-2023, SAXS Team, KEK-PF

"""
from ExcelCOM import (
        xlCategory, xlValue, xlLine,
        msoThemeColorText1, msoThemeColorAccent1,
        xlUpward, xlLow,
        msoTrue, xlXYScatter,
        RGB,
        xlMaximum )
from ExcelLogChart import ExcelLogChart
from HtmlColorNames import *
from .DefaultFont import set_default_bookfont_com

ADD_ERROR_BARS = True
ATSAS_COLOR = darkgray

def add_boundary_line( ws, chart, info ):
    j, text, yoffset = info
    if j is None:
        return

    # print( 'add_boundary_line: j=', j, text )

    point = ws.get_chart_data_point( chart, 0, j )
    # print( [j], 'point.Left=', point.Left, ', Point.Width=', point.Width )

    chart_area = chart.ChartArea
    X, Y, H = chart_area.Left, chart_area.Top, chart_area.Height
    x, y = point.Left, point.Top
    plot_area = chart.PlotArea
    pt, ph  = plot_area.Top, plot_area.Height

    # print( 'chart_area.Top=', chart_area.Top, 'chart_area.Height=', chart_area.Height )
    # print( 'plot_area.Top=', plot_area.Top, 'plot_area.Height=', plot_area.Height )

    line_bottom = Y+pt+ph+yoffset
    line = ws.draw_line( ( X+x, Y+pt+30 ), ( X+x, line_bottom ) )
    textbox = ws.draw_textbox( text, ( X+x-50, line_bottom ), 250, 24, visible=False )
    textbox.TextFrame2.TextRange.Font.Size = 16

def add_format_setting_to_extrapolated_chart( ws, chart, logscale=False, boundaries=[] ):
    
    # plot area frame lines
    line = chart.PlotArea.Format.Line
    line.Visible    = True
    line.ForeColor.ObjectThemeColor = msoThemeColorText1

    # grid lines
    chart.Axes(xlValue).MajorGridlines.Format.Line.Visible = False
    chart.Axes(xlCategory).MajorGridlines.Format.Line.Visible = False

    # text font size
    chart.ChartArea.Format.TextFrame2.TextRange.Font.Size = 16

    if logscale:
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

    for info in boundaries:
        add_boundary_line( ws, chart, info )

class ZeroExReportArgs:
    def __init__(self, title, book_obj, book_file, axis_direction_desc):
        self.title = title
        self.ud = book_obj.ud
        self.indeces = book_obj.indeces
        self.c_max = book_obj.c_max
        self.overlap_from_max = book_obj.overlap_from_max
        self.guinier_boundary = book_obj.guinier_boundary 
        self.boundary_j = book_obj.boundary_j
        self.lrf_boundary_j = book_obj.lrf_boundary_j
        self.num_rows = book_obj.num_rows
        self.need_bq = book_obj.need_bq
        self.book_file = book_file
        self.axis_direction_desc = axis_direction_desc

def add_result_format_setting(excel_client_, args, logger):
    # logger is not yet used

    book_file = args.book_file
    axis_direction_desc = args.axis_direction_desc

    ws = excel_client_.openWorksheet( book_file + '(' + args.title  + ')' )
    set_default_bookfont_com(ws.workbook)

    colors = [ green, orange, ATSAS_COLOR ]

    for i in range(3):
        p_chart = ws.get_chart( i )

        if i == 0:
            for j in [1, 2]:
                # TODO: move this to openpyxl
                series = p_chart.FullSeriesCollection(j)
                series.MarkerSize = 5
            if args.ud == 1 and axis_direction_desc == 1:
                axis = p_chart.Axes(xlCategory)
                axis.ReversePlotOrder = True
                axis.Crosses = xlMaximum

        # plot area frame lines
        line = p_chart.PlotArea.Format.Line
        line.Visible    = True
        line.ForeColor.ObjectThemeColor = msoThemeColorText1

        # grid lines
        p_chart.Axes(xlValue).MajorGridlines.Format.Line.Visible = False

        # text font size
        p_chart.ChartArea.Format.TextFrame2.TextRange.Font.Size = 12

        category = p_chart.Axes(xlCategory)
        tick_labels = category.TickLabels

        if i == 0:
            max_no = args.indeces[-1]
            if max_no >= 100:
                tick_labels.Orientation     = xlUpward
        else:
            tick_labels.NumberFormatLocal   = "#,##0.000_)"
            tick_labels.Orientation         = xlUpward
            if ADD_ERROR_BARS:
                for j in range(2):
                    series = p_chart.SeriesCollection(j+1)
                    series.ErrorBars.Format.Line.ForeColor.ObjectThemeColor = msoThemeColorAccent1

        # TODO: move this to openpyxl
        if i == 0:
            j_jist = [ 1, 2 ]
        else:
            j_jist = [ 1, 2, 3 ]
            p_chart = ws.get_chart( i )
            p_chart.ChartType = xlXYScatter
            axis = p_chart.Axes(xlCategory)
            axis.MaximumScale = args.c_max * 1.05
            axis.MinimumScale = 0

            if args.ud == 1 and axis_direction_desc == 0:
                axis.ReversePlotOrder = True
                axis.Crosses = xlMaximum

        for j in j_jist:
            series = p_chart.FullSeriesCollection(j)
            line = series.Format.Line
            line.Visible = msoTrue
            line.ForeColor.RGB = RGB( colors[ j-1 ] )   # necessary to remove the border. better way?
            line.Weight = 1
            # series.MarkerStyle = -4142  # no marker
            series.MarkerSize = 5
            p = 1 if args.ud == 0 else series.Points().Count
            point = series.Points(p)
            point.MarkerSize = 10

    atsas_j = args.overlap_from_max
    print( "boundary_j=", args.boundary_j )

    boundaries = []

    if args.lrf_boundary_j is None:
        if atsas_j is None or atsas_j > args.num_rows - 1:
            print( 'WARNING: ATSAS gave an exceeding boundary ', atsas_j )
            # TODO: annotate in the chart
        else:
            boundaries.append( [ atsas_j, "ATSAS Boundary", -160 ] )
    else:
        boundaries.append( [ args.lrf_boundary_j , "Rank Boundary", -80 ] )

    boundaries.append( [ args.guinier_boundary , "Guinier Boundary", -120 ] )
    boundaries.append( [ args.boundary_j , "Regression Boundary", -80 ] )

    ab_chart = ws.get_chart( 3 )
    add_format_setting_to_extrapolated_chart( ws, ab_chart, logscale=True, boundaries=boundaries )

    if args.need_bq:
        ab_chart = ws.get_chart( 4 )
        add_format_setting_to_extrapolated_chart( ws, ab_chart  )

    ws.zoom( 70 )
    scroll_column=16
    ws.freeze_panes( "B2", scroll_column=scroll_column )

    ws.workbook.Save()
    ws.workbook.Close()     # required in the more_multicore mode

class ZeroExOverlayArgs:
    def __init__(self, title, book_obj, book_file, boundary_indeces):
        self.title = title
        self.need_bq = book_obj.need_bq
        self.book_file = book_file
        self.boundary_indeces = boundary_indeces

def add_overlay_format_setting( excel_client, args, logger):
    # logger is not yet used

    need_bq = args.need_bq
    book_file = args.book_file
    boundary_indeces = args.boundary_indeces

    ws = excel_client.openWorksheet( book_file + '(' + args.title  + ')' )
    set_default_bookfont_com(ws.workbook)

    boundaries = []
    yoffset = -120
    for j, b in enumerate( boundary_indeces ):
        ad = 'Asc' if j == 0 else 'Desc'
        boundaries.append( [ b , "Regression Boundary (%s)" % ad, yoffset ] )
        yoffset += 40

    aq_chart = ws.get_chart( 0 )
    add_format_setting_to_extrapolated_chart( ws, aq_chart, logscale=True, boundaries=boundaries )

    if need_bq:
        bq_chart = ws.get_chart( 1 )
        add_format_setting_to_extrapolated_chart( ws, bq_chart  )

    ws.zoom( 70 )
    ws.freeze_panes( "B2", scroll_column=6 )

    ws.workbook.Save()
    ws.workbook.Close()     # required in the more_multicore mode
