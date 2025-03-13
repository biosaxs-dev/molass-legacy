"""
    GuinierExcelFormatter.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.NumpyUtils import get_safe_approximate_max
from molass_legacy.KekLib.ExcelCOM import xlValue, xlLine, compute_axis_max_value
from .DefaultFont import set_default_bookfont_com

NUM_CHARTS = 5
CHART_XFIX_300  = 14.5
CHART_AREA_LEFT = 816.0     # chart_area.Left of the basis environment
NUM_POINTS = 160            # num_points of the basis environment
ADJUST_PLOT_LEFT = False
TEXT_Y_ADJUST_UNIT = 20
TEXT_HALF_WIDTH = 120

def chart_xfix(xoffset, n, adjust_scale):
    return xoffset + CHART_XFIX_300 * n/300 * adjust_scale

class GuinierReportArgs:
    def __init__(self, title, book_obj, book_file, ranges):
        self.title = title
        self.num_points = book_obj.num_points
        self.j0 = book_obj.j0
        self.Izc_array = book_obj.Izc_array
        self.include_end_consistency = book_obj.include_end_consistency
        self.iz_array = book_obj.iz_array
        self.rg_array = book_obj.rg_array
        self.book_file = book_file
        self.ranges = ranges

def get_analysis_area(args, ole_ws, interval, logger, debug=False):

    if ADJUST_PLOT_LEFT:
        left_poss = []
        for k in range(NUM_CHARTS):
            chart = ole_ws.get_chart(k)
            left_poss.append(chart.PlotArea.Left)
        average_plot_left = np.average(left_poss, axis=0)

    top_chart = ole_ws.get_chart( 0 )
    top_area  = top_chart.ChartArea
    conc_chart = ole_ws.get_chart( 0 )
    chart_area = conc_chart.ChartArea
    plot_area = conc_chart.PlotArea

    adjust_scale = chart_area.Left/CHART_AREA_LEFT * NUM_POINTS/args.num_points
    # xoffset = plot_area.Width/args.num_points * 0.5
    xoffset = 0

    if debug:
        logger.info("get_analysis_area: j0=%d, num_points=%d, inverval=%s", args.j0, args.num_points, str(interval))
        logger.info("get_analysis_area: (average_plot_left, chart_area.Left, plot_area.Left) = %s", str((average_plot_left, chart_area.Left, plot_area.Left)))
        logger.info("get_analysis_area: (chart_area.Width, plot_area.Width) = %s", str((chart_area.Width, plot_area.Width)))

    x = []
    for j in interval:
        point = ole_ws.get_chart_data_point( conc_chart, 0, j )
        xpos = chart_xfix(xoffset, j, adjust_scale)
        x.append(point.Left + xpos)
        if debug:
            logger.info("%s (point.Left, point.Width, xpos) = %s", str([j]), str((point.Left, point.Width, xpos)))

    bottom_chart = ole_ws.get_chart( ole_ws.get_num_charts() - 1 )
    bottom_area = bottom_chart.ChartArea

    if debug:
        logger.info("(bottom_area.Top, bottom_area.Height) = %s", str((bottom_area.Top, bottom_area.Height)))

    if ADJUST_PLOT_LEFT:
        plot_area_adjust = average_plot_left - plot_area.Left
    else:
        plot_area_adjust = 0
    chart_left = chart_area.Left + plot_area_adjust
    x_left  = chart_left + x[0]
    x_right = chart_left + x[1]
    y_top   =    0
    y_btm   = top_area.Top + bottom_area.Top + bottom_area.Height

    return [ x_left, x_right, y_top, y_btm ]

def add_guinier_annonations(excel_client, args, logger, debug=False):
    # print( 'add_annonations: ranges=', ranges )

    book_file = args.book_file
    ranges = args.ranges

    book_path = os.path.abspath( book_file )

    ws = excel_client.openWorksheet( book_path + '(' + args.title  + ')' )
    set_default_bookfont_com(ws.workbook)

    izc_max_axis = None
    range_index_list = []
    last_x_center = None
    for k, interval in enumerate(ranges):
        x_left, x_right, y_top, y_btm = get_analysis_area(args, ws, interval, logger, debug=debug)

        if debug:
            logger.info("add_guinier_annonations: %s interval=%s", str([k]), str(interval))
            logger.info("(x_left, x_right, y_top, y_btm) = %s", str((x_left, x_right, y_top, y_btm)))

        range_index_list += list( range( interval[0], interval[1]+1 ) )

        x_center = ( x_left + x_right ) / 2

        if last_x_center is None or last_x_center + TEXT_HALF_WIDTH < x_center - TEXT_HALF_WIDTH:
            y_btm_adjust = 0
        else:
            y_btm_adjust = TEXT_Y_ADJUST_UNIT

        last_x_center = x_center

        y_btm += y_btm_adjust
        line1 = ws.draw_line( ( x_left,  y_top ), ( x_left,  y_btm ) )
        line2 = ws.draw_line( ( x_right, y_top ), ( x_right, y_btm ) )
        b_width = x_right - x_left
        x_ = x_left + b_width//2 - 10
        y_ = y_btm - b_width//2 + 12
        try:
            brace = ws.draw_right_brace( ( x_, y_ ), 20, b_width, 90 )
        except Exception as exc:
            logger.warning( str(exc) )
            continue

        for i, shape in enumerate( [ line1, line2, brace ] ):
            shape.Select()
            line_ = excel_client.selection().ShapeRange.Line
            line_.ForeColor.RGB =   0x50B000    # RGB(0, 176, 80)
            line_.Weight = 1

        interval_repo = [args.j0 + r for r in interval]
        ws.draw_textbox( "Concentration Dependency Analysis Interval %s" % str( interval_repo ), ( x_center - TEXT_HALF_WIDTH, y_btm + 30 ), 300, 20, visible=False )

        slice_ = slice( interval[0], interval[1]+1 )
        izc_array = args.Izc_array[slice_]
        # Be careful for inf values which make Excel hang!
        try:
            izc_max_axis_ =  np.percentile( izc_array[ np.isfinite(izc_array) ], 95 ) * 2
            if izc_max_axis is None or izc_max_axis_ > izc_max_axis:
                izc_max_axis = izc_max_axis_
        except:
            # i.e., in case of no finite values?
            # occured in 20151119/Kosugi3a_Backsub
            pass

    if izc_max_axis is None:
        # i.e., in case of no range
        izc_max_axis =  np.percentile( args.Izc_array[ np.isfinite(args.Izc_array) ], 95 ) * 2

    if len(range_index_list) > 0:
        range_index = np.array( range_index_list )
    else:
        range_index = slice(0,None)

    # modify QualityChar
    # print( 'modify QualityChar' )
    quality_chart = ws.get_chart( 1 )
    seriesno = 6 if args.include_end_consistency else 5
    series = quality_chart.FullSeriesCollection(seriesno)   # atsas quality
    series.ChartType = xlLine
    series.Format.Line.Weight = 1.5
    quality_chart.Axes(xlValue).MaximumScale = 1

    # modify IzChart
    # print( 'modify IzChart' )
    iz_max = get_safe_approximate_max( args.iz_array[range_index] )
    # iz_max_axis = float( '%.2g' % ( iz_max * 1.2 ) )
    iz_max_axis = compute_axis_max_value( iz_max * 1.2 )
    # print( 'iz_max=', iz_max, 'iz_max_axis=', iz_max_axis )

    iz_chart = ws.get_chart( 2 )
    iz_chart.Axes(xlValue).MaximumScale = iz_max_axis

    # modify z_devided_by_C_Chart
    # print( 'izc_max_axis=', izc_max_axis )
    izc_chart = ws.get_chart( 3 )
    izc_chart.Axes(xlValue).MinimumScale = 0
    izc_chart.Axes(xlValue).MaximumScale = izc_max_axis

    # modify RgChart
    # print( 'modify RgChart' )
    rg_primary_max = get_safe_approximate_max( args.rg_array[range_index] )

    # print("args.rg_array=", args.rg_array)
    args.rg_array[np.isnan(args.rg_array)] = 0  # temporary fix

    hc, hb = np.histogram( args.rg_array, bins=10 )
    # hb_ = [ ( hb[i]+hb[i+1] )/2  for i in range(len(hc)) ]
    rg_hist = sorted( zip( hc, hb[1:] ), reverse=True )
    rg_hist_max = np.max( rg_hist[:3], axis=0 )[1]

    rg_max = min( rg_primary_max * 2.0, max( rg_primary_max * 1.3, rg_hist_max ) )
    rg_max_axis = compute_axis_max_value( rg_max )
    # print( 'rg_primary_max=', rg_primary_max, 'rg_hist_max=', rg_hist_max, 'rg_max_axis=', rg_max_axis )

    rg_chart = ws.get_chart( 4 )
    rg_chart.Axes(xlValue).MaximumScale = rg_max_axis

    # Zoom and Scroll
    # print( 'Zoom and Scroll' )
    ws.zoom( 70 )
    scroll_column = 17 if args.include_end_consistency else 20
    ws.freeze_panes( "D2", scroll_column=scroll_column )

    ws.workbook.Save()
    ws.workbook.Close()     # required in the more_multicore mode
