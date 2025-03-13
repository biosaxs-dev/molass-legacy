# coding: utf-8
"""

    ExcelLogChart.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import numpy    as np
from ExcelCOM   import xlValue, xlCategory, xlNone, CellAddr, msoFalse
from Unicode    import to_superscripts

msoChartFieldRange  = 7
xlLabelPositionLeft = -4131

class ExcelLogChart:
    def __init__( self, chart, worksheet ):
        self.chart  = chart
        self.x_axis = chart.Axes( xlCategory )
        self.y_axis = chart.Axes( xlValue )
        self.ws     = worksheet

    def super_ticklabel( self, tick ):
        return '10' + to_superscripts( '%d' % tick )

    def plain_ticklabel( self, tick ):
        return '10^%d' % tick

    def change_y_labels( self, superscripts=False ):
        # self.x_axis.TickLabelPosition = xlNone
        self.y_axis.TickLabelPosition = xlNone

        print( 'MajorUnit=', self.y_axis.MajorUnit )
        print( 'MinimumScale=', self.y_axis.MinimumScale )
        print( 'MaximumScale=', self.y_axis.MaximumScale )

        minscale = int( np.floor( np.log10( self.y_axis.MinimumScale) ) )
        maxscale = int( np.ceil ( np.log10( self.y_axis.MaximumScale) ) )

        y_ticks  = np.arange( minscale, maxscale+1 )
        label_x = [ 0.0 ] * len(y_ticks)
        label_y = [ np.power( 10.0, float(tick) ) for tick in y_ticks ]
        if superscripts:
            ticklabel = self.super_ticklabel
        else:
            ticklabel = self.plain_ticklabel
        label_t = [ ticklabel(tick) for tick in y_ticks ]
        # print( 'label_x=', label_x )
        # print( 'label_y=', label_y )
        # print( 'label_t=', label_t )

        usedrange   = self.ws.UsedRange
        work_row    = usedrange.Row + usedrange.Rows.Count + 5
        work_col    = usedrange.Column + usedrange.Columns.Count
        work_col1   = work_col+1
        work_col2   = work_col+2
        work_col3   = work_col+3

        print( 'work_row=', work_row )

        for i in range( len(y_ticks) ):
            self.ws.Cells( work_row + i, work_col1 ).Value = label_x[i]
            self.ws.Cells( work_row + i, work_col2 ).Value = label_y[i]
            self.ws.Cells( work_row + i, work_col3 ).Value = label_t[i]

        series = self.chart.SeriesCollection().NewSeries()
        minrow = work_row
        maxrow = work_row + len(y_ticks)
        series.Name = ""
        series.XValues  = self.reference( (minrow,work_col1), (maxrow,work_col1) )
        series.Values   = self.reference( (minrow,work_col2), (maxrow,work_col2) )
        series.Format.Line.Visible = msoFalse
        series.ApplyDataLabels()
        datalabels = series.DataLabels()
        datalabels.ShowRange = True
        datalabels.ShowValue = False
        labeldata_ref = self.reference( (minrow,work_col3), (maxrow,work_col3) )
        datalabels.Format.TextFrame2.TextRange.InsertChartField( msoChartFieldRange, labeldata_ref, 0 )
        datalabels.Position = xlLabelPositionLeft

    def get_plotarea( self ):
        plotarea = self.chart.PlotArea
        return ( plotarea.Left, plotarea.Top, plotarea.Width, plotarea.Height )

    def change_plotarea( self, left=None, top=None, width=None, height=None ):
        plotarea = self.chart.PlotArea
        if left is not None:
            plotarea.Left = left
        if top is not None:
            plotarea.Top = top
        if width is not None:
            plotarea.Width = width
        if height is not None:
            plotarea.Height = height

    def get_yaxis_title_area( self ):
        yaxis_title = self.chart.Axes(xlValue).AxisTitle
        return ( yaxis_title.Left, yaxis_title.Top, yaxis_title.Width, yaxis_title.Hieght )

    def change_yaxis_title_area( self, left=None, top=None, width=None, height=None ):
        yaxis_title = self.chart.Axes(xlValue).AxisTitle
        if left is not None:
            yaxis_title.Left = left
        if top is not None:
            yaxis_title.Top = top
        if width is not None:
            yaxis_title.Width = width
        if height is not None:
            yaxis_title.Height = height

    def reference( self, cell_f, cell_t ):
        ref = "='%s'!%s:%s" % ( self.ws.Name, CellAddr(cell_f), CellAddr(cell_t) )
        # print( 'ref=', ref )
        return ref
