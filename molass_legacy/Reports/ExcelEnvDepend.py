"""

    ExcelEnvDepend.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
import os
import numpy as np
from time import sleep
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference
from molass_legacy._MOLASS.SerialSettings import get_setting

CHART_WIDTH     = 54

def add_a_chart( ws, name ):

    num_points = 160
    num_cols = 2

    title_row = []

    data = np.random.uniform(0, 1, (num_points, num_cols))

    for k in range(num_points):
        row = data[k,:]
        ws.append(list(row))

    c1 = LineChart()
    c1.title = name
    c1.style = 13
    c1.y_axis.title = 'Values'
    c1.x_axis.title = 'Eno'
    c1.width = CHART_WIDTH

    data = Reference(ws, min_col=1, min_row=1, max_col=num_cols, max_row=num_points)
    c1.add_data(data, titles_from_data=False)
    ws.add_chart(c1, "R10")

xlTypePDF = 0

def get_chart_width_adjust_ratio(excel_client):
    temp_folder = get_setting('temp_folder')
    book_path = os.path.join(temp_folder, "tempbook.xlsx")

    wb = Workbook()
    ws = wb.active
    add_a_chart(ws, "Test Chart")
    wb.save(book_path)
    wb.close()

    app = excel_client.excel

    ole_ws = excel_client.openWorksheet(book_path + "(Sheet)")
    top_chart = ole_ws.get_chart(0)
    top_area  = top_chart.ChartArea
    plot_area  = top_chart.PlotArea

    c_left = top_area.Left
    c_width = top_area.Width
    c_top = top_area.Top
    c_height = top_area.Height

    x_left = c_left
    x_right = c_left + c_width
    y_top = c_top
    y_btm = c_top + c_height

    line1 = ole_ws.draw_line( ( x_left,  y_top ), ( x_left,  y_btm ) )
    line2 = ole_ws.draw_line( ( x_right, y_top ), ( x_right, y_btm ) )

    for shape in line1, line2:
        shape.Select()
        line_ = excel_client.selection().ShapeRange.Line
        line_.ForeColor.RGB = 0x00FF00

    x_left = c_left + plot_area.Left
    x_right = x_left + plot_area.Width
    y_top = c_top
    y_btm = c_top + c_height

    line3 = ole_ws.draw_line( ( x_left,  y_top ), ( x_left,  y_btm ) )
    line4 = ole_ws.draw_line( ( x_right, y_top ), ( x_right, y_btm ) )

    for shape in line3, line4:
        shape.Select()
        line_ = excel_client.selection().ShapeRange.Line
        line_.ForeColor.RGB = 0xFF0000

    if False:

        app.Cells.Select()
        app.Selection.CopyPicture()
        active = app.Sheets.Add()
        active.Paste()

        # active.Pictures().Paste().Select()
        # active.Shapes.Range("Picture 1").Select()
        if True:
            img_path = os.path.join(temp_folder, "temp.jpg")
            active.Shapes(1).SaveAsPicture(img_path)
        else:
            pdf_path = os.path.join(temp_folder, "temp.pdf")
            active.ExportAsFixedFormat(xlTypePDF, pdf_path)

        # active.Pictures().SaveAsPicture(pic_path)
        # app.Selection.ExportAsFixedFormat(xlTypePDF, "temp.pdf")
        # print(active.Shapes.Count)
        # print(active.Shapes.Item(1))

        # active.Shapes.Item(1).SaveAsPicture("temp.jpg")
        # active.Shapes(1).SaveAsPicture("temp.jpg")
        # active.Shapes.Item(1).Chart.Export("temp.jpg")

    workbook = ole_ws.workbook
    workbook.Save()
"""
Sub Macro1()
'
' Macro1 Macro
'
    Cells.Select
    Application.Left = 174.25
    Application.Top = 25.75
    Selection.Copy
    Sheets.Add After:=ActiveSheet
    ActiveSheet.Pictures.Paste.Select
    ActiveSheet.Shapes.Range(Array("Picture 1")).Select
End Sub
"""
