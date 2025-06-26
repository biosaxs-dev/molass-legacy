"""
Test Excel Tick Labels Formatting
"""
# import pytest
import os
from win32com.client import DispatchEx
import pythoncom
from molass_legacy.KekLib.ExcelCOM import xlCategory, xlValue

# @pytest.fixture(scope="function")
def excel_app():
    pythoncom.CoInitialize()
    excel = DispatchEx("Excel.Application")
    excel.Visible = True
    workbook = excel.Workbooks.Add()
    sheet = workbook.Sheets(1)
    # Add some data
    sheet.Range("A1:A5").Value = [(i,) for i in range(1, 6)]
    sheet.Range("B1:B5").Value = [(i*2,) for i in range(1, 6)]
    # Add a chart (4 = xlLine)
    chart_obj = sheet.Shapes.AddChart2(4, 4, 100, 100, 300, 200)
    chart = chart_obj.Chart
    chart.SetSourceData(sheet.Range("A1:B5"))
    # yield chart
    test_ticklabels_number_format(chart)
    this_dir = os.path.dirname(__file__)
    workbook.SaveAs(os.path.join(this_dir, "test_ticklabels.xlsx"))
    workbook.Close(SaveChanges=False)
    excel.Quit()
    pythoncom.CoUninitialize()

def test_ticklabels_number_format(excel_app):
    chart = excel_app
    # X axis TickLabels (1 = xlCategory)
    x_axis = chart.Axes(xlCategory)
    tick_labels = x_axis.TickLabels
    tick_labels.NumberFormat = "0.0"
    assert tick_labels.NumberFormat == "0.0"

    # Y axis TickLabels (2 = xlValue)
    y_axis = chart.Axes(xlValue)
    y_tick_labels = y_axis.TickLabels
    y_tick_labels.NumberFormat = "0.00E+00"
    assert y_tick_labels.NumberFormat == "0.00E+00"

if __name__ == "__main__":
    excel_app()