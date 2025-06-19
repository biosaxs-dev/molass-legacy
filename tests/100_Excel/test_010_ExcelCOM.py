"""

"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from molass_legacy.KekLib.ExcelCOM import CoInitialize, CoUninitialize, ExcelComClient

def test_excel_com():
    CoInitialize()
    xlValue = 2
    save = True
    book_path = os.path.join( os.path.dirname(__file__), 'test_excel_com.xlsx' )
    exce_client = ExcelComClient(visible=True)
    ws = exce_client.openWorksheet( book_path + '(TestSheet)' )
    line1 = ws.draw_line( ( 100, 50 ), ( 100, 210 ) )
    line2 = ws.draw_line( ( 150, 50 ), ( 150, 210 ) )
    brace = ws.draw_right_brace( ( 110, 220 ), 20, 50, 90 )

    for i, shape in enumerate( [ line1, line2, brace ] ):
        shape.Select()
        line_ = exce_client.selection().ShapeRange.Line
        line_.ForeColor.RGB =   0x50B000    # RGB(0, 176, 80)
        line_.Weight = 2 if i < 2 else 1

    ws.draw_textbox( "テスト文字列", ( 110, 250 ), 100, 20, visible=False )
    ws.freeze_panes( "C2" )

    if False:
        # assert ws.get_num_charts() == 1

        chart = ws.get_chart( 0 )
        print( 'chart=', chart )
        chart.Axes(xlValue).MaximumScale = 70
        area = chart.ChartArea
        X, Y = area.Left, area.Top
        point = ws.get_chart_data_point( chart, 0, 4 )
        x, y = point.Left, point.Top
        ws.draw_oval( (X+x, Y+y), 10, 10, color=0x0000FF )

    if save:
        ws.workbook.Save()
        ws.workbook.Close()     # necessary for other processes to access

    exce_client.quit()
    CoUninitialize()