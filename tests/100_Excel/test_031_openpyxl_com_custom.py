import os
import openpyxl
from openpyxl.chart import LineChart, Reference

# --- 追加ここから ---
from openpyxl.packaging.extended import ExtendedProperties

class MyExtendedProperties(ExtendedProperties):
    def __init__(self, orig=None):
        assert orig is not None, "orig must be provided"
        self.__dict__ = orig.__dict__.copy()
        self.Application = "Microsoft Excel"
        self.AppVersion = "16.0300"  # 必要に応じて調整
# --- 追加ここまで ---

# 1. openpyxl で Excel ファイル作成
wb = openpyxl.Workbook()
ws = wb.active
for i in range(1, 11):
    ws.append([i, i * 2])

chart = LineChart()
data = Reference(ws, min_col=2, min_row=1, max_row=10)
chart.add_data(data, titles_from_data=False)
ws.add_chart(chart, "E5")

# --- 追加ここから ---
# Application名を上書き
print("properties:", wb.properties)
wb.properties = MyExtendedProperties(wb.properties)
# --- 追加ここまで ---

wb.save("test_chart.xlsx")

# 2. pywin32 で Excel を開く
import win32com.client

xlApp = win32com.client.Dispatch("Excel.Application")
xlApp.Visible = False
this_dir = os.path.dirname(os.path.abspath(__file__))
book_path = os.path.join(this_dir, "test_chart.xlsx")
xlBook = xlApp.Workbooks.Open(book_path)
xlSheet = xlBook.Worksheets(1)

# 3. グラフオブジェクト取得
chart_obj = xlSheet.ChartObjects(1)  # 1始まり
chart = chart_obj.Chart

# 4. y軸取得
xlValue = 2  # Excel 定数
y_axis = chart.Axes(xlValue)

# 5. TickLabels の存在確認
try:
    tick_labels = y_axis.TickLabels
    print("TickLabels:", tick_labels)
    # 例: 書式設定を試す
    tick_labels.NumberFormatLocal = "0.0E+00"
    print("NumberFormatLocal 設定成功")
except Exception as e:
    print("TickLabels アクセス時に例外:", e)

# 6. 後始末
xlBook.Close(SaveChanges=False)
xlApp.Quit()