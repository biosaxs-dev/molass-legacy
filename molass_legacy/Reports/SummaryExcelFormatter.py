"""

    SummaryExcelFormatter.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
from ExcelCOM import xlCategory, xlValue, xlLow, msoThemeColorText1
from ExcelLogChart import ExcelLogChart
from .DefaultFont import set_default_bookfont_com
from .CharacterFormatter import to_superscript_font, to_subscript_font, to_italic_font, to_italic_subscipt_font

class SummaryArgs:
    def __init__(self, title, book_file):
        self.title = title
        self.book_file = book_file

def add_summary_format_setting(excel_client, args, logger):
    title = args.title
    book_file = args.book_file
    book_path = "%s(%s)" % (book_file, title)
    logger.info("add_summary_format_setting start for %s", book_path)

    ws = excel_client.openWorksheet(book_path)
    set_default_bookfont_com(ws.workbook)

    for row in [7, 8, 11, 13, 14, 23, 42, 44, 48, 51, 53]:
        cell = ws.worksheet.Cells(row,1)
        to_superscript_font(cell)

    for row in [24]:    # H₂O
        cell = ws.worksheet.Cells(row,2)
        to_subscript_font(cell)

    for row in [23, 35, 42, 44, 45, 46, 47, 48, 51, 52, 54]:
        cell = ws.worksheet.Cells(row,1)
        to_italic_font(cell, ["Q", "I(0)", "P(r)", "Q×", "M"])

    for row in [43, 45, 49, 50]:
        cell = ws.worksheet.Cells(row,1)
        to_italic_subscipt_font(cell, ["Rg", "Dmax"])

    ws.workbook.Save()
    ws.workbook.Close()     # required in the more_multicore mode
