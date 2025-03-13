"""
    DefaultFont.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

from openpyxl.styles import DEFAULT_FONT
from molass_legacy._MOLASS.SerialSettings import get_setting

def set_default_font(name=None):
    if name is None:
        name = get_setting("report_default_font")

    if name is not None:
        DEFAULT_FONT.name = name

def set_default_bookfont(wb, name=None):
    # this is currently not used

    if name is None:
        name = get_setting("report_default_font")

    if name is not None:
        wb._named_styles['Normal'].font.name = name

def set_default_bookfont_com(wb, name=None):
    if name is None:
        name = get_setting("report_default_font")

    if name is not None:
        wb.Styles("Normal").Font.Name = name
