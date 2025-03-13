# coding: utf-8
"""

    OpenPyXlUtil.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF

"""

"""
from https://msdn.microsoft.com/en-us/library/bb264112(v=vs.85).aspx

EMU - english metrical unit
There are 914400 EMU per inch and 12700 EMU in a point.
"""
LINE_WIDTH      = int( 12700 * 1.5 )

def save_allowing_user_reply( wb, xlsx_file, parent=None ):
    ok = False

    while not ok:
        try:
            wb.save( xlsx_file )
        except Exception as ex:
            if str( ex ).find( 'Permission' ) >= 0:
                import OurMessageBox        as MessageBox
                oc = MessageBox.askokcancel(
                        'Permission Error Retry',
                        "'%s' is being used by another application.\nClose it and press 'OK' if you wish retry." % xlsx_file,
                        parent=parent )
                if oc:
                    continue
        ok = True
