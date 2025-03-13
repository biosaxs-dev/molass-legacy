# -*- coding: utf-8 -*-
"""

    ファイル名：   CuntomizedPandasTable.py

    処理内容：

       Pilatus 画像データ情報の一覧

"""
from __future__ import division, print_function, unicode_literals

import sys
import os
# import copy

if sys.version_info > (3,):
    import tkinter as Tk
    import tkinter.ttk as ttk
else:
    import Tkinter as Tk
    import ttk
    import tkFont

import pandas as pd
from pandastable import Table, TableModel

label_bg_color = 'gray25'

class PandasTable( Tk.Frame ):
    def __init__( self, parent, columns=[ [ 'col0', 100 ], [ 'col1', 100 ] , [ 'col2', 100 ] ], width=None, rowheight=16, height=None ):
        Tk.Frame.__init__( self, parent )

        # self.columns = copy.deepcopy( columns )
        self.columns = []
        for col in columns:
            self.columns.append( col[0] )

        self.table = None

    def destroy( self ):
        # Don't use this method
        if self.table:
            # self.table.destroy()
            pass

        self.table = None
        Tk.Frame.destroy( self  )

    def dummy( self, event ):
        pass

    def selectAll( self ):
        self.table.selectAll()

    def selectRows( self, begin, end ):
        self.table.setSelectedCells( begin, end, 0, len( self.columns ) )
        self.table.drawMultipleCells()

    def selectionClear( self ):
        self.table.selectNone()

    def import_array( self, data ):

        if len(data) > 0:
            columns_ = self.columns
        else:
            columns_ = []

        df = pd.DataFrame( data, columns=columns_ )

        if self.table == None:
            self.table = Table( self, dataframe=df )

            self.table.rowheaderwidth=30    # does not work. why?
            self.table.thefont = ( 'Microsoft Sans Serif', 9 )
            # self.table.thefont = ( 'Arial', 10 )
            # self.table.thefont = ( 'Arial Unicode MS', 10 )
            
            # self.table.thefont = ( 'ＭＳ Ｐゴシック', 10 )

            self.table.show()

            self.table.bind( "<Button-3>", self.dummy )                     # to remove popup-menu binding
            self.table.tablecolheader.bind( "<Button-3>", self.dummy )      # to remove popup-menu binding
            self.table.rowheader.bind( "<Button-3>", self.dummy )           # to remove popup-menu binding
            self.table.rowindexheader.bind( "<Button-3>", self.dummy )      # to remove popup-menu binding

        else:

            """
                borrowed from Table.clearTable
            """
            model = TableModel( dataframe=df )
            self.table.updateModel( model )
            self.table.redraw()
            self.table.show()

    def bindActionMenu( self, event, popup, corner=True ):
        self.table.bind( event, popup )
        self.table.rowheader.bind( event, popup )
        if corner:
            self.table.rowindexheader.bind( event, popup ) 

    def selectedRows( self ):
        # print( 'selectedRows: ', self.table.multiplerowlist )
        return self.table.multiplerowlist

    def selectedCell( self ):
        row = self.table.getSelectedRow()
        col = self.table.getSelectedColumn()
        print( [row, col] )
        return [row, col]
