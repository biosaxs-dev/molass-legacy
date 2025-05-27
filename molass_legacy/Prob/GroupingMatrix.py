"""
    GroupingMatrix.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from TkMiniTable import TkMiniTable
import tkinter.font as font

SELECTED_BG = 'green'

def num_to_char(i):
    if i is None:
        # occurred in tick formatter
        return ''
    else:
        return hex(i)[2]

def get_font_size(n):
    m = max(0, n-10)
    return 10 - round(m*4/6)

class GroupingMatrix(Tk.Frame):
    def __init__(self, parent, nrows=10, ncols=10, read_only=False):
        Tk.Frame.__init__(self, parent)

        table_frame = Tk.Frame(self)
        table_frame.pack()

        table = TkMiniTable(table_frame,
            columns = ['_'] + ['C%s' % num_to_char(j) for j in range(ncols)],
            default_colwidth=2,
            font = font.Font(self, family="Ariel", size=get_font_size(nrows)),
            cell_widget_class=Tk.Label,
            read_only=read_only,
            on_cell_click=None if read_only else self.on_cell_click,
            on_cell_click_only=True,
            )

        table.heading('_', text='G*' )
        table.column ('_', justify=Tk.CENTER, width=2)
        for j in range(ncols):
            Cn = 'C%s' % num_to_char(j)
            table.heading(Cn, text=Cn)

        table.pack()
        for i in range( 0, nrows ):
            Gn = 'G%s' % num_to_char(i)
            table.insert_row( [Gn] + [''] * ncols )

        self.table = table
        self.default_bg = table.cells_array[1,1].widget.cget('bg')
        self.set_default_grouping(nrows, ncols)

    def set_default_grouping(self, nrows, ncols):
        self.select_matrix = np.zeros((nrows, ncols), dtype=int)
        for i in range(nrows):
            self.select_matrix[i,i] = 1
            cell = self.table.cells_array[i+1, i+1]
            cell.widget.config(bg=SELECTED_BG)

    def on_cell_click(self, cell):
        # print('cell', cell)
        row = cell.row - 1
        col = cell.col - 1
        if row < 0 or col < 0:
            return

        selected = self.select_matrix[row, col]
        bg = self.default_bg if selected else SELECTED_BG
        cell.widget.config(bg=bg)

        self.select_matrix[row, col] = 1 - selected
        selected_col = self.select_matrix[:, col]
        if np.sum(selected_col) > 1:
            for i in range(len(selected_col)):
                if i == row:
                    continue
                cell = self.table.cells_array[i+1, col+1]
                cell.widget.config(bg=self.default_bg)
                selected_col[i] = 0

    def click(self, i, j):
        cell = self.table.cells_array[i+1,j+1]
        self.on_cell_click(cell)

    def reset(self):
        nrows = self.select_matrix.shape[0]
        for i in range(nrows):
            if self.select_matrix[i, i]:
                pass
            else:
                self.click(i, i)

    def get_matrix(self):
        return self.select_matrix
