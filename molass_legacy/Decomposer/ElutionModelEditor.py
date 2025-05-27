# coding: utf-8
"""
    ElutionModelEditor.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import os
import copy
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar, get_color, get_hex_color
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
import tkinter.font as font
from molass_legacy.KekLib.TkSupplements          import set_icon
from TkMiniTable            import TkMiniTable
from molass_legacy.Models.ElutionCurveModels     import EGHA, EMGA
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting

class ElutionModelEditor(Dialog):
    def __init__(self, parent, x, y, opt_recs):
        self.parent = parent
        self.x = x
        self.y = y
        self.opt_recs = opt_recs
        self.num_eltns = len(self.opt_recs)
        Dialog.__init__( self, parent, "Elutio nModel Editor", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame): # overrides parent class method
        set_icon(self)

        canvas_frame = Tk.Frame( body_frame )
        canvas_frame.pack(side=Tk.LEFT)
        editor_frame = Tk.Frame( body_frame )
        editor_frame.pack(side=Tk.LEFT)

        self.fig = plt.figure( figsize=(7, 6) )
        self.mpl_canvas = FigureCanvasTkAgg( self.fig, canvas_frame )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.ax = self.fig.gca()
        self.draw()
        self.fig.tight_layout()

        self.create_table(editor_frame)

    def draw(self):
        ax = self.ax
        x = self.x
        y = self.y
        ax.cla()
        ax.plot(x, y, color='orange', label='data')
        total_y = np.zeros(len(y))
        resid_y = copy.deepcopy(y)
        for k, rec in enumerate(self.opt_recs):
            func = rec[1]
            y_ = func(x)
            ax.plot(x, y_, ':', label='component %d' % (k+1), color=get_color(k), linewidth=3)
            total_y += y_
            resid_y -= y_

        ax.plot(x, total_y, ':', label='model total', color='red', linewidth=3)
        ax.plot(x, resid_y, label='residual', color='gray')
        ax.legend()
        self.mpl_canvas.draw()

    def create_table(self, frame):
        table = TkMiniTable( frame,
            columns = [ '_', 'A', 'B', 'C', 'D' ],
            default_colwidth = 20,
            # font = font.Font(self.parent, family="Ariel", size=16),
            font = ("", 12),
            cell_widget_class=Tk.Entry,
            )

        table.heading( '_', text='No' )
        table.column ( '_', justify=Tk.CENTER, width=5 )
        table.heading( 'A', text='Color' )
        table.heading( 'B', text='Type' )
        table.heading( 'C', text='Definition' )
        table.heading( 'D', text='Hide' )
        table.column ( [1], width=10 )
        table.column ( [2], width=10 )
        table.column ( [3], width=60 )
        table.column ( [4], width= 5 )

        table.insert_row( ( 0, "", "Data", "", "" ) )

        for k, rec in enumerate(self.opt_recs):
            evaluator = rec[1]
            type_ = evaluator.get_model_name()
            def_ = evaluator.get_model_def_expr()
            cells = table.insert_row( ( 1+k, "", type_, def_, "" ) )
            for c in cells[2:]:
                c.widget.config(bg='honeydew')

        table.insert_row( ( self.num_eltns+1, "", "Total", "", "" ) )
        table.insert_row( ( self.num_eltns+2, "", "Residual", "", "" ) )

        table.pack()
