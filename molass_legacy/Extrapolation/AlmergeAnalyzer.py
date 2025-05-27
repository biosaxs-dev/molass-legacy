# coding: utf-8
"""
    AlmergeAnalyzer.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class AlmergeAnalyzerDialog( Dialog ):
    def __init__( self, parent, mc_vector, sd, ranges ):
        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.mc_vector  = mc_vector
        self.sd         = sd
        self.mc_vector  = sd.mc_vector
        self.ranges     = ranges
        self.applied    = False

    def show( self ):
        title   = "Zero Concentration Extrapolation Solver"
        Dialog.__init__( self, self.parent, title )

    def body( self, body_fram ):
        tk_set_icon_portable( self )

        fframe = Tk.Frame( body_fram )
        fframe.pack()
        tframe = Tk.Frame( body_fram )
        tframe.pack( fill=Tk.X )

        cframe = Tk.Frame( fframe )
        cframe.pack( side=Tk.LEFT )
        pframe = Tk.Frame( fframe )
        pframe.pack( side=Tk.LEFT, fill=Tk.BOTH, padx=10 )

        self.fig = fig = plt.figure( figsize=(16, 8) )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas.draw()
        self.toolbar = NavigationToolbar( self.mpl_canvas, tframe )
        self.toolbar.update()

        self.mpl_canvas.draw()

    def apply( self ):  # overrides parent class method
        self.applied = True

        zx_penalty_matrix = []
        for panel in self.panel_list:
            zx_penalty_matrix.append( panel.get_penalty_weights() )

        set_setting( 'zx_penalty_matrix', zx_penalty_matrix )
