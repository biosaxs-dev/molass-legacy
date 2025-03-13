# coding: utf-8
"""

    ProgressChart.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""

class ProgressChart:
    def __init__( self, fig, size ):
        self.fig    = fig
        self.size   = size
        self.ax1    = fig.add_subplot( 111 )
        self.ax2    = self.ax1.twinx()

    def draw( self, sc_vector,  quality_array, rg_array ):
        self.ax1.cla()
        self.ax2.cla()
        self.ax1.set_axis_off()
        self.ax2.set_axis_off()

        self.ax1.plot( sc_vector, color='blue' )
        self.ax2.plot( rg_array, color='red' )
