# coding: utf-8
"""

    InputOutputDisplay.py

    Copyright (c) 2017-2018, Masatsuyo Takahashi, KEK-PF

"""

from molass_legacy.KekLib.OurTkinter             import Tk, Dialog, is_empty_val
from molass_legacy._MOLASS.SerialSettings         import get_setting

class InputOutputDisplay( Tk.Frame ):
    def __init__( self, parent, scatter_folder=None, absorb_file=None, book_path=None, label_witdh=100, value_width=300 ):
        Tk.Frame.__init__( self, parent )

        if scatter_folder is None:
            scatter_folder  = get_setting( 'in_folder' )

        if absorb_file is None:
            absorb_file     = '/'.join( [ get_setting( 'uv_folder' ), get_setting( 'uv_file' ) ] )

        if book_path is None:
            book_path       = '/'.join( [ get_setting( 'analysis_folder' ), get_setting( 'result_book' ) ] )

        label_width_frame = Tk.Frame( self, width=label_witdh )
        label_width_frame.grid( row=0, column=0 )
        value_width_frame = Tk.Frame( self, width=value_width )
        value_width_frame.grid( row=0, column=1 )

        scatter_label   = Tk.Label( self, text="Xray Scattering Data" )
        scatter_label.grid( row=0, column=0, sticky=Tk.E, pady=1 )

        scatter_value   = Tk.Label( self, text=scatter_folder, bg='white', anchor=Tk.W )
        scatter_value.grid( row=0, column=1, sticky=Tk.W + Tk.E, padx=5 )

        if get_setting('use_xray_conc') == 0:
            uv_state    = Tk.NORMAL
            uv_bg       = 'white'
        else:
            uv_state    = Tk.DISABLED
            uv_bg       = None
            absorb_file = "Using Xray-proportional concentration"

        absorb_label    = Tk.Label( self, text="UV Absorbance Data", state=uv_state )
        absorb_label.grid( row=1, column=0, sticky=Tk.E, pady=1 )

        absorb_value    = Tk.Label( self, text=absorb_file, bg=uv_bg, anchor=Tk.W, state=uv_state )
        absorb_value.grid( row=1, column=1, sticky=Tk.W + Tk.E, padx=5 )

        outbook_label   = Tk.Label( self, text="Result Book" )
        outbook_label.grid( row=2, column=0, sticky=Tk.E, pady=1 )

        outbook_value    = Tk.Label( self, text=book_path, bg='white', anchor=Tk.W )
        outbook_value.grid( row=2, column=1, sticky=Tk.W + Tk.E, padx=5 )
