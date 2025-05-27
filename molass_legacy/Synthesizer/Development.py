# -*- coding: utf-8 -*-
"""

    ファイル名：   Development.py

    処理内容：

        GUI のオプション設定情報

"""
from __future__ import division, print_function, unicode_literals

from PersistentInfo import PersistentInfo

DEFAULT_DEVELOPER_OPTIONS = {
    'adj_algorithm'         : 'round',
    'min_ratio'             : 0.5,
    'adj_output'            : 'NO',
    'postfix_adj'           : '_adj',
    'intermediate_results'  : 'NO',
    'defaultfont'           : ( 'TkDefaultFont', 10 ),
    'fixedfont'             : ( 'TkFixedFont', 10 ),
    'debug'                 : False,
    }

development_info    = PersistentInfo( 'development.dump', DEFAULT_DEVELOPER_OPTIONS )
development_        = development_info.get_dictionary()

from molass_legacy.KekLib.OurTkinter import Tk, Dialog, tk_set_icon_portable

DEBUG = False

def get_devel_info( item ):
    assert( item in DEFAULT_DEVELOPER_OPTIONS )
    return development_.get( item )

def set_devel_info( item, value ):
    assert( item in DEFAULT_DEVELOPER_OPTIONS )
    development_[item] = value

class DeveloperOptionsDialog( Dialog ):

    def __init__( self, parent, title ):
        self.grab = 'local'     # used in grab_set
        self.temporary          = False
        self.applied            = False

        Dialog.__init__(self, parent, title) # this calls body

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self, 'synthesizer' )

        iframe = Tk.Frame( body_frame );
        iframe.pack( expand=1, fill=Tk.BOTH, padx=10, pady=10 )

        label_spacing = '     '
        text_entry_width    = 12

        grid_row = -1

        grid_row += 1
        adj_algorithm_label = Tk.Label( iframe, text= 'Adjust Algorithm:' + label_spacing )
        adj_algorithm_label.grid( row=grid_row, column=0, sticky=Tk.E )
        self.adj_algorithm = Tk.StringVar()
        j = 0
        for t, method in ( [ 'Round', 'round' ], [ 'Fast', 'fast'], [ 'Slow', 'slow' ]  ):
            text_ = '%-20.20s' % ( t )
            b = Tk.Radiobutton( iframe, text=text_,
                            variable=self.adj_algorithm, value=method )
            j += 1
            b.grid( row=grid_row, column=j, sticky=Tk.W )

        grid_row += 1
        self.min_ratio = Tk.DoubleVar()
        self.min_ratio.set( get_devel_info( 'min_ratio' ) )
        min_ratio_label = Tk.Label( iframe, text= 'Acceptable Pixel Cover Ratio:' + label_spacing )
        min_ratio_label.grid( row=grid_row, column=0, sticky=Tk.E )
        min_ratio_entry = Tk.Entry( iframe, textvariable=self.min_ratio, width=text_entry_width, justify=Tk.CENTER )
        min_ratio_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        def adj_algorithm_trace( *args ):
            adj_algorithm = self.adj_algorithm.get()
            if adj_algorithm == 'round':
                state_ = 'disabled'
            else:
                state_ = 'normal'
            min_ratio_entry.config( state=state_ )

        self.adj_algorithm.trace( "w", adj_algorithm_trace )
        self.adj_algorithm.set( get_devel_info( 'adj_algorithm' ) )

        grid_row += 1
        adj_output_label = Tk.Label( iframe, text= 'Adjusted File Output:' + label_spacing )
        adj_output_label.grid( row=grid_row, column=0, sticky=Tk.E )
        self.adj_output = Tk.StringVar()
        j = 0
        for t, n in ( [ 'Yes', 'YES' ], [ 'No', 'NO' ] ):
            b = Tk.Radiobutton( iframe, text=t,
                            variable=self.adj_output, value=n )
            j += 1
            b.grid( row=grid_row, column=j, sticky=Tk.W )

        grid_row += 1
        self.postfix_adj = Tk.StringVar()
        self.postfix_adj.set( get_devel_info( 'postfix_adj' ) )
        postfix_adj_label = Tk.Label( iframe, text= 'Adjusted File Postfix:' + label_spacing )
        postfix_adj_label.grid( row=grid_row, column=0, sticky=Tk.E )
        postfix_adj_entry = Tk.Entry( iframe, textvariable=self.postfix_adj, width=text_entry_width, justify=Tk.CENTER )
        postfix_adj_entry.grid( row=grid_row, column=1, sticky=Tk.W )

        def adj_output_trace( *args ):
            adj_output = self.adj_output.get()
            if adj_output == 'NO':
                state_ = 'disabled'
            else:
                state_ = 'normal'
            postfix_adj_entry.config( state=state_ )

        self.adj_output.trace( "w", adj_output_trace )

        self.adj_output.set( get_devel_info( 'adj_output' ) )

        grid_row += 1
        intermediate_results_label = Tk.Label( iframe, text= 'Intermediate Results Output :' + label_spacing )
        intermediate_results_label.grid( row=grid_row, column=0, sticky=Tk.E )
        self.intermediate_results = Tk.StringVar()
        self.intermediate_results.set( get_devel_info( 'intermediate_results' ) )
        j = 0
        for t, n in ( [ 'Yes', 'YES' ], [ 'No', 'NO' ] ):
            b = Tk.Radiobutton( iframe, text=t,
                            variable=self.intermediate_results, value=n )
            j += 1
            b.grid( row=grid_row, column=j, sticky=Tk.W )

    def select_font( self ):
        font = askChooseFont( self )
        print( 'selected font=', font )
        pass

    def apply( self ):  # overrides parent class method
        print( "ok. apply" )

        set_devel_info( 'min_ratio',            self.min_ratio.get() )
        set_devel_info( 'adj_algorithm',        self.adj_algorithm.get() )
        set_devel_info( 'intermediate_results', self.intermediate_results.get() )

        old_adj_output = get_devel_info( 'adj_output' )
        new_adj_output = self.adj_output.get()
        if new_adj_output != old_adj_output:
            self.adj_output_changed = True
        set_devel_info( 'adj_output', new_adj_output )
        set_devel_info( 'postfix_adj', self.postfix_adj.get() )

        if not self.temporary:
            # development_info.save()
            pass    # don't save

        self.applied    = True
