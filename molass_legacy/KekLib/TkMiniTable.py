# coding: utf-8
"""

    TkMiniTable.py

    Copyright (c) 2016-2020, Masatsuyo Takahashi, KEK-PF

"""
from __future__ import division, print_function, unicode_literals
import sys
import numpy                as np
if sys.version_info > (3,):
    import tkinter as Tk
    import tkinter.ttk as ttk
else:
    import Tkinter as Tk
    import ttk

try:
    import pyperclip
    pyperclip_installed = True
except:
    pyperclip_installed = False

DEBUG = False

class Cell:
    def __init__( self, **entries ): 
        self.__dict__.update( entries )

    def __str__( self ):
        return 'TkMiniTable.Cell<' + str( (self.row, self.col) ) + '>'

class TkMiniTable( Tk.Frame ):
    def __init__( self, parent, columns=[],
        label_bg='gray40', label_fg='white',
        selected_bg='steelblue', selected_fg='white',
        default_colwidth=8,
        font=None,
        *args, **kw ):

        self.cell_widget_class = kw.pop('cell_widget_class', Tk.Label)
        self.on_cell_click = kw.pop('on_cell_click', None)
        self.read_only = kw.pop('read_only', False)
        self.on_cell_click_only = kw.pop('on_cell_click_only', False)

        Tk.Frame.__init__( self, parent, *args, **kw )

        self.parent         = parent
        self.label_bg       = label_bg
        self.label_fg       = label_fg
        self.selected_bg    = selected_bg
        self.selected_fg    = selected_fg
        self.default_colwidth   = default_colwidth
        self.font           = font
        self.active_cell    = None

        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        # self.columnconfigure(0, weight=1)

        self.L_frame = Tk.Frame( self )
        self.R_frame = Tk.Frame( self )
        self.L_frame.grid( row=0, column=0, sticky="nsew" )
        self.R_frame.grid( row=0, column=1, sticky="nsew" )

        """
            L_frame      R_frame
              L_frame1
              L_frame2
        """

        # self.L_frame.rowconfigure(0, weight=1)
        self.L_frame.columnconfigure(0, weight=1)
        self.L_frame.rowconfigure(1, weight=1)

        self.R_frame.rowconfigure(0, weight=1)

        self.L_frame1 = Tk.Frame( self.L_frame )
        self.L_frame1.grid( row=0, column=0, sticky="nsew" )

        vscrollbar = Tk.Scrollbar( self.R_frame, orient=Tk.VERTICAL )
        vscrollbar.grid( row=0, column=0, sticky="ns" )

        self.canvas = canvas = Tk.Canvas( self.L_frame, bd=0,
                                        highlightthickness=0,
                                        yscrollcommand=vscrollbar.set,
                                        )
        canvas.grid( row=1, column=0, sticky="nsew" )
        vscrollbar.config( command=canvas.yview )
        canvas.yview_moveto(0)

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Tk.Frame( canvas )
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=Tk.NW)

        # track changes to the canvas and frame width and sync them,
        # also updating the scrollbar
        def _configure_interior( canvas, interior, event ):
            # update the scrollbars to match the size of the inner frame
            canvas.config( scrollregion=( 0, 0, interior.winfo_reqwidth(), interior.winfo_reqheight() ) )
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the canvas's width to fit the inner frame
                canvas.config( width=interior.winfo_reqwidth() )
            if interior.winfo_reqheight() != canvas.winfo_height():
                # update the canvas's height to fit the inner frame
                canvas.config( height=interior.winfo_reqheight() )

        interior.bind('<Configure>', lambda event: _configure_interior( canvas, interior, event ) )

        # self.iframe = Tk.Frame( self )
        # self.iframe.pack( fill=Tk.BOTH, expand=1 )
        self.L_frame2   = interior
        self.columns    = columns

        self.cells_array = None

        self.reset_cells()
        self.bind( "<Control-c>", self._on_control_c )  # doesn't work

    def add_mousewheel_bind( self, top_widget ):
        self.top_widget = top_widget
        self.bind_all("<MouseWheel>", self._on_mousewheel)

    def reset_cells( self ):

        if self.cells_array is not None:
            for i in range( self.cells_array.shape[0] ):
                for j in range( self.cells_array.shape[1] ):
                    self.cells_array[i, j].widget.destroy()

        self.widgets_dict = {}
        self.selected_cells = []
        self.selected_rows  = []

        _cells = []
        self.columns_dict  = {}
        self.columns_array = []
        for j, col in enumerate( self.columns ):
            svar = Tk.StringVar()
            svar.set( col )

            w = self.cell_widget_class( self.L_frame1, textvariable=svar, relief=Tk.RIDGE, borderwidth=1,
                            width=self.default_colwidth,
                            font=self.font,
                            bg=self.label_bg, fg=self.label_fg )
            w.grid( row=0, column=j )
            w.bind( '<Button-1>', self._on_click )
            # w.bind( '<Control-c>', self._on_control_c )

            cell = Cell( row=0, col=j, svar=svar, widget=w, justify=Tk.CENTER, width=self.default_colwidth,
                        font=self.font,
                        inactive_bg=self.label_bg, inactive_fg=self.label_fg
                        )

            _cells.append( cell )

            head = Cell( col=j, justify=Tk.CENTER, width=self.default_colwidth, font=self.font,
                        bg='whitesmoke', fg='black'
                        )

            self.columns_dict[ col ] = head
            self.columns_array.append( head )

            self.widgets_dict[ w ] = cell

        self.cells_array    = np.array( [ _cells ] )

    def heading( self, colid, text=None, justify=Tk.CENTER ):
        j = self.columns_dict[ colid ].col
        cell = self.cells_array[ 0, j ]
        cell.svar.set( text )
        cell.widget.config( justify=justify )

    def column( self, colid, justify=Tk.CENTER, width=None, bg=None, **kwargs ):
        try:
            # if colid is iterable loop as expected.
            for i in colid:
                self.column_impl( i, justify, width, bg, **kwargs )
            return
        except Exception as exc:
            if DEBUG:
                raise exc
            pass

        self.column_impl(colid, justify, width, bg, **kwargs)

    def column_impl( self, colid, justify, width, bg, **kwargs):

        if type( colid ) == int:
            column = self.columns_array[ colid ]
        else:
            column = self.columns_dict[ colid ]

        if width is None:
            width = self.default_colwidth
        # print( colid, 'column width=', width )
        column.width = width
        if bg is not None:
            column.bg = bg

        n_ = column.col
        # print( 'column: n_=', n_ )
        column_cells = self.cells_array[ :, n_ ]
        for cell in column_cells:
            kwargs = { 'justify':justify, 'width':width }
            cell.widget.config( **kwargs )
            cell.width = width
        self.update()

    def clear_rows( self ):
        pass

    def insert_row( self, row_tuple ):
        id_ = self.cells_array.shape[0]

        _cells = []
        for j, col in enumerate( row_tuple ):

            column = self.columns_array[j]

            svar = Tk.StringVar()
            svar.set( col )

            if j == 0:
                bg = self.label_bg
                fg = self.label_fg
            else:
                # bg = 'whitesmoke'
                # fg = 'black'
                bg = column.bg
                fg = column.fg

            w = self.cell_widget_class( self.L_frame2, textvariable=svar,
                            justify=column.justify, width=column.width, relief=Tk.RIDGE, borderwidth=1,
                            font=self.font,
                            bg=bg, fg=fg )

            w.grid( row=id_-1, column=j )
            w.bind( '<Button-1>', self._on_click )
            w.bind( '<Button-3>', self._on_right_click )
            w.bind( '<Double-Button-1>', self._on_double_click )
            # w.bind( '<Control-c>', self._on_control_c )

            cell = Cell( row=id_, col=j, svar=svar, widget=w, inactive_bg=bg, inactive_fg=fg )
            _cells.append( cell )

            self.widgets_dict[ w ] = cell

        self.cells_array = np.vstack( [ self.cells_array, np.array( [ _cells ] ) ] )

        # print( 'cells_array.shape=', self.cells_array.shape )

        return _cells

    def _select_row( self, row ):
        if len( self.selected_rows ) > 0:
            prev_row = self.selected_rows[0]
            if row == prev_row: return

        for cell in self.selected_cells:
            cell.widget.config( bg=cell.inactive_bg, fg=cell.inactive_fg )

        if row == 0:
            self.selected_rows  = []
            return

        self.selected_rows  = [ row ]

        self.selected_cells = []
        row_cells = self.cells_array[ row, : ]
        for cell in row_cells:
            bg_ = cell.widget.cget( 'bg' )
            cell.inactive_bg = bg_
            fg_ = cell.widget.cget( 'fg' )
            cell.inactive_fg = fg_
            cell.widget.config( bg=self.selected_bg, fg=self.selected_fg )
            self.selected_cells.append( cell )

    def activate_cell( self, cell ):
        if DEBUG: print( 'activate_cell', cell )
        if self.active_cell is not None:
            a_cell = self.active_cell
            if a_cell.row in self.selected_rows:
                bg  = self.selected_bg
                fg  = self.selected_fg
            else:
                bg  = a_cell.inactive_bg
                fg  = a_cell.inactive_fg
            self.active_cell.widget.config( bg=bg, fg=fg )

        self.active_cell = cell
        if DEBUG: print( 'activate_cell active_cell=', self.active_cell )
        cell.widget.config( bg='blue', fg='white' )
        cell.widget.focus_force()

    def _on_click( self, event ):
        if DEBUG: print( '_on_click event.widget=', event.widget )
        if self.read_only:
            return

        cell = self.widgets_dict.get( event.widget )
        if cell is None:
            if DEBUG: print( '_on_click active_cell=', self.active_cell )
            self.inactivate()
            return

        if self.on_cell_click is not None:
            self.on_cell_click(cell)
            if self.on_cell_click_only:
                return

        # print( 'click', [ cell.row, cell.col ] )
        self._select_row( cell.row )

        self.activate_cell( cell )

        if hasattr( self.parent, '_on_click' ):
            self.parent._on_click( event )

    def select_row( self, row ):
        row_ = row + 1
        self._select_row(row_)
        cell = self.cells_array[ row_, 1 ]
        self.activate_cell(cell)
        move_y = max(0, row_ - 5) / self.cells_array.shape[0]
        self.canvas.yview_moveto(move_y)

    def inactivate( self ):
        if DEBUG: print( 'inactivate active_cell=', self.active_cell )

        if self.active_cell is None: return

        a_cell = self.active_cell
        if a_cell.row == 0 or a_cell.col == 0:
            bg  = self.label_bg
            fg  = self.label_fg
        else:
            bg  = self.selected_bg
            fg  = self.selected_fg
        self.active_cell.widget.config( bg=bg, fg=fg )
        self.active_cell = None

    def _on_double_click( self, event ):
        cell = self.widgets_dict.get( event.widget )
        if cell is None: return

        # print( 'double click', [ cell.row, cell.col ] )

        if hasattr( self.parent, '_on_double_click' ):
            self.parent._on_double_click( event )

    def _on_right_click( self, event ):
        if len( self.selected_rows ) == 0: return

        if hasattr( self.parent, '_on_right_click' ):
            self.parent._on_right_click( event )

    def _on_control_c( self, event ):
        # print( '_on_control_c' )
        if not pyperclip_installed:
            print( 'Install pyperclip if you need to copy to the clipboard.' )
            return
        if self.active_cell is None: return

        value = self.active_cell.svar.get()
        pyperclip.copy( value )
        print( 'active cell value copied to the clipboard', value )

    def _on_mousewheel( self, event ):
        """
        scroll only if the window is in focus
        TODO: any other way?
        
        http://stackoverflow.com/questions/10343759/determining-what-tkinter-window-is-currently-on-top
        Determining what tkinter window is currently on top
        """
        num_stacked = len( self.tk.eval( 'wm stackorder '+str(self.top_widget) ).split(' ') )
        if num_stacked == 1:
            self.canvas.yview_scroll( int( -1*(event.delta/120) ), "units")
