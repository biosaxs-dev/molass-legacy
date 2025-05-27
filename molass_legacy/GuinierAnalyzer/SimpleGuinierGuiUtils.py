# coding: utf-8
"""
    SimpleGuinierGuiUtils.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy        as np
from molass_legacy.KekLib.OurTkinter     import Tk, Font
from molass_legacy.KekLib.TkUtils        import adjusted_geometry

window_title    = 'Guinier Analyzer 0.0.1'

class CheckbuttonFrame( Tk.Frame ):
    def __init__( self, parent, types=None ):
        Tk.Frame.__init__( self, parent )

        if types is None:
            types = range(17)

        self.types = types
        self.clear_btn   = Tk.Button( self, text='clear all', command=self.cb_clear )
        self.clear_btn.grid( row=0, column=0 )

        self.cb_vars = []
        for i in types:
            cb_var = Tk.IntVar()
            cb_var.set( 1 )
            self.cb_vars.append( cb_var )
            cb = Tk.Checkbutton( self, variable=cb_var, text='type %d' % i )
            cb.grid( row=i+1, column=0, sticky=Tk.W )
            cb_var.trace( 'w', self.cb_vars_tracer )

        self.cb_vars_tracer_reset()

    def cb_vars_tracer_reset( self ):
        self.changed = False

    def cb_vars_tracer( self, *args ):
        self.changed = True

    def cb_clear( self ):
        text = self.clear_btn.cget( 'text' )
        val = 0 if text == 'clear all' else 1
        for cb_var in self.cb_vars:
            cb_var.set( val )
        text = 'check all' if val == 0 else 'clear all'
        self.clear_btn.config( text=text )

    def make_index( self, z ):
        types = []
        for t, v in enumerate( [ var.get() for var in self.cb_vars ] ):
            if v  == 1:
                types.append( self.types[t] )

        self.cb_vars_tracer_reset()

        if len( types ) == len( self.cb_vars ):
            return slice(None, None)
        else:
            return np.logical_or.reduce( [ z == t for t in types ] )

class TypeInfoPanel(Tk.Toplevel):
    def __init__( self, parent ):
        self.parent = parent
        Tk.Toplevel.__init__( self, parent )
        self.title( 'Type Help'  )
        self.fixed_font = Font.Font( family="Courier", size=9 )

    def show( self, type_names ):
        frame = Tk.Frame( self )
        frame.pack( padx=50, pady=20 )
        self.type_names = type_names
        for i, name in enumerate( self.type_names ):
            label = Tk.Label( frame, text='type %2d:    %s' % ( i, name ), font=self.fixed_font )
            label.grid( row=i, column=0, sticky=Tk.W )
        self.update()
        self.geometry( adjusted_geometry(self.geometry()) )

class FolderInfoPanel(Tk.Toplevel):
    def __init__( self, parent ):
        self.parent = parent
        Tk.Toplevel.__init__( self, parent )
        self.title( 'Folder Help'  )
        self.fixed_font = Font.Font( family="Courier", size=9 )

    def show( self, folder_paths ):
        frame = Tk.Frame( self )
        frame.pack( padx=20, pady=20 )
        self.folder_paths = folder_paths
        for i, path in enumerate( self.folder_paths ):
            button = Tk.Button( frame, text='folder %03d: %s' % ( i, path ), font=self.fixed_font,
                        command=lambda i_=i: self.draw_folder(i_),
                        relief=Tk.FLAT )
            j = i % 20
            k = i //20
            button.grid( row=j, column=k, sticky=Tk.W, padx=20 )
        self.update()
        self.geometry( adjusted_geometry(self.geometry()) )

    def draw_folder( self, i ):
        self.parent.draw_folder( folder_no=i )
