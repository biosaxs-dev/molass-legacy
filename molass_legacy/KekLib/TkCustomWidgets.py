# coding: utf-8
"""

    ファイル名：   TkCustomWidgets.py

    処理内容：

        FileNodeEntry ----- 上位クラス
        FolderEntry ------- Drag&drop 機能および選択ボタン付きの Folder 入力用 Entry
        FileEntry   ------- Drag&drop 機能および選択ボタン付きの File 入力用 Entry

    Copyright (c) 2017-2021, Masatsuyo Takahashi

"""
import os

from molass_legacy.KekLib.OurTkinter import Tk, FileDialog, is_empty_val
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import OurMessageBox    as MessageBox
from molass_legacy.KekLib.TkSupplements import SlimButton
from TkSizableWidgets import SizableEntry

class FileNodeEntry( Tk.Frame ):
    def __init__( self, parent, **kw ):

        sizable = kw.get( 'sizable' )
        if sizable is None:
            sizable = False
        else:
            del kw[ 'sizable' ]

        slimbutton = kw.get( 'slimbutton' )
        if slimbutton is None:
            slimbutton = False
        else:
            del kw[ 'slimbutton' ]

        Tk.Frame.__init__( self, parent )
        self.parent = parent
        var = kw.get( 'textvariable' )
        if var is None:
            var = Tk.StringVar()
            kw['textvariable'] = var
        self.variable = var
        self.on_entry_cb = kw.pop('on_entry_cb', None)

        if sizable:
            self.entry  = SizableEntry( self, **kw )
        else:
            self.entry  = Tk.Entry( self, **kw )

        if slimbutton:
            # self.button = SlimButton( self, text='...', command=self.on_button_click )
            self.button = SlimButton( self, text='...', command=self.on_button_click, width=18, height=22 )
        else:
            self.button = Tk.Button( self, text='...', command=self.on_button_click )

        self.entry.pack( side= Tk.LEFT, fill=Tk.BOTH, expand=1 )
        self.button.pack( side=Tk.LEFT )
        self.add_dnd_bind()

    def delete( self, *args ):
        self.entry.delete( *args )

    def insert( self, *args ):
        self.entry.insert( *args )

    def config( self, *args, **kwargs ):
        for w in [ self.entry, self.button ]:
            w.config( *args, **kwargs )

    def add_dnd_bind( self ):
        self.entry.register_drop_target("*")

        def dnd_handler( event ):
            event.widget.delete( 0, Tk.END )
            event.widget.insert( 0, event.data )
            self.on_entry()

        self.entry.bind("<<Drop>>", dnd_handler)

    def on_button_click( self ):
        f = self.select()
        if not f:
            return

        self.variable.set( f )
        self.check()
        self.on_entry()

    def select( self ):
        # must be overridden
        assert( False )

    def check( self ):
        # must be overridden
        assert( False )

    def on_entry( self, *args, **kwargs ):
        ok = self.check( *args )
        if not ok:
            return

        if self.on_entry_cb is not None:
            self.on_entry_cb( *args, **kwargs )

    def set_error( self ):
        self.entry.config( fg='red' )
        self.entry.focus_force()

    def bind(self, *args):
        self.entry.bind(*args)

class FolderEntry( FileNodeEntry ):
    def __init__( self, parent, **kw ):
        FileNodeEntry.__init__( self, parent, **kw )

    def select( self ):
        path = self.variable.get()
        dir_ = os.path.dirname( path ).replace( '/', '\\' )
        # print 'dir_=', dir_
        f = FileDialog.askdirectory( initialdir=dir_, parent=self.parent )
        return f

    def check( self, *args ):
        f = self.variable.get()
        ret = False
        if is_empty_val( f ):
            self.set_error()
            MessageBox.showerror( "Folder Input Error", "This folder input is required.", parent=self.parent )
        elif not os.path.exists( f ):
            self.set_error()
            MessageBox.showerror( "Folder Input Error", "'%s' does not exist." % f, parent=self.parent )
        elif not os.path.isdir( f ):
            self.set_error()
            MessageBox.showerror( "Folder Input Error", "'%s' is not a folder." % f, parent=self.parent )
        else:
            self.entry.config( fg='black' )
            ret = True

        if not ret:
            self.entry.focus_force()

        return ret

class FileEntry( FileNodeEntry ):
    def __init__( self, parent, **kw ):
        self.mode = kw.pop( 'mode', None )
        if self.mode is None:
            self.mode = 'r'

        FileNodeEntry.__init__( self, parent, **kw )

    def select( self ):
        path = self.variable.get()
        dir_ = os.path.dirname( path ).replace( '/', '\\' )
        # print 'dir_=', dir_
        if self.mode == 'w':
            f = FileDialog.asksaveasfilename ( initialdir=dir_, parent=self.parent )
        else:
            f = FileDialog.askopenfilename( initialdir=dir_, parent=self.parent )
        return f

    def check( self, *args ):
        f = self.variable.get()
        print( 'check: f=', f )
        ret = False
        if is_empty_val( f ):
            self.set_error()
            MessageBox.showerror( "File Input Error", "This file input is required.", parent=self.parent )
        elif not os.path.exists( f ):
            self.set_error()
            MessageBox.showerror( "File Input Error", "'%s' does not exist." % f, parent=self.parent )
        elif not os.path.isfile( f ):
            self.set_error()
            MessageBox.showerror( "File Input Error", "'%s' is not a file." % f, parent=self.parent )
        else:
            self.entry.config( fg='black' )
            ret = True

        if not ret:
            self.entry.focus_force()

        return ret

if __name__ == "__main__":
    root = Tk.Tk()
    var = Tk.StringVar()
    entry   = FolderEntry( root, textvariable=var, width=60 )
    # entry   = FileEntry( root, textvariable=var, width=60 )
    entry.pack()
    root.mainloop()
