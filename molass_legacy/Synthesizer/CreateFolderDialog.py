# -*- coding: utf-8 -*-
"""

    ファイル名：   CreateFolderDialog.py

    処理内容：

        GUI のオプション設定情報

"""
from __future__ import division, print_function, unicode_literals

import sys
import os

from molass_legacy.KekLib.OurTkinter     import Tk, Dialog
from molass_legacy.KekLib.TkSupplements  import tk_set_icon_portable

class CreateFolderDialog( Dialog ):

    def __init__( self, parent, title, suggested_folders ):
        self.grab = 'local'     # used in grab_set
        self.suggested_folders  = suggested_folders
        self.applied    = False
        self.created_folder = None

        Dialog.__init__(self, parent, title) # this calls body

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self, 'synthesizer' )

        iframe = Tk.Frame( body_frame );
        iframe.pack( expand=1, fill=Tk.BOTH, padx=10, pady=10 )

        question_label = Tk.Label( iframe, text= 'Select (and change if necessary) one of the new folders suggested below to create.' )
        question_label.pack( anchor=Tk.W )

        entry_frame = Tk.Frame( iframe );
        # entry_frame.pack( expand=1, fill=Tk.BOTH )
        entry_frame.pack( anchor=Tk.W )

        text_entry_width = 80
        self.folders = []

        self.selection = Tk.IntVar()
        self.selection.set( 0 )
        i = 0
        for f in self.suggested_folders:
            b = Tk.Radiobutton( entry_frame, text='', variable=self.selection, value=i )
            b.grid( row=i, column=0, sticky=Tk.W )
            var = Tk.StringVar()
            var.set( f )
            self.folders.append( var )
            entry_ = Tk.Entry( entry_frame, textvariable=var, width=text_entry_width )
            entry_.grid( row=i, column=1, sticky=Tk.W )
            i += 1

        guide_label = Tk.Label( iframe, text= 'If you cancel, the file selection dialog will follow.' )
        guide_label.pack( anchor=Tk.W )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

    def apply( self ):  # overrides parent class method
        print( "CreateFolderDialog: ok. apply" )

        i = self.selection.get()
        selected_folder =  self.folders[i].get()
        if not os.path.exists( selected_folder ):
            os.makedirs( selected_folder )
        self.created_folder = selected_folder
        self.applied    = True
