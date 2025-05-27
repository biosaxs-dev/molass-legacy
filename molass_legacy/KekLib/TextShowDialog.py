# coding: utf-8
"""

    ファイル名：   TextShowDialog.py

    処理内容：

        

"""
from __future__ import division, print_function, unicode_literals

from molass_legacy.KekLib.OurTkinter         import Tk, Dialog
from molass_legacy.KekLib.TkSupplements      import tk_set_icon_portable

class TextShowDialog( Dialog ):

    def __init__( self, parent, title, message='', width=80, height=50 ):
        self.grab = 'local'     # used in grab_set
        self.applied            = False
        self.message            = message
        self.width              = width
        self.height             = height

        Dialog.__init__(self, parent, title) # this calls body

    def body( self, body_frame ):   # overrides parent class method

        tk_set_icon_portable( self )

        iframe = Tk.Frame( body_frame );
        iframe.pack( expand=1, fill=Tk.BOTH, padx=10, pady=10 )

        message = Tk.Text( iframe, width=self.width, height=self.height )
        message.pack( expand=1 )
        message.insert( Tk.INSERT, self.message )
        # make it read-only
        message.config( state=Tk.DISABLED  )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

    def buttonbox(self):
        '''
        overrides the standard buttons
        '''

        box = Tk.Frame(self)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)

        box.pack( pady=10 )

    def apply( self ):  # overrides parent class method
        print( "ok" )
