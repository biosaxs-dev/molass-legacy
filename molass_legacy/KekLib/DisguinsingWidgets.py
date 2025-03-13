"""
    DisguinsingWidgets.py

    Copyright (c) 2016-2018, Masatsuyo Takahashi
"""
import os
import sys
if sys.version_info > (3,):
    import tkinter  as Tk
else:
    import Tkinter  as Tk

class CheckonLabel( Tk.Label ):
    def __init__( self, parent, **kwargs ):
        dir_, _ = os.path.split( __file__ )
        self.cb_image = Tk.PhotoImage( file=os.path.join( dir_, "checkon.gif") )
        Tk.Label.__init__( self, parent, image=self.cb_image, compound=Tk.LEFT, **kwargs )

class CheckoffLabel( Tk.Label ):
    def __init__( self, parent, **kwargs ):
        dir_, _ = os.path.split( __file__ )
        self.cb_image = Tk.PhotoImage( file=os.path.join( dir_, "checkoff.gif") )
        Tk.Label.__init__( self, parent, image=self.cb_image, compound=Tk.LEFT, **kwargs )

if __name__ == '__main__':
    root = Tk.Tk()
    frame = Tk.Frame( root )
    frame.pack( padx=30, pady=10 )
    cb1 = Tk.Checkbutton( frame, text="This is a normal checkbutton." )
    cb1.pack( anchor=Tk.W )

    cbvar = Tk.IntVar()
    cbvar.set( 1 )
    cb2 = Tk.Checkbutton( frame, text="This is a checked and disabled checkbutton.",
                          variable=cbvar, state=Tk.DISABLED)
    cb2.pack( anchor=Tk.W )
    cb3 = CheckonLabel( frame, text="This is a disguising label." )
    cb3.pack( anchor=Tk.W )
    root.mainloop()
