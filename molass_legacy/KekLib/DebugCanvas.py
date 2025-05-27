# coding: utf-8
"""

    ファイル名：   DebugCanvas.py

    処理内容：

        デバッグ用の plot 

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import numpy                            as np
import matplotlib.pyplot                as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter                         import Tk, Dialog
from molass_legacy.KekLib.TkUtils                            import adjusted_geometry

class DebugCanvas( Dialog ):
    def __init__( self, title, draw_func, parent=None, figsize=None, toolbar=False ):

        if parent is None:
            parent = Tk.Toplevel()
            parent.withdraw()
            self.created_parent = True
        else:
            self.created_parent = False

        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.title_     = title
        self.figsize    = figsize
        self.draw_func  = draw_func
        self.continue_  = None
        self.toolbar    = toolbar

    def show( self, cancelable=False, cunstom_button_cb=None, cursor_update=True ):
        self.cancelable = cancelable
        self.cunstom_button_cb = cunstom_button_cb
        self.cursor_update = cursor_update
        # self.parent.after( 0, self.adjust_geometry )
        Dialog.__init__( self, self.parent, self.title_ )
        # TODO: self.resizable(width=False, height=False)

    def destroy_parent( self ):
        if self.created_parent:
            self.parent.destroy()

    def body( self, body_frame ):   # overrides parent class method

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        figsize_ = ( 18, 9 ) if self.figsize is None else self.figsize

        fig = plt.figure( figsize=figsize_ )

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        if self.toolbar:
            from molass_legacy.KekLib.OurMatplotlib  import NavigationToolbar
            self.toolbar_widget = NavigationToolbar( self.mpl_canvas, cframe )

        # it seems that draw_func should be called after the creation of mpl_canvas
        # in order to enable 3d-rotation of the figure
        self.draw_func( fig )
        if self.cursor_update:
            self.parent.config( cursor='' )

        self.protocol( "WM_DELETE_WINDOW", self.ok )

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        if self.cunstom_button_cb is not None:
            self.cunstom_button_cb( self )
            return

        box = Tk.Frame(self)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        if self.cancelable:
            w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def ok( self ):
        self.destroy_parent()
        self.continue_  = True
        self.destroy()

    def cancel( self ):
        self.destroy_parent()
        self.continue_  = False
        self.destroy()

    def adjust_geometry( self ):
        self.update()
        # TODO: fix being placed to the exceeding left
        self.geometry( adjusted_geometry( self.geometry() ) )

