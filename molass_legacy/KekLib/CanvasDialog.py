# coding: utf-8
"""

    ファイル名：   CanvasDialog.py

    処理内容：

        デバッグ用の plot 

    Copyright (c) 2017-2018, Masatsuyo Takahashi, KEK-PF

"""
import matplotlib.pyplot                as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib          import NavigationToolbar
from molass_legacy.KekLib.TkUtils                import adjusted_geometry
import matplotlib

class CanvasDialog( Dialog ):
    def __init__( self, title, parent=None, adjust_geometry=False, figure=None ):

        if parent is None:
            parent = Tk.Toplevel()
            if adjust_geometry:
                parent.geometry( adjusted_geometry( parent.geometry() ) )
            parent.withdraw()
            self.created_parent = True
        else:
            self.created_parent = False

        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.title_     = title
        self.applied    = None
        self.caller_module = get_caller_module( level=2 )
        self.figure     = figure
        self.mplt_ge_2_2 = matplotlib.__version__ >= '2.2'

    def show( self, draw_func, figsize=None, message=None, button_labels=[ "OK", "Cancel" ],
                toolbar=False, parent_arg=False ):
        self.draw_func  = draw_func
        self.figsize    = figsize
        self.message    = message
        assert( len(button_labels) == 2 )
        self.button_labels  = button_labels
        self.toolbar    = toolbar
        self.parent_arg = parent_arg

        Dialog.__init__( self, self.parent, self.title_ )
        # TODO: self.resizable(width=False, height=False)

    def destroy_parent( self ):
        if self.created_parent:
            self.parent.destroy()

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        if self.message is not None:
            self.msg = Tk.Label( body_frame, text=self.message, bg='white' )
            self.msg.pack( fill=Tk.BOTH, expand=1, pady=20 )
            # msg.insert( Tk.INSERT, self.message )
            # msg.config( state=Tk.DISABLED )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        if self.figure is None:
            figsize_ = ( 18, 9 ) if self.figsize is None else self.figsize
            fig = plt.figure( figsize=figsize_ )
        else:
            fig = self.figure

        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        if self.mplt_ge_2_2:
            self.mpl_canvas.draw()
        else:
            self.mpl_canvas.show()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

        # it seems that draw_func should be called after the creation of mpl_canvas
        # in order to enable 3d-rotation of the figure
        if self.parent_arg:
            self.draw_func( fig, self.parent )
        else:
            self.draw_func( fig )
        self.parent.config( cursor='' )

        if self.toolbar:
            self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
            self.toolbar.update()

        self.protocol( "WM_DELETE_WINDOW", self.ok )

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        box = Tk.Frame(self)

        w = Tk.Button(box, text=self.button_labels[0], width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.ok_button = w
        w = Tk.Button(box, text=self.button_labels[1], width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.cancel_button = w

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def ok( self, *args ):
        self.destroy_parent()
        self.applied  = True
        self.destroy()

    def cancel( self, *arg ):
        self.destroy_parent()
        self.applied  = False
        self.destroy()

