"""
    ScrolledFrame.py

    original: http://tkinter.unpythonic.net/wiki/VerticalScrolledFrame

    Copyright (c) 2016-2024, Masatsuyo Takahashi, KEK-PF
"""
import tkinter      as Tk
import tkinter.constants    as Tkconstants
from molass_legacy.KekLib.TkSupplements  import SlimButton

class ScrolledFrame(Tk.Frame):
    """A pure Tkinter scrollable frame that actually works!
    * Use the 'interior' attribute to place widgets inside the scrollable frame
    * Construct and pack/place/grid normally
    * This frame only allows vertical scrolling

    """
    def __init__(self, parent, *args, **kw):
        self.sync_canvas_info = None
        Tk.Frame.__init__(self, parent, *args, **kw)            

        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # create a canvas object and a vertical scrollbar for scrolling it
        vscrollbar = Tk.Scrollbar(self, orient=Tk.VERTICAL)
        vscrollbar.grid( row=0, column=1, sticky=Tkconstants.NS )
        hscrollbar = Tk.Scrollbar(self, orient=Tk.HORIZONTAL)
        hscrollbar.grid( row=1, column=0, sticky=Tkconstants.EW )

        self.canvas = canvas = Tk.Canvas( self, bd=0, highlightthickness=0,
                            xscrollcommand=hscrollbar.set,
                            yscrollcommand=vscrollbar.set,
                            )
        canvas.grid( row=0, column=0, sticky=Tkconstants.NSEW )

        self.canvas_bottom = canvas_bottom = Tk.Canvas( self, bd=0, highlightthickness=0,
                            xscrollcommand=hscrollbar.set,
                            height=0
                            )

        canvas_bottom.grid( row=2, column=0, sticky=Tkconstants.NSEW )

        def xview_( *args ):
            # print( 'xview_' )
            canvas.xview( *args )
            canvas_bottom.xview( *args )
            if self.sync_canvas_info is not None:
                for sync_canvas in self.sync_canvas_info:
                    sync_canvas.xview( *args )

        hscrollbar.config( command=xview_ )
        vscrollbar.config( command=canvas.yview )

        # reset the view
        canvas.xview_moveto(0)
        canvas.yview_moveto(0)
        canvas_bottom.xview_moveto(0)

        self.canvas_info = canvas, canvas_bottom

        # create a frame inside the canvas which will be scrolled with it
        self.interior = interior = Tk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=Tk.NW)

        self.interior_bottom = interior_bottom = Tk.Frame(canvas_bottom)
        interior_bottom_id = canvas_bottom.create_window(0, 0, window=interior_bottom,
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
        interior_bottom.bind('<Configure>', lambda event: _configure_interior( canvas_bottom, interior_bottom, event ))

        def _configure_canvas( canvas, interior, interior_id, event, height_adjust=True ):
            # TODO: is this neccesary?
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # update the inner frame's width to fill the canvas
                canvas.itemconfigure( interior_id, width=canvas.winfo_width() )
            if height_adjust and interior.winfo_reqheight() != canvas.winfo_height():
                # update the inner frame's heght to fill the canvas
                canvas.itemconfigure( interior_id, height=canvas.winfo_height() )

        # canvas.bind('<Configure>', lambda event: _configure_canvas( canvas, interior, interior_id, event ) )
        # interior_bottom.bind('<Configure>', lambda event: _configure_canvas( canvas_bottom, interior_bottom, interior_bottom_id, event, height_adjust=False ) )

        self.lower_right_button = SlimButton( self, text='▲', height=18, width=18, command=self.lower_right_button_toggle )
        self.lower_right_button.grid( row=1, column=1 )
        self.lower_right_button_toggle()

    # thanks to: https://www.daniweb.com/programming/software-development/code/429838/simple-tkinter-toggle-button
    def lower_right_button_toggle( self ):
        text_ = self.lower_right_button.button['text']
        if text_ == '▼':
            text_   = '▲'
            self.canvas_bottom.grid( row=2, column=0, sticky=Tkconstants.NSEW  )
        else:
            text_  = '▼'
            self.canvas_bottom.grid_remove()
        self.lower_right_button.button['text'] = text_

    def destroy( self ):
        self.canvas.destroy()   # seems to have to be destroyed first in Python 3.7
        super(Tk.Frame, self).destroy()

    # thanks to: https://stackoverflow.com/questions/17355902/tkinter-binding-mousewheel-to-scrollbar
    def add_bind_mousewheel(self):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def set_sync_info(self, canvas_info):
        self.sync_canvas_info = canvas_info

if __name__ == "__main__":

    class SampleApp( Tk.Tk ):
        def __init__( self, *args, **kwargs ):
            root = Tk.Tk.__init__( self, *args, **kwargs )

            quit_button = Tk.Button( root, text="quit", command=self.quit )
            quit_button.pack( side=Tk.BOTTOM )

            self.frame = ScrolledFrame( root )
            self.frame.pack( fill=Tk.BOTH, expand=1 )
            self.frame2 = ScrolledFrame( root )
            self.frame2.pack( fill=Tk.BOTH, expand=1 )
            self.label = Tk.Label(self, text="Shrink the window to activate the scrollbar.")
            self.label.pack()
            buttons = []
            for i in range(10):
                buttons.append(Tk.Button( self.frame.interior, width=40, text="Button " + str(i)))
                buttons[-1].pack()

            self.frame.interior_bottom.configure( height=40 )
            for j in range(2):
                b = Tk.Button( self.frame.interior_bottom, width=20, text="Button B " + str(j) )
                b.grid( row=0, column=j )

            label = Tk.Label(self.frame2.interior, text="Frame to be sychronized", width=20)
            label.pack()
            b = Tk.Button( self.frame2.interior, width=40, text="Button S ")
            b.pack()

            self.frame.set_sync_info( self.frame2.canvas_info )
            self.frame2.set_sync_info( self.frame.canvas_info )

    app = SampleApp()
    app.mainloop()