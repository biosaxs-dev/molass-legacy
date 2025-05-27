# coding: utf-8
"""

    ScatteringBaseInvestigator.py

    Copyright (c) 2017-2019, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from CanvasFrame            import CanvasFrame

class InvestigatorDialog( Dialog ):
    def __init__( self, parent, corrector ):
        self.parent     = parent
        self.corrector  = corrector
        self.max_index  = len(self.corrector.qvector)-1

    def show( self ):
        title = "ScatteringBaseInvestigator"
        Dialog.__init__(self, self.parent, title )

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self )

        base_frame = Tk.Frame( body_frame );
        base_frame.pack( expand=1, fill=Tk.BOTH, padx=20, pady=10 )

        figsize = ( 16, 8 ) if is_low_resolution() else ( 16, 8 )
        self.canvas_frame = canvas_frame = CanvasFrame( base_frame, figsize=figsize )
        canvas_frame.pack()

        fig = canvas_frame.fig
        ax1 = fig.add_subplot( 121 )
        ax2 = fig.add_subplot( 122 )
        self.axes = [ ax1, ax2 ]

        self.index_var  = Tk.IntVar()
        self.index_var.set( 0 )
        self.inc_var    = Tk.IntVar()
        self.inc_var.set( 10 )
        spinbox_frame = Tk.Frame( base_frame )
        spinbox_frame.pack()

        label = Tk.Label( spinbox_frame, text="Investigating " )
        label.grid( row=0, column=0 )
        self.spinbox = Tk.Spinbox( spinbox_frame, textvariable=self.index_var,
                                from_=0, to=self.max_index, increment=self.inc_var.get(), 
                                justify=Tk.CENTER, width=6 )
        self.spinbox.grid( row=0, column=1 )
        label = Tk.Label( spinbox_frame, text="-th Q-Plane with spinbox increment" )
        label.grid( row=0, column=2 )
        inc_entry = Tk.Entry( spinbox_frame, textvariable=self.inc_var, justify=Tk.CENTER, width=3 )
        inc_entry.grid( row=0, column=3 )
        label = Tk.Label( spinbox_frame, text=" and plot options " )
        label.grid( row=0, column=4 )
        self.supress_clear = Tk.IntVar()
        self.supress_clear.set( 0 )
        cb = Tk.Checkbutton( spinbox_frame, text="without clearing", variable=self.supress_clear )
        cb.grid( row=0, column=5 )

        self.same_scaling = Tk.IntVar()
        self.same_scaling.set( 1 )
        cb = Tk.Checkbutton( spinbox_frame, text="same scaling", variable=self.same_scaling )
        cb.grid( row=0, column=6 )

        self.zero_base = Tk.IntVar()
        self.zero_base.set( 0 )
        cb = Tk.Checkbutton( spinbox_frame, text="zero base", variable=self.zero_base )
        cb.grid( row=0, column=7 )

        self.index_var.trace( 'w', self.index_var_tracer )
        self.index_var_tracer()
        self.inc_var.trace( 'w', self.inc_var_tracer )
        self.same_scaling.trace( 'w', self.redraw_tracer )
        self.zero_base.trace( 'w', self.redraw_tracer )

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        box = Tk.Frame(self)

        w = Tk.Button(box, text="Back", width=10, command=self.back_, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.back_button = w
        w = Tk.Button(box, text="Next", width=10, command=self.next_, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.next_button = w
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.cancel_button = w

        self.bind("<Escape>", self.cancel)

        box.pack()

    def index_var_tracer( self, *args ):
        try:
            i   = self.index_var.get()
        except:
            # possibly when ""
            return

        if i >= 0 and i <= self.max_index:
            self.draw_a_plane( i )
        else:
            # better be silent
            pass

    def inc_var_tracer( self, *args ):
        try:
            inc = self.inc_var.get()
        except:
            return

        if inc > 0 and inc < self.max_index//2:
            self.spinbox.config( increment=inc )

    def redraw_tracer( self, *args ):
        try:
            i   = self.index_var.get()
        except:
            # possibly when ""
            return

        self.draw_a_plane( max( 0, min( self.max_index, i ) ) )

    def next_( self ):
        try:
            i   = self.index_var.get()
            inc = self.inc_var.get()
        except:
            return
        self.index_var.set( max( 0, min( self.max_index, i + inc ) ) )

    def back_( self ):
        try:
            i   = self.index_var.get()
            inc = self.inc_var.get()
        except:
            return
        self.index_var.set( max( 0, min( self.max_index, i - inc ) ) )

    def draw_a_plane( self, i ):
        plot_closure = self.corrector.correct_a_single_q_plane( i, suppress_update=True, plot_always=True )
        if plot_closure is None:
            return

        self.axes[0].cla()
        if self.supress_clear.get() == 0:
            self.axes[1].cla()

        same_scaling    = self.same_scaling.get() == 1
        zero_base       = self.zero_base.get() == 1
        plot_closure( self.canvas_frame.fig, axes=self.axes, same_scaling=same_scaling, zero_base=zero_base )
        self.canvas_frame.show()
