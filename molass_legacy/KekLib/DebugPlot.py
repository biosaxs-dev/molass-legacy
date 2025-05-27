"""
    DebugPlot.py

    work-around to debug with matplotlib under tkinter

    Copyright (c) 2018-2024, Masatsuyo Takahashi, KEK-PF
"""
import matplotlib.pyplot as _plt
from matplotlib.pyplot import FuncFormatter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
import matplotlib
import gc
# from CallStack import CallStack

def _reset():
    global dp, dp_list
    dp = None
    dp_list = []

_reset()

kill_button = False
cancel_button = True
abort_button = True
aborting = False
g_ok_text = "OK"
g_ok_only = False

def set_global_opts(**kwargs):
    global kill_button, cancel_button, abort_button, g_ok_only, g_ok_text
    ok_only = kwargs.pop('ok_only', False)
    g_ok_text = kwargs.pop('ok_text', "OK")

    if ok_only:
        g_ok_only = True
        kill_button = False
        cancel_button = False
        abort_button = False
    else:
        kill_button = kwargs.pop('kill_button', False)
        cancel_button = kwargs.pop('cancel_button', True)

g_parent = None
threaded_parent = None

def set_plot_env( parent=None, sub_parent=None ):
    global g_parent
    if g_parent is None:
        if parent is None:
            parent = sub_parent
        if parent is None:
            from molass_legacy.KekLib.TkUtils import adjusted_geometry
            parent = Tk.Tk()
            parent.geometry( adjusted_geometry( parent.geometry() ) )
            parent.withdraw()
            parent.update()
        g_parent = parent

def get_parent():
    if threaded_parent is not None:
        return threaded_parent

    set_plot_env()
    if len(dp_list) > 0:
        parent = dp_list[-1].parent
    else:
        parent = g_parent
    return parent

def get_dp():
    if len(dp_list) == 0:
        parent = get_parent()
        dp = DebugPlot(parent)
        dp_list.append(dp)
    return dp_list[-1]

def push(**kwargs):
    parent = kwargs.pop('parent', None)
    if parent is None:
        parent = get_parent()
    dp_ = DebugPlot(parent, **kwargs)
    dp_list.append(dp_)
    return dp_

def pop():
    global aborting
    dp_ = dp_list.pop()
    _plt.close(dp_.fig)
    if aborting:
        aborting = False
        assert False

class Dp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __enter__(self):
        push(**self.kwargs)
    def __exit__(self, exc_type, exc_value, traceback):
        pop()

def debug_plot_close():
    # not yet successful
    if dp is not None:
        dp.destroy()

class DebugPlot( Dialog ):
    def __init__(self, parent,
                 scrollable=False,
                 window_title=None,
                 guide_message=None,
                 ok_only=None,
                 ok_text=None,
                 button_spec=None,
                 **kwargs):
        self.grab = 'local'     # used in grab_set
        self.parent = parent
        self.scrollable = scrollable
        self.mpl_canvas = None
        self.fig = None
        self.ax = None
        self.window_title = window_title
        self.ok_only = ok_only
        self.ok_text = ok_text
        self.button_spec = button_spec
        self.kwargs = kwargs
        self.guide_message = guide_message
        """
            Call to self.body must be delayed until the figure
            for FigureCanvasTkAgg become available.
            Be aware that this is not yet a Tk widget.
            E.g., you can't Dialog.destroy this instance.
        """

    def figure( self, *args, **kwargs ):
        # assert self.fig is None
        if self.fig is not None:
            # self.fig.close()
            self.fig.clf()

        # print(CallStack())
        self.fig = _plt.figure( *args, **kwargs )
        self.ax = None
        self._do_init(title=self.window_title)
        # in this case, there is no need to _do_init in self.show
        return self.fig

    def _do_init(self, title=None, visible=False, block=False):
        if title is None:
            title = "DebugPlot"
        Dialog.__init__( self, self.parent, title, visible=visible, block=block )

    def _get_fig(self):
        if self.fig is None:
            # import traceback
            # traceback.print_stack()
            self.fig = _plt.gcf()
        return self.fig

    def _get_ax(self):
        if self.ax is None:
            self.ax = self._get_fig().gca()
        return self.ax

    def show( self, block=True, pause=None ):
        """
        There are following two cases.
            Case (1)                Case (2)
            dp = DebugPlot(root)    dp = DebugPlot(root)
                                    fig = plt.figure()
                                    ax = fig.gca()
            plt.plot(...)           ax.plot()
            plt.show()              plt.show()
        """
        if self.mpl_canvas is None:
            # Case (1)
            # fig should be obtained by self._get_fig(), i.e. _plt.gcf(), in self.body
            self._do_init(title=self.window_title)
        else:
            # Case (2)
            # self._do_init() has been already done by self.figure
            pass

        self.mpl_canvas.draw()
        self.applied =False
        try:
            self._show(block=block, pause=pause)
        except:
            # _tkinter.TclError: bad window path name ".!debugplot"
            # ignore this exception for it seems harmless
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "DebugPlot._show: ")
        return self.applied

    def body(self, body_frame):     # overrides Dialog.body
        self.body_frame = body_frame
        if self.scrollable:
            from ScrolledFrame import ScrolledFrame
            button_frame = Tk.Frame(self)
            button_frame.pack(side=Tk.BOTTOM)
            sframe = ScrolledFrame(body_frame)
            bframe = sframe.interior
            sframe.pack(side=Tk.BOTTOM)
        else:
            bframe = body_frame
            button_frame = None
        cframe = Tk.Frame(bframe)
        cframe.pack()
        if self.guide_message is not None:
            label = Tk.Label(bframe, text=self.guide_message, bg="white")
            label.pack(side=Tk.TOP, fill=Tk.X)
        
        # figure should have been created before this call
        # (and before any creation of axes belonging to the figure)
        self.mpl_canvas = FigureCanvasTkAgg( self._get_fig(), cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.toolbar = NavigationToolbar( self.mpl_canvas, cframe )
        self.toolbar.update()
        if button_frame is None:
            button_frame = Tk.Frame(body_frame)
            button_frame.pack()
        self.my_buttonbox(button_frame)

    def draw(self):
        self.mpl_canvas.draw()

    def add_frame(self):
        frame = Tk.Frame(self.body_frame)
        frame.pack(side=Tk.LEFT)
        return frame

    def buttonbox(self):
        # nullify the default
        pass

    def my_buttonbox(self, frame):
        box = frame

        if self.button_spec is None:
            ok_button = True
            ok_text_ = g_ok_text
            if self.ok_text is not None:
                ok_text_ = self.ok_text

            if self.ok_only is not None and self.ok_only:
                # task: using globals is not desirable
                global cancel_button, abort_button, kill_button
                cancel_button = False
                abort_button = False
                kill_button = False
        else:
            ok_button = self.button_spec[0]
            if type(ok_button) is str:
                ok_text_ = ok_button
                ok_button = True
            elif type(ok_button) is bool:
                if ok_button:
                    ok_text_ = g_ok_text
            else:
                assert False, "unknown type of ok_button"
            if len(self.button_spec) > 1:
                cancel_button = self.button_spec[1]
            else:
                cancel_button = False
            if len(self.button_spec) > 2:
                abort_button = self.button_spec[2]
            else:
                abort_button = False
            if len(self.button_spec) > 3:
                kill_button = self.button_spec[3]
            else:
                kill_button = False

        if ok_button:
            w = Tk.Button(box, text=ok_text_, width=10, command=self.ok, default=Tk.ACTIVE)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

        if cancel_button:
            w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

        if abort_button:
            w = Tk.Button(box, text="Abort", width=10, command=self.abort)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

        if kill_button:
            w = Tk.Button(box, text="Kill", width=10, command=self.kill)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

        extra_specs = self.kwargs.get("extra_button_specs", [])
        for name, command in extra_specs:
            width = max(10, len(name)+2)
            w = Tk.Button(box, text=name, width=width, command=command)
            w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def destroy(self):
        '''Destroy the window'''
        # print( "overriden destroy" )
        r"""
        TODO fix: there may be an instance of DebugPlot which has not been destroyed properly

          File "F:\PyTools\pytools-1_2_2-develop\lib\KekLib\OurTkinter.py", line 173, in ok
            self.withdraw()
          File "c:\program files\python36\lib\tkinter\__init__.py", line 1992, in wm_withdraw
            return self.tk.call('wm', 'withdraw', self._w)
        _tkinter.TclError: bad window path name ".!debugplot"

        SEE ALSO: 
        https://ja.osdn.net/projects/pylaf/scm/hg/pylaf/blobs/tip/src/pylafiii/mplext.py
        """
        # print( "before unbind", self.bind("<Destroy>") )
        self.unbind("<Destroy>")    # don't know whether this is the right way
        # print( "after unbind", self.bind("<Destroy>") )
        self.initial_focus = None
        Tk.Toplevel.destroy(self)

        # to avoid "RuntimeError: main thread is not in main loop"
        # see also OurTkinter.Dialog
        gc.collect()

    def apply(self):
        self.applied = True

    def abort(self):
        from tkinter import messagebox
        global aborting

        ret = messagebox.askokcancel("Abort Confirmation", "Are you sure to abort DebugPlot?")
        if ret:
            aborting = True
            self.cancel()

    def kill(self):
        import sys
        print('kill')
        self.parent.destroy()
        sys.exit()

def debug_plot_ok():
    dp = get_dp()
    return dp.applied

def figure( *args, **kwargs ):
    dp = get_dp()
    return dp.figure( *args, **kwargs )

def gcf():
    dp = get_dp()
    return dp._get_fig()

def clf():
    dp = get_dp()
    return dp._get_fig().clf()

def cla():
    dp = get_dp()
    return dp._get_ax().cla()

def gca():
    fig = figure()
    return fig.gca()

def plot( *args, **kwargs ):
    return _plt.plot( *args, **kwargs )

def show( block=True, pause=None ):
    dp = get_dp()
    ret = dp.show( block=block, pause=pause )
    return ret

def legend():
    return gcf().legend()

def tight_layout():
    return gcf().tight_layout()

def annotate( *args, **kwargs ):
    dp = get_dp()
    return dp._get_ax().annotate( *args, **kwargs )

def subplots_adjust( *args, **kwargs ):
    return _plt.subplots_adjust( *args, **kwargs )

def subplots( *args, **kwargs ):
    dp = get_dp()
    fig, axes = _plt.subplots( *args, **kwargs )
    dp.fig = fig
    dp.axes = axes
    dp._do_init()       # required to enable tk methods, e.g., dp.geometry()
    return fig, axes

def axes( *args, **kwargs ):
    return _plt.axes( *args, **kwargs )

def close():
    _plt.close()

def update():
    """
    required in animation to get button events, etc.
    """
    dp.parent.update()

def setp(*args, **kwargs):
    _plt.setp(*args, **kwargs)

class DialogWrapper:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        set_global_opts(ok_only=True, ok_text="Close")
        push(**self.kwargs)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        show()
        pop()
        set_global_opts(ok_only=False)

def get_current_dialog():
    return dp

def switch_backend(*args):
    _plt.switch_backend(*args)

"""
    see Selective.PropOptimizerUtils.py for usage
"""
def exec_in_threaded_mainloop(closure):
    global threaded_parent

    threaded_parent = Tk.Tk()
    threaded_parent.withdraw()
    threaded_parent.after(0, closure)
    threaded_parent.mainloop()

def quit_threaded_mainloop():
    global threaded_parent

    threaded_parent.quit()
    threaded_parent = None
