"""
    ProgressMinDialog.py

    Copyright (c) 2018-2024, Masatsuyo Takahashi, KEK-PF
"""
import sys
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkSupplements import set_icon

class ProgressMinDialog(Dialog):
    def __init__(self, parent, title="ProgressMinDialog", message=None, num_steps=10, length=200,
                    progress_cb=None, geometry_info=None, cancelable=False, interval=500,
                    visible=True):
        self.parent = parent
        self.num_steps = num_steps
        self.length = length
        self.canceled = False
        self.message = message
        self.progress_cb = progress_cb
        self.geometry_info = geometry_info
        self.cancelable = cancelable
        self.interval = interval
        auto_geometry = geometry_info is None
        Dialog.__init__( self, self.parent, title, auto_geometry=auto_geometry, visible=visible)

    def show(self):
        """
        used when self has been constructed with visible=False
        """
        self._show()

    def body(self, body_frame):   # overrides Dialog.body
        set_icon(self)
        """
            https://stackoverflow.com/questions/7310511/how-to-create-downloading-progress-bar-in-ttk
        """
        if self.message is not None:
            label = Tk.Label(self, text=self.message)
            label.pack()

        self.mpb = ttk.Progressbar(body_frame,orient ="horizontal", length=self.length, mode="determinate")
        self.mpb.pack(padx=5, pady=5)
        self.mpb["maximum"] = self.num_steps
        self.update()

        if self.geometry_info is not None:
            self.geometry(self.geometry_info)
            self.update()

        self.step = 0
        self.refresh()

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        if self.cancelable:
            w = Tk.Button(box, text="Cancel", width=10, command=lambda: self.cancel(ask=True) )
            w.pack(side=Tk.LEFT, padx=5, pady=5)

    def cancel(self, ask=False):
        if ask:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            yn = MessageBox.askyesno( "Cancel comfirmation",
                "Are you sure to cancel this process?", parent=self )
            if yn:
                pass
            else:
                return

        self.canceled = True
        super(ProgressMinDialog, self).cancel()

    def refresh(self):
        self.mpb["value"] = self.step
        if self.progress_cb is None:
            self.step += 1
        else:
            self.step = self.progress_cb()

        if self.step < 0:
            self.cancel()
        elif self.step >= self.num_steps:
            self.ok()
        else:
            self.after( self.interval, self.refresh )
            self.update()

def run_with_progress(parent, proc, max_iter=10, title=None, on_return=None, debug=True):
    """
    TODO:
        make this cancelable for cases where it takes a long time.
    """
    import queue
    import threading
    from molass_legacy.KekLib.TkUtils import split_geometry
    if debug:
        import logging
        logger = logging.getLogger(__name__)

    w, h, x, y = split_geometry( parent.winfo_geometry() )
    geometry_info = "+%d+%d" % (parent.winfo_rootx() + int(w*0.7), parent.winfo_rooty() + int(h*0.8))

    exe_queue = queue.Queue()

    counter = [0]
    ret_info = [None, None]
    def progress_cb():
        # print('progress_cb:')
        error_info = None
        try:
            info = exe_queue.get(block=False)
            if debug:
                logger.info('progress_cb:', info[0])
            if info[0] == 1:
                ret_info[0] = info[1]
                if on_return is not None:
                    on_return(ret_info[0])
                counter[0] = max_iter + 1
            elif info[0] == 0:
                counter[0] = info[1] + 1
            else:
                # close the progressbar
                counter[0] = -1
                ret_info[1] = info[1]
        except:
            pass
        return counter[0]

    def thread_proc():
        try:
            ret = proc(exe_queue)
            exe_queue.put([1, ret])
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            exe_queue.put([-1, etb])

    thread = threading.Thread( target=thread_proc, name='ThreadInProgress', args=[] )
    thread.start()

    progress = ProgressMinDialog(parent,
                    title=title,
                    num_steps=max_iter + 1,
                    progress_cb=progress_cb, geometry_info=geometry_info,
                    visible=False )

    progress.show()

    thread.join()

    ret_tuple, error_info = ret_info
    return ret_tuple, error_info
