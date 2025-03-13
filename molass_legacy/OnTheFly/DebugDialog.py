"""
    OnTheFly.DebugDialog.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
if __name__ == '__main__':
    import os
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)

from molass_legacy.KekLib.OurTkinter import Tk, Dialog

class DebugDialog(Dialog):
    def __init__(self, parent=None, debug_info=None):
        if parent is None:
            from molass_legacy.KekLib.TkUtils import get_tk_root
            parent = get_tk_root()
        self.parent = parent
        self.debug_info = debug_info
        Dialog.__init__(self, parent, "Debug Dialog", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        label = Tk.Label(body_frame,
                            text="Tell the programmer if you are seeing this dialog.",
                            bg="white", fg="orange", font=('', 16))
        label.pack(padx=20, pady=10)
 
        run_button = Tk.Button(body_frame, text="Run", command=self.run)
        run_button.pack()

    def run(self):
        from importlib import reload
        import OnTheFly.DebugImpl
        reload(OnTheFly.DebugImpl)
        from OnTheFly.DebugImpl import debug_impl
        debug_impl(self)

if __name__ == '__main__':
    dialog = DebugDialog()
    dialog.show()
