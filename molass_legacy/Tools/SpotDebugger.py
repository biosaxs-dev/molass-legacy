"""
    SpotDebugger.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, Dialog

parents = []

class SpotDebugger(Dialog):
    def __init__(self, parent, debug_info):
        if parent is None:
            from molass_legacy.KekLib.TkUtils import adjusted_geometry
            parent = Tk.Tk()
            parents.append(parent)
            parent.geometry( adjusted_geometry( parent.geometry() ) )
            parent.withdraw()
            parent.update()

            """
            FIXME:
                _tkinter.TclError: grab failed: another application has grab occurred in ElutionDecomposer.add_other_peaks
            """

        self.debug_info = debug_info
        Dialog.__init__(self, parent, "Spot Debugger", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        button  = Tk.Button(body_frame, text="Debug", command=self.run_debug)
        button.pack()

    def run_debug(self):
        from importlib import reload
        import Tools.DebugCushion
        reload(Tools.DebugCushion)
        from molass_legacy.Tools.DebugCushion import debug_impl
        debug_impl(self.debug_info)
