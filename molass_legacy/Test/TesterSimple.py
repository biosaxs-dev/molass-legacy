"""

    TesterSimple.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog

class TesterSimple(Dialog):
    def __init__(self, parent, in_folder, sd, callback, mapper=None, debug=False):
        self.in_folder = in_folder
        self.sd = sd
        self.callback = callback
        self.mapper = mapper
        self.debug = debug
        Dialog.__init__(self, parent, "TesterSimple", visible=False)

    def show( self ):
        self._show()

    def body(self, body_frame):
        button = Tk.Button(body_frame, text="Run", command=self.run)
        button.pack(padx=50, pady=30)

    def run(self):
        kw_debug = {'debug':self.debug} if self.debug else {}

        if self.mapper is None:
            self.callback(self, self.in_folder, self.sd, **kw_debug)
        else:
            self.callback(self, self.in_folder, self.sd, self.mapper, **kw_debug)

def show_dialog(in_folder, callback, add_mapper=False, analysis_copy=False, debug=False):
    from molass_legacy._MOLASS.SerialSettings import set_setting
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.KekLib.TkUtils import get_tk_root

    set_setting('in_folder', in_folder)

    sp = StandardProcedure()
    if analysis_copy:
        from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
        sd_ = sp.load_old_way(in_folder)
        pre_recog = PreliminaryRecognition(sd_)
        sd = sd_._get_analysis_copy_impl(pre_recog)
        mapper = None
    elif add_mapper:
        sp.load(in_folder)
        sd = sp.get_corrected_sd()      # this method creates a mapper
        mapper = sp.mapper
    else:
        sd = sp.load_old_way(in_folder)
        mapper = None

    root = get_tk_root()

    def show_tester():
        ts = TesterSimple(root, in_folder, sd, callback, mapper=mapper, debug=debug)
        ts.show()
        root.quit()

    root.after(0, show_tester)
    root.mainloop()
    root.destroy()
