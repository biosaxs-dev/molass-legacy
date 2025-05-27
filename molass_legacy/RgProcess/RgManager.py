# coding: utf-8
"""
    RgProcess.RgManager.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""

class RgManager:
    def __init__(self, in_folder, sd):
        import multiprocessing as mp
        si = self.get_sharedinfo(in_folder, sd)
        self.process = mp.Process(target=self.run, args=(si,))
        self.process.start()

    def get_sharedinfo(self, in_folder, sd):
        from Processes.SharedInfo import SharedInfo
        si = SharedInfo()
        si.in_folder.set(in_folder)
        si.initilize_buffer(array=sd.intensity_array, ecurve=sd.get_xray_curve())
        return si

    def run(self, si):
        from molass_legacy.KekLib.TkUtils import get_tk_root
        from .RgAnalysis import RgAnalysisDialog

        si.get_buffer_ready()
        root = get_tk_root()

        def show_dialog():
            dialog  = RgAnalysisDialog(root, si)
            dialog.show()
            root.quit()

        root.after(0, show_dialog)
        root.mainloop()
        root.destroy()
