"""
    SSDC.FolderEntryDialog.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry

class FolderEntryDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, title="Input Folders Entry", visible=False)
        self.parent = parent
        self.applied = False
        self.info_list = None
    
    def show(self):
        self._show()
    
    def body(self, body_frame):
        self.entry1 = FolderEntry(body_frame, width=70)
        self.entry1.grid(row=0, column=0, padx=5, pady=5)
        self.entry2 = FolderEntry(body_frame, width=70)
        self.entry2.grid(row=1, column=0, padx=5, pady=5)
        nc_frame = Tk.Frame(body_frame)
        nc_frame.grid(row=2, column=0, padx=5, pady=5)
        self.num_components = Tk.IntVar()
        self.num_components.set(4)
        label = Tk.Label(nc_frame, text="Number of Components:")
        label.grid(row=0, column=0)
        self.spinbox = Tk.Spinbox(nc_frame, textvariable=self.num_components,
                                  from_=1, to=7, increment=1,
                                  justify=Tk.CENTER, width=6)        
        self.spinbox.grid(row=0, column=1)

        # self.entry1.variable.set('D:/PyTools/Data/20210727/data01')
        # self.entry2.variable.set('D:/PyTools/Data/20210727/data02')

        # self.entry1.variable.set('D:/PyTools/Data/20220716/BSA_201')
        # self.entry2.variable.set('D:/PyTools/Data/20220716/BSA_202')

        # self.entry1.variable.set('D:/PyTools/Data/20220716/FER_OA_301')
        # self.entry2.variable.set('D:/PyTools/Data/20220716/FER_OA_302')

        # self.entry1.variable.set('D:/PyTools/Data/20230706/ALD_OA001')
        # self.entry2.variable.set('D:/PyTools/Data/20230706/ALD_OA002')

        self.entry1.variable.set('D:/PyTools/Data/20230706/BSA001')
        self.entry2.variable.set('D:/PyTools/Data/20230706/BSA002')

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Test", width=10, command=self.test)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def apply(self):
        self.applied = True

    def test(self, devel=True):
        info_list = self.get_info_list()
        num_components = self.num_components.get()
        if devel:
            from importlib import reload
            import SSDC.SsdcAnalysis
            reload(SSDC.SsdcAnalysis)
        from SSDC.SsdcAnalysis import SsdcAnalysis
        try:
            analysis = SsdcAnalysis(self.parent, info_list, num_components)
            analysis.show()
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "Error in SsdcAnalysis: ", n=10)

    def get_folder_list(self):
        ret_list = []
        for entry in [self.entry1, self.entry2]:
            ret_list.append(entry.variable.get())
        print(ret_list)
        return ret_list
    
    def get_info_list(self):
        if self.info_list is None:
            from molass_legacy.Batch.LiteBatch import LiteBatch        
            info_list = []
            batch = LiteBatch()     # task: 
            for folder in self.get_folder_list():               
                lrf_src = batch.get_lrf_source(in_folder=folder)
                info_list.append((folder, lrf_src))
            self.info_list = info_list
        return self.info_list