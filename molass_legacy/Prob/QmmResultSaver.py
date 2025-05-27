# coding: utf-8
"""
    QmmResultSaver.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.TkUtils import rational_geometry
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry, FileEntry
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
from molass_legacy.KekLib.NumpyUtils import np_savetxt
try:
    import molass_legacy.KekLib.CustomMessageBox as MessageBox
except:
    import OurMessageBox as MessageBox

class QmmResultSaverDialog(Dialog):
    def __init__( self, parent, e11n):
        self.parent = parent
        self.e11n = e11n
        self.conc_dependence = get_setting('conc_dependence')

    def show(self):
        title = "Preview Data Saver"
        Dialog.__init__( self, self.parent, title, auto_geometry=False, geometry_cb=self.adjust_geometry )

    def adjust_geometry(self):
        rational_geometry(self, self.parent, 0.5, 0.7)

    def body(self, body_frame):
        set_icon( self )

        label_frame = Tk.Frame(body_frame)
        label_frame.pack( padx=50, pady=10 )

        guide = Tk.Label(label_frame, text='Select a folder to save into and press "OK"')
        guide.pack()

        detail_frame = Tk.Frame(body_frame)
        detail_frame.pack( padx=50, pady=10 )

        out_dir = (get_setting( 'analysis_folder' ) + '/qmm_results').replace('\\', '/')
        row = 0
        label = Tk.Label( detail_frame, text="Result Folder" )
        label.grid( row=row, column=0, sticky=Tk.W )
        self.folder = Tk.StringVar()
        self.folder.set( out_dir )
        folder_entry = FolderEntry( detail_frame, textvariable=self.folder, width=60 )
        folder_entry.grid( row=row, column=1, sticky=Tk.W, padx=10 )

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="Save", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def validate(self, notify=True):
        try:
            self.save_data()
            if notify:
                MessageBox.showinfo("INFO", "Saved successfully.", parent=self)
            ret = 1
        except Exception as exc:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            MessageBox.showerror("ERROR", "Save failed with the following error\n" + etb.last_line(), parent=self)
            ret = 0
        return ret

    def save_data(self):
        folder = self.folder.get()
        if not os.path.exists(folder):
            mkdirs_with_retry(folder)

        e11n = self.e11n
        Cinv = e11n.Cinv
        E = e11n.E
        P = e11n.P
        v = e11n.v
        size = e11n.size
        """
        P = MC⁺
        Pe = qsrt(E*E・C⁺*C⁺)
        """
        EE = E*E
        CC = Cinv*Cinv
        print('v.shape=', v.shape)
        print('EE.shape=', EE.shape)
        print('CC.shape=', CC.shape)

        size_ = P.shape[1] if size is None else size
        Pe = np.sqrt(EE@CC)
        for j in range(size_):
            data = np.vstack( [v, P[:,j], Pe[:,j]] ).T
            np_savetxt( folder + '/A%d.dat' % j, data )
