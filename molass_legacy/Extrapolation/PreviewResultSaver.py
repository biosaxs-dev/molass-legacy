# coding: utf-8
"""
    PreviewResultSaver.py

    Copyright (c) 2018-2021, SAXS Team, KEK-PF
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

class PreviewResultSaverDialog(Dialog):
    def __init__(self, parent, dialog):
        self.parent = parent
        self.dialog = dialog
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

        out_dir = (get_setting( 'analysis_folder' ) + '/preview_results').replace('\\', '/')
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

    def validate(self, notify=True, post_fix=""):
        try:
            self.save_data(post_fix=post_fix)
            if notify:
                MessageBox.showinfo("INFO", "Saved successfully.", parent=self)
            ret = 1
        except Exception as exc:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            MessageBox.showerror("ERROR", "Save failed with the following error\n" + etb.last_line(), parent=self)
            ret = 0
        return ret

    def save_data(self, post_fix=""):
        folder = self.folder.get()  + post_fix
        if not os.path.exists(folder):
            mkdirs_with_retry(folder)

        q = self.dialog.q

        self.save_extra_data(folder)

        for k, result in enumerate(self.dialog.solver_results):
            A, B, Z, E, lrf_info, C = result
            Ae, Be, Ze = E
            D = lrf_info.data
            # TODO: errors for A, B
            print('save_data: shapes=',  A.shape, B.shape, C.shape)
            A_data = np.vstack( [q, A, Ae] ).T
            np_savetxt( folder + '/A%d.dat' % k, A_data )
            ignore_bq = self.dialog.get_ignore_bq(k)

            if lrf_info.need_bq():
                B_data = np.vstack( [q, B, Be] ).T
                np_savetxt( folder + '/B%d.dat' % k, B_data )

            np_savetxt( folder + '/M%d.dat' % k, D )
            np_savetxt( folder + '/C%d.dat' % k, C )

    def save_extra_data(self, folder):
        extra_folder = os.path.join(folder, 'extra')
        if not os.path.exists(extra_folder):
            mkdirs_with_retry(extra_folder)

        dialog = self.dialog

        ranges_txt = os.path.join(extra_folder, 'ranges.txt')
        with open(ranges_txt, "w") as fh:
            fh.write(str([prange.get_fromto_list() for prange in dialog.cnv_ranges]) + "\n")

        sd = dialog.sd
        curve = sd.get_xray_curve()
        elution_txt = os.path.join(extra_folder, 'elution.txt')
        np_savetxt(elution_txt, np.array([curve.y]).T)

        angle_txt = os.path.join(extra_folder, 'angle.txt')
        np_savetxt(angle_txt, np.array([sd.qvector]).T)
