# coding: utf-8
"""
    SvdDataSaver.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
from molass_legacy.KekLib.NumpyUtils import np_savetxt

class SvdDataSaverDialog(Dialog):
    def __init__(self, parent, data_name, svd_results, num_components, location=None):
        self.svd_results = svd_results
        self.default_num = num_components
        self.data_name = data_name
        self.num_components = num_components
        Dialog.__init__(self, parent, "Svd Results Saver for " + data_name, visible=False, location=location)

    def body(self, body_frame):
        tk_set_icon_portable(self)

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        label = Tk.Label(iframe, text='Enter your output folder and press "Save" button.')
        label.pack()

        space = Tk.Frame(iframe, height=10)
        space.pack()

        input_frame = Tk.Frame(iframe)
        input_frame.pack()

        self.out_folder = Tk.StringVar()
        data_type = self.data_name.split(' ')[0].lower()
        default_folder = os.path.join(get_setting('analysis_folder'), 'svd-results' + '/' + data_type).replace('\\', '/')
        self.out_folder.set(default_folder)

        entry = FolderEntry(input_frame, textvariable=self.out_folder, width=60)
        entry.pack(side=Tk.LEFT)

        space = Tk.Frame(input_frame, width=20)
        space.pack(side=Tk.LEFT)

        label = Tk.Label(input_frame, text="number of components")
        label.pack(side=Tk.LEFT)

        self.num_components = Tk.IntVar()
        self.num_components.set(self.default_num)
        spinbox = Tk.Spinbox( input_frame, textvariable=self.num_components,
                                from_=1, to=10, increment=1,
                                justify=Tk.CENTER, width=6 )
        spinbox.pack(side=Tk.LEFT)

        space = Tk.Frame(iframe, height=10)
        space.pack()

        self.status = Tk.StringVar()
        self.status_label = Tk.Label(iframe, textvariable=self.status)
        self.status_label.pack()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Save", width=10, command=self.save, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def show(self):
        self._show()

    def save(self):
        out_folder = self.out_folder.get()

        if not os.path.exists(out_folder):
            mkdirs_with_retry(out_folder)

        x, U, s, V = self.svd_results
        n = self.num_components.get()

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            fig = plt.figure()
            ax = fig.gca()
            axt = ax.twinx()
            ax.plot(U[:,0:n], label='U')
            ax.plot(V[:,0:n], label='V')
            ax.legend()
            plt.show()

        x_name = "wavelengths.dat" if self.data_name.find("UV") >= 0 else "qvector.dat"
        for data, name in zip([x, U[:,0:n], s[0:n], V[:, 0:n]], [x_name, "U.dat", "s.dat", "V.dat"]):
            np_savetxt(os.path.join(out_folder, name), data)

        self.status.set("your svd results have been successfully saved.")
