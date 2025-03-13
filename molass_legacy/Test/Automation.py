"""
    Automation.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkCustomWidgets import FileEntry

class AutomationDialog(Dialog):
    def __init__(self, parent, parent_dialog):
        self.parent_dialog = parent_dialog
        self.logger = logging.getLogger(__name__)
        Dialog.__init__(self, parent, title="Automation", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        label = Tk.Label(body_frame, text="Select your automation script")
        label.pack()
        filepath_frame = Tk.Frame(body_frame)
        filepath_frame.pack()
        path_label = Tk.Label(filepath_frame, text="Script File Path: ")
        path_label.pack(side=Tk.LEFT)
        self.scriptfile_path = Tk.StringVar()
        homedir = os.path.abspath(os.path.dirname(__file__) + r"/../..")
        self.scriptfile_path.set(os.path.join(homedir, r"test\automation.py"))
        file_entry = FileEntry(filepath_frame, textvariable=self.scriptfile_path, width=60, on_entry_cb=self.on_entry )
        file_entry.pack(side=Tk.LEFT)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Run", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def on_entry(self):
        pass

    def ok(self):
        self.run()
        self.cancel()

    def run(self):
        from molass_legacy.KekLib.ImportUtils import import_module_from_path
        from molass_legacy.KekLib.TkTester import TestClient

        file_path = self.scriptfile_path.get()
        mod = import_module_from_path("Runner", file_path)
        client = TestClient(self.parent_dialog, mod.run)
        print("Done")
