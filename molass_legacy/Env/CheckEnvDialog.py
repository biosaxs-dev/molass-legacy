"""
    CheckEnvDialog.py

    Copyright (c) 2019-2022, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText
from .EnvMessages import get_env_messages

class CheckEnvDialog(Dialog):
    def __init__(self, parent):
        self.grab = 'local'     # used in grab_set
        self.parent = parent
        Dialog.__init__( self, parent, "Environment Info", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        iframe = Tk.Frame(body_frame)
        iframe.pack(side=Tk.LEFT, padx=10, pady=5)

        self.info_text = ReadOnlyText(iframe, width=80, height=10)
        self.info_text.pack()

        for line in get_env_messages():
            self.info_text.insert(Tk.END, line)

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()

        self.ok_btn = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        self.ok_btn.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
