"""
    DecompDummyDialog.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, Dialog

class DecompDummyDialog(Dialog):
    def __init__(self, parent, dialog, model):
        self.dialog = dialog
        self.model = model
        Dialog.__init__(self, parent, "DecompDummyDialog", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        from .DecompEditorFrame import DecompEditorFrame
        dialog = self.dialog
        frame = DecompEditorFrame(body_frame, dialog, dialog.sd, dialog.mapper, dialog.corbase_info, self.model)
        frame.pack()

def decompose(caller):
    dialog = caller.dialog
    model = dialog.get_current_frame().model
    dummy_dialog = DecompDummyDialog(dialog.parent, dialog, model)
    dummy_dialog.show()
