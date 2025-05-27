"""
    EmbedDialog.py

    edit this code while keeping the caller code unchanged

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import sys

def open_dialog(callback, *args, **kwargs):
    from molass_legacy.KekLib.OurTkinter import Tk, Dialog

    begin_main_loop = False
    parent = kwargs.pop('parent', None)
    if parent is None:
        parent = Tk.Tk()
        begin_main_loop = True

    class EmbedDialog(Dialog):
        def __init__(self, parent, callback, *args, **wkargs):
            self.callback = callback
            self.args = args
            self.kwargs = wkargs
            Dialog.__init__(self, parent, "EmbedDialog", visible=False)

        def show( self ):
            self._show()

        def body(self, body_frame):
            button = Tk.Button(body_frame, text="Run", command=self.run)
            button.pack(padx=50, pady=30)

        def run(self):
            self.callback(*self.args, **self.kwargs)

    dialog = EmbedDialog(parent, callback, *args, **kwargs)

    def show_dialog():
        dialog.show()
        if begin_main_loop:
            parent.quit()

    if begin_main_loop:
        parent.withdraw()
        parent.after(0, show_dialog)
        parent.mainloop()
    else:
        show_dialog()

if __name__ == '__main__':
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.dirname(this_dir)
    sys.path.append(lib_dir)
    from DevelUtils.LegacyImporters import import_legacy_modules
    import_legacy_modules()

    def test_cb(*args, **kwargs):
        print(*args)
        print(**kwargs)

    open_dialog(test_cb, 1, 2, 3)
