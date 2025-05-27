# coding: utf-8
"""

    LinkedToplevel.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""

from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.TkUtils import adjusted_geometry, split_geometry, join_geometry

class LinkedToplevel(Tk.Toplevel):
    def __init__(self, root, body=None):
        self.root = root
        self.built = False
        self.body = body
        Tk.Toplevel.__init__(self)

    def show(self):
        if not self.built:
            if self.body is not None:
                self.body(self, self.body_frame, self.navi_frame)
                self.built = True
        self.deiconify()

    def link(self, prev=None, next=None, body_kwargs=dict(width=400, height=300), navi_kwargs={}):
        self._prev = prev
        self._next = next

        self.body_frame = Tk.Frame(self, **body_kwargs)
        self.body_frame.pack(fill=Tk.BOTH, expand=1)
        self.navi_frame = Tk.Frame(self, **navi_kwargs)
        self.navi_frame.pack(fill=Tk.BOTH, expand=0)

        if self._prev is None:
            button = Tk.Button(self.navi_frame, text="◀ Exit", command=self.exit)
        else:
            button = Tk.Button(self.navi_frame, text="◀ Back", command=self._show_prev)
        button.pack(side=Tk.LEFT, padx=5, pady=5)
        self._prev_button = button

        if self._next is None:
            button = None
        else:
            button = Tk.Button(self.navi_frame, text="▶ Next", command=self._show_next)
            button.pack(side=Tk.RIGHT, padx=5)
        self._next_button = button
        self.update_geometry()
        self.restore_geometry()
        self.withdraw()

    def update_geometry(self):
        self.update()
        self.geometry_save = adjusted_geometry(self.geometry())

    def restore_geometry(self, invoked_geometry=None):
        if invoked_geometry is None:
            geometry = self.geometry_save
        else:
            _, _, x, y = split_geometry(invoked_geometry)
            w, h, _, _ = split_geometry(self.geometry_save)
            geometry = join_geometry(w, h, x, y)

        self.geometry(geometry)

    def rename_navi_buttons(self, names):
        for name, button in zip(names, [self._prev_button, self._next_button]):
            if name is None:
                continue
            button.config(text=name)

    def exit(self):
        toplevels = []
        next_ = self._next
        while True:
            if next_ is None:
                break
            toplevels.append(next_)
            next_ = next_._next

        for w in toplevels:
            w.destroy()

        self.destroy()
        self.root.quit()

    def _show_prev(self):
        geometry = self.geometry()
        self.withdraw()
        geometry = self.geometry()
        self._prev.show()
        self._prev.restore_geometry(invoked_geometry=geometry)

    def _show_next(self):
        geometry = self.geometry()
        self.withdraw()
        self._next.show()
        self._next.restore_geometry(invoked_geometry=geometry)
