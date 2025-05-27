# coding: utf-8
"""
    MethodAdjusterFrame.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk

class MethodAdjusterFrame(Tk.Frame):
    def __init__(self, parent):
        Tk.Frame.__init__(self, parent)
        self.body(self)

    def body(self, body_frame):
        space = Tk.Frame(body_frame, height=300)
        space.pack()
        label = Tk.Label(body_frame, text="In Preparation", font=('', 40), fg='gray')
        label.pack()

    def body_trash(self, body_frame):
        from PIL import Image, ImageTk
        dir_, _ = os.path.split(__file__)
        path = os.path.join(dir_, 'MethodFileAdjuster.png')
        img = Image.open(path)
        img = ImageTk.PhotoImage(img)
        canvas = Tk.Canvas(body_frame, bg='black', width=400, height=300)
        canvas.pack()
        canvas.create_image(0, 0, image=img, anchor=Tk.NW)
