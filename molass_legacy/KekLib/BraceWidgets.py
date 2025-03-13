"""
    BraceWidgets.py

    Copyright (c) 2022-2023, Masatsuyo Takahashi
"""
import os
import sys
import tkinter as Tk
from PIL import Image, ImageTk

# ClosingBrace or RightBrace
class ClosingBrace(Tk.Label):
    def __init__( self, parent, **kwargs ):
        dir_, _ = os.path.split(__file__)
        image = Image.open(os.path.join( dir_, "closing_brace.png"))
        w, h = image.size
        w_ = kwargs.pop("width", w)
        h_ = kwargs.pop("height", h)
        self.image = ImageTk.PhotoImage(image.resize((w_, h_)))
        Tk.Label.__init__( self, parent, image=self.image, compound=Tk.LEFT, **kwargs )

if __name__ == '__main__':
    root = Tk.Tk()
    frame = Tk.Frame( root )
    frame.pack( padx=30, pady=10 )
    brace = ClosingBrace(frame, height=50)
    # brace = ClosingBrace(frame)
    brace.pack()

    root.mainloop()
