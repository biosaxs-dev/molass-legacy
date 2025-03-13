"""
Python Tkinter Splash Screen

This script holds the class SplashScreen, which is simply a window without
the top bar/borders of a normal window.

The window width/height can be a factor based on the total screen dimensions
or it can be actual dimensions in pixels. (Just edit the useFactor property)

Very simple to set up, just create an instance of SplashScreen, and use it as
the parent to other widgets inside it.

www.sunjay-varma.com
"""

"""
The original code has been taken from
    http://code.activestate.com/recipes/577271-tkinter-splash-screen/
and modified by Masatsuyo Takahashi, KEK-PF
"""
import tkinter      as Tk
import tkinter.ttk  as ttk

class SplashScreen(Tk.Frame):
    def __init__(self, master=None, width=0.8, height=0.6, useFactor=True):
        Tk.Frame.__init__(self, master)
        self.pack(side=Tk.TOP, fill=Tk.BOTH, expand=Tk.YES)

        # get screen width and height
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        w = (useFactor and ws*width) or width
        h = (useFactor and ws*height) or height
        # calculate position x, y
        x = (ws/2) - (w/2) 
        y = (hs/2) - (h/2)
        self.master.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.master.overrideredirect(True)
        self.lift()

def demo(root):
    sp = SplashScreen(root)
    sp.config(bg="#3366ff")

    m = Tk.Label(sp, text="This is a test of the splash screen\n\n\nThis is only a test.\nwww.sunjay-varma.com")
    m.pack(side=Tk.TOP, expand=Tk.YES)
    m.config(bg="#3366ff", justify=Tk.CENTER, font=("calibri", 29))
    
    Tk.Button(sp, text="Press this button to kill the program", bg='red', command=root.destroy).pack(side=Tk.BOTTOM, fill=Tk.X)
    root.mainloop()

if __name__ == '__main__':
    root = Tk.Tk()
    demo(root)
