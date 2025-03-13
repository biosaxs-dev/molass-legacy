"""
    from:
        Recipe11.4.Adding Drag and Drop Reordering to a Tkinter Listbox
        https://flylib.com/books/en/2.9.1.230/1/
"""
import tkinter
class DDList(tkinter.Listbox):
    """ A tkinter listbox with drag'n'drop reordering of entries. """
    def __init__(self, master, **kw):
        kw['selectmode'] = tkinter.SINGLE
        tkinter.Listbox.__init__(self, master, kw)
        self.bind('<Button-1>', self.setCurrent)
        self.bind('<B1-Motion>', self.shiftSelection)
        self.curIndex = None
    def setCurrent(self, event):
        self.curIndex = self.nearest(event.y)
    def shiftSelection(self, event):
        i = self.nearest(event.y)
        if i < self.curIndex:
            x = self.get(i)
            self.delete(i)
            self.insert(i+1, x)
            self.curIndex = i
        elif i > self.curIndex:
            x = self.get(i)
            self.delete(i)
            self.insert(i-1, x)
            self.curIndex = i

if __name__ == '__main__':
    tk = tkinter.Tk( )
    length = 10
    dd = DDList(tk, height=length)
    dd.pack( )
    for i in range(length):
        dd.insert(tkinter.END, str(i))
    def show( ):
        ''' show the current ordering every 2 seconds '''
        for x in dd.get(0, tkinter.END):
            print(x),
        print()
    tk.after(2000, show)
    tk.after(2000, show)
    tk.mainloop( )
