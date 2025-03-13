"""
    ListingFrame.py

    adapted from DDList
        at
            Recipe11.4.Adding Drag and Drop Reordering to a Tkinter Listbox
            https://flylib.com/books/en/2.9.1.230/1/        

    Copyright (c) 2021-2022, Masatsuyo Takahashi, KEK-PF
"""
import tkinter as Tk

class ListingFrame(Tk.Frame):
    """ A frame with drag'n'drop reordering of entries. """
    def __init__(self, parent, **kw):
        self.numbering = kw.pop('numbering', False)
        self.debug = kw.pop('debug', False)
        Tk.Frame.__init__(self, parent, **kw)

        self.bind('<Button-1>', self.set_source)
        self.bind('<B1-Motion>', self.move_to_target)

        if self.numbering:
            self.number_labels = []
            self.widget_col = 1
        else:
            self.widget_col = 0
        self.pair_list = []
        self.h = None

    def insert(self, variable, widget):
        row=len(self.pair_list)
        if self.numbering:
            label = Tk.Label(self, text=str(row+1))
            label.grid(row=row, column=0, padx=5)
            self.number_labels.append(label)
        widget.grid(row=row, column=self.widget_col)
        widget.bind('<ButtonPress-1>', self.set_source)
        widget.bind('<B1-Motion>', self.move_widget_image)
        widget.bind('<ButtonRelease-1>', self.move_to_target)
        self.pair_list.append((variable, widget))

    def update_bind_geometry(self):
        self.update()
        self.width = self.winfo_reqwidth()
        self.height = self.winfo_reqheight()
        self.h = self.height/len(self.pair_list)
        if self.debug:
            print(self.width, self.height, self.h)

    def get_widget_index(self, event):
        rootx = self.winfo_rootx()
        rooty = self.winfo_rooty()
        x = event.x_root - rootx
        y = event.y_root - rooty
        i = int(y/self.h)
        if self.debug:
            print("get_widget_index", x, y, i)
        return i

    def set_source(self, event):
        self.source = self.get_widget_index(event)
        if self.debug:
            print("set_source", self.current)

    def move_widget_image(self, event):
        pass

    def move_to_target(self, event):
        target = self.get_widget_index(event)
        if self.debug:
            print("move_to_target", self.source, target)
        self.swap(self.source, target)

    def swap(self, i, j):
        if i == j:
            return

        source_pair = self.pair_list[i]
        target_pair = self.pair_list[j]

        self.pair_list[j] = source_pair
        source_pair[1].grid(row=j, column=self.widget_col)

        self.pair_list[i] = target_pair
        target_pair[1].grid(row=i, column=self.widget_col)

        self.update()

        if self.debug:
            for pair in self.pair_list:
                print(pair[0].get())

    def get_variables(self):
        return [pair[0] for pair in self.pair_list]
