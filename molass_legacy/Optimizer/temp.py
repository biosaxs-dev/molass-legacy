import tkinter as Tk

def test_it():

    root = Tk.Tk()

    for k, c in enumerate(["A", "B", "C", "D", "E", "F"]):
        label = Tk.Label(root, text=c)
        label.grid(row=0, column=k)

    for k, c in enumerate(["X", "Y", "Z"]):
        label = Tk.Label(root, text=c)
        label.grid(row=1, column=k*2, columnspan=2)

    for k, c in enumerate(["A", "B", "C", "D", "E", "F"]):
        label = Tk.Label(root, text=c)
        label.grid(row=2, column=k)

    root.mainloop()

if __name__ == "__main__":
    test_it()