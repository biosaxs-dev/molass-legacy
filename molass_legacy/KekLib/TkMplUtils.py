"""
    TkMplUtils.py

    leanred from Cody AI on 2024-02-02
"""

def adjust_the_tkframe_size(fig, canvas_widget, tkframe):
    """
    Adjust the size of the tkinter frame to fit the figure.
    We need this because this synchronization is not yet supported by FigureCanvasTkAgg.
    """
    w, h = fig.canvas.get_width_height()
    print("adjust_the_tkframe_size: w, h =", w, h)
    tkframe.config(width=w, height=h)
    canvas_widget.config(width=w, height=h)