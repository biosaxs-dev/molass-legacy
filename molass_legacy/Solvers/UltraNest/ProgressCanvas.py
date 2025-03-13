"""
    Solvers.UltraNest.ProressCanvas.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from multiprocessing import Process, Queue

def create_server(nrows, ncols, update_queue, parent=None, caller_pid=None):
    import tkinter
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    class ProressCanvasServer:
        def __init__(self, nrows, ncols, grid_unit=0.2, parent=None):
            self.parent = parent
            fig, ax = plt.subplots(figsize=(ncols*grid_unit, nrows*grid_unit))
            fig.tight_layout()
            self.mpl_canvas = FigureCanvasTkAgg(fig, parent)
            self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
            self.mpl_canvas_widget.pack(fill=tkinter.BOTH, expand=1)
            ax.set_axis_off()
            self.fig = fig
            self.ax = ax
            self.colormap = 'Purples'
            self.parent.protocol("WM_DELETE_WINDOW", self.quit)
            print("server created")
 
        def begin_poll(self, interval=500):
            print("begin_poll")
            self.interval = interval
            self.parent.after(interval, self.poll)

        def poll(self):
            print("polling")
            while True:
                try:
                    data = update_queue.get(False)
                    if np.isscalar(data) and data == -1:
                        # stop polling
                        print("stop polling")
                        self.quit()
                        return
                    else:
                        self.update(data)
                except:
                    break

            self.parent.after(self.interval, self.poll)

        def update(self, data):
            # print("updating data=", data)
            self.ax.imshow(data, cmap=self.colormap)
            self.mpl_canvas.draw_idle()

        def quit(self):
            print("ProressCanvasServer: quit")
            self.parent.quit()

    begin_loop = False
    if parent is None:
        parent = tkinter.Tk()
        caller_text = "" if caller_pid is None else " from caller pid=%d" % caller_pid
        parent.title("Proress Canvas Server pid=%d%s" % (os.getpid(), caller_text))
        begin_loop = True
    server = ProressCanvasServer(nrows, ncols, parent=parent)
    server.begin_poll()
    if begin_loop:
        parent.mainloop()

class ProressCanvasClient:
    def __init__(self, nrows, ncols):
        self.update_queue = Queue()
        # the server must run as a separate process to avoid matplotlib "main thread is not in main loop" problem
        args = (nrows, ncols, self.update_queue)
        kwargs = dict(caller_pid=os.getpid())
        self.server_process = Process(target=create_server, args=args, kwargs=kwargs)
        self.server_process.start()

    def update(self, data):
        self.update_queue.put(data)
    
    def quit(self):
        self.update_queue.put(-1)
        self.server_process.join()

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
    from time import sleep

    shape = (10, 40)

    canvas = ProressCanvasClient(*shape)
    sleep(1)
    data = np.zeros(shape)
    data[5,:] = 1
    data[:,20] = 1
    canvas.update(data)

    sleep(1)
    data = np.zeros(shape)
    canvas.update(data)

    sleep(1)
    canvas.quit()
