"""
    LoaderProcess.DataLoader.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time, sleep
import glob
import numpy as np

class DataLoader:
    def __init__(self, si, rg_queue):
        self.si = si
        self.rg_queue = rg_queue

    def load(self, in_folder, notify=True):
        t0 = time()
        paths = glob.glob(in_folder + '/*.dat')

        for i, path in enumerate(paths):
            print([i], path)
            d = np.loadtxt(path)
            if i == 0:
                shape = (len(paths), *d.shape)
                self.si.initilize_buffer(shape)
                if notify:
                    self.rg_queue.put(-1)   # notify to get ready for the folder
            self.si.set_xr_data(i, d)
            if notify:
                self.rg_queue.put(i)

        if notify:
            self.rg_queue.put(-2)   # notify to get finished for the folder
        print("It took %.3g seconds" % (time()-t0))
