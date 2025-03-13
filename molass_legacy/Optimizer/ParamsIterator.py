"""
    Optimizer.ParamsIterator.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import molass_legacy.KekLib.DebugPlot as plt

def iterator_test_from_dialog(caller):
    print("ParamsIterator.iterator_test_from_dialog")
    fv_list, xmax = caller.canvas.get_fv_array()
    fvvect = np.array([rec[1] for rec in fv_list])
    create_iterator(fvvect, debug=True)

def create_iterator(fvvect, debug=False):
    print("ParamsIterator.create_iterator: fvvect=", fvvect)
    last_fv = None
    iter_list = []
    for i in range(len(fvvect)-1):
        k = np.argmin(fvvect[0:i+1])
        if last_fv is None or fvvect[k] < last_fv:
            last_fv = fvvect[k]
            iter_list.append(k)

    if debug:
        x = np.arange(len(fvvect))
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.set_title("create_iterator: debug")
            ax.plot(x, fvvect)
            ax.plot(x[iter_list], fvvect[iter_list], "ro")
            fig.tight_layout()
            plt.show()
    
    return iter_list