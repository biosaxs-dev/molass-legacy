"""
    Solvers.UltraNest.GridspecLayoutMpl.py

    adapted from ipywidgets.GridspecLayout

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

class GridspecLayoutMpl:
    def __init__(self, nrows, ncols):
        from Solvers.UltraNest.ProgressCanvas import ProressCanvasClient
        self.nrows = nrows
        self.ncols = ncols
        self.data = np.zeros((nrows,ncols))
        self.canvas = ProressCanvasClient(nrows,ncols)
        self.canvas.update(self.data)

    def close(self):
        self.canvas.quit()

    def _get_indices_from_slice(self, row, column):
        "convert a two-dimensional slice to a list of rows and column indices"
        # borrowd from ipywidgets.GridspecLayout

        if isinstance(row, slice):
            start, stop, stride = row.indices(self.nrows)
            rows = range(start, stop, stride)
        else:
            rows = [row]

        if isinstance(column, slice):
            start, stop, stride = column.indices(self.ncols)
            columns = range(start, stop, stride)
        else:
            columns = [column]

        return rows, columns

    def __setitem__(self, key, value):
        row, column = key

        rows, columns = self._get_indices_from_slice(row, column)
        self.data[rows,columns] = value
        self.canvas.update(self.data)

    def __getitem__(self, key):
        rows, columns = self._get_indices_from_slice(*key)
        return self.data[rows,columns]

if __name__ == '__main__':
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))
    from time import sleep
    grid = GridspecLayoutMpl(6,10)
    grid[3,:] = 1
    grid[:,5] = 1
    grid[4,4] = 1
    print(grid.data)
    for i in range(10):
        sleep(1)
        grid[5,i] = 0.5
    sleep(3)
    grid.close()