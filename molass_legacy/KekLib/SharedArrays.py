# coding: utf-8
"""
    SharedArrays.py

    Creating multiple shared memories seems unstable as of
    Python 3.8.0 + numpy 1.17.4 + matplotlib 3.1.2

    Therefore, this module uses only one shared memory at a time,
    which contains multiple numpy arrays.

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import copy
import numpy as np

from multiprocessing.shared_memory import SharedMemory

NBYTES_FLOAT64 = 8

class SharedArrays:
    def __init__(self, arrays=None, tuples=None, name=None):
        if arrays is None:
            assert tuples is not None
            total_size = np.sum([np.prod(shape) * NBYTES_FLOAT64 for shape, dtype in tuples])
            existing_shm = SharedMemory(name=name, size=total_size)
            self.arrays = []
            self.tuples = copy.deepcopy(tuples)
            start = 0
            for shape, dtype in tuples:
                size = np.prod(shape) * NBYTES_FLOAT64
                stop = start + size
                sh_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf[start:stop])

                # deepcopy as a temporary fix to a suspected bug as of Python 3.8.0 + numpy 1.17.4 + matplotlib 3.1.2
                # therefore, these arrays should be used as read only
                self.arrays.append(copy.deepcopy(sh_array))

                start = stop
            self.shm = existing_shm
            self.name = name
        else:
            total_size = int(np.sum([a.nbytes for a in arrays]))
            shm = SharedMemory(create=True, size=total_size)
            name = shm.name
            self.arrays = []
            self.tuples = []
            start = 0
            for array in arrays:
                size = array.nbytes
                stop = start + size
                sh_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf[start:stop])
                sh_array[:] = array[:]  # Copy the original data into shared memory
                self.arrays.append(sh_array)
                self.tuples.append((array.shape, array.dtype))
                start = stop
            self.shm = shm
            self.name = shm.name

    def get_name(self):
        return self.name

    def get_arrays(self):
        return self.arrays

    def get_tuples(self):
        return self.tuples
