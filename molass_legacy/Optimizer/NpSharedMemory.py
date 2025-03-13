"""
    NpSharedMemory.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import numpy as np
from multiprocessing import shared_memory
from molass_legacy._MOLASS.SerialSettings import get_setting

class NpSharedMemory:
    def __init__(self, dtype=int, size=2, name=None):
        if name is None:
            a = np.zeros(size, dtype=dtype)
            self.shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
            self.name = self.shm.name
            self.array = np.ndarray(a.shape, dtype=dtype, buffer=self.shm.buf)
            self.array[:] = a[:]
        else:
            self.shm = shared_memory.SharedMemory(name=name)
            self.name = name
            self.array = np.ndarray((size,), dtype=dtype, buffer=self.shm.buf)

    def close(self):
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def unlink(self):
        if self.shm is not None:
            print("--------------- unlink")
            self.shm.close()
            self.shm.unlink()
            self.shm = None

    def __del__(self):
        self.close()

shm_singleton = None

def create_shm_singleton():
    global shm_singleton
    using_shared_memory = get_setting("using_shared_memory")
    if using_shared_memory:
        shm_singleton = NpSharedMemory()

def destroy_shm_singleton():
    global shm_singleton
    if shm_singleton is not None:
        shm_singleton.unlink()
        shm_singleton = None

def get_shm_singleton():
    if shm_singleton is None:
        create_shm_singleton()
    return shm_singleton

def get_shm_proxy(name):
    return NpSharedMemory(name=name)

def close_shared_memory():
    global shm_singleton
    if shm_singleton is not None:
        shm_singleton.unlink()
        shm_singleton = None
