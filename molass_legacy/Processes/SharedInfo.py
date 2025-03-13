"""
    Processes.SharedInfo.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from multiprocessing import Value, Array, RawArray

NUM_QUEUES = 4
QUEUE_GUI = 0
QUEUE_LOADER = 1
QUEUE_RG = 2
QUEUE_RECOG = 3
BUFFER_SIZE = 500*1500*3

class SharedString:
    def __init__(self, s="", maxlen=256):
        self.maxlen = Value('i', maxlen)
        self.strlen = Value('i', 0)
        self.buffer = Array('u', maxlen)
        self.set(s)

    def set(self, s):
        assert len(s) <= self.maxlen.value
        self.strlen.value = len(s)
        self.buffer[:self.strlen.value] = s

    def __str__(self):
        return ''.join(self.buffer[:self.strlen.value])

class SharedInfo:
    def __init__(self):
        self.value = Value('i', 0)
        self.in_folder = SharedString()
        self.proc_states = Array('i', np.zeros(10, dtype=int))
        self.buffer = RawArray('d', BUFFER_SIZE)
        self.buffer_shape = Array('i', [0,0,0])
        self.X = None
        self.curve_buffer = None
        self.curve_xy = None
        self.max_y = Value('d', 0)

    def __repr__(self):
        return str((self.value.value, str(self.in_folder), [v for v in self.proc_states]))

    def set_procstate(self, pn, state):
        self.proc_states[pn] = state

    def get_procstate(self, pn):
        return self.proc_states[pn]

    def initilize_buffer(self, shape=None, array=None, ecurve=None):
        if shape is None:
            assert array is not None
            shape = array.shape
        for i, n in enumerate(shape):
            self.buffer_shape[i] = n
        size = np.prod(shape)
        assert size <= BUFFER_SIZE
        self.X = np.frombuffer(self.buffer, count=size).reshape(shape)
        if array is None:
            array = np.zeros(shape)
        np.copyto(self.X, array)
        print("initilize_buffer:", shape, size)
        if ecurve is not None:
            size = len(ecurve.x)*2
            self.curve_buffer = RawArray('d', size)
            self.curve_xy = np.frombuffer(self.curve_buffer, count=size).reshape((2,size//2))
            self.max_y.value = ecurve.max_y
            np.copyto(self.curve_xy, np.array([ecurve.x, ecurve.y]))

    def set_xr_data(self, i, data):
        self.X[i,:,:] = data

    def get_buffer_ready(self):
        shape = [n for n in self.buffer_shape]
        size = np.prod(shape)
        assert size <= BUFFER_SIZE
        self.X = np.frombuffer(self.buffer, count=size).reshape(shape)
        print("get_buffer_ready:", shape, size)
        if self.curve_buffer is not None:
            size = self.X.shape[0]
            self.curve_xy = np.frombuffer(self.curve_buffer, count=size*2).reshape((2, size))

    def get_xr_data(self, i):
        return self.X[i,:,:]
