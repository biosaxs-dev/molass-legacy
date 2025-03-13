# coding: utf-8
"""
    GuPy.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
import pyopencl as cl
from pyopencl.array import Array
import pyclblast

default_dtype = np.float32

ctx = None
queue = None

def get_default_platform():
    platforms = cl.get_platforms()
    return platforms[0]

def initialize():
    global queue, ctx
    # print( 'np.float32=', np.float32 )

    platform = get_default_platform()
    print('platform name:', platform.name)
    devs = platform.get_devices()
    for d in devs:
        print('device name:', d.name)
        device_type = d.type
        device_type_str = cl.device_type.to_string(device_type)
        print('device type:', device_type_str)
        print('global memory size: ', d.global_mem_size / (1024**3), 'GB')
        if device_type_str == 'GPU':
            break

    if False:
        ctx = cl.create_some_context()
    else:
        ctx = cl.Context(
            dev_type=device_type,
            properties=[(cl.context_properties.PLATFORM, platform)])
    queue = cl.CommandQueue(ctx)

def terminate():
    global queue
    queue.finish()

def dot(a, b):
    global queue
    m, j = a.shape
    k, n = b.shape
    assert j == k
    c = np.zeros((m, n), dtype=default_dtype)
    cla = Array(queue, a.shape, a.dtype)
    clb = Array(queue, b.shape, b.dtype)
    clc = Array(queue, c.shape, c.dtype)
    cla.set(a)
    clb.set(b)
    clc.set(c)
    pyclblast.gemm(queue, m, n, k, cla, clb, clc, a_ld=k, b_ld=n, c_ld=n)
    return clc.get()
