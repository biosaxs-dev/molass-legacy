# coding: utf-8
"""
    Optimizer.WFM.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np

def proof():
    A = np.random.uniform(0, 1, (4,2))
    B = np.random.uniform(0, 1, (2,3))
    C = np.random.uniform(0, 1, (4,2))
    D = np.random.uniform(0, 1, (2,3))
    pw = np.random.uniform(0, 1, (4,1))
    cw = np.random.uniform(0, 1, (1,3))

    W = pw@cw

    print('---- (1)')
    print(pw*A)
    print(A*pw)
    print('---- (2)')
    print((pw*A)@(B*cw))
    print(pw*(A@B)*cw)
    print('---- (3)')
    print(pw*(A - C))
    print(pw*A - pw*C)
    print('---- (4)')
    print((B + D)*cw)
    print(B*cw + D*cw)
