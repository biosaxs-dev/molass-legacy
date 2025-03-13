# coding: utf-8
"""
    KL_Divergence.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.stats import entropy, mode
from scipy.optimize import minimize

def synchronize(pd1, pd2, distance=entropy):
    if pd1.N >= pd2.N:
        pd1_, pd2_ = pd1, pd2
    else:
        pd1_, pd2_ = pd2, pd1

    j = pd2_.x

    def obj_function(p):
        A, B = p
        print(A,B)
        i = A*j + B
        return distance(pd1_.pdf(i), pd2_.ny)    # return KL divergence

    A_init = pd1_.N/pd2_.N
    m1 = pd1_.mode
    m2 = pd2_.mode
    B_init = m1 - m2*A_init
    ratio = abs(B_init)/pd1.N
    if ratio > 0.03:
        print('using means', ratio)
        m1 = pd1_.M1
        m2 = pd2_.M1
        B_init = m1 - m2*A_init
    else:
        print('using modes', ratio)
    result = minimize(obj_function, [A_init, B_init], args=())

    A, B = result.x
    if pd1.N < pd2.N:
        A, B = 1/A, -B/A

    return A, B
