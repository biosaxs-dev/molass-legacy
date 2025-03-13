# coding: utf-8
"""
    SecTools.CorMap.AngularUnit.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np

def angstrom_qv(qv):
    if 3 < qv[-1] and qv[-1] < 8:
        qv = qv * 0.1
    return qv
