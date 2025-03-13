# coding: utf-8
"""
    CurveDecomposer.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
from molass_legacy.Models.ElutionCurveModels import EGH
from molass_legacy.ElutionDecomposer import ElutionDecomposer

def decompose(ecurve, model=EGH()):
    x = ecurve.x
    y = ecurve.y
    decomposer = ElutionDecomposer(ecurve, x, y, model=model, retry_valley=True, deeply=True)
    return decomposer.fit_recs
