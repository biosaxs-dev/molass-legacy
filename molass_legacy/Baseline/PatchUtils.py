# coding: utf-8
"""
    PatchUtils.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""

def make_area_points(x, y, bottomline=None):
    if bottomline is None:
        return [(x[0],0)] + list(zip(x,y)) + [(x[-1], 0)]
    else:
        r = list(zip(x,bottomline))
        r.reverse()
        return list(zip(x,y)) + r
