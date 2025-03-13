# coding: utf-8
"""
    OurProfiler.py

    Copyright (c) 2018-2019, Masatsuyo Takahashi, KEK-PF
"""
import time
import inspect
import cProfile, pstats, io

def profile(closure, site_packeges=False):
    pr = cProfile.Profile()
    pr.enable()

    closure()

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()

    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    file = mod.__file__ + '.prof'

    with open(file, "w") as fh:
        prof_str = s.getvalue()
        if site_packeges:
            fh.write(prof_str)
        else:
            for line in prof_str.split('\n'):
                if line.find('{') > 0 or line.find('<') > 0:
                    continue
                elif line.find('site-packages') > 0:
                    continue
                else:
                    fh.write( line + '\n' )

def take_time(closure, print_time=True):
    start = time.time()
    ret = closure()
    finish = time.time()
    t = finish - start
    if print_time:
        print( 'It took', t )
    return t, ret
