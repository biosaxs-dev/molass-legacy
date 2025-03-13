# coding: utf-8
"""
    SliceUtils.py

    Copyright (c) 2021, Masatsuyo Takahashi, KEK-PF
"""

import numpy as np

"""
    from 
    https://stackoverflow.com/questions/53361203/numpy-efficient-way-to-extract-range-of-consecutive-numbers
"""
def slice_consecutives(ary):
  pass
  res=[]
  tmp = [ary[0]]
  for idx, x in np.ndenumerate(ary[1:]):
    if x - ary[idx[0]] > 1 or idx[0] + 2 == len(ary):
      if tmp[0] != tmp[-1]: res.append([tmp[0], tmp[-1]])
      tmp = []
    tmp.append(x)
  if ary[-1] - res[-1][-1] == 1: res[-1][-1] = ary[-1]
  return res
