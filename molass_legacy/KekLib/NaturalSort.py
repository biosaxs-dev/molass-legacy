# coding: utf-8
"""
    NaturalSort.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF
"""
import re

"""
    ナチュラルソート
    https://qiita.com/peg/items/8bf13cff8259ab464d8b
"""
def natural_sorted(l):
    def alphanum_key(s):
        return [int(c) if c.isdecimal() else c for c in re.split('([0-9]+)', s)]
    return sorted(l, key=alphanum_key)

def natural_sorted_dict_items(d):
    def alphanum_key(pair):
        ret = []
        for p in pair:
            ret += [int(c) if c.isdecimal() else c for c in re.split('([0-9]+)', p)]
        return ret
    return sorted(d.items(), key=alphanum_key)
