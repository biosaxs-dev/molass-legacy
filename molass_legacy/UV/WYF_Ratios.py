"""
    UV.WYF_Ratios.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
from bisect import bisect_right

WAVELENGTHS = [280, 275, 258]

def get_wyf_indeces(wv):
    return [bisect_right(wv, wl) for wl in WAVELENGTHS]

def compute_ratios_from_Puv(wv, Puv, proportions=None, excl_prop=0.05):
    indecdes = get_wyf_indeces(wv)

    if proportions is None:
        colums = slice(None, None)
    else:
        colums = []
        for i, p in enumerate(proportions):
            if p > excl_prop:
                colums.append(i)

    ret_list = []
    for y in Puv.T[colums]:
        a280, a275, a258 = y[indecdes]
        r1 = a275/a280
        r2 = a258/a280
        ret_list.append((r1, r2))

    return ret_list

if __name__ == '__main__':
    pass
