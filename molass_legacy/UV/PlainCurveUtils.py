"""
    UV.PlainCurveUtils.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
from bisect import bisect_right
from molass_legacy._MOLASS.SerialSettings import get_setting

def get_flat_wavelength_simple():
    wavelength = get_setting('zero_absorbance')

    zero_absorbance_auto = get_setting('zero_absorbance_auto')
    if zero_absorbance_auto:
        flat_wavelength = get_setting('flat_wavelength')
        if flat_wavelength is not None:
            wavelength = flat_wavelength

    return wavelength

def get_flat_info_impl(wvector):
    w = get_flat_wavelength_simple()
    i = bisect_right(wvector, w)
    max_i = len(wvector) - 1
    i = min(i, max_i)
    if i == max_i:
        w = wvector[i]
    return w, i

def get_flat_info(sd):
    w, i = get_flat_info_impl(sd.lvector)
    return w, i, sd.conc_array[i,:]

def get_flat_wavelength(wvector=None):
    if wvector is None:
        w = get_flat_wavelength_simple()
    else:
        w, i = get_flat_info_impl(wvector)
    return w

def get_both_wavelengths(wvector=None):
    w = get_setting('absorbance_picking')
    return w, get_flat_wavelength(wvector)
