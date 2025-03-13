"""
    Trimming.TrimmingUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from bisect import bisect_right
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Trimming.TrimmingInfo import TrimmingInfo

def get_default_uv_wl_trimming(sd):
    # task: unifty with the similar coding in AutoRestrictor
    uv_wl_lower_bound = get_setting("uv_wl_lower_bound")
    assert uv_wl_lower_bound is not None
    start = bisect_right(sd.lvector, uv_wl_lower_bound)
    size = len(sd.lvector)
    return TrimmingInfo(1, start, size, size)