"""
   BaselineParams.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
from molass_legacy._MOLASS.SerialSettings import get_setting

def get_num_baseparams():
    unified_baseline_type = get_setting("unified_baseline_type")
    if unified_baseline_type == 1:
        return 2
    elif unified_baseline_type == 2:
        return 3
    elif unified_baseline_type == 3:
        return 3
    else:
        assert False
