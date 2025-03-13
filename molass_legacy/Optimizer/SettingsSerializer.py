"""
    SettingsSerializer.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np  # used in the eval, in cases where the value is like 'np.int64(140)'
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

def serialize(items):
    tuple_ = tuple()
    for name in items:
        tuple_ += (get_setting(name),)
    return str(tuple_).replace(" ", "")

def unserialize(items, serialized_str):
    tuple_ = eval(serialized_str)

    for name, value in zip(items, tuple_):
        set_setting(name, value)

ITEMS = ["poresize_bounds", "t0_upper_bound"]

def serialize_for_optimizer():
    return serialize(ITEMS)

def unserialize_for_optimizer(serialized_str):
    unserialize(ITEMS, serialized_str)
