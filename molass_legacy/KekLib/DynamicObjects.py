"""
    DynamicObjeccts.py

    Copyright (c) 2018, Masatsuyo Takahashi
"""

class FlexibleStruct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, attr):
        setattr(self, attr, FlexibleStruct())
        return getattr(self, attr)
