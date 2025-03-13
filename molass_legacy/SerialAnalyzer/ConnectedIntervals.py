# coding: utf-8
"""
    ConnectedIntervals.py

    連結区間の識別

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""

ALLOWANCE   = 10

class ConnectedIntervals:
    def __init__( self, index_array ):
        intervals = []
        last = None
        connected_interval = []
        for i in index_array:
            if not (last is None or ( i - last ) < ALLOWANCE ):
                intervals.append( connected_interval )
                connected_interval = []

            connected_interval.append( i )
            last = i

        intervals.append( connected_interval )
        self.intervals  = intervals
