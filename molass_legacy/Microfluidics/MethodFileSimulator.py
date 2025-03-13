# coding: utf-8
"""
    MethodFileSimulator.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from MethodFile import MethodFile

"""
 0  Pressurization
 1  FlowModulation
 2  Sample Filling (pumpB)
 3  FlowModulation
 4  Sample Filling (pumpA')
 5  FlowModulation
 6  Sample Filling (pumpC)
 7  FlowModulation
 8  FlowModulation
 9  u-fluidics wash fast flow II
10  Total flow rate modulation
11  100% pump A' (buffer)
12  pumpA' -> pumpA&C
13  pump A&C (Base Sample)
14  Y channel initialize
15  const flow rate (pump A, B, C)
16  FlowModulation
17  Waste
18  FlowModulation
19  const flow rate (pump A, B)
20  Linearly changing flow rate
21  const. uL/min (pump B)
22  FlowModulation
23  waste
24  FlowModulation
25  Pressurization & Waiting
"""

TOTAL_RATE_COLUMNS = [2, 4, 6, 10]
RATE_B_START_COLUMN = 4
RATE_B_STOP_COLUMN = 5

def set_default_params(self):
    self.delay_factor = 8
    if self.eno_interval is None:
        self.eno_interval = 1
    self.time_delta = self.eno_interval/2

class MethodFileSimulator(MethodFile):
    def __init__(self, filepath):
        MethodFile.__init__(self, filepath)
        self.eno_interval = None
        set_default_params(self)

    def set_params(self, delay_factor=None, default_init=False, debug=False):
        if default_init:
            # this call should not be a method call to allow uses without inheritance.
            set_default_params(self)

        if delay_factor is None:
            delay_factor = self.delay_factor

        i_, j_, k_ = self.guess_critical_points()
        p_ = i_ - 4     # temp
        j_1 = j_ - 1

        integ_duration = 0
        for k, values in enumerate(self.rows):
            assert values[0] == k + 1

            duration = values[1]/60
            integ_duration += duration
            if debug:
                print([k], values + [integ_duration])
            if k == j_1:    # const flow rate (pump A, B)
                self.start_duration = integ_duration
            elif k == p_:       # Total flow rate modulation
                self.pre_start_time = integ_duration
            elif k == i_:       # name == 'Y channel initialize
                frate_total = np.sum([values[k] for k in TOTAL_RATE_COLUMNS])
                self.y_channel_time = integ_duration + delay_factor/frate_total
            elif k == j_:       # Linearly changing flow rate
                self.stop_duration = integ_duration
                self.start_frate_total = np.sum([values[k] for k in TOTAL_RATE_COLUMNS])
                self.start_time = self.start_duration + delay_factor/self.start_frate_total
                self.start_conc = values[RATE_B_START_COLUMN]/self.start_frate_total
                rate_b_stop = values[RATE_B_STOP_COLUMN]
            elif k == k_:       # const. uL/min (pump B)
                self.stop_frate_total = np.sum([values[k] for k in TOTAL_RATE_COLUMNS])
                self.stop_time = self.stop_duration + delay_factor/self.stop_frate_total
                self.stop_conc = rate_b_stop/self.stop_frate_total
            last_values = values

        self.end_time = integ_duration + delay_factor/np.sum([last_values[k] for k in TOTAL_RATE_COLUMNS])

        if debug:
            for r in [  self.start_duration,
                        self.pre_start_time,
                        self.y_channel_time,
                        self.start_frate_total,
                        self.stop_duration,
                        self.stop_frate_total,
                        self.start_time,
                        self.stop_time,
                        self.start_conc,
                        self.stop_conc,
                        ]:
                print(r)
