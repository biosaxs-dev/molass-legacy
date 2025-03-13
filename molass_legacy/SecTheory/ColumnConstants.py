"""
    ColumnConstants.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
SECCONF_LOWER_BOUND = -2.5
BAD_CONFORMANCE_REDUCE = 1e-2       # used when it is positive
PORESIZE_BOUNDSS = (70, 100)         # deprecated and dynamically determined from Exclusion Limit

INJECTION_TIME = -1000
NUM_THEORETICAL_PLATES = 14400      # superceded by number_of_plates setting

Ti_LOWER = -5000        # -5047 with 20130303/NAC, G0665
Ti_UPPER = 0
Np_LOWER = 100
Np_UPPER = NUM_THEORETICAL_PLATES*2
