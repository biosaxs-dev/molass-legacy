# coding: utf-8
"""
    ReAtsas.Almerge.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""

import numpy as np
import molass_legacy.KekLib.DebugPlot as dplt
from SerialAtsasTools import AlmergeExecutor

class Almerge(AlmergeExecutor):
    def __init__(self):
        AlmergeExecutor.__init__(self)
