# coding: utf-8
"""
    Rgg.Callbacks.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from pomegranate.callbacks import Callback

class VisualizationtCallback(Callback):
    def __init__(self):
        self.model = None
        self.params = None

    def on_epoch_end(self, logs):
        epoch = logs['epoch']
        print("{} ----------- on_epoch_end".format(epoch))
