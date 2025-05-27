# coding: utf-8
"""
    MockMain.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import logging
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.BasicUtils import Struct

class MockMain(Dialog):
    def __init__(self, parent):
        self.logger = logging.getLogger(__name__)
        self.testing = False
        self.analyzer = Struct(app_logger=self.logger)
        Dialog.__init__(self, parent, title="Mock Main", visible=False)

    def show(self):
        self._show()
